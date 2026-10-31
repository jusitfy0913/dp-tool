# -*- coding: utf-8 -*-
"""训练/转换任务管理：启动子进程、解析 TRTFORGE_JSON、广播事件。

为什么不直接在服务进程内 import ultralytics 训练：
- 训练崩溃会拖垮 Web 服务；
- launcher 已有的参数体系与事件协议无需重写；
- 进程隔离让「停止训练」变成可靠的 terminate。

事件流模型：
    子进程 stdout ──► 读线程 ──► 解析 ──► 事件缓冲 + 所有订阅队列
                                              │
                                     WebSocket ◄┘

读线程是阻塞 IO，所以用独立线程；投递到 asyncio 队列时用
loop.call_soon_threadsafe 保证线程安全。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import EVENT_BUFFER_SIZE, LOG_DIR, TRAIN_EVENT_PREFIX


@dataclass
class Run:
    """一个正在执行或已结束的子进程任务。"""

    run_id: str
    kind: str  # "yolo-train" / "anomalib-train" / "engine-build"
    command: list[str]
    cwd: str
    status: str = "running"  # running / finished / failed / cancelled
    exit_code: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    process: subprocess.Popen | None = None
    # 期望生成的产物。退出码为 0 但该文件不存在时，任务判为失败。
    # 为什么需要它：trtexec 在 --saveEngine 路径不可写等情况下会以 0 退出
    # 却不落盘，只看退出码会把失败当成功（规格要求 5.4）。
    expect_file: Path | None = None
    # 最近事件，供断线重连补发
    events: list[dict] = field(default_factory=list)
    # 订阅者队列
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    log_file: Path | None = None

    def snapshot(self) -> dict:
        """给接口返回的任务摘要。"""
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": self.command,
        }


class RunManager:
    """管理所有子进程任务的注册、事件广播与生命周期。"""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """记录主事件循环，读线程投递事件时需要它。"""
        self._loop = loop

    def create(
        self,
        kind: str,
        command: list[str],
        cwd: str,
        env: dict[str, str] | None = None,
        expect_file: Path | None = None,
    ) -> Run:
        """启动子进程并立即开始读取输出。

        子进程启动失败会抛异常，由调用方转成 HTTP 错误。
        `expect_file` 用于「退出码 0 但没产出」的失败判定。
        """
        run_id = uuid.uuid4().hex[:12]
        log_file = LOG_DIR / f"{run_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        child_env = os.environ.copy()
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        if env:
            child_env.update(env)

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
            bufsize=0,
            # Windows 下不弹控制台窗口
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        run = Run(
            run_id=run_id,
            kind=kind,
            command=command,
            cwd=cwd,
            process=process,
            log_file=log_file,
            expect_file=expect_file,
        )
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(target=self._reader, args=(run,), daemon=True)
        thread.start()
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return [r.snapshot() for r in runs]

    def stop(self, run_id: str) -> bool:
        """终止任务。先 terminate，3 秒不退再 kill。"""
        run = self.get(run_id)
        if run is None or run.process is None:
            return False
        proc = run.process
        if proc.poll() is not None:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:  # noqa: BLE001 - 停止失败也要让状态可读
            return False
        run.status = "cancelled"
        return True

    def shutdown_all(self) -> None:
        """服务退出时清理，避免残留训练进程。"""
        with self._lock:
            run_ids = list(self._runs.keys())
        for run_id in run_ids:
            self.stop(run_id)

    # ------------------------------------------------------------------
    # 订阅
    # ------------------------------------------------------------------
    def subscribe(self, run_id: str) -> tuple[asyncio.Queue, list[dict]] | None:
        """注册订阅者，并返回已有的历史事件用于补发。"""
        run = self.get(run_id)
        if run is None:
            return None
        queue: asyncio.Queue = asyncio.Queue()
        with run.lock:
            history = list(run.events)
            run.subscribers.append(queue)
        return queue, history

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        run = self.get(run_id)
        if run is None:
            return
        with run.lock:
            if queue in run.subscribers:
                run.subscribers.remove(queue)

    # ------------------------------------------------------------------
    # 内部：读取子进程输出
    # ------------------------------------------------------------------
    def _reader(self, run: Run) -> None:
        """在独立线程中逐行读取输出，解析并广播。"""
        assert run.process is not None
        stream = run.process.stdout
        log_handle = None
        try:
            log_handle = open(run.log_file, "a", encoding="utf-8", errors="replace")
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                log_handle.write(line + "\n")
                log_handle.flush()
                payload = self._parse_line(line)
                self._publish(run, payload)
        except Exception as exc:  # noqa: BLE001 - 读线程不能静默死掉
            self._publish(
                run,
                {"type": "error", "code": "READER_CRASHED", "message": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            if log_handle is not None:
                log_handle.close()
            exit_code = None
            try:
                exit_code = run.process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            run.exit_code = exit_code
            run.finished_at = time.time()
            if run.status == "running":
                if exit_code == 0 and run.expect_file is not None and not run.expect_file.is_file():
                    # 退出码正常但产物没落盘，不能算成功
                    run.status = "failed"
                else:
                    run.status = "finished" if exit_code == 0 else "failed"

            # 退出码补一层语义化错误。
            # 原因：launcher 用 argparse 解析参数，参数不合法时 argparse 直接往 stderr
            # 打 usage 并 exit(2)，**不会**发 TRTFORGE_JSON error 事件。浏览器只看到
            # 退出码，看不出是"参数写错"还是"训练崩了"，所以这里补一条明确事件。
            synthetic = self._synthetic_error(run, exit_code)
            if synthetic is not None:
                self._publish(run, synthetic)

            # 产物缺失：退出码 0 但目标文件不存在（trtexec 可能静默不落盘）
            missing = self._missing_artifact(run, exit_code)
            if missing is not None:
                self._publish(run, missing)

            self._publish(
                run,
                {
                    "type": "finished",
                    "exit": exit_code,
                    "status": run.status,
                    "log_file": str(run.log_file),
                },
                close=True,
            )

    @staticmethod
    def _synthetic_error(run: Run, exit_code: int | None) -> dict | None:
        """把特殊退出码翻译成用户能理解的事件。

        - 2  ：argparse 参数错误（launcher 不会自己发 error 事件）
        - 130：被 Ctrl+C / 主动取消
        """
        if exit_code == 2:
            # 从日志尾部捞一行 usage 提示，帮助定位是哪个参数出错
            tail: list[str] = []
            if run.log_file and run.log_file.is_file():
                try:
                    lines = run.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = [ln for ln in lines[-12:] if ln.strip()]
                except OSError:
                    tail = []
            hint = "；".join(tail[-3:]) if tail else "请检查参数值格式"
            return {
                "type": "error",
                "code": "ARGUMENT_ERROR",
                "message": f"launcher 参数解析失败（退出码 2）：{hint}",
                "exit": exit_code,
            }
        if exit_code == 130:
            return {
                "type": "error",
                "code": "CANCELLED",
                "message": "任务被取消（退出码 130）",
                "exit": exit_code,
            }
        return None

    @staticmethod
    def _missing_artifact(run: Run, exit_code: int | None) -> dict | None:
        """退出码为 0 但产物没落盘时，给出可读错误。

        典型场景：trtexec 的 `--saveEngine` 指到不可写目录，进程正常退出却
        没有 .engine 文件。只看退出码会误判成功，用户点「下一步推理」才发现
        文件不存在。这里提前把失败暴露出来，并附上日志尾部方便定位。
        """
        if exit_code != 0 or run.expect_file is None:
            return None
        if run.expect_file.is_file():
            return None

        tail: list[str] = []
        if run.log_file and run.log_file.is_file():
            try:
                lines = run.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = [ln for ln in lines[-15:] if ln.strip()]
            except OSError:
                tail = []
        hint = "；".join(tail[-4:]) if tail else "（无日志输出）"

        return {
            "type": "error",
            "code": "ARTIFACT_MISSING",
            "message": (
                f"命令退出码为 0，但没有生成目标文件：{run.expect_file}。"
                f"常见原因是输出目录不存在或没有写权限。日志尾部：{hint}"
            ),
            "expect_file": str(run.expect_file),
            "exit": exit_code,
        }

    @staticmethod
    def _parse_line(line: str) -> dict:
        """把一行输出转成事件。

        带 TRTFORGE_JSON 前缀的是结构化事件；其余按普通日志处理。
        """
        if line.startswith(TRAIN_EVENT_PREFIX):
            body = line[len(TRAIN_EVENT_PREFIX):].strip()
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    return data
                return {"type": "log", "line": line}
            except json.JSONDecodeError:
                # 前缀在但 JSON 坏了，原样透出便于排查
                return {"type": "log", "line": line}
        return {"type": "log", "line": line}

    def _publish(self, run: Run, payload: dict[str, Any], close: bool = False) -> None:
        """把事件写入缓冲并投递给所有订阅者。"""
        with run.lock:
            run.events.append(payload)
            if len(run.events) > EVENT_BUFFER_SIZE:
                del run.events[: len(run.events) - EVENT_BUFFER_SIZE]
            queues = list(run.subscribers)

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in queues:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except RuntimeError:
                # 事件循环已关闭，忽略
                pass

        if close:
            with run.lock:
                run.subscribers.clear()


# 全局单例：FastAPI 的 lifespan 会绑定事件循环
manager = RunManager()
