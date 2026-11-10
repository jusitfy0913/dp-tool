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

import progress
import warmpool
import gpulock
from config import EVENT_BUFFER_SIZE, LOG_DIR, TRAIN_EVENT_PREFIX

#: 会独占 GPU（或会写同一批产物目录）的任务类型。
#:
#: 为什么需要这个分类：一台机器只有一块显卡，两个训练同时跑就是互相抢显存，
#: 结果通常是双双 OOM；而它们还可能写进同一个 ``runs/<name>`` 目录，
#: 后结束的那个会把前一个的产物盖掉。引擎构建同理——trtexec 的
#: ``--saveEngine`` 指向同一个文件时谁后写谁赢。
#:
#: 反过来，``*-check-env`` / ``yolo-export`` 这类探测任务又短又能重叠跑，
#: 一起拦掉只会让用户觉得「点什么都没反应」，所以只拦下面这些。
EXCLUSIVE_RUN_KINDS = frozenset(
    {
        "yolo-train",
        "anomalib-train",
        "engine-build",
        "infer-batch",
        "infer-video",
        # 自动实验循环：外层是 agent，内层会派生真正的训练进程。
        # 必须算独占，否则它和网页训练会同时抢显存（两块训练一起 OOM）。
        "autoexp-agent",
    }
)

#: 串联任务按阶段创建，kind 形如 ``pipeline-train``。同样是重任务，一并算独占。
EXCLUSIVE_RUN_PREFIX = "pipeline-"


def is_exclusive_kind(kind: str) -> bool:
    """该 kind 的任务是否独占 GPU / 产物目录。"""
    return kind in EXCLUSIVE_RUN_KINDS or kind.startswith(EXCLUSIVE_RUN_PREFIX)



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
    # 引擎构建专用：进度推断器 + 心跳线程停止信号。其它任务保持 None。
    # 之所以挂在 Run 上而不是 RunManager 里：进度天然属于某个任务的生命周期，
    # 任务一结束（_reader 的 finally）就该停心跳，放在 Run 上最不容易漏。
    tracker: Any = None
    progress_stop: threading.Event | None = None
    # 估算所需的信息（onnx_mb / precision / dynamic / imgsz），同时回传给前端，
    # 让「预计剩余」在第一个事件到达前就能显示。
    meta: dict = field(default_factory=dict)
    # 是否复用了常驻预热进程。前端据此显示「热启动」，日志与排障也用得上。
    warm: bool = False
    # 事件序号，从 1 开始自增。断线重连补发历史时前端靠它去重。
    seq: int = 0
    # 本次任务是否持有跨进程 GPU 锁（gpulock）。结束时要按它决定是否释放，
    # 不能无条件 release：抢不到锁的任务若也去 release，会删掉别人的锁。
    gpu_lock_held: bool = False

    def snapshot(self) -> dict:
        """给接口返回的任务摘要。"""
        end = self.finished_at or time.time()
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # 已用秒数：刷新页面后前端可以直接接着显示，不必从 0 重新数
            "elapsed": round(max(0.0, end - self.started_at), 1),
            "meta": dict(self.meta),
            "warm": self.warm,
            "command": self.command,
            # 日志文件路径：任务跑完、服务重启后内存里什么都没有了，
            # 只有这个文件还在，排障时前端要拿它去 /api/runs/{id}/log 取回全文。
            "log_file": str(self.log_file) if self.log_file else None,
        }


def _apply_offline_defaults(kind: str, env: dict[str, str]) -> None:
    """给训练类子进程关掉「每次开训都联网」的检查。

    Ultralytics 在 import 时读 ``YOLO_OFFLINE`` 决定要不要做 PyPI 版本检查
    （日志里那条 "New https://pypi.org/... available" 就是它）。网络慢时这几秒
    是纯等待，且对训练结果没有任何影响。该开关只管版本检查与遥测，
    不影响权重下载，所以默认打开是安全的；需要在线检查时设
    ``TRTFORGE_YOLO_ONLINE=1``。
    """
    online = os.environ.get("TRTFORGE_YOLO_ONLINE", "").strip().lower()
    if online not in {"1", "true", "yes", "on"} and kind.startswith("yolo"):
        env.setdefault("YOLO_OFFLINE", "1")
    # HuggingFace（Anomalib 的 EfficientAd 会用到）遥测与符号链接告警：纯噪音
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


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
        meta: dict | None = None,
        estimate: float | None = None,
        warm_kind: str | None = None,
    ) -> Run:
        """启动子进程并立即开始读取输出。

        子进程启动失败会抛异常，由调用方转成 HTTP 错误。
        `expect_file` 用于「退出码 0 但没产出」的失败判定。
        `meta` / `estimate` 只对 `kind == "engine-build"` 有意义：
        前者是估算依据（同时回传给前端），后者是本轮估算耗时（秒）。
        `warm_kind` 非空时优先复用常驻预热进程；取不到就照旧冷启动，
        因此这个参数只会更快，不会让原本能跑的任务跑不起来。
        """
        run_id = uuid.uuid4().hex[:12]
        log_file = LOG_DIR / f"{run_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # 允许调用方传字符串：这条路径上的代码在读取线程里跑，
        # 一旦类型不对就是「线程带异常死掉、run 永远 running」，不划算。
        if isinstance(expect_file, str):
            expect_file = Path(expect_file) if expect_file else None

        child_env = os.environ.copy()
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        _apply_offline_defaults(kind, child_env)
        if env:
            child_env.update(env)

        # 跨进程 GPU 占用锁：让命令行的 autoexp.py、以及本机其它 Web 实例
        # 都能看见「显卡正被谁占着」。抢不到**不在这里抛错**——理由有两个：
        #   1) 拒绝的职责在 app._reject_if_busy()，那里能返回带文案的 409；
        #   2) 同进程可以同时挂多个独占任务（既有并发测试依赖这一行为），
        #      抛错会改掉已生效的接口语义。
        lock_held = False
        if is_exclusive_kind(kind):
            lock_held, lock_reason = gpulock.acquire("web", kind, run_id)
            if not lock_held:
                message = f"GPU 已被占用，本任务未取得跨进程锁：{lock_reason}"
                # 直接打 stderr 而不是引 logging：runners 全模块没有 logger，
                # 为一行日志新建一个配置不划算，而这条信息要能在服务日志里看到。
                print(f"[runners] {message}", flush=True)

        worker = warmpool.pool.take(warm_kind) if warm_kind else None
        if worker is not None:
            try:
                return self._create_warm(
                    run_id, kind, worker, command, cwd, log_file, expect_file, meta,
                    gpu_lock_held=lock_held,
                )
            except Exception:  # noqa: BLE001 - 热启动任何异常都退回冷启动
                worker.terminate()

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
            meta=dict(meta or {}),
            gpu_lock_held=lock_held,
        )
        with self._lock:
            self._runs[run_id] = run

        # 第一条事件：让前端立刻知道「进程已经起来了」，而不是干等第一行日志。
        self._publish(
            run,
            {
                "type": "phase",
                "phase": "spawned",
                "label": "训练进程已启动",
                "pid": process.pid,
                "warm": False,
                "elapsed_ms": 0,
                "step_ms": 0,
            },
        )
        self._start_progress_heartbeat(kind, run, estimate)
        threading.Thread(target=self._reader, args=(run,), daemon=True).start()
        return run

    def _create_warm(
        self,
        run_id: str,
        kind: str,
        worker: warmpool.WarmWorker,
        command: list[str],
        cwd: str,
        log_file: Path,
        expect_file: Path | None,
        meta: dict | None,
        gpu_lock_held: bool = False,
    ) -> Run:
        """把任务交给预热进程。复用失败由 create() 捕获后回退冷启动。"""
        run = Run(
            run_id=run_id,
            kind=kind,
            command=command,
            cwd=cwd,
            process=worker.process,
            log_file=log_file,
            expect_file=expect_file,
            meta=dict(meta or {}),
            warm=True,
            gpu_lock_held=gpu_lock_held,
        )
        with self._lock:
            self._runs[run_id] = run

        pre_events = worker.bind(command[2:])
        # 第一条事件必须是 spawned：前端靠它立刻确认「任务已经起来了」。
        # 预热期的 boot / warm_ready 补发在它之后，用户能看出命中了热启动。
        self._publish(
            run,
            {
                "type": "phase",
                "phase": "spawned",
                "label": "复用预热进程（跳过依赖导入）",
                "pid": worker.process.pid if worker.process else None,
                "warm": True,
                "elapsed_ms": 0,
                "step_ms": 0,
            },
        )
        for payload in pre_events:
            self._publish(run, payload)
        threading.Thread(target=self._warm_reader, args=(run, worker), daemon=True).start()
        return run

    def _start_progress_heartbeat(self, kind: str, run: Run, estimate: float | None) -> None:
        """引擎构建起心跳线程：tactic 选择阶段 trtexec 一行都不打（实测空白 473 秒），
        只靠日志驱动的话进度条会长时间冻在原处。"""
        if kind != "engine-build":
            return
        run.tracker = progress.EngineBuildAnalyzer(
            estimate,
            meta=run.meta,
            stats=progress.default_store(),
            started_at=run.started_at,
        )
        run.progress_stop = threading.Event()
        threading.Thread(
            target=progress.run_heartbeat,
            args=(run.tracker, lambda payload: self._publish(run, payload), run.progress_stop),
            daemon=True,
        ).start()

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return [r.snapshot() for r in runs]

    def active_exclusive(self) -> list[dict]:
        """正在运行、且属于「独占」类的任务摘要（新的在前）。

        只返回 running 的：finished / failed / cancelled 都不再占资源。
        """
        with self._lock:
            runs = [r for r in self._runs.values() if r.status == "running" and is_exclusive_kind(r.kind)]
        runs.sort(key=lambda r: r.started_at, reverse=True)
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
            # 先标记再 terminate，顺序不能反。
            # 读线程在自己的 finally 里会 `if run.status == "running"` 才覆盖状态，
            # 并立刻广播 finished。若等 proc.wait() 返回后再标记，读线程可能已经
            # 把 terminate 导致的退出码 1 判成 "failed" 并广播出去，
            # 用户主动停止就会显示成「构建失败」（实测会随机复现）。
            run.status = "cancelled"
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:  # noqa: BLE001 - 停止失败也要让状态可读
            run.status = "running"
            return False
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
        log_handle = None
        try:
            log_handle = open(run.log_file, "a", encoding="utf-8", errors="replace")
            for raw in iter(run.process.stdout.readline, b""):
                self._consume(run, raw, log_handle)
        except Exception as exc:  # noqa: BLE001 - 读线程不能静默死掉
            self._publish_crash(run, exc)
        finally:
            self._close_log(log_handle)
            self._finalize(run)

    def _warm_reader(self, run: Run, worker: warmpool.WarmWorker) -> None:
        """预热进程的读线程：行来源换成 worker 的任务队列，其余流程与冷启动完全一致。"""
        log_handle = None
        try:
            log_handle = open(run.log_file, "a", encoding="utf-8", errors="replace")
            for raw in worker.read_task_lines():
                self._consume(run, raw, log_handle)
        except Exception as exc:  # noqa: BLE001
            self._publish_crash(run, exc)
        finally:
            self._close_log(log_handle)
            self._finalize(run)

    def _consume(self, run: Run, raw: bytes, log_handle: Any) -> None:
        """处理一行子进程输出：落盘 + 广播 + 喂给进度推断器。"""
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            return
        log_handle.write(line + "\n")
        log_handle.flush()
        payload = self._parse_line(line)
        self._publish(run, payload)
        # 引擎构建：同一行再喂给进度推断器。tracker 只在阶段变化时返回事件，
        # 普通日志行返回 None，所以不会多出事件。
        if run.tracker is not None:
            tracker_event = run.tracker.feed(line)
            if tracker_event is not None:
                self._publish(run, tracker_event)

    @staticmethod
    def _close_log(log_handle: Any) -> None:
        if log_handle is not None:
            log_handle.close()

    def _publish_crash(self, run: Run, exc: Exception) -> None:
        self._publish(
            run,
            {"type": "error", "code": "READER_CRASHED", "message": f"{type(exc).__name__}: {exc}"},
        )

    def _finalize(self, run: Run) -> None:
        """进程结束后收尾：定状态、补事件。冷启动与预热进程共用。"""
        # 先停心跳：进程都结束了还继续推进度，会和下面的 finished 抢语义
        if run.progress_stop is not None:
            run.progress_stop.set()
        # 释放跨进程 GPU 锁。必须在最前面做：下面的补发事件、写日志都可能抛异常，
        # 一旦抛在被释放之前，这把锁就会一直挂到进程死亡，把整台机器锁死。
        if run.gpu_lock_held:
            gpulock.release()
            run.gpu_lock_held = False
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

        # 进度收尾：成功补 100%，失败就停在原处（不假装走满）
        if run.tracker is not None:
            final_progress = run.tracker.finish(run.status, exit_code)
            if final_progress is not None:
                self._publish(run, final_progress)

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
            # 事件序号：断线重连时要靠它去重。
            # 重连时后端会把最近 EVENT_BUFFER_SIZE 条历史整段补发，没有序号的话
            # 前端只能再追加一遍——实测网络闪断一次，日志区就多出一整份重复内容，
            # 指标表也会重复记同一轮。
            run.seq += 1
            payload["seq"] = run.seq
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
