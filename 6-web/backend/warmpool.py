# -*- coding: utf-8 -*-
"""常驻预热进程池：把「解释器 + 重依赖导入」这笔固定开销提前付掉。

为什么需要它
------------
训练 launcher 每次冷启动都要重新 import torch / ultralytics / anomalib。本机实测：

    python -c pass                 0.05s
    import torch + ultralytics     1.30s
    import torch + Patchcore       5.70s

这段导入时间用户只能盯着空白日志等。预热进程在服务空闲时就把依赖导好，
点击训练时通过 stdin 把参数交给它，导入开销归零。

设计约束（都是为了让「提速」不换来新的不稳定）
----------------------------------------------
1. 只服务一个任务：任务跑完进程即退出。torch / Ultralytics 有大量进程级全局
   状态（CUDA 上下文、随机源、默认配置），复用同一进程跑第二个任务会让上一个
   任务的设置悄悄影响下一个，而日志上看不出来。
2. 纯增量、可回退：拿不到就绪的预热进程时，调用方走原来的冷启动路径，
   行为与改造前完全一致。
3. 可关闭：``TRTFORGE_WARM=0`` 时一个进程都不起，内存占用回到原样。
4. 真实隔离：训练仍在独立子进程里跑，训练崩溃不影响 Web 服务；
   「停止」仍然是 terminate 整个进程，语义不变。

进程协议
--------
启动：``python <launcher> --mode warm``
就绪：stdout 打 ``TRTFORGE_JSON {"type":"event","name":"warm_ready",...}``
下发：向 stdin 写一行 ``{"argv": ["--mode","train",...]}``
结束：launcher 执行完该任务后自然退出
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from config import TRAIN_EVENT_PREFIX

#: 预热进程从启动到 ready 的等待上限。Anomalib 冷导入实测约 6 秒，
#: GPU 驱动的首次初始化偶尔更慢，给到 60 秒；超时直接放弃，走冷启动。
READY_TIMEOUT_SECONDS = 60.0
#: 等待队列结束的哨兵
_EOF = object()


class WarmWorker:
    """一个已经（或正在）预热好的 launcher 进程。"""

    def __init__(self, kind: str, python: str, launcher: str, cwd: str, env: dict[str, str]) -> None:
        self.kind = kind
        self.python = python
        self.launcher = launcher
        self.cwd = cwd
        self._env = env
        self.process: subprocess.Popen | None = None
        self.started_at = time.time()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._bound = False
        self._pre_events: list[dict] = []  # 绑定前的事件（boot / warm_ready），绑定后补发
        self._lines: queue.Queue = queue.Queue()
        self._pump_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动进程并开始读它的输出（就绪前的输出只缓存，不丢）。"""
        command = [self.python, self.launcher, "--mode", "warm"]
        self.process = subprocess.Popen(
            command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._env,
            bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def bound(self) -> bool:
        return self._bound

    def wait_ready(self, timeout: float = READY_TIMEOUT_SECONDS) -> bool:
        """等就绪。进程提前退出或超时都返回 False（调用方据此回退冷启动）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._ready.wait(timeout=0.2):
                return self.alive
            if not self.alive:
                return False
        return False

    def bind(self, argv: list[str]) -> list[dict]:
        """把任务交给预热进程。

        返回绑定前缓存的事件（boot / warm_ready），由调用方补发到任务事件流里，
        这样用户能看到「这次命中了预热进程」。
        """
        with self._lock:
            if self._bound:
                raise RuntimeError("预热进程已经被占用")
            if not self.alive or self.process is None or self.process.stdin is None:
                raise RuntimeError("预热进程已退出")
            self._bound = True
            pre_events = list(self._pre_events)
            self._pre_events.clear()
        payload = json.dumps({"argv": argv}, ensure_ascii=False) + "\n"
        self.process.stdin.write(payload.encode("utf-8"))
        self.process.stdin.flush()
        return pre_events

    def read_task_lines(self) -> Iterator[bytes]:
        """在 bind 之后逐行读取任务输出，直到进程退出。"""
        while True:
            item = self._lines.get()
            if item is _EOF:
                return
            yield item

    def terminate(self) -> None:
        """回收进程（服务退出或空闲回收时用）。"""
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None and not self.process.stdin.closed:
                self.process.stdin.close()  # 关闭 stdin 让 warm 循环自然退出
            self.process.wait(timeout=3)
        except Exception:  # noqa: BLE001 - 回收失败只能强杀，不能让服务卡住
            try:
                self.process.kill()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 内部：读输出
    # ------------------------------------------------------------------
    def _pump(self) -> None:
        """独立线程读进程输出：就绪前缓存事件，绑定后交给任务队列。"""
        assert self.process is not None and self.process.stdout is not None
        try:
            for raw in iter(self.process.stdout.readline, b""):
                payload = _parse_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                with self._lock:
                    bound = self._bound
                    if not bound:
                        # 只保留结构化事件；就绪前的普通日志（pip 提示等）不补发，
                        # 否则会把「预热时的噪音」混进这次的训练日志里。
                        if len(self._pre_events) < 50 and payload.get("type") != "log":
                            payload["warm"] = True
                            self._pre_events.append(payload)
                if payload.get("name") == "warm_ready":
                    self._ready.set()
                if bound:
                    self._lines.put(raw)
        finally:
            # 进程退出：先让 wait_ready 不要死等，再结束任务读取循环
            self._ready.set()
            self._lines.put(_EOF)


def _parse_line(line: str) -> dict[str, Any]:
    """把一行输出转成事件（与 runners.RunManager._parse_line 保持一致）。"""
    if line.startswith(TRAIN_EVENT_PREFIX):
        body = line[len(TRAIN_EVENT_PREFIX):].strip()
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"type": "log", "line": line}


class WarmPool:
    """每类任务最多保留一个预热进程。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, WarmWorker] = {}
        self._specs: dict[str, dict[str, str]] = {}
        self._enabled = False
        self._idle_seconds = 1800
        self._started_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def configure(
        self,
        specs: dict[str, dict[str, str]],
        enabled: bool = True,
        idle_seconds: int = 1800,
    ) -> None:
        """注册可预热的任务类型。``specs`` 形如 ``{"yolo": {"python":..., "launcher":..., "cwd":...}}``。"""
        with self._lock:
            self._specs = {kind: dict(value) for kind, value in specs.items()}
            self._enabled = bool(enabled) and bool(specs)
            self._idle_seconds = max(60, int(idle_seconds))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def kinds(self) -> list[str]:
        return sorted(self._specs)

    # ------------------------------------------------------------------
    # 预热 / 取用
    # ------------------------------------------------------------------
    def ensure(self, kind: str, wait: bool = False) -> WarmWorker | None:
        """确保该类有预热进程在跑；已有则直接返回。"""
        if not self._enabled or kind not in self._specs:
            return None
        with self._lock:
            existing = self._workers.get(kind)
            if existing is not None and existing.alive:
                return existing
            worker = self._spawn_locked(kind)
        if worker is None:
            return None
        if wait:
            worker.wait_ready()
        return worker

    def take(self, kind: str) -> WarmWorker | None:
        """取一个已就绪的预热进程。取不到返回 None（调用方回退冷启动）。"""
        if not self._enabled:
            return None
        with self._lock:
            worker = self._workers.pop(kind, None)
        if worker is None:
            return None
        if not worker.alive or not worker.ready:
            # 还在导入或已经死掉：立刻回收并重新预热，本次任务走冷启动
            worker.terminate()
            self.ensure_async(kind)
            return None
        # 取走之后马上补一个新的，这样连续开训也能命中
        self.ensure_async(kind)
        return worker

    def ensure_async(self, kind: str) -> None:
        """后台补预热，不阻塞当前请求。"""
        if not self._enabled or kind not in self._specs:
            return
        threading.Thread(target=self.ensure, args=(kind,), daemon=True).start()

    def _spawn_locked(self, kind: str) -> WarmWorker | None:
        spec = self._specs.get(kind)
        if not spec:
            return None
        python = spec.get("python", "")
        launcher = spec.get("launcher", "")
        cwd = spec.get("cwd", "")
        if not python or not Path(python).is_file() or not launcher or not Path(launcher).is_file():
            return None
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        worker = WarmWorker(kind, python, launcher, cwd or str(Path(launcher).parent), env)
        try:
            worker.start()
        except Exception:  # noqa: BLE001 - 预热失败不该影响服务启动
            return None
        self._workers[kind] = worker
        self._started_at[kind] = time.time()
        return worker

    # ------------------------------------------------------------------
    # 状态 / 回收
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """给前端的预热状态。ready 的进程才是「下一次点击能立刻开训」。"""
        with self._lock:
            items = []
            for kind in sorted(self._specs):
                worker = self._workers.get(kind)
                if worker is None or not worker.alive:
                    state = "idle"
                elif worker.ready:
                    state = "ready"
                else:
                    state = "warming"
                items.append(
                    {
                        "kind": kind,
                        "state": state,
                        "pid": worker.process.pid if worker is not None and worker.process else None,
                        "age_seconds": round(time.time() - worker.started_at, 1) if worker is not None else None,
                    }
                )
        return {"enabled": self._enabled, "kinds": items}

    def reap_idle(self) -> None:
        """回收空闲太久的预热进程，避免白占几百 MB 内存。"""
        if not self._enabled:
            return
        now = time.time()
        with self._lock:
            for kind, worker in list(self._workers.items()):
                if worker.bound:
                    continue
                started = self._started_at.get(kind, worker.started_at)
                if now - started > self._idle_seconds:
                    worker.terminate()
                    self._workers.pop(kind, None)

    def shutdown(self) -> None:
        """服务退出时清掉所有预热进程，避免残留。"""
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.terminate()


#: 全局单例：app 的 lifespan 负责 configure / shutdown
pool = WarmPool()
