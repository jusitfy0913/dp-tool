# -*- coding: utf-8 -*-
"""推理代理：Web 进程通过它调用推理，自己不加载任何推理 DLL。

对外函数与 `infer` / `ocr_infer` 同名同参数（多一个可选的 ``source``，
只用于出错时告诉用户是哪张图），返回结构完全一致。

两种模式
--------
- 隔离（默认）：请求转给 `infer_worker.py` 子进程。子进程崩溃时当前请求报 503，
  随后后台自动拉起新进程；60 秒内连崩 3 次则熔断，状态里把 DLL 标成不可用，
  路由层原有的「DLL 不可用 → onnxruntime 降级」逻辑自然接管。
- 直通：``TRTFORGE_INFER_ISOLATION=0`` 时直接调用原模块，行为与 1.27.0 一致，
  仅用于排障。开关在每次调用时读取，测试可以随时切换。
"""

from __future__ import annotations

import collections
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

import config
import infer
import ocr_infer

BACKEND = Path(__file__).resolve().parent

ISOLATION_ENV = "TRTFORGE_INFER_ISOLATION"
TIMEOUT_ENV = "TRTFORGE_INFER_TIMEOUT"
#: 推理进程数。默认 1 = 与 1.30.x 完全一致（所有请求走同一个进程、同一个句柄集）。
#: 调到 2~4 才能让多个请求真正并行，代价是显存/内存按倍数增长。
WORKERS_ENV = "TRTFORGE_INFER_WORKERS"
#: 请求在「进程都忙」队列里最多等多久。等不到就报 503，让前端提示稍后重试。
QUEUE_TIMEOUT_ENV = "TRTFORGE_INFER_QUEUE_TIMEOUT"
START_TIMEOUT_SECONDS = 90.0
OCR_FIRST_LOAD_TIMEOUT_SECONDS = 180.0
CRASH_WINDOW_SECONDS = 60.0
CRASH_LIMIT = 3
#: 单个进程崩 3 次是它自己的问题；60 秒内全池累计崩 6 次，说明是同一张坏图
#: 在把每个进程挨个打崩，再自动重启也只是白烧显存，直接全池熔断。
POOL_CRASH_LIMIT = 6
TRIP_COOLDOWN_SECONDS = 300.0
SHUTDOWN_GRACE_SECONDS = 3.0
MAX_WORKERS = 8
QUEUE_TIMEOUT_SECONDS = 30.0
#: 显存达到这个量才自动开 2 个进程。低于它（小卡 / 无卡）只开 1 个，
#: 因为第二份 TRT 上下文会把显存挤爆，得不偿失。
AUTO_WORKERS_MIN_MB = 6 * 1024
#: nvidia-smi 查询结果的缓存；None 表示还没查过。
_GPU_MEMORY_MB: int | None = None


class WorkerError(RuntimeError):
    """推理进程层面的失败（崩溃、超时、熔断）。status 对应 HTTP 状态码。"""

    def __init__(self, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.status = status


class _Died(Exception):
    pass


class _TimedOut(Exception):
    pass


def isolation_enabled() -> bool:
    return os.environ.get(ISOLATION_ENV, "1").strip() != "0"


def _request_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get(TIMEOUT_ENV, "60")))
    except ValueError:
        return 60.0


def _gpu_memory_mb() -> int:
    """查最大一块显卡的显存（MB）。查不到（没有卡 / 没有驱动 / 没有 nvidia-smi）返回 0。

    只用 `nvidia-smi` 而不引第三方库：这是唯一在所有部署形态下都能用的零依赖探测，
    而且结果只取一次并缓存，不在每次请求路径上。
    """
    global _GPU_MEMORY_MB
    if _GPU_MEMORY_MB is not None:
        return _GPU_MEMORY_MB
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
        _GPU_MEMORY_MB = max(values) if values else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        _GPU_MEMORY_MB = 0
    return _GPU_MEMORY_MB


def _auto_workers() -> int:
    """按显存决定进程数。

    依据是 T1000 8GB 上的实测（4 个客户端同时推理）：
    2 个进程 QPS +28%、P95 从 139.8 ms 降到 19.3 ms；4 个进程吞吐不再增长
    而 P95 反而回到 48.7 ms——小卡的核心数撑不住更多并发上下文。
    所以自动档最多给到 2，想更多请显式设 `TRTFORGE_INFER_WORKERS`。
    """
    return 2 if _gpu_memory_mb() >= AUTO_WORKERS_MIN_MB else 1


def worker_count() -> int:
    """要起几个推理进程。

    优先级：显式环境变量 > 按显存自动判断。每次启动时读取，测试可以随时改。
    """
    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return _auto_workers()
        if value > 0:
            return max(1, min(MAX_WORKERS, value))
    return _auto_workers()


def queue_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get(QUEUE_TIMEOUT_ENV, QUEUE_TIMEOUT_SECONDS)))
    except ValueError:
        return QUEUE_TIMEOUT_SECONDS


def _unpack_blobs(reply: dict) -> dict:
    """把工作进程放在共享内存里的大二进制搬回 `result`。

    Anomalib 一次要回 overlay JPEG + heatmap PNG（约 1.8 MB），pickle 过管道实测
    155 ms，比模型本身还贵；放进共享内存只剩一次拷贝。约定见
    `infer_worker._pack_blobs`：字段值被置 None，偏移在 `reply["blobs"]` 里。

    读完立刻 `close()`，不 `unlink()`——那块内存归工作进程所有，下次还要复用。
    """
    blobs = reply.get("blobs")
    result = reply.get("result")
    if not blobs or not isinstance(result, dict):
        return reply
    try:
        shm = shared_memory.SharedMemory(name=blobs["shm"], track=False)
    except TypeError:  # Python < 3.13 没有 track 参数
        shm = shared_memory.SharedMemory(name=blobs["shm"])
    try:
        for key, (offset, length) in blobs["items"].items():
            result[key] = bytes(shm.buf[offset:offset + int(length)])
    finally:
        shm.close()
    reply.pop("blobs", None)
    return reply


def worker_command() -> list[str]:
    """与 `config.tool_command` 同一套规则：安装包里由编译后的程序执行。"""
    tool_exe = os.environ.get("TRTFORGE_TOOL_EXE", "").strip()
    if tool_exe:
        return [str(Path(tool_exe)), "--tool", "infer_worker"]
    return [sys.executable, str(BACKEND / "infer_worker.py")]


def _format_exit(code: int | None) -> str:
    if code is None:
        return "未知"
    unsigned = code & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return "0xC0000005（访问违例）"
    if unsigned > 0x7FFFFFFF:
        return f"0x{unsigned:08X}"
    return str(code)


def _status_from_dict(data: dict) -> infer.BackendStatus:
    return infer.BackendStatus(
        yolo_ok=bool(data.get("yolo_ok")),
        anomalib_ok=bool(data.get("anomalib_ok")),
        dll_dirs=list(data.get("dll_dirs") or []),
        detail=str(data.get("detail") or ""),
        onnx_ok=bool(data.get("onnx_ok")),
        onnx_detail=str(data.get("onnx_detail") or ""),
        onnx_provider=str(data.get("onnx_provider") or ""),
    )


def _unavailable_status(reason: str) -> dict:
    return {
        "yolo_ok": False,
        "anomalib_ok": False,
        "dll_dirs": [],
        "detail": reason,
        "onnx_ok": False,
        "onnx_detail": reason,
        "onnx_provider": "",
        "ocr": {"ok": False, "loaded": False, "detail": reason, "dll": ""},
    }


class _Worker:
    def __init__(self, index: int = 0) -> None:
        #: 0 号进程是「主进程」：状态查询、缓存统计、手动重启、OCR 都固定走它，
        #: 这样单进程时代的所有外部行为（接口字段、测试断言）都原样保留。
        self.index = index
        #: 多进程时日志必须分开写，否则两份输出交错到没法读。
        self._tag = "" if index == 0 else f"#{index + 1} "
        self._log_name = "infer_worker.log" if index == 0 else f"infer_worker-{index + 1}.log"
        #: 池子挂上来的回调：某次请求把进程打崩时通知池子累计（跨进程熔断用）。
        self.on_crash = None
        #: 返回 False 时禁止拉起进程。自动实验占显存期间用它挡住「空闲退出后自动重拉」。
        self.may_start = None
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._conn = None
        self._shm: shared_memory.SharedMemory | None = None
        self._status: dict = _unavailable_status("推理进程尚未启动")
        self._crashes: collections.deque[float] = collections.deque()
        self._tripped_at: float | None = None
        self._last_error = ""
        self._ocr_warm = False
        self._restarts = 0

    def tripped(self) -> bool:
        """本进程是否已熔断。熔断的进程不再参与分发，等手动重启恢复。"""
        return self._tripped_at is not None

    # ---------------- 生命周期 ----------------

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._conn is not None

    def _spawn_locked(self) -> None:
        key = secrets.token_hex(16)
        address = rf"\\.\pipe\trtforge-infer-{os.getpid()}-{secrets.token_hex(4)}"
        env = os.environ.copy()
        env["TRTFORGE_INFER_PIPE"] = address
        env["TRTFORGE_INFER_KEY"] = key
        env.setdefault("PYTHONIOENCODING", "utf-8")
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(config.LOG_DIR / self._log_name, "ab")  # noqa: SIM115 - 交给子进程持有
        try:
            self._process = subprocess.Popen(
                worker_command(),
                cwd=str(BACKEND),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log.close()

        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while True:
            if self._process.poll() is not None:
                raise WorkerError(
                    f"{self._tag}推理进程启动失败（退出码 {_format_exit(self._process.returncode)}），"
                    f"详见 {config.LOG_DIR / self._log_name}"
                )
            try:
                self._conn = Client(address, family="AF_PIPE", authkey=bytes.fromhex(key))
                break
            except (FileNotFoundError, OSError):
                if time.monotonic() > deadline:
                    self._kill_locked()
                    raise WorkerError(f"{self._tag}推理进程启动超时") from None
                time.sleep(0.1)

        hello = self._recv_locked(max(1.0, deadline - time.monotonic()))
        self._status = hello["result"]
        self._ocr_warm = False
        self._last_error = ""

    def _start_locked(self) -> None:
        if self.may_start is not None and not self.may_start():
            self._last_error = "显存已让给训练，暂不拉起推理进程"
            self._status = _unavailable_status(self._last_error)
            return
        try:
            self._spawn_locked()
        except (WorkerError, _Died, _TimedOut, OSError) as exc:
            reason = str(exc) or "推理进程启动失败"
            self._last_error = reason
            self._status = _unavailable_status(reason)
            self._kill_locked()

    def _kill_locked(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def startup(self) -> None:
        with self._lock:
            if self.may_start is not None and not self.may_start():
                return
            if not self._alive():
                self._start_locked()

    def shutdown(self) -> None:
        with self._lock:
            if self._alive():
                try:
                    self._conn.send({"op": "shutdown"})
                    self._recv_locked(SHUTDOWN_GRACE_SECONDS)
                    self._process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
                except (WorkerError, _Died, _TimedOut, OSError, subprocess.TimeoutExpired):
                    pass
            self._kill_locked()
            self._process = None
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
                self._shm = None
            self._status = _unavailable_status("推理进程已停止")

    def restart(self) -> dict:
        with self._lock:
            self._crashes.clear()
            self._tripped_at = None
            self._kill_locked()
            self._start_locked()
            return self.state()

    def _restart_in_background(self) -> None:
        def run() -> None:
            with self._lock:
                if self._tripped_at is None and not self._alive():
                    self._restarts += 1
                    self._start_locked()

        threading.Thread(target=run, name="infer-worker-restart", daemon=True).start()

    # ---------------- 熔断 ----------------

    def _record_crash_locked(self) -> bool:
        """记一次崩溃，返回是否触发熔断。"""
        now = time.monotonic()
        self._crashes.append(now)
        while self._crashes and now - self._crashes[0] > CRASH_WINDOW_SECONDS:
            self._crashes.popleft()
        if len(self._crashes) >= CRASH_LIMIT:
            self._tripped_at = now
            reason = f"推理进程 {CRASH_WINDOW_SECONDS:.0f} 秒内崩溃 {CRASH_LIMIT} 次，已暂停自动重启"
            self._status = _unavailable_status(reason)
            self._last_error = reason
            return True
        return False

    def _check_trip_locked(self) -> None:
        if self._tripped_at is None:
            return
        if time.monotonic() - self._tripped_at < TRIP_COOLDOWN_SECONDS:
            raise WorkerError(self._last_error, 503)
        self._tripped_at = None
        self._crashes.clear()

    # ---------------- 收发 ----------------

    def _recv_locked(self, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self._conn.poll(0.2):
                    return self._conn.recv()
            except (EOFError, OSError) as exc:
                raise _Died() from exc
            if self._process.poll() is not None:
                raise _Died()
            if time.monotonic() > deadline:
                raise _TimedOut()

    def _put_image_locked(self, image: np.ndarray) -> tuple[str, list[int]]:
        array = np.ascontiguousarray(image, dtype=np.uint8)
        if self._shm is None or self._shm.size < array.nbytes:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
            size = max(array.nbytes, 8 * 1024 * 1024)
            self._shm = shared_memory.SharedMemory(create=True, size=size)
        target = np.ndarray(array.shape, dtype=np.uint8, buffer=self._shm.buf)
        target[...] = array
        del target
        return self._shm.name, list(array.shape)

    def call(self, message: dict, image: np.ndarray | None = None, *, model: str = "", source: str = "",
             timeout: float | None = None):
        with self._lock:
            self._check_trip_locked()
            if not self._alive():
                if self._process is not None:
                    # 空闲时被外部结束或自行退出：不计入熔断（没有请求牵连），但要留痕
                    code = self._process.poll()
                    self._kill_locked()
                    self._last_error = f"{self._tag}推理进程在空闲时退出（退出码 {_format_exit(code)}），已重新拉起"
                    self._restarts += 1
                    print(f"[inferproxy] {self._last_error}", flush=True)
                previous = self._last_error
                self._start_locked()
                if not self._alive():
                    raise WorkerError(self._last_error or "推理进程不可用", 503)
                self._last_error = previous

            if image is not None:
                name, shape = self._put_image_locked(image)
                message = {**message, "shm": name, "shape": shape}

            if timeout is None:
                timeout = _request_timeout()
            try:
                self._conn.send(message)
                reply = self._recv_locked(timeout)
            except (_Died, OSError):
                code = None
                if self._process is not None:
                    # 管道先断、进程后退：稍等一下才能拿到真实退出码（如 0xC0000005）
                    try:
                        code = self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
                self._kill_locked()
                tripped = self._record_crash_locked()
                if self.on_crash is not None:
                    # 通知池子累计崩溃次数（跨进程熔断用）。回调只拿池子自己的锁，
                    # 不会与本进程的锁形成循环等待。
                    self.on_crash()
                where = "；".join(x for x in (f"模型：{model}" if model else "", f"图片：{source}" if source else "") if x)
                self._last_error = f"推理进程异常退出（退出码 {_format_exit(code)}）" + (f"，{where}" if where else "")
                if tripped:
                    raise WorkerError(f"{self._last_error}。{self._status['detail']}", 503) from None
                self._restart_in_background()
                raise WorkerError(f"{self._last_error}，已自动重启，请重试", 503) from None
            except _TimedOut:
                self._kill_locked()
                self._last_error = f"推理超过 {timeout:.0f} 秒未返回，已结束并重启推理进程"
                self._restart_in_background()
                raise WorkerError(self._last_error, 504) from None

            if reply.get("ok"):
                return _unpack_blobs(reply).get("result")
            kind = reply.get("kind")
            text = reply.get("message", "")
            if kind == "infer":
                raise infer.InferError(text)
            if kind == "ocr":
                raise ocr_infer.OcrError(text, int(reply.get("status", 400)))
            raise RuntimeError(text)

    # ---------------- 查询 ----------------

    def refresh_status(self) -> None:
        """只在不忙时向子进程要一次最新状态；忙时沿用缓存，不让诊断接口被推理卡住。"""
        if not self._lock.acquire(timeout=0.2):
            return
        try:
            if self._alive():
                self._conn.send({"op": "status"})
                self._status = self._recv_locked(10.0)["result"]
        except (_Died, _TimedOut, OSError):
            pass
        finally:
            self._lock.release()

    def status_dict(self) -> dict:
        return self._status

    def cache(self) -> dict:
        if not self._lock.acquire(timeout=0.5):
            return {"busy": True}
        try:
            if not self._alive():
                return {"running": False}
            self._conn.send({"op": "cache"})
            return self._recv_locked(10.0)["result"]
        except (_Died, _TimedOut, OSError):
            return {"running": False}
        finally:
            self._lock.release()

    def state(self) -> dict:
        if self._tripped_at is not None:
            name = "tripped"
        elif self._alive():
            name = "running"
        else:
            name = "stopped"
        return {
            "enabled": True,
            "state": name,
            "pid": self._process.pid if self._alive() else None,
            "restarts": self._restarts,
            "recent_crashes": len(self._crashes),
            "last_error": self._last_error,
        }


class _Pool:
    """推理进程池。

    每个进程有**完全独立**的 DLL 句柄缓存与共享内存，所以多个请求可以真正同时
    进 GPU——这正是单个 DLL 句柄做不到的事（同一个句柄并发推理会让显存写乱）。

    设计要点：

    - **分发**：队列里放的是「空闲进程令牌」，取一个、用一次、放回去。
      等不到令牌就报 503，绝不把请求无限挂着——用户宁可看到「忙」也不要看转圈。
    - **熔断两层**：单个进程 60 秒崩 3 次仍按老规则只停它自己（0 号进程的行为与
      1.30.x 完全一致，测试断言不受影响）；全池 60 秒累计 6 次才全池熔断，
      因为那说明是同一张坏图在把每个进程挨个打崩。
    - **0 号是主进程**：状态查询、缓存统计、手动重启、OCR 全固定走它，
      这样单进程时代的对外字段和测试一个字都不用改。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()        # 护住 _workers / _evicted 结构
        self._crash_lock = threading.Lock()  # 护住跨进程熔断计数（始终是最内层锁）
        self._workers: list[_Worker] = []
        self._evicted: list[_Worker] = []
        self._free: queue.Queue[_Worker] = queue.Queue()
        self._crashes: collections.deque[float] = collections.deque()
        self._tripped_at: float | None = None
        self._last_error = ""
        self._started = False
        #: 非空表示「显存已让给训练」。此期间禁止拉起推理进程，哪怕有推理请求进来。
        self._gpu_hold = ""
        self._primary = _Worker(0)
        # 必须走 _attach：它负责挂崩溃回调。漏挂 0 号会让跨进程熔断永远凑不满次数。
        self._attach(self._primary)

    # ---------------- 结构 ----------------

    @property
    def primary(self) -> _Worker:
        return self._primary

    @property
    def workers(self) -> list[_Worker]:
        with self._lock:
            return list(self._workers)

    def __len__(self) -> int:
        return len(self.workers)

    def _attach(self, worker: _Worker) -> None:
        worker.on_crash = self._record_crash
        worker.may_start = self._allow_worker_start
        self._workers.append(worker)

    def _allow_worker_start(self) -> bool:
        """自动实验占着显存时，任何路径（请求、崩溃重启、手动 startup）都不许拉起进程。"""
        return not self._gpu_hold

    def gpu_hold(self) -> str:
        return self._gpu_hold

    # ---------------- 跨进程熔断 ----------------

    def _record_crash(self) -> None:
        """由某个进程在「请求把它打崩」时回调。只拿 _crash_lock，不会死锁。"""
        with self._crash_lock:
            now = time.monotonic()
            self._crashes.append(now)
            while self._crashes and now - self._crashes[0] > CRASH_WINDOW_SECONDS:
                self._crashes.popleft()
            if self._tripped_at is None and len(self._crashes) >= POOL_CRASH_LIMIT:
                self._tripped_at = now
                self._last_error = (
                    f"{CRASH_WINDOW_SECONDS:.0f} 秒内累计崩溃 {POOL_CRASH_LIMIT} 次"
                    f"（{len(self._workers)} 个推理进程合计），已暂停自动重启"
                )
                print(f"[inferproxy] {self._last_error}", flush=True)

    def trip_reason(self) -> str:
        """当前是否处于全池熔断。冷却期结束后自动解除并返回空串。"""
        with self._crash_lock:
            if self._tripped_at is None:
                return ""
            if time.monotonic() - self._tripped_at >= TRIP_COOLDOWN_SECONDS:
                self._tripped_at = None
                self._crashes.clear()
                return ""
            return self._last_error or "推理进程频繁崩溃，已暂停服务"

    def _check_trip(self) -> None:
        reason = self.trip_reason()
        if reason:
            raise WorkerError(reason, 503)

    # ---------------- 生命周期 ----------------

    def startup(self) -> None:
        if self._gpu_hold:
            return
        with self._lock:
            if self._started:
                return
            target = worker_count()
            while len(self._workers) < target:
                self._attach(_Worker(len(self._workers)))
            self._evicted = []
            self._free = queue.Queue()
            for worker in self._workers:
                self._free.put(worker)
            rest = self._workers[1:]
            self._started = True
        if not isolation_enabled():
            return
        # 0 号同步起：/api/capabilities 一上线就要有真实状态可报。
        self._primary.startup()
        # 其余后台起，别让启动多等 N 次 DLL 加载。若首个请求先到，它会在该进程
        # 自己的锁上等加载完成——结果不变，只是那一次请求慢一点。
        for worker in rest:
            threading.Thread(target=self._start_one, args=(worker,),
                             name=f"infer-worker-start-{worker.index + 1}", daemon=True).start()

    @staticmethod
    def _start_one(worker: _Worker) -> None:
        try:
            worker.startup()
        except Exception as exc:  # noqa: BLE001 - 一个起不来不影响别人
            print(f"[inferproxy] {worker._tag}推理进程启动异常：{type(exc).__name__}: {exc}", flush=True)

    def shutdown(self) -> None:
        with self._lock:
            self._started = False
            workers = list(self._workers)
            self._workers = [self._primary]
            self._evicted = []
            self._free = queue.Queue()
        for worker in workers:
            try:
                worker.shutdown()
            except Exception as exc:  # noqa: BLE001 - 关闭路径尽量别抛
                print(f"[inferproxy] {worker._tag}停止异常：{type(exc).__name__}: {exc}", flush=True)
        with self._crash_lock:
            self._crashes.clear()
            self._tripped_at = None
            self._last_error = ""

    def yield_gpu(self, reason: str) -> dict:
        """停掉全部推理进程，并在解除之前拒绝重新拉起。

        自动实验的训练和推理进程抢的是同一块 8GB。推理进程是懒加载的：
        页面上点一次推理，它就把模型留在显存里不放。训练一开始再去抢，
        就会差几十 MB 然后 OOM，接着 Ultralytics 把 batch 偷偷减半。

        这里主动退出，而不是等它自己崩。解除时**不**马上拉起——
        下次真正的推理请求才会加载（见 ``release_gpu_hold``）。
        """
        reason = (reason or "训练").strip() or "训练"
        with self._lock:
            self._gpu_hold = reason
            self._started = False
            workers = list(self._workers) + list(self._evicted)
            self._evicted = []
            self._free = queue.Queue()
        stopped = 0
        for worker in workers:
            try:
                alive = worker._alive()
            except Exception:  # noqa: BLE001
                alive = False
            if alive:
                stopped += 1
            try:
                worker.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"[inferproxy] {worker._tag}让出显存时停止异常：{type(exc).__name__}: {exc}", flush=True)
        print(f"[inferproxy] 已退出推理进程（{stopped} 个），显存让给：{reason}", flush=True)
        return {
            "ok": True,
            "held": True,
            "mode": "worker",
            "reason": reason,
            "stopped": stopped,
        }

    def release_gpu_hold(self) -> dict:
        """允许下次推理重新加载。此刻不拉起进程，避免训练刚结束又把显存占回去。"""
        with self._lock:
            self._gpu_hold = ""
        return {
            "ok": True,
            "held": False,
            "mode": "worker",
            "reloaded": False,
        }

    def restart(self) -> None:
        if self._gpu_hold:
            print(f"[inferproxy] 训练占用显存（{self._gpu_hold}），忽略重启推理进程的请求", flush=True)
            return
        with self._crash_lock:
            self._crashes.clear()
            self._tripped_at = None
            self._last_error = ""
        with self._lock:
            workers = list(self._workers)
            evicted = list(self._evicted)
            self._evicted = []
        for worker in workers:
            try:
                worker.restart()
            except Exception as exc:  # noqa: BLE001
                print(f"[inferproxy] {worker._tag}重启异常：{type(exc).__name__}: {exc}", flush=True)
        # 被熔断踢出队列的进程现在恢复健康，重新放回队列。只放回「被踢出」的那几个：
        # 仍在队列里的进程再投一次令牌会让两个线程同时拿它推理，那是最坏的情况。
        for worker in evicted:
            self._free.put(worker)

    # ---------------- 分发 ----------------

    def call(self, message: dict, image: np.ndarray | None = None, *, model: str = "",
             source: str = "", timeout: float | None = None):
        if self._gpu_hold:
            raise WorkerError(
                f"推理进程已退出，把显存让给训练（{self._gpu_hold}）。"
                "训练结束后，下次推理会自动重新加载。",
                503,
            )
        if not self._started:
            # 与单进程时代一致：shutdown 之后再调用会自动拉起，而不是报错。
            # 上面的 gpu_hold 是唯一的例外：训练还没结束时不许拉起。
            self.startup()
        self._check_trip()
        with self._lock:
            all_tripped = bool(self._workers) and all(w.tripped() for w in self._workers)
        if all_tripped:
            raise WorkerError(self._last_error or "所有推理进程均已熔断，请手动重启推理服务", 503)

        deadline = time.monotonic() + queue_timeout()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                worker = self._free.get(timeout=remaining)
            except queue.Empty:
                break
            if worker.tripped():
                # 单个进程熔断：把它移出分发队列，等手动重启再放回来。
                # 不放回去是因为它每次被拿到都只会立刻报错，白占一次分发机会。
                with self._lock:
                    self._evicted.append(worker)
                print(f"[inferproxy] {worker._tag}已熔断，暂时退出进程池", flush=True)
                continue
            try:
                return worker.call(message, image, model=model, source=source, timeout=timeout)
            finally:
                self._free.put(worker)

        raise WorkerError(
            f"推理进程都在忙（共 {len(self.workers)} 个，已等 {queue_timeout():.0f} 秒），请稍后重试",
            503,
        )

    def state(self) -> dict:
        workers = self.workers
        data = self._primary.state()
        data["pool_size"] = len(workers)
        data["workers"] = [dict(w.state(), index=w.index) for w in workers]
        reason = self.trip_reason()
        data["pool_tripped"] = bool(reason)
        data["pool_error"] = reason
        hold = self.gpu_hold()
        data["gpu_hold"] = hold
        if hold:
            data["state"] = "yielded"
            data["last_error"] = f"已退出并把显存让给训练（{hold}），下次推理时重新加载"
        return data


_pool = _Pool()
#: 主进程（0 号）。状态、缓存、手动重启、OCR 固定走它——单进程时代的外部契约不变。
_worker = _pool.primary
#: 直通模式（不启子进程）时的让出标记。隔离模式的标记在池子里。
_DIRECT_HOLD = ""


def _yield_message(reason: str) -> str:
    return (
        f"推理进程已退出，把显存让给训练（{reason}）。"
        "训练结束后，下次推理会自动重新加载。"
    )


def _refuse_if_yielded() -> None:
    reason = _DIRECT_HOLD if not isolation_enabled() else _pool.gpu_hold()
    if reason:
        raise WorkerError(_yield_message(reason), 503)


# ---------------------------------------------------------------------------
# 对外接口（与 infer / ocr_infer 同名）
# ---------------------------------------------------------------------------

def startup() -> None:
    """服务启动时调用：隔离模式拉起工作进程（可能是多个），直通模式按原样加载 DLL。"""
    if _DIRECT_HOLD or (isolation_enabled() and _pool.gpu_hold()):
        return
    if isolation_enabled():
        _pool.startup()
    else:
        infer.load_libraries()


def shutdown() -> None:
    if isolation_enabled():
        _pool.shutdown()
    else:
        infer.release_handles()
        ocr_infer.release_handles()


def status() -> infer.BackendStatus:
    if not isolation_enabled():
        return infer.status()
    # 全池熔断时所有进程都不可信，直接把整个后端报成不可用，
    # 路由层原有的「DLL 不可用 → onnxruntime 降级」就会接管。
    reason = _pool.trip_reason()
    if reason:
        return _status_from_dict(_unavailable_status(reason))
    return _status_from_dict(_worker.status_dict())


def ocr_status() -> dict:
    if not isolation_enabled():
        return ocr_infer.status()
    return dict(_worker.status_dict().get("ocr") or _unavailable_status("")["ocr"])


def run_yolo(engine_path: str, task: str, image: np.ndarray, conf: float = 0.25, nms: float = 0.45,
             want_masks: bool = False, source: str = "") -> dict:
    _refuse_if_yielded()
    if not isolation_enabled():
        infer.status()  # 未加载时先加载，与 1.27.0 启动时加载的效果一致
        return infer.run_yolo(engine_path, task, image, conf=conf, nms=nms, want_masks=want_masks)
    kwargs = {"engine_path": engine_path, "task": task, "conf": conf, "nms": nms, "want_masks": want_masks}
    return _pool.call({"op": "yolo", "kwargs": kwargs}, image, model=engine_path, source=source)


def run_anomalib(engine_path: str, metadata_path: str, image: np.ndarray, model_type: str = "Auto",
                 want_overlay: bool = True, image_threshold: float = -1.0, pixel_threshold: float = -1.0,
                 colormap: str | None = None, want_heatmap: bool = True, source: str = "") -> dict:
    kwargs = {
        "engine_path": engine_path,
        "metadata_path": metadata_path,
        "model_type": model_type,
        "want_overlay": want_overlay,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "colormap": colormap,
        "want_heatmap": want_heatmap,
    }
    _refuse_if_yielded()
    if not isolation_enabled():
        infer.status()
        return infer.run_anomalib(image=image, **kwargs)
    return _pool.call({"op": "anomalib", "kwargs": kwargs}, image, model=engine_path, source=source)


def run_classify(onnx_path: str, image: np.ndarray, source: str = "") -> dict:
    """分类走与检测/异常检测同一条隔离通道。

    分类模型很小、几乎不会崩，但没有隔离时它一旦触发 ORT 的访问违规，
    整个 Web 进程会跟着退出——而分类恰恰是"随手试一下"的操作，
    用户不会预期它把服务带走。
    """
    _refuse_if_yielded()
    if not isolation_enabled():
        infer.status()
        return infer.run_classify(onnx_path, image)
    return _pool.call({"op": "classify", "kwargs": {"onnx_path": onnx_path}},
                      image, model=onnx_path, source=source)


def run_ocr(image: np.ndarray, options: dict, source: str = "") -> dict:
    """OCR 固定在 0 号进程。

    为什么不让它进池：`paddle_inference.dll` 首次加载后常驻约 1.7GB，
    每个进程各来一份就是 1.7GB × N。OCR 本身已是 Paddle 内部多线程，堆进程
    收益很小、代价极大。固定 0 号还有个好处：它加载期间其余进程照常做检测，
    用户不会再看到「点了 OCR 之后检测也卡住」。
    """
    _refuse_if_yielded()
    if not isolation_enabled():
        return ocr_infer.run(image, options)
    # 首次加载 Paddle（1.7GB 的 paddle_inference.dll）比普通推理慢得多
    timeout = None if _worker._ocr_warm else max(_request_timeout(), OCR_FIRST_LOAD_TIMEOUT_SECONDS)
    result = _worker.call({"op": "ocr", "options": options}, image,
                          model=str(options.get("det_dir", "")), source=source, timeout=timeout)
    _worker._ocr_warm = True
    return result


def release_handles() -> None:
    """释放所有进程的缓存句柄（服务退出路径）。逐个容错，一个挂了不影响别的。"""
    if not isolation_enabled():
        infer.release_handles()
        ocr_infer.release_handles()
        return
    for worker in _pool.workers:
        try:
            worker.call({"op": "release"})
        except (WorkerError, OSError):
            pass


def cache_stats() -> dict:
    if not isolation_enabled():
        return {**infer.cache_stats(), "ocr": ocr_infer.cache_stats()}
    return _worker.cache()


def worker_state() -> dict:
    if not isolation_enabled():
        return {"enabled": False, "state": "disabled"}
    # 顶层字段仍然描述 0 号进程（与单进程时代逐字段一致），
    # 另外多给 workers / pool_* 供诊断页展示多进程细节。
    return _pool.state()


def yield_gpu(reason: str = "训练") -> dict:
    """停掉推理进程并把显存让给训练。解除前不会自动重载。"""
    global _DIRECT_HOLD
    reason = (reason or "训练").strip() or "训练"
    if not isolation_enabled():
        try:
            infer.release_handles()
            ocr_infer.release_handles()
        except Exception as exc:  # noqa: BLE001
            print(f"[inferproxy] 直通模式释放句柄失败：{type(exc).__name__}: {exc}", flush=True)
        _DIRECT_HOLD = reason
        return {"ok": True, "held": True, "mode": "direct", "reason": reason, "stopped": 0}
    return _pool.yield_gpu(reason)


def release_gpu_hold() -> dict:
    """训练结束：允许下次推理按需重载，此刻不拉起进程。"""
    global _DIRECT_HOLD
    _DIRECT_HOLD = ""
    if not isolation_enabled():
        return {"ok": True, "held": False, "mode": "direct", "reloaded": False}
    return _pool.release_gpu_hold()


def restart_worker() -> dict:
    if not isolation_enabled():
        return worker_state()
    if _pool.gpu_hold():
        state = worker_state()
        state["refused"] = True
        return state
    _pool.restart()
    return worker_state()


def _test_call(op: str, **fields):
    """仅测试使用：需要子进程带 TRTFORGE_TEST_HOOKS=1 启动。固定走 0 号进程。"""
    timeout = fields.pop("timeout", None)
    return _worker.call({"op": op, **fields}, timeout=timeout)


def _test_pool_call(op: str, **fields):
    """仅测试使用：走进程池分发，用来验证「请求确实落在不同进程上」。"""
    timeout = fields.pop("timeout", None)
    return _pool.call({"op": op, **fields}, timeout=timeout)
