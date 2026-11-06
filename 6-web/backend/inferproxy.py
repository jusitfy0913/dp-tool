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
START_TIMEOUT_SECONDS = 90.0
OCR_FIRST_LOAD_TIMEOUT_SECONDS = 180.0
CRASH_WINDOW_SECONDS = 60.0
CRASH_LIMIT = 3
TRIP_COOLDOWN_SECONDS = 300.0
SHUTDOWN_GRACE_SECONDS = 3.0


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
    def __init__(self) -> None:
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
        log = open(config.LOG_DIR / "infer_worker.log", "ab")  # noqa: SIM115 - 交给子进程持有
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
                    f"推理进程启动失败（退出码 {_format_exit(self._process.returncode)}），"
                    f"详见 {config.LOG_DIR / 'infer_worker.log'}"
                )
            try:
                self._conn = Client(address, family="AF_PIPE", authkey=bytes.fromhex(key))
                break
            except (FileNotFoundError, OSError):
                if time.monotonic() > deadline:
                    self._kill_locked()
                    raise WorkerError("推理进程启动超时") from None
                time.sleep(0.1)

        hello = self._recv_locked(max(1.0, deadline - time.monotonic()))
        self._status = hello["result"]
        self._ocr_warm = False
        self._last_error = ""

    def _start_locked(self) -> None:
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
                    self._last_error = f"推理进程在空闲时退出（退出码 {_format_exit(code)}），已重新拉起"
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
                return reply.get("result")
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


_worker = _Worker()


# ---------------------------------------------------------------------------
# 对外接口（与 infer / ocr_infer 同名）
# ---------------------------------------------------------------------------

def startup() -> None:
    """服务启动时调用：隔离模式拉起工作进程，直通模式按原样加载 DLL。"""
    if isolation_enabled():
        _worker.startup()
    else:
        infer.load_libraries()


def shutdown() -> None:
    if isolation_enabled():
        _worker.shutdown()
    else:
        infer.release_handles()
        ocr_infer.release_handles()


def status() -> infer.BackendStatus:
    if not isolation_enabled():
        return infer.status()
    return _status_from_dict(_worker.status_dict())


def ocr_status() -> dict:
    if not isolation_enabled():
        return ocr_infer.status()
    return dict(_worker.status_dict().get("ocr") or _unavailable_status("")["ocr"])


def run_yolo(engine_path: str, task: str, image: np.ndarray, conf: float = 0.25, nms: float = 0.45,
             want_masks: bool = False, source: str = "") -> dict:
    if not isolation_enabled():
        infer.status()  # 未加载时先加载，与 1.27.0 启动时加载的效果一致
        return infer.run_yolo(engine_path, task, image, conf=conf, nms=nms, want_masks=want_masks)
    kwargs = {"engine_path": engine_path, "task": task, "conf": conf, "nms": nms, "want_masks": want_masks}
    return _worker.call({"op": "yolo", "kwargs": kwargs}, image, model=engine_path, source=source)


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
    if not isolation_enabled():
        infer.status()
        return infer.run_anomalib(image=image, **kwargs)
    return _worker.call({"op": "anomalib", "kwargs": kwargs}, image, model=engine_path, source=source)


def run_ocr(image: np.ndarray, options: dict, source: str = "") -> dict:
    if not isolation_enabled():
        return ocr_infer.run(image, options)
    # 首次加载 Paddle（1.7GB 的 paddle_inference.dll）比普通推理慢得多
    timeout = None if _worker._ocr_warm else max(_request_timeout(), OCR_FIRST_LOAD_TIMEOUT_SECONDS)
    try:
        result = _worker.call({"op": "ocr", "options": options}, image,
                              model=str(options.get("det_dir", "")), source=source, timeout=timeout)
        _worker._ocr_warm = True
        return result
    finally:
        _worker.refresh_status()


def release_handles() -> None:
    if not isolation_enabled():
        infer.release_handles()
        ocr_infer.release_handles()
        return
    try:
        _worker.call({"op": "release"})
    except WorkerError:
        pass


def cache_stats() -> dict:
    if not isolation_enabled():
        return {**infer.cache_stats(), "ocr": ocr_infer.cache_stats()}
    return _worker.cache()


def worker_state() -> dict:
    if not isolation_enabled():
        return {"enabled": False, "state": "disabled"}
    return _worker.state()


def restart_worker() -> dict:
    if not isolation_enabled():
        return worker_state()
    return _worker.restart()


def _test_call(op: str, **fields):
    """仅测试使用：需要子进程带 TRTFORGE_TEST_HOOKS=1 启动。"""
    timeout = fields.pop("timeout", None)
    return _worker.call({"op": op, **fields}, timeout=timeout)
