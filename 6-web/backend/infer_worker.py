# -*- coding: utf-8 -*-
"""推理工作进程：所有推理 DLL 只在这个进程里加载。

为什么单独一个进程
------------------
C++ 动态库里一次访问违例，Windows 会直接结束整个进程。DLL 若加载在 Web 进程里，
一张坏图就能让网页服务消失。放到这里以后，崩的只是本进程，Web 侧的
`inferproxy` 会把当前请求报错并重新拉起一个新的工作进程。

进程协议
--------
启动：``python infer_worker.py``，或安装包里 ``TRTForgeWeb.exe --tool infer_worker``。
      管道地址与认证密钥由环境变量 ``TRTFORGE_INFER_PIPE`` / ``TRTFORGE_INFER_KEY`` 传入。
连接：本进程作为 Listener 等待 Web 进程连接（只接受一个连接）。
消息：Web 发 ``{"op": ..., ...}``，本进程回 ``{"ok": True, "result": ...}`` 或
      ``{"ok": False, "kind": 错误类型, "message": ..., "status": HTTP 码}``。
图像：放在共享内存里，消息只带共享内存名与 shape，避免大图在管道里复制。
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import Listener
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

import infer  # noqa: E402
import ocr_infer  # noqa: E402

PIPE_ENV = "TRTFORGE_INFER_PIPE"
KEY_ENV = "TRTFORGE_INFER_KEY"
TEST_HOOKS_ENV = "TRTFORGE_TEST_HOOKS"


class _SharedImage:
    """按名字附着 Web 进程创建的共享内存；名字变了（扩容）才重新附着。"""

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None

    def view(self, name: str, shape: list[int]) -> np.ndarray:
        if self._shm is None or self._shm.name != name:
            self.close()
            # track=False：共享内存归 Web 进程管，本进程退出时不能把它注销掉
            try:
                self._shm = shared_memory.SharedMemory(name=name, track=False)
            except TypeError:  # Python < 3.13 没有 track 参数
                self._shm = shared_memory.SharedMemory(name=name)
        size = int(np.prod(shape))
        # 不拷贝：Web 侧持有请求锁直到收到回复，推理期间这块内存不会被改写
        return np.ndarray(tuple(shape), dtype=np.uint8, buffer=self._shm.buf[:size])

    def close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except BufferError:
                # 还有数组引用着旧块（极少见）：放弃关闭，句柄随进程退出释放
                pass
            self._shm = None


def _backend_status() -> dict:
    status = infer.status()
    return {
        "yolo_ok": status.yolo_ok,
        "anomalib_ok": status.anomalib_ok,
        "dll_dirs": status.dll_dirs,
        "detail": status.detail,
        "onnx_ok": status.onnx_ok,
        "onnx_detail": status.onnx_detail,
        "onnx_provider": status.onnx_provider,
        "ocr": ocr_infer.status(),
    }


def _handle(message: dict, images: _SharedImage) -> object:
    op = message.get("op")
    if op == "status":
        return _backend_status()
    if op == "yolo":
        image = images.view(message["shm"], message["shape"])
        return infer.run_yolo(image=image, **message["kwargs"])
    if op == "anomalib":
        image = images.view(message["shm"], message["shape"])
        return infer.run_anomalib(image=image, **message["kwargs"])
    if op == "ocr":
        image = images.view(message["shm"], message["shape"])
        return ocr_infer.run(image, message["options"])
    if op == "cache":
        return {**infer.cache_stats(), "ocr": ocr_infer.cache_stats()}
    if op == "release":
        infer.release_handles()
        ocr_infer.release_handles()
        return None
    if os.environ.get(TEST_HOOKS_ENV) == "1":
        if op == "crash":
            os.abort()
        if op == "sleep":
            time.sleep(float(message.get("seconds", 1)))
            return None
        if op == "pid":
            return os.getpid()
    raise ValueError(f"未知操作：{op}")


def _error_reply(exc: BaseException) -> dict:
    if isinstance(exc, ocr_infer.OcrError):
        return {"ok": False, "kind": "ocr", "message": str(exc), "status": exc.status}
    if isinstance(exc, infer.InferError):
        return {"ok": False, "kind": "infer", "message": str(exc), "status": 400}
    return {
        "ok": False,
        "kind": "internal",
        "message": f"{type(exc).__name__}: {exc}",
        "status": 500,
        "trace": traceback.format_exc(limit=8),
    }


def main(argv: list[str] | None = None) -> int:
    address = os.environ.get(PIPE_ENV, "")
    key = os.environ.get(KEY_ENV, "")
    if not address or not key:
        print(f"缺少 {PIPE_ENV} / {KEY_ENV}，推理工作进程只能由 Web 服务启动", file=sys.stderr)
        return 2

    infer.load_libraries()
    images = _SharedImage()
    with Listener(address, family="AF_PIPE", authkey=bytes.fromhex(key)) as listener:
        with listener.accept() as conn:
            conn.send({"ok": True, "result": _backend_status(), "pid": os.getpid()})
            while True:
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    break
                if message.get("op") == "shutdown":
                    conn.send({"ok": True, "result": None})
                    break
                try:
                    reply = {"ok": True, "result": _handle(message, images)}
                except Exception as exc:  # noqa: BLE001 - 原样回传给 Web 侧
                    reply = _error_reply(exc)
                conn.send(reply)

    images.close()
    infer.release_handles()
    ocr_infer.release_handles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
