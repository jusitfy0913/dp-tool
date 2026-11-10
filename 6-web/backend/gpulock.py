"""跨进程 GPU 占用锁。

## 为什么需要这个文件

Web 后端本来只有一道闸门 ``_reject_if_busy()``：它查自己内存里的任务表，
判断「本进程是否已有独占任务在跑」。这对「在同一个网页里连点两次」够用，
但漏掉了两种真实情况：

1. 命令行在跑 ``autoexp.py run``（自动实验循环），网页完全不知道；
2. 本机同时起了两个 Web 实例（比如 8060 和 8094 各一套），
   各自的 ``manager`` 互不可见。

两种情况都会让两个训练同时抢那 8 GB 显卡，结果通常是双双 OOM；
而 autoexp 还会把 batch 静默减半重试，于是「这次实验」和台账里别的实验
不再是同一件事，横向比较全部失真。

## 协议

一个 JSON 文件，位于 ``<repo>/runtime/gpu_lock.json``（已 gitignore）。
同一时刻最多一个持有者。

```json
{"owner": "autoexp", "pid": 12345, "kind": "autoexp-run",
 "label": "exp_0011 提高分辨率到 960", "since": "2026-11-10T13:20:00+08:00"}
```

判定规则：

- 文件不存在 → 空闲
- 文件存在但 ``pid`` 已死 → 视为**陈旧**，可被覆盖（进程被强杀时会留下这种）
- 文件存在且 ``pid`` 存活 → 被占用

## 为什么不用操作系统的文件锁

Windows 与 POSIX 的语义差别大（``msvcrt.locking`` / ``fcntl.flock``），
而且文件锁在进程被 ``TerminateProcess`` 硬杀时不会自动释放的平台上更难排查。
PID + 存活性探测更直白：拿到文件就能用记事本看懂「谁占着、占多久了」。

**注意**：存活性探测**不能**用 ``os.kill(pid, 0)``。在 POSIX 上它是只读检查，
但在 Windows 上 CPython 会把它变成 ``TerminateProcess(handle, sig)``，
等于把对方进程以退出码 0 杀掉。详见 :func:`pid_alive` 的注释。

## 这个模块的约束

**零第三方依赖**，只 import 标准库。原因有两个：
``autoexp.py`` 会 import 它（不能因此把 FastAPI 那一套拖进来），
而 Web 打包脚本会连整个 backend 目录一起冻结（放在 backend 里才不会漏文件）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

#: 仓库根目录。本文件位于 ``<repo>/6-web/backend/gpulock.py``。
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 锁文件路径。允许用环境变量覆盖，便于测试时指到临时目录。
LOCK_ENV = "TRTFORGE_GPU_LOCK"


def lock_path() -> Path:
    """锁文件位置。

    默认放在仓库根的 ``runtime/``（该目录已在 .gitignore 中，
    属于「运行产物」，不该进版本库）。
    """
    raw = os.environ.get(LOCK_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "runtime" / "gpu_lock.json"


def pid_alive(pid: int) -> bool:
    """探测进程是否还活着。**只探测，不产生任何副作用。**

    这里必须按平台分开写，原因是 Windows 上一个反直觉的事实：
    ``os.kill(pid, 0)`` 在 POSIX 上是「只检查不发信号」的探测，
    但在 CPython 的 Windows 实现里，只有 ``CTRL_C_EVENT`` / ``CTRL_BREAK_EVENT``
    走控制台事件，**其它所有值都直接交给 ``TerminateProcess(handle, sig)``**。
    也就是说 ``os.kill(pid, 0)`` 在 Windows 上会把那个进程以退出码 0 杀掉。
    （实测确认过：本进程调 ``os.kill(os.getpid(), 0)`` 后立刻无声退出。）

    用一把锁的代价去杀别人的进程显然不能接受，所以 Windows 走
    ``OpenProcess`` + ``GetExitCodeProcess``，只读状态不写。

    探测不到（权限不足等）时**保守地当作存活**：宁可多拒绝一次，
    也不要让两个训练同时吃掉显存。
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        return _pid_alive_windows(pid)

    try:
        os.kill(pid, 0)  # POSIX：sig=0 只做存在性与权限检查
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在，但不属于当前用户
    except OSError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Windows 专用的只读存活探测。失败一律返回 True（保守）。"""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    ERROR_INVALID_PARAMETER = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        if err == ERROR_INVALID_PARAMETER:
            return False  # 没有这个 pid
        if err == ERROR_ACCESS_DENIED:
            return True  # 进程存在，只是查不动（例如系统进程）
        return True  # 其它错误：保守当作存活
    try:
        code = wintypes.DWORD(0)
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == STILL_ACTIVE
        return True
    finally:
        kernel32.CloseHandle(handle)


def read() -> dict[str, Any] | None:
    """读当前持有者。返回 None 表示空闲（不存在、损坏或已陈旧）。"""
    path = lock_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        # 写了一半 / 内容坏了：当作没有占用。这里的取舍是「坏文件不能永久锁死机器」，
        # 代价是极端情况下会漏判一次并发，比让所有人都用不了强。
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid = int(data.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if not pid_alive(pid):
        return None  # 陈旧锁
    data["pid"] = pid
    return data


def describe(holder: dict[str, Any] | None) -> str:
    """把持有者翻译成一句人话。"""
    if not holder:
        return "空闲"
    owner = holder.get("owner") or "未知"
    kind = holder.get("kind") or ""
    label = holder.get("label") or ""
    pid = holder.get("pid")
    who = {
        "web": "网页端任务",
        "autoexp": "自动实验循环",
    }.get(owner, f"{owner} 任务")
    detail = f"{who}"
    if kind:
        detail += f"（{kind}）"
    if label:
        detail += f" {label}"
    return f"{detail}，PID {pid}"


def acquire(owner: str, kind: str = "", label: str = "") -> tuple[bool, str]:
    """申请占用。

    返回 ``(True, "")`` 表示拿到；``(False, 原因)`` 表示被别人占着。

    注意这里**不排队、不等待**：调用方都是「拿不到就立刻拒绝并提示用户」，
    等待只会让界面卡住，而用户需要的是马上知道「谁在占着 GPU」。
    """
    current = read()
    if current is not None and int(current.get("pid") or 0) != os.getpid():
        return False, describe(current)

    path = lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "owner": owner,
            "pid": os.getpid(),
            "kind": kind,
            "label": label,
            "since": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        # 先写临时文件再 rename：避免别的进程读到写了一半的 JSON。
        # rename 在同一分区上是原子的。
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # 写不了锁文件不能让训练本身跑不起来（比如只读目录）。
        # 降级为「无锁」并在调用方日志里留一句，属于可接受的退让。
        return True, f"锁文件写入失败（{type(exc).__name__}: {exc}），本次不做跨进程互斥"
    return True, ""


def release() -> None:
    """释放占用。

    只删自己的锁：如果文件里的 pid 不是本进程，说明已经换人了
    （例如自己被判定为陈旧后另一个进程接手），这时删掉会误伤别人。
    """
    path = lock_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    try:
        if int(data.get("pid") or 0) != os.getpid():
            return
    except (TypeError, ValueError):
        return
    try:
        path.unlink()
    except OSError:
        pass


class Hold:
    """``with Hold("web", "yolo-train", run_id):`` 形式的上下文管理器。

    刻意不用 ``contextlib.contextmanager``：这个文件要能被冻结进 exe，
    少引一个模块少一份风险，而且显式类比生成器更容易读懂。
    """

    def __init__(self, owner: str, kind: str = "", label: str = "") -> None:
        self.owner = owner
        self.kind = kind
        self.label = label
        self.ok = False
        self.reason = ""
        self.held = False

    def __enter__(self) -> "Hold":
        self.ok, self.reason = acquire(self.owner, self.kind, self.label)
        self.held = self.ok
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.held:
            release()
        return False


def self_check() -> int:
    """命令行自检：``python gpulock.py status``。"""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        holder = read()
        print(f"锁文件：{lock_path()}")
        print(f"状态：{describe(holder)}")
        return 0
    print(__doc__ or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
