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

**v1.42.0 起支持多个持有者**，按 ``pid + kind + label`` 区分：

```json
{"holders": {
  "web|yolo-train|run_abc": {"owner": "web", "pid": 12345, "kind": "yolo-train",
                             "label": "run_abc", "since": "2026-11-10T13:20:00+08:00"},
  "web|infer-video|run_jkl": {"owner": "web", "pid": 12345, "kind": "infer-video",
                              "label": "run_jkl", "since": "2026-11-10T13:21:00+08:00"}
}}
```

为什么不干脆「文件里只有一把锁」：同一个 Web 进程里可能有两个任务
（``infer-batch`` 与 ``yolo-train`` 之外，还包括「停止一个、另一个还在跑」的场景）。
单持有者版本下，后到的 ``release()`` 只比 pid 就把整个文件删掉，
另一个还在跑的任务等于失去了互斥保护——正是这个文件本来要防的事。
``release`` 现在只删自己那条，删空了才删文件。

**旧格式仍可读**（``{"owner": ..., "pid": ...}``，单持有者的扁平结构）。
autoexp 与老版本 Web 会写出这种形状，读到就当作一个持有者处理，
下次 ``acquire`` 时自然升级成新结构，不需要手工删文件。

判定规则：

- 文件不存在 → 空闲
- 某条目的 ``pid`` 已死 → 该条目视为**陈旧**，读取时忽略、写入时清掉
- 所有条目都陈旧 → 空闲
- 存在其它进程的存活条目 → 被占用

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


def holder_key(kind: str = "", label: str = "") -> str:
    """一个持有者在文件里的键：``kind|label``。

    为什么用这个组合：调用方本来就靠它区分任务（``kind`` 是任务类型，
    ``label`` 在 Web 端就是 ``run_id``，天然唯一）。同一进程两个任务因此
    各有各的条目，互不覆盖。两者都空时退化成 ``"|"``——
    那是「整进程一把锁」的老语义，仍然可用。
    """
    return f"{kind or ''}|{label or ''}"


def _split_holders(data: Any) -> dict[str, dict[str, Any]]:
    """把文件内容统一成 ``{key: holder}``，兼容旧的单持有者扁平格式。

    单独抽出来是因为「读到的到底是什么形状」有两个来源：
    新版本写 ``holders``，老版本（含 autoexp）写扁平字段。
    所有需要看懂锁的地方都必须走这里，否则迟早有一处漏了兼容。
    """
    if not isinstance(data, dict):
        return {}
    raw = data.get("holders")
    if isinstance(raw, dict):
        out: dict[str, dict[str, Any]] = {}
        for key, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                item["pid"] = int(item.get("pid") or 0)
            except (TypeError, ValueError):
                continue
            out[str(key)] = item
        return out
    # 旧格式：owner/pid/kind/label 平铺在顶层
    try:
        pid = int(data.get("pid") or 0)
    except (TypeError, ValueError):
        return {}
    if pid <= 0:
        return {}
    holder = dict(data)
    holder["pid"] = pid
    return {holder_key(str(holder.get("kind") or ""), str(holder.get("label") or "")): holder}


def holders() -> dict[str, dict[str, Any]]:
    """所有**存活**的持有者，键是 :func:`holder_key`。

    不存在、损坏、全部陈旧时返回 ``{}``（区别于「文件读不了」——
    两者在本模块的语义里都是「当作空闲」，因为坏文件不能永久锁死机器，
    代价是极端情况下漏判一次并发，比让所有人都用不了强）。
    """
    path = lock_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    alive = {
        key: item
        for key, item in _split_holders(data).items()
        if item.get("pid") and pid_alive(int(item["pid"]))
    }
    return alive


def read() -> dict[str, Any] | None:
    """读一个持有者（多持有者时取最早进入的那个）。

    保留这个签名是为了 ``web_state`` / ``autoexp_api`` 的现成调用
    （它们只需要「有没有人占着、是谁」）。要看全部请用 :func:`holders`。
    """
    alive = holders()
    if not alive:
        return None
    return min(alive.values(), key=lambda item: str(item.get("since") or ""))


def foreign_holders(pid: int | None = None) -> dict[str, dict[str, Any]]:
    """**其它进程**的存活持有者（默认排除本进程）。

    为什么不能再用 ``read()`` 做这件事：多持有者之后 ``read()`` 只返回一个，
    它可能恰好是本进程自己那条（例如网页正跑着推理），于是「另一个终端
    在跑 autoexp」就被漏掉了——而这正是这道闸门存在的理由。
    """
    me = os.getpid() if pid is None else pid
    return {
        key: item
        for key, item in holders().items()
        if int(item.get("pid") or 0) != me
    }


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


def describe_all(subjects: dict[str, dict[str, Any]] | None = None) -> str:
    """把全部持有者拼成一句人话（多任务时用）。

    ``release`` 失败、或用户点「谁占着显卡」时，只报一个持有者会让人以为
    另一个任务已经停了。这里按进入时间排序，逐个列出。
    """
    items = list((subjects if subjects is not None else holders()).values())
    if not items:
        return "空闲"
    items.sort(key=lambda item: str(item.get("since") or ""))
    return "；".join(describe(item) for item in items)


def _write(data: dict[str, Any]) -> None:
    """原子写锁文件（临时文件 + ``os.replace``）。"""
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时名带 pid：两个进程同时写时各自的临时文件不会互相覆盖。
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def acquire(owner: str, kind: str = "", label: str = "") -> tuple[bool, str]:
    """申请占用。

    返回 ``(True, "")`` 表示拿到；``(False, 原因)`` 表示被别人占着。

    注意这里**不排队、不等待**：调用方都是「拿不到就立刻拒绝并提示用户」，
    等待只会让界面卡住，而用户需要的是马上知道「谁在占着 GPU」。

    同进程的其它任务（已有条目、pid 相同）不算冲突——它们本来就是
    同一个进程里的两条流水线，本来就该各自记账。旧实现在这里直接
    「放行并覆盖」，把别人的条目冲掉，等于让先跑的那个任务失去保护。
    """
    alive = holders()
    for key, item in alive.items():
        if int(item.get("pid") or 0) != os.getpid():
            return False, describe(item)

    key = holder_key(kind, label)
    # 同 pid 的其它条目保持原样，只写自己这一条。
    # 唯一的例外：本进程已经有「同 kind 但还没写 label」的占位条目。
    # autoexp 的姿势就是先占位（exp_id 还没分配）、拿到 id 后再 acquire 一次补
    # label；不在这里清掉旧键的话，文件里会同时留着两条，看状态时像两个任务。
    if label:
        placeholder = holder_key(kind, "")
        if int(alive.get(placeholder, {}).get("pid") or 0) == os.getpid():
            alive.pop(placeholder, None)
    entry = {
        "owner": owner,
        "pid": os.getpid(),
        "kind": kind,
        "label": label,
        "since": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    alive[key] = entry
    try:
        _write({"holders": alive})
    except OSError as exc:
        # 写不了锁文件不能让训练本身跑不起来（比如只读目录）。
        # 降级为「无锁」并在调用方日志里留一句，属于可接受的退让。
        return True, f"锁文件写入失败（{type(exc).__name__}: {exc}），本次不做跨进程互斥"
    return True, ""


def release(kind: str = "", label: str = "") -> None:
    """释放占用。

    只删**自己那一条**（同一个 pid + 同一个 ``kind|label``）：

    - 文件里的 pid 不是本进程 → 说明已经换人了（例如自己被判定为陈旧后
      另一个进程接手），这时删掉会误伤别人；
    - pid 是本进程但有其它任务 → 那些条目必须留着，否则还在跑的任务
      在下一个进程眼里就变成「显卡空闲」。

    删空了才删文件。读写都按 :func:`_split_holders` 兼容旧格式。
    """
    path = lock_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    alive = _split_holders(data)
    key = holder_key(kind, label)
    if kind or label:
        # 有明确标识：只删自己那一条，且必须确实是本进程写的
        item = alive.get(key)
        mine = [key] if item is not None and int(item.get("pid") or 0) == os.getpid() else []
    else:
        # 不带任何标识：释放本进程的全部条目（老调用方与 Hold 的默认行为）
        mine = [k for k, v in alive.items() if int(v.get("pid") or 0) == os.getpid()]
    if not mine:
        return
    for k in mine:
        alive.pop(k, None)
    try:
        if alive:
            _write({"holders": alive})
        else:
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
            release(self.kind, self.label)
        return False


def self_check() -> int:
    """命令行自检：``python gpulock.py status``。"""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(f"锁文件：{lock_path()}")
        print(f"状态：{describe_all()}")
        return 0
    print(__doc__ or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
