# -*- coding: utf-8 -*-
"""服务端原生文件对话框。

为什么需要这个模块
------------------
浏览器有安全沙箱，网页**永远无法**弹 Windows 的系统文件框——这不是没实现，
是浏览器不允许。所以路径选择要么让用户手打，要么由后端"在它自己那台机器的
桌面上"弹框。本服务是前台脚本起的（``scripts/run_web_studio.ps1``），和浏览器
在同一个桌面会话，因此后端弹的框用户看得见。

适用范围与假设（不满足就如实降级，不假装能用）
- 仅 Windows；``tkinter`` 可用；
- 仅本机访问（127.0.0.1 / ::1）。远程用户点按钮时，框会弹在服务端屏幕上，
  等于没弹——调用方必须先判断来源，否则会变成一个"点了没反应"的鬼按钮。

实现要点
--------
1. **单独进程**：tkinter 的事件循环不能塞进 uvicorn 的线程池里。开发环境用
   ``pythonw.exe -c`` 跑下面的脚本；安装包里没有 Python，改由
   ``TRTForgeWeb.exe --tool native_dialog`` 跑同一段脚本（见 :func:`dialog_command`）。
2. **传参走文件**：中文、空格、引号在命令行里都是坑，payload 写临时 JSON，
   结果也写临时 JSON，子进程 stdout 不参与通信。
3. **单例锁**：同时弹两个模态框，用户会分不清哪个属于哪个字段。已有框打开时
   直接报 ``DialogBusy``，由接口转成 409。
4. **记住上次目录**：按字段名记在 ``runtime/dialog_state.json``，下次从那儿开。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

# ``RUNTIME_DIR`` 由 config 推导，这里不强依赖 config，避免循环导入。
_RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
STATE_FILE = _RUNTIME / "dialog_state.json"

# 对话框打开后最多等这么久。用户可能去倒杯水，但也别把线程永久占着。
TIMEOUT_SECONDS = float(os.environ.get("TRTFORGE_DIALOG_TIMEOUT", "300") or 300)
# 子进程起来后这么久还看不到自己的窗口，就认定「没弹出来」，立刻结束。
# 以前会干等 300 秒，网页按钮一直停在「…」。
SHOW_DEADLINE_SECONDS = float(os.environ.get("TRTFORGE_DIALOG_SHOW_DEADLINE", "12") or 12)

KINDS = ("file", "files", "dir", "save")

# 子进程脚本：只做一件事——按 payload 弹框、把结果写进 result 文件。
# 之所以写成字符串常量而不是独立 .py 文件：独立文件一旦被误删，
# 整条选择路径就会静默失效；内联脚本跟着模块走，少一个失败点。
_SCRIPT = r"""
import json, os, sys, threading, time, tkinter as tk
from tkinter import filedialog

payload = json.load(open(sys.argv[1], encoding="utf-8"))
result_path = sys.argv[2]
kind = payload.get("kind") or "file"
cancel_path = str(payload.get("cancel") or "")
show_deadline = float(payload.get("show_deadline") or 12)
stop = {"v": False}
shown = {"ok": False}

def write_result(paths, reason=""):
    body = {"paths": paths}
    if reason:
        body["reason"] = reason
    try:
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(body, handle, ensure_ascii=False)
    except Exception:
        pass

def bail(reason):
    write_result([], reason)
    os._exit(0)

def own_windows():
    # 只认本进程自己的可见窗口。隐藏的根窗口不算，避免「根窗口在、选择框没有」。
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []
    user32 = ctypes.windll.user32
    pid = os.getpid()
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        got = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(got))
        if got.value == pid:
            found.append(int(hwnd))
        return True

    user32.EnumWindows(visit, 0)
    return found

def lift(hwnd):
    # 浏览器经常占着前台。不把输入队列接上的话，Windows 会拒绝把选择框拉到最前，
    # 框其实开了，人却看不见，按钮就一直灰着。
    try:
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        fg = user32.GetForegroundWindow()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        my_tid = kernel32.GetCurrentThreadId()
        attached = False
        if fg_tid and fg_tid != my_tid:
            attached = bool(user32.AttachThreadInput(my_tid, fg_tid, True))
        user32.ShowWindow(hwnd, 9)
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
        user32.SetForegroundWindow(hwnd)
        if attached:
            user32.AttachThreadInput(my_tid, fg_tid, False)
    except Exception:
        pass

def guard():
    started = time.time()
    lift_until = 0.0
    while not stop["v"]:
        if cancel_path and os.path.exists(cancel_path):
            bail("cancelled")
        hwnds = own_windows()
        if hwnds:
            if not shown["ok"]:
                shown["ok"] = True
                lift_until = time.time() + 2.0
            if time.time() < lift_until:
                lift(hwnds[0])
        elif not shown["ok"] and time.time() - started > show_deadline:
            bail("dialog-not-shown")
        time.sleep(0.25)

threading.Thread(target=guard, daemon=True).start()

root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
    root.update()
except Exception:
    pass

common = {"title": payload.get("title") or "选择"}
initial = payload.get("initial") or ""
if initial:
    common["initialdir"] = initial

filetypes = []
for item in payload.get("filetypes") or []:
    label, pattern = item[0], item[1]
    filetypes.append((label, pattern))
if not filetypes:
    filetypes = [("所有文件", "*.*")]

chosen = ""
try:
    if kind in ("file", "files"):
        if kind == "file":
            chosen = filedialog.askopenfilename(parent=root, filetypes=filetypes, **common)
        else:
            chosen = filedialog.askopenfilenames(parent=root, filetypes=filetypes, **common)
    elif kind == "save":
        chosen = filedialog.asksaveasfilename(
            parent=root,
            filetypes=filetypes,
            defaultextension=payload.get("defaultextension") or "",
            **common
        )
    else:
        chosen = filedialog.askdirectory(parent=root, mustexist=bool(payload.get("must_exist", True)), **common)
finally:
    stop["v"] = True
    try:
        root.destroy()
    except Exception:
        pass

if isinstance(chosen, (tuple, list)):
    paths = [p for p in chosen if p]
elif chosen:
    paths = [chosen]
else:
    paths = []

write_result(paths)
"""

# 当前正在等的那个取消标记。另一个请求（网页上的「取消」）往这里写文件，
# 子进程看到文件就自己退出。没有进行中的对话框时是 None。
_ACTIVE_CANCEL: dict[str, Path | None] = {"path": None}

_LOCK = threading.Lock()
_TK_OK: bool | None = None


class DialogBusy(RuntimeError):
    """已经有一个对话框开着。"""


def available() -> bool:
    """当前环境能不能弹原生框。

    检查三件事：平台、显式禁用开关、tkinter 是否装好。故意**不**检查
    "有没有桌面会话"——那需要 win32 API，而真正的失败在子进程里会以
    非零退出码暴露，届时前端自动回退，比在这里猜更可靠。
    """
    if os.name != "nt":
        return False
    if os.environ.get("TRTFORGE_DIALOG_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return _tkinter_available()


def _tkinter_available() -> bool:
    global _TK_OK
    if _TK_OK is None:
        try:
            import tkinter  # noqa: F401
            _TK_OK = True
        except Exception:  # noqa: BLE001 - 缺 tkinter 就是不可用
            _TK_OK = False
    return bool(_TK_OK)


def packaged_exe() -> str:
    """安装包里的网页程序。没设置时返回空串，表示仍在开发环境。

    启动脚本会把 ``TRTFORGE_TOOL_EXE`` 指到 ``TRTForgeWeb.exe``。
    双击 exe 而没走启动脚本时，Nuitka 会把 ``sys.frozen`` 设上，这时解释器就是 exe 自己。
    """
    tool_exe = os.environ.get("TRTFORGE_TOOL_EXE", "").strip()
    if tool_exe:
        return tool_exe
    if getattr(sys, "frozen", False):
        return sys.executable
    return ""


def dialog_command(request_file: Path, result_file: Path) -> list[str]:
    """拼出弹窗子进程的命令。最后两项永远是请求文件和结果文件。

    开发环境是 ``python -c <脚本>``。安装包里的 ``TRTForgeWeb.exe`` 不认识 ``-c``，
    以前会把这次启动当成再开一个网页，窗口自然不会出现。
    """
    tool_exe = packaged_exe()
    if tool_exe:
        return [tool_exe, "--tool", "native_dialog", str(request_file), str(result_file)]
    return [interpreter(), "-c", _SCRIPT, str(request_file), str(result_file)]


def serve_dialog(request_path: str, result_path: str) -> int:
    """在当前进程里弹一次框。只给安装包的 ``--tool native_dialog`` 用。

    这里显式 import tkinter，是为了让 Nuitka 把对话框库编进 exe。
    真正的弹窗逻辑仍是上面的 ``_SCRIPT``，开发和安装包不会各写一份。
    """
    import tkinter  # noqa: F401
    from tkinter import filedialog  # noqa: F401

    sys.argv = ["native_dialog", request_path, result_path]
    exec(compile(_SCRIPT, "<native_dialog>", "exec"), {"__name__": "__dialog__"})
    return 0


def interpreter() -> str:
    """挑选运行对话框脚本的解释器。

    ``pythonw.exe`` 不分配控制台，弹框时不会闪一个黑窗口；没有就退回当前
    解释器，并用 ``CREATE_NO_WINDOW`` 抑制控制台。
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe":
        candidate = exe.with_name("pythonw.exe")
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def build_filetypes(patterns: list[str] | None, label: str = "") -> list[list[str]]:
    """把前端给的扩展名列表变成 tkinter 的 ``(描述, 通配)`` 列表。

    输入接受 ``.pt`` / ``*.pt`` / ``pt`` 三种写法，输出统一成 ``*.pt``。
    末尾永远补一个「所有文件」，否则用户的文件后缀不在白名单里时，
    对话框里什么都选不到，只会以为目录是空的。
    """
    cleaned: list[str] = []
    for raw in patterns or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("*."):
            text = text[1:]
        elif not text.startswith("."):
            text = "." + text
        if text not in cleaned:
            cleaned.append(text)

    filetypes: list[list[str]] = []
    if cleaned:
        title = label or "支持的文件"
        filetypes.append([f"{title} ({' '.join('*' + p for p in cleaned)})", " ".join("*" + p for p in cleaned)])
        # 只有一个后缀时，「组合项」和「单项」内容完全一样，
        # 再列一次会让下拉里出现两条一模一样的过滤器。
        if len(cleaned) > 1:
            for pattern in cleaned:
                filetypes.append([f"{pattern[1:].upper()} 文件 (*{pattern})", f"*{pattern}"])
    filetypes.append(["所有文件", "*.*"])
    return filetypes


def _load_state() -> dict:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, STATE_FILE)
    except OSError:
        # 记不住目录不影响本次选择，静默忽略
        pass


def _remember(field: str, path: str) -> None:
    if not field or not path:
        return
    state = _load_state()
    folder = path if Path(path).is_dir() else str(Path(path).parent)
    state[field] = folder
    _save_state(state)


def initial_dir(field: str, initial: str = "") -> str:
    """决定对话框从哪个目录打开。

    优先级：显式传入且存在 > 上次该字段选的目录 > 空（交给系统默认）。
    显式值可能是文件路径（字段里存的是 ``xxx.engine``），要取它的父目录，
    否则 tkinter 会忽略 initialdir 并从"文档"开始，用户又得重钻一遍。
    """
    text = str(initial or "").strip()
    if text:
        candidate = Path(text)
        folder = candidate if candidate.is_dir() else candidate.parent
        if folder.is_dir():
            return str(folder)
    remembered = str(_load_state().get(field) or "").strip()
    if remembered and Path(remembered).is_dir():
        return remembered
    return ""


def request_cancel() -> bool:
    """请当前对话框自己结束。没有正在打开的框时返回 False。"""
    path = _ACTIVE_CANCEL.get("path")
    if path is None:
        return False
    try:
        path.write_text("cancel", encoding="utf-8")
        return True
    except OSError:
        return False


def select(
    kind: str = "file",
    *,
    title: str = "",
    field: str = "",
    initial: str = "",
    patterns: list[str] | None = None,
    label: str = "",
    defaultextension: str = "",
    must_exist: bool = True,
) -> dict:
    """弹一次原生对话框，返回 ``{"paths": [...], "cancelled": bool, ...}``。

    调用方必须自己保证"是本机用户在选"——这个函数不做来源检查。
    用户取消不是错误：返回空 ``paths``，调用方不要改字段、不要报错。
    """
    if kind not in KINDS:
        raise ValueError(f"不支持的选择类型：{kind}")
    if not available():
        raise RuntimeError("当前环境不支持原生对话框")

    # 单例：同时开两个框，用户分不清哪个对应哪个字段。
    if not _LOCK.acquire(blocking=False):
        raise DialogBusy("已经有一个选择框打开了，请先在桌面上处理它")
    try:
        return _run(kind, title=title, field=field, initial=initial,
                    patterns=patterns, label=label,
                    defaultextension=defaultextension, must_exist=must_exist)
    finally:
        _LOCK.release()


def _run(kind: str, **kwargs) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix="tf_dialog_"))
    request_file = temp_dir / "request.json"
    result_file = temp_dir / "result.json"
    cancel_file = temp_dir / "cancel.flag"
    payload = {
        "kind": kind,
        "title": kwargs.get("title") or "选择路径",
        "initial": initial_dir(kwargs.get("field", ""), kwargs.get("initial", "")),
        "filetypes": build_filetypes(kwargs.get("patterns"), kwargs.get("label", "")),
        "defaultextension": kwargs.get("defaultextension") or "",
        "must_exist": kwargs.get("must_exist", True),
        "cancel": str(cancel_file),
        "show_deadline": SHOW_DEADLINE_SECONDS,
    }
    request_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    started = _now_ms()
    _ACTIVE_CANCEL["path"] = cancel_file
    try:
        try:
            completed = subprocess.run(
                dialog_command(request_file, result_file),
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired:
            return {
                "paths": [],
                "cancelled": True,
                "reason": "timeout",
                "elapsed_ms": _now_ms() - started,
            }

        elapsed = _now_ms() - started
        if completed.returncode != 0:
            stderr = (completed.stderr or b"").decode("utf-8", "replace")[-800:]
            return {
                "paths": [],
                "cancelled": True,
                "reason": "dialog-error",
                "message": stderr or f"对话框进程退出码 {completed.returncode}",
                "elapsed_ms": elapsed,
            }

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {
                "paths": [],
                "cancelled": True,
                "reason": "no-result",
                "message": "对话框进程没有返回结果（可能被系统强制关闭）",
                "elapsed_ms": elapsed,
            }
    finally:
        if _ACTIVE_CANCEL.get("path") == cancel_file:
            _ACTIVE_CANCEL["path"] = None
        _cleanup(temp_dir)

    paths = []
    for item in data.get("paths") or []:
        text = str(item or "").strip()
        if not text:
            continue
        # tkinter 在 Windows 上会把分隔符统一成 /（"D:/a/b"）。
        # 存进字段后要跟 Qt 的写法保持一致，否则同一个路径在两边看起来不一样。
        try:
            paths.append(str(Path(text)))
        except (OSError, ValueError):
            paths.append(text)
    if paths:
        _remember(kwargs.get("field", ""), paths[0])
    result = {
        "paths": paths,
        "cancelled": not paths,
        "elapsed_ms": elapsed,
    }
    # 子进程主动结束时会带原因：没弹出窗口、用户在网页上点了取消。
    # 不带原因的空路径才是用户在系统框里点了取消。
    reason = str(data.get("reason") or "").strip()
    if reason:
        result["reason"] = reason
    return result


def _cleanup(folder: Path) -> None:
    try:
        for item in folder.iterdir():
            item.unlink()
        folder.rmdir()
    except OSError:
        pass


def _now_ms() -> int:
    import time

    return int(time.perf_counter() * 1000)


def reset_state() -> None:
    """清掉记住的目录（测试用）。"""
    try:
        STATE_FILE.unlink()
    except OSError:
        pass
