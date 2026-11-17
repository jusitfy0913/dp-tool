# -*- coding: utf-8 -*-
"""安装包的启动器：双击 exe 就能用，不需要 bat，也不需要 PowerShell。

## 为什么要把这段逻辑从 PowerShell 搬到 Python

``packaging/templates/start_user.ps1`` 做的事是：读 ``config.json``、设置约 20 个
环境变量、起 job 轮询健康检查、打开浏览器、再启动 exe。这段逻辑本身没问题，
但对现场人员有三个坑：

1. 要先找到并双击 ``start_web.bat``，看起来不像一个软件；
2. 杀毒软件会把 ``powershell -ExecutionPolicy Bypass`` 当成可疑行为，
   而弹出来的是英文报错；
3. 出错时窗口一闪就没，用户拿不到任何可读信息。

把同一件事写进 exe 之后：双击即用、错误用中文消息框说清「发生了什么、为什么、
该怎么办」，并且环境变量只有**一处实现**——bat 与 ps1 不再各写一份。

## 什么时候不接管

``frozen_entry.py`` 里带 ``--tool`` 的调用是工具分支（``val_engine``、
``infer_worker``、``native_dialog`` 等），它们由服务进程自己拉起，
**不能**走启动流程。本模块只负责「无参数启动」。

## 线程分工（v1.41.0 起）

主线程归窗口（``desktop.run_window`` → ``webview.start()``，WinForms 的硬要求），
uvicorn 的 ``Server.run()`` 挪到后台线程。以前是反过来的（``uvicorn.run`` 占主线程、
后台线程去开浏览器），那个结构开不出窗口。

关窗口时把 ``server.should_exit`` 置 True，等它收尾后再返回，
这样「关窗口 = 关软件」是真的把端口放掉了，不是靠进程被杀。

## 控制台模式的前提

打包用 ``--windows-console-mode=attach``：双击时没有控制台，
``sys.stdout`` / ``sys.stderr`` 是 ``None``。因此 :func:`ensure_stdio` 必须
在 import ``app`` 之前执行——任何一处 ``print`` 撞上 ``None`` 都会抛
``AttributeError: 'NoneType' object has no attribute 'write'``。
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8060

#: 健康检查最长等待。冷启动要 import torch 之外的依赖，实测 3~8 秒；
#: 60 秒是给「机械硬盘 + 杀毒软件全盘扫描」留的余量。
HEALTH_TIMEOUT_SECONDS = 60.0

#: 关停等待的下限与上限（秒）。实际等待按运行中的任务数放大：
#: 每个任务最多 15 秒（``runners.kill_process_tree`` 的 taskkill 超时），
#: 所以 15 + 15×任务数 足够覆盖，下限 20 秒给「没有任务但服务在收尾」留余量。
SHUTDOWN_BASE_SECONDS = 20.0
SHUTDOWN_PER_TASK_SECONDS = 15.0
SHUTDOWN_MAX_SECONDS = 120.0

#: 单实例互斥量名字。用 ``Local\`` 而不是 ``Global\``：
#: 远程桌面与切换用户是各自独立的 Windows 会话，两个人各自开一个互不影响。
MUTEX_NAME = "Local\\TRTForgeWebStudio"

ERROR_ALREADY_EXISTS = 183

#: 界面模式的三种取值，见 ``resolve_ui_mode``。
UI_WINDOW = "window"
UI_BROWSER = "browser"
UI_AUTO = "auto"
UI_MODES = (UI_WINDOW, UI_BROWSER, UI_AUTO)


# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------

def is_compiled() -> bool:
    """本进程是不是被打包成 exe 了。

    为什么不能只用 ``sys.frozen``：Nuitka 的 standalone **不设** ``sys.frozen``
    （它设的是模块级的 ``__compiled__``）。以前只判 ``sys.frozen``，结果双击
    ``app\\TRTForgeWeb.exe`` 时 :func:`install_root` 一路走到「本文件上两级」，
    算出来是安装目录的**上一级**——实测日志里出现过
    「安装根=D:\\...\\dist」+「未找到 config.json」，前端目录跟着指错。
    通过 bat 启动时因为设了 ``TRTFORGE_HOME`` 才一直没暴露。
    """
    if getattr(sys, "frozen", False):
        return True
    # Nuitka 在每个被编译的模块里注入 __compiled__；glibc 的 PyInstaller 走上面的分支。
    return "__compiled__" in globals()


def _executable() -> Path:
    """当前在跑的 exe 的全路径。

    为什么不能直接用 ``sys.executable``：实测 Nuitka standalone（4.3.4）下
    ``sys.executable`` 是 ``<dist>\\python.exe``，**这个文件根本不存在**；真正在跑的
    是 ``sys.argv[0]``。用 ``sys.executable`` 取 ``.parent`` 会正好落进 ``app`` 目录，
    靠「父目录叫 app」这条判断歪打正着，换个布局就错。所以优先取存在的那一个。
    """
    for candidate in (sys.argv[0] if sys.argv else "", sys.executable):
        if not candidate:
            continue
        path = Path(candidate)
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return Path(sys.executable).resolve()


def install_root() -> Path:
    """定位安装根目录。

    顺序：
    1. ``TRTFORGE_HOME``（``start_web.bat`` 会设）；
    2. 可执行文件的上一级——安装包里 exe 在 ``<root>\\app\\``；
    3. 当前工作目录（开发环境用）；
    4. 本文件上两级（源码形态 ``6-web/backend/launcher.py`` → 仓库根）。
    """
    env = os.environ.get("TRTFORGE_HOME", "").strip()
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate.resolve()

    if is_compiled():
        parent = _executable().parent
        # 安装包布局：<root>/app/TRTForgeWeb.exe
        if parent.name.lower() == "app" and parent.parent.is_dir():
            return parent.parent
        return parent

    cwd = Path.cwd()
    # 源码形态：backend -> 6-web -> 仓库根
    source_root = Path(__file__).resolve().parents[2]
    for candidate in (cwd, source_root):
        if (candidate / "config.json").is_file():
            return candidate
    return source_root


def load_config(root: Path) -> dict[str, Any]:
    """读 ``config.json``。

    读不到或格式坏掉时返回默认值，**不阻止启动**：默认地址足以把软件跑起来，
    缺一个配置文件不该让用户看到一个错误框。
    """
    path = root / "config.json"
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_address(cfg: dict[str, Any]) -> tuple[str, int]:
    host = str(cfg.get("host") or DEFAULT_HOST).strip() or DEFAULT_HOST
    try:
        port = int(cfg.get("port") or DEFAULT_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    if not (1 <= port <= 65535):
        port = DEFAULT_PORT
    return host, port


def browse_host(host: str) -> str:
    """监听地址 → 浏览器地址。

    ``0.0.0.0`` 是「监听所有网卡」，浏览器打不开这个地址，要换成本机回环。
    """
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def page_url(host: str, port: int) -> str:
    return f"http://{browse_host(host)}:{port}/"


def resolve_ui_mode(cfg: dict[str, Any]) -> str:
    """读 ``config.json`` 的 ``ui`` 字段：``window`` / ``browser`` / ``auto``。

    ``auto``（默认）的语义：能开窗口就开窗口，开不了就退回系统浏览器。
    非法值一律当 ``auto``——配置写错不该让软件起不来，装不上窗口时还会自动降级，
    所以这个方向的容错是安全的。
    """
    raw = cfg.get("ui")
    text = str(raw).strip().lower() if isinstance(raw, str) else ""
    return text if text in UI_MODES else UI_AUTO


def should_use_window(cfg: dict[str, Any]) -> bool:
    """按配置和机器能力决定这次要不要开窗口。

    只有 ``auto`` 会去探 WebView2；``window`` 是用户的明确要求，直接尝试（失败了
    由 :func:`open_ui` 兜到浏览器）；``browser`` 连探测都不做，省掉 pythonnet
    那几十毫秒的 import。
    """
    mode = resolve_ui_mode(cfg)
    if mode == UI_BROWSER:
        return False
    if mode == UI_WINDOW:
        return True
    try:
        import desktop  # noqa: PLC0415

        return desktop.webview_available()
    except Exception:  # noqa: BLE001 - 窗口模块缺失/Nuitka 没编进去都当「不能开窗」
        return False


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

class Log:
    """启动日志。同时写文件和（有控制台时）打到标准错误。

    编码固定 UTF-8：安装路径和错误信息都可能有中文，用系统默认的 GBK 写出来
    在别的工具里打开是乱码。
    """

    def __init__(self, root: Path):
        self.path = root / "runtime" / "logs" / "launcher.log"
        self._handle = None
        self._lock = threading.Lock()

    def open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        except OSError:
            self._handle = None

    def _stderr_is_own_file(self) -> bool:
        """stderr 是不是已经被 :func:`ensure_stdio` 指到本日志文件上了。

        没有控制台时（双击 exe、pythonw 启动）``ensure_stdio`` 会把 stderr
        重定向到同一个 ``launcher.log``。此时再往 stderr 回显一次，日志里每条
        都会出现两遍，看起来像启动了两次，排查时非常容易误判。
        """
        try:
            name = getattr(sys.stderr, "name", "")
        except Exception:  # noqa: BLE001
            return False
        if not name:
            return False
        try:
            return Path(name).resolve() == self.path.resolve()
        except OSError:
            return False

    def write(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.write(line + "\n")
                    self._handle.flush()
                except OSError:
                    pass
        # 从命令行启动时（attach 模式会附着到调用方控制台）也让人看见
        if self._stderr_is_own_file():
            return
        try:
            if sys.stderr is not None:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


def ensure_stdio(log: Log | None = None) -> None:
    """没有控制台时把 stdout/stderr 重定向到日志文件。

    ``attach`` 模式下双击启动，``sys.stdout`` 与 ``sys.stderr`` 是 ``None``。
    只要有任何一处 ``print``，就会抛 ``AttributeError``，而且是在
    「import app」的过程中抛——用户看到的就是闪退。
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    target: Any = None
    if log is not None:
        try:
            log.path.parent.mkdir(parents=True, exist_ok=True)
            target = log.path.open("a", encoding="utf-8", buffering=1)
        except OSError:
            target = None
    if target is None:
        target = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - 全生命周期持有
    if sys.stdout is None:
        sys.stdout = target
    if sys.stderr is None:
        sys.stderr = target


# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------

def environment(root: Path, cfg: dict[str, Any]) -> dict[str, str]:
    """拼出运行需要的环境变量。

    这是**唯一**一份实现：``start_user.ps1`` 已不再设置这些变量（只负责用
    ``TRTFORGE_LEGACY_START`` 起排障模式），``test_launcher.py`` 会盯着这一点。
    返回而不是直接 ``os.environ.update``：这样单测能断言清单内容，
    不必真的污染当前进程。
    """
    host, port = resolve_address(cfg)

    python = str(cfg.get("python") or "").strip()
    if not python:
        bundled = root / "train-env" / "python.exe"
        if bundled.is_file():
            python = str(bundled)

    roots: list[str] = []
    raw_roots = cfg.get("data_roots")
    if isinstance(raw_roots, list):
        roots = [str(item) for item in raw_roots if item]
    elif isinstance(raw_roots, str) and raw_roots.strip():
        roots = [raw_roots.strip()]

    env: dict[str, str] = {
        "TRTFORGE_HOME": str(root),
        "TRTFORGE_REPO_ROOT": str(root),
        "TRTFORGE_FRONTEND_DIR": str(root / "frontend"),
        "TRTFORGE_BIN_DIR": str(root / "bin"),
        "TRTFORGE_YOLO_LAUNCHER_DIR": str(root / "launchers"),
        "TRTFORGE_ANOMALIB_LAUNCHER_DIR": str(root / "launchers"),
        "TRTFORGE_MODEL_DIR": str(root / "models"),
        "TRTFORGE_OCR_MODEL_DIR": str(root / "models" / "ocr"),
        "TRTFORGE_TRTEXEC": str(root / "bin" / "trtexec.exe"),
        "OPENCV_BIN_DIR": str(root / "bin"),
        "CUDA_BIN_DIR": str(root / "bin"),
        "TRTFORGE_WEB_PORT": str(port),
        "TRTFORGE_WEB_HOST": host,
        "TRTFORGE_BROWSE_ROOTS": os.pathsep.join(roots),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    # 自带工具分支（引擎构建、批量推理…）只有在**真的存在**自己的 exe 时才切过去。
    # 源码形态跑 launcher.py（开发用 6-web\.devroot）没有 app\TRTForgeWeb.exe，
    # 无条件写进去会让这些功能去调一个不存在的文件，而且报错信息很难看懂。
    tool_exe = root / "app" / "TRTForgeWeb.exe"
    if tool_exe.is_file():
        env["TRTFORGE_TOOL_EXE"] = str(tool_exe)
    # tkinter 的标准库位置随 exe 走；不存在就不设，让 Python 用默认值。
    for key, folder in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        candidate = root / "app" / folder
        if candidate.is_dir():
            env[key] = str(candidate)
    if python:
        env["TRTFORGE_PYTHON"] = python
    return env


def apply_environment(root: Path, cfg: dict[str, Any]) -> tuple[str, int]:
    """把环境变量写进当前进程，并返回 (host, port)。"""
    env = environment(root, cfg)
    os.environ.update(env)
    # PATH 前置 bin：推理 DLL 的依赖（OpenCV / CUDA / TensorRT）都在里面。
    # 放在最前面而不是最后面，避免系统里另一份同名 DLL 抢先被加载。
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = env["TRTFORGE_BIN_DIR"] + os.pathsep + current if current else env["TRTFORGE_BIN_DIR"]
    _, port = resolve_address(cfg)
    return str(env["TRTFORGE_WEB_HOST"]), port


# ---------------------------------------------------------------------------
# 单实例与端口
# ---------------------------------------------------------------------------

def acquire_mutex(name: str = MUTEX_NAME) -> tuple[Any, bool]:
    """创建命名互斥量，返回 ``(句柄, 是否新建)``。

    用互斥量而不是「探测端口是否空闲」：后者有竞态——两个进程可以同时
    发现端口空闲、同时开始 bind，第二个才失败，用户看到的是一个莫名其妙的报错。
    """
    if os.name != "nt":
        return None, True
    try:
        import ctypes  # noqa: PLC0415

        # use_last_error=True + get_last_error()：windll 的 GetLastError 会被
        # ctypes 内部调用覆盖，读到的可能不是 CreateMutexW 的错误码。
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 不声明 restype 会被当成 c_int，64 位下把句柄截断成 32 位。
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        handle = kernel32.CreateMutexW(None, True, name)
        already = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
        if not handle:
            return None, True
        return handle, not already
    except Exception:  # noqa: BLE001 - 拿不到互斥量不该阻止启动
        return None, True


def release_mutex(handle: Any) -> None:
    """释放并关闭句柄。

    句柄可能来自「已存在的互斥量」——那种情况下当前线程并没有所有权，
    ``ReleaseMutex`` 会失败。这里不看返回值：失败的语义恰好就是「本来就没持有」。
    """
    if handle is None or os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        pass


def port_free(host: str, port: int) -> bool:
    """端口是否可绑定。

    注意 ``SO_REUSEADDR`` 在 Windows 上的语义与 POSIX 不同：它允许绑定到
    「已被占用但处于 TIME_WAIT」的地址，不会掩盖「正在监听」的冲突。
    这里不用它，就是要让真实的占用暴露出来。
    """
    probe_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in probe_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((probe_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def health_version(host: str, port: int, timeout: float = 1.0) -> str | None:
    """读 ``/api/health``，返回版本号；读不到返回 ``None``。

    拿到版本号才能区分「端口上是本软件的旧实例」与「端口被别的程序占了」。
    这两种情况的处理方式完全不同，用一条笼统的「端口被占用」会误导人。
    """
    url = f"http://{browse_host(host)}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    return str(payload.get("version") or "")


def wait_healthy(host: str, port: int, timeout: float = HEALTH_TIMEOUT_SECONDS) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_version(host, port, timeout=1.0) is not None:
            return True
        time.sleep(0.5)
    return False


def _active_run_count() -> int:
    """当前运行中的任务数，用来放大关停等待时间。

    为什么按任务数放大：任务停止要走 ``taskkill /T /F`` 杀整棵进程树，
    每个任务最坏十几秒（等句柄、等 DLL 卸载）。固定 10 秒在「同时开着
    训练 + 自动实验」时必然不够，结果是窗口关了、进程还在，端口和显存
    都没释放——用户再点图标就会被单实例互斥量挡在门外，只能重启电脑。

    拿不到服务端状态时退回 0（只看下限）：宁可少等一点，也不能因为
    统计失败就把关窗口卡到两分钟。
    """
    try:
        import runners  # noqa: PLC0415 - 只在关停时用，避免拖慢启动
    except Exception:  # noqa: BLE001 - 打包环境下导入失败按 0 处理
        return 0
    try:
        return runners.manager.active_count()
    except Exception:  # noqa: BLE001 - 同上
        return 0


def _shutdown_timeout() -> float:
    """关停实际等待秒数：``BASE + PER_TASK × 任务数``，夹在上下限之间。"""
    count = _active_run_count()
    return min(
        SHUTDOWN_MAX_SECONDS,
        SHUTDOWN_BASE_SECONDS + SHUTDOWN_PER_TASK_SECONDS * count,
    )


def open_browser(url: str) -> None:
    try:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - 打不开浏览器也要把服务跑起来
        pass


# ---------------------------------------------------------------------------
# 错误提示
# ---------------------------------------------------------------------------

def _fallback_stderr() -> Any:
    """找一个能写的流。三个都可能是 ``None``（无控制台且日志没开起来）。"""
    if sys.stderr is not None:
        return sys.stderr
    if sys.stdout is not None:
        return sys.stdout
    return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - 进程退出即释放


def message_box(text: str, title: str = "TRTForge Web Studio") -> None:
    """弹一个 Windows 原生错误框。

    用 ``MessageBoxW`` 而不是 tkinter：进程启动失败的当下，tkinter 可能还没
    import 成功（它就是缺依赖的那一类），而 ``MessageBoxW`` 是系统 API。
    """
    if os.name != "nt":
        stream = _fallback_stderr()
        stream.write(f"{title}: {text}\n")
        stream.flush()
        return
    try:
        import ctypes  # noqa: PLC0415

        MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(0, text, title, MB_ICONERROR)
    except Exception:  # noqa: BLE001
        stream = _fallback_stderr()
        stream.write(f"{title}: {text}\n")
        stream.flush()


def _port_in_use_text(root: Path, port: int) -> str:
    return (
        f"启动失败：端口 {port} 已被其它程序占用\n\n"
        f"软件需要在这个端口提供网页服务，但该端口现在被别的程序占着。\n\n"
        f"请任选一种：\n"
        f"1. 用记事本打开 {root / 'config.json'}，把 port 改成 {port + 1} 后重新双击；\n"
        f"2. 结束占用 {port} 端口的程序，再重新双击。"
    )


def _window_fallback_text(error: str = "") -> str:
    text = (
        "窗口没能打开，已改用系统浏览器\n\n"
        "这台电脑缺少 Microsoft Edge WebView2 运行时，或者它没能正常加载。\n"
        "界面的功能不受影响：现在已经在默认浏览器里打开了同一个地址。\n\n"
        "想以后都直接用浏览器：把 config.json 里的 \"ui\" 改成 \"browser\"。\n"
        "想修好窗口：装一次微软的 WebView2 Runtime（免费，搜 WebView2 Runtime 下载）。"
    )
    if error:
        text = f"{text}\n\n技术细节：{error}"
    return text


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _make_server(host: str, port: int) -> Any:
    """造一个 uvicorn Server 实例，不启动。放最后一步：前面的检查都通过再 import app。"""
    import uvicorn  # noqa: PLC0415

    import app as app_module  # noqa: PLC0415

    config = uvicorn.Config(app_module.app, host=host, port=port, log_level="info")
    return uvicorn.Server(config)


def _serve_blocking(host: str, port: int) -> int:
    """启动服务并阻塞当前线程。排障模式（TRTFORGE_LEGACY_START）仍走这条路。"""
    server = _make_server(host, port)
    server.run()
    return 0


def _window_title(version: str) -> str:
    """窗口标题。desktop 模块拿不到时退回常量，保证两次启动能找到同一个窗口。"""
    try:
        import desktop  # noqa: PLC0415

        return desktop.window_title(version)
    except Exception:  # noqa: BLE001
        return f"TRTForge Web Studio {version}".strip()


def show_existing_ui(url: str, version: str, log: Log) -> None:
    """第二次双击时把已经开着的界面亮出来。

    窗口模式：按标题找回那扇窗并提到前面；找不到（比如它其实跑在浏览器模式）就开浏览器。
    浏览器模式：行为不变，让浏览器自己把标签页提到前面。
    """
    title = _window_title(version)
    try:
        import desktop  # noqa: PLC0415

        if desktop.webview_available() and desktop.activate_existing_window(title):
            log.write(f"已把现有窗口提到前面：{title}")
            return
    except Exception as exc:  # noqa: BLE001
        log.write(f"激活现有窗口失败：{type(exc).__name__}: {exc}")
    log.write(f"打开浏览器：{url}")
    open_browser(url)


def open_ui(
    url: str,
    root: Path,
    log: Log,
    stop: Callable[[], None],
    version: str = "",
) -> bool:
    """按配置把界面呈现给用户。返回 True 表示开的窗口，False 表示用了浏览器。

    ``stop`` 在窗口关闭时被调用（launcher 用它置 uvicorn 的 should_exit）。
    窗口起不来时**不抛异常**：退回浏览器并提示一次，用户至少还能干活。
    """
    if not should_use_window(load_config(root)):
        log.write(f"界面模式：浏览器（{url}）")
        open_browser(url)
        return False

    try:
        import desktop  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.write(f"窗口模块不可用，改用浏览器：{type(exc).__name__}: {exc}")
        message_box(_window_fallback_text(f"desktop 模块导入失败：{exc}"))
        open_browser(url)
        return False

    title = desktop.window_title(version)
    log.write(f"界面模式：窗口（{title}）")
    try:
        desktop.run_window(url, root, title=title, on_close=stop)
    except Exception as exc:  # noqa: BLE001 - 窗口失败不能让整个软件不可用
        detail = f"{type(exc).__name__}: {exc}"
        log.write(f"窗口启动失败，退回浏览器：{detail}")
        message_box(_window_fallback_text(detail))
        open_browser(url)
        return False
    return True


def run() -> int:
    """无参数启动的完整流程，返回进程退出码。"""
    root = install_root()
    log = Log(root)
    log.open()
    ensure_stdio(log)

    log.write("=" * 60)
    log.write(
        f"启动：安装根={root}  python={sys.version.split()[0]}  "
        f"frozen={getattr(sys, 'frozen', False)}  compiled={is_compiled()}"
    )

    # 排障回退：直接走原来的启动方式，不看互斥量也不开浏览器
    if os.environ.get("TRTFORGE_LEGACY_START", "").strip() in {"1", "true", "yes"}:
        log.write("TRTFORGE_LEGACY_START 已设置，跳过启动器逻辑")
        host, port = apply_environment(root, load_config(root))
        return _serve_blocking(host, port)

    cfg = load_config(root)
    if not (root / "config.json").is_file():
        log.write(f"未找到 config.json，使用默认地址 {DEFAULT_HOST}:{DEFAULT_PORT}")
    host, port = apply_environment(root, cfg)
    url = page_url(host, port)
    log.write(f"配置：host={host} port={port} url={url} ui={resolve_ui_mode(cfg)}")

    handle, created = acquire_mutex()
    if not created:
        # 已有一个实例：只要它是活的就把界面亮出来，不再起第二个
        version = health_version(host, port)
        if version is not None:
            log.write(f"已有实例在运行（版本 {version}），不重复启动")
            show_existing_ui(url, version, log)
            release_mutex(handle)
            log.close()
            return 0
        text = (
            "启动失败：上一个实例还没退干净\n\n"
            "程序检测到已经有 TRTForgeWeb 在运行，但它现在不响应。\n"
            "通常是上一次正在关闭，或者进程卡住了。\n\n"
            "请任选一种：\n"
            "1. 等 5 秒后重新双击；\n"
            "2. 打开任务管理器，结束 TRTForgeWeb.exe 后重新双击。"
        )
        log.write("互斥量已存在但健康检查不通，判定为残留实例")
        message_box(text)
        release_mutex(handle)
        log.close()
        return 1

    try:
        if not port_free(host, port):
            version = health_version(host, port)
            if version is not None:
                log.write(f"端口 {port} 上是本软件的实例（版本 {version}），不重复启动")
                show_existing_ui(url, version, log)
                return 0
            log.write(f"端口 {port} 被其它程序占用")
            message_box(_port_in_use_text(root, port))
            return 1

        # 服务跑后台线程，主线程留给窗口。顺序反了窗口根本出不来
        # （WinForms 要求主线程建窗口），所以这里不能用 uvicorn.run。
        stop = threading.Event()
        server = _make_server(host, port)

        def _serve() -> None:
            try:
                server.run()
            except Exception as exc:  # noqa: BLE001 - 记下来，主线程那边才知道为什么停了
                log.write(f"服务异常退出：{type(exc).__name__}: {exc}")
            finally:
                stop.set()

        def _stop_server() -> None:
            # 关窗口时调用。只置 uvicorn 的退出标志，让 `Server.run()` 自己走完
            # 优雅退出（等 lifespan 收尾、关 socket）后返回——**不要**在这里置
            # `stop`，否则下面的等待循环会立刻通过，等于没等，端口可能还在占用中。
            server.should_exit = True

        threading.Thread(target=_serve, name="uvicorn", daemon=True).start()

        if not wait_healthy(host, port):
            log.write(f"服务在 {HEALTH_TIMEOUT_SECONDS:.0f} 秒内没有就绪")
            _stop_server()
            message_box(
                "启动失败：服务没能起来\n\n"
                f"{HEALTH_TIMEOUT_SECONDS:.0f} 秒内没有响应健康检查。\n\n"
                f"详细信息已写入：\n{log.path}\n\n"
                "请把这个文件发给提供软件的人。"
            )
            return 1

        log.write(f"服务已就绪：{url}")
        used_window = open_ui(url, root, log, _stop_server, health_version(host, port) or "")
        if used_window:
            # 窗口关闭 = 关软件，等 uvicorn 真的收尾再返回，端口才能立刻释放。
            log.write("窗口已关闭，通知服务退出")
            _stop_server()
            timeout = _shutdown_timeout()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if stop.is_set():
                    break
                time.sleep(0.1)
            if not stop.is_set():
                # 超时说明有 daemon 线程卡住了（典型：某个任务正在写大文件、
                # 杀毒软件拦着句柄、或者杀进程树本身还没返回）。
                # 继续等只会让用户看到「关不掉」的窗口然后去任务管理器强杀，
                # 那会留下真正的半清理状态。这里主动强制退出：
                # 子进程树已在 lifespan 里被 taskkill /T /F 清过一遍，
                # 剩下的只是本进程的线程，交给操作系统回收是可控的。
                log.write(f"服务在 {timeout:.0f} 秒内没有退出，强制结束进程")
                log.close()
                os._exit(0)
        else:
            # 浏览器模式保持老行为：服务在后台一直跑，直到进程被杀。
            log.write("已打开浏览器，服务继续在后台运行")
            while not stop.is_set():
                time.sleep(1.0)
        log.write("服务退出")
        return 0
    except Exception as exc:  # noqa: BLE001 - 启动失败要让用户看到原因
        import traceback  # noqa: PLC0415

        detail = traceback.format_exc()
        log.write("启动异常：\n" + detail)
        message_box(
            "启动失败：服务没能启动\n\n"
            f"错误：{type(exc).__name__}: {exc}\n\n"
            f"详细信息已写入：\n{log.path}\n\n"
            "请把这个文件发给提供软件的人。"
        )
        return 1
    finally:
        release_mutex(handle)
        log.close()


if __name__ == "__main__":
    # 源码形态直接 `python launcher.py` 与打包后的入口等价（frozen_entry.main()
    # 也是调 run()）。少了这一段，源码形态直接运行会**静默退出、什么都不做**：
    # 进程退出码 0、端口没监听、日志文件都不生成，排查起来毫无线索。
    raise SystemExit(run())
