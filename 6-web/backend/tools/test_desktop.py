# -*- coding: utf-8 -*-
"""桌面窗口测试（desktop.py）。

    python tools/test_desktop.py

跑法：``tools/run_all_tests.py`` 会自动收录本文件（目录扫描 ``test_*.py``）。

测什么
------
1. **内核识别**：只有 ``edgechromium``（WebView2）才算「能开窗口」。
   ``mshtml``（老 IE 内核）与「import 失败」都必须判成 False —— 否则界面会在
   没装 WebView2 的机器上退化成 IE，页面直接错版，比开浏览器还难排查。
2. **窗口参数**：窗口收到的 URL / 标题正确；``private_mode=False`` 且
   ``storage_path`` 落在 ``<root>\\runtime\\webview``（localStorage 要持久化，
   并且不能让 pywebview 去删临时目录）。目录不存在时要自己建。
3. **关窗口语义**：``webview.start()`` 返回后必须调 ``on_close``（launcher 靠它
   停 uvicorn）；``webview.start()`` 抛异常时也要调，然后异常继续往上抛。
4. **window.open 转交**：页面里的 ``window.open`` 会被改成走系统浏览器
   （否则查看原始日志会点出一个空白窗口）。
5. **激活已有窗口**：``FindWindowW`` 找不到时返回 False，不抛异常。
6. **重量级依赖**：desktop 模块顶层不能 import webview / pythonnet，
   否则 ``ui=browser`` 的用户也要付这笔启动成本。

所有用例都用注入的假 ``webview`` 模块，不真的建窗口，跑起来是毫秒级。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from _harness import case, check_eq, finish, require, run_cases  # noqa: E402# noqa: E402

import desktop  # noqa: E402


# ---------------------------------------------------------------------------
# 假 webview
# ---------------------------------------------------------------------------

class FakeLoaded:
    """``window.events.loaded``：只要求支持 ``+=``。"""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class FakeEvents:
    def __init__(self) -> None:
        self.loaded = FakeLoaded()


class FakeWindow:
    def __init__(self, title, url, **kwargs) -> None:
        self.title = title
        self.url = url
        self.kwargs = kwargs
        self.events = FakeEvents()
        self.exposed: dict = {}
        self.scripts: list[str] = []

    def expose(self, *functions) -> None:
        for func in functions:
            self.exposed[func.__name__] = func

    def evaluate_js(self, script: str) -> None:
        self.scripts.append(script)

    def destroy(self) -> None:
        pass


class FakeWebview(types.ModuleType):
    """最小的 webview 替身：记录 create_window / start 收到的参数。"""

    def __init__(self, renderer: str = "edgechromium") -> None:
        super().__init__("webview")
        self.renderer = renderer
        self.windows: list[FakeWindow] = []
        self.start_calls: list[dict] = []
        self.start_error: Exception | None = None
        self.on_start = None

    def create_window(self, title, url, **kwargs):
        window = FakeWindow(title, url, **kwargs)
        self.windows.append(window)
        return window

    def start(self, **kwargs):
        self.start_calls.append(kwargs)
        if self.on_start is not None:
            self.on_start(self)
        if self.start_error is not None:
            raise self.start_error
        return None


def install_fake_webview(renderer: str = "edgechromium") -> FakeWebview:
    """把假 webview 塞进 sys.modules，并让 ``from webview.platforms import winforms`` 成立。"""
    fake = FakeWebview(renderer)
    platforms = types.ModuleType("webview.platforms")
    winforms = types.ModuleType("webview.platforms.winforms")
    winforms.renderer = renderer
    platforms.winforms = winforms
    fake.platforms = platforms
    sys.modules["webview"] = fake
    sys.modules["webview.platforms"] = platforms
    sys.modules["webview.platforms.winforms"] = winforms
    desktop.reset_cache()
    return fake


def remove_fake_webview() -> None:
    for name in ("webview.platforms.winforms", "webview.platforms", "webview"):
        sys.modules.pop(name, None)
    desktop.reset_cache()


# ---------------------------------------------------------------------------
# 1. 内核识别
# ---------------------------------------------------------------------------

@case("renderer=edgechromium 才算能开窗口")
def test_available_edgechromium():
    install_fake_webview("edgechromium")
    try:
        check_eq(desktop.renderer(), "edgechromium", "renderer")
        require(desktop.webview_available(), "edgechromium 应判为可用")
    finally:
        remove_fake_webview()


@case("mshtml（老 IE 内核）判为不可用")
def test_available_mshtml():
    install_fake_webview("mshtml")
    try:
        check_eq(desktop.renderer(), "mshtml", "renderer")
        require(not desktop.webview_available(), "mshtml 不能当 WebView2 用")
    finally:
        remove_fake_webview()


@case("webview 导入失败时返回 False 而不是抛异常")
def test_available_import_error():
    remove_fake_webview()
    # 真实环境里 webview 可能装着（开发机），这里直接模拟 import 抛错。
    class Boom(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError("pythonnet 没装")

    blocker = types.ModuleType("webview")
    blocker.platforms = Boom("webview.platforms")
    sys.modules["webview"] = blocker
    sys.modules.pop("webview.platforms", None)
    desktop.reset_cache()
    try:
        check_eq(desktop.renderer(), "", "import 失败应返回空串")
        require(not desktop.webview_available(), "import 失败应判为不可用")
    finally:
        remove_fake_webview()


@case("renderer 结果有缓存，reset_cache 后重新探测")
def test_renderer_cache():
    install_fake_webview("edgechromium")
    try:
        check_eq(desktop.renderer(), "edgechromium", "首次探测")
        # 换成 mshtml：改的是假 winforms 模块的属性，模拟「机器状态变了」
        sys.modules["webview.platforms.winforms"].renderer = "mshtml"
        check_eq(desktop.renderer(), "edgechromium", "缓存后不再探测")
        desktop.reset_cache()
        check_eq(desktop.renderer(), "mshtml", "清缓存后重新探测")
    finally:
        remove_fake_webview()


# ---------------------------------------------------------------------------
# 2. 窗口参数
# ---------------------------------------------------------------------------

@case("用户数据目录固定落在 runtime\\webview")
def test_storage_dir():
    root = Path(r"D:\软件\TRTForgeWeb")
    check_eq(desktop.storage_dir(root), root / "runtime" / "webview", "storage_dir")


@case("窗口尺寸在屏幕内且不小于 900x600")
def test_window_size():
    width, height = desktop.window_size()
    require(width >= 900, f"宽度太小：{width}")
    require(height >= 600, f"高度太小：{height}")
    require(width <= 3840 and height <= 2160, f"尺寸不合理：{width}x{height}")


@case("标题带版本号，没版本号时只有固定前缀")
def test_window_title():
    check_eq(desktop.window_title("1.41.0"), "TRTForge Web Studio 1.41.0", "带版本")
    check_eq(desktop.window_title(""), "TRTForge Web Studio", "无版本")


@case("run_window 传参正确：URL、标题、尺寸、private_mode、storage_path")
def test_run_window_params():
    fake = install_fake_webview("edgechromium")
    closed: list[int] = []
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = desktop.run_window(
                "http://127.0.0.1:8061/",
                root,
                title="TRTForge Web Studio 1.41.0",
                on_close=lambda: closed.append(1),
            )
            check_eq(code, 0, "退出码")
            check_eq(len(fake.windows), 1, "应只建一个窗口")
            window = fake.windows[0]
            check_eq(window.url, "http://127.0.0.1:8061/", "URL")
            check_eq(window.title, "TRTForge Web Studio 1.41.0", "标题")
            width, height = desktop.window_size()
            check_eq(window.kwargs.get("width"), width, "宽度")
            check_eq(window.kwargs.get("height"), height, "高度")

            check_eq(len(fake.start_calls), 1, "webview.start 调用次数")
            start = fake.start_calls[0]
            check_eq(start.get("private_mode"), False, "private_mode")
            check_eq(
                Path(start.get("storage_path")).resolve(),
                (root / "runtime" / "webview").resolve(),
                "storage_path",
            )
            check_eq(start.get("gui"), "edgechromium", "gui")
            require((root / "runtime" / "webview").is_dir(), "用户数据目录应被创建")
            check_eq(len(closed), 1, "窗口关闭后应回调 on_close")
    finally:
        remove_fake_webview()


@case("run_window 失败时仍回调 on_close 并继续抛异常")
def test_run_window_error():
    fake = install_fake_webview("edgechromium")
    fake.start_error = RuntimeError("WebView2 初始化失败")
    closed: list[int] = []
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            raised = None
            try:
                desktop.run_window(
                    "http://127.0.0.1:8060/",
                    Path(tmp),
                    on_close=lambda: closed.append(1),
                )
            except RuntimeError as exc:
                raised = exc
            require(raised is not None, "异常应继续往上抛（launcher 要靠它退回浏览器）")
            check_eq(len(closed), 1, "异常路径也要回调 on_close")
    finally:
        remove_fake_webview()


# ---------------------------------------------------------------------------
# 3. window.open 转交系统浏览器
# ---------------------------------------------------------------------------

@case("window.open 被改成走系统浏览器")
def test_external_open():
    fake = install_fake_webview("edgechromium")
    opened: list[str] = []
    saved = desktop.webbrowser.open
    desktop.webbrowser.open = lambda url: opened.append(url) or True
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            desktop.run_window("http://127.0.0.1:8060/", Path(tmp))
            window = fake.windows[0]
            require("open_external" in window.exposed, "必须暴露 open_external 给 JS")
            require(len(window.events.loaded.handlers) == 1, "应挂一个 loaded 回调")

            # 模拟页面加载完成：注入覆盖脚本
            window.events.loaded.handlers[0]()
            require(len(window.scripts) == 1, "loaded 后应注入一次脚本")
            require("window.open" in window.scripts[0], "脚本应覆盖 window.open")

            # 模拟 JS 调过来
            window.exposed["open_external"]("http://127.0.0.1:8060/api/runs/abc/log")
            check_eq(opened, ["http://127.0.0.1:8060/api/runs/abc/log"], "转交地址")

            # 空地址不能开一堆空白标签
            window.exposed["open_external"]("")
            check_eq(len(opened), 1, "空地址应被忽略")
    finally:
        desktop.webbrowser.open = saved
        remove_fake_webview()


# ---------------------------------------------------------------------------
# 4. 激活已有窗口
# ---------------------------------------------------------------------------

@case("找不到已有窗口时 activate_existing_window 返回 False")
def test_activate_missing_window():
    # 不注入假 webview，直接用真实的 ctypes 调用。标题用一个不可能存在的名字，
    # 避免误伤开发机上真开着的窗口。
    check_eq(desktop.activate_existing_window("__no_such_window_9d3f__"), False, "找不到应返回 False")


# ---------------------------------------------------------------------------
# 5. 顶层不 import 重依赖
# ---------------------------------------------------------------------------

@case("desktop.py 顶层不 import webview / pythonnet")
def test_no_heavy_toplevel_import():
    source = (BACKEND / "desktop.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if line != line.lstrip():
            continue  # 缩进的行在函数里，延迟 import 是允许的
        stripped = line.strip()
        if stripped.startswith(("import webview", "import clr", "import clr_loader", "import pythonnet")):
            raise AssertionError(f"desktop.py 顶层 import 了重模块：{stripped}")


def main() -> int:
    print("桌面窗口测试")
    print("=" * 60)
    run_cases(globals())

    print("-" * 60)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
