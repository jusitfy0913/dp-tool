# -*- coding: utf-8 -*-
"""桌面窗口：把网页装进自己的 exe 窗口（WebView2），不用 Qt。

## 为什么不把界面换成 Qt

这套界面本来就是网页。要让它"看起来像一个软件"，只差一层窗口；换成 Qt 等于把
整个前端重写一遍，而且 Qt 在部分客户机上有部署与权限问题。WebView2 用的是系统
自带的 Edge 运行时（Win10 1803 之后一般都有），既不用分发 GUI 框架，也不需要
管理员权限，更不写注册表。

## 三种模式（``config.json`` 的 ``ui`` 字段）

==========  ==========================================================
``window``  必须在窗口里打开；WebView2 不可用时退回系统浏览器并提示一次
``browser`` 永远用系统浏览器（v1.40 及以前的行为）
``auto``    默认。能用窗口就用窗口，否则浏览器
==========  ==========================================================

## 关窗口就等于关软件

``webview.start()`` 是阻塞调用，**最后一个窗口销毁后返回**。launcher 拿到返回值
之后把 uvicorn 的 ``should_exit`` 置 True，进程正常退出、端口释放。这与浏览器
模式不同（浏览器关掉服务还在跑），是刻意的：窗口模式看起来就是个桌面软件，
用户关窗口就是关软件。

## 主线程归谁

``webview.start()`` 必须在主线程调用（WinForms 的要求），所以服务要挪到后台线程。
uvicorn 0.54 的 ``Server.capture_signals`` 里有主线程判断，在子线程里跑不会崩。

## 打包形态下缺 DLL 会长什么样

实测（2026-11-16）：Nuitka standalone 编出来少了 ``ffi.dll``（``_ctypes`` 用）与
``libssl-3-x64.dll`` / ``libcrypto-3-x64.dll``（``ssl`` 用），症状是**空的
ImportError**，进程直接 0xC0000409。打包脚本的 PE 导入表传递闭包会补齐这几个，
所以本模块不需要自己找 DLL；但 ``frozen_entry._selfcheck`` 必须真的 import 一次
``webview``，否则缺库要到用户双击时才炸。
"""

from __future__ import annotations

import ctypes
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable

#: 窗口标题。第二实例靠这个标题找回已有窗口，所以它必须稳定：
#: 带上版本号是为了任务栏和排查时能一眼看出跑的是哪一版。
TITLE_PREFIX = "TRTForge Web Studio"

#: 首选窗口尺寸。实际会按屏幕可用区域收窄，避免在小笔记本上开到屏幕外面。
PREFERRED_WIDTH = 1440
PREFERRED_HEIGHT = 900

_renderer_cache: str | None = None
_renderer_lock = threading.Lock()


def window_title(version: str = "") -> str:
    """窗口标题。version 为空时只有固定前缀。"""
    text = TITLE_PREFIX
    if version:
        text = f"{text} {version}"
    return text


def storage_dir(root: Path) -> Path:
    """WebView2 的用户数据目录。

    为什么必须指定而不是让它自己挑临时目录：
    1. 默认是 ``%TEMP%\\tmpXXXX``，退出时 pywebview 会尝试删除，BrowserMetrics
       文件还锁着就报 ``WinError 5``（实测），日志里多一条看不懂的报错；
    2. 不指定就等于每次启动都是全新浏览器，localStorage 全丢——界面记住的
       分栏宽度、上次的任务 id 都会重置；
    3. WebView2 不支持在网络盘上建这个目录，放在安装目录下最稳。
    """
    return root / "runtime" / "webview"


def _probe_renderer() -> str:
    """Windows 上 pywebview 到底选了哪个内核。

    ``edgechromium`` 才是 WebView2；``mshtml`` 是老 IE 内核（界面会坏），
    ``cef`` 要额外运行时。空串表示 pywebview / pythonnet 根本没装好。
    """
    if os.name != "nt":
        return ""
    try:
        # 延迟 import：这个 import 会把 pythonnet 与 .NET 一起拉起来，几十毫秒起步，
        # 而且 ``ui=browser`` 的用户完全不需要付这个成本。
        from webview.platforms import winforms  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - 缺依赖、缺 .NET 都是「不能用窗口」
        return ""
    return str(getattr(winforms, "renderer", "") or "")


def renderer() -> str:
    """缓存版 :func:`_probe_renderer`。"""
    global _renderer_cache
    with _renderer_lock:
        if _renderer_cache is None:
            _renderer_cache = _probe_renderer()
        return _renderer_cache


def webview_available() -> bool:
    """能不能真的开一个 WebView2 窗口。"""
    return renderer() == "edgechromium"


def reset_cache() -> None:
    """丢掉缓存。只给测试用：单测要反复模拟「有/没有 WebView2」。"""
    global _renderer_cache
    with _renderer_lock:
        _renderer_cache = None


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


def _work_area() -> tuple[int, int]:
    """主屏可用区域（已扣掉任务栏）。取不到时退回整屏，再取不到给保守值。"""
    try:
        user32 = ctypes.windll.user32
        rect = _Rect()
        # SPI_GETWORKAREA=0x0030：返回「桌面可用区域」，任务栏占的地方已经扣掉。
        # 用整屏（GetSystemMetrics）也行，但窗口最大化时会压到任务栏下面。
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            width = int(rect.right - rect.left)
            height = int(rect.bottom - rect.top)
            if width > 0 and height > 0:
                return width, height
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:  # noqa: BLE001
        pass
    return PREFERRED_WIDTH, PREFERRED_HEIGHT


def window_size() -> tuple[int, int]:
    """按屏幕收窄后的窗口尺寸：最多占可用宽高的 94%，再小不小于 900x600。"""
    screen_w, screen_h = _work_area()
    width = min(PREFERRED_WIDTH, max(900, int(screen_w * 0.94)))
    height = min(PREFERRED_HEIGHT, max(600, int(screen_h * 0.94)))
    return width, height


def activate_existing_window(title: str) -> bool:
    """把已经开着的那扇窗口提到前面。找不到返回 False。

    为什么要这个：双击第二次时，launcher 只会发现「已有实例」然后退出。窗口
    模式下如果什么都不做，用户点两下会觉得「软件打不开」。浏览器模式不存在这个
    问题（浏览器自己会把标签页提到前面）。
    """
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
    except Exception:  # noqa: BLE001
        return False
    handle = user32.FindWindowW(None, title)
    if not handle:
        return False
    # SW_RESTORE=9：窗口被最小化过也要能拉回来。
    user32.ShowWindow(handle, 9)
    if not user32.SetForegroundWindow(handle):
        # 前台权限被系统收走时 SetForegroundWindow 会失败（Windows 的前台锁），
        # SwitchToThisWindow 是同一件事的老接口，这时通常仍然有效。
        try:
            user32.SwitchToThisWindow(handle, True)
        except Exception:  # noqa: BLE001
            pass
    return True


def _install_external_open(window: Any) -> None:
    """让页面里的 ``window.open`` 交给系统浏览器。

    窗口里再开一个 WebView2 窗口会得到一片空白（没地址栏、没内容）。而
    ``logs.js`` 正是用 ``window.open(url, "_blank")`` 让人看原始日志。
    与其让用户点出一个空窗，不如转给系统浏览器——那里有地址栏、能另存。

    暴露给 JS 的函数名必须叫 ``open_external``：pywebview 的约定是
    ``pywebview.api.<函数名>``。
    """

    def open_external(url: str = "") -> None:
        target = str(url or "").strip()
        if not target:
            return
        try:
            webbrowser.open(target)
        except Exception:  # noqa: BLE001 - 打不开浏览器不该影响主界面
            pass

    try:
        window.expose(open_external)
    except Exception:  # noqa: BLE001
        return

    script = """
    (() => {
      if (window.__trtforgeExternalOpen) { return; }
      window.__trtforgeExternalOpen = true;
      window.open = function (url) {
        let abs = "";
        try { abs = url ? new URL(String(url), location.href).href : ""; } catch (e) { abs = ""; }
        if (!abs) { return null; }
        const call = () => {
          try {
            if (window.pywebview && window.pywebview.api && window.pywebview.api.open_external) {
              window.pywebview.api.open_external(abs);
              return true;
            }
          } catch (e) { }
          return false;
        };
        // pywebview 的 api 是页面加载后才挂上去的，可能还没就绪，稍后再试一次。
        if (!call()) { setTimeout(call, 400); }
        return null;
      };
    })();
    """

    def on_loaded(*_: Any) -> None:
        try:
            window.evaluate_js(script)
        except Exception:  # noqa: BLE001
            pass

    try:
        window.events.loaded += on_loaded
    except Exception:  # noqa: BLE001
        pass


def run_window(
    url: str,
    root: Path,
    title: str = "",
    on_close: Callable[[], None] | None = None,
) -> int:
    """在主线程开窗口并阻塞到它被关掉。返回进程退出码。

    调用方必须在**主线程**调用；窗口关闭后 ``webview.start()`` 返回，这里再调
    ``on_close``（launcher 用它把 uvicorn 停掉）。

    任何异常都往上抛：launcher 要区分「窗口起不来」（可以退回浏览器）和
    「窗口正常关闭」。
    """
    import webview  # noqa: PLC0415 - 只有真要开窗口时才 import

    store = storage_dir(root)
    store.mkdir(parents=True, exist_ok=True)
    width, height = window_size()

    window = webview.create_window(
        title or TITLE_PREFIX,
        url,
        width=width,
        height=height,
        min_size=(900, 600),
    )
    _install_external_open(window)

    try:
        # private_mode=False + storage_path：要 localStorage 持久化，同时不让它
        # 退出时去删临时目录（那一步实测会报 WinError 5）。
        # gui 显式指定：renderer() 已经确认是 edgechromium，这里再钉一次，
        # 万一机器状态变了也不会悄悄退化成 IE 内核。
        webview.start(private_mode=False, storage_path=str(store), gui="edgechromium")
    finally:
        if on_close is not None:
            try:
                on_close()
            except Exception:  # noqa: BLE001
                pass
    return 0
