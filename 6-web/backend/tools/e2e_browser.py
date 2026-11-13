# -*- coding: utf-8 -*-
"""浏览器 e2e（可选）：真开一个 Chromium，把首页与标注页加载一遍。

什么时候用它：
- `test_full_chain.py` 只验接口契约，验不了「按钮绑没绑上」「模块路径 404」；
- 前端拆模块后最容易出的事故是 import 路径写错，浏览器控制台报一个 404 就白屏，
  这种问题只有真加载页面才能发现。

前置（不装也能跑，脚本会自己跳过）：
    pip install playwright
    playwright install chromium

用法：
    python 6-web/backend/tools/e2e_browser.py                 # 自动挑空闲端口
    python 6-web/backend/tools/e2e_browser.py --port 8123 --keep
    python 6-web/backend/tools/e2e_browser.py --headed        # 看得见浏览器（排障用）

注意：CI 里**不要**调用它——下载 Chromium 要几百 MB，且无头浏览器在容器里容易因
字体/沙箱失败，把它当成本地回归工具更划算。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent.parent
SHOTS = REPO / "runtime" / "e2e"
CONSOLE_LIMIT = 40


def free_port() -> int:
    """让系统分配一个空闲端口，避免和正在跑的开发服务撞车。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(url: str, timeout: float = 40.0) -> bool:
    """轮询 /api/health 直到起来：uvicorn 启动时间在不同机器上差很多。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - 还没起来就是连接失败，继续等
            pass
        time.sleep(0.4)
    return False


def start_server(port: int) -> subprocess.Popen:
    """起一个后台 uvicorn；日志走 stdout，失败时打印尾部。"""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(BACKEND),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class Failure(Exception):
    """某一个检查点没通过。"""


def _shot_on_failure(page) -> None:
    """失败时截一张图：只留控制台日志往往看不出"卡在哪个空白页"。"""
    if page is None:
        return
    target = SHOTS / "fail.png"
    try:
        page.screenshot(path=str(target), full_page=True)
        print(f"  失败截图：{target}")
    except Exception as exc:  # noqa: BLE001 - 页面已崩时截图也会失败，不能盖住原始错误
        print(f"  （截图失败：{type(exc).__name__}: {exc}）")


def _dump_console(console: list[str]) -> None:
    """失败时把控制台摘要落盘并打印尾部：报错信息是排障的全部线索。"""
    if not console:
        print("  （控制台没有输出）")
        return
    tail = console[-CONSOLE_LIMIT:]
    target = SHOTS / "console.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(tail), encoding="utf-8")
    print(f"  控制台摘要（末尾 {len(tail)} 条，已写入 {target}）：")
    for line in tail[-10:]:
        print(f"    {line}")


def run(port: int, headed: bool, keep: bool) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("跳过：未安装 playwright（pip install playwright && playwright install chromium）")
        return 0

    SHOTS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="trtforge_e2e_"))
    # 白名单默认只有仓库根；e2e 要用临时目录当工作区，必须显式放行
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(REPO)

    url = f"http://127.0.0.1:{port}"
    server = start_server(port)
    console: list[str] = []
    #: 失败时要截图，但 page 只在 with 块里存在，所以用一个可变容器往外传
    page_ref: dict[str, object] = {"page": None}
    passed = 0
    try:
        if not wait_health(url):
            print("✗ 服务没起来；uvicorn 输出：")
            server.terminate()
            out = (server.stdout.read() if server.stdout else b"").decode("utf-8", "replace")
            print(out[-2000:])
            return 1
        print(f"服务已就绪：{url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page_ref["page"] = page
            page.on("console", lambda msg: console.append(f"[{msg.type}] {msg.text}"))
            page.on("pageerror", lambda err: console.append(f"[pageerror] {err}"))

            def must(condition: bool, message: str) -> None:
                if not condition:
                    raise Failure(message)

            # 1. 首页与模块加载：拆分后的 js/annotate/*.js 必须都能 200，
            #    任何一个 404 都会让整页白屏，所以先看控制台有没有脚本错误。
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("#panel-train.active", timeout=15000)
            errors = [line for line in console if line.startswith("[pageerror]")]
            must(not errors, f"页面加载有脚本错误：{errors[:3]}")
            passed += 1
            print("  ✓ 首页加载完成，无脚本错误")

            # 2. 切到标注页：页签靠 .tab[data-tab] 切换，没有 hash 路由
            page.click('.tab[data-tab="annotate"]')
            page.wait_for_selector("#panel-annotate.active", timeout=10000)
            page.wait_for_selector(".anno-layout", timeout=10000)
            must(page.locator("#panel-annotate").is_visible(), "标注面板不可见")
            passed += 1
            print("  ✓ 标注页三栏渲染正常")

            # 3. 关键控件都在（拆分模块后 id 不能被改掉）
            for selector in ["#annoPath", "#annoUndo", "#annoRedo", "#annoSaveNow"]:
                must(page.locator(selector).count() > 0, f"找不到控件 {selector}")
            passed += 1
            print("  ✓ 标注关键控件齐全")

            # 4. 切到推理页：验证另一个模块的按需加载没被拆坏
            page.click('.tab[data-tab="infer"]')
            page.wait_for_selector("#panel-infer.active", timeout=10000)
            passed += 1
            print("  ✓ 推理页可切换")

            page.screenshot(path=str(SHOTS / "annotate.png"), full_page=True)
            browser.close()
    except Failure as exc:
        print(f"✗ e2e 失败（{passed} 个检查点已通过）：{exc}")
        _shot_on_failure(page_ref.get("page"))
        _dump_console(console)
        return 1
    except Exception as exc:  # noqa: BLE001 - playwright 的各种超时都走这里
        print(f"✗ e2e 异常：{type(exc).__name__}: {exc}")
        _shot_on_failure(page_ref.get("page"))
        _dump_console(console)
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - 只能强杀
            server.kill()
        if not keep:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    print(f"✓ e2e 通过：{passed} 个检查点")
    print(f"  截图目录：{SHOTS}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="TRTForge Web 浏览器 e2e（可选）")
    parser.add_argument("--port", type=int, default=0, help="服务端口，默认自动挑空闲端口")
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口")
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    args = parser.parse_args(argv[1:])

    port = args.port or free_port()
    print(f"浏览器 e2e（端口 {port}）")
    return run(port, args.headed, args.keep)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
