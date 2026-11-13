# -*- coding: utf-8 -*-
"""服务端原生文件对话框的验收脚本（不需要真的弹窗）。

为什么不能直接点：对话框是模态的，脚本点不了它的按钮。所以这里把「弹框那一步」
换成假的子进程，只验证**我们自己写的那部分逻辑**是否正确：

1. 扩展名归一化：``.pt`` / ``pt`` / ``*.pt`` 三种写法都要变成 ``*.pt``，
   且末尾必须有一条「所有文件」——否则用户选非白名单后缀时会以为目录是空的；
2. 起始目录：字段里是文件路径时要取父目录，不能把文件当目录传给系统框；
3. 四种类型（file / files / dir / save）都要映射到对应的 payload，不能串；
4. 单例锁：第二个请求必须立刻失败（``DialogBusy``），不能真的开第二个框；
5. 超时：到点按取消处理，不抛异常、不留下临时文件；
6. 子进程异常：返回 ``cancelled=True`` 并带上原因，让前端能回退；
7. 取消不写记忆目录；成功才记，且记住的是目录不是文件；
8. 白名单：``config.allow_path`` 之后该目录可访问，``clear_runtime_allowed`` 后失效；
9. 接口层：非本机来源必须 403（框会弹在服务端屏幕上），本机才放行并登记白名单；
10. ``/api/capabilities`` 必须带 ``native_dialog`` 字段，前端据此决定回退。

用法：
    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_native_dialog.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import nativedialog  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


class FakeRun:
    """替身 subprocess.run：把结果写进 result 文件，模拟用户的操作。

    ``outcome`` 决定行为：``paths`` 给出选中项，``error`` 造非零退出，
    ``timeout`` 抛 TimeoutExpired，``slow`` 让并发测试能撞上锁。
    """

    def __init__(self, paths: list[str] | None = None, *, error: str = "", timeout: bool = False, slow: float = 0.0):
        self.paths = paths or []
        self.error = error
        self.timeout = timeout
        self.slow = slow
        self.calls: list[list[str]] = []
        self.payloads: list[dict] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        # 临时目录在 select() 结束时就被清掉了，payload 必须在调用当场读出来
        try:
            self.payloads.append(json.loads(Path(args[-2]).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.payloads.append({})
        if self.slow:
            time.sleep(self.slow)
        if self.timeout:
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 1))
        if self.error:
            return subprocess.CompletedProcess(args, 1, b"", self.error.encode("utf-8"))
        result_path = Path(args[-1])
        result_path.write_text(json.dumps({"paths": self.paths}, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def payload(self, index: int = -1) -> dict:
        """取第 index 次调用真正发给子进程的 payload。"""
        return self.payloads[index]


def with_fake(fake: FakeRun, fn):
    """临时替换 subprocess.run，执行 fn，再还原。"""
    original = nativedialog.subprocess.run
    nativedialog.subprocess.run = fake
    try:
        return fn()
    finally:
        nativedialog.subprocess.run = original


def test_filetypes() -> None:
    groups = nativedialog.build_filetypes([".pt", "pth", "*.onnx"], "权重")
    patterns = [item[1] for item in groups]
    check(
        "扩展名：三种写法归一化成同一组",
        "*.pt *.pth *.onnx" in patterns,
        str(patterns),
    )
    check("扩展名：末尾有「所有文件」", groups[-1] == ["所有文件", "*.*"], str(groups[-1]))

    empty = nativedialog.build_filetypes([])
    check("扩展名：空列表只剩「所有文件」", len(empty) == 1 and empty[0][1] == "*.*", str(empty))

    dedup = nativedialog.build_filetypes([".yaml", "*.yaml", "yaml"])
    check("扩展名：重复项只留一份", len(dedup) == 2, str(dedup))


def test_initial_dir(tmp: Path) -> None:
    nativedialog.STATE_FILE = tmp / "state.json"
    nativedialog.reset_state()

    sample = tmp / "weights" / "best.pt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"x")
    found = nativedialog.initial_dir("expWeights", str(sample))
    check("起始目录：文件路径取父目录", Path(found) == sample.parent, found)

    folder = tmp / "datasets"
    folder.mkdir()
    check("起始目录：目录原样使用", Path(nativedialog.initial_dir("x", str(folder))) == folder)

    nativedialog._remember("expWeights", str(sample))
    check(
        "起始目录：记住的是目录不是文件",
        Path(nativedialog.initial_dir("expWeights", "")) == sample.parent,
        nativedialog.initial_dir("expWeights", ""),
    )
    check("起始目录：未知字段且无输入时留空", nativedialog.initial_dir("nope", "") == "")


def test_kinds() -> None:
    expected = {
        "file": "file",
        "files": "files",
        "dir": "dir",
        "save": "save",
    }
    for kind in expected:
        fake = FakeRun([str(Path(os.getcwd()) / "a.txt")])
        with_fake(fake, lambda: nativedialog.select(kind, field="k", patterns=[".pt"]))
        payload = fake.payload()
        check(f"类型：{kind} 原样传给子进程", payload.get("kind") == kind, json.dumps(payload, ensure_ascii=False)[:200])

    try:
        nativedialog.select("bogus")
        check("类型：未知类型报错", False, "没有抛异常")
    except ValueError:
        check("类型：未知类型报错", True)


def test_normalize(tmp: Path) -> None:
    """tkinter 回传的是正斜杠，字段里要跟 Qt 的写法一致（反斜杠）。"""
    nativedialog.STATE_FILE = tmp / "state.json"
    fake = FakeRun(["D:/data/weights/best.pt"])
    got = with_fake(fake, lambda: nativedialog.select("file", field="norm"))
    check(
        "返回路径：统一成 Windows 分隔符",
        got["paths"] == [str(Path("D:/data/weights/best.pt"))],
        str(got["paths"]),
    )


def test_single_flight(tmp: Path) -> None:
    nativedialog.STATE_FILE = tmp / "state.json"
    slow = FakeRun([str(tmp / "a.bin")], slow=0.6)
    results: list[str] = []

    def first():
        with_fake(slow, lambda: nativedialog.select("file", field="a"))
        results.append("first-done")

    thread = threading.Thread(target=first)
    thread.start()
    time.sleep(0.15)
    busy = False
    try:
        with_fake(FakeRun([str(tmp / "b.bin")]), lambda: nativedialog.select("file", field="b"))
    except nativedialog.DialogBusy:
        busy = True
    thread.join()
    check("并发：第二个请求被拒（不能开两个框）", busy, "第二个请求也弹了框")
    check("并发：第一个请求正常完成", results == ["first-done"], str(results))


def test_failures(tmp: Path) -> None:
    nativedialog.STATE_FILE = tmp / "state.json"

    timeout = FakeRun([], timeout=True)
    got = with_fake(timeout, lambda: nativedialog.select("file", field="t"))
    check(
        "超时：按取消处理并给出原因",
        got["cancelled"] is True and got.get("reason") == "timeout",
        json.dumps(got, ensure_ascii=False),
    )

    broken = FakeRun([], error="no display name and no $DISPLAY environment variable")
    got = with_fake(broken, lambda: nativedialog.select("file", field="e"))
    check(
        "子进程异常：返回 cancelled 并带错误信息",
        got["cancelled"] is True and "DISPLAY" in (got.get("message") or ""),
        json.dumps(got, ensure_ascii=False),
    )

    # 进程正常退出但没写结果文件（被系统强杀）
    no_result = FakeRun([])
    original = nativedialog.subprocess.run

    def run_without_result(args, **kwargs):
        no_result.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, b"", b"")

    nativedialog.subprocess.run = run_without_result
    try:
        got = nativedialog.select("file", field="n")
    finally:
        nativedialog.subprocess.run = original
    check(
        "无结果文件：reason=no-result",
        got.get("reason") == "no-result",
        json.dumps(got, ensure_ascii=False),
    )


def test_reason_and_cancel(tmp: Path) -> None:
    """子进程带回的 reason 必须原样交给前端，否则「没弹出窗口」会被当成用户取消。"""
    nativedialog.STATE_FILE = tmp / "state.json"
    original = nativedialog.subprocess.run

    def run_with_reason(args, **kwargs):
        result_path = Path(args[-1])
        result_path.write_text(
            json.dumps({"paths": [], "reason": "dialog-not-shown"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, b"", b"")

    nativedialog.subprocess.run = run_with_reason
    try:
        got = nativedialog.select("file", field="reason")
    finally:
        nativedialog.subprocess.run = original
    check(
        "没弹出窗口：reason 传回调用方",
        got.get("cancelled") is True and got.get("reason") == "dialog-not-shown",
        json.dumps(got, ensure_ascii=False),
    )

    flag = tmp / "cancel.flag"
    nativedialog._ACTIVE_CANCEL["path"] = flag
    try:
        ok = nativedialog.request_cancel()
    finally:
        nativedialog._ACTIVE_CANCEL["path"] = None
    check("取消：没有进行中的框时不写文件", nativedialog.request_cancel() is False)
    check("取消：进行中时写出标记", ok and flag.read_text(encoding="utf-8") == "cancel", str(ok))


def test_remember(tmp: Path) -> None:
    nativedialog.STATE_FILE = tmp / "state.json"
    nativedialog.reset_state()
    picked = tmp / "picked"
    picked.mkdir()
    target = picked / "shot.png"
    target.write_bytes(b"x")

    # 只比「本次运行新增的临时目录」：系统临时目录里可能躺着上次崩溃的残留，
    # 拿全局 glob 去断言会误报。
    before = {item.name for item in Path(tempfile.gettempdir()).glob("tf_dialog_*")}

    fake = FakeRun([str(target)])
    with_fake(fake, lambda: nativedialog.select("file", field="annoAddFiles"))
    state = nativedialog._load_state()
    check(
        "记忆：成功后记住所选文件所在目录",
        Path(state.get("annoAddFiles", "")) == picked,
        str(state),
    )

    cancel = FakeRun([])
    with_fake(cancel, lambda: nativedialog.select("file", field="other"))
    check("记忆：取消不记录", "other" not in nativedialog._load_state(), str(nativedialog._load_state()))

    after = {item.name for item in Path(tempfile.gettempdir()).glob("tf_dialog_*")}
    check("临时文件：跑完不留新的 tf_dialog_ 目录", after == before, str(after - before))


def test_allowlist(tmp: Path) -> None:
    outside = tmp / "outside"
    outside.mkdir()
    config.clear_runtime_allowed()
    check("白名单：仓库外目录默认拒绝", not config.is_path_allowed(outside))
    config.allow_path(outside)
    check("白名单：允许后可以访问", config.is_path_allowed(outside / "a.txt"))
    config.clear_runtime_allowed()
    check("白名单：清空后重新拒绝", not config.is_path_allowed(outside))

    # 传文件路径时放行的是它所在目录，否则父目录仍然被拒
    config.allow_path(outside / "a.txt")
    check("白名单：传文件路径时按父目录放行", config.is_path_allowed(outside / "b.txt"))
    config.clear_runtime_allowed()


def test_api(tmp: Path) -> None:
    from fastapi.testclient import TestClient

    import app as webapp

    prior_roots = os.environ.get("TRTFORGE_BROWSE_ROOTS", "")
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(BACKEND.parent)
    config.clear_runtime_allowed()
    try:
        client = TestClient(webapp.app)

        caps = client.get("/api/capabilities")
        check("接口：capabilities 带 native_dialog", "native_dialog" in caps.json(), caps.text[:200])

        # TestClient 的来源是 "testclient"，必须被当成远程拒掉
        remote = client.post("/api/browse/dialog", json={"kind": "file"})
        check("接口：非本机来源 403", remote.status_code == 403, f"{remote.status_code} {remote.text[:200]}")

        class FakeRequest:
            def __init__(self, host: str):
                self.client = type("C", (), {"host": host})()

        for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
            check(f"接口：{host} 视为本机", webapp._is_local_request(FakeRequest(host)))
        for host in ("192.168.1.9", "10.0.0.2", "testclient", ""):
            check(f"接口：{host or '(空)'} 视为远程", not webapp._is_local_request(FakeRequest(host)))

        # 本机路径：把 select 换成替身，验证「选中即放行」这条链路
        outside = tmp / "api_outside"
        outside.mkdir()
        picked = outside / "best.pt"
        picked.write_bytes(b"x")

        original_select = webapp.nativedialog.select
        original_local = webapp._is_local_request
        webapp.nativedialog.select = lambda *a, **k: {"paths": [str(picked)], "cancelled": False, "elapsed_ms": 5}
        webapp._is_local_request = lambda request: True
        try:
            response = client.post(
                "/api/browse/dialog",
                json={"kind": "file", "field": "expWeights", "patterns": [".pt"]},
            )
        finally:
            webapp.nativedialog.select = original_select
            webapp._is_local_request = original_local

        body = response.json()
        check("接口：本机来源返回选中路径", body.get("paths") == [str(picked)], response.text[:300])
        check(
            "接口：选中路径自动放行",
            any(Path(item) == outside for item in body.get("allowed") or []),
            response.text[:300],
        )
        check("接口：放行后图片接口不再 403", config.is_path_allowed(picked))
        config.clear_runtime_allowed()

        webapp.nativedialog._ACTIVE_CANCEL["path"] = tmp / "api-cancel.flag"
        webapp._is_local_request = lambda request: True
        try:
            cancelled = client.post("/api/browse/cancel")
        finally:
            webapp.nativedialog._ACTIVE_CANCEL["path"] = None
            webapp._is_local_request = original_local
        check(
            "接口：取消会写出标记",
            cancelled.status_code == 200 and (tmp / "api-cancel.flag").is_file(),
            cancelled.text[:300],
        )
    finally:
        if prior_roots:
            os.environ["TRTFORGE_BROWSE_ROOTS"] = prior_roots
        else:
            os.environ.pop("TRTFORGE_BROWSE_ROOTS", None)


def test_packaged_command(tmp: Path) -> None:
    """安装包必须走 --tool，不能把 -c 丢给 TRTForgeWeb.exe。"""
    request = tmp / "request.json"
    result = tmp / "result.json"
    saved = os.environ.get("TRTFORGE_TOOL_EXE")
    os.environ.pop("TRTFORGE_TOOL_EXE", None)
    try:
        dev = nativedialog.dialog_command(request, result)
        check("开发环境：用 -c 跑弹窗脚本", dev[1] == "-c", str(dev[:2]))
        check("开发环境：请求和结果文件在命令末尾", dev[-2:] == [str(request), str(result)], str(dev[-2:]))

        os.environ["TRTFORGE_TOOL_EXE"] = r"D:\TRTForge\app\TRTForgeWeb.exe"
        packed = nativedialog.dialog_command(request, result)
        check(
            "安装包：用 --tool native_dialog，而不是 -c",
            packed == [
                r"D:\TRTForge\app\TRTForgeWeb.exe",
                "--tool",
                "native_dialog",
                str(request),
                str(result),
            ],
            str(packed),
        )
    finally:
        if saved is None:
            os.environ.pop("TRTFORGE_TOOL_EXE", None)
        else:
            os.environ["TRTFORGE_TOOL_EXE"] = saved

    entry = (Path(__file__).resolve().parent.parent / "frozen_entry.py").read_text(encoding="utf-8")
    check("安装包入口认识 native_dialog", 'name == "native_dialog"' in entry, "frozen_entry.py 未处理该工具")


def main() -> int:
    print(f"原生对话框可用：{nativedialog.available()}  解释器：{nativedialog.interpreter()}")
    check("环境：Windows 上应可用", nativedialog.available() or os.name != "nt")

    with tempfile.TemporaryDirectory(prefix="tf_dialog_test_") as temp:
        tmp = Path(temp)
        test_filetypes()
        test_initial_dir(tmp)
        test_kinds()
        test_normalize(tmp)
        test_single_flight(tmp)
        test_failures(tmp)
        test_reason_and_cancel(tmp)
        test_remember(tmp)
        test_allowlist(tmp)
        test_api(tmp)
        test_packaged_command(tmp)

    print()
    if FAILED:
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        for name in FAILED:
            print(f"  失败：{name}")
        return 1
    print(f"通过 {len(PASSED)} 项，失败 0 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
