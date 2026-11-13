# -*- coding: utf-8 -*-
"""启动器测试（launcher.py + frozen_entry 的入口分派）。

    python tools/test_launcher.py

测什么
------
1. **地址解析**：空配置走默认 8060、非法端口回落、`0.0.0.0` 只用于监听不用于浏览器。
2. **配置容错**：文件缺失、JSON 坏掉、记事本存的 BOM，都不能让软件起不来。
3. **环境变量清单**：精确匹配一份固定名单 —— 这是「不准再出现第二份实现」的硬约束，
   不是「大概包含几个键」。同时断言 `start_user.ps1` 里已经没有 `$env:TRTFORGE_BIN_DIR`
   之类，防止有人把设置搬回脚本。
4. **单实例语义**：同名互斥量第二次拿到「已存在」，释放后能重新创建。
5. **端口三态**：空闲 / 上是本软件 / 被别的程序占用，三种结论必须能区分开。
6. **无控制台**：`sys.stdout is None` 时 `ensure_stdio()` 不抛异常且能写出内容。
7. **入口分派**：`--tool` 不进启动器，无参数才进。

所有用例都在临时目录里造数据，不写开发机的真实日志；
会真的起 HTTP 服务的用例只用回环地址和随机端口，跑完就关。
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND.parent.parent
sys.path.insert(0, str(BACKEND))

import launcher  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def case(name: str):
    def wrap(func):
        def runner():
            try:
                func()
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                print(f"  [FAIL] {name}")
            else:
                PASSED.append(name)
                print(f"  [ ok ] {name}")

        runner.__name__ = func.__name__
        return runner

    return wrap


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}：期望 {expected!r}，实际 {actual!r}")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def free_port(host: str = "127.0.0.1") -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
        if self.path.rstrip("/") == "/api/health":
            body = json.dumps({"ok": True, "version": "9.9.9"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):  # 测试输出里不要 HTTP 访问日志
        pass


class FakeHealthServer:
    """假装「端口上是本软件的旧实例」。"""

    def __init__(self, host: str = "127.0.0.1"):
        self.port = free_port(host)
        self._server = ThreadingHTTPServer((host, self.port), _HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeHealthServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class FakeListener:
    """假装「端口被别的程序占着」：只听不说，HTTP 请求拿不到合法响应。"""

    def __init__(self, host: str = "127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, 0))
        self.port = int(self.sock.getsockname()[1])
        self.sock.listen(4)

    def __enter__(self) -> "FakeListener":
        return self

    def __exit__(self, *exc_info) -> None:
        self.sock.close()


def make_root(base: Path, **cfg) -> Path:
    root = base / "install"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# 1. 地址解析
# ---------------------------------------------------------------------------

@case("空配置走默认 127.0.0.1:8060")
def test_default_address():
    check_eq(launcher.resolve_address({}), ("127.0.0.1", 8060), "默认地址")


@case("配置里的 host / port 生效")
def test_config_address():
    cfg = {"host": "0.0.0.0", "port": 9000}
    check_eq(launcher.resolve_address(cfg), ("0.0.0.0", 9000), "配置地址")


@case("非法端口一律回落 8060")
def test_bad_port():
    for bad in ("abc", 0, -1, 70000, None, []):
        check_eq(launcher.resolve_address({"port": bad})[1], 8060, f"port={bad!r}")


@case("host 留空回落 127.0.0.1")
def test_empty_host():
    check_eq(launcher.resolve_address({"host": "   "})[0], "127.0.0.1", "空白 host")


@case("0.0.0.0 只用于监听，浏览器仍打 127.0.0.1")
def test_browse_host():
    check_eq(launcher.browse_host("0.0.0.0"), "127.0.0.1", "0.0.0.0")
    check_eq(launcher.browse_host("::"), "127.0.0.1", "::")
    check_eq(launcher.browse_host("192.168.1.9"), "192.168.1.9", "局域网地址")
    check_eq(launcher.page_url("0.0.0.0", 8060), "http://127.0.0.1:8060/", "页面地址")


# ---------------------------------------------------------------------------
# 2. 配置容错
# ---------------------------------------------------------------------------

@case("config.json 不存在时不抛异常")
def test_config_missing():
    with tempfile.TemporaryDirectory() as tmp:
        check_eq(launcher.load_config(Path(tmp)), {}, "缺失配置")


@case("config.json 是坏 JSON 时返回空字典")
def test_config_broken():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text("{ this is not json", encoding="utf-8")
        check_eq(launcher.load_config(Path(tmp)), {}, "坏 JSON")


@case("config.json 是数组而不是对象时返回空字典")
def test_config_not_object():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text("[1, 2]", encoding="utf-8")
        check_eq(launcher.load_config(Path(tmp)), {}, "非对象")


@case("记事本存的 UTF-8 BOM 配置能读")
def test_config_bom():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text('{"port": 8070}', encoding="utf-8-sig")
        check_eq(launcher.load_config(Path(tmp)).get("port"), 8070, "BOM 配置")


@case("TRTFORGE_HOME 优先作为安装根")
def test_install_root_env():
    with tempfile.TemporaryDirectory() as tmp:
        saved = os.environ.get("TRTFORGE_HOME")
        os.environ["TRTFORGE_HOME"] = tmp
        try:
            check_eq(launcher.install_root(), Path(tmp).resolve(), "安装根")
        finally:
            if saved is None:
                os.environ.pop("TRTFORGE_HOME", None)
            else:
                os.environ["TRTFORGE_HOME"] = saved


# ---------------------------------------------------------------------------
# 3. 环境变量清单
# ---------------------------------------------------------------------------

#: 一份精确名单。少一个环境变量 = 某个功能静默失效（找不到前端、找不到 trtexec），
#: 多一个 = 有人又开了一条没文档的配置通道。所以断言「完全相等」，不是「包含」。
BASE_ENV_KEYS = {
    "TRTFORGE_HOME",
    "TRTFORGE_REPO_ROOT",
    "TRTFORGE_FRONTEND_DIR",
    "TRTFORGE_BIN_DIR",
    "TRTFORGE_YOLO_LAUNCHER_DIR",
    "TRTFORGE_ANOMALIB_LAUNCHER_DIR",
    "TRTFORGE_MODEL_DIR",
    "TRTFORGE_OCR_MODEL_DIR",
    "TRTFORGE_TRTEXEC",
    "TRTFORGE_TOOL_EXE",
    "OPENCV_BIN_DIR",
    "CUDA_BIN_DIR",
    "TRTFORGE_WEB_PORT",
    "TRTFORGE_WEB_HOST",
    "TRTFORGE_BROWSE_ROOTS",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
}


@case("环境变量清单与名单完全一致")
def test_env_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp))
        env = launcher.environment(root, launcher.load_config(root))
    check_eq(set(env), BASE_ENV_KEYS, "环境变量名单")


@case("环境变量值指向安装目录下的真实位置")
def test_env_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp), port=8088, host="0.0.0.0")
        env = launcher.environment(root, launcher.load_config(root))
    check_eq(env["TRTFORGE_HOME"], str(root), "HOME")
    check_eq(env["TRTFORGE_FRONTEND_DIR"], str(root / "frontend"), "前端目录")
    check_eq(env["TRTFORGE_BIN_DIR"], str(root / "bin"), "bin 目录")
    check_eq(env["TRTFORGE_OCR_MODEL_DIR"], str(root / "models" / "ocr"), "OCR 模型目录")
    check_eq(env["TRTFORGE_TRTEXEC"], str(root / "bin" / "trtexec.exe"), "trtexec")
    check_eq(env["TRTFORGE_TOOL_EXE"], str(root / "app" / "TRTForgeWeb.exe"), "自身 exe")
    check_eq(env["TRTFORGE_WEB_PORT"], "8088", "端口")
    check_eq(env["TRTFORGE_WEB_HOST"], "0.0.0.0", "监听地址")
    check_eq(env["PYTHONIOENCODING"], "utf-8", "输出编码")
    # 托盘图标和中文日志都指望它是 utf-8，写死能防止有人「优化」掉
    check_eq(env["PYTHONUNBUFFERED"], "1", "不缓冲")


@case("train-env\\python.exe 存在才写 TRTFORGE_PYTHON")
def test_env_python():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp))
        check("TRTFORGE_PYTHON" not in launcher.environment(root, {}), "没有训练环境时不该写")
        (root / "train-env").mkdir()
        (root / "train-env" / "python.exe").write_bytes(b"")
        env = launcher.environment(root, {})
        check_eq(env.get("TRTFORGE_PYTHON"), str(root / "train-env" / "python.exe"), "内置训练环境")


@case("config.json 里的 python 优先于内置训练环境")
def test_env_python_override():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp), python=r"D:\Custom\python.exe")
        (root / "train-env").mkdir()
        (root / "train-env" / "python.exe").write_bytes(b"")
        check_eq(launcher.environment(root, launcher.load_config(root)).get("TRTFORGE_PYTHON"),
                 r"D:\Custom\python.exe", "配置优先")


@case("TCL_LIBRARY / TK_LIBRARY 只在目录存在时设置")
def test_env_tcl():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp))
        env = launcher.environment(root, {})
        check("TCL_LIBRARY" not in env, "没有 tcl8.6 时不该写")
        (root / "app" / "tcl8.6").mkdir(parents=True)
        (root / "app" / "tk8.6").mkdir(parents=True)
        env = launcher.environment(root, {})
        check_eq(env.get("TCL_LIBRARY"), str(root / "app" / "tcl8.6"), "TCL_LIBRARY")
        check_eq(env.get("TK_LIBRARY"), str(root / "app" / "tk8.6"), "TK_LIBRARY")


@case("BROWSE_ROOTS 用系统路径分隔符连接")
def test_env_browse_roots():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp), data_roots=[r"D:\data\a", r"E:\data\b"])
        env = launcher.environment(root, launcher.load_config(root))
        check_eq(env["TRTFORGE_BROWSE_ROOTS"], os.pathsep.join([r"D:\data\a", r"E:\data\b"]),
                 "浏览根目录")
        root2 = make_root(Path(tmp) / "second")
        check_eq(launcher.environment(root2, {})["TRTFORGE_BROWSE_ROOTS"], "", "没有配置时为空串")


@case("apply_environment 把 bin 放到 PATH 最前且保留原 PATH")
def test_apply_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(Path(tmp), port=8099)
        saved = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = r"C:\Windows\system32"
            host, port = launcher.apply_environment(root, launcher.load_config(root))
            check_eq((host, port), ("127.0.0.1", 8099), "返回值")
            path = os.environ["PATH"]
            check(path.startswith(str(root / "bin") + os.pathsep), f"bin 不在最前：{path}")
            check(r"C:\Windows\system32" in path, "原 PATH 被覆盖")
        finally:
            os.environ["PATH"] = saved
            for key in BASE_ENV_KEYS | {"TRTFORGE_PYTHON"}:
                os.environ.pop(key, None)


@case("start_user.ps1 不再设置环境变量（只有启动器一份实现）")
def test_ps1_no_env():
    text = (REPO_ROOT / "packaging" / "templates" / "start_user.ps1").read_text(encoding="utf-8")
    # 这两项是解释器行为，且 TRTFORGE_HOME 必须由外层先给（否则 exe 找不到安装根），
    # 出现在排障脚本里不算配置漂移。其余一律不许再设。
    allowed = {"PYTHONIOENCODING", "PYTHONUNBUFFERED", "TRTFORGE_HOME"}
    for key in sorted(BASE_ENV_KEYS - allowed):
        check(f"$env:{key}" not in text, f"start_user.ps1 里又出现了 $env:{key}")
    check("TRTFORGE_LEGACY_START" in text, "排障脚本应显式走回退模式")


@case("start_web_cmd.bat 不再自带端口配置")
def test_cmd_bat_no_dup_config():
    raw = (REPO_ROOT / "packaging" / "templates" / "start_web_cmd.bat").read_text(encoding="utf-8")
    check("set \"PORT=" not in raw, "start_web_cmd.bat 又写死了 PORT")
    check("start_web.bat" in raw, "start_web_cmd.bat 应转调 start_web.bat")


# ---------------------------------------------------------------------------
# 4. 单实例
# ---------------------------------------------------------------------------

@case("同名互斥量第二次拿到「已存在」")
def test_mutex():
    if os.name != "nt":
        return
    name = f"Local\\TRTForgeTest-{uuid.uuid4().hex}"
    handle1, created1 = launcher.acquire_mutex(name)
    try:
        check(created1, "第一次创建应成功")
        handle2, created2 = launcher.acquire_mutex(name)
        try:
            check(not created2, "第二次应识别为已存在")
        finally:
            launcher.release_mutex(handle2)
    finally:
        launcher.release_mutex(handle1)
    # 全部释放后应当能重新创建，否则「关掉程序再打开」会一直判成已有实例
    handle3, created3 = launcher.acquire_mutex(name)
    try:
        check(created3, "释放后应能重新创建")
    finally:
        launcher.release_mutex(handle3)


@case("不同名字的互斥量互不影响")
def test_mutex_distinct():
    if os.name != "nt":
        return
    handle_a, created_a = launcher.acquire_mutex(f"Local\\TRTForgeTest-{uuid.uuid4().hex}")
    handle_b, created_b = launcher.acquire_mutex(f"Local\\TRTForgeTest-{uuid.uuid4().hex}")
    try:
        check(created_a and created_b, "不同用户会话应各自能开一个实例")
    finally:
        launcher.release_mutex(handle_a)
        launcher.release_mutex(handle_b)


# ---------------------------------------------------------------------------
# 5. 端口三态
# ---------------------------------------------------------------------------

@case("端口空闲：port_free 为真")
def test_port_free():
    port = free_port()
    check(launcher.port_free("127.0.0.1", port), "空端口应可绑定")
    check_eq(launcher.health_version("127.0.0.1", port, timeout=0.5), None, "空端口读不到版本")


@case("端口上是本软件：port_free 为假但能读到版本")
def test_port_our_instance():
    with FakeHealthServer() as server:
        check(not launcher.port_free("127.0.0.1", server.port), "已被监听应不可绑定")
        check_eq(launcher.health_version("127.0.0.1", server.port), "9.9.9", "版本号")


@case("端口被别的程序占用：port_free 为假且读不到版本")
def test_port_other_app():
    with FakeListener() as listener:
        check(not launcher.port_free("127.0.0.1", listener.port), "被占用应不可绑定")
        check_eq(launcher.health_version("127.0.0.1", listener.port, timeout=1.0), None, "版本号")


@case("健康检查返回非 JSON 时当作读不到")
def test_health_not_json():
    class Bad(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html>not api</html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Bad)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        check_eq(launcher.health_version("127.0.0.1", port), None, "非 JSON 响应")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@case("wait_healthy 超时返回 False 而不是抛异常")
def test_wait_healthy_timeout():
    port = free_port()
    check(not launcher.wait_healthy("127.0.0.1", port, timeout=1.0), "超时应返回 False")


@case("端口占用文案里带端口号和 config.json 路径")
def test_port_text():
    text = launcher._port_in_use_text(Path(r"D:\TRTForgeWeb"), 8060)
    check("8060" in text, "没写端口")
    check("8061" in text, "没给可改的替代端口")
    check("config.json" in text, "没指路到 config.json")
    check("已被其它程序占用" in text, "第一段没说清发生了什么")


# ---------------------------------------------------------------------------
# 6. 无控制台
# ---------------------------------------------------------------------------

@case("sys.stdout/stderr 为 None 时 ensure_stdio 不抛且能写")
def test_ensure_stdio_none():
    with tempfile.TemporaryDirectory() as tmp:
        log = launcher.Log(Path(tmp))
        log.open()
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout = None
        sys.stderr = None
        try:
            launcher.ensure_stdio(log)  # 不抛异常就是通过一半
            check(sys.stdout is not None, "stdout 没被重定向")
            check(sys.stderr is not None, "stderr 没被重定向")
            print("启动器测试：中文输出")
            sys.stdout.flush()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
            log.close()
        content = (Path(tmp) / "runtime" / "logs" / "launcher.log").read_text(encoding="utf-8")
        check("中文输出" in content, "重定向没写进日志文件")


@case("有控制台时 ensure_stdio 不改动 stdout")
def test_ensure_stdio_keep():
    saved_out, saved_err = sys.stdout, sys.stderr
    launcher.ensure_stdio(None)
    check_eq(sys.stdout, saved_out, "stdout")
    check_eq(sys.stderr, saved_err, "stderr")


@case("日志用 UTF-8 追加写，含时间戳")
def test_log_utf8():
    with tempfile.TemporaryDirectory() as tmp:
        log = launcher.Log(Path(tmp))
        log.open()
        log.write("安装根=C:\\软件\\TRTForgeWeb")
        log.close()
        path = Path(tmp) / "runtime" / "logs" / "launcher.log"
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        check("安装根" in text, "中文没写进去")
        check(path.read_text(encoding="utf-8").count("\n") == 1, "应只有一行")
        # 追加而不是覆盖：第二次启动不能把上次的记录冲掉
        log2 = launcher.Log(Path(tmp))
        log2.open()
        log2.write("第二次启动")
        log2.close()
        check_eq(len(path.read_text(encoding="utf-8").splitlines()), 2, "追加行数")
        # 时间戳形如 2026-11-13 15:37:01
        check(path.read_text(encoding="utf-8").startswith("20"), "缺时间戳")


# ---------------------------------------------------------------------------
# 7. 入口分派
# ---------------------------------------------------------------------------

@case("--tool 分支不进启动器")
def test_tool_branch():
    import frozen_entry

    saved = sys.argv
    try:
        sys.argv = ["TRTForgeWeb.exe", "--tool", "no_such_tool"]
        check_eq(frozen_entry.main(), 2, "未知工具退出码")
    finally:
        sys.argv = saved


@case("--tool 参数不足时不走启动流程")
def test_tool_short_args():
    import frozen_entry

    saved = sys.argv
    try:
        sys.argv = ["TRTForgeWeb.exe", "--tool"]
        calls = []
        saved_run = launcher.run
        launcher.run = lambda: calls.append(1) or 0
        try:
            frozen_entry.main()
        finally:
            launcher.run = saved_run
        check_eq(len(calls), 1, "无参数应进启动器")
    finally:
        sys.argv = saved


@case("无参数启动走 launcher.run")
def test_serve_branch():
    import frozen_entry

    saved = sys.argv
    try:
        sys.argv = ["TRTForgeWeb.exe"]
        calls = []
        saved_run = launcher.run
        launcher.run = lambda: calls.append(1) or 7
        try:
            check_eq(frozen_entry.main(), 7, "退出码应原样透传")
        finally:
            launcher.run = saved_run
        check_eq(len(calls), 1, "应调用一次启动器")
    finally:
        sys.argv = saved


@case("启动器顶层不 import uvicorn / app")
def test_no_heavy_import():
    """启动失败时如果 uvicorn 已经 import 失败，连错误框都弹不出来。

    launcher 的模块级只能有标准库；app 与 uvicorn 必须在 run() 里延迟 import。
    """
    source = (BACKEND / "launcher.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if line != line.lstrip():
            continue  # 缩进的行一定在函数里，延迟 import 是允许的
        stripped = line.strip()
        if stripped.startswith(("import uvicorn", "import app", "from app import")):
            raise AssertionError(f"launcher.py 顶层 import 了重模块：{stripped}")


@case("回退开关 TRTFORGE_LEGACY_START 被识别")
def test_legacy_flag():
    source = (BACKEND / "launcher.py").read_text(encoding="utf-8")
    check("TRTFORGE_LEGACY_START" in source, "回退开关不见了")
    check("_port_in_use_text" in source, "端口占用文案函数不见了")


def main() -> int:
    print("启动器测试")
    print("=" * 60)
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()

    print("-" * 60)
    if FAILED:
        for name, detail in FAILED:
            print(f"\n[FAIL] {name}\n{detail}")
        print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        return 1
    print(f"通过 {len(PASSED)} 项，失败 0 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
