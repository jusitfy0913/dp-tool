# -*- coding: utf-8 -*-
"""访问令牌 / 会话 / Host / Origin 的验收脚本。

覆盖的是 v1.42.0 新增的那一层（``backend/auth.py``），**不启动真实服务**：
用一个小 FastAPI 应用挂上同一个 ``auth.install``，再用 Starlette 的 TestClient
模拟不同来源。这样能精确控制「客户端地址」与「Host 头」这两个决定放行的输入。

    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_auth.py

判定标准（对应设计文档）：
- ``auto``（默认）：本机来源免令牌，其它来源必须带 Cookie 或 Bearer；
- ``on``：一律要令牌；``off``：一律不要（紧急回退）；
- 带点的域名 Host 一律 421（防 DNS 重绑定），IP / 单标签机器名放行；
- Origin 与 Host 不一致一律 403；
- 页面请求被拦时 302 到 ``/login.html``，接口被拦时 401；
- WebSocket 被拦时关闭码 4401（不是握手 403，前端才分得清原因）；
- 连续 10 次错令牌后 429。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402# noqa: E402

from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import auth  # noqa: E402
import config  # noqa: E402


def fake_request(client_host: str = "127.0.0.1", **headers: str) -> SimpleNamespace:
    """只带中间件会读的字段：client 地址、headers、cookies。

    ``host`` 是**连接来源**；要构造 Host 头请直接传 ``host_header``。
    """
    pairs = {key.replace("_", "-").lower(): value for key, value in headers.items()}
    return SimpleNamespace(
        client=SimpleNamespace(host=client_host),
        headers=pairs,
        cookies={},
    )


def build_app() -> FastAPI:
    """最小应用：一条受保护接口、一条公开接口、一个页面、一个 WebSocket。"""
    app = FastAPI()

    @app.get("/api/secret")
    def secret() -> dict:
        return {"ok": True}

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse("<h1>index</h1>")

    @app.websocket("/ws/runs/{run_id}")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        # 与 routers/runs.py 完全相同的顺序：先判权限，再 accept。
        if not auth.ws_authorized(websocket):
            await websocket.accept()
            await websocket.close(code=4401)
            return
        await websocket.accept()
        await websocket.send_json({"type": "ok", "run": run_id})
        await websocket.close()

    app.include_router(auth.router)
    if config.FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")
    auth.install(app)
    return app


def _env(**values: str) -> None:
    for key, value in values.items():
        os.environ[key] = value


def reset() -> None:
    """每个用例前清干净：会话、限流、令牌缓存、模式。"""
    auth.clear_sessions()
    auth.clear_limits()
    auth.reset_token_cache()
    os.environ.pop("TRTFORGE_AUTH", None)
    os.environ.pop("TRTFORGE_WEB_TOKEN", None)
    os.environ.pop("TRTFORGE_ALLOWED_HOSTS", None)
    os.environ.pop("TRTFORGE_SESSION_HOURS", None)


# ---------------------------------------------------------------------------
# 1. 模式与令牌
# ---------------------------------------------------------------------------
def test_mode(tmp: Path) -> None:
    os.environ.pop("TRTFORGE_AUTH", None)
    check("默认模式是 auto", auth.auth_mode() == "auto", auth.auth_mode())
    _env(TRTFORGE_AUTH="on")
    check("TRTFORGE_AUTH=on 生效", auth.auth_mode() == "on")
    _env(TRTFORGE_AUTH="OFF")
    check("模式不区分大小写", auth.auth_mode() == "off")
    _env(TRTFORGE_AUTH="whatever")
    check("非法模式回落 auto", auth.auth_mode() == "auto")


def test_token(tmp: Path) -> None:
    reset()
    path = auth.token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.unlink()
    token = auth.get_token()
    check("首次访问生成令牌", isinstance(token, str) and len(token) >= 20, token)
    check("令牌写入 runtime/web_token.txt", path.is_file(), str(path))
    check("文件内容与进程内一致", path.read_text(encoding="utf-8").strip() == token)
    check("重复调用返回同一把", auth.get_token() == token)

    auth.reset_token_cache()
    check("缓存清掉后从文件读回同一把", auth.get_token() == token)

    _env(TRTFORGE_WEB_TOKEN="fixed-token-for-test")
    auth.reset_token_cache()
    check("环境变量优先于文件", auth.get_token() == "fixed-token-for-test")
    check("环境变量令牌可匹配", auth.token_matches("fixed-token-for-test"))
    check("错误令牌不匹配", not auth.token_matches("wrong"))
    check("空令牌不匹配", not auth.token_matches(""))
    os.environ.pop("TRTFORGE_WEB_TOKEN")

    # 常量时间比较的"常量"没法在功能测试里量化，这里只确认它走的是 compare_digest
    source = (BACKEND / "auth.py").read_text(encoding="utf-8")
    check("令牌比较使用 hmac.compare_digest", "hmac.compare_digest" in source)


def test_session() -> None:
    reset()
    sid = auth.create_session()
    check("新建会话有效", auth.session_valid(sid))
    check("伪造会话无效", not auth.session_valid("not-a-session"))
    check("空会话无效", not auth.session_valid(""))
    auth.drop_session(sid)
    check("退出后会话失效", not auth.session_valid(sid))

    _env(TRTFORGE_SESSION_HOURS="0.0001")  # ≈0.36 秒
    short = auth.create_session()
    check("短会话创建时有效", auth.session_valid(short))
    time.sleep(0.5)
    check("过期会话自动失效", not auth.session_valid(short))
    os.environ.pop("TRTFORGE_SESSION_HOURS")


# ---------------------------------------------------------------------------
# 2. 来源判断（纯函数）
# ---------------------------------------------------------------------------
def test_host_and_origin() -> None:
    reset()
    check("Host=127.0.0.1 允许", auth.host_allowed("127.0.0.1:8060"))
    check("Host=localhost 允许", auth.host_allowed("localhost:8060"))
    check("Host=[::1] 允许", auth.host_allowed("[::1]:8060"))
    check("Host=LAN IP 允许", auth.host_allowed("192.168.1.7:8060"))
    check("Host=单标签机器名允许", auth.host_allowed("shopfloor-pc:8060"))
    check("Host=空（HTTP/1.0）允许", auth.host_allowed(""))
    check("Host=域名拒绝（防 DNS 重绑定）", not auth.host_allowed("evil.example.com"))
    check("Host=带点的伪 IP 也拒绝", not auth.host_allowed("127.0.0.1.evil.com"))

    _env(TRTFORGE_ALLOWED_HOSTS="studio.local;proxy.example.com")
    check("TRTFORGE_ALLOWED_HOSTS 里的域名放行", auth.host_allowed("studio.local:8060"))
    check("同清单第二项也放行", auth.host_allowed("proxy.example.com"))
    os.environ.pop("TRTFORGE_ALLOWED_HOSTS")

    ok = fake_request("127.0.0.1", origin="http://127.0.0.1:8060", host="127.0.0.1:8060")
    check("同源 Origin 允许", auth.origin_allowed(ok))
    bad = fake_request("127.0.0.1", origin="http://evil.example.com", host="127.0.0.1:8060")
    check("跨站 Origin 拒绝", not auth.origin_allowed(bad))
    null_origin = fake_request("127.0.0.1", origin="null", host="127.0.0.1:8060")
    check("Origin=null 拒绝", not auth.origin_allowed(null_origin))
    absent = fake_request("127.0.0.1", host="127.0.0.1:8060")
    check("没有 Origin（地址栏导航）允许", auth.origin_allowed(absent))


def test_needs_token() -> None:
    reset()
    local, remote = fake_request("127.0.0.1"), fake_request("192.168.1.9")
    test_client = fake_request("testclient")
    check("auto+本机：不要令牌", not auth.needs_token(local))
    check("auto+局域网：要令牌", auth.needs_token(remote))
    check("auto+TestClient：按本机处理", not auth.needs_token(test_client))
    _env(TRTFORGE_AUTH="on")
    check("on+本机：也要令牌", auth.needs_token(local))
    _env(TRTFORGE_AUTH="off")
    check("off+局域网：不要令牌", not auth.needs_token(remote))
    os.environ.pop("TRTFORGE_AUTH")

    req = fake_request("192.168.1.9")
    req.cookies = {auth.COOKIE_NAME: "bogus"}
    check("错误会话不算已登录", not auth.authenticated(req))
    req.cookies = {auth.COOKIE_NAME: auth.create_session()}
    check("有效会话算已登录", auth.authenticated(req))
    req.cookies = {}
    req.headers["authorization"] = f"Bearer {auth.get_token()}"
    check("Bearer 令牌算已登录", auth.authenticated(req))
    req.headers["authorization"] = "Bearer wrong"
    check("错误 Bearer 不算已登录", not auth.authenticated(req))


# ---------------------------------------------------------------------------
# 3. 中间件（HTTP 行为）
# ---------------------------------------------------------------------------
def test_middleware_local() -> None:
    reset()
    _env(TRTFORGE_AUTH="auto")
    with TestClient(build_app(), client=("127.0.0.1", 51000), follow_redirects=False) as client:
        check("本机 /api/secret 免令牌 200", client.get("/api/secret").status_code == 200)
        check("本机 / 正常返回", client.get("/").status_code == 200)
        status = client.get("/api/auth/status").json()
        check("状态接口 required=False", status["required"] is False)
        check("状态接口已登录语义", status["authenticated"] is True)
        check("状态接口带令牌文件路径", bool(status.get("token_file")))

    _env(TRTFORGE_AUTH="on")
    with TestClient(build_app(), client=("127.0.0.1", 51000), follow_redirects=False) as client:
        check("on 模式下本机也要令牌", client.get("/api/secret").status_code == 401)
        check("on 模式下健康检查仍公开", client.get("/api/health").status_code == 200)
        check("登录页本身公开", client.get("/login.html").status_code == 200)


def test_middleware_remote() -> None:
    reset()
    _env(TRTFORGE_AUTH="auto")
    base = build_app()
    with TestClient(base, client=("192.168.1.9", 51000), follow_redirects=False) as client:
        check("远程 /api/secret 401", client.get("/api/secret").status_code == 401)
        check("远程健康检查 200", client.get("/api/health").status_code == 200)
        check("远程静态资源放行", client.get("/static/css/style.css").status_code == 200)
        page = client.get("/", headers={"Accept": "text/html"})
        check("远程页面请求 302", page.status_code == 302, str(page.status_code))
        check("跳到登录页并带 next", page.headers.get("location", "").startswith("/login.html?next="),
              page.headers.get("location", ""))
        check("远程登录页可访问", client.get("/login.html").status_code == 200)

        # 登录 → 后续请求带 Cookie
        wrong = client.post("/api/auth/login", json={"token": "nope"})
        check("错误令牌 401", wrong.status_code == 401, str(wrong.status_code))
        good = client.post("/api/auth/login", json={"token": auth.get_token(), "next": "/"})
        check("正确令牌 200", good.status_code == 200)
        check("登录响应带回 next", good.json().get("next") == "/")
        check("Cookie 已下发", auth.COOKIE_NAME in client.cookies)
        check("登录后接口 200", client.get("/api/secret").status_code == 200)

        # Bearer 方式（脚本/自动化用）
        client.cookies.clear()
        bearer = client.get("/api/secret", headers={"Authorization": f"Bearer {auth.get_token()}"})
        check("Bearer 令牌可访问接口", bearer.status_code == 200)

        # 退出
        client.post("/api/auth/logout")
        check("退出后接口回到 401", client.get("/api/secret").status_code == 401)

        # 开放重定向防护
        ret = client.post("/api/auth/login", json={"token": auth.get_token(), "next": "//evil.com"})
        check("next 不接受跨站地址", ret.json().get("next") == "/", str(ret.json()))
        check("safe_next 拒绝 http 绝对地址", auth.safe_next("http://evil.com") == "/")
        check("safe_next 保留站内路径", auth.safe_next("/index.html?x=1") == "/index.html?x=1")


def test_headers_rejected() -> None:
    reset()
    _env(TRTFORGE_AUTH="auto")
    with TestClient(build_app(), client=("192.168.1.9", 51000), follow_redirects=False) as client:
        bad_host = client.get("/api/health", headers={"Host": "evil.example.com"})
        check("域名 Host → 421", bad_host.status_code == 421, str(bad_host.status_code))
        bad_origin = client.post(
            "/api/auth/login",
            json={"token": auth.get_token()},
            headers={"Origin": "http://evil.example.com"},
        )
        check("跨站 Origin → 403", bad_origin.status_code == 403, str(bad_origin.status_code))


def test_rate_limit() -> None:
    reset()
    _env(TRTFORGE_AUTH="auto")
    with TestClient(build_app(), client=("192.168.1.10", 51000), follow_redirects=False) as client:
        codes = [client.post("/api/auth/login", json={"token": "bad"}).status_code for _ in range(10)]
        check("前 10 次错误都是 401", codes == [401] * 10, str(codes))
        locked = client.post("/api/auth/login", json={"token": auth.get_token()})
        check("第 11 次被限流 429", locked.status_code == 429, str(locked.status_code))
        check("账号锁定期间正确令牌也拒绝", locked.status_code != 200)
    # 换一个来源不受影响：限流按来源，不是全局
    with TestClient(build_app(), client=("192.168.1.11", 51000), follow_redirects=False) as client:
        check("其它来源不受影响", client.post(
            "/api/auth/login", json={"token": auth.get_token()}
        ).status_code == 200)


def test_websocket() -> None:
    reset()
    _env(TRTFORGE_AUTH="auto")

    with TestClient(build_app(), client=("127.0.0.1", 51000), follow_redirects=False) as client:
        with client.websocket_connect("/ws/runs/abc") as ws:
            check("本机 WebSocket 可连接", ws.receive_json().get("type") == "ok")

    with TestClient(build_app(), client=("192.168.1.9", 51000), follow_redirects=False) as client:
        code = None
        try:
            with client.websocket_connect("/ws/runs/abc") as ws:
                ws.receive_json()
        except WebSocketDisconnect as exc:
            code = exc.code
        check("远程未登录 WebSocket 以 4401 关闭", code == 4401, str(code))

        # 带 ?token= 的脚本用法
        with client.websocket_connect(f"/ws/runs/abc?token={auth.get_token()}") as ws:
            check("WebSocket 支持 ?token=", ws.receive_json().get("type") == "ok")

        # 用 Cookie 登录后可连接
        client.post("/api/auth/login", json={"token": auth.get_token()})
        with client.websocket_connect("/ws/runs/abc") as ws:
            check("Cookie 登录后 WebSocket 可连接", ws.receive_json().get("type") == "ok")


def main() -> int:
    reset()
    with tempfile.TemporaryDirectory() as tmp:
        # 令牌文件改到临时目录，别覆盖开发者真实的那一份
        import config

        original_runtime = config.RUNTIME_DIR
        config.RUNTIME_DIR = Path(tmp)
        try:
            test_mode(Path(tmp))
            test_token(Path(tmp))
            test_session()
            test_host_and_origin()
            test_needs_token()
            test_middleware_local()
            test_middleware_remote()
            test_headers_rejected()
            test_rate_limit()
            test_websocket()
        finally:
            config.RUNTIME_DIR = original_runtime
            reset()
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
