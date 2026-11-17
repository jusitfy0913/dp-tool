# -*- coding: utf-8 -*-
"""访问令牌、会话与请求来源校验（v1.42.0）。

## 为什么需要这一层

Web Studio 默认只监听 ``127.0.0.1``，那时"能连上"等价于"坐在这台机器前"。
但车间多机使用时会把 ``host`` 改成 ``0.0.0.0``，一旦如此，**任何能 ping 到这台
机器的人都能训练、建引擎、读日志、改标注**——接口本身没有身份概念。

这一层只做三件事，都在路由之前：

1. **Host 校验**：挡住 DNS 重绑定。攻击者把 ``evil.com`` 解析到 ``127.0.0.1``
   后，浏览器仍会发 ``Host: evil.com``；域名一律拒绝，只认 IP、单标签机器名
   与 ``TRTFORGE_ALLOWED_HOSTS`` 里显式登记的名字。
2. **Origin 校验**：带 ``Origin`` 的请求必须与 ``Host`` 同源，挡住跨站伪造。
3. **令牌/会话校验**：见下。

## 令牌什么时候要

=============  ==========================================================
``auto``（默认） 本机来的请求不要令牌，其它来源一律要
``on``          一律要（想在本机也加锁时用）
``off``         一律不要（紧急回退；启动时会打一行告警）
=============  ==========================================================

注意 ``auto`` 判的是**连接的来源地址**（``request.client.host``），不是 Host 头
——Host 头是客户端可伪造的，用它判断"是不是本机"正是 DNS 重绑定能绕过的原因。
来源地址由 TCP 连接决定，伪造不了（除非攻击者已在本机，那就不在威胁模型内）。

## 令牌从哪来

``TRTFORGE_WEB_TOKEN`` 优先；没设时首次启动随机生成并写入
``runtime/web_token.txt``。现场人员能从文件里读到，不用记命令；删掉该文件
重启会换一把新的。

比对用 ``hmac.compare_digest``，防计时攻击。

## 为什么用 Cookie 而不是让前端加请求头

标注页的 ``<img src>``、批量结果的 ``<a download>``、日志新窗口、WebSocket
都无法自定义请求头，但都会自动带同源 Cookie。用 Cookie 意味着**前端现有
全部 URL 拼接一行都不用改**；脚本/第三方调用另有 ``Authorization: Bearer``。
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

import config

#: 会话 Cookie 名。改它等于让所有已登录的浏览器重新登录一次，别随手动。
COOKIE_NAME = "trtforge_session"

#: 本机来源地址。``::ffff:127.0.0.1`` 是双栈监听下 IPv4 映射后的形态；
#: ``testclient`` 是 Starlette 测试客户端的合成来源——它根本不走 TCP，
#: 直接调用 ASGI 应用，语义上就是"进程内"，按本机处理才符合事实。
LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1", "testclient"}

#: 免校验路径。**只放行登录必需的东西**：登录页、登录接口、状态查询、健康检查、
#: 静态资源。健康检查放行是因为它只回版本号，且前端与 launcher 都靠它判断
#: "端口上是不是本软件"；静态资源放行是因为登录页自己要用 CSS。
PUBLIC_PATHS = {
    "/login.html",
    "/api/auth/login",
    "/api/auth/status",
    "/api/health",
}

#: 登录失败限流：同一来源连续失败这么多次，锁这么久。
MAX_FAILURES = 10
LOCK_SECONDS = 300

router = APIRouter()

_TOKEN_LOCK = threading.Lock()
_TOKEN: str | None = None

_SESSION_LOCK = threading.Lock()
#: sid -> 过期时间戳（time.time() 秒）
_SESSIONS: dict[str, float] = {}

_FAIL_LOCK = threading.Lock()
#: 来源地址 -> (连续失败次数, 锁定到期时间戳)
_FAILURES: dict[str, tuple[int, float]] = {}


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------
def auth_mode() -> str:
    """当前令牌模式：``auto`` / ``on`` / ``off``。非法值按 ``auto`` 处理。

    每次调用都读环境变量：测试要能逐条改模式，缓存会让用例互相污染。
    这个查找是 dict 命中，比一次 ``fetch`` 便宜得多。
    """
    raw = os.environ.get("TRTFORGE_AUTH", "auto").strip().lower()
    return raw if raw in {"auto", "on", "off"} else "auto"


def session_hours() -> float:
    """会话有效期（小时），默认 7 天。"""
    raw = os.environ.get("TRTFORGE_SESSION_HOURS", "168").strip()
    try:
        value = float(raw)
    except ValueError:
        return 168.0
    return value if value > 0 else 168.0


def extra_allowed_hosts() -> set[str]:
    """``TRTFORGE_ALLOWED_HOSTS`` 里额外允许的主机名（分号分隔）。

    反向代理会改写 Host 成自己的域名，那种情况下必须显式登记，否则 421。
    """
    raw = os.environ.get("TRTFORGE_ALLOWED_HOSTS", "")
    return {item.strip().lower() for item in raw.split(";") if item.strip()}


# ---------------------------------------------------------------------------
# 令牌
# ---------------------------------------------------------------------------
def token_path() -> Path:
    """令牌文件位置。故意放在 runtime 下：和日志同级，现场好找。"""
    return config.RUNTIME_DIR / "web_token.txt"


def get_token() -> str:
    """取当前令牌。环境变量优先，否则读/生成 ``runtime/web_token.txt``。

    生成后写文件失败（只读安装目录）不算错误：进程内仍持有这个值，本次运行
    可用，只是重启后换一把——比启动失败好。
    """
    global _TOKEN
    env = os.environ.get("TRTFORGE_WEB_TOKEN", "").strip()
    if env:
        return env
    with _TOKEN_LOCK:
        if _TOKEN:
            return _TOKEN
        path = token_path()
        existing = ""
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if not existing:
            existing = secrets.token_urlsafe(24)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(existing + "\n", encoding="utf-8")
            except OSError:
                pass
        _TOKEN = existing
        return existing


def reset_token_cache() -> None:
    """清掉进程内令牌缓存（测试用；也用于改 ``TRTFORGE_WEB_TOKEN`` 后立即生效）。"""
    global _TOKEN
    with _TOKEN_LOCK:
        _TOKEN = None


def token_matches(candidate: str) -> bool:
    """常量时间比对。空令牌永远不通过。"""
    expected = get_token()
    if not expected or not candidate:
        return False
    return hmac.compare_digest(candidate, expected)


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
def create_session() -> str:
    """新建会话，返回 sid。顺手清掉过期项，避免字典无限增长。"""
    sid = secrets.token_urlsafe(32)
    expire = time.time() + session_hours() * 3600.0
    with _SESSION_LOCK:
        now = time.time()
        for key in [k for k, exp in _SESSIONS.items() if exp <= now]:
            _SESSIONS.pop(key, None)
        _SESSIONS[sid] = expire
    return sid


def session_valid(sid: str) -> bool:
    if not sid:
        return False
    with _SESSION_LOCK:
        expire = _SESSIONS.get(sid)
        if expire is None:
            return False
        if expire <= time.time():
            _SESSIONS.pop(sid, None)
            return False
    return True


def drop_session(sid: str) -> None:
    with _SESSION_LOCK:
        _SESSIONS.pop(sid, None)


def clear_sessions() -> None:
    """清空全部会话（测试用；服务重启本来就等于清空）。"""
    with _SESSION_LOCK:
        _SESSIONS.clear()


# ---------------------------------------------------------------------------
# 来源判断
# ---------------------------------------------------------------------------
def client_host(obj: Any) -> str:
    """连接来源地址，小写。取不到时返回空串（按"不是本机"处理）。"""
    client = getattr(obj, "client", None)
    host = getattr(client, "host", "") or ""
    return str(host).strip().lower()


def is_local_client(obj: Any) -> bool:
    """请求是否来自本机。判的是 TCP 来源地址，不是 Host 头。"""
    return client_host(obj) in LOOPBACK_CLIENTS


def host_allowed(host_header: str) -> bool:
    """Host 头是否是本机地址、IP 字面量、单标签机器名或显式登记的名字。

    拒绝的对象是**带点的域名**：DNS 重绑定必须用一个攻击者能控制的域名，
    而浏览器会把该域名原样放进 Host。IP 字面量与单标签机器名（NetBIOS）
    不可能由外部域名伪装出来。
    """
    host = (host_header or "").strip().lower()
    if not host:
        # HTTP/1.0 或无 Host 的请求：没有重绑定风险，但也没有来源信息，
        # 保守起见放行（后面的令牌校验仍会拦住远程访问）。
        return True
    if host.startswith("["):  # IPv6 形如 [::1]:8060
        host = host[1:].split("]", 1)[0]
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if not host:
        return False
    if host in LOOPBACK_CLIENTS:
        return True
    if host in extra_allowed_hosts():
        return True
    if "." not in host:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def origin_allowed(obj: Any) -> bool:
    """带 ``Origin`` 的请求必须与 Host 同源。

    浏览器对同源 POST 也会发 ``Origin``，所以正常操作一定能通过；跨站请求
    一定不匹配。``Origin: null``（file:// 或沙箱 iframe）一律拒绝。
    """
    origin = (obj.headers.get("origin") or "").strip()
    if not origin:
        return True
    if origin == "null":
        return False
    host = (obj.headers.get("host") or "").strip().lower()
    # Origin 形如 http://127.0.0.1:8060，取 "://" 之后的部分与 Host 比。
    _, _, netloc = origin.partition("://")
    return bool(netloc) and netloc.lower() == host


def needs_token(obj: Any) -> bool:
    """这次请求是否需要令牌。"""
    mode = auth_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return not is_local_client(obj)


def authenticated(obj: Any) -> bool:
    """Cookie 会话有效，或 ``Authorization: Bearer <令牌>`` 正确。"""
    sid = ""
    try:
        sid = obj.cookies.get(COOKIE_NAME) or ""
    except Exception:  # noqa: BLE001 - 没有 cookie 解析器时按未登录处理
        sid = ""
    if session_valid(sid):
        return True
    header = (obj.headers.get("authorization") or "").strip()
    if header.lower().startswith("bearer "):
        return token_matches(header[7:].strip())
    return False


# ---------------------------------------------------------------------------
# 登录限流
# ---------------------------------------------------------------------------
def _locked(ip: str) -> float:
    """返回剩余锁定秒数，0 表示没锁。"""
    with _FAIL_LOCK:
        count, until = _FAILURES.get(ip, (0, 0.0))
    remain = until - time.time()
    return remain if remain > 0 else 0.0


def _record_failure(ip: str) -> None:
    now = time.time()
    with _FAIL_LOCK:
        count, until = _FAILURES.get(ip, (0, 0.0))
        if until and until <= now:
            count = 0  # 上一轮锁定已过期，重新计数
        count += 1
        if count >= MAX_FAILURES:
            _FAILURES[ip] = (0, now + LOCK_SECONDS)
        else:
            _FAILURES[ip] = (count, 0.0)


def _clear_failures(ip: str) -> None:
    with _FAIL_LOCK:
        _FAILURES.pop(ip, None)


def clear_limits() -> None:
    """清空限流表（测试用）。"""
    with _FAIL_LOCK:
        _FAILURES.clear()


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    """登录参数。``next`` 只接受站内路径，避免开放重定向。"""

    token: str = Field(default="", description="访问令牌")
    next: str = Field(default="", description="登录后跳回的站内路径")


def safe_next(raw: str) -> str:
    """只允许站内绝对路径。``//evil.com`` 与 ``http://...`` 一律丢弃。"""
    value = (raw or "").strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


@router.get("/api/auth/status")
def auth_status(request: Request) -> dict:
    """前端开局调用：要不要登录、现在登录了没有。"""
    return {
        "mode": auth_mode(),
        "required": needs_token(request),
        "authenticated": (not needs_token(request)) or authenticated(request),
        "token_file": str(token_path()),
    }


@router.post("/api/auth/login")
def auth_login(payload: LoginRequest, request: Request) -> Response:
    """校验令牌并下发会话 Cookie。

    成功返回 ``{"ok": true, "next": ...}``；失败 401，错满 10 次后 429。
    """
    ip = client_host(request) or "unknown"
    remain = _locked(ip)
    if remain > 0:
        return JSONResponse(
            status_code=429,
            content={"detail": f"登录失败次数过多，请 {int(remain) + 1} 秒后再试"},
        )
    if not token_matches(payload.token.strip()):
        _record_failure(ip)
        return JSONResponse(status_code=401, content={"detail": "访问令牌不正确"})

    _clear_failures(ip)
    sid = create_session()
    body = {"ok": True, "next": safe_next(payload.next)}
    response = JSONResponse(content=body)
    # HttpOnly：脚本读不到，XSS 偷不走；SameSite=Strict：跨站请求不携带。
    # 不设 Secure：车间内网是 http，设了浏览器就完全不认这个 Cookie。
    response.set_cookie(
        COOKIE_NAME,
        sid,
        max_age=int(session_hours() * 3600),
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


@router.post("/api/auth/logout")
def auth_logout(request: Request) -> dict:
    """退出登录：删会话并清 Cookie。"""
    try:
        sid = request.cookies.get(COOKIE_NAME) or ""
    except Exception:  # noqa: BLE001
        sid = ""
    drop_session(sid)
    response = JSONResponse(content={"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.get("/login.html")
def login_page() -> FileResponse:
    """登录页。它自己也在 ``PUBLIC_PATHS`` 里，否则会无限重定向。"""
    path = config.FRONTEND_DIR / "login.html"
    if not path.is_file():
        return JSONResponse(status_code=500, content={"detail": "前端文件缺失：frontend/login.html"})
    return FileResponse(path)


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------
def _wants_html(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return "text/html" in accept


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static/")


def install(app: Any) -> None:
    """把校验挂成最外层 HTTP 中间件。

    顺序必须是「Host → Origin → 令牌」：Host/Origin 是零成本的头部比较，
    先做能挡掉大部分扫描流量；令牌校验要读文件、比对会话，放最后。
    """

    @app.middleware("http")
    async def _guard(request: Request, call_next):  # noqa: ANN001 - Starlette 约定
        path = request.url.path

        if not host_allowed(request.headers.get("host", "")):
            return JSONResponse(
                status_code=421,
                content={
                    "detail": (
                        "请求的 Host 不被接受。如果通过反向代理或域名访问，"
                        "请把该域名加进 TRTFORGE_ALLOWED_HOSTS 后重启服务。"
                    )
                },
            )

        if not origin_allowed(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "跨站请求被拒绝：Origin 与 Host 不一致"},
            )

        if _is_public(path):
            return await call_next(request)

        if not needs_token(request) or authenticated(request):
            return await call_next(request)

        if request.method == "GET" and _wants_html(request):
            target = f"/login.html?next={_quote(path)}"
            return RedirectResponse(target, status_code=302)
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    "需要访问令牌。请用浏览器打开页面输入令牌，"
                    f"或在请求头带 Authorization: Bearer <令牌>（令牌见 {token_path()}）"
                )
            },
        )


def _quote(value: str) -> str:
    from urllib.parse import quote  # noqa: PLC0415 - 只有重定向这一处用到

    return quote(safe_next(value), safe="/")


# ---------------------------------------------------------------------------
# WebSocket 与启动提示
# ---------------------------------------------------------------------------
def ws_authorized(websocket: Any) -> bool:
    """WebSocket 握手是否通过。

    浏览器的 WebSocket 不能自定义请求头，所以只认 Cookie；脚本可以改用
    ``?token=`` 查询串（自动化用，浏览器不会主动带）。
    """
    if not needs_token(websocket):
        return True
    if authenticated(websocket):
        return True
    query = getattr(websocket, "query_params", None)
    if query is not None:
        supplied = (query.get("token") or "").strip()
        if supplied and token_matches(supplied):
            return True
    return False


def announce() -> None:
    """启动时打一行「令牌在哪/是否关闭」的提示，只打印一次。"""
    mode = auth_mode()
    path = token_path()
    if mode == "off":
        print(
            "[auth] 警告：TRTFORGE_AUTH=off，任何能访问本端口的人都可以操作本软件。",
            flush=True,
        )
        return
    if mode == "on":
        print(f"[auth] 已启用访问令牌（TRTFORGE_AUTH=on），令牌文件：{path}", flush=True)
        return
    # auto：只在确实对外开放时才有意义，日志里点明判据，免得现场以为是 bug
    print(
        "[auth] 访问令牌：auto（本机访问免令牌，其它来源必须登录）。"
        f"令牌文件：{path}",
        flush=True,
    )
