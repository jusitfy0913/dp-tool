# -*- coding: utf-8 -*-
"""进入软件的操作密码。

和 ``auth.py`` 的局域网访问令牌不是一回事：

- 令牌防的是「别人的电脑连上这台服务就能操作」；
- 这里防的是「坐在这台电脑前的人，要先输对密码才能看到主界面」。

还没设过密码时（磁盘上没有记录文件），任何内容、包括空字符串都算通过。
在「环境说明」里设过之后，登录和改密码都要和哈希对上。

记录文件只存盐和 PBKDF2 哈希，不存明文。路径默认是
``runtime/operator_password.json``，测试可用环境变量
``TRTFORGE_OPERATOR_PASSWORD_FILE`` 指到临时文件。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import config
import jsonio

router = APIRouter()

ALGO = "pbkdf2-sha256"
ITERATIONS = 200_000
#: 连续输错这么多次就停 60 秒。比令牌那套（10 次 / 5 分钟）短，
#: 避免本机用户手滑几次就被关在门外很久。
MAX_FAILURES = 8
LOCK_SECONDS = 60
#: 防止有人贴进来一整篇文章，把 PBKDF2 拖成一次拒绝服务。
MAX_PASSWORD_CHARS = 128

_FAILS: dict[str, tuple[int, float]] = {}


class LoginBody(BaseModel):
    """登录只收一个密码。未设置时允许空字符串。"""

    password: str = ""


class PasswordBody(BaseModel):
    """设置 / 修改 / 清除共用。

    ``new_password`` 为空表示清除，但前提是已经设过密码、并且 ``old_password`` 正确。
    还没设过时旧密码被忽略。
    """

    old_password: str = ""
    new_password: str = ""


def store_path() -> Path:
    """密码记录文件。每次现读环境变量，测试才能在导入之后改路径。"""
    override = os.environ.get("TRTFORGE_OPERATOR_PASSWORD_FILE", "").strip()
    if override:
        return Path(override)
    return config.RUNTIME_DIR / "operator_password.json"


def clear_limits() -> None:
    """清掉输错计数。只给测试用，生产路径不调用。"""
    _FAILS.clear()


def _load() -> dict | None:
    """读出合法记录。文件不存在、损坏、算法对不上都当作「还没设密码」。

    损坏时放行而不是把人锁死：这是本机操作口令，锁死之后用户只能去删文件，
    和「文件本来就坏了」是同一件事，不如直接让他重新设。
    """
    data = jsonio.read_json(store_path(), default=None)
    if not isinstance(data, dict):
        return None
    if data.get("algo") != ALGO:
        return None
    if not isinstance(data.get("salt"), str) or not isinstance(data.get("hash"), str):
        return None
    return data


def _verify(password: str, record: dict) -> bool:
    """用记录里保存的迭代次数计算哈希，再做常时间比较。"""
    try:
        salt = bytes.fromhex(str(record["salt"]))
        expected = bytes.fromhex(str(record["hash"]))
        iterations = int(record["iterations"])
        if iterations < 1 or iterations > 2_000_000:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    if len(digest) != len(expected):
        return False
    return hmac.compare_digest(digest, expected)


def _write(password: str) -> None:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    jsonio.write_json(
        store_path(),
        {
            "algo": ALGO,
            "iterations": ITERATIONS,
            "salt": salt.hex(),
            "hash": digest.hex(),
        },
    )


def _clear_file() -> None:
    path = store_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _client_host(request: Request) -> str:
    if request.client is None or not request.client.host:
        return "unknown"
    return request.client.host


def _lock_remaining(host: str) -> float:
    """还要锁多少秒。到点了顺手把计数清掉，下一轮重新计。"""
    count, until = _FAILS.get(host, (0, 0.0))
    if count < MAX_FAILURES:
        return 0.0
    now = time.time()
    if now >= until:
        _FAILS.pop(host, None)
        return 0.0
    return until - now


def _mark_failure(host: str) -> None:
    count, _until = _FAILS.get(host, (0, 0.0))
    count += 1
    lock_until = time.time() + LOCK_SECONDS if count >= MAX_FAILURES else 0.0
    _FAILS[host] = (count, lock_until)


def _mark_success(host: str) -> None:
    _FAILS.pop(host, None)


def _reject_if_locked(host: str) -> None:
    remaining = _lock_remaining(host)
    if remaining > 0:
        raise HTTPException(
            status_code=429,
            detail=f"密码试错次数过多，请 {int(remaining) + 1} 秒后再试",
        )


def _check_length(password: str) -> None:
    if len(password) > MAX_PASSWORD_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"密码最长 {MAX_PASSWORD_CHARS} 个字符",
        )


@router.get("/api/operator/status")
def operator_status() -> dict:
    """告诉界面现在要不要真的核对密码。不返回哈希。"""
    return {"configured": _load() is not None}


@router.post("/api/operator/login")
def operator_login(body: LoginBody, request: Request) -> dict:
    """未设置时任意内容都成功；已设置时必须和哈希一致。"""
    host = _client_host(request)
    _reject_if_locked(host)
    _check_length(body.password)
    record = _load()
    if record is not None and not _verify(body.password, record):
        _mark_failure(host)
        # 刚刚这次把次数用满了，下一行会给出剩余秒数；没满则是普通的「不对」。
        _reject_if_locked(host)
        raise HTTPException(status_code=401, detail="密码不对")
    _mark_success(host)
    return {"ok": True, "configured": record is not None}


@router.post("/api/operator/password")
def operator_password(body: PasswordBody, request: Request) -> dict:
    """首次设置不看旧密码。已设置时旧密码必须正确。新密码为空就是清除。"""
    host = _client_host(request)
    _reject_if_locked(host)
    _check_length(body.old_password)
    _check_length(body.new_password)
    record = _load()
    if record is not None and not _verify(body.old_password, record):
        _mark_failure(host)
        _reject_if_locked(host)
        raise HTTPException(status_code=401, detail="当前密码不对")
    if body.new_password == "":
        if record is None:
            raise HTTPException(status_code=400, detail="请填写新密码")
        _clear_file()
        _mark_success(host)
        return {"ok": True, "configured": False}
    _write(body.new_password)
    _mark_success(host)
    return {"ok": True, "configured": True}
