# -*- coding: utf-8 -*-
"""进入密码：未设置时任意内容通过，设置后必须对上，磁盘上没有明文。

    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_operator_password.py

不启动真实服务，只用一个挂了同一份路由的 TestClient，记录文件指到临时目录。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import operator_auth  # noqa: E402


def _client(path: Path) -> TestClient:
    os.environ["TRTFORGE_OPERATOR_PASSWORD_FILE"] = str(path)
    operator_auth.clear_limits()
    app = FastAPI()
    app.include_router(operator_auth.router)
    return TestClient(app)


def main() -> int:
    secret = "correct-horse"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "operator_password.json"
        client = _client(path)

        status = client.get("/api/operator/status")
        check("未设置时 status 是 configured=false", status.json() == {"configured": False}, str(status.json()))
        empty = client.post("/api/operator/login", json={"password": ""})
        check("未设置时空密码可以进", empty.status_code == 200 and empty.json()["ok"] is True, empty.text)
        junk = client.post("/api/operator/login", json={"password": "随便写"})
        check("未设置时任意内容可以进", junk.status_code == 200, junk.text)
        check("登录没有把密码写进磁盘", not path.is_file())

        missing = client.post("/api/operator/password", json={"old_password": "", "new_password": ""})
        check("没设过密码时不能用空新密码", missing.status_code == 400, missing.text)

        created = client.post("/api/operator/password", json={"old_password": "", "new_password": secret})
        check("首次设置不需要旧密码", created.status_code == 200 and created.json()["configured"] is True, created.text)
        text = path.read_text(encoding="utf-8")
        check("磁盘上有哈希记录", path.is_file() and "pbkdf2-sha256" in text)
        check("磁盘上没有明文密码", secret not in text, text)

        wrong = client.post("/api/operator/login", json={"password": "不对"})
        check("设过之后错密码被拒绝", wrong.status_code == 401, wrong.text)
        right = client.post("/api/operator/login", json={"password": secret})
        check("设过之后对密码可以进", right.status_code == 200, right.text)

        sneak = client.post("/api/operator/password", json={"old_password": "", "new_password": "new-one"})
        check("已设置时不带旧密码改不了", sneak.status_code == 401, sneak.text)
        check("改失败后明文仍不在文件里", secret not in path.read_text(encoding="utf-8") and "new-one" not in path.read_text(encoding="utf-8"))

        changed = client.post(
            "/api/operator/password",
            json={"old_password": secret, "new_password": "new-one"},
        )
        check("旧密码正确才能修改", changed.status_code == 200, changed.text)
        check("旧密码不再有效", client.post("/api/operator/login", json={"password": secret}).status_code == 401)
        check("新密码可以进", client.post("/api/operator/login", json={"password": "new-one"}).status_code == 200)
        check("新密码的明文也不在文件里", "new-one" not in path.read_text(encoding="utf-8"))

        blocked = client.post("/api/operator/password", json={"old_password": "nope", "new_password": ""})
        check("清除时旧密码不对会被拒绝", blocked.status_code == 401, blocked.text)
        check("拒绝清除后文件还在", path.is_file())
        cleared = client.post("/api/operator/password", json={"old_password": "new-one", "new_password": ""})
        check("旧密码正确且新密码为空就是清除", cleared.status_code == 200 and cleared.json()["configured"] is False, cleared.text)
        check("清除后记录文件消失", not path.is_file())
        check("清除后任意密码又能进", client.post("/api/operator/login", json={"password": ""}).status_code == 200)

        client.post("/api/operator/password", json={"new_password": secret})
        codes = [
            client.post("/api/operator/login", json={"password": "bad"}).status_code
            for _ in range(operator_auth.MAX_FAILURES)
        ]
        # 第 8 次错就把锁加上了，这一次的响应已经是 429，不是再等下一次。
        check(
            "前 7 次是 401，第 8 次开始锁定",
            codes == [401] * (operator_auth.MAX_FAILURES - 1) + [429],
            str(codes),
        )
        locked = client.post("/api/operator/login", json={"password": secret})
        check("次数用满后连对密码也要等", locked.status_code == 429, locked.text)
        operator_auth.clear_limits()
        check(
            "锁被清掉后对密码又能进",
            client.post("/api/operator/login", json={"password": secret}).status_code == 200,
        )
        check("最终磁盘上仍然没有明文", secret not in path.read_text(encoding="utf-8"))

    return finish("进入密码")


if __name__ == "__main__":
    raise SystemExit(main())
