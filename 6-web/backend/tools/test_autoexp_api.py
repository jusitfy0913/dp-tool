# -*- coding: utf-8 -*-
"""自动实验 Web 接口（autoexp_api）的回归测试。

测什么：**接口契约与安全边界**，不测模型、不联网、不跑训练。

    python tools/test_autoexp_api.py

三条最重要的断言（任何一条被破坏都必须让测试失败）：

1. **Key 永不出现在响应里**。GET 只给 ``sk-****1a2b`` 遮罩；
2. **Key 永不进 argv**。启动命令里不能有 Key，只能出现在环境变量；
3. **没填 Key 时不许启动**。否则第一轮只是等一次必然失败的请求，
   用户看到的只有「模型调用失败」，很难联想到「你没填 Key」。

所有用例都把 ``LLM_STORE`` 指到临时文件，绝不碰真实的
``6-web/runtime/autoexp_llm.json``——跑测试把用户的 Key 覆盖掉，
是最不该发生也最难查的一类事故。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

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


class TempStore:
    """把 LLM_STORE 临时指到别处，并保证退出时恢复原值。"""

    def __enter__(self):
        import autoexp_api

        self.module = autoexp_api
        self.original = autoexp_api.LLM_STORE
        self._tmp = tempfile.TemporaryDirectory()
        autoexp_api.LLM_STORE = Path(self._tmp.name) / "autoexp_llm.json"
        return autoexp_api.LLM_STORE

    def __exit__(self, *exc_info):
        self.module.LLM_STORE = self.original
        self._tmp.cleanup()
        return False


class TempGpuLock:
    """把显卡锁指到临时文件。

    否则机器上「命令行正在跑的那一轮实验」会真实占着锁，测试被 409 挡住——
    这不是 bug，是保护生效了，但会让测试结果随机漂移。
    """

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get("TRTFORGE_GPU_LOCK")
        os.environ["TRTFORGE_GPU_LOCK"] = str(Path(self._tmp.name) / "gpu_lock.json")
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop("TRTFORGE_GPU_LOCK", None)
        else:
            os.environ["TRTFORGE_GPU_LOCK"] = self._saved
        self._tmp.cleanup()
        return False


class ClearKeyEnv:
    """临时清空 API Key 环境变量。

    不清的话，开发机上装了 ``DEEPSEEK_API_KEY`` 会让「未配置 Key」的用例
    变成真联网请求——既慢又花钱，还依赖网络。
    """

    def __enter__(self):
        import autoexp_api

        module = autoexp_api._load_llm()
        names = list(getattr(module, "KEY_ENVS", ())) if module else []
        self._saved = {name: os.environ.pop(name, None) for name in names}
        return self

    def __exit__(self, *exc_info):
        for name, value in self._saved.items():
            if value is not None:
                os.environ[name] = value
        return False


class IdleBusyCheck:
    """临时把并发闸门改成「永远空闲」。"""

    def __enter__(self):
        import autoexp_api

        self.module = autoexp_api
        autoexp_api.set_busy_check(lambda: None)
        return self

    def __exit__(self, *exc_info):
        # 恢复到 app.py 注册的那个（导入 app 时已完成注册）
        import app as web_app  # noqa: F401 - 仅为确保注册过

        self.module.set_busy_check(web_app._reject_if_busy)
        return False


def client():
    from fastapi.testclient import TestClient

    import app as web_app

    return TestClient(web_app.app)


# ---------------------------------------------------------------------------
# 1. 探活与登记
# ---------------------------------------------------------------------------


@case("任务类型 autoexp-agent 已登记为独占")
def test_kind_registered() -> None:
    import runners

    check(
        "autoexp-agent" in runners.EXCLUSIVE_RUN_KINDS,
        "autoexp-agent 必须登记在 EXCLUSIVE_RUN_KINDS，否则会和网页训练同时抢显卡",
    )


@case("/ping 可用且不自相矛盾")
def test_ping() -> None:
    with TempStore():
        response = client().get("/api/autoexp/ping")
    check_eq(response.status_code, 200, "状态码")
    body = response.json()
    check_eq(body.get("kind"), "autoexp-agent", "kind")
    check(body.get("registered_exclusive") is True, "registered_exclusive 必须为 True")


@case("/status 结构完整且 Key 只给遮罩")
def test_status_shape() -> None:
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": "sk-abcdefgh12345678"}), encoding="utf-8")
        response = client().get("/api/autoexp/status")
    check_eq(response.status_code, 200, "状态码")
    body = response.json()
    for key in ("available", "splits_ready", "noise", "llm", "ledger_rows"):
        check(key in body, f"缺少字段 {key}")
    check(body["llm"].get("has_key") is True, "has_key 应为 True")
    check(
        "sk-abcdefgh12345678" not in response.text,
        "/status 的响应体里出现了 Key 原文：这是不可接受的泄露",
    )


# ---------------------------------------------------------------------------
# 2. Key 的安全边界
# ---------------------------------------------------------------------------


@case("GET /llm-config 永不返回 Key 原文")
def test_llm_config_masks_key() -> None:
    secret = "sk-super-secret-9876"
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": secret, "model": "deepseek-chat"}), encoding="utf-8")
        response = client().get("/api/autoexp/llm-config")
    check_eq(response.status_code, 200, "状态码")
    body = response.json()
    check(secret not in response.text, "响应体里出现了 Key 原文")
    check_eq(body.get("key_masked"), "sk-****9876", "遮罩格式")
    check(body.get("has_key") is True, "has_key")


@case("保存配置时空 api_key 表示「保留原 Key」")
def test_save_keeps_key_when_blank() -> None:
    with TempStore():
        api = client()
        api.post("/api/autoexp/llm-config", json={
            "base_url": "https://api.deepseek.com", "model": "deepseek-chat",
            "api_key": "sk-first-0000",
        })
        # 第二次只改模型，Key 留空
        api.post("/api/autoexp/llm-config", json={
            "base_url": "https://api.deepseek.com", "model": "deepseek-reasoner", "api_key": "",
        })
        body = api.get("/api/autoexp/llm-config").json()
    check_eq(body.get("model"), "deepseek-reasoner", "模型应被更新")
    check(body.get("has_key") is True, "留空时不应把 Key 弄丢")


@case("__clear__ 能清除已保存的 Key")
def test_clear_key() -> None:
    with TempStore():
        api = client()
        api.post("/api/autoexp/llm-config", json={"api_key": "sk-temp-1111"})
        api.post("/api/autoexp/llm-config", json={"api_key": "__clear__"})
        body = api.get("/api/autoexp/llm-config").json()
    check(body.get("has_key") is False, "清除后 has_key 应为 False")


@case("接口地址与 temperature 的非法值被拒绝")
def test_reject_invalid_config() -> None:
    with TempStore():
        api = client()
        bad_url = api.post("/api/autoexp/llm-config", json={"base_url": "ftp://x"})
        bad_temp = api.post("/api/autoexp/llm-config", json={"temperature": 5})
    check_eq(bad_url.status_code, 400, "非法协议应 400")
    check_eq(bad_temp.status_code, 400, "temperature 越界应 400")


@case("max_tokens 可保存、会回显，并且太小/太大都被拒绝")
def test_max_tokens_roundtrip() -> None:
    """max_tokens 是「推理型模型输出被截断」的唯一解药，必须能在页面上调。

    事故背景：900 全被思考吃掉、正文剩 0，现象是「返回内容为空」和 JSON 半截。
    用户需要能自己把这个数字调大，而不是改配置文件和源码。
    """
    with TempStore():
        api = client()
        saved = api.post("/api/autoexp/llm-config", json={"max_tokens": 4096})
        check_eq(saved.status_code, 200, "保存应成功")
        got = api.get("/api/autoexp/llm-config").json()
        check_eq(got.get("max_tokens"), 4096, "回显应为保存的值")

        too_small = api.post("/api/autoexp/llm-config", json={"max_tokens": 64})
        too_big = api.post("/api/autoexp/llm-config", json={"max_tokens": 999999})
    check_eq(too_small.status_code, 422, "过小的预算应被校验拦下")
    check_eq(too_big.status_code, 422, "过大的预算应被校验拦下")


@case("/start 会把 max_errors 传给子进程（模型侧故障有自己的刹车）")
def test_start_passes_max_errors() -> None:
    """errors 和 stale 是两个刹车：一个管「搜索到瓶颈」，一个管「模型侧坏了」。

    以前它们共用一个计数器，于是「输出一直被截断」被报成「连续 N 轮没有进展」。
    """
    fake = FakeManager()
    with TempStore() as store:
        store.write_text(
            json.dumps({"api_key": "sk-test", "base_url": "https://x.invalid",
                        "model": "m", "max_tokens": 4096}),
            encoding="utf-8",
        )
        with TempGpuLock(), FakeManagerSwap(fake), IdleBusyCheck():
            response = client().post("/api/autoexp/start",
                                     json={"rounds": 2, "max_errors": 4})
    check_eq(response.status_code, 200, "状态码")
    argv = fake.calls[0]["command"]
    check("--max-errors" in argv, f"应传 --max-errors：{argv}")
    check_eq(argv[argv.index("--max-errors") + 1], "4", "max_errors 的值应透传")
    check("sk-test" not in " ".join(argv), "Key 绝不能进 argv")


@case("没填 Key 时 /start 直接拒绝，不发任何请求")
def test_start_requires_key() -> None:
    with TempStore(), ClearKeyEnv(), IdleBusyCheck():
        response = client().post("/api/autoexp/start", json={"rounds": 3})
    check_eq(response.status_code, 400, "状态码")
    check("API Key" in response.json().get("detail", ""), "错误信息应点明缺 Key")


# ---------------------------------------------------------------------------
# 3. 启动命令（Key 不能进 argv）
# ---------------------------------------------------------------------------


class FakeRun:
    run_id = "run_fake_0001"

    def snapshot(self):
        return {"run_id": self.run_id, "kind": "autoexp-agent", "status": "running"}


class FakeManager:
    """记录 create() 的入参，用来断言「Key 只在 env 里」。"""

    def __init__(self):
        self.calls: list[dict] = []

    def active_exclusive(self):
        return []

    def create(self, kind, command, cwd, env=None, on_finished=None, **_kwargs):
        self.calls.append({
            "kind": kind,
            "command": list(command),
            "cwd": cwd,
            "env": dict(env or {}),
            "on_finished": on_finished,
        })
        return FakeRun()

    def get(self, run_id):
        return None

    def stop(self, run_id):
        return False


class FakeManagerSwap:
    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        import autoexp_api

        self.module = autoexp_api
        self.original = autoexp_api.manager
        autoexp_api.manager = self.fake
        self.original_python = autoexp_api._resolve_python
        autoexp_api._resolve_python = lambda: "python"
        return self.fake

    def __exit__(self, *exc_info):
        self.module.manager = self.original
        self.module._resolve_python = self.original_python
        return False


@case("启动命令不含 Key，Key 只走环境变量")
def test_start_hides_key_from_argv() -> None:
    secret = "sk-must-not-appear-in-argv"
    fake = FakeManager()
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": secret}), encoding="utf-8")
        with TempGpuLock(), FakeManagerSwap(fake), IdleBusyCheck():
            response = client().post("/api/autoexp/start", json={
                "rounds": 7, "hours": 2.5, "max_stale": 3, "lock_wait": 120,
            })
    check_eq(response.status_code, 200, "状态码")
    check_eq(len(fake.calls), 1, "manager.create 调用次数")
    call = fake.calls[0]

    joined = " ".join(call["command"])
    check(secret not in joined, f"Key 出现在了命令行里：{joined}")
    check_eq(call["env"].get("AUTOEXP_API_KEY"), secret, "Key 应通过环境变量传入")
    # 锁委派必须精确到 pid：只传布尔开关会让子进程看不见「命令行正另跑一轮」
    check_eq(
        call["env"].get("AUTOEXP_GPU_LOCK_PID"), str(os.getpid()),
        "必须把后端 pid 传下去，子进程据此只跳过「自己家那把锁」",
    )
    check_eq(
        call["env"].get("AUTOEXP_INFER_YIELDED"), "1",
        "子进程必须知道推理进程已让出，不能在每一轮结束时提前归还显存",
    )
    check(callable(call["on_finished"]), "任务结束回调要负责解除让出")
    yielded = response.json().get("infer_yielded") or {}
    check(yielded.get("held") is True, f"启动前应先让出推理显存：{yielded}")

    # 轮数/时间上限等参数要真的传下去，否则界面上改了半天没用
    check("--rounds" in call["command"], "缺少 --rounds")
    check_eq(call["command"][call["command"].index("--rounds") + 1], "7", "rounds 取值")
    check_eq(call["command"][call["command"].index("--hours") + 1], "2.5", "hours 取值")
    check("--events" in call["command"], "缺少 --events，网页就收不到决策事件")
    check_eq(call["kind"], "autoexp-agent", "kind")


@case("已有任务在跑时 /start 返回 409，并且不得让出推理显存")
def test_start_rejects_busy() -> None:
    from fastapi import HTTPException

    import inferproxy

    yield_calls: list[str] = []
    original = inferproxy.yield_gpu

    def counting_yield(reason: str = "训练"):
        yield_calls.append(reason)
        return original(reason)

    inferproxy.yield_gpu = counting_yield
    try:
        with TempStore() as store:
            store.write_text(json.dumps({"api_key": "sk-busy-test"}), encoding="utf-8")

            def busy():
                raise HTTPException(status_code=409, detail="已有任务正在运行：yolo-train")

            import autoexp_api

            autoexp_api.set_busy_check(busy)
            try:
                with TempGpuLock():
                    response = client().post("/api/autoexp/start", json={"rounds": 3})
            finally:
                import app as web_app

                autoexp_api.set_busy_check(web_app._reject_if_busy)
    finally:
        inferproxy.yield_gpu = original
    check_eq(response.status_code, 409, "状态码")
    check_eq(yield_calls, [], "别人占着显卡时不能把推理进程停掉")


class BoomManager(FakeManager):
    """create() 直接失败，用来确认启动半路失败会把让出还回去。"""

    def create(self, kind, command, cwd, env=None, on_finished=None, **_kwargs):
        super().create(kind, command, cwd, env, on_finished, **_kwargs)
        raise OSError("模拟进程拉起失败")


@case("启动进程失败时必须立刻解除显存让出")
def test_start_failure_releases_yield() -> None:
    import inferproxy

    inferproxy.release_gpu_hold()
    fake = BoomManager()
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": "sk-boom"}), encoding="utf-8")
        with TempGpuLock(), FakeManagerSwap(fake), IdleBusyCheck():
            response = client().post("/api/autoexp/start", json={"rounds": 1})
    check_eq(response.status_code, 500, "状态码")
    check("启动自动实验进程失败" in response.json().get("detail", ""), f"应说明启动失败：{response.json()}")
    check(inferproxy._pool.gpu_hold() == "", f"失败后让出标记必须清掉，实际 {inferproxy._pool.gpu_hold()!r}")
    check_eq(len(fake.calls), 1, "应确实尝试过启动")


@case("/stop 在没有任务时返回 404")
def test_stop_without_run() -> None:
    with TempStore(), IdleBusyCheck():
        response = client().post("/api/autoexp/stop", params={"run_id": "run_nope"})
    check_eq(response.status_code, 404, "状态码")


# ---------------------------------------------------------------------------
# 4. 只读接口
# ---------------------------------------------------------------------------


@case("/history 返回台账行，不需要网络")
def test_history() -> None:
    response = client().get("/api/autoexp/history", params={"limit": 5})
    check_eq(response.status_code, 200, "状态码")
    body = response.json()
    check(isinstance(body.get("rows"), list), "rows 应为列表")
    check(isinstance(body.get("total"), int), "total 应为整数")
    check(len(body["rows"]) <= 5, "limit 应生效")


@case("/llm-test 在缺 Key 时返回 ok=false 而不是 500")
def test_llm_test_without_key() -> None:
    with TempStore(), ClearKeyEnv():
        response = client().post("/api/autoexp/llm-test")
    check_eq(response.status_code, 200, "状态码（失败也应是可解析的 200）")
    body = response.json()
    check(body.get("ok") is False, "ok 应为 False")
    check("Key" in (body.get("error") or ""), f"错误信息应提到 Key，实际：{body.get('error')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"autoexp Web 接口测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}")
    print()

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()

    print()
    print("=" * 60)
    total = len(PASSED) + len(FAILED)
    print(f"通过 {len(PASSED)} / {total}")
    if FAILED:
        print()
        for name, detail in FAILED:
            print(f"[FAIL] {name}")
            print(detail)
            print("-" * 60)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
