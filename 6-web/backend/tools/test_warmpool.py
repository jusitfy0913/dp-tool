"""常驻预热进程池的验证（不依赖 torch / ultralytics）。

用假 launcher 覆盖预热池的全部关键路径：
1. 正常预热 -> ready -> 取用 -> 下发任务 -> 任务跑完退出；
2. 预热进程死掉 / 还没就绪时，take() 必须返回 None，让调用方回退冷启动；
3. 关闭预热（TRTFORGE_WARM=0 的等价状态）时一个进程都不起；
4. 与 RunManager 的集成：命中预热时 run.warm=True，事件流仍然完整收到 finished。

为什么要测「回退」而不只是「成功」：
预热只是提速手段，任何情况下都不该让原本能跑的训练跑不起来。回退路径坏了，
用户看到的是「训练按钮没反应」，比慢几十秒严重得多。

用法：
    python backend/tools/test_warmpool.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402
import runners  # noqa: E402
import warmpool  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


#: 假 launcher：就绪后等一行 argv，然后按真实 launcher 的格式发事件并退出。
WARM_LAUNCHER = textwrap.dedent(
    """
    # -*- coding: utf-8 -*-
    import json, os, sys

    def emit(obj):
        print("TRTFORGE_JSON " + json.dumps(obj, ensure_ascii=False), flush=True)

    emit({"type": "event", "name": "warm_ready", "pid": os.getpid(), "elapsed_ms": 1})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        argv = payload.get("argv") or []
        emit({"type": "phase", "phase": "boot", "label": "python 进程已启动",
              "warm": True, "elapsed_ms": 0, "step_ms": 0})
        emit({"type": "event", "name": "launcher_train", "argv": argv})
        emit({"type": "event", "name": "finished", "exit": 0})
        sys.exit(0)
    """
).strip()


#: 立刻退出的假 launcher：模拟「预热进程已经死了」
DEAD_LAUNCHER = textwrap.dedent(
    """
    import sys
    sys.exit(3)
    """
).strip()


def write_launcher(directory: Path, name: str, body: str) -> str:
    path = directory / name
    path.write_text(body + "\n", encoding="utf-8")
    return str(path)


def make_pool(launcher: str, cwd: str) -> warmpool.WarmPool:
    pool = warmpool.WarmPool()
    pool.configure(
        {"fake": {"python": sys.executable, "launcher": launcher, "cwd": cwd}},
        enabled=True,
        idle_seconds=60,
    )
    return pool


def test_basic_cycle(tmp: Path) -> None:
    print("\n[1] 预热 -> 取用 -> 下发任务 -> 退出")
    launcher = write_launcher(tmp, "fake_warm.py", WARM_LAUNCHER)
    pool = make_pool(launcher, str(tmp))
    try:
        worker = pool.ensure("fake")
        check("ensure 返回 worker", worker is not None)
        check("wait_ready 成功", bool(worker and worker.wait_ready(20)))
        status = pool.status()
        check("status.enabled", status["enabled"] is True)
        check("status 为 ready", status["kinds"][0]["state"] == "ready", str(status))

        taken = pool.take("fake")
        check("take 拿到就绪进程", taken is not None)

        pre_events = taken.bind(["--mode", "train", "--epochs", "1"])
        check("绑定前缓存了 warm_ready", any(item.get("name") == "warm_ready" for item in pre_events))

        lines = []
        for raw in taken.read_task_lines():
            lines.append(raw.decode("utf-8", errors="replace"))
        text = "".join(lines)
        check("任务输出被完整读到", "launcher_train" in text and "finished" in text)
        # 读到 EOF 只代表 stdout 关了，进程回收要看 wait()。生产路径里由
        # _finalize() 的 process.wait(10) 完成，这里等价地等一次。
        check("任务结束后进程正常退出", taken.process.wait(5) == 0)

        # take 之后池子里应该已经自动补了一个新的
        started = time.time()
        refilled = False
        while time.time() - started < 10:
            if pool.status()["kinds"][0]["state"] == "ready":
                refilled = True
                break
            time.sleep(0.1)
        check("取用后自动补预热", refilled)
    finally:
        pool.shutdown()


def test_fallback(tmp: Path) -> None:
    print("\n[2] 回退：未就绪 / 进程已死 / 未启用")
    dead = write_launcher(tmp, "dead_warm.py", DEAD_LAUNCHER)
    pool = make_pool(dead, str(tmp))
    try:
        worker = pool.ensure("fake")
        check("死进程也能被创建", worker is not None)
        time.sleep(0.5)
        check("wait_ready 对死进程返回 False", worker.wait_ready(3) is False)
        check("take 对死进程返回 None（走冷启动）", pool.take("fake") is None)
    finally:
        pool.shutdown()

    launcher = write_launcher(tmp, "fake_warm2.py", WARM_LAUNCHER)
    disabled = warmpool.WarmPool()
    disabled.configure({}, enabled=False)
    check("未启用时 take 返回 None", disabled.take("fake") is None)

    unknown = make_pool(launcher, str(tmp))
    try:
        check("未注册的类型 take 返回 None", unknown.take("nope") is None)
    finally:
        unknown.shutdown()


def test_runner_integration(tmp: Path) -> None:
    print("\n[3] 与 RunManager 集成（命中热启动）")
    launcher = write_launcher(tmp, "fake_warm3.py", WARM_LAUNCHER)
    warmpool.pool.configure(
        {"fake": {"python": sys.executable, "launcher": launcher, "cwd": str(tmp)}},
        enabled=True,
        idle_seconds=60,
    )
    try:
        worker = warmpool.pool.ensure("fake")
        check("全局池预热成功", bool(worker and worker.wait_ready(20)))

        manager = runners.RunManager()
        loop = asyncio.new_event_loop()
        manager.bind_loop(loop)
        try:
            run = manager.create(
                "fake-train",
                [sys.executable, launcher, "--mode", "train"],
                str(tmp),
                warm_kind="fake",
            )
            check("run.warm=True", run.warm is True)
            check("快照带 warm 标记", run.snapshot().get("warm") is True)

            queue, history = manager.subscribe(run.run_id)
            events = list(history)

            async def collect() -> None:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=20)
                    except asyncio.TimeoutError:
                        return
                    events.append(event)
                    if event.get("type") == "finished":
                        return

            loop.run_until_complete(collect())
        finally:
            loop.close()

        first = events[0] if events else {}
        check("第一条事件是 phase", first.get("type") == "phase", str(first))
        check("spawned 标记为热启动", first.get("warm") is True and first.get("phase") == "spawned")
        check("收到 train_start 之外的任务事件", any(e.get("name") == "launcher_train" for e in events))
        finished = [e for e in events if e.get("type") == "finished"]
        check("收到 finished 且退出码 0", bool(finished) and finished[-1].get("exit") == 0)
        check("run 状态为 finished", run.status == "finished", run.status)
    finally:
        warmpool.pool.shutdown()


def test_offline_env() -> None:
    print("\n[4] 离线默认值")
    env: dict[str, str] = {}
    runners._apply_offline_defaults("yolo-train", env)
    check("yolo 任务默认 YOLO_OFFLINE=1", env.get("YOLO_OFFLINE") == "1")
    check("HF 遥测关闭", env.get("HF_HUB_DISABLE_TELEMETRY") == "1")

    env2: dict[str, str] = {}
    runners._apply_offline_defaults("engine-build", env2)
    check("非训练任务不加 YOLO_OFFLINE", "YOLO_OFFLINE" not in env2)


def main() -> int:
    print("常驻预热进程池验证")
    tmp = Path(tempfile.mkdtemp(prefix="warmpool_"))
    config.ensure_dirs()
    test_basic_cycle(tmp)
    test_fallback(tmp)
    test_runner_integration(tmp)
    test_offline_env()

    print(f"\n结果：通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
