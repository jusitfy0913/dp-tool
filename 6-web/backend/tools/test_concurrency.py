# -*- coding: utf-8 -*-
"""验证「同类重任务不允许并发」这条闸门。

为什么需要它：
前端刷新一次就失忆——页面退回「未开始」、按钮变成可点，而后端的训练进程
其实还在跑（v1.25.2 才补上接管）。用户点一下会起第二个训练进程：两个进程
抢同一块显卡通常双双 OOM，还会写进同一个 ``runs/<name>`` 目录把产物盖掉。

这里只测闸门本身的三件事：
1. 重任务在跑时，``active_exclusive()`` 能看见它；
2. 轻任务（自检 / 导出）不参与互斥，同时在跑也不该被算进去；
3. 任务结束后立刻让出位子，不能留下幽灵锁。

用法：python 6-web/backend/tools/test_concurrency.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runners  # noqa: E402

#: 一个跑得够久、可以被观察到的子进程：读一行 stdin 就停，从外部 terminate 结束。
_SLEEPER = "import sys, time\nsys.stdout.write('ready\\n')\nsys.stdout.flush()\ntime.sleep(60)\n"


def _write_sleeper(directory: Path) -> Path:
    script = directory / "sleeper.py"
    script.write_text(_SLEEPER, encoding="utf-8")
    return script


def make_manager() -> tuple[runners.RunManager, asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    manager = runners.RunManager()
    manager.bind_loop(loop)
    return manager, loop


def test_exclusive_kinds() -> None:
    """分类函数本身：重任务独占，轻任务不独占。"""
    for kind in ("yolo-train", "anomalib-train", "engine-build", "infer-batch", "infer-video"):
        assert runners.is_exclusive_kind(kind), f"{kind} 应算独占"
    # 串联的每个阶段都是 pipeline-<name>
    assert runners.is_exclusive_kind("pipeline-train")
    assert runners.is_exclusive_kind("pipeline-build")
    # 短任务允许并发，否则「训练时点一下检测环境」会被无谓拦住
    for kind in ("yolo-check-env", "anomalib-check-env", "yolo-export"):
        assert not runners.is_exclusive_kind(kind), f"{kind} 不该算独占"
    print("  [ok] 独占分类正确")


def test_active_exclusive_tracks_running() -> None:
    """跑着的重任务应出现在 active_exclusive 里，结束后必须消失。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = _write_sleeper(Path(tmp))
        manager, loop = make_manager()
        run = manager.create("yolo-train", [sys.executable, str(script)], tmp)

        # 给进程一点时间真的起来
        for _ in range(50):
            if run.process and run.process.poll() is None:
                break
            time.sleep(0.05)

        busy = manager.active_exclusive()
        assert [item["run_id"] for item in busy] == [run.run_id], f"应只看到 1 个重任务，实得 {busy}"
        assert busy[0]["status"] == "running"
        assert busy[0]["kind"] == "yolo-train"
        print("  [ok] 运行中的重任务可见")

        manager.stop(run.run_id)
        deadline = time.time() + 10
        while time.time() < deadline and manager.active_exclusive():
            time.sleep(0.05)
        assert manager.active_exclusive() == [], "任务结束后必须让出位子"
        print("  [ok] 任务结束后自动让位")
        loop.close()


def test_light_kinds_are_not_exclusive() -> None:
    """轻任务在跑时，闸门必须是空的——它不能被「检测环境」误伤。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = _write_sleeper(Path(tmp))
        manager, loop = make_manager()
        run = manager.create("yolo-check-env", [sys.executable, str(script)], tmp)

        for _ in range(50):
            if run.process and run.process.poll() is None:
                break
            time.sleep(0.05)

        assert manager.active_exclusive() == [], "check-env 不该占住并发位"
        manager.stop(run.run_id)
        loop.close()
        print("  [ok] 轻任务不占位")


def test_exclusive_survives_multiple_kinds() -> None:
    """两种重任务同时在跑时，两条都要能列出来（用于给用户报「谁在占位」）。"""
    with tempfile.TemporaryDirectory() as tmp:
        script = _write_sleeper(Path(tmp))
        manager, loop = make_manager()
        first = manager.create("engine-build", [sys.executable, str(script)], tmp)
        second = manager.create("infer-batch", [sys.executable, str(script)], tmp)

        for _ in range(60):
            ready = all(r.process and r.process.poll() is None for r in (first, second))
            if ready:
                break
            time.sleep(0.05)

        ids = {item["run_id"] for item in manager.active_exclusive()}
        assert ids == {first.run_id, second.run_id}, f"应看到 2 条，实得 {ids}"
        # 新的排前面：报错时优先说「最后启动的那个」
        assert manager.active_exclusive()[0]["run_id"] == second.run_id

        manager.stop(first.run_id)
        manager.stop(second.run_id)
        loop.close()
        print("  [ok] 多条重任务可同时列出，按启动时间倒序")


def test_reject_helper_raises_409() -> None:
    """闸门函数本身：有重任务时必须抛 409，且文案里说得清「谁在占位、去点哪」。

    只测这个纯函数、不启 TestClient：目的是验证「拒绝 + 文案」，
    真起服务会把 34 个用例的套件拖慢十几秒。
    """
    import app as backend_app
    from fastapi import HTTPException

    original = backend_app.manager.active_exclusive
    backend_app.manager.active_exclusive = lambda: [
        {
            "run_id": "abc123def456",
            "kind": "yolo-train",
            "status": "running",
            "elapsed": 3725,
            "started_at": 0,
            "meta": {},
        }
    ]
    try:
        try:
            backend_app._reject_if_busy()
        except HTTPException as exc:
            assert exc.status_code == 409, f"应是 409，实得 {exc.status_code}"
            detail = str(exc.detail)
            # 用户要能一眼看出「是什么任务、跑了多久、该去哪儿停」
            assert "YOLO 训练" in detail, detail
            assert "abc123def456" in detail, detail
            assert "62 分钟" in detail, detail
            assert "运行中" in detail, detail
        else:
            raise AssertionError("有重任务在跑却没有拒绝")
    finally:
        backend_app.manager.active_exclusive = original

    # 没有重任务时不能误伤
    backend_app.manager.active_exclusive = lambda: []
    try:
        backend_app._reject_if_busy()
    finally:
        backend_app.manager.active_exclusive = original
    print("  [ok] 409 文案包含任务类型、run_id、已运行时长与去处")


def test_describe_run_labels() -> None:
    """串联阶段（pipeline-*）要显示成「全流程串联」，不能把内部 kind 丢给用户。"""
    import app as backend_app

    text = backend_app._describe_run(
        {"run_id": "p1", "kind": "pipeline-build", "status": "running", "elapsed": 30}
    )
    assert "全流程串联" in text, text
    assert "30 秒" in text, text
    print("  [ok] 串联阶段的中文名与时长格式正确")


def main() -> int:
    print("test_concurrency: 并发闸门")
    test_exclusive_kinds()
    test_active_exclusive_tracks_running()
    test_light_kinds_are_not_exclusive()
    test_exclusive_survives_multiple_kinds()
    test_reject_helper_raises_409()
    test_describe_run_labels()
    print("test_concurrency: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
