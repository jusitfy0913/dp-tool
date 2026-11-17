# -*- coding: utf-8 -*-
"""关停要杀**整棵进程树**，不能只杀直接子进程。

为什么单独测这件事：真实链路是
``launcher -> ultralytics -> DataLoader worker × N``，Windows 上 worker 是
spawn 出来的独立进程。只 ``terminate()`` 直接子进程时，界面显示「已停止」，
显存却还被那几个 worker 占着，下一个任务立刻 OOM——「停止」看起来成功了，
所以这类问题最难联想到原因。

测的四个点：

1. ``kill_process_tree`` 能把孙进程一起带走；
2. ``RunManager.stop()`` 走的是同一条路，并把状态标成 cancelled（不是 failed）；
3. ``RunManager.shutdown_all()`` 并行收尾，多个任务的树都被清掉；
4. ``launcher._shutdown_timeout()`` 随任务数放大且夹在上下限之间
   （关窗时等待时间不够就会「窗口关了、进程还在」）。

    python 6-web/backend/tools/test_shutdown_tree.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from _harness import case, check_eq, finish, require, run_cases  # noqa: E402


#: 父进程再起一个孙进程，两者都长睡；把孙进程 pid 打到 stdout 供测试取用。
TREE_SCRIPT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', "
    "'import time\\nwhile True: time.sleep(0.2)'])\n"
    "print('CHILD %d' % child.pid, flush=True)\n"
    "while True:\n"
    "    time.sleep(0.2)\n"
)


def _write_script(directory: Path) -> Path:
    script = directory / "tree_launcher.py"
    script.write_text(TREE_SCRIPT, encoding="utf-8")
    return script


def _wait_child_pid(manager, run_id: str, timeout: float = 30.0) -> int:
    """从任务的日志事件里取出孙进程的 pid（父进程起来后会立刻打印）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = manager.get(run_id)
        require(run is not None, "任务记录应该还在")
        for event in list(run.events):
            line = str(event.get("line") or "")
            if line.startswith("CHILD "):
                return int(line.split()[1])
        if run.process is not None and run.process.poll() is not None:
            raise AssertionError(f"父进程提前退出了（exit={run.exit_code}），拿不到孙进程 pid")
        time.sleep(0.1)
    raise AssertionError("等不到孙进程 pid，父进程可能没起来")


@case("kill_process_tree 连孙进程一起清掉")
def test_kill_tree_direct() -> None:
    import gpulock
    import runners

    with tempfile.TemporaryDirectory(prefix="killtree_") as tmp:
        script = _write_script(Path(tmp))
        process = __import__("subprocess").Popen(
            [sys.executable, str(script)],
            cwd=tmp,
            stdout=-1,
            stderr=-2,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            line = process.stdout.readline() if process.stdout else ""
            require(line.startswith("CHILD"), f"父进程应打印孙进程 pid，实际：{line!r}")
            child_pid = int(line.split()[1])
            require(gpulock.pid_alive(child_pid), "孙进程应活着")
            require(runners.kill_process_tree(process), "kill_process_tree 应报告已退出")
            # 进程退出需要一点时间被系统回收，给一小段余量
            for _ in range(50):
                if not gpulock.pid_alive(child_pid):
                    break
                time.sleep(0.1)
            require(not gpulock.pid_alive(child_pid), "孙进程必须被杀掉（否则显存不会释放）")
            require(process.poll() is not None, "父进程也必须退出")
        finally:
            if process.poll() is None:
                process.kill()


@case("RunManager.stop() 杀树并把状态标成 cancelled")
def test_manager_stop_kills_tree() -> None:
    import gpulock
    import runners

    with tempfile.TemporaryDirectory(prefix="stoptree_") as tmp:
        script = _write_script(Path(tmp))
        loop = asyncio.new_event_loop()
        manager = runners.RunManager()
        manager.bind_loop(loop)
        try:
            run = manager.create("yolo-export", [sys.executable, str(script)], tmp)
            child_pid = _wait_child_pid(manager, run.run_id)
            require(gpulock.pid_alive(child_pid), "孙进程应活着")
            require(manager.stop(run.run_id), "stop 应返回 True")
            for _ in range(100):
                if not gpulock.pid_alive(child_pid):
                    break
                time.sleep(0.1)
            require(not gpulock.pid_alive(child_pid), "孙进程必须被杀掉")
            check_eq(run.status, "cancelled", "用户主动停止不算失败")
        finally:
            manager.shutdown_all()
            loop.close()


@case("shutdown_all() 并行收尾，多个任务的树都被清掉")
def test_shutdown_all_parallel() -> None:
    import gpulock
    import runners

    with tempfile.TemporaryDirectory(prefix="shutdownall_") as tmp:
        script = _write_script(Path(tmp))
        loop = asyncio.new_event_loop()
        manager = runners.RunManager()
        manager.bind_loop(loop)
        try:
            runs = [
                manager.create("yolo-export", [sys.executable, str(script)], tmp)
                for _ in range(2)
            ]
            child_pids = [_wait_child_pid(manager, run.run_id) for run in runs]
            check_eq(manager.active_count(), 2, "两个任务都应在运行")

            started = time.time()
            manager.shutdown_all()
            elapsed = time.time() - started

            for run in runs:
                check_eq(run.status, "cancelled", "关停后状态应为 cancelled")
            for pid in child_pids:
                for _ in range(100):
                    if not gpulock.pid_alive(pid):
                        break
                    time.sleep(0.1)
                require(not gpulock.pid_alive(pid), f"孙进程 {pid} 必须被杀掉")
            check_eq(manager.active_count(), 0, "关停后不该还有 running")
            # 并行而不是串行：串行做同样的事至少要等两轮 taskkill
            require(elapsed < 30, f"shutdown_all 花了 {elapsed:.1f}s，怀疑退化成了串行")
        finally:
            manager.shutdown_all()
            loop.close()


@case("_shutdown_timeout 随任务数放大且夹在上下限之间")
def test_shutdown_timeout_scaling() -> None:
    import launcher
    import runners

    base, per, cap = (
        launcher.SHUTDOWN_BASE_SECONDS,
        launcher.SHUTDOWN_PER_TASK_SECONDS,
        launcher.SHUTDOWN_MAX_SECONDS,
    )
    check_eq(launcher._shutdown_timeout(), base, "没有任务时应只等下限")
    require(base >= 20, "下限至少要覆盖一次 taskkill 的耗时")

    # 手工塞几条 running 记录，验证放大公式（不真起进程）
    with tempfile.TemporaryDirectory(prefix="timeout_") as tmp:
        saved = dict(runners.manager._runs)
        try:
            for index in range(3):
                run = runners.Run(
                    run_id=f"fake{index}",
                    kind="yolo-train",
                    command=[sys.executable, "-c", "pass"],
                    cwd=tmp,
                )
                runners.manager._runs[run.run_id] = run
            expected = min(cap, base + per * 3)
            check_eq(launcher._shutdown_timeout(), expected, "三个运行中任务应放大等待")
            require(expected <= cap, "放大结果不能超过上限")
        finally:
            runners.manager._runs.clear()
            runners.manager._runs.update(saved)


def main() -> int:
    print(f"关停杀进程树回归测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}")
    print()

    import config

    config.ensure_dirs()

    run_cases(globals())

    return finish()


if __name__ == "__main__":
    sys.exit(main())
