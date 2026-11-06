# -*- coding: utf-8 -*-
"""训练启动耗时基准：冷启动 vs 预览温进程复用。

为什么要有这个工具
------------------
「启动变快了」这种结论不能只靠体感。这个脚本把两件事分别计时：

    冷启动：spawn 解释器 -> 收到 env 事件（= import torch/ultralytics/anomalib 全做完）
    热启动：向已就绪的预热进程下发同一个任务 -> 收到 env 事件

两者之差就是预热省下的固定开销，也是 1.25.0 里「预热进程池」的收益上限。

用法：
    python backend/tools/bench_startup.py            # 两类任务都测
    python backend/tools/bench_startup.py yolo       # 只测 YOLO

注意：这是性能工具，不是单元测试，不进 run_all_tests.py（跑一次要十几秒）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402

PREFIX = "TRTFORGE_JSON "
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _read_event(process: subprocess.Popen, predicate, timeout: float) -> dict | None:
    """读一行事件直到 predicate 成立，返回 payload。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = process.stdout.readline() if process.stdout else b""
        if not raw:
            return None
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith(PREFIX):
            continue
        try:
            payload = json.loads(line[len(PREFIX):])
        except json.JSONDecodeError:
            continue
        if predicate(payload):
            return payload
    return None


def _spawn(python: str, launcher: Path, cwd: Path, extra: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [python, str(launcher), *extra],
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        bufsize=0,
        creationflags=NO_WINDOW,
    )


def bench_cold(python: str, launcher: Path, cwd: Path) -> float:
    """冷启动：spawn 到 env 事件。"""
    started = time.perf_counter()
    process = _spawn(python, launcher, cwd, ["--mode", "check-env"])
    try:
        event = _read_event(process, lambda e: e.get("type") == "env", 180)
        return (time.perf_counter() - started) * 1000 if event else -1.0
    finally:
        process.kill()


def bench_warm(python: str, launcher: Path, cwd: Path) -> tuple[float, float]:
    """预热：返回（预热本身耗时, 命中预热后跑同一任务的耗时）。"""
    started = time.perf_counter()
    process = _spawn(python, launcher, cwd, ["--mode", "warm"])
    try:
        ready = _read_event(process, lambda e: e.get("name") == "warm_ready", 180)
        warm_ms = (time.perf_counter() - started) * 1000
        if ready is None:
            return -1.0, -1.0
        assert process.stdin is not None
        task_started = time.perf_counter()
        process.stdin.write(json.dumps({"argv": ["--mode", "check-env"]}).encode("utf-8") + b"\n")
        process.stdin.flush()
        event = _read_event(process, lambda e: e.get("type") == "env", 180)
        task_ms = (time.perf_counter() - task_started) * 1000 if event else -1.0
        return warm_ms, task_ms
    finally:
        process.kill()


def fmt(value: float) -> str:
    return "失败" if value < 0 else f"{value / 1000:.2f}s"


def main(argv: list[str]) -> int:
    filters = [item.lower() for item in argv[1:]]
    python = config.resolve_python()
    if not python:
        print("找不到装 torch 的训练解释器，无法测。可用 TRTFORGE_PYTHON 指定。")
        return 1

    targets = [
        ("yolo", config.YOLO_LAUNCHER, config.YOLO_LAUNCHER_DIR),
        ("anomalib", config.ANOMALIB_LAUNCHER, config.ANOMALIB_LAUNCHER_DIR),
    ]
    targets = [item for item in targets if not filters or item[0] in filters]
    if not targets:
        print(f"没有匹配的任务类型：{filters}")
        return 1

    print(f"解释器：{python}")
    print(f"{'任务':<10}{'冷启动':>10}{'预热耗时':>12}{'命中后启动':>14}{'节省':>10}")
    print("-" * 58)
    for kind, launcher, cwd in targets:
        if not launcher.is_file():
            print(f"{kind:<10}找不到 launcher：{launcher}")
            continue
        cold = bench_cold(python, launcher, cwd)
        warm_ms, task_ms = bench_warm(python, launcher, cwd)
        saved = f"{(cold - task_ms) / 1000:.2f}s" if cold > 0 and task_ms > 0 else "—"
        print(f"{kind:<10}{fmt(cold):>10}{fmt(warm_ms):>12}{fmt(task_ms):>14}{saved:>10}")
    print("\n说明：预热耗时只在服务空闲时付一次；用户点击训练时等的是「命中后启动」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
