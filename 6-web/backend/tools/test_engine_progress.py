# -*- coding: utf-8 -*-
"""引擎构建进度与估算的验收脚本（纯函数，不需要 trtexec、不需要服务）。

覆盖四件事：
1. 用**真实日志**的时间戳回放，验证阶段顺序、百分比单调不减、静默 473 秒期间
   心跳能产出进度、以及「只有 PASSED 才到 100%」；
2. 估算值落在合理区间（样本真实 474.149 s）；
3. `engine_stats.json` 的读写、中位数选择、不匹配维度时回退公式；
4. `runners.create(kind="engine-build")` 的接线：日志喂进去能出 progress 事件，
   结束时补 100%，且失败时不假装走满。

用法：python 6-web/backend/tools/test_engine_progress.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import progress  # noqa: E402
import runners  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent

#: 真实日志 f066553e4df2.log 的关键行（含原始时间戳）。
#: 之所以内嵌一份而不是只读文件：runtime/ 是 gitignore 的，换台机器就没有了，
#: 但这份断言必须能在任何机器上跑。文件存在时下面还会额外回放全量日志。
SAMPLE_LINES = [
    "[10/31/2026-00:02:21] [I] &&&& RUNNING TensorRT.trtexec [TensorRT v8502]",
    "[10/31/2026-00:02:21] [I] Start parsing network model",
    "[10/31/2026-00:02:21] [I] Finish parsing network model",
    "[10/31/2026-00:02:21] [I] [TRT] Local timing cache in use. "
    "Profiling results in this builder pass will not be stored.",
    "[10/31/2026-00:10:14] [I] [TRT] Total Activation Memory: 4432243712",
    "[10/31/2026-00:10:14] [I] Engine built in 474.149 sec.",
    "[10/31/2026-00:10:14] [I] Engine deserialized in 0.0149783 sec.",
    "[10/31/2026-00:10:14] [I] Starting inference",
    "[10/31/2026-00:10:17] [I] Total Host Walltime: 3.02145 s",
    "[10/31/2026-00:10:17] [I] &&&& PASSED TensorRT.trtexec [TensorRT v8502]",
]

SAMPLE_META = {
    "onnx_mb": 12.39,
    "precision": "fp16",
    "dynamic": False,
    "imgsz": 960,
}

REAL_LOG = BACKEND_DIR.parent / "runtime" / "logs" / "f066553e4df2.log"


class FakeClock:
    """可手动推进的壁钟，用来把 8 分钟的日志压缩成毫秒级测试。"""

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def parse_stamp(line: str) -> float | None:
    """`[10/31/2026-00:02:21]` -> epoch 秒。取不到返回 None。"""
    if not line.startswith("["):
        return None
    end = line.find("]")
    if end < 0:
        return None
    try:
        return datetime.strptime(line[1:end], "%m/%d/%Y-%H:%M:%S").timestamp()
    except ValueError:
        return None


def replay(lines: list[str], estimate: float, store: progress.EngineStatsStore | None = None):
    """按日志时间戳推进假时钟，回放整段日志。

    返回 (事件列表, analyzer)。时间戳相同、或取不到时间戳时按 +0.1 s 兜底，
    保证不会出现「时间倒流」。
    """
    clock = FakeClock()
    analyzer = progress.EngineBuildAnalyzer(
        estimate, meta=SAMPLE_META, stats=store, started_at=clock(), clock=clock
    )
    events: list[dict] = []
    last = clock.now
    for line in lines:
        stamp = parse_stamp(line)
        if stamp is not None and stamp > last:
            clock.now = stamp
        else:
            clock.now = last + 0.1
        last = clock.now
        event = analyzer.feed(line)
        if event is not None:
            events.append(event)
    return events, analyzer, clock


def check_replay(lines: list[str], label: str, passed: list[str], failed: list[str]) -> None:
    """回放一段日志，断言阶段、单调性、静默期心跳、100% 时机。"""
    estimate = progress.estimate_seconds(SAMPLE_META, None)
    # 假时钟必须对齐日志的第一条时间戳：差 9 千万秒的话 elapsed/estimate 会瞬间
    # 顶到阶段上限，静默期就再也推不动百分比了（真实的坑，别删这段注释）。
    first_stamp = next(
        (stamp for stamp in (parse_stamp(line) for line in lines) if stamp is not None),
        1_700_000_000.0,
    )
    clock = FakeClock(first_stamp)
    analyzer = progress.EngineBuildAnalyzer(
        estimate, meta=SAMPLE_META, stats=None, started_at=clock(), clock=clock
    )
    events: list[dict] = []
    last = clock.now
    silent_ticks = 0
    silence_started = False

    for line in lines:
        stamp = parse_stamp(line)
        if stamp is not None and stamp > last:
            # 时间戳跳变说明中间静默了：按 2 秒一步模拟心跳线程
            if last and stamp - last >= 4 and silence_started:
                step = last + 2.0
                while step < stamp:
                    clock.now = step
                    tick_event = analyzer.tick()
                    if tick_event is not None:
                        events.append(tick_event)
                        silent_ticks += 1
                    step += 2.0
            clock.now = stamp
        else:
            clock.now = last + 0.1
        last = clock.now
        event = analyzer.feed(line)
        if event is not None:
            events.append(event)
            if event["stage"] == "build":
                silence_started = True

    stages = [event["stage"] for event in events]
    pcts = [event["pct"] for event in events]

    # 阶段顺序：必须是 STAGE_ORDER 的子序列（允许跳阶段，不允许倒退）
    indexes = [progress.STAGE_ORDER.index(item) for item in stages]
    if indexes == sorted(indexes):
        passed.append(f"{label}：阶段顺序单调前进（{len(set(stages))} 个阶段）")
    else:
        failed.append(f"{label}：阶段出现倒退 {stages}")

    if pcts == sorted(pcts):
        passed.append(f"{label}：百分比单调不减（{pcts[0]}% → {pcts[-1]}%）")
    else:
        failed.append(f"{label}：百分比出现回退 {pcts}")

    if silent_ticks >= 1:
        passed.append(f"{label}：8 分钟静默期内心跳产出 {silent_ticks} 条进度")
    else:
        failed.append(f"{label}：静默期内没有产出任何进度事件（心跳没生效）")

    before_done = [p for stage, p in zip(stages, pcts) if stage != "done"]
    if all(p < 100 for p in before_done) and pcts[-1] == 100.0:
        passed.append(f"{label}：只有 PASSED 才到 100%（结束前最高 {max(before_done):.1f}%）")
    else:
        failed.append(f"{label}：100% 时机不对 {pcts}")

    if analyzer.actual_seconds and abs(analyzer.actual_seconds - 474.149) < 0.01:
        passed.append(f"{label}：抓到真实构建耗时 {analyzer.actual_seconds} s")
    else:
        failed.append(f"{label}：未抓到真实构建耗时（{analyzer.actual_seconds}）")

    if 180 <= estimate <= 780:
        passed.append(f"{label}：估算 {estimate:.0f} s 与真实 474 s 同量级")
    else:
        failed.append(f"{label}：估算 {estimate:.0f} s 偏离真实 474 s 太多")


def test_analyzer(passed: list[str], failed: list[str]) -> None:
    check_replay(SAMPLE_LINES, "内嵌样本", passed, failed)

    if REAL_LOG.is_file():
        lines = REAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        check_replay(lines, "真实日志", passed, failed)
    else:
        passed.append("真实日志不存在，跳过全量回放（内嵌样本已覆盖同样断言）")

    # 失败构建：不假装走满
    analyzer = progress.EngineBuildAnalyzer(60.0, meta=SAMPLE_META, stats=None)
    analyzer.feed("[x] [I] Local timing cache in use.")
    event = analyzer.finish("failed", 1)
    if event is None and analyzer.pct < 100:
        passed.append("失败构建不发 100% 事件")
    else:
        failed.append(f"失败构建错误地走到了 {analyzer.pct}%（event={event}）")

    # 成功但日志里没有 PASSED（被杀掉前刚写完引擎）：finish 补 100%
    analyzer_ok = progress.EngineBuildAnalyzer(60.0, meta=SAMPLE_META, stats=None)
    analyzer_ok.feed("[x] [I] Local timing cache in use.")
    event = analyzer_ok.finish("finished", 0)
    if event is not None and event["pct"] == 100.0:
        passed.append("成功收尾补 100%")
    else:
        failed.append(f"成功收尾没有补 100%（event={event}）")

    # 普通日志行不该被当成阶段
    if progress.EngineBuildAnalyzer.stage_for("[x] [I] Some unrelated line") is None:
        passed.append("普通日志行不误判阶段")
    else:
        failed.append("普通日志行被误判成阶段")


def test_estimate(passed: list[str], failed: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = progress.EngineStatsStore(Path(tmp) / "engine_stats.json")

        # 空历史 -> 基线公式，应接近实测 474 s
        base = progress.estimate_seconds(SAMPLE_META, store)
        if 400 <= base <= 560:
            passed.append(f"无历史时基线外推 {base:.0f} s（实测 474 s）")
        else:
            failed.append(f"基线外推偏离过多：{base:.0f} s")

        # 写三条同维度记录（含一条异常值），应取中位数而不是平均值
        for seconds in (400.0, 500.0, 9000.0):
            store.record(
                {
                    "onnx_mb": 12.39,
                    "precision": "fp16",
                    "dynamic": False,
                    "imgsz": 960,
                    "seconds": seconds,
                    "ts": 0.0,
                }
            )
        median = progress.estimate_seconds(SAMPLE_META, store)
        if abs(median - 500.0) < 0.01:
            passed.append("历史命中时取中位数（400/500/9000 -> 500）")
        else:
            failed.append(f"中位数选择错误：{median}")

        # 精度不同 -> 不命中历史，回退公式（fp32 更快）
        fp32 = progress.estimate_seconds({**SAMPLE_META, "precision": "fp32"}, store)
        if abs(fp32 - base * 0.7) < 1.0:
            passed.append("精度不同时不命中历史，回退公式")
        else:
            failed.append(f"fp32 估算异常：{fp32}（期望约 {base * 0.7:.1f}）")

        # 分辨率差太多 -> 不命中历史。实测 320 / 12.1 MB 需要 317 s，
        # 幂律（指数 0.4）应给出 300 s 上下，绝不能沿用早期「按面积平方」的 53 s
        lowres = progress.estimate_seconds({**SAMPLE_META, "imgsz": 320}, store)
        if 250 <= lowres <= 380:
            passed.append(f"分辨率不匹配时不命中历史（320 -> {lowres:.0f} s，实测 317 s）")
        else:
            failed.append(f"320 分辨率估算异常：{lowres:.0f} s")

        # 拿不到体积 -> 兜底值
        unknown = progress.estimate_seconds({"onnx_mb": 0}, store)
        if unknown == progress.DEFAULT_ESTIMATE_SECONDS:
            passed.append(f"体积未知时兜底 {unknown:.0f} s")
        else:
            failed.append(f"兜底值异常：{unknown}")

        # 损坏文件 -> 当成没有历史，不抛异常
        (Path(tmp) / "bad.json").write_text("{ not json", encoding="utf-8")
        bad_store = progress.EngineStatsStore(Path(tmp) / "bad.json")
        if bad_store.load() == [] and progress.estimate_seconds(SAMPLE_META, bad_store) > 0:
            passed.append("统计文件损坏时降级为无历史")
        else:
            failed.append("统计文件损坏时没有正确降级")

        # 文件写坏后还能被后续写入修好（原子替换）
        store.record({"onnx_mb": 1, "precision": "fp16", "dynamic": False, "imgsz": 640, "seconds": 30})
        if len(store.load()) == 4:
            passed.append("统计条目追加正常（4 条）")
        else:
            failed.append(f"统计条目数量异常：{len(store.load())}")


def test_runners_wiring(passed: list[str], failed: list[str]) -> None:
    """验证 runners 把 tracker 串起来了：喂日志能出进度，收尾能补 100%。"""
    script_body = (
        "import time\n"
        "print('[x] &&&& RUNNING TensorRT.trtexec', flush=True)\n"
        "print('[x] [I] Start parsing network model', flush=True)\n"
        "print('[x] [I] Finish parsing network model', flush=True)\n"
        "print('[x] [I] Local timing cache in use.', flush=True)\n"
        "print('[x] [I] Engine built in 1.5 sec.', flush=True)\n"
        "print('[x] [I] &&&& PASSED TensorRT.trtexec', flush=True)\n"
        "time.sleep(0.2)\n"
    )

    events: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "fake_trtexec.py"
        script.write_text(script_body, encoding="utf-8")
        artifact = Path(tmp) / "fake.engine"
        artifact.write_text("engine", encoding="utf-8")

        # 统计写到临时目录，别让这条 1.5 秒的假日志污染真实基线
        old_stats = os.environ.get("TRTFORGE_ENGINE_STATS")
        os.environ["TRTFORGE_ENGINE_STATS"] = str(Path(tmp) / "stats.json")
        try:
            loop = asyncio.new_event_loop()
            manager = runners.RunManager()
            manager.bind_loop(loop)
            run = manager.create(
                "engine-build",
                [sys.executable, str(script)],
                tmp,
                expect_file=artifact,
                meta=dict(SAMPLE_META),
                estimate=2.0,
            )
            queue, _history = manager.subscribe(run.run_id)

            async def collect() -> None:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30)
                    except asyncio.TimeoutError:
                        break
                    events.append(event)
                    if event.get("type") == "finished":
                        break

            try:
                loop.run_until_complete(collect())
            finally:
                loop.close()
        finally:
            if old_stats is None:
                os.environ.pop("TRTFORGE_ENGINE_STATS", None)
            else:
                os.environ["TRTFORGE_ENGINE_STATS"] = old_stats

    progress_events = [event for event in events if event.get("type") == "progress"]
    stages = [event.get("stage") for event in progress_events]
    if progress_events:
        passed.append(f"runners 接线：产出 {len(progress_events)} 条 progress 事件")
    else:
        failed.append("runners 接线：一条 progress 事件都没有")

    if stages and stages[-1] == "done" and progress_events[-1]["pct"] == 100.0:
        passed.append("runners 接线：PASSED 后到 100%")
    else:
        failed.append(f"runners 接线：收尾阶段异常 {stages}")

    finished = [event for event in events if event.get("type") == "finished"]
    if finished and finished[0].get("status") == "finished":
        passed.append("runners 接线：任务判定为成功")
    else:
        failed.append(f"runners 接线：任务状态异常 {finished}")

    if all("elapsed" in event and "eta" in event for event in progress_events):
        passed.append("runners 接线：progress 事件带 elapsed / eta")
    else:
        failed.append("runners 接线：progress 事件缺 elapsed / eta")


def main() -> int:
    config.ensure_dirs()
    passed: list[str] = []
    failed: list[str] = []

    test_analyzer(passed, failed)
    test_estimate(passed, failed)
    test_runners_wiring(passed, failed)

    print()
    for item in passed:
        print(f"[PASS] {item}")
    for item in failed:
        print(f"[FAIL] {item}")
    print(f"\n通过 {len(passed)} 项，失败 {len(failed)} 项")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
