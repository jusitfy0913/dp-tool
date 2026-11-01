# -*- coding: utf-8 -*-
"""引擎构建进度推断 + 耗时估算 + 历史统计。

## 为什么需要这个模块（先读这段，别以为百分比是真的）

trtexec 在整个 tactic 选择阶段**一行日志都不打**。真实日志
`6-web/runtime/logs/f066553e4df2.log` 里：

```
[10/31/2026-00:02:21] [I] [TRT] Local timing cache in use. ...
[10/31/2026-00:10:14] [I] [TRT] Total Activation Memory: 4432243712
[10/31/2026-00:10:14] [I] Engine built in 474.149 sec.
```

中间 473 秒没有任何输出。所以界面上那个百分比**不可能是真实进度**，只能按
「历史耗时 + 阶段边界」推算。本模块提供三件事：

1. `EngineBuildAnalyzer`：从 trtexec 的每一行识别当前处于哪个阶段；
2. 用 `elapsed / estimate` 换算百分比，并夹在该阶段的上下限之间 ——
   保证 ① 百分比单调不减 ② 只有 `&&&& PASSED` 才到 100%；
3. `EngineStatsStore`：把真实耗时写进 `runtime/engine_stats.json`，
   下次估算立刻变准（越用越准，不需要人工维护基线）。

前端 `js/engine.js` 里的 `STAGES` 必须与本文件的 `STAGE_ORDER` 一一对应。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config

# ---------------------------------------------------------------------------
# 阶段定义
# ---------------------------------------------------------------------------
#: 阶段顺序，与前端 STAGES 数组严格对应。改这里就要同步改 engine.js。
STAGE_ORDER = [
    "start",
    "parse",
    "parsed",
    "build",
    "built",
    "deserialize",
    "benchmark",
    "done",
]

#: 给用户看的中文名。日志面板与进度条都显示它。
STAGE_LABELS = {
    "start": "启动 trtexec",
    "parse": "解析 ONNX",
    "parsed": "解析完成",
    "build": "构建与 tactic 选择",
    "built": "序列化引擎",
    "deserialize": "校验引擎",
    "benchmark": "推理基准",
    "done": "完成",
}

#: 每个阶段允许的百分比区间 (下限, 上限)。
#:
#: 上限刻意不衔接 100：`build` 封顶 94，`benchmark` 封顶 99，只有 PASSED 才给 100。
#: 原因是估算必然有误差（历史数据不足、GPU 被别人占用、模型改动），
#: 进度条冲到 100% 却还没结束，比停在 94% 更让人怀疑是不是卡死了。
STAGE_BANDS: dict[str, tuple[float, float]] = {
    "start": (1.0, 3.0),
    "parse": (3.0, 6.0),
    "parsed": (6.0, 8.0),
    "build": (8.0, 94.0),
    "built": (94.0, 97.0),
    "deserialize": (97.0, 98.0),
    "benchmark": (98.0, 99.0),
    "done": (100.0, 100.0),
}

#: 日志关键字 -> 阶段。按顺序匹配，同一行只认第一条命中。
#: 关键字全部取自真实日志，不要凭印象写。
STAGE_MARKERS: list[tuple[str, str]] = [
    ("&&&& PASSED", "done"),
    ("&&&& FAILED", "done"),
    ("Engine deserialized in", "deserialize"),
    ("Starting inference", "benchmark"),
    ("Starting calibration", "benchmark"),
    ("Total Host Walltime", "benchmark"),
    ("Engine built in", "built"),
    ("Local timing cache in use", "build"),
    ("Profiling results in this builder pass will not be stored", "build"),
    ("Finish parsing network model", "parsed"),
    ("Start parsing network model", "parse"),
    ("&&&& RUNNING", "start"),
]

#: `[10/31/2026-00:10:14] [I] Engine built in 474.149 sec.`
_ENGINE_BUILT_RE = re.compile(r"Engine built in\s+([0-9]+(?:\.[0-9]+)?)\s*sec")

# ---------------------------------------------------------------------------
# 估算基线
# ---------------------------------------------------------------------------
#: 每 MB ONNX 在 960×960、FP16 下的构建秒数。
#:
#: 来自实测：PCB 模型 12.39 MB / 960 输入 / FP16 / T1000，`Engine built in 474.149 sec.`
#: 所以 474.149 / 12.39 ≈ 38.3。
BASELINE_SECONDS_PER_MB = 38.3

#: 基线对应的输入边长。低于它按幂律折算，高于它同理。
BASELINE_IMGSZ = 960

#: 输入尺寸对构建耗时的影响指数。
#:
#: 一开始按「面积」取 2（12.39 MB @960 = 474 s，则 12.1 MB @320 只需 53 s），
#: 实测被打脸：同一网络结构在 320 上仍要 **317 s**（5 分 17 秒）。原因有两个：
#:   1. 耗时主要由「层数 × 每层候选 kernel 数」决定，ONNX 体积才是层数的代理，
#:      输入尺寸只影响每次候选试跑的成本；
#:   2. 分辨率小 → 激活值小 → 显存里能塞下更多 tactic，搜索空间反而更大。
#: 用 960 的 474 s 和 320 的 317 s 反解：log(317/474)/log(320/960) ≈ 0.37。
#: 取 0.4 作为保守值（宁可估长一点，也不要估短了让进度条提前顶到上限）。
IMGSZ_EXPONENT = 0.4

MIN_ESTIMATE_SECONDS = 30.0
MAX_ESTIMATE_SECONDS = 3600.0
#: 连 ONNX 体积都拿不到（探测失败）时的兜底值，对应实测的 8 分钟量级。
DEFAULT_ESTIMATE_SECONDS = 480.0
#: 历史条目最多保留多少条，超出丢弃最旧的。
MAX_STATS_ENTRIES = 50


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ratio_close(a: float, b: float, tolerance: float) -> bool:
    """两个正数是否在 tolerance 倍以内（1.0 = 必须相等）。"""
    if a <= 0 or b <= 0:
        return False
    ratio = a / b
    return (1.0 / tolerance) <= ratio <= tolerance


# ---------------------------------------------------------------------------
# 历史统计
# ---------------------------------------------------------------------------
class EngineStatsStore:
    """真实构建耗时的落盘存储。

    文件结构：一个 JSON 数组，每项形如
    `{"onnx_mb":12.39,"precision":"fp16","dynamic":false,"imgsz":960,"seconds":474.1,"ts":...}`

    读失败一律降级为「没有历史」：统计文件坏了不该让构建页打不开。
    写失败同理，只是下次估算还停在旧基线，不影响本次构建。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> list[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        try:
            data = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(data, list):
            return []
        # 只接受结构正确的条目，防止半截写入或人工编辑把估算带偏
        return [
            item
            for item in data
            if isinstance(item, dict) and isinstance(item.get("seconds"), (int, float)) and item["seconds"] > 0
        ]

    def record(self, entry: dict) -> None:
        """追加一条真实耗时。写临时文件再 replace，避免读到半截 JSON。"""
        with self._lock:
            items = self.load()
            items.append(entry)
            items = items[-MAX_STATS_ENTRIES:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(
                    json.dumps(items, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
            except OSError:
                # 统计是「锦上添花」，写不进去也不能影响构建本身
                pass


def default_store() -> EngineStatsStore:
    """统计文件的默认位置：`6-web/runtime/engine_stats.json`。

    可用环境变量 `TRTFORGE_ENGINE_STATS` 改到别处，测试会用它把统计写到临时目录，
    免得测试里那条假日志（`Engine built in 1.5 sec.`）污染真实估算基线。
    """
    override = os.environ.get("TRTFORGE_ENGINE_STATS", "").strip()
    if override:
        return EngineStatsStore(Path(override).expanduser())
    return EngineStatsStore(config.RUNTIME_DIR / "engine_stats.json")


def estimate_seconds(meta: dict, store: EngineStatsStore | None = None) -> float:
    """估算一次构建要多久（秒）。

    `meta` 需要 `onnx_mb` / `precision` / `dynamic` / `imgsz` 四项。

    优先级：
    1. 历史里精度、dynamic、体积（±67%）、分辨率（±20%）都对得上的条目 → 取中位数；
       用中位数而不是平均值，避免某次磁盘/GPU 抖动把估算带偏。
    2. 没有历史 → 按实测基线外推（每 MB × 尺寸幂律，见 BASELINE_* 常量）。
    3. 连体积都不知道 → 兜底 480 秒。
    """
    onnx_mb = float(meta.get("onnx_mb") or 0.0)
    imgsz = int(meta.get("imgsz") or 640)
    precision = str(meta.get("precision") or "fp16")
    dynamic = bool(meta.get("dynamic"))

    if store is not None and onnx_mb > 0:
        matches = [
            item
            for item in store.load()
            if str(item.get("precision") or "fp16") == precision
            and bool(item.get("dynamic")) == dynamic
            and _ratio_close(float(item.get("onnx_mb") or 0.0), onnx_mb, 1.67)
            and _ratio_close(float(item.get("imgsz") or 640), float(imgsz), 1.2)
        ]
        if matches:
            return _clamp(_median([float(item["seconds"]) for item in matches]), MIN_ESTIMATE_SECONDS, MAX_ESTIMATE_SECONDS)

    if onnx_mb <= 0:
        return DEFAULT_ESTIMATE_SECONDS

    seconds = (
        BASELINE_SECONDS_PER_MB
        * onnx_mb
        * (max(imgsz, 1) / float(BASELINE_IMGSZ)) ** IMGSZ_EXPONENT
    )
    if precision == "fp32":
        # fp32 的 kernel 候选更少（没有 FP16 的额外转换层），实测比 fp16 略快
        seconds *= 0.7
    return _clamp(seconds, MIN_ESTIMATE_SECONDS, MAX_ESTIMATE_SECONDS)


# ---------------------------------------------------------------------------
# 进度推断
# ---------------------------------------------------------------------------
class EngineBuildAnalyzer:
    """把 trtexec 的日志流翻译成「阶段 + 估算百分比」。

    线程安全：`feed` 在读线程里调用，`tick` 在心跳线程里调用，`elapsed` 属性
    给快照用，三者都会碰 `pct`，所以统一走 `_lock`。

    `started_at` 用 `time.time()`（壁钟）而不是 monotonic，因为 `Run.started_at`
    也是壁钟，两边必须同一把尺子才能算出正确的已用时间。
    """

    def __init__(
        self,
        estimate_seconds: float | None,
        *,
        meta: dict | None = None,
        stats: EngineStatsStore | None = None,
        started_at: float | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.estimate = float(estimate_seconds) if estimate_seconds else None
        self.meta = dict(meta or {})
        self.stats = stats
        self._clock = clock
        self.started_at = started_at if started_at is not None else clock()
        self.stage = "start"
        self.pct = STAGE_BANDS["start"][0]
        self.ok: bool | None = None
        self.actual_seconds: float | None = None
        self._lock = threading.Lock()
        self._recorded = False

    # -- 对外只读信息 ------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return max(0.0, self._clock() - self.started_at)

    # -- 日志喂入 ----------------------------------------------------------
    @staticmethod
    def stage_for(line: str) -> str | None:
        """一行日志对应哪个阶段；没有关键字就返回 None（绝大多数行都是 None）。"""
        for needle, stage in STAGE_MARKERS:
            if needle in line:
                return stage
        return None

    def feed(self, line: str) -> dict | None:
        """喂一行日志。只有阶段发生变化时才返回事件，普通行返回 None。"""
        stage = self.stage_for(line)
        if stage is None:
            return None

        with self._lock:
            if self.ok is not None:
                # 已经结束（PASSED/FAILED 之后还有几行收尾日志），不再改状态
                return None

            self._capture_actual_seconds(line)

            if stage == "done":
                self.stage = "done"
                self.pct = 100.0
                self.ok = "FAILED" not in line.upper()
                return self._event()

            if stage == self.stage:
                # 同一阶段的重复关键字（例如 build 阶段的两条标记），只把百分比往前推
                new_pct = max(self.pct, self._pct_now())
                if int(new_pct) <= int(self.pct):
                    return None
                self.pct = new_pct
                return self._event()

            self.stage = stage
            self.pct = max(self.pct, self._pct_now())
            return self._event()

    def tick(self) -> dict | None:
        """心跳：日志静默时靠它推进百分比。只有整数百分比变化才返回事件。

        只按整数变化发布，是为了压事件量：一次 1 小时的构建大约 100 条，
        不会把 `EVENT_BUFFER_SIZE`（2000）的历史缓冲挤掉有用的日志。
        """
        with self._lock:
            if self.ok is not None:
                return None
            new_pct = max(self.pct, self._pct_now())
            if int(new_pct) <= int(self.pct):
                return None
            self.pct = new_pct
            return self._event()

    def finish(self, status: str, exit_code: int | None = None) -> dict | None:
        """进程结束后收尾。

        成功（退出码 0）时补一条 100%；失败/取消时**不**发进度事件——
        百分比停在原处，由 `finished` 事件说明结果，避免「进度条冲到 100 但失败」。
        """
        with self._lock:
            if self.ok is not None:
                return None
            if status == "finished" and exit_code == 0:
                self.stage = "done"
                self.pct = 100.0
                self.ok = True
                return self._event()
            self.ok = False
            return None

    # -- 内部 --------------------------------------------------------------
    def _capture_actual_seconds(self, line: str) -> None:
        """从 `Engine built in 474.149 sec.` 抓真实耗时并落盘。

        抓的是「构建段」耗时（不含解析与推理基准），正是估算要预测的那一段，
        所以直接存它，不做换算。
        """
        if self._recorded:
            return
        match = _ENGINE_BUILT_RE.search(line)
        if not match:
            return
        try:
            seconds = float(match.group(1))
        except ValueError:
            return
        if seconds <= 0:
            return
        self._recorded = True
        self.actual_seconds = seconds
        if self.stats is None:
            return
        entry: dict[str, Any] = {
            "onnx_mb": round(float(self.meta.get("onnx_mb") or 0.0), 2),
            "precision": str(self.meta.get("precision") or "fp16"),
            "dynamic": bool(self.meta.get("dynamic")),
            "imgsz": int(self.meta.get("imgsz") or 640),
            "seconds": round(seconds, 2),
            "ts": round(time.time(), 1),
        }
        self.stats.record(entry)

    def _pct_now(self) -> float:
        low, high = STAGE_BANDS[self.stage]
        if not self.estimate or self.estimate <= 0:
            return low
        ratio = self.elapsed / self.estimate * 100.0
        return _clamp(ratio, low, high)

    def _event(self) -> dict:
        elapsed = self.elapsed
        eta = None if not self.estimate else max(0.0, self.estimate - elapsed)
        return {
            "type": "progress",
            "stage": self.stage,
            "stage_label": STAGE_LABELS[self.stage],
            "stage_index": STAGE_ORDER.index(self.stage),
            "pct": round(self.pct, 1),
            "elapsed": round(elapsed, 1),
            "eta": None if eta is None else round(eta, 1),
            "estimate": None if self.estimate is None else round(self.estimate, 1),
        }


def run_heartbeat(
    analyzer: EngineBuildAnalyzer,
    publish: Callable[[dict], None],
    stop_event: threading.Event,
    interval: float = 2.0,
) -> None:
    """心跳线程主体：每 interval 秒问一次进度，有变化就发布。

    单独抽成函数而不是写在 runners.py 里，是为了让测试可以直接调它，
    不必起子进程。
    """
    while not stop_event.wait(interval):
        try:
            payload = analyzer.tick()
        except Exception:  # noqa: BLE001 - 心跳线程不能因为一次异常整条死掉
            continue
        if payload is not None:
            publish(payload)
