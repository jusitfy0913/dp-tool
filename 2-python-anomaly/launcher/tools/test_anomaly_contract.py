# -*- coding: utf-8 -*-
"""输出挑选契约：导出图、标定用的挑选、DLL 的挑选必须是同一套规则。

`anomalib_infer.dll` 在 `anomalib_infer_c.cpp:454-471` 里选「哪个输出是异常图」，
规则里有一条很容易踩的陷阱：**任何**输出只要元素数比当前异常图大，就会顶替它。
所以多导出一个 `raw_mask [1,1,64,64]` 会让真正的 32×32 热力图被换掉，
推理不报错，只是结果全错。

这个测试做两件事：
1. 用两个假 ONNX（合规 / 多一个大输出）跑 ONNX Runtime，把三套实现的结果
   对到同一份：启动器的 `parse_outputs`、`trtforge_metadata.parse_anomaly_outputs`、
   以及按 C++ 源码逐行转写的 `dll_rule`；
2. 记下 `trtforge_metadata` 与 DLL 在一处边角上的已知分歧，说明为什么导出
   只允许两个输出（多输出时靠「碰巧一致」是不可接受的）。

    python 2-python-anomaly/launcher/tools/test_anomaly_contract.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parents[1]
ANOMALIB_DIR = LAUNCHER_DIR.parent / "anomalib-main"
sys.path.insert(0, str(LAUNCHER_DIR))
sys.path.insert(0, str(ANOMALIB_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tf_anomalydino_launcher as launcher  # noqa: E402
from trtforge_metadata import parse_anomaly_outputs as upstream_parse  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


class FakeOutputs(torch.nn.Module):
    """返回预设张量，但结果仍依赖输入，避免导出时 graph.input 被剪空。"""

    def __init__(self, spec: list[tuple[str, tuple[int, ...], float]]) -> None:
        super().__init__()
        self.spec = spec
        extra = 0.0

    def forward(self, x):
        bias = x.mean() * 0.0  # 只制造依赖，不改变数值
        values = []
        for _name, shape, fill in self.spec:
            values.append(torch.full(shape, fill, dtype=torch.float32) + bias)
        return tuple(values)


def export_fake(path: Path, spec: list[tuple[str, tuple[int, ...], float]]) -> None:
    model = FakeOutputs(spec)
    dummy = torch.zeros(1, 3, 56, 56)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=[name for name, _shape, _fill in spec],
        opset_version=16,
        do_constant_folding=True,
    )


def dll_rule(outputs: list[np.ndarray], names: list[str]) -> tuple[float, np.ndarray]:
    """`anomalib_infer_c.cpp:454-471` 的逐行转写（C++ 是权威实现）。

    顺序很重要：名字匹配与比大小写在同一个循环里，后者的结果可能覆盖前者。
    """
    arrays = [np.asarray(value) for value in outputs]
    map_idx: int | None = None
    score_idx: int | None = None
    for index, (name, array) in enumerate(zip(names, arrays, strict=False)):
        lower = name.lower()
        if "score" in lower:
            score_idx = index
        if "anomaly_map" in lower or ("map" in lower and "mask" not in lower):
            map_idx = index
        if map_idx is None or array.size > arrays[map_idx].size:
            map_idx = index
        if array.size == 1 and score_idx is None and "label" not in lower:
            score_idx = index
    if map_idx is None and arrays:
        map_idx = 0
    if score_idx == map_idx:
        score_idx = None
    anomaly_map = arrays[map_idx]
    if score_idx is not None:
        score = float(np.ravel(arrays[score_idx])[0])
    else:
        score = float(np.max(anomaly_map))
    return score, anomaly_map


def run(onnx_path: Path) -> tuple[list[np.ndarray], list[str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 56, 56), np.float32)})
    return list(outputs), names


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="anomaly_contract_"))
    try:
        # ---- 1. 合规导出：三套实现必须给出同一份结果 ----------------------
        good = tmp / "good.onnx"
        export_fake(
            good,
            [
                ("anomaly_map", (1, 1, 32, 32), 0.25),
                ("pred_score", (1, 1), 0.75),
            ],
        )
        outputs, names = run(good)
        check("合规模型有两个输出", names == ["anomaly_map", "pred_score"], str(names))
        launcher_score, launcher_map = launcher.parse_outputs(outputs, names)
        upstream_score, upstream_map = upstream_parse(outputs, names)
        dll_score, dll_map = dll_rule(outputs, names)
        check("合规模型：启动器分数 = 0.75", abs(launcher_score - 0.75) < 1e-6, str(launcher_score))
        check("合规模型：上游工具分数一致", abs(upstream_score - launcher_score) < 1e-6)
        check("合规模型：C++ 规则分数一致", abs(dll_score - launcher_score) < 1e-6)
        # 上游工具会 squeeze 掉单元素维度，所以比元素数而不是形状
        check("合规模型：三套实现挑中同一张图",
              launcher_map.size == upstream_map.size == dll_map.size == 1024,
              f"{launcher_map.size} {upstream_map.size} {dll_map.size}")

        # ---- 2. 危险导出：多一个元素数更大的 mask --------------------------
        # DLL 会把大的那个当异常图，且不报错——所以导出必须只保留两个输出。
        bad = tmp / "bad.onnx"
        export_fake(
            bad,
            [
                ("anomaly_map", (1, 1, 32, 32), 0.25),
                ("pred_score", (1, 1), 0.75),
                ("raw_mask", (1, 1, 64, 64), 0.5),
            ],
        )
        outputs, names = run(bad)
        launcher_score, launcher_map = launcher.parse_outputs(outputs, names)
        upstream_score, upstream_map = upstream_parse(outputs, names)
        dll_score, dll_map = dll_rule(outputs, names)
        check("危险模型：启动器也选大图（与 DLL 一致）", launcher_map.shape == (1, 1, 64, 64),
              str(launcher_map.shape))
        check("危险模型：C++ 规则选大图", dll_map.shape == (1, 1, 64, 64), str(dll_map.shape))
        check("危险模型：上游工具按名字留在小图（与 DLL 分歧）", upstream_map.size == 1024,
              str(upstream_map.shape))
        check("危险模型：分数仍来自 pred_score",
              abs(launcher_score - 0.75) < 1e-6 and abs(dll_score - 0.75) < 1e-6,
              f"{launcher_score} {dll_score}")

        # ---- 3. 名字匹配与比大小在同一个循环里：后者不一定赢 -----------------
        # C++ 是「名字匹配先赋值，再比大小」，所以后面那个叫 anomaly_map 的
        # 小图会盖掉前面的大图。这条容易被"元素数最多者胜"的说法误导。
        early = tmp / "early.onnx"
        export_fake(
            early,
            [
                ("big_map", (1, 1, 64, 64), 0.3),
                ("anomaly_map", (1, 1, 32, 32), 0.25),
            ],
        )
        outputs, names = run(early)
        launcher_score, launcher_map = launcher.parse_outputs(outputs, names)
        upstream_score, upstream_map = upstream_parse(outputs, names)
        dll_score, dll_map = dll_rule(outputs, names)
        check("大图在前：后出现的名字匹配胜出（三套实现一致）",
              launcher_map.size == upstream_map.size == dll_map.size == 1024,
              f"{launcher_map.shape} {upstream_map.shape} {dll_map.shape}")
        check("大图在前：没有分数时取异常图最大值",
              abs(launcher_score - 0.25) < 1e-6 and abs(dll_score - 0.25) < 1e-6,
              f"{launcher_score} {dll_score}")

        # ---- 4. 已知分歧：上游工具只按名字，不比大小 -----------------------
        # 记录它不是为了让分歧"合法"，而是为了说明：导出多输出时不能指望三套
        # 实现碰巧一致，所以 anomalydino 的导出固定只有 anomaly_map + pred_score。
        late = tmp / "late.onnx"
        export_fake(
            late,
            [
                ("anomaly_map", (1, 1, 8, 8), 0.25),
                ("extra", (1, 1, 64, 64), 0.4),
            ],
        )
        outputs, names = run(late)
        _launcher_score, launcher_map = launcher.parse_outputs(outputs, names)
        _upstream_score, upstream_map = upstream_parse(outputs, names)
        _dll_score, dll_map = dll_rule(outputs, names)
        check("边角：DLL 与启动器仍取大图", launcher_map.shape == dll_map.shape == (1, 1, 64, 64),
              f"{launcher_map.shape} {dll_map.shape}")
        check("边角：上游工具按名字留在小图（已知分歧）", upstream_map.size == 64,
              str(upstream_map.shape))

        # ---- 5. 真实导出图也过同一套规则 ----------------------------------
        import anomalydino_core as core

        class TinyBackbone:
            def __init__(self) -> None:
                generator = torch.Generator().manual_seed(1)
                self.tokens = torch.randn(1, 16, 384, generator=generator)

            def forward_features(self, batch):
                return torch.cat(
                    [torch.zeros(batch.shape[0], 1, 384), self.tokens * (1 + batch.mean())], dim=1
                )

            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

        bank = core.build_bank([torch.randn(16, 384) for _ in range(2)])
        wrapper = core.BakedAnomalyDINO(TinyBackbone(), bank, 4, 4, chunk_rows=8)
        real = tmp / "real.onnx"
        launcher.export_onnx(wrapper, 56, real)
        outputs, names = run(real)
        launcher_score, launcher_map = launcher.parse_outputs(outputs, names)
        upstream_score, upstream_map = upstream_parse(outputs, names)
        dll_score, dll_map = dll_rule(outputs, names)
        check("真实导出图：三套实现挑中同一张图",
              launcher_map.size == upstream_map.size == dll_map.size == 16,
              f"{launcher_map.shape} {upstream_map.shape} {dll_map.shape}")
        check("真实导出图：三套实现分数一致",
              abs(launcher_score - upstream_score) < 1e-6 and abs(launcher_score - dll_score) < 1e-6,
              f"{launcher_score} {upstream_score} {dll_score}")

        # ---- 6. metadata 里的阈值能被 DLL 的归一化接受 ---------------------
        meta = {
            "model_type": "AnomalyDINO",
            "normalization": "imagenet",
            "image_threshold": 0.4,
            "pixel_threshold": 0.6,
            "pred_scores.min": 0.05,
            "pred_scores.max": 1.1,
            "anomaly_maps.min": 0.0,
            "anomaly_maps.max": 1.3,
        }
        score_min = float(meta["pred_scores.min"])
        score_max = float(meta["pred_scores.max"])
        normalized = (launcher_score - score_min) / max(score_max - score_min, 1e-6)
        check("分数归一化后落在 [0,1]", 0.0 <= normalized <= 1.0, str(normalized))
        check("阈值也能被同一条归一化接受",
              0.0 <= (meta["image_threshold"] - score_min) / max(score_max - score_min, 1e-6) <= 1.0)
        check("metadata 是合法 JSON", isinstance(json.dumps(meta), str))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  !! {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
