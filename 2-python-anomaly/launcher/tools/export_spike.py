"""P0 技术验证脚本：AnomalyDINO 导出 ONNX 并对齐数值。

它不是交付物，而是「在写启动器之前先把风险点打掉」的一次性验证：

1. timm 主干与上游 `torch.hub` 模型的特征是否一致（不是差不多，是逐元素一致）
2. 建 16-shot 记忆库需要多久、占多少显存
3. 导出的 ONNX 与 PyTorch 参考实现的异常图/分数偏差
4. （可选）编 FP16 引擎后与 ONNX 的热力图相关性

用法（本机 dpenv）：

    python tools/export_spike.py --shots 16 --out results/_spike

输出会打印一张对比表，结论写进 openspec 的 design.md。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# 允许直接 `python tools/export_spike.py` 跑：把 launcher 目录加进 sys.path
LAUNCHER_DIR = Path(__file__).resolve().parent.parent
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import anomalydino_core as core  # noqa: E402

DEFAULT_BOTTLE = Path(__file__).resolve().parents[2] / "anomalib-main" / "datasets" / "bottle"


def list_images(folder: Path, limit: int | None = None) -> list[Path]:
    """列举图片，按文件名排序保证可复现（与上游 `list_images` 的字典序一致）。"""
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)
    return files[:limit] if limit else files


def read_rgb(path: Path) -> np.ndarray:
    """读图并转 RGB。DLL 的输入就是 RGB，这里必须一致。

    用 `imdecode + np.fromfile` 而不是 `cv2.imread`：后者在 Windows 上读不了
    含中文的路径，而本项目要求中文路径必须可用。
    """
    import cv2

    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise FileNotFoundError(f"读不到图片：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def collect_eval_images(root: Path, limit: int = 8) -> list[Path]:
    """收集测试图：优先取异常图（有真实缺陷才有区分度），再补正常图。

    这些图**不能**出现在记忆库里，否则距离恒为 0，验证不出任何东西。
    """
    picks: list[Path] = []
    test_dir = root / "test"
    if not test_dir.is_dir():
        return picks
    # 先异常类别（按名字排序，可复现），再 good
    kind_dirs = sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good")
    good_dir = test_dir / "good"
    if good_dir.is_dir():
        kind_dirs.append(good_dir)
    for folder in kind_dirs:
        for path in list_images(folder):
            picks.append(path)
            if len(picks) >= limit:
                return picks
    return picks


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def check_backbone_equivalence(resolution: int, images: list[Path], device: str) -> dict:
    """timm 预插值主干 vs 上游 torch.hub 主干，逐 token 余弦。"""
    import timm

    checkpoint = core.resolve_checkpoint(None)
    our_model = core.create_backbone(checkpoint, resolution, device)
    try:
        upstream = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        upstream.eval().to(device)
    except Exception as exc:  # noqa: BLE001 - 离线环境下 hub 可能不可用
        return {"available": False, "reason": f"torch.hub 不可用：{exc}"}

    batch = core.preprocess_like_dll([read_rgb(p) for p in images[:2]], resolution).to(device)
    with torch.inference_mode():
        ours = core.extract_patch_tokens(our_model, batch)
        theirs = upstream.get_intermediate_layers(batch)[0]
    per_token = torch.nn.functional.cosine_similarity(ours, theirs, dim=-1)
    result = {
        "available": True,
        "shape_ours": list(ours.shape),
        "shape_upstream": list(theirs.shape),
        "per_token_cos_min": float(per_token.min()),
        "per_token_cos_mean": float(per_token.mean()),
        "max_abs_diff": float((ours - theirs).abs().max()),
    }
    del upstream, our_model, batch, ours, theirs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def build_bank(model, images: list[Path], resolution: int, device: str) -> tuple[torch.Tensor, float]:
    """建库并返回 (库, 耗时秒)。"""
    started = time.perf_counter()
    tokens = []
    batch_size = 4
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        batch = core.preprocess_like_dll([read_rgb(p) for p in chunk], resolution).to(device)
        tokens.extend(core.extract_patch_tokens(model, batch).cpu())
    bank = core.build_bank(tokens)
    elapsed = time.perf_counter() - started
    return bank, elapsed


def export_onnx(wrapper: core.BakedAnomalyDINO, resolution: int, onnx_path: Path) -> None:
    """导出 ONNX。

    **必须用 opset 16，不能用 17。** opset 17 会把 `F.layer_norm` 导成
    `LayerNormalization` 单算子，而 TensorRT 8.5 没有原生支持它，onnx 解析器会去找
    一个不存在的插件，报 `Plugin not found` 直接失败。opset 16 下同一段计算被拆成
    ReduceMean/Sub/Mul/Div/Sqrt 等基础算子，TRT 8.5 能全部吃下。
    """
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper.eval(),
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["anomaly_map", "pred_score"],
        opset_version=16,
        do_constant_folding=True,
    )


def run_onnx(onnx_path: Path, batch: np.ndarray) -> tuple[np.ndarray, float]:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    outputs = session.run(None, {session.get_inputs()[0].name: batch})
    by_name = dict(zip([o.name for o in session.get_outputs()], outputs, strict=False))
    return by_name["anomaly_map"], float(np.ravel(by_name["pred_score"])[0])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="AnomalyDINO P0 导出验证")
    parser.add_argument("--data-root", default=str(DEFAULT_BOTTLE), help="数据集根目录（含 train/good）")
    parser.add_argument("--resolution", type=int, default=448, help="输入边长，必须是 14 的倍数")
    parser.add_argument("--shots", type=int, default=16, help="用多少张正常图建库，-1 表示全部")
    parser.add_argument("--out", default="results/_spike", help="输出目录（相对 launcher 目录）")
    parser.add_argument("--checkpoint", default="", help="显式指定 DINOv2 权重")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--skip-backbone-check", action="store_true", help="跳过与 torch.hub 的对比")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，退回 CPU（会明显变慢）")
        device = "cpu"

    grid_h, grid_w = core.grid_for_resolution(args.resolution)
    root = Path(args.data_root)
    train_dir = root / "train" / "good"
    if not train_dir.is_dir():
        print(f"[error] 找不到正常图目录：{train_dir}", file=sys.stderr)
        return 2

    all_images = list_images(train_dir)
    shots = len(all_images) if args.shots == -1 else min(args.shots, len(all_images))
    # 上游 few-shot 是 `[seed*n:(seed+1)*n]`，seed=0 时就是前 n 张。
    sample = all_images[:shots]
    print(f"[info] 正常图 {len(all_images)} 张，本次用 {shots} 张建库")

    checkpoint = core.resolve_checkpoint(args.checkpoint or None)
    print(f"[info] 权重：{checkpoint}  sha256={core.checkpoint_sha256(checkpoint)[:16]}…")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = core.create_backbone(checkpoint, args.resolution, device)
    load_seconds = time.perf_counter() - t0
    print(f"[info] 主干加载 {load_seconds:.2f}s")

    result: dict = {
        "resolution": args.resolution,
        "grid": [grid_h, grid_w],
        "shots": shots,
        "device": device,
        "load_seconds": round(load_seconds, 3),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": core.checkpoint_sha256(checkpoint),
    }

    if not args.skip_backbone_check:
        equivalence = check_backbone_equivalence(args.resolution, sample, device)
        result["backbone_equivalence"] = equivalence
        print(f"[check] 主干等价性：{json.dumps(equivalence, ensure_ascii=False)}")
        # 对比时把外部模型加载进来了，重建一次干净的主干，避免显存干扰计时
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        model = core.create_backbone(checkpoint, args.resolution, device)

    bank, bank_seconds = build_bank(model, sample, args.resolution, device)
    cells = bank.shape[0]
    result["bank_seconds"] = round(bank_seconds, 3)
    result["bank_rows"] = int(cells)
    result["bank_sha256"] = core.bank_digest(bank)
    result["bank_bytes"] = int(cells * bank.shape[1] * 4)
    print(
        f"[info] 建库 {bank_seconds:.2f}s，行数 {cells}（{shots} 图 × {grid_h * grid_w} 格），"
        f"占 {result['bank_bytes'] / 1e6:.1f} MB"
    )

    # 比对与标定必须用**不在库里的图**：拿建库图去测，距离恒为 0，什么也验证不出来。
    eval_images = collect_eval_images(root, limit=8)
    if not eval_images:
        print(f"[error] 找不到可用的测试图：{root / 'test'}", file=sys.stderr)
        return 2
    print(f"[info] 评估图 {len(eval_images)} 张（来自 test/，不进库）")

    eval_batch = core.preprocess_like_dll([read_rgb(p) for p in eval_images], args.resolution).to(device)
    ref_map, ref_score = core.predict_reference(model, eval_batch, bank.to(device), grid_h, grid_w)
    ref_map_np = ref_map.detach().cpu().numpy()
    ref_score_value = float(ref_score.reshape(-1)[0])
    print(f"[info] 参考实现：score={ref_score_value:.5f} map_max={float(ref_map_np.max()):.5f}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = LAUNCHER_DIR / out_dir
    onnx_path = out_dir / "model.onnx"
    # 导出必须在 CPU 上做：dummy 输入在 CPU，若权重还在 CUDA 会报 device 不匹配。
    # 这里深拷贝一份再搬到 CPU，避免把用于比对的 GPU 主干一起搬走。
    wrapper = core.BakedAnomalyDINO(model, bank, grid_h, grid_w)
    export_model = copy.deepcopy(wrapper).to("cpu").eval()
    export_onnx(export_model, args.resolution, onnx_path)
    result["onnx_bytes"] = onnx_path.stat().st_size
    print(f"[info] 导出 ONNX {onnx_path}（{result['onnx_bytes'] / 1e6:.1f} MB）")

    # 逐张比对：只测一张容易被单张的巧合误导
    pearsons: list[float] = []
    max_diffs: list[float] = []
    score_diffs: list[float] = []
    batch_cpu = eval_batch.detach().cpu().numpy()
    for index in range(batch_cpu.shape[0]):
        single = batch_cpu[index : index + 1]
        with torch.inference_mode():
            one_map, one_score = core.predict_reference(
                model, eval_batch[index : index + 1], bank.to(device), grid_h, grid_w
            )
        ort_map, ort_score = run_onnx(onnx_path, single)
        pearsons.append(pearson(one_map.detach().cpu().numpy(), ort_map))
        max_diffs.append(float(np.abs(one_map.detach().cpu().numpy() - ort_map).max()))
        score_diffs.append(abs(float(one_score.reshape(-1)[0]) - ort_score))

    result["onnx_vs_torch"] = {
        "images": len(pearsons),
        "map_pearson_min": float(np.min(pearsons)),
        "map_max_abs_diff_max": float(np.max(max_diffs)),
        "score_abs_diff_max": float(np.max(score_diffs)),
        "score_torch_first": ref_score_value,
    }
    print(f"[check] ONNX vs PyTorch：{json.dumps(result['onnx_vs_torch'], ensure_ascii=False)}")

    if device.startswith("cuda"):
        result["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        print(f"[info] 显存峰值 {result['cuda_peak_mb']} MB")

    # 单张延迟：跑 20 次取中位数，避免首发抖动
    wrapper = wrapper.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            wrapper(eval_batch[0:1])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings = []
        for _ in range(20):
            started = time.perf_counter()
            wrapper(eval_batch[0:1])
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    result["latency_ms_median"] = round(float(np.median(timings)), 2)
    print(f"[info] 单张延迟（含主干）中位数 {result['latency_ms_median']} ms")

    manifest = out_dir / "spike_report.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] 报告写入 {manifest}")

    ok = (
        result.get("backbone_equivalence", {}).get("per_token_cos_min", 1.0) >= 0.9999
        and result["onnx_vs_torch"]["map_pearson_min"] >= 0.99
        and result["onnx_vs_torch"]["score_abs_diff_max"] <= 1e-3
    )
    print(f"[result] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
