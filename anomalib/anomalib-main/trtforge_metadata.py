"""TRTForge metadata helpers for Anomalib ONNX/TRT deployment.

The Lightning checkpoint keeps Python model state, while TRTForge needs a small
JSON contract that C++ can read without importing Anomalib. This module writes
that contract and reuses the same parsing rules for ONNX comparison scripts.
When the exported ONNX already contains Anomalib post-processing outputs such as
``pred_score`` and ``anomaly_map``, the deployment contract is 0..1 and the
default image/pixel thresholds are both ``0.5``; TRTForge also uses
``normalization=none`` so it will not apply a second ImageNet normalization.

使用 **torch** 虚拟环境（或任意已安装 anomalib、torch、onnxruntime 的环境）在本目录生成 ``metadata.json``::

    # PowerShell 示例：先激活你的 torch 环境，再设置源码路径
    #   $env:TORCH_PY = "D:\\你的路径\\torch\\Scripts\\python.exe"   # 或直接 activate 后省略
    $env:PYTHONPATH = "$PWD\\src"
    python trtforge_metadata.py "D:\\...\\weights\\lightning" --calib-dir "D:\\...\\bottle\\test"

也可用同目录脚本 ``generate_trtforge_metadata.ps1``，通过环境变量 ``TORCH_PY`` 指向
``torch\\Scripts\\python.exe``，避免把解释器绑死在 PATH 上。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _to_float(value: Any, default: float | None = None) -> float | None:
    """Convert tensors, numpy scalars, and plain numbers to float."""
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_attr_path(obj: Any, path: str) -> Any:
    """Read a dotted attribute path, returning None when any segment is absent."""
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _first_float(obj: Any, paths: Iterable[str], default: float) -> float:
    for path in paths:
        value = _to_float(_get_attr_path(obj, path))
        if value is not None:
            return value
    return default


def extract_model_metadata(model: Any, model_type: str, input_size: tuple[int, int]) -> dict[str, Any]:
    """Extract thresholds and ranges learned by Anomalib post-processing."""
    post = getattr(model, "post_processor", None)
    height, width = input_size
    image_threshold = _first_float(
        post,
        ["image_threshold", "_image_threshold", "normalized_image_threshold"],
        0.5,
    )
    pixel_threshold = _first_float(
        post,
        ["pixel_threshold", "_pixel_threshold", "normalized_pixel_threshold"],
        image_threshold,
    )
    score_min = _first_float(post, ["image_min", "pred_score_min", "min"], 0.0)
    score_max = _first_float(post, ["image_max", "pred_score_max", "max"], 1.0)
    map_min = _first_float(post, ["pixel_min", "anomaly_map_min"], 0.0)
    map_max = _first_float(post, ["pixel_max", "anomaly_map_max"], 1.0)

    return {
        "schema": "trtforge.anomalib.metadata.v1",
        "created_by": "anomalib-main/trtforge_metadata.py",
        "created_at_unix": int(time.time()),
        "model_type": model_type,
        "input_size": [height, width],
        "transform": {"height": height, "width": width},
        "normalization": "imagenet" if model_type.lower() != "efficientad" else "none",
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "pred_scores": {"min": score_min, "max": score_max},
        "anomaly_maps": {"min": map_min, "max": map_max},
        "pred_scores.min": score_min,
        "pred_scores.max": score_max,
        "anomaly_maps.min": map_min,
        "anomaly_maps.max": map_max,
    }


def _metadata_number(metadata: dict[str, Any], dotted_key: str, default: float) -> float:
    if dotted_key in metadata and isinstance(metadata[dotted_key], (int, float)):
        return float(metadata[dotted_key])
    parent, child = dotted_key.split(".", 1)
    nested = metadata.get(parent)
    if isinstance(nested, dict) and isinstance(nested.get(child), (int, float)):
        return float(nested[child])
    return default


def normalize_with_metadata(value: np.ndarray | float, threshold: float, min_val: float, max_val: float):
    """Use the same threshold-centered normalization as Anomalib/TRTForge."""
    denom = max(max_val - min_val, 1.0e-12)
    return np.clip(((value - threshold) / denom) + 0.5, 0.0, 1.0)


def metadata_input_size(metadata: dict[str, Any], fallback: tuple[int, int] = (256, 256)) -> tuple[int, int]:
    """Return input size as (height, width), supporting old and new metadata keys."""
    if isinstance(metadata.get("input_size"), list) and len(metadata["input_size"]) >= 2:
        return int(metadata["input_size"][0]), int(metadata["input_size"][1])
    transform = metadata.get("transform")
    if isinstance(transform, dict) and "height" in transform and "width" in transform:
        return int(transform["height"]), int(transform["width"])
    return fallback


def preprocess_image_for_onnx(
    image_path: str | Path,
    metadata: dict[str, Any],
    model_type: str | None = None,
) -> np.ndarray:
    """Preprocess image to NCHW float32, matching TRTForge's anomaly path."""
    height, width = metadata_input_size(metadata)
    image = Image.open(image_path).convert("RGB").resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    normalization = str(metadata.get("normalization", "")).lower()
    effective_model = (model_type or str(metadata.get("model_type", ""))).lower()
    if normalization == "imagenet" or (normalization == "" and "efficient" not in effective_model):
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def parse_anomaly_outputs(
    outputs: list[np.ndarray],
    output_names: list[str] | None = None,
) -> tuple[float, np.ndarray | None]:
    """Parse Anomalib outputs using the same conservative rule as TRTForge C++."""
    if not outputs:
        raise ValueError("ONNX model returned no outputs")
    arrays = [np.asarray(out) for out in outputs]
    names = [name.lower() for name in output_names] if output_names else [""] * len(arrays)

    score_idx: int | None = None
    map_idx: int | None = None
    for idx, (name, arr) in enumerate(zip(names, arrays, strict=False)):
        if "score" in name:
            score_idx = idx
        if "anomaly_map" in name or ("map" in name and "mask" not in name):
            map_idx = idx

    if score_idx is None:
        for idx, (name, arr) in enumerate(zip(names, arrays, strict=False)):
            if arr.size == 1 and "label" not in name:
                score_idx = idx
                break
    if map_idx is None:
        map_idx = int(np.argmax([arr.size for arr in arrays]))

    anomaly_map = np.squeeze(arrays[map_idx])
    if score_idx is not None and score_idx != map_idx:
        score = float(np.ravel(arrays[score_idx])[0])
    else:
        score = float(np.max(anomaly_map))
    return score, anomaly_map.astype(np.float32, copy=False) if anomaly_map is not None else None


def run_onnx_inference(
    onnx_path: str | Path,
    image_path: str | Path,
    metadata: dict[str, Any],
    model_type: str | None = None,
) -> dict[str, Any]:
    """Run ONNX Runtime and return raw/normalized anomaly outputs."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    tensor = preprocess_image_for_onnx(image_path, metadata, model_type)
    start = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    raw_score, raw_map = parse_anomaly_outputs(outputs, output_names)

    image_threshold = float(metadata.get("image_threshold", 0.5))
    pixel_threshold = float(metadata.get("pixel_threshold", 0.5))
    score_min = _metadata_number(metadata, "pred_scores.min", 0.0)
    score_max = _metadata_number(metadata, "pred_scores.max", 1.0)
    map_min = _metadata_number(metadata, "anomaly_maps.min", 0.0)
    map_max = _metadata_number(metadata, "anomaly_maps.max", 1.0)
    normalized_score = float(normalize_with_metadata(raw_score, image_threshold, score_min, score_max))
    normalized_map = (
        normalize_with_metadata(raw_map, pixel_threshold, map_min, map_max)
        if raw_map is not None
        else None
    )
    return {
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "is_anomaly": normalized_score >= 0.5,
        "anomaly_map": raw_map,
        "normalized_map": normalized_map,
        "elapsed_ms": elapsed_ms,
    }


def calibrate_metadata_from_onnx(
    onnx_path: str | Path,
    image_paths: Iterable[str | Path],
    metadata: dict[str, Any],
    model_type: str | None = None,
    limit: int = 32,
) -> dict[str, Any]:
    """Update score/map min-max ranges by running ONNX on sample images.

    Calibration is best-effort. If ``onnxruntime`` is unavailable or no images are
    provided, the original metadata is returned unchanged.
    """
    paths = [Path(p) for p in image_paths if Path(p).is_file()][:limit]
    if not paths:
        return metadata

    try:
        import onnxruntime as ort
    except ImportError:
        print("metadata calibration skipped: onnxruntime is not installed")
        return metadata

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    scores: list[float] = []
    map_mins: list[float] = []
    map_maxs: list[float] = []
    for path in paths:
        tensor = preprocess_image_for_onnx(path, metadata, model_type)
        outputs = session.run(None, {input_name: tensor})
        score, anomaly_map = parse_anomaly_outputs(outputs, output_names)
        scores.append(float(score))
        if anomaly_map is not None:
            map_mins.append(float(np.min(anomaly_map)))
            map_maxs.append(float(np.max(anomaly_map)))

    if scores:
        metadata["pred_scores"]["min"] = min(scores)
        metadata["pred_scores"]["max"] = max(scores)
        metadata["pred_scores.min"] = min(scores)
        metadata["pred_scores.max"] = max(scores)
    if map_mins and map_maxs:
        metadata["anomaly_maps"]["min"] = min(map_mins)
        metadata["anomaly_maps"]["max"] = max(map_maxs)
        metadata["anomaly_maps.min"] = min(map_mins)
        metadata["anomaly_maps.max"] = max(map_maxs)
    metadata["calibration"] = {
        "source": "onnxruntime",
        "image_count": len(paths),
    }
    return metadata


def write_metadata_json(
    output_path: str | Path,
    metadata: dict[str, Any],
    ckpt_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
) -> Path:
    """Write metadata JSON with trace paths."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    if ckpt_path is not None:
        payload["ckpt_path"] = str(Path(ckpt_path).resolve())
    if onnx_path is not None:
        payload["onnx_path"] = str(Path(onnx_path).resolve())
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_anomalib_model_from_ckpt(ckpt_path: str | Path) -> tuple[Any, str]:
    """Load ``AnomalibModule`` from a Lightning ``model.ckpt`` file."""
    import torch

    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"checkpoint 中缺少 'model' 键：{path}")
    model = ckpt["model"]
    model.eval()
    return model, type(model).__name__


def infer_input_size_from_onnx(onnx_path: str | Path) -> tuple[int, int] | None:
    """Read static NCHW input shape from ONNX if all dimensions are fixed."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        if len(shape) < 4:
            return None
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h, w
    except Exception:
        pass
    return None


def onnx_has_postprocessed_outputs(onnx_path: str | Path) -> bool:
    """Return True when ONNX already exports Anomalib post-processed 0..1 outputs."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        names = {output.name.lower() for output in model.graph.output}
        return any("pred_score" in name for name in names) and any("anomaly_map" in name for name in names)
    except Exception:
        return False


def apply_postprocessed_output_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Use the 0..1 contract produced by Anomalib post-processing in exported ONNX."""
    metadata["output_contract"] = "anomalib_post_processed_0_1"
    metadata["normalization"] = "none"
    metadata["image_threshold"] = 0.5
    metadata["pixel_threshold"] = 0.5
    metadata["pred_scores"] = {"min": 0.0, "max": 1.0}
    metadata["anomaly_maps"] = {"min": 0.0, "max": 1.0}
    metadata["pred_scores.min"] = 0.0
    metadata["pred_scores.max"] = 1.0
    metadata["anomaly_maps.min"] = 0.0
    metadata["anomaly_maps.max"] = 1.0
    return metadata


def infer_model_type_from_context(weights_dir: Path, metadata_path: Path) -> str:
    """Infer model type when the checkpoint only stores a Lightning state_dict."""
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(existing.get("model_type"), str) and existing["model_type"]:
                return existing["model_type"]
        except Exception:
            pass
    for part in reversed(weights_dir.parts):
        lowered = part.lower()
        if lowered == "patchcore":
            return "Patchcore"
        if lowered == "padim":
            return "Padim"
        if lowered == "efficientad":
            return "EfficientAd"
    return "Patchcore"


def make_postprocessed_metadata(model_type: str, input_size: tuple[int, int]) -> dict[str, Any]:
    """Create metadata for ONNX models that already emit normalized Anomalib outputs."""
    height, width = input_size
    metadata = {
        "schema": "trtforge.anomalib.metadata.v1",
        "created_by": "anomalib-main/trtforge_metadata.py",
        "created_at_unix": int(time.time()),
        "model_type": model_type,
        "input_size": [height, width],
        "transform": {"height": height, "width": width},
        "normalization": "none",
    }
    return apply_postprocessed_output_contract(metadata)


def collect_calibration_images(root: Path, limit: int = 64) -> list[Path]:
    """Collect image paths under ``root`` for ONNX min/max calibration."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in suffixes and p.is_file()]
    return paths[:limit]


def generate_metadata_for_weights_dir(
    weights_dir: str | Path,
    *,
    input_size: tuple[int, int] | None = None,
    calib_root: str | Path | None = None,
    calib_limit: int = 32,
) -> Path:
    """Load ``model.ckpt`` / ``model.onnx`` from ``weights_dir`` and write ``metadata.json`` there.

    Parameters
    ----------
    weights_dir
        目录内应包含 ``model.ckpt``；若存在 ``model.onnx`` 则尽量读取静态输入尺寸并可选校准范围。
    input_size
        若 ONNX 为动态输入且未给出，则使用 ``(256, 256)``。
    calib_root
        若提供且存在 ``model.onnx``，则用该目录下图片做 ONNX 校准（扩大 pred_scores / anomaly_maps 范围）。
    """
    d = Path(weights_dir).resolve()
    ckpt_path = d / "model.ckpt"
    onnx_path = d / "model.onnx"
    out_path = d / "metadata.json"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 {ckpt_path}")

    h, w = (256, 256)
    if onnx_path.is_file():
        inferred = infer_input_size_from_onnx(onnx_path)
        if inferred is not None:
            h, w = inferred
    if input_size is not None:
        h, w = input_size

    postprocessed_outputs = onnx_path.is_file() and onnx_has_postprocessed_outputs(onnx_path)
    try:
        model, model_type = load_anomalib_model_from_ckpt(ckpt_path)
        metadata = extract_model_metadata(model, model_type=model_type, input_size=(h, w))
    except KeyError:
        if not postprocessed_outputs:
            raise
        model_type = infer_model_type_from_context(d, out_path)
        metadata = make_postprocessed_metadata(model_type, input_size=(h, w))

    if postprocessed_outputs and metadata.get("output_contract") != "anomalib_post_processed_0_1":
        metadata = apply_postprocessed_output_contract(metadata)

    if onnx_path.is_file() and calib_root is not None and not postprocessed_outputs:
        calib = Path(calib_root)
        if calib.is_dir():
            imgs = collect_calibration_images(calib, limit=calib_limit)
            if imgs:
                metadata = calibrate_metadata_from_onnx(
                    onnx_path,
                    imgs,
                    metadata,
                    model_type=model_type,
                    limit=calib_limit,
                )
        else:
            print(f"calibration skipped: calib dir 不存在 {calib}")

    return write_metadata_json(
        out_path,
        metadata,
        ckpt_path=ckpt_path,
        onnx_path=onnx_path if onnx_path.is_file() else None,
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="从 model.ckpt / model.onnx 生成 TRTForge metadata.json")
    p.add_argument(
        "weights_dir",
        type=Path,
        help="含 model.ckpt（及可选 model.onnx）的目录，例如 .../weights/lightning",
    )
    p.add_argument("--input-height", type=int, default=None, help="固定输入高（ONNX 为动态 shape 时使用）")
    p.add_argument("--input-width", type=int, default=None, help="固定输入宽")
    p.add_argument("--calib-dir", type=Path, default=None, help="校准用图片根目录（递归收集）")
    p.add_argument("--calib-limit", type=int, default=32, help="校准时最多使用多少张图")

    args = p.parse_args()
    inp: tuple[int, int] | None = None
    if args.input_height is not None and args.input_width is not None:
        inp = (args.input_height, args.input_width)
    elif args.input_height is not None or args.input_width is not None:
        p.error("请同时指定 --input-height 与 --input-width，或都不指定")

    path = generate_metadata_for_weights_dir(
        args.weights_dir,
        input_size=inp,
        calib_root=args.calib_dir,
        calib_limit=args.calib_limit,
    )
    print(f"wrote: {path}")


if __name__ == "__main__":
    _cli()

