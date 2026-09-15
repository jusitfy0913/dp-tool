# -*- coding: utf-8 -*-
"""TRTForge Anomalib training/export launcher.

Qt starts this script through QProcess. Ordinary stdout/stderr remains visible in
the TrainingTab runtime log, while TRTFORGE_JSON lines drive progress, artifact
lists and error dialogs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


JSON_PREFIX = "TRTFORGE_JSON "
SUPPORTED_MODELS = {"patchcore", "padim", "efficientad"}
SUPPORTED_MODES = {"check-env", "train-export"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def emit(event: dict[str, Any]) -> None:
    print(JSON_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_error(code: str, message: str, **extra: Any) -> None:
    payload = {"type": "error", "code": code, "message": message}
    payload.update(extra)
    emit(payload)


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {"ok": True, "version": str(getattr(module, "__version__", ""))}
    except Exception as exc:  # noqa: BLE001 - report dependency problems to UI
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def current_env_name() -> str:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if conda_env:
        return conda_env
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        return Path(virtual_env).name
    return Path(sys.prefix).name


def add_anomalib_source_path() -> Path:
    """Prefer the workspace anomalib-main source tree when available."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidates = [
            parent / "anomalib" / "anomalib-main",
            parent / "anomalib-main",
        ]
        for root in candidates:
            src = root / "src"
            if src.exists():
                for item in (str(root), str(src)):
                    if item not in sys.path:
                        sys.path.insert(0, item)
                return root
    return script_path.parent


def collect_gpu_status() -> dict[str, Any]:
    status: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            used_bytes = total_bytes - free_bytes
            status.update(
                {
                    "available": True,
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_total_mb": round(total_bytes / 1024 / 1024, 1),
                    "memory_used_mb": round(used_bytes / 1024 / 1024, 1),
                    "memory_free_mb": round(free_bytes / 1024 / 1024, 1),
                }
            )
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.check_output(
                [
                    smi,
                    "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2,
            ).strip()
            if output:
                parts = [part.strip() for part in output.splitlines()[0].split(",")]
                if len(parts) >= 4:
                    status.update(
                        {
                            "available": True,
                            "utilization_gpu_pct": float(parts[0]),
                            "memory_used_mb": float(parts[1]),
                            "memory_total_mb": float(parts[2]),
                            "name": parts[3] or status.get("name", ""),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            status.setdefault("nvidia_smi_error", f"{type(exc).__name__}: {exc}")
    return status


def emit_gpu_status(stage: str = "") -> None:
    payload = {"type": "gpu_status", "ts": time.time(), **collect_gpu_status()}
    if stage:
        payload["stage"] = stage
    emit(payload)


def normalize_model(value: str) -> str:
    key = value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key == "efficientad":
        return "efficientad"
    if key in {"patchcore", "padim"}:
        return key
    raise argparse.ArgumentTypeError(f"不支持的 Anomalib 模型：{value}，可选 patchcore/padim/efficientad")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(root: Path, value: str, required: bool = True) -> Path | None:
    text = value.strip()
    if not text:
        if required:
            raise FileNotFoundError("路径参数为空")
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def validate_image_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} 不存在：{path}")
    if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.rglob("*")):
        raise FileNotFoundError(f"{name} 中未找到图片：{path}")


def collect_images(roots: Iterable[Path], limit: int = 64) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root and root.exists():
            paths.extend(p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit]


def resolve_max_epochs(model_name: str, value: str) -> int:
    text = str(value).strip().lower()
    if text and text != "auto":
        return max(1, int(text))
    if model_name in {"patchcore", "padim"}:
        return 1
    return 20


def mode_check_env(_: argparse.Namespace) -> int:
    add_anomalib_source_path()
    deps = {
        "torch": import_status("torch"),
        "anomalib": import_status("anomalib"),
        "lightning": import_status("lightning"),
        "onnx": import_status("onnx"),
        "onnxruntime": import_status("onnxruntime"),
        "PIL": import_status("PIL"),
        "numpy": import_status("numpy"),
    }
    missing = [
        {"package": key, "install": f"python -m pip install {key}"}
        for key, status in deps.items()
        if not status.get("ok")
    ]
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # noqa: BLE001
        cuda["error"] = f"{type(exc).__name__}: {exc}"

    emit(
        {
            "type": "env",
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "env_name": current_env_name(),
            "sys_prefix": sys.prefix,
            **deps,
            "cuda": cuda,
            "missing": missing,
        }
    )
    if missing:
        print("环境检测：存在缺失依赖，请先安装后再训练。", flush=True)
    else:
        print("环境检测通过：Anomalib 训练导出依赖可用。", flush=True)
    return 0


def create_model(model_name: str, imagenet_dir: Path | None) -> Any:
    from anomalib.models import EfficientAd, Patchcore
    from anomalib.models.image.padim import Padim

    if model_name == "patchcore":
        return Patchcore()
    if model_name == "padim":
        return Padim()
    if imagenet_dir is None:
        raise FileNotFoundError("EfficientAd 需要 imagenet_dir。")
    return EfficientAd(imagenet_dir=imagenet_dir)


def find_checkpoint(engine: Any) -> Path:
    export_root = Path(engine.trainer.default_root_dir)
    ckpt_dir = export_root / "weights" / "lightning"
    ckpt_path = ckpt_dir / "model.ckpt"
    if ckpt_path.is_file():
        return ckpt_path
    callback = getattr(engine, "checkpoint_callback", None)
    if callback is not None and getattr(callback, "dirpath", None):
        alt = Path(callback.dirpath) / (getattr(callback, "filename", "model") + ".ckpt")
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"未找到 Lightning checkpoint：{ckpt_path}")


def patch_anomalib_version_dir_for_windows() -> None:
    """Avoid Windows symlink permission failures in Anomalib's latest run folder.

    Anomalib normally creates ``latest`` as a directory symlink. On Windows this
    requires elevated privileges or developer mode. The GUI training path should
    work for normal users, so we return the concrete ``vN`` directory instead.
    """

    try:
        import re
        import anomalib.engine.engine as engine_module

        def create_version_dir_without_symlink(root_dir: str | Path) -> Path:
            root = Path(root_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            version_pattern = re.compile(r"^v(\d+)$")
            highest = -1
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                match = version_pattern.match(child.name)
                if match:
                    highest = max(highest, int(match.group(1)))
            version_dir = root / f"v{highest + 1}"
            version_dir.mkdir(exist_ok=False)
            return version_dir

        engine_module.create_versioned_dir = create_version_dir_without_symlink
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: patch Anomalib version dir failed: {type(exc).__name__}: {exc}", flush=True)


def percentile(values: list[float] | Any, q: float) -> float:
    import numpy as np

    if len(values) == 0:
        raise ValueError("percentile input is empty")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def threshold_from_distributions(normal_values: list[float], abnormal_values: list[float]) -> float:
    normal_high = percentile(normal_values, 99.0)
    if not abnormal_values:
        return normal_high
    abnormal_low = percentile(abnormal_values, 5.0)
    if abnormal_low > normal_high:
        return (normal_high + abnormal_low) / 2.0
    span = max(max(normal_values + abnormal_values) - min(normal_values + abnormal_values), 1.0e-6)
    return normal_high + span * 0.02


def calibrate_efficientad_metadata(
    onnx_path: Path,
    normal_roots: list[Path],
    abnormal_roots: list[Path],
    metadata: dict[str, Any],
    model_type: str,
    limit: int,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from trtforge_metadata import parse_anomaly_outputs, preprocess_image_for_onnx

    normal_images = collect_images(normal_roots, limit)
    abnormal_images = collect_images(abnormal_roots, limit)
    if not normal_images:
        print("EfficientAd metadata 校准跳过：未找到 normal 校准图片。", flush=True)
        return metadata

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    normal_scores: list[float] = []
    abnormal_scores: list[float] = []
    normal_pixels: list[float] = []
    abnormal_pixels: list[float] = []
    map_mins: list[float] = []
    map_maxs: list[float] = []

    def infer_group(paths: list[Path], scores: list[float], pixels: list[float]) -> None:
        for image_path in paths:
            tensor = preprocess_image_for_onnx(image_path, metadata, model_type)
            outputs = session.run(None, {input_name: tensor})
            score, anomaly_map = parse_anomaly_outputs(outputs, output_names)
            scores.append(float(score))
            if anomaly_map is not None:
                arr = np.asarray(anomaly_map, dtype=np.float32)
                map_mins.append(float(np.min(arr)))
                map_maxs.append(float(np.max(arr)))
                pixels.append(percentile(arr.reshape(-1), 99.5))

    infer_group(normal_images, normal_scores, normal_pixels)
    infer_group(abnormal_images, abnormal_scores, abnormal_pixels)
    if not normal_scores:
        return metadata

    all_scores = normal_scores + abnormal_scores
    image_threshold = threshold_from_distributions(normal_scores, abnormal_scores)
    pixel_threshold = threshold_from_distributions(normal_pixels, abnormal_pixels) if normal_pixels else image_threshold
    metadata["output_contract"] = "efficientad_distribution_calibrated"
    metadata["normalization"] = "none"
    metadata["image_threshold"] = image_threshold
    metadata["pixel_threshold"] = pixel_threshold
    metadata["pred_scores"]["min"] = min(all_scores)
    metadata["pred_scores"]["max"] = max(max(all_scores), image_threshold)
    metadata["pred_scores.min"] = metadata["pred_scores"]["min"]
    metadata["pred_scores.max"] = metadata["pred_scores"]["max"]
    if map_mins and map_maxs:
        metadata["anomaly_maps"]["min"] = min(map_mins)
        metadata["anomaly_maps"]["max"] = max(max(map_maxs), pixel_threshold)
        metadata["anomaly_maps.min"] = metadata["anomaly_maps"]["min"]
        metadata["anomaly_maps.max"] = metadata["anomaly_maps"]["max"]
    metadata["calibration"] = {
        "source": "trtforge_gui_efficientad_distribution",
        "normal_image_count": len(normal_images),
        "abnormal_image_count": len(abnormal_images),
    }
    return metadata


def mode_train_export(args: argparse.Namespace) -> int:
    anomalib_root = add_anomalib_source_path()

    import torch

    # Local Lightning checkpoints contain Python objects; export must load trusted local ckpt fully.
    original_torch_load = torch.load

    def torch_load_trusted(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_trusted

    from anomalib.data import Folder
    from anomalib.deploy.export import ExportType
    from anomalib.engine import Engine
    from trtforge_metadata import (
        apply_postprocessed_output_contract,
        calibrate_metadata_from_onnx,
        extract_model_metadata,
        onnx_has_postprocessed_outputs,
        write_metadata_json,
    )
    patch_anomalib_version_dir_for_windows()

    model_name = normalize_model(args.model)
    root = Path(args.root).expanduser()
    normal_dir = resolve_path(root, args.normal_dir)
    abnormal_dir = resolve_path(root, args.abnormal_dir)
    normal_test_dir = resolve_path(root, args.normal_test_dir)
    assert normal_dir is not None and abnormal_dir is not None and normal_test_dir is not None
    validate_image_dir(normal_dir, "正常训练目录 normal_dir")
    validate_image_dir(abnormal_dir, "异常测试目录 abnormal_dir")
    validate_image_dir(normal_test_dir, "正常测试目录 normal_test_dir")

    input_size = int(args.input_size)
    max_epochs = resolve_max_epochs(model_name, args.max_epochs)
    train_batch_size = 1 if model_name == "efficientad" else max(1, int(args.train_batch_size))
    eval_batch_size = max(1, int(args.eval_batch_size))
    imagenet_dir = Path(args.imagenet_dir) if args.imagenet_dir else anomalib_root / "datasets" / "imagenette"
    if model_name == "efficientad" and not imagenet_dir.exists():
        raise FileNotFoundError(f"EfficientAd imagenet_dir 不存在：{imagenet_dir}")

    print(f"Anomalib train-export start: model={model_name}, root={root}", flush=True)
    print(f"normal_dir={normal_dir}", flush=True)
    print(f"abnormal_dir={abnormal_dir}", flush=True)
    print(f"normal_test_dir={normal_test_dir}", flush=True)
    print(f"max_epochs={max_epochs}, train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}", flush=True)
    emit({"type": "event", "name": "train_start", "model": model_name, "max_epochs": max_epochs})
    emit({"type": "progress", "stage": "setup", "epoch": 0, "total": max_epochs, "pct": 0.0})
    emit_gpu_status("setup")

    datamodule = Folder(
        name=args.name or root.name or "anomalib_dataset",
        root=root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        normal_test_dir=normal_test_dir,
        num_workers=max(0, int(args.num_workers)),
        val_split_ratio=float(args.val_split_ratio),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    datamodule.setup()
    model = create_model(model_name, imagenet_dir)
    emit(
        {
            "type": "anomaly_model",
            "model": model.__class__.__name__,
            "input_size": [input_size, input_size],
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
    )

    use_logger = max_epochs <= 1
    if not use_logger:
        print("logger=False：多 epoch 时避免 CSVLogger 表头冲突。", flush=True)
    engine = Engine(
        max_epochs=max_epochs,
        logger=use_logger,
        enable_progress_bar=False,
        default_root_dir=Path(args.project),
    )

    emit({"type": "progress", "stage": "train", "epoch": 0, "total": max_epochs, "pct": 5.0})
    engine.train(datamodule=datamodule, model=model)
    emit({"type": "progress", "stage": "train", "epoch": max_epochs, "total": max_epochs, "pct": 70.0})
    emit_gpu_status("train_done")

    ckpt_path = find_checkpoint(engine)
    export_root = Path(engine.trainer.default_root_dir)
    ckpt_dir = export_root / "weights" / "lightning"
    print(f"checkpoint: {ckpt_path}", flush=True)
    emit({"type": "artifact", "kind": "ckpt", "path": str(ckpt_path)})
    emit({"type": "artifact", "kind": "run_dir", "path": str(export_root)})

    print("Export ONNX...", flush=True)
    emit({"type": "progress", "stage": "export", "epoch": max_epochs, "total": max_epochs, "pct": 80.0})
    onnx_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=export_root,
        ckpt_path=ckpt_path,
        input_size=(input_size, input_size),
    )
    if onnx_path is None:
        emit_error("EXPORT_FAILED", "ONNX 导出失败，返回路径为空")
        return 1
    onnx_path = Path(onnx_path)
    target_onnx = ckpt_dir / onnx_path.name
    if onnx_path.resolve() != target_onnx.resolve():
        shutil.move(str(onnx_path), str(target_onnx))
        onnx_path = target_onnx
    print(f"onnx: {onnx_path}", flush=True)
    emit({"type": "artifact", "kind": "onnx", "path": str(onnx_path)})

    print("Generate metadata.json...", flush=True)
    emit({"type": "progress", "stage": "metadata", "epoch": max_epochs, "total": max_epochs, "pct": 90.0})
    metadata = extract_model_metadata(model=model, model_type=model.__class__.__name__, input_size=(input_size, input_size))
    if model_name == "efficientad" and parse_bool(args.calibrate_metadata):
        calib_normal = resolve_path(root, args.calib_normal_dir, required=False) or normal_dir
        calib_abnormal = resolve_path(root, args.calib_abnormal_dir, required=False) or abnormal_dir
        metadata = calibrate_efficientad_metadata(
            onnx_path=onnx_path,
            normal_roots=[normal_dir, normal_test_dir, calib_normal],
            abnormal_roots=[abnormal_dir, calib_abnormal],
            metadata=metadata,
            model_type=model.__class__.__name__,
            limit=max(1, int(args.calib_limit)),
        )
    elif onnx_has_postprocessed_outputs(onnx_path):
        metadata = apply_postprocessed_output_contract(metadata)
    elif parse_bool(args.calibrate_metadata):
        calibration_images = collect_images([normal_dir, normal_test_dir, abnormal_dir], max(1, int(args.calib_limit)))
        metadata = calibrate_metadata_from_onnx(
            onnx_path=onnx_path,
            image_paths=calibration_images,
            metadata=metadata,
            model_type=model.__class__.__name__,
        )

    metadata_path = write_metadata_json(
        output_path=ckpt_dir / "metadata.json",
        metadata=metadata,
        ckpt_path=ckpt_path,
        onnx_path=onnx_path,
    )
    print(f"metadata: {metadata_path}", flush=True)
    emit({"type": "artifact", "kind": "metadata", "path": str(metadata_path)})
    emit(
        {
            "type": "anomaly_export_summary",
            "model": model_name,
            "model_type": model.__class__.__name__,
            "output_contract": metadata.get("output_contract", ""),
            "normalization": metadata.get("normalization", ""),
            "image_threshold": metadata.get("image_threshold"),
            "pixel_threshold": metadata.get("pixel_threshold"),
            "ckpt": str(ckpt_path),
            "onnx": str(onnx_path),
            "metadata": str(metadata_path),
        }
    )
    emit({"type": "progress", "stage": "finished", "epoch": max_epochs, "total": max_epochs, "pct": 100.0})
    emit({"type": "event", "name": "finished", "exit": 0})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRTForge Anomalib training/export launcher")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--model", type=normalize_model, default="patchcore")
    parser.add_argument("--root", default="")
    parser.add_argument("--normal-dir", default="train/good")
    parser.add_argument("--abnormal-dir", default="test/abnormal")
    parser.add_argument("--normal-test-dir", default="test/good")
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="latest")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--max-epochs", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split-ratio", type=float, default=0.2)
    parser.add_argument("--imagenet-dir", default="")
    parser.add_argument("--calibrate-metadata", default="true")
    parser.add_argument("--calib-normal-dir", default="")
    parser.add_argument("--calib-abnormal-dir", default="")
    parser.add_argument("--calib-limit", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "check-env":
            return mode_check_env(args)
        if args.mode == "train-export":
            if not args.root.strip():
                emit_error("DATA_NOT_FOUND", "请先选择 Anomalib Folder 数据集 root。")
                return 2
            return mode_train_export(args)
        emit_error("UNSUPPORTED_MODE", f"未知 mode：{args.mode}")
        return 2
    except KeyboardInterrupt:
        emit_error("CANCELLED", "用户取消了任务")
        return 130
    except ModuleNotFoundError as exc:
        package = getattr(exc, "name", "") or "unknown"
        emit_error("MISSING_DEP", f"缺少 Python 依赖：{package}", package=package, install=f"python -m pip install {package}")
        traceback.print_exc()
        return 1
    except FileNotFoundError as exc:
        emit_error("DATA_NOT_FOUND", str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level launcher must report unknown failures
        emit_error(
            "UNHANDLED_EXCEPTION",
            f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
