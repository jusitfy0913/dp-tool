# -*- coding: utf-8 -*-
"""TRTForge YOLO training launcher.

This script is the stable Python boundary called by the Qt/C++ side through
QProcess. It keeps ordinary stdout/stderr visible while emitting structured
TRTFORGE_JSON lines for progress, metrics, artifacts and recoverable errors.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


JSON_PREFIX = "TRTFORGE_JSON "
SUPPORTED_TASKS = {"detect", "segment", "pose", "obb"}
SUPPORTED_MODES = {"check-env", "probe-model", "train", "val", "export"}
PLOT_FILES = {
    "BoxF1_curve": ["BoxF1_curve.png", "F1_curve.png"],
    "BoxPR_curve": ["BoxPR_curve.png", "PR_curve.png"],
    "BoxP_curve": ["BoxP_curve.png", "P_curve.png"],
    "BoxR_curve": ["BoxR_curve.png", "R_curve.png"],
    "confusion_matrix": ["confusion_matrix.png"],
    "confusion_matrix_normalized": ["confusion_matrix_normalized.png"],
    "labels": ["labels.jpg"],
    "results": ["results.png"],
}


def emit(event: Dict[str, Any]) -> None:
    """Print one structured event for Qt to parse."""
    print(JSON_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_error(code: str, message: str, **extra: Any) -> None:
    payload = {"type": "error", "code": code, "message": message}
    payload.update(extra)
    emit(payload)


def bool_or_path(value: Optional[str]) -> Any:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return value


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_batch(value: Any) -> Any:
    text = str(value).strip()
    if not text:
        return 16
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_cache(value: Any) -> Any:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return value


def optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def import_status(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        return {"ok": True, "version": str(version)}
    except Exception as exc:  # noqa: BLE001 - must report dependency problems to UI
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def ensure_import(module_name: str, pip_name: Optional[str] = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        package = pip_name or module_name
        emit_error(
            "MISSING_DEP",
            f"缺少 Python 库：{package}",
            package=package,
            install=f"python -m pip install {package}",
        )
        raise exc


def current_env_name() -> str:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if conda_env:
        return conda_env
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        return Path(virtual_env).name
    return Path(sys.prefix).name


def add_ultralytics_source_path() -> None:
    """Prefer the workspace 1-python-yolo/ultralytics-main source tree when available."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidates = [
            parent / "1-python-yolo" / "ultralytics-main",
            parent / "ultralytics-main",
        ]
        for ultra_root in candidates:
            if ultra_root.exists():
                ultra_path = str(ultra_root)
                if ultra_path not in sys.path:
                    sys.path.insert(0, ultra_path)
                return


def flatten_metrics(prefix: str, obj: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if obj is None:
        return result
    if isinstance(obj, dict):
        iterator = obj.items()
    else:
        iterator = ((name, getattr(obj, name)) for name in dir(obj) if not name.startswith("_"))
    for key, value in iterator:
        if isinstance(value, (int, float)):
            result[f"{prefix}{key}"] = float(value)
    return result


def emit_metrics(metrics_obj: Any, step: int = 0) -> None:
    values: Dict[str, float] = {}
    if hasattr(metrics_obj, "results_dict"):
        values.update(flatten_metrics("", getattr(metrics_obj, "results_dict")))
    values.update(flatten_metrics("", metrics_obj if isinstance(metrics_obj, dict) else None))
    speed = getattr(metrics_obj, "speed", None)
    if isinstance(speed, dict):
        values.update({f"speed/{k}": float(v) for k, v in speed.items() if isinstance(v, (int, float))})
    for name, value in sorted(values.items()):
        emit({"type": "metric", "name": name, "value": value, "step": step})


def to_float(value: Any) -> Optional[float]:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def emit_plot_artifacts(save_dir: Any) -> None:
    if not save_dir:
        return
    run_dir = Path(save_dir)
    for plot_name, candidates in PLOT_FILES.items():
        for filename in candidates:
            path = run_dir / filename
            if path.exists():
                emit(
                    {
                        "type": "plot",
                        "name": plot_name,
                        "path": str(path),
                        "mtime": path.stat().st_mtime,
                    }
                )
                emit({"type": "artifact", "kind": "plot", "name": plot_name, "path": str(path)})
                break


def collect_gpu_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {"available": False}
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
                first = output.splitlines()[0]
                parts = [part.strip() for part in first.split(",")]
                if len(parts) >= 4:
                    status.update(
                        {
                            "available": True,
                            "utilization_gpu_pct": to_float(parts[0]),
                            "memory_used_mb": to_float(parts[1]) or status.get("memory_used_mb"),
                            "memory_total_mb": to_float(parts[2]) or status.get("memory_total_mb"),
                            "name": parts[3] or status.get("name", ""),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            status.setdefault("nvidia_smi_error", f"{type(exc).__name__}: {exc}")
    return status


def emit_gpu_status(step: Optional[int] = None) -> None:
    payload = {"type": "gpu_status", "ts": time.time(), **collect_gpu_status()}
    if step is not None:
        payload["step"] = step
    emit(payload)


def extract_loss_values(trainer: Any) -> Dict[str, float]:
    values: Dict[str, float] = {}
    loss_names = getattr(trainer, "loss_names", []) or []
    tloss = getattr(trainer, "tloss", None)
    if tloss is None:
        return values
    try:
        loss_values: Iterable[Any] = list(tloss)
    except TypeError:
        loss_values = [tloss]
    for idx, value in enumerate(loss_values):
        numeric = to_float(value)
        if numeric is None:
            continue
        name = str(loss_names[idx]) if idx < len(loss_names) else f"loss/{idx}"
        values[name] = numeric
    return values


def build_epoch_summary(trainer: Any, epoch: int, total: int) -> Dict[str, Any]:
    metrics = {
        name: value
        for name, value in (getattr(trainer, "metrics", {}) or {}).items()
        if isinstance(value, (int, float))
    }
    losses = extract_loss_values(trainer)
    lr = getattr(trainer, "lr", {}) or {}
    lrs = {name: value for name, value in lr.items() if isinstance(value, (int, float))}
    return {
        "type": "epoch_summary",
        "epoch": epoch,
        "total": total,
        "pct": (epoch * 100.0 / total) if total > 0 else 0.0,
        "losses": losses,
        "metrics": metrics,
        "lr": lrs,
        "gpu": collect_gpu_status(),
    }


def emit_training_artifacts(save_dir: Any) -> None:
    if not save_dir:
        return
    run_dir = Path(save_dir)
    emit({"type": "artifact", "kind": "run_dir", "path": str(run_dir)})
    emit_plot_artifacts(run_dir)

    weights_dir = run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        path = weights_dir / name
        if path.exists():
            emit({"type": "artifact", "kind": "pt", "name": name, "path": str(path)})

    known_files = [
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]
    for name in known_files:
        path = run_dir / name
        if path.exists():
            emit({"type": "artifact", "kind": "file", "name": name, "path": str(path)})


def normalize_names(names: Any) -> List[str]:
    if isinstance(names, dict):
        result = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
            result.append(str(value))
        return result
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return []


def read_dataset_names(data_path: str) -> List[str]:
    if not data_path:
        return []
    path = Path(data_path)
    if not path.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return normalize_names(data.get("names"))
    except Exception:
        text = path.read_text(encoding="utf-8", errors="ignore")
        inline = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, re.M)
        if inline:
            return [item.strip().strip("'\"") for item in inline.group(1).split(",") if item.strip()]
        pairs = re.findall(r"^\s*(\d+)\s*:\s*['\"]?([^'\"\n]+)['\"]?\s*$", text, re.M)
        return [name.strip() for _, name in sorted(pairs, key=lambda pair: int(pair[0]))]


def safe_filename_part(value: str) -> str:
    value = value.strip() or "exp"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        numbered = path.with_name(f"{path.stem}_{stamp}_{idx}{path.suffix}")
        if not numbered.exists():
            return numbered
        idx += 1


def gui_configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "bin" / "configs"


def generate_labels_json(save_dir: Any, experiment_name: Optional[str], names: Any, data_path: str = "") -> List[str]:
    if not save_dir:
        return []
    classes = normalize_names(names) or read_dataset_names(data_path)
    if not classes:
        emit(
            {
                "type": "artifact",
                "kind": "labels_json",
                "name": "not_generated",
                "message": "未读取到类别 names，请检查 dataset.yaml 的 names 字段或 Ultralytics trainer.names。",
            }
        )
        return []
    run_dir = Path(save_dir)
    label_name = safe_filename_part(experiment_name or run_dir.name)
    payload = {"name": label_name, "classes": classes}
    filename = f"labels_{label_name}.json"

    written: List[str] = []
    targets = [run_dir / filename, gui_configs_dir() / filename]
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = unique_path(target)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(str(target))
            emit({"type": "artifact", "kind": "labels_json", "name": target.name, "path": str(target)})
        except Exception as exc:  # noqa: BLE001
            emit_error("LABEL_JSON_WRITE_FAILED", f"标签 JSON 写入失败：{target}", error=f"{type(exc).__name__}: {exc}")
    return written


def apply_onnx_last_output_transpose(onnx_path: str) -> str:
    onnx = ensure_import("onnx")
    helper = ensure_import("onnx.helper")
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    dst = src.with_name(src.stem + ".transd" + src.suffix)
    model = onnx.load(str(src))
    if not model.graph.node:
        raise RuntimeError("ONNX graph has no node")

    node = model.graph.node[-1]
    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            dims = specout.type.tensor_type.shape.dim
            if len(dims) >= 3:
                shape0 = dims[0]
                shape1 = dims[1]
                shape2 = dims[2]
                new_out = helper.make_tensor_value_info(
                    specout.name,
                    specout.type.tensor_type.elem_type,
                    [0, 0, 0],
                )
                new_dims = new_out.type.tensor_type.shape.dim
                new_dims[0].CopyFrom(shape0)
                new_dims[1].CopyFrom(shape2)
                new_dims[2].CopyFrom(shape1)
                specout.CopyFrom(new_out)
            break

    model.graph.node.append(helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1]))
    onnx.save(model, str(dst))
    return str(dst)


def default_model_for_task(task: str) -> str:
    return {
        "detect": "yolov8n.pt",
        "segment": "yolov8n-seg.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt",
    }.get(task, "yolov8n.pt")


def validate_task(task: str) -> None:
    if task not in SUPPORTED_TASKS:
        emit_error("UNSUPPORTED_TASK", f"不支持的 YOLO 任务头：{task}", supported=sorted(SUPPORTED_TASKS))
        raise ValueError(f"unsupported task: {task}")


def build_train_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.data or not Path(args.data).exists():
        emit_error("DATA_NOT_FOUND", "dataset.yaml 不存在或未选择", path=args.data or "")
        raise FileNotFoundError(args.data or "dataset.yaml")

    # Ultralytics 8.4 的 get_save_dir() 对「相对路径的 project」会再拼一层
    # <runs_dir>/<task>/，于是 --project runs/pcb 实际落到
    # runs/detect/runs/pcb（路径重复，用户按文档找不到产物）。
    # 这里提前解析成绝对路径，让它走 absolute 分支，行为回到直觉预期。
    project_dir = args.project
    if project_dir and not Path(project_dir).is_absolute():
        project_dir = str(Path(project_dir).resolve())

    kwargs: Dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "device": args.device,
        "project": project_dir,
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "cache": parse_cache(args.cache),
        "amp": args.amp,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "freeze": args.freeze,
        "resume": args.resume,
        "plots": args.plots,
        "verbose": True,
        # 增强参数：默认值与 Ultralytics 默认一致，只有用户显式指定才会改变行为
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "bgr": args.bgr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
    }

    pretrained = bool_or_path(args.pretrained)
    if pretrained is not None:
        kwargs["pretrained"] = pretrained

    if args.task == "segment":
        kwargs["overlap_mask"] = args.overlap_mask
        kwargs["mask_ratio"] = args.mask_ratio
    elif args.task == "pose":
        kwargs["pose"] = args.pose_loss
        kwargs["kobj"] = args.kobj
        kwargs["rle"] = args.rle
    elif args.task == "obb":
        kwargs["angle"] = args.angle

    return {k: v for k, v in kwargs.items() if v is not None}


def add_train_callbacks(model: Any, args: argparse.Namespace) -> None:
    last_gpu_emit = [0.0]

    def on_train_start(trainer: Any) -> None:
        total = int(getattr(getattr(trainer, "args", object()), "epochs", 0) or 0)
        emit({"type": "event", "name": "train_start", "ts": time.time(), "total": total})
        emit_gpu_status()

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        total = int(getattr(getattr(trainer, "args", object()), "epochs", 0) or 0)
        pct = (epoch * 100.0 / total) if total > 0 else 0.0
        emit({"type": "progress", "epoch": epoch, "total": total, "pct": pct})
        emit(build_epoch_summary(trainer, epoch, total))
        metrics = getattr(trainer, "metrics", {}) or {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                emit({"type": "metric", "name": str(name), "value": float(value), "step": epoch})
        for name, numeric in extract_loss_values(trainer).items():
            emit({"type": "metric", "name": name, "value": numeric, "step": epoch})
        emit_plot_artifacts(getattr(trainer, "save_dir", None))
        emit_gpu_status(epoch)

    def on_train_batch_end(trainer: Any) -> None:
        now = time.time()
        if now - last_gpu_emit[0] < 2.0:
            return
        last_gpu_emit[0] = now
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        emit_gpu_status(epoch if epoch > 0 else None)

    def on_model_save(trainer: Any) -> None:
        for kind, attr in (("last", "last"), ("best", "best")):
            path = getattr(trainer, attr, None)
            if path:
                emit({"type": "artifact", "kind": "pt", "name": kind, "path": str(path)})

    def on_train_end(trainer: Any) -> None:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            emit({"type": "artifact", "kind": "run_dir", "path": str(save_dir)})
            names = getattr(trainer, "names", None) or getattr(getattr(trainer, "model", None), "names", None)
            data = getattr(trainer, "data", {}) or {}
            if not names and isinstance(data, dict):
                names = data.get("names")
            generate_labels_json(save_dir, args.name, names, args.data)
            emit_plot_artifacts(save_dir)
        emit({"type": "event", "name": "train_finished", "ts": time.time()})

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_train_end", on_train_end)


def mode_check_env(_: argparse.Namespace) -> int:
    add_ultralytics_source_path()
    packages = {
        "torch": import_status("torch"),
        "ultralytics": import_status("ultralytics"),
        "onnx": import_status("onnx"),
        "onnxruntime": import_status("onnxruntime"),
        "polars": import_status("polars"),
    }
    torch_status = packages["torch"]
    cuda_status: Dict[str, Any] = {"available": False, "devices": []}
    if torch_status["ok"]:
        try:
            import torch

            cuda_status["available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                cuda_status["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:  # noqa: BLE001
            cuda_status["error"] = f"{type(exc).__name__}: {exc}"

    missing = []
    for package, status in packages.items():
        if not status["ok"]:
            missing.append(
                {
                    "package": package,
                    "error": status.get("error", ""),
                    "install": f'"{sys.executable}" -m pip install {package}',
                }
            )

    payload = {
        "type": "env",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "env_name": current_env_name(),
        "sys_prefix": sys.prefix,
        "sys_path0": sys.path[0] if sys.path else "",
        "torch": torch_status,
        "cuda": cuda_status,
        "ultralytics": packages["ultralytics"],
        "onnx": packages["onnx"],
        "onnxruntime": packages["onnxruntime"],
        "polars": packages["polars"],
        "missing": missing,
    }
    emit(payload)

    required = [item for item in missing if item["package"] in {"torch", "ultralytics", "polars"}]
    if required:
        emit_error(
            "MISSING_DEP",
            "当前 Python 虚拟环境缺少训练必需库：" + ", ".join(item["package"] for item in required),
            python=sys.executable,
            env_name=current_env_name(),
            missing=required,
        )
    return 0


def load_yolo(model_path: str) -> Any:
    add_ultralytics_source_path()
    YOLO = ensure_import("ultralytics").YOLO
    return YOLO(model_path)


def mode_probe_model(args: argparse.Namespace) -> int:
    weight = args.weights or args.model
    if not weight:
        emit_error("ARGUMENT_ERROR", "probe-model 需要 --weights 或 --model")
        return 2
    model = load_yolo(weight)
    names = getattr(model, "names", None)
    emit(
        {
            "type": "model",
            "path": weight,
            "task": getattr(model, "task", ""),
            "names": names if isinstance(names, dict) else {},
            "classes": len(names) if isinstance(names, dict) else None,
        }
    )
    return 0


def mode_train(args: argparse.Namespace) -> int:
    validate_task(args.task)
    model_path = args.model or default_model_for_task(args.task)
    model = load_yolo(model_path)
    add_train_callbacks(model, args)
    emit({"type": "event", "name": "launcher_train", "task": args.task, "model": model_path})
    results = model.train(**build_train_kwargs(args))
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    emit_training_artifacts(save_dir)
    emit_metrics(results, step=args.epochs)
    emit({"type": "event", "name": "finished", "exit": 0})
    return 0


def mode_val(args: argparse.Namespace) -> int:
    validate_task(args.task)
    weight = args.weights or args.model
    if not weight:
        emit_error("ARGUMENT_ERROR", "val 需要 --weights 或 --model")
        return 2
    if args.data and not Path(args.data).exists():
        emit_error("DATA_NOT_FOUND", "dataset.yaml 不存在或未选择", path=args.data)
        return 2
    model = load_yolo(weight)
    kwargs = {
        "data": optional_path(args.data),
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": args.half,
        "device": args.device,
        "plots": args.plots,
        "save_json": args.save_json,
        "save_txt": args.save_txt,
        "split": args.split,
        "workers": args.workers,
        "verbose": True,
    }
    metrics = model.val(**{k: v for k, v in kwargs.items() if v is not None})
    emit_metrics(metrics)
    emit({"type": "event", "name": "finished", "exit": 0})
    return 0


def mode_export(args: argparse.Namespace) -> int:
    validate_task(args.task)
    weight = args.weights or args.model
    if not weight:
        emit_error("ARGUMENT_ERROR", "export 需要 --weights 或 --model")
        return 2
    model = load_yolo(weight)
    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.export_batch,
        "device": args.device,
        "dynamic": args.dynamic,
        "half": args.half,
        "opset": args.opset,
        "simplify": args.simplify,
    }
    emit({"type": "event", "name": "export_start", "weights": weight, "format": args.format})
    exported = model.export(**{k: v for k, v in kwargs.items() if v is not None})
    onnx_path = str(exported) if exported else ""
    if not onnx_path or not Path(onnx_path).exists():
        emit_error("EXPORT_FAILED", "ONNX 导出失败，未找到导出文件")
        return 1
    emit({"type": "artifact", "kind": "onnx", "path": onnx_path})
    if str_to_bool(args.make_transd):
        transd = apply_onnx_last_output_transpose(onnx_path)
        emit({"type": "artifact", "kind": "transd_onnx", "path": transd})
    emit({"type": "event", "name": "finished", "exit": 0})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRTForge YOLO training launcher")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS), default="detect")
    parser.add_argument("--model", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/trtforge")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--freeze", type=int, default=None)
    # ------------------------------------------------------------------
    # 数据增强（默认值逐项对齐 Ultralytics 8.4 的 DEFAULT_CFG）：
    # 不填时与改动前行为完全一致，避免「升级后模型表现变了却不知道为什么」。
    # 小缺陷场景（如 PCB 70px 缺陷）最该调的是 mosaic 与 close_mosaic：
    # mosaic 会把 4 张图拼成 1 张，等于再缩小一半，缺陷可能小到看不见。
    # ------------------------------------------------------------------
    parser.add_argument("--hsv-h", type=float, default=0.015, help="色调抖动幅度（0~1）")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="饱和度抖动幅度（0~1）")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="亮度抖动幅度（0~1）")
    parser.add_argument("--degrees", type=float, default=0.0, help="随机旋转角度范围（±度）")
    parser.add_argument("--translate", type=float, default=0.1, help="随机平移比例（0~1）")
    parser.add_argument("--scale", type=float, default=0.5, help="随机缩放增益（0~1）")
    parser.add_argument("--shear", type=float, default=0.0, help="随机错切角度（±度）")
    parser.add_argument("--perspective", type=float, default=0.0, help="随机透视变换（0~0.001）")
    parser.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率（0~1）")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率（0~1）")
    parser.add_argument("--bgr", type=float, default=0.0, help="RGB/BGR 通道互换概率（0~1）")
    parser.add_argument("--mosaic", type=float, default=1.0, help="马赛克拼接概率（0~1），小缺陷建议降到 0~0.3")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp 概率（0~1）")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-Paste 概率（0~1，仅分割任务生效）")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="最后 N 个 epoch 关闭 mosaic，默认 10；小缺陷场景建议加大",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--pose-loss", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)
    parser.add_argument("--rle", type=float, default=1.0)
    parser.add_argument("--angle", type=float, default=1.0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--export-batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-transd", default="true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "check-env":
            return mode_check_env(args)
        if args.mode == "probe-model":
            return mode_probe_model(args)
        if args.mode == "train":
            return mode_train(args)
        if args.mode == "val":
            return mode_val(args)
        if args.mode == "export":
            return mode_export(args)
        emit_error("ARGUMENT_ERROR", f"未知 mode：{args.mode}")
        return 2
    except KeyboardInterrupt:
        emit_error("CANCELLED", "用户取消了任务")
        return 130
    except ModuleNotFoundError as exc:
        package = getattr(exc, "name", "") or str(exc).replace("No module named", "").strip(" '\"") or "unknown"
        emit_error(
            "MISSING_DEP",
            f"缺少 Python 依赖：{package}",
            package=package,
            install=f"python -m pip install {package}" if package != "unknown" else "python -m pip install <package>",
        )
        traceback.print_exc()
        return 1
    except FileNotFoundError as exc:
        emit_error("FILE_NOT_FOUND", str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level launcher must never hide errors
        emit_error(
            "UNHANDLED_EXCEPTION",
            f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
