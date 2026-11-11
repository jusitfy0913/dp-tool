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
SUPPORTED_TASKS = {"detect", "segment", "pose", "obb", "classify"}
SUPPORTED_MODES = {"check-env", "probe-model", "train", "val", "export", "warm"}
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


# ---------------------------------------------------------------------------
# 启动阶段计时
# ---------------------------------------------------------------------------
# 训练启动慢时，日志里只有 Ultralytics 自己的输出，用户看不出「在等解释器、
# 等导入、等权重，还是在扫数据集」。这里在关键节点各发一条 phase 事件：
#   elapsed_ms 是从本脚本加载起算的总耗时，step_ms 是距上一阶段的耗时。
# 两个都给：既回答「一共等多久」，也回答「慢在哪一段」。
# 用 perf_counter 而不是 time.time，避免系统时钟调整导致出现负耗时。
_T0 = time.perf_counter()
_last_phase_at = _T0
#: warm 进程收到真实任务的时间点。非 None 表示「本次 main 是复用预热进程在跑」，
#: 阶段时钟要从这一刻重新开始，否则会把自己预热等待的时间算进训练启动耗时里。
_TASK_T0: Optional[float] = None


def reset_phase_clock() -> bool:
    """把阶段时钟归零，返回本次是否为预热进程复用。"""
    global _T0, _last_phase_at, _TASK_T0
    warm_hit = _TASK_T0 is not None
    _T0 = time.perf_counter()
    _last_phase_at = _T0
    _TASK_T0 = None
    return warm_hit


def emit_phase(phase: str, label: str, **extra: Any) -> None:
    """报告一个启动阶段的耗时（TRTFORGE_JSON phase 事件）。"""
    global _last_phase_at
    now = time.perf_counter()
    payload: Dict[str, Any] = {
        "type": "phase",
        "phase": phase,
        "label": label,
        "pid": os.getpid(),
        "python": sys.executable,
        "elapsed_ms": int((now - _T0) * 1000),
        "step_ms": int((now - _last_phase_at) * 1000),
    }
    payload.update(extra)
    _last_phase_at = now
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


def _emit_classify_metrics(metrics_obj: Any) -> None:
    """分类只报准确率和混淆矩阵，不把检测 mAP 当成结果。

    训练结束（``model.train()`` 的返回值）和单独验证（``model.val()``）都走这里，
    两者返回的对象不完全一样：验证返回带 ``top1`` 属性的指标对象，
    训练有时返回 ``results_dict`` 字典。两种都要能取到值。
    """
    results = metrics_obj if isinstance(metrics_obj, dict) else getattr(metrics_obj, "results_dict", None)
    top1 = to_float(getattr(metrics_obj, "top1", None))
    if top1 is None and isinstance(results, dict):
        for key in ("metrics/accuracy_top1", "accuracy_top1", "top1"):
            if key in results:
                top1 = to_float(results[key])
                break
    matrix = None
    confusion = getattr(metrics_obj, "confusion_matrix", None)
    raw = getattr(confusion, "matrix", None) if confusion is not None else None
    if raw is not None and hasattr(raw, "tolist"):
        matrix = raw.tolist()
    emit({"type": "metric", "name": "accuracy_top1", "value": top1 if top1 is not None else 0.0, "step": 0})
    emit({"type": "event", "name": "classify_val", "top1": top1, "confusion_matrix": matrix})


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


def resolve_classify_data(data_path: str) -> str:
    """把分类任务的 ``dataset.yaml`` 还原成它描述的目录。

    Ultralytics 的分类路径（``BaseTrainer.get_dataset`` 与 ``BaseValidator.__call__``）
    都直接调用 ``check_cls_dataset(data)``，而它只接受**目录**：
    传 ``.yaml`` 会被当成 ``<datasets>/xxx.yaml``，后缀非空即抛
    ``Classification datasets must be a directory (...) not a file``。
    我们的 ``dataset.yaml`` 已经用 ``path/train/val/test`` 写了同样的目录结构，
    所以这里把 yaml 解析成根目录再交给 Ultralytics；非 yaml 原样返回。

    出错时抛 ``ValueError``，文案直接说明该改哪里，避免只看到
    ``RuntimeError: Dataset '...' error`` 这种看不出原因的外层错误。
    """
    raw = str(data_path or "").strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return raw
    try:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"分类数据集 yaml 读不了：{path}（{type(exc).__name__}: {exc}）") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"分类数据集 yaml 格式不对，应是「键: 值」：{path}")

    root = str(doc.get("path") or "").strip()
    if root and root != ".":
        candidate = Path(root).expanduser()
        base = candidate if candidate.is_absolute() else (path.parent / candidate)
    else:
        base = path.parent
    base = base.resolve()

    train_value = doc.get("train") or "train"
    if isinstance(train_value, (list, tuple)):
        train_value = train_value[0] if train_value else "train"
    train_name = str(train_value).strip().rstrip("/\\")
    if train_name not in {"train", "."}:
        raise ValueError(
            f"分类数据集 yaml 里 train 写的是「{train_name}」，但分类要求 {base} 下有 train 目录"
            "（每类一个子文件夹）。请把 train 改成 train，或改用 train/类别名/图片 的结构。"
        )
    if not (base / "train").is_dir():
        raise ValueError(
            f"分类数据集目录不完整：{base} 下没有 train 目录。"
            "分类要求 train/类别名/图片.jpg，val 和 test 可选。"
        )
    return str(base)


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


def _output_dims(output: Any) -> List[Any]:
    """把一个 graph output 的维度取成 list；动态维用 None 表示。"""
    dims: List[Any] = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            # dim_param / 未标注都当作动态维。动态批次不影响「通道在前」的判断，
            # 但通道与 anchors 两个位置都动态时就无法认定谁是预测头。
            dims.append(None)
    return dims


def _pick_prediction_head(model: Any, task: str) -> str:
    """按形状找出「检测头」输出名，而不是靠「图里最后一个节点」。

    为什么要改：旧实现取 `graph.node[-1]`，那是「导出顺序」的巧合，不是契约。
    Ultralytics 一改图结构（比如末尾多一个 Cast/Concat）就会转错张量，而错误是
    静默的——生成的 ONNX 依然能编译，只是推理结果全错。

    真实导出实测（ultralytics 8.4.192，imgsz=320）：
        detect  output0 [1, 84, 2100]          4 + 80 类
        segment output0 [1, 116, 2100]         4 + 80 类 + 32 mask 系数
                output1 [1, 32, 80, 80]        proto（4 维）
        pose    output0 [1, 56, 2100]          4 + 1 类 + 17*3 关键点
        obb     output0 [1, 20, 2100]          4 + 15 类 + 1 角度

    判据分两层：
    1. **预测头必然是 3 维输出**（proto 是 4 维），四种任务都只有它一个 3 维输出；
    2. 若出现多个 3 维输出（未来模型可能如此），再用「通道维 < anchors 维」裁决。
    不能只用第 2 条：imgsz 很小时 anchors 会小于通道数（如 64px 的检测头
    `[1, 84, 84]`），单看大小关系会判不出。
    """
    three_d: List[tuple[str, List[Any]]] = [
        (out.name, _output_dims(out))
        for out in model.graph.output
        if len(_output_dims(out)) == 3
    ]
    if not three_d:
        raise RuntimeError(
            "无法识别预测头输出：图里没有 3 维输出。"
            f"当前输出为 {[(o.name, _output_dims(o)) for o in model.graph.output]}"
        )

    if len(three_d) == 1:
        det_name, dims = three_d[0]
    else:
        candidates = [
            (name, dims)
            for name, dims in three_d
            if dims[1] is not None and dims[2] is not None and dims[1] < dims[2]
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"识别到多个 3 维输出 {[name for name, _ in three_d]}，无法确定该转置哪一个。"
                "请固定 export_batch=1、关闭 dynamic 后再试，或排查导出图结构。"
            )
        det_name, dims = candidates[0]

    _validate_head_channels(task, dims, model)
    return det_name


def _validate_head_channels(task: str, dims: List[Any], model: Any) -> None:
    """转置前校验通道数，把「任务选错」变成一句能看懂的话。

    不校验的代价：错误会一路带到 TensorRT，最后在推理时抛
    "V8Pose expects channels=56, got 116"，用户根本不知道是导出阶段就错了。
    """
    channels = dims[1]
    if task == "pose":
        # 4 框 + 1 类 + 17*3 关键点
        if channels != 56:
            raise RuntimeError(
                f"pose 模型通道数应为 56（4+1+17*3），实际 {channels}。"
                "请确认 --task pose 与权重匹配。"
            )
    elif task == "obb":
        # 4 框 + nc 类 + 1 角度；至少 1 个类别
        if channels < 6:
            raise RuntimeError(f"obb 模型通道数应 ≥ 6（4+nc+1），实际 {channels}")
    elif task == "segment":
        proto = [o for o in model.graph.output if len(_output_dims(o)) == 4]
        if not proto:
            raise RuntimeError(
                f"segment 模型应有 proto 输出（4 维），实际输出为 "
                f"{[(o.name, _output_dims(o)) for o in model.graph.output]}"
            )
        proto_channels = _output_dims(proto[0])[1]
        if proto_channels is None:
            raise RuntimeError("segment 的 proto 输出通道维是动态的，无法校验；请使用静态导出")
        # 4 框 + nc 类 + proto 通道（mask 系数），nc 至少 1
        if channels <= 4 + proto_channels:
            raise RuntimeError(
                f"segment 检测头通道数 {channels} 应大于 4 + proto 通道 {proto_channels}（= {4 + proto_channels}）"
            )


def _reorder_segment_outputs(model: Any, det_name: str) -> None:
    """把 proto 提到 graph.output 第 1 位（= 引擎的 binding 1）。

    这不是「风格问题」，是硬性契约：C++ 侧 `yolo.cu` 对 V8Seg 直接
        segment_head_dims_ = static_dims(1);   // 期望 proto
        bbox_head_dims_    = static_dims(2);   // 期望检测头
    而 Ultralytics 导出的顺序恰好相反（output0=检测头, output1=proto）。
    不重排的话：trtexec 能编译成功，推理时才报
    "V8Seg channel layout invalid (det=32, proto_dim=116)"。
    """
    proto = [o for o in model.graph.output if o.name != det_name and len(_output_dims(o)) == 4]
    if not proto:
        raise RuntimeError("segment 重排失败：找不到 proto 输出")
    others = [o for o in model.graph.output if o.name != proto[0].name and o.name != det_name]
    det = [o for o in model.graph.output if o.name == det_name]
    ordered = [proto[0], det[0], *others]
    del model.graph.output[:]
    model.graph.output.extend(ordered)


def make_transd_onnx(onnx_path: str, task: str = "detect") -> str:
    """生成「输出转置为 [batch, anchors, channels]」的 `.transd.onnx`。

    与旧实现的区别（旧函数名为 apply_onnx_last_output_transpose）：
    1. 不再假设「最后一个节点就是检测头」，改为按形状识别预测头；
    2. 转置前按任务校验通道数，任务与权重不匹配时直接报错；
    3. segment 额外把 proto 输出提到第一位，与 C++ 的 binding 约定对齐。

    原始 ONNX 保留不动：Ultralytics 自己的推理依赖原始布局，只有 TensorRT
    自定义引擎才需要转置版本。
    """
    onnx = ensure_import("onnx")
    helper = ensure_import("onnx.helper")
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    model = onnx.load(str(src))
    if not model.graph.node:
        raise RuntimeError("ONNX graph has no node")

    det_name = _pick_prediction_head(model, task)

    # 预测头输出若还被别的节点当作输入，改名会破坏那些节点，宁可报错也不产出坏图
    for node in model.graph.node:
        if det_name in node.input:
            raise RuntimeError(
                f"预测头输出 {det_name} 仍被节点 {node.op_type} 使用，无法安全插转置"
            )

    producer = None
    for node in model.graph.node:
        if det_name in node.output:
            producer = node
            break
    if producer is None:
        raise RuntimeError(f"找不到产生 {det_name} 的节点")

    # 一个节点可能有多路输出，只改名字匹配的那一路
    slot = list(producer.output).index(det_name)
    producer.output[slot] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == det_name:
            dims = specout.type.tensor_type.shape.dim
            if len(dims) != 3:
                raise RuntimeError(f"预测头 {det_name} 维度为 {len(dims)}，期望 3 维")
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0],
            )
            new_dims = new_out.type.tensor_type.shape.dim
            new_dims[0].CopyFrom(dims[0])
            new_dims[1].CopyFrom(dims[2])
            new_dims[2].CopyFrom(dims[1])
            specout.CopyFrom(new_out)
            break

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [det_name], perm=[0, 2, 1])
    )

    if task == "segment":
        _reorder_segment_outputs(model, det_name)

    dst = src.with_name(src.stem + ".transd" + src.suffix)
    onnx.save(model, str(dst))
    emit(
        {
            "type": "event",
            "name": "transd_written",
            "task": task,
            "path": str(dst),
            "outputs": [
                {"name": o.name, "dims": _output_dims(o)} for o in model.graph.output
            ],
        }
    )
    return str(dst)


def default_model_for_task(task: str) -> str:
    return {
        "detect": "yolov8n.pt",
        "segment": "yolov8n-seg.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt",
        "classify": "yolov8n-cls.pt",
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
        # 分类要走 check_cls_dataset，它只认目录；yaml 在这里换成目录，
        # 否则训练启动就会抛「must be a directory ... not a file」。
        "data": (
            resolve_classify_data(args.data) if args.task == "classify" else args.data
        ),
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
        "resume": str(Path(args.resume_from).resolve()) if args.resume_from else args.resume,
        "plots": args.plots,
        "verbose": True,
        # 微调写回同一个实验目录时必须置 True，否则 Ultralytics 会静默生成 exp2/exp3，
        # 用户以为覆盖了原实验，实际多出几个目录谁也不知道哪个是新的。
        "exist_ok": True if getattr(args, "exist_ok", False) else None,
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


def _effective_amp(trainer: Any) -> Optional[bool]:
    """从 trainer 上读出**真正生效**的混合精度标志。

    为什么不能看 ``args.amp``：黑名单显卡（如 T1000）上 ``args.amp`` 仍是 True，
    但 ``Trainer.__init__`` 里的 ``check_amp()`` 已经把它关掉了。真正的事实是
    ``trainer.amp``——一个 0 维张量（DDP 下会被 broadcast）。
    """
    value = getattr(trainer, "amp", None)
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)
    except (TypeError, ValueError):
        return None


def _current_gpu_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - 观测失败不影响训练
        return None
    return None


def _collect_amp_state(trainer: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """``amp_state`` 事件：请求值 vs 实际值。

    页面上「为什么慢一倍」只有这一个答案来源，所以即使拿不到也要发事件（值为 None），
    让前端能明确显示「未知」，而不是干脆不显示、让人以为在用 fp16。
    """
    effective = _effective_amp(trainer)
    requested = getattr(args, "amp", None)
    if effective is None:
        precision = "unknown"
    else:
        precision = "fp16" if effective else "fp32"
    return {
        "amp_requested": requested,
        "amp_effective": effective,
        "precision": precision,
        "gpu": _current_gpu_name(),
    }


def _collect_vram_peak(trainer: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """``vram_peak`` 事件：本次训练的峰值显存 + 可写入实测表的样本键。

    取 ``max_memory_reserved`` 而不是 ``allocated``：OOM 发生在向驱动申请的那一刻，
    预留（reserved）才是 PyTorch 真正占住的显存，allocated 会低估 30% 以上
    （本机 batch=8@640 实测 1769 MB，但同一组合在 2600 MB 可用时并不稳）。

    样本里的 batch 用 ``trainer.batch_size``（实际生效值），不是命令行请求值：
    Ultralytics 在首轮 OOM 时会悄悄把 batch 减半，用请求值记账会让下次预检误判。
    """
    peak_mb: Optional[float] = None
    allocated_mb: Optional[float] = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_mb = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
            allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001 - 读数失败只影响自校准
        peak_mb = None

    payload: Dict[str, Any] = {
        "peak_mb": peak_mb,
        "peak_allocated_mb": allocated_mb,
        "sample": None,
    }
    if not peak_mb or peak_mb <= 0:
        return payload

    trainer_args = getattr(trainer, "args", None)
    imgsz = getattr(trainer_args, "imgsz", None) or getattr(args, "imgsz", None)
    batch = getattr(trainer, "batch_size", None) or getattr(trainer_args, "batch", None)
    try:
        imgsz_int = int(imgsz)
        batch_int = int(batch)
    except (TypeError, ValueError):
        return payload
    if imgsz_int <= 0 or batch_int <= 0:
        return payload

    effective = _effective_amp(trainer)
    payload["sample"] = {
        "imgsz": imgsz_int,
        "batch": batch_int,
        "peak_mb": peak_mb,
        "peak_allocated_mb": allocated_mb,
        "precision": "fp32" if effective is False else ("fp16" if effective else "unknown"),
    }
    return payload


def write_dataset_meta(save_dir: Any, data_arg: Any) -> None:
    """把数据集指纹写进训练目录（``train_meta.json``）。

    为什么由 launcher 写而不是 Web 后端写：只有 trainer 知道**最终**的 ``save_dir``。
    用户填的 name 已存在时 Ultralytics 会自动改成 ``name2``，后端按请求路径写会写错位置。

    指纹内容由环境变量 ``TRTFORGE_DATASET_FP`` 传入，不走 argv：命令行会出现在
    进程列表、``runtime/logs/*.log`` 与前端「运行中」条的 tooltip 里，指纹没必要四处留。

    失败只打日志：这份文件是溯源凭据，写不出来不该让已经成功的训练变成失败。
    """
    raw = os.environ.get("TRTFORGE_DATASET_FP", "").strip()
    if not raw or not save_dir:
        return
    try:
        dataset = json.loads(raw)
    except ValueError:
        return
    if not isinstance(dataset, dict):
        return
    payload = {
        "dataset": dataset,
        "data_arg": str(data_arg or ""),
        "run_dir": str(save_dir),
        "run_id": os.environ.get("TRTFORGE_RUN_ID", "").strip(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        (Path(save_dir) / "train_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"写入 train_meta.json 失败（不影响训练）：{type(exc).__name__}: {exc}", flush=True)


def add_train_callbacks(model: Any, args: argparse.Namespace) -> None:
    last_gpu_emit = [0.0]

    def on_pretrain_routine_end(trainer: Any) -> None:
        # Ultralytics 在这之前已扫完数据集、做完 AMP 检查、建好优化器，
        # 之后才进入真正的训练循环。这是「启动」与「训练」的分界线。
        emit_phase("data", "数据集与优化器就绪")

    def on_train_start(trainer: Any) -> None:
        total = int(getattr(getattr(trainer, "args", object()), "epochs", 0) or 0)
        emit_phase("train_start", "开始训练", total=total)
        emit({"type": "event", "name": "train_start", "ts": time.time(), "total": total})
        emit({"type": "event", "name": "amp_state", "ts": time.time(), **_collect_amp_state(trainer, args)})
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
            write_dataset_meta(save_dir, args.data)
        emit({"type": "event", "name": "vram_peak", "ts": time.time(), **_collect_vram_peak(trainer, args)})
        emit({"type": "event", "name": "train_finished", "ts": time.time()})

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
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
    # import ultralytics 会连带 import torch，是启动耗时的大头之一
    emit_phase("deps", "训练依赖已导入", module="ultralytics")
    model = YOLO(model_path)
    emit_phase("model", "权重已加载", weights=str(model_path))
    return model


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
    if args.resume_from:
        checkpoint = Path(args.resume_from).expanduser()
        if not checkpoint.is_file():
            emit_error("CHECKPOINT_NOT_FOUND", "续训检查点不存在", path=str(checkpoint))
            return 2
        model_path = str(checkpoint.resolve())
    else:
        model_path = args.model or default_model_for_task(args.task)
    model = load_yolo(model_path)
    model_task = str(getattr(model, "task", "") or "").strip()
    if args.task == "classify" and model_task and model_task != "classify":
        emit_error(
            "UNSUPPORTED_TASK",
            f"当前权重是{model_task}，分类训练要换分类权重，例如 yolov8n-cls.pt",
            weights=model_path,
            model_task=model_task,
        )
        return 2
    add_train_callbacks(model, args)
    # 微调：起点是自己训好的 best.pt，参数按表单走（优化器与轮次重置）。
    # 单独发一条事件而不是只打普通日志，Web 端才能把「这次接着谁的权重」显示成结构化信息。
    if args.init_mode == "finetune":
        names = getattr(model, "names", None)
        emit(
            {
                "type": "event",
                "name": "finetune_from",
                "weights": model_path,
                "classes": list(names.values()) if isinstance(names, dict) else [],
                "exist_ok": bool(args.exist_ok),
            }
        )
        print(f"微调模式：从 {model_path} 继续训练（优化器与轮次重置）", flush=True)
    emit({"type": "event", "name": "launcher_train", "task": args.task, "model": model_path})
    if args.task == "classify" and int(args.imgsz or 0) > 320:
        # 分类网络默认 224 训练；直接用 640 会慢一个量级且收益很小。
        # 不强制改，只提醒一句，避免用户以为程序卡住。
        emit(
            {
                "type": "warn",
                "code": "CLASSIFY_IMGSZ_LARGE",
                "message": f"分类常用 imgsz=224，当前 {args.imgsz} 会明显变慢且精度收益很小",
            }
        )
    try:
        train_kwargs = build_train_kwargs(args)
    except ValueError as exc:
        # classify 的 yaml 解析失败是用户配置问题，给专用错误码，
        # 比落到最外层 UNHANDLED_EXCEPTION 更容易看懂。
        emit_error("CLASSIFY_DATA_INVALID", str(exc), data=args.data)
        return 2
    results = model.train(**train_kwargs)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    emit_training_artifacts(save_dir)
    if args.task == "classify":
        # 分类没有 mAP，emit_metrics 只会把 loss 和速度当指标发出去，
        # 训练页的「准确率」一栏永远是空的。这里改发 top-1 和混淆矩阵。
        _emit_classify_metrics(results)
    else:
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
    if args.task == "classify":
        try:
            classify_data = resolve_classify_data(args.data) if args.data else None
        except ValueError as exc:
            emit_error("CLASSIFY_DATA_INVALID", str(exc), data=args.data)
            return 2
        # 分类验证只认这些参数：conf/iou/max_det 是检测后处理，save_json/save_txt
        # 是检测结果落盘，传给分类验证要么被忽略，要么直接报参数不合法。
        kwargs = {
            "data": classify_data,
            "imgsz": args.imgsz,
            "batch": parse_batch(args.batch),
            "half": args.half,
            "device": args.device,
            "plots": args.plots,
            "split": args.split,
            "workers": args.workers,
            "verbose": True,
        }
    else:
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
    if args.task == "classify":
        _emit_classify_metrics(metrics)
    else:
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

    # 以权重自带的 task 为准：拿 seg 权重配 --task detect 导出，会生成一个
    # 通道数 116 的「检测头」，编译能过、推理时通道数校验才炸。
    # 这里不静默改，而是明确告诉用户改了什么，避免「我明明填了 detect」的困惑。
    model_task = str(getattr(model, "task", "") or "").strip()
    task = args.task
    if model_task and model_task in SUPPORTED_TASKS and model_task != args.task:
        emit(
            {
                "type": "event",
                "name": "task_overridden",
                "requested": args.task,
                "actual": model_task,
                "message": f"权重实际任务为 {model_task}，已忽略命令行传入的 {args.task}",
            }
        )
        task = model_task

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
    emit({"type": "event", "name": "export_start", "weights": weight, "format": args.format, "task": task})
    exported = model.export(**{k: v for k, v in kwargs.items() if v is not None})
    onnx_path = str(exported) if exported else ""
    if not onnx_path or not Path(onnx_path).exists():
        emit_error("EXPORT_FAILED", "ONNX 导出失败，未找到导出文件")
        return 1
    emit({"type": "artifact", "kind": "onnx", "path": onnx_path})
    if args.task == "classify":
        # 分类输出是类别分数，检测用的转置文件套上去会得到错误布局。
        emit({"type": "event", "name": "transd_skipped", "reason": "分类这一版不生成 .transd.onnx"})
        emit({"type": "event", "name": "finished", "exit": 0, "transd": False})
        return 0
    if str_to_bool(args.make_transd):
        try:
            transd = make_transd_onnx(onnx_path, task)
        except Exception as exc:  # noqa: BLE001 - 转置失败要给出可读原因
            emit_error(
                "TRANSD_FAILED",
                f"生成 .transd.onnx 失败：{type(exc).__name__}: {exc}",
                onnx=onnx_path,
                task=task,
            )
            # 原始 ONNX 已导出成功，转置属于附加产物，不算整个导出失败
            emit({"type": "event", "name": "finished", "exit": 0, "transd": False})
            return 0
        emit({"type": "artifact", "kind": "transd_onnx", "path": transd})
    emit({"type": "event", "name": "finished", "exit": 0, "transd": True})
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
    # 显式续训：拿到具体某个 last.pt，而不是让 Ultralytics 用 get_latest_run()
    # 去当前工作目录里猜「最近一次 run」。Web 端子进程的 cwd 是 launcher 目录，
    # 猜错就等于静默换了数据集/模型接着训，而且不会报错。
    parser.add_argument("--resume-from", default="", help="从指定的 .pt 检查点续训（优先级高于 --resume）")
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
    # 微调写回同一实验目录：默认 False 与 Ultralytics 一致，只有后端显式要求时才置 True。
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    # 起点语义。preset=官方预训练/新实验；finetune=用自己的 best.pt 微调（参数按表单）；
    # resume=断点续训（走 --resume-from，参数以检查点为准）。
    parser.add_argument("--init-mode", choices=["preset", "finetune", "resume"], default="preset")
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


def mode_warm(_: argparse.Namespace) -> int:
    """常驻预热进程：先把重依赖导入好，等后端通过 stdin 下发一次任务。

    为什么只服务一个任务就退出：
    - torch / Ultralytics 有大量进程级全局状态（CUDA 上下文、随机源、默认配置）；
    - 复用同一进程跑第二个任务，上一个任务的设置可能悄悄影响下一个，而日志上看不出来；
    - 只服务一个任务，「停止」仍然等价于 terminate 整个进程，语义与现在一致。

    预热只负责 import，不在预热期加载模型对象：训练会改写模型与 trainer 状态，
    把一份「已经练过」的对象交给下一个任务，比省下几百毫秒风险大得多。
    """
    global _TASK_T0
    add_ultralytics_source_path()
    ensure_import("torch")
    ensure_import("ultralytics")
    emit(
        {
            "type": "event",
            "name": "warm_ready",
            "pid": os.getpid(),
            "elapsed_ms": int((time.perf_counter() - _T0) * 1000),
        }
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            emit_error("BAD_WARM_COMMAND", f"预热进程收到非法指令：{line[:200]}")
            return 2
        argv = payload.get("argv") if isinstance(payload, dict) else None
        if not isinstance(argv, list) or not argv:
            emit_error("BAD_WARM_COMMAND", "预热指令缺少 argv 列表")
            return 2
        # 哨兵：告诉 main() 这是复用预热进程，阶段时钟要从「收到任务」开始算
        _TASK_T0 = time.perf_counter()
        return main([str(item) for item in argv])
    # stdin 被关闭 = 后端主动回收，静默退出即可
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # 第一条 phase 必须尽早发：Web 端靠它确认「进程真的起来了」，
    # 否则点击后这段时间界面完全空白，看起来像卡住。
    warm_hit = reset_phase_clock()
    emit_phase("boot", "python 进程已启动", warm=warm_hit)

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
        if args.mode == "warm":
            return mode_warm(args)
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
