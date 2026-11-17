"""TRTForge AnomalyDINO 启动器：建库 → 导出 ONNX → 标定阈值 → 写 metadata。

为什么单独一个启动器，而不是塞进 `tf_anomalib_launcher.py`：

- AnomalyDINO **不是 anomalib 的模型**。它的「训练」只是一次前向，没有 Lightning、
  没有优化器、没有 checkpoint。硬塞进 anomalib 那条路会把 `import lightning`（5 秒以上）
  变成必经步骤，而这条路根本用不到。
- 它的产物契约与 PatchCore 一致（ONNX + metadata + TRT 引擎），所以**下游一行都不用改**：
  同一个 `anomalib_infer.dll`、同一个推理页、同一个全流程卡片。

与 `tf_anomalib_launcher.py` 保持同一套 `TRTFORGE_JSON` 事件协议，前端不必新增渲染分支。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

JSON_PREFIX = "TRTFORGE_JSON "
SUPPORTED_MODES = {"check-env", "build-export", "warm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 少于这个数量的正常图建出来的库没有统计意义，宁可报错也不给一个假能用模型。
MIN_REFERENCE_IMAGES = 8

# 上游 AnomalyDINO 的旋转增强是 8 个角度（0/45/.../315），库里行数会 ×8。
ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

OUTPUT_CONTRACT = "trtforge_anomalydino_v1"
MODEL_LABEL = "AnomalyDINO"

_LAUNCHER_DIR = Path(__file__).resolve().parent
if str(_LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_DIR))


# ---------------------------------------------------------------------------
# 结构化事件：与 tf_anomalib_launcher.py 完全同一套字段
# ---------------------------------------------------------------------------

def emit(event: dict[str, Any]) -> None:
    print(JSON_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_error(code: str, message: str, **extra: Any) -> None:
    payload = {"type": "error", "code": code, "message": message}
    payload.update(extra)
    emit(payload)


_T0 = time.perf_counter()
_last_phase_at = _T0
_TASK_T0: float | None = None


def reset_phase_clock() -> bool:
    global _T0, _last_phase_at, _TASK_T0
    warm_hit = _TASK_T0 is not None
    _T0 = time.perf_counter()
    _last_phase_at = _T0
    _TASK_T0 = None
    return warm_hit


def emit_phase(phase: str, label: str, **extra: Any) -> None:
    global _last_phase_at
    now = time.perf_counter()
    payload: dict[str, Any] = {
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


def emit_progress(stage: str, pct: float, **extra: Any) -> None:
    payload = {"type": "progress", "stage": stage, "pct": round(float(pct), 1)}
    payload.update(extra)
    emit(payload)


def emit_artifact(kind: str, path: Path) -> None:
    emit({"type": "artifact", "kind": kind, "path": str(path)})


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def import_status(module_name: str) -> dict[str, Any]:
    """报告一个依赖能否 import 及版本号。"""
    import importlib

    try:
        module = importlib.import_module(module_name)
        return {"ok": True, "version": str(getattr(module, "__version__", ""))}
    except Exception as exc:  # noqa: BLE001 - 依赖问题要原样报给界面
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(root: Path, value: str, required: bool = True) -> Path | None:
    """相对路径按 root 解析，绝对路径原样返回。"""
    text = str(value or "").strip()
    if not text:
        if required:
            raise FileNotFoundError("路径为空")
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if required and not candidate.exists():
        raise FileNotFoundError(f"路径不存在：{candidate}")
    return candidate


def validate_image_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} 不是目录：{path}")
    if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir() if p.is_file()):
        raise FileNotFoundError(f"{name} 里没有图片：{path}")


def collect_images(roots: Iterable[Path], limit: int = 0) -> list[Path]:
    """收集图片，去重并按绝对路径排序，保证多次运行顺序一致。"""
    seen: set[str] = set()
    items: list[Path] = []
    for root in roots:
        if root is None or not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                key = os.path.normcase(str(path.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                items.append(path)
                if limit and len(items) >= limit:
                    return items
    return items


def next_version_dir(project: Path, name: str) -> Path:
    """给出 `project/AnomalyDINO/name/v<N>/`，N 取现有最大 +1。

    与 Anomalib 那条路的目录结构保持一致（它由 Lightning 的 default_root_dir 生成），
    这样 `models.py` 的扫描逻辑与全流程的延迟路径解析都不用改。
    """
    base = project / MODEL_LABEL / (name or "latest")
    base.mkdir(parents=True, exist_ok=True)

    def _index(path: Path) -> int:
        text = path.name[1:]
        return int(text) if path.name.startswith("v") and text.isdigit() else -1

    existing = [p for p in base.iterdir() if p.is_dir()]
    # 已存在编号但还没有产物的目录（上次失败留下的）应当被复用，否则每失败一次就多一个空版本。
    for candidate in existing:
        if (candidate / "weights" / "lightning" / "metadata.json").is_file():
            continue
        if _index(candidate) >= 0:
            return candidate
    next_index = max([_index(p) for p in existing] + [-1]) + 1
    return base / f"v{next_index}"


def percentile(values: list[float], q: float) -> float:
    import numpy as np

    if not values:
        raise ValueError("percentile 输入为空")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def threshold_from_distributions(normal_values: list[float], abnormal_values: list[float]) -> float:
    """按「正常 99 分位 / 异常 5 分位」取阈值，规则与 anomalib 那条路一致。

    这里有意复制了 `tf_anomalib_launcher.threshold_from_distributions` 的实现，
    而不是 import 它：那条路是已经在跑的旧代码，本次改动不改它一行；
    为一个 10 行纯函数建立「新启动器依赖旧启动器」的关系不划算。
    两边的口径由测试对齐（见 test_anomalydino_launcher.py）。
    """
    normal_high = percentile(normal_values, 99.0)
    if not abnormal_values:
        return normal_high
    abnormal_low = percentile(abnormal_values, 5.0)
    if abnormal_low > normal_high:
        return (normal_high + abnormal_low) / 2.0
    span = max(max(normal_values + abnormal_values) - min(normal_values + abnormal_values), 1.0e-6)
    return normal_high + span * 0.02


# ---------------------------------------------------------------------------
# 图像读取（中文路径必须可用）
# ---------------------------------------------------------------------------

def read_rgb(path: Path) -> Any:
    """读图转 RGB。

    用 `imdecode + np.fromfile`：`cv2.imread` 在 Windows 上读不了含中文的路径，
    而本项目要求中文路径可用。DLL 的输入是 RGB，这里必须同序。
    """
    import cv2
    import numpy as np

    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise FileNotFoundError(f"读不到图片：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def augment_rotations(image: Any) -> list[Any]:
    """8 个角度的旋转增强，复刻上游 `augment_image`（warpAffine + 默认边界）。"""
    import cv2
    import numpy as np

    centre = tuple(np.array(image.shape[1::-1]) / 2)
    outputs = []
    for angle in ROTATION_ANGLES:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        outputs.append(cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR))
    return outputs


# ---------------------------------------------------------------------------
# 建库与导出
# ---------------------------------------------------------------------------

def build_bank(
    model: Any,
    images: list[Path],
    resolution: int,
    device: str,
    rotation: bool,
    batch_size: int = 4,
) -> tuple[Any, dict[str, Any]]:
    """前向建库，返回 (库, 统计信息)。"""
    import anomalydino_core as core
    import torch

    started = time.perf_counter()
    tokens: list[Any] = []
    used_images = 0
    augmented_total = 0
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        pictures: list[Any] = []
        for path in chunk:
            image = read_rgb(path)
            variants = augment_rotations(image) if rotation else [image]
            augmented_total += len(variants)
            pictures.extend(variants)
            used_images += 1
            # 旋转增强会让单批张量翻 8 倍，512×512×3 的 float32 一个批次就可能上 GB，
            # 所以按「张」累计到 batch_size 就先跑一次前向。
            if len(pictures) >= batch_size:
                batch = core.preprocess_like_dll(pictures[:batch_size], resolution).to(device)
                tokens.extend(core.extract_patch_tokens(model, batch).cpu())
                pictures = pictures[batch_size:]
        if pictures:
            batch = core.preprocess_like_dll(pictures, resolution).to(device)
            tokens.extend(core.extract_patch_tokens(model, batch).cpu())
        emit_progress(
            "bank",
            20.0 + 40.0 * (start + len(chunk)) / max(len(images), 1),
            done=start + len(chunk),
            total=len(images),
        )

    bank = core.build_bank(tokens)
    stats = {
        "images": used_images,
        "augmented": augmented_total,
        "rows": int(bank.shape[0]),
        "seconds": round(time.perf_counter() - started, 3),
        "sha256": core.bank_digest(bank),
    }
    return bank, stats


def export_onnx(wrapper: Any, resolution: int, onnx_path: Path) -> None:
    """导出 ONNX，opset 固定 16。

    opset 17 会把 `F.layer_norm` 导成 `LayerNormalization` 单算子，TensorRT 8.5
    没有原生实现，onnx 解析器会去找一个不存在的插件并失败（实测
    `Plugin not found`）。opset 16 下它被拆成 ReduceMean/Sub/Mul/Div/Sqrt，TRT 全支持。
    """
    import torch

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


def open_onnx_session(onnx_path: Path) -> tuple[Any, str, list[str]]:
    """建 ONNX Runtime 会话，返回 (session, 输入名, 输出名列表)。"""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    # CUDA EP 在部分使用人员机器上会因 cuDNN 版本不匹配直接崩，标定用 CPU 足够
    # （几十张图，每张几十毫秒），换来的是「不会因为驱动问题导不出模型」。
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session, session.get_inputs()[0].name, [o.name for o in session.get_outputs()]


def parse_outputs(outputs: list[Any], names: list[str]) -> tuple[float, Any]:
    """按名字挑出分数与异常图，规则逐条复刻 `anomalib_infer.dll`。

    单一循环、判断顺序都与 `anomalib_infer_c.cpp:454-471` 一致：

    1. 名字含 `score` → 记为分数（**后者覆盖前者**，C++ 是无条件赋值）；
    2. 名字含 `anomaly_map`，或含 `map` 且不含 `mask` → 记为异常图；
    3. **任何**输出的元素数比当前异常图大 → 顶替它（这条是陷阱：多导出一个
       大 mask 就会把真正的热力图挤掉，且不报错）；
    4. 元素数为 1 且名字不含 `label` → 没有分数时当分数。

    必须一次循环完成：先把名字匹配都做完再比大小，结果会和 C++ 不同
    （`[大图, 名为 anomaly_map 的小图]` 这种导出的挑选结果就会反过来）。
    """
    import numpy as np

    arrays = [np.asarray(value) for value in outputs]
    score_idx: int | None = None
    map_idx: int | None = None
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
    if map_idx is None:
        raise ValueError("ONNX 没有任何输出，无法解析异常图")
    anomaly_map = arrays[map_idx]
    if score_idx is not None:
        score = float(np.ravel(arrays[score_idx])[0])
    else:
        score = float(np.max(anomaly_map))
    return float(score), anomaly_map


def calibrate_metadata(
    onnx_path: Path,
    resolution: int,
    normal_images: list[Path],
    abnormal_images: list[Path],
) -> dict[str, Any]:
    """用真图跑 ONNX 取原始分数与热力图，算出阈值与取值范围。

    **必须用与 DLL 完全一致的预处理**（拉伸 + /255 + ImageNet 归一），
    否则标定出来的阈值与线上判定是两张皮。
    """
    import anomalydino_core as core
    import numpy as np

    session, input_name, output_names = open_onnx_session(onnx_path)
    normal_scores: list[float] = []
    abnormal_scores: list[float] = []
    normal_pixels: list[float] = []
    abnormal_pixels: list[float] = []
    map_mins: list[float] = []
    map_maxs: list[float] = []

    def infer_group(paths: list[Path], scores: list[float], pixels: list[float]) -> None:
        for index, image_path in enumerate(paths):
            tensor = core.preprocess_like_dll([read_rgb(image_path)], resolution).numpy()
            outputs = session.run(None, {input_name: tensor})
            score, anomaly_map = parse_outputs(outputs, output_names)
            scores.append(float(score))
            array = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
            if array.size:
                map_mins.append(float(np.min(array)))
                map_maxs.append(float(np.max(array)))
                pixels.append(percentile(array.tolist(), 99.5))
            emit_progress(
                "calibrate",
                60.0 + 30.0 * index / max(len(paths), 1),
                done=index + 1,
                total=len(paths),
            )

    infer_group(normal_images, normal_scores, normal_pixels)
    infer_group(abnormal_images, abnormal_scores, abnormal_pixels)
    if not normal_scores:
        raise ValueError("标定失败：没有任何正常图参与标定")

    all_scores = normal_scores + abnormal_scores
    image_threshold = threshold_from_distributions(normal_scores, abnormal_scores)
    pixel_threshold = (
        threshold_from_distributions(normal_pixels, abnormal_pixels)
        if normal_pixels
        else image_threshold
    )
    metadata: dict[str, Any] = {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "pred_scores": {
            "min": float(min(all_scores)),
            # 上界至少覆盖阈值本身，否则阈值会被归一化函数夹到 1.0 之外
            "max": float(max(max(all_scores), image_threshold)),
        },
        "anomaly_maps": {
            "min": float(min(map_mins)) if map_mins else 0.0,
            "max": float(max(max(map_maxs), pixel_threshold)) if map_maxs else 1.0,
        },
        "calibration": {
            "source": "trtforge_anomalydino_distribution",
            "normal_image_count": len(normal_scores),
            "abnormal_image_count": len(abnormal_scores),
        },
    }
    return metadata


def write_metadata(
    output_path: Path,
    metadata: dict[str, Any],
    onnx_path: Path,
    resolution: int,
    bank_stats: dict[str, Any],
    shots: int,
    rotation: bool,
    seed: int,
    checkpoint: Path,
    checkpoint_sha: str,
) -> Path:
    """写 metadata.json。

    点号平铺键与嵌套对象两种写法都要有：DLL 认第一种
    （`anomalib_infer_c.cpp:598-605`），`6-web` 的 `bundle.py` 只透传嵌套块。
    """
    grid_side = resolution // 14
    payload: dict[str, Any] = {
        "model_type": MODEL_LABEL,
        "output_contract": OUTPUT_CONTRACT,
        # 主干是 DINOv2 + ImageNet 归一，与 PatchCore 同一条 DLL 分支
        "normalization": "imagenet",
        "input_size": [resolution, resolution],
        "created_by": "2-python-anomaly/launcher/tf_anomalydino_launcher.py",
        "backbone": "vit_small_patch14_dinov2",
        "backbone_checkpoint": checkpoint.name,
        "backbone_sha256": checkpoint_sha,
        "patch_size": 14,
        "grid": [grid_side, grid_side],
        "shots": shots,
        "rotation": bool(rotation),
        "masking": False,
        "seed": int(seed),
        "bank_rows": bank_stats["rows"],
        "bank_images": bank_stats["images"],
        "bank_augmented": bank_stats["augmented"],
        "bank_sha256": bank_stats["sha256"],
        "onnx": onnx_path.name,
    }
    for key in ("image_threshold", "pixel_threshold", "pred_scores", "anomaly_maps", "calibration"):
        if key in metadata:
            payload[key] = metadata[key]
    # 平铺副本
    payload["pred_scores.min"] = payload["pred_scores"]["min"]
    payload["pred_scores.max"] = payload["pred_scores"]["max"]
    payload["anomaly_maps.min"] = payload["anomaly_maps"]["min"]
    payload["anomaly_maps.max"] = payload["anomaly_maps"]["max"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------

def collect_gpu_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"available": False, "torch_cuda": str(torch.version.cuda)}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_mb": round(props.total_memory / 1024 / 1024, 1),
        "torch_cuda": str(torch.version.cuda),
    }


def mode_check_env(args: argparse.Namespace) -> int:
    """检查依赖与权重，不发出一条训练动作。"""
    import anomalydino_core as core

    env: dict[str, Any] = {
        "type": "env",
        "model": MODEL_LABEL,
        "python": sys.executable,
        "env_name": os.environ.get("CONDA_DEFAULT_ENV", "") or os.environ.get("VIRTUAL_ENV", ""),
    }
    modules = {
        "torch": import_status("torch"),
        "timm": import_status("timm"),
        "onnx": import_status("onnx"),
        "onnxruntime": import_status("onnxruntime"),
        "numpy": import_status("numpy"),
        "cv2": import_status("cv2"),
    }
    env["modules"] = modules
    env["gpu"] = collect_gpu_status()

    try:
        checkpoint = core.resolve_checkpoint(args.checkpoint or None)
        env["checkpoint"] = {
            "ok": True,
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": core.checkpoint_sha256(checkpoint),
        }
    except FileNotFoundError as exc:
        env["checkpoint"] = {"ok": False, "error": str(exc)}

    missing = [name for name, status in modules.items() if not status.get("ok")]
    if missing:
        emit(env)
        emit_error(
            "MISSING_DEP",
            f"缺少依赖：{', '.join(missing)}",
            missing=missing,
            install=f"python -m pip install {' '.join(missing)}",
        )
        return 1
    if not env["checkpoint"].get("ok"):
        emit(env)
        emit_error("MISSING_DEP", "缺少 DINOv2 权重 dinov2_vits14_pretrain.pth",
                   detail=env["checkpoint"].get("error", ""))
        return 1
    emit(env)
    emit({"type": "event", "name": "env_ok"})
    return 0


def mode_build_export(args: argparse.Namespace) -> int:
    """建库 → 导出 ONNX → 标定 → 写 metadata。"""
    import anomalydino_core as core
    import torch

    resolution = int(args.input_size)
    if resolution % core.PATCH_SIZE != 0:
        emit_error(
            "UNSUPPORTED_RESOLUTION",
            f"输入尺寸必须是 {core.PATCH_SIZE} 的倍数（例如 448），实际 {resolution}",
        )
        return 2
    grid_h, grid_w = core.grid_for_resolution(resolution)

    root = Path(args.root).expanduser().resolve()
    normal_dir = resolve_path(root, args.normal_dir)
    assert normal_dir is not None
    validate_image_dir(normal_dir, "正常图像目录 normal_dir")
    # 异常目录只用于标定，允许不存在（用户可能完全只有好品图）
    abnormal_dir = resolve_path(root, args.abnormal_dir, required=False)
    normal_test_dir = resolve_path(root, args.normal_test_dir, required=False)

    shots = int(args.shots)
    rotation = parse_bool(args.rotation)
    seed = int(args.seed)
    images = collect_images([normal_dir])
    if shots > 0:
        # 上游 few-shot 是 `[seed*n:(seed+1)*n]`，seed=0 时就是前 n 张
        start = max(seed, 0) * shots
        images = images[start : start + shots]
    if len(images) < MIN_REFERENCE_IMAGES:
        emit_error(
            "DATA_TOO_FEW",
            f"正常图只有 {len(images)} 张，少于下限 {MIN_REFERENCE_IMAGES} 张。"
            "记忆库太小会让判定随机，请补充正常图或把 shots 调大。",
            available=len(images),
            required=MIN_REFERENCE_IMAGES,
        )
        return 2

    device = str(args.device or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，退回 CPU（建库会明显变慢）", flush=True)
        device = "cpu"

    project = Path(args.project)
    if not project.is_absolute():
        project = _LAUNCHER_DIR / project
    version_dir = next_version_dir(project, args.name or root.name or "anomalydino")
    weights_dir = version_dir / "weights" / "lightning"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{weights_dir}", flush=True)
    emit_artifact("run_dir", version_dir)
    emit_progress("setup", 5.0)

    checkpoint = core.resolve_checkpoint(args.checkpoint or None)
    checkpoint_sha = core.checkpoint_sha256(checkpoint)
    print(f"权重：{checkpoint}（sha256 {checkpoint_sha[:16]}…）", flush=True)
    emit_phase("weights", "权重就绪", path=str(checkpoint))

    model = core.create_backbone(checkpoint, resolution, device)
    emit_phase("model", "主干已加载", backbone="vit_small_patch14_dinov2")
    emit(
        {
            "type": "anomaly_model",
            "model": MODEL_LABEL,
            "model_type": MODEL_LABEL,
            "input_size": [resolution, resolution],
            "grid": [grid_h, grid_w],
            "shots": shots,
            "rotation": rotation,
        }
    )
    emit_progress("bank", 20.0, done=0, total=len(images))

    bank, bank_stats = build_bank(model, images, resolution, device, rotation)
    emit_phase("bank", "记忆库已建好", rows=bank_stats["rows"], seconds=bank_stats["seconds"])
    print(f"记忆库：{bank_stats['rows']} 行（{bank_stats['augmented']} 张增强图），"
          f"{bank_stats['seconds']}s，sha256 {bank_stats['sha256']}", flush=True)

    onnx_path = weights_dir / "model.onnx"
    wrapper = core.BakedAnomalyDINO(model, bank, grid_h, grid_w)
    # 导出在 CPU 上做：dummy 输入在 CPU，权重还在 CUDA 会 device 不匹配。
    # 深拷贝一份，避免把用于后续标定的 GPU 主干一起搬走。
    import copy

    export_model = copy.deepcopy(wrapper).to("cpu").eval()
    export_onnx(export_model, resolution, onnx_path)
    emit_phase("export", "ONNX 已导出", path=str(onnx_path), bytes=onnx_path.stat().st_size)
    emit_artifact("onnx", onnx_path)
    emit_progress("export", 60.0)
    print(f"ONNX：{onnx_path}（{onnx_path.stat().st_size / 1e6:.1f} MB）", flush=True)

    if not parse_bool(args.calibrate_metadata):
        emit_error("METADATA_FAILED", "AnomalyDINO 必须标定阈值，不能关闭 calibrate_metadata")
        return 1

    calib_normal = resolve_path(root, args.calib_normal_dir, required=False)
    calib_abnormal = resolve_path(root, args.calib_abnormal_dir, required=False)
    normal_roots = [normal_dir, normal_test_dir, calib_normal]
    abnormal_roots = [abnormal_dir, calib_abnormal]
    limit = max(1, int(args.calib_limit))
    # 标定用的正常图不要与建库图完全重合：完全重合会让正常侧的分数恒为 0，
    # 阈值被压得过低，线上全是误报。优先用没进库的图，不够再回退。
    library_keys = {os.path.normcase(str(p.resolve())) for p in images}
    normal_pool = [p for p in collect_images(normal_roots, limit * 3)
                   if os.path.normcase(str(p.resolve())) not in library_keys]
    if not normal_pool:
        normal_pool = collect_images(normal_roots, limit)
    abnormal_pool = collect_images(abnormal_roots, limit)
    print(f"标定图：正常 {len(normal_pool)} 张，异常 {len(abnormal_pool)} 张", flush=True)
    emit_progress("calibrate", 60.0, done=0, total=len(normal_pool) + len(abnormal_pool))

    metadata = calibrate_metadata(onnx_path, resolution, normal_pool, abnormal_pool)
    emit_phase("calibrate", "阈值已标定",
               image_threshold=metadata["image_threshold"],
               pixel_threshold=metadata["pixel_threshold"])

    metadata_path = write_metadata(
        output_path=weights_dir / "metadata.json",
        metadata=metadata,
        onnx_path=onnx_path,
        resolution=resolution,
        bank_stats=bank_stats,
        shots=shots,
        rotation=rotation,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
    )
    emit_artifact("metadata", metadata_path)
    print(f"metadata：{metadata_path}", flush=True)

    emit(
        {
            "type": "anomaly_export_summary",
            "model": MODEL_LABEL,
            "model_type": MODEL_LABEL,
            "output_contract": OUTPUT_CONTRACT,
            "normalization": "imagenet",
            "image_threshold": metadata["image_threshold"],
            "pixel_threshold": metadata["pixel_threshold"],
            "onnx": str(onnx_path),
            "metadata": str(metadata_path),
            "bank_rows": bank_stats["rows"],
        }
    )
    emit_progress("finished", 100.0)
    emit({"type": "event", "name": "finished", "exit": 0, "run_dir": str(version_dir)})
    return 0


def mode_warm(_: argparse.Namespace) -> int:
    """常驻预热：先把 torch / timm / onnxruntime 导入好，等后端下发一次任务。

    与 anomalib 那条路的预热同构，但这里没有 Lightning，命中时的收益是
    torch + timm 的导入时间（本机约 2.4 秒）。不预建模型：模型带设备与权重状态，
    跨任务复用比重新构建风险高。
    """
    global _TASK_T0
    import torch  # noqa: F401 - 预热导入
    import timm  # noqa: F401
    import onnxruntime  # noqa: F401

    import anomalydino_core  # noqa: F401
    emit({"type": "event", "name": "warm_ready", "pid": os.getpid(),
          "elapsed_ms": int((time.perf_counter() - _T0) * 1000)})
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
        _TASK_T0 = time.perf_counter()
        return main([str(item) for item in argv])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRTForge AnomalyDINO 建库/导出启动器")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    # 保留 --model：训练页把它当通用字段发过来，这里接受但不参与分派（只有一种模型）
    parser.add_argument("--model", default="anomalydino")
    parser.add_argument("--root", default="")
    parser.add_argument("--normal-dir", default="train/good")
    parser.add_argument("--abnormal-dir", default="test/abnormal")
    parser.add_argument("--normal-test-dir", default="test/good")
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="latest")
    parser.add_argument("--input-size", type=int, default=448)
    # 零样本没有 epoch，保留参数只为让训练页的通用字段能原样传下来
    parser.add_argument("--max-epochs", default="1")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split-ratio", type=float, default=0.0)
    parser.add_argument("--imagenet-dir", default="")
    parser.add_argument("--calibrate-metadata", default="true")
    parser.add_argument("--calib-normal-dir", default="")
    parser.add_argument("--calib-abnormal-dir", default="")
    parser.add_argument("--calib-limit", type=int, default=64)
    # AnomalyDINO 专属
    parser.add_argument("--shots", type=int, default=16,
                        help="建库用多少张正常图，-1 表示全部；建议 8~32")
    parser.add_argument("--rotation", default="false", help="是否做 8 角度旋转增强")
    parser.add_argument("--seed", type=int, default=0, help="few-shot 取图偏移")
    parser.add_argument("--checkpoint", default="", help="显式指定 DINOv2 权重")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    warm_hit = reset_phase_clock()
    emit_phase("boot", "python 进程已启动", warm=warm_hit)
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "check-env":
            return mode_check_env(args)
        if args.mode == "build-export":
            if not str(args.root or "").strip():
                emit_error("DATA_NOT_FOUND", "请先选择数据集 root 目录。")
                return 2
            return mode_build_export(args)
        if args.mode == "warm":
            return mode_warm(args)
        emit_error("UNSUPPORTED_MODE", f"未知 mode：{args.mode}")
        return 2
    except KeyboardInterrupt:
        emit_error("CANCELLED", "用户取消了任务")
        return 130
    except ImportError as exc:
        package = getattr(exc, "name", "") or "unknown"
        emit_error("MISSING_DEP", f"缺少 Python 依赖：{package}", package=package,
                   install=f"python -m pip install {package}")
        traceback.print_exc()
        return 1
    except FileNotFoundError as exc:
        emit_error("DATA_NOT_FOUND", str(exc))
        traceback.print_exc()
        return 1
    except Exception as exc:  # noqa: BLE001 - 启动器必须把未知异常报给界面而不是静默退出
        emit_error(
            "UNHANDLED_EXCEPTION",
            f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
