# -*- coding: utf-8 -*-
"""onnxruntime 降级推理：DLL 不可用时的兜底路径。

为什么不直接复用 DLL：
DLL 是首选路径（前处理 / TensorRT / 解码 / 绘制全部在 C++ 里，性能最好）。
但当用户没编译 DLL 或缺少 OpenCV/CUDA 运行库时，服务只会返回 503，
整个推理页不可用。本模块用 onnxruntime 直接跑 ONNX，让页面至少能验证模型。

设计约束：
1. **返回结构与 infer.py 完全一致**，前端零改动；额外用 `backend` 字段标注真实后端，
   让用户一眼看出这次结果是 DLL 还是降级路径产出的。
2. **懒加载**：onnxruntime / cv2 只在真正降级时才 import。缺库时抛出带修复命令的错误，
   绝不因为降级依赖缺失而拖垮服务启动。
3. **不猜形状**：YOLO 的输入边长、输出布局（`[1,4+nc,n]` 还是 `[1,n,4+nc]`）、
   anomalib 的输出键名与维度，全部从模型本身读出来。本仓库里就有
   `[1,10,18900]`（未转置）和 `[1,18900,10]`（已转置）两种 YOLO 导出，
   PatchCore 的 `pred_score` 是 1 维、另两个模型是 2 维——写死任何一个都会崩。
4. **明确的能力边界**：只支持 detect / obb。segment / pose 的分支解码在 C++ 里，
   这里不做半吊子复刻，直接报错并提示走 DLL。

坐标语义（实测）：本仓库导出的 YOLO ONNX 输出的是**像素坐标**（0..input_size），
不是 0..1 归一化值。所以用 letterbox 的缩放比例还原即可。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# 降级支持的任务头。segment / pose 需要解析额外的输出张量并做掩码/关键点还原，
# 在 Python 里复刻容易与原实现产生细微差异，故不支持。
SUPPORTED_YOLO_TASKS = ("detect", "obb")

BACKEND_YOLO = "onnxruntime-fallback"
BACKEND_ANOMALIB = "onnxruntime-fallback"

# 缓存 session，避免每次请求重新加载模型（PatchCore ONNX 有 300MB）
_yolo_sessions: dict[str, object] = {}
_anom_sessions: dict[str, object] = {}


class FallbackError(RuntimeError):
    """降级路径的内部错误，由调用方转成 400/503。"""


def _import_ort():
    """按需导入 onnxruntime，给出可执行的修复命令。"""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise FallbackError(
            "降级推理需要 onnxruntime，当前解释器未安装。修复："
            "python -m pip install onnxruntime"
        ) from exc
    return ort


def _providers(ort) -> list[str]:
    """优先 CUDA，失败由 onnxruntime 自己回退，最后兜底 CPU。"""
    available = set(ort.get_available_providers())
    ordered = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return ordered or ["CPUExecutionProvider"]


def available() -> tuple[bool, str]:
    """探测降级能力，供 /api/capabilities 展示。"""
    try:
        ort = _import_ort()
    except FallbackError as exc:
        return False, str(exc)
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False, "降级推理需要 opencv-python。修复：python -m pip install opencv-python"
    return True, f"onnxruntime {ort.__version__}，providers={','.join(ort.get_available_providers())}"


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """等比缩放 + 灰边填充到 size×size，返回 (图, 缩放比, 左边距, 上边距)。

    与 Ultralytics 的 letterbox 一致：长边缩放到 size，居中填充 114 灰。
    """
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, ratio, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """标准 NMS，boxes 为 xyxy。"""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_r - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def _normalize_layout(raw: np.ndarray, num_classes: int) -> np.ndarray:
    """把输出统一成 [num_boxes, 4+nc (+1)] 的行布局。

    两种形态都要认：
      - `[1, 4+nc, n]`：未转置（本仓库 `best.onnx` 就是 `[1,10,18900]`）
      - `[1, n, 4+nc]`：已转置（`best.transd.onnx` 是 `[1,18900,10]`）
    """
    arr = np.squeeze(raw, axis=0) if raw.ndim == 3 else raw
    if arr.ndim != 2:
        raise FallbackError(f"无法识别的 YOLO 输出维度：{raw.shape}")

    rows, cols = arr.shape
    # 第二个维度等于 4+nc(+1) 时认为已转置；否则认为第一维是通道数
    if cols in (4 + num_classes, 5 + num_classes):
        return arr
    if rows in (4 + num_classes, 5 + num_classes):
        return arr.T
    raise FallbackError(
        f"输出形状 {raw.shape} 与类别数 {num_classes} 不匹配（期望 {4 + num_classes} 或 {5 + num_classes} 维）"
    )


def run_yolo(
    onnx_path: str,
    task: str,
    image: np.ndarray,
    conf: float = 0.25,
    nms: float = 0.45,
    num_classes: int | None = None,
) -> dict:
    """用 onnxruntime 跑 YOLO ONNX，返回与 DLL 路径一致的结构。"""
    if task not in SUPPORTED_YOLO_TASKS:
        raise FallbackError(
            f"降级推理只支持 {', '.join(SUPPORTED_YOLO_TASKS)}，当前任务 {task} 需要 yolo_infer.dll"
        )

    model_file = Path(onnx_path)
    if not model_file.is_file():
        raise FallbackError(f"模型文件不存在：{onnx_path}")

    # .transd.onnx 是我们自产的转置版本，onnxruntime 一样能跑，无需特殊处理
    key = f"{model_file.resolve()}|{task}"
    sess = _yolo_sessions.get(key)
    if sess is None:
        ort = _import_ort()
        sess = ort.InferenceSession(str(model_file), providers=_providers(ort))
        _yolo_sessions[key] = sess

    input_meta = sess.get_inputs()[0]
    shape = list(input_meta.shape)  # 可能是 ['batch', 3, 960, 960]
    if len(shape) != 4:
        raise FallbackError(f"YOLO 输入应为 4 维 NHWC/NCHW，实际 {shape}")
    size = shape[2] if isinstance(shape[2], int) else 640
    if not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise FallbackError(f"动态输入尺寸暂不支持，请固定导出：{shape}")
    size = int(shape[2])

    # 类别数：优先用调用方给的，否则从输出通道数反推（4+nc 或 5+nc，obb 多一个角度）
    out_meta = sess.get_outputs()[0]
    out_shape = [d if isinstance(d, int) else -1 for d in out_meta.shape]
    if num_classes is None:
        channels = None
        if len(out_shape) == 3:
            # 取较小的一维作为通道数（18900 远大于 4+nc）
            candidates = [d for d in out_shape[1:] if d > 0]
            if candidates:
                channels = min(candidates)
        if channels is None:
            raise FallbackError(f"无法从输出形状 {out_meta.shape} 推断类别数，请显式传入")
        extra = 1 if task == "obb" else 0
        num_classes = channels - 4 - extra
        if num_classes <= 0:
            raise FallbackError(f"推断出的类别数非正（{num_classes}），形状 {out_meta.shape}")

    letterboxed, ratio, pad_x, pad_y = _letterbox(image, size)
    import cv2

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    started = time.perf_counter()
    raw = sess.run(None, {input_meta.name: tensor})[0]
    infer_ms = (time.perf_counter() - started) * 1000

    rows = _normalize_layout(raw, num_classes)
    boxes_cxcywh = rows[:, :4]
    angle_idx = 4 + num_classes if task == "obb" else None
    angles = rows[:, angle_idx] if angle_idx is not None else None
    scores_all = rows[:, 4 : 4 + num_classes]
    class_ids = scores_all.argmax(axis=1)
    confidences = scores_all[np.arange(len(class_ids)), class_ids]

    keep_conf = confidences >= conf
    if not np.any(keep_conf):
        return _empty_yolo_result(task, image, infer_ms)

    boxes_cxcywh = boxes_cxcywh[keep_conf]
    class_ids = class_ids[keep_conf]
    confidences = confidences[keep_conf]
    if angles is not None:
        angles = angles[keep_conf]

    # cxcywh(像素) -> xyxy(像素)，再还原到原图坐标
    cx, cy, w, h = boxes_cxcywh.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio

    # 逐类 NMS。OBB 的旋转框这里按外接矩形做去重，与 DLL 的角度感知 NMS
    # 不完全等价，但足以在降级时给出可用的结果。
    keep: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        for k in _nms(xyxy[idx], confidences[idx], nms):
            keep.append(int(idx[k]))

    height, width = image.shape[:2]
    boxes = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        item = {
            "left": float(np.clip(x1, 0, width - 1)),
            "top": float(np.clip(y1, 0, height - 1)),
            "right": float(np.clip(x2, 0, width - 1)),
            "bottom": float(np.clip(y2, 0, height - 1)),
            "confidence": float(confidences[i]),
            "class_label": int(class_ids[i]),
            "has_mask": False,
            "is_obb": task == "obb",
        }
        if angles is not None:
            # DLL 用弧度；这里把 ONNX 的弧度原样透传，保持前端绘制一致
            item["obb"] = {
                "cx": float((cx[i] - pad_x) / ratio),
                "cy": float((cy[i] - pad_y) / ratio),
                "w": float(w[i] / ratio),
                "h": float(h[i] / ratio),
                "angle": float(angles[i]),
            }
        boxes.append(item)

    return {
        "backend": BACKEND_YOLO,
        "task": task,
        "image_width": width,
        "image_height": height,
        "count": len(boxes),
        "boxes": boxes,
        "timing": {
            # 降级路径不做 C++ 侧的分阶段计时，这里只给推理耗时，前/后处理记 0
            "preprocess_ms": 0.0,
            "inference_ms": float(infer_ms),
            "postprocess_ms": 0.0,
            "total_ms": float(infer_ms),
        },
    }


def _empty_yolo_result(task: str, image: np.ndarray, infer_ms: float) -> dict:
    height, width = image.shape[:2]
    return {
        "backend": BACKEND_YOLO,
        "task": task,
        "image_width": width,
        "image_height": height,
        "count": 0,
        "boxes": [],
        "timing": {
            "preprocess_ms": 0.0,
            "inference_ms": float(infer_ms),
            "postprocess_ms": 0.0,
            "total_ms": float(infer_ms),
        },
    }


# ---------------------------------------------------------------------------
# Anomalib
# ---------------------------------------------------------------------------
def _load_metadata(metadata_path: str) -> dict:
    path = Path(metadata_path)
    if not path.is_file():
        raise FallbackError(f"metadata.json 不存在：{metadata_path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FallbackError(f"metadata.json 解析失败：{exc}") from exc


def _normalize_score(value: float, meta: dict) -> float:
    """把原始分数映射到 0..1。

    `output_contract == anomalib_post_processed_0_1` 时 ONNX 输出已经归一化，直接用；
    否则（如 efficientad_distribution_calibrated）用 metadata 记录的 min/max 线性拉伸。
    """
    contract = str(meta.get("output_contract", "")).strip()
    if contract == "anomalib_post_processed_0_1":
        return float(np.clip(value, 0.0, 1.0))

    # 兼容两种写法：嵌套 pred_scores.min 与扁平 "pred_scores.min"
    lo = meta.get("pred_scores.min")
    hi = meta.get("pred_scores.max")
    if lo is None or hi is None:
        nested = meta.get("pred_scores") or {}
        lo = nested.get("min", 0.0)
        hi = nested.get("max", 1.0)
    lo, hi = float(lo), float(hi)
    if hi - lo <= 1e-12:
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def run_anomalib(
    onnx_path: str,
    metadata_path: str,
    image: np.ndarray,
    model_type: str = "Auto",
    want_overlay: bool = True,
) -> dict:
    """用 onnxruntime 跑 Anomalib ONNX，返回与 DLL 路径一致的结构。

    注意：这里接受的是 `.onnx`，而不是 `.engine`。TensorRT 引擎必须由 DLL 加载，
    onnxruntime 不能直接读 `.engine`（它不认识 TensorRT 的序列化格式）。
    """
    onnx_file = Path(onnx_path)
    if onnx_file.suffix.lower() != ".onnx":
        raise FallbackError(
            f"降级推理只能加载 .onnx，当前是 {onnx_file.suffix}。"
            "TensorRT 引擎（.engine/.trt）必须用 anomalib_infer.dll 加载。"
        )
    if not onnx_file.is_file():
        raise FallbackError(f"模型文件不存在：{onnx_path}")

    meta = _load_metadata(metadata_path)
    import cv2

    key = str(onnx_file.resolve())
    sess = _anom_sessions.get(key)
    if sess is None:
        ort = _import_ort()
        sess = ort.InferenceSession(key, providers=_providers(ort))
        _anom_sessions[key] = sess

    input_meta = sess.get_inputs()[0]
    shape = list(input_meta.shape)
    in_h = int(shape[2]) if isinstance(shape[2], int) else 256
    in_w = int(shape[3]) if isinstance(shape[3], int) else 256

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(
        (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    )

    started = time.perf_counter()
    outs = sess.run(None, {input_meta.name: tensor})
    infer_ms = (time.perf_counter() - started) * 1000

    named = {o.name: arr for o, arr in zip(sess.get_outputs(), outs)}
    score_arr = named.get("pred_score")
    if score_arr is None:
        raise FallbackError(f"模型输出缺少 pred_score，实际键：{list(named)}")
    # PatchCore 的 pred_score 是 [batch]，Padim/EfficientAd 是 [batch,1]
    raw_score = float(np.asarray(score_arr).reshape(-1)[0])

    map_arr = named.get("anomaly_map")
    map_2d = None
    if map_arr is not None:
        arr = np.asarray(map_arr)
        while arr.ndim > 2:
            arr = arr[0]
        map_2d = arr.astype(np.float32)

    normalized = _normalize_score(raw_score, meta)
    threshold = float(meta.get("image_threshold", 0.5))
    pixel_threshold = float(meta.get("pixel_threshold", threshold))
    is_anomaly = normalized >= threshold

    payload = {
        "backend": BACKEND_ANOMALIB,
        "raw_score": raw_score,
        "normalized_score": normalized,
        "is_anomaly": bool(is_anomaly),
        "metadata": {
            "model_type": str(meta.get("model_type", model_type)),
            "normalization": str(meta.get("normalization", "none")),
            "output_contract": str(meta.get("output_contract", "")),
            "image_threshold": threshold,
            "pixel_threshold": pixel_threshold,
            "input_h": in_h,
            "input_w": in_w,
        },
        "timing": {
            "preprocess_ms": 0.0,
            "inference_ms": float(infer_ms),
            "postprocess_ms": 0.0,
            "total_ms": float(infer_ms),
        },
        "overlay_png": None,
        "mask_png": None,
    }

    if want_overlay and map_2d is not None:
        # 归一化热力图，尺寸放大回原图，再与原图叠加
        lo = float(map_2d.min())
        hi = float(map_2d.max())
        norm = (map_2d - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(map_2d)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)

        # 掩码：按像素阈值二值化后放大回原图尺寸
        mask_small = (norm >= pixel_threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        payload["overlay_png"] = _encode_png(overlay)
        payload["mask_png"] = _encode_png(mask_rgb)

    return payload


def _encode_png(image: np.ndarray) -> bytes:
    """编码成 PNG **原始字节**。

    注意：这里必须返回 bytes 而非 base64 字符串。DLL 路径（infer.run_anomalib）
    返回的就是原始字节，由 app.py 统一做 base64。降级路径若提前编码，
    app.py 会二次编码直接报 `TypeError`。
    """
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise FallbackError("PNG 编码失败")
    return buffer.tobytes()


def reset_sessions() -> None:
    """释放缓存的 session（测试或模型切换时用）。"""
    _yolo_sessions.clear()
    _anom_sessions.clear()
