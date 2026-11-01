# -*- coding: utf-8 -*-
"""数据增强预览：拿训练集里的真实样本，按用户填的参数合成一张增强后的图。

**这是像素级近似，不是训练时的真实 pipeline。** 原因写在
``openspec/changes/add-training-console/design.md`` 里，简言之：

- 训练时的增强顺序与随机状态由 dataloader 决定，本来就不可能逐像素复现；
- 真实 pipeline 需要构造完整的 ``Dataset`` 对象，为了一张预览图不值当；
- 后端解释器没有 torch，能用的只有 ``cv2`` + ``numpy``。

因此这里手写参数可见的那部分变换，顺序尽量对齐 Ultralytics 的
``v8_transforms``：mosaic → 几何（旋转/缩放/平移/错切/透视）→ HSV → 上下翻 → 左右翻。

两个参数**不参与预览**，并在响应里明确说明：

- ``copy_paste``：只有分割任务有掩码，检测任务下它本就不生效；
- ``close_mosaic``：它是「最后 N 轮关掉 mosaic」的调度参数，不是像素变换。
"""

from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import traininfo

#: 与 ``1-python-yolo/launcher/tf_yolo_launcher.py`` 的 argparse 默认值保持一致。
#: 表单留空时用这里的默认值渲染，保证「不填参数」的预览等于「不填参数」的训练。
DEFAULTS: dict[str, float] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

#: 概率参数（0~1 之间按概率触发）；其余参数是「幅度」，每次都会施加。
PROBABILITY_KEYS = ("fliplr", "flipud", "bgr", "mosaic", "mixup", "copy_paste")

#: 纯幅度参数的合理上限，超出直接报错——避免用户填 500 度把图转成空白。
LIMITS: dict[str, tuple[float, float]] = {
    "hsv_h": (0.0, 1.0),
    "hsv_s": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "degrees": (-180.0, 180.0),
    "translate": (0.0, 1.0),
    "scale": (0.0, 1.0),
    "shear": (-90.0, 90.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "bgr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
}

PAD_VALUE = 114


class PreviewError(Exception):
    """预览无法生成（样本目录不存在、图片读不出等）。"""


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def normalize_params(raw: dict[str, Any] | None) -> tuple[dict[str, float], list[str]]:
    """把表单值转成 float，留空/非数字回落默认值。

    返回 ``(参数, 被回落的键)``；回落项会在响应里如实告知，避免用户以为
    「填了但没生效」。
    """
    normalized: dict[str, float] = {}
    fell_back: list[str] = []
    for key, default in DEFAULTS.items():
        value = None if raw is None else raw.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            normalized[key] = default
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            normalized[key] = default
            fell_back.append(key)
            continue
        low, high = LIMITS[key]
        if not low <= number <= high:
            raise PreviewError(f"{key} 超出允许范围 {low}~{high}，当前填的是 {number}")
        normalized[key] = number
    return normalized, fell_back


# ---------------------------------------------------------------------------
# 基础图像操作
# ---------------------------------------------------------------------------
def read_image(path: Path) -> np.ndarray:
    """读图。用 ``np.fromfile`` 而不是 ``cv2.imread``：后者在中文路径下返回 None。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise PreviewError(f"读取图片失败：{path}（{exc}）") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise PreviewError(f"图片无法解码：{path}")
    return image


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """等比缩放并灰边填充成**正好** ``size × size``。

    训练时 Ultralytics 的 ``LetterBox(new_shape=(imgsz, imgsz))`` 输出就是正方形，
    所以预览也必须是正方形——否则「原图/增强后」两张图尺寸不同，无法并排比较。
    """
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    new_w = max(1, min(size, int(round(width * scale))))
    new_h = max(1, min(size, int(round(height * scale))))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = size - new_w
    pad_h = size - new_h
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )


def _affine(image: np.ndarray, params: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """把 degrees / translate / scale / shear / perspective 合成一次透视变换。

    与 Ultralytics 的 ``RandomPerspective.affine_transform`` 同构：以画面中心为
    旋转缩放原点，最后叠加随机透视系数，越界部分填 114 灰。
    """
    height, width = image.shape[:2]
    center = np.array([[1.0, 0.0, width / 2], [0.0, 1.0, height / 2], [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(center)

    angle = float(rng.uniform(-params["degrees"], params["degrees"]))
    scale = float(rng.uniform(1 - params["scale"], 1 + params["scale"]))
    shear = float(rng.uniform(-params["shear"], params["shear"]))
    radians = np.radians(angle)
    cos, sin = np.cos(radians), np.sin(radians)

    rotation = np.array(
        [[cos * scale, sin * scale, 0.0], [-sin * scale, cos * scale, 0.0], [0.0, 0.0, 1.0]]
    )
    shear_matrix = np.array(
        [[1.0, np.tan(np.radians(shear)), 0.0], [np.tan(np.radians(shear)), 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    matrix = inverse @ rotation @ shear_matrix @ center
    matrix[0, 2] += float(rng.uniform(-params["translate"], params["translate"])) * width
    matrix[1, 2] += float(rng.uniform(-params["translate"], params["translate"])) * height

    if params["perspective"] > 0:
        perspective = np.eye(3)
        perspective[2, 0] = float(rng.uniform(-params["perspective"], params["perspective"]))
        perspective[2, 1] = float(rng.uniform(-params["perspective"], params["perspective"]))
        matrix = perspective @ matrix

    return cv2.warpPerspective(
        image, matrix, dsize=(width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )


def _hsv(image: np.ndarray, params: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """HSV 抖动。OpenCV 的 H 通道是 0~179，所以色调幅度按半圈算。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1, 1) * params["hsv_h"] * 90) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.uniform(-1, 1) * params["hsv_s"] * 255, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.uniform(-1, 1) * params["hsv_v"] * 255, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _mosaic(primary: np.ndarray, others: list[np.ndarray], size: int, rng: np.random.Generator) -> np.ndarray:
    """2×2 拼接。中心点在中三分之一的范围内随机，与 Ultralytics 的 mosaic 直觉一致。"""
    canvas = np.full((size * 2, size * 2, 3), PAD_VALUE, dtype=np.uint8)

    # 中心点决定四张图各自露多少，落在中三分之一避免出现细条
    center_x = int(size + rng.uniform(0.0, 1.0) * size)
    center_y = int(size + rng.uniform(0.0, 1.0) * size)
    quadrants = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
    for image, (dx, dy) in zip([primary] + others[:3], quadrants):
        patch = letterbox(image, size)
        x1, y1 = center_x + dx * size, center_y + dy * size
        x2, y2 = x1 + size, y1 + size
        sx1, sy1 = max(0, -x1), max(0, -y1)
        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(size * 2, x2), min(size * 2, y2)
        if dx2 <= dx1 or dy2 <= dy1:
            continue
        canvas[dy1:dy2, dx1:dx2] = patch[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def _mixup(image: np.ndarray, other: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """半透明叠加。Ultralytics 用 Beta(32,32) 取权重，这里用同形状的均匀分布。"""
    weight = float(rng.uniform(0.5, 1.0))
    blended = image.astype(np.float32) * weight + other.astype(np.float32) * (1 - weight)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _pick_samples(images: list[Path], index: int, rng: np.random.Generator, extra: int) -> list[Path]:
    """取主样本 + ``extra`` 个随机伙伴（mosaic / mixup 要用）。"""
    primary = images[index % len(images)]
    if extra <= 0:
        return [primary]
    pool = [p for p in images if p != primary] or [primary]
    picks = [pool[int(rng.integers(0, len(pool)))] for _ in range(extra)]
    return [primary] + picks


def render(
    data_yaml: str | Path,
    raw_params: dict[str, Any] | None,
    *,
    index: int = 0,
    seed: int | None = None,
    imgsz: int = 640,
    split: str = "train",
) -> dict:
    """生成一张预览，返回原图与增强图（base64 JPEG）以及本次触发了哪些增强。"""
    images = traininfo.list_split_images(data_yaml, split)
    if not images:
        raise traininfo.OverviewError(f"{split} 里没有可用图片，无法生成预览")

    params, fell_back = normalize_params(raw_params)
    actual_seed = int(seed) if seed is not None else random.randrange(0, 2 ** 31 - 1)
    rng = np.random.default_rng(actual_seed)

    size = max(64, int(imgsz))
    index = index % len(images)

    # mosaic / mixup 各要多一张图，先定下这次到底命中哪几个，再抽样本：
    # 顺序固定（先 mosaic 再 mixup），同一个 seed 才能稳定复现。
    mosaic_hit = params["mosaic"] > 0 and rng.random() < params["mosaic"]
    mixup_hit = params["mixup"] > 0 and rng.random() < params["mixup"]
    extra = 3 if mosaic_hit else (1 if mixup_hit else 0)
    picked = _pick_samples(images, index, rng, extra)
    primary_path = picked[0]
    primary = read_image(primary_path)

    applied: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    if mosaic_hit and len(picked) > 1:
        others = [read_image(p) for p in picked[1:4]]
        work = _mosaic(primary, others, size, rng)
        applied.append({"name": "mosaic", "detail": f"p={params['mosaic']:g} 命中，与另外 {len(others)} 张拼成 2×2"})
    else:
        work = letterbox(primary, size)
        skipped.append(
            {"name": "mosaic", "detail": "参数为 0" if params["mosaic"] <= 0 else f"p={params['mosaic']:g} 未命中"}
        )

    if mixup_hit and len(picked) > 1:
        work = _mixup(work, letterbox(read_image(picked[1]), size), rng)
        applied.append({"name": "mixup", "detail": f"p={params['mixup']:g} 命中，与另一张混合"})
    else:
        skipped.append(
            {"name": "mixup", "detail": "参数为 0" if params["mixup"] <= 0 else f"p={params['mixup']:g} 未命中"}
        )

    if any(params[key] > 0 for key in ("degrees", "translate", "scale", "shear", "perspective")):
        work = _affine(work, params, rng)
        detail = (
            f"旋转 ±{params['degrees']:g}° / 平移 {params['translate']:g} / "
            f"缩放 {params['scale']:g} / 错切 ±{params['shear']:g}° / 透视 {params['perspective']:g}"
        )
        applied.append({"name": "geometry", "detail": detail})
    else:
        skipped.append({"name": "geometry", "detail": "几何参数全为 0"})

    if any(params[key] > 0 for key in ("hsv_h", "hsv_s", "hsv_v")):
        work = _hsv(work, params, rng)
        applied.append(
            {
                "name": "hsv",
                "detail": f"h ±{params['hsv_h']:g} / s ±{params['hsv_s']:g} / v ±{params['hsv_v']:g}",
            }
        )
    else:
        skipped.append({"name": "hsv", "detail": "hsv 参数全为 0"})

    for key, label in (("flipud", "上下翻转"), ("fliplr", "左右翻转"), ("bgr", "R/B 通道互换")):
        if params[key] > 0 and rng.random() < params[key]:
            if key == "flipud":
                work = cv2.flip(work, 0)
            elif key == "fliplr":
                work = cv2.flip(work, 1)
            else:
                work = work[:, :, ::-1].copy()
            applied.append({"name": key, "detail": f"{label}，p={params[key]:g} 命中"})
        elif params[key] > 0:
            skipped.append({"name": key, "detail": f"{label}，p={params[key]:g} 未命中"})

    origin = letterbox(primary, size)
    return {
        "source": str(primary_path),
        "index": index,
        "total": len(images),
        "seed": actual_seed,
        "imgsz": size,
        "origin": _encode(origin),
        "augmented": _encode(work),
        "applied": applied,
        "skipped": skipped,
        "params": params,
        "fell_back": fell_back,
        # 这两个参数不参与像素变换，界面要照实说明，避免用户以为预览漏了东西
        "ignored": [
            {"name": "copy_paste", "detail": "只在分割任务下生效（需要掩码），预览不展示"},
            {"name": "close_mosaic", "detail": "控制「最后 N 轮关闭 mosaic」的调度，不是像素变换"},
        ],
    }


def _encode(image: np.ndarray) -> str:
    """编码成 data URI。单张 640 灰度 JPEG 约 40~90 KB，直接内联比落盘干净。"""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise PreviewError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
