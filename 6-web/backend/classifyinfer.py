# -*- coding: utf-8 -*-
"""YOLO 整图分类推理。

检测动态库只认框，读不了「一整排类别分数」。分类这一版只接受 ONNX，
用 onnxruntime 取出分数。TensorRT 引擎留到后面再做。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ENGINE_SUFFIXES = {".engine", ".trt", ".plan"}
_SESSIONS: dict[str, object] = {}


class ClassifyError(Exception):
    """分类推理无法继续。调用方把它转成 400。"""


def reject_engine(path: str) -> None:
    """分类不能拿检测引擎来跑。"""
    suffix = Path(path).suffix.lower()
    if suffix in ENGINE_SUFFIXES:
        raise ClassifyError("分类引擎还没做，请改用 ONNX。")
    if suffix != ".onnx":
        raise ClassifyError(f"分类只接受 .onnx，当前是 {suffix or '无后缀'}。")


def as_probabilities(values: list[float]) -> list[float]:
    """输出已经接近概率就原样用，否则做 softmax。"""
    if not values:
        return []
    total = float(sum(values))
    if min(values) < -1e-6 or abs(total - 1.0) > 0.05:
        peak = max(values)
        exps = [math.exp(value - peak) for value in values]
        denom = sum(exps) or 1.0
        return [item / denom for item in exps]
    return [float(item) for item in values]


def interpret(scores: list[float], names: list[str], conf: float) -> dict:
    """把一排分数收成一个类别。类别数对不上就拒绝，不能当成检测框。"""
    probs = as_probabilities([float(item) for item in scores])
    if names and len(probs) != len(names):
        raise ClassifyError(
            f"模型输出 {len(probs)} 个分数，标签有 {len(names)} 个类别，对不上。"
        )
    if not names:
        names = [f"cls{index}" for index in range(len(probs))]
    if not probs:
        raise ClassifyError("模型没有输出类别分数。")
    best = max(range(len(probs)), key=lambda index: probs[index])
    return {
        "class_id": best,
        "label": names[best],
        "score": float(probs[best]),
        "uncertain": float(probs[best]) < float(conf),
        "scores": [
            {"id": index, "name": names[index], "score": float(probs[index])}
            for index in range(len(probs))
        ],
    }


def _session(path: str):
    key = str(Path(path).resolve())
    cached = _SESSIONS.get(key)
    if cached is not None:
        return cached
    reject_engine(path)
    if not Path(path).is_file():
        raise ClassifyError(f"ONNX 不存在：{path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ClassifyError("没有安装 onnxruntime，分类推理跑不了。") from exc
    session = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    _SESSIONS[key] = session
    return session


def _input_hw(session) -> tuple[str, int, int]:
    meta = session.get_inputs()[0]
    shape = list(meta.shape)
    # NCHW: [1, 3, H, W]。动态维用 224，和常见分类权重一致。
    height = 224
    width = 224
    if len(shape) >= 4:
        if isinstance(shape[-2], int) and shape[-2] > 0:
            height = int(shape[-2])
        if isinstance(shape[-1], int) and shape[-1] > 0:
            width = int(shape[-1])
    return str(meta.name), height, width


def _vector(output) -> list[float]:
    array = np.array(output, dtype=np.float32).reshape(-1)
    return [float(item) for item in array.tolist()]


def parse_names(raw) -> list[str]:
    """把模型 metadata 里的 ``names`` 原文解析成有序类别名。

    Ultralytics 导出的是 Python dict 字面量（``{0: 'good', 1: 'bad'}``），
    也有版本写 list。两种都要认，解析不了就返回空表、交给调用方回退标签文件。

    抽成公共函数的原因：DLL 路径从 C 侧拿的是同一份原文
    （`oi_classify_result.names_utf8`），如果这里再写一份解析，
    两条路径在"dict 键是字符串"这种细节上就会分叉。
    """
    if not raw:
        return []
    parsed = None
    try:
        import ast

        parsed = ast.literal_eval(str(raw))
    except Exception:  # noqa: BLE001 - 老模型或非常规导出没有 metadata
        return []
    if isinstance(parsed, dict):
        pairs = []
        for key, value in parsed.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            pairs.append((index, str(value)))
        return [name for _, name in sorted(pairs, key=lambda item: item[0])]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []


def _model_names(session) -> list[str]:
    """从 ONNX metadata 读训练时的 names，顺序就是模型输出的顺序。

    Ultralytics 导出会把 ``names`` 写进 metadata（字符串形式的 dict）。
    这一步很重要：训练时类别名是 ``sorted(文件夹名)`` 排出来的，中文按 Unicode
    排序（不良 U+4E0D 在 良品 U+826F 前），很可能和 dataset.yaml 里写的 0/1 相反。
    以模型自带的顺序为准，才能保证「分数最高那一维」对得上正确的类别名。
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
    except Exception:  # noqa: BLE001 - 老模型或非常规导出没有 metadata
        return []
    return parse_names(meta.get("names"))


def preprocess(image, height: int, width: int):
    """按 Ultralytics 分类推理的方式缩放：短边等比缩放到目标边长，再中心裁剪。

    训练/验证用的是 ``classify_transforms``：``Resize(size)`` 把短边缩到 size
    （长宽比不变），``CenterCrop(size)`` 从中间裁出 size×size。若直接
    ``cv2.resize`` 到 W×H，非正方形的图会被横向或纵向压扁，物体比例和训练
    时不一样，置信度会偏低甚至被判到别的类。图越接近正方形两者差别越小。

    归一化只做 ÷255：本机 Ultralytics 分类的 ``DEFAULT_MEAN=(0,0,0)``、
    ``DEFAULT_STD=(1,1,1)``，即不做减均值除方差。
    """
    import cv2

    size_height = max(int(height), 1)
    size_width = max(int(width), 1)
    src_height, src_width = image.shape[:2]
    if src_height <= 0 or src_width <= 0:
        raise ClassifyError("图片尺寸是 0，没法分类。")

    # 非正方形输入：先等比缩放，再中心裁；正方形输入：直接缩放到目标尺寸。
    if size_height == size_width:
        target = size_height
        short = min(src_height, src_width)
        if short != target:
            scale = target / float(short)
            new_width = max(int(src_width * scale), target)
            new_height = max(int(src_height * scale), target)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image
        top = int(round((resized.shape[0] - target) / 2.0))
        left = int(round((resized.shape[1] - target) / 2.0))
        cropped = resized[top : top + target, left : left + target]
    else:
        cropped = cv2.resize(image, (size_width, size_height), interpolation=cv2.INTER_LINEAR)
    if cropped.size == 0:
        raise ClassifyError("缩放后没有可用像素，分类中止。")

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def run(model_path: str, image, names: list[str], conf: float = 0.25) -> dict:
    """对一张 BGR 图做分类。返回里 boxes 为空，结果在 classify。"""
    session = _session(model_path)
    name, height, width = _input_hw(session)
    blob = preprocess(image, height, width)
    model_names = _model_names(session)
    warning = ""
    if model_names:
        # 以模型自带的顺序为准；和传入的标签文件不一致时提示一句，
        # 但不静默改用户的标签文件。
        if names and list(names) != model_names:
            warning = "标签文件顺序与模型不一致，已按模型自带的类别顺序显示"
        names = model_names
    raw = session.run(None, {name: blob})
    verdict = interpret(_vector(raw[0]), list(names or []), conf)
    result = {
        "task": "classify",
        "backend": "onnxruntime",
        "boxes": [],
        "count": 0,
        "classify": verdict,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
    if warning:
        result["warning"] = warning
    return result


def result_from_raw(raw: dict, classes: list[str] | None, conf: float) -> tuple[dict, str]:
    """把 onnx_infer.dll 返回的原始分数组装成与 `run()` 完全一致的结构。

    为什么要有这个函数：网页单张、目录批量、视频三条入口都要把
    `{scores, names, timing}` 变成 `{classify, backend, ...}`。三处各写一遍
    组装逻辑，字段迟早会漂（一边有 timing 一边没有），前端就得按入口写分支。

    返回 `(result, warning)`：warning 非空表示标签顺序与模型不一致，
    交给调用方合并进最终响应的 `warning` 字段。
    """
    names = parse_names(raw.get("names") or "")
    warning = ""
    if names:
        # 与 `run()` 同一口径：模型自带顺序优先，和标签文件不一致时提示一句。
        if classes and list(classes) != names:
            warning = "标签文件顺序与模型不一致，已按模型自带的类别顺序显示"
        final_names = names
    else:
        final_names = list(classes or [])

    verdict = interpret(raw.get("scores") or [], final_names, conf)
    result = {
        "task": "classify",
        "backend": raw.get("backend") or "onnx_infer.dll",
        "boxes": [],
        "count": 0,
        "classify": verdict,
        "image_width": int(raw.get("image_width") or 0),
        "image_height": int(raw.get("image_height") or 0),
        "timing": raw.get("timing") or {},
    }
    return result, warning


def draw_label(image, verdict: dict):
    """在图上写类别和分数，不画框。返回 BGR 图。"""
    import cv2

    canvas = image.copy()
    text = f"{verdict.get('label', '')} {float(verdict.get('score', 0)):.2f}"
    if verdict.get("uncertain"):
        text += " 不确定"
    cv2.rectangle(canvas, (8, 8), (8 + 18 * len(text), 48), (12, 14, 18), thickness=-1)
    cv2.putText(canvas, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def overlay_png(image, verdict: dict) -> bytes:
    """画上类别名，编码成 PNG 字节。"""
    import cv2

    ok, encoded = cv2.imencode(".png", draw_label(image, verdict))
    if not ok:
        raise ClassifyError("结果图画不出来。")
    return encoded.tobytes()
