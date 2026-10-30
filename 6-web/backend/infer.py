# -*- coding: utf-8 -*-
"""推理适配器：用 ctypes 调用现有的 yolo_infer.dll / anomalib_infer.dll。

为什么不用 onnxruntime 重新实现：
封装好的 DLL 里已经有前处理、TensorRT 推理、解码、NMS、坐标还原、
异常热力图生成，重写一遍既容易出错又没意义。这里只负责「搬数据 + 转结构」。

关键约束（踩过的坑）：
1. Python 3.8+ 不再用 PATH 解析依赖 DLL，必须先 os.add_dll_directory
   把 OpenCV / CUDA 的 bin 目录加进去，否则会报「找不到模块或它的依赖项」。
2. C API 返回的指针由 DLL 句柄持有，只在下一次推理或 destroy 前有效，
   所以所有数组必须在函数内立即深拷贝成 Python 对象。
3. 一个句柄不能被多线程同时调用，本模块用锁串行化。
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import BIN_DIR, DLL_DIRS

# ---------------------------------------------------------------------------
# 结构体定义（必须与 C 头文件逐字段对应）
# ---------------------------------------------------------------------------
class YiBox(ctypes.Structure):
    """对应 yolo_infer_c.h 的 yi_box。"""

    _fields_ = [
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("right", ctypes.c_float),
        ("bottom", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_label", ctypes.c_int),
        ("has_mask", ctypes.c_int),
        ("mask_x", ctypes.c_int),
        ("mask_y", ctypes.c_int),
        ("mask_w", ctypes.c_int),
        ("mask_h", ctypes.c_int),
        ("mask_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("num_kpts", ctypes.c_int),
        ("kpts", ctypes.POINTER(ctypes.c_float)),
        ("is_obb", ctypes.c_int),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("ow", ctypes.c_float),
        ("oh", ctypes.c_float),
        ("angle", ctypes.c_float),
    ]


class YiFrameStat(ctypes.Structure):
    """对应 yolo_infer_c.h 的 yi_frame_stat。"""

    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("count", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


class AiMetadata(ctypes.Structure):
    """对应 anomalib_infer_c.h 的 ai_metadata。"""

    _fields_ = [
        ("model_type", ctypes.c_char * 64),
        ("normalization", ctypes.c_char * 32),
        ("output_contract", ctypes.c_char * 64),
        ("image_threshold", ctypes.c_float),
        ("pixel_threshold", ctypes.c_float),
        ("score_min", ctypes.c_float),
        ("score_max", ctypes.c_float),
        ("map_min", ctypes.c_float),
        ("map_max", ctypes.c_float),
        ("input_h", ctypes.c_int),
        ("input_w", ctypes.c_int),
    ]


class AiResult(ctypes.Structure):
    """对应 anomalib_infer_c.h 的 ai_result。"""

    _fields_ = [
        ("source_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("source_w", ctypes.c_int),
        ("source_h", ctypes.c_int),
        ("source_stride", ctypes.c_int),
        ("overlay_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("overlay_w", ctypes.c_int),
        ("overlay_h", ctypes.c_int),
        ("overlay_stride", ctypes.c_int),
        ("mask_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("mask_w", ctypes.c_int),
        ("mask_h", ctypes.c_int),
        ("mask_stride", ctypes.c_int),
        ("normalized_map", ctypes.POINTER(ctypes.c_float)),
        ("map_w", ctypes.c_int),
        ("map_h", ctypes.c_int),
        ("raw_score", ctypes.c_float),
        ("normalized_score", ctypes.c_float),
        ("is_anomaly", ctypes.c_int),
    ]


class AiFrameStat(ctypes.Structure):
    """对应 anomalib_infer_c.h 的 ai_frame_stat。"""

    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


# 任务枚举，与 yi_task 一致
TASK_DETECT = 0
TASK_SEGMENT = 1
TASK_POSE = 2
TASK_OBB = 3

TASK_NAMES = {
    "detect": TASK_DETECT,
    "segment": TASK_SEGMENT,
    "pose": TASK_POSE,
    "obb": TASK_OBB,
}


class InferError(RuntimeError):
    """推理相关错误，带可读原因。"""


@dataclass
class BackendStatus:
    """当前可用的推理后端。"""

    yolo_ok: bool
    anomalib_ok: bool
    dll_dirs: list[str]
    detail: str = ""


# ---------------------------------------------------------------------------
# DLL 加载
# ---------------------------------------------------------------------------
_load_lock = threading.Lock()
_yolo_lib: ctypes.WinDLL | None = None
_anomalib_lib: ctypes.WinDLL | None = None
_load_error: str = ""
_dll_handles: list = []


def _prepare_dll_search_path() -> None:
    """把 OpenCV / CUDA 的 bin 目录加入 DLL 搜索路径。

    Python 3.8+ 的 Windows 加载器默认不看 PATH，这是必须的一步。
    添加的句柄要保留引用，否则目录会被移出搜索路径。
    """
    if not hasattr(os, "add_dll_directory"):
        return
    existing = set()
    for handle in _dll_handles:
        existing.add(handle)
    for directory in DLL_DIRS:
        try:
            _dll_handles.append(os.add_dll_directory(str(directory)))
        except OSError:
            # 目录不存在或无权限时跳过，后续加载会给出更明确的错误
            continue


def _bind_yolo(lib: ctypes.WinDLL) -> None:
    """声明 yolo_infer 的函数签名。"""
    lib.yi_create.restype = ctypes.c_void_p
    lib.yi_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]

    lib.yi_infer_bgr.restype = ctypes.c_int
    lib.yi_infer_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(YiBox)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(YiFrameStat),
    ]

    lib.yi_destroy.restype = None
    lib.yi_destroy.argtypes = [ctypes.c_void_p]

    lib.yi_last_error.restype = ctypes.c_char_p
    lib.yi_last_error.argtypes = []


def _bind_anomalib(lib: ctypes.WinDLL) -> None:
    """声明 anomalib_infer 的函数签名。"""
    lib.ai_create.restype = ctypes.c_void_p
    lib.ai_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_float,
    ]

    lib.ai_get_metadata.restype = ctypes.c_int
    lib.ai_get_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(AiMetadata)]

    lib.ai_infer_rgb.restype = ctypes.c_int
    lib.ai_infer_rgb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(AiResult)),
        ctypes.POINTER(AiFrameStat),
    ]

    lib.ai_destroy.restype = None
    lib.ai_destroy.argtypes = [ctypes.c_void_p]

    lib.ai_last_error.restype = ctypes.c_char_p
    lib.ai_last_error.argtypes = []


def load_libraries() -> BackendStatus:
    """加载两个推理库，返回可用状态。

    加载失败不抛异常：Web 服务必须能起来，把问题在页面上说明。
    """
    global _yolo_lib, _anomalib_lib, _load_error

    with _load_lock:
        _prepare_dll_search_path()

        yolo_path = BIN_DIR / "yolo_infer.dll"
        anom_path = BIN_DIR / "anomalib_infer.dll"

        errors: list[str] = []

        if _yolo_lib is None:
            if yolo_path.is_file():
                try:
                    lib = ctypes.WinDLL(str(yolo_path))
                    _bind_yolo(lib)
                    _yolo_lib = lib
                except OSError as exc:
                    errors.append(f"yolo_infer.dll 加载失败：{exc}")
            else:
                errors.append(f"缺少文件：{yolo_path}")

        if _anomalib_lib is None:
            if anom_path.is_file():
                try:
                    lib = ctypes.WinDLL(str(anom_path))
                    _bind_anomalib(lib)
                    _anomalib_lib = lib
                except OSError as exc:
                    errors.append(f"anomalib_infer.dll 加载失败：{exc}")
            else:
                errors.append(f"缺少文件：{anom_path}")

        _load_error = "；".join(errors)
        return BackendStatus(
            yolo_ok=_yolo_lib is not None,
            anomalib_ok=_anomalib_lib is not None,
            dll_dirs=[str(d) for d in DLL_DIRS],
            detail=_load_error,
        )


def status() -> BackendStatus:
    """查询当前状态（未加载则先尝试加载）。"""
    if _yolo_lib is None and _anomalib_lib is None:
        return load_libraries()
    return BackendStatus(
        yolo_ok=_yolo_lib is not None,
        anomalib_ok=_anomalib_lib is not None,
        dll_dirs=[str(d) for d in DLL_DIRS],
        detail=_load_error,
    )


# ---------------------------------------------------------------------------
# 图像编解码
# ---------------------------------------------------------------------------
def decode_image(raw: bytes) -> np.ndarray:
    """把上传的图片字节解码成 BGR uint8 数组。

    用 cv2 解码，兼容 jpg/png/bmp；失败时给出明确错误。
    """
    import cv2

    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InferError("无法解析图片，请上传 jpg / png / bmp")
    return np.ascontiguousarray(image)


def _encode_png(image: np.ndarray) -> bytes:
    """把 BGR 数组编码成 PNG 字节。"""
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise InferError("结果图编码失败")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# YOLO 推理
# ---------------------------------------------------------------------------
_yolo_lock = threading.Lock()
_yolo_handles: dict[tuple[str, str, float, float], ctypes.c_void_p] = {}


def _yolo_handle(engine_path: str, task: str, conf: float, nms: float) -> ctypes.c_void_p:
    """取（或创建）YOLO 句柄。同参数复用，避免重复加载引擎。"""
    key = (os.path.abspath(engine_path), task, round(conf, 4), round(nms, 4))
    handle = _yolo_handles.get(key)
    if handle:
        return handle

    lib = _yolo_lib
    if lib is None:
        raise InferError(f"yolo_infer.dll 未加载：{_load_error or '未知原因'}")

    task_id = TASK_NAMES.get(task)
    if task_id is None:
        raise InferError(f"不支持的任务类型：{task}，可选 {', '.join(TASK_NAMES)}")

    engine_file = Path(engine_path)
    if not engine_file.is_file():
        raise InferError(f"引擎文件不存在：{engine_path}")

    handle = lib.yi_create(str(engine_file).encode("utf-8"), task_id, conf, nms)
    if not handle:
        raise InferError(_lib_error(lib, "yi_create"))

    _yolo_handles[key] = handle
    return handle


def _lib_error(lib, func_name: str) -> str:
    """读取 last_error 并格式化成可读信息。"""
    try:
        raw = lib.yi_last_error() if func_name.startswith("yi_") else lib.ai_last_error()
        message = raw.decode("utf-8", errors="replace") if raw else ""
    except Exception:  # noqa: BLE001
        message = ""
    return f"{func_name} 失败：{message or '（DLL 未返回错误信息）'}"


def run_yolo(
    engine_path: str,
    task: str,
    image: np.ndarray,
    conf: float = 0.25,
    nms: float = 0.45,
) -> dict:
    """执行一次 YOLO 推理，返回可 JSON 序列化的结果。"""
    lib = _yolo_lib
    if lib is None:
        raise InferError(f"yolo_infer.dll 未加载：{_load_error or '未知原因'}")

    height, width = image.shape[:2]
    stride = int(image.strides[0])
    buffer = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    with _yolo_lock:
        handle = _yolo_handle(engine_path, task, conf, nms)
        boxes_ptr = ctypes.POINTER(YiBox)()
        count = ctypes.c_int(0)
        stat = YiFrameStat()

        code = lib.yi_infer_bgr(
            handle, buffer, width, height, stride,
            ctypes.byref(boxes_ptr), ctypes.byref(count), ctypes.byref(stat),
        )
        if code != 0:
            raise InferError(_lib_error(lib, "yi_infer_bgr"))

        # 立即深拷贝：指针只在下一次推理前有效
        boxes = []
        for i in range(count.value):
            box = boxes_ptr[i]
            item = {
                "left": float(box.left),
                "top": float(box.top),
                "right": float(box.right),
                "bottom": float(box.bottom),
                "confidence": float(box.confidence),
                "class_label": int(box.class_label),
                "has_mask": bool(box.has_mask),
                "is_obb": bool(box.is_obb),
            }
            if box.is_obb:
                item["obb"] = {
                    "cx": float(box.cx),
                    "cy": float(box.cy),
                    "w": float(box.ow),
                    "h": float(box.oh),
                    "angle": float(box.angle),
                }
            if box.num_kpts > 0 and box.kpts:
                item["kpts"] = [
                    [float(box.kpts[j * 3]), float(box.kpts[j * 3 + 1]), float(box.kpts[j * 3 + 2])]
                    for j in range(box.num_kpts)
                ]
            boxes.append(item)

        timing = {
            "preprocess_ms": float(stat.preprocess_ms),
            "inference_ms": float(stat.inference_ms),
            "postprocess_ms": float(stat.postprocess_ms),
            "total_ms": float(stat.total_ms),
        }

    return {
        "backend": "yolo_infer.dll",
        "task": task,
        "image_width": width,
        "image_height": height,
        "count": len(boxes),
        "boxes": boxes,
        "timing": timing,
    }


def draw_yolo(image: np.ndarray, boxes: list[dict]) -> bytes:
    """把检测框画到图上，返回 PNG 字节。"""
    import cv2

    canvas = image.copy()
    thickness = max(2, int(round(min(canvas.shape[:2]) / 320)))
    font_scale = max(0.5, min(canvas.shape[:2]) / 900)

    for item in boxes:
        if item.get("is_obb"):
            obb = item["obb"]
            rect = ((obb["cx"], obb["cy"]), (obb["w"], obb["h"]), obb["angle"] * 180.0 / 3.141592653589793)
            points = cv2.boxPoints(rect).astype(int)
            cv2.polylines(canvas, [points], True, (0, 220, 120), thickness)
            origin = tuple(points[0])
        else:
            x1, y1 = int(item["left"]), int(item["top"])
            x2, y2 = int(item["right"]), int(item["bottom"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 200, 255), thickness)
            origin = (x1, max(0, y1 - 6))

        label = f"cls{item['class_label']} {item['confidence']:.2f}"
        cv2.putText(
            canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA,
        )

        for kpt in item.get("kpts", []):
            if len(kpt) >= 2:
                cv2.circle(canvas, (int(kpt[0]), int(kpt[1])), thickness + 1, (0, 255, 255), -1)

    return _encode_png(canvas)


# ---------------------------------------------------------------------------
# Anomalib 推理
# ---------------------------------------------------------------------------
_anom_lock = threading.Lock()
_anom_handles: dict[tuple[str, str, str], ctypes.c_void_p] = {}


def _anom_handle(engine_path: str, metadata_path: str, model_type: str) -> ctypes.c_void_p:
    """取（或创建）Anomalib 句柄。"""
    key = (os.path.abspath(engine_path), os.path.abspath(metadata_path), model_type)
    handle = _anom_handles.get(key)
    if handle:
        return handle

    lib = _anomalib_lib
    if lib is None:
        raise InferError(f"anomalib_infer.dll 未加载：{_load_error or '未知原因'}")

    engine_file = Path(engine_path)
    metadata_file = Path(metadata_path)
    if not engine_file.is_file():
        raise InferError(f"引擎文件不存在：{engine_path}")
    if not metadata_file.is_file():
        raise InferError(f"metadata.json 不存在：{metadata_path}")

    # 阈值 override 传负数表示沿用 metadata
    handle = lib.ai_create(
        str(engine_file).encode("utf-8"),
        str(metadata_file).encode("utf-8"),
        model_type.encode("utf-8"),
        ctypes.c_float(-1.0),
        ctypes.c_float(-1.0),
    )
    if not handle:
        raise InferError(_lib_error(lib, "ai_create"))

    _anom_handles[key] = handle
    return handle


def run_anomalib(
    engine_path: str,
    metadata_path: str,
    image: np.ndarray,
    model_type: str = "Auto",
    want_overlay: bool = True,
) -> dict:
    """执行一次 Anomalib 推理，返回分数、判定与叠加图。

    输入要求 RGB888，这里从 BGR 转换。
    """
    lib = _anomalib_lib
    if lib is None:
        raise InferError(f"anomalib_infer.dll 未加载：{_load_error or '未知原因'}")

    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    stride = int(rgb.strides[0])
    buffer = rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    with _anom_lock:
        handle = _anom_handle(engine_path, metadata_path, model_type)

        metadata = AiMetadata()
        if lib.ai_get_metadata(handle, ctypes.byref(metadata)) == 0:
            meta_dict = {
                "model_type": metadata.model_type.decode("utf-8", errors="replace"),
                "normalization": metadata.normalization.decode("utf-8", errors="replace"),
                "output_contract": metadata.output_contract.decode("utf-8", errors="replace"),
                "image_threshold": float(metadata.image_threshold),
                "pixel_threshold": float(metadata.pixel_threshold),
                "input_h": int(metadata.input_h),
                "input_w": int(metadata.input_w),
            }
        else:
            meta_dict = {}

        result_ptr = ctypes.POINTER(AiResult)()
        stat = AiFrameStat()
        code = lib.ai_infer_rgb(
            handle, buffer, width, height, stride,
            ctypes.byref(result_ptr), ctypes.byref(stat),
        )
        if code != 0:
            raise InferError(_lib_error(lib, "ai_infer_rgb"))

        result = result_ptr.contents
        payload = {
            "backend": "anomalib_infer.dll",
            "raw_score": float(result.raw_score),
            "normalized_score": float(result.normalized_score),
            "is_anomaly": bool(result.is_anomaly),
            "metadata": meta_dict,
            "timing": {
                "preprocess_ms": float(stat.preprocess_ms),
                "inference_ms": float(stat.inference_ms),
                "postprocess_ms": float(stat.postprocess_ms),
                "total_ms": float(stat.total_ms),
            },
            "overlay_png": None,
            "mask_png": None,
        }

        # 叠加图与轮廓图都是 DLL 持有的指针，必须立刻拷贝成 PNG
        if want_overlay and result.overlay_rgb and result.overlay_w > 0 and result.overlay_h > 0:
            overlay = _pointer_to_array(
                result.overlay_rgb, result.overlay_h, result.overlay_w, result.overlay_stride
            )
            payload["overlay_png"] = _encode_png(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if want_overlay and result.mask_rgb and result.mask_w > 0 and result.mask_h > 0:
            mask = _pointer_to_array(
                result.mask_rgb, result.mask_h, result.mask_w, result.mask_stride
            )
            payload["mask_png"] = _encode_png(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    return payload


def _pointer_to_array(pointer, height: int, width: int, stride: int) -> np.ndarray:
    """把 DLL 返回的 RGB 指针拷成 numpy 数组（去行对齐，转 BGR 由调用方做）。"""
    expected = stride * height
    raw = ctypes.string_at(ctypes.cast(pointer, ctypes.c_void_p), expected)
    buffer = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(buffer[:, : width * 3].reshape(height, width, 3))


def release_handles() -> None:
    """服务退出时释放所有缓存句柄。"""
    global _yolo_lib, _anomalib_lib
    with _yolo_lock:
        if _yolo_lib is not None:
            for handle in _yolo_handles.values():
                try:
                    _yolo_lib.yi_destroy(handle)
                except Exception:  # noqa: BLE001
                    pass
            _yolo_handles.clear()
    with _anom_lock:
        if _anomalib_lib is not None:
            for handle in _anom_handles.values():
                try:
                    _anomalib_lib.ai_destroy(handle)
                except Exception:  # noqa: BLE001
                    pass
            _anom_handles.clear()
