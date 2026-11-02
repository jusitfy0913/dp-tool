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
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import labels as labels_mod
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
# 句柄缓存（LRU）
# ---------------------------------------------------------------------------
# 为什么要有上限：
#   引擎句柄在创建时就把 conf / nms / 阈值**烧死**在里面，换参数必须重建。
#   一旦前端把阈值开放成滑块，用户拖动几十下就会产生几十个句柄，
#   而每个句柄都常驻一份显存（PatchCore 的引擎本身 275MB）。
#   实测过没有上限时显存会一路涨到 OOM。
#
# 为什么是 4：
#   典型现场并存量是「YOLO 两个任务 + Anomalib 两个模型」。超过 4 说明用户在
#   反复试参，此时旧句柄留着也没有命中收益，不如立刻腾出显存。
#
# 为什么不用 TTL：
#   显存压力来自"同时存在多少个"，不是"存在多久"。空转的句柄不会自己释放，
#   TTL 只是把 OOM 推迟到定时器触发那一刻。
HANDLE_CACHE_LIMIT = 4


class _HandleCache:
    """带上限的句柄缓存；超限时销毁最久未用项。

    淘汰时**立即**调用 destroy，而不是等进程退出：显存是有限资源，
    靠析构函数兜底在实际使用中根本等不到。
    """

    def __init__(self, limit: int, destroy) -> None:
        self._limit = max(1, int(limit))
        self._destroy = destroy
        self._items: OrderedDict[tuple, ctypes.c_void_p] = OrderedDict()
        # 供测试观察：记录被淘汰的键，用于断言"淘汰的是最久未用的那个"
        self.evicted: list[tuple] = []

    def get(self, key: tuple) -> ctypes.c_void_p | None:
        """命中则移到队尾（标记为最近使用）并返回句柄。"""
        handle = self._items.get(key)
        if handle:
            self._items.move_to_end(key)
            return handle
        return None

    def put(self, key: tuple, handle: ctypes.c_void_p) -> None:
        """写入新句柄；超过上限先淘汰最久未用项。"""
        self._items[key] = handle
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            old_key, old_handle = self._items.popitem(last=False)
            self.evicted.append(old_key)
            self._safe_destroy(old_handle)

    def _safe_destroy(self, handle: ctypes.c_void_p) -> None:
        try:
            self._destroy(handle)
        except Exception:  # noqa: BLE001 - 销毁失败不能让推理流程崩掉
            pass

    def clear(self) -> None:
        while self._items:
            _, handle = self._items.popitem(last=True)
            self._safe_destroy(handle)

    def keys(self) -> list[tuple]:
        """当前缓存键，按最近使用到最久未用排列。"""
        return list(self._items.keys())

    def __len__(self) -> int:
        return len(self._items)



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


def encode_image(image: np.ndarray, suffix: str = ".jpg", quality: int = 92) -> bytes:
    """按扩展名编码图片字节。

    批量推理默认要 JPEG：同样的 PCB 图，PNG 约 1.8MB、JPEG(q=92) 约 220KB，
    一次跑 5000 张时这个差别决定了「能不能一次性打包下载」。
    单图接口仍然用 PNG——只有一张，无损更划算。
    """
    import cv2

    ext = (suffix or ".jpg").lower()
    if ext in (".jpg", ".jpeg"):
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        ext = ".png"
        ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise InferError(f"结果图编码失败（{ext}）")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# YOLO 推理
# ---------------------------------------------------------------------------
_yolo_lock = threading.Lock()


def _destroy_yolo(handle: ctypes.c_void_p) -> None:
    """销毁一个 YOLO 句柄。库没加载时静默跳过（服务退出路径会走到）。"""
    lib = _yolo_lib
    if lib is not None and handle:
        lib.yi_destroy(handle)


_yolo_handles = _HandleCache(HANDLE_CACHE_LIMIT, _destroy_yolo)


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

    _yolo_handles.put(key, handle)
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
    """执行一次 YOLO 推理，返回可 JSON 序列化的结果。

    这里只产出整数类别号；类别名与配色统一由 `attach_labels()` 补上，
    这样 DLL 路径与 onnxruntime 降级路径的结构天然一致。
    """
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


def attach_labels(result: dict, classes: list[str] | None) -> dict:
    """给推理结果补上类别名与统一配色，返回同一个 dict。

    单独成函数的原因：DLL 路径与 onnxruntime 降级路径都要用，
    放在各自实现里迟早会漂移（一边有 label_name 一边没有），
    前端的绘制与表格就得写两套兜底逻辑。
    """
    boxes = result.get("boxes") or []
    for item in boxes:
        item["label_name"] = labels_mod.name_for(int(item.get("class_label", 0)), classes)

    class_count = len(classes) if classes else 0
    highest = max((int(b.get("class_label", 0)) for b in boxes), default=-1)
    # 没有标签文件时也要给配色：颜色数按实际出现的最大类别号 + 1，
    # 否则前端取不到颜色会全部退化成同一种。
    color_count = max(class_count, highest + 1, 1)

    result["class_names"] = list(classes) if classes else []
    result["class_colors"] = labels_mod.class_colors(color_count)
    return result


def draw_yolo(image: np.ndarray, boxes: list[dict], class_colors: dict | None = None) -> bytes:
    """把检测框画到图上，返回 PNG 字节。

    颜色取 `class_colors`（接口下发的同一份），据此保证"导出的图和屏幕上看到的一致"。
    没传配色时退回按类别号现算，结果与下发的一致（同一个纯函数）。
    """
    return _encode_png(render_yolo(image, boxes, class_colors))


def render_yolo(
    image: np.ndarray,
    boxes: list[dict],
    class_colors: dict | None = None,
) -> np.ndarray:
    """把检测框画到图上，返回 BGR 数组（不编码）。

    单独成函数的原因：批量推理要按扩展名选编码（JPEG 省 8 倍体积），
    视频要直接拿数组喂 VideoWriter，两者都不该为了拿 PNG 再解码一次。
    """
    import cv2

    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(height, width) / 320)))
    font_scale = max(0.5, min(height, width) / 900)

    def bgr_for(class_label: int) -> tuple[int, int, int]:
        """OpenCV 要 BGR，配色表存的是 RGB。"""
        rgb = None
        if class_colors:
            rgb = class_colors.get(str(class_label))
        if rgb is None:
            rgb = labels_mod.color_for(class_label)
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    for item in boxes:
        color = bgr_for(int(item.get("class_label", 0)))
        if item.get("is_obb"):
            obb = item["obb"]
            rect = ((obb["cx"], obb["cy"]), (obb["w"], obb["h"]), obb["angle"] * 180.0 / 3.141592653589793)
            points = cv2.boxPoints(rect).astype(int)
            cv2.polylines(canvas, [points], True, color, thickness)
            origin = tuple(points[0])
        else:
            x1, y1 = int(item["left"]), int(item["top"])
            x2, y2 = int(item["right"]), int(item["bottom"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            origin = (x1, max(0, y1 - 6))

        label = f"{item.get('label_name') or 'cls' + str(item.get('class_label', 0))} {item['confidence']:.2f}"
        # 先画一圈深色描边再画亮色字：PCB 这类背景本身就很花，单色字容易糊进去
        cv2.putText(
            canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (16, 18, 22), max(1, thickness + 1), cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA,
        )

        for kpt in item.get("kpts", []):
            if len(kpt) >= 2:
                cv2.circle(canvas, (int(kpt[0]), int(kpt[1])), thickness + 1, (0, 255, 255), -1)

    return canvas


# ---------------------------------------------------------------------------
# Anomalib 推理
# ---------------------------------------------------------------------------
_anom_lock = threading.Lock()


def _destroy_anom(handle: ctypes.c_void_p) -> None:
    """销毁一个 Anomalib 句柄。"""
    lib = _anomalib_lib
    if lib is not None and handle:
        lib.ai_destroy(handle)


_anom_handles = _HandleCache(HANDLE_CACHE_LIMIT, _destroy_anom)

# 默认色带：TURBO 的色差感知比 JET 均匀，异常强度的层次不会在黄绿段挤成一片。
DEFAULT_COLORMAP = "TURBO"
_FALLBACK_COLORMAPS = ("TURBO", "JET", "INFERNO", "VIRIDIS")


def resolve_colormap(name: str | None):
    """把色带名解析成 cv2 常量；非法名称回退默认，不抛错。

    cv2 的版本差异不小（COLORMAP_TURBO 是 4.1.1 之后才有的），
    所以逐个候选往下试，而不是直接 getattr 一次了事。
    """
    import cv2

    wanted = (name or DEFAULT_COLORMAP).strip().upper().replace("COLORMAP_", "")
    for candidate in (wanted, *_FALLBACK_COLORMAPS):
        value = getattr(cv2, f"COLORMAP_{candidate}", None)
        if value is not None:
            return value, candidate
    # 极端情况下连 JET 都没有，用 0（AUTUMN）兜底，保证接口永远能出图
    return 0, "AUTUMN"


def read_metadata_thresholds(metadata_path: str) -> dict:
    """从 metadata.json 里读出**文件原值**阈值。

    为什么不直接用 ai_get_metadata：
        ai_create 的 override 是**原地改写**内部 meta_（见 anomalib_infer_c.cpp
        `load()` 里的 `if (imageThresholdOverride >= 0) meta_.imageThreshold = ...`），
        所以句柄建立之后再查 metadata，拿到的已经是"生效值"，原值就此丢失。
        要在界面上并列显示"用户设定"与"metadata 原值"，只能自己读文件。

    读失败不抛错：原值只用于展示，不影响推理。
    """
    info = {"image_threshold": None, "pixel_threshold": None,
            "pred_scores": None, "anomaly_maps": None, "error": ""}
    try:
        import json

        raw = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        info["error"] = f"metadata 读取失败：{exc}"
        return info

    for key in ("image_threshold", "pixel_threshold"):
        value = raw.get(key)
        # bool 是 int 的子类，显式排除，免得 true 被当成 1.0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            info[key] = float(value)

    for key in ("pred_scores", "anomaly_maps"):
        block = raw.get(key)
        if isinstance(block, dict):
            low, high = block.get("min"), block.get("max")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                info[key] = [float(low), float(high)]
    return info


def _anom_handle(
    engine_path: str,
    metadata_path: str,
    model_type: str,
    image_threshold: float = -1.0,
    pixel_threshold: float = -1.0,
) -> ctypes.c_void_p:
    """取（或创建）Anomalib 句柄。

    阈值进入缓存键是必须的：句柄在创建时就把阈值烧进 meta_，
    用同一个句柄做两次不同阈值的推理，第二次会沿用第一次的判定标准，
    界面上看起来"改了阈值没反应"。
    """
    key = (
        os.path.abspath(engine_path),
        os.path.abspath(metadata_path),
        model_type,
        round(float(image_threshold), 6),
        round(float(pixel_threshold), 6),
    )
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

    # 阈值 override 传负数表示沿用 metadata —— 这是 DLL 自带的约定（见 anomalib_infer_c.h）
    handle = lib.ai_create(
        str(engine_file).encode("utf-8"),
        str(metadata_file).encode("utf-8"),
        model_type.encode("utf-8"),
        ctypes.c_float(float(image_threshold)),
        ctypes.c_float(float(pixel_threshold)),
    )
    if not handle:
        raise InferError(_lib_error(lib, "ai_create"))

    _anom_handles.put(key, handle)
    return handle


def run_anomalib_score(
    engine_path: str,
    metadata_path: str,
    image: np.ndarray,
    model_type: str = "Auto",
    image_threshold: float = -1.0,
    pixel_threshold: float = -1.0,
) -> dict:
    """只取分数与判定，不渲染任何 PNG。

    精度评估要逐张跑几千次，`run_anomalib` 每次都会编码叠加图/轮廓图/热能图/原图
    四张 PNG 并拷贝出指针内容，一张 1080p 的图光编码就要几十毫秒，整体耗时
    会被图片编码彻底主导（推理本身只有几毫秒）。这里只搬运三个标量。

    判定用 DLL 的 `is_anomaly`：评估报告里的漏检率必须与页面上单张推理的判定
    完全一致，自己拿 raw_score 再判一次是引入分歧的最短路径。
    """
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    stride = int(rgb.strides[0])
    buffer = rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    lib = _anomalib_lib
    if lib is None:
        raise InferError(f"anomalib_infer.dll 未加载：{_load_error or '未知原因'}")

    with _anom_lock:
        handle = _anom_handle(
            engine_path, metadata_path, model_type, image_threshold, pixel_threshold
        )
        result_ptr = ctypes.POINTER(AiResult)()
        stat = AiFrameStat()
        code = lib.ai_infer_rgb(
            handle, buffer, width, height, stride,
            ctypes.byref(result_ptr), ctypes.byref(stat),
        )
        if code != 0:
            raise InferError(_lib_error(lib, "ai_infer_rgb"))
        result = result_ptr.contents
        return {
            "raw_score": float(result.raw_score),
            "normalized_score": float(result.normalized_score),
            "is_anomaly": bool(result.is_anomaly),
            "total_ms": float(stat.total_ms),
            "image_width": width,
            "image_height": height,
        }


def run_anomalib(
    engine_path: str,
    metadata_path: str,
    image: np.ndarray,
    model_type: str = "Auto",
    want_overlay: bool = True,
    image_threshold: float = -1.0,
    pixel_threshold: float = -1.0,
    colormap: str | None = None,
    want_heatmap: bool = True,
) -> dict:
    """执行一次 Anomalib 推理，返回分数、判定、叠加图、轮廓图与热能图。

    输入要求 RGB888，这里从 BGR 转换。

    阈值语义：`>= 0` 覆盖 metadata，`< 0` 沿用 metadata（与 DLL 一致）。
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

    originals = read_metadata_thresholds(metadata_path)

    with _anom_lock:
        handle = _anom_handle(
            engine_path, metadata_path, model_type, image_threshold, pixel_threshold
        )

        metadata = AiMetadata()
        if lib.ai_get_metadata(handle, ctypes.byref(metadata)) == 0:
            meta_dict = {
                "model_type": metadata.model_type.decode("utf-8", errors="replace"),
                "normalization": metadata.normalization.decode("utf-8", errors="replace"),
                "output_contract": metadata.output_contract.decode("utf-8", errors="replace"),
                # 注意：这两个值是**生效值**（override 已经写进去了），
                # 文件原值在下面的 thresholds.metadata 里
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
            "heatmap_png": None,
            "source_png": None,
            "image_width": width,
            "image_height": height,
        }

        # 叠加图 / 轮廓图 / 原图 / 热能图都是 DLL 持有的指针，
        # 只在下一次推理前有效，所以必须在这里立刻拷贝成 PNG。
        if result.source_rgb and result.source_w > 0 and result.source_h > 0:
            source = _pointer_to_array(
                result.source_rgb, result.source_h, result.source_w, result.source_stride
            )
            payload["source_png"] = _encode_png(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
            payload["image_width"] = int(result.source_w)
            payload["image_height"] = int(result.source_h)

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

        colormap_name = DEFAULT_COLORMAP
        if want_heatmap:
            heatmap, colormap_name = _render_heatmap(result, metadata, colormap)
            payload["heatmap_png"] = heatmap
        else:
            _, colormap_name = resolve_colormap(colormap)
        payload["heatmap_colormap"] = colormap_name

    # 生效阈值 = 覆盖值（>=0）或文件原值；两者都拿不到时退回 DLL 报的值
    effective_image = (
        float(image_threshold) if float(image_threshold) >= 0
        else originals["image_threshold"]
        if originals["image_threshold"] is not None
        else meta_dict.get("image_threshold")
    )
    effective_pixel = (
        float(pixel_threshold) if float(pixel_threshold) >= 0
        else originals["pixel_threshold"]
        if originals["pixel_threshold"] is not None
        else meta_dict.get("pixel_threshold")
    )
    payload["thresholds"] = {
        "image": {
            "effective": effective_image,
            "metadata": originals["image_threshold"],
            "overridden": float(image_threshold) >= 0,
        },
        "pixel": {
            "effective": effective_pixel,
            "metadata": originals["pixel_threshold"],
            "overridden": float(pixel_threshold) >= 0,
        },
    }
    return payload


def _pointer_to_array(pointer, height: int, width: int, stride: int) -> np.ndarray:
    """把 DLL 返回的 RGB 指针拷成 numpy 数组（去行对齐，转 BGR 由调用方做）。"""
    expected = stride * height
    raw = ctypes.string_at(ctypes.cast(pointer, ctypes.c_void_p), expected)
    buffer = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(buffer[:, : width * 3].reshape(height, width, 3))


def _render_heatmap(result: AiResult, metadata: AiMetadata, colormap: str | None = None):
    """把 `normalized_map` 渲染成假彩色热能图，返回 (PNG 字节 | None, 实际色带名)。

    为什么不直接用 DLL 给的叠加图：
        叠加图是"原图 + 半透明热力"的成品，判读时无法把"原图纹理"与"异常强度"分开看。
        热能图是把异常强度单独摊开，配合叠加图一起看才能判断"是误检还是真异常"。

    归一化用 metadata 的 anomaly_maps.min/max（`AiMetadata.map_min/map_max`）：
        同一批图用同一个范围，不同图之间的颜色才可比。若 metadata 没给范围，
        才退回本图自身的最值（此时只能看"这张图里哪里最异常"，不能横向比）。

    顺序是"先放大灰度值图，再上色"：
        反过来先上色再放大，颜色会在线性插值下互相污染，弱异常区会出现脏色带。
    """
    import cv2

    _, name_used = resolve_colormap(colormap)

    if not result.normalized_map or result.map_w <= 0 or result.map_h <= 0:
        return None, name_used

    count = int(result.map_w) * int(result.map_h)
    raw = ctypes.string_at(ctypes.cast(result.normalized_map, ctypes.c_void_p), count * 4)
    values = np.frombuffer(raw, dtype=np.float32).reshape(int(result.map_h), int(result.map_w))
    values = values.astype(np.float32, copy=True)

    low, high = float(metadata.map_min), float(metadata.map_max)
    if not (high > low):
        low, high = float(values.min()), float(values.max())
    if high > low:
        scaled = (values - low) / (high - low)
    else:
        # 全图同值（正常图常见）：全给 0，不要除以 0 也不要伪造"有异常"
        scaled = np.zeros_like(values)

    scaled = np.clip(scaled, 0.0, 1.0)
    gray = (scaled * 255.0).round().astype(np.uint8)

    # 放大到原图尺寸；尺寸取 source 的，没有就保持 map 的原尺寸
    target_w = int(result.source_w) if result.source_w > 0 else int(result.map_w)
    target_h = int(result.source_h) if result.source_h > 0 else int(result.map_h)
    if (target_w, target_h) != (int(result.map_w), int(result.map_h)):
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    colormap_id, name_used = resolve_colormap(colormap)
    colored = cv2.applyColorMap(gray, colormap_id)
    return _encode_png(colored), name_used


def release_handles() -> None:
    """服务退出时释放所有缓存句柄。"""
    with _yolo_lock:
        _yolo_handles.clear()
    with _anom_lock:
        _anom_handles.clear()


def cache_stats() -> dict:
    """当前句柄缓存状态，供测试与诊断页查看。"""
    return {
        "limit": HANDLE_CACHE_LIMIT,
        "yolo": len(_yolo_handles),
        "anomalib": len(_anom_handles),
        "yolo_keys": [list(map(str, k)) for k in _yolo_handles.keys()],
        "anomalib_keys": [list(map(str, k)) for k in _anom_handles.keys()],
    }
