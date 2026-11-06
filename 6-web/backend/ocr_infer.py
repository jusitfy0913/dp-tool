# -*- coding: utf-8 -*-
"""OCR 推理桥：用 ctypes 调用 ocr_infer.dll。

检测、识别、方向分类和后处理都在 C++ 动态库里完成。
本模块只做三件事：填 oi_config、把 BGR 图像交给 oi_infer_bgr、把 oi_result 深拷贝成 dict。

和 infer.py 分开，是因为 onnx_infer.dll 的导出也以 oi_ 开头，放在一起容易绑错函数。

懒加载：第一次 OCR 请求才加载 ocr_infer.dll 和约 1.7GB 的 paddle_inference.dll。
加载失败只让 OCR 不可用，不影响 YOLO / Anomalib / ONNX。
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

import ocr_models
from config import BIN_DIR, DLL_DIRS, REPO_ROOT

HANDLE_CACHE_LIMIT = 2


class OcrError(RuntimeError):
    """OCR 错误。status 对应 HTTP 状态码。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class OiTextLine(ctypes.Structure):
    """对应 ocr_infer_c.h 的 oi_text_line。"""

    _fields_ = [
        ("index", ctypes.c_int),
        ("point_count", ctypes.c_int),
        ("xs", ctypes.POINTER(ctypes.c_float)),
        ("ys", ctypes.POINTER(ctypes.c_float)),
        ("rec_score", ctypes.c_float),
        ("textline_angle", ctypes.c_int),
        ("text_utf8", ctypes.c_char_p),
    ]


class OiResult(ctypes.Structure):
    """对应 ocr_infer_c.h 的 oi_result。指针只在下一次推理前有效。"""

    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("line_count", ctypes.c_int),
        ("lines", ctypes.POINTER(OiTextLine)),
        ("overlay_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("overlay_w", ctypes.c_int),
        ("overlay_h", ctypes.c_int),
        ("overlay_stride", ctypes.c_int),
        ("raw_json_utf8", ctypes.c_char_p),
    ]


class OiFrameStat(ctypes.Structure):
    """对应 ocr_infer_c.h 的 oi_frame_stat。"""

    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("count", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("io_ms", ctypes.c_double),
        ("pipeline_ms", ctypes.c_double),
        ("result_ms", ctypes.c_double),
        ("device_utf8", ctypes.c_char * 16),
        ("precision_utf8", ctypes.c_char * 16),
        ("run_mode_utf8", ctypes.c_char * 32),
        ("mkldnn_enabled", ctypes.c_int),
        ("cpu_threads", ctypes.c_int),
        ("rec_batch_size", ctypes.c_int),
    ]


class OiConfig(ctypes.Structure):
    """对应 ocr_infer_c.h 的 oi_config。字符串在 oi_create 返回前必须保持存活。"""

    _fields_ = [
        ("det_model_dir_utf8", ctypes.c_char_p),
        ("rec_model_dir_utf8", ctypes.c_char_p),
        ("textline_orientation_model_dir_utf8", ctypes.c_char_p),
        ("doc_orientation_model_dir_utf8", ctypes.c_char_p),
        ("doc_unwarping_model_dir_utf8", ctypes.c_char_p),
        ("device_utf8", ctypes.c_char_p),
        ("precision_utf8", ctypes.c_char_p),
        ("enable_mkldnn", ctypes.c_int),
        ("use_textline_orientation", ctypes.c_int),
        ("use_doc_orientation_classify", ctypes.c_int),
        ("use_doc_unwarping", ctypes.c_int),
        ("cpu_threads", ctypes.c_int),
        ("rec_batch_size", ctypes.c_int),
        ("det_thresh", ctypes.c_float),
        ("det_box_thresh", ctypes.c_float),
        ("det_unclip_ratio", ctypes.c_float),
        ("rec_score_thresh", ctypes.c_float),
    ]


class _Entry:
    def __init__(self, handle: int) -> None:
        self.handle = handle
        self.lock = threading.Lock()
        self.in_use = 0


_load_lock = threading.Lock()
_create_lock = threading.Lock()
_cache_lock = threading.Lock()
_lib: ctypes.WinDLL | None = None
_load_error = ""
_loaded = False
_dll_dirs: list = []
_cache: OrderedDict[tuple, _Entry] = OrderedDict()
_hits = 0
_creates = 0


def _mkl_dirs() -> list[Path]:
    """mklml.dll 不在 bin 里，Paddle 运行时仍会按名字加载它。"""
    roots = [
        BIN_DIR / "paddle_inference",
        REPO_ROOT / "ocr" / "ocr_infer" / "third_party" / "paddle_inference",
    ]
    found = []
    for root in roots:
        lib = root / "third_party" / "install" / "mklml" / "lib"
        if (lib / "mklml.dll").is_file():
            found.append(lib)
    return found


def _prepare_dll_search_path() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    for directory in list(DLL_DIRS) + _mkl_dirs():
        try:
            _dll_dirs.append(os.add_dll_directory(str(directory)))
        except OSError:
            continue


def _bind(lib: ctypes.WinDLL) -> None:
    lib.oi_create.restype = ctypes.c_void_p
    lib.oi_create.argtypes = [ctypes.POINTER(OiConfig)]
    lib.oi_infer_bgr.restype = ctypes.c_int
    lib.oi_infer_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(OiResult)),
        ctypes.POINTER(OiFrameStat),
    ]
    lib.oi_destroy.restype = None
    lib.oi_destroy.argtypes = [ctypes.c_void_p]
    lib.oi_last_error.restype = ctypes.c_char_p
    lib.oi_last_error.argtypes = []


def _last_error() -> str:
    if _lib is None:
        return _load_error or "ocr_infer.dll 未加载"
    raw = _lib.oi_last_error()
    if not raw:
        return "ocr_infer.dll 没有返回错误信息"
    return raw.decode("utf-8", errors="replace")


def status() -> dict:
    """不触发加载。文件不存在或上次加载失败时 ok 为 False。"""
    dll = BIN_DIR / "ocr_infer.dll"
    if _load_error:
        return {"ok": False, "loaded": False, "detail": _load_error, "dll": str(dll)}
    if not dll.is_file():
        return {"ok": False, "loaded": False, "detail": f"缺少 {dll}", "dll": str(dll)}
    return {"ok": True, "loaded": _lib is not None, "detail": "", "dll": str(dll)}


def ensure_loaded() -> ctypes.WinDLL:
    """首次调用才加载。失败结果会记住，避免每次请求都再撞一次。"""
    global _lib, _load_error, _loaded
    if _lib is not None:
        return _lib
    with _load_lock:
        if _lib is not None:
            return _lib
        if _loaded and _load_error:
            raise OcrError(_load_error, status=503)
        _loaded = True
        dll = BIN_DIR / "ocr_infer.dll"
        if not dll.is_file():
            _load_error = f"缺少 OCR 动态库：{dll}"
            raise OcrError(_load_error, status=503)
        _prepare_dll_search_path()
        try:
            lib = ctypes.WinDLL(str(dll))
            _bind(lib)
        except OSError as exc:
            _load_error = f"加载 ocr_infer.dll 失败：{exc}"
            raise OcrError(_load_error, status=503) from exc
        _lib = lib
        return lib


def _check_ranges(options: dict) -> None:
    device = options["device"]
    if device not in ("cpu", "gpu:0"):
        raise OcrError("device 只能是 cpu 或 gpu:0")
    threads = options["cpu_threads"]
    batch = options["rec_batch_size"]
    if not 1 <= threads <= 64:
        raise OcrError("cpu_threads 必须在 1 到 64 之间")
    if not 1 <= batch <= 64:
        raise OcrError("rec_batch_size 必须在 1 到 64 之间")
    for name in ("det_thresh", "det_box_thresh", "rec_score_thresh"):
        value = options[name]
        if not 0.0 <= value <= 1.0:
            raise OcrError(f"{name} 必须在 0 到 1 之间")
    unclip = options["det_unclip_ratio"]
    if not 0.5 <= unclip <= 5.0:
        raise OcrError("det_unclip_ratio 必须在 0.5 到 5 之间")


def _cache_key(options: dict) -> tuple:
    def norm(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        return str(Path(text).expanduser().resolve())

    return (
        norm(options["det_dir"]),
        norm(options["rec_dir"]),
        norm(options["textline_dir"]),
        options["device"],
        int(options["enable_mkldnn"]),
        int(options["cpu_threads"]),
        int(options["rec_batch_size"]),
        round(float(options["det_thresh"]), 4),
        round(float(options["det_box_thresh"]), 4),
        round(float(options["det_unclip_ratio"]), 4),
        round(float(options["rec_score_thresh"]), 4),
    )


def _destroy(handle: int) -> None:
    if _lib is not None and handle:
        _lib.oi_destroy(handle)


def _evict_unlocked() -> None:
    """只淘汰没人正在用的最久未用句柄。调用方必须已持有 _cache_lock。"""
    while len(_cache) > HANDLE_CACHE_LIMIT:
        victim_key = None
        for key, entry in _cache.items():
            if entry.in_use == 0:
                victim_key = key
                break
        if victim_key is None:
            return
        entry = _cache.pop(victim_key)
        _destroy(entry.handle)


def _create(lib: ctypes.WinDLL, key: tuple) -> int:
    (
        det_dir, rec_dir, textline_dir, device, mkldnn, threads, batch,
        det_thresh, box_thresh, unclip, rec_score,
    ) = key
    held = [
        det_dir.encode("utf-8"),
        rec_dir.encode("utf-8"),
        textline_dir.encode("utf-8"),
        b"",
        b"",
        device.encode("utf-8"),
        b"fp32",
    ]
    cfg = OiConfig()
    cfg.det_model_dir_utf8 = held[0]
    cfg.rec_model_dir_utf8 = held[1]
    cfg.textline_orientation_model_dir_utf8 = held[2]
    cfg.doc_orientation_model_dir_utf8 = held[3]
    cfg.doc_unwarping_model_dir_utf8 = held[4]
    cfg.device_utf8 = held[5]
    cfg.precision_utf8 = held[6]
    cfg.enable_mkldnn = mkldnn
    cfg.use_textline_orientation = 1 if textline_dir else 0
    cfg.use_doc_orientation_classify = 0
    cfg.use_doc_unwarping = 0
    cfg.cpu_threads = threads
    cfg.rec_batch_size = batch
    cfg.det_thresh = det_thresh
    cfg.det_box_thresh = box_thresh
    cfg.det_unclip_ratio = unclip
    cfg.rec_score_thresh = rec_score
    handle = lib.oi_create(ctypes.byref(cfg))
    # held 必须活到 oi_create 返回：DLL 在函数内把路径拷进 std::string。
    del held
    if not handle:
        raise OcrError(f"创建 OCR 句柄失败：{_last_error()}", status=500)
    return int(handle)


def _acquire(lib: ctypes.WinDLL, key: tuple) -> _Entry:
    global _hits, _creates
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            entry.in_use += 1
            _cache.move_to_end(key)
            _hits += 1
            return entry
    with _create_lock:
        with _cache_lock:
            entry = _cache.get(key)
            if entry is not None:
                entry.in_use += 1
                _cache.move_to_end(key)
                _hits += 1
                return entry
        handle = _create(lib, key)
        entry = _Entry(handle)
        with _cache_lock:
            entry.in_use = 1
            _cache[key] = entry
            _creates += 1
            _evict_unlocked()
        return entry


def _release(entry: _Entry) -> None:
    with _cache_lock:
        entry.in_use = max(0, entry.in_use - 1)


def _decode_text(raw) -> str:
    if not raw:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _copy_result(result: OiResult, stat: OiFrameStat) -> dict:
    """在持有句柄锁时把 DLL 指针拷成纯 Python 对象。"""
    lines = []
    count = int(result.line_count)
    if count > 0 and result.lines:
        for index in range(count):
            line = result.lines[index]
            points = max(0, int(line.point_count))
            poly = []
            for point in range(points):
                poly.append([float(line.xs[point]), float(line.ys[point])])
            lines.append({
                "index": int(line.index),
                "poly": poly,
                "text": _decode_text(line.text_utf8),
                "score": float(line.rec_score),
                "angle": int(line.textline_angle),
            })
    return {
        "backend": "ocr_infer.dll",
        "device": _decode_text(bytes(stat.device_utf8).split(b"\x00", 1)[0]),
        "precision": _decode_text(bytes(stat.precision_utf8).split(b"\x00", 1)[0]),
        "run_mode": _decode_text(bytes(stat.run_mode_utf8).split(b"\x00", 1)[0]),
        "mkldnn": bool(stat.mkldnn_enabled),
        "cpu_threads": int(stat.cpu_threads),
        "rec_batch_size": int(stat.rec_batch_size),
        "src_w": int(result.src_w),
        "src_h": int(result.src_h),
        "count": len(lines),
        "lines": lines,
        "raw_json": _decode_text(result.raw_json_utf8),
        "timing": {
            "io_ms": float(stat.io_ms),
            "pipeline_ms": float(stat.pipeline_ms),
            "result_ms": float(stat.result_ms),
            "total_ms": float(stat.total_ms),
            "preprocess_ms": float(stat.io_ms),
            "inference_ms": float(stat.pipeline_ms),
            "postprocess_ms": float(stat.result_ms),
        },
    }


def run(image: np.ndarray, options: dict) -> dict:
    """对一张 BGR 图做 OCR。options 里的路径和数值见设计文档 D5。"""
    _check_ranges(options)
    try:
        ocr_models.preflight(options["det_dir"], options["rec_dir"], options.get("textline_dir", ""))
    except ocr_models.OcrModelError as exc:
        raise OcrError(str(exc), status=400) from exc

    array = np.ascontiguousarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise OcrError("OCR 输入必须是 uint8 的 BGR 图像")

    lib = ensure_loaded()
    key = _cache_key(options)
    entry = _acquire(lib, key)
    try:
        with entry.lock:
            out = ctypes.POINTER(OiResult)()
            stat = OiFrameStat()
            ptr = array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            code = lib.oi_infer_bgr(
                entry.handle,
                ptr,
                int(array.shape[1]),
                int(array.shape[0]),
                int(array.strides[0]),
                ctypes.byref(out),
                ctypes.byref(stat),
            )
            if code != 0 or not out:
                raise OcrError(f"OCR 推理失败：{_last_error()}", status=500)
            return _copy_result(out.contents, stat)
    finally:
        _release(entry)


def release_handles() -> None:
    """进程退出时释放缓存句柄。"""
    with _cache_lock:
        while _cache:
            _key, entry = _cache.popitem()
            _destroy(entry.handle)


def cache_stats() -> dict:
    with _cache_lock:
        keys = []
        for key in reversed(_cache):
            det, rec, textline, device, *_rest = key
            keys.append({
                "det": Path(det).name if det else "",
                "rec": Path(rec).name if rec else "",
                "textline": Path(textline).name if textline else "",
                "device": device,
            })
        return {
            "limit": HANDLE_CACHE_LIMIT,
            "size": len(_cache),
            "hits": _hits,
            "creates": _creates,
            "keys": keys,
        }
