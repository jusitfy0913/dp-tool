# -*- coding: utf-8 -*-
"""Python ctypes 示例：调用 anomalib_infer.dll 推理单张图片。"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np


class AiMetadata(ctypes.Structure):
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


class AiFrameStat(ctypes.Structure):
    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


class AiResult(ctypes.Structure):
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


def _utf8(path: str | Path) -> bytes:
    return str(path).encode("utf-8")


def _copy_rgb(ptr: ctypes.POINTER(ctypes.c_ubyte), width: int, height: int, stride: int) -> np.ndarray:
    raw = ctypes.string_at(ptr, height * stride)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return arr[:, : width * 3].reshape(height, width, 3).copy()


def load_library(path: Path) -> ctypes.CDLL:
    dll = ctypes.CDLL(str(path))
    dll.ai_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float]
    dll.ai_create.restype = ctypes.c_void_p
    dll.ai_get_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(AiMetadata)]
    dll.ai_get_metadata.restype = ctypes.c_int
    dll.ai_infer_rgb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(AiResult)),
        ctypes.POINTER(AiFrameStat),
    ]
    dll.ai_infer_rgb.restype = ctypes.c_int
    dll.ai_destroy.argtypes = [ctypes.c_void_p]
    dll.ai_destroy.restype = None
    dll.ai_last_error.argtypes = []
    dll.ai_last_error.restype = ctypes.c_char_p
    return dll


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Anomalib TensorRT inference through anomalib_infer.dll")
    parser.add_argument("--dll", required=True, type=Path, help="Path to anomalib_infer.dll")
    parser.add_argument("--engine", required=True, type=Path, help="Path to .trt/.engine")
    parser.add_argument("--metadata", required=True, type=Path, help="Path to metadata.json")
    parser.add_argument("--image", required=True, type=Path, help="Input image")
    parser.add_argument("--out-dir", default=Path("outputs"), type=Path, help="Output directory")
    args = parser.parse_args()

    dll = load_library(args.dll)
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    handle = dll.ai_create(_utf8(args.engine), _utf8(args.metadata), b"Auto", -1.0, -1.0)
    if not handle:
        raise RuntimeError(dll.ai_last_error().decode("utf-8", errors="replace"))

    try:
        meta = AiMetadata()
        if dll.ai_get_metadata(handle, ctypes.byref(meta)) != 0:
            raise RuntimeError(dll.ai_last_error().decode("utf-8", errors="replace"))

        result_ptr = ctypes.POINTER(AiResult)()
        stat = AiFrameStat()
        rc = dll.ai_infer_rgb(
            handle,
            image_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            image_rgb.shape[1],
            image_rgb.shape[0],
            image_rgb.strides[0],
            ctypes.byref(result_ptr),
            ctypes.byref(stat),
        )
        if rc != 0:
            raise RuntimeError(dll.ai_last_error().decode("utf-8", errors="replace"))

        result = result_ptr.contents
        overlay_rgb = _copy_rgb(result.overlay_rgb, result.overlay_w, result.overlay_h, result.overlay_stride)
        mask_rgb = _copy_rgb(result.mask_rgb, result.mask_w, result.mask_h, result.mask_stride)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_dir / "overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.out_dir / "mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

        status = "NG" if result.is_anomaly else "OK"
        print(
            f"model={meta.model_type.decode('utf-8', errors='replace').rstrip(chr(0))} "
            f"score={result.normalized_score:.3f} raw={result.raw_score:.3f} "
            f"status={status} infer_ms={stat.inference_ms:.1f}"
        )
    finally:
        dll.ai_destroy(handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
