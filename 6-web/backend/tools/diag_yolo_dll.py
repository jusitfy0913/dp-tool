# -*- coding: utf-8 -*-
"""诊断 yolo_infer.dll 返回的 box 数据为何错乱。

思路：不经过 Web 层，直接用 ctypes 调 DLL，把
  1) yi_frame_stat（src_w/src_h/count/耗时）
  2) 前几个 box 的原始 float32 内存（按 22 个 float 视窗扫）
全部打印出来，判断是「句柄/图像传入错误」还是「结构体错位」。

编码 UTF-8。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "4-qt-studio" / "bin"
ENGINE = REPO / "1-python-yolo" / "launcher" / "runs" / "pcb" / "detect_v1" / "weights" / "best.transd_fp16.engine"
IMG = sorted((REPO / "PCB data" / "PCB缺陷检测" / "images" / "val").glob("*.jpg"))[0]


class YiBox(ctypes.Structure):
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
    _fields_ = [
        ("src_w", ctypes.c_int),
        ("src_h", ctypes.c_int),
        ("count", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


def main() -> int:
    print(f"结构体 sizeof(YiBox) = {ctypes.sizeof(YiBox)}")

    if hasattr(os, "add_dll_directory"):
        pass
    # 复用 Web 后端的加载逻辑（它已经处理好 OpenCV/CUDA 的 DLL 目录）
    sys.path.insert(0, str(REPO / "6-web" / "backend"))
    import infer as backend_infer

    status = backend_infer.load_libraries()
    print(f"加载状态: yolo_ok={status.yolo_ok} detail={status.detail}")
    if not status.yolo_ok:
        return 1
    lib = backend_infer._yolo_lib
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

    h = lib.yi_create(str(ENGINE).encode(), 0, 0.25, 0.45)
    print(f"yi_create -> {h}")
    if not h:
        print(f"创建失败: {lib.yi_last_error()}")
        return 1

    import cv2

    # cv2.imread 在 Windows 上不支持非 ASCII 路径（本数据集目录含中文），
    # 必须先用 numpy 读字节再 imdecode —— Web 后端的 decode_image 也是这个思路。
    bgr = cv2.imdecode(np.fromfile(str(IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! 图片解码失败: {IMG}")
        return 2
    bgr = np.ascontiguousarray(bgr)
    print(f"图像 {IMG.name}: {bgr.shape[1]}x{bgr.shape[0]} dtype={bgr.dtype}")

    boxes_ptr = ctypes.POINTER(YiBox)()
    count = ctypes.c_int(0)
    stat = YiFrameStat()
    code = lib.yi_infer_bgr(
        h,
        bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        bgr.shape[1],
        bgr.shape[0],
        bgr.strides[0],
        ctypes.byref(boxes_ptr),
        ctypes.byref(count),
        ctypes.byref(stat),
    )
    print(f"yi_infer_bgr -> {code}, 错误={lib.yi_last_error()}")
    print(
        f"stat: src={stat.src_w}x{stat.src_h} count={stat.count} "
        f"pre={stat.preprocess_ms:.2f}ms inf={stat.inference_ms:.2f}ms "
        f"post={stat.postprocess_ms:.2f}ms total={stat.total_ms:.2f}ms"
    )
    print(f"out_count={count.value}")

    # 按 float32 视窗扫首元素，看是否存在“整体错位”
    raw = ctypes.cast(boxes_ptr, ctypes.POINTER(ctypes.c_float))
    n = min(count.value, 3)
    for i in range(n):
        base = i * (ctypes.sizeof(YiBox) // 4)
        vals = [round(raw[base + k], 3) for k in range(22)]
        print(f"  box[{i}] 首22个float: {vals}")

    lib.yi_destroy(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
