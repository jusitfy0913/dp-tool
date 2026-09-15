# -*- coding: utf-8 -*-
"""
TRTForge DLL API 自动化冒烟测试
覆盖：yolo_infer.dll (DET/SEG/POSE/OBB) + anomalib_infer.dll (PatchCore)
运行方式：
    python dll_api_smoke_test.py
    python dll_api_smoke_test.py --out-dir D:/test_results
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Windows 下兼容中文/Unicode 路径的图片读取，替代 cv2.imread。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# ─────────────────────────────────────────────
# 路径配置（相对于项目根目录自动推导）
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # deeplearning/
BIN  = ROOT / "trtexec/gui/bin"

ASSET = {
    "yolo_dll":        BIN / "yolo_infer.dll",
    "anomalib_dll":    BIN / "anomalib_infer.dll",
    "det_engine":      ROOT / "trtmodel/trtexcemodel/yolov8-detection.trans.trt",
    "seg_engine":      ROOT / "trtmodel/trtexcemodel/yolov8-seg.transd.trt",
    "pose_engine":     ROOT / "trtmodel/trtexcemodel/yolov8s-pose.transd.trt",
    "obb_engine":      ROOT / "trtmodel/trtexcemodel/yolov8n-obb.transd.trt",
    "coco_labels":     BIN  / "configs/labels_coco.json",
    "obb_labels":      BIN  / "configs/labels_obb_dota_v1.json",
    "bus_img":         ROOT / "YOLOv8-det/workspace/images/bus.jpg",
    "zidane_img":      ROOT / "YOLOv8-det/workspace/images/zidane.jpg",
    "plane_img":       ROOT / "YOLOv8_obb/normal/plane.jpeg",
    "plane2_img":      ROOT / "YOLOv8_obb/normal/plane2.jpg",
    "anomalib_engine": ROOT / "results/Patchcore/demo_5/latest/weights/lightning/model.trt",
    "anomalib_meta":   ROOT / "results/Patchcore/demo_5/latest/weights/lightning/metadata.json",
    "abnormal_dir":    ROOT / "anomalib/bottle/test/abnormal",
}

# ─────────────────────────────────────────────
# ctypes 结构体定义（与头文件保持一致）
# ─────────────────────────────────────────────

class YiBox(ctypes.Structure):
    _fields_ = [
        ("left",       ctypes.c_float),
        ("top",        ctypes.c_float),
        ("right",      ctypes.c_float),
        ("bottom",     ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_label",ctypes.c_int),
        ("has_mask",   ctypes.c_int),
        ("mask_x",     ctypes.c_int),
        ("mask_y",     ctypes.c_int),
        ("mask_w",     ctypes.c_int),
        ("mask_h",     ctypes.c_int),
        ("mask_data",  ctypes.c_void_p),
        ("num_kpts",   ctypes.c_int),
        ("kpts",       ctypes.c_void_p),
        ("is_obb",     ctypes.c_int),
        ("cx",         ctypes.c_float),
        ("cy",         ctypes.c_float),
        ("ow",         ctypes.c_float),
        ("oh",         ctypes.c_float),
        ("angle",      ctypes.c_float),
    ]


class YiFrameStat(ctypes.Structure):
    _fields_ = [
        ("src_w",           ctypes.c_int),
        ("src_h",           ctypes.c_int),
        ("count",           ctypes.c_int),
        ("preprocess_ms",   ctypes.c_double),
        ("inference_ms",    ctypes.c_double),
        ("postprocess_ms",  ctypes.c_double),
        ("total_ms",        ctypes.c_double),
    ]


class AiMetadata(ctypes.Structure):
    _fields_ = [
        ("model_type",       ctypes.c_char * 64),
        ("normalization",    ctypes.c_char * 32),
        ("output_contract",  ctypes.c_char * 64),
        ("image_threshold",  ctypes.c_float),
        ("pixel_threshold",  ctypes.c_float),
        ("score_min",        ctypes.c_float),
        ("score_max",        ctypes.c_float),
        ("map_min",          ctypes.c_float),
        ("map_max",          ctypes.c_float),
        ("input_h",          ctypes.c_int),
        ("input_w",          ctypes.c_int),
    ]


class AiFrameStat(ctypes.Structure):
    _fields_ = [
        ("src_w",           ctypes.c_int),
        ("src_h",           ctypes.c_int),
        ("preprocess_ms",   ctypes.c_double),
        ("inference_ms",    ctypes.c_double),
        ("postprocess_ms",  ctypes.c_double),
        ("total_ms",        ctypes.c_double),
    ]


class AiResult(ctypes.Structure):
    _fields_ = [
        ("source_rgb",      ctypes.POINTER(ctypes.c_ubyte)),
        ("source_w",        ctypes.c_int),
        ("source_h",        ctypes.c_int),
        ("source_stride",   ctypes.c_int),
        ("overlay_rgb",     ctypes.POINTER(ctypes.c_ubyte)),
        ("overlay_w",       ctypes.c_int),
        ("overlay_h",       ctypes.c_int),
        ("overlay_stride",  ctypes.c_int),
        ("mask_rgb",        ctypes.POINTER(ctypes.c_ubyte)),
        ("mask_w",          ctypes.c_int),
        ("mask_h",          ctypes.c_int),
        ("mask_stride",     ctypes.c_int),
        ("normalized_map",  ctypes.POINTER(ctypes.c_float)),
        ("map_w",           ctypes.c_int),
        ("map_h",           ctypes.c_int),
        ("raw_score",       ctypes.c_float),
        ("normalized_score",ctypes.c_float),
        ("is_anomaly",      ctypes.c_int),
    ]


# ─────────────────────────────────────────────
# 测试结果记录
# ─────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id:  str
    name:     str
    passed:   bool
    detail:   str
    warn:     str = ""
    elapsed:  float = 0.0


@dataclass
class TestReport:
    results: List[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult):
        self.results.append(r)
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.case_id} {r.name}")
        print(f"        {r.detail}")
        if r.warn:
            print(f"        WARN: {r.warn}")

    @property
    def passed_count(self): return sum(1 for r in self.results if r.passed)
    @property
    def failed_count(self): return sum(1 for r in self.results if not r.passed)


# ─────────────────────────────────────────────
# DLL 加载工具
# ─────────────────────────────────────────────

def _utf8(p) -> bytes:
    return str(p).encode("utf-8")


def load_yolo_dll(dll_path: Path) -> ctypes.CDLL:
    dll = ctypes.CDLL(str(dll_path))
    dll.yi_create.argtypes   = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    dll.yi_create.restype    = ctypes.c_void_p
    dll.yi_infer_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(YiBox)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(YiFrameStat),
    ]
    dll.yi_infer_bgr.restype  = ctypes.c_int
    dll.yi_destroy.argtypes   = [ctypes.c_void_p]
    dll.yi_destroy.restype    = None
    dll.yi_last_error.argtypes = []
    dll.yi_last_error.restype  = ctypes.c_char_p
    return dll


def load_anomalib_dll(dll_path: Path) -> ctypes.CDLL:
    dll = ctypes.CDLL(str(dll_path))
    dll.ai_create.argtypes   = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float]
    dll.ai_create.restype    = ctypes.c_void_p
    dll.ai_get_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(AiMetadata)]
    dll.ai_get_metadata.restype  = ctypes.c_int
    dll.ai_infer_rgb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(AiResult)),
        ctypes.POINTER(AiFrameStat),
    ]
    dll.ai_infer_rgb.restype  = ctypes.c_int
    dll.ai_destroy.argtypes   = [ctypes.c_void_p]
    dll.ai_destroy.restype    = None
    dll.ai_last_error.argtypes = []
    dll.ai_last_error.restype  = ctypes.c_char_p
    return dll


# ─────────────────────────────────────────────
# YOLO 测试实现
# ─────────────────────────────────────────────

TASK_MAP = {
    "DET":  (0, "YI_DETECT"),
    "SEG":  (1, "YI_SEGMENT"),
    "POSE": (2, "YI_POSE"),
    "OBB":  (3, "YI_OBB"),
}


def run_yolo_case(
    report: TestReport,
    dll: ctypes.CDLL,
    case_id: str,
    task_name: str,
    engine: Path,
    images: list[Path],
    out_dir: Path,
) -> None:
    """单个 YOLO 任务完整测试：创建句柄 → 多图推理 → 销毁 → 错误路径"""
    task_int, _ = TASK_MAP[task_name]

    # ── SMK：句柄创建
    t0 = time.perf_counter()
    handle = None
    try:
        handle = dll.yi_create(_utf8(engine), task_int, 0.25, 0.45)
        elapsed = time.perf_counter() - t0
        if not handle:
            err = (dll.yi_last_error() or b"").decode("utf-8", errors="replace")
            report.add(CaseResult(case_id + "-CREATE", f"YOLO {task_name} 句柄创建", False,
                                   f"yi_create 返回 null，错误={err}"))
            return
        report.add(CaseResult(case_id + "-CREATE", f"YOLO {task_name} 句柄创建",
                               True, f"成功，耗时 {elapsed:.2f}s", elapsed=elapsed))
    except Exception as e:
        report.add(CaseResult(case_id + "-CREATE", f"YOLO {task_name} 句柄创建",
                               False, f"异常: {e}"))
        return

    try:
        for idx, img_path in enumerate(images):
            img_no = idx + 1
            bgr = imread_unicode(img_path)
            if bgr is None:
                report.add(CaseResult(f"{case_id}-IMG{img_no}", f"YOLO {task_name} 图{img_no} 读取",
                                       False, f"cv2.imread 失败：{img_path}"))
                continue

            bgr = np.ascontiguousarray(bgr)
            h, w = bgr.shape[:2]
            stride = bgr.strides[0]
            boxes_ptr  = ctypes.POINTER(YiBox)()
            count      = ctypes.c_int(0)
            stat       = YiFrameStat()

            t1 = time.perf_counter()
            rc = dll.yi_infer_bgr(
                handle,
                bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                w, h, stride,
                ctypes.byref(boxes_ptr),
                ctypes.byref(count),
                ctypes.byref(stat),
            )
            elapsed = time.perf_counter() - t1

            if rc != 0:
                err = (dll.yi_last_error() or b"").decode("utf-8", errors="replace")
                report.add(CaseResult(f"{case_id}-IMG{img_no}", f"YOLO {task_name} 图{img_no} 推理",
                                       False, f"yi_infer_bgr rc={rc}，错误={err}"))
                continue

            n = count.value
            warn = ""
            if n == 0:
                warn = "检测结果为 0，请确认图片与模型是否匹配"

            timing = (f"pre={stat.preprocess_ms:.1f}ms "
                      f"infer={stat.inference_ms:.1f}ms "
                      f"post={stat.postprocess_ms:.1f}ms "
                      f"total={stat.total_ms:.1f}ms")
            detail = f"检测到 {n} 个目标，{timing}，图片={img_path.name}"

            # 保存可视化结果
            vis = bgr.copy()
            for i in range(n):
                box = boxes_ptr[i]
                cv2.rectangle(vis, (int(box.left), int(box.top)),
                              (int(box.right), int(box.bottom)), (0, 255, 0), 2)
            out_path = out_dir / f"{case_id}_img{img_no}_{img_path.stem}.jpg"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), vis)

            report.add(CaseResult(f"{case_id}-IMG{img_no}", f"YOLO {task_name} 图{img_no} 推理",
                                   True, detail, warn=warn, elapsed=elapsed))

        # ── 模型缓存验证：同引擎、换图，不重建句柄
        # 再推理一次（第一张图），确认句柄可复用
        bgr2 = imread_unicode(images[0])
        if bgr2 is not None:
            bgr2 = np.ascontiguousarray(bgr2)
            h2, w2 = bgr2.shape[:2]
            boxes_ptr2 = ctypes.POINTER(YiBox)()
            count2     = ctypes.c_int(0)
            stat2      = YiFrameStat()
            t2 = time.perf_counter()
            rc2 = dll.yi_infer_bgr(
                handle,
                bgr2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                w2, h2, bgr2.strides[0],
                ctypes.byref(boxes_ptr2),
                ctypes.byref(count2),
                ctypes.byref(stat2),
            )
            e2 = time.perf_counter() - t2
            passed2 = (rc2 == 0)
            report.add(CaseResult(
                f"{case_id}-CACHE",
                f"YOLO {task_name} 句柄复用（模型缓存）",
                passed2,
                f"复用句柄再推理，rc={rc2}，total={stat2.total_ms:.1f}ms，目标={count2.value}",
                elapsed=e2,
            ))

    finally:
        dll.yi_destroy(handle)
        report.add(CaseResult(f"{case_id}-DESTROY", f"YOLO {task_name} 句柄销毁",
                               True, "yi_destroy 完成，无崩溃"))

    # ── 错误路径：无效引擎
    bad_handle = dll.yi_create(b"invalid/path/to/non_existent.trt", task_int, 0.25, 0.45)
    if bad_handle:
        dll.yi_destroy(bad_handle)
        report.add(CaseResult(f"{case_id}-ERRPATH", f"YOLO {task_name} 错误路径",
                               False, "无效路径应返回 null，但返回了非 null 句柄"))
    else:
        err = (dll.yi_last_error() or b"").decode("utf-8", errors="replace")
        report.add(CaseResult(f"{case_id}-ERRPATH", f"YOLO {task_name} 错误路径",
                               True, f"返回 null 且有错误信息：{err[:120]}"))


# ─────────────────────────────────────────────
# Anomalib 测试实现
# ─────────────────────────────────────────────

def run_anomalib_case(
    report: TestReport,
    dll: ctypes.CDLL,
    engine: Path,
    meta_path: Path,
    abnormal_images: list[Path],
    out_dir: Path,
) -> None:
    # ── 句柄创建
    t0 = time.perf_counter()
    handle = dll.ai_create(_utf8(engine), _utf8(meta_path), b"Auto", -1.0, -1.0)
    elapsed = time.perf_counter() - t0
    if not handle:
        err = (dll.ai_last_error() or b"").decode("utf-8", errors="replace")
        report.add(CaseResult("ANO-CREATE", "Anomalib PatchCore 句柄创建", False,
                               f"ai_create 失败: {err}"))
        return
    report.add(CaseResult("ANO-CREATE", "Anomalib PatchCore 句柄创建",
                           True, f"成功，耗时 {elapsed:.2f}s"))

    try:
        # ── Metadata 读取
        meta = AiMetadata()
        rc = dll.ai_get_metadata(handle, ctypes.byref(meta))
        if rc != 0:
            err = (dll.ai_last_error() or b"").decode("utf-8", errors="replace")
            report.add(CaseResult("ANO-META", "Anomalib metadata 读取", False,
                                   f"ai_get_metadata 失败: {err}"))
        else:
            model_type  = meta.model_type.decode("utf-8", errors="replace").rstrip("\x00")
            norm        = meta.normalization.decode("utf-8", errors="replace").rstrip("\x00")
            contract    = meta.output_contract.decode("utf-8", errors="replace").rstrip("\x00")
            detail = (f"model={model_type} norm={norm} contract={contract} "
                      f"img_thr={meta.image_threshold:.3f} pix_thr={meta.pixel_threshold:.3f} "
                      f"input={meta.input_w}x{meta.input_h}")
            report.add(CaseResult("ANO-META", "Anomalib metadata 读取", True, detail))

        # ── 多张异常图推理（取前 5 张）
        test_imgs = abnormal_images[:5]
        for idx, img_path in enumerate(test_imgs):
            img_no = idx + 1
            bgr = imread_unicode(img_path)
            if bgr is None:
                report.add(CaseResult(f"ANO-IMG{img_no}", f"Anomalib 异常图{img_no} 读取",
                                       False, f"cv2.imread 失败：{img_path}"))
                continue
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            h, w = rgb.shape[:2]

            result_ptr = ctypes.POINTER(AiResult)()
            stat       = AiFrameStat()
            t1 = time.perf_counter()
            rc = dll.ai_infer_rgb(
                handle,
                rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                w, h, rgb.strides[0],
                ctypes.byref(result_ptr),
                ctypes.byref(stat),
            )
            elapsed = time.perf_counter() - t1

            if rc != 0:
                err = (dll.ai_last_error() or b"").decode("utf-8", errors="replace")
                report.add(CaseResult(f"ANO-IMG{img_no}", f"Anomalib 异常图{img_no} 推理",
                                       False, f"ai_infer_rgb rc={rc}: {err}"))
                continue

            res = result_ptr.contents
            status_str = "NG" if res.is_anomaly else "OK"
            timing = (f"pre={stat.preprocess_ms:.1f}ms "
                      f"infer={stat.inference_ms:.1f}ms "
                      f"total={stat.total_ms:.1f}ms")
            warn = ""
            if not res.is_anomaly:
                warn = f"异常图判为 OK（score={res.normalized_score:.3f}），请核查阈值与 output_contract"

            detail = (f"{status_str} score={res.normalized_score:.3f} "
                      f"raw={res.raw_score:.3f} {timing} 图={img_path.name}")

            # 保存三画布输出
            out_dir.mkdir(parents=True, exist_ok=True)
            if res.overlay_rgb and res.overlay_w > 0:
                raw_ovl = ctypes.string_at(res.overlay_rgb, res.overlay_h * res.overlay_stride)
                ovl = np.frombuffer(raw_ovl, dtype=np.uint8).reshape(res.overlay_h, res.overlay_stride)
                ovl = ovl[:, :res.overlay_w * 3].reshape(res.overlay_h, res.overlay_w, 3)
                cv2.imwrite(str(out_dir / f"ano_img{img_no}_overlay.jpg"),
                            cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))
            if res.mask_rgb and res.mask_w > 0:
                raw_mask = ctypes.string_at(res.mask_rgb, res.mask_h * res.mask_stride)
                mask = np.frombuffer(raw_mask, dtype=np.uint8).reshape(res.mask_h, res.mask_stride)
                mask = mask[:, :res.mask_w * 3].reshape(res.mask_h, res.mask_w, 3)
                cv2.imwrite(str(out_dir / f"ano_img{img_no}_mask.jpg"),
                            cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

            report.add(CaseResult(f"ANO-IMG{img_no}", f"Anomalib 异常图{img_no} 推理",
                                   True, detail, warn=warn, elapsed=elapsed))

        # ── 模型缓存验证（换图后复用句柄）
        if test_imgs:
            bgr_r = imread_unicode(test_imgs[0])
            if bgr_r is not None:
                rgb_r = np.ascontiguousarray(cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB))
                rp2   = ctypes.POINTER(AiResult)()
                st2   = AiFrameStat()
                t3 = time.perf_counter()
                rc2 = dll.ai_infer_rgb(
                    handle,
                    rgb_r.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    rgb_r.shape[1], rgb_r.shape[0], rgb_r.strides[0],
                    ctypes.byref(rp2), ctypes.byref(st2),
                )
                e3 = time.perf_counter() - t3
                report.add(CaseResult(
                    "ANO-CACHE", "Anomalib 句柄复用（模型缓存）",
                    rc2 == 0,
                    f"复用句柄再推理 rc={rc2}，total={st2.total_ms:.1f}ms",
                    elapsed=e3,
                ))

    finally:
        dll.ai_destroy(handle)
        report.add(CaseResult("ANO-DESTROY", "Anomalib 句柄销毁",
                               True, "ai_destroy 完成，无崩溃"))

    # ── 错误路径：无效 engine
    bad_h = dll.ai_create(b"bad/path.trt", b"bad/meta.json", b"Auto", -1.0, -1.0)
    if bad_h:
        dll.ai_destroy(bad_h)
        report.add(CaseResult("ANO-ERRPATH", "Anomalib 错误路径",
                               False, "无效路径应返回 null"))
    else:
        err = (dll.ai_last_error() or b"").decode("utf-8", errors="replace")
        report.add(CaseResult("ANO-ERRPATH", "Anomalib 错误路径",
                               True, f"正确返回 null：{err[:120]}"))

    # ── 阈值覆盖验证：用极高阈值（2.0）迫使全 OK
    h2 = dll.ai_create(_utf8(engine), _utf8(meta_path), b"Auto", 2.0, 2.0)
    if h2:
        bgr_v = imread_unicode(abnormal_images[0])
        if bgr_v is not None:
            rgb_v = np.ascontiguousarray(cv2.cvtColor(bgr_v, cv2.COLOR_BGR2RGB))
            rp_v  = ctypes.POINTER(AiResult)()
            st_v  = AiFrameStat()
            rc_v  = dll.ai_infer_rgb(
                h2,
                rgb_v.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                rgb_v.shape[1], rgb_v.shape[0], rgb_v.strides[0],
                ctypes.byref(rp_v), ctypes.byref(st_v),
            )
            if rc_v == 0:
                r_v = rp_v.contents
                forced_ok = (not r_v.is_anomaly)
                report.add(CaseResult(
                    "ANO-THR",
                    "Anomalib 阈值覆盖验证（image_thr=2.0 应全 OK）",
                    forced_ok,
                    f"is_anomaly={r_v.is_anomaly} score={r_v.normalized_score:.3f}",
                    warn="" if forced_ok else "高阈值仍判 NG，阈值覆盖逻辑异常",
                ))
        dll.ai_destroy(h2)


# ─────────────────────────────────────────────
# 停止重启循环测试
# ─────────────────────────────────────────────

def run_restart_loop(
    report: TestReport,
    dll: ctypes.CDLL,
    engine: Path,
    task_int: int,
    task_name: str,
    img_path: Path,
    loops: int = 3,
) -> None:
    """模拟 GUI 中多次 开始→停止→开始 操作"""
    bgr = imread_unicode(img_path)
    if bgr is None:
        report.add(CaseResult(f"LOOP-{task_name}", f"停止重启循环 {task_name}",
                               False, f"无法读图: {img_path}"))
        return
    bgr = np.ascontiguousarray(bgr)

    all_ok = True
    for i in range(loops):
        h = dll.yi_create(_utf8(engine), task_int, 0.25, 0.45)
        if not h:
            err = (dll.yi_last_error() or b"").decode("utf-8", errors="replace")
            report.add(CaseResult(f"LOOP-{task_name}-{i+1}", f"循环{i+1} 创建",
                                   False, f"句柄创建失败: {err}"))
            all_ok = False
            continue
        boxes_ptr = ctypes.POINTER(YiBox)()
        count     = ctypes.c_int(0)
        stat      = YiFrameStat()
        rc = dll.yi_infer_bgr(
            h,
            bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            bgr.shape[1], bgr.shape[0], bgr.strides[0],
            ctypes.byref(boxes_ptr), ctypes.byref(count), ctypes.byref(stat),
        )
        dll.yi_destroy(h)
        if rc != 0:
            err = (dll.yi_last_error() or b"").decode("utf-8", errors="replace")
            report.add(CaseResult(f"LOOP-{task_name}-{i+1}", f"循环{i+1} 推理",
                                   False, f"rc={rc}: {err}"))
            all_ok = False

    if all_ok:
        report.add(CaseResult(f"LOOP-{task_name}", f"停止重启循环 {task_name} ×{loops}",
                               True, f"连续 {loops} 次创建/推理/销毁，无崩溃"))


# ─────────────────────────────────────────────
# 生成 Markdown 报告
# ─────────────────────────────────────────────

def write_report(report: TestReport, out_path: Path, test_time: str) -> None:
    lines = [
        "# TRTForge DLL API 自动化冒烟测试报告",
        "",
        "## 测试结论",
        f"- **结论**：{'通过' if report.failed_count == 0 else '部分通过' if report.passed_count > 0 else '不通过'}",
        f"- **测试时间**：{test_time}",
        f"- **通过 / 失败**：{report.passed_count} / {report.failed_count}",
        f"- **YOLO DLL**：{ASSET['yolo_dll']}",
        f"- **Anomalib DLL**：{ASSET['anomalib_dll']}",
        "",
        "## 覆盖范围",
        "- YOLO DET / SEG / POSE / OBB：句柄创建、多图推理、模型缓存复用、错误路径",
        "- Anomalib PatchCore：句柄创建、metadata 读取、多张异常图推理、模型缓存、阈值覆盖、错误路径",
        "- 停止重启循环：DET / OBB × 3 轮",
        "",
        "## 详细结果",
        "",
        "| ID | 用例名称 | 结果 | 耗时(s) | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        detail = r.detail.replace("|", "｜")
        if r.warn:
            detail += f"  ⚠️ {r.warn.replace('|', '｜')}"
        elapsed_str = f"{r.elapsed:.2f}" if r.elapsed > 0 else "-"
        lines.append(f"| {r.case_id} | {r.name} | {status} | {elapsed_str} | {detail} |")

    if report.failed_count > 0:
        lines += ["", "## 失败项汇总", ""]
        for r in report.results:
            if not r.passed:
                lines.append(f"- **{r.case_id}** {r.name}：{r.detail}")

    lines += [
        "",
        "## 关键风险",
        "- Anomalib 只提供 abnormal 图，OK 场景（normal 图）未覆盖",
        "- 纯 UI 交互（画布缩放、进度弹窗、停止按钮状态）需配合方案 A 手动验证",
        "",
        "## 建议下一步",
        "- 提供 `anomalib/bottle/test/good/*.png` 以补充 OK 场景测试",
        "- 执行方案 A（日志驱动半手动测试）覆盖 UI 交互层",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{out_path}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="TRTForge DLL API 自动化冒烟测试")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "trtexec/gui/tests/test_output",
                        help="测试输出目录（截图、报告）")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_time = time.strftime("%Y-%m-%d %H:%M:%S")
    report    = TestReport()

    print("=" * 60)
    print("TRTForge DLL API 自动化冒烟测试")
    print(f"时间：{test_time}")
    print(f"输出：{out_dir}")
    print("=" * 60)

    # ── 切换工作目录到 bin，确保依赖 DLL 可被找到
    os.chdir(str(BIN))

    # ── 加载 DLL
    print("\n[1/3] 加载 DLL...")
    try:
        yolo_dll = load_yolo_dll(ASSET["yolo_dll"])
        print(f"  yolo_infer.dll  已加载：{ASSET['yolo_dll']}")
    except Exception as e:
        print(f"  FATAL: yolo_infer.dll 加载失败: {e}")
        return 1

    try:
        ano_dll = load_anomalib_dll(ASSET["anomalib_dll"])
        print(f"  anomalib_infer.dll 已加载：{ASSET['anomalib_dll']}")
    except Exception as e:
        print(f"  FATAL: anomalib_infer.dll 加载失败: {e}")
        return 1

    # ── YOLO 测试
    print("\n[2/3] YOLO 推理测试...")

    yolo_cases = [
        ("INF-01", "DET",  ASSET["det_engine"],  [ASSET["bus_img"],    ASSET["zidane_img"]]),
        ("INF-02", "SEG",  ASSET["seg_engine"],  [ASSET["bus_img"],    ASSET["zidane_img"]]),
        ("INF-03", "POSE", ASSET["pose_engine"], [ASSET["bus_img"],    ASSET["zidane_img"]]),
        ("INF-04", "OBB",  ASSET["obb_engine"],  [ASSET["plane_img"],  ASSET["plane2_img"]]),
    ]

    for case_id, task_name, engine, images in yolo_cases:
        print(f"\n  -- {case_id} YOLO {task_name} --")
        try:
            run_yolo_case(report, yolo_dll, case_id, task_name, engine, images,
                          out_dir / "yolo_vis")
        except Exception:
            report.add(CaseResult(f"{case_id}-CRASH", f"YOLO {task_name} 执行异常",
                                   False, traceback.format_exc()[-300:]))

    # ── YOLO 停止重启循环
    print("\n  -- LOOP DET/OBB 停止重启 ×3 --")
    run_restart_loop(report, yolo_dll, ASSET["det_engine"], 0, "DET", ASSET["bus_img"])
    run_restart_loop(report, yolo_dll, ASSET["obb_engine"], 3, "OBB", ASSET["plane_img"])

    # ── Anomalib 测试
    print("\n[3/3] Anomalib PatchCore 测试...")
    abnormal_pngs = sorted(ASSET["abnormal_dir"].glob("*.png"))
    if not abnormal_pngs:
        print("  WARN: 找不到 abnormal PNG，跳过 Anomalib 测试")
    else:
        try:
            run_anomalib_case(report, ano_dll,
                              ASSET["anomalib_engine"], ASSET["anomalib_meta"],
                              abnormal_pngs, out_dir / "anomalib_vis")
        except Exception:
            report.add(CaseResult("ANO-CRASH", "Anomalib 执行异常",
                                   False, traceback.format_exc()[-300:]))

    # ── 输出统计
    print("\n" + "=" * 60)
    print(f"测试完成：共 {len(report.results)} 项，"
          f"通过 {report.passed_count}，失败 {report.failed_count}")

    # ── 写报告
    report_path = out_dir / f"dll_api_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    write_report(report, report_path, test_time)

    return 0 if report.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
