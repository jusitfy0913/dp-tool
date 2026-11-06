# -*- coding: utf-8 -*-
"""onnx_infer.dll 的 C API 验收脚本（不需要启动 Web 服务）。

为什么要单独测 DLL：
    `tools/test_backend_consistency.py` 走的是 Python 包装层（infer.py），
    它验证「分派对不对」；这里直接用 ctypes 打 C ABI，验证「DLL 本身守不守约定」：
    错误必须返回码而不是崩溃、失败原因必须可读、结构体内存必须按文档有效。

覆盖：
1. 加载与版本：DLL 可加载，provider / ort_version 可读；
2. 错误路径：空路径、文件不存在、任务与模型不匹配、空句柄，都要返回失败码且不崩溃；
3. 四个任务：detect / segment / pose / obb 各跑一次，检查掩码、关键点、旋转框字段；
4. 三个异常模型：PatchCore / PaDiM / EfficientAd，检查分数、归一化图范围、判定；
5. 耗时字段：preprocess / inference / postprocess / total 必须为正且大致自洽。

用法：
    python 6-web/backend/tools/test_onnx_infer_dll.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
BIN = REPO / "4-qt-studio" / "bin"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []

OI_DETECT, OI_SEGMENT, OI_POSE, OI_OBB = 0, 1, 2, 3


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name} {detail}".rstrip())
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}".rstrip())


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name}（{reason}）")


# ---------------------------------------------------------------------------
# C ABI 结构定义（与 onnx_infer_c.h 逐字段一致）
# ---------------------------------------------------------------------------

class OiBox(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_float), ("top", ctypes.c_float),
        ("right", ctypes.c_float), ("bottom", ctypes.c_float),
        ("confidence", ctypes.c_float), ("class_label", ctypes.c_int),
        ("has_mask", ctypes.c_int),
        ("mask_x", ctypes.c_int), ("mask_y", ctypes.c_int),
        ("mask_w", ctypes.c_int), ("mask_h", ctypes.c_int),
        ("mask_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("num_kpts", ctypes.c_int), ("kpts", ctypes.POINTER(ctypes.c_float)),
        ("is_obb", ctypes.c_int),
        ("cx", ctypes.c_float), ("cy", ctypes.c_float),
        ("ow", ctypes.c_float), ("oh", ctypes.c_float), ("angle", ctypes.c_float),
    ]


class OiFrameStat(ctypes.Structure):
    _fields_ = [
        ("src_w", ctypes.c_int), ("src_h", ctypes.c_int),
        ("count", ctypes.c_int),
        ("preprocess_ms", ctypes.c_double), ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double), ("total_ms", ctypes.c_double),
    ]


class OiAnomalyResult(ctypes.Structure):
    _fields_ = [
        ("source_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("source_w", ctypes.c_int), ("source_h", ctypes.c_int), ("source_stride", ctypes.c_int),
        ("overlay_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("overlay_w", ctypes.c_int), ("overlay_h", ctypes.c_int), ("overlay_stride", ctypes.c_int),
        ("mask_rgb", ctypes.POINTER(ctypes.c_ubyte)),
        ("mask_w", ctypes.c_int), ("mask_h", ctypes.c_int), ("mask_stride", ctypes.c_int),
        ("normalized_map", ctypes.POINTER(ctypes.c_float)),
        ("map_w", ctypes.c_int), ("map_h", ctypes.c_int),
        ("raw_score", ctypes.c_float), ("normalized_score", ctypes.c_float),
        ("is_anomaly", ctypes.c_int),
    ]


class OiInfo(ctypes.Structure):
    _fields_ = [
        ("provider", ctypes.c_char * 32), ("ort_version", ctypes.c_char * 32),
        ("task", ctypes.c_int), ("input_w", ctypes.c_int), ("input_h", ctypes.c_int),
        ("num_classes", ctypes.c_int), ("num_keypoints", ctypes.c_int),
    ]


def load() -> ctypes.WinDLL:
    dll = ctypes.WinDLL(str(BIN / "onnx_infer.dll"))
    dll.oi_create_yolo.restype = ctypes.c_void_p
    dll.oi_create_yolo.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    dll.oi_create_anomalib.restype = ctypes.c_void_p
    dll.oi_create_anomalib.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float,
    ]
    dll.oi_infer_yolo.restype = ctypes.c_int
    dll.oi_infer_yolo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.POINTER(OiBox)), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(OiFrameStat),
    ]
    dll.oi_infer_anomalib.restype = ctypes.c_int
    dll.oi_infer_anomalib.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.POINTER(OiAnomalyResult)),
        ctypes.POINTER(OiFrameStat),
    ]
    dll.oi_get_info.restype = ctypes.c_int
    dll.oi_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(OiInfo)]
    dll.oi_destroy.restype = None
    dll.oi_destroy.argtypes = [ctypes.c_void_p]
    dll.oi_last_error.restype = ctypes.c_char_p
    dll.oi_last_error.argtypes = []
    return dll


def last_error(dll) -> str:
    raw = dll.oi_last_error()
    return raw.decode("utf-8", "replace") if raw else ""


def to_bytes(image: np.ndarray) -> tuple[np.ndarray, ctypes.POINTER(ctypes.c_ubyte), int]:
    image = np.ascontiguousarray(image)
    pointer = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    return image, pointer, image.shape[1] * image.shape[2]


# ---------------------------------------------------------------------------
# 1. 错误路径
# ---------------------------------------------------------------------------

def test_errors(dll, detect_model: Path) -> None:
    handle = dll.oi_create_yolo(None, OI_DETECT, 0.25, 0.45)
    check("错误：空路径返回 nullptr", not handle and last_error(dll) != "", last_error(dll))

    handle = dll.oi_create_yolo(str(Path("D:/不存在/model.onnx")).encode("utf-8"), OI_DETECT, 0.25, 0.45)
    check("错误：文件不存在返回 nullptr", not handle and "不存在" in last_error(dll), last_error(dll))

    # 任务与模型不匹配：检测模型当分割用，掩码通道不存在，必须在创建阶段就失败
    if detect_model.is_file():
        handle = dll.oi_create_yolo(str(detect_model).encode("utf-8"), OI_SEGMENT, 0.25, 0.45)
        ok = not handle and last_error(dll) != ""
        check("错误：任务与模型不匹配在创建阶段报错", ok, last_error(dll))
        if handle:
            dll.oi_destroy(handle)
    else:
        skip("错误：任务与模型不匹配", "缺检测模型")

    info = OiInfo()
    code = dll.oi_get_info(None, ctypes.byref(info))
    check("错误：空句柄取信息返回失败码", code != 0, f"code={code}")

    # 空句柄销毁必须是空操作，不能让后端在无关路径上崩掉
    dll.oi_destroy(None)
    check("错误：空句柄销毁不崩溃", True)


# ---------------------------------------------------------------------------
# 2. 四个任务
# ---------------------------------------------------------------------------

def run_yolo_case(dll, task: int, model: Path, image: np.ndarray, label: str) -> dict | None:
    if not model.is_file():
        skip(f"YOLO {label}", f"缺模型 {model.name}")
        return None

    handle = dll.oi_create_yolo(str(model).encode("utf-8"), task, 0.25, 0.45)
    if not handle:
        check(f"YOLO {label}：创建句柄", False, last_error(dll))
        return None

    try:
        info = OiInfo()
        code = dll.oi_get_info(handle, ctypes.byref(info))
        provider = info.provider.decode("utf-8", "replace")
        version = info.ort_version.decode("utf-8", "replace")
        check(
            f"YOLO {label}：信息可读",
            code == 0 and provider.strip().lower() in ("cuda", "cpu") and version and info.input_w > 0,
            f"provider={provider} ort={version} input={info.input_w}x{info.input_h}",
        )

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        holder, pointer, stride = to_bytes(bgr)
        boxes = ctypes.POINTER(OiBox)()
        count = ctypes.c_int(0)
        stat = OiFrameStat()
        code = dll.oi_infer_yolo(
            handle, pointer, holder.shape[1], holder.shape[0], stride,
            ctypes.byref(boxes), ctypes.byref(count), ctypes.byref(stat),
        )
        check(f"YOLO {label}：推理成功", code == 0, last_error(dll))
        if code != 0:
            return None

        items = [boxes[i] for i in range(count.value)]
        spans_ok = all(
            -1.0 <= b.left <= b.right <= holder.shape[1] + 1 and
            -1.0 <= b.top <= b.bottom <= holder.shape[0] + 1
            for b in items
        )
        check(f"YOLO {label}：框在图像范围内", spans_ok, f"count={count.value}")
        check(
            f"YOLO {label}：耗时字段自洽",
            stat.total_ms > 0 and stat.inference_ms > 0 and
            stat.total_ms + 1e-3 >= stat.preprocess_ms + stat.inference_ms and
            stat.src_w == holder.shape[1],
            f"{stat.preprocess_ms:.1f}/{stat.inference_ms:.1f}/{stat.postprocess_ms:.1f}/{stat.total_ms:.1f}ms",
        )

        if task == OI_SEGMENT:
            with_mask = [b for b in items if b.has_mask]
            check(
                f"YOLO {label}：框带掩码",
                bool(items) and len(with_mask) == len(items) and
                all(b.mask_w > 0 and b.mask_h > 0 for b in with_mask),
                f"{len(with_mask)}/{len(items)}",
            )
        if task == OI_POSE:
            check(
                f"YOLO {label}：关键点数量一致",
                bool(items) and all(
                    b.num_kpts > 0 and b.num_kpts == info.num_keypoints for b in items
                ),
                f"num_kpts={info.num_keypoints}",
            )
        if task == OI_OBB:
            check(
                f"YOLO {label}：旋转框字段有效",
                bool(items) and all(
                    b.is_obb == 1 and abs(b.angle) <= np.pi + 1e-3 and b.ow > 0 and b.oh > 0
                    for b in items
                ),
                f"count={count.value}",
            )

        return {
            "count": count.value,
            "preprocess_ms": stat.preprocess_ms,
            "inference_ms": stat.inference_ms,
            "postprocess_ms": stat.postprocess_ms,
            "total_ms": stat.total_ms,
            "provider": info.provider.decode("utf-8", "replace"),
            "input": (info.input_w, info.input_h),
        }
    finally:
        dll.oi_destroy(handle)


# ---------------------------------------------------------------------------
# 3. 三个异常模型
# ---------------------------------------------------------------------------

def run_anomaly_case(dll, model: Path, metadata: Path, image: np.ndarray, label: str) -> dict | None:
    if not (model.is_file() and metadata.is_file()):
        skip(f"Anomalib {label}", "缺模型或 metadata")
        return None

    handle = dll.oi_create_anomalib(
        str(model).encode("utf-8"), str(metadata).encode("utf-8"),
        label.encode("utf-8"), -1.0, -1.0,
    )
    if not handle:
        check(f"Anomalib {label}：创建句柄", False, last_error(dll))
        return None

    try:
        info = OiInfo()
        dll.oi_get_info(handle, ctypes.byref(info))
        rgb = np.ascontiguousarray(cv2.resize(image, (256, 256)))
        holder, pointer, stride = to_bytes(rgb)
        result = ctypes.POINTER(OiAnomalyResult)()
        stat = OiFrameStat()
        code = dll.oi_infer_anomalib(
            handle, pointer, holder.shape[1], holder.shape[0], stride,
            ctypes.byref(result), ctypes.byref(stat),
        )
        check(f"Anomalib {label}：推理成功", code == 0, last_error(dll))
        if code != 0:
            return None

        item = result.contents
        values = np.ctypeslib.as_array(item.normalized_map, shape=(item.map_h * item.map_w,))
        check(
            f"Anomalib {label}：归一化图值域 [0,1]",
            item.map_w > 0 and item.map_h > 0 and float(values.min()) >= -1e-6
            and float(values.max()) <= 1.0 + 1e-6,
            f"{item.map_w}x{item.map_h} [{values.min():.3f},{values.max():.3f}]",
        )
        check(
            f"Anomalib {label}：三张渲染图有效",
            item.source_w > 0 and item.overlay_w > 0 and item.mask_stride >= item.mask_w * 3
            and item.source_rgb and item.overlay_rgb and item.mask_rgb,
            f"src={item.source_w}x{item.source_h} mask={item.mask_w}x{item.mask_h}",
        )
        check(
            f"Anomalib {label}：分数与判定自洽",
            np.isfinite(item.raw_score) and 0.0 <= item.normalized_score <= 1.0
            and int(item.is_anomaly) in (0, 1),
            f"score={item.raw_score:.4f} norm={item.normalized_score:.3f} anomaly={item.is_anomaly}",
        )
        return {
            "score": float(item.raw_score),
            "normalized": float(item.normalized_score),
            "is_anomaly": int(item.is_anomaly),
            "map": (item.map_w, item.map_h),
            "total_ms": stat.total_ms,
            "provider": info.provider.decode("utf-8", "replace"),
            "input": (info.input_w, info.input_h),
        }
    finally:
        dll.oi_destroy(handle)


def main() -> int:
    dll_path = BIN / "onnx_infer.dll"
    if not dll_path.is_file():
        print(f"找不到 {dll_path}，请先运行 scripts/build_infer_dlls.ps1 -Targets onnx_infer")
        return 3

    dll = load()
    print(f"已加载 {dll_path}")

    detect_model = REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd.onnx"
    probe = REPO / "6-web/runtime/export_probe"
    seg_model = probe / "yolov8n-seg.transd.onnx"
    pose_model = probe / "yolov8n-pose.transd.onnx"
    obb_model = probe / "yolov8n-obb.transd.onnx"

    image_path = REPO / "6-web/runtime/smoke_batch/in/bus.jpg"
    if not image_path.is_file():
        image_path = REPO / "6-web/runtime/smoke_batch/in/zidane.jpg"
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    print(f"测试图像 {image_path.name} {image.shape[1]}x{image.shape[0]}")

    test_errors(dll, detect_model)

    timings: dict[str, dict] = {}
    # OBB 模型是 DOTA 权重，bus.jpg 里没有它认识的类别，单给它一张飞机图，
    # 否则「旋转框字段有效」会因为 0 个框而永远无法验证。
    obb_image_path = REPO / "6-web/runtime/smoke_batch/in/plane2.jpg"
    obb_image = image
    if obb_image_path.is_file():
        obb_image = cv2.cvtColor(cv2.imread(str(obb_image_path)), cv2.COLOR_BGR2RGB)

    for task, model, label, sample in (
        (OI_DETECT, detect_model, "detect", image),
        (OI_SEGMENT, seg_model, "segment", image),
        (OI_POSE, pose_model, "pose", image),
        (OI_OBB, obb_model, "obb", obb_image),
    ):
        result = run_yolo_case(dll, task, model, sample, label)
        if result:
            timings[label] = result

    anom_root = REPO / "2-python-anomaly/launcher/results"
    for label, rel in (
        ("PatchCore", "Patchcore/web_smoke2/v0/weights/lightning"),
        ("PaDiM", "Padim/cmp_padim/v0/weights/lightning"),
        ("EfficientAd", "EfficientAd/cmp_efficientad/v0/weights/lightning"),
    ):
        folder = anom_root / rel
        result = run_anomaly_case(
            dll, folder / "model.onnx", folder / "metadata.json", image, label
        )
        if result:
            timings[f"anomalib-{label}"] = result

    print()
    if timings:
        print("耗时参考（毫秒）")
        for name, item in timings.items():
            if "preprocess_ms" in item:
                print(
                    f"  {name:<18} 预处理 {item['preprocess_ms']:6.2f}  "
                    f"推理 {item['inference_ms']:7.2f}  后处理 {item['postprocess_ms']:6.2f}  "
                    f"合计 {item['total_ms']:7.2f}  输入 {item['input'][0]}x{item['input'][1]}"
                    f"  provider={item['provider']}"
                )
            else:
                print(
                    f"  {name:<18} 合计 {item['total_ms']:7.2f}  "
                    f"输入 {item['input'][0]}x{item['input'][1]}"
                    f"  score={item['score']:.4f}  anomaly={item['is_anomaly']}"
                )
    print()
    print(f"通过 {len(PASSED)}，失败 {len(FAILED)}，跳过 {len(SKIPPED)}")
    if FAILED:
        for name in FAILED:
            print(f"  [FAIL] {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
