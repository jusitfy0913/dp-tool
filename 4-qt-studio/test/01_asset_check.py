# -*- coding: utf-8 -*-
"""
TRTForge 测试第一步：资产完整性与 metadata 契约验证
无需 GPU / DLL，纯 Python 运行。
运行方式：
    python 4-qt-studio/test/01_asset_check.py
"""
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # dp-tool/

ASSETS = {
    "TRTForge.exe":          BASE / "4-qt-studio/bin/TRTForge.exe",
    "yolo_infer.dll":        BASE / "4-qt-studio/bin/yolo_infer.dll",
    "anomalib_infer.dll":    BASE / "4-qt-studio/bin/anomalib_infer.dll",
    "labels_coco.json":      BASE / "4-qt-studio/bin/configs/labels_coco.json",
    "labels_obb_dota_v1.json": BASE / "4-qt-studio/bin/configs/labels_obb_dota_v1.json",
    "YOLO DET .trt":         BASE / "trtmodel/trtexcemodel/yolov8-detection.trans.trt",
    "YOLO SEG .trt":         BASE / "trtmodel/trtexcemodel/yolov8-seg.transd.trt",
    "YOLO OBB .trt":         BASE / "trtmodel/trtexcemodel/yolov8n-obb.transd.trt",
    "YOLO POSE .trt":        BASE / "trtmodel/trtexcemodel/yolov8s-pose.transd.trt",
    "test image bus.jpg":    BASE / "5-reference/YOLOv8-det/workspace/images/bus.jpg",
    "test image zidane.jpg": BASE / "5-reference/YOLOv8-det/workspace/images/zidane.jpg",
    "OBB image plane.jpeg":  BASE / "5-reference/YOLOv8_obb/normal/plane.jpeg",
    "OBB image plane2.jpg":  BASE / "5-reference/YOLOv8_obb/normal/plane2.jpg",
    "Anomalib model.trt":    BASE / "results/Patchcore/demo_5/latest/weights/lightning/model.trt",
    "Anomalib model.onnx":   BASE / "results/Patchcore/demo_5/latest/weights/lightning/model.onnx",
    "Anomalib metadata.json":BASE / "results/Patchcore/demo_5/latest/weights/lightning/metadata.json",
}

ANOMALIB_ABNORMAL_DIR = BASE / "2-python-anomaly/anomalib-main/datasets/bottle/test/abnormal"

REQUIRED_METADATA_FIELDS = [
    "schema", "model_type", "input_size", "normalization",
    "image_threshold", "pixel_threshold", "output_contract",
]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

results = []


def check(label, ok, detail=""):
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append((label, ok))
    return ok


print("=" * 60)
print("TRTForge 资产完整性检查")
print("=" * 60)

print("\n[1] 核心文件存在性")
for name, path in ASSETS.items():
    sz = path.stat().st_size if path.exists() else 0
    check(name, path.exists(), f"大小: {sz:,} bytes" if path.exists() else str(path))

print("\n[2] Anomalib 异常图目录")
abnormal_imgs = sorted(ANOMALIB_ABNORMAL_DIR.glob("*.png")) + \
                sorted(ANOMALIB_ABNORMAL_DIR.glob("*.jpg"))
# 仅使用纯 ASCII 文件名的图片（避免乱码文件名导致 TRTForge 中文路径问题）
ascii_imgs = [p for p in abnormal_imgs if all(ord(c) < 128 for c in p.name)]
check("abnormal 目录存在", ANOMALIB_ABNORMAL_DIR.exists())
check("异常图数量 >= 5", len(abnormal_imgs) >= 5, f"共 {len(abnormal_imgs)} 张")
check("有效 ASCII 文件名图片 >= 5", len(ascii_imgs) >= 5, f"纯 ASCII 文件名 {len(ascii_imgs)} 张")
if ascii_imgs:
    print(f"    推荐使用前5张: {[p.name for p in ascii_imgs[:5]]}")

print("\n[3] labels_coco.json 内容检查")
label_path = ASSETS["labels_coco.json"]
if label_path.exists():
    with open(label_path, encoding="utf-8") as f:
        coco_labels = json.load(f)
    if isinstance(coco_labels, list):
        check("labels_coco.json 格式为 list", True, f"共 {len(coco_labels)} 个类别")
    elif isinstance(coco_labels, dict) and "names" in coco_labels:
        check("labels_coco.json 含 names 字段", True, f"共 {len(coco_labels['names'])} 个类别")
    else:
        check("labels_coco.json 格式可识别", False, f"类型: {type(coco_labels)}")
else:
    check("labels_coco.json 可读", False)

print("\n[4] Anomalib metadata.json 契约验证")
meta_path = ASSETS["Anomalib metadata.json"]
if meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for field in REQUIRED_METADATA_FIELDS:
        check(f"metadata 含字段 '{field}'", field in meta,
              str(meta.get(field, "缺失")))
    # 契约规则
    check("output_contract = anomalib_post_processed_0_1",
          meta.get("output_contract") == "anomalib_post_processed_0_1",
          meta.get("output_contract"))
    check("normalization = none",
          str(meta.get("normalization", "")).lower() == "none",
          meta.get("normalization"))
    check("model_type = Patchcore",
          "patchcore" in str(meta.get("model_type", "")).lower(),
          meta.get("model_type"))
    check("input_size 含2个元素",
          isinstance(meta.get("input_size"), list) and len(meta["input_size"]) == 2,
          str(meta.get("input_size")))
    h, w = meta.get("input_size", [0, 0])
    check("input_size > 0", h > 0 and w > 0, f"{h}x{w}")
    check("image_threshold in [0,1]",
          0.0 <= float(meta.get("image_threshold", -1)) <= 1.0,
          str(meta.get("image_threshold")))
    check("pixel_threshold in [0,1]",
          0.0 <= float(meta.get("pixel_threshold", -1)) <= 1.0,
          str(meta.get("pixel_threshold")))
else:
    check("metadata.json 可读", False)

print("\n[5] labels_obb_dota_v1.json 内容检查")
obb_path = ASSETS["labels_obb_dota_v1.json"]
if obb_path.exists():
    with open(obb_path, encoding="utf-8") as f:
        obb_labels = json.load(f)
    if isinstance(obb_labels, list):
        check("labels_obb_dota_v1.json 格式为 list", True, f"共 {len(obb_labels)} 个类别")
    elif isinstance(obb_labels, dict):
        check("labels_obb_dota_v1.json 可解析为 dict", True)
    else:
        check("labels_obb_dota_v1.json 格式可识别", False)
else:
    check("labels_obb_dota_v1.json 可读", False)

print()
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print("=" * 60)
print(f"总计 {total} 项，通过 {passed}，失败 {failed}")
if failed == 0:
    print("资产验证全部通过，可进行 DLL 冒烟与 UI 回归测试。")
else:
    print("存在失败项，请修复后再继续。")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
