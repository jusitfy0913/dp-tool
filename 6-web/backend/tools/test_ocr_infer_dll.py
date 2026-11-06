#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ocr_infer.dll 实图验收。会加载 Paddle Inference，CPU 上大约需要十几秒。

覆盖：空参数、缺文件、示例图至少一行且含预期文字、字段对齐、连续两次一致、耗时大于 0。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

import ocr_infer  # noqa: E402
import ocr_models  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
REPO = BACKEND.parents[1]
DET = REPO / "9-python-paddleOCR" / "model" / "legacy_v4" / "ch_PP-OCRv4_det_infer"
REC = REPO / "9-python-paddleOCR" / "model" / "legacy_v4" / "ch_PP-OCRv4_rec_infer"
IMAGE = REPO / "ocr" / "PaddleOCR-main" / "docs" / "version2.x" / "ppocr" / "images" / "ch_ppocr_mobile_v2.0" / "00006737.jpg"
# 这张示例图是登机牌，稳定包含「登机牌」三个字。
EXPECTED = "登机牌"


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  通过  {name}")
    else:
        FAILED.append(name)
        print(f"  失败  {name}  {detail}")


def options(**overrides) -> dict:
    base = {
        "det_dir": str(DET),
        "rec_dir": str(REC),
        "textline_dir": "",
        "device": "cpu",
        "enable_mkldnn": 1,
        "cpu_threads": 4,
        "rec_batch_size": 6,
        "det_thresh": 0.3,
        "det_box_thresh": 0.6,
        "det_unclip_ratio": 1.5,
        "rec_score_thresh": 0.0,
    }
    base.update(overrides)
    return base


def main() -> int:
    print("OCR 动态库")
    # DLL 与模型都不进 git，CI 上必然缺；和其他 DLL 测试一样跳过而不是失败。
    dll_status = ocr_infer.status()
    if not dll_status["ok"]:
        print(f"  [SKIP] 缺少 ocr_infer.dll（{dll_status['detail']}）")
        return 0
    if not DET.is_dir() or not REC.is_dir() or not IMAGE.is_file():
        print("  [SKIP] 缺少老版本模型或示例图")
        return 0

    try:
        ocr_infer.run(np.zeros((8, 8, 3), np.uint8), options(det_dir=""))
        check("空检测目录被拒绝", False, "没有抛错")
    except ocr_infer.OcrError as exc:
        check("空检测目录被拒绝", exc.status == 400, str(exc))

    try:
        ocr_models.preflight(str(DET / "missing"), str(REC), "")
        check("缺文件目录被拒绝", False, "没有抛错")
    except ocr_models.OcrModelError as exc:
        check("缺文件目录被拒绝", "不存在" in str(exc), str(exc))

    image = cv2.imdecode(np.fromfile(str(IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)
    check("示例图可读", image is not None and image.size > 0)
    if image is None:
        return 1

    first = ocr_infer.run(image, options())
    text = "".join(line["text"] for line in first["lines"])
    print(f"  识别：{text}")
    check("至少一行", first["count"] >= 1 and len(first["lines"]) == first["count"], str(first["count"]))
    check("包含预期文字", EXPECTED in text, text)
    aligned = all(
        {"index", "poly", "text", "score", "angle"} <= set(line)
        and len(line["poly"]) >= 4
        for line in first["lines"]
    )
    check("行字段一一对应", aligned)
    check("耗时大于 0", first["timing"]["total_ms"] > 0, str(first["timing"]))
    check("后端名称", first["backend"] == "ocr_infer.dll")

    second = ocr_infer.run(image, options())
    second_text = "".join(line["text"] for line in second["lines"])
    check("连续两次文字一致", text == second_text, second_text)
    stats = ocr_infer.cache_stats()
    check("第二次命中缓存", stats["hits"] >= 1 and stats["creates"] == 1, str(stats))

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
