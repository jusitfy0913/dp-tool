#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OCR 网页接口验收：模型扫描、预检、参数错误、动态库缺失。

不加载 1.7GB 的 paddle_inference.dll。真正出字的验收在 test_ocr_infer_dll.py。
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi import HTTPException, UploadFile  # noqa: E402

import app  # noqa: E402
import inferproxy  # noqa: E402
import ocr_infer  # noqa: E402
import ocr_models  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
REPO = BACKEND.parents[1]
LEGACY = REPO / "9-python-paddleOCR" / "model" / "legacy_v4"
V6_DET = REPO / "9-python-paddleOCR" / "model" / "PP-OCRv6_medium_det_infer"
V6_REC = REPO / "9-python-paddleOCR" / "model" / "PP-OCRv6_medium_rec_infer"


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  通过  {name}")
    else:
        FAILED.append(name)
        print(f"  失败  {name}  {detail}")


def tiny_upload() -> UploadFile:
    import cv2

    ok, encoded = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
    if not ok:
        raise RuntimeError("生成测试图失败")
    return UploadFile(filename="blank.png", file=io.BytesIO(encoded.tobytes()))


def expect_http(fn, status: int, name: str) -> None:
    try:
        fn()
    except HTTPException as exc:
        check(name, exc.status_code == status, f"实际 {exc.status_code} {exc.detail}")
        return
    check(name, False, "没有抛出 HTTPException")


def main() -> int:
    print("OCR 接口")
    scanned = ocr_models.scan_models()
    names = {item["name"]: item for item in scanned["entries"]}
    check("扫到老版检测模型", "ch_PP-OCRv4_det_infer" in names)
    det = names.get("ch_PP-OCRv4_det_infer") or {}
    check("老版检测可兼容", det.get("compatible") is True and det.get("format") == "pdmodel", str(det))
    v6 = names.get("PP-OCRv6_medium_det_infer") or {}
    check("v6 检测标为不兼容", v6.get("format") == "json" and v6.get("compatible") is False, str(v6.get("reason")))
    check("能力接口带 OCR 字段", "infer_ocr" in app.capabilities() and "paddle_version" in app.capabilities())

    try:
        ocr_models.preflight(str(V6_DET), str(V6_REC), "")
        check("json 模型预检被拒绝", False, "没有抛错")
    except ocr_models.OcrModelError as exc:
        check("json 模型预检被拒绝", "3" in str(exc), str(exc))

    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "bad_rec"
        broken.mkdir()
        (broken / "inference.pdmodel").write_bytes(b"x")
        (broken / "inference.pdiparams").write_bytes(b"x")
        (broken / "inference.yml").write_text("Global:\n  model_name: Demo_rec\n", encoding="utf-8")
        try:
            ocr_models.preflight(str(LEGACY / "ch_PP-OCRv4_det_infer"), str(broken), "")
            check("缺字典预检被拒绝", False, "没有抛错")
        except ocr_models.OcrModelError as exc:
            check("缺字典预检被拒绝", "character_dict" in str(exc), str(exc))

    det_dir = str(LEGACY / "ch_PP-OCRv4_det_infer")
    rec_dir = str(LEGACY / "ch_PP-OCRv4_rec_infer")
    expect_http(
        lambda: app.infer_ocr(
            image=tiny_upload(), det_dir=det_dir, rec_dir=rec_dir, textline_ori_dir="",
            device="tpu", enable_mkldnn=1, cpu_threads=8, rec_batch_size=6,
            det_thresh=0.3, det_box_thresh=0.6, det_unclip_ratio=1.5, rec_score_thresh=0.0,
        ),
        400,
        "非法 device 返回 400",
    )
    expect_http(
        lambda: app.infer_ocr(
            image=tiny_upload(), det_dir=str(V6_DET), rec_dir=str(V6_REC), textline_ori_dir="",
            device="cpu", enable_mkldnn=1, cpu_threads=8, rec_batch_size=6,
            det_thresh=0.3, det_box_thresh=0.6, det_unclip_ratio=1.5, rec_score_thresh=0.0,
        ),
        400,
        "json 模型请求返回 400",
    )

    import infer
    yolo_before = infer.status().yolo_ok

    # 回归：飞桨 2.6 的 dynamic_loader 用 LoadLibrary("mklml.dll") 自己找 MKL，
    # 只认 PATH，add_dll_directory 对它无效，少了就报 error code 126（CPU/GPU 都起不来）。
    mkl_dirs = ocr_infer._mkl_dirs()
    check("能定位 mklml.dll", bool(mkl_dirs), "源码树里找不到 mklml，详见 ocr/ocr_infer/README.md")
    ocr_infer._prepare_dll_search_path()
    path_parts = [p.lower() for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    check(
        "mklml 目录已进 PATH",
        all(str(d).lower() in path_parts for d in mkl_dirs),
        f"mkl_dirs={[str(d) for d in mkl_dirs]}",
    )
    check("bin 目录已进 PATH", str(ocr_infer.BIN_DIR).lower() in path_parts, str(ocr_infer.BIN_DIR))

    saved = (ocr_infer.BIN_DIR, ocr_infer._lib, ocr_infer._loaded, ocr_infer._load_error)
    try:
        ocr_infer.BIN_DIR = Path(tempfile.gettempdir()) / "ocr-dll-missing"
        ocr_infer._lib = None
        ocr_infer._loaded = False
        ocr_infer._load_error = ""
        try:
            ocr_infer.ensure_loaded()
            check("缺少 DLL 返回 503", False, "没有抛错")
        except ocr_infer.OcrError as exc:
            check("缺少 DLL 返回 503", exc.status == 503, str(exc))
        check("OCR 失败不影响 YOLO 状态", infer.status().yolo_ok == yolo_before)
    finally:
        ocr_infer.BIN_DIR, ocr_infer._lib, ocr_infer._loaded, ocr_infer._load_error = saved

    # ---- preflight / inspect_dir 缓存 -------------------------------------
    # 识别的 inference.yml 带 6000+ 条 character_dict，每次 preflight 都解析一遍
    # 要 50 ms 左右。这里用「换掉 _load_yaml 计数」验证第二次真的没再解析。
    with tempfile.TemporaryDirectory() as tmp:
        cached = Path(tmp) / "Demo_rec"
        cached.mkdir()
        (cached / "inference.pdmodel").write_bytes(b"x")
        (cached / "inference.pdiparams").write_bytes(b"x")
        (cached / "inference.yml").write_text(
            "Global:\n  model_name: Demo_rec\nPostProcess:\n  character_dict: [a, b]\n",
            encoding="utf-8",
        )
        calls = {"n": 0}
        original_load = ocr_models._load_yaml

        def counting_load(path):
            calls["n"] += 1
            return original_load(path)

        ocr_models._load_yaml = counting_load
        try:
            first = ocr_models.inspect_dir(cached, 2)
            after_first = calls["n"]
            second = ocr_models.inspect_dir(cached, 2)
            cached_hits = calls["n"] == after_first
            # 改 yml 的 mtime（内容也变），缓存必须失效
            (cached / "inference.yml").write_text(
                "Global:\n  model_name: Demo_rec\nPostProcess:\n  character_dict: [a]\n",
                encoding="utf-8",
            )
            os.utime(cached / "inference.yml", (1, 1))
            third = ocr_models.inspect_dir(cached, 2)
            invalidated = calls["n"] > after_first
        finally:
            ocr_models._load_yaml = original_load
        check("inspect_dir 第二次命中缓存", cached_hits, f"解析次数 {calls['n']}")
        check("inspect_dir 缓存结果一致", first == second)
        check("inference.yml 变了缓存失效", invalidated, f"解析次数 {calls['n']}")
        check("失效后重新解析出条目", third.get("kind") == "rec", str(third))

    # ---- /api/ocr/warm 预热接口 -------------------------------------------
    # 用假的 run_ocr 顶掉真实推理：预热接口的职责是「后台建句柄 + 幂等」，
    # 不是真的再加载一遍 1.7 GB 的飞桨。
    calls = []
    original_run_ocr = inferproxy.run_ocr
    inferproxy.run_ocr = lambda image, options, source="": calls.append(dict(options)) or {"lines": []}
    try:
        app._OCR_WARM_STATE.update({"running": False, "ok": None, "ms": None, "error": "", "source": ""})
        started = app.warm_ocr(app.OcrWarmRequest(det_dir=det_dir, rec_dir=rec_dir))
        check("预热接口返回已启动", started.get("started") is True, str(started))
        deadline = time.time() + 5.0
        while time.time() < deadline and app._OCR_WARM_STATE["running"]:
            time.sleep(0.01)
        check("预热只调用一次 run_ocr", len(calls) == 1, str(len(calls)))
        check("预热成功后标记 ok", app._OCR_WARM_STATE.get("ok") is True, str(app._OCR_WARM_STATE))
        check("预热把模型目录传下去了", calls and calls[0].get("det_dir") == det_dir, str(calls[:1]))
        again = app.warm_ocr(app.OcrWarmRequest(det_dir=det_dir, rec_dir=rec_dir))
        check("重复预热被幂等拦下", again.get("started") is False, str(again))
        check("重复预热不再调 run_ocr", len(calls) == 1, str(len(calls)))
    finally:
        inferproxy.run_ocr = original_run_ocr
        app._OCR_WARM_STATE.update({"running": False, "ok": None, "ms": None, "error": "", "source": ""})

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
