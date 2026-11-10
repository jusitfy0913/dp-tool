#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理工作进程验收。

覆盖：
1. 隔离模式下 Web 进程不加载任何推理 DLL；
2. 工作进程崩溃 → 当前请求 503 → 下一次请求由新进程完成；
3. 60 秒内崩 3 次熔断，状态把 DLL 标成不可用，手动重启后恢复；
4. 超时 504 并重启；空闲时被外部结束，下次请求自动拉起并留痕、不计入熔断；
5. 有模型时：YOLO、Anomalib、OCR 在隔离与直通两种模式下结果一致；``--bench`` 时给出耗时对比；
6. 直通模式（TRTFORGE_INFER_ISOLATION=0）不启动子进程。

没有 DLL / 模型的环境（CI）只跑 1–4、6，模型相关项跳过。
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ["TRTFORGE_TEST_HOOKS"] = "1"
os.environ["TRTFORGE_INFER_ISOLATION"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import infer  # noqa: E402
import inferproxy  # noqa: E402
import ocr_infer  # noqa: E402

REPO = BACKEND.parents[1]
YOLO_ENGINE = REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd_fp16.engine"
YOLO_ONNX = REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd.onnx"
YOLO_IMAGES = REPO / "PCB data/PCB缺陷检测"
ANOMALIB_DIR = REPO / "2-python-anomaly/launcher/results/Patchcore/cmp_patchcore/v0/weights/lightning"
OCR_DET = REPO / "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_det_infer"
OCR_REC = REPO / "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_rec_infer"
OCR_IMAGE = REPO / "ocr/PaddleOCR-main/docs/version2.x/ppocr/images/ch_ppocr_mobile_v2.0/00006737.jpg"

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name}（{reason}）")


def expect_worker_error(name: str, func, status: int) -> None:
    try:
        func()
    except inferproxy.WorkerError as exc:
        check(name, exc.status == status, f"{exc.status} {exc}")
        return
    check(name, False, "没有抛 WorkerError")


def wait_running(timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if inferproxy.worker_state()["state"] == "running":
            return True
        time.sleep(0.2)
    return False


def read_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def sorted_boxes(result: dict) -> list[tuple]:
    """GPU 上的 NMS 不保证输出顺序，同一模式连跑两次框的顺序也会变，按内容排序后再比。"""
    keys = ("class_label", "left", "top", "right", "bottom", "confidence")
    return sorted(tuple(round(float(box[k]), 3) for k in keys) for box in result["boxes"])


def test_lifecycle() -> None:
    inferproxy.startup()
    state = inferproxy.worker_state()
    check("工作进程启动", state["state"] == "running", str(state))
    check("Web 进程未加载 yolo/anomalib/onnx DLL",
          infer._yolo_lib is None and infer._anomalib_lib is None and infer._onnx_lib is None)
    check("Web 进程未加载 OCR DLL", ocr_infer._lib is None)
    status = inferproxy.status()
    check("状态来自工作进程", isinstance(status, infer.BackendStatus), status.detail)

    pid = inferproxy._test_call("pid")
    check("工作进程 pid 与本进程不同", pid != os.getpid(), str(pid))

    expect_worker_error("崩溃时当前请求 503", lambda: inferproxy._test_call("crash"), 503)
    check("崩溃后自动重启", wait_running(), str(inferproxy.worker_state()))
    new_pid = inferproxy._test_call("pid")
    check("重启后是新进程", new_pid != pid, f"{pid} → {new_pid}")

    expect_worker_error("超时 504", lambda: inferproxy._test_call("sleep", seconds=5, timeout=1), 504)
    check("超时后重启", wait_running(), str(inferproxy.worker_state()))

    inferproxy.restart_worker()
    for i in range(inferproxy.CRASH_LIMIT):
        expect_worker_error(f"连续崩溃第 {i + 1} 次 503", lambda: inferproxy._test_call("crash"), 503)
        wait_running(30)
    state = inferproxy.worker_state()
    check("连续崩溃后熔断", state["state"] == "tripped", str(state))
    status = inferproxy.status()
    check("熔断后 DLL 标为不可用（路由会走 onnxruntime 降级）",
          not status.yolo_ok and not status.anomalib_ok and not status.onnx_ok, status.detail)
    expect_worker_error("熔断期间请求直接 503", lambda: inferproxy._test_call("pid"), 503)

    state = inferproxy.restart_worker()
    check("手动重启解除熔断", state["state"] == "running", str(state))
    check("缓存查询可用", isinstance(inferproxy.cache_stats(), dict))

    idle_pid = inferproxy._test_call("pid")
    before = inferproxy.worker_state()["restarts"]
    subprocess.run(["taskkill", "/F", "/PID", str(idle_pid)], capture_output=True)
    time.sleep(0.5)
    again = inferproxy._test_call("pid")
    state = inferproxy.worker_state()
    check("空闲被杀后下次请求自动拉起", again != idle_pid and state["state"] == "running", f"{idle_pid} → {again}")
    check("空闲退出留痕且不计入熔断",
          state["restarts"] == before + 1 and "空闲" in state["last_error"] and state["recent_crashes"] == 0, str(state))


def test_yolo_consistency() -> None:
    images = []
    if YOLO_IMAGES.is_dir():
        images = sorted(p for p in YOLO_IMAGES.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})[:5]
    if not images:
        skip("YOLO 一致性", "缺少 PCB 示例图")
        return
    models = [p for p in (YOLO_ENGINE, YOLO_ONNX) if p.is_file()]
    status = inferproxy.status()
    if not models or not (status.yolo_ok or status.onnx_ok):
        skip("YOLO 一致性", f"缺少模型或 DLL：{status.detail}")
        return

    arrays = [read_image(p) for p in images]
    for model in models:
        if model.suffix == ".onnx" and not status.onnx_ok:
            skip(f"一致性 {model.name}", "onnx_infer.dll 不可用")
            continue
        if model.suffix == ".engine" and not status.yolo_ok:
            skip(f"一致性 {model.name}", "yolo_infer.dll 不可用")
            continue
        isolated = [inferproxy.run_yolo(str(model), "detect", a, source="t") for a in arrays]
        os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
        try:
            direct = [inferproxy.run_yolo(str(model), "detect", a) for a in arrays]
        finally:
            os.environ["TRTFORGE_INFER_ISOLATION"] = "1"
        same = all(sorted_boxes(x) == sorted_boxes(y) for x, y in zip(isolated, direct))
        total = sum(len(x["boxes"]) for x in isolated)
        check(f"一致性 {model.name}：框完全相同", same and total > 0, f"{len(arrays)} 张，共 {total} 框")


def test_anomalib_consistency() -> None:
    folder = ANOMALIB_DIR
    engine, metadata = folder / "model_fp16.engine", folder / "metadata.json"
    images = []
    if YOLO_IMAGES.is_dir():
        images = sorted(p for p in YOLO_IMAGES.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})[:3]
    if not (engine.is_file() and metadata.is_file() and images):
        skip("Anomalib 一致性", "缺少引擎、metadata 或图片")
        return
    if not inferproxy.status().anomalib_ok:
        skip("Anomalib 一致性", inferproxy.status().detail)
        return
    arrays = [read_image(p) for p in images]
    isolated = [inferproxy.run_anomalib(str(engine), str(metadata), a) for a in arrays]
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    try:
        direct = [inferproxy.run_anomalib(str(engine), str(metadata), a) for a in arrays]
    finally:
        os.environ["TRTFORGE_INFER_ISOLATION"] = "1"
    fields = ("score", "is_anomaly", "heatmap_png", "mask_png", "overlay_jpg")
    diffs = [k for x, y in zip(isolated, direct) for k in fields if x.get(k) != y.get(k)]
    check("Anomalib 一致性：分数、判定与热力图字节相同", not diffs, f"{len(arrays)} 张，不一致字段 {sorted(set(diffs))}")


def bench(model: Path, shape: tuple[int, int], rounds: int) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(shape[1], shape[0], 3), dtype=np.uint8)

    def timed() -> float:
        for _ in range(5):
            inferproxy.run_yolo(str(model), "detect", image)
        samples = []
        for _ in range(rounds):
            start = time.perf_counter()
            inferproxy.run_yolo(str(model), "detect", image)
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    isolated = timed()
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    try:
        direct = timed()
    finally:
        os.environ["TRTFORGE_INFER_ISOLATION"] = "1"
    print(f"[INFO] 耗时 {model.name} {shape[0]}x{shape[1]}：隔离 {isolated:.2f} ms，"
          f"直通 {direct:.2f} ms，差 {isolated - direct:+.2f} ms（中位数，{rounds} 次）")


def test_ocr_consistency() -> None:
    if not (OCR_DET.is_dir() and OCR_REC.is_dir() and OCR_IMAGE.is_file()):
        skip("OCR 一致性", "缺少模型或示例图")
        return
    if not inferproxy.ocr_status()["ok"]:
        skip("OCR 一致性", inferproxy.ocr_status()["detail"])
        return
    options = {
        "det_dir": str(OCR_DET), "rec_dir": str(OCR_REC), "textline_dir": "",
        "device": "cpu", "enable_mkldnn": 1, "cpu_threads": 4, "rec_batch_size": 6,
        "det_thresh": 0.3, "det_box_thresh": 0.6, "det_unclip_ratio": 1.5, "rec_score_thresh": 0.0,
    }
    image = read_image(OCR_IMAGE)
    isolated = inferproxy.run_ocr(image, options, source=OCR_IMAGE.name)
    check("OCR 隔离模式下 Web 进程仍未加载 OCR DLL", ocr_infer._lib is None)
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    try:
        direct = inferproxy.run_ocr(image, options)
    finally:
        os.environ["TRTFORGE_INFER_ISOLATION"] = "1"
    a = [(line["text"], line["poly"]) for line in isolated["lines"]]
    b = [(line["text"], line["poly"]) for line in direct["lines"]]
    check("OCR 一致性：文字与多边形完全相同", a == b and len(a) > 0, f"{len(a)} 行")


def test_passthrough() -> None:
    inferproxy.shutdown()
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    try:
        check("直通模式状态为 disabled", inferproxy.worker_state()["state"] == "disabled")
    finally:
        os.environ["TRTFORGE_INFER_ISOLATION"] = "1"


def main() -> int:
    print("推理工作进程")
    try:
        test_lifecycle()
        test_yolo_consistency()
        test_anomalib_consistency()
        test_ocr_consistency()
        if "--bench" in sys.argv:
            model = YOLO_ENGINE if inferproxy.status().yolo_ok and YOLO_ENGINE.is_file() else None
            if model is None:
                skip("耗时对比", "缺少 TensorRT 引擎")
            else:
                bench(model, (1920, 1080), 100)
                bench(model, (3840, 2160), 100)
        test_passthrough()
    finally:
        inferproxy.shutdown()

    print()
    print(f"通过 {len(PASSED)}，失败 {len(FAILED)}，跳过 {len(SKIPPED)}")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
