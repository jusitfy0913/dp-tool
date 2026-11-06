# -*- coding: utf-8 -*-
"""onnxruntime 降级推理的回归验证。

覆盖：
1. YOLO 两种输出布局（未转置 `[1,4+nc,n]` 与已转置 `[1,n,4+nc]`）都能正确解码
2. 坐标从 letterbox 空间还原回原图，且框落在图像范围内
3. 返回结构与 DLL 路径一致（字段名、类型齐全）
4. anomalib 的 `pred_score` 一维/二维两种形状都能取到分数
5. 错误路径：segment/pose 明确拒绝、.engine 明确拒绝、缺文件明确报错

用法：
    python backend/tools/test_onnx_fallback.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np

import onnx_fallback as F

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def imread_unicode(path: str) -> np.ndarray | None:
    """Windows 下 cv2.imread 读不了非 ASCII 路径，必须走 bytes 解码。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def find_test_image() -> tuple[str, np.ndarray] | None:
    """优先找 PCB 真实图，退化到任意可用的大图。"""
    patterns = [
        "1-python-yolo/**/*.jpg",
        "1-python-yolo/**/*.png",
        "5-reference/**/*.jpg",
    ]
    fallback = None
    for pat in patterns:
        for p in REPO_ROOT.glob(pat):
            img = imread_unicode(str(p))
            if img is None or img.shape[0] < 200:
                continue
            if "pcb" in p.name.lower() or "pcb" in str(p.parent).lower():
                return str(p), img
            if fallback is None:
                fallback = (str(p), img)
    return fallback


def test_yolo() -> None:
    print("\n[1] YOLO 降级推理")
    model = REPO_ROOT / "1-python-yolo" / "launcher" / "runs" / "pcb" / "detect_v1" / "weights" / "best.onnx"
    transd = model.with_name("best.transd.onnx")
    if not model.is_file():
        print(f"  [SKIP] 缺少模型：{model}")
        return

    found = find_test_image()
    if not found:
        print("  [SKIP] 找不到测试图")
        return
    img_path, image = found
    print(f"  测试图：{Path(img_path).name} {image.shape}")
    height, width = image.shape[:2]

    # 1a. 未转置布局
    result = F.run_yolo(str(model), "detect", image, conf=0.25, nms=0.45)
    check("未转置 [1,4+nc,n] 能跑通", result["backend"] == F.BACKEND_YOLO)
    check("返回结构含 DLL 路径的全部字段",
          set(result) >= {"backend", "task", "image_width", "image_height", "count", "boxes", "timing"})
    check("timing 四阶段齐全", set(result["timing"]) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"})

    # 1b. 已转置布局必须得到同样数量级的结果（布局自适应是关键）
    if transd.is_file():
        result_t = F.run_yolo(str(transd), "detect", image, conf=0.25, nms=0.45)
        check("已转置 [1,n,4+nc] 能跑通", result_t["backend"] == F.BACKEND_YOLO)
        check(
            f"两种布局框数接近（{result['count']} vs {result_t['count']}）",
            abs(result["count"] - result_t["count"]) <= max(3, result["count"] // 5),
        )

    # 1c. 框必须落在原图范围内（验证 letterbox 还原正确）
    in_bounds = all(
        0 <= b["left"] <= width and 0 <= b["top"] <= height
        and 0 <= b["right"] <= width and 0 <= b["bottom"] <= height
        and b["right"] > b["left"] and b["bottom"] > b["top"]
        for b in result["boxes"]
    )
    check(f"框全部落在原图内且非退化（{result['count']} 个）", in_bounds)

    # 1d. 置信度过滤生效
    low = F.run_yolo(str(model), "detect", image, conf=0.01, nms=0.45)
    check(f"conf 从 0.25 降到 0.01 框数不减少（{result['count']} -> {low['count']}）",
          low["count"] >= result["count"])
    check("所有框置信度 >= conf", all(b["confidence"] >= 0.01 for b in low["boxes"]))

    # 1e. 不支持的任务头要明确拒绝
    for task in ("segment", "pose"):
        try:
            F.run_yolo(str(model), task, image)
            check(f"{task} 应被拒绝", False, "未抛异常")
        except F.FallbackError as exc:
            check(f"{task} 被明确拒绝", "yolo_infer.dll" in str(exc), str(exc))


def test_anomalib() -> None:
    print("\n[2] Anomalib 降级推理")
    base = REPO_ROOT / "2-python-anomaly" / "launcher" / "results"
    cases = []
    for model_dir in base.glob("*"):
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.glob("*"):
            onnx = run_dir / "v0" / "weights" / "lightning" / "model.onnx"
            meta = run_dir / "v0" / "weights" / "lightning" / "metadata.json"
            if onnx.is_file() and meta.is_file():
                cases.append((model_dir.name, onnx, meta))
    if not cases:
        print("  [SKIP] 没有可用的 anomalib 模型")
        return

    # 用 bottle 的异常图；没有就用任意图验证链路
    abnormal = None
    for p in (REPO_ROOT / "2-python-anomaly" / "anomalib-main" / "datasets" / "bottle" / "test" / "abnormal").glob("*"):
        img = imread_unicode(str(p))
        if img is not None:
            abnormal = img
            break
    if abnormal is None:
        found = find_test_image()
        if not found:
            print("  [SKIP] 找不到测试图")
            return
        abnormal = found[1]

    seen_models = set()
    for name, onnx, meta in cases:
        if name in seen_models:
            continue
        seen_models.add(name)
        try:
            result = F.run_anomalib(str(onnx), str(meta), abnormal)
        except F.FallbackError as exc:
            check(f"{name} 推理执行", False, str(exc))
            continue
        ok_fields = set(result) >= {
            "backend", "raw_score", "normalized_score", "is_anomaly",
            "metadata", "timing", "overlay_png", "mask_png",
        }
        check(f"{name} 返回结构完整", ok_fields)
        check(
            f"{name} normalized_score 在 0~1（{result['normalized_score']:.4f}）",
            0.0 <= result["normalized_score"] <= 1.0,
        )
        check(f"{name} 生成叠加图与掩码", bool(result["overlay_png"]) and bool(result["mask_png"]))
        # 必须是原始 PNG 字节，而不是 base64 字符串（app.py 统一做编码）
        check(
            f"{name} overlay 为 PNG 字节（非 base64 字符串）",
            isinstance(result["overlay_png"], (bytes, bytearray))
            and bytes(result["overlay_png"][:8]) == b"\x89PNG\r\n\x1a\n",
        )

    # .engine 必须明确拒绝
    fake_engine = BACKEND_DIR / "dummy.engine"
    try:
        F.run_anomalib(str(fake_engine), str(cases[0][2]), abnormal)
        check(".engine 被明确拒绝", False, "未抛异常")
    except F.FallbackError as exc:
        check(".engine 被明确拒绝并解释原因", ".onnx" in str(exc), str(exc))

    # 缺文件
    try:
        F.run_anomalib(str(BACKEND_DIR / "nope.onnx"), str(cases[0][2]), abnormal)
        check("缺模型文件时报错", False, "未抛异常")
    except F.FallbackError:
        check("缺模型文件时报错", True)


def test_capability() -> None:
    print("\n[3] 能力探测")
    ok, detail = F.available()
    check("available() 返回可用", ok, detail)
    check("detail 含 provider 信息", "provider" in detail.lower() or "onnxruntime" in detail)


def test_api_fallback() -> None:
    """模拟 DLL 损坏，确认 API 自动降级且返回结构不变。"""
    print("\n[4] API 降级端到端（模拟 DLL 不可用）")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 缺少 fastapi TestClient：{exc}")
        return

    import app as backend_app
    import infer

    model = REPO_ROOT / "1-python-yolo" / "launcher" / "runs" / "pcb" / "detect_v1" / "weights" / "best.onnx"
    found = find_test_image()
    if not (model.is_file() and found):
        print("  [SKIP] 缺少模型或测试图")
        return
    _, image = found
    ok, png = cv2.imencode(".png", image)
    if not ok:
        print("  [SKIP] 测试图编码失败")
        return
    image_bytes = png.tobytes()

    import inferproxy

    # 路由经 inferproxy 取状态、调推理，模拟「DLL 缺失」要替换这一层
    original_status = inferproxy.status
    original_yolo = inferproxy.run_yolo
    original_anom = inferproxy.run_anomalib
    original_isolation = os.environ.get("TRTFORGE_INFER_ISOLATION")

    def broken_status():
        return infer.BackendStatus(
            yolo_ok=False, anomalib_ok=False, dll_dirs=[], detail="测试模拟：DLL 缺失"
        )

    def boom(*_args, **_kwargs):
        raise AssertionError("降级时不应调用 DLL 路径")

    # 降级路径与推理进程无关，用直通模式免得 TestClient 启动时拉起真实工作进程
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    inferproxy.status = broken_status
    inferproxy.run_yolo = boom
    inferproxy.run_anomalib = boom
    try:
        with TestClient(backend_app.app) as client:
            # 4a. YOLO：给 .onnx，应走降级并成功
            resp = client.post(
                "/api/infer/yolo",
                files={"image": ("t.png", image_bytes, "image/png")},
                data={"engine": str(model), "task": "detect", "conf": "0.25", "nms": "0.45"},
            )
            check("DLL 缺失时 YOLO 不再返回 503", resp.status_code == 200, resp.text[:150])
            if resp.status_code == 200:
                body = resp.json()
                check("响应标注了降级后端", body.get("backend") == F.BACKEND_YOLO, str(body.get("backend")))
                check("带 warning 说明降级原因", "降级" in str(body.get("warning", "")))
                check("仍返回标注图", bool(body.get("annotated_png")))

            # 4b. 给 .engine 且同目录没有 .onnx —— 应 503 并说清怎么修
            fake_engine = BACKEND_DIR / "no_such_engine.engine"
            resp = client.post(
                "/api/infer/yolo",
                files={"image": ("t.png", image_bytes, "image/png")},
                data={"engine": str(fake_engine), "task": "detect"},
            )
            check("找不到对应 ONNX 时返回 503", resp.status_code == 503, f"实际 {resp.status_code}")
            check("503 里给出修复指引", ".onnx" in resp.text, resp.text[:150])

            # 4c. Anomalib：engine 用 .onnx，metadata 用真实文件，应走降级
            base = REPO_ROOT / "2-python-anomaly" / "launcher" / "results"
            case = None
            for onnx in base.glob("*/*/v0/weights/lightning/model.onnx"):
                meta = onnx.with_name("metadata.json")
                if meta.is_file():
                    case = (onnx, meta)
                    break
            if case:
                resp = client.post(
                    "/api/infer/anomalib",
                    files={"image": ("t.png", image_bytes, "image/png")},
                    data={
                        "engine": str(case[0]),
                        "metadata": str(case[1]),
                        "model_type": "Auto",
                    },
                )
                check("DLL 缺失时 Anomalib 降级成功", resp.status_code == 200, resp.text[:150])
                if resp.status_code == 200:
                    body = resp.json()
                    check("Anomalib 标注降级后端", body.get("backend") == F.BACKEND_ANOMALIB)
                    check("Anomalib 返回 base64 叠加图", isinstance(body.get("overlay_png"), str))
                    check("Anomalib 带 warning", "降级" in str(body.get("warning", "")))
            else:
                print("  [SKIP] 没有 anomalib 模型")
    finally:
        inferproxy.status = original_status
        inferproxy.run_yolo = original_yolo
        inferproxy.run_anomalib = original_anom
        if original_isolation is None:
            os.environ.pop("TRTFORGE_INFER_ISOLATION", None)
        else:
            os.environ["TRTFORGE_INFER_ISOLATION"] = original_isolation


def main() -> int:
    test_capability()
    test_yolo()
    test_anomalib()
    test_api_fallback()
    print(f"\n{'=' * 56}")
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 56)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
