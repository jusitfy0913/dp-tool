# -*- coding: utf-8 -*-
"""ONNX 动态库接入的验收脚本（不需要 GPU，也不需要启动 Web 服务）。

覆盖本次变更里最容易悄悄坏掉的部分：
1. 框匹配：同类贪心匹配、IoU 门槛、两侧都空时视为一致；
2. 阈值判定：引擎与 ONNX 的差异必须让工具返回「不通过」，而不是只打印警告；
3. 张量对齐：形状相同、最后两维转置、元素数不同三种情况；
4. 张量配对：TensorRT 的 binding 顺序与 ONNX 输出顺序不同时必须按名字配；
5. 报告编码：CSV 必须是 utf-8-sig，Excel 打开不乱码；
6. 后端分派：`.onnx` 走 onnx_infer.dll，`.engine` 走 TRT 库（有模型时实测）。

用法：
    python 6-web/backend/tools/test_backend_consistency.py
    python 6-web/backend/tools/test_backend_consistency.py --engine <x.engine> --onnx <x.onnx> --images <dir>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (BACKEND, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

import compare_backends as cb  # noqa: E402
import compare_raw_tensors as crt  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name}（{reason}）")


def box(left, top, right, bottom, conf=0.9, cls=0, **extra) -> dict:
    item = {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "confidence": conf, "class_label": cls,
    }
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# 1. 框匹配
# ---------------------------------------------------------------------------

def test_matching() -> None:
    a = [box(0, 0, 10, 10), box(20, 20, 30, 30, conf=0.8)]
    b = [box(0, 0, 10, 10, conf=0.89), box(20, 20, 31, 30, conf=0.79)]
    pairs, unmatched_a, unmatched_b = cb.match_boxes(a, b)
    check("匹配：两个框各配一对", len(pairs) == 2 and not unmatched_a and not unmatched_b)

    # 类别不同不能配对，否则会把「猫」和「狗」算成同一个框的偏差
    c = [box(0, 0, 10, 10, cls=0)]
    d = [box(0, 0, 10, 10, cls=1)]
    pairs, unmatched_a, unmatched_b = cb.match_boxes(c, d)
    check("匹配：类别不同不配对", not pairs and len(unmatched_a) == 1 and len(unmatched_b) == 1)

    pairs, _, _ = cb.match_boxes([box(0, 0, 10, 10)], [box(50, 50, 60, 60)])
    check("匹配：IoU 不达标不配对", not pairs)


def test_empty_frames() -> None:
    """两侧都没检出时必须算一致：这是最常见的正常图，不能报失败。"""
    args = argparse.Namespace(
        task="detect", min_match_rate=0.95, min_iou=0.9, max_conf_diff=0.02,
        min_mask_iou=0.9, max_kpt_dist=3.0, max_angle_diff=5.0,
    )
    metrics = cb.compare_yolo_frame({"boxes": []}, {"boxes": []}, args)
    check("空框：匹配率 1.0", metrics["match_rate"] == 1.0, str(metrics["match_rate"]))
    check("空框：判定通过", not cb.judge_yolo(metrics, args), str(cb.judge_yolo(metrics, args)))


# ---------------------------------------------------------------------------
# 2. 阈值判定
# ---------------------------------------------------------------------------

def test_judging() -> None:
    args = argparse.Namespace(
        task="detect", min_match_rate=0.95, min_iou=0.9, max_conf_diff=0.02,
        min_mask_iou=0.9, max_kpt_dist=3.0, max_angle_diff=5.0,
    )
    bad = {"matched": 2, "match_rate": 1.0, "min_iou": 0.5, "max_conf_diff": 0.001}
    check("判定：IoU 低于阈值必须失败", bool(cb.judge_yolo(bad, args)))

    good = {"matched": 2, "match_rate": 1.0, "min_iou": 0.95, "max_conf_diff": 0.001}
    check("判定：全部达标通过", not cb.judge_yolo(good, args))

    anom_args = argparse.Namespace(
        max_score_diff=0.10, max_map_diff=0.05, min_mask_agree=0.95,
    )
    anom_ok = {
        "score_diff": 0.02, "verdict_equal": True, "map_diff": 0.01,
        "mask_agree": 0.98, "verdict_engine": True, "verdict_onnx": True,
    }
    check("判定：异常检测达标通过", not cb.judge_anomalib(anom_ok, anom_args))

    anom_bad = dict(anom_ok, verdict_equal=False, verdict_onnx=False)
    check("判定：判定翻转必须失败", bool(cb.judge_anomalib(anom_bad, anom_args)))


# ---------------------------------------------------------------------------
# 3. 张量对齐与配对
# ---------------------------------------------------------------------------

def test_alignment() -> None:
    a = np.arange(24, dtype=np.float32)
    b = np.arange(24, dtype=np.float32)
    _, _, note = crt.align(a, [1, 2, 12], b, [1, 2, 12])
    check("对齐：形状相同直接比较", note == "形状一致", note)

    t = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
    transposed = t.transpose(0, 2, 1).reshape(-1)
    aligned_a, aligned_b, note = crt.align(transposed, [1, 12, 2], t.reshape(-1), [1, 2, 12])
    ok = aligned_a is not None and np.allclose(aligned_a, aligned_b)
    check("对齐：最后两维转置后逐元素相等", ok, f"{note}")

    aligned_a, _, note = crt.align(np.arange(6, dtype=np.float32), [1, 6],
                                   np.arange(4, dtype=np.float32), [1, 4])
    check("对齐：元素数不同必须报错", aligned_a is None and "元素数不同" in note, note)


def test_pairing() -> None:
    """PatchCore 的真实顺序：TRT 是 score/map/label/mask，ONNX 是 score/label/map/mask。"""
    onnx_entries = [
        {"backend": "onnx_infer", "run": 1, "phase": "output", "index": 0, "name": "pred_score", "shape": [1]},
        {"backend": "onnx_infer", "run": 1, "phase": "output", "index": 1, "name": "pred_label", "shape": [1]},
        {"backend": "onnx_infer", "run": 1, "phase": "output", "index": 2, "name": "anomaly_map", "shape": [1, 1, 4, 4]},
    ]
    trt_entries = [
        {"backend": "anomalib_infer", "run": 1, "phase": "output", "index": 0, "name": "pred_score", "shape": [1]},
        {"backend": "anomalib_infer", "run": 1, "phase": "output", "index": 1, "name": "anomaly_map", "shape": [1, 1, 4, 4]},
        {"backend": "anomalib_infer", "run": 1, "phase": "output", "index": 2, "name": "pred_label", "shape": [1]},
    ]
    pairs = crt.compare_entries(onnx_entries, trt_entries)
    mapping = {item["key"][1]: item["trt"]["name"] for item in pairs}
    check(
        "配对：按名字配对而不是序号",
        mapping.get("anomaly_map") == "anomaly_map" and mapping.get("pred_label") == "pred_label",
        str(mapping),
    )


# ---------------------------------------------------------------------------
# 4. 报告编码
# ---------------------------------------------------------------------------

def test_report_encoding() -> None:
    report = {
        "meta": {"kind": "yolo"},
        "images": [{"image": "D:/数据/图片.jpg", "match_rate": 1.0, "failures": ""}],
    }
    with tempfile.TemporaryDirectory() as temp:
        json_path, csv_path = cb.write_reports(Path(temp), report)
        raw = csv_path.read_bytes()
        check("报告：CSV 使用 utf-8-sig", raw.startswith(b"\xef\xbb\xbf"))
        check("报告：JSON 落盘", json_path.is_file() and json_path.stat().st_size > 0)
        text = csv_path.read_text(encoding="utf-8-sig")
        check("报告：中文路径不丢字", "图片.jpg" in text)


# ---------------------------------------------------------------------------
# 5. 真实后端分派
#
# 基线不比「能不能跑」，比「跑的是谁」。缺样例时必须明确失败：一个永远 SKIP 的
# 验收项等于没有验收，等真出问题时才发现它从没跑过。
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]  # parents[0]=tools → parents[3]=dp-tool
DEFAULT_SAMPLES = {
    "onnx": REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd.onnx",
    "engine": REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd_fp16.engine",
    "images": REPO / "PCB data/PCB缺陷检测",
}


def resolve_samples(engine: str, onnx: str, images: str) -> tuple[Path, Path, Path]:
    """显式参数优先，其次仓库内的默认样例，保证不带参也有东西可测。"""
    return (
        Path(engine) if engine else DEFAULT_SAMPLES["engine"],
        Path(onnx) if onnx else DEFAULT_SAMPLES["onnx"],
        Path(images) if images else DEFAULT_SAMPLES["images"],
    )


def test_live_dispatch(engine: str, onnx: str, images: str, skip_live: bool) -> None:
    if skip_live:
        skip("实测分派", "命令行显式 --skip-live（无模型的环境用）")
        return

    engine_path, onnx_path, image_dir = resolve_samples(engine, onnx, images)
    missing = [
        str(path) for path in (engine_path, onnx_path, image_dir) if not path.exists()
    ]
    if missing:
        check("分派：样例齐全", False, f"缺 {', '.join(missing)}；用 --engine/--onnx/--images 指定，或加 --skip-live")
        return

    import infer

    status = infer.status()
    if not status.onnx_ok:
        check("分派：onnx_infer.dll 可用", False, status.onnx_detail or "未知原因")
        return
    if not status.yolo_ok:
        check("分派：yolo_infer.dll 可用", False, status.detail or "未知原因")
        return

    targets = cb.collect_images(image_dir)
    if not targets:
        check("分派：图片目录非空", False, str(image_dir))
        return
    print(f"实测样例：{onnx_path.name} vs {engine_path.name}，图片 {len(targets)} 张")

    image = infer.decode_image(targets[0].read_bytes())
    onnx_result = infer.run_yolo(str(onnx_path), "detect", image, conf=0.25, nms=0.45)
    check("分派：.onnx 走 onnx_infer.dll", onnx_result.get("backend") == "onnx_infer.dll",
          str(onnx_result.get("backend")))

    engine_result = infer.run_yolo(str(engine_path), "detect", image, conf=0.25, nms=0.45)
    check("分派：.engine 走 yolo_infer.dll",
          engine_result.get("backend") == "yolo_infer.dll",
          str(engine_result.get("backend")))
    check(
        "分派：两条后端框数量一致",
        len(engine_result.get("boxes") or []) == len(onnx_result.get("boxes") or []),
        f"{len(engine_result.get('boxes') or [])} vs {len(onnx_result.get('boxes') or [])}",
    )

    infer.release_handles()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ONNX 动态库接入验收")
    parser.add_argument("--engine", default="", help=".engine 路径（默认用仓库内 PCB 样例）")
    parser.add_argument("--onnx", default="", help=".onnx 路径（默认用仓库内 PCB 样例）")
    parser.add_argument("--images", default="", help="图片目录（默认用仓库内 PCB 数据）")
    parser.add_argument("--skip-live", action="store_true", help="跳过实测分派（没有模型的环境用）")
    args = parser.parse_args(argv)

    test_matching()
    test_empty_frames()
    test_judging()
    test_alignment()
    test_pairing()
    test_report_encoding()
    test_live_dispatch(args.engine, args.onnx, args.images, args.skip_live)

    print()
    print(f"通过 {len(PASSED)}，失败 {len(FAILED)}，跳过 {len(SKIPPED)}")
    if FAILED:
        for name in FAILED:
            print(f"  [FAIL] {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
