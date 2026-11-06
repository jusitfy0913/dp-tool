# -*- coding: utf-8 -*-
"""比对同一条模型的 TensorRT 引擎与 ONNX 两条后端，逐图给出差异指标。

为什么需要它：
    ONNX 路径的下游（Web 推理页、批量、视频）与 TRT 路径共用同一套后处理语义，
    但两条后端的数值来源完全不同（fp16 引擎 vs fp32/CPU ORT、不同的预处理实现）。
    「看起来框差不多」不能作为验收依据，必须在相同输入与相同参数下逐图量化：
    框数量、匹配率、最小 IoU、置信度差、关键点距离、旋转角差、掩码 IoU、
    异常分数差、判定一致性、异常图差值。

用法（在 6-web/backend 目录下）：

    python tools/compare_backends.py --kind yolo \
        --engine runs/pcb/weights/best_fp16.engine \
        --onnx  runs/pcb/weights/best.transd.onnx \
        --images "PCB data/PCB缺陷检测/images/val" --task detect

    python tools/compare_backends.py --kind anomalib \
        --engine results/Patchcore/.../model_fp16.engine \
        --onnx  results/Patchcore/.../model.onnx \
        --metadata results/Patchcore/.../metadata.json \
        --images "bottle/test/broken_large"

产物（默认写到 6-web/runtime/compare/<时间戳>/）：
    compare_report.json   完整明细 + 本次使用的阈值
    compare_report.csv    每图一行，utf-8-sig，Excel 直接打开不乱码

退出码：
    0  全部指标在阈值内
    1  有图超出阈值（报告里指出图片与指标）
    2  无法比对（文件缺失 / DLL 不可用 / 目录为空 / 参数错误）
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import infer  # noqa: E402

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNAVAILABLE = 2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT 引擎与 ONNX 输出一致性比对")
    parser.add_argument("--kind", choices=("yolo", "anomalib"), required=True)
    parser.add_argument("--engine", required=True, help=".engine / .trt 路径")
    parser.add_argument("--onnx", required=True, help=".onnx 路径")
    parser.add_argument("--images", required=True, help="图片目录或单张图片")
    parser.add_argument("--task", default="detect", choices=("detect", "segment", "pose", "obb"))
    parser.add_argument("--metadata", default="", help="anomalib 的 metadata.json")
    parser.add_argument("--model-type", default="Auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="最多比对多少张，0 = 全部")
    parser.add_argument("--out", default="", help="报告输出目录，留空用 runtime/compare/<时间戳>")

    # 阈值：默认值按「可以上线」定，不按「差不多」定
    parser.add_argument("--min-match-rate", type=float, default=0.95)
    parser.add_argument("--min-iou", type=float, default=0.90)
    parser.add_argument("--max-conf-diff", type=float, default=0.02)
    parser.add_argument("--min-mask-iou", type=float, default=0.90)
    parser.add_argument("--max-kpt-dist", type=float, default=3.0, help="关键点欧氏距离（像素）")
    parser.add_argument("--max-angle-diff", type=float, default=5.0, help="旋转框角度差（度）")
    parser.add_argument("--max-score-diff", type=float, default=0.10)
    parser.add_argument("--max-map-diff", type=float, default=0.05)
    parser.add_argument("--min-mask-agree", type=float, default=0.95)
    return parser.parse_args(argv)


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in IMAGE_SUFFIXES else []
    if not target.is_dir():
        return []
    return sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def box_iou(a: dict, b: dict) -> float:
    """轴对齐 IoU。旋转框也先用外接轴对齐框匹配，角度差异单独比较。"""
    ax1, ay1, ax2, ay2 = a["left"], a["top"], a["right"], a["bottom"]
    bx1, by1, bx2, by2 = b["left"], b["top"], b["right"], b["bottom"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(a_boxes: list[dict], b_boxes: list[dict], iou_thr: float = 0.5):
    """贪心匹配：同类、IoU 最高且达到阈值。返回 (pairs, unmatched_a, unmatched_b)。"""
    pairs: list[tuple[int, int, float]] = []
    used_b: set[int] = set()
    unmatched_a: list[int] = []

    # 置信度高的先匹配，避免低分框抢走高分框的对应关系
    order = sorted(range(len(a_boxes)), key=lambda i: -float(a_boxes[i].get("confidence", 0.0)))
    for i in order:
        best_j, best_iou = -1, 0.0
        for j, candidate in enumerate(b_boxes):
            if j in used_b:
                continue
            if int(a_boxes[i].get("class_label", -1)) != int(candidate.get("class_label", -1)):
                continue
            value = box_iou(a_boxes[i], candidate)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_j >= 0 and best_iou >= iou_thr:
            used_b.add(best_j)
            pairs.append((i, best_j, best_iou))
        else:
            unmatched_a.append(i)

    unmatched_b = [j for j in range(len(b_boxes)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def decode_png(value) -> np.ndarray | None:
    """解码 PNG。DLL 包装层直接返回 bytes，base64 字符串也接受，方便前端/报告复用。"""
    if not value:
        return None
    if isinstance(value, str):
        raw_bytes = base64.b64decode(value)
    else:
        raw_bytes = bytes(value)
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def decode_mask(item: dict) -> np.ndarray | None:
    """还原 segment 掩码：DLL 给的是灰度 uint8，宽高来自框内的裁剪区域。"""
    text = item.get("mask_b64")
    if not text:
        return None
    width, height = int(item.get("mask_w", 0)), int(item.get("mask_h", 0))
    if width <= 0 or height <= 0:
        return None
    raw = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    if raw.size != width * height:
        return None
    return raw.reshape(height, width)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(union) if union else 1.0


def angle_diff_deg(a: float, b: float) -> float:
    """旋转框角度差；先折到 [-90, 90)，避免 179° 与 -1° 被算成大差异。"""
    delta = (float(a) - float(b)) % math.pi
    if delta > math.pi / 2:
        delta -= math.pi
    return abs(delta) * 180.0 / math.pi


def kpt_distance(a: list | None, b: list | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    worst = 0.0
    for pa, pb in zip(a, b):
        worst = max(worst, math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1])))
    return worst


# ---------------------------------------------------------------------------
# 单图比对
# ---------------------------------------------------------------------------

def compare_yolo_frame(a: dict, b: dict, args: argparse.Namespace) -> dict:
    a_boxes = a.get("boxes") or []
    b_boxes = b.get("boxes") or []
    pairs, unmatched_a, unmatched_b = match_boxes(a_boxes, b_boxes)

    ious = [value for _, _, value in pairs]
    conf_diffs = [
        abs(float(a_boxes[i].get("confidence", 0.0)) - float(b_boxes[j].get("confidence", 0.0)))
        for i, j, _ in pairs
    ]
    mask_ious: list[float] = []
    angle_diffs: list[float] = []
    kpt_dists: list[float] = []

    for i, j, _ in pairs:
        item_a, item_b = a_boxes[i], b_boxes[j]
        if args.task == "segment":
            value = mask_iou(decode_mask(item_a), decode_mask(item_b))
            if value is not None:
                mask_ious.append(value)
        if args.task == "obb" and item_a.get("obb") and item_b.get("obb"):
            angle_diffs.append(angle_diff_deg(item_a["obb"]["angle"], item_b["obb"]["angle"]))
        if args.task == "pose" and item_a.get("kpts") and item_b.get("kpts"):
            value = kpt_distance(item_a["kpts"], item_b["kpts"])
            if value is not None:
                kpt_dists.append(value)

    # 两侧都没框等价于完全一致；用较大的一侧做分母，多出一个框就会拉低匹配率
    denominator = max(len(a_boxes), len(b_boxes))
    match_rate = 1.0 if denominator == 0 else len(pairs) / denominator

    return {
        "count_engine": len(a_boxes),
        "count_onnx": len(b_boxes),
        "matched": len(pairs),
        "unmatched_engine": len(unmatched_a),
        "unmatched_onnx": len(unmatched_b),
        "match_rate": round(match_rate, 4),
        "min_iou": round(min(ious), 4) if ious else None,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
        "max_conf_diff": round(max(conf_diffs), 4) if conf_diffs else None,
        "min_mask_iou": round(min(mask_ious), 4) if mask_ious else None,
        "max_angle_diff_deg": round(max(angle_diffs), 3) if angle_diffs else None,
        "max_kpt_dist_px": round(max(kpt_dists), 3) if kpt_dists else None,
    }


def compare_anomalib_frame(a: dict, b: dict) -> dict:
    """异常检测的结果级差异。

    map 差值用渲染后的热能图灰度差近似：DLL 不把归一化 map 数组交给 Python，
    而渲染链路三个后端共用一份实现，因此灰度差能代表 map 的归一化差异。
    """
    mask_a = decode_png(a.get("mask_png"))
    mask_b = decode_png(b.get("mask_png"))
    heat_a = decode_png(a.get("heatmap_png"))
    heat_b = decode_png(b.get("heatmap_png"))

    mask_agree = None
    if mask_a is not None and mask_b is not None and mask_a.shape == mask_b.shape:
        gray_a = cv2.cvtColor(mask_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(mask_b, cv2.COLOR_BGR2GRAY)
        mask_agree = float((gray_a == gray_b).mean())

    map_diff = None
    if heat_a is not None and heat_b is not None and heat_a.shape == heat_b.shape:
        gray_a = cv2.cvtColor(heat_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(heat_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        map_diff = float(np.abs(gray_a - gray_b).mean() / 255.0)

    return {
        "score_engine": round(float(a.get("raw_score", 0.0)), 6),
        "score_onnx": round(float(b.get("raw_score", 0.0)), 6),
        "score_diff": round(abs(float(a.get("raw_score", 0.0)) - float(b.get("raw_score", 0.0))), 6),
        "verdict_engine": bool(a.get("is_anomaly")),
        "verdict_onnx": bool(b.get("is_anomaly")),
        "verdict_equal": bool(a.get("is_anomaly")) == bool(b.get("is_anomaly")),
        "mask_agree": round(mask_agree, 4) if mask_agree is not None else None,
        "map_diff": round(map_diff, 4) if map_diff is not None else None,
    }


# ---------------------------------------------------------------------------
# 阈值判定
# ---------------------------------------------------------------------------

def judge_yolo(metrics: dict, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    rate = metrics.get("match_rate")
    if rate is not None and rate < args.min_match_rate:
        failures.append(f"match_rate {rate:.4f} < {args.min_match_rate}")

    if metrics.get("matched"):
        min_iou = metrics.get("min_iou")
        if min_iou is not None and min_iou < args.min_iou:
            failures.append(f"min_iou {min_iou:.4f} < {args.min_iou}")
        max_conf = metrics.get("max_conf_diff")
        if max_conf is not None and max_conf > args.max_conf_diff:
            failures.append(f"max_conf_diff {max_conf:.4f} > {args.max_conf_diff}")

    if args.task == "segment":
        value = metrics.get("min_mask_iou")
        if value is not None and value < args.min_mask_iou:
            failures.append(f"min_mask_iou {value:.4f} < {args.min_mask_iou}")
    if args.task == "pose":
        value = metrics.get("max_kpt_dist_px")
        if value is not None and value > args.max_kpt_dist:
            failures.append(f"max_kpt_dist {value:.3f} > {args.max_kpt_dist}")
    if args.task == "obb":
        value = metrics.get("max_angle_diff_deg")
        if value is not None and value > args.max_angle_diff:
            failures.append(f"max_angle_diff {value:.3f} > {args.max_angle_diff}")
    return failures


def judge_anomalib(metrics: dict, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if metrics["score_diff"] > args.max_score_diff:
        failures.append(f"score_diff {metrics['score_diff']:.4f} > {args.max_score_diff}")
    if not metrics["verdict_equal"]:
        failures.append(
            f"verdict engine={metrics['verdict_engine']} onnx={metrics['verdict_onnx']}"
        )
    value = metrics.get("map_diff")
    if value is not None and value > args.max_map_diff:
        failures.append(f"map_diff {value:.4f} > {args.max_map_diff}")
    value = metrics.get("mask_agree")
    if value is not None and value < args.min_mask_agree:
        failures.append(f"mask_agree {value:.4f} < {args.min_mask_agree}")
    return failures


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def write_reports(out_dir: Path, report: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compare_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "compare_report.csv"
    rows = report["images"]
    columns = list(rows[0].keys()) if rows else ["image"]
    # utf-8-sig：Excel 双击打开中文路径不乱码，这是交付给现场看的表
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if value is None else value) for key, value in row.items()
            })
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    engine = Path(args.engine).expanduser()
    onnx = Path(args.onnx).expanduser()
    images = collect_images(Path(args.images).expanduser())

    missing: list[str] = []
    if not engine.is_file():
        missing.append(f"引擎不存在：{engine}")
    if not onnx.is_file():
        missing.append(f"ONNX 不存在：{onnx}")
    if not images:
        missing.append(f"图片目录为空或没有图片：{args.images}")

    metadata = Path(args.metadata).expanduser() if args.metadata.strip() else None
    if args.kind == "anomalib":
        if metadata is None or not metadata.is_file():
            missing.append(f"anomalib 需要 metadata.json：{args.metadata or '(未提供)'}")

    status = infer.status()
    if not status.onnx_ok:
        missing.append(f"onnx_infer.dll 不可用：{status.onnx_detail or '未知原因'}")
    if args.kind == "yolo" and not status.yolo_ok:
        missing.append(f"yolo_infer.dll 不可用：{status.detail or '未知原因'}")
    if args.kind == "anomalib" and not status.anomalib_ok:
        missing.append(f"anomalib_infer.dll 不可用：{status.detail or '未知原因'}")

    if missing:
        for text in missing:
            print(f"[无法比对] {text}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    out_dir = Path(args.out).expanduser() if args.out.strip() else (
        BACKEND_DIR.parent / "runtime" / "compare" / time.strftime("%Y%m%d_%H%M%S")
    )

    rows: list[dict] = []
    failed_images: list[str] = []
    backends = {"engine": "", "onnx": ""}
    started = time.perf_counter()

    for index, image_path in enumerate(images):
        try:
            image = infer.decode_image(image_path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - 单张坏图不能让整次比对白跑
            rows.append({"image": str(image_path), "error": f"解码失败：{exc}"})
            failed_images.append(str(image_path))
            continue

        try:
            if args.kind == "yolo":
                a = infer.run_yolo(
                    str(engine), args.task, image, conf=args.conf, nms=args.nms,
                    want_masks=args.task == "segment",
                )
                b = infer.run_yolo(
                    str(onnx), args.task, image, conf=args.conf, nms=args.nms,
                    want_masks=args.task == "segment",
                )
                backends["engine"], backends["onnx"] = a.get("backend", ""), b.get("backend", "")
                metrics = compare_yolo_frame(a, b, args)
                failures = judge_yolo(metrics, args)
            else:
                a = infer.run_anomalib(
                    str(engine), str(metadata), image, model_type=args.model_type,
                )
                b = infer.run_anomalib(
                    str(onnx), str(metadata), image, model_type=args.model_type,
                )
                backends["engine"], backends["onnx"] = a.get("backend", ""), b.get("backend", "")
                metrics = compare_anomalib_frame(a, b)
                failures = judge_anomalib(metrics, args)
        except Exception as exc:  # noqa: BLE001 - 记录到该图，继续下一张
            rows.append({"image": str(image_path), "error": str(exc)})
            failed_images.append(str(image_path))
            continue

        row = {"image": str(image_path), **metrics, "failures": "; ".join(failures)}
        rows.append(row)
        if failures:
            failed_images.append(str(image_path))
        if (index + 1) % 10 == 0 or index + 1 == len(images):
            print(f"[{index + 1}/{len(images)}] {image_path.name}", flush=True)

    elapsed = time.perf_counter() - started
    thresholds = {
        "min_match_rate": args.min_match_rate,
        "min_iou": args.min_iou,
        "max_conf_diff": args.max_conf_diff,
        "min_mask_iou": args.min_mask_iou,
        "max_kpt_dist": args.max_kpt_dist,
        "max_angle_diff": args.max_angle_diff,
        "max_score_diff": args.max_score_diff,
        "max_map_diff": args.max_map_diff,
        "min_mask_agree": args.min_mask_agree,
    }
    report = {
        "meta": {
            "kind": args.kind,
            "task": args.task if args.kind == "yolo" else None,
            "engine": str(engine),
            "onnx": str(onnx),
            "metadata": str(metadata) if metadata else None,
            "images": str(Path(args.images).expanduser()),
            "image_count": len(images),
            "conf": args.conf,
            "nms": args.nms,
            "backends": backends,
            "thresholds": thresholds,
            "elapsed_seconds": round(elapsed, 2),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "total": len(rows),
            "failed": len(failed_images),
            "passed": len(rows) - len(failed_images),
        },
        "failed_images": failed_images,
        "images": rows,
    }

    json_path, csv_path = write_reports(out_dir, report)
    print(f"报告：{json_path}")
    print(f"CSV ：{csv_path}")
    print(f"结果：{report['summary']['passed']}/{report['summary']['total']} 张通过")

    if failed_images:
        for path in failed_images[:10]:
            print(f"  未通过：{path}", file=sys.stderr)
        if len(failed_images) > 10:
            print(f"  …… 其余 {len(failed_images) - 10} 张见报告", file=sys.stderr)
        return EXIT_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
