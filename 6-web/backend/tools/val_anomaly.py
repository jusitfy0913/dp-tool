# -*- coding: utf-8 -*-
"""用**编译好的 TensorRT 引擎**评估异常检测精度，产出可交付的报告。

为什么要单独有这个东西：训练日志里的分数是 torch 模型在验证集上的结果，
而产线跑的是 `.engine`。中间隔着 ONNX 导出、阈值校准、fp16 舍入，
任何一环偏了训练日志都看不出来——只有拿引擎把 test 集真跑一遍才知道。

与 YOLO 的 `val_engine.py` 的区别（口径不同，不能套用）：

| | YOLO | Anomalib |
| --- | --- | --- |
| 标签 | 每张图若干框 | 每张图 OK / NG 一个标签 |
| 主指标 | mAP50 / mAP50-95 | 图像级 AUROC |
| 阈值 | `conf` 用来过滤框 | metadata 的 `image_threshold` 决定 OK/NG |
| 验收口径 | mAP 越高越好 | 漏检 = 0，误报率 < 5% |

用法（在 6-web/backend 目录下）：

    python tools/val_anomaly.py --engine <x.engine> --metadata <metadata.json> \
        --normal-dir <root>/test/good --abnormal-dir <root>/test/abnormal \
        --out <目录>

产物：
    anomaly_val_report.json   指标 + 每张图的分数与判定（供前端展示与回溯）
    anomaly_val_report.csv    每张图一行，给 Excel

事件：每 10 张（或 0.5 秒）打一条 `TRTFORGE_JSON {...}`，与其他脚本同一套协议。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import anomalymetrics  # noqa: E402
import infer  # noqa: E402
import traininfo  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def emit(payload: dict) -> None:
    print("TRTFORGE_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def emit_error(code: str, message: str) -> None:
    emit({"type": "error", "code": code, "message": message, "error": message})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 TensorRT 引擎评估异常检测精度")
    parser.add_argument("--engine", required=True, help=".engine / .trt 绝对路径")
    parser.add_argument("--metadata", required=True, help="metadata.json 绝对路径")
    parser.add_argument("--normal-dir", default="", help="正常测试图目录（label = 0）")
    parser.add_argument("--abnormal-dir", default="", help="异常测试图目录（label = 1）")
    parser.add_argument("--normal-train-dir", default="",
                        help="可选：训练时的正常目录，用来算「训练集误报率」")
    parser.add_argument("--model-type", default="Auto", help="传给 DLL 的模型类型，一般 Auto")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个类别最多评估多少张，0 = 全部（调试用）")
    parser.add_argument("--out", required=True, help="报告输出目录")
    return parser.parse_args(argv)


def collect(directory: Path, limit: int) -> list[Path]:
    """递归收图，复用 traininfo 的规则（后缀、排序），不在这里另写一份。"""
    if not directory.is_dir():
        return []
    images = traininfo.collect_images(directory)
    if limit > 0:
        images = images[:limit]
    return images


def read_image(path: Path) -> np.ndarray | None:
    """按字节解码：`cv2.imread` 在中文路径上直接返回 None。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def write_csv(path: Path, rows: list[dict]) -> None:
    """每张图一行；`utf-8-sig` + `\\n`，Excel 打开中文路径不乱码。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["file", "label", "raw_score", "normalized_score", "pred", "correct", "total_ms"]
        )
        for row in rows:
            writer.writerow([
                row["file"],
                "NG" if row["label"] == 1 else "OK",
                f"{row['raw_score']:.6f}",
                f"{row['normalized_score']:.6f}",
                "NG" if row["pred"] == 1 else "OK",
                "1" if row["pred"] == row["label"] else "0",
                f"{row.get('total_ms', 0.0):.2f}",
            ])


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    engine_path = Path(args.engine).expanduser()
    metadata_path = Path(args.metadata).expanduser()
    if not engine_path.is_file():
        emit_error("ENGINE_NOT_FOUND", f"引擎不存在：{engine_path}")
        return 2
    if not metadata_path.is_file():
        emit_error("METADATA_NOT_FOUND", f"metadata.json 不存在：{metadata_path}")
        return 2

    status = infer.status()
    if not status.anomalib_ok:
        emit_error(
            "DLL_UNAVAILABLE",
            f"anomalib_infer.dll 不可用（{status.detail}），无法验证引擎。"
            "降级到 onnxruntime 验的是另一份产物，不能代表引擎精度。",
        )
        return 3

    normal_dir = Path(args.normal_dir).expanduser() if args.normal_dir else None
    abnormal_dir = Path(args.abnormal_dir).expanduser() if args.abnormal_dir else None
    if (normal_dir is None or not normal_dir.is_dir()) and (
        abnormal_dir is None or not abnormal_dir.is_dir()
    ):
        emit_error(
            "SPLIT_EMPTY",
            "正常与异常测试目录都不存在或为空，无法评估："
            f"normal={args.normal_dir!r} abnormal={args.abnormal_dir!r}",
        )
        return 4

    normal_images = collect(normal_dir, args.limit) if normal_dir else []
    abnormal_images = collect(abnormal_dir, args.limit) if abnormal_dir else []
    if not normal_images and not abnormal_images:
        emit_error("SPLIT_EMPTY", "测试目录里没有图片（或后缀都不在支持列表内）")
        return 4

    originals = infer.read_metadata_thresholds(str(metadata_path))
    if originals.get("error"):
        emit_error("METADATA_INVALID", originals["error"])
        return 4
    image_threshold = originals["image_threshold"]
    if image_threshold is None:
        # 没有 image_threshold 时 AUROC 仍可算，但「漏检率/误报率」无从谈起
        emit(
            {
                "type": "warning",
                "message": "metadata.json 里没有 image_threshold，判定阈值将无法复现，"
                           "报告只给 AUROC 与最优阈值建议。",
            }
        )

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, int]] = [(path, 0) for path in normal_images]
    jobs += [(path, 1) for path in abnormal_images]

    emit({
        "type": "started",
        "images": len(jobs),
        "normal": len(normal_images),
        "abnormal": len(abnormal_images),
        "engine": engine_path.name,
        "image_threshold": image_threshold,
    })

    started = time.time()
    rows: list[dict] = []
    failed: list[dict] = []
    last_emit = 0.0

    for index, (image_path, label) in enumerate(jobs, start=1):
        image = read_image(image_path)
        if image is None:
            failed.append({"file": str(image_path), "error": "decode failed"})
        else:
            try:
                result = infer.run_anomalib_score(
                    str(engine_path),
                    str(metadata_path),
                    image,
                    model_type=args.model_type,
                )
                rows.append({
                    "file": str(image_path),
                    "label": label,
                    "raw_score": result["raw_score"],
                    "normalized_score": result["normalized_score"],
                    "pred": 1 if result["is_anomaly"] else 0,
                    "total_ms": result.get("total_ms", 0.0),
                })
            except Exception as exc:  # noqa: BLE001 - 单张失败不该毁掉整次评估
                failed.append({"file": str(image_path), "error": f"{type(exc).__name__}: {exc}"})

        now = time.time()
        if index == len(jobs) or index % 10 == 0 or now - last_emit >= 0.5:
            last_emit = now
            emit({
                "type": "progress",
                "done": index,
                "total": len(jobs),
                "pct": round(index / len(jobs) * 100, 1),
                "failed": len(failed),
            })

    if not rows:
        emit_error("NO_VALID_IMAGE", "所有图片都无法解码或推理失败，无法评估")
        return 6

    scores = [row["raw_score"] for row in rows]
    labels = [row["label"] for row in rows]
    predictions = [row["pred"] for row in rows]

    auroc = anomalymetrics.image_auroc(scores, labels)
    best = anomalymetrics.best_f1_threshold(scores, labels)

    # 与 DLL 判定口径的一致性自检：用「raw >= image_threshold」重算一遍，
    # 和 DLL 的 is_anomaly 比。不一致说明分数与阈值不是同一套语义
    # （例如 metadata 被手工改过、或引擎导出的输出契约变了），
    # 这种情况下报告的漏检率不可信，必须显式告诉用户。
    at_metadata: dict | None = None
    mismatches = 0
    if image_threshold is not None:
        at_metadata = anomalymetrics.confusion_at(scores, labels, image_threshold)
        recomputed = [1 if score >= image_threshold else 0 for score in scores]
        mismatches = sum(1 for a, b in zip(recomputed, predictions) if a != b)

    counts = {
        "normal": sum(1 for label in labels if label == 0),
        "abnormal": sum(1 for label in labels if label == 1),
    }
    report: dict = {
        "kind": "anomaly",
        "model": {
            "engine": str(engine_path),
            "metadata": str(metadata_path),
            "image_threshold": image_threshold,
            "pixel_threshold": originals.get("pixel_threshold"),
            "score_range": originals.get("pred_scores"),
        },
        "num_images": len(rows),
        "counts": counts,
        "failed_images": failed[:50],
        "image": {
            "auroc": None if auroc is None else round(auroc, 6),
            "auroc_note": None
            if auroc is not None
            else "样本只有单一类别，AUROC 无定义（既没有正常图或没有异常图）",
            "at_metadata_threshold": at_metadata,
            "best_f1": best,
            "decision_mismatch": mismatches,
        },
        "pixel": {
            "available": False,
            "reason": "本版只做图像级：像素级需要 ground_truth 掩码，"
                      "且 PRO/AUPRO 的成本远高于图像级，等有真实需求再加。",
        },
        "elapsed": round(time.time() - started, 2),
        "note": "AUROC 用秩统计（Mann-Whitney U，同分取平均秩）；"
                "阈值下指标以 metadata 的 image_threshold 判定，"
                "与 DLL 的 normalized_score >= 0.5 等价。",
    }

    json_path = out_dir / "anomaly_val_report.json"
    csv_path = out_dir / "anomaly_val_report.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    summary = {
        "type": "summary",
        "image_auroc": report["image"]["auroc"],
        "images": report["num_images"],
        "normal": counts["normal"],
        "abnormal": counts["abnormal"],
        "failed": len(failed),
        "elapsed": report["elapsed"],
        "report": str(json_path),
    }
    if at_metadata:
        summary.update({
            "precision": at_metadata["precision"],
            "recall": at_metadata["recall"],
            "f1": at_metadata["f1"],
            "fpr": at_metadata["fpr"],
            "fnr": at_metadata["fnr"],
        })
    emit(summary)

    text = f"[val_anomaly] images={report['num_images']} " + (
        f"AUROC={report['image']['auroc']:.4f}" if report["image"]["auroc"] is not None else "AUROC=n/a"
    )
    if at_metadata:
        text += (
            f" 阈值下 P={at_metadata['precision']:.4f} R={at_metadata['recall']:.4f}"
            f" 漏检={at_metadata['fnr']:.2%} 误报={at_metadata['fpr']:.2%}"
        )
    if mismatches:
        text += f" [警告] 与引擎判定有 {mismatches} 张不一致"
    print(text + f" 报告={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
