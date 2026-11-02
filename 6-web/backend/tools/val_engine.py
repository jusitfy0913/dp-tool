# -*- coding: utf-8 -*-
"""用**编译好的 TensorRT 引擎**在数据集某个 split 上算精度，产出可直接交付的报告。

为什么要单独有这个东西：训练末尾 Ultralytics 给的 mAP 是 `.pt` 的，
而产线跑的是 `.engine`。两者之间隔着 ONNX 导出、输出转置、fp16 舍入、后处理，
任何一环错了训练日志都看不出来——只有拿引擎真跑一遍才知道。

用法（在 6-web/backend 目录下）：

    python tools/val_engine.py --engine <x.engine> --data <dataset.yaml> \
        --labels <labels_x.json> --split val --conf 0.25 --nms 0.45 --out <目录>

产物：
    val_report.json   指标 + 每类 AP + P-R 曲线（供前端画图）
    val_report.csv    每类一行，给 Excel

事件：每 10 张（或 0.5 秒）打一条 `TRTFORGE_JSON {...}` 到 stdout，
     与训练/批量推理同一套协议，后端 runners 直接能转成 WebSocket 事件。

限制：只支持 detect。segment / pose / obb 的 GT 需要掩码与关键点匹配，
口径完全不同，宁可明确拒绝，也不要算出一个看着像样的错数。
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

import infer  # noqa: E402  依赖 backend 目录在 sys.path 上
import labels as labels_mod  # noqa: E402
import traininfo  # noqa: E402
import valmetrics  # noqa: E402


def emit(payload: dict) -> None:
    """单行 JSON 事件，前端/后端都按行解析。"""
    print("TRTFORGE_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def emit_error(code: str, message: str, error: str = "") -> None:
    emit({"type": "error", "code": code, "message": message, "error": error or message})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 TensorRT 引擎评估检测精度")
    parser.add_argument("--engine", required=True, help=".engine / .trt 绝对路径")
    parser.add_argument("--data", required=True, help="dataset.yaml 绝对路径")
    parser.add_argument("--labels", default="", help="labels_*.json（只用于报告里的类别名）")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--task", default="detect")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="最多评估多少张，0 = 全部")
    parser.add_argument("--min-score", type=float, default=0.001,
                        help="参与 AP 计算的最低置信度，默认与 COCO 一致取 0.001")
    parser.add_argument("--out", required=True, help="报告输出目录")
    return parser.parse_args(argv)


def labels_dir_for(images_dir: Path, yaml_dir: Path, split: str) -> Path:
    """沿用 traininfo 的推导规则（images/xx → labels/xx），不在本文件另写一份。"""
    return traininfo.labels_dir_for(images_dir, yaml_dir, split)


def read_yolo_labels(path: Path, width: int, height: int) -> dict[int, list[list[float]]]:
    """读 YOLO detect 标签，返回 {cls: [[x1,y1,x2,y2], ...]}（像素坐标）。

    容忍的情况：文件不存在、空文件、行尾多余空格、列数不足。真实数据集里这几种
    都存在，直接抛异常会让整次评估白跑。
    """
    result: dict[int, list[list[float]]] = {}
    if not path.is_file():
        return result
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        result.setdefault(cls, []).append([x1, y1, x2, y2])
    return result


def select_images(split_dirs: list[Path], limit: int) -> list[Path]:
    images: list[Path] = []
    for directory in split_dirs:
        images.extend(traininfo.collect_images(directory))
    if limit > 0:
        images = images[:limit]
    return images


def write_csv(path: Path, report: dict, classes: list[str]) -> None:
    """每类一行；与批量推理的 CSV 一致用 utf-8-sig + \\n，Excel 打开不乱码。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["class_id", "class_name", "instances", "predictions", "ap50", "ap"])
        for row in report["per_class"]:
            cls = int(row["id"])
            name = classes[cls] if 0 <= cls < len(classes) else f"cls{cls}"
            writer.writerow([
                cls,
                name,
                row["instances"],
                row.get("predictions", 0),
                "" if row.get("ap50") is None else f"{row['ap50']:.4f}",
                "" if row.get("ap") is None else f"{row['ap']:.4f}",
            ])


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    engine_path = Path(args.engine).expanduser()
    if not engine_path.is_file():
        emit_error("ENGINE_NOT_FOUND", f"引擎不存在：{engine_path}")
        return 2
    if args.task != "detect":
        emit_error(
            "TASK_UNSUPPORTED",
            f"精度评估目前只支持 detect，收到 {args.task}。"
            "分割/姿态/旋转框的 GT 匹配口径不同，需单独实现，不能套用检测的算法。",
        )
        return 2

    status = infer.status()
    if not status.yolo_ok:
        emit_error(
            "DLL_UNAVAILABLE",
            f"yolo_infer.dll 不可用（{status.detail}），无法验证引擎。"
            "降级到 onnxruntime 验的是另一份产物，不能代表引擎精度。",
        )
        return 3

    try:
        splits = traininfo.resolve_splits(args.data)
    except traininfo.OverviewError as exc:
        emit_error("DATA_INVALID", str(exc))
        return 4

    split_dirs = splits["splits"].get(args.split) or []
    if not split_dirs:
        emit_error("SPLIT_EMPTY", f"dataset.yaml 里没有配置 {args.split} 目录")
        return 4

    images = select_images(split_dirs, args.limit)
    if not images:
        emit_error("SPLIT_EMPTY", f"{args.split} 目录里没有图片：{split_dirs}")
        return 4

    classes = [
        entry["name"] if isinstance(entry, dict) else str(entry)
        for entry in (splits.get("classes") or [])
    ]
    if args.labels:
        label_info = labels_mod.load_labels(args.labels)
        if label_info.get("classes"):
            classes = list(label_info["classes"])
    if not classes:
        emit_error("CLASSES_UNKNOWN", "dataset.yaml 与 labels 文件里都没有类别名，无法映射类别")
        return 5

    yaml_dir = Path(splits["yaml_dir"])
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    emit({
        "type": "started",
        "images": len(images),
        "split": args.split,
        "classes": len(classes),
        "engine": engine_path.name,
        "conf": args.conf,
        "nms": args.nms,
    })

    started = time.time()
    evaluated: list[dict] = []
    failed: list[dict] = []
    last_emit = 0.0

    for index, image_path in enumerate(images, start=1):
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
        if image is None:
            failed.append({"file": str(image_path), "error": "decode failed"})
        else:
            height, width = image.shape[:2]
            # 标签目录按 split 目录反推：images/val → labels/val
            for images_dir in split_dirs:
                try:
                    relative = image_path.relative_to(images_dir)
                except ValueError:
                    continue
                label_path = labels_dir_for(images_dir, yaml_dir, args.split) / relative.with_suffix(".txt")
                break
            else:
                label_path = image_path.with_suffix(".txt")

            gts = read_yolo_labels(label_path, width, height)
            try:
                result = infer.run_yolo(
                    str(engine_path), args.task, image, conf=args.min_score, nms=args.nms
                )
                preds = [
                    {
                        "box": [box["left"], box["top"], box["right"], box["bottom"]],
                        "score": box["confidence"],
                        "cls": box["class_label"],
                    }
                    for box in (result.get("boxes") or [])
                ]
            except Exception as exc:  # noqa: BLE001 - 单张失败不应毁掉整次评估
                failed.append({"file": str(image_path), "error": f"{type(exc).__name__}: {exc}"})
                preds = []
                gts = {}
            evaluated.append({"file": str(image_path), "preds": preds, "gts": gts})

        now = time.time()
        if index == len(images) or index % 10 == 0 or now - last_emit >= 0.5:
            last_emit = now
            emit({
                "type": "progress",
                "done": index,
                "total": len(images),
                "pct": round(index / len(images) * 100, 1),
                "failed": len(failed),
            })

    if not evaluated:
        emit_error("NO_VALID_IMAGE", "所有图片都无法解码，无法评估")
        return 6

    report = valmetrics.evaluate(evaluated, len(classes), conf=args.conf)
    report.update({
        "split": args.split,
        "engine": str(engine_path),
        "data": str(Path(args.data).resolve()),
        "labels": args.labels or None,
        "classes": classes,
        "elapsed": round(time.time() - started, 2),
        "failed_images": failed[:50],
        "note": "AP 按 COCO 口径计算（IoU 0.50:0.95、全点插值）；P/R 取自 conf 阈值下 IoU 0.5 的整体统计。",
    })

    json_path = out_dir / "val_report.json"
    csv_path = out_dir / "val_report.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, report, classes)

    emit({
        "type": "summary",
        "map50": report["map50"],
        "map5095": report["map5095"],
        "precision": report["precision"],
        "recall": report["recall"],
        "images": report["num_images"],
        "gts": report["num_gts"],
        "failed": len(failed),
        "elapsed": report["elapsed"],
        "report": str(json_path),
    })
    print(f"[val] mAP50={report['map50']:.4f} mAP50-95={report['map5095']:.4f} "
          f"P={report['precision']:.4f} R={report['recall']:.4f} "
          f"报告={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
