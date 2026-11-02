#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目录批量推理 CLI（由 6-web 后端以子进程方式调用）。

为什么是子进程而不是在服务里开线程：
    批量跑 5000 张要几分钟，用户随时可能点「停止」或直接关页面。
    子进程能被 terminate 干净杀掉；服务内线程杀掉只能靠标志位，遇到
    DLL 内部阻塞（PatchCore 首帧加载）时根本停不下来。

事件协议（stdout，供 runners.py 解析后经 WebSocket 推送）：
    started   总张数、后端、产物目录
    progress  每完成 1/N 张推一次（N = max(1, total//100)，最多约 100 条）
    warning   单张失败等非致命问题
    summary   汇总 + 产物路径 + zip 路径
    error     致命错误（此时退出码非 0）

退出码：
    0 全部完成（个别图片失败也算完成，失败项记在 results.csv 的 error 列）
    1 致命错误（引擎加载失败、目录里没有图片、产物写不进去）

用法示例：
    python infer_batch.py --engine best_fp16.engine --task detect \\
        --labels labels_v1.json --source D:\\pcb\\images --output D:\\pcb\\out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import infer  # noqa: E402
import inferjobs  # noqa: E402
import labels as labels_mod  # noqa: E402
import onnx_fallback  # noqa: E402


def emit(payload: dict) -> None:
    """按 TRTFORGE_JSON 协议输出一行事件。"""
    print(config.TRAIN_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


class EngineRunner:
    """统一「DLL 优先 / ONNX 降级」两种后端，循环里不再判断。

    降级只影响速度，不影响结果结构：两种后端返回的字典字段是一致的，
    所以批量循环、CSV 列、前端展示都不用分叉。
    """

    def __init__(self, args: argparse.Namespace, classes: list[str]) -> None:
        self.kind = args.kind
        self.classes = classes
        status = infer.status()

        if self.kind == "yolo":
            self.use_dll = status.yolo_ok
            self.onnx_path = ""
            if not self.use_dll:
                self.onnx_path, reason = onnx_fallback.resolve_onnx_path(args.engine, "")
                if self.onnx_path is None:
                    raise RuntimeError(
                        f"yolo_infer.dll 不可用（{status.detail or '未知原因'}），且无法降级：{reason}"
                    )
            self.backend = "yolo_infer.dll" if self.use_dll else "onnxruntime"
        else:
            self.use_dll = status.anomalib_ok
            self.onnx_path = ""
            if not self.use_dll:
                self.onnx_path, reason = onnx_fallback.resolve_onnx_path(args.engine, args.metadata)
                if self.onnx_path is None:
                    raise RuntimeError(
                        f"anomalib_infer.dll 不可用（{status.detail or '未知原因'}），且无法降级：{reason}"
                    )
            if not Path(args.metadata).is_file():
                raise RuntimeError(f"metadata.json 不存在：{args.metadata}")
            self.backend = "anomalib_infer.dll" if self.use_dll else "onnxruntime"

    def run(self, args: argparse.Namespace, array) -> dict:
        if self.kind == "yolo":
            if self.use_dll:
                return infer.run_yolo(args.engine, args.task, array, conf=args.conf, nms=args.nms)
            return onnx_fallback.run_yolo(
                self.onnx_path, args.task, array,
                conf=args.conf, nms=args.nms,
                num_classes=len(self.classes) or None,
            )

        if self.use_dll:
            return infer.run_anomalib(
                args.engine, args.metadata, array,
                model_type=args.model_type,
                image_threshold=args.image_threshold,
                pixel_threshold=args.pixel_threshold,
                colormap=args.colormap,
                want_overlay=True,
                want_heatmap=args.heatmap,
            )
        result = onnx_fallback.run_anomalib(
            self.onnx_path, args.metadata, array, model_type=args.model_type
        )
        # 降级路径没有 C++ 侧渲染，补齐字段让下游只有一条分支
        result.setdefault("heatmap_png", None)
        result.setdefault("source_png", None)
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="目录批量推理（YOLO / Anomalib）")
    parser.add_argument("--kind", choices=("yolo", "anomalib"), default="yolo")
    parser.add_argument("--engine", required=True, help=".engine / .trt / .onnx")
    parser.add_argument("--task", default="detect", choices=("detect", "segment", "pose", "obb"))
    parser.add_argument("--labels", default="", help="labels_*.json；留空则用 cls{id}")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--source", required=True, help="图片目录（或单张图片）")
    parser.add_argument("--output", default="", help="产物目录；留空自动建在 runtime/infer 下")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    parser.add_argument("--no-recursive", action="store_true", help="只扫顶层，不进子目录")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--no-zip", action="store_true", help="不生成 annotated.zip")
    parser.add_argument("--no-full-json", action="store_true", help="results.json 里不写每个框")
    # Anomalib 专用
    parser.add_argument("--metadata", default="")
    parser.add_argument("--model-type", default="Auto")
    parser.add_argument("--image-threshold", type=float, default=-1.0)
    parser.add_argument("--pixel-threshold", type=float, default=-1.0)
    parser.add_argument("--colormap", default="TURBO")
    parser.add_argument("--heatmap", action="store_true", help="额外保存热能图（每张多一个文件）")
    return parser.parse_args(argv)


YOLO_COLUMNS = ["index", "file", "count", "top_class", "max_conf", "elapsed_ms", "output", "error"]
ANOMALIB_COLUMNS = [
    "index", "file", "raw_score", "normalized_score", "is_anomaly",
    "image_threshold", "elapsed_ms", "output", "error",
]


def _top_class(boxes: list[dict]) -> tuple[str, float]:
    """取置信度最高的框的类别名与分数，用于 CSV 快速扫读。"""
    if not boxes:
        return "", 0.0
    best = max(boxes, key=lambda b: float(b.get("confidence", 0.0)))
    return str(best.get("label_name") or best.get("class_label", "")), float(best.get("confidence", 0.0))


def _yolo_record(index: int, file: Path, result: dict, elapsed_ms: float, output: str) -> dict:
    boxes = result.get("boxes") or []
    label, confidence = _top_class(boxes)
    return {
        "index": index,
        "file": str(file),
        "count": len(boxes),
        "top_class": label,
        "max_conf": confidence,
        "elapsed_ms": elapsed_ms,
        "output": output,
        "error": "",
        "boxes": boxes,
        "image_width": result.get("image_width"),
        "image_height": result.get("image_height"),
    }


def _anomalib_record(index: int, file: Path, result: dict, elapsed_ms: float, output: str) -> dict:
    thresholds = result.get("thresholds") or {}
    image_thr = (thresholds.get("image") or {}).get("effective")
    return {
        "index": index,
        "file": str(file),
        "raw_score": result.get("raw_score"),
        "normalized_score": result.get("normalized_score"),
        "is_anomaly": bool(result.get("is_anomaly")),
        "image_threshold": image_thr,
        "elapsed_ms": elapsed_ms,
        "output": output,
        "error": "",
    }


def _error_record(index: int, file: Path, message: str) -> dict:
    """单张失败的记录。批量里一张坏图不能拖死整批，必须继续跑完并留痕。"""
    return {
        "index": index,
        "file": str(file),
        "count": 0,
        "top_class": "",
        "max_conf": 0.0,
        "raw_score": None,
        "normalized_score": None,
        "is_anomaly": False,
        "image_threshold": None,
        "elapsed_ms": 0.0,
        "output": "",
        "error": message,
    }


def _flush(job: Path, args: argparse.Namespace, records: list[dict], extra: dict) -> tuple[Path, Path]:
    """把结果写成 JSON + CSV，返回 (json 路径, csv 路径)。"""
    columns = YOLO_COLUMNS if args.kind == "yolo" else ANOMALIB_COLUMNS
    rows = [
        record if not args.no_full_json
        else {k: v for k, v in record.items() if k != "boxes"}
        for record in records
    ]
    json_path = inferjobs.write_json(job / "results.json", {
        "kind": args.kind,
        "task": args.task,
        "engine": str(Path(args.engine).resolve()),
        "labels": args.labels or None,
        "conf": args.conf,
        "nms": args.nms if args.kind == "yolo" else None,
        "source": str(Path(args.source).resolve()),
        **extra,
        "files": rows,
    })
    csv_path = inferjobs.write_csv(job / "results.csv", columns, records)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    started = time.perf_counter()
    source = Path(args.source).expanduser()
    try:
        images = inferjobs.collect_images(
            source, recursive=not args.no_recursive, limit=args.limit
        )
    except (OSError, FileNotFoundError) as exc:
        emit({"type": "error", "code": "SOURCE_INVALID", "message": str(exc)})
        return 1

    if not images:
        emit({
            "type": "error",
            "code": "NO_IMAGES",
            "message": f"目录里没有可推理的图片（支持 {', '.join(sorted(config.IMAGE_SUFFIXES))}）：{source}",
        })
        return 1

    try:
        job = Path(args.output).expanduser() if args.output.strip() else inferjobs.new_job_dir("batch")
        (job / "annotated").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        emit({"type": "error", "code": "OUTPUT_INVALID", "message": f"产物目录不可用：{exc}"})
        return 1

    label_info = labels_mod.load_labels(args.labels)
    if args.labels and not label_info["ok"]:
        emit({"type": "warning", "code": "LABELS_UNUSED", "message": label_info["error"]})

    try:
        runner = EngineRunner(args, label_info["classes"])
    except Exception as exc:  # noqa: BLE001 - 引擎加载失败必须变成可读事件
        emit({"type": "error", "code": "ENGINE_UNAVAILABLE", "message": str(exc)})
        return 1

    total = len(images)
    emit({
        "type": "started",
        "kind": args.kind,
        "backend": runner.backend,
        "total": total,
        "source": str(source),
        "output": str(job),
        "task": args.task,
        "labels_ok": label_info["ok"],
        "class_count": len(label_info["classes"]),
    })

    records: list[dict] = []
    class_counter: dict[str, int] = {}
    failed = 0
    step = max(1, total // 100)

    try:
        for index, file in enumerate(images, start=1):
            file_started = time.perf_counter()
            try:
                array = infer.decode_image(file.read_bytes())
                result = runner.run(args, array)

                if args.kind == "yolo":
                    infer.attach_labels(result, label_info["classes"])
                out_name = inferjobs.annotated_name(
                    index, file, ".jpg" if args.kind == "yolo" else ".png"
                )
            except Exception as exc:  # noqa: BLE001 - 单张失败不影响整批
                failed += 1
                message = f"{type(exc).__name__}: {exc}"
                records.append(_error_record(index, file, message))
                emit({"type": "warning", "code": "IMAGE_FAILED", "file": str(file), "message": message})
            else:
                elapsed_ms = (time.perf_counter() - file_started) * 1000.0
                relative = f"annotated/{out_name}"
                if args.kind == "yolo":
                    canvas = infer.render_yolo(array, result.get("boxes") or [], result.get("class_colors"))
                    (job / "annotated" / out_name).write_bytes(
                        infer.encode_image(canvas, ".jpg", args.jpeg_quality)
                    )
                    record = _yolo_record(index, file, result, elapsed_ms, relative)
                    for box in result.get("boxes") or []:
                        key = str(box.get("label_name") or box.get("class_label", ""))
                        class_counter[key] = class_counter.get(key, 0) + 1
                else:
                    overlay = result.get("overlay_png")
                    if overlay:
                        (job / "annotated" / out_name).write_bytes(overlay)
                    record = _anomalib_record(index, file, result, elapsed_ms, relative)
                    verdict = "NG" if record["is_anomaly"] else "OK"
                    class_counter[verdict] = class_counter.get(verdict, 0) + 1
                    if args.heatmap and result.get("heatmap_png"):
                        (job / "annotated" / f"{Path(out_name).stem}_heat.png").write_bytes(
                            result["heatmap_png"]
                        )
                records.append(record)

            if index % step == 0 or index == total:
                done_elapsed = time.perf_counter() - started
                emit({
                    "type": "progress",
                    "done": index,
                    "total": total,
                    "percent": round(index * 100.0 / total, 1),
                    "file": str(file),
                    "failed": failed,
                    "detections": sum(int(r.get("count") or 0) for r in records),
                    "elapsed": round(done_elapsed, 2),
                    "eta": round(done_elapsed / index * (total - index), 1),
                })
    except KeyboardInterrupt:
        emit({"type": "warning", "code": "INTERRUPTED", "message": "收到中断，正在保存已完成的结果"})

    elapsed = time.perf_counter() - started
    annotated = sorted((job / "annotated").glob("*"))
    zip_path: Path | None = None
    json_path: Path | None = None
    csv_path: Path | None = None

    try:
        json_path, csv_path = _flush(job, args, records, {
            "total": total,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 2),
            "backend": runner.backend,
            "class_histogram": class_counter,
        })
        if not args.no_zip:
            zip_path = inferjobs.make_zip(
                job / "annotated.zip",
                files=annotated + [json_path, csv_path],
                base_dir=job,
            )
        inferjobs.write_json(job / "job.json", {
            "kind": "batch",
            "args": vars(args),
            "output": str(job),
            "zip": str(zip_path) if zip_path else None,
        })
    except OSError as exc:
        emit({"type": "error", "code": "WRITE_FAILED", "message": f"产物写入失败：{exc}"})
        return 1

    emit({
        "type": "summary",
        "total": total,
        "processed": len(records),
        "failed": failed,
        "annotated": len(annotated),
        "detections": sum(int(r.get("count") or 0) for r in records),
        "class_histogram": class_counter,
        "elapsed_seconds": round(elapsed, 2),
        "avg_ms": round(sum(float(r.get("elapsed_ms") or 0) for r in records) / max(1, len(records)), 2),
        "output": str(job),
        "results_json": str(json_path) if json_path else None,
        "results_csv": str(csv_path) if csv_path else None,
        "zip": str(zip_path) if zip_path else None,
        "backend": runner.backend,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
