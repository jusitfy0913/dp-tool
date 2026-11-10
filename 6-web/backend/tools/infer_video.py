#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视频推理 CLI（由 6-web 后端以子进程方式调用）。

产出：带检测框的 mp4 + 汇总 JSON（可选逐帧检测明细）。

三个关键取舍：

1. **抽帧而不是逐帧**。30fps 的 1080p 视频逐帧推理要跑几千次，
   而相邻帧的结果差异肉眼看不出来。默认每帧都跑，``--stride N`` 可以
   降到每 N 帧一次；输出视频只写处理过的帧，仍用 ``原fps / N`` 作为帧率，
   所以播放时长与原视频一致（不会变成快进）。

2. **编码器先探测再使用**。OpenCV 的 ``avc1``（H.264）依赖 openh264，
   某些机器静默失败：``isOpened()`` 返回 True，但写出来的文件是 0 字节。
   所以先用一帧黑图试写并检查文件大小，通过才算可用；依次退到 ``mp4v``。
   mp4v（MPEG-4 Part 2）浏览器多半不能直接播，但 Windows 播放器可以，
   实测代码会明确告诉用户实际用了哪个编码器。

3. **进度按时间节流**。逐帧发事件在 60fps 视频上会把 WebSocket 打满；
   这里每 0.5 秒最多一条，且一定带 ETA。
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

#: 编码器候选顺序。avc1/H264 浏览器可播；mp4v 作为最后兜底。
CODEC_CANDIDATES = ("avc1", "H264", "mp4v")


def emit(payload: dict) -> None:
    """按 TRTFORGE_JSON 协议输出一行事件。"""
    print(config.TRAIN_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频推理，输出带框的 mp4")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--task", default="detect", choices=("detect", "segment", "pose", "obb", "classify"))
    parser.add_argument("--labels", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--source", required=True, help="视频文件（mp4 / avi / mov / mkv）")
    parser.add_argument("--output", default="", help="输出 mp4 路径；留空自动建在 runtime/infer 下")
    parser.add_argument("--stride", type=int, default=1, help="每 N 帧处理一次，默认 1")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限")
    parser.add_argument("--max-side", type=int, default=0, help="输出长边上限，0 表示保持原尺寸")
    parser.add_argument("--codec", default="auto", choices=("auto",) + CODEC_CANDIDATES)
    parser.add_argument("--save-detections", action="store_true", help="额外写逐帧检测明细 JSON")
    return parser.parse_args(argv)


def probe_codec(job: Path, fps: float, size: tuple[int, int], wanted: str) -> tuple[str | None, str]:
    """探测可用编码器，返回 (fourcc 或 None, 说明)。

    必须用「写一帧再看文件大小」的方式判断：``isOpened()`` 对 H.264 会撒谎，
    文件 0 字节时它依然返回 True，等到整段跑完才发现白干一场。
    """
    import cv2
    import numpy as np

    candidates = CODEC_CANDIDATES if wanted == "auto" else (wanted,)
    notes: list[str] = []
    probe_file = job / "codec-probe.mp4"

    for name in candidates:
        writer = None
        try:
            writer = cv2.VideoWriter(
                str(probe_file), cv2.VideoWriter_fourcc(*name), max(1.0, fps), size
            )
            if not writer.isOpened():
                notes.append(f"{name}：编码器未打开")
                continue
            writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
            writer.release()
            writer = None
            size_bytes = probe_file.stat().st_size if probe_file.is_file() else 0
            if size_bytes > 0:
                return name, f"使用 {name}（探测文件 {size_bytes} 字节）"
            notes.append(f"{name}：写入 0 字节")
        except Exception as exc:  # noqa: BLE001 - 探测失败继续试下一个
            notes.append(f"{name}：{type(exc).__name__}: {exc}")
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:  # noqa: BLE001
                    pass
            if probe_file.is_file():
                try:
                    probe_file.unlink()
                except OSError:
                    pass

    return None, "；".join(notes) or "没有可用的视频编码器"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import cv2

    # 关掉 OpenCV/FFmpeg 自己的 stderr 噪声：没有 openh264 时它会打印
    # "Failed to load OpenH264 library"，但随后仍能正常用内置编码器写出 H.264。
    # 这些行会被 RunManager 当成日志推给用户，看起来像失败，实际任务成功。
    # 真正的编码失败由 probe_codec 的「写一帧看字节数」与显式 error 事件兜住。
    cv2.setLogLevel(0)

    source = Path(args.source).expanduser()
    if not source.is_file():
        emit({"type": "error", "code": "SOURCE_INVALID", "message": f"视频文件不存在：{source}"})
        return 1

    if args.output.strip():
        out_path = Path(args.output).expanduser()
        if out_path.suffix.lower() != ".mp4":
            out_path = out_path.with_suffix(".mp4")
        job = out_path.parent
    else:
        job = inferjobs.new_job_dir("video")
        out_path = job / f"{inferjobs.safe_stem(source.stem)}_annotated.mp4"

    try:
        job.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        emit({"type": "error", "code": "OUTPUT_INVALID", "message": f"输出目录不可用：{exc}"})
        return 1

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        emit({"type": "error", "code": "VIDEO_UNREADABLE", "message": f"无法打开视频：{source}"})
        return 1

    src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= src_fps <= 240.0):
        # 容器没写帧率时 CAP_PROP_FPS 会返回 0 或 nan，按 25fps 处理并告知
        src_fps = 25.0
    src_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(args.stride))

    out_w, out_h, scale = src_w, src_h, 1.0
    if args.max_side and max(src_w, src_h) > args.max_side:
        scale = float(args.max_side) / float(max(src_w, src_h))
        out_w, out_h = max(2, int(src_w * scale)), max(2, int(src_h * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2

    label_info = labels_mod.load_labels(args.labels)
    if args.labels and not label_info["ok"]:
        emit({"type": "warning", "code": "LABELS_UNUSED", "message": label_info["error"]})
    if args.task == "classify" and not infer.is_onnx_model(args.engine):
        emit({
            "type": "error",
            "code": "ENGINE_UNAVAILABLE",
            "message": "分类引擎还没做，请改用 ONNX。",
        })
        capture.release()
        return 1

    backend = "yolo_infer.dll"
    onnx_path = ""
    status = infer.status()
    is_onnx = infer.is_onnx_model(args.engine)
    dll_ok = status.onnx_ok if is_onnx else status.yolo_ok
    if args.task == "classify":
        # 分类只有 .onnx 一条路（TensorRT 分类未做），但要区分是 DLL 还是 onnxruntime。
        if infer.classify_available() and Path(args.engine).is_file():
            backend = "onnx_infer.dll"
        else:
            backend = "onnxruntime"
    elif dll_ok:
        backend = "onnx_infer.dll" if is_onnx else "yolo_infer.dll"
    else:
        if is_onnx:
            # 用户直接给了 .onnx：只是退到 Python 兜底，模型路径本身就是 onnx
            if not Path(args.engine).is_file():
                emit({
                    "type": "error",
                    "code": "ENGINE_UNAVAILABLE",
                    "message": f"ONNX 模型不存在：{args.engine}",
                })
                capture.release()
                return 1
            onnx_path = args.engine
        else:
            onnx_path, reason = onnx_fallback.resolve_onnx_path(args.engine, "")
            if onnx_path is None:
                emit({
                    "type": "error",
                    "code": "ENGINE_UNAVAILABLE",
                    "message": f"yolo_infer.dll 不可用（{status.detail or '未知原因'}），且无法降级：{reason}",
                })
                capture.release()
                return 1
        backend = "onnxruntime"

    # 输出帧率 = 原帧率 / 抽帧步长：处理过的帧按这个帧率播放，时长才与原视频一致
    out_fps = max(1.0, src_fps / stride)
    codec, codec_note = probe_codec(job, out_fps, (out_w, out_h), args.codec)
    if codec is None:
        emit({"type": "error", "code": "ENCODER_UNAVAILABLE", "message": f"无法创建视频编码器：{codec_note}"})
        capture.release()
        return 1

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), out_fps, (out_w, out_h))
    if not writer.isOpened():
        emit({"type": "error", "code": "ENCODER_UNAVAILABLE", "message": f"{codec} 写入器创建失败"})
        capture.release()
        return 1

    total_estimate = int(src_count / stride) if src_count > 0 else 0
    if args.max_frames:
        total_estimate = min(total_estimate, args.max_frames) if total_estimate else args.max_frames

    emit({
        "type": "started",
        "backend": backend,
        "codec": codec,
        "codec_note": codec_note,
        "source": str(source),
        "output": str(out_path),
        "src_fps": round(src_fps, 2),
        "out_fps": round(out_fps, 2),
        "src_size": [src_w, src_h],
        "out_size": [out_w, out_h],
        "stride": stride,
        "total_frames": total_estimate,
        "duration_seconds": round(src_count / src_fps, 1) if src_count else None,
    })

    started = time.perf_counter()
    read_index = 0
    processed = 0
    written = 0
    detections = 0
    class_counter: dict[str, int] = {}
    frame_records: list[dict] = []
    warn_count = 0
    last_emit = 0.0

    try:
        while True:
            if args.max_frames and processed >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            read_index += 1
            if (read_index - 1) % stride != 0:
                continue

            try:
                frame_classify: dict | None = None
                if args.task == "classify":
                    import classifyinfer

                    if backend == "onnx_infer.dll":
                        raw = infer.run_classify(args.engine, frame)
                        classified, _warning = classifyinfer.result_from_raw(
                            raw, label_info["classes"], args.conf
                        )
                    else:
                        classified = classifyinfer.run(
                            args.engine, frame, label_info["classes"], args.conf,
                        )
                    verdict = classified.get("classify") or {}
                    frame_classify = verdict
                    items = []
                    canvas = classifyinfer.draw_label(frame, verdict)
                    label_key = str(verdict.get("label") or "")
                    class_counter[label_key] = class_counter.get(label_key, 0) + 1
                    detections += 1
                else:
                    boxes = infer.run_yolo(args.engine, args.task, frame, conf=args.conf, nms=args.nms) \
                        if backend in ("yolo_infer.dll", "onnx_infer.dll") \
                        else onnx_fallback.run_yolo(
                            onnx_path, args.task, frame, conf=args.conf, nms=args.nms,
                            num_classes=len(label_info["classes"]) or None,
                        )
                    result_boxes = infer.attach_labels(boxes, label_info["classes"])
                    items = result_boxes.get("boxes") or []
                    canvas = infer.render_yolo(frame, items, result_boxes.get("class_colors"))
            except Exception as exc:  # noqa: BLE001 - 单帧失败不能中断整段视频
                warn_count += 1
                if warn_count <= 20:
                    emit({
                        "type": "warning",
                        "code": "FRAME_FAILED",
                        "frame": read_index,
                        "message": f"{type(exc).__name__}: {exc}",
                    })
                continue

            if scale != 1.0:
                canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(canvas)
            written += 1
            processed += 1
            detections += len(items)
            for box in items:
                key = str(box.get("label_name") or box.get("class_label", ""))
                class_counter[key] = class_counter.get(key, 0) + 1
            if args.save_detections:
                if frame_classify is not None:
                    # 分类没有框，「逐帧明细」给的是这一帧判成了什么、多少分。
                    frame_records.append({
                        "frame": read_index,
                        "classify": {
                            "label": frame_classify.get("label", ""),
                            "score": frame_classify.get("score", 0.0),
                            "uncertain": bool(frame_classify.get("uncertain")),
                        },
                    })
                else:
                    frame_records.append({
                        "frame": read_index,
                        "count": len(items),
                        "boxes": items,
                    })

            now = time.perf_counter()
            if now - last_emit >= 0.5 or (args.max_frames and processed >= args.max_frames):
                last_emit = now
                elapsed = now - started
                percent = round(processed * 100.0 / total_estimate, 1) if total_estimate else None
                eta = round(elapsed / processed * (total_estimate - processed), 1) \
                    if total_estimate and processed else None
                emit({
                    "type": "progress",
                    "frame": read_index,
                    "processed": processed,
                    "written": written,
                    "total_frames": total_estimate,
                    "percent": percent,
                    "detections": detections,
                    "fps": round(processed / elapsed, 2) if elapsed > 0 else None,
                    "elapsed": round(elapsed, 2),
                    "eta": eta,
                })
    finally:
        try:
            writer.release()
        except Exception:  # noqa: BLE001
            pass
        capture.release()

    elapsed = time.perf_counter() - started
    if written == 0:
        emit({
            "type": "error",
            "code": "NO_FRAMES_WRITTEN",
            "message": "没有成功写入任何帧：请检查视频能否解码，或调小 --stride",
        })
        return 1

    detections_path: Path | None = None
    if args.save_detections:
        detections_path = inferjobs.write_json(job / "detections.json", {
            "source": str(source),
            "stride": stride,
            "frames": frame_records,
        })

    summary = {
        "type": "summary",
        "source": str(source),
        "output": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2) if out_path.is_file() else 0,
        "codec": codec,
        "codec_note": codec_note,
        "src_fps": round(src_fps, 2),
        "out_fps": round(out_fps, 2),
        "stride": stride,
        "frames_processed": processed,
        "frames_written": written,
        "detections": detections,
        "class_histogram": class_counter,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(processed / elapsed, 2) if elapsed > 0 else None,
        "detections_json": str(detections_path) if detections_path else None,
        "backend": backend,
    }
    inferjobs.write_json(job / "job.json", {"kind": "video", "args": vars(args), **summary})
    emit(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
