# -*- coding: utf-8 -*-
"""视频推理的命令拼装、抽帧与编码器探测验收，不需要 GPU / DLL / 真实权重。

守的是三类「跑完才发现」的问题：

1. **编码器撒谎**：OpenCV 的 ``avc1`` 在没有 openh264 时会 ``isOpened()==True``
   却写出 0 字节文件。只看返回值会把失败当成功，整段视频白跑。
   这里断言 ``probe_codec`` 必须真的写出字节才算通过。
2. **抽帧后时长变短**：``--stride 2`` 时如果输出仍用原帧率，
   视频会变成两倍速。正确做法是 ``out_fps = src_fps / stride``。
3. **事件里没有可显示的进度**：视频任务动辄几分钟，
   没有 percent/eta 的话页面只能显示"转圈"，用户会以为卡死。

用法：
    python 6-web/backend/tools/test_infer_video.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import app  # noqa: E402
import config  # noqa: E402
import infer_video  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def expect_http(call, status: int, label: str) -> None:
    try:
        call()
        check(label, False, "没有抛 HTTPException")
    except HTTPException as exc:
        check(label, exc.status_code == status, f"{exc.status_code} != {status}：{exc.detail}")


def make_video(path: Path, frames: int = 12, fps: float = 12.0, size=(64, 48)) -> bool:
    """生成一小段 mp4；编码器不可用时返回 False，让调用方跳过相关断言。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        return False
    try:
        for index in range(frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :] = (index * 12 % 255, 80, 160)
            writer.write(frame)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def run_video(argv: list[str]) -> tuple[int, list[dict]]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = infer_video.main(argv)
    events = []
    for line in buffer.getvalue().splitlines():
        if line.startswith(config.TRAIN_EVENT_PREFIX):
            events.append(json.loads(line[len(config.TRAIN_EVENT_PREFIX):]))
    return code, events


class FakeStatus:
    yolo_ok = True
    anomalib_ok = False
    detail = ""


def fake_boxes(array):
    return {
        "boxes": [{
            "left": 2.0, "top": 3.0, "right": 20.0, "bottom": 22.0,
            "confidence": 0.88, "class_label": 0, "has_mask": False, "is_obb": False,
        }],
        "image_width": int(array.shape[1]),
        "image_height": int(array.shape[0]),
        "backend": "fake",
        "timing": {"total_ms": 1.0},
    }


def main() -> int:
    config.ensure_dirs()
    root = Path(tempfile.mkdtemp(prefix="test-infer-video-", dir=str(config.RUNTIME_DIR)))
    original_status = infer_video.infer.status
    original_run_yolo = infer_video.infer.run_yolo

    try:
        engine = root / "best_fp16.engine"
        engine.write_bytes(b"fake-engine")
        labels = root / "labels.json"
        labels.write_text(json.dumps({"classes": ["cat"]}), encoding="utf-8")
        video = root / "clip.mp4"
        video_ok = make_video(video)
        check("测试视频生成成功（mp4v 可用）", video_ok, str(video))

        # ---- 1. 命令拼装 -----------------------------------------------------
        request = app.VideoInferRequest(
            engine=str(engine), task="detect", labels=str(labels), conf=0.3, nms=0.55,
            source=str(video), stride=3, max_frames=100, max_side=720,
            codec="auto", save_detections=True,
        )
        output = root / "job" / "out.mp4"
        command = app.build_video_command(request, "python.exe", output)
        pairs = {}
        for index, token in enumerate(command):
            if token.startswith("--") and index + 1 < len(command):
                if not command[index + 1].startswith("--"):
                    pairs[token] = command[index + 1]
        check("调用的是 infer_video.py", any(t.endswith("infer_video.py") for t in command), str(command[:2]))
        check("engine 透传", pairs.get("--engine") == str(engine), str(pairs.get("--engine")))
        check("source 透传", pairs.get("--source") == str(video), str(pairs.get("--source")))
        check("output 透传", pairs.get("--output") == str(output), str(pairs.get("--output")))
        check("stride 透传", pairs.get("--stride") == "3", str(pairs.get("--stride")))
        check("max-frames 透传", pairs.get("--max-frames") == "100", str(pairs.get("--max-frames")))
        check("max-side 透传", pairs.get("--max-side") == "720", str(pairs.get("--max-side")))
        check("conf 透传", pairs.get("--conf") == "0.3", str(pairs.get("--conf")))
        check("nms 透传", pairs.get("--nms") == "0.55", str(pairs.get("--nms")))
        check("labels 透传", pairs.get("--labels") == str(labels), str(pairs.get("--labels")))
        check("逐帧明细开关生效", "--save-detections" in command, str(command))
        check("默认 codec=auto", pairs.get("--codec") == "auto", str(pairs.get("--codec")))

        # ---- 2. 参数校验 -----------------------------------------------------
        not_video = root / "notes.txt"
        not_video.write_text("x", encoding="utf-8")
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(engine=str(engine), source=str(not_video)), "python.exe", output,
            ),
            400, "非视频扩展名被拒",
        )
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(engine=str(engine), source=str(root / "missing.mp4")), "python.exe", output,
            ),
            400, "视频不存在被拒",
        )
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(engine=str(engine), source=str(video), codec="h265"), "python.exe", output,
            ),
            400, "不支持的编码器被拒",
        )
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(engine=str(engine), source=str(video), task="classify"), "python.exe", output,
            ),
            400, "不支持的任务类型被拒",
        )
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(engine=str(root / "none.engine"), source=str(video)), "python.exe", output,
            ),
            400, "引擎不存在被拒",
        )

        # ---- 3. 编码器探测 ---------------------------------------------------
        probe_dir = root / "probe"
        probe_dir.mkdir()
        codec, note = infer_video.probe_codec(probe_dir, 12.0, (64, 48), "auto")
        if codec is None:
            check("编码器探测（本机无可用编码器，已跳过）", True, note)
        else:
            check(f"探测到可用编码器 {codec}", codec in infer_video.CODEC_CANDIDATES, codec)
            check("探测说明可读", bool(note), note)
            check("探测文件已清理", not (probe_dir / "codec-probe.mp4").exists())

        # ---- 4. 端到端（伪造推理） -------------------------------------------
        infer_video.infer.status = lambda: FakeStatus()
        infer_video.infer.run_yolo = lambda engine_path, task, array, conf=0.25, nms=0.45: fake_boxes(array)

        if not video_ok:
            check("视频端到端（测试视频不可用，已跳过）", True, "cv2 编码器不可用")
        else:
            job1 = root / "job1"
            code, events = run_video([
                "--engine", str(engine), "--task", "detect", "--labels", str(labels),
                "--source", str(video), "--output", str(job1 / "out.mp4"),
            ])
            summary = next((e for e in events if e.get("type") == "summary"), None)
            started = next((e for e in events if e.get("type") == "started"), None)
            check("视频任务退出码 0", code == 0, str(code))
            check("产物 mp4 存在且非空", (job1 / "out.mp4").is_file() and (job1 / "out.mp4").stat().st_size > 0)
            check("summary 报告 12 帧", summary and summary["frames_written"] == 12, str(summary and summary["frames_written"]))
            check("summary 检出 12 个", summary and summary["detections"] == 12, str(summary and summary["detections"]))
            check("summary 有类别分布", summary and summary["class_histogram"].get("cat") == 12,
                  str(summary and summary["class_histogram"]))
            check("started 报告原帧率", started and abs(started["src_fps"] - 12.0) < 0.6, str(started and started["src_fps"]))
            check("started 报告输出帧率", started and abs(started["out_fps"] - 12.0) < 0.6, str(started and started["out_fps"]))
            check("started 报告编码器", bool(started and started.get("codec")), str(started and started.get("codec")))

            progress = [e for e in events if e.get("type") == "progress"]
            check("进度事件带 percent", progress and progress[-1].get("percent") is not None, str(progress[-1:]))

            # ---- 5. 抽帧：时长不变 -------------------------------------------
            job2 = root / "job2"
            code, events = run_video([
                "--engine", str(engine), "--source", str(video), "--output", str(job2 / "out.mp4"),
                "--stride", "2",
            ])
            summary = next((e for e in events if e.get("type") == "summary"), None)
            started = next((e for e in events if e.get("type") == "started"), None)
            check("stride=2 退出码 0", code == 0, str(code))
            check("stride=2 只写 6 帧", summary and summary["frames_written"] == 6,
                  str(summary and summary["frames_written"]))
            check(
                "stride=2 输出帧率减半（时长不变）",
                started and abs(started["out_fps"] - started["src_fps"] / 2) < 0.1,
                str(started and (started["src_fps"], started["out_fps"])),
            )

            # ---- 6. max-frames 上限 -----------------------------------------
            job3 = root / "job3"
            code, events = run_video([
                "--engine", str(engine), "--source", str(video), "--output", str(job3 / "out.mp4"),
                "--max-frames", "5",
            ])
            summary = next((e for e in events if e.get("type") == "summary"), None)
            check("max-frames=5 只处理 5 帧", code == 0 and summary and summary["frames_written"] == 5,
                  str(summary and summary["frames_written"]))

            # ---- 7. 逐帧明细 -------------------------------------------------
            job4 = root / "job4"
            code, events = run_video([
                "--engine", str(engine), "--source", str(video), "--output", str(job4 / "out.mp4"),
                "--save-detections", "--max-frames", "3",
            ])
            summary = next((e for e in events if e.get("type") == "summary"), None)
            detections_file = Path(summary["detections_json"]) if summary and summary.get("detections_json") else None
            check("detections.json 已生成", bool(detections_file and detections_file.is_file()),
                  str(summary and summary.get("detections_json")))
            if detections_file and detections_file.is_file():
                payload = json.loads(detections_file.read_text(encoding="utf-8"))
                check("detections.json 记录 3 帧", len(payload["frames"]) == 3, str(len(payload["frames"])))
                check("每帧带框", all(frame["count"] == 1 for frame in payload["frames"]))

        # ---- 8. 全部帧推理失败 -----------------------------------------------
        def broken(engine_path, task, array, conf=0.25, nms=0.45):
            raise RuntimeError("模拟引擎异常")

        infer_video.infer.run_yolo = broken
        if video_ok:
            code, events = run_video([
                "--engine", str(engine), "--source", str(video), "--output", str(root / "job5" / "out.mp4"),
            ])
            check("全帧失败退出码非 0", code == 1, str(code))
            check("全帧失败报 NO_FRAMES_WRITTEN",
                  any(e.get("code") == "NO_FRAMES_WRITTEN" for e in events), str(events[-1:]))
            warnings = [e for e in events if e.get("type") == "warning"]
            check("警告最多 20 条（不刷屏）", 0 < len(warnings) <= 20, str(len(warnings)))

        # ---- 9. 无法解码的视频 -----------------------------------------------
        broken_video = root / "broken.mp4"
        broken_video.write_bytes(b"not a video")
        code, events = run_video([
            "--engine", str(engine), "--source", str(broken_video), "--output", str(root / "job6" / "out.mp4"),
        ])
        check("无法解码退出码非 0", code == 1, str(code))
        check("无法解码报 VIDEO_UNREADABLE",
              any(e.get("code") == "VIDEO_UNREADABLE" for e in events), str(events[-1:]))

        code, events = run_video(["--engine", str(engine), "--source", str(root / "nope.mp4")])
        check("源文件不存在报 SOURCE_INVALID",
              code == 1 and any(e.get("code") == "SOURCE_INVALID" for e in events), str(events[-1:]))
    finally:
        infer_video.infer.status = original_status
        infer_video.infer.run_yolo = original_run_yolo
        shutil.rmtree(root, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
