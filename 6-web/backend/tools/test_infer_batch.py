# -*- coding: utf-8 -*-
"""目录批量推理的命令拼装、脚本执行与摄像头单帧接口验收，不需要 GPU / DLL。

守的是「表单填了、命令行没传」和「批处理被一张坏图拖死」这两类问题：

1. 参数漏传：``infer_batch.py`` 有 20 多个可选参数，漏一个不会报错，
   只会悄悄用默认值——用户看到的是"我明明勾了全图 JSON，怎么没有框"。
   所以这里逐条断言命令行里的键值。
2. 单张失败中断整批：现场目录里混着 PNG 截图、0 字节文件是常态。
   一张坏图必须记进 CSV 的 error 列并继续，不能整批退出。
3. 摄像头接口回传整图：单图接口带 base64 标注图，摄像头若照抄，
   1080p 每帧 400KB、10fps 就是 4MB/s，页面会直接卡死。
   这里断言 `/api/infer/frame` 的响应里**没有** annotated_png。

用法：
    python 6-web/backend/tools/test_infer_batch.py
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
from fastapi import HTTPException, UploadFile  # noqa: E402

import app  # noqa: E402
import config  # noqa: E402
import infer_batch  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def as_pairs(command: list[str]) -> dict[str, str]:
    """把 --key value 形式解析成字典（标记型参数不出现在结果里）。"""
    out: dict[str, str] = {}
    for index, token in enumerate(command):
        if token.startswith("--") and index + 1 < len(command):
            nxt = command[index + 1]
            if not nxt.startswith("--"):
                out[token] = nxt
    return out


def expect_http(call, status: int, label: str) -> None:
    try:
        call()
        check(label, False, "没有抛 HTTPException")
    except HTTPException as exc:
        check(label, exc.status_code == status, f"{exc.status_code} != {status}：{exc.detail}")


def write_image(path: Path, color: tuple[int, int, int] = (10, 120, 200)) -> None:
    """写一张真实可解码的小图（cv2 编码，不依赖磁盘上的样例图）。"""
    import cv2

    array = np.zeros((48, 64, 3), dtype=np.uint8)
    array[:, :] = color
    ok, buffer = cv2.imencode(".png", array)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def run_batch(argv: list[str]) -> tuple[int, list[dict]]:
    """在进程内跑一次批处理脚本，收集它打出的 TRTFORGE_JSON 事件。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = infer_batch.main(argv)
    events = []
    for line in buffer.getvalue().splitlines():
        if line.startswith(config.TRAIN_EVENT_PREFIX):
            events.append(json.loads(line[len(config.TRAIN_EVENT_PREFIX):]))
    return code, events


class FakeStatus:
    """伪造推理库状态：两个后端都可用，脚本就不会走 ONNX 降级分支。"""

    yolo_ok = True
    anomalib_ok = True
    detail = ""


def main() -> int:
    config.ensure_dirs()
    root = Path(tempfile.mkdtemp(prefix="test-infer-batch-", dir=str(config.RUNTIME_DIR)))
    original_status = infer_batch.infer.status
    original_run_yolo = infer_batch.infer.run_yolo
    original_once = app._run_yolo_once

    try:
        engine = root / "best_fp16.engine"
        engine.write_bytes(b"fake-engine")
        labels = root / "labels_v1.json"
        labels.write_text(json.dumps({"name": "v1", "classes": ["cat", "dog"]}), encoding="utf-8")
        images = root / "images"
        for name in ("a.png", "b.png", "sub/c.png"):
            write_image(images / name)

        job = root / "job"
        job.mkdir()

        # ---- 1. 命令拼装：YOLO ------------------------------------------------
        request = app.BatchInferRequest(
            kind="yolo", engine=str(engine), task="detect", labels=str(labels),
            conf=0.3, nms=0.5, source=str(images), limit=10, jpeg_quality=88,
        )
        command = app.build_batch_command(request, "python.exe", job)
        pairs = as_pairs(command)
        check("调用的是 infer_batch.py", any(token.endswith("infer_batch.py") for token in command), str(command[:2]))
        check("kind 透传", pairs.get("--kind") == "yolo", str(pairs.get("--kind")))
        check("engine 透传", pairs.get("--engine") == str(engine), str(pairs.get("--engine")))
        check("source 透传", pairs.get("--source") == str(images), str(pairs.get("--source")))
        check("output 指向任务目录", pairs.get("--output") == str(job), str(pairs.get("--output")))
        check("task 透传", pairs.get("--task") == "detect", str(pairs.get("--task")))
        check("conf 透传", pairs.get("--conf") == "0.3", str(pairs.get("--conf")))
        check("nms 透传", pairs.get("--nms") == "0.5", str(pairs.get("--nms")))
        check("limit 透传", pairs.get("--limit") == "10", str(pairs.get("--limit")))
        check("jpeg 质量透传", pairs.get("--jpeg-quality") == "88", str(pairs.get("--jpeg-quality")))
        check("labels 透传", pairs.get("--labels") == str(labels), str(pairs.get("--labels")))
        check("默认递归（没传 --no-recursive）", "--no-recursive" not in command, str(command))
        check("默认打包（没传 --no-zip）", "--no-zip" not in command, str(command))
        check("默认全量 JSON（没传 --no-full-json）", "--no-full-json" not in command, str(command))
        check("YOLO 不传 anomalib 参数", "--metadata" not in command, str(command))

        # ---- 2. 命令拼装：Anomalib -------------------------------------------
        metadata = root / "metadata.json"
        metadata.write_text(json.dumps({"image_threshold": 0.4, "pixel_threshold": 0.5}), encoding="utf-8")
        anom = app.BatchInferRequest(
            kind="anomalib", engine=str(engine), source=str(images), metadata=str(metadata),
            model_type="PatchCore", image_threshold=0.42, pixel_threshold=0.55,
            colormap="VIRIDIS", heatmap=True, recursive=False, make_zip=False, full_json=False,
        )
        anom_cmd = app.build_batch_command(anom, "python.exe", job)
        anom_pairs = as_pairs(anom_cmd)
        check("anomalib kind 透传", anom_pairs.get("--kind") == "anomalib", str(anom_pairs.get("--kind")))
        check("metadata 透传", anom_pairs.get("--metadata") == str(metadata), str(anom_pairs.get("--metadata")))
        check("模型类型透传", anom_pairs.get("--model-type") == "PatchCore", str(anom_pairs.get("--model-type")))
        check("图像阈值透传", anom_pairs.get("--image-threshold") == "0.42", str(anom_pairs.get("--image-threshold")))
        check("像素阈值透传", anom_pairs.get("--pixel-threshold") == "0.55", str(anom_pairs.get("--pixel-threshold")))
        check("色带透传", anom_pairs.get("--colormap") == "VIRIDIS", str(anom_pairs.get("--colormap")))
        check("热力图开关生效", "--heatmap" in anom_cmd, str(anom_cmd))
        check("anomalib 不传 task", "--task" not in anom_cmd, str(anom_cmd))
        check("关掉递归生效", "--no-recursive" in anom_cmd, str(anom_cmd))
        check("关掉打包生效", "--no-zip" in anom_cmd, str(anom_cmd))
        check("关掉全量 JSON 生效", "--no-full-json" in anom_cmd, str(anom_cmd))

        # ---- 3. 参数校验 -----------------------------------------------------
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="coco", engine=str(engine), source=str(images)),
                "python.exe", job,
            ),
            400, "非法 kind 被拒",
        )
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(root / "nope.engine"), source=str(images)),
                "python.exe", job,
            ),
            400, "引擎不存在被拒",
        )
        bad_suffix = root / "model.bin"
        bad_suffix.write_bytes(b"x")
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(bad_suffix), source=str(images)),
                "python.exe", job,
            ),
            400, "引擎扩展名非法被拒",
        )
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(engine), source=str(root / "no-dir")),
                "python.exe", job,
            ),
            400, "数据源不存在被拒",
        )
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(engine), source=str(images), labels=str(root / "no.json")),
                "python.exe", job,
            ),
            400, "标签文件不存在被拒",
        )
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="anomalib", engine=str(engine), source=str(images)),
                "python.exe", job,
            ),
            400, "Anomalib 缺 metadata 被拒",
        )
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(engine), source=str(images), task="classify"),
                "python.exe", job,
            ),
            400, "不支持的任务类型被拒",
        )
        # 仓库外的路径：白名单只放仓库根，正常会被 403 拦住
        outside = Path(tempfile.gettempdir()) / "outside-infer-source"
        outside.mkdir(exist_ok=True)
        expect_http(
            lambda: app.build_batch_command(
                app.BatchInferRequest(kind="yolo", engine=str(engine), source=str(outside)),
                "python.exe", job,
            ),
            403, "仓库外路径被拒",
        )

        # ---- 4. 记录聚合 -----------------------------------------------------
        boxes = [
            {"confidence": 0.4, "label_name": "dog", "class_label": 1},
            {"confidence": 0.9, "label_name": "cat", "class_label": 0},
        ]
        label_name, confidence = infer_batch._top_class(boxes)
        check("_top_class 取最高分", label_name == "cat" and abs(confidence - 0.9) < 1e-9, f"{label_name} {confidence}")
        check("空框的 _top_class 返回空", infer_batch._top_class([]) == ("", 0.0))
        record = infer_batch._yolo_record(1, Path("a.png"), {"boxes": boxes}, 12.5, "annotated/00001_a.jpg")
        check("_yolo_record 统计框数", record["count"] == 2, str(record["count"]))
        check("_yolo_record 保留最高分类别", record["top_class"] == "cat", record["top_class"])
        error_record = infer_batch._error_record(2, Path("b.png"), "解码失败")
        check("错误记录带 error 文案", error_record["error"] == "解码失败")
        check("错误记录框数为 0", error_record["count"] == 0)

        # ---- 5. 脚本端到端（伪造推理函数） ------------------------------------
        infer_batch.infer.status = lambda: FakeStatus()

        def fake_run(engine_path, task, array, conf=0.25, nms=0.45):
            return {
                "boxes": [{
                    "left": 1.0, "top": 2.0, "right": 11.0, "bottom": 12.0,
                    "confidence": 0.91, "class_label": 0, "has_mask": False, "is_obb": False,
                }],
                "image_width": int(array.shape[1]),
                "image_height": int(array.shape[0]),
                "backend": "fake",
                "timing": {"total_ms": 1.0},
            }

        infer_batch.infer.run_yolo = fake_run

        code, events = run_batch([
            "--engine", str(engine), "--task", "detect", "--labels", str(labels),
            "--source", str(images), "--output", str(job),
        ])
        summary = next((e for e in events if e.get("type") == "summary"), None)
        check("批处理退出码 0", code == 0, str(code))
        check("有 summary 事件", summary is not None)
        check("处理了 3 张", summary and summary["processed"] == 3, str(summary and summary["processed"]))
        check("没有失败", summary and summary["failed"] == 0, str(summary and summary["failed"]))
        check("检出 3 个框", summary and summary["detections"] == 3, str(summary and summary["detections"]))
        check("类别分布统计正确", summary and summary["class_histogram"].get("cat") == 3, str(summary and summary["class_histogram"]))
        check("progress 事件不超过 100 条", len([e for e in events if e.get("type") == "progress"]) <= 100)
        check("started 事件报告总数", any(e.get("type") == "started" and e.get("total") == 3 for e in events))

        annotated = sorted((job / "annotated").glob("*.jpg"))
        check("标注图 3 张", len(annotated) == 3, str([p.name for p in annotated]))
        check("标注图非空", all(p.stat().st_size > 0 for p in annotated))
        check("标注图按序号排序", annotated[0].name.startswith("00001_"), annotated[0].name)

        result_json = json.loads((job / "results.json").read_text(encoding="utf-8"))
        check("results.json 记录 3 条", len(result_json["files"]) == 3, str(len(result_json["files"])))
        check("类别名已补上", result_json["files"][0]["boxes"][0]["label_name"] == "cat", str(result_json["files"][0]["boxes"][0]))
        check("results.json 有汇总", result_json.get("class_histogram", {}).get("cat") == 3)

        csv_text = (job / "results.csv").read_text(encoding="utf-8-sig")
        check("results.csv 有表头与 3 行", csv_text.count("\n") == 4, csv_text[:120])
        check("CSV 首列是序号", csv_text.splitlines()[1].startswith("1,"), csv_text.splitlines()[1])

        check("annotated.zip 生成", (job / "annotated.zip").is_file())
        import zipfile

        with zipfile.ZipFile(job / "annotated.zip") as archive:
            names = archive.namelist()
        check("zip 含 3 张图 + json + csv", len([n for n in names if n.endswith(".jpg")]) == 3
              and any(n.endswith(".json") for n in names)
              and any(n.endswith(".csv") for n in names), str(names))

        # ---- 6. 坏图不拖死整批 ------------------------------------------------
        bad_dir = root / "images-bad"
        write_image(bad_dir / "good1.png")
        (bad_dir / "broken.png").write_bytes(b"this is not an image")
        bad_job = root / "job-bad"
        code, events = run_batch([
            "--engine", str(engine), "--labels", str(labels),
            "--source", str(bad_dir), "--output", str(bad_job),
        ])
        summary = next((e for e in events if e.get("type") == "summary"), None)
        check("有坏图仍退出 0", code == 0, str(code))
        check("坏图计入 failed", summary and summary["failed"] == 1, str(summary and summary["failed"]))
        check("好图照常处理", summary and summary["processed"] == 2, str(summary and summary["processed"]))
        bad_rows = json.loads((bad_job / "results.json").read_text(encoding="utf-8"))["files"]
        check("坏图在结果里留痕", any(row["error"] for row in bad_rows), str([r["error"] for r in bad_rows]))
        check("坏图有 warning 事件", any(e.get("type") == "warning" for e in events))

        # ---- 7. 空目录与引擎不可用 --------------------------------------------
        empty = root / "empty"
        empty.mkdir()
        code, events = run_batch(["--engine", str(engine), "--source", str(empty), "--output", str(root / "job-empty")])
        check("空目录退出码非 0", code == 1, str(code))
        check("空目录报 NO_IMAGES", any(e.get("code") == "NO_IMAGES" for e in events), str(events[-1:]))

        infer_batch.infer.status = lambda: type("S", (), {"yolo_ok": False, "anomalib_ok": False, "detail": "DLL 缺失"})()
        code, events = run_batch(["--engine", str(engine), "--source", str(images), "--output", str(root / "job-nodll")])
        check("引擎不可用退出码非 0", code == 1, str(code))
        check("引擎不可用报 ENGINE_UNAVAILABLE", any(e.get("code") == "ENGINE_UNAVAILABLE" for e in events), str(events[-1:]))

        # ---- 8. 摄像头单帧接口 ------------------------------------------------
        frame_png = (images / "a.png").read_bytes()
        fake_result = {
            "boxes": [{"left": 1, "top": 1, "right": 5, "bottom": 5, "confidence": 0.8, "class_label": 0}],
            "count": 1,
            "class_names": ["cat"],
            "class_colors": {"0": [34, 211, 238]},
            "image_width": 64,
            "image_height": 48,
            "timing": {"inference_ms": 3.2},
        }
        app._run_yolo_once = lambda *args, **kwargs: (dict(fake_result), None)
        upload = UploadFile(filename="frame.jpg", file=io.BytesIO(frame_png))
        response = app.infer_frame(image=upload, engine=str(engine), task="detect", conf=0.25, nms=0.45, labels=str(labels))
        payload = json.loads(response.body)
        check("单帧接口返回框", payload.get("count") == 1, str(payload.get("count")))
        check("单帧接口不回传标注图", "annotated_png" not in payload, str(list(payload.keys())))
        check("单帧接口带配色", payload.get("class_colors", {}).get("0") == [34, 211, 238])
        check("单帧接口带耗时", payload.get("timing", {}).get("inference_ms") == 3.2)

        huge = UploadFile(filename="big.jpg", file=io.BytesIO(b"0" * (9 * 1024 * 1024)))
        expect_http(
            lambda: app.infer_frame(image=huge, engine=str(engine), task="detect", conf=0.25, nms=0.45, labels=""),
            413, "超大单帧被拒",
        )

        # ---- 9. OCR 目录批量的参数与拒绝 ------------------------------------------
        det = config.REPO_ROOT / "9-python-paddleOCR" / "model" / "legacy_v4" / "ch_PP-OCRv4_det_infer"
        rec = config.REPO_ROOT / "9-python-paddleOCR" / "model" / "legacy_v4" / "ch_PP-OCRv4_rec_infer"
        if det.is_dir() and rec.is_dir():
            ocr_request = app.BatchInferRequest(
                kind="ocr", det_dir=str(det), rec_dir=str(rec), source=str(images),
            )
            ocr_cmd = app.build_batch_command(ocr_request, sys.executable, root / "ocr-job")
            check("OCR 批量传入检测目录", "--det-dir" in ocr_cmd and str(det) in ocr_cmd)
            check("OCR 批量传入识别目录", "--rec-dir" in ocr_cmd and str(rec) in ocr_cmd)
        code, events = run_batch([
            "--kind", "ocr",
            "--det-dir", str(root / "missing-det"),
            "--rec-dir", str(root / "missing-rec"),
            "--source", str(images),
            "--output", str(root / "job-ocr"),
        ])
        check("OCR 模型目录不存在时失败", code == 1, str(code))
        check(
            "OCR 预检失败记为引擎不可用",
            any(e.get("code") == "ENGINE_UNAVAILABLE" for e in events),
            str(events[-1:]),
        )
        expect_http(
            lambda: app.build_video_command(
                app.VideoInferRequest(kind="ocr", engine=str(engine), source=str(images / "a.png")),
                sys.executable,
                root / "out.mp4",
            ),
            400,
            "视频拒绝 OCR",
        )

        # ---- 10. 产物下载接口 -------------------------------------------------
        infer_root = config.RUNTIME_DIR / "infer"
        infer_root.mkdir(parents=True, exist_ok=True)
        artifact = infer_root / "test-artifact.txt"
        artifact.write_text("ok", encoding="utf-8")
        file_response = app.infer_artifact(str(artifact), download=True)
        check("产物下载返回文件", Path(file_response.path) == artifact.resolve(), str(file_response.path))
        check("下载时带文件名", "attachment" in file_response.headers.get("content-disposition", ""),
              file_response.headers.get("content-disposition", ""))
        expect_http(
            lambda: app.infer_artifact(str(root / "outside.txt"), download=False),
            403, "产物目录外被拒",
        )
        artifact.unlink()
    finally:
        infer_batch.infer.status = original_status
        infer_batch.infer.run_yolo = original_run_yolo
        app._run_yolo_once = original_once
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
