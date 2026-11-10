# -*- coding: utf-8 -*-
"""分类真机闭环冒烟测试（需要联网 + GPU，不是单测）。

单测只能验证「参数拼得对不对」，验证不了「和 Ultralytics 的约定对不对」。
这个脚本用官方权重和官方数据集，把真实链路走一遍：

    dataset.yaml -> launcher 训练 -> 导出 ONNX -> Ultralytics 预测 vs Web 预测

为什么不用 test_ 前缀：它要下载约 100 MB、要跑训练，不该进 run_all_tests.py。
单测见 test_classify.py。

用法（在 dp-tool 目录下）：

    python 6-web/backend/tools/smoke_classify_e2e.py --prepare   # 只下载和造数据
    python 6-web/backend/tools/smoke_classify_e2e.py --run       # 训练 + 导出 + 对比
    python 6-web/backend/tools/smoke_classify_e2e.py --run --skip-train  # 复用已有 best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
SIX_WEB = HERE.parents[2]                    # 6-web
REPO = SIX_WEB.parent                        # dp-tool
RUNTIME = SIX_WEB / "runtime" / "classify_smoke"
LAUNCHER = REPO / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py"
BACKEND = SIX_WEB / "backend"

WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt"
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenette160.zip"
WEIGHT = RUNTIME / "yolov8n-cls.pt"
DATASET_ZIP = RUNTIME / "imagenette160.zip"
DATA = RUNTIME / "data"

# 两个类别名故意和 Ultralytics 的排序相反。
# Ultralytics 用 sorted(文件夹名) 生成 names：不良(U+4E0D) < 良品(U+826F)，
# 所以模型输出 0 是「不良」。而 Web 导出的 dataset.yaml 写的是 0=良品。
# 这正是真实场景，用来验证推理端是否以 ONNX metadata 为准。
CN2_NAMES = ["良品", "不良"]
CN2_CLASSES = ["n01440764", "n02102040"]   # tench / English springer，仅作两类素材


def log(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def download(url: str, target: Path) -> None:
    """带百分比进度地下载；已存在且非空就跳过。"""
    if target.is_file() and target.stat().st_size > 1024:
        log(f"已存在，跳过下载：{target.name}（{target.stat().st_size / 1048576:.1f} MB）")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    log(f"下载 {url}")
    started = time.time()
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                if int(pct) % 10 == 0:
                    print(f"\r  {pct:5.1f}%  {done / 1048576:6.1f}/{total / 1048576:.1f} MB", end="", flush=True)
    print()
    tmp.replace(target)
    log(f"完成，用时 {time.time() - started:.1f}s，{target.stat().st_size / 1048576:.1f} MB")


def collect_images(directory: Path, limit: int = 0) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in suffixes]
    return files[:limit] if limit else files


def write_yaml(root: Path, names: list[str]) -> Path:
    """按 Web 导出的格式写 dataset.yaml：path 绝对、train/val/test 是目录名。"""
    yaml_path = root / "dataset.yaml"
    lines = [
        f"path: {root.as_posix()}",
        "train: train",
        "val: val",
        "test: test",
        "task: classify",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def prepare() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    download(WEIGHT_URL, WEIGHT)
    download(DATASET_URL, DATASET_ZIP)

    extracted = RUNTIME / "imagenette"
    ready = extracted / "train"
    if not ready.is_dir():
        log("解压 imagenette160")
        with zipfile.ZipFile(DATASET_ZIP) as archive:
            archive.extractall(RUNTIME)
        # 压缩包内可能是 imagenette2-160/ 或 imagenette160/，找带 train 的那层
        for candidate in RUNTIME.glob("imagenette*"):
            if candidate.is_dir() and (candidate / "train").is_dir():
                extracted = candidate
                break
    if not (extracted / "train").is_dir():
        raise SystemExit(f"解压后找不到 train 目录：{extracted}")

    classes = sorted(p.name for p in (extracted / "train").iterdir() if p.is_dir())
    log(f"imagenette 类别数：{len(classes)}")

    # ---- multi：全类别，每类限 40 张 train / 10 张 val，跑得快 ----
    multi = DATA / "multi"
    if not (multi / "train").is_dir():
        for split, limit in (("train", 40), ("val", 10)):
            for name in classes:
                source = extracted / split / name
                if not source.is_dir():
                    continue
                target = multi / split / name
                target.mkdir(parents=True, exist_ok=True)
                for index, image in enumerate(collect_images(source, limit)):
                    shutil.copy2(image, target / f"{index:04d}{image.suffix.lower()}")
        write_yaml(multi, classes)
        log(f"multi 数据集就绪：{multi}")

    # ---- cn2：两类改名成良品/不良，专测中文排序导致的类别错位 ----
    cn2 = DATA / "cn2"
    if not (cn2 / "train").is_dir():
        picked = [name for name in CN2_CLASSES if (extracted / "train" / name).is_dir()]
        if len(picked) < 2:
            # 兜底：取图片最多的两个类别
            ranked = sorted(
                ((name, len(collect_images(extracted / "train" / name))) for name in classes),
                key=lambda item: -item[1],
            )
            picked = [name for name, _ in ranked[:2]]
        for split, limit in (("train", 40), ("val", 10)):
            for source_name, target_name in zip(picked, CN2_NAMES):
                source = extracted / split / source_name
                if not source.is_dir():
                    continue
                target = cn2 / split / target_name
                target.mkdir(parents=True, exist_ok=True)
                for index, image in enumerate(collect_images(source, limit)):
                    shutil.copy2(image, target / f"{index:04d}{image.suffix.lower()}")
        write_yaml(cn2, CN2_NAMES)
        log(f"cn2 数据集就绪：{cn2}（源类别 {picked} -> {CN2_NAMES}）")


def run_launcher(args: list[str]) -> tuple[int, list[dict]]:
    """跑 launcher 并解析结构化事件；返回退出码和事件列表。"""
    command = [sys.executable, str(LAUNCHER), *args]
    log("运行：" + " ".join(command))
    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.run(
        command, cwd=str(LAUNCHER.parent), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    events: list[dict] = []
    for line in (process.stdout or "").splitlines():
        marker = line.find('{"')
        if marker < 0:
            continue
        try:
            events.append(json.loads(line[marker:]))
        except json.JSONDecodeError:
            continue
    errors = [event for event in events if event.get("type") == "error"]
    for event in errors:
        log(f"launcher 报错 {event.get('code')}: {event.get('message')}")
    if process.returncode != 0 and not errors:
        log((process.stderr or "")[-2000:])
    return process.returncode, events


def find_artifact(events: list[dict], kind: str) -> str:
    for event in events:
        if event.get("type") == "artifact" and event.get("kind") == kind:
            return str(event.get("path") or "")
    return ""


def train(data_yaml: Path, name: str, epochs: int, imgsz: int) -> Path:
    project = RUNTIME / "runs"
    code, events = run_launcher(
        [
            "--mode", "train",
            "--task", "classify",
            "--data", str(data_yaml),
            "--model", str(WEIGHT),
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", "16",
            "--device", "0",
            "--project", str(project),
            "--name", name,
            "--workers", "2",
        ]
    )
    if code != 0:
        raise SystemExit(f"训练失败，退出码 {code}")
    top1 = None
    for event in events:
        if event.get("type") == "metric" and event.get("name") == "accuracy_top1":
            top1 = event.get("value")
    log(f"训练完成，top1={top1}")
    best = project / name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练结束但找不到权重：{best}")
    return best


def val(weights: Path, data_yaml: Path) -> None:
    code, events = run_launcher(
        [
            "--mode", "val",
            "--task", "classify",
            "--weights", str(weights),
            "--data", str(data_yaml),
            "--imgsz", "160",
            "--batch", "16",
            "--device", "0",
        ]
    )
    if code != 0:
        raise SystemExit(f"验证失败，退出码 {code}")
    top1 = None
    first_labels = ""
    for event in events:
        if event.get("type") == "metric" and event.get("name") == "accuracy_top1":
            top1 = event.get("value")
        if event.get("type") == "artifact" and event.get("kind") == "labels_json":
            first_labels = str(event.get("path") or "")
    log(f"验证完成，top1={top1}，labels={first_labels}")


def export(weights: Path, imgsz: int) -> Path:
    code, events = run_launcher(
        [
            "--mode", "export",
            "--task", "classify",
            "--weights", str(weights),
            "--format", "onnx",
            "--imgsz", str(imgsz),
            "--device", "0",
        ]
    )
    if code != 0:
        raise SystemExit(f"导出失败，退出码 {code}")
    onnx_path = find_artifact(events, "onnx")
    if not onnx_path:
        raise SystemExit("导出事件里没有 onnx 产物")
    log(f"导出完成：{onnx_path}")
    return Path(onnx_path)


def compare(onnx_path: Path, data_yaml: Path, limit: int = 20) -> list[dict]:
    """逐张对比 Ultralytics 官方预处理和 Web 侧 classifyinfer 的结果。"""
    import cv2

    sys.path.insert(0, str(BACKEND))
    import classifyinfer  # noqa: E402
    import traininfo  # noqa: E402

    images = traininfo.list_split_images(data_yaml, "val")[:limit]
    if not images:
        raise SystemExit("验证集没有图片，无法对比")

    # Ultralytics 官方预测
    from ultralytics import YOLO

    model = YOLO(str(onnx_path), task="classify")
    rows: list[dict] = []
    mismatches = 0
    for image_path in images:
        official = model.predict(str(image_path), imgsz=160, verbose=False)[0]
        official_id = int(official.probs.top1)
        official_name = str(official.names[official_id])
        official_score = float(official.probs.top1conf)

        image = cv2.imdecode(__import__("numpy").fromfile(str(image_path), dtype="uint8"), cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            web = classifyinfer.run(str(onnx_path), image, [], conf=0.25)
            verdict = web["classify"]
            web_name = verdict["label"]
            web_score = verdict["score"]
            web_id = verdict["class_id"]
        except classifyinfer.ClassifyError as exc:
            web_name, web_score, web_id = f"错误：{exc}", 0.0, -1
        same = official_id == web_id and official_name == web_name
        if not same:
            mismatches += 1
        rows.append(
            {
                "file": image_path.name,
                "official": official_name,
                "official_score": round(official_score, 4),
                "web": web_name,
                "web_score": round(web_score, 4),
                "same": same,
            }
        )
        log(
            f"  {image_path.name[:26]:26s} 官方={official_name}({official_score:.3f})  "
            f"Web={web_name}({web_score:.3f})  {'一致' if same else '不一致'}"
        )
    log(f"对比 {len(rows)} 张，不一致 {mismatches} 张")
    (RUNTIME / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def run_batch(onnx_path: Path, data_yaml: Path, labels_json: Path | None) -> None:
    """走真实的 infer_batch.py（Web 批量推理用的就是它）。"""
    source = DATA / "cn2" / "val"
    if not source.is_dir():
        source = DATA / "multi" / "val"
    output = RUNTIME / "batch_out"
    if output.exists():
        shutil.rmtree(output, ignore_errors=True)
    command = [
        sys.executable,
        str(BACKEND / "tools" / "infer_batch.py"),
        "--kind", "yolo",
        "--engine", str(onnx_path),
        "--source", str(source),
        "--output", str(output),
        "--task", "classify",
        "--conf", "0.25",
        "--no-zip",
    ]
    if labels_json and labels_json.is_file():
        command.extend(["--labels", str(labels_json)])
    log("批量推理：" + " ".join(command))
    process = subprocess.run(command, cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (process.stdout or "")[-1500:]
    log(f"批量退出码 {process.returncode}")
    if process.returncode != 0:
        log(tail)
        return
    csv_files = list(output.rglob("results.csv"))
    if not csv_files:
        log("批量没有生成 results.csv")
        return
    text = csv_files[0].read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    log(f"results.csv 表头：{lines[0]}")
    for line in lines[1:4]:
        log(f"  {line}")
    if lines[0].startswith("index,file,class_name,score,uncertain"):
        log("CSV 列符合分类约定")
    else:
        log("CSV 列不符合分类约定")


def run_video(onnx_path: Path, data_yaml: Path, labels_json: Path | None) -> None:
    """合成一段视频，走真实的 infer_video.py，确认每帧只给一个类别。"""
    import cv2
    import numpy as np

    images = [p for p in traininfo_images(data_yaml)][:12]
    if not images:
        log("没有图片，跳过视频")
        return
    video_in = RUNTIME / "smoke.mp4"
    first = cv2.imdecode(np.fromfile(str(images[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        log("首帧读不出来，跳过视频")
        return
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(video_in), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    if not writer.isOpened():
        log("视频写入器打不开，跳过视频")
        return
    for image_path in images:
        frame = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    log(f"合成视频：{video_in}（{len(images)} 帧）")

    output = RUNTIME / "video_out.mp4"
    command = [
        sys.executable,
        str(BACKEND / "tools" / "infer_video.py"),
        "--engine", str(onnx_path),
        "--task", "classify",
        "--source", str(video_in),
        "--output", str(output),
        "--conf", "0.25",
        "--nms", "0.45",
        "--stride", "1",
    ]
    if labels_json and labels_json.is_file():
        command.extend(["--labels", str(labels_json)])
    process = subprocess.run(command, cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(f"视频退出码 {process.returncode}")
    if process.returncode != 0:
        log((process.stdout or "")[-1500:])
        log((process.stderr or "")[-800:])
        return
    for line in (process.stdout or "").splitlines():
        if '"class_counter"' in line or '"type":"summary"' in line.replace(" ", ""):
            log(f"  {line[:200]}")


def traininfo_images(data_yaml: Path) -> list[Path]:
    sys.path.insert(0, str(BACKEND))
    import traininfo  # noqa: E402

    return traininfo.list_split_images(data_yaml, "val")


def main() -> int:
    parser = argparse.ArgumentParser(description="分类真机闭环冒烟测试")
    parser.add_argument("--prepare", action="store_true", help="只下载官方权重和数据集并造两个测试集")
    parser.add_argument("--run", action="store_true", help="训练、导出、对比、批量")
    parser.add_argument("--skip-train", action="store_true", help="复用已有 best.pt")
    parser.add_argument("--multi", action="store_true", help="额外跑一遍 10 类数据集，验证多分类")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=160)
    args = parser.parse_args()
    if not args.prepare and not args.run:
        parser.print_help()
        return 1

    prepare()
    if not args.run:
        return 0

    cn2_yaml = DATA / "cn2" / "dataset.yaml"
    multi_yaml = DATA / "multi" / "dataset.yaml"
    if not cn2_yaml.is_file():
        raise SystemExit("cn2 数据集没准备好，先跑 --prepare")

    best = RUNTIME / "runs" / "cn2" / "weights" / "best.pt"
    if not args.skip_train:
        log("=== 1/4 训练 cn2（中文两类，故意与排序相反）===")
        best = train(cn2_yaml, "cn2", args.epochs, args.imgsz)
        log("=== 2/4 验证 cn2 ===")
        val(best, cn2_yaml)

    if not best.is_file():
        raise SystemExit(f"没有可用的权重：{best}")
    log("=== 3/4 导出 ONNX ===")
    onnx_path = export(best, args.imgsz)

    log("=== 4/4 逐张对比 + 批量 ===")
    rows = compare(onnx_path, cn2_yaml)
    labels = best.parent.parent / f"labels_{best.parent.parent.name}.json"
    run_batch(onnx_path, cn2_yaml, labels)
    run_video(onnx_path, cn2_yaml, labels)

    mismatches = sum(1 for row in rows if not row["same"])
    log("=" * 60)
    log(f"结论：对比 {len(rows)} 张，不一致 {mismatches} 张")

    if args.multi and multi_yaml.is_file():
        log("=== 附加：10 类数据集多分类 ===")
        multi_best = train(multi_yaml, "multi", 3, args.imgsz)
        multi_onnx = export(multi_best, args.imgsz)
        multi_rows = compare(multi_onnx, multi_yaml, limit=20)
        multi_bad = sum(1 for row in multi_rows if not row["same"])
        log(f"多分类：对比 {len(multi_rows)} 张，不一致 {multi_bad} 张")
        mismatches += multi_bad

    if mismatches:
        log("有差异，需要继续对齐预处理或类别顺序")
        return 1
    log("全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
