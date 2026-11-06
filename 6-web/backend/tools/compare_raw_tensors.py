# -*- coding: utf-8 -*-
"""张量级比对：把同一条模型的引擎与 ONNX 侧的输入/输出原始张量对齐后算差值。

为什么还要张量级：
    结果级（compare_backends.py）回答「能不能上线」，一旦不通过却回答不了「差在哪」。
    有了两侧的原始张量，就能把差异分成三段：预处理输入 → 模型数值 → 后处理。
    本工具依赖动态库的诊断导出开关 `TRTFORGE_RAW_DUMP`，未设置时库本身不产生任何文件。

用法（在 6-web/backend 目录下）：

    python tools/compare_raw_tensors.py --kind yolo \
        --engine runs/.../best_fp16.engine --onnx runs/.../best.transd.onnx \
        --image "PCB data/xxx.jpg"

    python tools/compare_raw_tensors.py --kind anomalib \
        --engine .../model_fp16.engine --onnx .../model.onnx \
        --metadata .../metadata.json --image bottle.png

产物（默认 6-web/runtime/raw_compare/<时间戳>/）：
    raw_compare.json    每个张量的形状、最大绝对差、平均绝对差、余弦相似度
    onnx/ trt/          两侧的 .f32 原始数据与 manifest.jsonl，便于用其它工具复查

退出码：
    0  全部张量在阈值内
    1  有张量超出阈值
    2  无法比对（文件缺失 / DLL 不可用 / 某一侧没导出张量）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="引擎与 ONNX 原始张量比对")
    parser.add_argument("--kind", choices=("yolo", "anomalib"), required=True)
    parser.add_argument("--engine", required=True, help=".engine / .trt 路径")
    parser.add_argument("--onnx", required=True, help=".onnx 路径")
    parser.add_argument("--image", required=True, help="单张图片（张量比对不做批量）")
    parser.add_argument("--task", default="detect", choices=("detect", "segment", "pose", "obb"))
    parser.add_argument("--metadata", default="", help="anomalib 的 metadata.json")
    parser.add_argument("--model-type", default="Auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--out", default="", help="输出目录，留空用 runtime/raw_compare/<时间戳>")
    parser.add_argument("--max-abs", type=float, default=0.0,
                        help="单张量最大绝对差上限，0 表示不按绝对值判定（YOLO 坐标通道是像素尺度）")
    parser.add_argument("--max-rel", type=float, default=0.10,
                        help="按张量量级归一化后的最大相对差上限")
    parser.add_argument("--min-cosine", type=float, default=0.99, help="余弦相似度下限")
    parser.add_argument("--min-binary-agree", type=float, default=0.95,
                        help="0/1 张量（如 pred_mask）的像素一致率下限")
    parser.add_argument(
        "--skip", default="pred_label",
        help="逗号分隔的张量名，不参与比对。默认跳过 pred_label：它是模型图内的"
             "二值后处理，单元素翻转就是 100% 差异，而业务判定以归一化分数为准",
    )

    # 子进程入口：主进程用这两个参数分别触发两侧导出
    parser.add_argument("--worker", choices=("onnx", "trt"), help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--dump", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 子进程：只跑一次推理，把张量交给动态库落盘
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    import infer  # 延迟到子进程里导入，主进程不需要加载 DLL

    # 子进程是全新进程，必须先加载三个 DLL；status() 内部会做一次性加载
    status = infer.status()
    if args.kind == "yolo" and not (status.onnx_ok if infer.is_onnx_model(args.model) else status.yolo_ok):
        print(f"DLL 不可用：{status.detail} {status.onnx_detail}", file=sys.stderr)
        return 2

    image = infer.decode_image(Path(args.image).read_bytes())
    if args.kind == "yolo":
        result = infer.run_yolo(
            args.model, args.task, image, conf=args.conf, nms=args.nms
        )
    else:
        result = infer.run_anomalib(
            args.model, args.metadata, image, model_type=args.model_type
        )
    print(json.dumps({"backend": result.get("backend", "")}, ensure_ascii=False))
    return 0


def run_side(args: argparse.Namespace, side: str, dump_dir: Path) -> tuple[int, str]:
    command = [
        sys.executable, str(Path(__file__)), "--worker", side,
        "--kind", args.kind,
        "--engine", args.engine,
        "--onnx", args.onnx,
        "--image", args.image,
        "--task", args.task,
        "--conf", str(args.conf),
        "--nms", str(args.nms),
        "--metadata", args.metadata,
        "--model-type", args.model_type,
        "--model", args.engine if side == "trt" else args.onnx,
        "--dump", str(dump_dir),
    ]
    env = dict(os.environ)
    env["TRTFORGE_RAW_DUMP"] = str(dump_dir)
    # DLL 的日志与 Python 的打印可能混用编码，固定 utf-8 + 容错，避免读输出时崩掉
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return completed.returncode, (completed.stderr or completed.stdout or "").strip()


# ---------------------------------------------------------------------------
# 清单与对齐
# ---------------------------------------------------------------------------

def read_manifest(dump_dir: Path) -> list[dict]:
    path = dump_dir / "manifest.jsonl"
    if not path.is_file():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def load_tensor(dump_dir: Path, entry: dict) -> np.ndarray:
    name = f"{entry['backend']}_run{entry['run']:04d}_{entry['phase']}{entry['index']}.f32"
    raw = np.fromfile(dump_dir / name, dtype=np.float32)
    return raw


def transpose_last_two(array: np.ndarray, shape: list[int]) -> np.ndarray:
    axes = list(range(len(shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return array.reshape(shape).transpose(axes).reshape(-1)


def align(a: np.ndarray, shape_a: list[int], b: np.ndarray, shape_b: list[int]):
    """返回 (对齐后的 a, 对齐后的 b, 说明)；无法对齐时返回 (None, None, 原因)。"""
    if a.size != b.size:
        return None, None, f"元素数不同：{a.size} vs {b.size}"
    if list(shape_a) == list(shape_b):
        return a, b, "形状一致"
    if len(shape_a) == len(shape_b) >= 2:
        swapped = list(shape_a)
        swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
        if swapped == list(shape_b):
            return transpose_last_two(a, list(shape_a)), b, "已对齐最后两维转置"
    return None, None, f"形状不同且无法对齐：{shape_a} vs {shape_b}"


def compare_entries(onnx_entries: list[dict], trt_entries: list[dict]) -> list[dict]:
    """按 phase + name 配对。

    不能用序号：TensorRT 的 binding 顺序与 ONNX 图输出顺序不保证一致
    （实测 PatchCore 的 TRT 是 score/map/label/mask，ONNX 是 score/label/map/mask），
    按序号配会把 map 和 label 拿去相减，得出毫无意义的巨大差异。
    只有某一侧完全没有名字时才退回序号。
    """
    def keyed(entries: list[dict]) -> dict:
        table: dict = {}
        nameless: dict = {}
        for entry in entries:
            name = str(entry.get("name") or "")
            if name:
                table[(entry["phase"], name)] = entry
            else:
                nameless[(entry["phase"], str(entry["index"]))] = entry
        return table if table else nameless

    onnx_table = keyed(onnx_entries)
    trt_table = keyed(trt_entries)

    pairs: list[dict] = []
    for key, onnx_entry in onnx_table.items():
        trt_entry = trt_table.get(key)
        if trt_entry is None:
            continue
        pairs.append({"key": key, "onnx": onnx_entry, "trt": trt_entry})
    return pairs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.worker:
        return run_worker(args)

    engine = Path(args.engine).expanduser()
    onnx = Path(args.onnx).expanduser()
    image = Path(args.image).expanduser()

    missing: list[str] = []
    if not engine.is_file():
        missing.append(f"引擎不存在：{engine}")
    if not onnx.is_file():
        missing.append(f"ONNX 不存在：{onnx}")
    if not image.is_file():
        missing.append(f"图片不存在：{image}")
    if args.kind == "anomalib" and not Path(args.metadata or "").is_file():
        missing.append(f"anomalib 需要 metadata.json：{args.metadata or '(未提供)'}")
    if missing:
        for text in missing:
            print(f"[无法比对] {text}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser() if args.out.strip() else (
        BACKEND_DIR.parent / "runtime" / "raw_compare" / time.strftime("%Y%m%d_%H%M%S")
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    onnx_dir, trt_dir = out_dir / "onnx", out_dir / "trt"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)

    print("导出 ONNX 侧张量 …", flush=True)
    code, output = run_side(args, "onnx", onnx_dir)
    if code != 0:
        print(f"[无法比对] ONNX 侧推理失败：{output}", file=sys.stderr)
        return 2

    print("导出 TRT 侧张量 …", flush=True)
    code, output = run_side(args, "trt", trt_dir)
    if code != 0:
        print(f"[无法比对] TRT 侧推理失败：{output}", file=sys.stderr)
        return 2

    onnx_entries = read_manifest(onnx_dir)
    trt_entries = read_manifest(trt_dir)
    if not onnx_entries or not trt_entries:
        print(
            "[无法比对] 某一侧没有导出张量。"
            f" onnx={len(onnx_entries)} trt={len(trt_entries)}",
            file=sys.stderr,
        )
        return 2

    pairs = compare_entries(onnx_entries, trt_entries)
    if not pairs:
        print("[无法比对] 两侧张量无法按名称或序号配对", file=sys.stderr)
        return 2

    rows: list[dict] = []
    failures: list[str] = []
    skip_names = {item.strip() for item in args.skip.split(",") if item.strip()}
    for item in pairs:
        onnx_entry, trt_entry = item["onnx"], item["trt"]
        name = onnx_entry.get("name") or trt_entry.get("name") or ""
        if name in skip_names:
            continue
        a = load_tensor(onnx_dir, onnx_entry)
        b = load_tensor(trt_dir, trt_entry)
        aligned_a, aligned_b, note = align(
            a, list(onnx_entry.get("shape") or []), b, list(trt_entry.get("shape") or [])
        )
        row = {
            "phase": onnx_entry["phase"],
            "name": name,
            "shape_onnx": onnx_entry.get("shape"),
            "shape_trt": trt_entry.get("shape"),
            "alignment": note,
        }
        if aligned_a is None:
            row["error"] = note
            failures.append(f"{row['phase']}/{row['name']}：{note}")
            rows.append(row)
            continue

        diff = np.abs(aligned_a - aligned_b)
        denom = float(np.linalg.norm(aligned_a) * np.linalg.norm(aligned_b))
        cosine = float(np.dot(aligned_a, aligned_b) / denom) if denom > 0 else 1.0
        max_abs = float(diff.max()) if diff.size else 0.0
        # 坐标通道是像素尺度（0~输入尺寸），绝对差天然比归一化通道大一个量级；
        # 用张量自身量级缩放后再判定，才不会把 fp16 的正常舍入当成错误。
        scale = max(1.0, float(np.abs(aligned_a).max()), float(np.abs(aligned_b).max()))
        max_rel = max_abs / scale
        row.update({
            "max_abs_diff": round(max_abs, 6),
            "max_rel_diff": round(max_rel, 6),
            "mean_abs_diff": round(float(diff.mean()) if diff.size else 0.0, 6),
            "cosine": round(cosine, 8),
        })

        # 0/1 张量（pred_mask 等）不能用相对差判定：一个像素翻转就是 rel=1。
        # 这类张量看「一致率」，与业务上「同一块区域被判为缺陷」直接对应。
        unique_a = np.unique(aligned_a) if aligned_a.size else np.array([])
        unique_b = np.unique(aligned_b) if aligned_b.size else np.array([])
        binary = (
            unique_a.size > 0 and unique_b.size > 0
            and np.all(np.isin(unique_a, (0.0, 1.0)))
            and np.all(np.isin(unique_b, (0.0, 1.0)))
        )
        if binary:
            agree = float((aligned_a == aligned_b).mean())
            row["binary_agree"] = round(agree, 6)
            if agree < args.min_binary_agree:
                failures.append(
                    f"{row['phase']}/{row['name']}：binary_agree {agree:.4f} "
                    f"< {args.min_binary_agree}"
                )
        else:
            if args.max_abs > 0 and max_abs > args.max_abs:
                failures.append(
                    f"{row['phase']}/{row['name']}：max_abs {row['max_abs_diff']} > {args.max_abs}"
                )
            if max_rel > args.max_rel:
                failures.append(
                    f"{row['phase']}/{row['name']}：max_rel {row['max_rel_diff']} > {args.max_rel}"
                )
            if cosine < args.min_cosine:
                failures.append(
                    f"{row['phase']}/{row['name']}：cosine {row['cosine']} < {args.min_cosine}"
                )
        rows.append(row)

    report = {
        "meta": {
            "kind": args.kind,
            "task": args.task if args.kind == "yolo" else None,
            "engine": str(engine),
            "onnx": str(onnx),
            "image": str(image),
            "max_abs": args.max_abs,
            "max_rel": args.max_rel,
            "min_cosine": args.min_cosine,
            "min_binary_agree": args.min_binary_agree,
            "skip": sorted(skip_names),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tensors": rows,
        "failures": failures,
    }
    report_path = out_dir / "raw_compare.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"  {row['phase']:<6} {row['name']:<16} "
            f"max={row.get('max_abs_diff')} rel={row.get('max_rel_diff')} "
            f"mean={row.get('mean_abs_diff')} cos={row.get('cosine')} "
            f"{row.get('alignment', '')}"
        )
    print(f"报告：{report_path}")

    if failures:
        print(f"未通过：{len(failures)} 项", file=sys.stderr)
        for text in failures[:10]:
            print(f"  {text}", file=sys.stderr)
        return 1
    print("张量级比对通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
