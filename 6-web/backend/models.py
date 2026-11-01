# -*- coding: utf-8 -*-
"""训练产物扫描：把 `runs/` 与 `results/` 里的引擎列出来给前端选。

为什么不做索引落盘：
    产物数量是几十级别，遍历一遍在毫秒量级；建索引反而要处理"什么时候失效"，
    而模型文件恰恰是频繁改名、覆盖、删除的那类东西。直接实时遍历最不容易出错。

为什么必须容忍残缺目录：
    实测目录里大量存在"有目录没引擎"或"有引擎没 metadata"的中间状态
    （训练中断、只导出了 ckpt、只跑了 val）。扫描器遇到这些必须跳过而不是报错，
    否则一个训练到一半的实验就能让整个页面打不开。

匹配规则（来自实测的真实现象，不是猜的）：
    引擎名普遍带精度后缀：`best_fp16.engine` / `best.transd_fp16.engine` / `model_fp16.engine`，
    而 ONNX 叫 `best.onnx` / `best.transd.onnx` / `model.onnx`——
    **没有任何一个引擎与 ONNX 严格同名**。所以配对要先去精度后缀再比。

    YOLO 的 `labels_*.json` 在 run 根目录（引擎的**上一级**），同目录一定找不到；
    Anomalib 没有标签文件，只有同目录的 `metadata.json`。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import config

# 扫描根与它们对应的产物类型。顺序决定返回顺序的次要排序。
SCAN_TARGETS = [
    ("yolo", lambda: config.YOLO_LAUNCHER_DIR / "runs"),
    ("anomalib", lambda: config.ANOMALIB_LAUNCHER_DIR / "results"),
]

ENGINE_SUFFIXES = {".engine", ".trt"}

# 限制遍历深度与目录数，防止有人把 runs 指到一个巨大的目录上把页面拖死
MAX_DEPTH = 6
MAX_ENTRIES = 80

# 这些目录里的东西与"可推理引擎"无关，进去只会浪费时间
SKIP_DIR_NAMES = {
    "lightning_logs", "images", "onnx", "__pycache__", ".git",
    "train_batch", "val_batch", "wandb", "weights_backup",
}

# 精度后缀：`best_fp16` -> `best`，`model_fp32` -> `model`
_PRECISION_SUFFIX = re.compile(r"_(fp16|fp32|int8|bf16)$", re.IGNORECASE)

# 任务类型关键字，按"越长越优先"匹配，避免 obb 被 detect 抢走这类误判
_TASK_KEYWORDS = ("segment", "detect", "pose", "obb")


def _strip_precision(stem: str) -> str:
    """去掉精度后缀，用于把引擎名和 ONNX 名对齐。"""
    return _PRECISION_SUFFIX.sub("", stem)


def _is_transposed_onnx(path: Path) -> bool:
    """判断 ONNX 是否已经转置。

    launcher 导出的可推理 ONNX 名为 `*.transd.onnx`；
    `best.onnx` 是 Ultralytics 原始输出布局，用它编译的引擎会让 DLL 解码越界
    （README 里有实测记录）。所以这个标志必须带给前端，让用户提前知道选错了。
    """
    return ".transd." in path.name.lower() or path.stem.lower().endswith(".transd")


def _find_onnx(engine: Path) -> tuple[Path | None, bool]:
    """为引擎找配套 ONNX，返回 (路径, 是否已转置)。

    优先级：
    1. 去精度后缀后严格同名（`best.transd_fp16` -> `best.transd.onnx`）；
    2. 再退一步去掉 `.transd`（`best.transd` -> `best.onnx`）；
    3. 同目录里唯一的 .onnx（只有一个候选时不会有歧义）；
    4. 同目录里优先取已转置的那个。
    """
    try:
        siblings = list(engine.parent.iterdir())
    except OSError:
        return None, False

    onnx_files = [p for p in siblings if p.suffix.lower() == ".onnx" and p.is_file()]
    if not onnx_files:
        return None, False

    base = _strip_precision(engine.stem)
    for candidate in onnx_files:
        if candidate.stem.lower() == base.lower():
            return candidate, _is_transposed_onnx(candidate)

    # `best.transd` -> 再去掉 .transd，试 `best.onnx`
    transd_less = re.sub(r"\.transd$", "", base, flags=re.IGNORECASE)
    for candidate in onnx_files:
        if candidate.stem.lower() == transd_less.lower():
            return candidate, _is_transposed_onnx(candidate)

    if len(onnx_files) == 1:
        return onnx_files[0], _is_transposed_onnx(onnx_files[0])

    transposed = [p for p in onnx_files if _is_transposed_onnx(p)]
    if transposed:
        return transposed[0], True
    return onnx_files[0], _is_transposed_onnx(onnx_files[0])


def _run_dir(engine: Path) -> Path:
    """引擎所在实验目录。

    YOLO 的引擎放在 `<run>/weights/best.engine`，其余产物在 `<run>/` 根；
    统一在这里判定一次，避免每个调用点各写一遍而大小写判断不一致。
    """
    return engine.parent.parent if engine.parent.name.lower() == "weights" else engine.parent


def _find_labels(engine: Path, kind: str) -> Path | None:
    """找标签文件。

    YOLO 的标签在 run 根（引擎上一级），文件名形如 `labels_<实验名>.json`；
    同目录先找一遍是为了兼容将来把标签放进 weights 的布局。
    Anomalib 没有标签文件，直接返回 None。
    """
    if kind != "yolo":
        return None

    for directory in (engine.parent, engine.parent.parent):
        try:
            found = sorted(p for p in directory.glob("labels_*.json") if p.is_file())
        except OSError:
            found = []
        if not found:
            continue
        # 实验名 = 引擎所在 run 目录名；优先精确匹配，避免一个 run 里多份标签时选错
        run_name = _run_dir(engine).name
        for candidate in found:
            if candidate.stem == f"labels_{run_name}":
                return candidate
        return found[0]
    return None


def _find_metadata(engine: Path) -> tuple[Path | None, Path | None]:
    """找 (metadata.json, metadata_calibrated.json)。"""
    plain = engine.parent / "metadata.json"
    calibrated = engine.parent / "metadata_calibrated.json"
    return (
        plain if plain.is_file() else None,
        calibrated if calibrated.is_file() else None,
    )


def _read_json_quietly(path: Path) -> dict:
    """读一个 json，失败返回空字典（扫描器不能因为一个坏文件就崩）。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _infer_task(engine: Path, kind: str) -> str | None:
    """推断任务类型，供前端预填。

    优先级：
    1. YOLO 的 `args.yaml` 里 Next 的 `task:` 字段（训练时写下的真实任务）；
    2. 路径里的关键字（`detect_v1` / `segment_v2`）；
    3. Anomalib 一律是异常检测，用 metadata 里的 model_type 表示模型，任务固定 detect 语义。
    """
    if kind == "anomalib":
        return "anomalib"

    # 引擎在 weights/ 里，args.yaml 在 run 根
    args_file = _run_dir(engine) / "args.yaml"
    if args_file.is_file():
        try:
            text = args_file.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            text = ""
        found = re.search(r"^task:\s*([A-Za-z]+)", text, re.MULTILINE)
        if found and found.group(1).lower() in _TASK_KEYWORDS:
            return found.group(1).lower()

    haystack = str(engine).lower()
    for keyword in _TASK_KEYWORDS:
        if keyword in haystack:
            return keyword
    return "detect"


def _iter_engines(root: Path, max_depth: int = MAX_DEPTH):
    """在 root 下递归找引擎文件，带深度与目录跳过限制。"""
    if not root.is_dir():
        return

    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for item in children:
            name_lower = item.name.lower()
            if item.is_dir():
                if name_lower in SKIP_DIR_NAMES or name_lower.startswith("."):
                    continue
                stack.append((item, depth + 1))
            elif item.is_file() and item.suffix.lower() in ENGINE_SUFFIXES:
                yield item


def _relative_label(path: Path, root: Path) -> str:
    """相对根的短标签，用于下拉列表显示（绝对路径太长，一屏放不下几个）。"""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def scan_models(limit: int = MAX_ENTRIES) -> dict:
    """扫描全部产物根，返回引擎条目列表。

    返回结构：
        {
          "entries": [...],      # 按修改时间倒序
          "roots": [{kind, path, ok, reason}],
          "truncated": bool,
        }
    """
    entries: list[dict] = []
    roots_info: list[dict] = []

    for kind, resolver in SCAN_TARGETS:
        try:
            root = resolver()
        except Exception as exc:  # noqa: BLE001 - 路径推导失败不能拖垮接口
            roots_info.append({"kind": kind, "path": "", "ok": False,
                               "reason": f"路径推导失败：{exc}"})
            continue

        if not root.is_dir():
            roots_info.append({"kind": kind, "path": str(root), "ok": False,
                               "reason": "目录不存在"})
            continue

        # 白名单外不扫描，但要说明原因，不能静默返回空列表让人以为"没有模型"
        if not config.is_path_allowed(root):
            roots_info.append({
                "kind": kind, "path": str(root), "ok": False,
                "reason": "目录不在允许范围内，已跳过。可设置 TRTFORGE_BROWSE_ROOTS 追加。",
            })
            continue

        roots_info.append({"kind": kind, "path": str(root), "ok": True, "reason": ""})

        for engine in _iter_engines(root):
            try:
                stat = engine.stat()
            except OSError:
                continue

            onnx_path, transposed = _find_onnx(engine)
            labels_path = _find_labels(engine, kind)
            metadata_path, calibrated_path = _find_metadata(engine)

            task = _infer_task(engine, kind)

            model_type = None
            input_size = None
            if kind == "anomalib" and metadata_path is not None:
                meta = _read_json_quietly(metadata_path)
                model_type = meta.get("model_type")
                size = meta.get("input_size")
                if isinstance(size, list) and len(size) == 2:
                    input_size = [int(size[0]), int(size[1])]

            entries.append({
                "path": str(engine),
                "label": _relative_label(engine, root),
                "name": engine.name,
                "kind": kind,
                "task": task,
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "root": str(root),
                "run_dir": str(_run_dir(engine)),
                "onnx": str(onnx_path) if onnx_path else None,
                "transposed": bool(transposed),
                "labels": str(labels_path) if labels_path else None,
                "metadata": str(metadata_path) if metadata_path else None,
                "metadata_calibrated": str(calibrated_path) if calibrated_path else None,
                "model_type": model_type,
                "input_size": input_size,
            })

    entries.sort(key=lambda item: item["mtime"], reverse=True)
    truncated = len(entries) > limit
    return {
        "entries": entries[:limit],
        "total": len(entries),
        "truncated": truncated,
        "roots": roots_info,
    }
