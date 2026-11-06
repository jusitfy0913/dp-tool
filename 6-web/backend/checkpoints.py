# -*- coding: utf-8 -*-
"""历史训练产物的发现与解析（断点续训用）。

为什么**不加载** ``.pt``：

- 后端的解释器（FastAPI 那个）没有装 torch，``torch.load`` 直接不可用；
- 硬塞 pickle 解析一个包含张量的 checkpoint 既不安全也不可靠；
- 训练目录里本来就有 Ultralytics 自己写下的 ``args.yaml`` 与 ``results.csv``，
  前者给参数，后者给「跑了几轮、最好多少」，续训需要的信息足够了。

目录约定（Ultralytics 8.x）：``<project>/<name>/args.yaml`` + ``results.csv`` +
``weights/last.pt`` + ``weights/best.pt``。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import dataset

#: 扫描深度上限：``<root>/<project>/<name>/args.yaml`` 已经是 3 层，
#: 留一层余量给 ``runs/detect/xxx`` 这种嵌套 project。
MAX_DEPTH = 4

#: 返回条数上限，避免一次把几百次实验全推给浏览器。
DEFAULT_LIMIT = 50

#: results.csv 里可能出现的列名（不同任务头不一样）。
MAP_KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95(P)", "metrics/mAP50-95(O)")


def _depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return 0


def find_run_dirs(roots: Iterable[Path], limit_paths: int = 500) -> list[Path]:
    """在若干根目录下找所有含 ``args.yaml`` 的训练目录。"""
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for args_file in root.rglob("args.yaml"):
            try:
                if _depth(args_file, root) > MAX_DEPTH:
                    continue
            except OSError:
                continue
            run_dir = args_file.parent
            key = str(run_dir.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(run_dir)
            if len(found) >= limit_paths:
                return found
    return found


def read_args_yaml(path: Path) -> dict[str, str]:
    """只取顶层标量。args.yaml 是 Ultralytics 用 yaml.dump 写的扁平字典。"""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or raw[0] in " \t":
            continue
        head, sep, tail = raw.partition(":")
        if not sep:
            continue
        key = head.strip()
        value = tail.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.lower()] = value
    return values


def read_results(csv_path: Path) -> dict[str, Any]:
    """读 results.csv，返回已完成轮数、最后一轮指标、最佳 mAP。"""
    summary: dict[str, Any] = {
        "epochs_done": 0,
        "last_epoch": 0,
        "best_map": None,
        "best_epoch": None,
        "last_metrics": {},
        "train_seconds": None,
    }
    if not csv_path.is_file():
        return summary
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if (row.get("epoch") or "").strip()]
    except OSError:
        return summary

    summary["epochs_done"] = len(rows)
    if not rows:
        return summary

    last = rows[-1]
    try:
        summary["last_epoch"] = int(float(last.get("epoch", 0)))
        summary["train_seconds"] = round(float(last.get("time", 0)), 1)
    except (TypeError, ValueError):
        summary["last_epoch"] = len(rows)

    best_value: float | None = None
    best_epoch: int | None = None
    for row in rows:
        for key in MAP_KEYS:
            raw = row.get(key)
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if best_value is None or value > best_value:
                best_value = value
                try:
                    best_epoch = int(float(row.get("epoch", 0)))
                except (TypeError, ValueError):
                    best_epoch = None
            break
    summary["best_map"] = round(best_value, 4) if best_value is not None else None
    summary["best_epoch"] = best_epoch
    summary["last_metrics"] = {
        key: round(float(value), 4)
        for key, value in last.items()
        if key and key not in {"epoch", "time"} and _is_number(value)
    }
    return summary


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def describe(run_dir: Path) -> dict:
    """把一个训练目录整理成前端可直接渲染的一行。"""
    args = read_args_yaml(run_dir / "args.yaml")
    results = read_results(run_dir / "results.csv")

    last_weight = run_dir / "weights" / "last.pt"
    best_weight = run_dir / "weights" / "best.pt"
    data_path = args.get("data", "")
    data_exists = bool(data_path) and Path(data_path).is_file()

    classes: list[dict] = []
    if data_exists:
        try:
            text = Path(data_path).read_text(encoding="utf-8", errors="replace")
            classes = dataset.parse_names_from_yaml(text)
        except OSError:
            classes = []

    try:
        modified = (run_dir / "args.yaml").stat().st_mtime
    except OSError:
        modified = 0.0

    return {
        "run_dir": str(run_dir),
        "name": args.get("name") or run_dir.name,
        "task": args.get("task", ""),
        "model": args.get("model", ""),
        "data": data_path,
        "data_exists": data_exists,
        "imgsz": _to_int(args.get("imgsz")),
        "batch": args.get("batch", ""),
        "optimizer": args.get("optimizer", ""),
        "epochs_planned": _to_int(args.get("epochs")),
        "epochs_done": results["epochs_done"],
        "last_epoch": results["last_epoch"],
        "best_map": results["best_map"],
        "best_epoch": results["best_epoch"],
        "train_seconds": results["train_seconds"],
        "classes": len(classes),
        # 类别名要回传给前端做「微调兼容性检查」：Ultralytics 是按类名重映射检测头的，
        # 名称对不上的旧类权重会被静默丢弃，不提前说清楚就会「越训越差还不知道为什么」。
        "class_names": [item["name"] for item in classes],
        "last_pt": str(last_weight) if last_weight.is_file() else "",
        "last_pt_mb": round(last_weight.stat().st_size / 1024 / 1024, 1) if last_weight.is_file() else 0.0,
        "best_pt": str(best_weight) if best_weight.is_file() else "",
        "best_pt_mb": round(best_weight.stat().st_size / 1024 / 1024, 1) if best_weight.is_file() else 0.0,
        # 续训必须有 last.pt，且数据集还在；两条都不满足时前端要禁用按钮
        "resumable": last_weight.is_file() and data_exists,
        "blocked_reason": _blocked_reason(last_weight, data_path, data_exists),
        # 微调只需要 best.pt（优化器重置，不需要优化器状态）；数据集不在只是警告级问题
        "finetunable": best_weight.is_file(),
        "finetune_blocked_reason": "" if best_weight.is_file() else "缺少 weights/best.pt（本次训练还没产出最佳权重）",
        "modified": round(modified, 1),
    }


def _blocked_reason(last_weight: Path, data_path: str, data_exists: bool) -> str:
    if not last_weight.is_file():
        return "缺少 weights/last.pt（训练可能刚开始就中断了，没有可续训的检查点）"
    if not data_path:
        return "args.yaml 里没有记录数据集路径"
    if not data_exists:
        return f"数据集已不存在：{data_path}"
    return ""


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def list_checkpoints(roots: Iterable[Path], limit: int = DEFAULT_LIMIT) -> dict:
    """扫描并排序（新的在前）。"""
    run_dirs = find_run_dirs(roots)
    items = [describe(run_dir) for run_dir in run_dirs]
    items.sort(key=lambda item: item["modified"], reverse=True)
    limited = items[: max(1, limit)]
    return {
        "total": len(items),
        "shown": len(limited),
        "roots": [str(root) for root in roots if root.is_dir()],
        "items": limited,
    }


# ---------------------------------------------------------------------------
# 微调检查：用自己的 best.pt 当起点之前，先说清楚会发生什么
# ---------------------------------------------------------------------------
# 微调（init_mode=finetune）与断点续训（resume）是两件事：
#   - 续训：从 last.pt 恢复优化器、轮次与全部超参，表单参数不生效；
#   - 微调：从 best.pt 载入权重，优化器与轮次全部重置，表单参数生效。
# 这里只做「能不能接上」的静态检查，不加载 .pt（后端解释器没有 torch）。


def describe_weights(weights: Path) -> dict | None:
    """从 .pt 路径反推训练目录并读出元信息；找不到训练目录返回 None。

    Ultralytics 的目录约定是 ``<run>/weights/best.pt``，只往上找两层即可，
    不需要 torch.load（那既不安全，也需要后端装 torch）。
    """
    try:
        weights = weights.expanduser().resolve()
    except OSError:
        return None
    run_dir = weights.parent.parent if weights.parent.name.lower() == "weights" else weights.parent
    if (run_dir / "args.yaml").is_file():
        return describe(run_dir)
    return None


def _read_class_names(data_yaml: Path) -> list[str]:
    try:
        text = data_yaml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [item["name"] for item in dataset.parse_names_from_yaml(text)]


def finetune_check(
    weights: Path,
    data_yaml: Path,
    task: str = "detect",
    imgsz: int | None = None,
) -> dict:
    """检查「用这个 best.pt 继续训练」是否可行。

    blockers：硬性问题，必须拦住（文件不存在、任务头不一致）；
    warnings：会被静默改变行为的情况，要让用户先看到再决定。
    """
    blockers: list[str] = []
    warnings: list[str] = []

    if not weights.is_file():
        blockers.append(f"权重文件不存在：{weights}")

    source = describe_weights(weights) or {}
    source_info = {
        "run_dir": source.get("run_dir", ""),
        "name": source.get("name", ""),
        "task": source.get("task", ""),
        "imgsz": source.get("imgsz"),
        "epochs_done": source.get("epochs_done", 0),
        "best_map": source.get("best_map"),
        "class_names": source.get("class_names", []),
        "data": source.get("data", ""),
    }

    source_task = str(source_info["task"] or "")
    if source_task and task and source_task != task:
        blockers.append(
            f"任务头不一致：权重是 {source_task}，当前表单是 {task}。"
            "不同任务头的检测头结构不同，不能直接互相接续。"
        )

    if not source_info["run_dir"]:
        warnings.append(
            "这个权重旁边没有 args.yaml，无法读出它训练时的类别。"
            "如果当前的类别和它不一致，检测头会被重映射，对不上的旧类权重将被丢弃。"
        )
    else:
        old_names = [str(name) for name in source_info["class_names"] or []]
        new_names = _read_class_names(data_yaml)
        if old_names and new_names:
            if old_names == new_names:
                pass
            elif sorted(old_names) == sorted(new_names):
                warnings.append(
                    "类别集合相同但顺序不同。Ultralytics 按类名重映射，结果通常正确，"
                    "但建议核对一下序号是否与你的标签一致。"
                )
            else:
                missing = [name for name in old_names if name not in new_names]
                added = [name for name in new_names if name not in old_names]
                parts = []
                if missing:
                    parts.append(f"新数据集缺少旧类别 {missing}（这部分权重会丢失）")
                if added:
                    parts.append(f"新增类别 {added}（这部分检测头将随机初始化）")
                warnings.append("类别不一致：" + "；".join(parts))

        source_imgsz = source_info["imgsz"]
        if imgsz and source_imgsz and int(imgsz) != int(source_imgsz):
            warnings.append(
                f"输入尺寸与原实验不同（原 {source_imgsz}px，现在 {imgsz}px）。"
                "微调可以改尺寸，但小缺陷场景建议先用原尺寸跑通再调。"
            )

    if source_info["run_dir"]:
        warnings.append(
            f"结果将写回原实验目录：{source_info['run_dir']}。"
            "开训前会自动把旧的 args.yaml / results.csv / weights 备份到 _prev/<时间戳>/。"
        )

    return {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "source": source_info,
    }
