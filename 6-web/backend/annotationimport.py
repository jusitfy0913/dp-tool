# -*- coding: utf-8 -*-
"""YOLO 数据集 -> ``.tfanno`` 工程导入层。

## 为什么单独一层

标注页原来的「批量加入」只扫描图片、把 ``annotations`` 写成空列表，因此
`data.yaml + images/ + labels/` 这种已经标好的数据集进不了标注页——只能
"看到图、框全丢"。这一层负责把 txt 标签读成真正的标注对象。

三件事必须复用，不能另写一份：

1. ``traininfo.resolve_splits``：解析 ``data.yaml`` 的 ``path:`` 与 train/val/test，
   它已经处理了「path 是相对 yaml 所在目录」这类真实数据集里的坑；
2. ``dataset.parse_yolo_label``：归一化坐标 -> 像素坐标，detect/segment/obb/pose
   四种布局的换算与「数据集预览」完全一致，否则同一个框在两个页面里会不一样；
3. ``annotation.normalize_object``：点序折叠、点数对齐、默认值不落盘，
   保证导入结果和手工画的框是同一种数据。

## 坐标清洗

YOLO 标签里的归一化值偶尔会略微越界（``0.9999`` / ``1.0001``，标注工具导出时
常见）。这类值夹回 ``[0, 1]`` 再换算，否则导出校验会报「点落在图像范围外」；
但明显越界（超出 5%）的整条丢弃并计数——那种基本是解码错了，写进工程等于
把脏数据固化成"标注"。

## 路径

默认把图片路径存成**相对工程文件所在目录**（Qt 的写法），这样整个工程可以随
数据集一起搬走；跨盘无法相对时回退绝对路径，并在报告里说明。
读回来时由 ``annotation.with_resolved_paths`` 补绝对路径给浏览器用。
"""

from __future__ import annotations

import os
from pathlib import Path

import annotation
import dataset
import traininfo

#: 支持的导入方式
MODES = ("new", "merge")
#: 同名图片的三种处理策略
DUPLICATES = ("skip", "replace", "append")
#: 允许的任务类型（与 .tfanno 的 task 字段一致）
IMPORT_TASKS = ("detect", "segment", "pose", "obb", "classify")

#: 归一化坐标允许的越界容差。超出这个范围的整条标签会被丢弃。
COORD_TOLERANCE = 0.05
#: 单次导入的图片上限，避免误选一个巨型目录把浏览器卡死
MAX_IMAGES = 5000
#: 报告里每一类问题最多列几个样例文件名
SAMPLE_LIMIT = 3


class AnnotationImportError(annotation.AnnotationError):
    """导入阶段可读的错误：数据源结构不对、任务类型冲突、没有可用图片。"""


# ---------------------------------------------------------------------------
# 数据源解析
# ---------------------------------------------------------------------------
def resolve_dataset(source: str | Path) -> dict:
    """把导入来源解析成 ``{yaml, yaml_dir, base, classes, nc, splits}``。

    支持三种输入：

    - ``data.yaml`` 文件：完整读取 ``path`` / ``names`` / 三个 split；
    - 数据集根目录（含 data.yaml）：同上；
    - 只给了图片目录或 ``images/`` 的父目录：按 Ultralytics 目录约定推断，
      类别名从标签里的类别号补 ``class_N``。

    不能直接用 ``dataset.load_yolo``：它忽略 ``data.yaml`` 里的 ``path:``，
    而真实数据集（包括本仓库的 PCB）常用 ``path:`` 指向另一层目录。
    """
    path = Path(source).expanduser()
    if not path.exists():
        raise AnnotationImportError(f"数据源不存在：{path}")

    if path.is_file():
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise AnnotationImportError(f"只接受 data.yaml 或数据集目录，收到：{path.name}")
        return _resolve_yaml(path)

    if not path.is_dir():
        raise AnnotationImportError(f"不是目录：{path}")

    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        candidate = path / name
        if candidate.is_file():
            return _resolve_yaml(candidate)

    # 没有 yaml：按目录约定推断。images/train、images/val 这种结构优先。
    splits: dict[str, list[Path]] = {name: [] for name in traininfo.SPLITS}
    images_root = path / "images"
    if images_root.is_dir():
        for name in traininfo.SPLITS:
            candidate = images_root / name
            if candidate.is_dir():
                splits[name].append(candidate.resolve())
        if not any(splits.values()):
            splits["train"] = [images_root.resolve()]
    else:
        splits["train"] = [path.resolve()]

    return {
        "source": str(path),
        "yaml": None,
        "yaml_dir": path,
        "base": path,
        "classes": [],
        "nc": None,
        "splits": splits,
    }


def _resolve_yaml(yaml_path: Path) -> dict:
    try:
        resolved = traininfo.resolve_splits(yaml_path)
    except traininfo.OverviewError as exc:
        raise AnnotationImportError(str(exc)) from exc
    resolved["source"] = str(yaml_path)
    return resolved


# ---------------------------------------------------------------------------
# 标签读取与坐标清洗
# ---------------------------------------------------------------------------
def _coord_indices(count: int, task: str) -> set[int]:
    """返回一行标签里哪些位置是「归一化坐标」。

    pose 的每一组关键点是 ``x y visibility``，visibility 是 0/1/2 的枚举值，
    把它当坐标夹到 ``[0,1]`` 会把 2 改成 1，关键点可见性就错了。
    """
    if task == "pose":
        return {index for index in range(count) if index < 4 or (index - 4) % 3 != 2}
    return set(range(count))


def sanitize_label(text: str, task: str) -> tuple[str, dict]:
    """清洗一段 YOLO 标签文本，返回 ``(可解析文本, 统计)``。

    做三件事：畸形行丢弃、明显越界行丢弃、轻微越界坐标夹回 ``[0, 1]``。
    输出仍是 YOLO 文本，交给 :func:`dataset.parse_yolo_label` 做统一换算——
    这样几何公式只有一份。
    """
    stats = {"malformed": 0, "out_of_range": 0, "clamped": 0, "lines": 0, "tokens": []}
    kept: list[str] = []

    for raw in text.splitlines():
        tokens = raw.split()
        if not tokens:
            continue
        if len(tokens) < 5:
            stats["malformed"] += 1
            continue
        try:
            int(float(tokens[0]))
            values = [float(token) for token in tokens[1:]]
        except ValueError:
            stats["malformed"] += 1
            continue

        coords = _coord_indices(len(values), task)
        out_of_range = False
        clamped = False
        for index in coords:
            value = values[index]
            if value < -COORD_TOLERANCE or value > 1 + COORD_TOLERANCE:
                out_of_range = True
                break
            if value < 0.0:
                values[index] = 0.0
                clamped = True
            elif value > 1.0:
                values[index] = 1.0
                clamped = True
        if out_of_range:
            stats["out_of_range"] += 1
            continue
        if clamped:
            stats["clamped"] += 1

        rendered = [tokens[0]]
        for index, value in enumerate(values):
            # 非坐标字段（pose 的 visibility）保留原始写法，避免浮点化后失去整数语义
            rendered.append(f"{value:.6f}" if index in coords else tokens[index + 1])
        kept.append(" ".join(rendered))
        stats["lines"] += 1
        stats["tokens"].append(len(tokens))

    return "\n".join(kept), stats


def _label_path(image: Path, images_dir: Path, yaml_dir: Path, split: str) -> Path | None:
    """按 Ultralytics 的约定找这张图对应的标签文件。

    优先用「把路径里最后一个 images 段换成 labels」的方式，这样嵌套目录
    （``images/train/sub/a.jpg``）也能对上 ``labels/train/sub/a.txt``；
    找不到再退回按 split 的标签根目录拼接。返回 ``None`` 表示这张图没有标签。
    """
    parts = list(image.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        candidate = Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
        if candidate.is_file():
            return candidate

    try:
        relative = image.relative_to(images_dir).with_suffix(".txt")
    except ValueError:
        return None
    candidate = traininfo.labels_dir_for(images_dir, yaml_dir, split) / relative
    return candidate if candidate.is_file() else None


def _abs_key(path: Path) -> str:
    """比较两张图是否同一张时用的规范化键（不区分大小写，解析 ``..``）。"""
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(os.path.abspath(str(path)))


# ---------------------------------------------------------------------------
# 图片路径写法
# ---------------------------------------------------------------------------
def store_path(image: Path, project_path: str) -> tuple[str, str]:
    """返回 ``(写进 .tfanno 的路径, 实际使用的模式)``。

    有工程路径时优先存相对路径（工程可整体搬移）；跨盘无法相对时回退绝对路径。
    ``project_path`` 为空（只预览、还没决定存哪）时存绝对路径，并在报告里说明。
    """
    if not project_path:
        return str(image), "absolute"
    base = Path(project_path).expanduser().resolve().parent
    try:
        relative = os.path.relpath(str(image), str(base))
    except ValueError:
        return str(image), "absolute"
    return relative, "relative"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build(
    source: str | Path,
    *,
    project_path: str = "",
    base_project: dict | None = None,
    splits: list[str] | None = None,
    mode: str = "new",
    duplicate: str = "skip",
    task: str = "auto",
    limit: int = MAX_IMAGES,
) -> tuple[dict, dict]:
    """构建导入后的工程与报告，返回 ``(project, report)``。

    ``mode="new"`` 生成一份全新工程；``mode="merge"`` 把数据并进 ``base_project``
    （类别按名字复用，图片按真实路径去重）。两种模式都只返回内存对象，
    是否落盘由调用方决定——导入失败时磁盘上不会留下半份工程。
    """
    if mode not in MODES:
        raise AnnotationImportError(f"未知导入方式：{mode}")
    if duplicate not in DUPLICATES:
        raise AnnotationImportError(f"未知重复处理方式：{duplicate}")
    if task != "auto" and task not in IMPORT_TASKS:
        raise AnnotationImportError(f"未知任务类型：{task}")
    limit = max(1, min(int(limit or MAX_IMAGES), MAX_IMAGES))
    if task == "classify":
        return _build_classify(
            source,
            project_path=project_path,
            base_project=base_project,
            mode=mode,
            duplicate=duplicate,
            limit=limit,
        )

    resolved = resolve_dataset(source)
    requested_splits = _pick_splits(resolved, splits)
    entries, truncated, split_warnings = _collect_entries(resolved, requested_splits, limit)
    if not entries:
        raise AnnotationImportError(
            "所选 split 下没有找到图片。请确认 data.yaml 里的 train/val/test 目录存在。"
        )

    # 1) 先读全部标签原文：任务类型要先按「每行字段数」推断出来，再决定怎么清洗
    #    （pose 的 visibility 是枚举值，不能当坐标夹；顺序反过来会把 2 夹成 1，
    #     并且会把整行判成"越界"直接丢掉，最后任务被误判成 detect）
    prepared = []
    token_counts: list[int] = []
    for entry in entries:
        text = entry["label"].read_text(encoding="utf-8", errors="replace") if entry["label"] else ""
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) >= 5:
                token_counts.append(len(tokens))
        width, height = dataset._image_size(entry["path"])  # noqa: SLF001 - 只读图片头
        prepared.append({**entry, "text": text, "width": width, "height": height})

    resolved_task = dataset.guess_task(token_counts) if task == "auto" else task
    if resolved_task not in IMPORT_TASKS:
        raise AnnotationImportError(f"推断出的任务类型不受支持：{resolved_task}")

    class_hits: dict[int, int] = {}
    issues = {
        "malformed_lines": 0,
        "out_of_range_objects": 0,
        "clamped_lines": 0,
    }
    for item in prepared:
        sanitized, stats = sanitize_label(item["text"], resolved_task)
        issues["malformed_lines"] += stats["malformed"]
        issues["out_of_range_objects"] += stats["out_of_range"]
        issues["clamped_lines"] += stats["clamped"]
        for line in sanitized.splitlines():
            tokens = line.split()
            if tokens:
                class_id = int(float(tokens[0]))
                class_hits[class_id] = class_hits.get(class_id, 0) + 1
        item["sanitized"] = sanitized

    # 2) 类别表：names 优先，没有就用标签里的类别号补占位名
    names = [str(item.get("name", "")).strip() for item in resolved["classes"]]
    if not names:
        upper = max(class_hits, default=-1) + 1
        names = [f"class_{index}" for index in range(upper)]
    warnings: list[str] = list(split_warnings)
    if not resolved["classes"]:
        warnings.append("数据源里没有 data.yaml 的 names，类别名用 class_0 / class_1 占位")
    if any(not name for name in names):
        warnings.append("data.yaml 里有类别名为空，已按 class_<序号> 补名")
        names = [name or f"class_{index}" for index, name in enumerate(names)]

    base = None
    if mode == "merge":
        if not base_project:
            raise AnnotationImportError("合并模式需要先打开一个工程")
        try:
            base, base_warnings = annotation.normalize_project(base_project)
        except annotation.AnnotationError as exc:
            raise AnnotationImportError(f"当前工程内容不合法：{exc}") from exc
        warnings.extend(base_warnings)
        if base.get("task") != resolved_task:
            raise AnnotationImportError(
                f"当前工程是 {base.get('task')} 任务，导入数据是 {resolved_task}；"
                "一个 .tfanno 只能有一种任务类型，请改用「新建工程」"
            )

    classes, class_map = _merge_classes(base, names)

    # 3) 逐图转标注
    if base is not None:
        project = base
        existing = {
            _abs_key(annotation.resolve_image_path(project_path, image.get("path") or "")): image
            for image in base.get("images") or []
        }
    else:
        project = annotation.new_project(task=resolved_task, classes=classes)
        existing = {}

    used_ids = {str(image.get("id") or "") for image in project.get("images") or []}
    split_stats: dict[str, dict] = {}
    added_images = 0
    added_objects = 0
    skipped_existing = 0
    replaced_images = 0
    empty_images = 0
    unknown_class_objects = 0
    clamped_points = 0
    unreadable_images = 0
    path_modes: set[str] = set()

    for item in prepared:
        split = item["split"]
        bucket = split_stats.setdefault(split, {"images": 0, "objects": 0})

        # 宽高读不到就没法把归一化坐标换成像素；这种图只进工程不导框
        if item["width"] <= 0 or item["height"] <= 0:
            unreadable_images += 1

        objects, unknown, clamped = _build_objects(item, resolved_task, class_map, names)
        unknown_class_objects += unknown
        clamped_points += clamped

        bucket["images"] += 1
        bucket["objects"] += len(objects)

        if not objects:
            empty_images += 1

        absolute = item["path"]
        key = _abs_key(absolute)
        stored, mode_used = store_path(absolute, project_path)
        path_modes.add(mode_used)

        if key in existing:
            if duplicate == "skip":
                skipped_existing += 1
                continue
            if duplicate == "replace":
                # 保留原有 id 与 split：用户可能已经手工调整过集合划分
                existing[key]["annotations"] = objects
                existing[key]["width"] = item["width"] or existing[key].get("width", 0)
                existing[key]["height"] = item["height"] or existing[key].get("height", 0)
                replaced_images += 1
                added_objects += len(objects)
                continue

        image_id = _unique_id(absolute.stem, used_ids)
        used_ids.add(image_id)
        image = annotation.new_image(
            image_id, stored, item["width"], item["height"], item["split"], objects
        )
        project["images"].append(image)
        existing[key] = image
        added_images += 1
        added_objects += len(objects)

    project["task"] = resolved_task
    project["classes"] = classes

    report = {
        "source": resolved.get("source") or str(source),
        "root": str(resolved.get("base") or ""),
        "yaml": str(resolved.get("yaml") or ""),
        "task": resolved_task,
        "mode": mode,
        "duplicate": duplicate,
        "classes": [{"id": int(c["id"]), "name": str(c["name"])} for c in classes],
        "splits": [name for name in traininfo.SPLITS if name in requested_splits],
        "images": len(prepared),
        "objects": sum(bucket["objects"] for bucket in split_stats.values()),
        "per_split": split_stats,
        "added_images": added_images,
        "added_objects": added_objects,
        "skipped_existing": skipped_existing,
        "replaced_images": replaced_images,
        "empty_images": empty_images,
        "unknown_class_objects": unknown_class_objects,
        "clamped_points": clamped_points,
        "unreadable_images": unreadable_images,
        "orphan_labels": 0,
        "truncated": truncated,
        "path_mode": "relative" if path_modes == {"relative"} else "absolute",
        **issues,
        "warnings": [],
    }

    orphan, orphan_samples = (0, [])
    if not truncated:
        # 截断时「没被用到的标签」可能是被切掉的图，报成孤儿会误导；
        # 宁可少报一条提示，也不要给人一个假的数字。
        used_labels = {_abs_key(item["label"]) for item in prepared if item["label"]}
        orphan, orphan_samples = _count_orphan_labels(resolved, requested_splits, used_labels)
    report["orphan_labels"] = orphan

    warnings.extend(_build_warnings(report, orphan_samples, prepared))
    report["warnings"] = warnings[:20]
    return project, report


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _pick_splits(resolved: dict, splits: list[str] | None) -> list[str]:
    """确认要导入哪些 split。未指定时导入全部存在的 split。"""
    available = [
        name
        for name in traininfo.SPLITS
        if any(Path(item).is_dir() for item in resolved["splits"].get(name) or [])
    ]
    if not available:
        raise AnnotationImportError(
            "data.yaml 里的 train/val/test 都指向不存在的目录。"
            "请检查 path: 与各 split 的相对路径。"
        )
    if not splits:
        return available
    unknown = [name for name in splits if name not in traininfo.SPLITS]
    if unknown:
        raise AnnotationImportError(f"未知的 split：{', '.join(unknown)}")
    picked = [name for name in traininfo.SPLITS if name in splits and name in available]
    if not picked:
        raise AnnotationImportError("所选的 split 下没有目录")
    return picked


def _collect_entries(
    resolved: dict, splits: list[str], limit: int
) -> tuple[list[dict], bool, list[str]]:
    """按 split 收集图片条目。条目含图片、标签路径与所属 split。"""
    entries: list[dict] = []
    warnings: list[str] = []
    truncated = False
    yaml_dir = Path(resolved["yaml_dir"])

    for split in splits:
        for images_dir in resolved["splits"].get(split) or []:
            images_dir = Path(images_dir)
            if not images_dir.is_dir():
                warnings.append(f"{split} 目录不存在，已跳过：{images_dir}")
                continue
            for image in traininfo.collect_images(images_dir):
                if len(entries) >= limit:
                    truncated = True
                    break
                entries.append(
                    {
                        "path": image,
                        "split": split,
                        "label": _label_path(image, images_dir, yaml_dir, split),
                    }
                )
            if truncated:
                break
        if truncated:
            break
    return entries, truncated, warnings


def _merge_classes(base: dict | None, names: list[str]) -> tuple[list[dict], dict[int, int]]:
    """生成目标工程的类别表，并返回 ``data.yaml 类别号 -> 工程类别 id`` 映射。

    合并模式按**名字**复用已有类别：这样同一批数据反复导入不会出现
    「missing_hole (0)」和「missing_hole (6)」两个类，训练时会各占一个检测头。
    """
    if base is None:
        classes = [annotation.new_class(index, name) for index, name in enumerate(names)]
        return classes, {index: index for index in range(len(names))}

    classes = [dict(item) for item in base.get("classes") or []]
    by_name: dict[str, int] = {}
    for item in classes:
        by_name.setdefault(str(item.get("name") or ""), int(item.get("id", 0)))
    next_id = max((int(item.get("id", 0)) for item in classes), default=-1) + 1

    class_map: dict[int, int] = {}
    for index, name in enumerate(names):
        if name in by_name:
            class_map[index] = by_name[name]
            continue
        classes.append(annotation.new_class(next_id, name))
        by_name[name] = next_id
        class_map[index] = next_id
        next_id += 1
    return classes, class_map


def _build_objects(
    item: dict, task: str, class_map: dict[int, int], names: list[str]
) -> tuple[list[dict], int, int]:
    """把一张图的标签行转成标注对象，返回 ``(对象列表, 未知类别数, 越界被夹的对象数)``。

    坐标换算复用 ``dataset.parse_yolo_label``；这里额外把点夹回图像范围：
    YOLO 允许框有一部分在图外，但 `.tfanno` 的导出校验会把「点在图外」报成问题，
    用户会以为是导入坏了。夹回边界并计数，比留着一堆红字更诚实。
    """
    if item["width"] <= 0 or item["height"] <= 0:
        return [], 0, 0

    width = float(item["width"])
    height = float(item["height"])
    parsed = dataset.parse_yolo_label(item["sanitized"], item["width"], item["height"], task)
    # obb 用 4 点多边形而不是 rotated_rect：.tfanno 的旋转框只存「2 点 + rotation」，
    # 而 dataset 导出 obb 时只认前 4 个点——存成旋转框会让导出的角点丢掉角度。
    # 用 4 个角点存成多边形，标注页照样能拖动顶点，导回 YOLO 时形状一字不差。
    shape = {
        "detect": "rectangle",
        "segment": "polygon",
        "obb": "polygon",
        "pose": "keypoints",
    }[task]

    objects: list[dict] = []
    unknown = 0
    clamped = 0
    for index, raw in enumerate(parsed, start=1):
        target = class_map.get(int(raw.class_id))
        if target is None:
            unknown += 1
            continue

        points = []
        touched = False
        for point in raw.points:
            x = min(max(float(point[0]), 0.0), width)
            y = min(max(float(point[1]), 0.0), height)
            if (x, y) != (float(point[0]), float(point[1])):
                touched = True
            points.append([x, y])
        if touched:
            clamped += 1

        candidate = {
            "id": f"o{index}",
            "class_id": target,
            "label": names[int(raw.class_id)] if 0 <= int(raw.class_id) < len(names) else "",
            "shape_type": shape,
            "points": points,
        }
        if task == "pose":
            candidate["keypoint_visibility"] = [int(v) for v in (raw.visibility or [])]
        try:
            objects.append(annotation.normalize_object(candidate))
        except annotation.AnnotationError:
            # 单个对象形状不合法不该让整张图导入失败；调用方只关心数量
            continue

    # 同一张图里编号必须唯一：被跳过的类别会让 o<行号> 出现空档但不重复，
    # 这里仍重新编号，保证导出校验（图内 id 唯一）一定通过。
    for index, obj in enumerate(objects, start=1):
        obj["id"] = f"o{index}"
    return objects, unknown, clamped


def _count_orphan_labels(resolved: dict, splits: list[str], used: set[str]) -> tuple[int, list[str]]:
    """统计「有 txt 但没有同名图片」的孤儿标签。

    孤儿标签不能导入（没有图就没有坐标系），但必须报出来：它通常意味着
    数据集本身少了一张图，或者 labels 与 images 的目录配错了。
    """
    count = 0
    samples: list[str] = []
    seen: set[str] = set()
    yaml_dir = Path(resolved["yaml_dir"])
    for split in splits:
        for images_dir in resolved["splits"].get(split) or []:
            images_dir = Path(images_dir)
            label_dir = traininfo.labels_dir_for(images_dir, yaml_dir, split)
            if not label_dir.is_dir():
                continue
            for label in label_dir.rglob("*.txt"):
                key = _abs_key(label)
                # 两个 split 可能共用同一个标签目录，按绝对路径去重，避免重复计数
                if key in seen:
                    continue
                seen.add(key)
                if key in used:
                    continue
                count += 1
                if len(samples) < SAMPLE_LIMIT:
                    samples.append(label.name)
    return count, samples


def _unique_id(stem: str, used: set[str]) -> str:
    """图片 id 去重：不同子目录可能同名，id 重复会让草稿串图。"""
    base = stem or "image"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _build_warnings(report: dict, orphan_samples: list[str], entries: list[dict]) -> list[str]:
    """把统计数字翻译成人能看懂的提示，顺序按「会不会让结果变错」排。"""
    warnings: list[str] = []
    if report["unknown_class_objects"]:
        warnings.append(
            f"{report['unknown_class_objects']} 个框的类别号超出 data.yaml 的 names 范围，已跳过；"
            "请检查标签与类别表是否配对"
        )
    if report["malformed_lines"]:
        warnings.append(f"{report['malformed_lines']} 行标签格式不合法（字段数不足），已跳过")
    if report["out_of_range_objects"]:
        warnings.append(
            f"{report['out_of_range_objects']} 行标签的坐标明显越界（超出 ±{COORD_TOLERANCE:g}），已跳过"
        )
    if report["clamped_lines"]:
        warnings.append(
            f"{report['clamped_lines']} 行标签的坐标轻微越界，已夹回 [0,1] 再换算"
        )
    if report["clamped_points"]:
        warnings.append(
            f"{report['clamped_points']} 个框有点落在图像外，已夹到图像边界"
            "（YOLO 允许框出界，但工程内保持一致更好核对）"
        )
    if report["orphan_labels"]:
        sample = "、".join(orphan_samples)
        warnings.append(
            f"{report['orphan_labels']} 个标签找不到同名图片，未导入"
            + (f"（如 {sample}）" if sample else "")
        )
    if report["unreadable_images"]:
        warnings.append(f"{report['unreadable_images']} 张图读不到尺寸，只导入图片、不导入框")
    if report["empty_images"]:
        warnings.append(f"{report['empty_images']} 张图没有标注，工程里以 0 框显示")
    if report["truncated"]:
        warnings.append(f"图片超过上限 {MAX_IMAGES} 张，只导入了前 {report['images']} 张")
    if report["path_mode"] == "absolute" and report["added_images"]:
        warnings.append("图片按绝对路径写入：未指定 .tfanno 路径，或数据集与工程不在同一个盘")
    if report["task"] == "obb" and report["added_objects"]:
        warnings.append(
            "obb 的 4 个角点按多边形导入（可拖顶点）：.tfanno 的旋转框只存 2 点 + 角度，"
            "存成旋转框会让再次导出的 YOLO 标签丢掉角度"
        )
    return warnings


def _build_classify(
    source: str | Path,
    *,
    project_path: str,
    base_project: dict | None,
    mode: str,
    duplicate: str,
    limit: int,
) -> tuple[dict, dict]:
    """从「类别文件夹」导入整图标签。检测用的 labels 目录直接拒绝。"""
    root = Path(source).expanduser()
    if not root.exists():
        raise AnnotationImportError(f"数据源不存在：{root}")
    if root.is_file():
        if root.suffix.lower() in {".yaml", ".yml"}:
            root = root.parent
        else:
            raise AnnotationImportError("分类导入请选类别文件夹，或选里面有 train/类别名 的目录。")
    if (root / "images").is_dir() and (root / "labels").is_dir():
        raise AnnotationImportError("这是检测数据集（有 labels 目录），不能导入分类工程。")

    found: list[tuple[str, str, Path]] = []
    split_names = [name for name in ("train", "val", "test") if (root / name).is_dir()]
    if split_names:
        for split in split_names:
            for class_dir in sorted((p for p in (root / split).iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                found.extend(_classify_files(class_dir, split, class_dir.name))
    else:
        class_dirs = [p for p in root.iterdir() if p.is_dir()]
        if not class_dirs:
            raise AnnotationImportError("没有找到类别文件夹。期望 良品/图片.jpg，或 train/良品/图片.jpg。")
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            found.extend(_classify_files(class_dir, "train", class_dir.name))
    if not found:
        raise AnnotationImportError("类别文件夹里没有图片。")

    truncated = len(found) > limit
    found = found[:limit]
    classes: list[dict] = []
    name_to_id: dict[str, int] = {}
    if mode == "merge" and base_project:
        for item in base_project.get("classes") or []:
            classes.append(dict(item))
            name_to_id[str(item.get("name"))] = int(item.get("id", len(name_to_id)))
    for _split, class_name, _path in found:
        if class_name not in name_to_id:
            class_id = len(classes)
            name_to_id[class_name] = class_id
            classes.append(annotation.new_class(class_id, class_name))

    images = []
    if mode == "merge" and base_project:
        images = [dict(item) for item in base_project.get("images") or []]
    existing = {str(item.get("path")) for item in images}
    added = 0
    skipped_existing = 0
    for split, class_name, path in found:
        key = str(path)
        if key in existing:
            skipped_existing += 1
            if duplicate != "replace":
                continue
            for item in images:
                if str(item.get("path")) == key:
                    item["class_id"] = name_to_id[class_name]
                    item["split"] = split
                    item["annotations"] = []
            continue
        images.append(
            annotation.new_image(path.stem, str(path), split=split, class_id=name_to_id[class_name])
        )
        existing.add(key)
        added += 1

    project = annotation.new_project("classify", classes, images)
    report = {
        "task": "classify",
        "classes": [{"id": item["id"], "name": item["name"]} for item in classes],
        "images": len(found),
        "objects": added,
        "added_images": added,
        "added_objects": added,
        "skipped_existing": skipped_existing,
        "per_split": {},
        "warnings": ["分类导入按文件夹名作为整图类别，不读取检测框。"],
        "path_mode": "absolute",
        "truncated": truncated,
    }
    return project, report


def _classify_files(folder: Path, split: str, class_name: str) -> list[tuple[str, str, Path]]:
    rows = []
    for path in sorted(folder.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.suffix.lower() in dataset.IMAGE_EXTS:
            rows.append((split, class_name, path))
    return rows
