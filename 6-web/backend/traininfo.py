# -*- coding: utf-8 -*-
"""训练数据集体检：解析 ``dataset.yaml`` 并统计每个 split 的真实内容。

为什么单独写一个模块而不复用 ``dataset.load_project``：

- ``dataset`` 模块的职责是「划分 + 导出」，它把整个数据集读成对象列表，
  每张图都要开一次图像头，还要为 ``.tfanno`` 走专门分支；
- 训练页只要一份「已经导出的 YOLO 目录健康报告」，多读的东西都是白花时间；
- 这里统计只要标签文件名 + 每行首列类别号，不做任何几何换算。

口径约定（与 Ultralytics 保持一致）：

- ``dataset.yaml`` 的 ``path`` 为相对路径时，以 yaml 所在目录为基准；
- ``train`` / ``val`` / ``test`` 是相对 ``path`` 的目录（也接受绝对路径）；
- 图片目录 ``images/<split>`` 对应标签目录 ``labels/<split>``，
  若路径里没有 ``images`` 这一段，则退化为 ``<yaml 目录>/labels/<split>``。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import dataset

#: 参与统计的图片扩展名。比 config.IMAGE_SUFFIXES 多一个 webp，
#: 因为数据集常常是从别处抓来的，.webp 很常见。
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: 三个 split 的固定顺序，前端按这个顺序渲染，避免字典顺序漂移。
SPLITS = ("train", "val", "test")

#: 读文件头统计分辨率的上限；超过就等距抽样，避免几万张图卡住页面。
MAX_SIZE_SAMPLES = 2000

#: 每个 split 最多解析多少行标签；防一手异常大的标签文件。
MAX_LABEL_LINES = 2_000_000

#: 概览缓存的有效期（秒）。用户连点「扫描数据集」时不该重复遍历磁盘。
CACHE_TTL = 60.0

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


class OverviewError(Exception):
    """数据集结构不可用（路径不存在、yaml 读不出等）。"""


# ---------------------------------------------------------------------------
# dataset.yaml 解析
# ---------------------------------------------------------------------------
def _strip_quotes(token: str) -> str:
    token = token.strip().rstrip(",")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1].strip()
    return token


def parse_scalars(text: str) -> dict[str, str]:
    """抽取 ``dataset.yaml`` 顶层的标量键。

    只认 ``path`` / ``train`` / ``val`` / ``test`` / ``nc`` 这几个训练真正用到的键；
    缩进行（``names:`` 下面那种）与注释行直接跳过。之所以手写而不引入 PyYAML：
    本项目的 ``dataset.py`` 已经用同一套手写解析，多一个依赖不值当。
    """
    wanted = {"path", "train", "val", "test", "nc"}
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        # 缩进行属于 names 之类的块内容，不是顶层标量
        if line[0] in " \t":
            continue
        head, sep, tail = line.partition(":")
        if not sep:
            continue
        key = head.strip().lower()
        if key not in wanted:
            continue
        value = _strip_quotes(tail)
        if value:
            found[key] = value
    return found


def _split_dirs(value: str) -> list[str]:
    """``train: images/train`` 或 ``train: [a, b]`` 都转成字符串列表。"""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [item for item in (_strip_quotes(part) for part in inner.split(",")) if item]
    return [value] if value else []


def _labels_dir(images_dir: Path, yaml_dir: Path, split: str) -> Path:
    """按 Ultralytics 的约定推导标签目录。"""
    parts = list(images_dir.parts)
    if "images" in parts:
        # 从后往前替换最后一个 images 段，兼容 images/train 与 a/b/images/train
        index = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :])
    return yaml_dir / "labels" / split


def resolve_splits(yaml_path: str | Path) -> dict[str, Any]:
    """把 dataset.yaml 解析成 ``{base, classes, nc, splits:{train:Path|None,...}}``。

    供本模块的 ``scan`` 与 ``augpreview`` 共用：预览也要知道训练集在哪。
    """
    path = Path(yaml_path).expanduser()
    if not path.is_file():
        raise OverviewError(f"dataset.yaml 不存在：{path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OverviewError(f"读取 dataset.yaml 失败：{exc}") from exc

    scalars = parse_scalars(text)
    yaml_dir = path.parent

    raw_root = scalars.get("path", "")
    if raw_root:
        base = Path(raw_root).expanduser()
        if not base.is_absolute():
            base = (yaml_dir / base).resolve()
    else:
        base = yaml_dir

    splits: dict[str, list[Path]] = {}
    for name in SPLITS:
        value = scalars.get(name, "")
        dirs: list[Path] = []
        for item in _split_dirs(value):
            candidate = Path(item).expanduser()
            if not candidate.is_absolute():
                candidate = (base / candidate) if base else (yaml_dir / candidate)
            dirs.append(candidate.resolve())
        splits[name] = dirs

    classes = dataset.parse_names_from_yaml(text)
    nc_raw = scalars.get("nc", "")
    nc = int(nc_raw) if nc_raw.isdigit() else None

    return {
        "yaml": str(path.resolve()),
        "yaml_dir": yaml_dir,
        "base": base,
        "text": text,
        "classes": classes,
        "nc": nc,
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# 单个 split 统计
# ---------------------------------------------------------------------------
def _collect_files(root: Path) -> list[Path]:
    """递归收集图片（嵌套目录在真实数据集里很常见）。"""
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def parse_label_file(path: Path) -> tuple[dict[int, int], int, int]:
    """解析一个 YOLO 标签文件。

    返回 ``(每类实例数, 总行数, 畸形行数)``。只取首列类别号——概览要的是
    「每类多少个」，坐标对不对交给训练本身去暴露（导出侧已有单测覆盖几何）。
    """
    per_class: dict[int, int] = {}
    lines = 0
    malformed = 0
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return per_class, 0, 1
    for raw in content.splitlines():
        lines += 1
        if lines > MAX_LABEL_LINES:
            break
        stripped = raw.strip()
        if not stripped:
            malformed += 1
            continue
        token = stripped.split()[0]
        try:
            class_id = int(float(token))
        except ValueError:
            malformed += 1
            continue
        if class_id < 0:
            malformed += 1
            continue
        per_class[class_id] = per_class.get(class_id, 0) + 1
    return per_class, lines, malformed


def _image_size(path: Path) -> tuple[int, int]:
    """只读文件头拿尺寸，不解码像素。"""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001 - 单张读不出不该中断整个统计
        return 0, 0


def _sample(items: list[Path], limit: int) -> tuple[list[Path], bool]:
    """等距抽样：比 ``items[:limit]`` 更能代表整个数据集。"""
    if len(items) <= limit:
        return items, False
    step = len(items) / float(limit)
    return [items[min(len(items) - 1, int(i * step))] for i in range(limit)], True


def stat_split(dirs: list[Path], split: str, yaml_dir: Path) -> dict:
    """统计一个 split（可能对应多个目录）。"""
    images: list[Path] = []
    for d in dirs:
        images.extend(_collect_files(d))

    label_files: list[Path] = []
    for d in dirs:
        label_root = _labels_dir(d, yaml_dir, split)
        if label_root.is_dir():
            label_files.extend(p for p in label_root.rglob("*.txt") if p.is_file())

    # 同名标签在嵌套目录里重复时只保留第一个，并单独记数
    labels_by_stem: dict[str, Path] = {}
    duplicate_stems: list[str] = []
    for item in label_files:
        if item.stem in labels_by_stem:
            duplicate_stems.append(item.name)
            continue
        labels_by_stem[item.stem] = item

    image_stems = {p.stem for p in images}
    per_class: dict[int, int] = {}
    empty = 0
    malformed = 0
    used_labels: set[str] = set()
    for image in images:
        label = labels_by_stem.get(image.stem)
        if label is None:
            continue
        used_labels.add(image.stem)
        counts, lines, bad = parse_label_file(label)
        malformed += bad
        if lines == 0:
            empty += 1
        for class_id, number in counts.items():
            per_class[class_id] = per_class.get(class_id, 0) + number

    orphans = sorted(stem for stem in labels_by_stem if stem not in image_stems)
    missing = sorted(p.stem for p in images if p.stem not in labels_by_stem)

    return {
        "split": split,
        "configured": bool(dirs),
        "dirs": [str(d) for d in dirs],
        "missing_dirs": [str(d) for d in dirs if not d.is_dir()],
        "images": len(images),
        "labels": len(labels_by_stem),
        "empty_labels": empty,
        "missing_labels": len(missing),
        "missing_examples": missing[:10],
        "orphan_labels": len(orphans),
        "orphan_examples": orphans[:10],
        "duplicate_labels": len(duplicate_stems),
        "malformed_lines": malformed,
        "instances": sum(per_class.values()),
        "class_instances": {str(k): v for k, v in sorted(per_class.items())},
        "label_coverage": round(len(used_labels) / len(images), 4) if images else 0.0,
    }


def resolution_profile(images: list[Path]) -> dict:
    """分辨率分布：单边尺寸的最小/中位/最大 + 最常见的几种尺寸。"""
    sample, sampled = _sample(images, MAX_SIZE_SAMPLES)
    sizes: list[tuple[int, int]] = []
    for item in sample:
        width, height = _image_size(item)
        if width > 0 and height > 0:
            sizes.append((width, height))
    if not sizes:
        return {"sampled": sampled, "count": 0, "width": [0, 0, 0], "height": [0, 0, 0], "top": []}

    widths = sorted(w for w, _ in sizes)
    heights = sorted(h for _, h in sizes)
    counter: dict[tuple[int, int], int] = {}
    for size in sizes:
        counter[size] = counter.get(size, 0) + 1
    top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    def _median(values: list[int]) -> int:
        return values[len(values) // 2]

    return {
        "sampled": sampled,
        "count": len(sizes),
        "width": [widths[0], _median(widths), widths[-1]],
        "height": [heights[0], _median(heights), heights[-1]],
        "top": [{"width": w, "height": h, "count": c} for (w, h), c in top],
    }


# ---------------------------------------------------------------------------
# 警告
# ---------------------------------------------------------------------------
def build_warnings(result: dict) -> list[dict]:
    """把统计结果翻译成人能看懂的警告。等级只用 ``info`` / ``warn`` / ``bad``。"""
    warnings: list[dict] = []
    splits = result["splits"]
    classes = result["classes"]
    total_images = sum(s["images"] for s in splits.values())

    configured = [name for name, stat in splits.items() if stat["configured"]]
    if not configured:
        warnings.append(
            {"level": "bad", "code": "no_split", "text": "dataset.yaml 里没有 train / val / test 任何一个目录"}
        )
    elif "train" not in configured:
        warnings.append(
            {"level": "bad", "code": "no_train", "text": "缺少 train 目录，Ultralytics 无法开始训练"}
        )

    for name, stat in splits.items():
        if stat["missing_dirs"]:
            warnings.append(
                {
                    "level": "bad",
                    "code": "split_missing",
                    "text": f"{name} 目录不存在：{', '.join(stat['missing_dirs'])}",
                }
            )
        if stat["images"] == 0 and stat["configured"]:
            # test 目录在 80/20 划分下必然为空（导出脚本照样建目录），
            # 这不是问题，只在 info 级别提一句；train/val 空了才是真问题。
            if name == "test":
                warnings.append(
                    {"level": "info", "code": "split_empty", "text": "test 目录为空（本次划分没有分配测试集）"}
                )
            else:
                warnings.append(
                    {"level": "warn", "code": "split_empty", "text": f"{name} 目录里没有图片"}
                )

    if total_images == 0:
        warnings.append({"level": "bad", "code": "no_image", "text": "所有 split 加起来一张图片都没有"})

    total_empty = sum(s["empty_labels"] for s in splits.values())
    total_missing = sum(s["missing_labels"] for s in splits.values())
    total_orphan = sum(s["orphan_labels"] for s in splits.values())

    if total_missing:
        examples = []
        for stat in splits.values():
            examples.extend(stat["missing_examples"])
        warnings.append(
            {
                "level": "bad",
                "code": "missing_labels",
                "text": f"{total_missing} 张图片没有对应标签文件（会当作负样本训练）："
                + "、".join(examples[:5]),
            }
        )
    if total_orphan:
        examples = []
        for stat in splits.values():
            examples.extend(stat["orphan_examples"])
        warnings.append(
            {
                "level": "warn",
                "code": "orphan_labels",
                "text": f"{total_orphan} 个标签文件找不到同名图片："
                + "、".join(f"{name}.txt" for name in examples[:5]),
            }
        )
    if total_empty:
        ratio = total_empty / max(1, total_images)
        warnings.append(
            {
                "level": "warn" if ratio > 0.1 else "info",
                "code": "empty_labels",
                "text": f"{total_empty} 个标签文件是空的（占 {ratio * 100:.1f}%），这些图会作为纯背景图参与训练",
            }
        )

    zero = [item["name"] for item in classes if result["class_instances"].get(str(item["id"]), 0) == 0]
    if classes and zero:
        warnings.append(
            {
                "level": "bad",
                "code": "zero_instance",
                "text": f"这些类别一个框都没有：{'、'.join(zero)}（训练时会被当成永远检测不到）",
            }
        )

    nc = result["nc"]
    if nc is not None and classes and nc != len(classes):
        warnings.append(
            {
                "level": "bad",
                "code": "nc_mismatch",
                "text": f"nc={nc} 与 names 里的类别数 {len(classes)} 不一致，Ultralytics 会据此建错检测头",
            }
        )

    beyond = sorted(
        int(key) for key in result["class_instances"]
        if classes and int(key) >= len(classes)
    )
    if beyond:
        warnings.append(
            {
                "level": "bad",
                "code": "class_out_of_range",
                "text": f"标签里出现了超出 names 范围的类别号：{beyond[:10]}",
            }
        )

    if result["resolutions"]["sampled"]:
        warnings.append(
            {
                "level": "info",
                "code": "sampled",
                "text": f"图片超过 {MAX_SIZE_SAMPLES} 张，分辨率统计已等距抽样",
            }
        )

    ratio_parts = []
    for name in SPLITS:
        stat = splits[name]
        if stat["configured"] and total_images:
            ratio_parts.append(f"{name} {stat['images'] * 100 // max(1, total_images)}%")
    if ratio_parts:
        warnings.append({"level": "info", "code": "split_ratio", "text": "划分比例：" + " / ".join(ratio_parts)})

    positive = {k: v for k, v in result["class_instances"].items() if v > 0}
    if len(positive) >= 2:
        values = sorted(positive.values())
        imbalance = values[-1] / max(1, values[0])
        if imbalance >= 10:
            warnings.append(
                {
                    "level": "warn",
                    "code": "imbalance",
                    "text": f"类别极不平衡：最多的类 {values[-1]} 个框，最少的类 {values[0]} 个（{imbalance:.0f}:1）",
                }
            )
    return warnings


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def scan(yaml_path: str | Path, *, use_cache: bool = True) -> dict:
    """扫描 ``dataset.yaml`` 指向的数据集，返回体检报告。"""
    path = Path(yaml_path).expanduser()
    try:
        stat = path.stat()
    except OSError as exc:
        raise OverviewError(f"dataset.yaml 不可访问：{path}（{exc}）") from exc

    key = str(path.resolve())
    if use_cache:
        with _cache_lock:
            hit = _cache.get(key)
            if hit and hit[0] > time.time() - CACHE_TTL and hit[1]["_stamp"] == (stat.st_mtime_ns, stat.st_size):
                cached = dict(hit[1])
                cached.pop("_stamp", None)
                cached["cached"] = True
                return cached

    started = time.perf_counter()
    resolved = resolve_splits(path)
    yaml_dir = resolved["yaml_dir"]

    splits: dict[str, dict] = {}
    total_class_instances: dict[str, int] = {}
    for name in SPLITS:
        stat_split_result = stat_split(resolved["splits"][name], name, yaml_dir)
        splits[name] = stat_split_result
        for class_id, number in stat_split_result["class_instances"].items():
            total_class_instances[class_id] = total_class_instances.get(class_id, 0) + number

    # 分辨率只统计 train：训练输入只关心它，val/test 少一批读文件头
    train_images: list[Path] = []
    for directory in resolved["splits"]["train"]:
        train_images.extend(_collect_files(directory))

    result = {
        "yaml": resolved["yaml"],
        "base": str(resolved["base"]),
        "nc": resolved["nc"],
        "classes": resolved["classes"],
        "splits": splits,
        "counts": {
            "total": sum(s["images"] for s in splits.values()),
            "train": splits["train"]["images"],
            "val": splits["val"]["images"],
            "test": splits["test"]["images"],
            "instances": sum(total_class_instances.values()),
        },
        "class_instances": dict(sorted(total_class_instances.items(), key=lambda kv: int(kv[0]))),
        "resolutions": resolution_profile(train_images),
        "elapsed_ms": 0,
        "cached": False,
    }
    result["warnings"] = build_warnings(result)
    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

    with _cache_lock:
        _cache[key] = (time.time(), {**result, "_stamp": (stat.st_mtime_ns, stat.st_size)})
    return result


def list_split_images(yaml_path: str | Path, split: str = "train") -> list[Path]:
    """按稳定顺序返回某个 split 的图片，供增强预览取样本。"""
    resolved = resolve_splits(yaml_path)
    if split not in resolved["splits"]:
        raise OverviewError(f"未知 split：{split}")
    images: list[Path] = []
    for directory in resolved["splits"][split]:
        images.extend(_collect_files(directory))
    return images
