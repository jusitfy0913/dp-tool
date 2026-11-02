"""数据集工作台：扫描 YOLO / .tfanno，按固定哈希划分，导出 YOLO / COCO / VOC。

与 Qt 版的关系
--------------
Qt 的 ``DatasetSplitPolicy`` 用 ``qHash(i + "_2026")`` 排序。qHash 的实现在不同的
Qt 版本、编译选项与运行时环境下都可能变化，历史划分无法逐张复现。本模块改用
FNV-1a 64 位，盐值仍保留 ``_2026``，好处是：

- 同一份数据 + 同一比例 → 任何机器、任何进程、任何 Python 版本结果完全一致；
- 不依赖 Qt，Web 端可以独立复算，不需要保存划分状态。

比例语义刻意与 Qt 保持一致：``train = n*train//100``、``val = n*val//100``，
剩下的全部归 ``test``。所以 80/20 时 test 为 0，70/20/10 时余数会落进 test。

坐标约定
--------
内存里一律用 **像素坐标**（与 Qt 的 AnnotationObject 一致），导出时再归一化，
这样 COCO 与 YOLO 两条导出路径不会因为反复归一化而漂移。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SALT = "_2026"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
TASKS = ("detect", "segment", "pose", "obb")
# COCO 17 关键点的左右翻转配对，Ultralytics 导出 pose 时需要
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


class DatasetError(Exception):
    """可读的业务错误，路由层统一转成 HTTP 400。"""


# ---------------------------------------------------------------------------
# 划分算法
# ---------------------------------------------------------------------------
def fnv1a64(text: str) -> int:
    """FNV-1a 64 位哈希（跨平台、跨版本稳定）。"""
    value = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK64
    return value


def rank_order(count: int) -> list[int]:
    """按固定哈希排出划分优先级（下标列表，已排序）。"""
    return sorted(range(count), key=lambda i: fnv1a64(f"{i}{SALT}"))


def assign_splits(count: int, train: int, val: int) -> list[str]:
    """给每张图分配 train/val/test，索引与 ``rank_order`` 无关，按原顺序返回。

    注意 train/val 是「向下取整」，余数全部进 test —— 与 Qt 版完全一致。
    """
    if count <= 0:
        return []
    for name, value in (("train", train), ("val", val)):
        if not 0 <= value <= 100:
            raise DatasetError(f"{name} 比例必须在 0~100 之间，当前为 {value}")
    if train + val > 100:
        raise DatasetError(f"train({train}%) + val({val}%) 不能超过 100%")

    train_n = count * train // 100
    val_n = count * val // 100
    result = ["test"] * count
    for rank, index in enumerate(rank_order(count)):
        if rank < train_n:
            result[index] = "train"
        elif rank < train_n + val_n:
            result[index] = "val"
        else:
            result[index] = "test"
    return result


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class DatasetObject:
    """一个标注对象，points 为像素坐标。"""

    class_id: int
    points: list[tuple[float, float]] = field(default_factory=list)
    visibility: list[int] = field(default_factory=list)


@dataclass
class DatasetImage:
    index: int
    path: Path
    rel: str
    width: int
    height: int
    objects: list[DatasetObject] = field(default_factory=list)
    current_split: str = "train"


@dataclass
class DatasetProject:
    source: Path
    kind: str  # yolo / tfanno
    task: str
    classes: list[dict] = field(default_factory=list)
    images: list[DatasetImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def class_name(self, class_id: int) -> str:
        for item in self.classes:
            if item["id"] == class_id:
                return item["name"]
        return f"class_{class_id}"


# ---------------------------------------------------------------------------
# data.yaml 解析（不引入 PyYAML 依赖，兼容三种常见写法）
# ---------------------------------------------------------------------------
def _strip_quotes(token: str) -> str:
    token = token.strip().rstrip(",")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1].strip()
    return token


def parse_names_from_yaml(text: str) -> list[dict]:
    """从 data.yaml 文本里抽取类别表，支持三种写法：

    1. ``names: [ "a", "b" ]``（含跨行）
    2. ``names:`` 下的 ``- a`` 列表
    3. 旧式字典 ``0: a`` 逐行
    """
    classes: list[dict] = []

    # 写法 1：行内列表
    lowered = text.lower()
    pos = lowered.find("names:")
    while pos != -1:
        segment = text[pos:]
        open_at = segment.find("[")
        close_at = segment.find("]")
        if open_at != -1 and close_at > open_at:
            inner = segment[open_at + 1 : close_at]
            for token in inner.split(","):
                name = _strip_quotes(token)
                if name:
                    classes.append({"id": len(classes), "name": name})
            if classes:
                return classes
        break

    # 写法 2：names: 下的 - 列表
    in_names = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not in_names:
            if stripped.lower().startswith("names:"):
                if "[" in stripped:
                    continue
                in_names = True
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            name = _strip_quotes(stripped[1:])
            if name:
                classes.append({"id": len(classes), "name": name})
            continue
        if ":" in stripped and not line.startswith((" ", "\t")):
            break
    if classes:
        return classes

    # 写法 3：逐行 0: name
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, sep, tail = stripped.partition(":")
        if not sep or not head.strip().isdigit():
            continue
        name = _strip_quotes(tail.split("#", 1)[0])
        if name:
            classes.append({"id": int(head.strip()), "name": name})
    classes.sort(key=lambda c: c["id"])
    return classes


# ---------------------------------------------------------------------------
# 扫描：YOLO 目录
# ---------------------------------------------------------------------------
def _image_size(path: Path) -> tuple[int, int]:
    """只读图片头拿尺寸，不整张解码。"""
    try:
        from PIL import Image  # 延迟导入：没装 Pillow 时只在需要尺寸处报错

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001 - 尺寸读不到不该中断整个扫描
        return 0, 0


def _guess_task(token_counts: list[int]) -> str:
    """按标签文件每行的字段数猜任务类型。"""
    if not token_counts:
        return "detect"
    pose_hits = sum(1 for n in token_counts if n == 5 + 17 * 3)
    if pose_hits and pose_hits * 2 >= len(token_counts):
        return "pose"
    obb_hits = sum(1 for n in token_counts if n == 9)
    if obb_hits and obb_hits * 2 >= len(token_counts):
        return "obb"
    detect_hits = sum(1 for n in token_counts if n == 5)
    if detect_hits * 2 >= len(token_counts):
        return "detect"
    return "segment"


def _parse_yolo_label(text: str, width: int, height: int, task: str) -> list[DatasetObject]:
    """把 YOLO 标签行解析成像素坐标对象。"""
    objects: list[DatasetObject] = []
    for raw in text.splitlines():
        tokens = raw.split()
        if len(tokens) < 5:
            continue
        try:
            class_id = int(float(tokens[0]))
            values = [float(v) for v in tokens[1:]]
        except ValueError:
            continue

        if task == "detect":
            cx, cy, bw, bh = values[:4]
            points = [
                ((cx - bw / 2) * width, (cy - bh / 2) * height),
                ((cx + bw / 2) * width, (cy + bh / 2) * height),
            ]
        elif task == "obb":
            if len(values) < 8:
                continue
            points = [(values[i] * width, values[i + 1] * height) for i in range(0, 8, 2)]
        elif task == "pose":
            if len(values) < 4 + 17 * 3:
                continue
            cx, cy, bw, bh = values[:4]
            # 用外接框占位前两个点，后续导出时会重新算框，关键点单独保存
            points = [
                ((cx - bw / 2) * width, (cy - bh / 2) * height),
                ((cx + bw / 2) * width, (cy + bh / 2) * height),
            ]
            visibility = []
            keypoints: list[tuple[float, float]] = []
            for i in range(4, 4 + 17 * 3, 3):
                keypoints.append((values[i] * width, values[i + 1] * height))
                visibility.append(int(values[i + 2]))
            objects.append(DatasetObject(class_id, keypoints, visibility))
            continue
        else:  # segment
            if len(values) < 6 or len(values) % 2:
                continue
            points = [(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)]

        objects.append(DatasetObject(class_id, points))
    return objects


def parse_yolo_label(text: str, width: int, height: int, task: str) -> list[DatasetObject]:
    """公开入口：把一段 YOLO 标签文本解析成像素坐标对象。

    标注导入（``annotationimport.py``）与数据集载入必须共用同一份几何换算。
    各写一份的话，同一份 txt 在「数据集预览」和「标注页」里可能差半个像素、
    甚至上下颠倒——而这种错不会报错，只会让标注框看起来"稍微有点歪"。
    """
    return _parse_yolo_label(text, width, height, task)


def guess_task(token_counts: list[int]) -> str:
    """按标签每行字段数推断任务类型（公开版，供标注导入复用同一套判据）。"""
    return _guess_task(token_counts)


def _collect_yolo_images(root: Path, warnings: list[str]) -> list[tuple[Path, str, Path | None]]:
    """返回 (图片路径, 原划分, 标签路径) 列表。"""
    found: list[tuple[Path, str, Path | None]] = []
    images_root = root / "images"
    labels_root = root / "labels"

    split_dirs = [(name, images_root / name, labels_root / name) for name in SPLITS]
    split_dirs = [item for item in split_dirs if item[1].is_dir()]
    if not split_dirs and images_root.is_dir():
        # 扁平结构：images/ + labels/
        split_dirs = [("train", images_root, labels_root)]

    if not split_dirs:
        raise DatasetError(
            f"没有找到图片目录：{images_root}。"
            "期望结构为 images/train、images/val（可再加 images/test），或直接 images/。"
        )

    for split_name, img_dir, lbl_dir in split_dirs:
        files = sorted(
            (p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
        if not files:
            warnings.append(f"{split_name} 集没有图片：{img_dir}")
        for path in files:
            label = lbl_dir / f"{path.stem}.txt"
            found.append((path, split_name, label if label.is_file() else None))
    return found


def load_yolo(source: Path, task: str = "auto") -> DatasetProject:
    """扫描一个 YOLO 数据集目录（或它里面的 data.yaml）。"""
    warnings: list[str] = []
    yaml_text = ""

    if source.is_file():
        if source.suffix.lower() in {".yaml", ".yml"}:
            root = source.parent
            yaml_text = source.read_text(encoding="utf-8", errors="replace")
        else:
            raise DatasetError(f"YOLO 输入应是目录或 data.yaml，收到：{source}")
    else:
        root = source
        for candidate in ("data.yaml", "dataset.yaml", "data.yml"):
            path = root / candidate
            if path.is_file():
                yaml_text = path.read_text(encoding="utf-8", errors="replace")
                break

    if not root.is_dir():
        raise DatasetError(f"目录不存在：{root}")

    classes = parse_names_from_yaml(yaml_text) if yaml_text else []
    if not classes:
        warnings.append("没有从 data.yaml 读到类别表，导出时类别名会用 class_0 / class_1 占位")

    entries = _collect_yolo_images(root, warnings)

    # 先读标签，用于推断任务类型
    token_counts: list[int] = []
    parsed: list[tuple[Path, str, list[DatasetObject], int, int]] = []
    for path, split_name, label in entries:
        width, height = _image_size(path)
        objects: list[DatasetObject] = []
        if label is not None:
            for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                tokens = line.split()
                if len(tokens) >= 5:
                    token_counts.append(len(tokens))
        parsed.append((path, split_name, objects, width, height))

    resolved_task = task if task in TASKS else _guess_task(token_counts)
    if task not in TASKS and task != "auto":
        raise DatasetError(f"未知任务类型：{task}，可选 {', '.join(TASKS)} 或 auto")

    images: list[DatasetImage] = []
    empty = 0
    for index, (path, split_name, _objects, width, height) in enumerate(parsed):
        label = entries[index][2]
        objects = []
        if label is not None:
            objects = _parse_yolo_label(
                label.read_text(encoding="utf-8", errors="replace"), width, height, resolved_task
            )
        if not objects:
            empty += 1
        images.append(
            DatasetImage(
                index=index,
                path=path,
                rel=str(path.relative_to(root)).replace("\\", "/"),
                width=width,
                height=height,
                objects=objects,
                current_split=split_name,
            )
        )

    if empty:
        warnings.append(f"{empty} 张图片没有标注或标签文件缺失，导出时会写成空标签")
    if any(img.width <= 0 for img in images):
        warnings.append("部分图片读不到尺寸，导出 COCO / VOC 时这些图会被跳过")

    return DatasetProject(
        source=root, kind="yolo", task=resolved_task, classes=classes, images=images, warnings=warnings
    )


# ---------------------------------------------------------------------------
# 扫描：.tfanno 标注工程
# ---------------------------------------------------------------------------
def load_tfanno(source: Path) -> DatasetProject:
    """读取 Qt 版的 .tfanno 工程文件。"""
    try:
        data = json.loads(source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"读取 .tfanno 失败：{exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError(".tfanno 内容不是 JSON 对象")

    task = str(data.get("task") or "detect").lower()
    if task not in TASKS:
        raise DatasetError(f".tfanno 里的任务类型不受支持：{task}")

    classes = []
    for item in data.get("classes") or []:
        if isinstance(item, dict):
            classes.append({"id": int(item.get("id", len(classes))), "name": str(item.get("name", ""))})

    warnings: list[str] = []
    images: list[DatasetImage] = []
    for index, item in enumerate(data.get("images") or []):
        if not isinstance(item, dict):
            continue
        raw_path = Path(str(item.get("path", "")))
        path = raw_path if raw_path.is_absolute() else (source.parent / raw_path)
        if not path.is_file():
            warnings.append(f"图片不存在，已跳过：{raw_path}")
            continue

        objects: list[DatasetObject] = []
        for obj in item.get("annotations") or []:
            if not isinstance(obj, dict):
                continue
            points = []
            for pair in obj.get("points") or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    points.append((float(pair[0]), float(pair[1])))
            visibility = [int(v) for v in (obj.get("keypoint_visibility") or []) if str(v).isdigit()]
            objects.append(DatasetObject(int(obj.get("class_id", 0)), points, visibility))

        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = _image_size(path)

        split_name = str(item.get("split") or "train").lower()
        if split_name not in SPLITS:
            split_name = "train"

        images.append(
            DatasetImage(
                index=len(images),
                path=path,
                rel=str(path).replace("\\", "/"),
                width=width,
                height=height,
                objects=objects,
                current_split=split_name,
            )
        )

    if not images:
        raise DatasetError(".tfanno 里没有可用的图片")
    if not classes:
        classes = [{"id": 0, "name": "object"}]
        warnings.append(".tfanno 没有类别表，已回退为单类别 object")

    return DatasetProject(
        source=source, kind="tfanno", task=task, classes=classes, images=images, warnings=warnings
    )


def load_project(source: str, task: str = "auto") -> DatasetProject:
    path = Path(source).expanduser()
    if not path.exists():
        raise DatasetError(f"路径不存在：{source}")
    if path.is_file() and path.suffix.lower() == ".tfanno":
        return load_tfanno(path)
    return load_yolo(path, task)


# ---------------------------------------------------------------------------
# 扫描结果摘要
# ---------------------------------------------------------------------------
def split_counts(splits: list[str]) -> dict:
    return {name: sum(1 for s in splits if s == name) for name in SPLITS}


def preview(project: DatasetProject, train: int, val: int, limit: int = 200) -> dict:
    """给前端预览用：不落盘，只算分配结果。"""
    splits = assign_splits(len(project.images), train, val)
    counts = split_counts(splits)
    counts["total"] = len(project.images)
    counts["empty"] = sum(1 for img in project.images if not img.objects)

    class_counter = {item["id"]: 0 for item in project.classes}
    for img in project.images:
        for obj in img.objects:
            class_counter[obj.class_id] = class_counter.get(obj.class_id, 0) + 1

    images = []
    for img, split in list(zip(project.images, splits))[:limit]:
        images.append(
            {
                "index": img.index,
                "rel": img.rel,
                "width": img.width,
                "height": img.height,
                "objects": len(img.objects),
                "current_split": img.current_split,
                "split": split,
            }
        )

    return {
        "kind": project.kind,
        "source": str(project.source),
        "task": project.task,
        "classes": project.classes,
        "class_objects": class_counter,
        "counts": counts,
        "images": images,
        "truncated": len(project.images) > limit,
        "warnings": project.warnings,
    }


# ---------------------------------------------------------------------------
# 导出辅助
# ---------------------------------------------------------------------------
def _norm(value: float) -> str:
    return f"{min(1.0, max(0.0, value)):.6f}"


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """返回 (minx, miny, maxx, maxy)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _prepare_output(output: Path, overwrite: bool, source: Path) -> None:
    """检查输出目录，防止误覆盖与自我递归。"""
    if output == source or source in output.parents:
        raise DatasetError(f"输出目录不能放在数据源内部：{output}（会导致自我复制）")
    if output.exists():
        if not output.is_dir():
            raise DatasetError(f"输出路径已存在且不是目录：{output}")
        if any(output.iterdir()) and not overwrite:
            raise DatasetError(f"输出目录非空：{output}。确认要覆盖请勾选「允许覆盖」")
    else:
        output.mkdir(parents=True, exist_ok=True)


def _yolo_line(obj: DatasetObject, task: str, width: int, height: int) -> str:
    if width <= 0 or height <= 0 or not obj.points:
        return ""
    parts = [str(obj.class_id)]

    if task == "detect":
        x1, y1, x2, y2 = _bounds(obj.points)
        parts += [
            _norm((x1 + x2) / 2 / width),
            _norm((y1 + y2) / 2 / height),
            _norm((x2 - x1) / width),
            _norm((y2 - y1) / height),
        ]
    elif task == "segment":
        if len(obj.points) < 3:
            return ""
        for x, y in obj.points:
            parts += [_norm(x / width), _norm(y / height)]
    elif task == "pose":
        # 关键点对象：前两个点只用来占位，真正的框由关键点外接框决定
        keypoints = obj.points[:17]
        if not keypoints:
            return ""
        x1, y1, x2, y2 = _bounds(keypoints)
        parts += [
            _norm((x1 + x2) / 2 / width),
            _norm((y1 + y2) / 2 / height),
            _norm((x2 - x1) / width),
            _norm((y2 - y1) / height),
        ]
        for i in range(17):
            if i < len(keypoints):
                x, y = keypoints[i]
            else:
                x = y = 0.0
            visible = obj.visibility[i] if i < len(obj.visibility) else 2
            parts += [_norm(x / width), _norm(y / height), str(visible)]
    else:  # obb
        points = obj.points
        if len(points) < 4:
            x1, y1, x2, y2 = _bounds(points)
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for x, y in points[:4]:
            parts += [_norm(x / width), _norm(y / height)]
    return " ".join(parts)


def _copy_image(src: Path, dst_dir: Path) -> Path:
    """复制图片；同目录重名时自动加序号，避免静默覆盖。"""
    dst = dst_dir / src.name
    counter = 2
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# 导出：YOLO
# ---------------------------------------------------------------------------
def export_yolo(project: DatasetProject, output: Path, splits: list[str], overwrite: bool) -> dict:
    _prepare_output(output, overwrite, project.source)
    for name in SPLITS:
        (output / "images" / name).mkdir(parents=True, exist_ok=True)
        (output / "labels" / name).mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {name: 0 for name in SPLITS}
    skipped: list[str] = []
    for image, split in zip(project.images, splits):
        if not image.path.is_file():
            skipped.append(f"图片不存在：{image.path}")
            continue
        target = _copy_image(image.path, output / "images" / split)
        lines = []
        for obj in image.objects:
            line = _yolo_line(obj, project.task, image.width, image.height)
            if line:
                lines.append(line)
        label = output / "labels" / split / f"{target.stem}.txt"
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[split] += 1

    # dataset.yaml：path 用正斜杠，Ultralytics 在 Windows 上也能吃
    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(project.classes)}",
    ]
    if project.task == "pose":
        yaml_lines += ["kpt_shape: [17, 3]", f"flip_idx: {FLIP_IDX}"]
    yaml_lines.append("names:")
    for item in project.classes:
        yaml_lines.append(f"  {item['id']}: {item['name']}")
    (output / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "format": "yolo",
        "output": str(output),
        "counts": written,
        "skipped": skipped,
        "extra_files": ["dataset.yaml"],
    }


# ---------------------------------------------------------------------------
# 导出：COCO
# ---------------------------------------------------------------------------
def export_coco(project: DatasetProject, output: Path, splits: list[str], overwrite: bool) -> dict:
    if project.task == "obb":
        raise DatasetError("标准 COCO 不支持旋转框，请改用 YOLO OBB 导出")
    _prepare_output(output, overwrite, project.source)

    images_json = []
    annotations_json = []
    skipped: list[str] = []
    ann_id = 1
    empty = 0

    for image, split in zip(project.images, splits):
        if image.width <= 0 or image.height <= 0:
            skipped.append(f"读不到图片尺寸，已跳过：{image.path}")
            continue
        if not image.path.is_file():
            skipped.append(f"图片不存在，已跳过：{image.path}")
            continue
        images_json.append(
            {
                "id": image.index + 1,
                "file_name": image.path.name,
                "width": image.width,
                "height": image.height,
                "split": split,
            }
        )
        if not image.objects:
            empty += 1
        for obj in image.objects:
            if not obj.points:
                continue
            x1, y1, x2, y2 = _bounds(obj.points)
            item = {
                "id": ann_id,
                "image_id": image.index + 1,
                "category_id": obj.class_id,
                "iscrowd": 0,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "area": round((x2 - x1) * (y2 - y1), 2),
            }
            if project.task == "segment" and len(obj.points) >= 3:
                polygon = []
                for x, y in obj.points:
                    polygon += [round(x, 2), round(y, 2)]
                item["segmentation"] = [polygon]
            elif project.task == "pose":
                keypoints = []
                visible = 0
                for i in range(17):
                    if i < len(obj.points):
                        x, y = obj.points[i]
                    else:
                        x = y = 0.0
                    flag = obj.visibility[i] if i < len(obj.visibility) else 2
                    if flag > 0:
                        visible += 1
                    keypoints += [round(x, 2), round(y, 2), flag]
                item["keypoints"] = keypoints
                item["num_keypoints"] = visible
            else:
                item["segmentation"] = []
            annotations_json.append(item)
            ann_id += 1

    doc = {
        "info": {"description": "TRTForge Web Studio 导出", "version": "1.0"},
        "images": images_json,
        "annotations": annotations_json,
        "categories": [
            {"id": item["id"], "name": item["name"], "supercategory": "object"}
            for item in project.classes
        ],
    }
    target = output / "annotations.json"
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "format": "coco",
        "output": str(output),
        "counts": {"images": len(images_json), "annotations": len(annotations_json), "empty": empty},
        "skipped": skipped,
        "extra_files": [target.name],
        "note": "COCO 只写标注文件，不复制图片；图片路径以 file_name 记录",
    }


# ---------------------------------------------------------------------------
# 导出：VOC
# ---------------------------------------------------------------------------
def _voc_xml(image: DatasetImage, objects: list[DatasetObject], project: DatasetProject, filename: str) -> str:
    lines = [
        "<annotation>",
        "  <folder>JPEGImages</folder>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{image.width}</width>",
        f"    <height>{image.height}</height>",
        "    <depth>3</depth>",
        "  </size>",
    ]
    for obj in objects:
        if not obj.points:
            continue
        x1, y1, x2, y2 = _bounds(obj.points)
        lines += [
            "  <object>",
            f"    <name>{project.class_name(obj.class_id)}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            "    <bndbox>",
            f"      <xmin>{round(x1)}</xmin>",
            f"      <ymin>{round(y1)}</ymin>",
            f"      <xmax>{round(x2)}</xmax>",
            f"      <ymax>{round(y2)}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    lines.append("</annotation>")
    return "\n".join(lines) + "\n"


def export_voc(project: DatasetProject, output: Path, splits: list[str], overwrite: bool) -> dict:
    if project.task != "detect":
        raise DatasetError(
            f"VOC 只支持标准水平矩形检测，当前任务是 {project.task}。"
            "segment / pose / obb 请改用 YOLO 或 COCO 导出。"
        )
    _prepare_output(output, overwrite, project.source)
    (output / "Annotations").mkdir(exist_ok=True)
    (output / "JPEGImages").mkdir(exist_ok=True)

    written = 0
    skipped: list[str] = []
    for image, _split in zip(project.images, splits):
        if not image.path.is_file():
            skipped.append(f"图片不存在：{image.path}")
            continue
        target = _copy_image(image.path, output / "JPEGImages")
        xml = _voc_xml(image, image.objects, project, target.name)
        (output / "Annotations" / f"{target.stem}.xml").write_text(xml, encoding="utf-8")
        written += 1

    return {
        "format": "voc",
        "output": str(output),
        "counts": {"images": written},
        "skipped": skipped,
        "extra_files": ["Annotations/", "JPEGImages/"],
        "note": "VOC 本身没有 train/val 划分，划分比例仅用于统计",
    }


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
FORMATS = ("yolo", "coco", "voc")


def run_export(
    source: str, task: str, train: int, val: int, fmt: str, output: str, overwrite: bool
) -> dict:
    """扫描 → 划分 → 导出，一次请求走完，避免前端保存中间状态。"""
    if fmt not in FORMATS:
        raise DatasetError(f"未知导出格式：{fmt}，可选 {', '.join(FORMATS)}")
    if not output.strip():
        raise DatasetError("请填写输出目录")

    project = load_project(source, task)
    splits = assign_splits(len(project.images), train, val)
    target = Path(output).expanduser()

    if fmt == "yolo":
        result = export_yolo(project, target, splits, overwrite)
    elif fmt == "coco":
        result = export_coco(project, target, splits, overwrite)
    else:
        result = export_voc(project, target, splits, overwrite)

    result.update(
        {
            "task": project.task,
            "source_kind": project.kind,
            "split_counts": split_counts(splits),
            "warnings": project.warnings,
        }
    )
    # 比例向下取整后可能出现空集（例如 3 张图配 33% 的 train 会算成 0 张），
    # 这里显式提示，避免用户拿到「训练集空」的数据集还以为是 bug。
    for name, ratio in (("train", train), ("val", val)):
        if ratio > 0 and result["split_counts"].get(name, 0) == 0:
            result["warnings"].append(
                f"{name} 比例填了 {ratio}%，但样本量向下取整后为 0 张。请提高比例或增加图片。"
            )
    return result
