"""标注工程：`.tfanno` 的读写与规范化。

本模块是 Web 标注工作台的**唯一数据出入口**，字段名、类型、字符串映射逐项对齐
`4-qt-studio/src/annotation/AnnotationProject.cpp`（Qt 版是格式权威实现）。

设计要点（决策依据见 openspec/changes/add-web-annotation-workbench/design.md）：

1. **矩形 2 点制**：Qt 的矩形是 ``points[0]`` 左上、``points[1]`` 右下（对角线）。
   外部标注服务给的是 4 点，读入时在此折叠，写出时永远只有 2 点。
2. **关键点单对象模型**：同一目标的关键点放在一个对象里，靠
   ``keypoint_visibility`` 对齐；绝不拆成多个 ``point`` 对象。
3. **扩展字段只加键不改结构**：``difficult`` / ``flags`` / ``attributes`` 是顶层新增键，
   Qt 的 ``objectFromJson`` 逐键读取，未知键会被静默忽略，因此安全。
4. **修订号存根级 ``rev``**：同样是被 Qt 忽略的新增键，用于乐观锁。

所有形状/集合的字符串映射与 Qt 一致，且对未知值回退（矩形 / train）并记录警告，
不抛异常——宁可让用户看到「这个形状被降级成矩形了」，也不要让整个工程打不开。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量：与 Qt 端逐项对齐
# ---------------------------------------------------------------------------

CURRENT_VERSION = 1

TASKS = ("detect", "segment", "pose", "obb")
SHAPES = ("rectangle", "rotated_rect", "polygon", "line", "point", "keypoints")
SPLITS = ("train", "val", "test")

TASK_ALIASES = {
    "detect": "detect",
    "detection": "detect",
    "segment": "segment",
    "segmentation": "segment",
    "pose": "pose",
    "keypoint": "pose",
    "keypoints": "pose",
    "obb": "obb",
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}

SHAPE_ALIASES = {
    "rectangle": "rectangle",
    "rect": "rectangle",
    "box": "rectangle",
    "rotated_rect": "rotated_rect",
    "rotatedrect": "rotated_rect",
    "rotated_rectangle": "rotated_rect",
    "polygon": "polygon",
    "mask": "polygon",
    "line": "line",
    "point": "point",
    "keypoints": "keypoints",
    "keypoint": "keypoints",
    "pose": "keypoints",
}

#: 每个形状的语义点数。``None`` 表示点数不固定（多边形 / 关键点）。
POINT_ARITY = {
    "rectangle": 2,
    "rotated_rect": 2,
    "polygon": None,
    "line": 2,
    "point": 1,
    "keypoints": None,
}

#: 需要闭合边形的形状。
CLOSED_SHAPES = {"rectangle", "rotated_rect", "polygon"}

#: 扩展字段在对象里的键名 -> 默认值。默认值不落盘，避免文件膨胀与无意义 diff。
EXTENSION_DEFAULTS = {
    "difficult": False,
    "flags": {},
    "attributes": {},
}


class AnnotationError(Exception):
    """标注工程的通用错误（文件读写失败、内容不合法等）。"""


class RevisionConflict(AnnotationError):
    """保存时修订号不一致。

    ``disk`` 携带磁盘上的真实状态，让前端能显示「你基于 rev=3 提交，磁盘已是 rev=5」，
    而不是一句「保存失败」。
    """

    def __init__(self, message: str, reason: str, disk: dict, expected: dict | None = None):
        super().__init__(message)
        self.reason = reason  # "rev-mismatch" / "external-modification"
        self.disk = disk
        self.expected = expected or {}


# ---------------------------------------------------------------------------
# 字符串映射
# ---------------------------------------------------------------------------

def task_to_str(task: str) -> str:
    value = (task or "").strip().lower()
    return value if value in TASKS else "detect"


def task_from_str(text: str, warnings: list[str] | None = None) -> str:
    raw = (text or "").strip().lower()
    resolved = TASK_ALIASES.get(raw)
    if resolved is None:
        if warnings is not None and raw:
            warnings.append(f"未知任务类型「{text}」，已回退为 detect")
        return "detect"
    return resolved


def shape_to_str(shape: str) -> str:
    value = (shape or "").strip().lower()
    return value if value in SHAPES else "rectangle"


def shape_from_str(text: str, warnings: list[str] | None = None, object_id: str = "") -> str:
    raw = (text or "").strip().lower()
    resolved = SHAPE_ALIASES.get(raw)
    if resolved is None:
        if warnings is not None:
            who = f"对象「{object_id}」" if object_id else "标注"
            warnings.append(f"{who}的形状「{text}」不受支持，已回退为矩形")
        return "rectangle"
    return resolved


def split_to_str(split: str) -> str:
    value = (split or "").strip().lower()
    return value if value in SPLITS else "train"


def split_from_str(text: str, warnings: list[str] | None = None) -> str:
    raw = (text or "").strip().lower()
    if not raw:
        return "train"
    resolved = SPLIT_ALIASES.get(raw)
    if resolved is None:
        if warnings is not None:
            warnings.append(f"未知数据集集合「{text}」，已回退为 train")
        return "train"
    return resolved


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def new_class(class_id: int, name: str = "", color: list[int] | None = None) -> dict:
    name = (name or "").strip() or f"class_{class_id}"
    return {"id": int(class_id), "name": name, "color": list(color) if color else default_color(class_id)}


def new_object(
    object_id: str,
    class_id: int = 0,
    label: str = "",
    shape: str = "rectangle",
    points: list | None = None,
) -> dict:
    return {
        "id": str(object_id),
        "class_id": int(class_id),
        "label": label or "",
        "description": "",
        "group_id": 0,
        "shape_type": shape_to_str(shape),
        "points": [[float(p[0]), float(p[1])] for p in (points or [])],
        "rotation": 0.0,
        "keypoint_visibility": [],
        "score": -1.0,
    }


def new_image(
    image_id: str,
    path: str,
    width: int = 0,
    height: int = 0,
    split: str = "train",
    annotations: list | None = None,
) -> dict:
    return {
        "id": str(image_id) or Path(path).stem,
        "path": str(path),
        "width": int(width or 0),
        "height": int(height or 0),
        "split": split_to_str(split),
        "annotations": list(annotations or []),
    }


def new_project(task: str = "detect", classes: list | None = None, images: list | None = None) -> dict:
    return {
        "version": CURRENT_VERSION,
        "task": task_to_str(task),
        "classes": list(classes if classes is not None else [new_class(0, "object")]),
        "images": list(images or []),
        "rev": 0,
    }


# ---------------------------------------------------------------------------
# 配色：与 Qt 的 QColor::fromHsv((seed * 47) % 360, 210, 240) 完全一致
# ---------------------------------------------------------------------------

def default_color(seed: int) -> list[int]:
    """按种子生成配色，与 Qt 侧 ``colorFromJson`` 的兜底公式逐位一致。

    这条公式必须复刻而不能"差不多"：否则同一个工程在 Qt 与 Web 里显示的颜色不同，
    用户会以为是数据被改了。
    """
    return hsv_to_rgb((int(seed) * 47) % 360, 210, 240)


def hsv_to_rgb(h: int, s: int, v: int) -> list[int]:
    """HSV(0-359, 0-255, 0-255) -> RGB(0-255)，按 Qt 的取整规则（四舍五入）。"""
    if s == 0:
        return [v, v, v]
    h = max(0, min(359, int(h)))
    sector = h / 60.0
    index = int(sector) % 6
    fraction = sector - int(sector)
    p = v * (1.0 - s / 255.0)
    q = v * (1.0 - s / 255.0 * fraction)
    t = v * (1.0 - s / 255.0 * (1.0 - fraction))
    table = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    return [int(round(channel)) for channel in table[index]]


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------

def _as_points(raw) -> list[list[float]]:
    """把 ``[[x,y], ...]`` / ``[{"x":..,"y":..}]`` 统一成二维浮点数组。

    容忍：空值、字符串数字、长度不足 2 的对、``null``。任何无法解析的项直接丢弃，
    而不是让整个工程读取失败。
    """
    points: list[list[float]] = []
    if not isinstance(raw, (list, tuple)):
        return points
    for item in raw:
        try:
            if isinstance(item, dict):
                points.append([float(item.get("x", 0.0)), float(item.get("y", 0.0))])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def as_int(value, default: int) -> int:
    """把外部输入转成 int，失败就用 ``default``。

    `.tfanno` 可能来自 Qt、手工编辑或外部标注服务，单个字段类型不对
    不该让整份工程读不出来，所以转换失败在这里统一吞掉。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default: float) -> float:
    """与 :func:`as_int` 同理，用于浮点字段。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fold_rectangle(points: list[list[float]], shape: str) -> list[list[float]]:
    """把 4 个角点折叠成 Qt 的 2 点对角线表示。

    - ``rectangle``：取所有点的包围盒，保证 ``points[0]`` 一定是左上、``points[1]`` 一定是右下。
      这样即使外部服务按「左上 → 右上 → 右下 → 左下」给点，结果也正确。
    - ``rotated_rect``：取 ``points[0]`` 与距它最远的顶点，保留对角线方向，
      旋转角由 ``rotation`` 字段单独承载。
    """
    if len(points) <= 2:
        if len(points) == 1:
            return [points[0], list(points[0])]
        return points
    if shape == "rotated_rect":
        first = points[0]
        farthest = max(points[1:], key=lambda p: (p[0] - first[0]) ** 2 + (p[1] - first[1]) ** 2)
        return [list(first), list(farthest)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


def normalize_object(raw: dict, warnings: list[str] | None = None) -> dict:
    """把任意来源的标注对象规范成 Qt 兼容结构。

    做了四件事：形状回退、点数对齐、矩形折叠、扩展字段剥离默认值。
    """
    if not isinstance(raw, dict):
        raise AnnotationError("标注对象必须是 JSON 对象")

    object_id = str(raw.get("id") or "").strip()
    shape = shape_from_str(str(raw.get("shape_type") or raw.get("shape") or "rectangle"), warnings, object_id)

    points = _as_points(raw.get("points"))
    arity = POINT_ARITY[shape]
    if arity is not None:
        if shape in ("rectangle", "rotated_rect"):
            points = _fold_rectangle(points, shape)
        elif len(points) > arity:
            points = points[:arity]
        elif len(points) < arity and warnings is not None:
            warnings.append(f"标注「{object_id}」的 {shape} 点数不足（{len(points)} < {arity}）")
    elif shape == "polygon" and len(points) < 3 and warnings is not None:
        warnings.append(f"多边形「{object_id}」只有 {len(points)} 个点，无法围成面积")

    visibility = []
    for value in raw.get("keypoint_visibility") or []:
        try:
            visibility.append(int(value))
        except (TypeError, ValueError):
            visibility.append(2)  # Qt 的缺失默认值
    if shape == "keypoints" and visibility:
        # 可见性数组与点数必须一一对应，否则 Qt 侧索引会错位。
        if len(visibility) < len(points):
            visibility.extend([2] * (len(points) - len(visibility)))
        elif len(visibility) > len(points):
            visibility = visibility[: len(points)]

    obj = {
        "id": object_id,
        "class_id": as_int(raw.get("class_id"), 0),
        "label": str(raw.get("label") or ""),
        "description": str(raw.get("description") or ""),
        "group_id": as_int(raw.get("group_id"), 0),
        "shape_type": shape,
        "points": points,
        "rotation": as_float(raw.get("rotation"), 0.0),
        "keypoint_visibility": visibility,
        "score": as_float(raw.get("score", -1.0), -1.0),
    }
    # 扩展字段：只在显式出现时保留，默认值丢弃（不落盘）。
    for key in ("difficult", "flags", "attributes"):
        if key in raw and raw[key] is not None:
            value = raw[key]
            if key == "difficult":
                obj[key] = bool(value)
            elif isinstance(value, dict) and value:
                obj[key] = {str(k): v for k, v in value.items()}
    return obj


def normalize_image(raw: dict, warnings: list[str] | None = None) -> dict:
    if not isinstance(raw, dict):
        raise AnnotationError("图片条目必须是 JSON 对象")

    path = str(raw.get("path") or "")
    image_id = str(raw.get("id") or "").strip() or Path(path).stem
    annotations = []
    for item in raw.get("annotations") or []:
        try:
            annotations.append(normalize_object(item, warnings))
        except AnnotationError as exc:
            if warnings is not None:
                warnings.append(f"图片「{image_id}」有一条标注被跳过：{exc}")
    return new_image(
        image_id,
        path,
        as_int(raw.get("width"), 0),
        as_int(raw.get("height"), 0),
        split_from_str(str(raw.get("split") or "train"), warnings),
        annotations,
    )


def normalize_project(raw: dict) -> tuple[dict, list[str]]:
    """把任意来源的工程规范成 Qt 兼容结构，返回 (工程, 警告列表)。"""
    if not isinstance(raw, dict):
        raise AnnotationError("工程内容必须是 JSON 对象")

    warnings: list[str] = []
    task = task_from_str(str(raw.get("task") or "detect"), warnings)

    classes = []
    for index, item in enumerate(raw.get("classes") or []):
        if not isinstance(item, dict):
            continue
        class_id = as_int(item.get("id"), index)
        color = item.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            try:
                color = [int(color[0]), int(color[1]), int(color[2])]
            except (TypeError, ValueError):
                color = None
        else:
            color = None
        classes.append(new_class(class_id, str(item.get("name") or ""), color))
    if not classes:
        # 与 Qt 的 ensureDefaultClass() 行为一致。
        classes = [new_class(0, "object")]

    images = []
    for item in raw.get("images") or []:
        try:
            images.append(normalize_image(item, warnings))
        except AnnotationError as exc:
            warnings.append(f"一条图片记录被跳过：{exc}")

    version = as_int(raw.get("version", CURRENT_VERSION), CURRENT_VERSION)
    rev = max(0, as_int(raw.get("rev"), 0))

    project = {
        "version": version if version > 0 else CURRENT_VERSION,
        "task": task,
        "classes": classes,
        "images": images,
        "rev": rev,
    }
    return project, warnings


# ---------------------------------------------------------------------------
# 序列化：写出时严格按 Qt 的键顺序与"默认值不落盘"规则
# ---------------------------------------------------------------------------

def object_to_json(obj: dict) -> dict:
    out = {
        "id": obj.get("id", ""),
        "class_id": int(obj.get("class_id", 0)),
        "label": obj.get("label", ""),
        "description": obj.get("description", ""),
        "group_id": int(obj.get("group_id", 0)),
        "shape_type": shape_to_str(obj.get("shape_type", "rectangle")),
        "points": [[float(p[0]), float(p[1])] for p in obj.get("points") or []],
        "rotation": float(obj.get("rotation", 0.0)),
    }
    score = float(obj.get("score", -1.0))
    if score >= 0.0:
        out["score"] = score
    out["keypoint_visibility"] = [int(v) for v in obj.get("keypoint_visibility") or []]
    # 扩展字段：默认值不落盘。
    if obj.get("difficult"):
        out["difficult"] = True
    for key in ("flags", "attributes"):
        value = obj.get(key)
        if isinstance(value, dict) and value:
            out[key] = dict(value)
    return out


def image_to_json(image: dict) -> dict:
    return {
        "id": image.get("id", ""),
        "path": image.get("path", ""),
        "width": int(image.get("width", 0)),
        "height": int(image.get("height", 0)),
        "split": split_to_str(image.get("split", "train")),
        "annotations": [object_to_json(obj) for obj in image.get("annotations") or []],
    }


def project_to_json(project: dict) -> dict:
    return {
        "version": int(project.get("version", CURRENT_VERSION)),
        "task": task_to_str(project.get("task", "detect")),
        "rev": max(0, int(project.get("rev", 0))),
        "classes": [
            {
                "id": int(item.get("id", index)),
                "name": str(item.get("name", "")),
                "color": [int(c) for c in (item.get("color") or default_color(int(item.get("id", index))))][:3],
            }
            for index, item in enumerate(project.get("classes") or [])
        ],
        "images": [image_to_json(image) for image in project.get("images") or []],
    }


def dumps(project: dict) -> str:
    """序列化成文本。``ensure_ascii=False`` 让中文类别名保持可读，
    Qt 用 UTF-8 读写，两边一致。"""
    return json.dumps(project_to_json(project), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

#: 纳秒时间戳约 1.79e18，超过 JavaScript ``Number.MAX_SAFE_INTEGER``（2^53-1）。
#: 浏览器 JSON.parse 会把末几位四舍五入，原样回传后看起来就像「文件被外部改过」。
#: 因此对外一律用**字符串**传递 ``mtime_ns``；万一收到浮点数，只做容差比较。
STAMP_TOLERANCE_NS = 4096


def _stamp_ns(value) -> tuple[int, bool] | None:
    """把客户端回传的 mtime 解析成 ``(纳秒, 是否精确)``；无法解析返回 ``None``。

    只有字符串才算精确值——JSON 里的整数可能已经被浏览器按 float64 舍入过，
    无法分辨真伪，因此按容差处理（见 :data:`STAMP_TOLERANCE_NS`）。
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value, False
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text), True
    except ValueError:
        pass
    try:
        return int(float(text)), False
    except (TypeError, ValueError):
        return None


def read_stamp(path: str | Path) -> dict:
    """读取磁盘上的修订号与文件指纹（mtime / size）。

    文件不存在时返回 ``rev=-1``，让调用方能区分「新建」与「已存在 rev=0」。

    ``mtime_ns`` 以字符串返回，避免经过浏览器 float64 丢精度（见 STAMP_TOLERANCE_NS）。
    """
    target = Path(path)
    if not target.is_file():
        return {"rev": -1, "mtime_ns": "0", "size": 0, "exists": False}
    stat = target.stat()
    # 读不出 rev 就当 0：这个值只用于乐观锁，不该让「看看文件状态」失败。
    rev = 0
    try:
        raw = json.loads(target.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, json.JSONDecodeError):
        raw = None
    if isinstance(raw, dict):
        rev = max(0, as_int(raw.get("rev"), 0))
    return {"rev": rev, "mtime_ns": str(stat.st_mtime_ns), "size": stat.st_size, "exists": True}


def load(path: str | Path) -> tuple[dict, dict]:
    """读取工程文件。

    返回 ``(工程, 元信息)``。元信息含 ``rev`` / ``mtime_ns`` / ``size``，
    前端保存时把它们原样回传，用于乐观锁与「文件被外部修改」检测。
    """
    target = Path(path)
    if not target.is_file():
        raise AnnotationError(f"工程文件不存在：{path}")

    # utf-8-sig：Qt 在 Windows 上写文件时可能带 BOM，直接 utf-8 会解析失败。
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AnnotationError(f"读取工程失败：{exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"工程文件不是合法 JSON：{exc}") from exc

    project, warnings = normalize_project(raw)
    stat = target.stat()
    meta = {
        "rev": project["rev"],
        # 字符串：纳秒时间戳超出 JS 安全整数范围，用数字会被浏览器四舍五入。
        "mtime_ns": str(stat.st_mtime_ns),
        "size": stat.st_size,
        "warnings": warnings,
    }
    return project, meta


_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _lock_for(target: Path) -> threading.Lock:
    """按规范化后的绝对路径取一把写锁。

    uvicorn 的同步路由跑在线程池里，同一个工程可能被两次请求同时保存。
    「读 rev → 校验 → 写回 rev+1」这段必须串行，否则两次保存都能通过校验，
    后写的直接抹掉前一次的内容（丢更新）。
    """
    key = os.path.abspath(str(target))
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[key] = lock
        return lock


def _atomic_write(target: Path, text: str) -> None:
    """先写临时文件再替换。

    直接覆盖写在中途失败（磁盘满、进程被杀）时会留下半截 JSON，
    用户的工程就废了。原子替换把损失限制在「没保存成功」。

    临时文件名带 uuid：只用 pid 的话，同进程内的两个请求会写同一个临时文件，
    互相把对方的半截内容 replace 成正式文件。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + f".tmp{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, target)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def save(
    path: str | Path,
    project: dict,
    expected_rev: int | None = None,
    expected_stamp: dict | None = None,
    force: bool = False,
) -> dict:
    """保存工程，成功后把 ``rev`` 加一并写回文件。

    - ``expected_rev``：客户端读到的修订号。``None`` 表示不校验（新建文件或强制覆盖）。
    - ``expected_stamp``：客户端读到的文件指纹，用于发现「有人用 Qt/记事本改过文件」。
    - ``force=True``：跳过全部校验（用户在冲突弹窗里选了「强制覆盖」）。

    冲突时抛 :class:`RevisionConflict`，**不写盘**。
    """
    target = Path(path)
    with _lock_for(target):
        return _save_locked(target, project, expected_rev, expected_stamp, force)


def _save_locked(
    target: Path,
    project: dict,
    expected_rev: int | None,
    expected_stamp: dict | None,
    force: bool,
) -> dict:
    """``save`` 的实现体，调用方**必须**已持有该路径的写锁。"""
    disk = read_stamp(target)
    normalized, warnings = normalize_project(project)

    if not force:
        if expected_rev is not None and expected_rev != disk["rev"]:
            raise RevisionConflict(
                f"工程已被他人修改（你基于 rev={expected_rev}，磁盘是 rev={disk['rev']}），保存已取消",
                "rev-mismatch",
                disk,
                {"rev": expected_rev},
            )
        if expected_stamp:
            same_rev = expected_rev is None or expected_rev == disk["rev"]
            parsed = _stamp_ns(expected_stamp.get("mtime_ns") or expected_stamp.get("mtime"))
            if same_rev and disk["exists"] and parsed is not None:
                client_ns, exact = parsed
                drift = abs(client_ns - int(disk["mtime_ns"]))
                size_mismatch = as_int(expected_stamp.get("size"), 0) != disk["size"]
                # 精确值必须完全一致；浮点数最多容忍 STAMP_TOLERANCE_NS 的末位舍入。
                if size_mismatch or drift > (0 if exact else STAMP_TOLERANCE_NS):
                    # 修订号没变但文件指纹变了：说明有人绕过本接口改了文件。
                    raise RevisionConflict(
                        "工程文件已被外部程序修改（修订号未变但文件内容变了），请重新加载后再保存",
                        "external-modification",
                        disk,
                        expected_stamp,
                    )

    normalized["rev"] = max(0, disk["rev"]) + 1 if disk["exists"] else 1
    _atomic_write(target, dumps(normalized))

    stat = target.stat()
    return {
        "rev": normalized["rev"],
        "mtime_ns": str(stat.st_mtime_ns),
        "size": stat.st_size,
        "warnings": warnings,
    }


def touch(path: str | Path, project: dict) -> dict:
    """按当前内容重写一次（用于导出前把内存状态落成临时文件）。不动 rev 语义。"""
    target = Path(path)
    normalized, warnings = normalize_project(project)
    # 与 save 共用同一把锁：导出前重写和用户手动保存撞在一起同样会写出半截文件。
    with _lock_for(target):
        normalized["rev"] = max(0, read_stamp(target)["rev"])
        _atomic_write(target, dumps(normalized))
    return {"rev": normalized["rev"], "warnings": warnings}


# ---------------------------------------------------------------------------
# 图片解析
# ---------------------------------------------------------------------------

def resolve_image_path(project_path: str | Path, image_path: str) -> Path:
    """解析图片的真实路径。

    `.tfanno` 里存的是相对工程目录的路径（Qt 的写法），也兼容绝对路径。
    中文与空格不需要特殊处理——``pathlib`` 在 Windows 上按 Unicode 处理，
    真正的坑是编码：历史工程可能是 GBK 存的，那种情况在读取时会替换成 U+FFFD，
    此时回退到「按顺序尝试常见编码」再解析一遍。
    """
    raw = str(image_path or "")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    base = Path(project_path).parent
    direct = base / raw
    if direct.exists():
        return direct
    # 路径里可能带 ``..`` 或正反斜杠混用，normalize 后再试一次。
    normalized = Path(os.path.normpath(str(direct)))
    if normalized.exists():
        return normalized
    return direct


def to_dataset_project(project: dict, source: str | Path | None = None):
    """把内存中的标注工程转成 ``dataset.DatasetProject``，供既有导出实现复用。

    直接构造 ``DatasetProject`` 而不是写临时文件再走 ``load_tfanno``，
    是因为后者会丢掉内存中尚未落盘的编辑——而导出必须与屏幕一致。

    ``source`` 传工程文件路径，导出层用它做「输出目录不能放在数据源内部」的防呆；
    不传时给一个不存在的占位路径，避免误判。
    """
    from dataset import DatasetImage, DatasetObject, DatasetProject, _image_size

    images = []
    warnings: list[str] = []
    for index, item in enumerate(project.get("images") or []):
        path = Path(str(item.get("path") or ""))
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = _image_size(path)
            if width <= 0 or height <= 0:
                warnings.append(f"读不到图片尺寸，导出 COCO/VOC 时会被跳过：{path}")
        objects = []
        for obj in item.get("annotations") or []:
            points = [(float(p[0]), float(p[1])) for p in obj.get("points") or []]
            visibility = [int(v) for v in obj.get("keypoint_visibility") or []]
            objects.append(DatasetObject(int(obj.get("class_id", 0)), points, visibility))
        images.append(
            DatasetImage(
                index=index,
                path=path,
                rel=str(path).replace("\\", "/"),
                width=width,
                height=height,
                objects=objects,
                current_split=split_to_str(item.get("split", "train")),
            )
        )

    return DatasetProject(
        source=Path(source) if source else Path("__tfanno_in_memory__"),
        kind="tfanno",
        task=task_to_str(project.get("task", "detect")),
        classes=[{"id": int(c.get("id", 0)), "name": str(c.get("name", ""))} for c in project.get("classes") or []],
        images=images,
        warnings=warnings,
    )


def validate(project: dict) -> list[str]:
    """导出前的一致性校验：只报告"会让导出结果变错"的问题，不纠结样式。"""
    issues: list[str] = []
    class_ids = {int(item.get("id", -1)) for item in project.get("classes") or []}

    # 两类别 id 必须分开算：图片 id 是全局的，标注 id 只需在单张图内唯一
    # （Qt 端和历史工程都是从 1 开始、逐图重置）。共用一个集合会把正常工程误判成重复。
    seen_image_ids: set[str] = set()
    for image in project.get("images") or []:
        image_id = str(image.get("id") or "")
        if not image_id:
            issues.append("有图片缺少 id")
        elif image_id in seen_image_ids:
            issues.append(f"图片 id 重复：{image_id}")
        seen_image_ids.add(image_id)

        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        seen_object_ids: set[str] = set()
        for obj in image.get("annotations") or []:
            object_id = str(obj.get("id") or "")
            if object_id in seen_object_ids:
                issues.append(f"标注 id 重复：{image_id} / {object_id}")
            seen_object_ids.add(object_id)

            if int(obj.get("class_id", 0)) not in class_ids:
                issues.append(f"标注「{object_id}」引用了不存在的类别 {obj.get('class_id')}")

            points = obj.get("points") or []
            shape = str(obj.get("shape_type") or "")
            if shape in CLOSED_SHAPES and len(points) < 2:
                issues.append(f"标注「{object_id}」的 {shape} 点数不足")
            if shape == "polygon" and len(points) < 3:
                issues.append(f"标注「{object_id}」的多边形不足 3 点")
            if shape == "keypoints" and not points:
                issues.append(f"标注「{object_id}」没有任何关键点")

            if width > 0 and height > 0:
                outside = [
                    p
                    for p in points
                    if float(p[0]) < -1 or float(p[1]) < -1 or float(p[0]) > width + 1 or float(p[1]) > height + 1
                ]
                if outside:
                    issues.append(f"标注「{object_id}」有 {len(outside)} 个点落在图像范围外（{width}x{height}）")
    return issues


def stats(project: dict) -> dict:
    """给前端图片列表用的汇总：每图标注数与全局计数。"""
    images = project.get("images") or []
    per_image = []
    total = 0
    for image in images:
        count = len(image.get("annotations") or [])
        total += count
        per_image.append(
            {
                "id": image.get("id", ""),
                "path": image.get("path", ""),
                "width": int(image.get("width") or 0),
                "height": int(image.get("height") or 0),
                "split": split_to_str(image.get("split", "train")),
                "count": count,
            }
        )
    counts = {name: sum(1 for item in per_image if item["split"] == name) for name in SPLITS}
    return {
        "images": per_image,
        "total_images": len(per_image),
        "total_objects": total,
        "split_counts": counts,
        "class_counts": {
            int(item.get("id", 0)): sum(
                1
                for image in images
                for obj in image.get("annotations") or []
                if int(obj.get("class_id", 0)) == int(item.get("id", 0))
            )
            for item in project.get("classes") or []
        },
        "now": int(time.time()),
    }
