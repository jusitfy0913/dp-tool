# -*- coding: utf-8 -*-
"""标注工程数据层与接口验收脚本（不需要 GPU，也不需要手动启动 Web 服务）。

覆盖变更 3 的七类风险点：
1. `.tfanno` 往返一致：读 → 写 → 再读，字段逐项相等；
2. Qt 兼容：按 `AnnotationProject.cpp` 的读取逻辑模拟一遍，确认字段名与类型都对得上，
   且新增键（rev / difficult / flags / attributes）不会影响 Qt 解析；
3. 扩展字段：默认值不落盘、非默认值落盘；
4. 矩形 4 点折叠为 2 点，且顺序一定是「左上 → 右下」；
5. 未知形状与未知集合回退，并给出警告而不是抛错；
6. 修订号冲突：旧修订号被拒且磁盘内容一字未改；
7. 导出复用：导出 YOLO 再读回来，标注数量与类别一致。

用法：
    python 6-web/backend/tools/test_annotation_project.py

设计说明（为什么用"Qt 模拟器"而不是真编译 Qt 程序）：
    真跑 Qt 需要 1~2 分钟编译且依赖 Qt 运行库，放不进每次都要跑的验收脚本。
    这里把 `AnnotationProject.cpp` 的 `load()` / `objectFromJson()` 逐键搬成 Python，
    只读取 Qt 真实会读的键、类型不符时按 Qt 的 `toInt()/toString()` 语义退化。
    这验证的是"契约"——键名、类型、缺省行为——正是跨语言兼容唯一会出错的地方。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import annotation  # noqa: E402
import autolabel  # noqa: E402
import dataset  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def close(actual, expected, tol: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Qt 读取端模拟（逐键对齐 AnnotationProject.cpp）
# ---------------------------------------------------------------------------
def qt_load(path: Path) -> dict:
    """按 Qt 的 `AnnotationProject::load()` 逻辑解析工程。

    与 Qt 一致的三条行为，正是兼容性的判断依据：
    - 只读已知键，**未知键一律忽略**（所以扩展字段安全）；
    - `shapeFromString` 未知值回落 rectangle，`splitFromString` 回落 train；
    - `score` 缺失时取 -1.0。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    def shape_from(text: str) -> str:
        value = str(text or "").strip().lower()
        return value if value in ("rectangle", "rotated_rect", "polygon", "line", "point", "keypoints") else "rectangle"

    def split_from(text: str) -> str:
        value = str(text or "").strip().lower()
        if value in ("val", "valid", "validation"):
            return "val"
        if value == "test":
            return "test"
        return "train"

    def points_from(arr) -> list[list[float]]:
        out = []
        for pair in arr or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out.append([float(pair[0]), float(pair[1])])
        return out

    classes = []
    for index, item in enumerate(raw.get("classes") or []):
        class_id = int(item.get("id", index))
        color = item.get("color")
        classes.append(
            {
                "id": class_id,
                "name": str(item.get("name", f"class_{class_id}")),
                "color": [int(c) for c in color[:3]] if isinstance(color, list) and len(color) >= 3 else None,
            }
        )
    if not classes:
        classes = [{"id": 0, "name": "object", "color": None}]

    images = []
    for item in raw.get("images") or []:
        objects = []
        for obj in item.get("annotations") or []:
            objects.append(
                {
                    "id": str(obj.get("id", "")),
                    "class_id": int(obj.get("class_id", 0)),
                    "label": str(obj.get("label", "")),
                    "description": str(obj.get("description", "")),
                    "group_id": int(obj.get("group_id", 0)),
                    "shape_type": shape_from(obj.get("shape_type", "")),
                    "points": points_from(obj.get("points")),
                    "rotation": float(obj.get("rotation", 0.0)),
                    "score": float(obj["score"]) if "score" in obj else -1.0,
                    "keypoint_visibility": [int(v) for v in obj.get("keypoint_visibility") or []],
                }
            )
        images.append(
            {
                "id": str(item.get("id") or Path(str(item.get("path", ""))).stem),
                "path": str(item.get("path", "")),
                "width": int(item.get("width", 0)),
                "height": int(item.get("height", 0)),
                "split": split_from(item.get("split", "")),
                "annotations": objects,
            }
        )

    return {
        "version": int(raw.get("version", 1)),
        "task": str(raw.get("task", "detect")),
        "classes": classes,
        "images": images,
    }


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def make_image(path: Path, width: int = 320, height: int = 240) -> None:
    """生成一张纯色 PNG，避免测试依赖仓库里的真实数据。"""
    import numpy as np

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(canvas, (10, 10), (100, 80), (255, 120, 30), -1)
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok, "测试图片编码失败"
    buffer.tofile(str(path))


def build_project(root: Path) -> dict:
    """构造一份覆盖全部形状与扩展字段的工程。"""
    classes = [
        annotation.new_class(0, "missing_hole", [31, 182, 255]),
        annotation.new_class(1, "mouse_bite", [255, 200, 60]),
        annotation.new_class(2, "spur", [120, 255, 160]),
    ]
    images = [
        annotation.new_image(
            "img_a",
            str(root / "images" / "a.png"),
            320,
            240,
            "train",
            [
                annotation.new_object("o1", 0, "missing_hole", "rectangle", [[10, 20], [110, 90]]),
                annotation.new_object("o2", 1, "mouse_bite", "polygon", [[5, 5], [60, 5], [60, 50], [5, 50]]),
                annotation.new_object("o3", 2, "spur", "line", [[1, 1], [50, 60]]),
                annotation.new_object("o4", 0, "missing_hole", "point", [[200, 200]]),
            ],
        ),
        annotation.new_image(
            "img_b",
            str(root / "images" / "b.png"),
            320,
            240,
            "val",
            [
                annotation.new_object(
                    "o5", 1, "mouse_bite", "keypoints", [[10, 10], [30, 40], [70, 20]]
                ),
                annotation.new_object("o6", 2, "spur", "rotated_rect", [[100, 100], [180, 140]]),
            ],
        ),
    ]
    images[1]["annotations"][0]["keypoint_visibility"] = [2, 1, 0]
    images[1]["annotations"][1]["rotation"] = 0.35
    project = annotation.new_project("detect", classes, images)
    return project


# ---------------------------------------------------------------------------
# 1. 往返一致
# ---------------------------------------------------------------------------
def test_roundtrip(tmp: Path) -> None:
    root = tmp / "proj"
    make_image(root / "images" / "a.png")
    make_image(root / "images" / "b.png")
    path = root / "demo.tfanno"

    source = build_project(root)
    meta = annotation.save(path, source, expected_rev=None, force=True)
    check("往返：首次保存 rev 从 1 开始", meta["rev"] == 1, str(meta))

    loaded, read_meta = annotation.load(path)
    check("往返：读回修订号一致", read_meta["rev"] == 1, str(read_meta))
    check("往返：版本号保留", loaded["version"] == 1, str(loaded.get("version")))
    check("往返：任务类型保留", loaded["task"] == "detect", str(loaded.get("task")))
    check("往返：类别数一致", len(loaded["classes"]) == 3, str(len(loaded["classes"])))
    check(
        "往返：类别名与配色一致",
        [(c["id"], c["name"], c["color"]) for c in loaded["classes"]]
        == [(c["id"], c["name"], c["color"]) for c in source["classes"]],
        str(loaded["classes"]),
    )
    check("往返：图片数一致", len(loaded["images"]) == 2, str(len(loaded["images"])))
    check(
        "往返：集合划分一致",
        [i["split"] for i in loaded["images"]] == ["train", "val"],
        str([i["split"] for i in loaded["images"]]),
    )

    flat_src = [(o["id"], o["class_id"], o["shape_type"], o["points"]) for img in source["images"] for o in img["annotations"]]
    flat_out = [(o["id"], o["class_id"], o["shape_type"], o["points"]) for img in loaded["images"] for o in img["annotations"]]
    check("往返：六个对象的 id/类别/形状/顶点逐项相等", flat_src == flat_out, f"{flat_src} != {flat_out}")

    kp = [o for img in loaded["images"] for o in img["annotations"] if o["shape_type"] == "keypoints"][0]
    check("往返：关键点可见性数组保留", kp["keypoint_visibility"] == [2, 1, 0], str(kp["keypoint_visibility"]))
    check("往返：关键点仍是单对象", len(kp["points"]) == 3, str(len(kp["points"])))

    rot = [o for img in loaded["images"] for o in img["annotations"] if o["shape_type"] == "rotated_rect"][0]
    check("往返：旋转角保留", close(rot["rotation"], 0.35), str(rot["rotation"]))

    # 二次保存必须再推进修订号，且内容不变
    meta2 = annotation.save(path, loaded, expected_rev=1, force=False)
    check("往返：二次保存 rev 递增", meta2["rev"] == 2, str(meta2))
    again, _ = annotation.load(path)
    flat_again = [(o["id"], o["points"]) for img in again["images"] for o in img["annotations"]]
    check(
        "往返：二次保存内容不变",
        flat_again == [(o["id"], o["points"]) for img in loaded["images"] for o in img["annotations"]],
    )


# ---------------------------------------------------------------------------
# 2. Qt 兼容
# ---------------------------------------------------------------------------
def test_qt_compat(tmp: Path) -> None:
    root = tmp / "qt"
    make_image(root / "images" / "a.png")
    path = root / "from_qt.tfanno"

    # 完全按 Qt 的写出格式手写一份（无 rev、无扩展字段、score 省略）
    qt_raw = {
        "version": 1,
        "task": "segment",
        "classes": [
            {"id": 0, "name": "缺陷", "color": [31, 182, 255]},
            {"id": 1, "name": "划痕", "color": [255, 200, 60]},
        ],
        "images": [
            {
                "id": "a",
                "path": "images/a.png",
                "width": 320,
                "height": 240,
                "split": "train",
                "annotations": [
                    {
                        "id": "obj_1",
                        "class_id": 1,
                        "label": "划痕",
                        "description": "右上角",
                        "group_id": 3,
                        "shape_type": "polygon",
                        "points": [[10, 20], [80, 20], [80, 70]],
                        "rotation": 0.0,
                        "keypoint_visibility": [],
                    }
                ],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qt_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded, meta = annotation.load(path)
    check("Qt：能读取 Qt 写出的文件", len(loaded["images"]) == 1, str(meta))
    check("Qt：任务类型 segment 保留", loaded["task"] == "segment", str(loaded["task"]))
    check("Qt：中文类别名正确解码", loaded["classes"][0]["name"] == "缺陷", str(loaded["classes"]))
    check("Qt：缺少 rev 时按 0 处理", meta["rev"] == 0, str(meta["rev"]))
    obj = loaded["images"][0]["annotations"][0]
    check("Qt：多边形顶点数保留", len(obj["points"]) == 3, str(obj["points"]))
    check("Qt：描述字段保留", obj["description"] == "右上角", obj["description"])
    check("Qt：分组号保留", obj["group_id"] == 3, str(obj["group_id"]))
    check("Qt：缺少 score 时取 -1", close(obj["score"], -1.0), str(obj["score"]))

    # 写入后让"Qt 模拟器"再读一遍，验证键名与类型都合规
    annotation.save(path, loaded, expected_rev=0)
    replayed = qt_load(path)
    check("Qt：写回后 Qt 能解析出同样的类别", [c["name"] for c in replayed["classes"]] == ["缺陷", "划痕"], str(replayed["classes"]))
    check("Qt：写回后 Qt 能解析出同样的任务", replayed["task"] == "segment", replayed["task"])
    replayed_obj = replayed["images"][0]["annotations"][0]
    check("Qt：写回后描述字段仍正确", replayed_obj["description"] == "右上角", str(replayed_obj))
    check("Qt：写回后形状仍是 polygon", replayed_obj["shape_type"] == "polygon", replayed_obj["shape_type"])

    # 新增键（rev / 扩展字段）不应影响 Qt 解析
    loaded["images"][0]["annotations"][0]["difficult"] = True
    loaded["images"][0]["annotations"][0]["attributes"] = {"批次": "A1"}
    annotation.save(path, loaded, expected_rev=1)
    replayed2 = qt_load(path)
    check("Qt：新增键不影响 Qt 解析（值仍一致）", replayed2["images"][0]["annotations"][0]["label"] == "划痕")
    check(
        "Qt：Qt 侧看不到扩展键（逐键读取）",
        "difficult" not in replayed2["images"][0]["annotations"][0],
        str(replayed2["images"][0]["annotations"][0].keys()),
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    check("Qt：扩展键确实写进了文件", raw["images"][0]["annotations"][0].get("difficult") is True)
    check("Qt：rev 确实写进了文件根级", isinstance(raw.get("rev"), int) and raw["rev"] >= 1, str(raw.get("rev")))


# ---------------------------------------------------------------------------
# 3. 扩展字段默认值不落盘
# ---------------------------------------------------------------------------
def test_extensions(tmp: Path) -> None:
    root = tmp / "ext"
    path = root / "ext.tfanno"
    project = annotation.new_project(
        "detect",
        [annotation.new_class(0, "object")],
        [
            annotation.new_image(
                "i",
                str(root / "i.png"),
                100,
                100,
                "train",
                [
                    annotation.new_object("plain", 0, "", "rectangle", [[0, 0], [10, 10]]),
                    {**annotation.new_object("rich", 0, "", "rectangle", [[0, 0], [20, 20]]),
                     "difficult": True, "flags": {"occluded": True}, "attributes": {"颜色": "蓝"}},
                ],
            )
        ],
    )
    annotation.save(path, project, force=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    objects = {o["id"]: o for o in raw["images"][0]["annotations"]}

    check("扩展：未设置时不写 difficult", "difficult" not in objects["plain"], str(objects["plain"]))
    check("扩展：未设置时不写 flags", "flags" not in objects["plain"], str(objects["plain"]))
    check("扩展：未设置时不写 attributes", "attributes" not in objects["plain"], str(objects["plain"]))
    check("扩展：显式 True 时落盘", objects["rich"].get("difficult") is True)
    check("扩展：flags 落盘", objects["rich"].get("flags") == {"occluded": True}, str(objects["rich"].get("flags")))
    check("扩展：attributes 落盘（中文键）", objects["rich"].get("attributes") == {"颜色": "蓝"}, str(objects["rich"].get("attributes")))

    # 空字典也应视为默认值
    project["images"][0]["annotations"][0]["flags"] = {}
    project["images"][0]["annotations"][0]["attributes"] = {}
    project["images"][0]["annotations"][0]["difficult"] = False
    annotation.save(path, project, force=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    plain = {o["id"]: o for o in raw["images"][0]["annotations"]}["plain"]
    check("扩展：空字典与 False 不落盘",
          "flags" not in plain and "attributes" not in plain and "difficult" not in plain, str(plain))


# ---------------------------------------------------------------------------
# 4. 矩形 4 点折叠
# ---------------------------------------------------------------------------
def test_rectangle_fold() -> None:
    # 外部服务常见顺序：左上 → 右上 → 右下 → 左下
    obj = annotation.normalize_object(
        {"id": "r1", "shape_type": "rectangle", "points": [[10, 20], [110, 20], [110, 90], [10, 90]]}
    )
    check("折叠：4 点变 2 点", len(obj["points"]) == 2, str(obj["points"]))
    check("折叠：第 0 点是左上", obj["points"][0] == [10.0, 20.0], str(obj["points"]))
    check("折叠：第 1 点是右下", obj["points"][1] == [110.0, 90.0], str(obj["points"]))

    # 乱序输入也应收敛到左上/右下（包围盒）
    obj2 = annotation.normalize_object(
        {"id": "r2", "shape_type": "rectangle", "points": [[110, 90], [10, 90], [10, 20], [110, 20]]}
    )
    check("折叠：乱序输入仍得到左上/右下", obj2["points"] == [[10.0, 20.0], [110.0, 90.0]], str(obj2["points"]))

    # 已经是 2 点时不改动
    obj3 = annotation.normalize_object({"id": "r3", "shape_type": "rectangle", "points": [[1, 2], [3, 4]]})
    check("折叠：2 点原样保留", obj3["points"] == [[1.0, 2.0], [3.0, 4.0]], str(obj3["points"]))

    rot = annotation.normalize_object(
        {"id": "r4", "shape_type": "rotated_rect", "rotation": 0.5,
         "points": [[0, 0], [100, 0], [100, 40], [0, 40]]}
    )
    check("折叠：旋转矩形保留对角线", len(rot["points"]) == 2 and rot["points"][1] == [100.0, 40.0], str(rot["points"]))
    check("折叠：旋转矩形的 rotation 保留", close(rot["rotation"], 0.5), str(rot["rotation"]))


# ---------------------------------------------------------------------------
# 5. 未知值回退
# ---------------------------------------------------------------------------
def test_fallbacks() -> None:
    warnings: list[str] = []
    obj = annotation.normalize_object(
        {"id": "u1", "shape_type": "squiggle", "points": [[0, 0], [10, 10]]}, warnings
    )
    check("回退：未知形状降级为矩形", obj["shape_type"] == "rectangle", obj["shape_type"])
    check("回退：未知形状有警告", any("squiggle" in w for w in warnings), str(warnings))

    warnings2: list[str] = []
    check("回退：未知集合降级为 train", annotation.split_from_str("holdout", warnings2) == "train")
    check("回退：未知集合有警告", any("holdout" in w for w in warnings2), str(warnings2))
    check("回退：val 别名 valid 被识别", annotation.split_from_str("valid") == "val")
    check("回退：空集合默认 train", annotation.split_from_str("") == "train")
    check("回退：未知任务降级为 detect", annotation.task_from_str("nope") == "detect")
    check("分类任务保持 classify", annotation.task_from_str("classify") == "classify")

    # 整份工程读取时也不能因为一条坏数据就失败
    project, warns = annotation.normalize_project(
        {
            "task": "mystery",
            "classes": [],
            "images": [
                {"path": "x.png", "split": "weird", "annotations": [{"id": "z", "shape_type": "blob", "points": []}]}
            ],
        }
    )
    check("回退：整份工程仍可解析", project["task"] == "detect" and len(project["images"]) == 1, str(project["task"]))
    check("回退：没有类别表时补默认 object", project["classes"][0]["name"] == "object", str(project["classes"]))
    check("回退：坏集合被纠正为 train", project["images"][0]["split"] == "train", project["images"][0]["split"])
    check("回退：警告里记录了问题", bool(warns), str(warns))

    # 配对：可见性数组长度与点数对齐
    kp = annotation.normalize_object(
        {"id": "k", "shape_type": "keypoints", "points": [[1, 1], [2, 2], [3, 3]], "keypoint_visibility": [2]}
    )
    check("回退：可见性数组补足到点数", kp["keypoint_visibility"] == [2, 2, 2], str(kp["keypoint_visibility"]))

    # line 点数过多时截断
    line = annotation.normalize_object({"id": "l", "shape_type": "line", "points": [[0, 0], [1, 1], [2, 2]]})
    check("回退：线段只保留 2 点", len(line["points"]) == 2, str(line["points"]))


# ---------------------------------------------------------------------------
# 6. 修订号冲突
# ---------------------------------------------------------------------------
def test_revision_conflict(tmp: Path) -> None:
    root = tmp / "rev"
    path = root / "rev.tfanno"
    project = annotation.new_project(
        "detect", [annotation.new_class(0, "object")],
        [annotation.new_image("i", str(root / "i.png"), 10, 10, "train", [])],
    )
    meta = annotation.save(path, project, force=True)
    check("冲突：初始 rev=1", meta["rev"] == 1, str(meta))

    before = path.read_bytes()

    # A 基于 rev=1 保存成功
    project["images"][0]["annotations"].append(annotation.new_object("a1", 0, "", "rectangle", [[0, 0], [5, 5]]))
    meta_a = annotation.save(path, project, expected_rev=1)
    check("冲突：基于正确 rev 保存成功", meta_a["rev"] == 2, str(meta_a))

    # B 仍基于 rev=1 保存 —— 必须被拒
    stale = annotation.new_project(
        "detect", [annotation.new_class(0, "object")],
        [annotation.new_image("i", str(root / "i.png"), 10, 10, "train", [])],
    )
    disk_before = path.read_bytes()
    raised = None
    try:
        annotation.save(path, stale, expected_rev=1)
    except annotation.RevisionConflict as exc:
        raised = exc
    check("冲突：过期修订号被拒", raised is not None)
    check("冲突：reason=rev-mismatch", raised is not None and raised.reason == "rev-mismatch",
          raised.reason if raised else "")
    check("冲突：异常里带磁盘真实 rev", raised is not None and raised.disk["rev"] == 2,
          str(raised.disk) if raised else "")
    check("冲突：磁盘内容未被修改", path.read_bytes() == disk_before, "文件被改写了")

    # 强制覆盖应放行
    meta_force = annotation.save(path, stale, expected_rev=1, force=True)
    check("冲突：force=True 可覆盖", meta_force["rev"] == 3, str(meta_force))
    reloaded, _ = annotation.load(path)
    check("冲突：覆盖后标注被清空（证明真的写进去了）",
          len(reloaded["images"][0]["annotations"]) == 0, str(reloaded["images"][0]["annotations"]))

    # 外部程序改了文件（rev 不变但指纹变了）应被识别
    ok_meta = annotation.load(path)[1]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["task"] = "segment"
    time.sleep(0.02)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    external = None
    try:
        annotation.save(path, raw, expected_rev=ok_meta["rev"], expected_stamp=ok_meta)
    except annotation.RevisionConflict as exc:
        external = exc
    check("冲突：外部改动被识别", external is not None and external.reason == "external-modification",
          external.reason if external else "未触发")
    check("冲突：外部改动时磁盘未被覆盖", json.loads(path.read_text(encoding="utf-8"))["task"] == "segment")

    check("冲突：before 快照非空（防呆）", len(before) > 0)

    # 回归：纳秒指纹超出 JS 安全整数范围，必须用字符串回传，否则被误判为外部修改
    annotation.save(path, raw, force=True)
    meta_b = annotation.load(path)[1]
    check("指纹：meta.mtime_ns 以字符串下发", isinstance(meta_b["mtime_ns"], str),
          type(meta_b["mtime_ns"]).__name__)
    check("指纹：字符串指纹可被解析为整数", int(meta_b["mtime_ns"]) > 0, meta_b["mtime_ns"])

    project_b = annotation.load(path)[0]
    meta_ok = annotation.save(
        path, project_b, expected_rev=meta_b["rev"],
        expected_stamp={"mtime_ns": meta_b["mtime_ns"], "size": meta_b["size"]},
    )
    check("指纹：原样回传可正常保存", meta_ok["rev"] == meta_b["rev"] + 1, str(meta_ok))

    # 模拟浏览器精度丢失（float64）后回传：容差内必须放行
    meta_c = annotation.load(path)[1]
    lossy = int(float(meta_c["mtime_ns"]))
    check("指纹：float 转换的偏差在容差内",
          abs(lossy - int(meta_c["mtime_ns"])) <= annotation.STAMP_TOLERANCE_NS,
          f"{lossy} vs {meta_c['mtime_ns']}")
    project_c = annotation.load(path)[0]
    meta_ok2 = annotation.save(
        path, project_c, expected_rev=meta_c["rev"],
        expected_stamp={"mtime_ns": lossy, "size": meta_c["size"]},
    )
    check("指纹：浮点指纹容差放行", meta_ok2["rev"] == meta_c["rev"] + 1, str(meta_ok2))

    # 大小与记录不符，仍要拦住
    meta_d = annotation.load(path)[1]
    tampered = None
    try:
        annotation.save(
            path, annotation.load(path)[0], expected_rev=meta_d["rev"],
            expected_stamp={"mtime_ns": meta_d["mtime_ns"], "size": meta_d["size"] + 1},
        )
    except annotation.RevisionConflict as exc:
        tampered = exc
    check("指纹：size 不符判定为外部修改",
          tampered is not None and tampered.reason == "external-modification",
          tampered.reason if tampered else "未触发")


# ---------------------------------------------------------------------------
# 7. 导出复用
# ---------------------------------------------------------------------------
def test_export(tmp: Path) -> None:
    root = tmp / "exp"
    make_image(root / "images" / "a.png")
    make_image(root / "images" / "b.png")
    path = root / "exp.tfanno"
    project = build_project(root)
    annotation.save(path, project, force=True)
    loaded, _ = annotation.load(path)

    # 一致性校验：这份工程应当没有硬性问题
    issues = annotation.validate(loaded)
    check("导出：干净工程无一致性告警", not issues, str(issues))

    ds_project = annotation.to_dataset_project(loaded, source=path)
    splits = [image.current_split for image in ds_project.images]
    check("导出：集合划分沿用工程设置", splits == ["train", "val"], str(splits))

    output = tmp / "out_yolo"
    result = dataset.export_yolo(ds_project, output, splits, overwrite=True)
    counts = result.get("counts") or {}
    check("导出：YOLO 导出成功", counts.get("train") == 1 and counts.get("val") == 1, str(result))

    back = dataset.load_tfanno(path)
    check("导出：回读工程图片数一致", len(back.images) == 2, str(len(back.images)))
    total_before = sum(len(img.objects) for img in ds_project.images)
    total_after = sum(len(img.objects) for img in back.images)
    check("导出：回读标注数一致", total_before == total_after, f"{total_before} vs {total_after}")
    check(
        "导出：回读类别数一致",
        len(back.classes) == len(ds_project.classes),
        f"{len(back.classes)} vs {len(ds_project.classes)}",
    )

    # 导出的 YOLO 标签行数应与对象数吻合（每行一个对象）
    label_dir = output / "labels"
    label_files = sorted(label_dir.rglob("*.txt"))
    lines = 0
    for item in label_files:
        lines += len([ln for ln in item.read_text(encoding="utf-8").splitlines() if ln.strip()])
    check("导出：YOLO 标签行数等于对象数", lines == total_before, f"{lines} vs {total_before}")

    # 校验器：越界点应被报出来
    bad = json.loads(json.dumps(loaded))
    bad["images"][0]["annotations"][0]["points"] = [[-50, -50], [5000, 5000]]
    issues2 = annotation.validate(bad)
    check("导出：越界点被校验器捕获", any("范围外" in item for item in issues2), str(issues2))

    # 校验器：引用不存在类别
    bad2 = json.loads(json.dumps(loaded))
    bad2["images"][0]["annotations"][0]["class_id"] = 99
    issues3 = annotation.validate(bad2)
    check("导出：不存在的类别被捕获", any("不存在的类别" in item for item in issues3), str(issues3))


# ---------------------------------------------------------------------------
# 8. 自动标注（不需要真引擎）
# ---------------------------------------------------------------------------
def test_autolabel(tmp: Path) -> None:
    boxes = [
        {"left": 10, "top": 20, "right": 110, "bottom": 90, "confidence": 0.87,
         "class_label": 2, "label_name": "missing_hole"},
        {"left": 300, "top": 40, "right": 200, "bottom": 10, "confidence": 0.5,
         "class_label": 0, "label_name": "ok"},
    ]
    drafts = autolabel.drafts_from_yolo({"boxes": boxes}, "detect")
    check("预标注：每个框一个草稿", len(drafts) == 2, str(len(drafts)))
    check("预标注：草稿是矩形且 2 点", all(d["shape_type"] == "rectangle" and len(d["points"]) == 2 for d in drafts))
    check("预标注：反向坐标被纠正", drafts[1]["points"] == [[200.0, 10.0], [300.0, 40.0]], str(drafts[1]["points"]))
    check("预标注：保留置信度", close(drafts[0]["score"], 0.87), str(drafts[0]["score"]))
    check("预标注：保留类别名", drafts[0]["label"] == "missing_hole", drafts[0]["label"])
    check("预标注：标记为草稿", all(d.get("_draft") is True for d in drafts))

    obb = autolabel.drafts_from_yolo(
        {"boxes": [{"left": 0, "top": 0, "right": 50, "bottom": 30, "confidence": 0.9,
                    "class_label": 1, "is_obb": True,
                    "obb": {"cx": 25, "cy": 15, "w": 50, "h": 30, "angle": 0.3}}]},
        "obb",
    )
    check("预标注：OBB 转旋转矩形", obb[0]["shape_type"] == "rotated_rect", obb[0]["shape_type"])
    check("预标注：OBB 保留旋转角", close(obb[0]["rotation"], 0.3), str(obb[0]["rotation"]))

    pose = autolabel.drafts_from_yolo(
        {"boxes": [{"kpts": [[1, 2, 2], [3, 4, 1]], "class_label": 0, "label_name": "k"}]}, "pose"
    )
    check("预标注：姿态聚合成单对象", len(pose) == 1 and pose[0]["shape_type"] == "keypoints", str(pose))
    check("预标注：姿态点数正确", len(pose[0]["points"]) == 2, str(pose[0]["points"]))
    check("预标注：姿态可见性保留", pose[0]["keypoint_visibility"] == [2, 1], str(pose[0]["keypoint_visibility"]))

    seg = autolabel.drafts_from_yolo(
        {"boxes": [{"left": 0, "top": 0, "right": 10, "bottom": 10, "confidence": 0.6,
                    "class_label": 0, "has_mask": True}]},
        "segment",
    )
    check("预标注：分段退回矩形并说明原因", seg[0]["shape_type"] == "rectangle" and "_note" in seg[0], str(seg[0]))

    # 外部服务的 4 点矩形
    shapes = [{"label": "x", "shape_type": "rectangle", "points": [[0, 0], [10, 0], [10, 10], [0, 10]], "score": 0.7}]
    ext = autolabel.drafts_from_shapes(shapes, "detect")
    check("预标注：外部 4 点矩形折叠为 2 点", len(ext[0]["points"]) == 2, str(ext[0]["points"]))

    # 外部服务的姿态散点按 group_id 聚合
    groups = [
        {"label": "p", "shape_type": "point", "points": [1, 2], "group_id": 7},
        {"label": "p", "shape_type": "point", "points": [3, 4], "group_id": 7},
        {"label": "p", "shape_type": "point", "points": [5, 6], "group_id": 9},
    ]
    agg = autolabel.drafts_from_shapes(groups, "pose")
    check("预标注：姿态散点按组聚合", len(agg) == 2, str(len(agg)))
    check("预标注：聚合后点数正确", len(agg[0]["points"]) == 2 and len(agg[1]["points"]) == 1,
          f"{agg[0]['points']} / {agg[1]['points']}")

    # 服务未配置 / 不可达：都应显式报错，不静默
    raised_none = None
    try:
        autolabel.external_predict("", str(tmp / "nope.png"))
    except autolabel.AutoLabelError as exc:
        raised_none = exc
    check("预标注：未配置服务时报 no-service", raised_none is not None and raised_none.kind == "no-service",
          raised_none.kind if raised_none else "未抛出")

    image = tmp / "al.png"
    make_image(image)
    raised_unreach = None
    try:
        autolabel.external_predict("http://127.0.0.1:9", image, timeout=1.0)
    except autolabel.AutoLabelError as exc:
        raised_unreach = exc
    check("预标注：不可达时报 unreachable", raised_unreach is not None and raised_unreach.kind == "unreachable",
          raised_unreach.kind if raised_unreach else "未抛出")
    check("预标注：不可达的提示里带服务地址",
          raised_unreach is not None and "127.0.0.1:9" in str(raised_unreach),
          str(raised_unreach))


# ---------------------------------------------------------------------------
# 9. HTTP 接口（TestClient，不需要真启动服务）
# ---------------------------------------------------------------------------
def test_api(tmp: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        check("接口：跳过（未安装 httpx）", True)
        return

    import app as web_app

    # 白名单默认只有仓库根；测试目录在系统临时区，必须显式放行，
    # 否则每个接口都会先返回 403，测不到真正想测的逻辑。
    # `config.allowed_roots()` 每次调用都重读环境变量，所以这里设置是生效的。
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(web_app.config.REPO_ROOT)

    client = TestClient(web_app.app)
    root = tmp / "api"
    make_image(root / "images" / "a.png")
    make_image(root / "images" / "b.png")
    path = root / "api.tfanno"
    project = build_project(root)

    saved = client.post("/api/annotation/project", json={"path": str(path), "project": project, "force": True})
    check("接口：首次保存 200", saved.status_code == 200, f"{saved.status_code} {saved.text[:200]}")
    body = saved.json()
    check("接口：返回新修订号", body.get("rev") == 1, str(body.get("rev")))
    check("接口：返回统计", body.get("stats", {}).get("total_objects") == 6, str(body.get("stats")))

    loaded = client.get("/api/annotation/project", params={"path": str(path)})
    check("接口：读取 200", loaded.status_code == 200, loaded.text[:200])
    check("接口：exists=true", loaded.json().get("exists") is True)
    check("接口：返回 meta 供回传", "mtime_ns" in loaded.json().get("meta", {}), str(loaded.json().get("meta")))

    conflict = client.post("/api/annotation/project", json={"path": str(path), "project": project, "rev": 0})
    check("接口：旧 rev 返回 409", conflict.status_code == 409, f"{conflict.status_code} {conflict.text[:200]}")
    check("接口：409 里带 error 码", conflict.json().get("error") == "revision-conflict", conflict.text[:200])
    check("接口：409 里带磁盘 rev", conflict.json().get("disk", {}).get("rev") == 1, conflict.text[:200])

    missing = client.get("/api/annotation/project", params={"path": str(root / "none.tfanno")})
    check("接口：不存在的工程返回空骨架", missing.status_code == 200 and missing.json().get("exists") is False,
          missing.text[:200])

    img = client.get("/api/annotation/image", params={"path": str(root / "images" / "a.png")})
    check("接口：图片流 200", img.status_code == 200, str(img.status_code))
    check("接口：图片 MIME 正确", img.headers.get("content-type", "").startswith("image/png"),
          img.headers.get("content-type", ""))
    check("接口：图片字节非空", len(img.content) > 0, str(len(img.content)))

    outside = Path(tempfile.gettempdir()).parent / "definitely_not_allowed.png"
    blocked = client.get("/api/annotation/image", params={"path": str(outside)})
    check("接口：白名单外被拒 403", blocked.status_code == 403, f"{blocked.status_code} {blocked.text[:200]}")

    meta = client.get("/api/annotation/image-meta", params={"path": str(root / "images" / "a.png")})
    check("接口：image-meta 返回尺寸", meta.json().get("width") == 320 and meta.json().get("height") == 240, meta.text[:200])

    scan = client.post("/api/annotation/scan-images", params={"root": str(root)})
    scan_body = scan.json()
    check("接口：扫描目录找到图片", scan_body.get("count", 0) >= 2, scan.text[:200])
    # 图片数正好等于 limit 时不能误报截断（否则前端会提示「已截断」却一张没少）
    exact = client.post(
        "/api/annotation/scan-images",
        params={"root": str(root), "limit": scan_body.get("count", 0)},
    )
    check(
        "接口：正好 limit 张不误报截断",
        exact.json().get("truncated") is False,
        exact.text[:200],
    )
    capped = client.post("/api/annotation/scan-images", params={"root": str(root), "limit": 1})
    check("接口：超过 limit 才报截断", capped.json().get("truncated") is True, capped.text[:200])

    exported = client.post(
        "/api/annotation/export",
        json={"project": project, "source": str(path), "format": "yolo",
              "output": str(tmp / "api_out"), "overwrite": True},
    )
    check("接口：导出 200", exported.status_code == 200, exported.text[:300])
    check(
        "接口：导出张数正确",
        (exported.json().get("counts") or {}).get("train") == 1
        and (exported.json().get("counts") or {}).get("val") == 1,
        exported.text[:300],
    )
    check("接口：导出沿用工程划分", exported.json().get("split_counts", {}).get("val") == 1,
          str(exported.json().get("split_counts")))

    bad_fmt = client.post(
        "/api/annotation/export",
        json={"project": project, "format": "xml", "output": str(tmp / "x"), "overwrite": True},
    )
    check("接口：未知导出格式被拒 400", bad_fmt.status_code == 400, str(bad_fmt.status_code))


# ---------------------------------------------------------------------------
# 10. 中文与空格路径
# ---------------------------------------------------------------------------
def test_unicode_paths(tmp: Path) -> None:
    root = tmp / "中文 目录"
    image = root / "图片 一张.png"
    make_image(image)
    path = root / "工程 文件.tfanno"
    project = annotation.new_project(
        "detect",
        [annotation.new_class(0, "缺陷")],
        [annotation.new_image("i", str(image), 320, 240, "train",
                              [annotation.new_object("o", 0, "缺陷", "rectangle", [[1, 2], [30, 40]])])],
    )
    meta = annotation.save(path, project, force=True)
    check("中文：工程可写", meta["rev"] == 1, str(meta))
    check("中文：工程文件存在", path.is_file())

    loaded, _ = annotation.load(path)
    check("中文：中文类别名读写一致", loaded["classes"][0]["name"] == "缺陷", str(loaded["classes"]))
    check("中文：图片路径原样保留", loaded["images"][0]["path"] == str(image), loaded["images"][0]["path"])

    resolved = annotation.resolve_image_path(path, "图片 一张.png")
    check("中文：相对路径能解析到真实文件", resolved.is_file(), str(resolved))

    # 文件内容必须是可读的 UTF-8（不是 \uXXXX 转义），便于人工排查
    text = path.read_text(encoding="utf-8")
    check("中文：落盘为可读 UTF-8 而非转义", "缺陷" in text, text[:200])


# ---------------------------------------------------------------------------
# 11. 配色与 Qt 一致
# ---------------------------------------------------------------------------
def test_palette() -> None:
    # Qt: QColor::fromHsv((seed * 47) % 360, 210, 240)
    check("配色：seed 0 与 Qt 一致", annotation.default_color(0) == [240, 42, 42], str(annotation.default_color(0)))
    check("配色：seed 8 回绕到 16 度", annotation.default_color(8) == annotation.hsv_to_rgb(376 % 360, 210, 240),
          str(annotation.default_color(8)))
    check("配色：饱和度 0 时为灰", annotation.hsv_to_rgb(123, 0, 100) == [100, 100, 100])
    check("配色：各通道在 0~255", all(0 <= c <= 255 for s in range(12) for c in annotation.default_color(s)))


def test_hardening(tmp: Path) -> None:
    """回归：并发保存与校验器的两类「看起来正常却被判错/写坏」的问题。"""
    import threading

    root = tmp / "hard"
    make_image(root / "images" / "a.png")
    make_image(root / "images" / "b.png")
    path = root / "hard.tfanno"
    project = build_project(root)
    annotation.save(path, project, force=True)
    loaded, _ = annotation.load(path)

    # 1) 标注 id 只在单图内唯一：两张图各有一个 "1" 不该报警
    twins = json.loads(json.dumps(loaded))
    for image in twins["images"]:
        for index, obj in enumerate(image.get("annotations") or []):
            obj["id"] = str(index + 1)
    issues = annotation.validate(twins)
    check(
        "并发/校验：跨图同号标注不再误报重复",
        not any("标注 id 重复" in item for item in issues),
        str(issues),
    )

    # 2) 同一张图内重复才真的算错
    dupes = json.loads(json.dumps(twins))
    objects = dupes["images"][0].get("annotations") or []
    if len(objects) >= 2:
        objects[1]["id"] = objects[0]["id"]
        issues_dup = annotation.validate(dupes)
        check("并发/校验：单图内重复 id 仍被抓到", any("标注 id 重复" in item for item in issues_dup), str(issues_dup))

    # 3) 同一路径并发保存：只能有一个成功（其余撞修订号），文件必须仍是完整 JSON
    results: list[str] = []
    errors: list[str] = []

    def worker() -> None:
        try:
            annotation.save(path, loaded, expected_rev=loaded.get("rev"), force=False)
            results.append("ok")
        except annotation.RevisionConflict:
            results.append("conflict")
        except Exception as exc:  # noqa: BLE001 - 任何其它异常都算失败
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    check("并发保存：无意外异常", not errors, str(errors))
    check("并发保存：有且仅有一个成功", results.count("ok") == 1, str(results))
    after, _ = annotation.load(path)
    check("并发保存：文件仍是完整工程", len(after.get("images") or []) == len(loaded.get("images") or []))
    check(
        "并发保存：修订号只加一",
        int(after.get("rev") or 0) == int(loaded.get("rev") or 0) + 1,
        f"{after.get('rev')} vs {loaded.get('rev')}",
    )
    leftovers = list(root.glob("*.tmp*"))
    check("并发保存：无残留临时文件", not leftovers, str(leftovers))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tfanno_test_") as raw:
        tmp = Path(raw)
        test_roundtrip(tmp)
        test_qt_compat(tmp)
        test_extensions(tmp)
        test_rectangle_fold()
        test_fallbacks()
        test_revision_conflict(tmp)
        test_export(tmp)
        test_autolabel(tmp)
        test_api(tmp)
        test_unicode_paths(tmp)
        test_palette()
        test_hardening(tmp)

    print()
    if FAILED:
        print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}，失败 {len(FAILED)} 项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print(f"通过 {len(PASSED)} / {len(PASSED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
