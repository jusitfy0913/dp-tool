# -*- coding: utf-8 -*-
"""YOLO 数据集 -> .tfanno 导入层的验收脚本（不需要 GPU / 服务 / trtexec）。

覆盖的是「导入会不会把标注弄错」这件事，而不是接口连通性：

1. detect 的归一化坐标 -> 像素坐标必须逐位准确；
2. segment / obb / pose 三种特殊布局的形状类型与点数正确，
   **关键点的 visibility 不能被当成坐标夹到 [0,1]**（2 变 1 就是错答案）；
3. 空标签、孤儿标签、畸形行、明显越界、轻微越界分别计数，且不中断整批；
4. 类别号超出 names 范围必须跳过并计数，不能猜成别的类；
5. 合并模式按类别名复用 id，同名图片 skip / replace / append 三种策略；
6. 任务类型不一致时拒绝合并（一个 .tfanno 只能有一种 task）；
7. `data.yaml` 的 `path:` 指向别的目录时也要跟着走；
8. 真实 PCB 数据集存在时做一次冒烟（不存在则跳过，CI 上没有这份数据）。

用法：
    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_annotation_import.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import annotation  # noqa: E402
import annotationimport  # noqa: E402

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


def make_image(path: Path, width: int = 100, height: int = 50) -> None:
    """写一张纯色 PNG。用 PIL 而不是 cv2：导入层只需要读图片头。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (30, 40, 50)).save(path)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_dataset(
    root: Path,
    labels: dict[str, str],
    names: str = '["missing_hole", "mouse_bite"]',
    yaml_path: Path | None = None,
    extra_images: tuple[str, ...] = (),
) -> Path:
    """构造一个最小的 YOLO 数据集，返回 data.yaml 路径。

    ``labels`` 的键是相对 ``images/train`` 的文件名，值是标签文本。
    """
    for name, text in labels.items():
        make_image(root / "images" / "train" / name)
        write(root / "labels" / "train" / f"{Path(name).stem}.txt", text)
    for name in extra_images:
        make_image(root / "images" / "train" / name)

    yaml = yaml_path or (root / "data.yaml")
    write(
        yaml,
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names.split(','))}\n"
        f"names: {names}\n",
    )
    return yaml


# ---------------------------------------------------------------------------
# 1. detect：坐标换算
# ---------------------------------------------------------------------------
def test_detect() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name) / "ds"
        yaml = make_dataset(
            root,
            {
                "a.png": "0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.2\n",
                "b.png": "",  # 空标签文件：0 框
            },
        )
        project, report = annotationimport.build(yaml, project_path=str(root / "pcb.tfanno"))

        check("detect：任务类型推断", report["task"] == "detect", report["task"])
        check("detect：图片数", report["images"] == 2, report["images"])
        check("detect：框数", report["objects"] == 2, report["objects"])
        check("detect：类别名来自 names", [c["name"] for c in project["classes"]] == ["missing_hole", "mouse_bite"])
        check("detect：split 取自目录", project["images"][0]["split"] == "train")

        image = next(item for item in project["images"] if item["id"] == "a")
        first, second = image["annotations"]
        check("detect：形状是矩形", first["shape_type"] == "rectangle")
        check(
            "detect：坐标逐位准确",
            close(first["points"][0][0], 40) and close(first["points"][0][1], 15)
            and close(first["points"][1][0], 60) and close(first["points"][1][1], 35),
            first["points"],
        )
        check("detect：class_id 正确", first["class_id"] == 0 and second["class_id"] == 1)
        check("detect：label 写类别名", second["label"] == "mouse_bite", second["label"])
        check("detect：空标签图 0 框", next(i for i in project["images"] if i["id"] == "b")["annotations"] == [])
        check("detect：空标签计数", report["empty_images"] == 1, report["empty_images"])
        check("detect：无校验问题", annotation.validate(project) == [], annotation.validate(project))

        # 相对路径：工程与数据集同盘时必须能搬移
        stored = image["path"]
        check(
            "detect：存相对工程目录的路径",
            stored == os.path.join("images", "train", "a.png"),
            stored,
        )
        check("detect：路径模式报告", report["path_mode"] == "relative", report["path_mode"])


# ---------------------------------------------------------------------------
# 2. 脏数据：畸形 / 越界 / 未知类别 / 孤儿
# ---------------------------------------------------------------------------
def test_dirty_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name) / "ds"
        yaml = make_dataset(
            root,
            {
                "c.png": (
                    "0 0.5 0.5\n"              # 字段不足 -> 畸形行
                    "0 1.5 0.5 0.2 0.2\n"      # 明显越界 -> 整条丢弃
                    "0 1.01 0.5 0.2 0.2\n"     # 轻微越界 -> 夹回
                    "1 0.2 0.2 0.1 0.1\n"      # 正常
                ),
                "d.png": "9 0.5 0.5 0.2 0.2\n",  # 类别号超出 names
            },
        )
        # 孤儿标签：有 txt 没有同名图片
        write(root / "labels" / "train" / "ghost.txt", "0 0.5 0.5 0.1 0.1\n")

        project, report = annotationimport.build(yaml, project_path=str(root / "x.tfanno"))

        check("脏数据：畸形行计数", report["malformed_lines"] == 1, report["malformed_lines"])
        check("脏数据：明显越界被丢弃", report["out_of_range_objects"] == 1, report["out_of_range_objects"])
        check("脏数据：轻微越界计数", report["clamped_lines"] == 1, report["clamped_lines"])
        check("脏数据：越界点被夹回图像内", report["clamped_points"] == 1, report["clamped_points"])
        check("脏数据：未知类别被跳过", report["unknown_class_objects"] == 1, report["unknown_class_objects"])
        check("脏数据：孤儿标签计数", report["orphan_labels"] == 1, report["orphan_labels"])
        check("脏数据：孤儿标签不进工程", report["images"] == 2, report["images"])
        check("脏数据：没有校验问题", annotation.validate(project) == [], annotation.validate(project))

        warnings = "\n".join(report["warnings"])
        check("脏数据：警告里出现孤儿标签", "找不到同名图片" in warnings, warnings)


# ---------------------------------------------------------------------------
# 3. segment / obb / pose
# ---------------------------------------------------------------------------
def test_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        base = Path(tmp_name)

        seg = base / "seg"
        yaml = make_dataset(
            seg,
            {"s.png": "0 0.1 0.1 0.5 0.2 0.6 0.8\n"},
            names='["poly"]',
        )
        project, report = annotationimport.build(yaml)
        obj = project["images"][0]["annotations"][0]
        check("segment：任务推断", report["task"] == "segment", report["task"])
        check("segment：形状是多边形", obj["shape_type"] == "polygon", obj["shape_type"])
        check("segment：3 个点", len(obj["points"]) == 3, len(obj["points"]))
        check(
            "segment：坐标准确",
            close(obj["points"][1][0], 50) and close(obj["points"][2][1], 40),
            obj["points"],
        )

        obb = base / "obb"
        yaml = make_dataset(
            obb,
            {"o.png": "0 0.1 0.2 0.6 0.2 0.6 0.8 0.1 0.8\n"},
            names='["box"]',
        )
        project, report = annotationimport.build(yaml)
        obj = project["images"][0]["annotations"][0]
        check("obb：任务推断", report["task"] == "obb", report["task"])
        check("obb：4 个角点", len(obj["points"]) == 4, len(obj["points"]))
        # 存成多边形是为了「导回 YOLO 时角点不变」：旋转框只存 2 点 + 角度，会丢角度
        check("obb：用多边形保住角点", obj["shape_type"] == "polygon", obj["shape_type"])
        check(
            "obb：角点坐标准确",
            close(obj["points"][0][0], 10) and close(obj["points"][1][0], 60)
            and close(obj["points"][2][1], 40),
            obj["points"],
        )
        warnings = "\n".join(report["warnings"])
        check("obb：报告解释为什么用多边形", "4 个角点按多边形导入" in warnings, warnings)

        pose = base / "pose"
        # 前 4 个值是框，后面 17 组 x y visibility；visibility 全写 2
        keypoints = " ".join(f"{0.5} {0.5} {2}" for _ in range(17))
        yaml = make_dataset(
            pose,
            {"p.png": f"0 0.5 0.5 0.4 0.6 {keypoints}\n"},
            names='["person"]',
        )
        project, report = annotationimport.build(yaml)
        obj = project["images"][0]["annotations"][0]
        check("pose：任务推断", report["task"] == "pose", report["task"])
        check("pose：形状是关键点", obj["shape_type"] == "keypoints", obj["shape_type"])
        check("pose：17 个关键点", len(obj["points"]) == 17, len(obj["points"]))
        check(
            "pose：visibility 没被当成坐标夹成 1",
            len(obj["keypoint_visibility"]) == 17 and set(obj["keypoint_visibility"]) == {2},
            obj["keypoint_visibility"][:5],
        )


# ---------------------------------------------------------------------------
# 4. data.yaml 的 path: 指向别的目录
# ---------------------------------------------------------------------------
def test_yaml_path_redirection() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        base = Path(tmp_name)
        data = base / "data"
        make_image(data / "images" / "val" / "v.png")
        write(data / "labels" / "val" / "v.txt", "0 0.5 0.5 0.2 0.2\n")

        cfg = base / "cfg"
        yaml = cfg / "pcb.yaml"
        write(
            yaml,
            "path: ../data\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  - missing_hole\n",
        )

        project, report = annotationimport.build(yaml, splits=["val"])
        check("path 重定向：找到 val 图片", report["images"] == 1, report["images"])
        check("path 重定向：框数", report["objects"] == 1, report["objects"])
        check("path 重定向：类别来自块状 names", [c["name"] for c in project["classes"]] == ["missing_hole"])
        check("path 重定向：无孤儿", report["orphan_labels"] == 0, report["orphan_labels"])


# ---------------------------------------------------------------------------
# 5. 合并模式
# ---------------------------------------------------------------------------
def build_base(root: Path, task: str = "detect") -> dict:
    image = root / "images" / "train" / "a.png"
    return annotation.new_project(
        task=task,
        classes=[annotation.new_class(0, "missing_hole"), annotation.new_class(7, "extra")],
        images=[
            annotation.new_image(
                "a",
                str(image),
                100,
                50,
                "train",
                [
                    {
                        "id": "old",
                        "class_id": 0,
                        "label": "missing_hole",
                        "shape_type": "rectangle",
                        "points": [[1, 1], [9, 9]],
                    }
                ],
            )
        ],
    )


def test_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name) / "ds"
        yaml = make_dataset(
            root,
            {"a.png": "0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1\n"},
        )
        base = build_base(root)

        project, report = annotationimport.build(
            yaml, project_path=str(root / "x.tfanno"), base_project=base, mode="merge"
        )
        ids = [item["id"] for item in project["classes"]]
        check("合并：类别名复用已有 id", ids == [0, 7, 8], ids)
        check("合并：同名类别没有被重复添加", [c["name"] for c in project["classes"]].count("missing_hole") == 1)
        check("合并：新类别追加在末尾", project["classes"][-1]["name"] == "mouse_bite", project["classes"][-1])

        # 同名图片走 skip：原图标注不能被换成 txt 里的
        image = next(item for item in project["images"] if item["id"] == "a")
        check("合并 skip：图片数不变", len(project["images"]) == 1, len(project["images"]))
        check("合并 skip：计数正确", report["skipped_existing"] == 1, report["skipped_existing"])
        check("合并 skip：原标注保留", [o["id"] for o in image["annotations"]] == ["old"], image["annotations"])

        # replace：用 txt 覆盖标注，但保留图片 id 与 split
        replaced, replaced_report = annotationimport.build(
            yaml, project_path=str(root / "x.tfanno"), base_project=build_base(root),
            mode="merge", duplicate="replace",
        )
        image = next(item for item in replaced["images"] if item["id"] == "a")
        check("合并 replace：标注被覆盖", len(image["annotations"]) == 2, image["annotations"])
        check("合并 replace：计数正确", replaced_report["replaced_images"] == 1, replaced_report["replaced_images"])
        check("合并 replace：图片没有重复", len(replaced["images"]) == 1, len(replaced["images"]))

        # append：同一张图再加一份，id 必须唯一
        appended, _ = annotationimport.build(
            yaml, project_path=str(root / "x.tfanno"), base_project=build_base(root),
            mode="merge", duplicate="append",
        )
        check("合并 append：变成两张", len(appended["images"]) == 2, len(appended["images"]))
        check("合并 append：id 唯一", len({i["id"] for i in appended["images"]}) == 2)
        check("合并 append：无校验问题", annotation.validate(appended) == [], annotation.validate(appended))

        # 任务不一致必须拒绝
        try:
            annotationimport.build(
                yaml, project_path=str(root / "x.tfanno"),
                base_project=build_base(root, task="segment"), mode="merge",
            )
            check("合并：任务不一致要报错", False, "没有抛错")
        except annotationimport.AnnotationImportError as exc:
            check("合并：任务不一致要报错", "任务" in str(exc), str(exc))


# ---------------------------------------------------------------------------
# 6. 参数与路径模式
# ---------------------------------------------------------------------------
def test_options() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name) / "ds"
        yaml = make_dataset(root, {"a.png": "0 0.5 0.5 0.2 0.2\n"})

        # 未指定工程路径 -> 绝对路径
        project, report = annotationimport.build(yaml)
        check("无工程路径：存绝对路径", os.path.isabs(project["images"][0]["path"]), project["images"][0]["path"])
        check("无工程路径：报告写 absolute", report["path_mode"] == "absolute", report["path_mode"])

        # 只选一个不存在的 split：必须明确报错，而不是导出一个空工程
        try:
            annotationimport.build(yaml, splits=["val"])
            check("split 过滤：空 split 要报错", False, "没有抛错")
        except annotationimport.AnnotationImportError as exc:
            check("split 过滤：空 split 要报错", "split" in str(exc), str(exc))

        try:
            annotationimport.build(yaml, splits=["nope"])
            check("split 过滤：未知 split 要报错", False, "没有抛错")
        except annotationimport.AnnotationImportError as exc:
            check("split 过滤：未知 split 要报错", "未知" in str(exc), str(exc))

        # 数据源不存在
        try:
            annotationimport.build(root / "missing")
            check("数据源不存在要报错", False, "没有抛错")
        except annotationimport.AnnotationImportError:
            check("数据源不存在要报错", True)


# ---------------------------------------------------------------------------
# 7. 真实 PCB 数据集冒烟
# ---------------------------------------------------------------------------
def test_pcb_smoke() -> None:
    yaml = Path(r"D:\test\deeplearning\dp-tool\PCB data\PCB缺陷检测\data.yaml")
    if not yaml.is_file():
        print("[SKIP] 没有 PCB 数据集，跳过冒烟")
        return
    project, report = annotationimport.build(
        yaml, project_path=str(yaml.parent / "pcb.tfanno"), splits=["val"]
    )
    check("PCB：val 139 图", report["images"] == 139, report["images"])
    check("PCB：val 598 框", report["objects"] == 598, report["objects"])
    check("PCB：6 个类别", len(project["classes"]) == 6, len(project["classes"]))
    check("PCB：detect 任务", report["task"] == "detect", report["task"])
    check("PCB：无校验问题", annotation.validate(project) == [], annotation.validate(project))


def main() -> int:
    test_detect()
    test_dirty_labels()
    test_shapes()
    test_yaml_path_redirection()
    test_merge()
    test_options()
    test_pcb_smoke()

    print()
    if FAILED:
        print(f"失败：{len(PASSED)} 项通过，{len(FAILED)} 项失败")
        for name in FAILED:
            print(f"  ✗ {name}")
        return 1
    print(f"全部通过：{len(PASSED)} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
