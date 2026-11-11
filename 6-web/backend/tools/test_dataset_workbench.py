# -*- coding: utf-8 -*-
"""数据集工作台验收脚本（不需要启动 Web 服务）。

覆盖四类风险点：
1. 划分算法是否稳定（同输入两次结果必须逐项一致）；
2. 比例语义是否与 Qt 版一致（train/val 向下取整，余数归 test）；
3. 导出是否可逆（YOLO 标签写回后与源标签逐值相同）；
4. 错误路径是否被拦住（比例超限、输出在源内、格式不兼容、非空未覆盖）。

用法：
    python 6-web/backend/tools/test_dataset_workbench.py
    python 6-web/backend/tools/test_dataset_workbench.py --dataset "D:/.../PCB缺陷检测"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# 允许直接以脚本方式运行（把 backend 加入 sys.path）
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import dataset  # noqa: E402
import datasetqc  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_ok_samples(work: Path) -> None:
    """良品图的八条硬规则。

    这里全部用临时目录 + 空图片文件搭数据，不依赖真实数据集，所以每次都会跑。
    最容易错的地方是「良品」与「漏标」的边界：两者在磁盘上都表现为没有框，
    但前者要算进过杀率、后者必须排除——混在一起过杀率就会变成一个好看的假数。
    """
    root = work / "okcase"
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [defect]\n".format(
            root.as_posix()
        ),
        encoding="utf-8",
    )
    # 缺陷图：一张有标注、一张空标签（空标签在 images 里表示「这张没缺陷」也是漏标嫌疑）
    _touch(root / "images" / "train" / "d1.jpg")
    _touch(root / "labels" / "train" / "d1.txt", "0 0.5 0.5 0.2 0.2\n")
    _touch(root / "images" / "train" / "d2.jpg")
    _touch(root / "labels" / "train" / "d2.txt")
    # 良品目录：两张图，都没有标签文件
    _touch(root / "ok" / "g_ok1.jpg")
    _touch(root / "ok" / "g_ok2.jpg")

    project = dataset.load_project(str(root), "detect", ok_dirs=["ok"], ok_split="val")
    ok_items = [image for image in project.images if image.is_ok]
    check("良品目录被记录", project.ok_dirs == ["ok"], str(project.ok_dirs))
    check("良品图计入扫描结果", len(ok_items) == 2, f"ok={len(ok_items)}")
    check("良品图默认进 val", all(image.current_split == "val" for image in ok_items))
    check("缺陷图不受影响", sum(1 for image in project.images if not image.is_ok) == 2)
    summary = dataset.ok_summary(project, "val")
    check("良品统计张数正确", summary["images"] == 2, str(summary))
    check("良品统计带说明", bool(summary["note"]))
    preview = dataset.preview(project, 70, 20)
    check("预览里带良品统计", preview["ok"]["images"] == 2, str(preview.get("ok")))

    # ---- 良品图必须归到验收划分 ----
    splits = ["train"] * len(project.images)
    moved = dataset.apply_ok_split(project, splits, "test")
    ok_positions = [i for i, image in enumerate(project.images) if image.is_ok]
    check("良品图被移动到指定划分", moved == 2, f"moved={moved}")
    check(
        "良品图划分全部改为 test",
        all(splits[i] == "test" for i in ok_positions),
        str([splits[i] for i in ok_positions]),
    )
    check(
        "非良品图的划分没有被改动",
        all(splits[i] == "train" for i in range(len(splits)) if i not in ok_positions),
        str(splits),
    )

    # ---- 良品目录里的非空标注必须拦住 ----
    _touch(root / "ok" / "bad.jpg")
    _touch(root / "ok" / "bad.txt", "0 0.5 0.5 0.2 0.2\n")
    try:
        dataset.load_project(str(root), "detect", ok_dirs=["ok"], ok_split="val")
        check("良品目录有标注时报错", False, "未拦截")
    except dataset.DatasetError as exc:
        check("良品目录有标注时报错", "非空标注" in str(exc), str(exc))
    (root / "ok" / "bad.jpg").unlink()
    (root / "ok" / "bad.txt").unlink()

    # ---- 空标签文件 + 已被正常扫描的图：只打标记，不重复计数 ----
    _touch(root / "images" / "val" / "g3.jpg")
    _touch(root / "labels" / "val" / "g3.txt")
    project2 = dataset.load_project(str(root), "detect", ok_dirs=["ok", "images/val"], ok_split="val")
    stems = [image.path.stem for image in project2.images]
    check("同一张图不会重复计数", len(stems) == len(set(stems)), str(stems))
    dup = next(image for image in project2.images if image.path.stem == "g3")
    check("已扫描到的良品图被补打标记", dup.is_ok is True)
    check("良品张数仍按图算", sum(1 for image in project2.images if image.is_ok) == 3,
          str(sum(1 for image in project2.images if image.is_ok)))

    # ---- 良品划分参数必须合法 ----
    try:
        dataset.load_project(str(root), "detect", ok_dirs=["ok"], ok_split="trainx")
        check("非法良品划分被拒", False, "未拦截")
    except dataset.DatasetError:
        check("非法良品划分被拒", True)

    # ---- 体检验收：良品目录里的图不算漏标 ----
    report = datasetqc.inspect(root, "detect", ok_dirs=["ok", "images/val"])
    check("体检报告标出良品图数", report.get("ok_images") == 3, str(report.get("ok_images")))
    ok_issues = [
        item for item in report["issues"]
        if item.get("code") in ("LBL_MISSING", "LBL_EMPTY") and "g_ok" in str(item.get("rel") or "")
    ]
    check("良品图不被报成漏标/空标注", not ok_issues, str(ok_issues))

    # ---- 导出：良品图写空标签，且只出现在验收划分 ----
    out = work / "okexport"
    result = dataset.run_export(
        str(root), "detect", 70, 20, "yolo", str(out), False,
        ok_dirs=["ok"], ok_split="val",
    )
    check(
        "导出图片数包含良品图",
        sum(result["counts"].values()) == 5,
        str(result["counts"]),
    )
    ok_in_train = [
        path.name for path in (out / "images" / "train").glob("*")
        if path.stem.startswith("g_ok")
    ]
    check("良品图没有进入训练集", not ok_in_train, str(ok_in_train))
    val_labels = sorted((out / "labels" / "val").glob("*.txt"))
    empty_labels = [path for path in val_labels if not path.read_text(encoding="utf-8").strip()]
    check("良品图导出空标签文件（不是缺文件）", len(empty_labels) >= 2,
          str([path.name for path in val_labels]))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("manifest 记录良品块", manifest.get("ok", {}).get("images") == 2, str(manifest.get("ok")))
    check("manifest 记录良品划分", manifest.get("ok", {}).get("split") == "val")
    check(
        "split_mode 记录 ok_split",
        result["split_mode"].get("ok_split") == "val",
        str(result["split_mode"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(BACKEND.parents[1] / "PCB data" / "PCB缺陷检测"),
        help="用于验证的 YOLO 数据集目录",
    )
    args = parser.parse_args()

    # 良品图相关的验收用临时目录自造数据，不依赖本机是否有真实数据集
    ok_work = Path(tempfile.mkdtemp(prefix="tfds_ok_"))
    try:
        test_ok_samples(ok_work)
    finally:
        shutil.rmtree(ok_work, ignore_errors=True)

    source = Path(args.dataset)
    if not source.is_dir():
        print(f"[SKIP] 找不到数据集：{source}")
        return 0

    project = dataset.load_project(str(source))
    total = len(project.images)
    check("扫描到图片", total > 0, f"total={total}")
    check("识别到类别", len(project.classes) > 0, f"classes={len(project.classes)}")
    check("任务类型合法", project.task in dataset.TASKS, project.task)

    # 1. 稳定性
    first = dataset.assign_splits(total, 70, 20)
    second = dataset.assign_splits(total, 70, 20)
    check("划分可复现（同输入两次一致）", first == second)
    check(
        "FNV-1a 哈希稳定",
        dataset.fnv1a64("0_2026") == dataset.fnv1a64("0_2026")
        and dataset.fnv1a64("0_2026") != dataset.fnv1a64("1_2026"),
    )

    # 2. 比例语义
    counts = dataset.split_counts(first)
    check(
        "70/20/10 向下取整",
        counts["train"] == total * 70 // 100 and counts["val"] == total * 20 // 100,
        str(counts),
    )
    check("余数归 test", counts["train"] + counts["val"] + counts["test"] == total, str(counts))
    counts8020 = dataset.split_counts(dataset.assign_splits(total, 80, 20))
    check("80/20 时 test 为余数", counts8020["test"] == total - total * 80 // 100 - total * 20 // 100)

    tmp = Path(tempfile.mkdtemp(prefix="tfds_test_"))
    try:
        # 3. YOLO 导出可逆
        result = dataset.run_export(str(source), "auto", 70, 20, "yolo", str(tmp / "yolo"), False)
        yaml_text = (tmp / "yolo" / "dataset.yaml").read_text(encoding="utf-8")
        check("dataset.yaml 含 nc", "nc:" in yaml_text)
        check("dataset.yaml 含全部类别名", all(c["name"] in yaml_text for c in project.classes))
        check(
            "导出图片数等于源图片数",
            sum(result["counts"].values()) == total,
            str(result["counts"]),
        )
        # 抽第一张有标注的图，逐行比对标签
        sample = next(img for img in project.images if img.objects)
        label = next((tmp / "yolo" / "labels").rglob(f"{sample.path.stem}.txt"), None)
        check("抽样标签文件存在", label is not None)
        if label is not None:
            original_path = next((source / "labels").rglob(f"{sample.path.stem}.txt"), None)
            if original_path is not None:
                src_lines = [ln.split() for ln in original_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                dst_lines = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
                same = len(src_lines) == len(dst_lines) and all(
                    a[0] == b[0] and all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a[1:], b[1:]))
                    for a, b in zip(src_lines, dst_lines)
                )
                check("YOLO 标签往返无损", same, f"{len(src_lines)} 行")

        # 4. COCO / VOC
        coco = dataset.run_export(str(source), "auto", 70, 20, "coco", str(tmp / "coco"), False)
        doc = json.loads((tmp / "coco" / "annotations.json").read_text(encoding="utf-8"))
        check(
            "COCO 结构完整",
            {"images", "annotations", "categories"} <= set(doc)
            and len(doc["categories"]) == len(project.classes),
            str(coco["counts"]),
        )
        voc = dataset.run_export(str(source), "auto", 70, 20, "voc", str(tmp / "voc"), False)
        xml_count = len(list((tmp / "voc" / "Annotations").glob("*.xml")))
        check("VOC 生成 XML", xml_count == voc["counts"]["images"] and xml_count > 0, f"xml={xml_count}")

        # 5. 错误路径
        cases = {
            "比例超限被拒": lambda: dataset.run_export(str(source), "auto", 90, 20, "yolo", str(tmp / "e1"), False),
            "输出在源内被拒": lambda: dataset.run_export(str(source), "auto", 70, 20, "yolo", str(source / "out"), False),
            "非空目录需显式覆盖": lambda: dataset.run_export(str(source), "auto", 70, 20, "yolo", str(tmp / "coco"), False),
            "VOC 拒绝非 detect": lambda: dataset.run_export(str(source), "segment", 70, 20, "voc", str(tmp / "e2"), False),
            "COCO 拒绝 OBB": lambda: dataset.run_export(str(source), "obb", 70, 20, "coco", str(tmp / "e3"), False),
        }
        for name, fn in cases.items():
            try:
                fn()
                check(name, False, "未拦截")
            except dataset.DatasetError:
                check(name, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
