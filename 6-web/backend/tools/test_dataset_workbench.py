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

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(BACKEND.parents[1] / "PCB data" / "PCB缺陷检测"),
        help="用于验证的 YOLO 数据集目录",
    )
    args = parser.parse_args()
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
