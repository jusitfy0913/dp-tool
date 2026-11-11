# -*- coding: utf-8 -*-
"""数据集体检验收脚本（不需要启动 Web 服务，不依赖真实数据集）。

覆盖六类风险点：
1. 分组是否把「同底板」分对、是否会把整批图塌成一组；
2. 逐字节重复是否被单独识别（唯一允许删的级别）；
3. 缓存是否命中、命中后结论是否一致；
4. 标注质检能否定位到行号与原始内容；
5. 按组切分是否保证「同组不跨划分」，默认行为是否与升级前一致；
6. 体检是否真的只读。

    10|用法：
    python 6-web/backend/tools/test_dataset_hygiene.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# 缓存指向临时文件：测试不能污染真实运行目录，也不能反过来被真实缓存干扰
_TMP_CACHE = Path(tempfile.mkdtemp(prefix="dsqc_cache_")) / "cache.json"
os.environ["TRTFORGE_DSQC_CACHE"] = str(_TMP_CACHE)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

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


def _base_plate(seed: int, level: int) -> np.ndarray:
    """造一块「底板」：灰度底 + 固定纹理，同一 seed 完全一致。"""
    rng = np.random.default_rng(seed)
    plate = np.full((240, 320), 80 + level * 35, dtype=np.uint8)
    for _ in range(30):
        y = int(rng.integers(0, 236))
        x = int(rng.integers(0, 260))
        plate[y : y + 2, x : x + int(rng.integers(20, 60))] = 190
    return plate


def build_dataset(root: Path) -> None:
    """12 张图 / 3 个底板组，另有 1 张逐字节副本被放进 val，用来验证泄漏与精确重复。"""
    for name in ("train", "val"):
        (root / "images" / name).mkdir(parents=True, exist_ok=True)
        (root / "labels" / name).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnc: 2\nnames: [defect, other]\n".format(
            root.as_posix()
        ),
        encoding="utf-8",
    )
    for group, (seed, level) in enumerate(((11, 0), (22, 1), (33, 2)), start=1):
        plate = _base_plate(seed, level)
        for index in range(4):
            array = plate.copy()
            # 局部缺陷斑块：位置随张变化，大小固定 —— 模拟「同底板不同缺陷实例」
            px, py = 20 + index * 55, 40 + index * 35
            array[py : py + 20, px : px + 20] = 30
            image = Image.fromarray(array, "L").convert("RGB")
            name = f"{group:02d}_defect_{index + 1:02d}"
            image.save(root / "images" / "train" / f"{name}.jpg", quality=95)
            (root / "labels" / "train" / f"{name}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
    # 第 1 组的第 1 张逐字节复制一份放到 val：既是精确重复，也制造跨划分泄漏
    source = root / "images" / "train" / "01_defect_01.jpg"
    shutil.copyfile(source, root / "images" / "val" / "01_defect_01_copy.jpg")
    shutil.copyfile(
        root / "labels" / "train" / "01_defect_01.txt",
        root / "labels" / "val" / "01_defect_01_copy.txt",
    )


def build_broken_dataset(root: Path) -> None:
    """每个坏规则各造一张图，用来验证规则码与行号定位。"""
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nnc: 2\nnames: [defect, other]\n".format(root.as_posix()),
        encoding="utf-8",
    )
    plate = _base_plate(7, 0)
    cases = {
        "b01_field": "0 0.5 0.5 0.1\n",  # 只有 4 个字段
        "b02_parse": "0 0.5 abc 0.1 0.1\n",  # 数值无法解析
        "b03_class": "9 0.5 0.5 0.1 0.1\n",  # 类别越界
        "b04_coord": "0 1.5 0.5 0.1 0.1\n",  # 坐标越界
        "b05_zero": "0 0.5 0.5 0 0.1\n",  # 零面积
        "b06_tiny": "0 0.5 0.5 0.001 0.001\n",  # 极小框
        "b07_huge": "0 0.5 0.5 0.9 0.9\n",  # 超大框
        "b08_border": "0 0.005 0.5 0.01 0.2\n",  # 贴边
        "b09_dup": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",  # 同图重复框
        "b10_empty": "",  # 空标注
    }
    for name, text in cases.items():
        Image.fromarray(plate, "L").convert("RGB").save(
            root / "images" / "train" / f"{name}.jpg", quality=95
        )
        (root / "labels" / "train" / f"{name}.txt").write_text(text, encoding="utf-8")
    # 缺标注文件
    Image.fromarray(plate, "L").convert("RGB").save(
        root / "images" / "train" / "b11_missing.jpg", quality=95
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="dsqc_case_"))
    try:
        root = work / "clean"
        build_dataset(root)

        # ---------- 只读性：体检前后所有标签文件的摘要与 mtime 不变 ----------
        labels = sorted((root / "labels").rglob("*.txt"))
        before = {str(path): (file_digest(path), path.stat().st_mtime_ns) for path in labels}

        report = datasetqc.inspect(root, "detect", threshold=12)

        after = {str(path): (file_digest(path), path.stat().st_mtime_ns) for path in labels}
        check("体检不修改任何标签文件", before == after)

        # ---------- 分组 ----------
        check("扫描到全部图片", report["counts"]["images"] == 13, str(report["counts"]))
        check(
            "分组数等于底板数（没有塌成一组）",
            report["groups"]["total"] == 3,
            f"groups={report['groups']['total']}",
        )
        biggest = report["groups"]["items"][0]["size"] if report["groups"]["items"] else 0
        check("最大分组没有被全量吞并", biggest < report["counts"]["images"], f"biggest={biggest}")
        check(
            "跨划分泄漏被发现",
            report["groups"]["cross_split"] == 1,
            f"cross={report['groups']['cross_split']}",
        )
        check("泄漏图片数大于 0", report["groups"]["leak_images"] > 0)

        # ---------- 精确重复 ----------
        check(
            "逐字节副本被判为精确重复",
            report["exact_group_total"] == 1,
            f"exact={report['exact_group_total']}",
        )

        # ---------- 干净数据不产生错误级问题 ----------
        check("干净数据的错误级问题为 0", report["summary"]["error"] == 0, str(report["summary"]))

        # ---------- 缓存 ----------
        second = datasetqc.inspect(root, "detect", threshold=12)
        check("第二次体检命中缓存", second["cache"]["hits"] == 13, str(second["cache"]))
        check(
            "缓存命中后结论一致",
            second["groups"]["total"] == report["groups"]["total"]
            and second["summary"] == report["summary"],
        )
        check(
            "缓存后更快",
            second["elapsed_ms"] <= report["elapsed_ms"] + 200,
            f"{report['elapsed_ms']} -> {second['elapsed_ms']}",
        )

        # ---------- 分组方式 ----------
        check(
            "prefix 取第一个下划线之前",
            dataset.group_key("images/train/01_defect_03.jpg", "prefix") == "01",
        )
        check(
            "dir 在标准 YOLO 布局下不退化成 images",
            dataset.group_key("images/train/01_defect_03.jpg", "dir") != "images",
        )
        check(
            "dir 在有子目录时取子目录名",
            dataset.group_key("images/train/lineA/01.jpg", "dir") == "lineA",
        )
        check("none 返回空组键", dataset.group_key("images/train/a.jpg", "none") == "")

        # ---------- 按组切分 ----------
        keys = [dataset.group_key(image.rel, "prefix") for image in dataset.load_project(str(root)).images]
        splits, notes = dataset.assign_splits_grouped(keys, 60, 20)
        by_group: dict[str, set[str]] = {}
        for key, split in zip(keys, splits):
            by_group.setdefault(key, set()).add(split)
        check("按组切分同组不跨划分", all(len(value) == 1 for value in by_group.values()), str(by_group))
        check("按组切分给出说明", any("按组划分" in note for note in notes), str(notes))

        # ---------- 导出 ----------
        out_default = work / "out_default"
        result_default = dataset.run_export(
            str(root), "detect", 70, 20, "yolo", str(out_default), False
        )
        check(
            "默认导出仍是按比例划分",
            result_default["split_mode"]["mode"] == "ratio",
            str(result_default["split_mode"]),
        )
        manifest = json.loads((out_default / "manifest.json").read_text(encoding="utf-8"))
        check("默认 manifest 没有 group_signature", "group_signature" not in manifest)
        check("默认 manifest 没有 qc 块", "qc" not in manifest)

        out_group = work / "out_group"
        result_group = dataset.run_export(
            str(root), "detect", 60, 20, "yolo", str(out_group), False,
            group_by="prefix", with_qc=True,
        )
        check(
            "按组导出记录 group 模式",
            result_group["split_mode"]["mode"] == "group:prefix",
            str(result_group["split_mode"]),
        )
        seen: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            for path in (out_group / "images" / split).glob("*"):
                seen.setdefault(path.name.split("_")[0], set()).add(split)
        check("导出后没有前缀跨划分", all(len(value) == 1 for value in seen.values()), str(seen))

        manifest_group = json.loads((out_group / "manifest.json").read_text(encoding="utf-8"))
        check("manifest 记录 group_signature", "group_signature" in manifest_group)
        check("manifest 记录 qc 摘要", "qc" in manifest_group)
        check(
            "qc 摘要与体检结论一致",
            manifest_group["qc"]["groups"]["total"] == report["groups"]["total"],
        )
        check(
            "qc 块不参与 unchanged 判定",
            "unchanged" in result_group and "qc" not in json.dumps(
                {"unchanged": result_group["unchanged"]}
            ),
        )

        # ---------- 坏标注 ----------
        broken = work / "broken"
        build_broken_dataset(broken)
        bad = datasetqc.inspect(broken, "detect", threshold=12)
        by_rule = {code: item["count"] for code, item in bad["by_rule"].items()}
        for code in (
            "LBL_FIELD",
            "LBL_PARSE",
            "LBL_CLASS_RANGE",
            "LBL_COORD_RANGE",
            "LBL_ZERO_AREA",
            "LBL_TINY",
            "LBL_HUGE",
            "LBL_BORDER",
            "LBL_DUP_BOX",
            "LBL_EMPTY",
            "LBL_MISSING",
        ):
            check(f"检出 {code}", by_rule.get(code, 0) > 0, f"by_rule={by_rule}")
        field_issue = next(
            (item for item in bad["issues"] if item["code"] == "LBL_FIELD"), None
        )
        check(
            "坏行能定位到行号与原始内容",
            bool(field_issue and field_issue["line"] == 1 and "0 0.5 0.5 0.1" in field_issue["raw"]),
            str(field_issue),
        )
        check("坏数据产生错误级问题", bad["summary"]["error"] > 0, str(bad["summary"]))
        check("问题列表可截断", datasetqc.inspect(broken, "detect", limit=1)["issues_truncated"])

        # ---------- 阈值与版本 ----------
        check("报告带算法版本", report["algo"] == datasetqc.ALGO)
        check("报告带阈值", report["threshold"] == 12)
        strict = datasetqc.inspect(root, "detect", threshold=0)
        check(
            "阈值收紧不会多分组",
            strict["groups"]["total"] <= report["groups"]["total"],
            f"0->{strict['groups']['total']} 12->{report['groups']['total']}",
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(_TMP_CACHE.parent, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败：{name}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
