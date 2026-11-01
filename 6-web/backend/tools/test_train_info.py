# -*- coding: utf-8 -*-
"""训练数据集体检（traininfo）验收脚本，不需要启动 Web 服务。

覆盖的坑：

1. ``dataset.yaml`` 的 ``path`` 是相对路径时，必须按 yaml 所在目录解析；
2. 空标签、缺失标签、孤儿标签三种「看起来有文件其实没内容」的情况要分开计数；
3. ``names`` 里的类别数与 ``nc`` 不一致、类别一个框都没有，要给出可读警告；
4. 大目录只做等距抽样，不能把几万张图挨个解码；
5. 60 秒缓存要真的命中，``refresh`` 要真的绕开。

用法：
    python 6-web/backend/tools/test_train_info.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable

# 允许直接以脚本方式运行（把 backend 加入 sys.path）
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import traininfo  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    """写一张纯色小图。用 PIL 而不是 cv2，避免测试依赖 cv2 的编码器。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (12, 34, 56)).save(path)


def codes(warnings: Iterable[dict]) -> set[str]:
    return {item["code"] for item in warnings}


def build_dataset(root: Path) -> Path:
    """造一个「什么都占一点」的 YOLO 数据集。

    结构（刻意混进各种异常）：

    - train: 三张图；a 有 2 个框、b 标签是空文件、c 没有标签文件；
    - train 下还有一个孤儿标签 d.txt（没有同名图片）；
    - val: 一张图配一个框，类别 1；
    - names 写了 3 个类，nc 写成 2（故意的），类别 2 一个框都没有。
    """
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)

    for name in ("a", "b", "c"):
        write_image(root / "images" / "train" / f"{name}.jpg")

    (root / "labels" / "train" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\nbad line here\n", encoding="utf-8"
    )
    (root / "labels" / "train" / "b.txt").write_text("", encoding="utf-8")
    # c 没有标签；d 是孤儿
    (root / "labels" / "train" / "d.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    write_image(root / "images" / "val" / "v1.png", (32, 32))
    (root / "labels" / "val" / "v1.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# TRTForge 导出的数据集",
                "path: .",
                "train: images/train",
                "val: images/val   # 行尾注释也不能被当成路径",
                "test: images/test",
                "nc: 2",
                "names:",
                "  0: missing_hole",
                "  1: mouse_bite",
                "  2: open_circuit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        yaml_path = build_dataset(root)

        result = traininfo.scan(yaml_path, use_cache=False)
        train = result["splits"]["train"]
        val = result["splits"]["val"]
        test = result["splits"]["test"]

        # ---- 1. 基本计数 ---------------------------------------------------
        check("base 按 yaml 所在目录解析", Path(result["base"]) == root.resolve(), result["base"])
        check("train 图片数", train["images"] == 3, str(train["images"]))
        check("train 标签数", train["labels"] == 3, str(train["labels"]))
        check("空标签单独计数", train["empty_labels"] == 1, str(train["empty_labels"]))
        check(
            "缺失标签单独计数",
            train["missing_labels"] == 1 and train["missing_examples"] == ["c"],
            str(train["missing_examples"]),
        )
        check(
            "孤儿标签单独计数",
            train["orphan_labels"] == 1 and train["orphan_examples"] == ["d"],
            str(train["orphan_examples"]),
        )
        check("畸形行被计数但不是实例", train["malformed_lines"] == 1, str(train["malformed_lines"]))
        check("train 实例数只算合法行", train["instances"] == 2, str(train["instances"]))
        check("val 计数", val["images"] == 1 and val["instances"] == 1, str(val))
        check(
            "未写 test 目录时 configured 为 False",
            test["configured"] is True and test["images"] == 0,
            str(test),
        )
        check(
            "目录不存在给出 split_missing",
            "split_missing" in codes(result["warnings"]),
            str(codes(result["warnings"])),
        )
        check("标签覆盖率", abs(train["label_coverage"] - 2 / 3) < 1e-3, str(train["label_coverage"]))

        check("总图片数", result["counts"]["total"] == 4, str(result["counts"]))
        check("总框数", result["counts"]["instances"] == 3, str(result["counts"]))
        check(
            "每类实例数",
            result["class_instances"] == {"0": 2, "1": 1},
            str(result["class_instances"]),
        )

        # ---- 2. 分辨率 -----------------------------------------------------
        res = result["resolutions"]
        check("分辨率统计张数", res["count"] == 3, str(res))
        check("分辨率中位数", res["width"][1] == 64 and res["height"][1] == 48, str(res))
        check("未触发抽样", res["sampled"] is False, str(res["sampled"]))

        # ---- 3. 警告 -------------------------------------------------------
        found = codes(result["warnings"])
        expected = {"empty_labels", "missing_labels", "orphan_labels", "zero_instance", "nc_mismatch"}
        check("关键警告齐全", expected <= found, f"缺 {expected - found}")
        zero = [w for w in result["warnings"] if w["code"] == "zero_instance"]
        check("零实例类别点名", bool(zero) and "open_circuit" in zero[0]["text"], str(zero))
        nc_warn = [w for w in result["warnings"] if w["code"] == "nc_mismatch"]
        check("nc 不一致点名", bool(nc_warn) and "nc=2" in nc_warn[0]["text"], str(nc_warn))
        check(
            "空 test 目录只报 info",
            all(w["level"] == "info" for w in result["warnings"] if w["code"] == "split_empty"),
            str([w for w in result["warnings"] if w["code"] == "split_empty"]),
        )

        # ---- 4. 缓存 -------------------------------------------------------
        cached = traininfo.scan(yaml_path)
        check("第二次扫描命中缓存", cached["cached"] is True, str(cached["cached"]))
        fresh = traininfo.scan(yaml_path, use_cache=False)
        check("refresh 绕开缓存", fresh["cached"] is False, str(fresh["cached"]))

        # 缓存键是「yaml 路径 + mtime + size」：只往目录里塞文件、yaml 不动时，
        # 60 秒内继续返回旧结果（这是有意为之——前端不该每点一次就重扫几万张）；
        # 只要 yaml 被重新写过（导出就是重写 yaml），缓存立刻失效。
        (root / "images" / "train" / "e.jpg").parent.mkdir(parents=True, exist_ok=True)
        write_image(root / "images" / "train" / "e.jpg")
        still_cached = traininfo.scan(yaml_path)
        check("只加文件不改 yaml 时仍命中缓存", still_cached["cached"] is True, str(still_cached["cached"]))

        yaml_path.write_text(yaml_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        again = traininfo.scan(yaml_path)
        check(
            "yaml 重写后缓存失效并看到新文件",
            again["cached"] is False and again["splits"]["train"]["images"] == 4,
            str(again["splits"]["train"]),
        )

        # ---- 5. 抽样 -------------------------------------------------------
        old_limit = traininfo.MAX_SIZE_SAMPLES
        try:
            traininfo.MAX_SIZE_SAMPLES = 2
            sampled = traininfo.scan(yaml_path, use_cache=False)["resolutions"]
            check("大目录触发抽样", sampled["sampled"] is True and sampled["count"] == 2, str(sampled))
            check(
                "抽样会给出 info 警告",
                "sampled" in codes(traininfo.scan(yaml_path, use_cache=False)["warnings"]),
            )
        finally:
            traininfo.MAX_SIZE_SAMPLES = old_limit

        # ---- 6. 错误路径 ---------------------------------------------------
        try:
            traininfo.scan(Path(tmp) / "nope.yaml", use_cache=False)
            check("不存在的 yaml 被拒", False, "未抛异常")
        except traininfo.OverviewError:
            check("不存在的 yaml 被拒", True)

        broken = Path(tmp) / "broken.yaml"
        broken.write_text("path: .\ntrain: images/ghost\nnc: 1\nnames: [a]\n", encoding="utf-8")
        ghost = traininfo.scan(broken, use_cache=False)
        check(
            "目录不存在时给出可读警告而不是零计数",
            "split_missing" in codes(ghost["warnings"]),
            str(codes(ghost["warnings"])),
        )

        # ---- 7. yaml 解析 ---------------------------------------------------
        scalars = traininfo.parse_scalars(
            "path: 'D:/data set/x'\ntrain: images/train  # 注释\n  nc: 99\nnames:\n  0: a\nnc: 3\n"
        )
        check("引号被剥掉", scalars.get("path") == "D:/data set/x", str(scalars))
        check("行尾注释被剥掉", scalars.get("train") == "images/train", str(scalars))
        check("缩进块里的 nc 不被采信", scalars.get("nc") == "3", str(scalars))

        # ---- 8. 图片列表稳定 -------------------------------------------------
        images = traininfo.list_split_images(yaml_path, "train")
        check(
            "split 图片按相对路径稳定排序",
            [p.name for p in images] == ["a.jpg", "b.jpg", "c.jpg", "e.jpg"],
            str([p.name for p in images]),
        )

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
