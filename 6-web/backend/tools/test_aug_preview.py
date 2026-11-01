# -*- coding: utf-8 -*-
"""数据增强预览（augpreview）验收脚本，不需要启动 Web 服务。

预览是「像素级近似」，但近似也得守住几条硬底线，否则用户会被误导：

1. 同 seed + 同参数 + 同一张图 ⇒ 字节完全一致（否则「换个种子对比」这件事没意义）；
2. 参数全 0 ⇒ 增强图与原图逐像素相同（说明没有任何隐藏变换偷偷生效）；
3. 概率参数为 0 时绝不能触发（``applied`` 与像素两头都要对得上）；
4. 输出尺寸必须等于 imgsz（训练输入是正方形，预览也得是）；
5. ``copy_paste`` / ``close_mosaic`` 不参与预览，但要在响应里说明理由；
6. 越界参数直接报错，不要默默截断。

用法：
    python 6-web/backend/tools/test_aug_preview.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 允许直接以脚本方式运行（把 backend 加入 sys.path）
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import augpreview  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def decode(uri: str) -> np.ndarray:
    raw = base64.b64decode(uri.split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def build_dataset(root: Path, count: int = 4) -> Path:
    """造一个小训练集：颜色块 + 一个白框，方便看出几何/颜色变换。"""
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for index in range(count):
        # 用彩色底而不是灰色底：灰色（饱和度=0）做色调抖动看不出任何变化，
        # 会让「hsv 是否生效」这类断言失去意义。
        canvas = np.full((80, 120, 3), (30 + index * 10, 90 + index * 15, 200), dtype=np.uint8)
        cv2.rectangle(canvas, (20 + index * 5, 20), (70, 60), (255, 255, 255), 2)
        cv2.circle(canvas, (95, 40), 12, (40, 220, 180), -1)
        # 用 imencode + tofile 写盘：中文路径下 cv2.imwrite 会失败
        ok, buffer = cv2.imencode(".jpg", canvas)
        assert ok
        buffer.tofile(str(images_dir / f"img{index}.jpg"))
        (labels_dir / f"img{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/train",
                "nc: 1",
                "names:",
                "  0: defect",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def neutral_params() -> dict[str, str]:
    """所有能产生像素差别的参数都置 0（copy_paste / close_mosaic 本就不参与）。"""
    return {
        "hsv_h": "0",
        "hsv_s": "0",
        "hsv_v": "0",
        "degrees": "0",
        "translate": "0",
        "scale": "0",
        "shear": "0",
        "perspective": "0",
        "flipud": "0",
        "fliplr": "0",
        "bgr": "0",
        "mosaic": "0",
        "mixup": "0",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        yaml_path = build_dataset(root)

        # ---- 1. 同 seed 完全一致 -------------------------------------------
        params = {"mosaic": "1.0", "fliplr": "0.5", "degrees": "10", "hsv_v": "0.3"}
        first = augpreview.render(yaml_path, params, index=0, seed=1234, imgsz=256)
        second = augpreview.render(yaml_path, params, index=0, seed=1234, imgsz=256)
        check("同 seed 增强图字节一致", first["augmented"] == second["augmented"])
        check("同 seed 原图字节一致", first["origin"] == second["origin"])

        third = augpreview.render(yaml_path, params, index=0, seed=4321, imgsz=256)
        check("不同 seed 结果不同", third["augmented"] != first["augmented"])

        # ---- 2. 尺寸 -------------------------------------------------------
        check("增强图尺寸等于 imgsz", decode(first["augmented"]).shape == (256, 256, 3), str(decode(first["augmented"]).shape))
        check("原图尺寸等于 imgsz", decode(first["origin"]).shape == (256, 256, 3))
        long_side = augpreview.render(yaml_path, {}, index=0, seed=1, imgsz=320)
        check("非方形原图也被 letterbox 成正方形", decode(long_side["origin"]).shape == (320, 320, 3))

        # ---- 3. 参数全 0 ⇒ 与原图逐像素相同 --------------------------------
        neutral = augpreview.render(yaml_path, neutral_params(), index=1, seed=7, imgsz=192)
        check("参数全 0 时增强图 == 原图", neutral["augmented"] == neutral["origin"])
        check("参数全 0 时没有任何 applied", neutral["applied"] == [], str(neutral["applied"]))
        check(
            "参数全 0 时 mosaic/mixup 在 skipped 里说明原因",
            {item["name"] for item in neutral["skipped"]} >= {"mosaic", "mixup"},
            str([item["name"] for item in neutral["skipped"]]),
        )

        # ---- 4. 概率参数为 0 绝不触发 --------------------------------------
        only_flip = dict(neutral_params())
        only_flip["fliplr"] = "1.0"
        flipped = augpreview.render(yaml_path, only_flip, index=0, seed=5, imgsz=192)
        check("概率 1.0 时 flip 必定生效", any(item["name"] == "fliplr" for item in flipped["applied"]))
        expect = cv2.flip(decode(flipped["origin"]), 1)
        # 不能要求逐字节相同：JPEG 是 8×8 分块有损编码，「先翻转后编码」与
        # 「先解码再翻转」的量化误差不在同一位置，像素会有 1~2 级的正常差异。
        diff = np.abs(decode(flipped["augmented"]).astype(np.int16) - expect.astype(np.int16))
        check("flip 的像素结果等于翻转原图", diff.mean() < 2.0, f"平均差 {diff.mean():.2f}")

        # ---- 5. 几何 / 颜色参数确实改像素 ----------------------------------
        rotated = augpreview.render(yaml_path, {**neutral_params(), "degrees": "45"}, index=0, seed=2, imgsz=192)
        check("旋转改变像素", rotated["augmented"] != rotated["origin"])

        colored = augpreview.render(yaml_path, {**neutral_params(), "hsv_h": "1.0"}, index=0, seed=3, imgsz=192)
        check("色调抖动改变像素", colored["augmented"] != colored["origin"])

        mix = augpreview.render(yaml_path, {**neutral_params(), "mixup": "1.0"}, index=0, seed=4, imgsz=192)
        check("mixup=1.0 必定生效", any(item["name"] == "mixup" for item in mix["applied"]))

        # ---- 6. 不参与预览的参数要说明 -------------------------------------
        check(
            "copy_paste / close_mosaic 出现在 ignored 里",
            {item["name"] for item in first["ignored"]} == {"copy_paste", "close_mosaic"},
            str([item["name"] for item in first["ignored"]]),
        )
        check(
            "ignored 里带原因",
            all(item["detail"] for item in first["ignored"]),
        )

        # ---- 7. 参数校验 ---------------------------------------------------
        try:
            augpreview.normalize_params({"hsv_h": "5"})
            check("越界参数被拒", False, "未抛异常")
        except augpreview.PreviewError:
            check("越界参数被拒", True)

        normalized, fell_back = augpreview.normalize_params({"hsv_h": "abc", "mosaic": ""})
        check("非数字回落默认值并记录", fell_back == ["hsv_h"], str(fell_back))
        check("空串按默认值处理", normalized["mosaic"] == augpreview.DEFAULTS["mosaic"], str(normalized["mosaic"]))
        check(
            "默认值与 launcher 的 mosaic 默认一致（1.0）",
            augpreview.DEFAULTS["mosaic"] == 1.0 and augpreview.DEFAULTS["fliplr"] == 0.5,
            str(augpreview.DEFAULTS),
        )

        # ---- 8. 索引与样本数 -----------------------------------------------
        wrapped = augpreview.render(yaml_path, neutral_params(), index=9, seed=1, imgsz=128)
        check("index 取模而不是报错", wrapped["index"] == 1 and wrapped["total"] == 4, f"{wrapped['index']}/{wrapped['total']}")
        check("响应里回显实际 seed", first["seed"] == 1234, str(first["seed"]))
        check("响应里回显原图路径", first["source"].endswith("img0.jpg"), first["source"])

        # ---- 9. 空目录 -----------------------------------------------------
        empty_root = Path(tmp) / "empty"
        (empty_root / "images" / "train").mkdir(parents=True)
        empty_yaml = empty_root / "dataset.yaml"
        empty_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: [a]\n", encoding="utf-8")
        try:
            augpreview.render(empty_yaml, {}, index=0, seed=1)
            check("空训练集给出可读错误", False, "未抛异常")
        except Exception as exc:  # noqa: BLE001 - 这里就要它抛
            check("空训练集给出可读错误", "没有可用图片" in str(exc), str(exc))

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
