# -*- coding: utf-8 -*-
"""像素级诊断的验收脚本，不需要 GPU / DLL / cv2。

这个脚本守的是「没有掩码也能说清定位好坏」这件事不被做歪：

- 判定线必须是 0.5 而不是 pixel_threshold：DLL 返回的 normalized_map 已经以
  pixel_threshold 为中心归一化（阈值处 = 0.5），拿 0.172 去比会把 40% 的像素
  判成异常，报告立刻变成"好板上到处是缺陷"；
- 「没有样本」与「全是 0」必须能区分：前者是 None，后者是 0.0，
  合并成 0 会让人以为测过了；
- 无掩码时 `auroc` 必须是 None：拿整图标签冒充像素标签算出来的数看着像精度，
  其实在惩罚模型"把好区域判正常"。

用法：
    python 6-web/backend/tools/test_anomaly_pixel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import anomalymetrics  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def near(a, b, eps: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= eps


def make_map(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def main() -> int:
    # ---- 1. 判定线：0.5，不是 pixel_threshold ----------------------------
    half = np.full((10, 10), 0.5, dtype=np.float32)
    check(
        "恰好 0.5 的像素算异常（>= 而不是 >）",
        near(anomalymetrics.pixel_ratio_above(half), 1.0),
        f"得到 {anomalymetrics.pixel_ratio_above(half)}",
    )
    check(
        "0.499 不算异常",
        near(anomalymetrics.pixel_ratio_above(np.full((4, 4), 0.499, np.float32)), 0.0),
    )
    check(
        "0.501 算异常",
        near(anomalymetrics.pixel_ratio_above(np.full((4, 4), 0.501, np.float32)), 1.0),
    )
    # 这条是防回归的核心：如果哪天有人把判定线改成 pixel_threshold（例如 0.172），
    # 一张 0.2 的图会被判成"整幅异常"，本断言立刻失败。
    check(
        "0.2 的图不会被当成整幅异常（判定线与 pixel_threshold 无关）",
        near(anomalymetrics.pixel_ratio_above(np.full((8, 8), 0.2, np.float32)), 0.0),
    )

    # ---- 2. 占比计算 -----------------------------------------------------
    check(
        "全 0 图占比 0",
        near(anomalymetrics.pixel_ratio_above(np.zeros((5, 5), np.float32)), 0.0),
    )
    check(
        "全 1 图占比 1",
        near(anomalymetrics.pixel_ratio_above(np.ones((5, 5), np.float32)), 1.0),
    )
    quarter = np.zeros((4, 4), np.float32)
    quarter[:2, :2] = 1.0
    check(
        "四分之一像素超标 → 0.25",
        near(anomalymetrics.pixel_ratio_above(quarter), 0.25),
        f"得到 {anomalymetrics.pixel_ratio_above(quarter)}",
    )

    # ---- 3. 无效输入一律返回 None，不抛异常 ------------------------------
    check("None 图返回 None", anomalymetrics.pixel_ratio_above(None) is None)
    check(
        "空数组返回 None",
        anomalymetrics.pixel_ratio_above(np.zeros((0, 0), np.float32)) is None,
    )
    check(
        "一维数组返回 None（内存布局搬错时不能瞎算）",
        anomalymetrics.pixel_ratio_above(np.zeros(100, np.float32)) is None,
    )
    check(
        "三维数组返回 None",
        anomalymetrics.pixel_ratio_above(np.zeros((4, 4, 3), np.float32)) is None,
    )

    # ---- 4. 好图乱指率 / 定位面积 ----------------------------------------
    samples = [
        {"label": 0, "pred": 0, "above": 0.0, "size": (256, 256)},
        {"label": 0, "pred": 0, "above": 0.0, "size": (256, 256)},
        {"label": 0, "pred": 0, "above": 0.02, "size": (256, 256)},
        {"label": 1, "pred": 1, "above": 0.10, "size": (256, 256)},
        {"label": 1, "pred": 1, "above": 0.30, "size": (256, 256)},
    ]
    result = anomalymetrics.pixel_diagnostics(samples)
    diag = result["diagnostic"]
    check("有异常图时 available=True", result["available"] is True)
    check("mode 标记为 diagnostic", result["mode"] == "diagnostic")
    check(
        "无掩码时 auroc 必须为 None（不拿整图标签冒充像素标签）",
        result["auroc"] is None,
    )
    check(
        "好图乱指率取正常图均值（(0+0+0.02)/3）",
        near(diag["normal_false_pixel_rate"], 0.00666667, 1e-6),
        str(diag["normal_false_pixel_rate"]),
    )
    check(
        "定位面积取异常图均值（(0.10+0.30)/2）",
        near(diag["abnormal_marked_rate"], 0.2),
        str(diag["abnormal_marked_rate"]),
    )
    check("正常 / 异常张数分别统计", diag["images"] == {"normal": 3, "abnormal": 2}, str(diag["images"]))
    check("异常图尺寸记入报告", diag["map_size"] == [256, 256], str(diag["map_size"]))
    check("判定线回显为 0.5", near(diag["cutoff"], 0.5))
    check("无跳过时 pixel_skipped=0", diag["pixel_skipped"] == 0)

    # ---- 5. 指不出位置 ---------------------------------------------------
    undirected = [
        {"label": 1, "pred": 1, "above": 0.0, "size": (64, 64)},
        {"label": 1, "pred": 1, "above": 0.05, "size": (64, 64)},
        {"label": 0, "pred": 0, "above": 0.0, "size": (64, 64)},
    ]
    diag2 = anomalymetrics.pixel_diagnostics(undirected)["diagnostic"]
    check(
        "判 NG 但异常图全无达标像素 → 指不出位置 1 张",
        diag2["abnormal_undirected"] == 1,
        str(diag2["abnormal_undirected"]),
    )
    check(
        "判 OK 且无达标像素不算「指不出位置」",
        anomalymetrics.pixel_diagnostics(
            [{"label": 0, "pred": 0, "above": 0.0, "size": (8, 8)}]
        )["diagnostic"]["abnormal_undirected"]
        == 0,
    )
    check(
        "正常图被误报 NG 且无边也会计入（同样是撑不住的结论）",
        anomalymetrics.pixel_diagnostics(
            [{"label": 0, "pred": 1, "above": 0.0, "size": (8, 8)}]
        )["diagnostic"]["abnormal_undirected"]
        == 1,
    )

    # ---- 6. 样本缺失 / 跳过 ----------------------------------------------
    empty = anomalymetrics.pixel_diagnostics([])
    check("没有任何样本时 available=False", empty["available"] is False)
    check("没有任何样本时给出原因", bool(empty.get("reason")), str(empty))
    check("没有任何样本时 diagnostic 为 None", empty["diagnostic"] is None)

    all_null = anomalymetrics.pixel_diagnostics(
        [{"label": 0, "pred": 0, "above": None, "size": (0, 0)}]
    )
    check("样本全无有效异常图时 available=False", all_null["available"] is False)
    check("无有效异常图与「没有样本」的原因不同", all_null["reason"] != empty["reason"])

    skipped = anomalymetrics.pixel_diagnostics(samples, skipped=3)
    check("跳过张数原样带出", skipped["diagnostic"]["pixel_skipped"] == 3)
    check("有跳过时 reason 说明张数", "3" in (skipped.get("reason") or ""), str(skipped.get("reason")))

    # ---- 7. 只有正常图 / 只有异常图 --------------------------------------
    only_normal = anomalymetrics.pixel_diagnostics(
        [{"label": 0, "pred": 0, "above": 0.01, "size": (32, 32)}]
    )["diagnostic"]
    check(
        "只有正常图时定位面积为 None（不是 0，避免误读成「定位很准」）",
        only_normal["abnormal_marked_rate"] is None,
        str(only_normal["abnormal_marked_rate"]),
    )
    only_abnormal = anomalymetrics.pixel_diagnostics(
        [{"label": 1, "pred": 1, "above": 0.2, "size": (32, 32)}]
    )["diagnostic"]
    check(
        "只有异常图时好图乱指率为 None",
        only_abnormal["normal_false_pixel_rate"] is None,
        str(only_abnormal["normal_false_pixel_rate"]),
    )

    # ---- 8. 尺寸不统一要说出来 -------------------------------------------
    mixed = anomalymetrics.pixel_diagnostics([
        {"label": 0, "pred": 0, "above": 0.0, "size": (256, 256)},
        {"label": 1, "pred": 1, "above": 0.1, "size": (512, 512)},
    ])
    check("尺寸不统一时给出原因", "尺寸" in (mixed.get("reason") or ""), str(mixed.get("reason")))

    # ---- 9. 与 DLL 同源的模拟：0.5 中心的两张图 --------------------------
    # 好图：全在 0.5 以下；异常图：一小块高于 0.5 —— 这是 DLL 归一化后的真实形态
    good_map = make_map(np.full((16, 16), 0.30))
    bad_map = make_map(np.full((16, 16), 0.30))
    bad_map[4:8, 4:8] = 0.85  # 16 个像素 / 256 = 6.25%
    good_ratio = anomalymetrics.pixel_ratio_above(good_map)
    bad_ratio = anomalymetrics.pixel_ratio_above(bad_map)
    check("好图模拟：乱指率 0", near(good_ratio, 0.0), str(good_ratio))
    check("异常图模拟：定位面积 6.25%", near(bad_ratio, 0.0625), str(bad_ratio))
    simulated = anomalymetrics.pixel_diagnostics([
        {"label": 0, "pred": 0, "above": good_ratio, "size": (16, 16)},
        {"label": 1, "pred": 1, "above": bad_ratio, "size": (16, 16)},
    ])
    check(
        "模拟场景下 available=True 且无硬伤",
        simulated["available"] is True
        and simulated["diagnostic"]["abnormal_undirected"] == 0,
        str(simulated.get("diagnostic")),
    )

    print()
    total = len(PASSED) + len(FAILED)
    if FAILED:
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print(f"全部通过：{total} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
