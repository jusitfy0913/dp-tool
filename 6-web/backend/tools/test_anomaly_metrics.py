# -*- coding: utf-8 -*-
"""异常检测精度口径的验收脚本，不需要 GPU / DLL / cv2。

这个模块的每一条断言都在守「数字看着合理但其实错了」：
- AUROC 用错方向（把 1 当成正常）：模型越好分数越低，报告却一片绿；
- 同分样本按先后顺序定秩：同一批图换个文件名顺序，AUROC 就变了；
- 阈值判定用 `>` 而不是 `>=`：恰好等于阈值的图被判 OK，漏检率虚低；
- 单一类别时返回 0.5：被当成「模型只会瞎猜」，比"没有这个数"更误导。

用法：
    python 6-web/backend/tools/test_anomaly_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> int:
    # ---- 1. 归一化与判定口径（必须与 DLL 一致）----------------------------
    check(
        "value == threshold 归一化为 0.5",
        near(anomalymetrics.normalize_with_range(0.5, 0.5, 0.0, 1.0), 0.5),
    )
    check(
        "低于阈值归一化 < 0.5",
        anomalymetrics.normalize_with_range(0.2, 0.5, 0.0, 1.0) < 0.5,
    )
    check(
        "高于阈值归一化 > 0.5",
        anomalymetrics.normalize_with_range(0.8, 0.5, 0.0, 1.0) > 0.5,
    )
    check(
        "极端值被 clamp 到 [0,1]",
        near(anomalymetrics.normalize_with_range(1e9, 0.5, 0.0, 1.0), 1.0)
        and near(anomalymetrics.normalize_with_range(-1e9, 0.5, 0.0, 1.0), 0.0),
    )
    check(
        "max == min 时退化为 clamp（不除零）",
        near(anomalymetrics.normalize_with_range(0.7, 0.5, 0.3, 0.3), 0.7),
        f"得到 {anomalymetrics.normalize_with_range(0.7, 0.5, 0.3, 0.3)}",
    )
    check(
        "NaN 归一化为 0（与 C++ clamp01 一致）",
        near(anomalymetrics.normalize_with_range(float("nan"), 0.5, 0.0, 1.0), 0.0),
    )

    # ---- 2. AUROC --------------------------------------------------------
    check(
        "完美分离 AUROC = 1",
        near(anomalymetrics.image_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0),
    )
    check(
        "完全反向 AUROC = 0",
        near(anomalymetrics.image_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0),
    )
    check(
        "全部同分 AUROC = 0.5",
        near(anomalymetrics.image_auroc([0.5, 0.5, 0.5], [0, 1, 1]), 0.5),
    )
    check(
        "跨类别同分按平均秩 = 0.5",
        near(anomalymetrics.image_auroc([0.5, 0.5], [0, 1]), 0.5),
    )
    # 手算：正常 0.1/0.4 秩 1/3，异常 0.35/0.8/0.9 秩 2/4/5
    # AUC = (2+4+5 - 3*4/2) / (3*2) = 5/6
    check(
        "混合样本 AUROC = 5/6（手算核对）",
        near(
            anomalymetrics.image_auroc([0.1, 0.4, 0.35, 0.8, 0.9], [0, 0, 1, 1, 1]),
            5 / 6,
        ),
        f"得到 {anomalymetrics.image_auroc([0.1, 0.4, 0.35, 0.8, 0.9], [0, 0, 1, 1, 1])}",
    )
    check(
        "只有正常图时 AUROC = None（不是 0.5）",
        anomalymetrics.image_auroc([0.1, 0.2], [0, 0]) is None,
    )
    check(
        "只有异常图时 AUROC = None",
        anomalymetrics.image_auroc([0.1, 0.2], [1, 1]) is None,
    )
    check(
        "长度不一致时 AUROC = None",
        anomalymetrics.image_auroc([0.1, 0.2], [1]) is None,
    )
    check(
        "空输入 AUROC = None",
        anomalymetrics.image_auroc([], []) is None,
    )
    check(
        "AUROC 与输入顺序无关",
        near(
            anomalymetrics.image_auroc([0.1, 0.4, 0.35, 0.8, 0.9], [0, 0, 1, 1, 1]),
            anomalymetrics.image_auroc([0.9, 0.35, 0.1, 0.8, 0.4], [1, 1, 0, 1, 0]),
        ),
    )

    # ---- 3. 阈值下混淆矩阵 -------------------------------------------------
    stats = anomalymetrics.confusion_at(
        [0.6, 0.4, 0.55, 0.2], [1, 1, 0, 0], 0.5
    )
    check("TP/FP/TN/FN 计数正确",
          (stats["tp"], stats["fp"], stats["tn"], stats["fn"]) == (1, 1, 1, 1),
          str(stats))
    check("阈值 0.5 时 P=R=F1=0.5",
          near(stats["precision"], 0.5) and near(stats["recall"], 0.5) and near(stats["f1"], 0.5))
    check("误报率与漏检率各 0.5",
          near(stats["fpr"], 0.5) and near(stats["fnr"], 0.5))
    check("准确率 0.5", near(stats["accuracy"], 0.5))

    boundary = anomalymetrics.confusion_at([0.5], [1], 0.5)
    check(
        "分数恰好等于阈值判为异常（>= 而非 >）",
        boundary["tp"] == 1 and boundary["fn"] == 0,
        str(boundary),
    )
    perfect = anomalymetrics.confusion_at([0.1, 0.9], [0, 1], 0.5)
    check(
        "完美判定时 F1 = 1 且误报/漏检为 0",
        near(perfect["f1"], 1.0) and near(perfect["fpr"], 0.0) and near(perfect["fnr"], 0.0),
    )
    empty = anomalymetrics.confusion_at([0.1], [0], 0.5)
    check(
        "没有任何异常样本时 recall/F1 为 0（不抛异常）",
        near(empty["recall"], 0.0) and near(empty["f1"], 0.0),
        str(empty),
    )

    # ---- 4. 最优阈值 ------------------------------------------------------
    best = anomalymetrics.best_f1_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    check(
        "可完美分离时最优 F1 = 1",
        best is not None and near(best["f1"], 1.0),
        str(best),
    )
    check(
        "最优阈值落在两类之间的空档里",
        best is not None and 0.2 < best["threshold"] < 0.8,
        str(best),
    )
    check(
        "最优阈值同 F1 时优先更低的误报率",
        best is not None and near(best["fpr"], 0.0),
        str(best),
    )
    check(
        "单一类别时最优阈值为 None",
        anomalymetrics.best_f1_threshold([0.1, 0.2], [0, 0]) is None,
    )

    # ---- 5. 直方图像素 AUROC ----------------------------------------------
    check(
        "正负像素完全分离时 AUROC = 1",
        near(
            anomalymetrics.pixel_auroc_from_histograms([0, 0, 0, 10], [10, 0, 0, 0]),
            1.0,
        ),
        f"得到 {anomalymetrics.pixel_auroc_from_histograms([0, 0, 0, 10], [10, 0, 0, 0])}",
    )
    check(
        "正负分布相同时 AUROC = 0.5",
        near(
            anomalymetrics.pixel_auroc_from_histograms([5, 5, 0, 0], [5, 5, 0, 0]),
            0.5,
        ),
        f"得到 {anomalymetrics.pixel_auroc_from_histograms([5, 5, 0, 0], [5, 5, 0, 0])}",
    )
    check(
        "没有正像素时 AUROC = None",
        anomalymetrics.pixel_auroc_from_histograms([0, 0], [1, 1]) is None,
    )
    check(
        "没有负像素时 AUROC = None",
        anomalymetrics.pixel_auroc_from_histograms([1, 1], [0, 0]) is None,
    )

    print()
    total = len(PASSED) + len(FAILED)
    if FAILED:
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print(f"通过 {total} 项，失败 0 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
