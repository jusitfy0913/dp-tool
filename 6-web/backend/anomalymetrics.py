# -*- coding: utf-8 -*-
"""异常检测（Anomalib）的精度口径：图像级 AUROC / F1 / 误报率 / 漏检率，
以及无需 ground_truth 掩码的像素级诊断（乱指率 / 定位面积 / 指不出位置）。

## 为什么不复刻 anomalib 自己的 metric

Anomalib 的 `AnomalyScoreThreshold` / `AUROC` 是挂在 Lightning 训练流程上的，
要在编排末尾算一次就得再起一个 Python 进程加载 torch 与整个训练集——
而这一刻我们手里只有「引擎吐出来的分数」这一件事要评。
本模块只吃两个数组：分数与 0/1 标签，纯 numpy、无 GPU、无 DLL，可直接单测。

## 判定口径必须与 DLL 一致

DLL（`anomalib_infer_c.cpp:121`、`:508`）的判定是：

    normalized = clamp01(((raw - image_threshold) / (score_max - score_min)) + 0.5)
    is_anomaly = normalized >= 0.5

也就是说「raw >= image_threshold 即判 NG」，而 `normalized` 只是给人看的分值。
如果这里改用「raw >= 0.5」之类想当然的阈值，报告上的漏检率会和页面上单张推理的
判定**互相矛盾**——同一个模型、同一张图，一边 NG 一边 OK，用户只会不知道该信谁。
所以判定一律走 `normalize_with_range`，不自己发明。

## AUROC 用秩统计而不是扫阈值

逐样本扫阈值是 O(n²)，且分数同分时要额外处理。秩统计（Mann-Whitney U）
把 AUROC 变成一次排序：同分取平均秩，天然处理并存分数，复杂度 O(n log n)。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def clamp01(value: float) -> float:
    """与 C++ 的 `clamp01` 一致：NaN 归 0，超界贴合端点。"""
    if value != value:  # NaN
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def normalize_with_range(
    value: float, threshold: float, min_value: float, max_value: float
) -> float:
    """复刻 DLL 的分数归一化：`((value - threshold) / (max - min)) + 0.5`。

    `max == min`（分数饱和时很常见）直接 clamp 原值，不做除法——
    DLL 那边也是这么兜底的（`std::abs(denom) < 1e-6`）。
    """
    denominator = float(max_value) - float(min_value)
    if abs(denominator) < 1e-6:
        return clamp01(float(value))
    return clamp01(((float(value) - float(threshold)) / denominator) + 0.5)


def image_auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """图像级 AUROC（label 1 = 异常）。只有单一类时返回 None。

    返回 None 而不是 0.5 的原因：全正常或全异常时 AUROC 没有定义，
    给个 0.5 会被当成「模型只会瞎猜」，比"没有这个数"更有误导性。
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.size != s.size:
        return None
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None

    # 平均秩：同分样本共享秩，避免"谁排前面"影响结果
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    index = 0
    while index < s.size:
        end = index
        while end + 1 < s.size and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        # 秩从 1 开始，同分取平均
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1

    positive_rank_sum = float(ranks[y == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def confusion_at(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    """在给定原始分数阈值下的混淆矩阵与派生指标。

    `threshold` 是**原始分数**阈值（metadata 的 `image_threshold`），比较方式
    `score >= threshold` 与 DLL 的 `normalized >= 0.5` 等价。
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    predicted = s >= float(threshold)
    actual = y == 1

    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    # 误报率 = 正常被判异常的比例；漏检率 = 异常被判正常的比例
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "accuracy": round(accuracy, 6),
    }


def best_f1_threshold(
    scores: Sequence[float], labels: Sequence[int]
) -> dict[str, Any] | None:
    """扫全部候选阈值取最大 F1。

    候选取「每个唯一分数」与「唯一分数之间的中点」：异常判定是 `>=`，
    阈值落在一段空档里时结果与落在右端样本上完全一样，中点只是把
    「可以调到 0.73 还是 0.74」这件事说得更准确一点。
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0 or y.size != s.size:
        return None
    if not np.any(y == 1) or not np.any(y == 0):
        return None

    unique = np.unique(s)
    candidates = [float(unique[0]) - 1e-6]
    for left, right in zip(unique[:-1], unique[1:]):
        candidates.append(float((left + right) / 2.0))
    candidates.append(float(unique[-1]))

    best: dict[str, Any] | None = None
    for threshold in candidates:
        stats = confusion_at(s, y, threshold)
        # 同分时优先更小的误报率：同一个 F1，产线上更愿意少报
        if best is None or (stats["f1"], -stats["fpr"]) > (best["f1"], -best["fpr"]):
            best = stats
    return best


def pixel_auroc_from_histograms(
    positive_hist: Iterable[float],
    negative_hist: Iterable[float],
    bin_edges: Sequence[float] | None = None,
) -> float | None:
    """由「异常像素 / 正常像素」两张同桶直方图算像素级 AUROC。

    为什么用直方图而不是留全部像素：一张 1024×1024 的图有一百万个 float，
    几百张图就是几百 MB；直方图把内存压成固定桶数，代价是分辨率下降，
    对"这个模型大概能不能定位"这个用途完全够。像素级指标在本版只作参考，
    主指标是图像级。
    """
    pos = np.asarray(list(positive_hist), dtype=np.float64)
    neg = np.asarray(list(negative_hist), dtype=np.float64)
    if pos.size == 0 or pos.size != neg.size:
        return None
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return None

    # 从高到低累加：以「分数 >= 第 i 桶」为阈值时的 TPR / FPR
    tpr = np.cumsum(pos[::-1]) / total_pos
    fpr = np.cumsum(neg[::-1]) / total_neg
    # 曲线下面积用梯形法，并在两端补 (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    area = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return max(0.0, min(1.0, area))


# ---------------------------------------------------------------------------
# 像素级诊断（无需 ground_truth 掩码）
#
# ## 为什么判定线是 0.5 而不是 pixel_threshold
#
# DLL（`anomalib_infer_c.cpp:479`）在返回异常图之前已经做过一次归一化：
#
#     normalizedMap = ((rawMap - pixel_threshold) / (mapMax - mapMin)) + 0.5
#
# 也就是说「恰好等于 pixel_threshold 的像素」在返回的图上就是 **0.5**，
# 高于阈值 → > 0.5，低于阈值 → < 0.5（并夹到 [0,1]）。DLL 自己生成缺陷掩码时
# 也是 `threshold(displayMap, mask, 0.5, ...)`（`:499`）。
#
# 所以这里必须用 0.5 当判定线。若照字面拿 `pixel_threshold` 去比（比如 0.172），
# 会把几乎所有像素都算成异常，得出"好板上 40% 是缺陷"这种荒谬结论——
# 与图像级那个 `normalized >= 0.5` 的坑是同一个，只是换到了像素上。
# ---------------------------------------------------------------------------

PIXEL_MAP_CUTOFF = 0.5


def pixel_ratio_above(
    anomaly_map: Any, cutoff: float = PIXEL_MAP_CUTOFF
) -> float | None:
    """归一化异常图里被判为异常的像素占比；拿不到有效图时返回 None。

    `anomaly_map` 是 DLL 返回的 `normalized_map`（float32 二维，已以
    `pixel_threshold` 为中心归一化），判定线固定 `>= 0.5`，见上方说明。
    """
    if anomaly_map is None:
        return None
    values = np.asarray(anomaly_map, dtype=np.float64)
    if values.size == 0:
        return None
    if values.ndim != 2:
        # 一维或三维说明搬错了内存布局，宁可不算也不要给一个错的占比
        return None
    above = int(np.count_nonzero(values >= float(cutoff)))
    return above / float(values.size)


def _mean_or_none(values: Sequence[float]) -> float | None:
    """均值；空列表返回 None（不是 0）——"没有样本"和"全是 0"必须能区分。"""
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return round(sum(items) / len(items), 6)


def pixel_diagnostics(
    samples: Sequence[dict],
    skipped: int = 0,
    cutoff: float = PIXEL_MAP_CUTOFF,
) -> dict:
    """由逐图统计汇总像素级诊断结论。

    `samples` 每项形如：
        {"label": 0 | 1, "pred": 0 | 1, "above": float | None, "size": (w, h)}

    其中 `label` 是人工标签（0 正常 / 1 异常），`pred` 是引擎判定，
    `above` 是该图异常图中被判为异常的像素占比（`pixel_ratio_above` 的结果）。

    三个指标回答三个不同问题：
      - `normal_false_pixel_rate`：好板上被乱指的比例。期望接近 0。
      - `abnormal_marked_rate`：异常图上报出缺陷的面积占比。小而集中才算定位成功。
      - `abnormal_undirected`：引擎说这张图坏、但图上没有一个像素达标。
        这是硬伤——分数与位置自相矛盾，比"定位面积偏大"严重得多。
    """
    valid = [s for s in samples if s.get("above") is not None]
    if not valid:
        return {
            "available": False,
            "mode": "diagnostic",
            "auroc": None,
            "reason": (
                "引擎未返回可用的归一化异常图，无法做像素级诊断"
                if not samples
                else "所有图片的异常图均为空或尺寸为 0，无法做像素级诊断"
            ),
            "diagnostic": None,
        }

    normal = [s for s in valid if int(s.get("label", 0)) == 0]
    abnormal = [s for s in valid if int(s.get("label", 0)) == 1]
    # 「判 NG 却指不出位置」按引擎判定统计，正常图被误报 NG 且无边也一样计入：
    # 那同样是一句无法支撑的结论，只是错的方向不同。
    undirected = sum(
        1
        for s in valid
        if int(s.get("pred", 0)) == 1 and float(s["above"]) <= 0.0
    )

    sizes = [tuple(int(v) for v in s["size"]) for s in valid if s.get("size")]
    map_size = list(sizes[0]) if sizes else None
    # 尺寸不统一说明中途换了引擎或输入尺寸，指标之间不可比，得说出来
    size_mixed = any(size != tuple(map_size) for size in sizes) if map_size else False

    reason = None
    if size_mixed:
        reason = "异常图尺寸不统一，各图占比不可直接横向比较"
    elif skipped > 0:
        reason = f"有 {skipped} 张图未参与像素统计（异常图缺失或尺寸为 0）"

    return {
        "available": True,
        "mode": "diagnostic",
        "auroc": None,
        "reason": reason,
        "diagnostic": {
            "cutoff": float(cutoff),
            "normal_false_pixel_rate": _mean_or_none([s["above"] for s in normal]),
            "abnormal_marked_rate": _mean_or_none([s["above"] for s in abnormal]),
            "abnormal_undirected": undirected,
            "pixel_skipped": int(skipped),
            "map_size": map_size,
            "images": {"normal": len(normal), "abnormal": len(abnormal)},
        },
    }
