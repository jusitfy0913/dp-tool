# -*- coding: utf-8 -*-
"""异常检测（Anomalib）的精度口径：图像级 AUROC / F1 / 误报率 / 漏检率。

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
