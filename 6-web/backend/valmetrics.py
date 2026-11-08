# -*- coding: utf-8 -*-
"""目标检测精度指标（纯计算，不碰 I/O、不依赖 cv2 / DLL）。

为什么自己写一份 AP 而不是调 pycocotools：

1. **口径要能写清楚**。产线上"mAP 0.91"和"训练日志里的 mAP 0.93"对不上时，
   必须能一句话解释差在哪——这里全部按 COCO 的口径实现（按类别、按置信度降序、
   IoU 贪心匹配、全点插值），代码就是口径本身。
2. **不能让验证依赖重依赖**。`pycocotools` 在 Windows 上要编译，CI 里装不上；
   `ultralytics` 会连带拖进 torch。这个模块只依赖 numpy，能被单测直接覆盖。

匹配细节（与 COCO 一致的地方、刻意简化的地方都写在这里）：

- 同一张图里，一个 GT 只能被一个预测框匹配（贪心，按 score 降序先到先得）；
- 匹配成功的判据是 IoU >= 阈值（COCO 用 `>=`，不是 `>`）；
- 每个类别独立算 AP，然后对"该类真的有 GT"的类别取平均；
- 没有 GT 的类别 AP 记为 None 并排除在均值外，避免把"没数据"算成 0 分拉低总分。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# COCO 的 10 个 IoU 阈值：0.50, 0.55 ... 0.95
DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * i, 2) for i in range(10)
)
RECALL_POINTS = 101


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 [x1, y1, x2, y2] 框的 IoU。

    退化框（宽或高 <= 0）面积为 0，IoU 直接返回 0：真实数据集里确实有这种
    标注（标注工具里手抖拖出来的），不能让它把分母变成负数。
    """
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _normalize_preds(preds: Iterable[dict]) -> list[dict]:
    out = []
    for item in preds:
        box = item.get("box") or [
            item.get("left", item.get("x1", 0)),
            item.get("top", item.get("y1", 0)),
            item.get("right", item.get("x2", 0)),
            item.get("bottom", item.get("y2", 0)),
        ]
        out.append(
            {
                "box": [float(v) for v in box[:4]],
                "score": float(item.get("score", item.get("confidence", 0.0))),
                "cls": int(item.get("cls", item.get("class_label", 0))),
            }
        )
    return out


def _collect_class_preds(
    images: Sequence[dict], cls: int
) -> list[tuple[float, int, list[float]]]:
    """把某类别的预测框汇总成 (score, image_index, box)，按 score 降序。"""
    collected: list[tuple[float, int, list[float]]] = []
    for index, item in enumerate(images):
        for pred in _normalize_preds(item.get("preds") or []):
            if pred["cls"] == cls:
                collected.append((pred["score"], index, pred["box"]))
    collected.sort(key=lambda row: row[0], reverse=True)
    return collected


def _count_class_gts(images: Sequence[dict], cls: int) -> int:
    total = 0
    for item in images:
        total += len((item.get("gts") or {}).get(cls) or [])
    return total


def _match_flags(
    class_preds: Sequence[tuple[float, int, list[float]]],
    images: Sequence[dict],
    cls: int,
    threshold: float,
) -> list[int]:
    """返回与 class_preds 等长的 0/1 序列：1 = 命中 GT（TP）。

    贪心匹配的"已用 GT"按图隔离：跨图复用同一 GT 会把 Recall 算爆。
    """
    used: dict[int, set[int]] = {}
    flags: list[int] = []
    for _score, image_index, box in class_preds:
        gts = (images[image_index].get("gts") or {}).get(cls) or []
        used_here = used.setdefault(image_index, set())
        best_index = -1
        best_iou = -1.0
        for gt_index, gt in enumerate(gts):
            if gt_index in used_here:
                continue
            value = iou_xyxy(box, gt)
            # 边界容差：IoU 恰好 0.5 时浮点结果是 0.4999999999999999，
            # 直接 >= 会把它判成漏检，理由是"差了一个 ulp"。COCO 的判据本来
            # 就是 IoU >= 阈值，这里用 1e-9 的容差把"恰好等于"还原回来。
            if value >= threshold - 1e-9 and value > best_iou:
                best_iou = value
                best_index = gt_index
        if best_index >= 0:
            used_here.add(best_index)
            flags.append(1)
        else:
            flags.append(0)
    return flags


def _average_precision(flags: Sequence[int], total_gts: int) -> tuple[float, list[float], list[float]]:
    """按 COCO 的全点插值算 AP，并返回原始 (recall, precision) 曲线。"""
    if total_gts <= 0:
        return 0.0, [], []
    if not flags:
        return 0.0, [0.0], [0.0]

    tp = np.cumsum(np.asarray(flags, dtype=np.float64))
    fp = np.cumsum(1 - np.asarray(flags, dtype=np.float64))
    recall = tp / float(total_gts)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # 从右往左取包络：precision 必须随 recall 单调不增，否则插值会高估
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    index = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(((mrec[index + 1] - mrec[index]) * mpre[index + 1]).sum())
    return ap, recall.tolist(), precision.tolist()


def _sample_curve(recall: Sequence[float], precision: Sequence[float]) -> dict[str, list[float]]:
    """把曲线重采样到固定 recall 网格，便于前端直接画折线。"""
    if not recall:
        return {"recall": [], "precision": []}
    grid = np.linspace(0.0, 1.0, RECALL_POINTS)
    values = np.zeros_like(grid)
    recall_array = np.asarray(recall, dtype=np.float64)
    precision_array = np.asarray(precision, dtype=np.float64)
    # 单调查找：对每个 recall 网格点取"不小于它的第一个预测点"的 precision
    for i, point in enumerate(grid):
        idx = int(np.searchsorted(recall_array, point, side="left"))
        if idx >= len(precision_array):
            values[i] = precision_array[-1] if len(precision_array) else 0.0
        else:
            values[i] = precision_array[idx]
    return {"recall": grid.round(4).tolist(), "precision": values.round(4).tolist()}


def _micro_counts(
    images: Sequence[dict], num_classes: int, conf: float, threshold: float = 0.5
) -> tuple[int, int]:
    """按 conf 过滤后的整体 TP / FP（用于报表上的 P / R，不参与 AP）。"""
    tp = fp = 0
    for cls in range(num_classes):
        preds = [row for row in _collect_class_preds(images, cls) if row[0] >= conf]
        flags = _match_flags(preds, images, cls, threshold)
        tp += sum(flags)
        fp += len(flags) - sum(flags)
    return tp, fp


def evaluate(
    images: Sequence[dict],
    num_classes: int,
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    conf: float = 0.25,
) -> dict[str, Any]:
    """计算 mAP50 / mAP50-95 / 每类 AP / P-R 曲线。

    入参 ``images`` 每项形如::

        {"preds": [{"box": [x1,y1,x2,y2], "score": 0.87, "cls": 3}],
         "gts":   {3: [[x1,y1,x2,y2], ...]}}

    坐标一律是**像素**坐标；归一化坐标的换算在调用方（读标签时）完成，
    这样指标层不需要知道图片尺寸。
    """
    num_classes = max(int(num_classes), 0)
    per_class: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}

    for cls in range(num_classes):
        class_preds = _collect_class_preds(images, cls)
        total_gts = _count_class_gts(images, cls)
        if total_gts == 0:
            per_class.append(
                {
                    "id": cls,
                    "instances": 0,
                    "ap50": None,
                    "ap": None,
                    "excluded": True,
                }
            )
            continue

        aps: list[float] = []
        ap50 = 0.0
        curve: dict[str, list[float]] = {"recall": [], "precision": []}
        for threshold in iou_thresholds:
            flags = _match_flags(class_preds, images, cls, threshold)
            ap, recall, precision = _average_precision(flags, total_gts)
            aps.append(ap)
            if abs(threshold - 0.5) < 1e-9:
                ap50 = ap
                curve = _sample_curve(recall, precision)

        per_class.append(
            {
                "id": cls,
                "instances": total_gts,
                "predictions": len(class_preds),
                "ap50": round(ap50, 4),
                "ap": round(float(np.mean(aps)), 4) if aps else 0.0,
                "excluded": False,
            }
        )
        curves[str(cls)] = curve

    counted = [row for row in per_class if not row["excluded"]]
    map50 = round(float(np.mean([row["ap50"] for row in counted])), 4) if counted else 0.0
    map5095 = round(float(np.mean([row["ap"] for row in counted])), 4) if counted else 0.0

    tp, fp = _micro_counts(images, num_classes, conf)
    total_gts = sum(row["instances"] for row in per_class)
    recall_at_conf = round(tp / total_gts, 4) if total_gts else 0.0
    precision_at_conf = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    found = diagnosis(images, num_classes, conf=conf)

    return {
        "map50": map50,
        "map5095": map5095,
        "precision": precision_at_conf,
        "recall": recall_at_conf,
        "conf": round(float(conf), 4),
        "num_images": len(images),
        "num_gts": total_gts,
        "num_predictions": sum(len(_normalize_preds(item.get("preds") or [])) for item in images),
        "iou_thresholds": [round(float(t), 2) for t in iou_thresholds],
        "per_class": per_class,
        "curves": curves,
        "diagnosis": found,
    }


def _gt_list(item: dict) -> list[dict]:
    """把一张图的 GT 摊平成 [{cls, box}]。类别键可能是 int，也可能是从 JSON 读回来的 str。"""
    rows: list[dict] = []
    for key, boxes in (item.get("gts") or {}).items():
        try:
            cls = int(key)
        except (TypeError, ValueError):
            continue
        for box in boxes or []:
            rows.append({"cls": cls, "box": [float(v) for v in box[:4]]})
    return rows


def _best_gt(box: list[float], gts: list[dict], used: set[int], threshold: float, *, cls: int | None) -> tuple[int, float]:
    """在还没被占用的 GT 里找 IoU 最大的一个。cls 给出时只看同一类。"""
    best_index = -1
    best_iou = -1.0
    for index, gt in enumerate(gts):
        if index in used:
            continue
        if cls is not None and gt["cls"] != cls:
            continue
        value = iou_xyxy(box, gt["box"])
        if value >= threshold - 1e-9 and value > best_iou:
            best_iou = value
            best_index = index
    return best_index, best_iou


def diagnosis(
    images: Sequence[dict],
    num_classes: int,
    *,
    conf: float = 0.25,
    iou: float = 0.5,
) -> dict[str, Any]:
    """在 conf 与 IoU 0.5 下给出混淆矩阵和有问题的框。

    配对分两步，保证和 ``evaluate`` 的精确率、召回率是同一批框：

    1. 先按类别、按置信度从高到低，把预测框配给同类 GT。配上的是「判对」。
    2. 剩下的框里，重叠够大但类别不同的配成「判错」。
       再剩下的预测框是「过检」，没被配上的 GT 是「漏检」。

    因此：精确率 = 判对 / (判对 + 判错 + 过检)，召回率 = 判对 / (判对 + 判错 + 漏检)。

    矩阵比类别数多一行一列，下标 ``num_classes`` 是背景：
    最后一列是漏检，最后一行是过检，其余非对角格是判错。
    """
    num_classes = max(int(num_classes), 0)
    size = num_classes + 1
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    errors: list[dict[str, Any]] = []
    correct = wrong = miss = extra = 0

    for item in images:
        file_name = str(item.get("file") or "")
        preds = [
            pred for pred in _normalize_preds(item.get("preds") or [])
            if pred["score"] >= conf and 0 <= pred["cls"] < num_classes
        ]
        preds.sort(key=lambda pred: pred["score"], reverse=True)
        gts = [gt for gt in _gt_list(item) if 0 <= gt["cls"] < num_classes]
        used_gt: set[int] = set()
        used_pred: set[int] = set()

        for pred_index, pred in enumerate(preds):
            gt_index, _value = _best_gt(pred["box"], gts, used_gt, iou, cls=pred["cls"])
            if gt_index < 0:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            matrix[pred["cls"]][pred["cls"]] += 1
            correct += 1

        for pred_index, pred in enumerate(preds):
            if pred_index in used_pred:
                continue
            gt_index, value = _best_gt(pred["box"], gts, used_gt, iou, cls=None)
            if gt_index < 0:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            gt = gts[gt_index]
            matrix[gt["cls"]][pred["cls"]] += 1
            wrong += 1
            errors.append({
                "file": file_name,
                "kind": "wrong",
                "gt_class": gt["cls"],
                "pred_class": pred["cls"],
                "score": round(pred["score"], 4),
                "iou": round(value, 4),
            })

        for gt_index, gt in enumerate(gts):
            if gt_index in used_gt:
                continue
            matrix[gt["cls"]][num_classes] += 1
            miss += 1
            errors.append({
                "file": file_name,
                "kind": "miss",
                "gt_class": gt["cls"],
                "pred_class": None,
                "score": None,
                "iou": None,
            })

        for pred_index, pred in enumerate(preds):
            if pred_index in used_pred:
                continue
            matrix[num_classes][pred["cls"]] += 1
            extra += 1
            errors.append({
                "file": file_name,
                "kind": "extra",
                "gt_class": None,
                "pred_class": pred["cls"],
                "score": round(pred["score"], 4),
                "iou": None,
            })

    return {
        "iou": round(float(iou), 2),
        "background": num_classes,
        "matrix": matrix,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "extra": extra,
        "errors": errors,
    }


def cap_errors(found: dict[str, Any], limit: int = 500) -> dict[str, Any]:
    """JSON 里只留前 limit 条问题框。调用前应已把全量写进 CSV。"""
    errors = list(found.get("errors") or [])
    found["errors_total"] = len(errors)
    if len(errors) > limit:
        found["errors"] = errors[:limit]
        found["errors_omitted"] = len(errors) - limit
    else:
        found["errors_omitted"] = 0
    return found


def box_iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """批量 IoU 矩阵，给需要一次性算 N×M 的调用方用（当前主流程不依赖它）。"""
    a_array = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b_array = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a_array.size == 0 or b_array.size == 0:
        return np.zeros((len(a_array), len(b_array)), dtype=np.float64)
    matrix = np.zeros((len(a_array), len(b_array)), dtype=np.float64)
    for i, row in enumerate(a_array):
        for j, col in enumerate(b_array):
            matrix[i, j] = iou_xyxy(row, col)
    return matrix
