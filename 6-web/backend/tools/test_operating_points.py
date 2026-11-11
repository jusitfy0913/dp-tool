# -*- coding: utf-8 -*-
"""操作点（漏检-过杀曲线）验收脚本，不需要 GPU / DLL / cv2。

这个模块每条断言守的都是「曲线画出来了但数学是错的」这类问题：
- 良品图（标签文件存在且为空）被当成漏标 → 过杀率虚高成 0；
- 漏标图（标签文件不存在）混进良品统计 → 过杀率被稀释成好看的数字；
- 推荐操作点取最小 conf → 明明可以少报的框全报出来，方向反了；
- 没有良品图时返回 0 而不是 null → 看不出「没测过」和「一个没误报」的区别；
- 前缀累积与 :func:`valmetrics.evaluate` 的计数对不上 → 同一份数据两个页面两个数。

用法：
    python 6-web/backend/tools/test_operating_points.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import valmetrics  # noqa: E402

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
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= eps


def _box(x: float = 0.0, y: float = 0.0) -> list[float]:
    return [x, y, x + 10.0, y + 10.0]


def main() -> int:
    # ---- 1. conf 列表归一化 ----------------------------------------------
    confs = valmetrics._normalize_confs([0.5, 0.5, 0.1, 1.5, -0.2])
    check("conf 去重排序并夹到 [0,1]", confs == [0.0, 0.1, 0.5, 1.0], str(confs))
    check(
        "conf 点数上限",
        len(valmetrics._normalize_confs([i / 200 for i in range(201)]))
        <= valmetrics.MAX_OPERATING_POINTS,
    )
    default = valmetrics._normalize_confs(None)
    check("默认扫描 conf 从低到高", default == sorted(default) and len(default) >= 2, str(default))
    check("默认最低 conf 足够低", default[0] <= 0.1, str(default[0]))

    # ---- 2. GT 状态统计 --------------------------------------------------
    images_states = [
        {"gt_state": "labeled", "gts": {0: [_box()]}},
        {"gt_state": "ok", "gts": {}},
        {"gt_state": "ok", "gts": {}},
        {"gt_state": "missing", "gts": {}},
        {"gt_state": "unparsed", "gts": {}},
    ]
    stats = valmetrics.gt_stats(images_states)
    check(
        "GT 状态分类计数",
        (stats["instances"], stats["ok"], stats["missing"], stats["unparsed"]) == (1, 2, 1, 1),
        str(stats),
    )
    # 老调用方不带 gt_state：没有框就当良品，这是兜底而不是首选
    fallback = valmetrics.gt_stats([{"gts": {}}, {"gts": {0: [_box()]}}])
    check("缺省状态兜底为良品", (fallback["ok"], fallback["missing"]) == (1, 0), str(fallback))

    # ---- 3. 曲线基本形状 --------------------------------------------------
    # 一张有缺陷图 + 一张良品图：良品图里有一个低分误报框
    sample = [
        {
            "file": "defect.jpg",
            "gt_state": "labeled",
            "preds": [
                {"box": _box(), "score": 0.90, "cls": 0},
                {"box": _box(100, 100), "score": 0.40, "cls": 0},
            ],
            "gts": {0: [_box()]},
        },
        {
            "file": "ok.jpg",
            "gt_state": "ok",
            "preds": [{"box": _box(50, 50), "score": 0.30, "cls": 0}],
            "gts": {},
        },
    ]
    report = valmetrics.operating_points(sample, 1, confs=[0.1, 0.35, 0.5, 0.95])
    points = report["points"]
    check("每个 conf 一个点", len(points) == 4, str(len(points)))
    check("图片数与 GT 数", report["num_images"] == 2 and report["num_gts"] == 1, str(report))
    check("良品图计数", report["ok_images"] == 1, str(report["ok_images"]))
    check("漏标图不混进良品", report["missing_label_images"] == 0)

    confs_out = [point["conf"] for point in points]
    check("点按 conf 升序", confs_out == sorted(confs_out), str(confs_out))

    tps = [point["tp"] for point in points]
    fps = [point["fp"] for point in points]
    check("tp 随 conf 单调不增", all(a >= b for a, b in zip(tps, tps[1:])), str(tps))
    check("fp 随 conf 单调不增", all(a >= b for a, b in zip(fps, fps[1:])), str(fps))

    low = points[0]
    check("低 conf 抓全缺陷：recall = 1", near(low["recall"], 1.0), str(low["recall"]))
    check("低 conf 漏检率 = 0", near(low["miss_rate"], 0.0), str(low["miss_rate"]))
    check("低 conf 误报 2 个（含良品图 1 个）", low["fp"] == 2, str(low["fp"]))
    check("过杀 = 误报/图片数 = 1.0", near(low["fp_per_image"], 1.0), str(low["fp_per_image"]))
    check("良品图被误报", low["ok_fp_images"] == 1, str(low["ok_fp_images"]))
    check("良品误报率 = 1", near(low["ok_fp_rate"], 1.0), str(low["ok_fp_rate"]))

    high = points[-1]
    check("高 conf 全漏检", near(high["miss_rate"], 1.0), str(high["miss_rate"]))
    check("高 conf 无过杀", high["fp"] == 0 and near(high["fp_per_image"], 0.0), str(high["fp"]))
    check("高 conf 良品误报率 = 0", near(high["ok_fp_rate"], 0.0), str(high["ok_fp_rate"]))

    # ---- 4. 没有良品图 → ok_fp_rate 必须是 null ---------------------------
    no_ok = [
        {
            "file": "a.jpg",
            "gt_state": "labeled",
            "preds": [{"box": _box(), "score": 0.9, "cls": 0}],
            "gts": {0: [_box()]},
        },
        {"file": "b.jpg", "gt_state": "missing", "preds": [], "gts": {}},
    ]
    report = valmetrics.operating_points(no_ok, 1, confs=[0.5])
    point = report["points"][0]
    check("无良品图时 ok_fp_rate 为 None", point["ok_fp_rate"] is None, str(point["ok_fp_rate"]))
    check("无良品图时 ok_images = 0", report["ok_images"] == 0)
    check("漏标图单独计数", report["missing_label_images"] == 1, str(report["missing_label_images"]))
    check("良品误报数不会凭空出现", point["ok_fp_images"] == 0)

    # ---- 5. 与 evaluate 的计数一致 ---------------------------------------
    # 同一份数据、同一个 conf，曲线上的点必须等于 evaluate 的统计：
    # 两个页面各显示一套数字是最难查的那类 bug。
    for conf in (0.1, 0.35, 0.5):
        curve_point = next(
            point for point in valmetrics.operating_points(sample, 1, confs=[conf])["points"]
        )
        flat = valmetrics.evaluate(sample, 1, conf=conf)
        same_precision = (
            (curve_point["precision"] is None and flat["precision"] is None)
            or near(curve_point["precision"], flat["precision"], 1e-4)
        )
        check(
            f"conf={conf} 与 evaluate 精度一致",
            same_precision,
            f"曲线 {curve_point['precision']} vs 评估 {flat['precision']}",
        )
        check(
            f"conf={conf} 与 evaluate 召回一致",
            near(curve_point["recall"], flat["recall"], 1e-4),
            f"曲线 {curve_point['recall']} vs 评估 {flat['recall']}",
        )

    # ---- 6. 逐类明细 -----------------------------------------------------
    multi = [
        {
            "file": "c.jpg",
            "gt_state": "labeled",
            "preds": [{"box": _box(), "score": 0.9, "cls": 0}],
            "gts": {0: [_box()], 1: [_box(40, 40)]},
        }
    ]
    point = valmetrics.operating_points(multi, 2, confs=[0.5])["points"][0]
    rows = {row["id"]: row for row in point["per_class"]}
    check("有 GT 的类别才有明细行", sorted(rows) == [0, 1], str(sorted(rows)))
    check("类别 0 命中", rows[0]["tp"] == 1 and rows[0]["fn"] == 0, str(rows[0]))
    check("类别 1 全漏检", rows[1]["tp"] == 0 and rows[1]["fn"] == 1, str(rows[1]))
    check(
        "整体漏检率 = 漏检框 / GT 框",
        near(point["miss_rate"], 0.5),
        str(point["miss_rate"]),
    )

    # ---- 7. 推荐操作点：取满足目标的最高 conf -----------------------------
    target_report = valmetrics.operating_points(sample, 1, confs=[0.1, 0.35, 0.5, 0.95])
    target = valmetrics.recommend_operating_point(target_report, target_miss_rate=0.0)
    check("有可行点时 feasible = True", target["feasible"] is True, target["reason"])
    chosen = target["point"]
    check("推荐点漏检率满足目标", chosen["miss_rate"] <= 0.0 + 1e-9, str(chosen["miss_rate"]))
    feasible_confs = [
        point["conf"]
        for point in target_report["points"]
        if point["miss_rate"] is not None and point["miss_rate"] <= 1e-9
    ]
    check(
        "推荐点是可行解里 conf 最高的（过杀最低）",
        near(chosen["conf"], max(feasible_confs)),
        f"选中 {chosen['conf']}，可行 {feasible_confs}",
    )
    check("推荐点不是最低 conf", chosen["conf"] > min(target_report["confs"]))

    # ---- 8. 无可行点必须说实话 ------------------------------------------
    impossible = valmetrics.recommend_operating_point(
        target_report, target_miss_rate=-1.0
    )
    check("目标不可能时 feasible = False", impossible["feasible"] is False, str(impossible))
    check("无解也给最接近的点", impossible["point"] is not None)
    check("无解时理由非空", bool(impossible["reason"]))
    empty = valmetrics.recommend_operating_point({"points": []}, target_miss_rate=0.0)
    check("空报告不崩且不可行", empty["feasible"] is False and empty["point"] is None)

    # ---- 9. 过杀目标约束 --------------------------------------------------
    fp_target = valmetrics.recommend_operating_point(
        target_report, target_miss_rate=None, target_fp_per_image=0.0
    )
    check("只给过杀目标时可行", fp_target["feasible"] is True, fp_target["reason"])
    check(
        "过杀目标被满足",
        near(fp_target["point"]["fp_per_image"], 0.0),
        str(fp_target["point"]["fp_per_image"]),
    )

    # ---- 10. 空输入 ------------------------------------------------------
    empty_report = valmetrics.operating_points([], 0, confs=[0.5])
    check("空输入不崩", empty_report["num_images"] == 0 and len(empty_report["points"]) == 1)
    check("空输入 recall 为 None", empty_report["points"][0]["recall"] is None)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  !! {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
