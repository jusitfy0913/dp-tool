# -*- coding: utf-8 -*-
"""精度指标验收脚本，不需要 GPU / DLL / cv2。

这个模块的每一条断言都在守「数字看着合理但其实错了」：
- IoU 边界用 `>=` 还是 `>`：差一个等号，边界框全判成漏检；
- 一个 GT 被两个预测框同时匹配：Recall 会超过 1；
- 没有 GT 的类别算成 0 分：把总分无端拉低；
- 重复框没被算成误检：Precision 虚高。

用法：
    python 6-web/backend/tools/test_val_metrics.py
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
    return abs(float(a) - float(b)) <= eps


def main() -> int:
    # ---- 1. IoU 基础 ------------------------------------------------------
    box = [0.0, 0.0, 10.0, 10.0]
    check("同框 IoU = 1", near(valmetrics.iou_xyxy(box, box), 1.0))
    check(
        "相离 IoU = 0",
        near(valmetrics.iou_xyxy(box, [20, 20, 30, 30]), 0.0),
    )
    check(
        "一半重叠 IoU = 1/3",
        near(valmetrics.iou_xyxy(box, [5, 0, 15, 10]), 1 / 3),
        f"得到 {valmetrics.iou_xyxy(box, [5, 0, 15, 10])}",
    )
    check(
        "退化框（宽 0）IoU = 0",
        near(valmetrics.iou_xyxy([5, 0, 5, 10], box), 0.0),
    )
    check(
        "包含关系 IoU = 小/大",
        near(valmetrics.iou_xyxy(box, [2, 2, 4, 4]), 4 / 100),
    )

    # ---- 2. 完美预测 ------------------------------------------------------
    images = [
        {
            "preds": [{"box": [0, 0, 10, 10], "score": 0.9, "cls": 0}],
            "gts": {0: [[0, 0, 10, 10]]},
        }
    ]
    report = valmetrics.evaluate(images, 1, conf=0.25)
    check("完美预测 mAP50 = 1", near(report["map50"], 1.0), str(report["map50"]))
    check("完美预测 mAP50-95 = 1", near(report["map5095"], 1.0), str(report["map5095"]))
    check("完美预测 precision = 1", near(report["precision"], 1.0))
    check("完美预测 recall = 1", near(report["recall"], 1.0))
    check("统计图片数", report["num_images"] == 1)
    check("统计 GT 数", report["num_gts"] == 1)
    check("统计预测数", report["num_predictions"] == 1)

    # ---- 3. 全漏检 --------------------------------------------------------
    report = valmetrics.evaluate([{"preds": [], "gts": {0: [[0, 0, 10, 10]]}}], 1)
    check("全漏检 mAP50 = 0", near(report["map50"], 0.0))
    check("全漏检 recall = 0", near(report["recall"], 0.0))
    check("全漏检 AP 仍是数值", report["per_class"][0]["ap50"] == 0.0)

    # ---- 4. 全是误检 ------------------------------------------------------
    report = valmetrics.evaluate(
        [{"preds": [{"box": [100, 100, 110, 110], "score": 0.9, "cls": 0}], "gts": {0: [[0, 0, 10, 10]]}}],
        1,
    )
    check("全误检 mAP50 = 0", near(report["map50"], 0.0))
    check("全误检 precision = 0", near(report["precision"], 0.0))

    # ---- 5. 重复框：第二个必须算误检 --------------------------------------
    duplicate = [
        {
            "preds": [
                {"box": [0, 0, 10, 10], "score": 0.9, "cls": 0},
                {"box": [0, 0, 10, 10], "score": 0.8, "cls": 0},
            ],
            "gts": {0: [[0, 0, 10, 10]]},
        }
    ]
    report = valmetrics.evaluate(duplicate, 1)
    check("重复框 recall 不超过 1", report["recall"] <= 1.0, str(report["recall"]))
    check("重复框 AP50 仍为 1（首个已命中）", near(report["map50"], 1.0))
    check("重复框 precision = 0.5", near(report["precision"], 0.5), str(report["precision"]))
    check(
        "重复框的实例数只算 1 个 GT",
        report["per_class"][0]["instances"] == 1,
    )

    # ---- 6. 一个 GT 不能被两个框同时匹配（跨图也不能串） ------------------
    two_images = [
        {"preds": [{"box": [0, 0, 10, 10], "score": 0.9, "cls": 0}], "gts": {0: [[0, 0, 10, 10]]}},
        {"preds": [{"box": [0, 0, 10, 10], "score": 0.8, "cls": 0}], "gts": {0: []}},
    ]
    report = valmetrics.evaluate(two_images, 1)
    check("第二张图的预测算误检", near(report["precision"], 0.5), str(report["precision"]))
    check("GT 只在第一张图里", report["num_gts"] == 1)

    # ---- 7. IoU 阈值边界 --------------------------------------------------
    # 构造一个 IoU 恰好 0.5 的预测框：宽 10 的框右移 10/3 后，
    # 交 = 66.67、并 = 133.33 → IoU = 0.5。
    # 必须在 0.50 上算命中，在 0.55 上算失败。
    shifted = [10 / 3, 0, 10 + 10 / 3, 10]
    boundary = [{"preds": [{"box": shifted, "score": 0.9, "cls": 0}], "gts": {0: [[0, 0, 10, 10]]}}]
    check(
        "构造的边界框 IoU ≈ 0.5",
        near(valmetrics.iou_xyxy(shifted, [0, 0, 10, 10]), 0.5, 1e-9),
        str(valmetrics.iou_xyxy(shifted, [0, 0, 10, 10])),
    )
    report = valmetrics.evaluate(boundary, 1, iou_thresholds=(0.5, 0.55))
    check("IoU=0.5 在阈值 0.50 上命中", report["per_class"][0]["ap50"] == 1.0,
          str(report["per_class"][0]["ap50"]))
    check(
        "IoU=0.5 在阈值 0.55 上不命中，AP50-95 被拉低",
        report["map5095"] < 0.6,
        f"mAP50-95={report['map5095']}",
    )

    # ---- 8. 多类别平均 ----------------------------------------------------
    multi = [
        {
            "preds": [{"box": [0, 0, 10, 10], "score": 0.9, "cls": 0}],
            "gts": {0: [[0, 0, 10, 10]], 1: [[20, 20, 30, 30]]},
        }
    ]
    report = valmetrics.evaluate(multi, 2)
    check("一类满分一类零分 → mAP50 = 0.5", near(report["map50"], 0.5), str(report["map50"]))
    check("零分类别的 AP 是 0", report["per_class"][1]["ap50"] == 0.0)
    check("两类都参与平均", len([r for r in report["per_class"] if not r["excluded"]]) == 2)

    # ---- 9. 没有 GT 的类别必须排除在均值外 --------------------------------
    only_class0 = [
        {
            "preds": [
                {"box": [0, 0, 10, 10], "score": 0.9, "cls": 0},
                {"box": [50, 50, 60, 60], "score": 0.9, "cls": 1},
            ],
            "gts": {0: [[0, 0, 10, 10]]},
        }
    ]
    report = valmetrics.evaluate(only_class0, 2)
    check("无 GT 类别被标记 excluded", report["per_class"][1]["excluded"] is True)
    check("无 GT 类别不拉低 mAP50", near(report["map50"], 1.0), str(report["map50"]))

    # ---- 10. 曲线形状 -----------------------------------------------------
    check("P-R 曲线重采样到 101 点",
          len(report["curves"]["0"]["recall"]) == valmetrics.RECALL_POINTS,
          str(len(report["curves"]["0"]["recall"])))
    check("曲线 recall 从 0 到 1",
          report["curves"]["0"]["recall"][0] == 0.0 and report["curves"]["0"]["recall"][-1] == 1.0)
    check("无 GT 的类别没有曲线", "1" not in report["curves"])

    # ---- 11. conf 过滤只影响 P/R，不影响 AP ------------------------------
    low_conf = [
        {
            "preds": [
                {"box": [0, 0, 10, 10], "score": 0.9, "cls": 0},
                {"box": [100, 100, 110, 110], "score": 0.1, "cls": 0},
            ],
            "gts": {0: [[0, 0, 10, 10]]},
        }
    ]
    report = valmetrics.evaluate(low_conf, 1, conf=0.25)
    check("低分误检不进入 conf 统计", near(report["precision"], 1.0), str(report["precision"]))
    check("低分误检仍影响 AP（排在后面但计入曲线）", report["map50"] <= 1.0)

    # ---- 12. 混淆矩阵与问题框 --------------------------------------------
    # 判对优先于判错：类别 0 的预测框盖住类别 0 的 GT，即使也盖住类别 1，也算判对。
    mixed = [
        {
            "file": "a.jpg",
            "preds": [
                {"box": [0, 0, 10, 10], "score": 0.9, "cls": 0},
                {"box": [0, 0, 10, 10], "score": 0.8, "cls": 1},
                {"box": [50, 50, 60, 60], "score": 0.7, "cls": 0},
            ],
            "gts": {0: [[0, 0, 10, 10]], 1: [[20, 20, 30, 30]]},
        }
    ]
    report = valmetrics.evaluate(mixed, 2, conf=0.25)
    found = report["diagnosis"]
    # 三个预测框：一个判对，一个盖在已被占用的 GT 上所以是过检，一个落在空白处也是过检。类别 1 的 GT 没人检到。
    check("判对 1、判错 0、漏检 1、过检 2",
          (found["correct"], found["wrong"], found["miss"], found["extra"]) == (1, 0, 1, 2),
          str((found["correct"], found["wrong"], found["miss"], found["extra"])))
    check("精确率 = 判对 / (判对+判错+过检)",
          near(report["precision"], 0.3333, 1e-4), str(report["precision"]))
    check("召回率 = 判对 / (判对+判错+漏检)",
          near(report["recall"], 1 / 2), str(report["recall"]))
    kinds = sorted(row["kind"] for row in found["errors"])
    check("问题清单只有漏检和过检", kinds == ["extra", "extra", "miss"], str(kinds))

    swapped = [
        {
            "file": "b.jpg",
            "preds": [{"box": [0, 0, 10, 10], "score": 0.95, "cls": 1}],
            "gts": {0: [[0, 0, 10, 10]]},
        }
    ]
    found = valmetrics.diagnosis(swapped, 2, conf=0.25)
    check("类别判错进非对角格", found["matrix"][0][1] == 1 and found["wrong"] == 1, str(found["matrix"]))
    check("判错不计入漏检和过检", found["miss"] == 0 and found["extra"] == 0)
    check("判错行带文件名和两个类别",
          found["errors"][0]["file"] == "b.jpg"
          and found["errors"][0]["gt_class"] == 0
          and found["errors"][0]["pred_class"] == 1)

    low = valmetrics.diagnosis(
        [{"preds": [{"box": [0, 0, 10, 10], "score": 0.1, "cls": 0}], "gts": {0: [[0, 0, 10, 10]]}}],
        1,
        conf=0.25,
    )
    check("低于 conf 的预测框当成没检出来", low["miss"] == 1 and low["correct"] == 0, str(low))

    capped = valmetrics.cap_errors({"errors": [{"kind": "miss"}] * 3}, limit=2)
    check("问题框截断只影响列表",
          capped["errors_total"] == 3 and capped["errors_omitted"] == 1 and len(capped["errors"]) == 2)

    # ---- 13. 空输入 ------------------------------------------------------
    empty = valmetrics.evaluate([], 0)
    check("空输入不崩", empty["map50"] == 0.0 and empty["num_images"] == 0)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  !! {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
