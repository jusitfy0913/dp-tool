# -*- coding: utf-8 -*-
"""断点续训的目录解析（checkpoints）验收脚本，不需要启动 Web 服务。

这里守的是「续训猜错数据集」这类静默故障：

1. 已完成轮数与最佳 mAP 只从 ``results.csv`` 读，绝不加载 ``.pt``（后端没有 torch）；
2. ``args.yaml`` 里的 ``data`` 已不存在时必须标成不可续训，并说明原因；
3. 缺少 ``weights/last.pt`` 时同样不可续训；
4. ``results.csv`` 缺失（训练刚起就中断）不能崩，只能给出 0 轮；
5. 多个训练目录按时间倒序，``limit`` 生效。

用法：
    python 6-web/backend/tools/test_checkpoints.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# 允许直接以脚本方式运行（把 backend 加入 sys.path）
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import checkpoints  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


RESULTS_CSV = (
    "epoch,time,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss\n"
    "1,10.0,1.1,0.20,0.10,0.9\n"
    "2,22.5,0.9,0.40,0.30,0.8\n"
    "3,35.1,0.8,0.45,0.35,0.7\n"
)


def build_run(
    root: Path,
    project: str,
    name: str,
    *,
    data_path: str | None,
    with_results: bool = True,
    with_last: bool = True,
    epochs: int = 20,
    offset: float = 0.0,
) -> Path:
    """造一个 Ultralytics 风格的训练目录。"""
    run_dir = root / project / name
    (run_dir / "weights").mkdir(parents=True)
    args = [
        "task: detect",
        "model: yolov8n.pt",
        f"data: {data_path if data_path is not None else ''}",
        f"epochs: {epochs}",
        "imgsz: 640",
        "batch: 8",
        "name: " + name,
        "optimizer: auto",
    ]
    (run_dir / "args.yaml").write_text("\n".join(args) + "\n", encoding="utf-8")
    if with_results:
        (run_dir / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
    if with_last:
        # 写够体积，才能验证 MB 换算（14 字节的假文件四舍五入是 0.0 MB）
        (run_dir / "weights" / "last.pt").write_bytes(b"fake-checkpoint" * 40000)
        (run_dir / "weights" / "best.pt").write_bytes(b"fake-checkpoint" * 40000)
    # 控制排序用的 mtime
    stamp = time.time() - offset
    for path in run_dir.rglob("*"):
        import os

        os.utime(path, (stamp, stamp))
    return run_dir


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "runs"
        data_yaml = Path(tmp) / "dataset.yaml"
        data_yaml.write_text(
            "path: .\ntrain: images/train\nnc: 2\nnames:\n  0: a\n  1: b\n", encoding="utf-8"
        )

        good = build_run(root, "trtforge", "exp_ok", data_path=str(data_yaml), offset=0)
        orphan = build_run(root, "trtforge", "exp_gone", data_path=str(Path(tmp) / "gone.yaml"), offset=10)
        broken = build_run(root, "trtforge", "exp_nolast", data_path=str(data_yaml), with_last=False, offset=20)
        fresh = build_run(root, "trtforge", "exp_fresh", data_path=str(data_yaml), with_results=False, offset=30)

        # ---- 1. 正常目录 ---------------------------------------------------
        item = checkpoints.describe(good)
        check("已完成轮数来自 results.csv", item["epochs_done"] == 3, str(item["epochs_done"]))
        check("计划轮数来自 args.yaml", item["epochs_planned"] == 20, str(item["epochs_planned"]))
        check("最佳 mAP 取最大值", item["best_map"] == 0.35, str(item["best_map"]))
        check("最佳轮次可定位", item["best_epoch"] == 3, str(item["best_epoch"]))
        check("可续训", item["resumable"] is True and item["blocked_reason"] == "", str(item["blocked_reason"]))
        check("类别数从 data 里读出", item["classes"] == 2, str(item["classes"]))
        check("last.pt 体积被记录", item["last_pt_mb"] > 0, str(item["last_pt_mb"]))
        check("imgsz / batch 被解析", item["imgsz"] == 640 and item["batch"] == "8", f"{item['imgsz']}/{item['batch']}")

        # ---- 2. 数据集已删除 -------------------------------------------------
        gone = checkpoints.describe(orphan)
        check("数据集不存在时不可续训", gone["resumable"] is False, str(gone["resumable"]))
        check("原因里点出数据集路径", "数据集已不存在" in gone["blocked_reason"], gone["blocked_reason"])

        # ---- 3. 缺 last.pt ---------------------------------------------------
        no_last = checkpoints.describe(broken)
        check("缺 last.pt 时不可续训", no_last["resumable"] is False, str(no_last["resumable"]))
        check("原因里点出 last.pt", "last.pt" in no_last["blocked_reason"], no_last["blocked_reason"])

        # ---- 4. 没有 results.csv --------------------------------------------
        no_results = checkpoints.describe(fresh)
        check("缺 results.csv 时轮数为 0", no_results["epochs_done"] == 0, str(no_results["epochs_done"]))
        check("缺 results.csv 时最佳 mAP 为空", no_results["best_map"] is None, str(no_results["best_map"]))
        check("缺 results.csv 但仍可续训", no_results["resumable"] is True, str(no_results["resumable"]))

        # ---- 5. 列表排序与 limit ---------------------------------------------
        listing = checkpoints.list_checkpoints([root], limit=10)
        names = [entry["name"] for entry in listing["items"]]
        check("找到全部训练目录", listing["total"] == 4, str(listing["total"]))
        check("按时间倒序（最新在前）", names[0] == "exp_ok", str(names))
        limited = checkpoints.list_checkpoints([root], limit=2)
        check("limit 生效", limited["shown"] == 2 and limited["total"] == 4, str(limited["shown"]))

        # ---- 6. 空根目录不崩 -------------------------------------------------
        empty = checkpoints.list_checkpoints([Path(tmp) / "nothing"], limit=5)
        check("根目录不存在时返回空列表", empty["total"] == 0 and empty["items"] == [], str(empty))

        # ---- 7. args.yaml 解析细节 -------------------------------------------
        parsed = checkpoints.read_args_yaml(good / "args.yaml")
        check("args.yaml 键名小写", "epochs" in parsed and parsed["imgsz"] == "640", str(parsed.get("imgsz")))
        results = checkpoints.read_results(good / "results.csv")
        check("results 解析出最后一轮指标", results["last_metrics"].get("metrics/mAP50-95(B)") == 0.35, str(results["last_metrics"]))
        check("训练时长取最后一轮累计时间", results["train_seconds"] == 35.1, str(results["train_seconds"]))

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
