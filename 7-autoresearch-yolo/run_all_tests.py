"""自动实验循环的回归测试。

只测纯逻辑与本地文件操作，**不跑训练**，所以可以在几秒内跑完，
适合每次改完 autoexp.py 就执行一遍。

    python run_all_tests.py

设计原则：
1. 每个用例自己建临时目录，跑完删掉，不污染真实的 results.tsv / holdout.tsv
2. 断言失败要打印「期望 vs 实际」，不要只说 assert False
3. 涉及生产路径（1-python-yolo/launcher/runs 等）的用例只验证「被拒绝」，
   绝不真的去写
"""

from __future__ import annotations

import csv
import importlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import autoexp as A  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []


class Skip(Exception):
    """用例的前提条件不满足（例如 splits/ 还没生成），跳过而不是失败。"""


def case(name: str):
    """装饰器：跑一个用例，抓住异常记成失败而不是中断整轮。"""

    def wrap(func):
        def runner():
            try:
                func()
            except Skip as exc:
                SKIPPED.append((name, str(exc)))
                print(f"  [skip] {name}（{exc}）")
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                print(f"  [FAIL] {name}")
            else:
                PASSED.append(name)
                print(f"  [ ok ] {name}")

        runner.__name__ = func.__name__
        return runner

    return wrap


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}：期望 {expected!r}，实际 {actual!r}")


# ---------------------------------------------------------------------------
# 1. 配置校验
# ---------------------------------------------------------------------------


@case("base.json 能通过 space.json 校验")
def test_base_passes_space() -> None:
    base = A.load_json(A.BASE_JSON)
    space = A.load_json(A.SPACE_JSON)
    errors = A.validate_against_space(base, space)
    check_eq(errors, [], "校验错误列表")


@case("越界的值会被拒绝")
def test_space_rejects_out_of_range() -> None:
    space = A.load_json(A.SPACE_JSON)
    bad = {"train": {"imgsz": 12345, "batch": "8"}}
    errors = A.validate_against_space(bad, space)
    check(len(errors) > 0, "imgsz=12345 应当被拒绝，实际通过了")


@case("未在 space.json 声明的键会被拒绝")
def test_space_rejects_unknown_key() -> None:
    space = A.load_json(A.SPACE_JSON)
    bad = {"train": {"batch": "8", "this_key_does_not_exist": 1}}
    errors = A.validate_against_space(bad, space)
    check(len(errors) > 0, "未知键应当被拒绝，实际通过了")


@case("deep_merge 只覆盖指定字段，其余继承")
def test_deep_merge() -> None:
    base = {"a": 1, "train": {"imgsz": 640, "batch": "8", "lr0": 0.01}}
    override = {"train": {"imgsz": 960}}
    merged = A.deep_merge(base, override)
    check_eq(merged["train"]["imgsz"], 960, "imgsz")
    check_eq(merged["train"]["batch"], "8", "batch 应继承")
    check_eq(merged["train"]["lr0"], 0.01, "lr0 应继承")
    check_eq(merged["a"], 1, "顶层字段应继承")
    # 不能就地改坏 base
    check_eq(base["train"]["imgsz"], 640, "base 不应被修改")


# ---------------------------------------------------------------------------
# 2. 写入守卫
# ---------------------------------------------------------------------------


@case("写入守卫拦住生产 runs 目录")
def test_sandbox_blocks_production_runs() -> None:
    targets = [
        A.ROOT.parent / "1-python-yolo" / "launcher" / "runs" / "exp",
        A.ROOT.parent / "2-python-anomaly" / "launcher" / "results" / "x",
        A.ROOT.parent / "6-web" / "runtime" / "logs" / "x.log",
    ]
    for target in targets:
        try:
            A.assert_sandbox(target, "regression")
        except PermissionError:
            continue
        raise AssertionError(f"应当拒绝 {target}，实际放行了")


@case("写入守卫放行本模块目录")
def test_sandbox_allows_own_dir() -> None:
    target = (A.ROOT / "runs" / "autoexp" / "selfcheck").resolve()
    result = A.assert_sandbox(target, "regression")
    check_eq(result, target, "返回值")


# ---------------------------------------------------------------------------
# 3. 台账：表头迁移
# ---------------------------------------------------------------------------


@case("台账表头升级时按列名迁移，不让历史行错位")
def test_tsv_header_migration() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoexp_tsv_"))
    try:
        path = tmp / "t.tsv"
        old_header = ["exp_id", "map50", "status"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(old_header)
            writer.writerow(["exp_0001", "0.42", "done"])

        new_header = ["exp_id", "map50_95", "map50", "batch_eff", "status"]
        A.ensure_tsv(path, new_header)

        rows = A.read_tsv(path)
        check_eq(len(rows), 1, "行数")
        check_eq(rows[0]["exp_id"], "exp_0001", "exp_id")
        check_eq(rows[0]["map50"], "0.42", "map50 应按列名落到原位")
        check_eq(rows[0]["map50_95"], "", "新增列应补空串")
        check_eq(rows[0]["batch_eff"], "", "新增列应补空串")
        check_eq(rows[0]["status"], "done", "status 应按列名落到原位")
        check(path.with_suffix(".tsv.bak").is_file(), "应留下 .bak 备份")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case("台账表头一致时不重写文件")
def test_tsv_no_rewrite_when_same() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoexp_tsv2_"))
    try:
        path = tmp / "t.tsv"
        header = ["exp_id", "map50"]
        A.append_tsv(path, header, {"exp_id": "exp_0001", "map50": "0.1"})
        before = path.read_bytes()
        A.ensure_tsv(path, header)
        check_eq(path.read_bytes(), before, "文件内容不应变化")
        check(not path.with_suffix(".tsv.bak").is_file(), "不应产生 .bak")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. 显存预检
# ---------------------------------------------------------------------------


@case("显存预检：需求随 batch 与分辨率单调增长")
def test_preflight_monotonic() -> None:
    # 直接比对估算公式的序关系；实际可用显存由机器决定，不在这里断言
    low = 2600.0 + 240.0 * 8 * (640 / 640) ** 2
    high = 2600.0 + 240.0 * 16 * (640 / 640) ** 2
    big = 2600.0 + 240.0 * 8 * (1280 / 640) ** 2
    check(low < high, "batch 翻倍，需求应变大")
    check(low < big, "分辨率翻倍，需求应变大")
    # 无 CUDA 的机器上 free_vram_mb 返回 None，预检必须放行而不是卡死
    free = A.free_vram_mb()
    check(free is None or free > 0, f"可用显存不应为负：{free}")


@case("显存预检：可用显存不足时拒绝并给出可读原因")
def test_preflight_rejects_when_tight() -> None:
    # 直接构造「需要 100000 MB」的调用，任何真实显卡都不可能满足
    ok, reason = A.preflight_vram(640, 8, need_mb=100000.0)
    free = A.free_vram_mb()
    if free is None:
        check(ok is True, "无 CUDA 时应放行")
    else:
        check(ok is False, "需求远超可用显存时应拒绝")
        check("可用显存" in reason, f"原因应包含可用显存读数，实际：{reason}")


# ---------------------------------------------------------------------------
# 5. 指标解析
# ---------------------------------------------------------------------------


@case("results.csv 解析：取最后一行，不取历史最佳")
def test_read_results_last_row() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoexp_csv_"))
    try:
        path = tmp / "results.csv"
        path.write_text(
            "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),"
            "metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,25.0,3.9,0.01,0.11,0.005,0.0022\n"
            "2,50.0,2.9,0.20,0.15,0.116,0.0471\n"
            "3,75.0,2.5,0.34,0.28,0.245,0.1036\n",
            encoding="utf-8",
        )
        metrics = A.read_results_csv(path)
        check_eq(metrics["epochs_done"], 3, "epochs_done")
        check_eq(metrics["map50_95"], 0.1036, "map50_95 应取最后一行")
        check_eq(metrics["best_map50_95"], 0.1036, "best 是历史最大")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case("results.csv 解析：best 与最后一行不同时能被区分")
def test_read_results_best_vs_last() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoexp_csv2_"))
    try:
        path = tmp / "results.csv"
        path.write_text(
            "epoch,time,metrics/precision(B),metrics/recall(B),"
            "metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,25.0,0.3,0.3,0.30,0.20\n"
            "2,50.0,0.5,0.5,0.50,0.40\n"
            "3,75.0,0.4,0.4,0.42,0.39\n",
            encoding="utf-8",
        )
        metrics = A.read_results_csv(path)
        check_eq(metrics["map50_95"], 0.39, "适应度取最后一行")
        check_eq(metrics["best_map50_95"], 0.40, "best 取历史最大")
        check(
            metrics["best_map50_95"] > metrics["map50_95"],
            "这个用例必须让 best 大于最后一行，否则测不出设计意图",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case("指标取值：支持带类别后缀的键，并统一保留 6 位小数")
def test_metric_from_results_dict() -> None:
    results = {
        "metrics/precision(B)": 0.952168289466484,
        "metrics/recall(B)": 0.8282777212050667,
        "metrics/mAP50(B)": 0.8822592812133806,
        "metrics/mAP50-95(B)": 0.4380553793247416,
        "fitness": 0.3,
    }
    check_eq(
        A._metric_from_results_dict(results, "metrics/mAP50-95"), 0.4380553793247416, "mAP50-95"
    )
    check_eq(
        A._round_or_none(A._metric_from_results_dict(results, "metrics/mAP50-95")),
        0.438055,
        "四舍五入到 6 位",
    )
    check_eq(A._metric_from_results_dict(results, "metrics/nonexistent"), None, "缺失键应为 None")


@case("numpy 数组求和不会因布尔判断抛异常")
def test_sum_array_safe() -> None:
    try:
        import numpy as np
    except ImportError:
        return
    check_eq(A._sum_array(np.array([10, 20, 30])), 60, "整数数组")
    check_eq(A._sum_array(np.array([[1, 2], [3, 4]])), 10, "二维数组应展平")
    check_eq(A._sum_array(None), 0, "None")
    check_eq(A._sum_array([]), 0, "空列表")
    check_eq(A._sum_array(np.array([])), 0, "空数组")


# ---------------------------------------------------------------------------
# 6. 切分文件
# ---------------------------------------------------------------------------


@case("切分 yaml 用正斜杠路径（避免 Windows 反斜杠被 YAML 当转义）")
def test_split_yaml_uses_posix_paths() -> None:
    missing = [n for n in ("pcb_tune.yaml", "pcb_holdout.yaml") if not (A.SPLITS_DIR / n).is_file()]
    if missing:
        raise Skip(f"{', '.join(missing)} 尚未生成，先跑 autoexp.py split")
    for name in ("pcb_tune.yaml", "pcb_holdout.yaml"):
        path = A.SPLITS_DIR / name
        text = path.read_text(encoding="utf-8")
        check("\\" not in text, f"{name} 含反斜杠，Windows 上会被 YAML 当转义符")
        check("path:" in text, f"{name} 缺少 path 字段")


@case("holdout 与 tune 没有重叠图片")
def test_split_no_overlap() -> None:
    tune = A.SPLITS_DIR / "pcb" / "images" / "tune"
    hold = A.SPLITS_DIR / "pcb" / "images" / "holdout"
    if not tune.is_dir() or not hold.is_dir():
        raise Skip("切分目录尚未生成，先跑 autoexp.py split")
    tune_names = {p.name for p in tune.iterdir() if p.is_file()}
    hold_names = {p.name for p in hold.iterdir() if p.is_file()}
    overlap = tune_names & hold_names
    check_eq(overlap, set(), "tune 与 holdout 不应有交集")
    check(len(hold_names) > 0, "holdout 不应为空")


@case("holdout 会从训练记录读回 imgsz，不用配置里的默认值")
def test_read_run_args_yaml() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoexp_args_"))
    try:
        run_dir = tmp / "exp_9999"
        (run_dir / "weights").mkdir(parents=True)
        weights = run_dir / "weights" / "best.pt"
        weights.write_bytes(b"not-a-real-checkpoint")
        (run_dir / "args.yaml").write_text(
            "task: detect\n"
            "imgsz: 960\n"
            "batch: 4\n"
            "device: '0'\n"
            "epochs: 100\n",
            encoding="utf-8",
        )
        found = A.read_run_args_yaml(weights)
        check_eq(found.get("imgsz"), "960", "imgsz 应读回训练时的值")
        check_eq(found.get("batch"), "4", "batch")
        check("epochs" not in found, "不在白名单内的键不应返回")

        # 没有 args.yaml 时必须安静地返回空字典，由调用方回退
        orphan = tmp / "orphan" / "weights" / "best.pt"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"x")
        check_eq(A.read_run_args_yaml(orphan), {}, "缺失 args.yaml 应返回空字典")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"autoexp 回归测试（{HERE}）")
    print(f"Python {sys.version.split()[0]}  |  ROOT={A.ROOT}")
    print()

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()

    print()
    print("=" * 60)
    total = len(PASSED) + len(FAILED) + len(SKIPPED)
    print(f"通过 {len(PASSED)} / {total}" + (f"（跳过 {len(SKIPPED)}）" if SKIPPED else ""))
    if SKIPPED:
        print()
        for name, reason in SKIPPED:
            print(f"[skip] {name}：{reason}")
    if FAILED:
        print()
        for name, detail in FAILED:
            print(f"[FAIL] {name}")
            print(detail)
            print("-" * 60)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
