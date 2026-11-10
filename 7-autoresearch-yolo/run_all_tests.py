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


class CleanLlmEnv:
    """临时摘掉 LLM 相关的环境变量。

    测试必须自己控制环境：开发机上经常残留 ``AUTOEXP_MODEL`` / ``AUTOEXP_BASE_URL``
    （比如刚跑过离线 mock），不清掉的话用例结果会随「谁的终端开着」而变。
    """

    NAMES = ("AUTOEXP_BASE_URL", "AUTOEXP_MODEL", "AUTOEXP_LLM_CONFIG")

    def __enter__(self):
        import os as _os
        self._saved = {name: _os.environ.pop(name, None) for name in self.NAMES}
        return self

    def __exit__(self, *exc):
        import os as _os
        for name, value in self._saved.items():
            if value is not None:
                _os.environ[name] = value
        return False


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
# 8. LLM 决策层（agent）
# ---------------------------------------------------------------------------


def _agent_fixtures():
    base = A.load_json(A.BASE_JSON)
    space = A.load_json(A.SPACE_JSON)
    return base, space


@case("提议只改一个键时，其余参数从基线继承")
def test_proposal_merge_keeps_baseline() -> None:
    base, space = _agent_fixtures()
    errors, merged = A.validate_proposal({"train": {"batch": "4", "imgsz": 960}}, base, space)
    check_eq(errors, [], "不应有错误")
    check_eq(merged["train"]["imgsz"], 960, "imgsz 应被覆盖")
    check_eq(merged["train"]["batch"], "4", "batch 应被覆盖")
    # 没提到的键必须原样保留，否则模型每轮都要重抄 40 多个参数，迟早漏
    check_eq(merged["train"]["lr0"], base["train"]["lr0"], "lr0 应继承基线")
    check_eq(merged["data"], base["data"], "data 应保持基线")
    check_eq(merged["budget_seconds"], base["budget_seconds"], "预算应保持基线")


@case("提议不许改 data（holdout 纪律的硬红线）")
def test_proposal_rejects_changing_data() -> None:
    base, space = _agent_fixtures()
    errors, merged = A.validate_proposal({"data": "splits/pcb_holdout.yaml"}, base, space)
    check(merged is None, "越权提议不应返回配置")
    check(any("冻结" in item for item in errors), f"应报冻结错误，实际：{errors}")


@case("即使搜索空间放开了 data，也仍然不许指向 holdout")
def test_proposal_rejects_holdout_even_if_space_allows() -> None:
    base, _space = _agent_fixtures()
    # 模拟「有人为了让 agent 更自由而把 data 加进 space.json」这种事
    lax_space = {
        "train": {},
        "data": {"enum": ["splits/pcb_tune.yaml", "splits/pcb_holdout.yaml"]},
    }
    errors, merged = A.validate_proposal({"data": "splits/pcb_holdout.yaml"}, base, lax_space)
    check(merged is None, "必须拒绝")
    check(any("冻结" in item for item in errors), f"冻结字段应优先拦住，实际：{errors}")


@case("提议越界（imgsz 不在枚举里）被拒绝")
def test_proposal_rejects_out_of_space() -> None:
    base, space = _agent_fixtures()
    errors, merged = A.validate_proposal({"train": {"imgsz": 5000}}, base, space)
    check(merged is None, "应拒绝")
    check(any("imgsz" in item for item in errors), f"错误应点名 imgsz，实际：{errors}")


@case("提议用数字代替字符串枚举（batch=4）被拒绝")
def test_proposal_rejects_wrong_type_enum() -> None:
    base, space = _agent_fixtures()
    # space.json 里 batch 是字符串枚举，模型很可能直接给数字，这是高频真实错误
    errors, _merged = A.validate_proposal({"train": {"batch": 4}}, base, space)
    check(errors, "数字 4 不在 [\"4\",\"8\",\"16\",\"-1\"] 内，应被拒绝")


@case("提议里出现搜索空间未声明的参数被拒绝")
def test_proposal_rejects_unknown_train_key() -> None:
    base, space = _agent_fixtures()
    errors, _merged = A.validate_proposal({"train": {"magic_param": 1}}, base, space)
    check(any("magic_param" in item for item in errors), f"应点名未知参数，实际：{errors}")


@case("提议里出现冻结的训练参数被拒绝")
def test_proposal_rejects_frozen_train_key() -> None:
    base, space = _agent_fixtures()
    errors, _merged = A.validate_proposal({"train": {"workers": 8}}, base, space)
    check(any("冻结" in item for item in errors), f"workers 应被冻结，实际：{errors}")


@case("提议里出现不允许的顶层字段被拒绝")
def test_proposal_rejects_extra_top_level() -> None:
    base, space = _agent_fixtures()
    errors, _merged = A.validate_proposal({"epochs": 300, "train": {}}, base, space)
    check(any("顶层字段" in item for item in errors), f"应拒绝未知顶层字段，实际：{errors}")


@case("判定只认噪声阈值：阈值内一律 DISCARD")
def test_judge_uses_noise_threshold() -> None:
    threshold = 0.0064
    verdict, _reason = A.judge_round(0.6037, 0.6000, threshold)
    check_eq(verdict, "DISCARD", "+0.0037 是噪声，必须 DISCARD")
    verdict, _reason = A.judge_round(0.6063, 0.6000, threshold)
    check_eq(verdict, "DISCARD", "+0.0063 仍未超过 0.0064")
    verdict, _reason = A.judge_round(0.6065, 0.6000, threshold)
    check_eq(verdict, "KEEP", "+0.0065 超过阈值应 KEEP")
    verdict, _reason = A.judge_round(0.5900, 0.6000, threshold)
    check_eq(verdict, "DISCARD", "下降必须 DISCARD")


@case("首轮与崩溃轮的判定")
def test_judge_first_and_crash() -> None:
    verdict, _ = A.judge_round(0.5, None, 0.0064)
    check_eq(verdict, "KEEP", "首轮没有基准，应直接 KEEP")
    verdict, _ = A.judge_round(None, 0.5, 0.0064)
    check_eq(verdict, "DISCARD", "无指标应 DISCARD")


@case("判定：真实负向要说「下降」，不能一律说「无法与随机波动区分」")
def test_judge_reports_real_regression() -> None:
    """2026-11-10 那场里 -0.029 / -0.067 都被写成「无法与随机波动区分」，
    读起来像「没什么差别」，实际是两个远超阈值的真实退步。"""
    verdict, reason = A.judge_round(0.4297, 0.45862, 0.006368)
    check_eq(verdict, "DISCARD", "下降必须 DISCARD")
    check("下降" in reason, f"应点明是下降：{reason}")
    check("负向" in reason, f"应说明这是真实负向：{reason}")


@case("判定：阈值未校准时必须在理由里说清楚")
def test_judge_warns_uncalibrated() -> None:
    """用别的分辨率测出来的 σ 去判 960，可能把噪声当信号、也可能漏掉真提升。
    这件事必须出现在台账和界面上，否则人会去怀疑模型而不是去补测噪声。"""
    _v, reason = A.judge_round(0.43, 0.42, 0.006368, calibrated=False,
                               group="imgsz=960|batch=4")
    check("没测过" in reason, f"应说明噪声底没测过：{reason}")
    check("imgsz=960" in reason, f"应点名是哪一组：{reason}")
    _v, reason = A.judge_round(0.43, 0.42, 0.006368, calibrated=True,
                               group="imgsz=960|batch=4")
    check("没测过" not in reason, f"已校准就不该再警告：{reason}")


@case("噪声底分组：优先用实测 groups，其次用配置相同的重复实验，最后才借全局")
def test_noise_threshold_for_groups() -> None:
    """这是「KEEP 永远是 0」那个坑的核心：640 的 σ 不能拿去量 960。"""
    base, _space = _agent_fixtures()
    train_960 = {**base["train"], "imgsz": 960, "batch": "4"}
    key = A.noise_group_key(960, "4")
    check_eq(key, "imgsz=960|batch=4", "分组键格式")

    # 必须隔离真实的 _noise_summary.json：它测过 960 之后，"没有组数据"这个前提
    # 就不成立了，测试会变成在验证「真文件里有没有 960」——那不是这段要测的东西。
    empty_noise = Path(tempfile.mkdtemp()) / "no_summary.json"
    original = A.NOISE_JSON
    A.NOISE_JSON = empty_noise
    try:
        # 1) 没有任何组数据：标 calibrated=False，并明说「这是别的配置的尺子」
        got = A.noise_threshold_for(
            [], train_960,
            global_threshold=0.006368, global_sigma=0.003184, global_source="全局实测",
        )
        check_eq(got["calibrated"], False, "没测过就必须标未校准")
        check_eq(got["threshold"], 0.006368, "退回全局阈值")
        check("全局" in got["source"], f"来源要说明是全局：{got['source']}")

        # 2) _noise_summary.json 里有该组实测：必须优先用它
        noise_path = Path(tempfile.mkdtemp()) / "_noise_summary.json"
        noise_path.write_text(json.dumps({
            "sigma": 0.003184,
            "groups": {key: {"values": [0.42, 0.45, 0.44], "sigma": 0.012, "mean": 0.4367}},
        }), encoding="utf-8")
        A.NOISE_JSON = noise_path
        got = A.noise_threshold_for(
            [], train_960,
            global_threshold=0.006368, global_sigma=0.003184, global_source="全局实测",
        )
    finally:
        A.NOISE_JSON = original

    check_eq(got["calibrated"], True, "有组实测就该是已校准")
    check_eq(got["sigma"], 0.012, "应取该组的 sigma，而不是全局的 0.003184")
    check_eq(got["threshold"], 0.024, "阈值 = 2 sigma")
    check("groups" in got["source"], f"来源应指向 groups：{got['source']}")


@case("噪声底分组：台账里配置完全相同的重复实验可作为校准依据")
def test_noise_threshold_uses_replicates() -> None:
    base, _space = _agent_fixtures()
    train_cfg = {**base["train"], "imgsz": 960, "batch": "4"}
    signature = A.train_signature(train_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp)
        # 三次配置完全相同、指标不同的实验
        for exp_id, metrics in (("e1", 0.44), ("e2", 0.45), ("e3", 0.43)):
            (exp_dir / f"{exp_id}.json").write_text(json.dumps({
                "exp_id": exp_id,
                "resolved": {**base, "train": train_cfg},
            }), encoding="utf-8")
        rows = [
            {"exp_id": "e1", "status": "done", "fitness_map50_95": "0.44"},
            {"exp_id": "e2", "status": "done", "fitness_map50_95": "0.45"},
            {"exp_id": "e3", "status": "done", "fitness_map50_95": "0.43"},
        ]
        original = A.EXPERIMENTS_DIR
        original_noise = A.NOISE_JSON
        A.EXPERIMENTS_DIR = exp_dir
        # 同样要隔离 NOISE_JSON：真实文件里一旦有了该组的实测，groups 就会抢先命中，
        # 这段「用重复实验校准」的逻辑根本不会被执行，测试就成了摆设。
        A.NOISE_JSON = exp_dir / "_no_summary.json"
        try:
            got = A.noise_threshold_for(
                rows, train_cfg,
                global_threshold=0.006368, global_sigma=0.003184, global_source="全局实测",
            )
            # 一个跑崩的行不能算进样本
            got_with_crash = A.noise_threshold_for(
                rows + [{"exp_id": "e4", "status": "crash", "fitness_map50_95": ""}],
                train_cfg,
                global_threshold=0.006368, global_sigma=0.003184, global_source="全局实测",
            )
        finally:
            A.EXPERIMENTS_DIR = original
            A.NOISE_JSON = original_noise

    check(signature, "签名不应为空")
    check_eq(got["calibrated"], True, "有重复实验就该是已校准")
    check_eq(got["samples"], 3, "应找到 3 个重复样本")
    check(abs(got["sigma"] - 0.008165) < 0.0001,
          f"0.43/0.44/0.45 的 pstdev 约 0.008165，实际 {got['sigma']}")
    check_eq(got_with_crash["samples"], 3, "crash 行不该被算进样本")


@case("Ultralytics 的降 batch 提示必须能被识别（这是中止训练的触发条件）")
def test_batch_downgrade_pattern() -> None:
    lines = [
        "WARNING CUDA out of memory with batch=4. Reducing to batch=2 and retrying (1/3).",
        "CUDA out of memory with batch=8. Reducing to batch=4 and retrying (1/3).",
    ]
    for line in lines:
        hit = A.BATCH_DOWNGRADE_PATTERN.search(line)
        check(hit is not None, f"应命中：{line}")

    hit = A.BATCH_DOWNGRADE_PATTERN.search(lines[1])
    check_eq(hit.group(1), "4", "应抓到降级后的 batch 值")

    # 正常日志不能被误判成降级，否则每一轮都会在开头被自己中止
    for line in (
        "TRTFORGE_JSON {\"type\":\"progress\",\"epoch\":1}",
        "4/10  2.2G  2.008  2.12  1.162  15  960: 76% 106/139",
        "batch=4 imgsz=960",
    ):
        check(A.BATCH_DOWNGRADE_PATTERN.search(line) is None, f"不应命中：{line}")


@case("提议继承：未写的键要取「当前最优」，不是「基线」")
def test_proposal_inherits_incumbent() -> None:
    """exp_0017 的真实事故：模型写 `changed: imgsz 960 不变` 但只给了 warmup，
    程序按基线合并成 640，整轮验证的根本不是同一个假设。"""
    base, space = _agent_fixtures()
    inherit = {"imgsz": 960, "batch": "4"}

    calls: list[list[dict]] = []

    class FakeLlm:
        LLMError = RuntimeError
        LLMTruncated = type("LLMTruncated", (RuntimeError,), {})

        @staticmethod
        def chat(messages, **_kwargs):
            calls.append(list(messages))
            return {
                "content": json.dumps({
                    "hypothesis": "只改 warmup",
                    "changed": "warmup 3 -> 1，imgsz 960 / batch 4 不变",
                    "reasoning": "短",
                    "train": {"warmup_epochs": 1.0},
                }, ensure_ascii=False),
                "usage": {},
            }

    proposal, meta = A._ask_proposal(
        FakeLlm, [{"role": "user", "content": "x"}], base, space,
        inherit_train=inherit,
    )
    check(proposal is not None, f"应拿到提议，错误：{meta.get('error')}")
    check_eq(meta["resolved"]["train"]["imgsz"], 960,
             "未写 imgsz 时必须继承当前最优的 960，而不是基线的 640")
    check_eq(meta["resolved"]["train"]["batch"], "4", "batch 同理")
    check_eq(meta["resolved"]["train"]["warmup_epochs"], 1.0, "本轮改动仍要生效")
    # 提议原文必须保持模型写的样子，审计要能对上「它当时说了什么」
    check_eq(meta["proposal"]["train"], {"warmup_epochs": 1.0}, "原始提议不该被改写")


@case("噪声阈值恒为正数且不超过模型的判断空间")
def test_noise_threshold_sane() -> None:
    threshold, sigma, source = A.noise_threshold()
    check(threshold > 0, f"阈值必须为正，否则任何波动都会被判成 KEEP（实际 {threshold}）")
    check(threshold < 0.05, f"阈值不应大到任何改进都无法通过（实际 {threshold}）")
    if sigma is not None:
        check(abs(threshold - 2 * sigma) < 1e-6, "阈值应等于 2 sigma")
    check(isinstance(source, str) and source, "来源说明不能为空")


@case("模型输出带代码块或闲聊也能解析")
def test_parse_proposal_tolerates_formatting() -> None:
    fenced = '```json\n{"hypothesis":"h","changed":"c","train":{"imgsz":960}}\n```'
    check_eq(A.parse_proposal(fenced)["train"]["imgsz"], 960, "代码块内的 JSON")
    chatty = '好的，我的提议是：{"hypothesis":"h","train":{}}，请审查。'
    check_eq(A.parse_proposal(chatty)["train"], {}, "前后有闲聊也要能解析")
    spaced = '  \n {"hypothesis":"h","train":{"mosaic":0.5}} \n '
    check_eq(A.parse_proposal(spaced)["train"]["mosaic"], 0.5, "首尾空白")


@case("模型输出完全不是 JSON 时抛 ProposalError")
def test_parse_proposal_garbage() -> None:
    for bad in ("", "我不知道该改什么。", "[1,2,3]"):
        try:
            A.parse_proposal(bad)
        except A.ProposalError:
            continue
        raise AssertionError(f"应抛 ProposalError，实际接受了 {bad!r}")


@case("历史 delta 以「当时的历史最佳」为基准")
def test_history_delta_uses_running_best() -> None:
    rows = [
        {"exp_id": "e1", "status": "done", "fitness_map50_95": "0.60", "changed": "a"},
        {"exp_id": "e2", "status": "done", "fitness_map50_95": "0.59", "changed": "b"},
        {"exp_id": "e3", "status": "done", "fitness_map50_95": "0.61", "changed": "c"},
        {"exp_id": "e4", "status": "crash", "fitness_map50_95": "", "changed": "d"},
    ]
    original = A.EXPERIMENTS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        A.EXPERIMENTS_DIR = Path(tmp)  # 没有快照文件，params_diff 为空
        try:
            entries = A.history_with_delta(rows, {"train": {}})
        finally:
            A.EXPERIMENTS_DIR = original
    check_eq(entries[0]["delta_vs_best_then"], None, "第一条没有基准")
    check_eq(entries[1]["delta_vs_best_then"], -0.01, "第二条应相对 0.60 为 -0.01")
    check_eq(entries[2]["delta_vs_best_then"], 0.01, "第三条应相对 0.60 为 +0.01")
    check_eq(entries[3]["fitness"], None, "崩溃行没有指标")


@case("给模型的上下文里绝不出现 holdout 字样")
def test_agent_context_hides_holdout() -> None:
    base, space = _agent_fixtures()
    context = A.build_agent_context([], base, space, 1, 10)
    text = json.dumps(context, ensure_ascii=False)
    check("holdout" not in text.lower(), "上下文里出现了 holdout：这是纪律红线")
    # 冻结字段要明确告诉模型，否则它会一直尝试
    check("data" in context["frozen_fields"], "data 必须在冻结清单里")
    check("model" in context["frozen_fields"], "第一版 model 应冻结")
    # 允许改的清单里不能混进冻结项
    check("cache" not in context["allowed_train"], "cache 被冻结，不应出现在 allowed_train")
    check("imgsz" in context["allowed_train"], "imgsz 应可调")


@case("上下文必须给出允许的**取值**，不只是键名")
def test_agent_context_includes_space_values() -> None:
    """曾经只给键名，模型只能猜，于是提 imgsz=1024 / batch=2 这种必然越界的值。

    一轮被这样废掉就是 11 分钟，所以这条是硬要求，不是「有更好」。
    """
    base, space = _agent_fixtures()
    context = A.build_agent_context([], base, space, 1, 10)
    space_out = context.get("search_space") or {}
    check(space_out, "上下文里必须有 search_space")
    train_rules = space_out.get("train") or {}
    check("imgsz" in train_rules, "train.imgsz 的取值约束必须给出")
    enum = train_rules["imgsz"].get("enum")
    check(enum, f"imgsz 应带 enum，实际 {train_rules['imgsz']!r}")
    check(960 in enum, f"enum 里应有 960：{enum}")
    check(1024 not in enum, f"1024 不在 enum，不该出现：{enum}")
    joined = json.dumps(train_rules, ensure_ascii=False)
    check('"4"' in joined or "4" in joined, "batch 的允许值应给出")

    # 冻结键不该出现在 search_space 里：写出来只会诱导模型去改
    for frozen in ("cache", "workers", "patience"):
        if frozen in train_rules:
            raise AssertionError(f"冻结键 {frozen} 不该出现在 search_space")


@case("停止判据：errors 与 stale 必须分开报，文案不能共用")
def test_evaluate_stop() -> None:
    # 都没到上限：继续
    check_eq(A.evaluate_stop(stale=0, max_stale=5, errors=0, max_errors=3,
                             now=100.0, deadline=None), None, "未到上限应继续")
    # 搜索无进展
    code, message = A.evaluate_stop(stale=5, max_stale=5, errors=0, max_errors=3,
                                    now=100.0, deadline=None)
    check_eq(code, "stale", "stale 满应报 stale")
    check("没有进展" in message, f"文案应说「没有进展」：{message}")
    # 模型侧故障：这条以前被错报成「连续 5 轮没有进展」
    code, message = A.evaluate_stop(stale=0, max_stale=5, errors=3, max_errors=3,
                                    now=100.0, deadline=None)
    check_eq(code, "errors", "errors 满应报 errors")
    check("max_tokens" in message, f"应直接点名 max_tokens 这个可能原因：{message}")
    check("没有进展" not in message, "模型侧故障不能说成「搜索没有进展」")
    # 两者同时到：优先报 errors，因为它才是可操作的那条
    code, _ = A.evaluate_stop(stale=9, max_stale=5, errors=9, max_errors=3,
                              now=100.0, deadline=None)
    check_eq(code, "errors", "两条同时触发时应先报 errors")
    # 时间上限
    code, _ = A.evaluate_stop(stale=0, max_stale=5, errors=0, max_errors=3,
                              now=200.0, deadline=100.0)
    check_eq(code, "deadline", "超时应报 deadline")


@case("系统提示词存在且要求 JSON 输出")
def test_prompt_exists() -> None:
    check(A.PROMPT_MD.is_file(), f"缺少提示词文件 {A.PROMPT_MD}")
    text = A.PROMPT_MD.read_text(encoding="utf-8")
    # response_format=json_object 要求 messages 里出现 json 字样，否则上游直接 400
    check("json" in text.lower(), "提示词里必须出现 json 字样，否则上游会拒绝 json_object 模式")
    check("train" in text, "提示词应说明 train 字段")
    messages = A.render_messages({"round": 1, "history": []})
    check_eq(len(messages), 2, "应产生 system + user 两条消息")
    check_eq(messages[0]["role"], "system", "第一条是 system")


@case("emit_event 默认静默，开启后才写 stdout")
def test_emit_event_is_gated() -> None:
    import io as _io
    from contextlib import redirect_stdout

    original = A.EVENTS_ENABLED
    try:
        A.EVENTS_ENABLED = False
        buffer = _io.StringIO()
        with redirect_stdout(buffer):
            A.emit_event({"type": "agent", "name": "x"})
        check_eq(buffer.getvalue(), "", "关闭时必须一个字节都不输出")

        A.EVENTS_ENABLED = True
        buffer = _io.StringIO()
        with redirect_stdout(buffer):
            A.emit_event({"type": "agent", "name": "x", "中文": "值"})
        text = buffer.getvalue()
        check(text.startswith(A.JSON_PREFIX), f"应带统一前缀，实际：{text!r}")
        check("中文" in text, "中文必须原样输出（ensure_ascii=False），否则界面上是乱码")
    finally:
        A.EVENTS_ENABLED = original


@case("update_best 只在 KEEP 时被调用，并写出可追溯的证据")
def test_update_best_writes_evidence() -> None:
    original_best = A.BEST_JSON
    original_noise = A.NOISE_JSON
    with tempfile.TemporaryDirectory() as tmp:
        A.BEST_JSON = Path(tmp) / "best.json"
        A.NOISE_JSON = A.ROOT / "experiments" / "_noise_summary.json"
        try:
            row = {"exp_id": "exp_test", "fitness_map50_95": "0.6100"}
            cfg = A.deep_merge(A.load_json(A.BASE_JSON), {"train": {"imgsz": 1152, "batch": "4"}})
            A.update_best("exp_test", row, cfg, 0.006368, 0.003184)
            payload = json.loads(A.BEST_JSON.read_text(encoding="utf-8"))
        finally:
            A.BEST_JSON = original_best
            A.NOISE_JSON = original_noise

    check_eq(payload["exp_id"], "exp_test", "exp_id")
    check_eq(payload["train"], {"imgsz": 1152, "batch": "4"}, "只记录相对基线的改动")
    evidence = payload["evidence"]
    check_eq(evidence["tune_map50_95"], 0.61, "应记录本次指标")
    check_eq(evidence["keep_threshold_2sigma"], 0.006368, "应记录判定阈值")
    # holdout 必须留空并写明要人工验收：搜索期自动跑 holdout 会破坏纪律
    check_eq(evidence["holdout_map50_95"], None, "holdout 不应被自动填写")
    check("holdout" in payload["note"], "note 里应提示 holdout 需人工验收")


# ---------------------------------------------------------------------------
# 9. llm.py（不联网）
# ---------------------------------------------------------------------------


@case("llm.mask_key 不泄露 Key")
def test_llm_mask_key() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")
    check_eq(llm.mask_key("sk-abcdefgh1234"), "sk-****1234", "遮罩格式")
    check_eq(llm.mask_key("short"), "*****", "短 Key 必须全遮")
    check_eq(llm.mask_key(""), "", "空 Key")


@case("llm.chat 用假 transport 跑通，且 400 时自动去掉 json_object 重试")
def test_llm_chat_offline() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")

    calls: list[dict] = []

    def fake_post(url, body, api_key, timeout):
        calls.append(dict(body))
        if "response_format" in body:
            # 模拟「兼容接口不认识 response_format」
            raise llm.LLMError("bad request", status=400, retryable=False)
        return {
            "choices": [{"message": {"content": '{"hypothesis":"h","train":{}}'}}],
            "usage": {"total_tokens": 12},
            "model": "fake-model",
        }

    original = llm._post_json
    llm._post_json = fake_post
    try:
        result = llm.chat(
            [{"role": "user", "content": "请给 json"}],
            config={"base_url": "https://example.invalid", "max_retries": 3},
            key="sk-test-key",
        )
    finally:
        llm._post_json = original

    check_eq(len(calls), 2, "第一次被 400 拒后应去掉 response_format 再试一次")
    check("response_format" not in calls[-1], "重试时不应再带 response_format")
    check_eq(result["content"], '{"hypothesis":"h","train":{}}', "内容")
    check_eq(result["usage"]["total_tokens"], 12, "usage 应透传")
    # Key 绝不能出现在请求体里
    check("sk-test-key" not in json.dumps(calls, ensure_ascii=False), "Key 不应出现在请求体")


@case("llm.resolve_key 优先显式传入，其次环境变量")
def test_llm_resolve_key() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")
    import os as _os

    saved = {name: _os.environ.pop(name, None) for name in llm.KEY_ENVS}
    try:
        check_eq(llm.resolve_key(), "", "没有 Key 时应返回空串")
        _os.environ[llm.KEY_ENVS[0]] = "sk-from-env"
        check_eq(llm.resolve_key(), "sk-from-env", "应读环境变量")
        check_eq(llm.resolve_key("sk-explicit"), "sk-explicit", "显式传入优先")
    finally:
        for name, value in saved.items():
            _os.environ.pop(name, None)
            if value is not None:
                _os.environ[name] = value


@case("llm.load_config 会把 api_key 字段剔除（防止随配置流进日志）")
def test_llm_config_drops_key() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")
    with CleanLlmEnv():
        cfg = llm.load_config({"api_key": "sk-should-be-dropped", "model": "m"})
    check("api_key" not in cfg, "api_key 不应留在配置字典里")
    check_eq(cfg["model"], "m", "其它字段应保留")


@case("llm.chat 在 finish_reason=length 时自动加预算，而不是把半截 JSON 丢给上层")
def test_llm_chat_grows_budget_on_truncation() -> None:
    """这是 2026-11-10 那场实测事故的回归用例。

    现场：max_tokens=900，推理型模型思考用掉 900，正文剩 0。
    现象被报成「返回内容不是合法 JSON」，人只会去改提示词，方向完全错。
    正确行为：加预算重来（思考 + 正文共用预算，只能靠加大预算解决）。
    """
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")

    calls: list[dict] = []

    def fake_post(url, body, api_key, timeout):
        calls.append(dict(body))
        if len(calls) == 1:
            # 第一次：思考吃掉全部预算，正文只有半截 JSON
            return {
                "choices": [{"message": {"content": '{"hypothesis": "推到 1024'},
                             "finish_reason": "length"}],
                "usage": {"completion_tokens": 900,
                          "completion_tokens_details": {"reasoning_tokens": 900}},
                "model": "fake-reasoner",
            }
        return {
            "choices": [{"message": {"content": '{"hypothesis":"h","train":{}}'},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 500,
                      "completion_tokens_details": {"reasoning_tokens": 300}},
            "model": "fake-reasoner",
        }

    original = llm._post_json
    llm._post_json = fake_post
    try:
        result = llm.chat(
            [{"role": "user", "content": "请给 json"}],
            config={"base_url": "https://example.invalid", "max_retries": 3,
                    "max_tokens": 900, "temperature": 0.7},
            key="sk-test-key",
        )
    finally:
        llm._post_json = original

    check_eq(len(calls), 2, "截断后应重试一次")
    check_eq(calls[0]["max_tokens"], 900, "第一次用配置里的预算")
    check(calls[1]["max_tokens"] > calls[0]["max_tokens"],
          f"第二次必须加预算，实际 {calls[0]['max_tokens']} -> {calls[1]['max_tokens']}")
    check_eq(result["content"], '{"hypothesis":"h","train":{}}', "应返回重试后的完整内容")
    check_eq(result["finish_reason"], "stop", "应透出 finish_reason")
    check_eq(result["reasoning_tokens"], 300, "应透出推理 token 数")


@case("llm.chat 预算加到上限仍被截断时，报「被截断」而不是「不是合法 JSON」")
def test_llm_chat_truncation_error_is_actionable() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")

    budgets: list[int] = []

    def fake_post(url, body, api_key, timeout):
        budgets.append(int(body.get("max_tokens") or 0))
        return {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"completion_tokens": int(body.get("max_tokens") or 0),
                      "completion_tokens_details": {"reasoning_tokens": int(body.get("max_tokens") or 0)}},
        }

    original = llm._post_json
    llm._post_json = fake_post
    try:
        try:
            llm.chat(
                [{"role": "user", "content": "请给 json"}],
                config={"base_url": "https://example.invalid", "max_retries": 3,
                        "max_tokens": 900},
                key="sk-test-key",
            )
        except llm.LLMTruncated as exc:
            message = str(exc)
        else:
            raise AssertionError("应当抛出 LLMTruncated")
    finally:
        llm._post_json = original

    check(len(budgets) == 3, f"3 次尝试都应打满，实际 {budgets}")
    check(budgets == sorted(budgets) and budgets[0] < budgets[-1],
          f"预算应逐次加大，实际 {budgets}")
    # 必须能一句话定位到该改哪个数字
    check("max_tokens" in message, f"报错要点名 max_tokens：{message}")
    check("截断" in message, f"报错要说「截断」：{message}")
    check("4096" in message, f"报错要给出建议值：{message}")
    check("不是合法 JSON" not in message, "截断绝不能说成「不是合法 JSON」")


@case("llm.ping 遇到推理型模型截断时仍报「连通」")
def test_llm_ping_tolerates_truncation() -> None:
    """自检用极小预算，推理型模型必然被截断。那是预期行为，不是配置错。

    这个用例防的是一类很坑的回归：修好截断逻辑之后，顺手把「测试连通性」
    弄成了永远失败，用户会以为自己的 Key 坏了。
    """
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")

    def fake_post(url, body, api_key, timeout):
        return {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 64,
                      "completion_tokens_details": {"reasoning_tokens": 64}},
            "model": "fake-reasoner",
        }

    original = llm._post_json
    llm._post_json = fake_post
    try:
        result = llm.ping(config={"base_url": "https://example.invalid", "max_retries": 1},
                          key="sk-test-key")
    finally:
        llm._post_json = original

    check_eq(result["ok"], True, "截断不应被当成连通性失败")
    check_eq(result["truncated"], True, "应标明这是截断")
    check(result["reply"], "reply 应给出人话解释")


@case("llm.load_config 的优先级：显式指定 > 环境变量 > 入库文件")
def test_llm_config_precedence() -> None:
    llm = A.load_llm()
    if llm is None:
        raise Skip("找不到 llm.py")
    import os as _os

    with CleanLlmEnv():
        # 环境变量能压过入库的 configs/llm.json（CI 换端点不用改仓库）
        _os.environ["AUTOEXP_MODEL"] = "from-env"
        check_eq(llm.load_config()["model"], "from-env", "环境变量应压过入库文件")
        # 但压不过本次显式指定：否则页面/命令行上的选择会被 shell 里的陈旧
        # 变量静默改掉，用户会以为界面坏了
        got = llm.load_config({"model": "explicit"})["model"]
        check_eq(got, "explicit", "显式指定必须压过环境变量")


# ---------------------------------------------------------------------------
# 5. 事件转发与状态判定（网页端每一轮都会走这条路）
# ---------------------------------------------------------------------------


@case("derive_result_status：没有 proc 也能定状态（--events 分支曾经 NameError）")
def test_derive_result_status() -> None:
    # 成功但结果文件里只写 ok：统一归一成 done，台账和界面都认这个词
    check_eq(A.derive_result_status({"status": "ok"}, 0), "done", "ok 归一为 done")
    # 结果文件自带 done 时不能被改坏
    check_eq(A.derive_result_status({"status": "done"}, 0), "done", "done 保持不变")
    # 子进程非零退出但结果文件缺失：判 crash 而不是当成完成
    check_eq(A.derive_result_status({}, 1), "crash", "非零退出应判 crash")
    # 子进程零退出但结果文件缺失：也只能算 ok→done，不能凭空判 crash
    check_eq(A.derive_result_status({}, 0), "done", "零退出且无结果按 done")
    # 训练自己写下的失败原因必须原样保留，不能被覆盖成 done
    check_eq(A.derive_result_status({"status": "oom"}, 0), "oom", "保留子进程给的状态")


@case("_exec_tee 把子进程输出同时写进日志和 stdout，并返回退出码")
def test_exec_tee_passthrough() -> None:
    import contextlib
    import io as _io

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "tee.log"
        # 用真实子进程而不是 mock：这里要看的就是「逐行转发 + 落盘」是否真的成立
        code = (
            "import sys, json;"
            "print('TRTFORGE_JSON ' + json.dumps({'type': 'progress', 'pct': 1.0}));"
            "print('普通日志一行');"
            "sys.exit(0)"
        )
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code_rc = A._exec_tee([sys.executable, "-c", code], log_path, "exp_t", "2026-01-01T00:00:00")
        check_eq(code_rc, 0, "退出码")
        forwarded = buffer.getvalue()
        check("TRTFORGE_JSON" in forwarded, f"事件没转发到 stdout：{forwarded!r}")
        check("普通日志一行" in forwarded, f"普通日志没转发：{forwarded!r}")
        saved = log_path.read_text(encoding="utf-8")
        check("TRTFORGE_JSON" in saved, "事件没落盘，事后无法复盘")
        check("普通日志一行" in saved, "普通日志没落盘")

        # 失败退出码要如实返回：上层据此判 crash
        buffer2 = _io.StringIO()
        with contextlib.redirect_stdout(buffer2):
            fail_rc = A._exec_tee(
                [sys.executable, "-c", "import sys; sys.exit(7)"], log_path, "exp_t", "x",
            )
        check_eq(fail_rc, 7, "子进程退出码应原样返回")


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
