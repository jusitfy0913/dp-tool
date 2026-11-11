# -*- coding: utf-8 -*-
"""显存预检（实测峰值表 + 安全余量 + 自校准）与 AMP 可见性的验收脚本。

不需要 GPU、不需要 torch、不需要 DLL：所有硬件读数都被显式注入或打桩，
判定逻辑本身必须与机器无关。

守的是这几类问题（都是 v1.33.x 上真实发生过的）：

1. **公式与实测脱节**：960/batch=4 按旧公式 4760 MB 放行，实际 OOM。
   现在有样本就必须用样本；没有样本才回落到（更保守的）公式。
2. **精度混键**：同一张卡 fp16 与 fp32 的激活显存差近一倍。若共用表键，
   一张卡上的 fp16 样本会放行另一台机器的 fp32 组合。
3. **自校准污染**：一次「别的进程抢显存」的坏读数如果进了表，会长期误拦。
   因此只保留合理范围内的最大值，且失败只记日志。
4. **AMP 静默降级**：T1000 在 Ultralytics 黑名单里，训练实际是 fp32，
   页面必须能说清楚，台账必须记下来。

用法：
    python 6-web/backend/tools/test_vram_amp.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import ampcheck  # noqa: E402
import vram  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))


class _TempStore:
    """把实测峰值表指到系统 temp，跑完删掉。

    不这么做的后果：测试会往真实的 `experiments/vram_samples.json` 里塞假样本，
    之后每一次真机训练都被这些假峰值影响，而且根本看不出是测试写的。
    """

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory(prefix="trtforge_vram_")
        self.path = Path(self._dir.name) / "vram_samples.json"
        self._old = os.environ.get("TRTFORGE_VRAM_SAMPLES")
        os.environ["TRTFORGE_VRAM_SAMPLES"] = str(self.path)
        return self.path

    def __exit__(self, *_exc) -> None:
        if self._old is None:
            os.environ.pop("TRTFORGE_VRAM_SAMPLES", None)
        else:
            os.environ["TRTFORGE_VRAM_SAMPLES"] = self._old
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# 基础：键与精度归一
# ---------------------------------------------------------------------------
def test_keys_and_precision() -> None:
    check("表键格式可读", vram.sample_key(960, 4, "fp32") == "960x4xfp32", vram.sample_key(960, 4, "fp32"))
    check("布尔 True 归一为 fp16", vram.normalize_precision(True) == "fp16")
    check("布尔 False 归一为 fp32", vram.normalize_precision(False) == "fp32")
    check("字符串 bf16 归到 fp16 量级", vram.normalize_precision("bf16") == "fp16")
    check("字符串 fp32 保持不变", vram.normalize_precision("FP32") == "fp32")
    check("None 视为未知", vram.normalize_precision(None) == "unknown")
    check("0 维张量式取值可归一", vram.normalize_precision(_Scalar(0)) == "fp32")


class _Scalar:
    """模拟 trainer.amp（0 维张量）：bool(张量) 会抛异常，真实现都走 .item()。"""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


# ---------------------------------------------------------------------------
# 实测表：写入、取最大值、容错
# ---------------------------------------------------------------------------
def test_store_roundtrip() -> None:
    with _TempStore() as store:
        check("空表读出空字典", vram.load_samples() == {}, str(vram.load_samples()))

        entry = vram.record_sample(640, 8, 1769.5, precision="fp32", peak_allocated_mb=1500.0, exp_id="exp_0001")
        check("首次写入生成条目", entry.get("peak_mb") == 1769.5, str(entry))
        check("首次写入 count=1", entry.get("count") == 1, str(entry))
        check("记录了 allocated 参考值", entry.get("peak_allocated_mb") == 1500.0, str(entry))
        check("记录了实验号", entry.get("last_exp") == "exp_0001", str(entry))
        check("文件确实落盘", store.is_file(), str(store))

        again = vram.record_sample(640, 8, 2100.0, precision="fp32", exp_id="exp_0002")
        check("更大峰值覆盖旧值", again.get("peak_mb") == 2100.0, str(again))
        check("count 累加", again.get("count") == 2, str(again))

        smaller = vram.record_sample(640, 8, 1800.0, precision="fp32")
        check("更小峰值不覆盖", smaller.get("peak_mb") == 2100.0, str(smaller))
        check("count 仍然累加", smaller.get("count") == 3, str(smaller))

        loaded = vram.load_samples()
        check("读回同一条", list(loaded) == ["640x8xfp32"], str(loaded))
        check("字段完整", loaded["640x8xfp32"]["precision"] == "fp32", str(loaded))


def test_store_guards() -> None:
    with _TempStore() as store:
        check("过小的峰值被拒", "skipped" in vram.record_sample(640, 8, 5.0, "fp32"))
        check("过大的峰值被拒", "skipped" in vram.record_sample(640, 8, 5_000_000.0, "fp32"))
        check("非数字被拒", "skipped" in vram.record_sample(640, 8, "abc", "fp32"))  # type: ignore[arg-type]
        check("被拒时不写文件", not store.is_file(), str(store))

        # 文件损坏：当作空表继续跑，绝不抛异常（预检只是「更好的估算」，不能弄挂训练）
        store.write_text("{ 这不是 JSON", encoding="utf-8")
        check("损坏文件读成空表", vram.load_samples() == {}, str(vram.load_samples()))
        entry = vram.record_sample(640, 8, 2000.0, "fp32")
        check("损坏后仍能重建", entry.get("peak_mb") == 2000.0, str(entry))
        payload = json.loads(store.read_text(encoding="utf-8"))
        check("重建后结构完整", "samples" in payload and "640x8xfp32" in payload["samples"], str(payload))


# ---------------------------------------------------------------------------
# 估算与预检
# ---------------------------------------------------------------------------
def test_estimate() -> None:
    with _TempStore():
        est = vram.estimate(960, 4, "fp32")
        expected = 2600.0 + 300.0 * 4 * (960 / 640.0) ** 2
        check("无样本时用公式", est["source"] == "formula", str(est))
        check("公式系数已提到 300", abs(est["needed_mb"] - expected) < 0.2, f"{est['needed_mb']} vs {expected}")
        check("公式值同时回显", abs(est["formula_mb"] - expected) < 0.2, str(est))

        vram.record_sample(960, 4, 4000.0, "fp32")
        est = vram.estimate(960, 4, "fp32")
        check("有样本时用实测", est["source"] == "measured", str(est))
        check("实测含 15% 余量与 400MB 预留", est["needed_mb"] == round(4000.0 * 1.15 + 400, 1), str(est))
        check("实测值回显", est["measured_mb"] == 4000.0, str(est))
        check("样本次数回显", est["sample_count"] == 1, str(est))

        # 精度隔离：fp16 的样本不得用于 fp32 的判定
        est_fp16 = vram.estimate(960, 4, "fp16")
        check("fp16 不命中 fp32 样本", est_fp16["source"] == "formula", str(est_fp16))
        vram.record_sample(960, 4, 2200.0, "fp16")
        est_fp16 = vram.estimate(960, 4, "fp16")
        check("fp16 命中自己的样本", est_fp16["needed_mb"] == round(2200.0 * 1.15 + 400, 1), str(est_fp16))
        est_fp32 = vram.estimate(960, 4, "fp32")
        check("fp32 仍用自己的样本", est_fp32["measured_mb"] == 4000.0, str(est_fp32))

        # 精度未知：不查表，走保守公式（fp32 口径）
        est_unknown = vram.estimate(960, 4, "unknown")
        check("未知精度走公式", est_unknown["source"] == "formula", str(est_unknown))

        # batch 命中不到：比请求小的样本按线性外推（方向保守）
        est_b8 = vram.estimate(960, 8, "fp32")
        check("更大 batch 用实测外推", est_b8["source"] == "scaled", str(est_b8))
        check("外推按 batch 比例", est_b8["measured_mb"] == 8000.0, str(est_b8))
        check("外推后仍加余量", est_b8["needed_mb"] == round(8000.0 * 1.15 + 400, 1), str(est_b8))

        # 比请求大的样本：显存随 batch 不减，所以它是需求上界。
        # 用 640 举例：公式在 batch=7 要 4700，而 batch=8 实测只有 2132，
        # 「调小 batch 反而更吃显存」是荒唐结论，必须被上界压住。
        vram.record_sample(640, 8, 2132.0, "fp32")
        est_b7 = vram.estimate(640, 7, "fp32")
        check("更小 batch 用上方样本封顶", est_b7["source"] == "capped", str(est_b7))
        check("上界即大 batch 的实测需求", est_b7["needed_mb"] == round(2132.0 * 1.15 + 400, 1), str(est_b7))

        # 单调性：batch 减小，需求不能变大（否则「按建议调小」会越调越红）
        previous = None
        for batch in (1, 2, 3, 4, 6, 8, 12, 16):
            needed = vram.estimate(960, batch, "fp32")["needed_mb"]
            if previous is not None:
                check(
                    f"估算随 batch 单调不减（b={batch}）",
                    needed >= previous,
                    f"{previous} -> {needed}",
                )
            previous = needed


def test_preflight() -> None:
    with _TempStore():
        vram.record_sample(640, 8, 3000.0, "fp32")
        needed = round(3000.0 * 1.15 + 400, 1)

        ok = vram.preflight(640, 8, "fp32", free_mb=needed + 100)
        check("显存充足放行", ok["ok"] is True, str(ok))
        check("放行理由含实测依据", "实测" in ok["reason"], ok["reason"])

        blocked = vram.preflight(640, 8, "fp32", free_mb=needed - 100)
        check("显存不足拦截", blocked["ok"] is False, str(blocked))
        check("拦截理由含可用与需求", "可用显存" in blocked["reason"] and "需求" in blocked["reason"], blocked["reason"])
        check("拦截时给出建议", bool(blocked.get("hint")), str(blocked))
        check("按可用量反推 batch", isinstance(blocked.get("suggest_batch"), int), str(blocked))

        # 建议值必须真的能通过同一套估算（否则这条建议是骗人的）
        suggestion = blocked["suggest_batch"]
        check(
            "建议的 batch 确实能放下",
            vram.estimate(640, suggestion, "fp32")["needed_mb"] <= needed - 100,
            f"suggest={suggestion}",
        )

        # 批量自动：不适合估算，返回跳过（不能拦、更不能抛）
        skipped = vram.preflight(640, "auto", "fp32", free_mb=100)  # type: ignore[arg-type]
        check("非法 batch 返回跳过", skipped.get("skipped") is not None, str(skipped))
        check("跳过时不拦截", skipped.get("ok") is True, str(skipped))


def test_preflight_without_gpu() -> None:
    """没有 CUDA / nvidia-smi 时必须放行：这台机器本来也训不了 CUDA 训练。"""
    original = vram.gpu_info
    vram.gpu_info = lambda: None  # type: ignore[assignment]
    try:
        report = vram.preflight(640, 16, "fp32")
        check("无 GPU 时放行", report["ok"] is True, str(report))
        check("无 GPU 时说明已跳过", "跳过" in report["reason"], report["reason"])
        check("无 GPU 时可用显存为空", report["free_mb"] is None, str(report))
    finally:
        vram.gpu_info = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# AMP 判定
# ---------------------------------------------------------------------------
def test_amp_status() -> None:
    t1000 = ampcheck.amp_status("NVIDIA T1000 8GB")
    check("T1000 判定不可用", t1000["available"] is False, str(t1000))
    check("T1000 命中型号", t1000["matched"] == "t1000", str(t1000))
    check("说明里写清原因", "黑名单" in t1000["reason"], t1000["reason"])
    check("说明里写清无法强制开启", "无法强制开启" in t1000["note"], t1000["note"])
    check("预计精度 fp32", t1000["expected_precision"] == "fp32", str(t1000))

    other_black = ampcheck.amp_status("NVIDIA GeForce GTX 1660 Ti")
    check("1660 也在黑名单", other_black["available"] is False and other_black["matched"] == "1660", str(other_black))

    rtx = ampcheck.amp_status("NVIDIA GeForce RTX 4090")
    check("4090 判定可用", rtx["available"] is True, str(rtx))
    check("4090 预计 fp16", rtx["expected_precision"] == "fp16", str(rtx))

    unknown = ampcheck.amp_status(None)
    check("拿不到显卡名时状态未知", unknown["available"] is None, str(unknown))
    check("未知时精度也是未知", unknown["expected_precision"] == "unknown", str(unknown))

    check("显式关闭时预测 fp32", ampcheck.expected_precision(False, "NVIDIA GeForce RTX 4090") == "fp32")
    check("显式尝试且卡可用时预测 fp16", ampcheck.expected_precision(True, "NVIDIA GeForce RTX 4090") == "fp16")
    check("默认在黑名单卡上预测 fp32", ampcheck.expected_precision(None, "NVIDIA T1000 8GB") == "fp32")


# ---------------------------------------------------------------------------
# Web 后端接线
# ---------------------------------------------------------------------------
def test_train_command_amp() -> None:
    import app as backend_app  # noqa: PLC0415 - 导入 app 会拉起一批模块，放到用到处

    payload = {
        "task": "detect",
        "model": "yolov8n.pt",
        "data": str(Path(__file__).resolve()),
        "imgsz": 640,
        "batch": "8",
        "project": "runs/test",
        "name": "amp",
        "init_mode": "preset",
    }
    command = backend_app.build_train_command(backend_app.YoloTrainRequest(**payload), "python")
    check("默认不拼 --amp", "--amp" not in command and "--no-amp" not in command, " ".join(command))

    command = backend_app.build_train_command(
        backend_app.YoloTrainRequest(**payload, amp=False), "python"
    )
    check("amp=false 拼 --no-amp", "--no-amp" in command, " ".join(command))

    command = backend_app.build_train_command(
        backend_app.YoloTrainRequest(**payload, amp=True), "python"
    )
    check("amp=true 拼 --amp", "--amp" in command and "--no-amp" not in command, " ".join(command))

    check("auto 解析为 None", backend_app._amp_from_mode("auto") is None)
    check("off 解析为 False", backend_app._amp_from_mode("off") is False)
    check("on 解析为 True", backend_app._amp_from_mode("on") is True)


def test_vram_guard() -> None:
    from fastapi import HTTPException  # noqa: PLC0415

    import app as backend_app  # noqa: PLC0415

    payload = {
        "data": str(Path(__file__).resolve()),
        "imgsz": 960,
        "batch": "4",
        "init_mode": "preset",
    }
    original = vram.preflight
    vram.preflight = lambda *a, **kw: {  # type: ignore[assignment]
        "ok": False,
        "reason": "可用显存 4000 MB < 需求 5000 MB",
        "hint": "把 batch 降到 3",
    }
    try:
        request = backend_app.YoloTrainRequest(**payload)
        try:
            backend_app.vram_guard(request)
            check("预检不通过时抛 409", False, "没有抛异常")
        except HTTPException as exc:
            check("预检不通过时抛 409", exc.status_code == 409, str(exc.status_code))
            check("detail 带 VRAM_PREFLIGHT 前缀", str(exc.detail).startswith("VRAM_PREFLIGHT"), str(exc.detail))
            check("detail 含建议", "batch 降到 3" in str(exc.detail), str(exc.detail))

        forced = backend_app.YoloTrainRequest(**payload, allow_low_vram=True)
        report = backend_app.vram_guard(forced)
        check("allow_low_vram 时放行", report.get("ok") is False, str(report))
        check("放行时保留原始结论", "需求" in str(report.get("reason")), str(report))
    finally:
        vram.preflight = original  # type: ignore[assignment]


def test_preflight_endpoint() -> None:
    """端点只读、不启动训练：直接调函数，不需要起服务。"""
    import app as backend_app  # noqa: PLC0415

    original = vram.gpu_info
    vram.gpu_info = lambda: {  # type: ignore[assignment]
        "name": "NVIDIA T1000 8GB",
        "total_mb": 8192.0,
        "free_mb": 7000.0,
        "source": "torch",
    }
    try:
        with _TempStore() as store:
            vram.record_sample(640, 8, 3000.0, "fp32")
            data = backend_app.train_preflight(imgsz=640, batch="8", amp="auto")
            check("端点返回可用", data["ok"] is True, str(data.get("vram")))
            check("端点报告 GPU 名称", data["vram"]["gpu"] == "NVIDIA T1000 8GB", str(data.get("vram")))
            check("端点按实测估算", data["vram"]["source"] == "measured", str(data.get("vram")))
            check("端点给出 AMP 结论", data["amp"]["available"] is False, str(data.get("amp")))
            check("端点把 T1000 判成 fp32", data["precision"] == "fp32", str(data.get("precision")))
            check("端点报告样本数", data["sample_count"] == 1, str(data.get("sample_count")))
            check("端点报告表位置", data["store"] == str(store), str(data.get("store")))

            # 显式关闭混合精度：精度口径固定 fp32，与显卡无关
            data = backend_app.train_preflight(imgsz=640, batch="8", amp="off")
            check("amp=off 时精度为 fp32", data["precision"] == "fp32", str(data.get("precision")))

            # batch=auto：跳过估算但不能报错，前端要能正常显示
            data = backend_app.train_preflight(imgsz=640, batch="auto", amp="auto")
            check("batch=auto 不报错", data["vram"].get("skipped") is not None, str(data.get("vram")))
            check("batch=auto 仍然放行", data["ok"] is True, str(data.get("vram")))
    finally:
        vram.gpu_info = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# autoexp 接线
# ---------------------------------------------------------------------------
def test_autoexp_wiring() -> None:
    autoexp_dir = BACKEND.parent.parent / "7-autoresearch-yolo"
    if not (autoexp_dir / "autoexp.py").is_file():
        check("找到 autoexp.py", False, str(autoexp_dir))
        return
    if str(autoexp_dir) not in sys.path:
        sys.path.insert(0, str(autoexp_dir))
    import autoexp as A  # noqa: PLC0415

    check("台账含 amp 列", "amp" in A.RESULTS_HEADER, str(A.RESULTS_HEADER))
    check("台账含 reserved 峰值列", "peak_vram_reserved_mb" in A.RESULTS_HEADER, str(A.RESULTS_HEADER))
    check("amp 列紧跟 batch_eff", A.RESULTS_HEADER.index("amp") == A.RESULTS_HEADER.index("batch_eff") + 1)

    # 表头升级：旧行按列名迁移，新列补空（不能整体错位一格）
    with tempfile.TemporaryDirectory(prefix="trtforge_tsv_") as tmp:
        path = Path(tmp) / "results.tsv"
        old_header = [name for name in A.RESULTS_HEADER if name not in {"amp", "peak_vram_reserved_mb"}]
        old_row = {name: f"v_{name}" for name in old_header}
        A.append_tsv(path, old_header, old_row)
        A.ensure_tsv(path, A.RESULTS_HEADER)
        rows = A.read_tsv(path)
        check("旧行数量不变", len(rows) == 1, str(rows))
        check("旧列值按名迁移", rows[0]["exp_id"] == "v_exp_id", str(rows[0]))
        check("新列补空", rows[0]["amp"] == "", str(rows[0]))
        check("升级前备份了旧文件", path.with_suffix(".tsv.bak").is_file(), "没有 .bak")

    # preflight_vram 必须委托给共享实现（而不是又写一条公式）
    original_free = A.free_vram_mb
    original_load = A.load_vram
    try:
        class _Fake:
            called: list[tuple] = []

            @staticmethod
            def preflight(imgsz, batch, precision=None, free_mb=None):
                _Fake.called.append((imgsz, batch, precision, free_mb))
                return {"ok": False, "reason": "来自共享实现"}

        A.free_vram_mb = lambda: 4000.0  # type: ignore[assignment]
        A.load_vram = lambda: _Fake  # type: ignore[assignment]
        ok, reason = A.preflight_vram(960, 4)
        check("委托给共享实现", reason == "来自共享实现", reason)
        check("把可用显存传下去", _Fake.called and _Fake.called[0][3] == 4000.0, str(_Fake.called))
        check("拦截结果透传", ok is False, str(ok))

        # 显式关闭混合精度时，精度提示必须是 fp32（表键才不会错）
        A.preflight_vram(960, 4, precision="fp32")
        check("精度提示透传", _Fake.called[-1][2] == "fp32", str(_Fake.called[-1]))

        # 缺 vram.py 时降级为公式：960/4 -> 2600 + 300*4*2.25 = 5300 > 4000
        A.load_vram = lambda: None  # type: ignore[assignment]
        ok, reason = A.preflight_vram(960, 4)
        check("缺模块时降级为公式", ok is False and "5300" in reason, reason)
    finally:
        A.free_vram_mb = original_free  # type: ignore[assignment]
        A.load_vram = original_load  # type: ignore[assignment]

    # trainer.amp 读取：它是「实际生效值」，与 args.amp 不是一回事
    class _Trainer:
        def __init__(self, value):
            self.amp = value

    check("trainer.amp=0 读成 False", A.read_trainer_amp(_Trainer(_Scalar(0))) is False)
    check("trainer.amp=1 读成 True", A.read_trainer_amp(_Trainer(_Scalar(1))) is True)
    check("没有 amp 属性时返回 None", A.read_trainer_amp(object()) is None)


def test_runner_absorbs_hardware_events() -> None:
    """Web 训练路径：launcher 上报的 amp_state / vram_peak 必须被吸收。

    这条链路是「自校准」在 Web 侧的唯一入口：事件只在读线程里出现一次，
    丢了就再也没有了（日志文件里是文本，再解析一遍既慢又脆）。
    """
    import runners  # noqa: PLC0415

    with _TempStore():
        run = runners.Run(
            run_id="test-run",
            kind="yolo-train",
            command=["python"],
            cwd=".",
            log_file=Path("test.log"),
        )
        runners.RunManager._absorb_hardware(
            run,
            {
                "type": "event",
                "name": "amp_state",
                "amp_requested": True,
                "amp_effective": False,
                "precision": "fp32",
                "gpu": "NVIDIA T1000 8GB",
            },
        )
        check("实际精度写进 run.meta", run.meta.get("amp_effective") is False, str(run.meta))
        check("精度字符串写进 run.meta", run.meta.get("precision") == "fp32", str(run.meta))
        check("显卡名写进 run.meta", run.meta.get("gpu") == "NVIDIA T1000 8GB", str(run.meta))

        runners.RunManager._absorb_hardware(
            run,
            {
                "type": "event",
                "name": "vram_peak",
                "peak_mb": 2600.0,
                "peak_allocated_mb": 1800.0,
                "sample": {
                    "imgsz": 640,
                    "batch": 8,
                    "peak_mb": 2600.0,
                    "peak_allocated_mb": 1800.0,
                    "precision": "fp32",
                },
            },
        )
        check("峰值写进 run.meta", run.meta.get("peak_vram_mb") == 2600.0, str(run.meta))
        samples = vram.load_samples()
        check("样本写进实测表", "640x8xfp32" in samples, str(samples))
        check("样本峰值正确", samples["640x8xfp32"]["peak_mb"] == 2600.0, str(samples))

        # 没有 sample 的事件（例如 CPU 训练）：不能崩，也不能写脏数据
        runners.RunManager._absorb_hardware(run, {"type": "event", "name": "vram_peak", "peak_mb": None})
        check("空样本不写表", len(vram.load_samples()) == 1, str(vram.load_samples()))
        # 普通日志行：必须原样忽略
        runners.RunManager._absorb_hardware(run, {"type": "log", "line": "epoch 1"})
        check("日志行不被误处理", len(vram.load_samples()) == 1, str(vram.load_samples()))


def main() -> int:
    print("== 显存预检与 AMP 可见性验收 ==")
    test_keys_and_precision()
    test_store_roundtrip()
    test_store_guards()
    test_estimate()
    test_preflight()
    test_preflight_without_gpu()
    test_amp_status()
    test_train_command_amp()
    test_vram_guard()
    test_preflight_endpoint()
    test_runner_absorbs_hardware_events()
    test_autoexp_wiring()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
