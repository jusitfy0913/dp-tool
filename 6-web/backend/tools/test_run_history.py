# -*- coding: utf-8 -*-
"""任务台账、日志滚动、自动续训的验收脚本（不需要 GPU / torch / DLL）。

守的是四类「只有出事那天才会发现」的问题：

1. **服务重启后历史全空**：`/api/runs` 只返回本进程的任务，排障要看的是
   「上一次那条训练到底怎么结束的」，所以必须有落盘台账。
2. **僵尸 running**：进程被强杀后台账里留着 running，如果不改判成
   interrupted，用户会一直以为它还在跑。
3. **日志把磁盘写满**：单次任务日志必须有上限，而且滚动后还能读到前面的内容。
4. **自动续训变成无限重跑**：没有「只续一次」的闸门，一次环境问题会让训练
   无人值守地重跑一整夜。

用法：
    python 6-web/backend/tools/test_run_history.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import logstore  # noqa: E402
import runledger  # noqa: E402
import runners  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))


class _TempPaths:
    """把台账与日志目录临时指到系统 temp，跑完恢复。

    不这么做的后果：测试会往真实 `runtime/runs.jsonl` 里塞假任务，
    开发机上的历史列表立刻变脏，而且下次真机验证时根本分不清哪条是真的。
    """

    def __enter__(self) -> "_TempPaths":
        self._dir = tempfile.TemporaryDirectory(prefix="trtforge_ledger_")
        root = Path(self._dir.name)
        self.ledger = root / "runs.jsonl"
        self.logs = root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self._old_ledger = config.RUNS_LEDGER
        self._old_logs = config.LOG_DIR
        config.RUNS_LEDGER = self.ledger
        config.LOG_DIR = self.logs
        return self

    def __exit__(self, *_exc) -> None:
        config.RUNS_LEDGER = self._old_ledger
        config.LOG_DIR = self._old_logs
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# 台账
# ---------------------------------------------------------------------------
def test_ledger_fold() -> None:
    """后写的记录覆盖先写的：一个任务最后算一个状态。"""
    with _TempPaths():
        runledger.record_started("aaaa00000001", "yolo-train", ["python", "x.py"], "a.log", started_at=100.0)
        runledger.record_finished(
            "aaaa00000001", "yolo-train", "finished", 0, started_at=100.0, finished_at=160.0,
            command=["python", "x.py"], log_file="a.log",
        )
        latest = runledger.latest_states()
        check("台账按 run_id 折叠", len(latest) == 1, str(len(latest)))
        record = latest["aaaa00000001"]
        check("折叠取最后一条状态", record["status"] == "finished", str(record["status"]))
        check("结束记录带退出码", record["exit_code"] == 0, str(record["exit_code"]))

        items = runledger.history(10)
        check("历史快照标记为只读", items and items[0].get("history") is True, str(items[:1]))
        check("历史快照算出时长", items[0]["elapsed"] == 60.0, str(items[0]["elapsed"]))


def test_ledger_broken_line() -> None:
    """进程被强杀时最后一行可能半截，读取必须跳过它而不是整份报废。"""
    with _TempPaths() as paths:
        runledger.record_started("bbbb00000001", "engine-build", ["trtexec"], "b.log", started_at=200.0)
        with paths.ledger.open("a", encoding="utf-8") as handle:
            handle.write('{"run_id": "cccc00000001", "event": "star')  # 故意半截

        latest = runledger.latest_states()
        check("坏行被跳过", "bbbb00000001" in latest, str(list(latest)))
        check("坏行不产生假任务", "cccc00000001" not in latest, str(list(latest)))


def test_ledger_write_failure_is_quiet() -> None:
    """台账写不进去也不能抛异常：任务该跑还得跑。"""
    with _TempPaths() as paths:
        # 用一个目录当台账文件 → 必然 OSError
        config.RUNS_LEDGER = paths.logs
        ok = runledger.append({"run_id": "dddd00000001", "status": "running"})
        check("台账不可写时返回 False 而不抛异常", ok is False, str(ok))


def test_recover_interrupted() -> None:
    """重启后把僵尸 running 改判为 interrupted，且再恢复一次不会重复改。"""
    with _TempPaths():
        runledger.record_started("eeee00000001", "yolo-train", ["python"], "e.log", started_at=300.0)
        runledger.record_started("eeee00000002", "engine-build", ["trtexec"], "f.log", started_at=310.0)
        runledger.record_finished(
            "eeee00000002", "engine-build", "finished", 0, started_at=310.0, finished_at=330.0
        )

        recovered = runledger.recover_interrupted(now=400.0)
        check("只恢复仍为 running 的任务", [r["run_id"] for r in recovered] == ["eeee00000001"], str(recovered))

        record = runledger.latest_states()["eeee00000001"]
        check("改判为 interrupted", record["status"] == "interrupted", str(record["status"]))
        check("标记为孤儿任务", record.get("orphan") is True, str(record.get("orphan")))
        check("补上结束时间", record["finished_at"] == 400.0, str(record["finished_at"]))

        again = runledger.recover_interrupted(now=500.0)
        check("重复恢复不再改动", again == [], str(again))


def test_history_limit() -> None:
    """历史按开始时间倒序，limit 生效。"""
    with _TempPaths():
        for index in range(5):
            runledger.record_finished(
                f"ffff0000000{index}", "yolo-train", "finished", 0,
                started_at=100.0 + index * 10, finished_at=105.0 + index * 10,
            )
        items = runledger.history(3)
        check("limit 生效", len(items) == 3, str(len(items)))
        check(
            "新的在前",
            [item["started_at"] for item in items] == [140.0, 130.0, 120.0],
            str([item["started_at"] for item in items]),
        )


# ---------------------------------------------------------------------------
# 日志滚动
# ---------------------------------------------------------------------------
def test_log_rotation() -> None:
    """写满就滚动，且滚动后还能跨分片读到老内容。"""
    with _TempPaths() as paths:
        run_id = "999900000001"
        max_bytes = 200
        sink = logstore.RunLogSink(paths.logs / f"{run_id}.log", max_bytes=max_bytes, backups=2)
        for index in range(60):
            sink.write(f"line-{index:04d}")
        sink.close()

        shards = logstore.shard_paths(run_id)
        check("产生了滚动分片", len(shards) >= 2, str([p.name for p in shards]))
        check("分片数不超过 1+backups", len(shards) <= 1 + 2, str([p.name for p in shards]))

        total = sum(path.stat().st_size for path in shards)
        # 每行 10 字节；上限 200 字节 × 3 份 = 600，允许最后一份再超一点点
        check("总量受上限约束", total <= max_bytes * 3 + 64, f"{total}")

        tail = logstore.read_tail(run_id, 3)
        check("尾部跨分片取到最新行", tail.splitlines()[-1] == "line-0059", repr(tail[-40:]))

        full = logstore.read_all(run_id)
        check("全文按旧→新拼接", full.splitlines()[-1].endswith("line-0059"), repr(full[-40:]))

        lines = full.splitlines()
        check("全文本身是递增的", lines == sorted(lines), "顺序错乱")


def test_log_endpoint_reads_shards() -> None:
    """/api/runs/{id}/log 必须能读到滚动出去的内容。"""
    import app as backend_app
    from fastapi import HTTPException

    with _TempPaths() as paths:
        run_id = "888800000001"
        sink = logstore.RunLogSink(paths.logs / f"{run_id}.log", max_bytes=120, backups=2)
        for index in range(40):
            sink.write(f"row-{index:03d}")
        sink.close()

        try:
            text = backend_app.get_run_log(run_id, lines=5).body.decode("utf-8")
            check("滚动后仍能读到最新行", text.splitlines()[-1] == "row-039", repr(text[-40:]))
            check("lines 仍然被夹紧", len(text.splitlines()) <= 5, str(len(text.splitlines())))
        except HTTPException as exc:  # noqa: PERF203
            check("滚动后仍能读到最新行", False, f"HTTP {exc.status_code}: {exc.detail}")

        # 只剩分片（当前那份被手工删掉）：仍然要能读出内容，而不是 404
        current = paths.logs / f"{run_id}.log"
        if current.is_file():
            current.unlink()
        try:
            shard_text = backend_app.get_run_log(run_id, lines=5).body.decode("utf-8")
            check(
                "只剩分片时仍能读",
                shard_text.strip() != "" and shard_text.splitlines()[-1].startswith("row-"),
                repr(shard_text[-40:]),
            )
        except HTTPException as exc:  # noqa: PERF203
            check("只剩分片时仍能读", False, f"HTTP {exc.status_code}: {exc.detail}")

        try:
            backend_app.get_run_log("777700000001", lines=5)
        except HTTPException as exc:
            check("没有日志时返回 404", exc.status_code == 404, str(exc.status_code))
        else:
            check("没有日志时返回 404", False, "没有抛异常")


# ---------------------------------------------------------------------------
# 历史接口
# ---------------------------------------------------------------------------
def test_history_endpoint() -> None:
    """HTTP 层的夹紧与排序。"""
    import app as backend_app

    with _TempPaths():
        runledger.record_finished(
            "123400000001", "yolo-train", "finished", 0, started_at=10.0, finished_at=20.0
        )
        payload = backend_app.list_run_history(limit=50)
        check("历史接口返回 runs", len(payload.get("runs") or []) == 1, str(payload))

        clamped = backend_app.list_run_history(limit=99999)
        check("limit 超上限被夹到 500", clamped.get("limit") == 500, str(clamped.get("limit")))


# ---------------------------------------------------------------------------
# 自动续训
# ---------------------------------------------------------------------------
def test_resume_guards() -> None:
    """四个闸门：关配置 / 没权重 / 已续过 / 非 YOLO，任一命中都不该重跑。"""
    import app as backend_app

    with _TempPaths() as paths:
        calls: list[dict] = []

        def fake_create(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            return runners.Run(
                run_id="resume000001", kind=kwargs["kind"], command=kwargs["command"], cwd="."
            )

        manager = runners.RunManager()
        manager.create = fake_create  # type: ignore[method-assign]

        run_dir = paths.logs / "exp42"
        weights = run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)

        def make_run(**overrides):
            base = dict(
                run_id="orig00000001",
                kind="yolo-train",
                command=["python", "train.py"],
                cwd=".",
                status="failed",
                resume_dir=run_dir,
                resume_command=lambda: ["python", "train.py", "--resume-from", "x"],
            )
            base.update(overrides)
            return runners.Run(**base)

        old_auto = config.AUTO_RESUME
        try:
            config.AUTO_RESUME = False
            (weights / "last.pt").write_text("x", encoding="utf-8")
            manager._maybe_resume(make_run())
            check("关闭开关后不续训", calls == [], str(calls))

            config.AUTO_RESUME = True
            (weights / "last.pt").unlink()
            manager._maybe_resume(make_run())
            check("没有 last.pt 不续训", calls == [], str(calls))

            (weights / "last.pt").write_text("x", encoding="utf-8")
            manager._maybe_resume(make_run(resume_command=None))
            check("没有重建命令不续训", calls == [], str(calls))

            manager._maybe_resume(make_run(resumed_from="older0000001"))
            check("续训衍生的任务不再续训", calls == [], str(calls))

            manager._maybe_resume(make_run(kind="anomalib-train"))
            check("Anomalib 不走自动续训", calls == [], str(calls))

            manager._maybe_resume(make_run(status="finished"))
            check("正常结束不续训", calls == [], str(calls))

            manager._maybe_resume(make_run())
            check("满足条件时续训一次", len(calls) == 1, str(calls))
            check("续训记录来源 run", calls and calls[0].get("resumed_from") == "orig00000001", str(calls[:1]))
        finally:
            config.AUTO_RESUME = old_auto


def test_resume_request_switches_mode() -> None:
    """自动续训用的是 --resume-from，而不是重新开一次训练。"""
    import app as backend_app

    with tempfile.TemporaryDirectory(prefix="trtforge_resume_") as temp_dir:
        weights = Path(temp_dir) / "exp1" / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        last_pt = weights / "last.pt"
        last_pt.write_text("fake", encoding="utf-8")

        request = backend_app.YoloTrainRequest(
            data=str(Path(__file__).resolve()),
            model="yolov8n.pt",
            project=str(Path(temp_dir) / "exp1"),
            name="exp1",
            init_mode="finetune",
            resume=True,
        )
        patched = backend_app._resume_request(request, last_pt)
        check("续训请求指向 last.pt", patched.resume_from.endswith("last.pt"), patched.resume_from)
        check("续训请求不再走微调", patched.init_mode == "preset", patched.init_mode)
        check("续训请求带 exist_ok", patched.exist_ok is True, str(patched.exist_ok))
        check("续训请求关掉布尔 resume", patched.resume is False, str(patched.resume))
        check("原始请求没被就地修改", request.init_mode == "finetune", request.init_mode)

        command = backend_app.build_train_command(patched, "python")
        check("命令里带 --resume-from", "--resume-from" in command, " ".join(command))
        check("命令里带 --exist-ok", "--exist-ok" in command, " ".join(command))


def test_prune_logs() -> None:
    """过期日志族整族删除；同目录里不认识的日志一律不碰。"""
    import os

    with _TempPaths() as paths:
        old_run = "111100000001"
        fresh_run = "222200000001"
        for run_id in (old_run, fresh_run):
            (paths.logs / f"{run_id}.log").write_text("x\n", encoding="utf-8")
            (paths.logs / f"{run_id}.log.1").write_text("y\n", encoding="utf-8")
        # 推理进程自己的日志：名字不是 12 位十六进制，必须留着
        other = paths.logs / "infer_worker.log"
        other.write_text("keep\n", encoding="utf-8")

        stamp = 1_700_000_000.0
        for name in (f"{old_run}.log", f"{old_run}.log.1"):
            os.utime(paths.logs / name, (stamp, stamp))
        for name in (f"{fresh_run}.log", f"{fresh_run}.log.1"):
            os.utime(paths.logs / name, (stamp, stamp))

        # 30 天保留期：老任务 100 天前，新任务 1 天前
        for name in (f"{fresh_run}.log", f"{fresh_run}.log.1"):
            os.utime(paths.logs / name, (stamp + 99 * 86400, stamp + 99 * 86400))

        result = logstore.prune_logs(max_age_days=30, log_dir=paths.logs, now=stamp + 100 * 86400)
        check("整族删除过期任务", result["families"] == 1, str(result))
        check("连分片一起删", result["files"] == 2, str(result))
        check("旧任务日志已消失", not (paths.logs / f"{old_run}.log").exists(), "文件还在")
        check("新任务日志保留", (paths.logs / f"{fresh_run}.log").is_file(), "被误删")
        check("不认识的日志不碰", other.is_file(), "被误删")

        disabled = logstore.prune_logs(max_age_days=0, log_dir=paths.logs, now=stamp + 999 * 86400)
        check("保留期 0 表示不清理", disabled["files"] == 0, str(disabled))


def main() -> int:
    test_ledger_fold()
    test_ledger_broken_line()
    test_ledger_write_failure_is_quiet()
    test_recover_interrupted()
    test_history_limit()
    test_log_rotation()
    test_log_endpoint_reads_shards()
    test_prune_logs()
    test_history_endpoint()
    test_resume_guards()
    test_resume_request_switches_mode()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
