# -*- coding: utf-8 -*-
"""全流程串联的状态机验收，不需要 GPU / trtexec / DLL。

守的是「编排错了但每段日志都正常」这一类问题：
- 某段失败后仍然继续跑下一段：拿旧权重编引擎，看起来一切成功；
- 停止只停了当前段，后面几段接着跑；
- 报告文件没被收进 metrics，页面永远显示"验证中"；
- 状态推进时读到写了一半的 JSON（持久化不是原子替换）。

用假 run 管理器把整条链跑完，所以能在 CI 里跑，也跑得够快。

用法：
    python 6-web/backend/tools/test_pipeline_stages.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import pipeline  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


class FakeRun:
    def __init__(self, run_id: str, command: list[str], expect_file: str | None):
        self.run_id = run_id
        self.command = command
        self.expect_file = expect_file
        self.status = "running"
        self.outcome = "finished"
        self.polls_left = 1

    def snapshot(self) -> dict:
        return {"run_id": self.run_id, "status": self.status, "command": self.command}


class FakeRunManager:
    """按预设结果推进的假 run 管理器。

    ``polls_before_finish`` 让状态先保持 running 一次，保证编排器真的走了轮询分支，
    而不是"创建即成功"这种没测到等待逻辑的假通过。
    """

    def __init__(self, outcomes: list[str] | None = None, polls_before_finish: int | None = 1) -> None:
        self.runs: dict[str, FakeRun] = {}
        self.created: list[tuple[str, list[str], str | None]] = []
        self.expect_file_seen: list[object] = []
        self.outcomes = list(outcomes or [])
        # None 表示"一直跑，直到有人 stop"，用来测停止分支
        self.polls_before_finish = polls_before_finish
        self.stopped: list[str] = []

    def create(self, kind, command, cwd="", expect_file=None, meta=None, **kwargs):
        run_id = f"run-{len(self.runs) + 1}"
        run = FakeRun(run_id, list(command), expect_file)
        run.outcome = self.outcomes.pop(0) if self.outcomes else "finished"
        run.polls_left = self.polls_before_finish
        self.runs[run_id] = run
        self.created.append((kind, list(command), expect_file))
        self.expect_file_seen.append(expect_file)
        return run

    def get(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            return None
        if run.status == "running" and run.polls_left is not None:
            run.polls_left -= 1
            if run.polls_left <= 0:
                run.status = run.outcome
        return run

    def stop(self, run_id):
        run = self.runs.get(run_id)
        if run and run.status == "running":
            run.status = "cancelled"
            self.stopped.append(run_id)
            return True
        return False


def wait_done(manager: pipeline.PipelineManager, pipeline_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(pipeline_id)
        if record and record["status"] != pipeline.CHAIN_RUNNING and record["status"] != pipeline.PENDING:
            return record
        time.sleep(0.01)
    return manager.get(pipeline_id)


def make_temp(tag: str) -> Path:
    config.ensure_dirs()
    return Path(tempfile.mkdtemp(prefix=f"test-{tag}-", dir=str(config.RUNTIME_DIR)))


def main() -> int:
    root = make_temp("pipeline")
    try:
        # ---- 1. 三段全部成功 -------------------------------------------------
        fake = FakeRunManager()
        store = root / "store1"
        manager = pipeline.PipelineManager(fake, store, wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan("train", "训练", command=["py", "train.py"], expect_file="best.pt"),
            pipeline.StagePlan("export", "导出", command=["py", "export.py"], expect_file="best.transd.onnx"),
            pipeline.StagePlan("build", "编译", command=["trtexec", "--onnx=a"], expect_file="a.engine"),
        ]
        record = manager.create("yolo", "测试链", plans, artifacts={"run_dir": "r"})
        done = wait_done(manager, record["id"])
        check("三段全部成功", done["status"] == pipeline.CHAIN_SUCCESS, str(done["status"]))
        check("阶段状态全是 success", all(s["status"] == pipeline.SUCCESS for s in done["stages"]))
        check("按顺序创建了三个 run", [c[0] for c in fake.created] == [
            "pipeline-train", "pipeline-export", "pipeline-build"
        ])
        check("命令原样传给 run 管理器", fake.created[0][1] == ["py", "train.py"])
        # runners 对 expect_file 调 .is_file()：传字符串会让读取线程带异常死掉，
        # run 永远停在 running，整条链静默卡住。这里守住类型。
        check("expect_file 交给 run 管理器时已是 Path",
              isinstance(fake.expect_file_seen[0], Path), str(fake.expect_file_seen[0]))
        check("产物记录含三段", set(done["artifacts"]) >= {"train", "export", "build", "run_dir"})
        check("产物路径取 expect_file", done["artifacts"]["build"] == "a.engine")
        check("有耗时字段", done["elapsed"] is not None and done["elapsed"] >= 0)

        # ---- 1b. 产物去重 ---------------------------------------------------
        # app.py 创建时就把 engine/transd_onnx 等路径放进 artifacts，
        # 阶段结束时如果再按阶段名补一份，表格里会出现两行同样的路径。
        fake = FakeRunManager()
        manager = pipeline.PipelineManager(fake, root / "store1b", wait_interval=0, sleep=lambda _: None)
        record = manager.create(
            "yolo", "去重链",
            [pipeline.StagePlan("build", "编译", command=["trtexec"], expect_file="a.engine")],
            artifacts={"engine": "a.engine"},
        )
        done = wait_done(manager, record["id"])
        check("同一路径不再按阶段名重复记录", "build" not in done["artifacts"], str(done["artifacts"]))
        check("原有产物键保留", done["artifacts"].get("engine") == "a.engine")

        # ---- 2. 某段失败后不再继续 -------------------------------------------
        fake = FakeRunManager(outcomes=["finished", "failed"])
        manager = pipeline.PipelineManager(fake, root / "store2", wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan("train", "训练", command=["py", "train.py"]),
            pipeline.StagePlan("export", "导出", command=["py", "export.py"]),
            pipeline.StagePlan("build", "编译", command=["trtexec"]),
        ]
        record = manager.create("yolo", "失败链", plans)
        done = wait_done(manager, record["id"])
        check("第二段失败 → 整链 failed", done["status"] == pipeline.CHAIN_FAILED, str(done["status"]))
        check("第一段成功", done["stages"][0]["status"] == pipeline.SUCCESS)
        check("第二段 failed", done["stages"][1]["status"] == pipeline.FAILED)
        check("第三段被跳过", done["stages"][2]["status"] == pipeline.SKIPPED)
        check("只有两个 run 被创建", len(fake.created) == 2, str(len(fake.created)))
        check("失败原因写进整链", bool(done["error"]), str(done["error"]))

        # ---- 3. 可执行阶段（callable） ---------------------------------------
        calls: list[str] = []

        def make_callable(value: str):
            def action():
                calls.append(value)
                return {"artifact": f"/tmp/{value}.json"}
            return action

        def explode():
            raise ValueError("标签生成失败：读不到 names")

        fake = FakeRunManager()
        manager = pipeline.PipelineManager(fake, root / "store3", wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan("labels", "补标签", callable=make_callable("labels")),
            pipeline.StagePlan("boom", "炸", callable=explode),
        ]
        record = manager.create("yolo", "callable 链", plans, artifacts={})
        done = wait_done(manager, record["id"])
        check("callable 被调用", calls == ["labels"])
        check("callable 的产物被记录", done["artifacts"].get("labels") == "/tmp/labels.json")
        check("callable 抛异常 → failed", done["status"] == pipeline.CHAIN_FAILED)
        check("异常信息带类型", "ValueError" in (done["error"] or ""), str(done["error"]))
        check("callable 阶段不创建 run", fake.created == [])

        # callable 返回的产物与已有路径相同时不能重复记一行：
        # probe 阶段返回的就是 transd_onnx，而它早就在初始 artifacts 里了。
        fake = FakeRunManager()
        manager = pipeline.PipelineManager(fake, root / "store3b", wait_interval=0, sleep=lambda _: None)
        record = manager.create(
            "yolo", "callable 去重链",
            [pipeline.StagePlan("probe", "校验", callable=make_callable("transd"))],
            artifacts={"transd_onnx": "/tmp/transd.json"},
        )
        done = wait_done(manager, record["id"])
        check("callable 产物与已有路径重复时不新增键",
              "probe" not in done["artifacts"], str(done["artifacts"]))

        # ---- 4. 停止：当前段取消，后续不跑 -----------------------------------
        fake = FakeRunManager(polls_before_finish=None)
        manager = pipeline.PipelineManager(fake, root / "store4", wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan("train", "训练", command=["py", "train.py"]),
            pipeline.StagePlan("export", "导出", command=["py", "export.py"]),
        ]
        record = manager.create("yolo", "停止链", plans)
        pid = record["id"]
        # 等到第一段真的建了 run 再停，否则测到的是"还没开跑就被停"这条分支
        deadline = time.time() + 2.0
        while time.time() < deadline:
            current = manager.get(pid)
            if current and current.get("run_id"):
                break
            time.sleep(0.01)
        check("进行中的链可以停止", manager.stop(pid) is True)
        done = wait_done(manager, pid)
        check("停止后整链 cancelled", done["status"] == pipeline.CHAIN_CANCELLED, str(done["status"]))
        check("当前段标记 cancelled", done["stages"][0]["status"] == pipeline.CANCELLED)
        check("后续段标记 skipped", done["stages"][1]["status"] == pipeline.SKIPPED)
        check("真的调用了 run_manager.stop", fake.stopped == ["run-1"], str(fake.stopped))
        check("只有第一段被创建", len(fake.created) == 1)
        check("已结束的链不能再停止", manager.stop(pid) is False)

        # ---- 5. 指标报告收集 -------------------------------------------------
        report_dir = root / "val"
        report_dir.mkdir(parents=True)
        report_path = report_dir / "val_report.json"
        report_path.write_text(
            json.dumps({"map50": 0.8123, "map5095": 0.541, "per_class": []}),
            encoding="utf-8",
        )
        fake = FakeRunManager()
        manager = pipeline.PipelineManager(fake, root / "store5", wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan(
                "val", "验证", command=["py", "val.py"],
                report_file=str(report_path), expect_file=str(report_path),
            )
        ]
        record = manager.create("yolo", "报告链", plans)
        done = wait_done(manager, record["id"])
        check("报告被读进 metrics", (done.get("metrics") or {}).get("map50") == 0.8123,
              str(done.get("metrics")))

        # ---- 6. 报告缺失时不崩 -----------------------------------------------
        fake = FakeRunManager()
        manager = pipeline.PipelineManager(fake, root / "store6", wait_interval=0, sleep=lambda _: None)
        plans = [
            pipeline.StagePlan("val", "验证", command=["py", "val.py"],
                               report_file=str(root / "missing.json")),
        ]
        record = manager.create("yolo", "缺报告", plans)
        done = wait_done(manager, record["id"])
        check("报告缺失不影响整链成功", done["status"] == pipeline.CHAIN_SUCCESS)
        check("metrics 保持 None", done.get("metrics") is None)

        # ---- 7. 快照隔离 -----------------------------------------------------
        record = manager.get(record["id"])
        record["stages"][0]["status"] = "被外部改了"
        again = manager.get(record["id"])
        check("修改快照不影响内部状态", again["stages"][0]["status"] != "被外部改了")

        # ---- 8. 列表与持久化 -------------------------------------------------
        items = manager.list()
        check("list 返回记录", len(items) >= 1, str(len(items)))
        check("list 按创建时间倒序",
              all(items[i]["created_at"] >= items[i + 1]["created_at"] for i in range(len(items) - 1)))
        check("limit 生效", len(manager.list(limit=1)) == 1)

        pid = items[0]["id"]
        persisted = root / "store6" / f"{pid}.json"
        check("状态写入磁盘", persisted.is_file(), str(persisted))
        restored = pipeline.read_persisted(root / "store6", pid)
        check("可以从磁盘读回", (restored or {}).get("id") == pid)
        broken = root / "store6" / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        check("损坏文件返回 None", pipeline.read_persisted(root / "store6", "broken") is None)
        check("不存在的记录返回 None", pipeline.read_persisted(root / "store6", "nope") is None)

        # ---- 9. 重启后：磁盘上的遗留记录 -------------------------------------
        # 断电/强杀进程时来不及走 shutdown_all()，磁盘上会留下 status=running 的记录。
        # 不处理的话：历史列表看不到它（内存里没有），按 id 查却能查到 running，
        # 前端就会永远转圈，点「停止」只拿到 409。
        store = root / "store9"
        store.mkdir(parents=True)
        stale = {
            "id": "stale-1", "kind": "yolo", "title": "老记录", "status": "running",
            "created_at": 1.0, "started_at": 1.0, "finished_at": None, "elapsed": None,
            "stage_index": 1, "stage": "build", "run_id": "run-x",
            "stages": [
                {"name": "train", "label": "训练", "status": "success", "run_id": "r1"},
                {"name": "build", "label": "编译", "status": "running", "run_id": "run-x"},
                {"name": "val", "label": "验证", "status": "pending", "run_id": None},
            ],
            "artifacts": {"pt": "/x/best.pt"}, "metrics": None, "error": None,
            "meta": {}, "stop_requested": False,
        }
        (store / "stale-1.json").write_text(
            json.dumps(stale, ensure_ascii=False), encoding="utf-8"
        )
        restarted = pipeline.PipelineManager(
            FakeRunManager(), store, wait_interval=0, sleep=lambda _: None
        )
        check("遗留记录不在内存里", restarted.get("stale-1") is None)
        listed = restarted.list()
        check("list 从磁盘补历史记录",
              any(item["id"] == "stale-1" for item in listed), str([i["id"] for i in listed]))
        stale_after = next(item for item in listed if item["id"] == "stale-1")
        check("遗留 running 被标成 cancelled", stale_after["status"] == "cancelled",
              str(stale_after["status"]))
        check("中断原因写清楚", "中断" in (stale_after["error"] or ""), str(stale_after["error"]))
        check("中断的段标成 cancelled", stale_after["stages"][1]["status"] == "cancelled")
        check("没跑的段标成 skipped", stale_after["stages"][2]["status"] == "skipped")
        check("已完成的段不动", stale_after["stages"][0]["status"] == "success")
        check("历史记录标 live=false", stale_after["live"] is False)

        # ---- 10. 参数校验 ----------------------------------------------------
        manager = pipeline.PipelineManager(FakeRunManager(), None, wait_interval=0, sleep=lambda _: None)
        record = manager.create("yolo", "空链", [pipeline.StagePlan("x", "空")])
        done = wait_done(manager, record["id"])
        check("既无命令也无动作的阶段报错", done["status"] == pipeline.CHAIN_FAILED)
        check("错误信息点明阶段", "x" in (done["error"] or ""), str(done["error"]))

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  !! {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
