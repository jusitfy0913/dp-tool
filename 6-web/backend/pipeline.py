# -*- coding: utf-8 -*-
"""全流程串联：把「训练 → 导出 → 编译 → 补标签 → 引擎验证」串成一条可停可看的链。

## 为什么不做一个大脚本去 spawn 一堆子进程

Windows 上杀父进程不会连带杀子进程，`proc.terminate()` 只终止直接子进程。
如果编排器自己 spawn 训练进程，用户点「停止」后训练还在后台占着 GPU，
需要 Job Object（ctypes）才能收干净。而现成的 `runners.py` 已经把
"单个任务"的停止、日志、事件流、产物校验都做完了。

所以这里的选择是：**每一段都是一个普通的 run**，编排器只负责决定
"下一段跑什么、跑完算不算成功"。好处：

- 停止 = 停当前那一个 run，没有孤儿进程；
- 每段都有独立日志与 `WS /ws/runs/{id}`，前端复用现成的日志面板；
- 某一段失败时，前面已完成的产物照常保留；
- 后端重启时随 `shutdown_all()` 一起停，状态标成 cancelled，不会假装还在跑。

## 状态机

    pending ──start──> running ──ok──> success ──> 下一段
                          │
                          └──fail/stop──> failed / cancelled（整条链就此停止）

失败不自动重试：训练 OOM、数据配错这类问题重试一次只是再烧一遍时间。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 阶段状态
PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
CANCELLED = "cancelled"
SKIPPED = "skipped"

# 整条链状态
CHAIN_RUNNING = "running"
CHAIN_SUCCESS = "success"
CHAIN_FAILED = "failed"
CHAIN_CANCELLED = "cancelled"


@dataclass
class StagePlan:
    """一段要执行的事情。

    三种形态：给 ``command`` 就跑子进程（走 runners）；给 ``callable`` 就在
    编排线程里同步执行（用于"读 yaml 写一个 labels.json"这种毫秒级动作，
    为它起一个进程反而更慢、日志也更碎）；给 ``resolve`` 则在阶段开始前
    用记录里的产物派生命令（路径依赖上一段才知道的版本号时用）。
    """

    name: str
    label: str
    command: list[str] | None = None
    cwd: str = ""
    #: 必须存在的产物。runners 会拿它做「退出码 0 但没产出」的判定，
    #: 那边要求 Path 而不是字符串——直接传 str 会在读取线程里抛
    #: AttributeError，表现得像「进程跑完了但日志突然断掉」。
    expect_file: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    callable: Callable[[], dict[str, Any]] | None = None
    report_file: str | None = None
    kind: str = ""  # 传给 runners 的任务类型名，留空按 pipeline-<name> 推导
    #: 命令要等上一段产物确定之后才能拼出来时用（异常检测的 ONNX 落在
    #: `results/<模型类>/<name>/vN/...`，版本号 `N` 训练完才知道，猜 `v0`
    #: 在第二次训练时就错）。入参是整条记录，返回
    #: `{"command": [...], "expect_file": ..., "report_file": ...}`。
    resolve: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    #: 非空时这一段不执行，直接标成跳过，并把这句话显示在阶段上。
    skip_reason: str = ""


    def run_kind(self, pipeline_id: str) -> str:
        return self.kind or f"pipeline-{self.name}"

    def __post_init__(self) -> None:
        # 调用方（app.py）习惯直接写路径字符串，这里统一转成 Path：
        # runners 拿 str 会在读取线程里 `.is_file()` 抛 AttributeError，
        # 那条线程一死，run 就永远停在 running，整条链跟着卡住。
        if isinstance(self.expect_file, str):
            self.expect_file = Path(self.expect_file) if self.expect_file else None


class PipelineManager:
    """管线注册表 + 监督线程。

    ``run_manager`` 只用到三个方法：``create`` / ``get`` / ``stop``，
    单测里换成一个假实现就能把整条状态机跑一遍，不需要 GPU、不需要真实进程。
    """

    def __init__(
        self,
        run_manager: Any,
        store_dir: Path | str | None = None,
        *,
        wait_interval: float = 0.5,
        wait_timeout: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.run_manager = run_manager
        self.store_dir = Path(store_dir) if store_dir else None
        self.wait_interval = wait_interval
        self.wait_timeout = wait_timeout
        self._sleep = sleep
        self._pipelines: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._mark_interrupted()

    def _mark_interrupted(self) -> None:
        """把上次进程遗留的「还在跑」的记录标成已中断。

        正常退出会走 ``shutdown_all()``，但断电、任务管理器结束进程、崩溃都不会。
        这些记录的 status 会一直是 running，前端按它轮询就永远转圈，
        而且点「停止」只能拿到 409（内存里没有这条）。
        """
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("status") not in (PENDING, CHAIN_RUNNING):
                continue
            record["status"] = CHAIN_CANCELLED
            record["error"] = "服务重启，任务已中断"
            record["finished_at"] = time.time()
            record["stage"] = None
            record["run_id"] = None
            if record.get("started_at"):
                record["elapsed"] = round(record["finished_at"] - record["started_at"], 2)
            for stage_state in record.get("stages") or []:
                if stage_state.get("status") in (PENDING, RUNNING):
                    stage_state["status"] = SKIPPED if stage_state.get("status") == PENDING else CANCELLED
            try:
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                continue

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def create(
        self,
        kind: str,
        title: str,
        stages: list[StagePlan],
        artifacts: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pipeline_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record: dict[str, Any] = {
            "id": pipeline_id,
            "kind": kind,
            "title": title,
            "status": PENDING,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed": None,
            "stage_index": -1,
            "stage": None,
            "run_id": None,
            "stages": [
                {
                    "name": stage.name,
                    "label": stage.label,
                    "status": PENDING,
                    "run_id": None,
                    "started_at": None,
                    "finished_at": None,
                    "elapsed": None,
                    "error": None,
                }
                for stage in stages
            ],
            "artifacts": dict(artifacts or {}),
            "metrics": None,
            "error": None,
            "meta": dict(meta or {}),
            "stop_requested": False,
        }
        with self._lock:
            self._pipelines[pipeline_id] = record
        self._persist(record)

        thread = threading.Thread(
            target=self._run_chain, args=(pipeline_id, stages), daemon=True
        )
        thread.start()
        return self.snapshot(pipeline_id)

    def snapshot(self, pipeline_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._pipelines.get(pipeline_id)
            if record is None:
                return None
            # 深拷一份，避免调用方在序列化时改到内部状态
            return json.loads(json.dumps(record, ensure_ascii=False))

    def get(self, pipeline_id: str) -> dict[str, Any] | None:
        return self.snapshot(pipeline_id)

    def is_live(self, pipeline_id: str) -> bool:
        """这条记录是否还在本进程内存里（false = 只能从磁盘读到历史）。"""
        with self._lock:
            return pipeline_id in self._pipelines

    def list(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._pipelines.values())
            known = set(self._pipelines)
        # 内存里没有的记录从磁盘补：刷新页面/重启服务后，用户仍然要能看到
        # 「上次跑到哪一段」，否则历史列表会突然变空，看起来像数据丢了。
        if self.store_dir:
            for path in self.store_dir.glob("*.json"):
                if path.stem in known:
                    continue
                record = read_persisted(self.store_dir, path.stem)
                if record is not None:
                    record["live"] = False
                    records.append(record)
        records.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
        return [json.loads(json.dumps(item, ensure_ascii=False)) for item in records[:limit]]

    def stop(self, pipeline_id: str) -> bool:
        """停止整条链：标记意图 + 立刻停当前这一段。"""
        with self._lock:
            record = self._pipelines.get(pipeline_id)
            if record is None:
                return False
            if record["status"] not in (PENDING, CHAIN_RUNNING):
                return False
            record["stop_requested"] = True
            run_id = record.get("run_id")
        if run_id:
            self.run_manager.stop(run_id)
        self._persist(record)
        return True

    def shutdown_all(self) -> None:
        """服务退出时调用：把还在跑的链标记成取消，并停掉当前阶段。"""
        with self._lock:
            running = [
                item["id"]
                for item in self._pipelines.values()
                if item["status"] in (PENDING, CHAIN_RUNNING)
            ]
        for pipeline_id in running:
            self.stop(pipeline_id)

    # ------------------------------------------------------------------
    # 监督线程
    # ------------------------------------------------------------------
    def _run_chain(self, pipeline_id: str, stages: list[StagePlan]) -> None:
        record = self._pipelines[pipeline_id]
        record["status"] = CHAIN_RUNNING
        record["started_at"] = time.time()
        self._persist(record)

        try:
            for index, stage in enumerate(stages):
                if record.get("stop_requested"):
                    self._mark_remaining(record, index, SKIPPED)
                    self._finish(record, CHAIN_CANCELLED, "已按要求停止")
                    return

                stage_state = record["stages"][index]
                if stage.skip_reason:
                    stage_state["status"] = SKIPPED
                    stage_state["error"] = stage.skip_reason
                    stage_state["started_at"] = time.time()
                    stage_state["finished_at"] = stage_state["started_at"]
                    stage_state["elapsed"] = 0
                    self._persist(record)
                    continue

                record["stage_index"] = index
                record["stage"] = stage.name
                record["run_id"] = None
                stage_state["status"] = RUNNING
                stage_state["started_at"] = time.time()
                self._persist(record)

                ok, error = self._run_stage(record, stage, stage_state)
                stage_state["finished_at"] = time.time()
                stage_state["elapsed"] = round(
                    stage_state["finished_at"] - (stage_state["started_at"] or time.time()), 2
                )

                if not ok:
                    stage_state["status"] = CANCELLED if record.get("stop_requested") else FAILED
                    stage_state["error"] = error
                    code = CHAIN_CANCELLED if record.get("stop_requested") else CHAIN_FAILED
                    # 先标掉后续段再收尾：否则轮询方可能看到"整链已失败 + 后面还是 pending"
                    # 这种中间态，前端就会显示出一段永远不动的 pending。
                    self._mark_remaining(record, index + 1, SKIPPED)
                    self._finish(record, code, error)
                    return

                stage_state["status"] = SUCCESS
                stage_state["run_id"] = stage_state.get("run_id")
                self._persist(record)

            self._finish(record, CHAIN_SUCCESS, None)
        except Exception as exc:  # noqa: BLE001 - 监督线程不能静默死掉
            self._finish(record, CHAIN_FAILED, f"编排异常：{type(exc).__name__}: {exc}")

    def _run_stage(
        self, record: dict[str, Any], stage: StagePlan, stage_state: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """执行一段，返回 (是否成功, 失败原因)。"""
        if stage.resolve is not None:
            # 命令依赖上一段的产物（路径里有训练时才确定的版本号），
            # 所以在这里才派生；失败就当作本段失败，不去猜路径硬跑。
            try:
                info = stage.resolve(record) or {}
            except Exception as exc:  # noqa: BLE001
                return False, f"派生 {stage.name} 命令失败：{type(exc).__name__}: {exc}"
            # 记下实际命令行：出错时用户能直接从接口看到「到底跑的是什么」，
            # 而不是只知道「编译失败了」。
            stage_state["command"] = list(info.get("command") or [])
            if info.get("expect_file") is not None:
                stage.expect_file = (
                    Path(info["expect_file"]) if info["expect_file"] else None
                )
            if info.get("report_file"):
                stage.report_file = str(info["report_file"])
            stage.command = list(info.get("command") or [])

        if stage.callable is not None:
            try:
                detail = stage.callable() or {}
            except Exception as exc:  # noqa: BLE001
                return False, f"{type(exc).__name__}: {exc}"
            if detail.get("artifact"):
                self._record_artifact(record, stage.name, detail["artifact"])
            # 一段可能同时产出多个东西（onnx + metadata + ckpt），
            # 后面几段的命令要读它们，只记一个就得让下游重新 glob 一遍。
            for key, value in (detail.get("artifacts") or {}).items():
                if value:
                    self._record_artifact(record, key, value)
            return True, None

        if not stage.command:
            return False, f"阶段 {stage.name} 既没有命令也没有可执行动作"

        try:
            run = self.run_manager.create(
                stage.run_kind(record["id"]),
                list(stage.command),
                stage.cwd,
                expect_file=stage.expect_file,
                meta=dict(stage.meta or {}),
            )
        except OSError as exc:
            return False, f"启动进程失败：{exc}"

        run_id = getattr(run, "run_id", None)
        stage_state["run_id"] = run_id
        record["run_id"] = run_id
        self._persist(record)

        status = self._wait(run_id)
        if status == SUCCESS:
            self._collect_artifacts(record, stage)
            return True, None
        if status == CANCELLED:
            return False, "已停止"
        if status is None:
            return False, "等待任务状态超时"
        return False, f"阶段 {stage.name} 失败（run 状态：{status}）"

    def _wait(self, run_id: str | None) -> str | None:
        """轮询等待某个 run 结束。

        轮询而不是订阅：订阅是 asyncio 的，这里是普通线程；而且 run 结束后
        状态已经是终态，多花 0.5 秒读一次比跨线程投递事件简单可靠得多。
        """
        if not run_id:
            return FAILED
        deadline = None if self.wait_timeout is None else time.time() + self.wait_timeout
        while True:
            run = self.run_manager.get(run_id)
            if run is None:
                return FAILED
            status = getattr(run, "status", None)
            if status in ("finished", "failed", "cancelled"):
                return SUCCESS if status == "finished" else status
            if deadline is not None and time.time() > deadline:
                return None
            self._sleep(self.wait_interval)

    def _record_artifact(self, record: dict[str, Any], name: str, path: Any) -> None:
        """按阶段名记一份产物路径，但**同一路径只出现一次**。

        初始 artifacts 里已经有 engine / transd_onnx 这些路径，阶段结束时再补一份
        只会让产物表出现两行一模一样的路径（用户会以为跑出了两个文件）。
        """
        artifacts = record.setdefault("artifacts", {})
        text = str(path)
        if text not in artifacts.values():
            artifacts[name] = text

    def _collect_artifacts(self, record: dict[str, Any], stage: StagePlan) -> None:
        if stage.expect_file:
            self._record_artifact(record, stage.name, stage.expect_file)
        if stage.report_file:
            path = Path(stage.report_file)
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    record["metrics_error"] = f"报告解析失败：{exc}"
                else:
                    record["metrics"] = payload
                    self._record_artifact(record, stage.name, path)
        self._persist(record)

    def _mark_remaining(self, record: dict[str, Any], start: int, status: str) -> None:
        for stage_state in record["stages"][start:]:
            if stage_state["status"] in (PENDING, RUNNING):
                stage_state["status"] = status
        self._persist(record)

    def _finish(self, record: dict[str, Any], status: str, error: str | None) -> None:
        record["status"] = status
        record["error"] = error
        record["finished_at"] = time.time()
        if record.get("started_at"):
            record["elapsed"] = round(record["finished_at"] - record["started_at"], 2)
        record["stage"] = None
        record["run_id"] = None
        self._persist(record)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _persist(self, record: dict[str, Any]) -> None:
        """写盘用「临时文件 + 替换」，避免刷新页面时读到半个 JSON。"""
        if not self.store_dir:
            return
        path = self.store_dir / f"{record['id']}.json"
        temp = path.with_suffix(".json.part")
        try:
            temp.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp, path)
        except OSError:
            # 写盘失败不该让整条链挂掉：状态还在内存里，页面照样能看
            pass


def read_persisted(store_dir: Path | str, pipeline_id: str) -> dict[str, Any] | None:
    """服务重启后从磁盘恢复查看（只读，不恢复执行）。"""
    path = Path(store_dir) / f"{pipeline_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
