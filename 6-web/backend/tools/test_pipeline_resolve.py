# -*- coding: utf-8 -*-
"""串联的「延迟派生命令」与「多产物记录」验收脚本，不需要 GPU / trtexec / DLL。

守的是异常检测串联特有的两类错：

1. **路径猜错**：Anomalib 的产物落在 `<project>/<模型类名>/<name>/vN/...`，
   版本号训练完才知道。硬拼 `v0` 在第二次训练时会指向上一次的旧模型，
   旧模型同样能编译、同样能出分数——错得毫无提示。所以命令必须等到
   上一段产物确定后再派生（`StagePlan.resolve`），且找不到时要明确失败。
2. **产物只记了一个**：一段同时产出 onnx + metadata + ckpt 时，只记一个会让
   下游重新 glob 一遍，两处规则迟早分叉。

用法：
    python 6-web/backend/tools/test_pipeline_resolve.py
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

import app as backend_app  # noqa: E402
import pipeline  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))


class FakeRun:
    def __init__(self, run_id: str, command: list[str], expect_file):
        self.run_id = run_id
        self.command = command
        self.expect_file = expect_file
        self.status = "running"
        self.outcome = "finished"


class FakeRunManager:
    """立刻成功的假 run 管理器：这里只验证「命令拼得对不对」。"""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.runs: dict[str, FakeRun] = {}

    def create(self, kind, command, cwd="", expect_file=None, meta=None, **kwargs):
        run_id = f"run-{len(self.runs) + 1}"
        run = FakeRun(run_id, list(command), expect_file)
        self.runs[run_id] = run
        self.created.append(
            {"kind": kind, "command": list(command), "expect_file": expect_file, "cwd": cwd}
        )
        return run

    def get(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            return None
        # 第一次查询就当作「跑完了」：这里只验证编排，不验证等待
        if run.status == "running":
            run.status = run.outcome
        return run

    def stop(self, run_id):
        run = self.runs.get(run_id)
        if run is not None:
            run.status = "cancelled"
        return True


def make_manager() -> pipeline.PipelineManager:
    # store_dir=None：这些用例不关心持久化，别往 runtime 里写垃圾记录
    return pipeline.PipelineManager(
        FakeRunManager(),
        store_dir=None,
        wait_interval=0.0,
        wait_timeout=5.0,
        sleep=lambda _: None,
    )


def wait_done(manager: pipeline.PipelineManager, pipeline_id: str, timeout: float = 10.0) -> dict:
    """等监督线程跑到终态。

    `create()` 只是把线程启起来就返回快照，立刻读会读到 running——
    这不是产品缺陷（前端本来就是轮询），但断言必须等它跑完。
    """
    deadline = time.time() + timeout
    record = manager.get(pipeline_id) or {}
    while record.get("status") in (pipeline.PENDING, pipeline.CHAIN_RUNNING):
        if time.time() > deadline:
            return record
        time.sleep(0.01)
        record = manager.get(pipeline_id) or {}
    return record


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="pipe_resolve_"))
    try:
        # ---- 1. resolve 从上一段产物派生命令 ------------------------------
        onnx = tmp_root / "model.onnx"
        onnx.write_bytes(b"fake-onnx")
        metadata = tmp_root / "metadata.json"
        metadata.write_text("{}", encoding="utf-8")
        report = tmp_root / "report.json"

        manager = make_manager()
        fake = manager.run_manager

        def stage_one() -> dict:
            return {
                "artifact": str(metadata),
                "artifacts": {
                    "onnx": str(onnx),
                    "metadata": str(metadata),
                    "ckpt": str(tmp_root / "model.ckpt"),
                },
            }

        def resolve_two(record: dict) -> dict:
            report.write_text(json.dumps({"ok": True}), encoding="utf-8")
            return {
                "command": ["trtexec", f"--onnx={record['artifacts']['onnx']}"],
                "expect_file": onnx.with_name("model_fp16.engine"),
                "report_file": report,
            }

        record = manager.create(
            kind="anomalib",
            title="t",
            stages=[
                pipeline.StagePlan(name="probe", label="校验", callable=stage_one),
                pipeline.StagePlan(name="build", label="编译", resolve=resolve_two),
            ],
        )
        result = wait_done(manager, record["id"])
        check("整链成功", result["status"] == "success", str(result.get("error")))
        recorded = set(result["artifacts"].values())
        check(
            "多产物都记进 artifacts（同路径只记一次）",
            {str(onnx), str(metadata), str(tmp_root / "model.ckpt")} <= recorded,
            str(result["artifacts"]),
        )
        check("probe 的 artifact 也记了", "probe" in result["artifacts"], str(result["artifacts"]))
        check(
            "resolve 生成的命令被真正执行",
            bool(fake.created) and fake.created[0]["command"] == ["trtexec", "--onnx=" + str(onnx)],
            str(fake.created),
        )
        check(
            "expect_file 统一成 Path（runners 要求）",
            isinstance(fake.created[0]["expect_file"], Path),
            type(fake.created[0]["expect_file"]).__name__,
        )
        check(
            "resolve 的 report_file 被收进 metrics",
            (result.get("metrics") or {}).get("ok") is True,
            str(result.get("metrics")),
        )
        build_state = next(s for s in result["stages"] if s["name"] == "build")
        check(
            "阶段记录实际命令行（排障能直接看到）",
            build_state.get("command") == ["trtexec", "--onnx=" + str(onnx)],
            str(build_state.get("command")),
        )

        # ---- 2. resolve 抛异常 → 该段失败、整链 failed、后续 skipped --------
        def resolve_boom(_record: dict) -> dict:
            raise FileNotFoundError("找不到 v1 目录")

        manager2 = make_manager()
        record2 = manager2.create(
            kind="anomalib",
            title="t2",
            stages=[
                pipeline.StagePlan(
                    name="probe", label="校验", callable=lambda: {"artifact": str(metadata)}
                ),
                pipeline.StagePlan(name="build", label="编译", resolve=resolve_boom),
                pipeline.StagePlan(name="val", label="验证", command=["python", "val.py"]),
            ],
        )
        result2 = wait_done(manager2, record2["id"])
        check("resolve 失败时整链 failed", result2["status"] == "failed", str(result2["status"]))
        check(
            "错误信息说明是派生命令失败",
            "派生" in (result2.get("error") or ""),
            str(result2.get("error")),
        )
        check("resolve 失败时不会起子进程", not manager2.run_manager.created)
        statuses = {s["name"]: s["status"] for s in result2["stages"]}
        check(
            "失败的段是 failed，后续段 skipped",
            statuses.get("build") == "failed" and statuses.get("val") == "skipped",
            str(statuses),
        )

        # ---- 3. 产物去重：与初始 artifacts 同路径只出现一次 ------------------
        manager3 = make_manager()
        record3 = manager3.create(
            kind="anomalib",
            title="t3",
            stages=[
                pipeline.StagePlan(
                    name="probe",
                    label="校验",
                    callable=lambda: {"artifacts": {"onnx": str(onnx)}},
                )
            ],
            artifacts={"onnx": str(onnx)},
        )
        result3 = wait_done(manager3, record3["id"])
        values = list(result3["artifacts"].values())
        check(
            "同一路径不重复记录",
            values.count(str(onnx)) == 1 and "probe" not in result3["artifacts"],
            str(result3["artifacts"]),
        )

        # ---- 4. 产物定位：选最新版本、拒绝残缺目录 --------------------------
        project = tmp_root / "project"
        for version in ("v0", "v1"):
            lightning = project / "Patchcore" / "demo" / version / "weights" / "lightning"
            lightning.mkdir(parents=True)
            (lightning / "metadata.json").write_text(
                json.dumps({"image_threshold": 0.5, "pixel_threshold": 0.4}),
                encoding="utf-8",
            )
            (lightning / "model.onnx").write_bytes(b"onnx")
        # v1 上再放一个更新的 onnx，验证「同目录取最新」
        newer = project / "Patchcore" / "demo" / "v1" / "weights" / "lightning" / "model_b.onnx"
        newer.write_bytes(b"onnx2")

        located = backend_app._locate_anomaly_artifacts(project, "demo")
        check("定位到最新版本 v1", located["version"] == "v1", str(located))
        check(
            "同目录多个 onnx 取最新的那个",
            Path(located["onnx"]).name == "model_b.onnx",
            located["onnx"],
        )
        check(
            "metadata 与 onnx 在同一目录",
            Path(located["metadata"]).parent == Path(located["onnx"]).parent,
        )

        try:
            backend_app._locate_anomaly_artifacts(project, "does-not-exist")
            check("找不到产物时抛异常", False)
        except ValueError as exc:
            check(
                "找不到产物时抛异常并给出目录",
                "does-not-exist" in str(exc) and str(project) in str(exc),
                str(exc),
            )

        empty = tmp_root / "empty_lightning"
        empty.mkdir()
        (empty / "metadata.json").write_text("{}", encoding="utf-8")
        try:
            backend_app._anomaly_artifacts_in(empty)
            check("目录里没有 onnx 时抛异常", False)
        except ValueError as exc:
            check("目录里没有 onnx 时抛异常", ".onnx" in str(exc), str(exc))

        # ---- 5. metadata 校验：缺阈值要在编译前拦住 --------------------------
        missing = tmp_root / "missing_meta.json"
        missing.write_text(json.dumps({"pixel_threshold": 0.4}), encoding="utf-8")
        try:
            backend_app._probe_anomaly_metadata(
                {"metadata": str(missing), "onnx": str(onnx)}
            )
            check("缺 image_threshold 时拒绝继续", False)
        except ValueError as exc:
            check("缺 image_threshold 时拒绝继续", "image_threshold" in str(exc), str(exc))

        bool_threshold = tmp_root / "bool_meta.json"
        bool_threshold.write_text(
            json.dumps({"image_threshold": True, "pixel_threshold": 0.4}), encoding="utf-8"
        )
        try:
            backend_app._probe_anomaly_metadata(
                {"metadata": str(bool_threshold), "onnx": str(onnx)}
            )
            check("阈值是 bool 时也拒绝（True 会被当成 1.0）", False)
        except ValueError as exc:
            check(
                "阈值是 bool 时也拒绝（True 会被当成 1.0）",
                "image_threshold" in str(exc),
                str(exc),
            )

        check(
            "引擎路径按精度加后缀",
            backend_app._anomaly_engine_path(Path("a/model.onnx"), "fp16").name
            == "model_fp16.engine"
            and backend_app._anomaly_engine_path(Path("a/model.onnx"), "fp32").name
            == "model_fp32.engine",
        )
        check(
            "项目根按 launcher 目录解析相对路径",
            backend_app._anomaly_project_root("results").name == "results"
            and backend_app._anomaly_project_root("results").is_absolute(),
        )

        # ---- 6. 动态 batch：anomalib 的 ONNX 恒为 batch 动态，必须能编译 -----
        # Anomalib `to_onnx` 固定 `dynamic_axes={"input": {0: "batch_size"}}`，
        # `_probe_onnx` 把符号维度读成 -1。校验不能把它当错误，编译要补形状。
        args = backend_app._anomaly_shape_args(
            {"input_name": "input", "input_shape": [-1, 3, 256, 256]}, 8
        )
        check(
            "动态 batch 补 min/opt/max 形状（opt 取 1 与 DLL 一致）",
            args
            == [
                "--minShapes=input:1x3x256x256",
                "--optShapes=input:1x3x256x256",
                "--maxShapes=input:8x3x256x256",
            ],
            str(args),
        )
        check(
            "max_batch 小于实际 batch 时以实际 batch 为准",
            backend_app._anomaly_shape_args(
                {"input_name": "input", "input_shape": [-1, 3, 64, 64]}, 2
            )[-1]
            == "--maxShapes=input:2x3x64x64",
        )
        check(
            "静态形状不需要形状参数",
            backend_app._anomaly_shape_args(
                {"input_name": "input", "input_shape": [1, 3, 256, 256]}, 8
            )
            == [],
        )
        dynamic_hw = tmp_root / "dynamic_hw_meta.json"
        dynamic_hw.write_text(
            json.dumps({"image_threshold": 0.5, "pixel_threshold": 0.4}), encoding="utf-8"
        )
        fake_probe = {"input_shape": [-1, 3, -1, -1], "input_name": "input"}
        original_probe = backend_app._probe_onnx
        try:
            backend_app._probe_onnx = lambda _path: fake_probe  # type: ignore[assignment]
            try:
                backend_app._probe_anomaly_metadata(
                    {"metadata": str(dynamic_hw), "onnx": str(onnx)}
                )
                check("高/宽动态时拒绝编译", False)
            except ValueError as exc:
                check("高/宽动态时拒绝编译", "动态" in str(exc), str(exc))

            fake_probe = {"input_shape": [-1, 3, 256, 256], "input_name": "input"}
            detail = backend_app._probe_anomaly_metadata(
                {"metadata": str(dynamic_hw), "onnx": str(onnx)}
            )
            check(
                "batch 动态时通过校验，形状按 batch=1 记",
                detail["dynamic_batch"] is True and detail["input_shape"] == [1, 3, 256, 256],
                str(detail),
            )
        finally:
            backend_app._probe_onnx = original_probe  # type: ignore[assignment]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    total = len(PASSED) + len(FAILED)
    if FAILED:
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print(f"通过 {total} 项，失败 0 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
