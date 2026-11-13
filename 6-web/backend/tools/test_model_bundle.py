# -*- coding: utf-8 -*-
"""模型包契约测试（bundle.py + models.scan_models 的模型包分支）。

    python tools/test_model_bundle.py

测什么
------
1. **造包**：引擎与 ONNX 两条路径、清单字段、说明文件、来源文件、硬链接。
2. **原始产物不被改动**：这是「造包是只读操作」的硬保证，内容与修改时间都不能变。
3. **三档结论**：schema / 越界路径 / 哈希 / 缺文件 / 种类 / 任务 / TensorRT 主小版本 /
   显卡架构，每一档都断言 level，避免「该拦的只给了警告」。
4. **散装兼容**：没有任何模型包时模型列表照常返回散装条目。
5. **枚举不把包内文件当成散装引擎**：否则下拉里同一个模型出现两次。

所有用例都在临时目录里造数据，绝不碰仓库里真实的训练产物。
TensorRT 版本与显卡用 monkeypatch 固定，保证在任何机器上结论一致。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import bundle  # noqa: E402
import models as models_mod  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []

ENGINE_BYTES = b"FAKE-TENSORRT-ENGINE" * 8
ONNX_BYTES = b"FAKE-ONNX" * 8


def case(name: str):
    def wrap(func):
        def runner():
            try:
                func()
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


class FixedRuntime:
    """把运行时探测固定住。

    真实环境里 TensorRT 版本取决于 bin 目录、显卡取决于 nvidia-smi，
    直接把断言写成「等于本机版本」会让这个测试只能在某一台机器上通过。
    """

    def __init__(self, trt: str | None = "8.5.2.2", cap: str | None = "7.5", gpu: str | None = "NVIDIA T1000"):
        self.trt = trt
        self.cap = cap
        self.gpu = gpu

    def __enter__(self):
        self._orig = (bundle.trt_version, bundle.cuda_version, bundle.gpu_snapshot)
        bundle.trt_version = lambda: self.trt
        bundle.cuda_version = lambda: "11.8"
        bundle.gpu_snapshot = lambda: {"name": self.gpu, "compute_cap": self.cap}
        return self

    def __exit__(self, *exc_info):
        bundle.trt_version, bundle.cuda_version, bundle.gpu_snapshot = self._orig
        return False


class TempTree:
    """一个临时工作区：<tmp>/runs/proj/detect_v1/weights/ 与 <tmp>/bundles。"""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.weights = self.root / "runs" / "proj" / "detect_v1" / "weights"
        self.weights.mkdir(parents=True)
        self.run_dir = self.weights.parent
        self.out = self.root / "bundles"
        self.out.mkdir()
        self.engine = self.weights / "best_fp16.engine"
        self.engine.write_bytes(ENGINE_BYTES)
        self.onnx = self.weights / "best.transd.onnx"
        self.onnx.write_bytes(ONNX_BYTES)
        self.labels = self.run_dir / "labels_detect_v1.json"
        self.labels.write_text(
            json.dumps({"names": {"0": "missing_hole", "1": "short"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.metadata = self.weights / "metadata.json"
        self.metadata.write_text(
            json.dumps(
                {"model_type": "EfficientAd", "input_size": [256, 256],
                 "image_threshold": 0.51, "pixel_threshold": 0.42},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return self

    def __exit__(self, *exc_info):
        self._tmp.cleanup()
        return False


def _build(tree: TempTree, **overrides):
    """用固定运行时造一个引擎包，返回 (结果, 包目录)。"""
    kwargs = {
        "labels": tree.labels,
        "kind": "yolo",
        "task": "detect",
        "precision": "fp16",
        "imgsz": 960,
    }
    kwargs.update(overrides)
    with FixedRuntime():
        result = bundle.build_bundle(tree.engine, tree.out, "pcb_v1", **kwargs)
    return result, Path(result["path"]) if result.get("path") else None


# ---------------------------------------------------------------------------
# 造包
# ---------------------------------------------------------------------------

@case("造引擎包：目录、清单、说明文件、来源文件都齐全")
def test_build_engine_bundle():
    with TempTree() as tree:
        result, target = _build(tree)
        check(result["ok"], f"造包应成功：{result.get('error')}")
        check(target.is_dir(), "包目录应存在")

        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        check_eq(manifest["schema"], bundle.SCHEMA, "schema")
        check_eq(manifest["id"], "pcb_v1", "id")
        check_eq(manifest["kind"], "yolo", "kind")
        check_eq(manifest["task"], "detect", "task")
        check_eq(manifest["runtime"]["format"], "engine", "runtime.format")
        check_eq(manifest["runtime"]["precision"], "fp16", "runtime.precision")
        check_eq(manifest["runtime"]["trt_version"], "8.5.2.2", "runtime.trt_version")
        check_eq(manifest["runtime"]["compute_cap"], "7.5", "runtime.compute_cap")
        check_eq(manifest["runtime"]["imgsz"], [960, 960], "runtime.imgsz")
        check_eq(manifest["labels"]["count"], 2, "labels.count")
        check_eq(manifest["labels"]["names"], ["missing_hole", "short"], "labels.names")

        roles = {item["role"]: item["path"] for item in manifest["files"]}
        check_eq(roles.get("model"), "model.engine", "model 文件名")
        check_eq(roles.get("labels"), "labels.json", "labels 文件名")

        check((target / "README.txt").is_file(), "应生成 README.txt")
        check("换机器之前务必确认" in (target / "README.txt").read_text(encoding="utf-8"),
              "README 应说明换机要重编引擎")

        origin = json.loads((target / "origin.meta.json").read_text(encoding="utf-8"))
        check_eq(origin["source_model_name"], "best_fp16.engine", "origin.source_model_name")


@case("造 ONNX 包：format=onnx，且不记 TensorRT 与显卡")
def test_build_onnx_bundle():
    with TempTree() as tree:
        with FixedRuntime():
            result = bundle.build_bundle(
                tree.onnx, tree.out, "onnx_only", labels=tree.labels, kind="yolo", task="detect"
            )
        check(result["ok"], f"造包应成功：{result.get('error')}")
        manifest = result["manifest"]
        check_eq(manifest["runtime"]["format"], "onnx", "runtime.format")
        check_eq(manifest["runtime"]["trt_version"], None, "ONNX 不应记 TensorRT 版本")
        check_eq(manifest["runtime"]["gpu_name"], None, "ONNX 不应记显卡")
        check_eq(manifest["runtime"]["onnx_layout"], "transposed", "应识别 .transd.onnx")


@case("造包不改动原始产物（内容与修改时间都不变）")
def test_build_does_not_touch_source():
    with TempTree() as tree:
        before_bytes = tree.engine.read_bytes()
        before_mtime = tree.engine.stat().st_mtime_ns
        _build(tree)
        check_eq(tree.engine.read_bytes(), before_bytes, "引擎内容被改了")
        check_eq(tree.engine.stat().st_mtime_ns, before_mtime, "引擎修改时间被改了")
        check(tree.onnx.is_file(), "ONNX 不应被删除")


@case("同分区用硬链接，linked=true")
def test_build_links_on_same_volume():
    with TempTree() as tree:
        result, target = _build(tree)
        check(result["ok"], "造包应成功")
        check_eq(result["manifest"]["linked"], True, "同分区应硬链接")
        check_eq((target / "model.engine").stat().st_size, len(ENGINE_BYTES), "硬链接大小应与原文件一致")


@case("目标已存在时拒绝覆盖，显式 overwrite 才允许")
def test_build_overwrite_guard():
    with TempTree() as tree:
        first, _ = _build(tree)
        check(first["ok"], "第一次应成功")
        second, _ = _build(tree)
        check(not second["ok"], "重复造包应被拒绝")
        check("已存在" in second["error"], f"错误信息应说明已存在：{second.get('error')}")
        third, _ = _build(tree, overwrite=True)
        check(third["ok"], "显式 overwrite 应成功")


@case("模型文件不存在或不支持的后缀都给出可读错误")
def test_build_rejects_bad_input():
    with TempTree() as tree:
        missing = bundle.build_bundle(tree.root / "nope.engine", tree.out, "x")
        check(not missing["ok"], "不存在的文件应失败")
        check("不存在" in missing["error"], "错误应说明文件不存在")

        weird = tree.root / "weights.pt"
        weird.write_bytes(b"x")
        bad = bundle.build_bundle(weird, tree.out, "y")
        check(not bad["ok"], "不支持的 suffix 应失败")
        check("后缀" in bad["error"], "错误应说明后缀不支持")


@case("包名里的路径分隔符被收敛，不能建到目录外")
def test_build_sanitizes_bundle_id():
    with TempTree() as tree:
        with FixedRuntime():
            result = bundle.build_bundle(tree.engine, tree.out, "../../evil")
        check(result["ok"], "应成功但落在 out_root 内")
        check_eq(Path(result["path"]).parent.resolve(), tree.out.resolve(), "不能建到 out_root 之外")


# ---------------------------------------------------------------------------
# 校验：结构与完整性
# ---------------------------------------------------------------------------

def _verify(target: Path, **kwargs):
    with FixedRuntime():
        return bundle.verify_bundle(target, **kwargs)


@case("校验通过：ok，且每项检查都有名字")
def test_verify_ok():
    with TempTree() as tree:
        _, target = _build(tree)
        verdict = _verify(target, kind="yolo", task="detect")
        check_eq(verdict["level"], bundle.LEVEL_OK, "结论")
        names = {item["name"] for item in verdict["checks"]}
        for expected in ("清单版本", "必填字段", "包内路径", "文件存在", "文件完整性",
                         "种类匹配", "任务匹配", "TensorRT 版本"):
            check(expected in names, f"缺少检查项：{expected}")


@case("schema 不认识 → block")
def test_verify_unknown_schema():
    with TempTree() as tree:
        _, target = _build(tree)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["schema"] = "trtforge.bundle.v999"
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "未知 schema 必须 block")


@case("缺少必填字段 → block 且逐项列出")
def test_verify_missing_fields():
    with TempTree() as tree:
        _, target = _build(tree)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        del manifest["runtime"]
        del manifest["task"]
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "缺必填字段必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "必填字段")
        check("runtime" in detail and "task" in detail, f"应列出缺失字段：{detail}")


@case("包内路径越界（绝对路径 / ..）→ block，且不打开该路径")
def test_verify_path_escape():
    with TempTree() as tree:
        _, target = _build(tree)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["files"].append(
            {"role": "labels", "path": "../../../Windows/System32/drivers/etc/hosts", "sha256": ""}
        )
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "越界路径必须 block")
        names = {item["name"] for item in verdict["checks"]}
        check("包内路径" in names, "应报告包内路径检查")


@case("哈希不一致 → block，并给出登记值与实际值")
def test_verify_hash_mismatch():
    with TempTree() as tree:
        _, target = _build(tree)
        (target / "model.engine").write_bytes(b"TAMPERED")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "哈希不符必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "文件完整性")
        check("model.engine" in detail, f"应指明文件：{detail}")


@case("文件缺失 → block")
def test_verify_missing_file():
    with TempTree() as tree:
        _, target = _build(tree)
        (target / "model.engine").unlink()
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "缺文件必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "文件存在")
        check("model.engine" in detail, f"应指明缺失文件：{detail}")


@case("清单不可读 → block，不抛异常")
def test_verify_unreadable_manifest():
    with TempTree() as tree:
        _, target = _build(tree)
        (target / "manifest.json").write_text("{ not json", encoding="utf-8")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "坏清单必须 block")
        check_eq(verdict["manifest"], None, "坏清单时不返回 manifest")


# ---------------------------------------------------------------------------
# 校验：语义与运行时
# ---------------------------------------------------------------------------

@case("种类冲突 → block")
def test_verify_kind_conflict():
    with TempTree() as tree:
        _, target = _build(tree)
        verdict = _verify(target, kind="anomalib")
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "种类冲突必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "种类匹配")
        check("yolo" in detail, f"应说明包的种类：{detail}")


@case("任务冲突 → block")
def test_verify_task_conflict():
    with TempTree() as tree:
        _, target = _build(tree)
        verdict = _verify(target, kind="yolo", task="segment")
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "任务冲突必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "任务匹配")
        check("detect" in detail, f"应说明训练时的任务：{detail}")


@case("TensorRT 主版本不同 → block")
def test_verify_trt_major_differs():
    with TempTree() as tree:
        _, target = _build(tree)
        with FixedRuntime(trt="10.0.1.6"):
            verdict = bundle.verify_bundle(target)
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "主版本不同必须 block")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "TensorRT 版本")
        check("重新编译" in (next(item["fix"] for item in verdict["checks"]
                                 if item["name"] == "TensorRT 版本")), f"应给出修复建议：{detail}")


@case("TensorRT 小版本不同 → warn（不阻止推理）")
def test_verify_trt_minor_differs():
    with TempTree() as tree:
        _, target = _build(tree)
        with FixedRuntime(trt="8.6.1.6"):
            verdict = bundle.verify_bundle(target)
        check_eq(verdict["level"], bundle.LEVEL_WARN, "小版本不同应为 warn")


@case("运行时字段缺失 → warn 而不是 block（旧产物不能被升级挡住）")
def test_verify_missing_runtime_fields():
    with TempTree() as tree:
        _, target = _build(tree)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["runtime"]["trt_version"] = None
        manifest["runtime"]["compute_cap"] = None
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        verdict = _verify(target)
        check_eq(verdict["level"], bundle.LEVEL_WARN, "缺运行时字段应为 warn")


@case("显卡架构不同 → warn，并给出记录的与本机的型号")
def test_verify_gpu_arch_differs():
    with TempTree() as tree:
        _, target = _build(tree)
        with FixedRuntime(cap="8.9", gpu="NVIDIA RTX 4060"):
            verdict = bundle.verify_bundle(target)
        check_eq(verdict["level"], bundle.LEVEL_WARN, "架构不同应为 warn")
        detail = next(item["detail"] for item in verdict["checks"] if item["name"] == "显卡架构")
        check("T1000" in detail and "4060" in detail, f"应给出两个型号：{detail}")


@case("异常检测包缺 metadata → block")
def test_verify_anomalib_requires_metadata():
    with TempTree() as tree:
        with FixedRuntime():
            result = bundle.build_bundle(
                tree.engine, tree.out, "anomaly_no_meta", kind="anomalib", task="anomaly"
            )
        check(result["ok"], "造包应成功（造包不强制 metadata）")
        verdict = _verify(Path(result["path"]))
        check_eq(verdict["level"], bundle.LEVEL_BLOCK, "异常检测缺 metadata 必须 block")


@case("异常检测包带 metadata → ok，且阈值被记进清单")
def test_build_anomalib_with_metadata():
    with TempTree() as tree:
        with FixedRuntime():
            result = bundle.build_bundle(
                tree.engine, tree.out, "anomaly_ok",
                metadata=tree.metadata, kind="anomalib", task="anomaly",
            )
        check(result["ok"], "造包应成功")
        check_eq(result["manifest"]["anomaly"]["image_threshold"], 0.51, "图像阈值应进清单")
        verdict = _verify(Path(result["path"]), kind="anomalib")
        check_eq(verdict["level"], bundle.LEVEL_OK, "带 metadata 的异常检测包应通过")


# ---------------------------------------------------------------------------
# 枚举与解析
# ---------------------------------------------------------------------------

@case("按包 id 与按路径都能找到包")
def test_find_bundle():
    with TempTree() as tree:
        _, target = _build(tree)
        original = bundle.default_root
        bundle.default_root = lambda: tree.out
        try:
            by_id = bundle.find_bundle("pcb_v1")
            check(by_id is not None, "按 id 应能找到")
            check_eq(by_id.resolve(), target.resolve(), "按 id 找到的路径")
            by_path = bundle.find_bundle(str(target))
            check(by_path is not None, "按路径应能找到")
            check(bundle.find_bundle("不存在") is None, "找不到应返回 None")
        finally:
            bundle.default_root = original


@case("resolve_bundle 给出模型、标签、metadata 与种类")
def test_resolve_bundle():
    with TempTree() as tree:
        with FixedRuntime():
            result = bundle.build_bundle(
                tree.engine, tree.out, "resolvable",
                labels=tree.labels, metadata=tree.metadata, kind="anomalib", task="anomaly",
            )
        resolved = bundle.resolve_bundle(result["path"])
        check(resolved["ok"], f"应解析成功：{resolved.get('error')}")
        check_eq(resolved["kind"], "anomalib", "kind")
        check(resolved["model"].endswith("model.engine"), f"模型路径：{resolved['model']}")
        check(resolved["labels"].endswith("labels.json"), f"标签路径：{resolved['labels']}")
        check(resolved["metadata"].endswith("metadata.json"), f"metadata 路径：{resolved['metadata']}")


@case("quick_check 不读内容也能给出结论")
def test_quick_check():
    with TempTree() as tree:
        _, target = _build(tree)
        verdict = bundle.quick_check(target)
        check_eq(verdict["level"], bundle.LEVEL_OK, "完整包应 ok")
        (target / "model.engine").unlink()
        broken = bundle.quick_check(target)
        check_eq(broken["level"], bundle.LEVEL_BLOCK, "缺文件应 block")


@case("模型列表带模型包，且不把包内 model.engine 当散装条目")
def test_models_reports_bundles_and_skips_internal():
    with TempTree() as tree:
        # 一个散装引擎（不在包里）
        loose = tree.weights / "other_fp16.engine"
        loose.write_bytes(b"LOOSE")

        original_targets = models_mod.SCAN_TARGETS
        original_config = models_mod.config
        original_default = bundle.default_root
        models_mod.SCAN_TARGETS = [("yolo", lambda: tree.root / "runs")]
        bundle.default_root = lambda: tree.out
        models_mod.config = type("Stub", (), {"is_path_allowed": staticmethod(lambda _p: True)})()
        try:
            with FixedRuntime():
                built = bundle.build_bundle(
                    tree.engine, tree.out, "listed", labels=tree.labels, kind="yolo", task="detect"
                )
            check(built["ok"], "造包应成功")
            payload = models_mod.scan_models()
        finally:
            models_mod.SCAN_TARGETS = original_targets
            models_mod.config = original_config
            bundle.default_root = original_default

        paths = {item["path"] for item in payload["entries"]}
        check(any(path.endswith("other_fp16.engine") for path in paths), "散装引擎应仍被列出")
        check(
            not any("model.engine" in path.replace("\\", "/") for path in paths),
            f"包内 model.engine 不应被当成散装条目：{paths}",
        )
        check_eq(len(payload["bundles"]), 1, "应列出 1 个模型包")
        check_eq(payload["bundles"][0]["verify"]["level"], bundle.LEVEL_OK, "包的轻量校验结论")


@case("没有模型包时模型列表照常返回散装条目")
def test_models_without_bundles():
    with TempTree() as tree:
        original_targets = models_mod.SCAN_TARGETS
        original_config = models_mod.config
        original_default = bundle.default_root
        models_mod.SCAN_TARGETS = [("yolo", lambda: tree.root / "runs")]
        bundle.default_root = lambda: tree.root / "no_such_dir"
        models_mod.config = type("Stub", (), {"is_path_allowed": staticmethod(lambda _p: True)})()
        try:
            payload = models_mod.scan_models()
        finally:
            models_mod.SCAN_TARGETS = original_targets
            models_mod.config = original_config
            bundle.default_root = original_default
        check_eq(payload["bundles"], [], "没有包时 bundles 为空")
        check(any(item["path"].endswith("best_fp16.engine") for item in payload["entries"]),
              "散装条目应照常返回")


@case("四个模型包接口都已注册")
def test_routes_registered():
    import app  # noqa: PLC0415 - 导入即构造路由表

    table = {(m, getattr(route, "path", "")) for route in app.app.routes
             for m in (getattr(route, "methods", None) or [])}
    for expected in (("GET", "/api/bundles"), ("POST", "/api/bundles/build"),
                     ("POST", "/api/bundles/verify"), ("GET", "/api/bundles/{bundle_id}/manifest")):
        check(expected in table, f"缺少路由：{expected}")


@case("推理接口接受 bundle 参数且 engine 变为可选")
def test_infer_accepts_bundle():
    import inspect  # noqa: PLC0415

    import app  # noqa: PLC0415

    signature = inspect.signature(app.infer_yolo)
    check("bundle" in signature.parameters, "infer_yolo 应有 bundle 参数")
    check_eq(signature.parameters["engine"].default.default, "", "engine 应可选（默认空串）")

    anomaly_signature = inspect.signature(app.infer_anomalib)
    check("bundle" in anomaly_signature.parameters, "infer_anomalib 应有 bundle 参数")
    check_eq(anomaly_signature.parameters["metadata"].default.default, "", "metadata 应可选")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"模型包契约测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}  schema={bundle.SCHEMA}")
    print()

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()

    print()
    print("=" * 60)
    total = len(PASSED) + len(FAILED)
    print(f"通过 {len(PASSED)} / {total}")
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
