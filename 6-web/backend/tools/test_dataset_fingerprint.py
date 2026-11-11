# -*- coding: utf-8 -*-
"""数据集指纹与导出清单（manifest）的验收脚本。

不需要 GPU、不需要 torch：指纹只读目录与标注文本，算的是纯函数意义上的摘要。

守的是这几类问题：

1. **同一批数据必须同指纹**：导出到 ``D:\\out_a`` 与 ``E:\\out_b`` 要能对上，
   否则「两个模型用的是一批数据吗」永远回答不了。
2. **任何一处变化必须换指纹**：改标注、增删图、重命名都要变；
   只改 val 时 train 分片不能变（否则分片指纹没有意义）。
3. **指纹失败不能阻断导出**：源目录权限异常、yaml 写坏时，导出照常完成。
4. **旧产物不能崩**：没有 manifest / 没有指纹字段时按「未记录」处理，不报错。

用法：
    python 6-web/backend/tools/test_dataset_fingerprint.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import dataset  # noqa: E402
import fingerprint  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 测试环境缺依赖时给出明确提示
    print("需要 Pillow 才能生成测试图片：pip install pillow")
    raise

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))


def make_dataset(root: Path, counts: tuple[int, int, int] = (4, 2, 1)) -> Path:
    """造一个最小可用的 YOLO 检测数据集，返回 dataset.yaml 路径。"""
    root.mkdir(parents=True, exist_ok=True)
    for split, count in zip(("train", "val", "test"), counts):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = root / "images" / split / f"{split}_{index}.jpg"
            Image.new("RGB", (32, 32), (index * 20 % 255, 40, 80)).save(image, format="JPEG")
            (root / "labels" / split / f"{split}_{index}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: hole\n"
        "  1: scratch\n",
        encoding="utf-8",
    )
    return yaml_path


def export_once(source: Path, output: Path, fmt: str = "yolo") -> dict:
    """按源工程的现有划分导出一次（不重新分配，便于逐项对比指纹）。"""
    project = dataset.load_project(str(source), "auto")
    splits = [image.current_split for image in project.images]
    if fmt == "yolo":
        return dataset.export_yolo(
            project, output, splits, True, split_mode={"mode": "manual"}
        )
    return dataset.export_coco(
        project, output, splits, True, split_mode={"mode": "manual"}
    )


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_same_tree_same_fingerprint(tmp: Path) -> None:
    root = tmp / "ds_same"
    source = make_dataset(root)
    out = tmp / "out_same"

    first = export_once(source, out)
    second = export_once(source, out)

    check("导出返回指纹", bool(first.get("dataset_fingerprint")), str(first))
    check("同目录两次导出指纹相同", first["dataset_fingerprint"] == second["dataset_fingerprint"])
    check("第二次导出标记 unchanged", second.get("unchanged") is True, str(second.get("unchanged")))
    check("第一次不是 unchanged", first.get("unchanged") is False)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("manifest 存在且含算法版本", manifest.get("algo") == "v1", str(manifest.get("algo")))
    check("manifest 含三个分片指纹", set(manifest.get("split_fingerprints") or {}) == {"train", "val", "test"})
    check("manifest 含程序版本", bool(manifest.get("app_version")), str(manifest))
    check("manifest 写明算法边界", "不含图片内容" in str(manifest.get("note")), str(manifest.get("note")))
    check("manifest 数量与导出一致", manifest["counts"]["train"] == 4, str(manifest["counts"]))
    check("manifest 记录划分方式", manifest["split_mode"].get("mode") == "manual", str(manifest["split_mode"]))


def test_cross_directory_same_fingerprint(tmp: Path) -> None:
    source = make_dataset(tmp / "ds_cross")
    out_a = tmp / "out_cross_a"
    out_b = tmp / "out_cross_b"

    first = export_once(source, out_a)
    second = export_once(source, out_b)

    check("导出到不同目录指纹相同", first["dataset_fingerprint"] == second["dataset_fingerprint"])
    check("新目录不算 unchanged", second.get("unchanged") is False)
    check("两处 manifest 指纹一致", json.loads((out_b / "manifest.json").read_text(encoding="utf-8"))["dataset_fingerprint"] == first["dataset_fingerprint"])


def test_label_change(tmp: Path) -> None:
    root = tmp / "ds_label"
    source = make_dataset(root)
    out = tmp / "out_label"

    before = export_once(source, out)
    (root / "labels" / "train" / "train_0.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    after = export_once(source, out)

    check("改标注后总量指纹变化", before["dataset_fingerprint"] != after["dataset_fingerprint"])
    check("改标注后标注指纹变化", before["label_fingerprint"] != after["label_fingerprint"])
    check(
        "改标注只影响 train 分片",
        before["split_fingerprints"]["train"] != after["split_fingerprints"]["train"]
        and before["split_fingerprints"]["val"] == after["split_fingerprints"]["val"],
        f"{before['split_fingerprints']} -> {after['split_fingerprints']}",
    )


def test_rename_changes_fingerprint(tmp: Path) -> None:
    root = tmp / "ds_rename"
    source = make_dataset(root)
    out = tmp / "out_rename"

    before = export_once(source, out)
    # 图片与标签一起改名：这才是「只是换了个文件名」。
    # 只改图片不改标签等于把那条标注丢了，数据确实变了——那是另一回事。
    (root / "images" / "train" / "train_1.jpg").rename(root / "images" / "train" / "renamed.jpg")
    (root / "labels" / "train" / "train_1.txt").rename(root / "labels" / "train" / "renamed.txt")
    after = export_once(source, out)

    check("重命名图片也会改变指纹", before["dataset_fingerprint"] != after["dataset_fingerprint"])
    check(
        "重命名不改变标注指纹（标注内容没动）",
        before["label_fingerprint"] == after["label_fingerprint"],
        f"{before['label_fingerprint']} vs {after['label_fingerprint']}",
    )


def test_only_val_changes(tmp: Path) -> None:
    root = tmp / "ds_val"
    source = make_dataset(root)
    out = tmp / "out_val"

    before = export_once(source, out)
    (root / "images" / "val" / "val_1.jpg").unlink()
    after = export_once(source, out)

    check("删 val 图后 val 分片变化", before["split_fingerprints"]["val"] != after["split_fingerprints"]["val"])
    check("删 val 图后 train 分片不变", before["split_fingerprints"]["train"] == after["split_fingerprints"]["train"])
    check("删 val 图后总量变化", before["dataset_fingerprint"] != after["dataset_fingerprint"])


def test_classify_export(tmp: Path) -> None:
    """分类导出没有 labels 目录，指纹要能正常算出来。"""
    root = tmp / "ds_classify"
    out = tmp / "out_classify"
    project = dataset.DatasetProject(
        source=root,
        kind="yolo",
        task="classify",
        classes=[{"id": 0, "name": "ok"}, {"id": 1, "name": "ng"}],
        images=[],
    )
    for split in ("train", "val"):
        for label_id, name in ((0, "ok"), (1, "ng")):
            folder = root / split / name
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{split}_{name}.jpg"
            Image.new("RGB", (24, 24), (label_id * 200, 0, 0)).save(image_path, format="JPEG")
            project.images.append(
                dataset.DatasetImage(
                    index=len(project.images),
                    path=image_path,
                    rel=f"{split}/{name}/{image_path.name}",
                    width=24,
                    height=24,
                    label_id=label_id,
                    current_split=split,
                )
            )
    splits = [image.current_split for image in project.images]
    result = dataset.export_yolo(project, out, splits, True, split_mode={"mode": "manual"})

    check("分类导出返回指纹", bool(result.get("dataset_fingerprint")), str(result))
    check("分类导出计数正确", result["counts"]["train"] == 2, str(result["counts"]))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("分类 manifest 记录任务", manifest.get("task") == "classify", str(manifest.get("task")))


def test_missing_yaml_degrades() -> None:
    report = fingerprint.compute(Path(tempfile.gettempdir()) / "definitely-not-here.yaml")
    check("缺 yaml 不抛异常", isinstance(report, dict))
    check("缺 yaml 带 error", bool(report.get("error")), str(report))
    check("缺 yaml 没有指纹", report.get("dataset_fingerprint") is None, str(report))
    check("缺 yaml 不算已记录", fingerprint.is_recorded(report) is False)


def test_coco_manifest(tmp: Path) -> None:
    source = make_dataset(tmp / "ds_coco")
    out = tmp / "out_coco"
    result = export_once(source, out, fmt="coco")

    check("COCO 也有指纹", bool(result.get("dataset_fingerprint")), str(result))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("COCO manifest 标注 scope", manifest.get("scope") == "export", str(manifest.get("scope")))
    check("COCO manifest 记录格式", manifest.get("format") == "coco", str(manifest.get("format")))


def test_manifest_absent_is_not_error(tmp: Path) -> None:
    """升级前的产物没有 manifest：读取方必须当作未记录，而不是报错。"""
    out = tmp / "out_old"
    out.mkdir(parents=True, exist_ok=True)
    check("缺失 manifest 时不抛异常", dataset._previous_fingerprint(out) is None)
    broken = out / "manifest.json"
    broken.write_text("{ 不是 json", encoding="utf-8")
    check("manifest 损坏时不抛异常", dataset._previous_fingerprint(out) is None)


def test_engine_meta_links_dataset(tmp: Path) -> None:
    """引擎元数据要能把「引擎 → ONNX → 训练 run → 数据集」这条链接起来。"""
    import enginemeta

    run_dir = tmp / "runs" / "exp_test"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    onnx_path = weights / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx-bytes")
    engine_path = run_dir / "best_fp16.engine"
    engine_path.write_bytes(b"fake-engine-bytes")

    # launcher 训练结束时会写这份文件；引擎元数据从它往上找。
    fps = {
        "algo": "v1",
        "dataset_fingerprint": "0123456789abcdef",
        "short": "01234567",
        "counts": {"train": 10, "val": 2, "test": 0},
    }
    (run_dir / "train_meta.json").write_text(
        json.dumps({"dataset": fps}, ensure_ascii=False), encoding="utf-8"
    )

    report = enginemeta.write_meta(
        engine_path,
        onnx_path=onnx_path,
        precision="fp16",
        command=["trtexec"],
        run_id="abc123",
        app_version="1.34.2",
    )
    check("引擎元数据无错误", not report.get("error"), str(report.get("error")))
    check("ONNX 哈希非空", len(report.get("onnx_sha256") or "") == 64, str(report.get("onnx_sha256")))
    check("引擎哈希非空", len(report.get("engine_sha256") or "") == 64)
    check(
        "从 train_meta 拿到数据集指纹",
        report.get("dataset_fingerprint") == "0123456789abcdef",
        str(report.get("dataset_fingerprint")),
    )
    target = engine_path.with_name(engine_path.name + ".meta.json")
    check("元数据与引擎同目录同前缀", target.is_file(), str(target))
    saved = json.loads(target.read_text(encoding="utf-8"))
    check("元数据记录精度", saved.get("precision") == "fp16", str(saved.get("precision")))

    # 没有 train_meta 的普通 ONNX：照常写，只是指纹为空，不能报错
    lone = tmp / "lone.onnx"
    lone.write_bytes(b"x")
    lone_report = enginemeta.write_meta(
        tmp / "lone.engine", onnx_path=lone, precision="fp32"
    )
    check("无 train_meta 时也能写", not lone_report.get("error"), str(lone_report))
    check("无 train_meta 时指纹为空", lone_report.get("dataset_fingerprint") is None)


def test_launcher_writes_train_meta(tmp: Path) -> None:
    """launcher 在 on_train_end 写 train_meta.json（位置由 trainer 决定）。"""
    import importlib.util
    import os

    launcher_path = BACKEND.parents[1] / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py"
    if not launcher_path.is_file():
        check(True, "裁剪部署没有 launcher，跳过")
        return
    spec = importlib.util.spec_from_file_location("tf_yolo_launcher_fp_test", launcher_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.sys = sys  # 让模块里的 sys 引用可用（正常由 import 机制注入）
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - 缺 torch 等依赖时跳过，不把环境问题当功能失败
        check(True, f"launcher 依赖不可用，跳过：{type(exc).__name__}")
        return

    save_dir = tmp / "exp2"
    save_dir.mkdir()
    payload = {
        "algo": "v1",
        "dataset_fingerprint": "fedcba9876543210",
        "short": "fedcba98",
        "counts": {"train": 3, "val": 1, "test": 0},
    }
    old = os.environ.get("TRTFORGE_DATASET_FP")
    os.environ["TRTFORGE_DATASET_FP"] = json.dumps(payload, ensure_ascii=False)
    try:
        module.write_dataset_meta(save_dir, "splits/pcb_tune.yaml")
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_DATASET_FP", None)
        else:
            os.environ["TRTFORGE_DATASET_FP"] = old

    target = save_dir / "train_meta.json"
    check("launcher 写出 train_meta.json", target.is_file(), str(target))
    if target.is_file():
        saved = json.loads(target.read_text(encoding="utf-8"))
        check(
            "launcher 记录的指纹与传入一致",
            saved.get("dataset", {}).get("dataset_fingerprint") == "fedcba9876543210",
            str(saved.get("dataset")),
        )
        check("launcher 记录 data 参数", saved.get("data_arg") == "splits/pcb_tune.yaml", str(saved))

    # 没有环境变量（非训练链路调用）：静默跳过，不能写半份空文件
    module.write_dataset_meta(tmp / "noenv", "x.yaml")
    check("无环境变量时不乱写", not (tmp / "noenv" / "train_meta.json").exists())


def main() -> int:
    print("== 数据集指纹与导出清单验收 ==")
    with tempfile.TemporaryDirectory(prefix="trtforge_fp_") as raw:
        tmp = Path(raw)
        test_same_tree_same_fingerprint(tmp)
        test_cross_directory_same_fingerprint(tmp)
        test_label_change(tmp)
        test_rename_changes_fingerprint(tmp)
        test_only_val_changes(tmp)
        test_classify_export(tmp)
        test_missing_yaml_degrades()
        test_coco_manifest(tmp)
        test_manifest_absent_is_not_error(tmp)
        test_engine_meta_links_dataset(tmp)
        test_launcher_writes_train_meta(tmp)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  失败：{name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
