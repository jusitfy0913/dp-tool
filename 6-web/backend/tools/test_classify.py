# -*- coding: utf-8 -*-
"""YOLO 分类：整图类别的保存、导出和分数解释。不启动训练，也不加载真实模型。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import annotation  # noqa: E402
import annotationimport  # noqa: E402
import classifyinfer  # noqa: E402
import dataset  # noqa: E402
import traininfo  # noqa: E402

LAUNCHER_PATH = Path(__file__).resolve().parents[3] / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py"


def _load_launcher():
    """按路径加载 launcher。它是脚本不是包，不能 import 名字。"""
    spec = importlib.util.spec_from_file_location("tf_yolo_launcher_under_test", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  通过  {name}")
    else:
        FAILED.append(name)
        print(f"  失败  {name}  {detail}")


def main() -> int:
    print("分类工程")
    project = annotation.new_project(
        "classify",
        [annotation.new_class(0, "良品"), annotation.new_class(1, "不良")],
        [
            annotation.new_image("a", "a.jpg", class_id=0),
            annotation.new_image("b", "b.jpg", class_id=1),
            annotation.new_image("c", "c.jpg"),
        ],
    )
    text = annotation.dumps(project)
    loaded, warnings = annotation.normalize_project(json.loads(text))
    check("任务保持 classify", loaded["task"] == "classify", loaded["task"])
    check("未标注不写成 0", loaded["images"][2].get("class_id") is None, str(loaded["images"][2]))
    check("已标注类别还在", loaded["images"][0].get("class_id") == 0 and loaded["images"][1].get("class_id") == 1)
    check("读回没有警告", not warnings, str(warnings))

    shaped = json.loads(text)
    shaped["images"][0]["annotations"] = [{"id": "box", "class_id": 0, "shape_type": "rectangle", "points": [[0, 0], [2, 2]]}]
    try:
        annotation.normalize_project(shaped)
        check("分类工程拒绝画框", False, "没有拒绝")
    except annotation.AnnotationError as exc:
        check("分类工程拒绝画框", "不能保存画框" in str(exc), str(exc))

    print("分类导出")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (root / name).write_bytes(b"not-a-real-image")
        images = [
            annotation.new_image("a", str(root / "a.jpg"), class_id=0, split="train"),
            annotation.new_image("b", str(root / "b.jpg"), class_id=1, split="val"),
            annotation.new_image("c", str(root / "c.jpg"), split="train"),
        ]
        project = annotation.new_project(
            "classify",
            [annotation.new_class(0, "良品"), annotation.new_class(1, "不良")],
            images,
        )
        ds = annotation.to_dataset_project(project, source=str(root / "job.tfanno"))
        out = root / "out"
        result = dataset.export_yolo(ds, out, [img.current_split for img in ds.images], True)
        check("良品目录有图", (out / "train" / "良品" / "a.jpg").is_file())
        check("不良目录有图", (out / "val" / "不良" / "b.jpg").is_file())
        check("未标注不导出", not (out / "train" / "c.jpg").exists() and result["skipped_unlabeled"] == 1, str(result["skipped"]))
        check("没有检测标签目录", not (out / "labels").exists())
        yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8")
        check("yaml 标明分类", "task: classify" in yaml_text and "train: train" in yaml_text, yaml_text)

        bad = annotation.new_project("classify", [annotation.new_class(0, "良/品")], images[:1])
        bad_ds = annotation.to_dataset_project(bad, source=str(root / "job.tfanno"))
        try:
            dataset.export_yolo(bad_ds, root / "bad", ["train"], True)
            check("非法类别名被拒绝", False, "没有拒绝")
        except dataset.DatasetError as exc:
            check("非法类别名被拒绝", "文件夹" in str(exc), str(exc))

        try:
            dataset.export_coco(ds, root / "coco", ["train", "val", "train"], True)
            check("分类拒绝 COCO", False, "没有拒绝")
        except dataset.DatasetError as exc:
            check("分类拒绝 COCO", "COCO" in str(exc), str(exc))

    print("分层划分")
    labels = [0, 0, 0, 0, 1]
    splits, notes = dataset.assign_splits_stratified(labels, 50, 50)
    check("两个类别都进了划分", set(splits) <= {"train", "val", "test"} and "train" in splits and "val" in splits)
    check("少样本类别写明分不出来", any("类别 1" in note and "分不出" in note for note in notes), str(notes))

    print("分数")
    verdict = classifyinfer.interpret([0.1, 0.9], ["良品", "不良"], 0.5)
    check("取得分最高的类别", verdict["label"] == "不良" and not verdict["uncertain"], str(verdict))
    low = classifyinfer.interpret([0.4, 0.6], ["良品", "不良"], 0.8)
    check("低于阈值标不确定", low["uncertain"] and low["label"] == "不良", str(low))
    logits = classifyinfer.as_probabilities([1.0, 3.0])
    check("logits 会变成概率", abs(sum(logits) - 1) < 1e-6 and logits[1] > logits[0], str(logits))
    try:
        classifyinfer.interpret([0.2, 0.8], ["只有一个对不上"], 0.5)
        check("类别数对不上要拒绝", False, "没有拒绝")
    except classifyinfer.ClassifyError as exc:
        check("类别数对不上要拒绝", "对不上" in str(exc), str(exc))
    try:
        classifyinfer.reject_engine("best.engine")
        check("引擎路径被拒绝", False, "没有拒绝")
    except classifyinfer.ClassifyError as exc:
        check("引擎路径被拒绝", "分类引擎还没做" in str(exc), str(exc))

    print("DLL 分数封装")
    # v1.30.6：分类走 onnx_infer.dll 时，C 侧只回原始分数 + metadata names，
    # 组装成前端结构的工作在 result_from_raw 里。这组用例不加载真实 DLL，
    # 只验证"同一份分数在两条路径上得到同一个结论"。
    names_text = "{0: 'alpha', 1: 'beta', 2: 'gamma'}"
    check("metadata dict 解析", classifyinfer.parse_names(names_text) == ["alpha", "beta", "gamma"])
    check("metadata list 解析", classifyinfer.parse_names("['a', 'b']") == ["a", "b"])
    check("metadata 坏数据给空表", classifyinfer.parse_names("not a dict") == [])
    check("metadata 空串给空表", classifyinfer.parse_names("") == [])

    raw = {
        "backend": "onnx_infer.dll",
        "scores": [0.1, 0.7, 0.2],
        "names": names_text,
        "timing": {"total_ms": 5.0, "inference_ms": 3.0},
        "image_width": 100,
        "image_height": 50,
    }
    result, warning = classifyinfer.result_from_raw(raw, ["x", "y", "z"], 0.5)
    check("模型 names 覆盖标签文件", [item["name"] for item in result["classify"]["scores"]] == ["alpha", "beta", "gamma"])
    check("顺序不一致给提示", "顺序与模型不一致" in warning, warning)
    check("top1 与分数一致", result["classify"]["class_id"] == 1 and abs(result["classify"]["score"] - 0.7) < 1e-6, str(result["classify"]))
    check("timing 原样透传", result["timing"]["total_ms"] == 5.0, str(result["timing"]))
    check("DLL 后端名保留", result["backend"] == "onnx_infer.dll", result["backend"])
    check("分类结果不带框", result["boxes"] == [] and result["count"] == 0)

    no_names = dict(raw, names="", scores=[0.2, 0.3, 0.5])
    result, warning = classifyinfer.result_from_raw(no_names, ["x", "y", "z"], 0.5)
    check("没有 names 时用标签文件", result["classify"]["label"] == "z", str(result["classify"]))
    check("没有 names 不给顺序提示", warning == "", warning)

    try:
        classifyinfer.result_from_raw(dict(raw, names="", scores=[0.5, 0.5]), ["x", "y", "z"], 0.5)
        check("分数与标签数量不符要拒绝", False, "没有拒绝")
    except classifyinfer.ClassifyError as exc:
        check("分数与标签数量不符要拒绝", "对不上" in str(exc), str(exc))

    print("分类导入")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        detect = root / "detect"
        (detect / "images").mkdir(parents=True)
        (detect / "labels").mkdir()
        try:
            annotationimport.build(str(detect), task="classify")
            check("检测 labels 不能导入分类", False, "没有拒绝")
        except annotationimport.AnnotationImportError as exc:
            check("检测 labels 不能导入分类", "检测数据集" in str(exc), str(exc))

    print("分类数据集概览")
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "train" / "良品"
        bad = root / "train" / "不良"
        good.mkdir(parents=True)
        bad.mkdir(parents=True)
        import cv2

        def _write_jpg(path: Path, height: int, width: int) -> None:
            # Windows 上 cv2.imwrite 处理不了含中文的路径（良品 / 不良），会静默失败，
            # 所以先编码再自己写字节。
            ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
            if not ok:
                raise RuntimeError("测试图片编码失败")
            path.write_bytes(buffer.tobytes())

        _write_jpg(good / "a.jpg", 150, 200)
        _write_jpg(bad / "b.jpg", 160, 160)
        _write_jpg(bad / "c.jpg", 160, 160)
        yaml = root / "dataset.yaml"
        yaml.write_text("path: .\ntrain: train\nval: val\ntask: classify\nnames:\n  0: 良品\n  1: 不良\n", encoding="utf-8")
        overview = traininfo.scan(yaml, use_cache=False)
        by_name = {item["name"]: overview["class_instances"].get(str(item["id"]), 0) for item in overview["classes"]}
        check("概览按图片计数", by_name.get("良品") == 1 and by_name.get("不良") == 2, str(overview["class_instances"]))
        check("概览不报缺标签", not any(item["code"] in {"label_missing", "label_empty"} for item in overview["warnings"]), str(overview["warnings"]))
        check("概览能读到分辨率", overview["resolutions"]["count"] == 3, str(overview["resolutions"]))

        print("分类训练的数据路径")
        launcher = _load_launcher()
        resolved = launcher.resolve_classify_data(str(yaml))
        check("yaml 换成它描述的目录", Path(resolved) == root.resolve(), resolved)
        check("目录原样返回", launcher.resolve_classify_data(str(root)) == str(root))
        absolute = root / "abs.yaml"
        absolute.write_text(f"path: {root.as_posix()}\ntrain: train\n", encoding="utf-8")
        check("绝对 path 也能解析", Path(launcher.resolve_classify_data(str(absolute))) == root.resolve())

        broken = root / "broken"
        broken.mkdir()
        broken_yaml = broken / "dataset.yaml"
        broken_yaml.write_text("path: .\ntrain: train\n", encoding="utf-8")
        try:
            launcher.resolve_classify_data(str(broken_yaml))
            check("缺少 train 目录要报错", False, "没有报错")
        except ValueError as exc:
            check("缺少 train 目录要报错", "没有 train 目录" in str(exc), str(exc))

        renamed_yaml = root / "renamed.yaml"
        renamed_yaml.write_text("path: .\ntrain: images/train\n", encoding="utf-8")
        try:
            launcher.resolve_classify_data(str(renamed_yaml))
            check("train 不是 train 要报错", False, "没有报错")
        except ValueError as exc:
            check("train 不是 train 要报错", "train 写的是" in str(exc), str(exc))

    print("分类预处理")
    wide = np.zeros((480, 640, 3), dtype=np.uint8)
    wide[:, :10] = 255   # 只在最左侧 10 列点亮
    blob = classifyinfer.preprocess(wide, 224, 224)
    check("预处理输出 NCHW", blob.shape == (1, 3, 224, 224), str(blob.shape))
    check("预处理裁掉两侧而不是压扁", float(blob.mean()) < 0.005, f"mean={float(blob.mean()):.4f}")
    square = np.full((320, 320, 3), 128, dtype=np.uint8)
    square_blob = classifyinfer.preprocess(square, 160, 160)
    check("正方形图直接缩放", square_blob.shape == (1, 3, 160, 160), str(square_blob.shape))

    class _Meta:
        custom_metadata_map = {"names": "{0: '不良', 1: '良品'}"}

    class _Session:
        def get_modelmeta(self):
            return _Meta()

    check("从 ONNX metadata 读类别顺序", classifyinfer._model_names(_Session()) == ["不良", "良品"], str(classifyinfer._model_names(_Session())))

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
