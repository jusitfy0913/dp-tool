# -*- coding: utf-8 -*-
"""`.transd.onnx` 生成逻辑（四种任务头）验收脚本，不需要 torch / GPU / DLL。

守的是三类「编译能过、推理才炸」的静默故障：

1. **转错张量**：旧实现取 `graph.node[-1]`，图末尾多一个节点就会转错，
   而生成的 ONNX 依然能编译成功，只是结果全错。现在按形状识别预测头。
2. **分割输出顺序**：C++ 侧 `yolo.cu` 对 V8Seg 硬性要求 binding1=proto、
   binding2=检测头；Ultralytics 导出的顺序正好相反，必须重排。
3. **任务与权重不匹配**：通道数不对时在导出阶段就报错，不要留到推理时报
   "V8Pose expects channels=56, got 116"。

用法：
    python 6-web/backend/tools/test_transd_onnx.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py"

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
EVENTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def load_launcher():
    """按路径加载 launcher（文件名不是合法模块名，只能走 spec）。"""
    spec = importlib.util.spec_from_file_location("tf_yolo_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # emit 会往 stdout 打 TRTFORGE_JSON，测试里改成收集，顺便断言事件内容
    module.emit = lambda event: EVENTS.append(dict(event))
    return module


def _constant_node(name: str, dims: list[int]) -> tuple:
    count = 1
    for d in dims:
        count *= d
    tensor = helper.make_tensor(f"{name}_value", TensorProto.FLOAT, dims, [0.0] * count)
    return helper.make_node("Constant", [], [name], value=tensor)


def build_onnx(
    path: Path,
    head_dims: list[int],
    proto_dims: list[int] | None = None,
    dead_node: bool = False,
) -> Path:
    """造一个最小图：images -> output0（以及可选的 output1=proto）。

    `dead_node=True` 时在末尾追加一个不接任何图输出的节点，用来证明
    「最后一个节点」不等于「预测头」。
    """
    nodes = [_constant_node("output0", head_dims)]
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, head_dims)]
    if proto_dims is not None:
        nodes.append(_constant_node("output1", proto_dims))
        outputs.append(helper.make_tensor_value_info("output1", TensorProto.FLOAT, proto_dims))
    if dead_node:
        nodes.append(_constant_node("dead_end", [1, 2, 3]))

    graph = helper.make_graph(
        nodes,
        "synthetic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def dims_of(output) -> list:
    out = []
    for d in output.type.tensor_type.shape.dim:
        out.append(int(d.dim_value) if d.HasField("dim_value") else None)
    return out


def count_transpose(model) -> int:
    return sum(1 for n in model.graph.node if n.op_type == "Transpose")


def main() -> int:
    launcher = load_launcher()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ---- 1. detect：基本转置 -------------------------------------------
        det = build_onnx(root / "det.onnx", [1, 84, 100])
        out = Path(launcher.make_transd_onnx(str(det), "detect"))
        model = onnx.load(str(out))
        check("生成的文件名带 .transd", out.name == "det.transd.onnx", out.name)
        check("detect 输出转成 [1, anchors, channels]", dims_of(model.graph.output[0]) == [1, 100, 84], str(dims_of(model.graph.output[0])))
        check("只追加一个 Transpose", count_transpose(model) == 1, str(count_transpose(model)))
        trans = [n for n in model.graph.node if n.op_type == "Transpose"][0]
        check("Transpose 的 perm 为 [0,2,1]", list(trans.attribute[0].ints) == [0, 2, 1], str(list(trans.attribute[0].ints)))
        check("Transpose 写出原输出名", list(trans.output) == ["output0"], str(list(trans.output)))
        # 原文件必须保持原布局（Ultralytics 自己还要用）
        original = onnx.load(str(det))
        check("原始 ONNX 未被改动", dims_of(original.graph.output[0]) == [1, 84, 100], str(dims_of(original.graph.output[0])))
        check("转置事件带 task 与输出形状", EVENTS[-1].get("name") == "transd_written" and EVENTS[-1].get("task") == "detect", str(EVENTS[-1]))

        # ---- 2. 预测头不是最后一个节点 -------------------------------------
        late = build_onnx(root / "late.onnx", [1, 84, 100], dead_node=True)
        out = Path(launcher.make_transd_onnx(str(late), "detect"))
        model = onnx.load(str(out))
        check("末尾有无关节点时仍转置预测头", dims_of(model.graph.output[0]) == [1, 100, 84], str(dims_of(model.graph.output[0])))

        # ---- 3. pose：56 通道照常转 ----------------------------------------
        pose = build_onnx(root / "pose.onnx", [1, 56, 100])
        out = Path(launcher.make_transd_onnx(str(pose), "pose"))
        model = onnx.load(str(out))
        check("pose 输出转成 [1, 100, 56]", dims_of(model.graph.output[0]) == [1, 100, 56], str(dims_of(model.graph.output[0])))

        # ---- 4. pose 通道数不对必须报错 ------------------------------------
        wrong = build_onnx(root / "wrong_pose.onnx", [1, 84, 100])
        try:
            launcher.make_transd_onnx(str(wrong), "pose")
            check("pose 通道数不对时报错", False, "没有抛异常")
        except RuntimeError as exc:
            check("pose 通道数不对时报错", "56" in str(exc), str(exc))

        # ---- 5. obb：通道过少必须报错 --------------------------------------
        tiny = build_onnx(root / "tiny_obb.onnx", [1, 4, 100])
        try:
            launcher.make_transd_onnx(str(tiny), "obb")
            check("obb 通道过少时报错", False, "没有抛异常")
        except RuntimeError as exc:
            check("obb 通道过少时报错", "6" in str(exc), str(exc))

        # ---- 6. segment：转置检测头 + proto 提到第一位 ----------------------
        seg = build_onnx(root / "seg.onnx", [1, 116, 100], proto_dims=[1, 32, 10, 10])
        out = Path(launcher.make_transd_onnx(str(seg), "segment"))
        model = onnx.load(str(out))
        names = [o.name for o in model.graph.output]
        check("segment 输出顺序变成 [proto, det]", names == ["output1", "output0"], str(names))
        check("segment proto 保持 4 维不变", dims_of(model.graph.output[0]) == [1, 32, 10, 10], str(dims_of(model.graph.output[0])))
        check("segment 检测头被转置", dims_of(model.graph.output[1]) == [1, 100, 116], str(dims_of(model.graph.output[1])))

        # ---- 7. segment 缺 proto 必须报错 ----------------------------------
        no_proto = build_onnx(root / "no_proto.onnx", [1, 116, 100])
        try:
            launcher.make_transd_onnx(str(no_proto), "segment")
            check("segment 缺 proto 时报错", False, "没有抛异常")
        except RuntimeError as exc:
            check("segment 缺 proto 时报错", "proto" in str(exc), str(exc))

        # ---- 8. 动态维：仍能转置，且保留 dim_param（不能丢掉 batch 动态性）----
        dyn = helper.make_model(
            helper.make_graph(
                [_constant_node("output0", [1, 84, 100])],
                "dyn",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
                [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 84, "anchors"])],
            ),
            opset_imports=[helper.make_opsetid("", 12)],
        )
        dyn.ir_version = 8
        dyn_path = root / "dyn.onnx"
        onnx.save(dyn, str(dyn_path))
        try:
            out = Path(launcher.make_transd_onnx(str(dyn_path), "detect"))
            model = onnx.load(str(out))
            dims = model.graph.output[0].type.tensor_type.shape.dim
            names = [d.dim_param if not d.HasField("dim_value") else d.dim_value for d in dims]
            check("动态维仍能转置且保留符号维度", names == ["batch", "anchors", 84], str(names))
        except Exception as exc:  # noqa: BLE001
            check("动态维仍能转置且保留符号维度", False, f"{type(exc).__name__}: {exc}")

        # ---- 9. 文件不存在时给出明确异常 -----------------------------------
        try:
            launcher.make_transd_onnx(str(root / "nope.onnx"), "detect")
            check("文件不存在时报错", False, "没有抛异常")
        except FileNotFoundError:
            check("文件不存在时报错", True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
