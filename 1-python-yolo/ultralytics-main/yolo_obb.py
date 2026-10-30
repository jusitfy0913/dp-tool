# -*- coding: utf-8 -*-
"""YOLOv8-OBB 推理、导出 ONNX，并额外生成 .transd.onnx（最后输出追加 Transpose(0,2,1)）。

说明：
- 原始导出的 .onnx 会照常保留，Ultralytics 后处理依赖该布局，因此可视化对比仍使用原始 .onnx。
- .transd.onnx 仅供 TensorRT 等按 [batch, anchors, channels] 布局实现的自定义引擎使用，
  不要再用 YOLO() 加载它进行推理（会导致维度/后处理错误）。
- 流程与 yolo_pose.py 保持一致。
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")  # 须先于 ultralytics：避免被切换为 Agg 后端导致弹窗失败

import matplotlib.pyplot as plt
import onnx
import onnx.helper as helper
from PIL import Image
from ultralytics import YOLO

# 测试图片：默认取仓库内 5-reference 对照工程，可用 TRTFORGE_OBB_IMAGE 环境变量覆盖
_REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = os.environ.get("TRTFORGE_OBB_IMAGE", "").strip() or str(
    _REPO_ROOT / "5-reference" / "YOLOv8_obb" / "normal" / "plane2.jpg"
)
WEIGHTS = "yolov8n-obb.pt"


def apply_onnx_last_output_transpose(onnx_path: str) -> str:
    """将最后一层输出改名为 pre_transpose，再追加 Transpose(perm=[0,2,1])。

    与 yolo_pose.py / workspace/transform.py 一致：
    例如 yolov8n-obb.onnx -> yolov8n-obb.transd.onnx
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)

    prefix, suffix = os.path.splitext(onnx_path)
    dst = prefix + ".transd" + suffix

    model = onnx.load(onnx_path)
    node = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0],
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return dst


def _show_yolo_result_window(results, title: str, window_title: str, save_path: Optional[str]):
    """单张图结果：一个独立 Figure 窗口；r.plot() 为 BGR，显示前转 RGB。"""
    if not results:
        return
    r = results[0]
    im_bgr = r.plot()
    im_rgb = im_bgr[..., ::-1]

    fig = plt.figure(figsize=(10, 8))
    try:
        mgr = fig.canvas.manager
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(window_title)
    except Exception:
        pass
    plt.imshow(im_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        Image.fromarray(im_rgb).save(save_path)


# 1) PyTorch 推理
model_pt = YOLO(WEIGHTS)
results_pt = model_pt(IMAGE_PATH)

# 2) 导出 ONNX，并额外生成 .transd.onnx（原 onnx 同时保留）
exported = model_pt.export(format="onnx")
onnx_path = str(exported) if exported else None
transd_onnx_path = None
if onnx_path:
    transd_onnx_path = apply_onnx_last_output_transpose(onnx_path)
    print(
        "已写入 .transd.onnx，仅供自定义/TensorRT 使用；"
        "Ultralytics 对比推理使用未改动的导出 ONNX。"
    )

# 3) 使用未改动的导出 ONNX 与 PT 同图对比
results_onnx = None
if onnx_path and os.path.isfile(onnx_path):
    model_onnx = YOLO(onnx_path)
    results_onnx = model_onnx(IMAGE_PATH)

_show_yolo_result_window(
    results_pt,
    title="PyTorch (.pt) — OBB",
    window_title="YOLOv8-OBB — PyTorch (.pt)",
    save_path="results_obb_pt.jpg",
)
if results_onnx is not None:
    _show_yolo_result_window(
        results_onnx,
        title="ONNX (export, 未 transpose) — OBB",
        window_title="YOLOv8-OBB — ONNX (export)",
        save_path="results_obb_onnx.jpg",
    )

plt.show(block=True)
