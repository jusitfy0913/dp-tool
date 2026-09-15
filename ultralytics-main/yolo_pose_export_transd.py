# -*- coding: utf-8 -*-
"""YOLOv8-pose: 导出 ONNX 并生成 .transd.onnx，专供 yolo_infer (TensorRT) 使用。

背景：
- Ultralytics 默认导出的 yolov8s-pose.onnx 输出形状为 [1, 56, 8400]
  （channels-first：4(box) + 1(score) + 17*3(kpts) = 56）。
- yolo_infer 的 GPU 解码 kernel 假设布局是 [1, anchors, channels]，
  因此必须在 ONNX 末尾追加一个 Transpose(perm=[0,2,1])，
  得到 [1, 8400, 56] 后再转 TensorRT engine。

复用 yolo_segement.py 中的 apply_onnx_last_output_transpose。
.transd.onnx 不能用 Ultralytics 的 YOLO() 直接推理（后处理布局不匹配），
仅给 trtexec / yolo_infer 等按该布局实现的引擎使用。

用法：
  python yolo_pose_export_transd.py [pt_weights]
  默认 pt_weights = "yolov8s-pose.pt"
"""

import os
import sys

from ultralytics import YOLO

from yolo_segement import apply_onnx_last_output_transpose


def main(pt_weights: str = "yolov8s-pose.pt") -> None:
    if not os.path.exists(pt_weights):
        print(f"[info] {pt_weights} 不存在，将由 ultralytics 自动下载。")

    model = YOLO(pt_weights)
    exported = model.export(format="onnx")
    onnx_path = str(exported) if exported else None
    if not onnx_path or not os.path.isfile(onnx_path):
        raise RuntimeError("ONNX 导出失败")

    print(f"[ok] 原始 ONNX: {onnx_path}")
    transd = apply_onnx_last_output_transpose(onnx_path)
    print(f"[ok] transd ONNX: {transd}")
    print()
    print("下一步（用 trtexec 转 engine）示例：")
    print(f"  trtexec --onnx={transd} "
          f"--saveEngine={os.path.splitext(transd)[0]}.engine --fp16")


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8s-pose.pt"
    main(weights)
