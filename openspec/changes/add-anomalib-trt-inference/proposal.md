# Change: Add Anomalib TensorRT Inference

## Why
TRTForge 当前推理演示只面向 YOLO 检测、分割、姿态和旋转框任务，无法直接验证 Anomalib 训练后导出的 TensorRT 异常检测模型。

用户需要在同一个推理演示页中加载已有 Anomalib TensorRT engine 和部署用 `metadata.json`，查看异常分数、热力图和 mask，用于工业缺陷检测模型的快速验证。由于 Anomalib 2.x 训练默认只保存 Lightning `model.ckpt`，TRTForge 还需要在 ONNX 导出后生成一份 C++ 可读取的 metadata 契约。

## What Changes
- 在 TRTForge 推理演示中新增 `异常检测 ANOMALY` 模式。
- 支持加载已有 Anomalib TensorRT engine，不包含训练和导出。
- 支持加载 `metadata.json` 并展示/使用 `transform`、阈值、归一化范围等参数。
- 在 Anomalib 示例训练/导出链路中生成 `metadata.json`，并提供 `model.ckpt` 与 `model.onnx` 的推理对比脚本。
- 支持 EfficientAd、PaDiM、PatchCore 常见输出结构，自动根据输出数量解析 anomaly map 与 score。
- 推理输出显示热力图、mask、叠加图、异常分数和 OK/NG 状态。
- 保留图像、视频、相机输入和中文路径兼容策略。

## Impact
- Affected specs: `anomalib-trt-inference`
- Affected code:
  - `trtexec/gui/include/InferenceTab.h`
  - `trtexec/gui/src/InferenceTab.cpp`
  - `trtexec/gui/CMakeLists.txt`
  - `anomalib/anomalib-main/train.py`
  - `anomalib/anomalib-main/trtforge_metadata.py`
  - `anomalib/anomalib-main/compare_ckpt_onnx_inference.py`
  - 新增 Anomalib TRT 推理 worker 和 metadata loader
- Affected docs:
  - `readme.md`
  - `trtexec/gui/README.md`
  - `docs/trtexec_gui_YOLO_TRT推理流程说明.md`
