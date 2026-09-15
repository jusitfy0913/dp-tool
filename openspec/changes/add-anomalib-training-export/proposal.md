# Change: 新增 Anomalib 异常检测训练导出

## Why

TRTForge 已经具备 YOLO 训练导出、ONNX 转 TensorRT 和 Anomalib TensorRT 推理能力，但异常检测模型仍需要用户手工运行 `anomalib/anomalib-main/train.py`、整理 `model.onnx` 与 `metadata.json`。这会让 PatchCore、PaDiM、EfficientAd 的训练导出链路分散，且容易再次出现 metadata 阈值契约不一致的问题。

本变更在现有训练导出页新增 `Anomalib 异常检测` 模式，沿用 `QProcess + Python launcher + TRTFORGE_JSON`，让用户在 Qt 界面里完成环境检测、Folder 数据集配置、训练、ONNX 导出和 metadata 生成。

## What Changes

- 在 `pencil-new.pen` 增加 Anomalib 训练导出原型页，展示训练类型、Folder 数据集、模型参数、导出产物和验证闭环。
- 新增规格 `anomalib-training-export`，覆盖 PatchCore、PaDiM、EfficientAd 的训练、导出、metadata 生成和 UI 行为。
- 设计 Python 入口 `tf_anomalib_launcher.py`，支持 `check-env` 与 `train-export` 两种模式，后续可扩展 `compare`。
- 定义 Anomalib 专用 `TRTFORGE_JSON` 事件，包括环境、训练进度、GPU 状态、异常模型摘要、导出摘要、产物和错误。
- 设计 `TrainingTab` 模式切换、参数控件显隐、按钮策略和 `QSettings` 保存策略。
- 保留现有 YOLO `detect / segment / pose / obb` 训练导出能力，不修改既有 YOLO 参数语义。

## Impact

- Affected specs: `anomalib-training-export`
- Affected design files: `pencil-new.pen`
- Affected future code:
  - `trtexec/gui/training/scripts/tf_anomalib_launcher.py`
  - `trtexec/gui/include/TrainingTab.h`
  - `trtexec/gui/src/training/TrainingTab.cpp`
  - `trtexec/gui/include/YoloTrainController.h`
  - `trtexec/gui/src/training/YoloTrainController.cpp`
  - `anomalib/anomalib-main/trtforge_metadata.py`
- Affected docs:
  - `trtexec/gui/README.md`
  - `readme.md`
- Constraints:
  - 不修改仓库根目录 `bin/`、`include/`、`lib/`。
  - 不把 Python 嵌入 Qt 主进程。
  - 不直接破坏现有 YOLO 训练导出流程。
  - 所有脚本和文档使用 UTF-8。
  - 设计确认前不进入 Qt/C++ 与 Python 实现。
