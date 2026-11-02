# Change: 新增 YOLO 训练、验证与 ONNX 导出平台

## Why

TRTForge 已经具备标注、数据集导出、ONNX 转 TensorRT 引擎和推理验证能力，但还缺少从 YOLO 数据集到训练权重、验证指标、ONNX 产物的统一入口。用户目前需要把 `1-python-yolo/ultralytics-main` 集成进 TRTForge，并让 Qt/C++ 能稳定调用 Python 训练流程。

训练任务耗时长、依赖复杂，若直接嵌入 Python 会增加 Qt 主程序崩溃风险。因此本变更采用方案 A：Qt 通过 `QProcess` 调用独立 Python 子进程，并预留方案 C：后续替换为本地训练服务。

## What Changes

- 新增 YOLO 训练平台规格，覆盖 `detect`、`segment`、`pose`、`obb` 四种任务头。
- 设计统一 Python 入口 `tf_yolo_launcher.py`，供 C++/Qt 调用 `check-env`、`train`、`val`、`export`、`probe-model`。
- 定义 `TRTFORGE_JSON` 结构化输出协议，同时保留普通 stdout/stderr 日志原样回显。
- 定义 Python 环境检测与缺库错误提示策略，确保缺少依赖时弹窗报错、不拖崩主程序。
- 定义训练、验证、导出参数的 UI 暴露分层，避免把 Ultralytics 全量参数平铺给用户。
- 设计 Pencil UI 原型页面 `TRT Forge - YOLO Training Lab`，作为后续 Qt 页面实现依据。
- 规定 ONNX 导出时保留原始 `.onnx`，并可生成适配 TRTForge 推理链路的 `.transd.onnx`。

## Impact

- Affected specs: `yolo-training-platform`
- Affected design files: `pencil-new.pen`
- Affected docs: `docs/Qt_Cpp_Python训练平台_架构设计.md`、`4-qt-studio/README.md`
- Affected future code:
  - `1-python-yolo/launcher/tf_yolo_launcher.py`
  - `4-qt-studio/include/*Training*.h`
  - `4-qt-studio/src/training/*`
  - `4-qt-studio/src/MainWindow.cpp`
- Constraints:
  - 不修改仓库根目录 `bin/`、`include/`、`lib/`。
  - 不直接大改 `1-python-yolo/ultralytics-main` 源码。
  - 所有脚本和文档使用 UTF-8。
