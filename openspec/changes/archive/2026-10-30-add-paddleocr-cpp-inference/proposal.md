# Change: Add PaddleOCR C++ Inference

## Why
TRTForge 当前推理演示已经支持 YOLO TensorRT 与 Anomalib TensorRT，但还不能直接演示 PaddleOCR C++ OCR 推理。用户需要在推理演示页中选择 OCR 任务，加载 PaddleOCR 检测/识别模型，查看文本框、识别文本、分数和原始多边形数据。

## What Changes
- 新增独立 `ocr_infer.dll`，封装 PaddleOCR C++ Paddle Inference 推理能力，TRTForge 主程序只通过 C API 调用。
- 在推理演示页新增 `OCR 文本识别 OCR` 任务类型，提供 OCR 模型路径、设备、线程和阈值配置。
- OCR 推理结果同时返回可视化图片和原始数据：每个文本行的序号、多边形点、识别文本、识别分数、文本行方向角必须一一对应。
- 默认使用 `4-qt-studio/bin/paddle_inference` 作为 Paddle Inference 依赖根目录，检测模型示例目录为 `4-qt-studio/bin/paddle_inference/ch_PP-OCRv4_det_infer`。
- 使用 `ocr/PaddleOCR-main/docs/datasets/images` 作为基础测试图片根目录。
- 更新 README 和 docs，说明 OCR DLL C API、参数、模型文件要求、测试流程和与旧版本差异。

## Impact
- Affected specs: `paddleocr-cpp-inference`
- Affected code:
  - `4-qt-studio/CMakeLists.txt`
  - `4-qt-studio/include/InferenceTab.h`
  - `4-qt-studio/src/InferenceTab.cpp`
  - 新增 `ocr/ocr_infer/`
  - 新增 `4-qt-studio/include/OcrInferWorker.h`
  - 新增 `4-qt-studio/src/OcrInferWorker.cpp`
  - 新增 `4-qt-studio/include/inference/OcrInferenceAdapter.h`
  - 新增 `4-qt-studio/src/inference/OcrInferenceAdapter.cpp`
- Affected docs:
  - `readme.md`
  - `4-qt-studio/README.md`
  - `docs/TRTForge_操作手册.md`
  - `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`
