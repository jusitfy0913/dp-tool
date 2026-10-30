## 1. Specification And Pencil
- [x] 1.1 完成 Pencil OCR 推理演示窗口设计
- [x] 1.2 创建 OpenSpec proposal/design/spec/tasks
- [x] 1.3 通过 `openspec validate add-paddleocr-cpp-inference --strict`

## 2. OCR Inference DLL
- [x] 2.1 新增 `ocr/ocr_infer` 子工程
- [x] 2.2 定义 `ocr_infer_c.h` C API 和结果内存生命周期
- [x] 2.3 接入 PaddleOCR C++ 推理源码与 Paddle Inference / OpenCV / third_party 依赖
- [x] 2.4 实现 `oi_create`、`oi_infer_file`、`oi_infer_bgr`、`oi_destroy`、`oi_last_error`
- [x] 2.5 将文本行多边形、文本、分数、角度一一对应地写入 `oi_result`
- [x] 2.6 添加 DLL 级 C++ 示例或验证入口

## 3. TRTForge GUI Integration
- [x] 3.1 新增 `OcrInferWorker` 和 `OcrInferenceAdapter`
- [x] 3.2 扩展 `InferenceTab`，新增 `OCR 文本识别 OCR` 任务类型
- [x] 3.3 增加 OCR 模型路径、设备、线程和阈值配置控件
- [x] 3.4 接入 OCR Worker 加载和帧推理分支
- [x] 3.5 显示 OCR 框、序号、文本、分数和多边形坐标列表
- [x] 3.6 支持点击列表行高亮对应 OCR 多边形

## 4. Build And Runtime Packaging
- [x] 4.1 更新 `4-qt-studio/CMakeLists.txt` 添加 `ocr_infer`
- [x] 4.2 配置 Paddle Inference include/lib 路径，默认指向 `4-qt-studio/bin/paddle_inference`
- [x] 4.3 核验并记录 `paddle_inference.lib`、运行时 DLL、MKL/oneDNN 依赖
- [x] 4.4 确认 `ocr_infer.dll` 输出到 `4-qt-studio/bin`
- [x] 4.5 将 OCR 子工程默认从 TRTForge 主解决方案拆出，主工程默认导入 `bin/ocr_infer.dll/lib`

## 5. Documentation And Validation
- [x] 5.1 更新 `readme.md`
- [x] 5.2 更新 `4-qt-studio/README.md`
- [x] 5.3 更新 `docs/TRTForge_操作手册.md`
- [x] 5.4 更新 `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`
- [x] 5.5 使用 `ocr/PaddleOCR-main/docs/datasets/images/CDLA_demo` 做测试图片验证（`val_0633.jpg` 55 行、`val_0941.jpg` 83 行）
- [x] 5.6 验证 OCR 原始数据契约：文本、多边形、分数一一对应（`ocr_basic` 输出与 `raw_json` 中 `rec_texts`/`rec_scores`/`rec_polys` 长度一致）
