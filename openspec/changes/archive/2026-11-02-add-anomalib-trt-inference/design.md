## Context
现有 TRTForge 推理演示通过 `CaptureWorker` 采集图像/视频/相机帧，再交给 `InferWorker` 调用 YOLO TensorRT C API，最后在 `InferenceTab` 叠加检测结果。

Anomalib TensorRT 推理与 YOLO 最大区别是：
- 不输出检测框类别，而是输出 anomaly map 和/或 anomaly score。
- 后处理强依赖 `metadata.json` 中的阈值、归一化范围和输入尺寸；Anomalib 2.x 训练默认只保存 Lightning `model.ckpt`，不会自动在 checkpoint 旁边生成 TRTForge 可读取的 metadata。
- EfficientAd、PaDiM、PatchCore 的输出头数量和语义不同。

## Goals / Non-Goals
- Goals:
  - 在推理演示页新增异常检测模式。
  - 使用已有 TensorRT engine 和 `metadata.json` 完成单帧推理。
  - 输出热力图、mask、叠加图、异常分数和 OK/NG 状态。
  - 对中文路径保持兼容。
  - 在示例训练/ONNX 导出后生成部署 metadata，并提供 ckpt/onnx 推理对比脚本。
  - 将 Anomalib TensorRT 推理核心封装为独立 `anomalib_infer.dll`，供 TRTForge GUI、外部 C++ 和 Python ctypes 调用。
- Non-Goals:
  - 不实现 Anomalib 训练。
  - 不实现 ONNX 导出或 engine 构建。
  - 不修改 YOLO 推理 DLL 的接口契约。

## Decisions
- 新增独立 `AnomalyInferWorker`，不把异常检测逻辑塞进 YOLO `InferWorker`。
  - 原因：YOLO 输出是 boxes，Anomalib 输出是 map/score，数据结构和 UI 展示完全不同。
- 新增独立 `anomalib_infer` 动态库，不把异常检测逻辑塞进 `yolo_infer`。
  - 原因：YOLO 输出是 boxes，Anomalib 输出是 map/score/heatmap/mask，C API 契约不同；独立 DLL 更适合外部 C++/Python 复用。
- `AnomalyInferWorker` 保留为 Qt 薄适配层。
  - 原因：GUI 仍需要 `QObject`、`QThread`、`QImage` 和信号槽，但 TensorRT Runtime、metadata、预处理、后处理和可视化由 DLL 统一维护。
- `anomalib_infer.dll` 使用不依赖 Qt 的 C API 读取 `metadata.json`。
  - 原因：外部 C++/Python 调用不应额外依赖 Qt；Windows 中文路径由 DLL 内部按 UTF-8 转文件系统路径读取。
  - 兼容字段：嵌套字段 `pred_scores.min/max`、`anomaly_maps.min/max`，以及实际导出中常见的扁平键 `pred_scores.min`、`anomaly_maps.max`。
- 在 `anomalib-main/train.py` 的 ONNX 导出后写出 `weights/lightning/metadata.json`。
  - 原因：`model.ckpt` 是 Python/Lightning 状态文件，C++/TensorRT 不应读取它来推断阈值。
  - metadata 同时写 `transform.height/width`、`input_size`、嵌套范围和 dotted-key 范围，保证 Anomalib 脚本与 TRTForge C++ 读取同一套参数。
- 新增 `compare_ckpt_onnx_inference.py` 放在 `anomalib-main` 根目录。
  - 原因：训练用户可以在同目录直接对 `model.ckpt` 和 `model.onnx` 做性能与一致性验证，再进入 TensorRT 构建和 TRTForge 推理。
- 在 `anomalib_infer.dll` 内直接使用 TensorRT Runtime 反序列化 engine。
  - 原因：让 Anomalib 推理像 YOLO 一样拥有独立动态库边界，并支持外部 C++/Python 程序调用。
- C API 结果内存由 DLL 句柄持有。
  - 原因：避免跨 DLL 分配/释放内存不一致；调用方必须在下一次 `ai_infer_rgb` 或 `ai_destroy` 前深拷贝结果。
- 输出解析采用保守策略：
  - 1 个输出：输出即 anomaly map，score 取 map 最大值。
  - 2 个输出：第 1 个按 score，第 2 个按 anomaly map。
  - 3 个及以上输出：按 EfficientAd 规则优先取第 3 个输出作为 anomaly map，score 取 map 最大值。

## Risks / Trade-offs
- 不同 Anomalib 版本的 metadata 字段层级可能不同。
  - Mitigation: loader 同时兼容扁平字段和 `anomaly_maps` / `pred_scores` 嵌套字段，缺失时给出明确 UI 错误。
- 部分 metadata 不包含 `transform.height/width`。
  - Mitigation: 静态 engine 回填自身输入尺寸；动态 engine 暂按 `256×256` 兜底，并在 UI 文档中说明。
- engine 输入名、输出名可能不同。
  - Mitigation: 运行时遍历 TensorRT IO tensor，根据输入/输出模式识别，不依赖固定名称。
- EfficientAd 预处理可能不使用 ImageNet mean/std。
  - Mitigation: UI 暴露模型类型，默认 EfficientAd 不做 mean/std，PaDiM/PatchCore 做 ImageNet mean/std。
- C API 被外部程序调用时可能误用结果指针生命周期。
  - Mitigation: 在 `anomalib_infer_c.h`、README、C++ 示例和 Python ctypes 示例中明确“DLL 持有，调用方立即深拷贝”的规则。

## Validation Plan
- 使用 EfficientAd 样例 engine 和 metadata 验证：
  - metadata 能加载并显示参数。
  - 图像输入能输出热力图、mask、叠加图和异常分数。
  - score 大于 image threshold 时 UI 显示 NG，否则显示 OK。
- 对中文路径的 engine、metadata、图像进行加载验证。
- 使用外部 C++ 示例调用 `anomalib_infer.dll`，保存 overlay/mask 图片。
- 使用 Python ctypes 示例调用 `anomalib_infer.dll`，保存 overlay/mask 图片。
