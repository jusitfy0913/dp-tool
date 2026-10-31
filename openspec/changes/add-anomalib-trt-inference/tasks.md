## 1. Specification
- [x] 1.1 完成 Pencil 原型：异常检测模式、metadata 参数区、热力图和 mask 输出区
- [x] 1.2 创建 OpenSpec proposal/design/spec/tasks
- [x] 1.3 通过 `openspec validate add-anomalib-trt-inference --strict`

## 2. Metadata And Inference Core
- [x] 2.1 新增 `AnomalyInferWorker` 相关头文件和实现文件
- [x] 2.2 实现 `metadata.json` 读取，兼容扁平字段和嵌套字段
- [x] 2.3 实现 TensorRT engine 反序列化、IO tensor 枚举和显存分配
- [x] 2.4 实现 EfficientAd / PaDiM / PatchCore 输出解析、归一化、heatmap、mask 和 overlay 生成

## 3. Inference UI
- [x] 3.1 扩展 `InferenceTab`，新增 `异常检测 ANOMALY` 任务模式
- [x] 3.2 增加 metadata 路径选择、加载状态、阈值/归一化参数显示
- [x] 3.3 异常检测模式下隐藏 labels.json/NMS，显示 image threshold / pixel threshold
- [x] 3.4 接入 `AnomalyInferWorker`，显示 score、OK/NG、热力图/mask/叠加图

## 4. Build And Documentation
- [x] 4.1 更新 `CMakeLists.txt` 加入新增文件
- [x] 4.2 更新 `readme.md`、`4-qt-studio/README.md` 和推理流程文档
- [x] 4.3 使用可用样例或编译检查验证改动
- [x] 4.4 新增 `anomalib_infer.dll` 子工程，导出 `ai_*` C API
- [x] 4.5 将 `AnomalyInferWorker` 改为调用 `anomalib_infer.dll` 的 Qt 薄适配层
- [x] 4.6 增加外部 C++ 和 Python ctypes 调用示例，并说明结果内存生命周期

## 5. Anomalib Export Metadata And Comparison
- [x] 5.1 在 `anomalib-main/train.py` 的 ONNX 导出后生成 `metadata.json`
- [x] 5.2 新增 `trtforge_metadata.py`，统一 metadata 写入、ONNX 预处理和输出解析
- [x] 5.3 在 `anomalib-main` 根目录新增 `compare_ckpt_onnx_inference.py`
- [x] 5.4 TRTForge metadata loader 兼容 `input_size` 新字段并优化缺失提示
- [x] 5.5 运行 `compare_ckpt_onnx_inference.py` 完成三模型样例验证（PatchCore 81.1% / PaDiM 89.0% / EfficientAd 82.7% 判定一致）。修复了脚本三个真实缺陷：① `import anomalib` 落到 user-site 1.2.0 而非仓库内 2.0.0b3（ckpt 由后者训练，导致分数恒为 0.5）→ 改为优先注入 `src` 路径；② 新版 `Engine.predict` 返回 `list[dict]` 且键为复数（`pred_scores`/`anomaly_maps`），原代码用 `getattr` 取单数属性，全部拿到 `None`；③ 新增 `_print_summary` 解释「分数差呈近常数偏移 = 两侧阈值基准不同，判定以 metadata 阈值为准」。报告见 `anomalib-main/compare_outputs/`
