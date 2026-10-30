## 1. Design Approval

- [x] 1.1 更新 `pencil-new.pen` 训练导出页原型，增加 Anomalib 异常检测训练模式与参数区。
- [x] 1.2 编写 `proposal.md`，说明新增 Anomalib 训练导出的原因、范围和影响。
- [x] 1.3 编写 `design.md`，明确 Python launcher、TRTFORGE_JSON、metadata 和 Qt UI 设计。
- [x] 1.4 编写 `specs/anomalib-training-export/spec.md`，覆盖需求与场景。
- [x] 1.5 运行 `openspec validate add-anomalib-training-export --strict` 并修复问题。
- [x] 1.6 已进入等待用户确认设计的实现闸门，确认前不进入 Qt/C++ 与 Python 实现。

## 2. Implementation

- [x] 2.1 新增 `2-python-anomaly/launcher/tf_anomalib_launcher.py`。
- [x] 2.2 实现 `check-env`，输出 Python、torch、CUDA、anomalib、lightning、onnx、onnxruntime、PIL、numpy 状态。
- [x] 2.3 实现 `train-export`，支持 PatchCore、PaDiM、EfficientAd 的 Folder 数据集训练参数。
- [x] 2.4 复用 `2-python-anomaly/anomalib-main/trtforge_metadata.py` 的 metadata 生成逻辑。
- [x] 2.5 在 `TrainingTab` 增加 YOLO / Anomalib 模式切换与 Anomalib 参数控件。
- [x] 2.6 拆分 `TrainingTab` 参数组装逻辑为 `baseYoloArgs()` 与 `baseAnomalibArgs()`。
- [x] 2.7 让控制器按训练类型选择 `tf_yolo_launcher.py` 或 `tf_anomalib_launcher.py`。
- [x] 2.8 解析 `anomaly_model`、`anomaly_export_summary`、`artifact(kind=metadata)` 等事件。
- [x] 2.9 保存与恢复 Anomalib 参数到独立 `QSettings` 键。
- [x] 2.10 更新 `4-qt-studio/README.md` 与根目录 `readme.md`。
- [x] 2.11 用小型 Folder 数据集验证 PaDiM 命令行训练导出。
- [ ] 2.12 用生成的 ONNX 和 metadata 在 TRTForge 异常检测推理页验证 OK/NG。

## 3. Verification

- [x] 3.1 `tf_anomalib_launcher.py --mode check-env` 输出环境状态。
- [x] 3.2 `tf_anomalib_launcher.py --mode train-export --model padim` 成功生成 `model.ckpt`、`model.onnx`、`metadata.json`。
- [x] 3.3 `cmake --build --preset vs2022-x64 -j 8` 编译通过。
