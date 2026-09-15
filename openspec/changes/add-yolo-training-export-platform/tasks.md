## 1. 方案与原型确认
- [x] 1.1 创建本 OpenSpec 变更文档，覆盖 proposal、design、tasks、spec。
- [x] 1.2 在 `pencil-new.pen` 中新增 `TRT Forge - YOLO Training Lab` 页面原型。
- [x] 1.3 确认四种任务头 `detect`、`segment`、`pose`、`obb` 的第一版范围。
- [x] 1.4 确认 UI 参数分层：常用参数、训练高级参数、任务专属参数、验证参数、导出参数。
- [x] 1.4.1 确认训练初始化方式：加载预训练权重微调、从模型结构从零训练、从 `last.pt` 断点恢复。
- [x] 1.5 经用户确认后再进入代码实现。

## 2. Python 统一入口
- [x] 2.1 新增 `trtexec/gui/training/scripts/tf_yolo_launcher.py`，作为 C++/Qt 唯一调用入口。
- [x] 2.2 支持 `--mode check-env`，检测 Python、torch、CUDA、ultralytics、onnx、onnxruntime。
- [x] 2.3 支持 `--mode probe-model`，读取模型路径、任务头、类别与基础信息。
- [x] 2.4 支持 `--mode train`，调用 Ultralytics `YOLO(...).train(...)`。
- [x] 2.5 支持 `--mode val`，调用 Ultralytics `YOLO(...).val(...)` 并输出指标。
- [x] 2.6 支持 `--mode export`，导出原始 ONNX。
- [x] 2.7 支持 `--make-transd true`，基于原始 ONNX 生成 `.transd.onnx`。

## 3. 结构化日志协议
- [x] 3.1 定义并实现 `TRTFORGE_JSON` 输出函数。
- [x] 3.2 普通 stdout、stderr、Ultralytics 日志必须原样输出。
- [x] 3.3 训练进度输出 `progress`，至少包含 `epoch`、`total`、`pct`。
- [x] 3.4 指标输出 `metric`，覆盖 loss、precision、recall、mAP。
- [x] 3.5 产物输出 `artifact`，覆盖 `pt`、`onnx`、`transd_onnx`。
- [x] 3.6 错误输出 `error`，覆盖缺库、数据集不存在、导出失败、用户取消、未知异常。
- [x] 3.7 训练监控输出 `epoch_summary`、`plot`、`gpu_status`，每个 epoch 刷新最新数表和图表。
- [x] 3.8 训练完成输出 `labels_json` 产物，生成 `labels_<实验名称>.json`。

## 4. 参数映射
- [x] 4.1 映射常用训练参数：`task`、`model`、`data`、`epochs`、`imgsz`、`batch`、`device`、`project`、`name`。
- [x] 4.2 映射训练初始化参数：`model`、`pretrained`、`resume`，区分预训练微调、从零训练和断点恢复。
- [x] 4.3 映射高级训练参数：`patience`、`workers`、`cache`、`amp`、`optimizer`、`lr0`、`lrf`、`momentum`、`weight_decay`、`freeze`。
- [x] 4.4 映射 `segment` 专属参数：`overlap_mask`、`mask_ratio`。
- [x] 4.5 映射 `pose` 专属参数：`pose`、`kobj`、`rle`。
- [x] 4.6 映射 `obb` 专属参数：`angle`。
- [x] 4.7 映射验证参数：`split`、`conf`、`iou`、`max_det`、`half`、`plots`、`save_json`、`save_txt`。
- [x] 4.8 映射导出参数：`format`、`imgsz`、`batch`、`device`、`dynamic`、`half`、`opset`、`simplify`、`make_transd`。

## 5. Qt/C++ 后续接入
- [x] 5.1 新增 Python 环境选择与自动检测逻辑。
- [x] 5.2 新增 `YoloTrainController`，用 `QProcess` 启动 Python。
- [x] 5.3 解析 `TRTFORGE_JSON`，转换为 Qt 信号。
- [x] 5.4 普通 stdout/stderr 输出到训练日志面板。
- [x] 5.5 缺库、数据集错误、导出失败时弹窗提示，不崩溃。
- [x] 5.6 支持停止训练，先优雅中断，超时后强制结束。
- [x] 5.7 新增训练监控页签，显示指标表、最新训练图表、GPU 状态和标签 JSON 路径。
- [x] 5.7.1 最新训练图表改为深色曲线小图控件，支持双击单图放大和再次双击恢复多图。
- [x] 5.7.2 曲线小图补充坐标轴、刻度数字和网格线，混淆矩阵改为 PNG 缩略图控件。
- [x] 5.7.3 GPU 状态补充显存占用和利用率进度条，标签 JSON 显示生成状态和失败原因。
- [x] 5.8 开始训练前清空上一轮监控数据，避免旧数值混入。

## 6. 验证与文档
- [ ] 6.1 使用极小 YOLO detect 数据集验证 `check-env/train/val/export`。
- [ ] 6.2 使用极小 YOLO OBB 数据集验证 `.transd.onnx` 生成。
- [ ] 6.3 使用已有标注工作台导出的 `dataset.yaml` 验证 Ultralytics 可识别。
- [x] 6.3.1 使用短训练验证指标表、图表刷新、GPU 状态和 `labels_*.json` 生成。
- [x] 6.4 更新 `docs/Qt_Cpp_Python训练平台_架构设计.md`。
- [x] 6.5 更新 `trtexec/gui/README.md`。
- [x] 6.6 如根目录仍无 `readme.md`，补充项目总说明。
