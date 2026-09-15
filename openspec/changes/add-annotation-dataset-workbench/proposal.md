# Change: Add TRTForge Annotation Dataset Workbench

## Why
TRTForge 当前已经具备 ONNX 到 TensorRT 引擎构建、YOLO TensorRT 推理验证能力，但缺少前置的数据标注、标准格式导入导出和训练集划分工作台。为了形成面向 YOLO 检测、实例分割、姿态识别和 OBB 的工业视觉闭环，需要在 TRTForge 内新增原生 Qt 标注 UI，并支持导出可直接用于训练的数据集结构。

## What Changes
- 新增 TRTForge 标注工作台 UI，界面布局参考 X-AnyLabeling 的左侧工具栏、中间画布、右侧标签/对象/属性面板，并保持 TRTForge 暗色工业风格。
- 支持单项目单任务模式：`detect`、`segment`、`pose`、`obb`。
- 支持绘制与编辑：水平矩形、旋转矩形、多边形、线、点、COCO 17 点姿态关键点。
- 画框或绘制对象完成后弹出标注确认对话框，用户确认标签、描述、分组和数据集归属后，再同步更新标签列表、对象列表和属性描述。
- 支持数据集划分：按比例自动划分 `train` / `val` / `test`，并允许用户手动调整单张图片归属。
- 标注项目文件后缀使用 `.tfanno`，用于保存 TRTForge 标注项目的任务、类别、图片、标注和划分状态。
- 强调支持标准格式导出：
  - `YOLO`：导出 `detect`、`segment`、`pose`、`obb` 数据集，生成 Ultralytics 可用的 `dataset.yaml`；OBB 使用 Ultralytics 官方四角点归一化格式。
  - `COCO`：导出 `detect`、`segment`、`pose` 标注 JSON。
  - `VOC`：导出标准水平矩形检测 XML；非检测任务导出 VOC 时必须提示不支持。
- 支持将本工具导出的 COCO、VOC、YOLO 标注文件重新加载到 UI 中，修改标注内容后再次导出，形成可反复修订的数据闭环。
- 支持导入现有 TRTForge `labels_*.json` 类别文件，也允许在项目内手动维护类别。
- 导出训练数据集时复制图片到标准目录结构，避免训练路径依赖原始散乱目录。

## Non-Goals
- 本变更不实现 YOLO 训练启动、训练进度监控和训练日志解析。
- 本变更不优先实现自动标注；后续可复用 `yolo_infer` C API 增加预标注能力。
- 本变更不直接嵌入 X-AnyLabeling 的 PyQt6 窗口，不引入 PyQt6 到 TRTForge 主程序。
- 本变更不要求导入 X-AnyLabeling 自有 JSON 标注格式。
- 本变更不支持 VOC 的非标准扩展字段；VOC 仅作为标准目标检测格式。

## UI Approval Gate
标注 UI 开发前必须先使用 Pencil 制作 UI 原型给用户确认。当前第一版原型文件为 `pencil-new.pen`，已经包含左侧绘图工具栏、中间标注画布、右侧标签/对象/属性面板、底部数据集划分与格式导出区域，以及绘制完成后的标注确认弹窗。

## Impact
- Affected specs: `annotation-dataset-workbench`
- Affected code:
  - `trtexec/gui/src/MainWindow.cpp`
  - `trtexec/gui/include/MainWindow.h`
  - `trtexec/gui/src/InferenceTab.cpp`
  - `trtexec/gui/include/InferenceTab.h`
  - 新增 `trtexec/gui/src/annotation/` 标注画布与数据模型
  - 新增 `trtexec/gui/src/dataset/` 数据集划分与导出模块
  - 新增或扩展 `trtexec/gui/bin/configs/` 下的项目/类别配置样例

