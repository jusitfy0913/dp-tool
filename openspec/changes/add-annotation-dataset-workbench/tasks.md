## 1. UI 原型与需求确认
- [x] 1.1 基于 `pencil-new.pen` 完成标注工作台 UI 评审。
- [x] 1.2 根据用户反馈调整 Pencil 原型，确认左侧工具栏、中间画布、右侧面板、底部图片/划分区域。
- [x] 1.3 确认绘制完成后弹出标注确认对话框的字段：标签、描述、分组、数据集归属、置信度。

## 2. 标注项目与数据模型
- [x] 2.1 定义 `.tfanno` TRTForge 标注项目文件结构，保存任务类型、类别、图片列表、标注对象和划分状态。
- [x] 2.2 实现单项目单任务约束：`detect`、`segment`、`pose`、`obb`。
- [x] 2.3 定义统一标注对象模型，覆盖矩形、旋转矩形、多边形、线、点和 COCO 17 点姿态。
- [x] 2.4 支持导入现有 `labels_*.json` 类别文件，并支持项目内新增、重命名、删除类别。

## 3. 标注 UI 与交互
- [x] 3.1 新增 TRTForge 标注工作台页签或模块入口。
- [x] 3.2 实现图片目录加载、图片列表、画布缩放、平移和当前图片切换。
- [x] 3.3 实现水平矩形、旋转矩形、多边形、线、点、COCO 17 点姿态绘制。
- [x] 3.4 实现标注对象选择、移动、删除、基础编辑和撤销。
- [x] 3.5 实现绘制完成后的标注确认对话框，确认后同步更新标签列表、对象列表和属性描述。

## 4. 数据集划分
- [x] 4.1 实现按比例自动划分 `train` / `val` / `test`，支持随机种子保证可复现。
- [x] 4.2 实现单张图片手动调整 `train` / `val` / `test` 归属。
- [x] 4.3 在 UI 中展示划分数量、比例和每张图片当前归属。

## 5. 标准格式导入导出
- [x] 5.1 实现 YOLO detect 导入导出。
- [x] 5.2 实现 YOLO segment 多边形导入导出。
- [x] 5.3 实现 YOLO pose COCO 17 点导入导出。
- [x] 5.4 实现 YOLO OBB 导入导出。
- [x] 5.4.1 YOLO OBB 必须使用 Ultralytics 官方四角点归一化格式：`class_index x1 y1 x2 y2 x3 y3 x4 y4`。
- [x] 5.5 实现 COCO detect、segment、pose 导入导出。
- [x] 5.6 实现 VOC 标准水平矩形检测导入导出。
- [x] 5.7 VOC 遇到 segment、pose、obb 项目时给出明确不支持提示，不生成非标准 VOC。
- [x] 5.8 导出训练集时复制图片到 `images/train`、`images/val`、`images/test`，并生成对应标签目录。
- [x] 5.9 YOLO 导出必须生成 Ultralytics 可直接使用的 `dataset.yaml`。
- [x] 5.10 支持重新加载由 TRTForge 导出的 COCO、VOC、YOLO 标注文件，在 UI 中修改后再次导出。
- [x] 5.11 明确不支持导入 X-AnyLabeling 自有 JSON 标注格式。

## 6. 验证与文档
- [x] 6.1 使用小样例验证 COCO、VOC、YOLO 导入导出结果。
    - 通过 Web 数据集工作台验证：YOLO 标签往返逐值无损、COCO 结构完整、VOC XML 数量一致
    - 脚本 `6-web/backend/tools/test_dataset_workbench.py`，20 项通过
    - 说明：本次验证的是「YOLO/.tfanno 导入 → 划分 → 导出」，标注绘制与 COCO/VOC **反向导入**仍属 Qt 版能力
- [x] 6.2 验证导出的 YOLO 数据集目录能被 Ultralytics 识别。
    - `check_det_dataset(dataset.yaml)` → nc=6、6 个类名、train/val/test 三个路径全部解析成功
- [x] 6.3 更新 TRTForge 相关 README 或 docs，说明标注 UI、项目文件、格式导出和数据集划分用法。
- [x] 6.4 记录 VOC 仅支持检测水平框的限制。

