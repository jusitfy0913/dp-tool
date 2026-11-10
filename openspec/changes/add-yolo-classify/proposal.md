# Change: YOLO 整图分类（标注、导出、训练、ONNX 推理）

## Why

现在的标注、导出和推理都是为「图上画框」准备的。新建标注工程被写成检测，训练只能选检测、分割、姿态、旋转框，推理结果是一张框表。现场要做的「整张图是良品还是不良」用这套框流程做不到：分类没有框，Ultralytics 认的是 `train/类别名/图片.jpg`，不是 `labels/*.txt`。

## What Changes

- **标注页**：工程可以选择任务「分类」。分类工程不显示画框工具。点一个类别就标完整张图，并跳到下一张没标的图。一张图只有一个类别，可以改、可以清空。数字键 1、2、3 对应前几个类别。没标的图不导出。
- **导入**：可以从已经按类别分好的文件夹读入整图标签。检测用的 `labels/*.txt` 不能导进分类工程。
- **数据集页**：任务增加「分类」。导出写成 `train/<类别>/图片`、`val/<类别>/图片`、`test/<类别>/图片`，并写一份分类用的 `dataset.yaml`。划分按类别分层，避免某一类全部进了训练。COCO、VOC 对分类不可用。
- **训练导出页**：任务增加 `classify`。开训前拒绝用检测权重去训分类。数据集概览显示每个类别多少张图，不再数框。分类验证给出准确率和混淆矩阵，不报检测用的 mAP、漏检、过检。
- **全流程**：分类可以训练并导出 ONNX。转置检测布局、编译 TensorRT 引擎、检测精度验证这三段跳过，页面写明原因。
- **推理演示页**：任务增加「分类」。单张图不画框，显示判定类别、分数和其余类别的分数。目录批量每张图一行，可导出 CSV。视频和摄像头每一帧一个类别。选 TensorRT 引擎时拒绝并说明分类引擎还没做。推理读 ONNX 的类别分数，不走现有检测动态库。
- `APP_VERSION` 1.29.0 → 1.30.0。
- **BREAKING**：无。检测、分割、姿态、旋转框的标注、txt 导出、训练和画框推理保持原样。已有 `.tfanno` 不改任务字段。

## Impact

- Affected specs:
  - `web-annotation-workbench`（分类标注交互）
  - `web-studio`（分类导出不再要求标签目录）
  - `yolo-training-platform`（launcher 接受 `classify`）
  - `web-training-console`（分类数据集概览）
  - `web-yolo-pipeline`（分类全流程停在 ONNX）
  - `web-infer-console`（整图分类结果）
  - `web-infer-batch`（批量、视频、摄像头的分类结果）
- Affected code:
  - `6-web/backend/annotation.py`、`dataset.py`、`app.py`
  - `6-web/frontend/index.html`、`js/annotate.js`、`js/dataset.js`、`js/pipeline.js`、`js/infer.js`、`js/inferbatch.js`
  - `1-python-yolo/launcher/tf_yolo_launcher.py`
  - `6-web/README.md`、根 `readme.md`、`docs/WORKFLOW_GUIDE.md`
- 不改动:
  - 引擎构建页
  - `3-cpp-tensorrt/yolo_infer` 与 `onnx_infer` 的检测解码
  - 仓库根目录 `bin` / `include` / `lib`
  - Anomalib、OCR
  - Qt 标注台界面（`4-qt-studio`）。桌面版若打开 `task=classify` 的工程，不得把它改写成检测后再存回去
- 非目标（Non-Goals）:
  - 不做分类 TensorRT 引擎，也不改引擎构建页
  - 不把分类并进 Anomalib 的 OK/NG 分数
  - 分类工程不做检测预标注（点、框、文本提示生成草稿框）
  - 不改检测、分割、姿态、旋转框的画框和 txt 导出
