## Context

Web Studio 的标注权威格式是 UTF-8 JSON `.tfanno`。`annotation.py` 的任务只认 `detect` / `segment` / `pose` / `obb`，不认识的值会退回检测。标注页新建工程写死 `task: "detect"`。导出走 `dataset.py`，检测一行五个数：`类别 中心x 中心y 宽 高`。

Ultralytics 分类认的是文件夹：

```text
dataset/
  train/良品/a.jpg
  train/不良/b.jpg
  val/良品/c.jpg
  val/不良/d.jpg
  dataset.yaml
```

`dataset.yaml` 只需要 `path`、`train`、`val`、`names`。没有 `labels/*.txt`。

分类模型的 ONNX 输出是一整排类别分数，不是检测框。`yolo_infer` 和 `onnx_infer` 只会解框，所以这一版推理用 Python 读 ONNX 分数，不进这两份动态库。

## Goals / Non-Goals

- Goals:
  - 分类工程里，一张图只保存一个类别，界面用点选完成，不画框
  - 导出成 Ultralytics 能直接训练的分类目录
  - 训练任务增加 `classify`，验证用准确率和混淆矩阵
  - 推理页对一张图给出一个类别和各类分数
  - 检测四类任务的现有行为不变
- Non-Goals:
  - 分类 TensorRT 引擎、INT8、引擎构建页
  - Qt 标注台的分类界面
  - 用检测动态库解码分类输出
  - 分类工程的框式预标注

## Decisions

- Decision: `.tfanno` 用 `task=classify`。每张图增加 `class_id`（整数或空）。空表示没标。不使用形状列表存分类标签。
  - 理由：和检测框分开存，打开分类工程时不会画出一个不存在的框。
- Decision: 分类工程隐藏画框工具、框表和预标注入口。类别列表上的名字可以点，点中即写入 `class_id` 并跳到下一张 `class_id` 为空的图。
  - 理由：分类没有形状。保留画框会让人以为还要框住缺陷。
- Decision: 导出复制图片到 `train|val|test/<类别名>/`，并写 `dataset.yaml`。类别名里不能有路径分隔符。没标的图跳过，并在结果里写明跳过张数。
  - 理由：这是 Ultralytics 分类的目录约定。复制而不是移动，原图还在。
- Decision: 自动划分按类别分层。每个类别内部再按 train/val/test 比例抽，比例仍必须合计 100%。
  - 理由：随机抽整库时，少的那一类可能全部进训练，验证里没有它。
- Decision: `tf_yolo_launcher.py` 的 `SUPPORTED_TASKS` 增加 `classify`。权重文件名或模型任务头不是分类时，开训前失败，错误码沿用 `UNSUPPORTED_TASK` 或单独的可读错误，说明要换 `yolov8n-cls.pt` 这类分类权重。
- Decision: 分类导出 ONNX 不生成检测用的 `.transd.onnx`。全流程在导出 ONNX 后结束，`probe`、`build`、`val` 标为跳过，原因写在阶段上。
  - 理由：转置和引擎验证都假定输出是检测框。
- Decision: 分类验证指标是 top1 准确率、每类正确数和 `类别数 × 类别数` 的混淆矩阵。没有背景行、没有漏检和过检。
- Decision: 推理只接受 `.onnx`。请求 `.engine` 时返回 400，说明分类引擎还没做。分数用 ONNX 输出做 softmax（输出已经是概率时不再二次 softmax，实现时以和为 1 的容差判断）。低于置信度阈值时类别旁显示「不确定」，仍给出最高分类别。
- Decision: 目录批量的 CSV 用 `utf-8-sig`，列至少包括文件名、类别、分数、是否不确定。不要求标注图上画框。视频和摄像头每帧只叠一个类别名和分数。

## Risks / Trade-offs

- 桌面版不认识 `classify` → 实现前先看 Qt 打开 `.tfanno` 时会不会把未知任务写成 `detect` 并保存。若会，这一版至少要让桌面版对 `classify` 拒绝打开或原样保留，不能改写后存盘。
- 分类 ONNX 的输出布局各版本不完全一样 → 推理前检查输出最后一维等于类别数。对不上就返回可读错误，不把向量硬当成框。
- 类别名含 `\`、`/`、`:` 会变成非法目录 → 导出前拒绝并指出哪个名字。
- 图片很多时按类别复制占磁盘 → 与现在导出检测数据集一样，接受复制。不在这一版做硬链接。

## Migration Plan

- 已有工程的 `task` 不是 `classify` 时，读写和导出与现在相同。
- 新工程默认仍是检测。只有用户选了分类才走新路径。
- 回退：去掉分类入口后，旧的检测工程不受影响。已经导出的分类目录可以留在磁盘上。

## Open Questions

- 无。类别数量不限，良品 / 不良只是示例名字。分类引擎留到后续变更。
