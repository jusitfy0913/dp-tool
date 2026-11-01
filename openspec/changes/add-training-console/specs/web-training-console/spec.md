## ADDED Requirements

### Requirement: 训练数据集概览

系统 SHALL 在训练页提供「扫描数据集」功能，解析用户指定的 `dataset.yaml`，按 `train` / `val` / `test` 三个 split 返回图片数、标签数、空标签数、孤儿标签数、缺失标签数、每类实例数与分辨率分布。

#### Scenario: 正常数据集

- **WHEN** 用户对已导出的 YOLO `dataset.yaml` 点击「扫描数据集」
- **THEN** 接口返回三个 split 的计数与每类实例数，`total` 等于实际图片文件数
- **AND** 前端展示计数表与每类实例条形图，不出现警告

#### Scenario: 数据集存在结构问题

- **WHEN** 数据集中存在 0 行标签文件、存在无对应图片的标签、或某个类别实例数为 0
- **THEN** 响应 `warnings` 中分别给出空标签比例、孤儿标签文件名（最多 10 个）、零实例类别名
- **AND** 前端以醒目样式展示这些警告，而不是静默忽略

#### Scenario: yaml 的 path 为相对路径

- **WHEN** `dataset.yaml` 只写了 `train: images/train` 且 `path` 为相对路径
- **THEN** 扫描以 `dataset.yaml` 所在目录为基准解析真实图片目录
- **AND** 图片目录不存在时返回可读错误而不是零计数

#### Scenario: 重复扫描

- **WHEN** 60 秒内对同一 `dataset.yaml`（mtime 与大小未变）重复点击扫描
- **THEN** 系统返回缓存结果，不重复遍历磁盘

### Requirement: 数据增强效果预览

系统 SHALL 使用用户在表单中填写的增强参数，对训练集真实样本生成一次「原图 / 增强后」对比预览，并在响应中说明本次实际触发了哪些增强项。

#### Scenario: 相同种子结果可复现

- **WHEN** 用同一 `seed`、同一参数、同一张样本图连续请求两次预览
- **THEN** 两次返回的增强后图像字节完全相同

#### Scenario: 参数参与情况可核对

- **WHEN** 用户设置 `fliplr=0`、`hsv_h=0.5`、`mosaic=0` 后请求预览
- **THEN** 响应 `applied` 中不包含 `fliplr` 与 `mosaic`
- **AND** 界面明确标注该预览是像素级近似，实际训练由 Ultralytics 随机触发

#### Scenario: 概率参数未触发

- **WHEN** `fliplr=0.5` 的某次抽样未命中
- **THEN** 图像保持左右方向不变，且 `applied` 如实不列出 `fliplr`

#### Scenario: 仅分割任务的参数

- **WHEN** 用户设置 `copy_paste > 0`
- **THEN** 预览不应用复制粘贴，界面注明「仅分割任务生效，预览不展示」

### Requirement: 断点续训入口

系统 SHALL 列出历史训练产物并允许用户从指定的 `last.pt` 继续训练，续训目标 MUST 由显式路径决定，不得依赖「最近一次 run」的自动推断。

#### Scenario: 列出可续训的历史

- **WHEN** 用户点击「扫描历史训练」
- **THEN** 返回每个 `runs/**` 训练目录的名称、任务、已完成轮数、最佳 `mAP50-95`、`last.pt` 是否存在
- **AND** 已完成的轮数来自 `results.csv`，不通过加载 `.pt` 获取

#### Scenario: 从指定检查点续训

- **WHEN** 用户对某条历史点击「继续训练」并提交
- **THEN** 后端以 `--resume-from <last.pt 绝对路径>` 启动训练
- **AND** 界面提示「epochs 等参数以检查点内 `args.yaml` 为准」

#### Scenario: 数据集已失效

- **WHEN** 某条历史的 `args.yaml` 中 `data` 指向的 `dataset.yaml` 已不存在
- **THEN** 该行标记为不可续训并禁用按钮，不发出训练请求

#### Scenario: 兼容旧调用方

- **WHEN** 调用方仍使用旧的 `--resume` 布尔开关（例如 Qt 版）
- **THEN** 行为与升级前一致，`--resume-from` 未出现时不改变任何既有参数语义
