## MODIFIED Requirements

### Requirement: 训练数据集概览

系统 SHALL 在训练页提供「扫描数据集」功能，解析用户指定的 `dataset.yaml`。检测、分割、姿态、旋转框按 `train` / `val` / `test` 返回图片数、标签数、空标签数、孤儿标签数、缺失标签数、每类实例数与分辨率分布。分类数据集按类别返回图片张数。

#### Scenario: 正常数据集

- **WHEN** 用户对已导出的检测、分割、姿态或旋转框 `dataset.yaml` 点击「扫描数据集」
- **THEN** 接口返回三个 split 的计数与每类实例数，`total` 等于实际图片文件数
- **AND** 前端展示计数表与每类实例条形图，不出现警告

#### Scenario: 数据集存在结构问题

- **WHEN** 检测、分割、姿态或旋转框数据集中存在 0 行标签文件、存在无对应图片的标签、或某个类别实例数为 0
- **THEN** 响应 `warnings` 中分别给出空标签比例、孤儿标签文件名（最多 10 个）、零实例类别名
- **AND** 前端以醒目样式展示这些警告，而不是静默忽略

#### Scenario: yaml 的 path 为相对路径

- **WHEN** `dataset.yaml` 只写了 `train: images/train` 且 `path` 为相对路径
- **THEN** 扫描以 `dataset.yaml` 所在目录为基准解析真实图片目录
- **AND** 图片目录不存在时返回可读错误而不是零计数

#### Scenario: 重复扫描

- **WHEN** 60 秒内对同一 `dataset.yaml`（mtime 与大小未变）重复点击扫描
- **THEN** 系统返回缓存结果，不重复遍历磁盘

#### Scenario: 分类数据集

- **WHEN** 用户扫描分类 `dataset.yaml`
- **THEN** 接口必须返回每个 split 里每个类别的图片张数
- **AND** 不得因为没有 `labels/*.txt` 而报告空标签、孤儿标签或缺失标签
