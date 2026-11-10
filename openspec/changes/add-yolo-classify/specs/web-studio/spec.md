## MODIFIED Requirements

### Requirement: Annotation Workflow Preserved

系统 SHALL 提供数据集标注与导出能力，且 SHALL 保持 `train / val / test` 比例划分规则。检测、分割、姿态、旋转框的 `dataset.yaml` 字段格式保持不变。分类导出使用分类目录和只含路径与类别名的 `dataset.yaml`。

#### Scenario: 自动划分

- **WHEN** 用户设置合计为 100% 的划分比例并触发自动划分
- **THEN** 系统 SHALL 按该比例分配图片归属

#### Scenario: 比例非法

- **WHEN** 用户设置的划分比例合计不等于 100%
- **THEN** 系统 SHALL 拒绝执行并提示具体合计值

#### Scenario: 导出 YOLO 数据集

- **WHEN** 用户导出检测、分割、姿态或旋转框工程
- **THEN** 系统 SHALL 生成图片目录、标签目录与 `dataset.yaml`

## ADDED Requirements

### Requirement: 分类数据集按类别文件夹导出

系统 SHALL 把分类工程导出为 Ultralytics 分类目录，而不是 `labels/*.txt`。

#### Scenario: 导出目录结构

- **WHEN** 用户导出一个含「良品」「不良」两类且图片已划分的分类工程
- **THEN** 输出必须包含 `train/良品/`、`train/不良/`、`val/良品/`、`val/不良/` 中实际有图的目录，以及 `dataset.yaml`
- **AND** `dataset.yaml` 的 `names` 必须与工程类别顺序一致
- **AND** 不得生成 `labels/*.txt`

#### Scenario: 未标注图片

- **WHEN** 工程中有图片没有类别
- **THEN** 这些图片不得被复制到导出目录
- **AND** 导出结果必须给出跳过张数

#### Scenario: 按类别分层划分

- **WHEN** 用户对分类工程做自动划分且比例合计为 100%
- **THEN** 系统必须在每个类别内部按该比例分配图片
- **AND** 某个类别图片少到无法分进三个子集时，允许 val 或 test 中没有该类别，并在结果中说明

#### Scenario: 非法类别名

- **WHEN** 某个类别名包含路径分隔符或盘符
- **THEN** 导出必须拒绝并指出该类别名，不得创建对应目录

#### Scenario: 分类不支持 COCO 与 VOC

- **WHEN** 用户对分类工程请求 COCO 或 VOC 导出
- **THEN** 系统必须拒绝并说明分类只导出类别文件夹

#### Scenario: 从类别文件夹导入

- **WHEN** 用户选择已经按类别分好的图片目录导入到分类工程
- **THEN** 每张图片的类别必须等于它所在文件夹的名字
- **AND** 检测标签 txt 不得被读成分类工程的标注

#### Scenario: 扫描显示张数

- **WHEN** 用户在数据集页扫描分类数据源
- **THEN** 预览必须给出每个类别的图片张数，不得把缺少 txt 显示成标注缺失
