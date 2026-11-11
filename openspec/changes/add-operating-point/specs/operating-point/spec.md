## ADDED Requirements

### Requirement: 良品图声明与统计

系统 SHALL 允许为数据集显式声明「良品目录」，目录内的图片 MUST 被视为良品（OK）图。

系统 MUST NOT 把「没有标签文件」的图片自动当作良品：该状态 MUST 单独计数并在报告中提示
可能是漏标。

良品目录内若存在非空标注文件，系统 MUST 报错并拒绝按良品处理——否则「良品」这个词
会被静默污染。

良品图 MUST 在导出时写成空标签文件，且默认 MUST 只进入验收划分（`val` 或 `test`），
不参与训练。良品图数量与目标划分 MUST 记入导出 `manifest.json`。

#### Scenario: 良品目录被识别
- **WHEN** 用户在扫描请求里给出 `ok_dirs=["ok"]` 且该目录下 50 张图都没有标注
- **THEN** 报告 `ok.images` 为 50，导出后这 50 张全在 `val` 且对应 `.txt` 为空文件

#### Scenario: 良品目录里有标注就报错
- **WHEN** 声明为良品的目录里有一张图存在非空标签
- **THEN** 接口返回错误并指明该文件，不产生任何导出

#### Scenario: 漏标不算良品
- **WHEN** 数据集中有 3 张图没有标签文件，但用户没有把它们所在目录声明为良品
- **THEN** 报告的 `missing_label_images` 为 3，`ok.images` 不包含它们

### Requirement: 漏检-过杀曲线

系统 SHALL 提供 `operating_points()`，在给定 IoU 下按 conf 阈值扫描，输出每个阈值点的
混淆计数与派生指标。

每个点 MUST 至少含：`conf`、`tp`、`fp`、`fn`、`recall`、`miss_rate`、`precision`、`f1`、
`fp_per_image`、`ok_fp_rate` 与逐类明细。

`miss_rate` MUST 定义为 `1 - recall`；`fp_per_image` MUST 定义为 `FP / 评估图片数`；
`ok_fp_rate` MUST 定义为「被判出至少一个框的良品图数 / 良品图数」。

`ok_fp_rate` 在良品图数为 0 时 MUST 为 `null` 而不是 `0`——「没有良品图」与
「良品全部通过」必须是不同的结论。

系统 SHALL 提供「满足目标漏检率的最高 conf」反查（过杀最低的点）；无可行点时 MUST 返回
`feasible: false` 并说明最接近点，不得伪造一个满足条件的点。

#### Scenario: 曲线与单点指标自洽
- **WHEN** 曲线中某个 conf 等于报告使用的 conf
- **THEN** 该点的 `tp/fp/fn` 与 `confusion_at` 在同一 conf 下的结果一致

#### Scenario: 目标漏检反查
- **WHEN** 以目标漏检率 1% 反查且存在满足条件的点
- **THEN** 返回满足条件的最高 conf（即过杀最低的点），并附该点的 `fp_per_image` 与 `ok_fp_rate`

#### Scenario: 无可行点
- **WHEN** 所有阈值点的漏检率都高于目标
- **THEN** 返回 `feasible: false` 与最接近目标的点，不返回「已满足」

#### Scenario: 没有良品图
- **WHEN** 评估集里没有良品图
- **THEN** 每个点的 `ok_fp_rate` 为 `null`，界面显示「无良品图」而不是 0%

### Requirement: 工作点保存与推理默认值

系统 SHALL 允许把选定工作点（conf、IoU）保存到 `<engine>.meta.json` 的
`operating_point` 字段。

保存 MUST NOT 修改 `metadata.json` 或引擎文件，MUST NOT 影响既有推理结果，
直到用户在推理页使用该默认值。

推理页选择引擎时，若旁挂文件含 `operating_point.conf`，MUST 预填 conf 并显示来源与
保存时间；用户手动改过之后 MUST 以用户输入为准。

#### Scenario: 保存后推理页预填
- **WHEN** 引擎旁挂文件含 `operating_point.conf = 0.42`
- **THEN** 选中该引擎时推理页 conf 输入框为 0.42，并显示「来自引擎工作点」

#### Scenario: 旧引擎不受影响
- **WHEN** 引擎旁挂文件不存在或没有 `operating_point`
- **THEN** 推理页 conf 使用原默认值，不报错

### Requirement: 自动实验的过杀列与 KEEP 约束

`results.tsv` 与 holdout 台账 SHALL 增加 `fp_per_image` 列，旧文件按列名迁移。

自动实验 KEEP 判定 MUST 在候选与现任基线都有 `fp_per_image` 时检查过杀：
候选的过杀率相对基线恶化超过容差（`max(0.01, 基线 × 5%)`）时，
即使 mAP 提升超过噪声阈值，也 MUST NOT 判 KEEP，并 MUST 在理由中写明两者数值。

该值 MAY 由 P/R 与 GT 数量反推，但文档 MUST 说明它对应 best-F1 conf、且不区分良品图。

#### Scenario: 过杀恶化拒绝 KEEP
- **WHEN** 候选 mAP 提升 0.02（超过 2σ）但 fp_per_image 由 0.10 升到 0.30
- **THEN** 结论为 DISCARD，理由含两者的过杀率数值

#### Scenario: 容差内照常判定
- **WHEN** 候选 fp_per_image 由 0.100 升到 0.102
- **THEN** 容差内不拦截，按原 2σ 规则判定

#### Scenario: 没有过杀数据不拦
- **WHEN** 基线或候选的 `fp_per_image` 为空
- **THEN** 判定与升级前完全一致
