## ADDED Requirements

### Requirement: Batch Auto Labeling
系统 SHALL 支持对整个标注工程做批量预标注，把推理结果按图片写入草稿池，
让人的工作从"逐张触发"降为"只做审核"。

#### Scenario: 整工程一键预标注
- **WHEN** 用户在已打开的工程上触发批量预标注
- **THEN** 系统 SHALL 对工程内的每张图片执行一次预标注，并把结果按图片 id 写入草稿池

#### Scenario: 进度可见
- **WHEN** 批量预标注正在进行
- **THEN** 界面 SHALL 显示已完成数 / 总数、当前图片与累计草稿数，且 MUST NOT 阻塞画布交互

#### Scenario: 可中途停止
- **WHEN** 用户在批量执行期间点击停止
- **THEN** 系统 SHALL 在当前图片处理完成后停止，已生成的草稿 SHALL 保留

#### Scenario: 跳过已完成图片
- **WHEN** 用户选择"只跑还没有标注或草稿的图片"
- **THEN** 系统 SHALL 跳过这些图片并单独汇报跳过数量

#### Scenario: 单张失败不中断
- **WHEN** 批量过程中某张图片推理失败
- **THEN** 系统 SHALL 记录该图片的错误并继续处理其余图片，整批 MUST NOT 因一张失败而终止

#### Scenario: 不覆盖正式标注
- **WHEN** 批量预标注写入某张图片的草稿
- **THEN** 该图片已有的正式标注 MUST NOT 被修改或删除

### Requirement: Training Artifact Engine Picker
标注页 SHALL 直接列出训练产物中的可用引擎，免去手工翻目录填路径。

#### Scenario: 下拉列出训练产物
- **WHEN** 用户展开预标注参数或点击刷新
- **THEN** 系统 SHALL 从训练产物扫描结果中列出 YOLO 引擎，至少包含名称、体积与是否已转置

#### Scenario: 选中即回填
- **WHEN** 用户在下拉中选中一个引擎
- **THEN** 引擎路径输入框 SHALL 被回填，且配套标签文件存在时一并回填

#### Scenario: 保留手工输入
- **WHEN** 训练产物扫描不到任何引擎，或用户要用的引擎不在列表里
- **THEN** 用户 SHALL 仍能手工填写路径或用文件对话框选择，两条通道均不得被禁用

### Requirement: Draft Persistence
预标注草稿 SHALL 持久化到工程旁的旁挂文件 `<工程名>.drafts.json`，
且 MUST NOT 写入 `.tfanno` 工程文件本身。

#### Scenario: 刷新后可恢复
- **WHEN** 用户在草稿未处理完时刷新页面并重新打开同一个工程
- **THEN** 草稿池 SHALL 还原为刷新前的内容，包括每张图片的草稿与置信度

#### Scenario: 不污染工程文件
- **WHEN** 草稿被持久化
- **THEN** `.tfanno` 工程文件的内容 SHALL 保持不变，Qt 版打开该工程 SHALL NOT 看到任何草稿

#### Scenario: 增量写入
- **WHEN** 用户只处理了某一张图片的草稿
- **THEN** 系统 SHALL 只提交该图片的草稿，其他图片在磁盘上的草稿 MUST NOT 被清空

#### Scenario: 旁挂文件损坏
- **WHEN** 旁挂文件不是合法 JSON 或某个图片条目结构异常
- **THEN** 系统 SHALL 返回可用的空草稿池或跳过坏条目并给出警告，接口 MUST NOT 报 500

### Requirement: Draft Review Shortcuts
草稿审核 SHALL 提供键盘快捷键，并给出可发现的按钮入口。

#### Scenario: 接受当前草稿
- **WHEN** 用户在标注页按下 `A`
- **THEN** 当前草稿 SHALL 转为正式标注；若没有选中任何草稿，SHALL 取置信度最高的一条

#### Scenario: 丢弃当前草稿
- **WHEN** 用户在标注页按下 `D`
- **THEN** 当前草稿 SHALL 被丢弃，其余草稿不受影响

#### Scenario: 下一张图片
- **WHEN** 用户按下空格
- **THEN** 系统 SHALL 切换到下一张图片；已是最后一张时给出提示且 MUST NOT 循环回第一张

#### Scenario: 不干扰输入
- **WHEN** 焦点在文本框、数字框或下拉框内
- **THEN** 上述快捷键 MUST NOT 触发任何标注行为

### Requirement: Draft Confidence Tiers
草稿 SHALL 按置信度分档着色，让高置信度与低置信度的结果在画布上一眼可辨。

#### Scenario: 三档阈值
- **WHEN** 草稿置信度大于等于 `0.70`
- **THEN** 其边框与标签 SHALL 使用绿色；`0.40` 至 `0.70` 使用黄色；低于 `0.40` 使用红色

#### Scenario: 列表同步分档
- **WHEN** 草稿出现在右侧列表
- **THEN** 列表项 SHALL 使用与画布相同的分档颜色并带"高/中/低"标识

#### Scenario: 无置信度
- **WHEN** 草稿来自不返回置信度的通道（`score < 0`）
- **THEN** 系统 SHALL 按"未知"呈现，MUST NOT 将其误判为低置信度红色
