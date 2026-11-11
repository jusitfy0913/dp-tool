## ADDED Requirements

### Requirement: 底板分组检测

系统 SHALL 提供唯一的分组实现 `6-web/backend/datasetqc.py`，供 Web 后端与
`7-autoresearch-yolo/labelqc.py` 共用。

分组 MUST 分两级判定：A 级精确重复（文件 sha256 相同）、B 级同底板分组（感知相似）。
两级的判定依据 MUST 在输出中可见。

B 级 MUST 同时满足「256 位 dHash 汉明距离 ≤ 阈值」与「32×32 灰度平均绝对差 ≤ 2.0」，
两者任一不满足则不得并组。

系统 MUST NOT 把 B 级同组图片判为重复样本，也 MUST NOT 据此删除任何文件。

「同组 + 同标注」MAY 作为提示输出，但 MUST NOT 被表述为「可删除的重复样本」：
同一块板在同一位置反复出现同类缺陷是产线常态，图像相似度无法区分它与副本。

分组算法 MUST 带版本号（当前 `dsqc-v1`）与阈值，使不同参数下的结果不可直接比较。

#### Scenario: 同底板不同缺陷不判为重复
- **WHEN** 两张图来自同一张底片、仅局部缺陷斑块不同、标注类别不同
- **THEN** 两者 MUST 进入同一 B 级分组，但 MUST NOT 出现在任何「可删除」清单里

#### Scenario: 文件副本判为精确重复
- **WHEN** 同一文件被复制成两个不同文件名
- **THEN** 两者进入同一 A 级分组，且不计入 B 级分组

#### Scenario: 全量相同数据不塌缩
- **WHEN** 一份 692 张、含 10 个底板组的 PCB 数据集被体检
- **THEN** 分组数 MUST 为 10 且每组 MUST 对应一个底板，不得出现「全部并成一组」

#### Scenario: 缓存命中不改结论
- **WHEN** 同一数据集在文件未变化时连续体检两次
- **THEN** 两次报告的指纹等价结果一致，第二次 MUST 明显更快且 `cache.hits` 大于 0

### Requirement: 跨划分泄漏警告

体检报告 MUST 指出每个分组在各 split 中的成员数量，并 MUST 单独列出跨 split 的分组。

存在跨 split 分组时，系统 MUST 给出明确警告，措辞必须说明「同一底板同时出现在训练与验证，
指标不可视为泛化能力」。

系统 MUST NOT 因发现泄漏而修改数据或阻止导出/训练。

#### Scenario: 泄漏可见
- **WHEN** 某分组同时含 train 与 val 成员
- **THEN** 报告的 `groups.cross_split` 计数包含该组，界面显示红色警告

#### Scenario: 无泄漏时不报警
- **WHEN** 每个分组只出现在一个 split 中
- **THEN** 报告 `groups.cross_split` 为 0，界面不出现泄漏警告

### Requirement: 按组切分

数据集扫描预览与导出 SHALL 支持 `group_by` 参数，取值 `none`、`prefix`、`dir`。

`prefix` MUST 取文件名第一个 `_` 之前的片段作为组键；`dir` MUST 取图片所在目录名作为组键，
目录名带 split 语义（`train` / `val` / `test` / `tune` / `holdout`）时 MUST 退回上一级目录名。

按组切分时，同一组的全部图片 MUST 进入同一个 split。比例无法精确达成时 MUST 优先保证组完整，
并在 `warnings` 中报告实际张数与目标比例的偏差。

导出 manifest MUST 记录 `split_mode.mode = "group:<mode>"` 与 `group_signature`
（组键与各组张数），使事后能判断「这次划分是按组做的」。

`group_by` 默认 MUST 为 `none`，保证既有导出行为与指纹不变。

#### Scenario: 同组不跨划分
- **WHEN** 以 `group_by=prefix` 导出含 10 个底板组的数据集
- **THEN** 每个前缀的全部图片出现在同一个 split 中

#### Scenario: 比例偏差如实报告
- **WHEN** 按组划分后 train 实际占比与请求比例相差超过 5 个百分点
- **THEN** 返回的 `warnings` 中含实际张数与偏差说明

#### Scenario: 默认行为不变
- **WHEN** 不传 `group_by` 导出
- **THEN** 划分结果与升级前一致，manifest 的 `split_mode.mode` 仍为 `ratio`

### Requirement: 标注质检

系统 SHALL 提供只读的标注质检，覆盖 YOLO 标签文件的逐行校验，并 MUST 报告行号与原始行内容。

质检 MUST 至少覆盖以下规则：标注文件缺失、字段数不符、数值不可解析、类别编号越界、
坐标越界、零面积框、极小框（面积占比 < 0.01%）、超大框（> 60%）、框贴边、
同图重复框（IoU > 0.9）、某类别在某 split 中为 0。

每条问题 MUST 带级别（`error` / `warn` / `info`）与稳定规则码，便于界面分组与后续自动化。

质检 MUST NOT 修改任何文件，也 MUST NOT 阻断扫描、导出或训练。

#### Scenario: 坏行可定位
- **WHEN** 某标签文件第 3 行只有 4 个字段
- **THEN** 报告含该文件相对路径、行号 3、原始行内容与规则码

#### Scenario: 零问题不产生噪声
- **WHEN** 数据集全部标签合法
- **THEN** `summary.error` 与 `summary.warn` 为 0，界面显示「未发现问题」

#### Scenario: 只报告不改文件
- **WHEN** 体检发现任意级别的问题
- **THEN** 标签文件 mtime 与内容 MD5 在体检前后保持一致

### Requirement: 体检接口与结果缓存

系统 SHALL 提供只读接口 `POST /api/dataset/qc`，接受数据源、任务类型、分组阈值与条数上限。

接口 MUST 一次返回分组结果与标注问题，不得要求前端发起两次请求。

实现 MUST 按「绝对路径 + 文件大小 + mtime_ns」缓存每个文件的 sha256、dHash 与缩略图，
缓存文件位于 `runtime/dsqc_cache.json`，写入失败 MUST NOT 影响报告返回。

响应 MUST 包含 `algo`、`threshold`、`summary`、`by_rule`、`groups`、`issues`、`elapsed_ms`
与 `cache` 统计。

问题列表 MUST 支持条数上限，超出时 MUST 标记 `truncated`，避免万张数据集把响应撑爆。

#### Scenario: 单次请求拿到两类结果
- **WHEN** 调用该接口
- **THEN** 响应同时含 `groups` 与 `issues`，且两者基于同一次读图

#### Scenario: 大列表截断
- **WHEN** 数据集产生 5000 条问题而 `limit` 为 500
- **THEN** `issues` 返回 500 条并标记 `truncated: true`，`by_rule` 仍为完整计数

### Requirement: 体检在界面上可见

数据集页 MUST 提供「体检」入口，结果 MUST 以分组列表与问题列表呈现。

跨划分泄漏 MUST 以醒目样式提示，并 MUST 说明「指标不可视为泛化能力」。

问题列表 MUST 按规则码分组折叠，每条显示相对路径、行号、原始行与解释。

无体检记录的数据集 MUST 显示「未体检」，不得报错，也不得伪造结果。

#### Scenario: 泄漏提示可见
- **WHEN** 体检发现 10 个分组全部跨划分
- **THEN** 页面顶部显示泄漏警告，分组列表中每个跨划分组有标记

#### Scenario: 未体检状态
- **WHEN** 用户刚打开数据集页、尚未点击体检
- **THEN** 体检区域显示「未体检」，其余扫描/导出功能正常

### Requirement: 命令行质检与实验门禁

系统 SHALL 提供 `7-autoresearch-yolo/labelqc.py`，复用同一份体检实现。

CLI MUST 支持 `--json` 输出与 `--fail-on {none,error,warn}` 退出码控制。

`autoexp.py` SHALL 提供可选的 `--qc-gate` 开关；开启时若体检存在 `error` 级问题 MUST 中止本次实验
并打印可执行的修复建议，关闭时 MUST 完全不影响既有流程。

#### Scenario: 门禁拦下坏数据
- **WHEN** 以 `--qc-gate` 运行且数据存在 error 级问题
- **THEN** 实验不启动，退出码非 0，输出含问题文件与行号

#### Scenario: 默认不拦
- **WHEN** 不带 `--qc-gate` 运行
- **THEN** 实验照常启动，不计算体检结果
