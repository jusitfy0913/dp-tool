## ADDED Requirements

### Requirement: Prompt Based Auto Labeling
系统 SHALL 支持用点、框、文本三种提示直接生成预标注草稿，且 MUST NOT 要求用户
先训练出自有引擎。

#### Scenario: 点提示生成多边形
- **WHEN** 用户在标注页放置至少一个正点后触发提示式预标注
- **THEN** 系统 SHALL 用分割模型生成多边形草稿，并把点集映射回原图坐标系

#### Scenario: 框提示生成多边形
- **WHEN** 用户拖出一个框后触发提示式预标注
- **THEN** 系统 SHALL 以该框的左上与右下两点作为框提示，生成多边形草稿

#### Scenario: 负点排除区域
- **WHEN** 用户在同一张图上同时放置正点与负点
- **THEN** 系统 SHALL 把负点作为排除提示送入模型，结果 MUST NOT 覆盖负点所在区域

#### Scenario: 无提示时不执行
- **WHEN** 用户没有任何点或框却触发提示式预标注
- **THEN** 系统 SHALL 给出提示并 MUST NOT 发起推理请求

#### Scenario: 掩码转多边形有上限
- **WHEN** 模型输出的掩码轮廓点数超过上限
- **THEN** 系统 SHALL 做轮廓近似把点数压到上限以内，且 MUST 保留闭合多边形语义

#### Scenario: 提示式草稿不声明置信度
- **WHEN** 草稿来自提示式分割通道
- **THEN** 其置信度 SHALL 标记为未知（负值），MUST NOT 用分割模型的内部打分冒充检测置信度

### Requirement: Open Vocabulary Text Labeling
系统 SHALL 支持用一段文字提示直接检出目标，且 MUST NOT 依赖训练时的固定类别表。

#### Scenario: 文本提示出框
- **WHEN** 用户输入文本提示并触发文本预标注
- **THEN** 系统 SHALL 返回与该提示匹配的矩形草稿，`label` 使用模型解出的短语

#### Scenario: 空提示不执行
- **WHEN** 文本提示为空
- **THEN** 系统 SHALL 给出提示并 MUST NOT 发起推理请求

#### Scenario: 两个阈值分别生效
- **WHEN** 用户调整框阈值与文本阈值
- **THEN** 两者 SHALL 分别作用于框得分与短语得分，低于对应阈值的候选 MUST NOT 出现在草稿里

#### Scenario: 可选细化为多边形
- **WHEN** 用户勾选"细化为多边形"
- **THEN** 系统 SHALL 对每个检出框调用一次分割模型生成多边形，且同一张图的分割编码 MUST 只做一次

#### Scenario: 模型未就绪时显式提示
- **WHEN** 文本通道所需的模型文件不存在
- **THEN** 系统 SHALL 明确提示需要先下载哪个模型，MUST NOT 静默失败或回退到其他通道

### Requirement: Builtin Model Repository
标注页 SHALL 能列出并下载提示式预标注所需的模型，且 MUST NOT 让用户手工去外部网站找文件。

#### Scenario: 列出模型与本地状态
- **WHEN** 用户打开模型仓库区域
- **THEN** 系统 SHALL 列出每个模型的名称、体积、用途、许可证与是否已下载

#### Scenario: 下载有进度
- **WHEN** 用户点击下载
- **THEN** 界面 SHALL 显示下载进度与已下载/总字节，且 MUST NOT 阻塞标注页其他操作

#### Scenario: 可取消
- **WHEN** 用户在下载过程中点击取消
- **THEN** 系统 SHALL 停止下载并清理未完成的分片文件

#### Scenario: 断点续传
- **WHEN** 下载中断后用户再次点击下载
- **THEN** 系统 SHALL 从已下载的字节处继续，MUST NOT 从头重新开始

#### Scenario: 模型目录可覆盖
- **WHEN** 设置了模型目录环境变量
- **THEN** 系统 SHALL 使用该目录存放模型，MUST NOT 写死仓库内路径

### Requirement: Review State Tracking
系统 SHALL 记录每张图片的审核状态，让用户知道整个工程审到了哪里。

#### Scenario: 标为已审核并跳到下一张未审核
- **WHEN** 用户在标注页按下标记已审核的快捷键
- **THEN** 当前图片 SHALL 被标为已审核，并自动切换到下一张未审核的图片

#### Scenario: 取消已审核
- **WHEN** 用户对已审核的图片再次执行取消操作
- **THEN** 该图片 SHALL 回到未审核状态

#### Scenario: 列表显示审核状态
- **WHEN** 图片出现在左侧列表
- **THEN** 已审核的图片 SHALL 带可见的已审核标识，未审核的图片 MUST NOT 带该标识

#### Scenario: 进度可见
- **WHEN** 工程中部分图片已被审核
- **THEN** 界面 SHALL 显示已审核张数与总张数

#### Scenario: 刷新后状态不丢
- **WHEN** 用户刷新页面并重新打开同一个工程
- **THEN** 审核状态 SHALL 与刷新前一致

#### Scenario: 审核状态与正式标注分离
- **WHEN** 审核状态被持久化
- **THEN** `.tfanno` 工程文件的内容 SHALL 保持不变，审核状态 MUST NOT 写入工程文件

#### Scenario: 旧版旁挂文件可读
- **WHEN** 旁挂文件是旧版结构（没有审核块）
- **THEN** 系统 SHALL 正常读出草稿并把审核状态视为全部未审核，MUST NOT 报错

#### Scenario: 清除草稿不影响审核
- **WHEN** 用户清空全部草稿
- **THEN** 审核状态 SHALL 保留，除非用户显式要求整份替换
