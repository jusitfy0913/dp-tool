## ADDED Requirements

（无。本次变更不引入新能力，只把已有需求的验收标准补严。）

## MODIFIED Requirements

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

#### Scenario: 提示输入容忍数组类型
- **WHEN** 提示点集在进入推理前已被几何工具转成数值数组（而非 Python 列表）
- **THEN** 系统 SHALL 正常生成草稿，MUST NOT 因"数组真值判断歧义"而失败

> 为什么补这一条：v1.24 的 `build_prompt()` 用 `points or []` 判空，而上游
> `clamp_points()` 返回的是 numpy 数组，ndarray 的 `__bool__` 在多元素时直接抛
> `ValueError`——**任何一次点/框提示都是 500**。单元测试全喂 list，绕过了这条真实路径，
> 所以需求文本里必须写明"上游可能给数组"，否则同样的错会被再写一次。

#### Scenario: 内置权重上真机可跑通
- **WHEN** 使用模型仓库内置的分割权重（未替换为其他导出）执行点或框提示
- **THEN** 系统 SHALL 在无人工干预的情况下产出至少一个多边形草稿
- **AND** MUST 按模型声明的输入签名（维度与通道顺序）构造输入，MUST NOT 假定固定的张量布局

> 为什么补这一条：v1.24 把 encoder 输入写死成 NCHW `(1,3,H,W)`，而本项目权重声明的是
> rank-3 HWC `[H,W,3]`，onnxruntime 直接拒收。单测用的是假 session，永远发现不了
> "布局与真权重不匹配"，所以验收必须锚定"用内置权重跑通"。

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

#### Scenario: 体积与实际文件一致
- **WHEN** 清单中某个模型被标记为已就绪，或一次下载被标记为完成
- **THEN** 清单声明的体积 SHALL 与服务端 `Content-Length` 一致
- **AND** 进度 MUST 能走到 100%，MUST NOT 在未下满时提前显示完成

> 为什么补这一条：v1.24 清单里的体积是手估的（GroundingDINO 写 700 MB，实际 171.28 MB）。
> 体积同时是进度条分母和"文件是否完整"的判据，估大了会让进度条卡在 24% 就"完成"，
> 用户会以为下载被截断——实测 214.3 MiB 才是对的。

#### Scenario: 回退地址可用
- **WHEN** 某个下载源连接失败或返回错误
- **THEN** 系统 SHALL 依次尝试其余镜像，且每个候选地址 MUST 是实测可返回该文件的形式
- **AND** MUST NOT 把未经探活的地址写进候选列表

> 为什么补这一条：v1.24 为每个文件配了三条 `hf-mirror.com/CVHub520/...` 回退，
> **实测全部 404**（那些仓库不存在），主源超时后只是白等一次。
