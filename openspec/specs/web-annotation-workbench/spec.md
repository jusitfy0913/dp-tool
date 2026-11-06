# web-annotation-workbench Specification

## Purpose
定义浏览器内的标注工作台：与 Qt 版共用 `.tfanno` 工程格式，提供画布绘制/撤销重做/显示控制/
自动保存，能用自有引擎、点框提示（SAM）或文本提示（GroundingDINO）生成草稿，并用旁挂草稿池
与审核状态支撑「整工程跑一遍、人只审核」的批量流程。设计目标是「不装 Qt、不训模型也能把框画完
并交给训练」。
## Requirements
### Requirement: Tfanno Project Interoperability
后端 SHALL 读写 `.tfanno` 标注工程文件，且读写的字段结构 MUST 与 Qt 版实现兼容。

#### Scenario: 读取 Qt 生成的工程
- **WHEN** 后端读取一个由 Qt 版生成的 `.tfanno` 文件
- **THEN** 它 SHALL 完整还原任务类型、类别表、图片列表与每张图的标注对象，不得丢失字段

#### Scenario: 写出的工程可被 Qt 打开
- **WHEN** 后端写出一个 `.tfanno` 文件
- **THEN** 该文件的根结构与标注对象字段 SHALL 与 Qt 读取端所期望的名称和类型一致，且 SHALL NOT 被 Qt 判为损坏

#### Scenario: 未知形状回退
- **WHEN** 输入文件包含 Qt 形状枚举之外的类型
- **THEN** 后端 SHALL 回退为矩形并在响应中给出提示，接口不得报错

#### Scenario: 矩形点数统一
- **WHEN** 输入标注的矩形使用四个角点
- **THEN** 后端 SHALL 将其折叠为由左上与右下构成的二点表示

#### Scenario: 关键点保持单对象模型
- **WHEN** 输入或输出包含关键点标注
- **THEN** 同一目标的关键点 SHALL 存放在单个对象中，并携带可见性数组

### Requirement: Extended Annotation Fields
标注对象 SHALL 支持难例标记、对象级布尔标记与结构化属性三项扩展字段，且扩展 MUST NOT 破坏与 Qt 版的兼容性。

#### Scenario: 写入扩展字段
- **WHEN** 用户为某对象设置难例标记或属性
- **THEN** 后端 SHALL 将其写入工程的扩展键中

#### Scenario: 默认值不落盘
- **WHEN** 某对象未设置任何扩展字段
- **THEN** 该对象的扩展键 SHALL NOT 出现在输出文件中

#### Scenario: 兼容性不受影响
- **WHEN** 一个包含扩展字段的工程被 Qt 版打开
- **THEN** Qt 的解析 SHALL 不受影响，原有字段 SHALL 全部正确还原

### Requirement: Annotation Project API
后端 SHALL 提供标注工程的读取、保存、图片流与导出接口。

#### Scenario: 保存时校验修订号
- **WHEN** 客户端提交的修订号与磁盘上的当前修订号不一致
- **THEN** 后端 SHALL 拒绝保存并返回冲突，不得覆盖磁盘内容

#### Scenario: 保存成功推进修订号
- **WHEN** 保存成功
- **THEN** 后端 SHALL 递增修订号并在响应中返回新值

#### Scenario: 图片流受白名单约束
- **WHEN** 客户端请求图片流且路径不在允许范围内
- **THEN** 后端 SHALL 拒绝并说明原因，不得返回文件内容

#### Scenario: 路径含非 ASCII 字符
- **WHEN** 请求的图片路径包含中文或空格
- **THEN** 后端 SHALL 正确读取并返回图片，不得报路径不存在

#### Scenario: 复用既有导出能力
- **WHEN** 客户端请求导出标注工程
- **THEN** 后端 SHALL 复用既有的 YOLO / COCO / VOC 导出实现，不得另写一套

### Requirement: Annotation Canvas
前端 SHALL 提供 Canvas 2D 标注画布，支持六种形状的绘制与编辑。

#### Scenario: 六种形状可绘制
- **WHEN** 用户选择矩形、旋转矩形、多边形、线、点或关键点工具并完成绘制
- **THEN** 画布 SHALL 生成对应类型的标注对象，且点数与形状语义相符

#### Scenario: 光标锚点缩放
- **WHEN** 用户在画布上滚动滚轮
- **THEN** 光标下方的图像像素 SHALL 保持不动，缩放行为与推理页查看器一致

#### Scenario: 拖动顶点编辑
- **WHEN** 用户拖动已选中形状的某个顶点
- **THEN** 仅该顶点位置变化，矩形与旋转矩形 SHALL 保持正交或保角约束

#### Scenario: 多边形插点与删点
- **WHEN** 用户在多边形某条边上插入点，或按住修饰键点击某个顶点
- **THEN** 画布 SHALL 分别增加或删除该顶点，形状其余部分不变

#### Scenario: 整体平移
- **WHEN** 用户拖动形状内部区域
- **THEN** 该形状的全部顶点 SHALL 同步平移

### Requirement: Undo And Redo
标注画布 SHALL 提供撤销与重做，且历史记录深度 MUST 不少于 50 步。

#### Scenario: 撤销可回退
- **WHEN** 用户连续编辑后触发撤销
- **THEN** 画布 SHALL 逐级回退到编辑前状态，最多 50 步

#### Scenario: 重做可前进
- **WHEN** 用户撤销后触发重做
- **THEN** 画布 SHALL 恢复到被撤销的状态

#### Scenario: 新编辑清空重做
- **WHEN** 用户在撤销状态下产生新的编辑
- **THEN** 重做历史 SHALL 被清空

#### Scenario: 快捷键可触达
- **WHEN** 用户按下撤销或重做的快捷键
- **THEN** 对应操作 SHALL 立即执行

### Requirement: Annotation Display Controls
标注画布 SHALL 支持对象显隐、填充透明度调整与在图上显示置信度。

#### Scenario: 单个对象隐藏
- **WHEN** 用户关闭某个对象的显隐开关
- **THEN** 该对象 SHALL 从画布上隐藏，其余对象不受影响

#### Scenario: 填充透明度可调
- **WHEN** 用户调整填充透明度
- **THEN** 全部标注的填充不透明度 SHALL 按新值渲染，边框可读性不受影响

#### Scenario: 图上显示置信度
- **WHEN** 用户开启置信度显示
- **THEN** 标注标签 SHALL 同时包含类别名与置信度数值

### Requirement: Annotation Navigation Aids
标注画布 SHALL 提供导航缩略图、十字准线与吸附辅助。

#### Scenario: 缩略图定位
- **WHEN** 用户在导航缩略图上点击或拖动
- **THEN** 主画布 SHALL 平移到对应区域，缩略图 SHALL 显示当前视口范围

#### Scenario: 十字准线
- **WHEN** 用户开启十字准线并移动光标
- **THEN** 画布 SHALL 显示跟随光标的水平与垂直辅助线

#### Scenario: 多边形闭合吸附
- **WHEN** 绘制多边形时光标接近起点
- **THEN** 光标位置 SHALL 吸附到起点，便于闭合

#### Scenario: 正交吸附与临时关闭
- **WHEN** 用户绘制线段时按住修饰键，或按下临时关闭吸附的修饰键
- **THEN** 前者 SHALL 约束为水平/垂直方向，后者 SHALL 临时取消吸附

### Requirement: Debounced Auto Save
标注画布 SHALL 在编辑后自动保存，且 MUST NOT 把保存动作计入撤销历史。

#### Scenario: 编辑后自动保存
- **WHEN** 用户完成一次编辑且停止操作达到防抖间隔
- **THEN** 后端 SHALL 收到一次保存请求

#### Scenario: 连续编辑合并
- **WHEN** 用户在一次防抖间隔内连续进行多次编辑
- **THEN** 后端 SHALL 只收到一次保存请求

#### Scenario: 保存不进入撤销栈
- **WHEN** 自动保存完成
- **THEN** 撤销历史 SHALL NOT 因保存动作而增长

### Requirement: Auto Labeling From Own Inference
系统 SHALL 支持用自有推理引擎对当前图片生成预标注草稿。

#### Scenario: 生成草稿
- **WHEN** 用户在当前图片上触发预标注
- **THEN** 系统 SHALL 调用自有推理引擎，并把检测结果转为矩形标注草稿

#### Scenario: 草稿不直接落盘
- **WHEN** 预标注完成
- **THEN** 草稿 SHALL 以独立样式显示，MUST NOT 自动写入工程文件

#### Scenario: 逐个接受与拒绝
- **WHEN** 用户对某个草稿选择接受或拒绝
- **THEN** 接受者转为正式标注，拒绝者被丢弃

#### Scenario: 全部接受
- **WHEN** 用户选择接受全部草稿
- **THEN** 全部草稿 SHALL 转为正式标注，且该操作可被撤销

#### Scenario: 保留置信度
- **WHEN** 草稿转为正式标注
- **THEN** 该标注 SHALL 保留推理得到的置信度数值

### Requirement: External Auto Labeling Service
系统 SHALL 支持通过 HTTP 调用独立进程形式的外部标注服务，用于提示分割与文字提示检测。

#### Scenario: 服务未配置时不影响主流程
- **WHEN** 未配置外部服务地址
- **THEN** 仅自有推理通道可用，界面不得出现错误提示阻塞操作

#### Scenario: 服务不可达时显式提示
- **WHEN** 已配置外部服务但其不可达
- **THEN** 系统 SHALL 明确提示不可达并回退到自有推理通道，MUST NOT 静默失败

#### Scenario: 结果转为标注
- **WHEN** 外部服务返回形状列表
- **THEN** 系统 SHALL 将其转换为本项目标注对象并作为草稿呈现

#### Scenario: 姿态结果聚合
- **WHEN** 外部服务返回按组标识关联的多个关键点
- **THEN** 系统 SHALL 按组聚合为单个关键点对象

#### Scenario: 许可边界
- **WHEN** 系统打包分发
- **THEN** 该外部服务的代码 MUST NOT 出现在分发物中，调用 SHALL 仅通过网络进行

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

