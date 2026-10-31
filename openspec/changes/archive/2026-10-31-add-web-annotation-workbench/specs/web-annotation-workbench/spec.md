## ADDED Requirements

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
