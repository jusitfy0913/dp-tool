# web-infer-console Specification

## Purpose
定义 Web 单张推理控制台：YOLO 与 Anomalib 共用一个页面，阈值/置信度可覆盖且能追溯来源，
检出框以矢量叠加层渲染、任意倍率清晰，句柄按参数缓存并有上限，结果可表格化查看与导出。
Anomalib 的推荐阈值在**推理之前**即可读，不需要先跑一次才知道会用多少。

## Requirements
### Requirement: Anomalib Threshold Override
推理接口 SHALL 允许客户端覆盖异常检测的图像阈值与像素阈值，未提供时 SHALL 沿用 metadata 中的值。

#### Scenario: 覆盖图像阈值
- **WHEN** 客户端提交不小于 0 的图像阈值
- **THEN** 后端 SHALL 把该值传给 `ai_create` 的 `image_threshold_override`，且响应中 `is_anomaly` 依据该阈值判定

#### Scenario: 沿用 metadata 默认值
- **WHEN** 客户端未提交阈值字段，或提交负数
- **THEN** 后端 SHALL 传入负值使 DLL 使用 metadata 中的阈值

#### Scenario: 响应区分生效值与原值
- **WHEN** 客户端覆盖了阈值并完成一次推理
- **THEN** 响应 SHALL 同时给出本次生效的阈值与 metadata 中记录的原阈值

#### Scenario: 阈值进入句柄缓存键
- **WHEN** 客户端用不同的阈值连续推理
- **THEN** 后端 SHALL 为不同阈值使用不同的句柄，不得复用按旧阈值创建的句柄

### Requirement: Label File Mapping
后端 SHALL 支持读取 YOLO 标签文件中的类别表，把整数类别号映射为类别名，并 SHALL 下发统一的类别配色。

#### Scenario: 类别名生效
- **WHEN** 客户端提交标签文件且该文件包含类别数组
- **THEN** 推理响应的每个检测框 SHALL 带有对应的类别名

#### Scenario: 缺少标签文件时回退
- **WHEN** 客户端未提交标签文件，或文件缺少类别数组
- **THEN** 后端 SHALL 回退为整数类别标识，接口不得报错

#### Scenario: 配色前后端一致
- **WHEN** 推理结果同时包含服务端绘制的标注图与前端矢量叠加层
- **THEN** 二者 SHALL 使用响应中下发的同一份类别配色

### Requirement: Training Artifact Discovery
后端 SHALL 提供接口枚举训练产物目录中的可用引擎，并附带其配套文件信息。

#### Scenario: 枚举产物
- **WHEN** 客户端请求模型列表
- **THEN** 后端 SHALL 返回 YOLO 与 Anomalib 产物根目录下所有引擎，含路径、体积、修改时间

#### Scenario: 附带配套文件
- **WHEN** 某引擎同目录或上级存在同名 ONNX、标签文件或 metadata
- **THEN** 该引擎条目 SHALL 附带这些文件的路径，并标注 ONNX 是否已转置

#### Scenario: 推断任务类型
- **WHEN** 枚举到一个引擎
- **THEN** 后端 SHALL 依据其所在目录结构推断所属任务类型，供前端预填

#### Scenario: 最近的排在前
- **WHEN** 接口返回多个引擎
- **THEN** 结果 SHALL 按修改时间倒序排列

#### Scenario: 白名单外目录不参与扫描
- **WHEN** 配置的产物根目录不在允许范围内
- **THEN** 该目录 SHALL NOT 被扫描，且接口 SHALL 说明原因

### Requirement: Bounded Inference Handle Cache
推理句柄缓存 SHALL 有上限，并在超限时销毁最久未使用的句柄。

#### Scenario: 超限淘汰
- **WHEN** 缓存中的句柄数量达到上限且需要创建新句柄
- **THEN** 后端 SHALL 先销毁最久未使用的句柄，再创建新句柄

#### Scenario: 命中即刷新
- **WHEN** 请求命中一个已存在的句柄
- **THEN** 该句柄 SHALL 被标记为最近使用，不得成为下一次被淘汰的对象

#### Scenario: 上限内的参数不重复加载
- **WHEN** 客户端用已缓存过的参数组合推理
- **THEN** 后端 SHALL 复用现有句柄，不得重建引擎

### Requirement: Anomaly Heatmap
异常检测推理 SHALL 返回基于归一化异常图渲染的热能图。

#### Scenario: 渲染热能图
- **WHEN** DLL 返回的归一化异常图宽高均大于 0 且指针非空
- **THEN** 后端 SHALL 将其归一化后用假彩色渲染成热能图并返回

#### Scenario: 无异常图时不报错
- **WHEN** 归一化异常图缺失或尺寸为 0
- **THEN** 热能图字段 SHALL 为空，接口仍返回成功

#### Scenario: 色带可选
- **WHEN** 客户端指定色带名称
- **THEN** 后端 SHALL 使用对应色带渲染，非法名称时回退为默认色带

#### Scenario: 原图一并返回
- **WHEN** 一次异常检测推理完成
- **THEN** 响应 SHALL 包含原图，供前端显示，不得要求前端重复解码上传内容

### Requirement: Adjustable Inference Parameters UI
推理页 SHALL 以滑块与数字框联动的方式暴露全部可调参数，并 SHALL 说明参数改动的影响。

#### Scenario: 滑块与数字框联动
- **WHEN** 用户拖动某个参数滑块
- **THEN** 对应数字框 SHALL 同步更新；反之亦然

#### Scenario: 提示会重建引擎
- **WHEN** 用户修改 `conf`、`nms` 或任一阈值
- **THEN** 界面 SHALL 提示该改动会导致引擎重建、首次推理较慢

#### Scenario: 显示生效阈值
- **WHEN** 异常检测使用 metadata 默认阈值
- **THEN** 界面 SHALL 显示实际生效的阈值数值，而不是只显示空值

#### Scenario: 沿用 metadata 时回显推荐值
- **WHEN** 用户填写或选择一个可读的 `metadata.json`，且「沿用 metadata 默认阈值」处于勾选状态
- **THEN** 界面 SHALL 在推理之前就显示文件内记录的两个阈值，并保留至少 4 位小数
- **AND** 两个只读输入框的值 SHALL 同步为该推荐值，不得停留在与文件无关的默认值
- **AND** 界面 SHALL 一并显示该文件的模型类型与输入尺寸，供用户核对引擎是否配套

#### Scenario: 标注推荐值来源
- **WHEN** 所选 `metadata.json` 含有 `calibration` 区块（阈值经过现场数据重新标定）
- **THEN** 界面 SHALL 标注「已校准」并给出样本判对率；无该区块时 SHALL 标注为文件原值

#### Scenario: 推荐值读取失败
- **WHEN** `metadata.json` 不存在、无法解析，或路径不在允许范围内
- **THEN** 界面 SHALL 清空已显示的推荐值并说明失败原因
- **AND** 界面 SHALL NOT 继续显示上一个文件遗留的数值

#### Scenario: 手动覆盖时保留原值对照
- **WHEN** 用户取消勾选「沿用 metadata 默认阈值」并改为手动填写
- **THEN** 界面 SHALL 同时呈现所用阈值与 metadata 中记录的原值，以便判断是否被覆盖
- **AND** 阈值的可调精度 SHALL 足以复现文件中的 4 位小数值

### Requirement: Model Selection From Artifacts
推理页 SHALL 提供模型下拉列表，选中后自动填充相关路径字段。

#### Scenario: 自动填充
- **WHEN** 用户从下拉列表选择一个引擎
- **THEN** 界面 SHALL 自动填充引擎路径，并在可用时填充标签文件、metadata 与任务类型

#### Scenario: 标注未转置的引擎
- **WHEN** 所选引擎对应的 ONNX 尚未转置
- **THEN** 界面 SHALL 给出提示，说明该引擎不适合当前推理链路

#### Scenario: 手工输入仍可用
- **WHEN** 用户在路径输入框中直接输入路径
- **THEN** 界面 SHALL 照常接受，不得强制只能从下拉选择

### Requirement: Detection Overlay Rendering
检出框 SHALL 以矢量方式叠加显示，在任意缩放倍率下保持清晰，并 SHALL 同时展示类别名与置信度。

#### Scenario: 高倍率下仍然清晰
- **WHEN** 用户把结果图放大到 4 倍以上
- **THEN** 框线与标签文字 SHALL 保持锐利，不得出现位图放大产生的模糊或锯齿

#### Scenario: 标签内容完整
- **WHEN** 结果图中显示一个检测框
- **THEN** 其标签 SHALL 同时包含类别名与置信度数值

#### Scenario: 标签不越界
- **WHEN** 检测框靠近图像上边缘导致标签会超出显示区域
- **THEN** 标签 SHALL 自动翻转到框内侧

#### Scenario: 叠加层可独立关闭
- **WHEN** 用户关闭「显示框」或「显示标签」
- **THEN** 对应元素 SHALL 隐藏，另一项不受影响

#### Scenario: 同类别同色
- **WHEN** 同一张图中存在多个同类别的检测框
- **THEN** 它们的框线 SHALL 使用相同颜色，且与响应下发的类别配色一致

### Requirement: Inference Result Inspection
推理页 SHALL 提供结果明细表、置信度分布、计时分解与结果导出。

#### Scenario: 明细表可排序
- **WHEN** 一次 YOLO 推理返回多个检测框
- **THEN** 明细表 SHALL 逐行列出类别名、置信度与坐标，并支持按列排序

#### Scenario: 行与框联动
- **WHEN** 用户点击明细表中的某一行
- **THEN** 结果图中对应的检测框 SHALL 被高亮

#### Scenario: 导出结果
- **WHEN** 用户点击导出
- **THEN** 界面 SHALL 生成包含本次结果的 JSON 或 CSV 文件供下载，且导出内容与屏幕显示一致

#### Scenario: 计时分解
- **WHEN** 一次推理完成
- **THEN** 界面 SHALL 分项展示预处理、推理、后处理各自的耗时

### Requirement: Metadata Threshold Inspection
后端 SHALL 提供只读接口读取 `metadata.json` 中的推荐参数，供推理页在推理之前展示。

#### Scenario: 读取成功
- **WHEN** 客户端请求 `GET /api/infer/anomalib/metadata?path=…` 且该文件可解析
- **THEN** 响应 SHALL 包含图像阈值、像素阈值、模型类型、输入尺寸、归一化方式、
  分数区间、异常图区间与校准精度
- **AND** 文件里缺失的字段 SHALL 以空值返回，不得因为缺字段而使整个请求失败

#### Scenario: 零阈值是有效值
- **WHEN** 文件中记录的阈值为 `0.0`（校准后正常样本分数恒为 0 时会出现）
- **THEN** 响应 SHALL 原样返回 `0.0`，界面 SHALL 将其当作有效阈值而非"未读到"

#### Scenario: 不加载模型
- **WHEN** 客户端用任意路径调用该接口
- **THEN** 后端 SHALL 只读取文件，不得加载引擎、创建推理句柄或占用显存

#### Scenario: 非法路径
- **WHEN** 路径为空、文件不存在或 JSON 无法解析
- **THEN** 后端 SHALL 返回 400 并给出可读原因

#### Scenario: 越界路径
- **WHEN** 路径不在允许浏览的范围内
- **THEN** 后端 SHALL 返回 403，且不得读取该文件

