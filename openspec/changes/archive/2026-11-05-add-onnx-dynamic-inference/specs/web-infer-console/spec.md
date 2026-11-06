## ADDED Requirements

### Requirement: 推理页直接选择 ONNX 模型

推理页 SHALL 允许用户把 `.onnx` 模型当作一等推理输入，与 `.trt` / `.engine` 一样通过路径输入、文件浏览与训练产物下拉选择，并在结果中如实标注本次使用的后端。

#### Scenario: 浏览并选择 ONNX 文件

- **WHEN** 用户在 YOLO 或 Anomalib 的模型路径输入框点击浏览
- **THEN** 文件对话框必须同时接受 `.onnx`、`.engine`、`.trt`、`.plan`，且选中 `.onnx` 后路径原样填入输入框

#### Scenario: 训练产物下拉提供 ONNX 条目

- **WHEN** 某个训练产物条目存在配套的 `.onnx` 文件
- **THEN** 下拉列表必须额外提供一条标注为 ONNX 直推的选项，选中后填入该 `.onnx` 的路径，并照常填充标签文件、`metadata.json` 与任务类型

#### Scenario: 未转置警告只针对引擎

- **WHEN** 用户选择的是引擎且其配套 ONNX 未转置
- **THEN** 界面必须给出该引擎解码不可信的警告；当用户直接选择 `.onnx` 时不得显示该警告，因为 ONNX 直推支持两种输出布局

#### Scenario: 手工输入 ONNX 路径

- **WHEN** 用户在路径输入框中直接输入一个 `.onnx` 路径
- **THEN** 界面必须照常接受并提交，不得要求必须从下拉选择

#### Scenario: 后端如实回显

- **WHEN** 一次推理完成
- **THEN** 结果区的后端指标必须显示服务端返回的后端名称（ONNX 动态库、引擎动态库或 Python 兜底），不得写死或推断

#### Scenario: 能力接口暴露 ONNX 后端

- **WHEN** 客户端请求能力接口
- **THEN** 响应必须包含 ONNX 动态库是否可用、其不可用的原因，以及实际的 ONNX Runtime 提供程序

### Requirement: 推理后端选择顺序与降级

后端 SHALL 按「ONNX 动态库 → 对应引擎动态库 → Python 兜底」的顺序为请求选择执行后端，并在降级时给出可执行的修复提示，而不是笼统报错。

#### Scenario: ONNX 走 ONNX 动态库

- **WHEN** 请求的模型是 `.onnx` 且 ONNX 动态库可用
- **THEN** 后端必须通过 ONNX 动态库推理，响应中的后端字段为 ONNX 动态库名称，且不得走 Python 兜底

#### Scenario: 引擎走引擎动态库

- **WHEN** 请求的模型是 `.engine` / `.trt` / `.plan` 且对应引擎动态库可用
- **THEN** 后端必须通过该引擎动态库推理，行为与新增 ONNX 支持之前完全一致

#### Scenario: ONNX 动态库缺失时降级

- **WHEN** 请求的模型是 `.onnx` 但 ONNX 动态库不可用
- **THEN** 后端必须在 Python 侧仍可运行时降级执行，并在响应中给出降级原因；两者都不可用时必须返回可读错误，说明要么提供 ONNX 动态库、要么安装 Python 推理依赖

#### Scenario: 摄像头与单图共用同一套选择

- **WHEN** 同一模型同时用于单图推理与摄像头逐帧推理
- **THEN** 两者必须使用同一套后端选择逻辑与同一份句柄缓存，框字段保持同构

#### Scenario: 句柄缓存不串后端

- **WHEN** 用户先后用同一个模型的 `.engine` 与 `.onnx` 推理
- **THEN** 后端必须为两种后端分别持有句柄，不得互相命中对方的缓存条目
