# web-infer-batch Specification

## Purpose
定义除单张图片之外的三种推理形态：目录批量、视频、摄像头逐帧。三者共用推理设置
（引擎 / 标签 / 任务类型 / `conf` / `nms`），但执行模型不同：批量与视频是可停止、
可观测的子进程任务；摄像头是同步单帧接口，帧不出本机。
## Requirements
### Requirement: Batch Directory Inference
后端 SHALL 提供目录批量推理接口，把目录内（可递归）的图片逐张推理，并产出结构化的产物包。

#### Scenario: 启动批量任务
- **WHEN** 客户端提交源目录、引擎与已编译好的推理参数
- **THEN** 后端 SHALL 创建子进程任务并返回 `run_id`，进度与日志经 `WS /ws/runs/{id}` 推送

#### Scenario: 产物结构固定
- **WHEN** 批量任务正常结束
- **THEN** 产物目录 SHALL 包含标注图目录、`results.csv`、`results.json`、`job.json`，
  且默认包含 `annotated.zip`

#### Scenario: CSV 编码与换行
- **WHEN** 写出 `results.csv`
- **THEN** 文件 SHALL 为 `utf-8-sig` 编码且以 `\n` 换行，保证 Excel 打开中文类别名不乱码、不多空行

#### Scenario: 单张失败不中断整批
- **WHEN** 目录中存在损坏或无法解码的图片
- **THEN** 该行 SHALL 标记为 `status=error` 并写明原因，其余图片照常处理，任务不得整体失败

#### Scenario: 处理上限
- **WHEN** 客户端指定最多处理张数
- **THEN** 后端 SHALL 只处理排序后的前 N 张，且 `limit=0` 表示不限制

#### Scenario: 取消任务
- **WHEN** 客户端请求停止批量任务
- **THEN** 后端 SHALL 终止子进程，已写出的文件 SHALL 保留，未完成的压缩包不得出现

### Requirement: Video Inference
后端 SHALL 提供视频推理接口，逐帧（或按步长抽样）推理并产出可播放的标注视频。

#### Scenario: 编码器可用性探测
- **WHEN** 客户端选择自动编码器
- **THEN** 后端 SHALL 依次尝试候选编码器，并**实际写入一帧后检查文件非空**，只有产出非空文件才算可用

#### Scenario: 抽帧保持时长
- **WHEN** 抽帧步长大于 1
- **THEN** 输出视频的帧率 SHALL 写为 `源帧率 ÷ 步长`，使播放时长与原视频一致

#### Scenario: 帧数上限
- **WHEN** 客户端指定最多处理帧数
- **THEN** 后端 SHALL 在达到上限时正常收尾并产出完整视频，不得报错

#### Scenario: 逐帧明细
- **WHEN** 客户端开启逐帧明细
- **THEN** 产物 SHALL 额外包含每帧检测框的 `detections.json`

#### Scenario: 视频仅支持 YOLO
- **WHEN** 客户端对视频请求 Anomalib 类型
- **THEN** 后端 SHALL 明确拒绝并说明原因，不得静默降级

### Requirement: Frame Inference for Camera
后端 SHALL 提供同步单帧推理接口，供浏览器摄像头逐帧调用，只返回结构化结果，不返回图像。

#### Scenario: 只回结构化结果
- **WHEN** 客户端提交一帧图像
- **THEN** 响应 SHALL 包含尺寸、框列表、类别配色与耗时，**不得包含 base64 图像**

#### Scenario: 不落盘
- **WHEN** 反复调用单帧接口
- **THEN** 后端 SHALL NOT 创建运行记录、产物目录或磁盘文件

#### Scenario: 与单张推理同构
- **WHEN** 同一引擎同时用于单张推理与单帧推理
- **THEN** 两者的框字段（含 OBB 角度、分割多边形、关键点）SHALL 同构，且共用引擎句柄缓存

#### Scenario: 帧率由客户端控制
- **WHEN** 客户端设置的帧率上限低于上一帧返回速度
- **THEN** 客户端 SHALL 跳过本次采集而不是排队，使表现是掉帧而非延迟累积

### Requirement: Artifact Download Sandbox
产物下载接口 SHALL 只允许访问批量产物根目录之内已生成的文件。

#### Scenario: 越界拒绝
- **WHEN** 请求路径解析后落在产物根目录之外（含 `..` 穿越）
- **THEN** 后端 SHALL 返回 404，且响应不得透露目标文件是否存在

#### Scenario: 内联与下载
- **WHEN** 客户端请求产物且未声明下载
- **THEN** 后端 SHALL 以内联方式返回，使视频可在页面直接播放；声明下载时以附件形式返回

#### Scenario: 不存在或非文件
- **WHEN** 目标路径不存在或指向目录
- **THEN** 后端 SHALL 返回 404，不得抛出未处理异常

### Requirement: Artifact Naming
批量产物中的图片文件名 SHALL 可安全落盘且保持输入顺序。

#### Scenario: 非法字符净化
- **WHEN** 输入文件名包含 Windows 非法字符
- **THEN** 产物文件名 SHALL 替换这些字符，同时保留中文与空格

#### Scenario: 顺序与去重
- **WHEN** 目录中多张图片净化后重名
- **THEN** 文件名 SHALL 带自增序号前缀，保证一一对应且不覆盖

#### Scenario: 扩展名规范化
- **WHEN** 输入为任意受支持格式
- **THEN** 标注图 SHALL 统一以实际编码格式的扩展名落盘

### Requirement: 批量与视频支持 ONNX 模型

目录批量与视频推理 SHALL 接受 `.onnx` 模型，并通过 ONNX 动态库执行，使三种推理形态（批量、视频、摄像头）与单张推理在同一个模型下得到一致的后端选择与结果字段。

#### Scenario: 批量以 ONNX 启动

- **WHEN** 客户端提交 `.onnx` 模型启动目录批量推理，且 ONNX 动态库可用
- **THEN** 批量任务必须通过 ONNX 动态库逐张推理，不得回退到 Python 兜底，也不得因为模型不是引擎而拒绝任务

#### Scenario: 视频以 ONNX 启动

- **WHEN** 客户端提交 `.onnx` 模型启动视频推理
- **THEN** 视频任务必须通过 ONNX 动态库逐帧推理并正常产出标注视频

#### Scenario: 后端记录到产物

- **WHEN** 批量或视频任务结束
- **THEN** 任务的元数据与结果汇总必须记录本次实际使用的后端名称，便于事后判断产物来自哪条链路

#### Scenario: 与单张推理结果同构

- **WHEN** 同一张图片分别经单张推理与批量推理处理
- **THEN** 两者在相同 `conf` / `nms` 下的框数量、坐标、置信度与类别号必须一致（差异仅允许浮点末位）

#### Scenario: ONNX 动态库缺失时可降级

- **WHEN** 批量或视频任务的模型是 `.onnx` 且 ONNX 动态库不可用
- **THEN** 任务必须在 Python 兜底可用时继续执行并在元数据中注明降级原因；兜底不支持该任务类型时必须让任务以可读错误失败，不得静默产出一份不完整的结果

#### Scenario: 不支持的任务类型明确拒绝

- **WHEN** 批量或视频请求的任务类型超出当前后端能力（如兜底路径下的 `segment` / `pose`）
- **THEN** 任务必须在开始前失败并说明缺少哪个动态库或哪个任务不受支持

### Requirement: 目录批量支持 OCR

目录批量推理 SHALL 支持 `kind=ocr`，通过 `ocr_infer.dll` 逐张推理并使用与单图相同的模型预检；视频与摄像头推理 SHALL 不支持 OCR。

#### Scenario: OCR 批量产物

- **WHEN** 客户端以兼容模型启动 `kind=ocr` 的目录批量推理
- **THEN** 每张图片必须产出一份与单图响应同构的 `<name>.ocr.json`，任务结束时产出 `ocr_summary.csv`（UTF-8 BOM，含文件名、行数、全文、平均置信度、耗时），元数据记录实际后端

#### Scenario: 批量前预检失败

- **WHEN** 批量请求的模型未通过预检（格式不兼容、yml 缺字段或文件缺失）
- **THEN** 任务必须在开始前失败并给出与单图接口相同的原因，不得产出部分结果

#### Scenario: 与单图结果同构

- **WHEN** 同一张图片在相同模型与参数下分别经单图与批量处理
- **THEN** 两者的行数、文字、多边形坐标与置信度必须一致

#### Scenario: 视频与摄像头拒绝 OCR

- **WHEN** 客户端以 `kind=ocr` 请求视频推理，或在推理页选择 OCR 后切到视频 / 摄像头
- **THEN** 后端必须返回 400 并说明 OCR 不支持实时推理，前端必须禁用对应按钮并给出原因

