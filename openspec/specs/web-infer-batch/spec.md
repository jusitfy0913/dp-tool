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
