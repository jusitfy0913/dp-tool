## MODIFIED Requirements
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

## ADDED Requirements
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
