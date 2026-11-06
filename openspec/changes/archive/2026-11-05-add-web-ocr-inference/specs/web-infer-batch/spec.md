## ADDED Requirements

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
