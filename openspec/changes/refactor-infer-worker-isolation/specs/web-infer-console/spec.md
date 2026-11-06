## ADDED Requirements

### Requirement: 推理动态库在独立工作进程中执行

Web 服务进程 SHALL NOT 直接加载推理动态库。单图推理、摄像头单帧、OCR 与 YOLO 自动标注 SHALL 通过常驻推理工作进程调用 `yolo_infer.dll`、`anomalib_infer.dll`、`onnx_infer.dll` 与 `ocr_infer.dll`；接口的请求与返回结构必须与进程内调用时完全一致。

#### Scenario: 结果与进程内调用一致

- **WHEN** 同一张图片、同一模型与参数分别在 `TRTFORGE_INFER_ISOLATION=1` 与 `0` 下推理
- **THEN** 两次返回的框、分数、掩码、热力图与文字必须完全一致

#### Scenario: 动态库崩溃不影响 Web 服务

- **WHEN** 推理过程中工作进程因访问违例退出
- **THEN** 当前请求必须返回 503，信息包含模型路径与图片名
- **AND** Web 服务进程必须继续运行，训练任务记录与实时日志不受影响
- **AND** 下一次推理请求必须由自动重启的工作进程正常完成

#### Scenario: 崩溃循环熔断

- **WHEN** 工作进程在 60 秒内连续崩溃 3 次
- **THEN** 后端必须停止自动重启，并把各推理动态库报告为不可用
- **AND** 后续推理请求必须走既有的 onnxruntime 降级路径；找不到可降级的 ONNX 模型时返回 503
- **AND** `/api/capabilities` 必须报告推理进程处于熔断状态，用户可通过 `POST /api/infer/worker/restart` 恢复

#### Scenario: 推理卡死

- **WHEN** 单次推理超过超时上限
- **THEN** 当前请求必须返回 504，工作进程必须被强制结束并重启

#### Scenario: 排障时可回到进程内调用

- **WHEN** 设置 `TRTFORGE_INFER_ISOLATION=0`
- **THEN** 后端必须在 Web 进程内直接调用推理模块，行为与 1.27.0 一致
