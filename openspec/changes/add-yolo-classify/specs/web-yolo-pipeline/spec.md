## ADDED Requirements

### Requirement: 分类全流程停在 ONNX

任务为 `classify` 时，全流程 SHALL 完成训练和 ONNX 导出后结束。转置检测布局、编译 TensorRT 引擎和检测精度验证 SHALL 标记为跳过，并在阶段上写明原因。

#### Scenario: 分类串联的阶段

- **WHEN** 用户以分类任务启动全流程且训练与导出成功
- **THEN** `train` 与 `export` 必须完成
- **AND** `probe`、`build`、`val` 必须为跳过
- **AND** 跳过原因必须说明分类这一版不编译引擎、也不做检测精度验证

#### Scenario: 不生成检测转置文件

- **WHEN** 分类训练导出 ONNX
- **THEN** 产物必须包含分类模型的 `.onnx`
- **AND** 不得把该文件再转成检测用的 `.transd.onnx` 后当作成功产物

#### Scenario: 页面说明

- **WHEN** 训练页的任务选为分类
- **THEN** 全流程卡片必须在开始前说明引擎编译和检测验证会被跳过
