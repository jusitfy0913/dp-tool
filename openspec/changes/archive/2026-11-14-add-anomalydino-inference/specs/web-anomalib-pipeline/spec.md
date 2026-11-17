## MODIFIED Requirements

### Requirement: Anomalib Pipeline Orchestration
后端 SHALL 提供异常检测全流程串联接口，按固定顺序推进四个阶段，每阶段独立可观测。

#### Scenario: 启动串联
- **WHEN** 客户端提交 `POST /api/pipeline/anomalib` 且 `model ∈ {patchcore, padim, efficientad, anomalydino}`
- **THEN** 后端 SHALL 创建整链记录并返回 `id`，在后台线程推进
- **AND** 非支持的模型或精度 SHALL 返回 400，不得启动任何子进程

#### Scenario: 阶段顺序固定
- **WHEN** 串联正常执行
- **THEN** 阶段顺序 SHALL 为 `train_export → probe → build → val`
- **AND** `probe` MUST 排在 `build` 之前，`val` MUST 排在 `build` 之后

#### Scenario: 训练与导出不拆分
- **WHEN** 未跳过训练
- **THEN** 训练、ONNX 导出与 metadata 阈值校准 SHALL 在同一个 launcher 进程内完成，
  不得为凑阶段数拆成多个进程

#### Scenario: 零样本模型复用同一阶段划分
- **WHEN** `model` 为 `anomalydino`
- **THEN** `train_export` 阶段 SHALL 由 AnomalyDINO 启动器完成「建库 + 导出 + 标定」，
  阶段名与顺序 MUST 与其它模型一致
- **AND** 后端 MUST NOT 为它新增接口或新增阶段

#### Scenario: 跳过训练
- **WHEN** 客户端设置 `skip_train=true`
- **THEN** 后端 SHALL 从 `probe` 开始，不执行 `train_export`
- **AND** 未提供含 `metadata.json` 与 `.onnx` 的 `artifacts_dir` 时 SHALL 返回 400

#### Scenario: 关闭精度验证
- **WHEN** 客户端设置 `run_val=false`
- **THEN** 整链 SHALL 在 `build` 完成后结束，不启动 `val`

#### Scenario: 失败与停止
- **WHEN** 任一阶段失败或被停止
- **THEN** 整链 SHALL 标为 `failed` / `cancelled`，后续段标 `skipped`，
  已完成阶段的产物 SHALL 保留

### Requirement: Metadata Preflight
编译前 SHALL 校验 metadata 与 ONNX 形状，避免编译白跑。

#### Scenario: 阈值缺失
- **WHEN** `metadata.json` 缺少 `image_threshold` 或 `pixel_threshold`
- **THEN** `probe` SHALL 失败并指出缺失字段（DLL 会拒绝加载）

#### Scenario: 动态 batch
- **WHEN** ONNX 输入为 `[-1, 3, H, W]`（Anomalib 导出恒为动态 batch）
- **THEN** `probe` SHALL 通过，并记录 `dynamic_batch=true`
- **AND** `build` SHALL 补充 `--minShapes` / `--optShapes` / `--maxShapes`，
  其中 opt 批次取 1、max 批次取请求的 `max_batch`

#### Scenario: 静态 batch
- **WHEN** ONNX 输入为 `[1, 3, H, W]` 且三轴均为固定值（AnomalyDINO 导出为静态）
- **THEN** `probe` SHALL 通过，并记录 `dynamic_batch=false`
- **AND** `build` MUST NOT 追加任何 `--minShapes` / `--optShapes` / `--maxShapes` 参数

#### Scenario: 高宽与 metadata 不一致
- **WHEN** ONNX 的静态高宽与 metadata 的 `input_size` 不一致
- **THEN** `probe` SHALL 失败并同时报出两个值

#### Scenario: 动态高宽
- **WHEN** ONNX 的高或宽是动态维
- **THEN** `probe` SHALL 拒绝并提示带固定 `input_size` 重新导出
