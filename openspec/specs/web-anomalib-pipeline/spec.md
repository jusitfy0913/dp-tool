# web-anomalib-pipeline Specification

## Purpose
定义 Web 端的 Anomalib 全流程串联：把「训练 + 导出 + 阈值校准 → 校验 metadata → 编译引擎 →
引擎精度验证」压成一条可停止、可看日志的链，并在产物路径只有训练完才知道的前提下延迟派生命令。
## Requirements
### Requirement: Anomalib Pipeline Orchestration
后端 SHALL 提供异常检测全流程串联接口，按固定顺序推进四个阶段，每阶段独立可观测。

#### Scenario: 启动串联
- **WHEN** 客户端提交 `POST /api/pipeline/anomalib` 且 `model ∈ {patchcore, padim, efficientad}`
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

### Requirement: Deferred Command Resolution
命令依赖训练后才确定的产物路径时，后端 SHALL 在阶段执行前派生，不得猜测路径。

#### Scenario: 版本目录延迟定位
- **WHEN** 训练产物的实际路径为 `results/<模型类名>/<name>/v<N>/weights/lightning/`
- **THEN** `build` 与 `val` 的命令 SHALL 在该阶段开始前按已记录产物拼装
- **AND** 拼装失败 SHALL 使该阶段失败并说明原因，不得退回 `v0` 之类的猜测

#### Scenario: 多产物记录
- **WHEN** 校验阶段确认了 onnx / metadata / ckpt
- **THEN** 三者 SHALL 同时记入整链 `artifacts`，同路径只出现一次

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

#### Scenario: 动态高宽
- **WHEN** ONNX 的高或宽是动态维
- **THEN** `probe` SHALL 拒绝并提示带固定 `input_size` 重新导出

### Requirement: Anomaly Metrics
后端 SHALL 计算与 DLL 判定口径一致的图像级指标，并在没有掩码时给出免掩码的像素级诊断。

#### Scenario: 阈值语义对齐
- **WHEN** 报告在 `image_threshold` 下判定异常
- **THEN** 该判定 MUST 与 DLL 的 `clamp01((raw - min) / (max - min)) >= 0.5` 等价
- **AND** 报告 SHALL 给出报告判定与引擎 `is_anomaly` 不一致的张数

#### Scenario: 指标内容
- **WHEN** 评估完成
- **THEN** 报告 SHALL 含 image AUROC、阈值下的 TP/FP/TN/FN、precision、recall、F1、
  FPR（误报率）、FNR（漏检率）与最优 F1 阈值

#### Scenario: AUROC 同分
- **WHEN** 多张图片分数相同
- **THEN** AUROC SHALL 使用平均秩（Mann-Whitney U），不得因排序不稳定而抖动

#### Scenario: 像素级指标的两种模式
- **WHEN** 评估完成且引擎返回了可用的归一化异常图
- **THEN** 报告 SHALL 返回 `pixel.available=true`，`pixel.mode` 为 `diagnostic`
  （无掩码，只做诊断）或 `mask`（有掩码，可算像素级 AUROC）

#### Scenario: 像素级不可用
- **WHEN** 客户端请求像素级指标
- **THEN** 报告 SHALL 返回 `pixel.available=false` 并说明原因（缺 ground_truth 掩码），
  不得返回猜测值

#### Scenario: 完全拿不到异常图
- **WHEN** 引擎未返回异常图，或异常图尺寸恒为 0
- **THEN** 报告 SHALL 返回 `pixel.available=false` 并在 `pixel.reason` 说明原因

#### Scenario: 报告产物
- **WHEN** `val` 成功
- **THEN** 产物 SHALL 包含 `anomaly_val_report.json` 与同名 `.csv`（`utf-8-sig`）

### Requirement: Anomalib Pipeline UI
训练页串联卡片 SHALL 支持异常检测模式并展示指标。

#### Scenario: 模式切换
- **WHEN** 用户在「类型」下拉选择异常检测
- **THEN** 卡片 SHALL 显示异常检测参数区（模型 / 输入尺寸 / 数据集根目录 / 跳过训练 /
  产物目录 / 精度 / 是否验证 / 评估上限），并隐藏 YOLO 专属参数

#### Scenario: 精度卡片
- **WHEN** 报告存在
- **THEN** 卡片 SHALL 显示 AUROC、F1、漏检率、误报率、precision、recall、
  评估张数与最优阈值
- **AND** `decision_mismatch > 0` 时 SHALL 显示告警，提示归一化口径可能不一致

#### Scenario: 像素级诊断入卡
- **WHEN** 报告含可用的像素级诊断
- **THEN** 卡片 SHALL 显示好图乱指率、异常图定位面积与「指不出位置」的张数
- **AND** 「指不出位置」大于 0 时该行 SHALL 标红并加告警标记
- **AND** 像素级完全不可用时应显示一行原因，不得留空数字

#### Scenario: 产物衔接
- **WHEN** 串联结束
- **THEN** 引擎与 metadata 行 SHALL 提供跳转到推理页的入口，任意行可复制绝对路径

### Requirement: Pixel Diagnostics
评估脚本 SHALL 在无掩码条件下，用引擎返回的归一化异常图给出像素级诊断结论。

#### Scenario: 无掩码时不给 AUROC 伪值
- **WHEN** 数据集没有 ground_truth 掩码
- **THEN** 报告 SHALL 把 `pixel.auroc` 置为 `null` 并给出原因，
  不得用整图标签冒充像素标签计算伪 AUROC

#### Scenario: 好图乱指率
- **WHEN** 一批标签为正常（`label=0`）的图片完成评估且给出了异常图
- **THEN** 报告 SHALL 给出每张正常图中被判为异常（分数不低于 `pixel_threshold`）的像素
  占比，并给出这批图上的平均值

#### Scenario: 定位面积
- **WHEN** 一批标签为异常（`label=1`）的图片完成评估
- **THEN** 报告 SHALL 给出每张异常图中被判为异常的像素占比及其平均值，
  以便区分「小而集中」与「糊成一片」

#### Scenario: 指不出位置
- **WHEN** 某张图被引擎判为异常，但其异常图中没有任何像素达到 `pixel_threshold`
- **THEN** 该图 SHALL 计入 `abnormal_undirected`，报告据此提示「分数报坏但指不出位置」

#### Scenario: 异常图缺失不中断评估
- **WHEN** 某张图未返回异常图或异常图尺寸为 0
- **THEN** 该图 SHALL 跳过像素统计并计入 `pixel_skipped`，其余图片的统计不受影响

#### Scenario: 缺少像素阈值
- **WHEN** `metadata.json` 中没有 `pixel_threshold`
- **THEN** 报告 SHALL 不给出任何「超阈值」类指标，只给出异常图尺寸与跳过张数，
  并在 `pixel.reason` 说明缺少该阈值

#### Scenario: 直方图口径
- **WHEN** 累加异常图直方图
- **THEN** 各图的异常图 SHALL 归一到同一组桶边界后累加，且落在边界外的像素夹到端点，
  不得丢弃像素

