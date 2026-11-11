## ADDED Requirements

### Requirement: 显存预检使用实测峰值表

系统 SHALL 维护实测峰值表 `7-autoresearch-yolo/experiments/vram_samples.json`，
按 `(imgsz, batch, precision)` 记录历史最大峰值显存。

需求值 MUST 按以下优先级取值（`free >= needed` 时放行）：

1. 该组合有样本：`peak * 1.15 + 400`；
2. 同分辨率同精度只有**更小 batch** 的样本：按 batch 线性外推到请求 batch；
3. 同分辨率同精度只有**更大 batch** 的样本：以它的实测需求作为上界（取较小者）；
4. 都没有：公式 `2600 + 300 * batch * (imgsz/640)^2`。

估算 MUST 随 batch 单调不减：不允许出现「batch 调小反而需求更大」。

峰值 MUST 优先取 `torch.cuda.max_memory_reserved()`，取不到时才用 `max_memory_allocated()`。

#### Scenario: 有样本时用实测值而非公式
- **WHEN** 表中 `960x4xfp32` 记录 peak=4000 MB，当前可用 6000 MB
- **THEN** 判定为放行，需求值按 4000×1.15+400=5000 MB 计算，且结果标明来源为「实测」

#### Scenario: 没有同组合样本时按 batch 线性外推
- **WHEN** 表中只有 `960x4xfp32` 的 peak=4000 MB，请求 batch=8
- **THEN** 需求值为 8000×1.15+400=9600 MB，并标明「按 batch=4 的实测值线性外推」

#### Scenario: 只有更大 batch 的样本时用上界封顶
- **WHEN** 表中只有 `640x8xfp32` 的 peak=2132 MB，请求 batch=7（公式算出来是 4700 MB）
- **THEN** 需求值被封顶为 2132×1.15+400=2852 MB，而不是公式值

#### Scenario: 什么都没跑过时回落到保守公式
- **WHEN** 表中不存在 `1152x4xfp32`
- **THEN** 需求值为 2600+300×4×(1152/640)²≈6488 MB，结果标明来源为「公式」

#### Scenario: 同一组合不同精度分开计
- **WHEN** 表中只有 `960x4xfp16`（fp16）的样本
- **THEN** 查询 fp32 的同组合时不得使用该样本，必须回落到公式

#### Scenario: 无 CUDA 时跳过预检
- **WHEN** 探测不到显卡（没有 torch 也没有 nvidia-smi）
- **THEN** 预检返回放行，并注明「无法读取显存，已跳过」

### Requirement: 训练结束自动记录显存样本

系统 SHALL 在每次训练正常结束时，把该次的峰值显存写入实测峰值表（自校准）。

同一组合多次训练时 MUST 保留最大值，并累计次数与最后更新时间。写入失败 MUST NOT
影响训练结果。

#### Scenario: 同组合第二次更大时更新
- **WHEN** 表中 `640x8xfp32` 已是 1769.5，新一次实测 2100
- **THEN** 表中该键的 peak 变为 2100，count 加 1

#### Scenario: 新一次更小时保留旧最大值
- **WHEN** 表中 `640x8xfp32` 已是 2200，新一次实测 1800
- **THEN** 表中该键的 peak 仍为 2200

#### Scenario: 表文件损坏
- **WHEN** `vram_samples.json` 内容不是合法 JSON
- **THEN** 读取视为空表、继续走公式，不抛异常；下一次写入重建文件

### Requirement: 混合精度实际状态可见

系统 SHALL 在训练前给出本机混合精度（AMP）可用性判定，并在训练结束后记录**实际**生效的精度。

判定 MUST 与 Ultralytics 的 AMP 黑名单保持一致（含 `t1000`、`t1200`、`t2000`、`1660`、
`1650`、`1630`、`t400`、`t550`、`t600`、`k40m`），并 MUST 说明「黑名单显卡即使传
`--amp` 也会被框架关闭」。

实际精度 MUST 写入训练台账（`results.tsv` 的 `amp` 列）与实验记录 JSON。

#### Scenario: 黑名单显卡给出明确结论
- **WHEN** 显卡名称为 `NVIDIA T1000 8GB`
- **THEN** 判定为「混合精度不可用」，理由指出命中的型号，并提示训练将走 fp32、显存与耗时都会增加

#### Scenario: 常规显卡判定可用
- **WHEN** 显卡名称为 `NVIDIA GeForce RTX 4090`
- **THEN** 判定为「混合精度可用」，不给出降级提示

#### Scenario: 训练结束记录实际精度
- **WHEN** 一次训练在 T1000 上以默认 `amp` 跑完
- **THEN** 台账该行的 `amp` 为 `fp32`，实验 JSON 的 `amp_effective` 为 `false`

### Requirement: 显存预检拦截训练启动

Web 训练接口 SHALL 在启动 YOLO 训练前执行显存预检。预检不放行且请求未显式要求
忽略时，MUST 返回 HTTP 409，响应中 MUST 包含可用量、需求量、依据与可执行建议。

请求带 `allow_low_vram=true` 时 MUST 放行并照常启动。

#### Scenario: 显存不足时拒绝启动
- **WHEN** 可用显存 4200 MB、需求 6491 MB，请求未带 `allow_low_vram`
- **THEN** 返回 409，detail 写明「可用 4200 MB < 需求 6491 MB（依据：公式）」

#### Scenario: 用户显式承担风险
- **WHEN** 同样的条件下请求带 `allow_low_vram=true`
- **THEN** 训练照常启动，日志里保留一条预检告警

#### Scenario: 预估不影响非 YOLO 任务
- **WHEN** 启动 Anomalib 训练或引擎构建
- **THEN** 不经过该显存闸门（它们的显存模型不同，混用只会误拦）

### Requirement: 精度覆盖参数

YOLO 训练接口 SHALL 接受三态 `amp` 参数：`null`（跟随检测，默认）、`false`（显式关闭）、
`true`（显式尝试）。`false` 时命令行 MUST 传 `--no-amp`，`true` 时 MUST 传 `--amp`，
`null` 时 MUST 不传该参数。

#### Scenario: 显式关闭
- **WHEN** 请求 `amp=false`
- **THEN** 命令行包含 `--no-amp`，页面提示「已显式关闭混合精度」

#### Scenario: 默认不干预
- **WHEN** 请求不带 `amp` 字段
- **THEN** 命令行不含 `--amp` / `--no-amp`，行为与升级前完全一致
