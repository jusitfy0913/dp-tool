# anomalydino-export Specification

## Purpose
TBD - created by archiving change add-anomalydino-inference. Update Purpose after archive.
## Requirements
### Requirement: Zero-Shot Memory Bank Export

系统 SHALL 提供 AnomalyDINO 的零样本导出路子：只用正常图像构建特征记忆库，不得要求任何标注文件，不得执行梯度训练。

#### Scenario: 只提供正常图像目录

- **WHEN** 用户给出数据集根目录与正常图像子目录（默认 `train/good`）
- **THEN** 启动器 SHALL 只读取正常图像来构建记忆库
- **AND** 它 MUST NOT 读取任何掩码或缺陷类别目录
- **AND** 它 MUST NOT 调用反向传播或优化器

#### Scenario: 正常图像数量不足

- **WHEN** 可用正常图像少于下限（8 张）
- **THEN** 启动器 SHALL 以 `DATA_TOO_FEW` 错误终止
- **AND** 它 MUST NOT 静默降级为更少的样本继续导出

#### Scenario: 记忆库规模可追溯

- **WHEN** 建库完成
- **THEN** 库的行数 SHALL 等于「参与建库的图数（含旋转增强）× 网格单元数」
- **AND** 该行数 SHALL 写入 `metadata.json` 的 `bank_rows`

### Requirement: Backbone Fidelity

系统 SHALL 使用 timm 的 `vit_small_patch14_dinov2.lvd142m` 加载官方 DINOv2 权重，并按上游 `interpolate_offset=0.1` 的公式预插值位置编码，使 patch 特征与上游 `torch.hub` 实现一致。

#### Scenario: 位置编码按上游公式插值

- **WHEN** 目标分辨率为 `R`，patch 尺寸为 14，原始位置编码网格为 `M`
- **THEN** 位置编码 SHALL 以 `mode="bicubic"`、`antialias=False`、
  `scale_factor=((R/14 + 0.1) / M, (R/14 + 0.1) / M)` 插值
- **AND** 它 MUST NOT 使用 `size=(grid, grid)` 的写法（实测会导致逐 token 余弦只有 0.991）

#### Scenario: 与上游数值一致

- **WHEN** 同一个真实图像分别经上游 `torch.hub` 模型与本系统主干前向
- **THEN** 两者的 patch token SHALL 逐 token 余弦相似度 ≥ 0.9999
- **AND** 验证脚本 SHALL 把该结论写成可复核的测试

#### Scenario: 运行期不依赖 torch.hub

- **WHEN** 使用人员机器无法访问 GitHub
- **THEN** 导出流程 SHALL 仍能正常完成
- **AND** 它 MUST NOT 在运行期调用 `torch.hub.load`

### Requirement: Baked-Bank ONNX Contract

导出的 ONNX SHALL 把记忆库固化为图内常量，并严格满足 `anomalib_infer.dll` 的输出挑选规则。

#### Scenario: 输入契约

- **WHEN** 导出 ONNX
- **THEN** 输入 SHALL 为单个 `float32` 张量 `[1, 3, H, W]`，H 与 W 均为 14 的倍数且非动态维
- **AND** 通道顺序 SHALL 为 RGB，像素值口径 SHALL 为除以 255 后按 ImageNet 均值方差归一

#### Scenario: 输出契约

- **WHEN** 导出 ONNX
- **THEN** 输出 SHALL 只有两个张量：`anomaly_map` 形状 `[1, 1, gh, gw]` 与 `pred_score` 形状 `[1, 1]`
- **AND** `anomaly_map` 的元素数 MUST 大于 `pred_score`
- **AND** 不得导出任何比 `anomaly_map` 元素数更大的张量（否则 DLL 会把它当作异常图）

#### Scenario: 库分块计算

- **WHEN** 记忆库行数超过 4096
- **THEN** 相似度计算 SHALL 按行分块进行并逐块取最大值
- **AND** 中间张量的元素数 MUST NOT 随库行数线性增长

#### Scenario: 分数口径

- **WHEN** 计算图像级分数
- **THEN** 分数 SHALL 为全网格最近邻余弦距离降序前 1% 的均值
- **AND** 网格固定时该比例对应的整型 `k` SHALL 作为编译期常量，不得在图中做动态取整

### Requirement: AnomalyDINO Metadata Contract

导出的 `metadata.json` SHALL 同时满足 DLL 读取口径与 Web 侧元数据透传口径。

#### Scenario: 必填字段

- **WHEN** 导出完成
- **THEN** metadata SHALL 含 `model_type`、`output_contract`、`normalization`、`input_size`、
  `image_threshold`、`pixel_threshold`
- **AND** 它 SHALL 含 `pred_scores.min` / `pred_scores.max` / `anomaly_maps.min` / `anomaly_maps.max`
  的点号平铺键与其嵌套对象两种写法
- **AND** `normalization` SHALL 为 `imagenet`，`output_contract` SHALL 为 `trtforge_anomalydino_v1`

#### Scenario: 取值范围来自真实标定

- **WHEN** 同时提供了正常与异常标定图
- **THEN** 分数与热力图的 `min` / `max` SHALL 由标定集实测得出
- **AND** 它们 MUST NOT 硬编码为 0 与 1（余弦距离的量纲与 0..1 后处理模型不同）

#### Scenario: 溯源字段

- **WHEN** 导出完成
- **THEN** metadata SHALL 记录 `backbone`、`patch_size`、`grid`、`shots`、`bank_rows`、
  `bank_sha256`、`masking`、`seed`
- **AND** 两个库内容不同的模型 MUST NOT 出现相同的 `bank_sha256`

#### Scenario: 缺少阈值

- **WHEN** 标定失败或未提供标定图
- **THEN** 启动器 SHALL 以 `METADATA_FAILED` 报错
- **AND** 它 MUST NOT 写出缺少 `image_threshold` 或 `pixel_threshold` 的 metadata

### Requirement: AnomalyDINO Launcher Modes

启动器 SHALL 支持 `check-env`、`build-export`、`warm` 三个模式，并以 `TRTFORGE_JSON` 事件回传进度。

#### Scenario: 环境检查

- **WHEN** 以 `check-env` 启动
- **THEN** 启动器 SHALL 报告 torch、CUDA、timm、onnx、onnxruntime 与权重文件的状态
- **AND** 任一必需项缺失时 SHALL 以 `MISSING_DEP` 错误结束，并指出缺失项

#### Scenario: 不支持的模式

- **WHEN** 传入未知模式
- **THEN** 启动器 SHALL 以 `UNSUPPORTED_MODE` 在开始任何计算之前失败

#### Scenario: 进度事件

- **WHEN** 建库、导出、标定各阶段推进
- **THEN** 启动器 SHALL 发出带阶段名与进度的结构化事件
- **AND** 失败时 SHALL 发出带错误码与可读原因的错误事件，且不得抛未捕获异常

### Requirement: Offline Weights

系统 SHALL 让 AnomalyDINO 在完全离线的机器上可用。

#### Scenario: 权重不存在

- **WHEN** 本机没有 DINOv2 权重
- **THEN** 启动器 SHALL 通过现有模型库（modelzoo）机制按清单拉取并校验 sha256
- **AND** 下载中断后 SHALL 支持续传，不得留下半截文件被当作可用权重

#### Scenario: 依赖缺失即停止打包

- **WHEN** 打包完成后检查发现目标环境无法 `import timm`
- **THEN** 打包脚本 SHALL 失败并停止产出安装包

