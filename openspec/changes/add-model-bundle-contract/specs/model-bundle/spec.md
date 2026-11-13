## ADDED Requirements

### Requirement: Bundle Directory Shape

一个可交付模型 SHALL 以一个自描述目录 `<名字>.tfbundle/` 表达，目录内 MUST 含 `manifest.json` 与至少一个 `role` 为 `model` 的模型文件；标签、异常检测 metadata、来源元数据 SHALL 在同一目录内，不得散落到目录外。

#### Scenario: 目录自包含

- **WHEN** 用户把 `<名字>.tfbundle/` 整个目录复制到另一台机器的任意路径
- **THEN** 该模型包 SHALL 仍然可被枚举、校验与推理，不得依赖原训练目录中的任何文件

#### Scenario: 必需文件缺失

- **WHEN** 枚举到的目录没有 `manifest.json`，或 `manifest.json` 的 `files` 中没有 `role=model` 的条目
- **THEN** 该目录 SHALL NOT 被当作模型包，枚举结果 SHALL 说明缺什么

#### Scenario: 目录内文件可读

- **WHEN** 模型包位于含中文或空格的路径下
- **THEN** 枚举、校验与推理 SHALL 全部正常，不得出现编码错误

#### Scenario: 来源说明随包携带

- **WHEN** 模型包由引擎构建产生
- **THEN** 目录内 SHALL 含 `origin.meta.json`，记录原 ONNX 路径与哈希、训练运行标识、数据集指纹与构建命令

### Requirement: Manifest Schema Versioning

`manifest.json` MUST 含 `schema` 字段，当前版本为 `trtforge.bundle.v1`；读取方 SHALL 拒绝不认识的版本并给出可读原因，不得按「尽量解析」的方式继续。

#### Scenario: 版本匹配

- **WHEN** `manifest.json` 的 `schema` 等于 `trtforge.bundle.v1`
- **THEN** 读取方 SHALL 按该版本解析全部字段

#### Scenario: 版本不识别

- **WHEN** `schema` 为其它取值或缺失
- **THEN** 校验结论 SHALL 为 `block`，原因 SHALL 说明当前软件支持的版本

#### Scenario: 未知字段向前兼容

- **WHEN** `manifest.json` 含当前版本未定义的额外字段
- **THEN** 读取方 SHALL 忽略这些字段并正常解析，不得报错

#### Scenario: 清单必填字段

- **WHEN** `manifest.json` 缺少 `kind`、`task`、`runtime` 或 `files` 中任一项
- **THEN** 校验结论 SHALL 为 `block`，并逐项列出缺失字段名

### Requirement: Bundle Integrity Verification

校验 SHALL 逐个比对 `files[]` 中登记的 `sha256` 与实际文件内容，并 SHALL 拒绝含绝对路径、盘符或 `..` 的包内路径。

#### Scenario: 哈希一致

- **WHEN** 包内每个文件的 sha256 与清单登记值一致
- **THEN** 完整性检查 SHALL 通过

#### Scenario: 哈希不一致

- **WHEN** 某个文件的 sha256 与清单不一致
- **THEN** 校验结论 SHALL 为 `block`，原因 SHALL 给出该文件路径、登记哈希与实际哈希

#### Scenario: 路径越界

- **WHEN** `files[].path` 是绝对路径、含盘符或含 `..`
- **THEN** 校验结论 SHALL 为 `block`，且读取方 MUST NOT 打开该路径

#### Scenario: 文件缺失

- **WHEN** 清单登记的文件在包目录内不存在
- **THEN** 校验结论 SHALL 为 `block`，原因 SHALL 指明缺失文件名

### Requirement: Runtime Compatibility Verdict

校验 SHALL 比对清单记录的运行时环境与当前机器，并把结论分为 `ok`、`warn`、`block` 三档；`block` MUST 只用于确定性错误，`warn` 不得阻止推理。

#### Scenario: 全部匹配

- **WHEN** 品牌型号、compute capability 与 TensorRT 主版本均与当前机器一致
- **THEN** 结论 SHALL 为 `ok`

#### Scenario: TensorRT 主版本不同

- **WHEN** 清单记录的 TensorRT 主版本与当前环境不同
- **THEN** 结论 SHALL 为 `block`，原因 SHALL 说明需在本机重新编译引擎

#### Scenario: TensorRT 小版本不同

- **WHEN** 主版本相同而小版本不同
- **THEN** 结论 SHALL 为 `warn`，原因 SHALL 说明可能加载失败及确认方法

#### Scenario: GPU 架构不同

- **WHEN** 清单记录的 compute capability 与当前 GPU 不同
- **THEN** 结论 SHALL 为 `warn`，原因 SHALL 给出记录的型号与本机型号

#### Scenario: 运行时字段缺失

- **WHEN** 清单由旧版本产生、`runtime` 中缺少 `trt_version` 或 `compute_cap`
- **THEN** 结论 SHALL 为 `warn`（不得为 `block`），并提示重新构建模型包可补全

#### Scenario: 种类与推理模式冲突

- **WHEN** 用户以异常检测模式提交一个 `kind` 为 `yolo` 的模型包
- **THEN** 结论 SHALL 为 `block`，原因 SHALL 指明该包的种类与实际请求的模式

#### Scenario: 任务与推理模式冲突

- **WHEN** 用户以分割模式提交一个 `task` 为 `detect` 的模型包
- **THEN** 结论 SHALL 为 `block`，原因 SHALL 列出该包支持的任务

### Requirement: Bundle Build From Loose Artifacts

后端 SHALL 提供接口把现有散装产物构建成模型包，构建 SHALL 采集运行时环境与各文件哈希，且 MUST NOT 修改或删除原始产物。

#### Scenario: 从引擎构建

- **WHEN** 客户端提交引擎路径、标签文件、可选的 metadata 与包名
- **THEN** 后端 SHALL 在指定父目录下创建 `<包名>.tfbundle/`，写入 manifest、模型文件、标签、metadata 与来源元数据

#### Scenario: 采集运行时环境

- **WHEN** 构建以 `.engine` 为模型文件
- **THEN** manifest 的 `runtime` SHALL 含 TensorRT 版本、CUDA 版本、GPU 名称、compute capability、输入尺寸、是否动态 shape 与 ONNX 是否转置

#### Scenario: 以 ONNX 构建

- **WHEN** 构建以 `.onnx` 为模型文件
- **THEN** manifest 的 `runtime.format` SHALL 为 `onnx`，且 SHALL NOT 记录 TensorRT 与 GPU 字段

#### Scenario: 原始产物不受影响

- **WHEN** 构建完成
- **THEN** 被引用的原引擎、ONNX、标签与 metadata 文件 SHALL 保持原路径与原内容不变

#### Scenario: 同分区不占额外空间

- **WHEN** 包目录与模型文件在同一分区
- **THEN** 后端 SHALL 优先使用硬链接，manifest 的 `linked` SHALL 为 `true`

#### Scenario: 跨分区退回复制

- **WHEN** 包目录与模型文件不在同一分区
- **THEN** 后端 SHALL 复制文件，manifest 的 `linked` SHALL 为 `false`，界面 SHALL 显示实际占用体积

#### Scenario: 目标已存在

- **WHEN** 目标包目录已存在
- **THEN** 后端 SHALL NOT 静默覆盖，且 SHALL 要求客户端显式确认覆盖

#### Scenario: 引擎构建后自动造包

- **WHEN** 引擎构建成功且自动造包开关开启
- **THEN** 后端 SHALL 自动为该引擎生成模型包，造包失败 SHALL NOT 把已经成功的构建判为失败

### Requirement: Legacy Loose Artifact Compatibility

散装产物 SHALL 继续支持推理，读取方 MUST NOT 要求用户必须先构建模型包。

#### Scenario: 散装文件照常推理

- **WHEN** 客户端以引擎路径或 ONNX 路径提交推理，未提供模型包标识
- **THEN** 后端 SHALL 按现有逻辑完成推理，行为与本变更之前一致

#### Scenario: 枚举结果标记来源

- **WHEN** 客户端请求模型列表
- **THEN** 属于某个模型包的条目 SHALL 附带该包标识，散装条目 SHALL 标注为兼容来源

#### Scenario: 不因缺包而报错

- **WHEN** 模型产物目录里没有任何模型包
- **THEN** 模型列表 SHALL 正常返回散装条目，不得返回空列表或错误
