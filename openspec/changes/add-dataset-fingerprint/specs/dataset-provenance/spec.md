## ADDED Requirements

### Requirement: 数据集指纹计算

系统 SHALL 提供唯一的数据集指纹实现 `6-web/backend/fingerprint.py`，供 Web 后端与
`7-autoresearch-yolo/autoexp.py` 共用。

指纹算法 MUST 带版本号，且 MUST 同时产出三个粒度：整批指纹、每个 split 的指纹、
仅标注的指纹。

指纹输入 MUST 为每个 split 的「相对路径 + 文件字节数 + 标注文件原始字节哈希」，
排序后做 sha256。

指纹 MUST NOT 依赖文件修改时间（复制、解压、检出都会改变它，会产生假的「数据变了」）。
指纹 MUST NOT 读取图片内容。

相对路径 MUST 以该 split 的 images 目录为基准，使同一批数据导出到不同目录时指纹相同。

#### Scenario: 同一批数据导出到两个目录
- **WHEN** 同一数据集分别导出到 `D:\out_a` 与 `E:\out_b`
- **THEN** 两处的 `dataset_fingerprint`、三个 `split_fingerprints`、`label_fingerprint` 全部相同

#### Scenario: 修改一个标注
- **WHEN** 把某张 train 图片的标注文件里一个类别编号由 `0` 改成 `1`
- **THEN** `label_fingerprint`、`split_fingerprints.train`、`dataset_fingerprint` 三者都变化

#### Scenario: 只改验证集
- **WHEN** 只删除一张 val 图片
- **THEN** `split_fingerprints.val` 与 `dataset_fingerprint` 变化，`split_fingerprints.train` 不变

#### Scenario: 重命名图片
- **WHEN** 把一张图片改名而内容不动
- **THEN** 指纹变化（路径计入指纹，用户已确认该语义）

#### Scenario: 算法版本可区分
- **WHEN** 读取任意指纹记录
- **THEN** 其中 MUST 有 `algo` 字段（当前为 `v1`），不得把不同版本的指纹当作同一份

#### Scenario: 指纹计算失败不阻断
- **WHEN** 标注目录不可读或 yaml 无法解析
- **THEN** 返回带 `error` 的结果而不是抛出异常，导出/训练流程照常继续

### Requirement: 导出产物携带数据集清单

数据集导出 SHALL 在输出目录写 `manifest.json`，与 `dataset.yaml` 同级。

manifest MUST 包含：算法版本、三个粒度的指纹、各 split 数量、类别表、任务类型、
来源路径与来源类型、划分方式、导出时间、程序版本。

同一输出目录已存在 manifest 且指纹相同时，导出结果 MUST 带 `unchanged: true`。

manifest MUST NOT 改变 `dataset.yaml` 的任何字段（Ultralytics 会拒绝未知键）。

#### Scenario: 首次导出
- **WHEN** 把 1200 张图导出为 YOLO 格式
- **THEN** 输出目录出现 `manifest.json`，其中 `counts.train + counts.val` 等于 1200，
  且 `dataset_fingerprint` 为 16 位十六进制

#### Scenario: 内容未变的重复导出
- **WHEN** 同一批数据再次导出到同一目录
- **THEN** 新 manifest 的指纹与旧的一致，接口返回 `unchanged: true`

#### Scenario: 导出到新目录
- **WHEN** 同一批数据导出到一个空目录
- **THEN** 返回 `unchanged: false`，指纹与旧目录一致

### Requirement: 训练记录数据集指纹

启动 YOLO 训练时，系统 SHALL 计算该次训练所用 `dataset.yaml` 的指纹，并写入任务记录。

训练完成（或失败）后，训练产物目录 SHALL 存在 `train_meta.json`，记录数据路径、
三个指纹、各 split 数量与计算时间。

任务台账 `runtime/runs.jsonl` SHALL 记录 `data_fp` 短码（整批指纹前 8 位）。

训练页 MUST 在开训前显示本次将使用的指纹短码与 train/val 数量。

#### Scenario: 训练前可见
- **WHEN** 训练页选定 `dataset.yaml` 并填入 batch
- **THEN** 页面显示「本次将使用 `<8 位短码>`（N train / M val）」

#### Scenario: 训练完成后可追溯
- **WHEN** 一次训练结束
- **THEN** 该 run 目录的 `train_meta.json` 中 `dataset_fingerprint` 与开训前页面显示的一致，
  且 `runs.jsonl` 该任务的 `data_fp` 是该指纹的前 8 位

#### Scenario: 旧模型无指纹
- **WHEN** 查看升级前完成的训练产物
- **THEN** 界面显示「未记录」，不报错，也不伪造指纹

### Requirement: 实验台账携带数据集指纹

`7-autoresearch-yolo` SHALL 在每个实验开跑前计算数据指纹，并写入
`experiments/exp_NNNN.json` 的 `dataset` 块与 `results.tsv` 的 `data_fp` 列。

`best.json` 的 `evidence` MUST 记录该次 KEEP 所依据的数据指纹。

`splits/_manifest.json` MUST 记录 `data_fingerprint`，使 holdout 验收结论能对应到具体数据。

新增列 MUST 由既有 `ensure_tsv()` 按列名迁移，旧文件备份为 `.bak`，旧行补空。

#### Scenario: 新增列不破坏历史
- **WHEN** 升级后第一次写入 `results.tsv`
- **THEN** 旧文件备份为 `results.tsv.bak`，旧行的 `data_fp` 为空，其余列不错位

#### Scenario: best.json 可追溯
- **WHEN** 某次实验被 KEEP 并更新 `best.json`
- **THEN** `best.json` 的 `evidence` 中含 `data_fingerprint`，与对应 `exp_NNNN.json` 一致

#### Scenario: holdout 数据被改动
- **WHEN** `splits/_manifest.json` 记录的指纹与当前实际数据不一致
- **THEN** 工具 MUST 明确提示「holdout 数据已变化，之前的验收结论不可沿用」

#### Scenario: 共享实现缺失
- **WHEN** `autoexp.py` 找不到 `fingerprint.py`
- **THEN** `dataset` 块标为 `unavailable`，实验照常进行，不中断

### Requirement: 引擎元数据可溯源

TensorRT 引擎构建成功后，系统 SHALL 在引擎文件旁写 `<engine>.meta.json`。

内容 MUST 包含：ONNX 路径与其 sha256、引擎路径、精度、trtexec 参数、构建时间、
程序版本；能确定时 MUST 包含来源训练 run 与数据集指纹。

写元数据失败 MUST NOT 影响构建结果。

读取方 MUST 把缺失的元数据视为「未记录」，而不是错误。

#### Scenario: 构建后生成元数据
- **WHEN** 把一个 ONNX 编译为 fp16 引擎
- **THEN** 同目录出现 `model_fp16.engine.meta.json`，其中 `onnx_sha256` 与实际 ONNX 文件一致

#### Scenario: 元数据与训练对上
- **WHEN** 该 ONNX 来自一次训练导出的权重
- **THEN** 元数据里的 `dataset_fingerprint` 与该训练 run 的 `train_meta.json` 一致

#### Scenario: 写元数据失败
- **WHEN** 目标目录只读
- **THEN** 引擎构建仍返回成功，日志中记录一条元数据写入失败

### Requirement: 指纹在界面上可见

数据集页、训练页与自动实验列表 MUST 显示数据集指纹短码。

自动实验列表 MUST 按指纹分组区分，跨指纹的行 MUST 明确提示不可直接横向比较。

指纹旁 MUST 注明算法版本与覆盖范围（路径 + 大小 + 标注），避免被误当作图片级校验。

#### Scenario: 自动实验列表分组
- **WHEN** 列表中同时存在指纹 `a3f9c1e2` 与 `7b2d4f10` 的实验
- **THEN** 两组视觉上可区分，且跨组的比较提示可见

#### Scenario: 长指纹可展开
- **WHEN** 用户点击短码
- **THEN** 显示完整 16 位指纹与算法版本
