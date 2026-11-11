# Change: 给数据集算指纹，并把指纹写进导出产物、训练记录、实验台账与引擎元数据

## Why

现在有三件事无法回答：

1. **这个模型是用哪批图训练的**。训练目录里只有一个 `data: D:\...\dataset.yaml` 路径。
   图片被重新标注、增删、重划分之后，路径没变但内容已经不同，事后无法分辨。
2. **两次实验能不能直接比**。`7-autoresearch-yolo` 的台账只记参数与指标。换了数据集后
   「`0.4109` 对 `0.4150`」到底是参数赢了还是数据变了，无从判断。
3. **线上引擎来自哪里**。`.engine` 是纯二进制，精度掉了只能靠人回忆它对应哪个 ONNX、
   哪次训练、哪批数据。

附带一个问题：`splits/_manifest.json` 只记录**文件名列表 + seed**。holdout 的图片被改过
标注后，之前「holdout 验收通过」的结论已经失效，但没有任何标记。

## What Changes

- 新增 `6-web/backend/fingerprint.py`，作为**指纹计算的唯一实现**（Web 与 autoexp 共用）：
  - 指纹算法 `v1`：对每个 split 取 `排序后的 "<split 内相对路径>\t<文件字节数>\t<标注文本哈希>"`，
    整体 sha256 取前 16 位；
  - 产出三个粒度：`dataset_fingerprint`（整批）、`split_fingerprints`（三个 split 各自）、
    `label_fingerprint`（只算标注，用来区分「模型变了」还是「标注变了」）；
  - **路径计入指纹**：重命名图片即视为数据变化（用户确认）；
  - **不做图片内容哈希**：只读 `stat` 与标注文本，目标 5000 张 ≤ 2 秒（用户确认）；
  - 算法带 `algo: "v1"` 版本号，以后换算法不会把新旧指纹混为一谈。
- 导出产物新增 `<输出目录>/manifest.json`：三个指纹、各 split 数量、类别表、来源路径与
  kind、划分方式、导出时间、程序版本。**同目录重复导出且指纹相同时返回 `unchanged=true`**，
  便于「内容没变就别重导」。
- 训练：开训前算一次指纹写进 `run.meta`，训练目录写 `train_meta.json`，任务台账
  （`runtime/runs.jsonl`）带短码。
- autoexp：`exp_NNNN.json` 增 `dataset` 块；`results.tsv` 新增 `data_fp` 列（由既有
  `ensure_tsv()` 按列名迁移，旧文件备份 `.bak`）；`best.json` 记录证据行指纹；
  `splits/_manifest.json` 增 `data_fingerprint`。
- 引擎构建完成后在 `.engine` 旁写 `<engine>.meta.json`：来源 ONNX 与其 sha256、weights、
  训练 run、数据集指纹、trtexec 参数、编译时间。**不改 DLL 加载逻辑**（纯旁挂只读文件）。
- Web 界面：数据集页显示完整指纹；训练页显示「本次将使用 `a3f9c1e2…`（1234 train / 156 val）」；
  自动实验列表按指纹分组着色，跨指纹的行标注「数据不同，不可直接横比」。
- 新增只读接口 `GET /api/dataset/fingerprint?source=<yaml>`，供界面与外部脚本查询。
- `APP_VERSION` 1.34.1 → 1.34.2；autoexp v0.2.3 → v0.2.4。

## Impact

- Affected specs: `dataset-provenance`（新增能力）
- Affected code:
  - `6-web/backend/fingerprint.py`（新增）、`dataset.py`、`app.py`、`runners.py`、`pipeline.py`
  - `6-web/frontend/index.html`、`js/dataset.js`（或等价模块）、`js/train.js`、`js/autoexp*`
  - `7-autoresearch-yolo/autoexp.py`、`run_all_tests.py`、`experiments/*.json`（新增字段）
  - `6-web/backend/tools/test_dataset_fingerprint.py`（新增测试）
- **BREAKING**：无。所有新字段都是增量；旧模型、旧实验没有指纹时显示「未记录」，不回填
  （回填需要当时的原始数据，凭空补只会造假）。
