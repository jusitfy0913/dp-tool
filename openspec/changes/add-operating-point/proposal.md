# Change: 漏检-过杀工作点与良品图验收

## Why

产线验收的第一口径是「漏检率」和「过杀率」，而现在的引擎精度报告只给一个 conf 下的
P/R 和一个 best-F1 点（`6-web/backend/valmetrics.py:246`），回答不了「漏检压到 1% 时，
每张图会误报几次」。同时数据集里没有「良品图」的概念：无标签图一律按「可能漏标」处理
（`6-web/backend/dataset.py:526`），于是过杀率根本无处统计。

## What Changes

- 数据集扫描/导出支持**显式勾选良品目录**（OK 图）：这些图 MUST 带标注时报错，
  导出时写成空标签，默认只进验收划分，不参与训练。
- `valmetrics.py` 新增 `operating_points()`：按 conf 扫描输出漏检率、每图过杀数、
  良品误报率与逐类明细；并给出「满足目标漏检率的最小 conf」反查。
- `tools/val_engine.py` 报告与终端摘要带曲线与推荐工作点；区分「良品（空标签）」与
  「漏标（无标签文件）」，后者不计入良品误报率。
- 精度卡片显示「漏检-过杀」曲线与关键工作点，可把选定点写进 `<engine>.meta.json`，
  推理页选择该引擎时自动预填 conf 并标注来源。
- `autoexp.py`：`results.tsv` 与 holdout 台账补 `fp_per_image` 列；KEEP 判定增加
  「过杀不劣化」约束（由 P/R 反推的近似值，仅在数据能算出该值时生效）。

## Impact

- 受影响规范：新增 `operating-point` 能力
- 受影响代码：`6-web/backend/{valmetrics,dataset,datasetqc,enginemeta,app,config}.py`、
  `6-web/backend/tools/val_engine.py`、`6-web/frontend/{index.html,js/{pipeline,dataset,infer}.js,css/style.css}`、
  `7-autoresearch-yolo/autoexp.py`
- 不影响：DLL C API、metadata.json 语义、既有划分结果（`ok_dirs` 为空时行为完全不变）
