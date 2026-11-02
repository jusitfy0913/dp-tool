# Change: Anomalib 全流程串联与引擎精度验证

## Why

v1.17 的串联只覆盖 YOLO。异常检测（Patchcore / PaDiM / EfficientAd）仍然要靠人接力：
训练页跑 launcher → 去 `results/<类名>/<name>/vN/weights/lightning/` 里翻版本号 →
手敲 trtexec（还要自己知道输入是 `input`）→ 最后**没有任何精度数字**，
阈值是 0.5 还是校准值、漏检几张、误报几张，全靠打开几张热力图「感觉还行」。

三个具体问题：

- **产物路径训练前不可知**。版本目录 `vN` 每次训练递增，编排器若按 `v0` 拼路径，
  第二次训练就会编译上一次的旧模型——旧模型同样能编译、同样能出分数，错得毫无提示。
- **导出 ONNX 的 batch 恒为动态轴**。Anomalib 的 `to_onnx` 固定
  `dynamic_axes={"input": {0: "batch_size"}}`，输入形状是 `[-1, 3, H, W]`；
  把它当「非法形状」拒绝，会让训练成功的模型卡在编译前。
- **图像级精度没有口径**。AUROC、误报率、漏检率都需要与 DLL 的判定逻辑严格对齐，
  否则报告与线上行为对不上，比没有数字更危险。

## What Changes

- **新增异常检测四段链**：`train_export`（训练 + ONNX 导出 + metadata 阈值校准）→
  `probe`（校验 metadata 字段与 ONNX 形状）→ `build`（trtexec 编译）→
  `val`（引擎逐张推理出报告）。
- **`StagePlan.resolve`**：编排层新增「延迟派生命令」，在阶段真正开始前按
  `record["artifacts"]` 拼命令与产物路径，解决 `vN` 训练前不可知的问题。
- **callable 多产物记录**：一段可返回 `{"artifact": ..., "artifacts": {...}}`，
  同时记录 onnx / metadata / ckpt，下游不再重复 glob。
- **动态 batch 支持**：`probe` 允许 batch 维为动态、拒绝 H/W 动态；
  `build` 自动补 `--minShapes/--optShapes/--maxShapes`（opt=1 与 DLL 一致，
  max 由 `max_batch` 控制，默认 8）。
- **新增 `backend/anomalymetrics.py`**：CLAMP01 归一化、Mann-Whitney U 图像级 AUROC
  （同分取平均秩）、阈值混淆矩阵（TP/FP/TN/FN、P/R/F1、FPR、FNR）、最优 F1 阈值搜索、
  直方图 AUROC。
- **新增 `tools/val_anomaly.py`**：走 `anomalib_infer.dll` 在测试集上逐张打分，
  产出 `anomaly_val_report.json` 与 `.csv`（`utf-8-sig`）。
- **`infer.run_anomalib_score`**：轻量打分入口，只取原始分数与判定，
  不生成 / 不拷贝 source / overlay / mask / heatmap 四张 PNG。
- **接口**：`POST /api/pipeline/anomalib`（复用 `GET /api/pipeline`、`/{id}`、
  `/{id}/stop`、`/{id}/log`）。
- **前端**：串联卡片新增「类型」下拉（YOLO / 异常检测），异常检测参数区与
  精度卡片（AUROC / F1 / 漏检率 / 误报率 / 最优阈值 / 判定不一致告警）。
- 版本升至 `1.18.0`。

## Impact

- Affected specs: 新增 `web-anomalib-pipeline`
- Affected code:
  - 新增 `6-web/backend/anomalymetrics.py`、`6-web/backend/tools/val_anomaly.py`
  - 修改 `6-web/backend/pipeline.py`（`resolve` + callable 多产物）、
    `6-web/backend/infer.py`（`run_anomalib_score`）、
    `6-web/backend/app.py`（`/api/pipeline/anomalib`、产物定位、metadata 校验、动态 batch）
  - 前端：`index.html`（`#pipeKind` / `#pipeAnomalyOptions`）、`js/pipeline.js`（双模式）
  - 测试：`test_pipeline_resolve.py`（26 项）、`test_anomaly_metrics.py`（31 项）

## Non-Goals

- 不做像素级 PRO / AUPRO：需要 ground_truth 掩码，成本远高于图像级；
  报告里明确 `pixel.available=false` 并写原因，而不是给空白数字。
- 不把 `train_export` 拆成三个阶段：launcher 一个进程就完成训练、导出与阈值校准，
  拆开只会多两次模型加载，并把「导出的不是刚训的模型」这类错埋进编排层。
- 不做 Anomalib 的断点续训：launcher 侧尚未支持，串联层不擅自模拟。
- 不回写 `best_f1_threshold` 到 metadata：在测试集上搜出来的阈值会过拟合，
  只作为参考显示。

## 验收

- `run_all_tests.py` 20 个脚本全绿；前端 `test_contrast.mjs` 与 `node --check` 通过。
- 真链路（bottle，Patchcore / 256 / fp16，20 正常 + 44 异常）：
  `train_export` 199.7 秒 → `probe` 0.7 秒 → `build` 1016 秒 → `val` 4.0 秒，整链 `success`；
  报告 image AUROC 1.000、F1 1.000、FPR 0.000、FNR 0.000、`decision_mismatch=0`。
- 首轮真实缺陷回归：动态 batch 被 `probe` 误判为非法形状，训练 199 秒后失败；
  修复后新增 4 条断言（动态 batch 形状、max_batch 夹取、静态不缺参数、H/W 动态拒绝）。
- `skip_train` 路径：用同一份 `weights/lightning` 只跑 `probe → build → val`，
  1020.95 秒 `success`，证明可反复调编译参数而不必重训。
