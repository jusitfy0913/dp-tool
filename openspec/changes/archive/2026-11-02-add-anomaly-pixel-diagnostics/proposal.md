# Change: 异常检测加免掩码的像素级诊断

## Why
现在异常检测的精度只有一个数字：**图像级 AUROC**——「这张图有没有缺陷」。
但产线上更常见的失败不是判错 OK/NG，而是**判对了却说不出缺陷在哪**：分数报了 NG，
热能图却一片平坦，或者好板上到处点红点。

这类问题在现有报告里完全不可见：`anomaly_val_report.json` 的 `pixel` 块是写死的常量
`{"available": false, "reason": "本版只做图像级"}`,而不是算出来的。

而 `anomalymetrics.pixel_auroc_from_histograms()` 早已写好却**没有任何调用方**——
它缺两样东西：异常图没从 DLL 搬出来，以及像素级真值（掩码）根本不存在
（`Folder` 数据模块只配了 `normal_dir` / `abnormal_dir`，没有掩码目录）。

## What Changes
- `infer.run_anomalib_score()` 增加 `want_map` 开关：开启时把 `normalized_map`
  拷成 numpy 数组一起返回（不编码 PNG，评估链路不能被编码拖慢）。
- `anomalymetrics.py` 新增两个纯函数：`accumulate_map_histogram()`（把一张异常图
  累加进正/负两张直方图）与 `pixel_diagnostics()`（好图乱指率 / 定位面积 / 指不出位置）。
- `tools/val_anomaly.py` 的 `pixel` 块改为真实计算，模式记为 `diagnostic`；
  异常图缺失或尺寸为 0 的图单独计数，不影响其余图。
- 推理页精度卡片增加 3 行像素指标，「指不出位置」大于 0 时标红。
- **不做**真像素级 AUROC / PRO / AUPRO：三者都需要 ground_truth 掩码，
  当前数据集没有。掩码目录的两种约定（显式 `--mask-dir` 优先、
  其次 MVTec 风格 `ground_truth/<类别>/*_mask.png`）已记录为下一版接口约定。

## Impact
- 影响能力：`web-anomalib-pipeline`（精度报告的口径）
- 影响代码：
  - `6-web/backend/anomalymetrics.py`
  - `6-web/backend/infer.py`
  - `6-web/backend/tools/val_anomaly.py`
  - `6-web/frontend/js/pipeline.js`
  - `6-web/backend/tools/test_anomaly_pixel.py`（新增）
- 不涉及：DLL 契约、`metadata.json` 结构、图像级判定口径、串联编排逻辑
- 报告向后兼容：`pixel.available` / `pixel.reason` 字段保留，旧前端不会读崩
