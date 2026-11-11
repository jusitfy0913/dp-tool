# Change: 数据集体检——底板分组、按组切分与标注质检

## Why

批次 2.1 的数据集指纹回答了「这次和上次是不是同一份数据」，但回答不了「这份数据本身干不干净」。
两类问题都不会报错，只会让指标虚高或让模型学错：

1. **重复/同源图泄漏**。`splits/pcb`（692 张）实测：逐字节重复 0 组，但 64 位 dHash 汉明距离
   ≤6 时 692 张塌成 **10 组**，分组结果与文件名前缀 `01_ / 04_ / 06_…` 完全一致；
   全分辨率复核显示同前缀两张图 `MAE=0.04/255`、仅 0.035% 像素差大于 40——同一张底片配不同缺陷斑块。
   **10 个底板组全部同时出现在 train / tune / holdout**，holdout 的 42 张 100% 能在 train 里找到同底板兄弟。
   换言之，2.1 加固过的 holdout 纪律在这份数据上「独立」不成立。
2. **标注问题被静默吞掉**。`dataset.py` 解析标签时，字段数不对或数值解析失败的行直接 `continue` 丢弃，
   页面只显示「空标注 N 张」，用户无法区分「真没缺陷」与「标签写坏了」。

## What Changes

- 新增 `6-web/backend/datasetqc.py`：一次解码同时产出**底板分组**与**标注质检**报告，按
  `路径 + 大小 + mtime` 缓存到 `runtime/dsqc_cache.json`。
- 分组判定改为**分级**，不再用单一 dHash 阈值当「重复」：
  - A 级 精确重复（文件 sha256 相同）——唯一允许删的级别
  - B 级 同底板分组（256 位 dHash + 32×32 灰度差复核）——**只分组，绝不据此删除**
  - 「同组 + 同标注」只作提示：实测无法与「同位置的不同样本」区分
- 新增 `POST /api/dataset/qc`：只读体检接口，返回分组、跨划分泄漏、标注问题与规则计数。
- 划分新增**按组**模式：`group_by: none | prefix | dir`，同一底板组只进一个 split；比例无法精确达成时
  如实报告实际张数，不做静默取舍。
- `dataset.py` 的 `preview()` / `run_export()` 支持 `group_by`，manifest 记录 `split_mode=group:prefix`
  与 `group_signature`（组键 + 各组成员数）。
- 新增 `7-autoresearch-yolo/labelqc.py` CLI，供无人值守场景复检；`autoexp.py` 增可选 `--qc-gate`。
- 数据集页新增「体检」面板：分组列表、跨划分泄漏红色警告、按规则折叠的标注问题。
- 版本号提升到 `1.34.3`。

**明确不做**：

- 不自动修标注（会静默改数据）。
- 不因质检失败阻断导出/训练（用户已确认只报告）。
- 不因发现同底板跨划分而自动重划数据（用户已确认只警告）。

## Impact

- 受影响规范：`dataset-hygiene`（新增）
- 受影响代码：
  - `6-web/backend/datasetqc.py`（新增）
  - `6-web/backend/dataset.py`（分组切分、preview/run_export 参数、manifest 字段）
  - `6-web/backend/app.py`（`/api/dataset/qc`、scan/export 参数）
  - `6-web/frontend/js/dataset.js`、`index.html`、`css/style.css`
  - `7-autoresearch-yolo/labelqc.py`（新增）、`autoexp.py`
- 性能：692 张首次约 6~10 秒，命中缓存后 <0.3 秒；不读像素以外的任何外部资源，不占 GPU。
