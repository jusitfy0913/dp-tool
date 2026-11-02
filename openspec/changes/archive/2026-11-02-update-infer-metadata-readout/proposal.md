# Change: 推理页回显 metadata 推荐阈值

## Why
勾选「沿用 metadata 默认阈值」后，两个阈值输入框只被置灰，框里留着默认 `0.5`。
`metadata.json` 里记录的推荐值在界面上没有任何出口，用户必须先跑一次推理、
再去结果行看 `（metadata）` 才知道实际会用多少——推荐值实际是被藏起来的。
现有 spec 的「显示生效阈值」场景只约束了**推理之后**的结果展示，没有约束**推理之前**的可读性。

## What Changes
- 新增只读接口 `GET /api/infer/anomalib/metadata?path=…`：读取 `metadata.json`，
  返回 `image_threshold` / `pixel_threshold` / `model_type` / `input_size` /
  `normalization` / `pred_scores` / `anomaly_maps` / `sample_accuracy`。
  **只读文件、不建立引擎句柄**，因此可以随路径变化随时刷新。
- `infer.read_metadata_thresholds()` 扩展返回上面这些附加字段（纯加法，不影响既有调用方）。
- 推理页新增「metadata 推荐值」卡片，按 4 位小数展示两个阈值与模型/输入尺寸，
  并标注来源（有 `calibration` 区块 → 「已校准 N%」，否则「文件原值」）。
- 沿用状态下把推荐值写进只读框；取消勾选后保留推荐值作为手动覆盖的起点，
  提示语并列显示 metadata 原值。
- 读取失败（路径不存在 / 坏 JSON / 越界）时清空数值并显示原因，不得沿用上一次的数字。
- 两个阈值滑块步长 `0.01` → `0.0001`，使 `0.172` 这类校准值可被手动复现。

## Impact
- 影响能力：`web-infer-console`
- 影响代码：
  - `6-web/backend/app.py`（新接口）
  - `6-web/backend/infer.py`（`read_metadata_thresholds` 扩字段）
  - `6-web/frontend/index.html` / `js/infer.js` / `js/api.js` / `css/style.css`
  - `6-web/backend/tools/test_metadata_readout.py`（新增 37 项验收）
- 不涉及：推理判定口径、DLL 侧调用契约、句柄缓存策略
