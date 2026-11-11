# Change: 显存预检改实测表并自校准；混合精度（AMP）实际状态可见并可覆盖

## Why

`7-autoresearch-yolo/autoexp.py` 的 `preflight_vram()` 现在是一条写死的经验公式：

```python
estimate = 2600.0 + 240.0 * batch * (imgsz / 640.0) ** 2
```

问题有两个，都已在 v1.33.2 的真实实验里出现过：

1. **公式与实测脱节**。台账里 batch=8 @640 的 `peak_vram_mb` 记的是 1769.5，
   而公式给同一组合估 4520；到了 960/batch=4，公式估 4760 就放行，实际仍 OOM。
   机器的显存状态（推理常驻进程、桌面占用）与 `imgsz` 是**线性还是平方**增长，
   不是一条固定系数能覆盖的。
2. **混合精度是个隐形变量**。本机 NVIDIA T1000 在 Ultralytics 的 AMP 黑名单里
   （`utils/checks.py` 的 `check_amp()` 用正则匹配 `t1000`），训练被静默降为 fp32：
   显存占用显著更高、速度约慢一倍，而页面上看不出任何区别。于是「同一个 batch，
   昨天放行今天 OOM」这种问题无从解释，实验之间也不再可比。

## What Changes

- 新增 `6-web/backend/vram.py`，作为**显存预检的唯一实现**，Web 训练页与 autoexp 共用：
  - 实测峰值表 `7-autoresearch-yolo/experiments/vram_samples.json`，按
    `(imgsz, batch, precision)` 记录历史最大峰值（`max_memory_reserved` 优先）；
  - 精确命中：放行条件 `free >= peak * 1.15 + 400MB`；
  - 只有更小 batch 的样本：按 batch 线性外推；只有更大 batch 的样本：用它的实测值作上界；
  - 什么都没跑过：回落到公式，系数由 240 提到 300（`2600 + 300 * batch * (imgsz/640)^2`）；
  - 每次训练结束自动记录样本 = **自校准**，不需要人维护这张表。
- 新增 `6-web/backend/ampcheck.py`：复刻 Ultralytics 8.4.192 的 AMP 黑名单判定，
  输出「本机能不能用混合精度、为什么、影响多大」。
- `1-python-yolo/launcher` 训练结束上报两条事件：`amp_state`（请求值 / 实际生效值 / GPU 名）
  与 `vram_peak`（分配 / 保留峰值）。**不改任何训练参数语义**，只增加观测。
- Web 训练接口新增 `GET /api/train/preflight`（算力体检）与 `YoloTrainRequest.amp`
  （`null` = 跟随检测 / `false` = 显式关闭 / `true` = 显式尝试）。
  开训前显存不放行时返回 **409** 并给出可执行建议，前端可「仍然开始」重发。
- 训练页新增「算力体检」块：GPU、可用/需求显存、依据（实测 or 公式）、AMP 状态与
  三态选择；训练结束后在日志里写明**实际**精度与峰值显存。
- autoexp：`preflight_vram()` 委托给 `vram`（签名不变），`results.tsv` 新增 `amp` 列，
  `exp_NNNN.json` 记录 `amp_effective`。
- `APP_VERSION` 1.34.0 → 1.34.1。
- **BREAKING**：无。新接口是增量；`results.tsv` 加列由既有 `ensure_tsv()` 按列名迁移
  （旧文件自动备份为 `.tsv.bak`）。

## Impact

- Affected specs: `web-training-console`（新增「训练前算力体检」）
- Affected code:
  - `6-web/backend/vram.py`（新增）、`ampcheck.py`（新增）、`app.py`、`runners.py`、`config.py`
  - `6-web/frontend/js/train.js`、`index.html`、`css/style.css`
  - `1-python-yolo/launcher/tf_yolo_launcher.py`（只加两条 emit）
  - `7-autoresearch-yolo/autoexp.py`、`run_all_tests.py`
  - `6-web/backend/tools/test_vram_amp.py`（新增测试）
