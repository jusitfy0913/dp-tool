## Context

Anomalib 的训练产物结构由框架自己决定：

```
results/<模型类名>/<name>/v<N>/weights/lightning/{model.ckpt, model.onnx, metadata.json}
```

- `<模型类名>` 是 `Patchcore` / `Padim` / `EfficientAd` 这类类名，不是请求里的 `model`；
- `<N>` 每训一次递增，训练开始前不可知；
- 阈值由 launcher 在导出后用 ONNX 推理校准，写在 `metadata.json`。

同时 ONNX 的 batch 恒为动态轴（`to_onnx` 里写死 `dynamic_axes={"input": {0: "batch_size"}}`），
而 DLL 在 `ai_create` 时按 metadata 的 `input_size` 调 `setBindingDimensions(1x3xHxW)`。

## Goals / Non-Goals

- Goals：一次点击跑完训练→导出→编译→精度报告；产物路径不靠猜；指标与 DLL 判定严格一致；
  失败可定位到具体阶段；跳过训练可反复调编译参数。
- Non-Goals：像素级 PRO/AUPRO；把 `train-export` 拆成多进程；Anomalib 断点续训；
  把最优阈值回写 metadata。

## Decisions

### 1. 延迟派生（`StagePlan.resolve`）而不是预拼路径

**问题**：`build` / `val` 的命令需要 `vN` 路径，但只有 `probe` 阶段结束才知道。

**选择**：`resolve(record) -> {command, expect_file, report_file}`，在阶段开始前调用，
读取 `record["artifacts"]`（`probe` 记录进去的 onnx / metadata / ckpt）。

**否决的方案**：

- 在 `train_export` 之后加一个「识别产物」阶段再回填 —— 需要可变全局状态，且持久化结构要改；
- 让 launcher 输出固定路径 —— 改动训练侧行为，已有产物目录全部失效。

### 2. `train_export` 单进程

launcher 的 `train-export` 一次完成训练、导出、阈值校准。拆成三个进程除了让阶段列表好看，
只会多两次模型加载，并制造「导出用的不是刚训的模型」的可能。因此阶段名就叫 `train_export`，
标签写清楚它包含三件事。

### 3. 动态 batch：允许并自动补形状

| 维度 | 处理 | 理由 |
| --- | --- | --- |
| batch `-1` | 允许，补 `min=1 / opt=1 / max=max_batch` | DLL 每次只推一张（`setBindingDimensions(1x3xHxW)`），opt=1 才是实际最快 |
| H/W 动态 | 拒绝 | DLL 用 metadata 的 `input_size` 定形状，范围对不上会静默算错 |

`max_batch` 默认 8：给以后批量推理留余量，又不至于让 tactic 搜索范围过大
（实测动态 batch 1…8 的 Patchcore 编译 1016 秒，静态形状约 900 秒）。

### 4. 指标口径以 DLL 为准

DLL（`anomalib_infer_c.cpp`）的判定是
`normalized = clamp01((raw - min) / (max - min)); is_anomaly = normalized >= 0.5`。
由于 `clamp01` 单调且 `(thr - min)/(max - min) + 0.5` 不一定等于 0.5，**不能用「raw > 0.5」**
判断。报告一律按 `raw >= image_threshold` 判，并把「报告判定 vs 引擎 `is_anomaly`」
的不一致张数写进报告：一旦归一化口径漂移，这个数字会立刻不为 0。

AUROC 用 Mann-Whitney U 秩统计并对同分取平均秩：分数大量相同时（异常检测很常见，
比如全 0）按出现顺序排序会让 AUROC 抖动，平均秩才是无偏口径。

### 5. 评估走轻量打分入口

`infer.run_anomalib_score` 只取 `raw_score` / `normalized_score` / `is_anomaly`，
不生成 source / overlay / mask / heatmap 四张 PNG。64 张图 4 秒跑完；
若复用单张推理入口，光是写图和编码就要几十秒，而且产物没人看。

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 动态 batch 让编译变慢 | build 涨到 ~17 分钟 | 文档写明「只推单张可把 `max_batch` 设 1」；进度条有心跳不会假死 |
| 最优 F1 阈值被误当默认值 | 上测试集过拟合 | 卡片标注「仅参考，不回写 metadata」 |
| launcher 输出目录结构变化 | `_locate_anomaly_artifacts` 找不到 | 定位失败时直接报出找过的根目录，不猜路径 |
| 跳过训练时目录选错版本 | 编译了旧模型 | 要求显式填 `artifacts_dir`，并在产物表显示实际 `run_dir` 与 `version` |

## Migration Plan

纯新增能力，无数据迁移。旧记录（v1.17 的 YOLO 串联）在 `runtime/pipelines/*.json` 里
带 `kind=yolo`，前端按 `kind` 决定渲染哪套参数与指标卡片，互不影响。

## Open Questions

- 像素级 PRO/AUPRO 需要的 `ground_truth` 目录约定，等真实数据到位再定。
- 是否支持 EfficientAd 的 imagenet 辅助数据在串联里显式指定（目前透传 `imagenet_dir`）。
