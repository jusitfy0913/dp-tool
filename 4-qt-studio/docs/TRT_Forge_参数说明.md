# TRT Forge 参数说明

本文档说明 **TRT Forge**（`4-qt-studio`）中 ONNX → TensorRT 引擎构建相关的全部参数：界面控件、TensorRT 含义、**额外参数**文本框支持的 token，以及静态/动态输入的注意点。

---

## 1. 文档与代码对应关系

| 数据结构 | 文件 |
| --- | --- |
| 界面收集的选项 | `include/BuildOptions.h` |
| 构建逻辑与 profile 规则 | `src/BuilderWorker.cpp` |
| 额外参数解析 | `BuilderWorker::parseExtraArgsInto` |

---

## 2. 模型路径（区块 01）

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| **ONNX 模型路径** | 文本 / 浏览 | 待转换的 `.onnx` 文件绝对路径。 |
| **输出引擎路径** | 文本 / 另存为 | 生成的 `.trt`（或 `.engine`）保存路径；若 ONNX 已选且输出为空，浏览 ONNX 后会自动生成默认路径（同目录、同名改后缀）。 |

**约束**：两者均非空且 ONNX 文件必须存在，否则构建在开始阶段失败。

---

## 3. 精度与运行时（区块 02）

### 3.1 精度模式（下拉框）

| 界面选项 | 内部枚举 | TensorRT 行为（摘要） |
| --- | --- | --- |
| FP32 - 单精度 | `Precision::FP32` | 不启用 FP16/INT8 标志。 |
| FP16 - 半精度 (推荐) | `Precision::FP16` | 设置 `BuilderFlag::kFP16`。 |
| INT8 - 量化 | `Precision::INT8` | 设置 `BuilderFlag::kINT8`；**强烈建议**配合标定缓存（见 §6）。 |
| BEST - FP16+INT8 最佳 | `Precision::BEST` | 同时设置 `kFP16` 与 `kINT8`。 |
| TF32 - Tensor Float32 | `Precision::TF32` | 仅设置枚举值；**不**自动打开 `kFP32/kFP16/kINT8`。 |

说明：

- **TF32 计算**：TensorRT 的 `BuilderFlag::kTF32` **仅由界面复选框「TF32」**（`opt.tf32`）控制，与精度下拉框第 5 项无直接绑定。若需要 TF32，请**勾选「TF32」**；下拉选「FP32」或「TF32」时若不勾选，则不会启用 `kTF32`。
- **INT8**：在精度为 **INT8** 且未提供 `--calib=` 时，程序会打出**警告**（`BEST` 不会触发同一条 INT8 标定警告，但仍建议为 INT8 部分准备标定）。

### 3.2 运行时模式（下拉框）

| 界面选项 | 内部值 |
| --- | --- |
| full (默认) | `RuntimeMode::Full` |
| lean | `RuntimeMode::Lean` |
| dispatch | `RuntimeMode::Dispatch` |

**当前版本**：该字段仅保存在 `BuildOptions` 中，**构建流程尚未根据此项切换 TensorRT Runtime API**（预留扩展）。实际引擎构建仍走标准 `IBuilder` + `buildSerializedNetwork`。

---

## 4. 构建参数（区块 03）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| **工作空间 (MiB)** | 4096 | 对应 `IBuilderConfig::setMemoryPoolLimit(kWORKSPACE, bytes)`，单位为 MiB，下限在代码中钳制为不小于 64。 |
| **最大 Batch** | 1 | 写入 `BuildOptions::maxBatch`；**当前构建代码未使用此字段**（显式 batch 网络一般由输入维度或 dynamic profile 表达）。预留与文档同步。 |
| **DLA 核心** | -1 | `-1` 表示不使用 DLA；`≥0` 时设置 `DefaultDeviceType::kDLA` 与 `setDLACore`。 |
| **目标设备** | 0 | `cudaSetDevice(device)` 使用的 GPU 索引。 |

### 复选框（chip）

| 名称 | 字段 | TensorRT |
| --- | --- | --- |
| 稀疏化 | `sparsity` | `BuilderFlag::kSPARSE_WEIGHTS` |
| 可重拟合 | `refit` | `BuilderFlag::kREFIT` |
| 版本兼容 | `versionCompatible` | `BuilderFlag::kVERSION_COMPATIBLE` |
| 允许GPU回退 | `gpuFallback` | `BuilderFlag::kGPU_FALLBACK`（默认勾选） |
| TF32 | `tf32` | `BuilderFlag::kTF32` |

---

## 5. 动态形状（区块 04）

| 参数 | 说明 |
| --- | --- |
| **输入名称** | 要与 ONNX 中该输入张量名称一致（如 YOLO 常见为 `images`）。 |
| **最小 / 最优 / 最大形状** | 字符串格式：`维度1x维度2x…`，分隔符支持 `x`、`X`、`,`（如 `1x3x640x640`）。 |

### 5.1 静态 ONNX 与 Profile 规则（重要）

程序在解析 ONNX 之后会检查**是否存在动态维**（任一分量为 `-1`）：

- **全部为静态正整数**（例如 `[1,3,640,640]`）：**不会**添加 `IOptimizationProfile`，并在日志中提示实际静态 shape。此时界面里填的 min/opt/max **不会参与构建**，避免与 TensorRT 报错类似：`kOPT dimensions ... but input has static dimensions`。
- **存在动态维**：为指定 **输入名称** 添加一个 profile，并设置 MIN/OPT/MAX 三组维度。

若需要 **动态 batch**，请在导出 ONNX 时使用动态轴（例如 Ultralytics：`export(..., dynamic=True)`），使输入维上出现 `-1` 或动态符号，再在本工具中填写一致的 min/opt/max。

---

## 6. INT8 与标定

| 字段 | 说明 |
| --- | --- |
| `int8CalibCache` | 通过额外参数 `--calib=路径` 传入；选择 INT8/BEST 时若为空会警告。 |
| `int8CalibDir` | 结构体预留；**当前构建流程未读取目录标定**，仅文档占位。 |

完整 INT8 标定流程（Entropy 等）需自行生成缓存文件后通过 `--calib=` 指定。

---

## 7. 额外参数文本框

多行文本，按**空白**拆成多个 token。支持 `--key=value` 与 `--flag` 形式（单 `-` 前缀同样会去掉一层再解析）。

解析在**构建开始前**执行，会**覆盖**界面上已填的同名含义项（例如 `--workspace=8192` 覆盖工作空间）。

### 7.1 已识别键一览

| Token 形式 | 作用 |
| --- | --- |
| `--fp16` | 精度 → FP16 |
| `--int8` | 精度 → INT8 |
| `--best` | 精度 → BEST |
| `--tf32` | `tf32 = true` |
| `--sparsity` | `sparsity = true` |
| `--refit` | `refit = true` |
| `--versionCompatible` | `versionCompatible = true` |
| `--allowGpuFallback` | `gpuFallback = true` |
| `--workspace=N` | 工作空间 MiB = N |
| `--device=N` | GPU 设备号 = N |
| `--useDLACore=N` | DLA 核心 = N（实现里键名为 `usedlacore`，与 trtexec 写法对应） |
| `--minShapes=输入名:1x3x640x640` | 只取 **最后一个 `:` 之后** 的维度串写入 `minShape` |
| `--optShapes=...` | 同上 → `optShape` |
| `--maxShapes=...` | 同上 → `maxShape` |
| `--shapes=输入名:WxHx...` | 将 **min / opt / max 三者设为同一维度串** |
| `--calib=路径` | INT8 标定缓存路径 |
| `--timingCacheFile=路径` | 写入 `timingCacheFile` 字段 |
| `--saveTimingCache=路径` | 同上（合并到同一字段） |
| `--loadTimingCache=路径` | 同上（合并到同一字段） |
| `--onnx=路径` | 覆盖 ONNX 路径 |
| `--saveEngine=路径` | 覆盖输出引擎路径 |

**未识别的 token** 会在日志中以 `[警告] 额外参数未识别（已忽略）: ...` 输出，**不会**中止构建。

### 7.2 与 trtexec 的差异

- 本程序**不**启动 `trtexec.exe`；仅**模仿**部分命令行开关的**语义**并映射到 `BuildOptions`。
- 许多 trtexec 选项（如 `--useCudaGraph`、详细 profiling）**未实现**，会显示为未识别并忽略。
- `--timingCacheFile` / `--saveTimingCache` / `--loadTimingCache`：值会保存到 `BuildOptions::timingCacheFile`，**当前 `BuilderWorker::runImpl` 尚未把该文件接入 `IBuilderConfig` 的 timing cache API**；若需缩短重复构建时间，需后续版本接入或自行用 trtexec 生成 cache。

---

## 7.5 引擎自检（构建后冒烟测试）

| 控件 | 字段 | 说明 |
| --- | --- | --- |
| 构建后自动自检 | `BuildOptions::testEnabled`（默认 `true`） | 构建成功后立刻跑一次自检；可关闭。 |
| Warmup | `BuildOptions::testWarmup`（默认 10） | 不计时的预热次数。 |
| Iterations | `BuildOptions::testIterations`（默认 100） | 计时迭代次数。 |
| 对已有 .trt 自检 | — | 不构建，直接对选中的 `.trt` 文件做反序列化 + 随机数据推理 + 报告。 |

**自检会做的事**：

1. `IRuntime::deserializeCudaEngine` → `IExecutionContext`；
2. 遍历 `engine->getNbIOTensors()`，列出每个张量的 IO 模式 / dtype / shape / 字节数；
3. 对动态输入按"动态形状/最优形状"或 profile 的 `kOPT` 设置一组 shape；
4. 用 `std::mt19937` 在 host 端生成随机数据，按 dtype 转换后 `cudaMemcpy` 到 device；
5. `cudaStream` + `cudaEvent` 计时 enqueueV3 的 min/avg/max ms 与 QPS；
6. 把第一个输出张量的前 8 个数值拷回 host，作为粗略一瞥；
7. 通过 `cudaMemGetInfo` 报告当前显存占用。

**自检失败可能原因**：引擎与当前 TRT 版本不兼容；动态 shape 无法匹配；显存不足；输入张量大小为 0。

---

## 8. 典型示例

### 8.1 静态 YOLOv8 ONNX（如 `yolov8s.onnx`，输入 `1x3x640x640`）

- 精度选 **FP16**，工作空间按需调大（如 4096）。
- **无需**修改「动态形状」；构建时会自动跳过 profile。
- 额外参数可留空。

### 8.2 动态 batch 导出后的 ONNX

- 导出：`export(..., dynamic=True, ...)`，确保输入存在动态维。
- 在界面填写 **输入名称** 与 **min/opt/max** 三档形状，或使用额外参数：

```text
--minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
```

（注意：解析器取 `:` 后的部分作为维度串，输入名在 `minShapes` 里仅用于与 trtexec 习惯一致，**界面仍以「输入名称」控件为准**。）

### 8.3 命令行式一键覆盖路径与精度

```text
--onnx=D:\models\a.onnx --saveEngine=D:\models\a_fp16.trt --fp16 --workspace=8192
```

---

## 9. 常见错误与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `kOPT dimensions ... but input has static dimensions` | 对静态输入仍添加了 optimization profile | 已在新版本中自动跳过；或使用静态 ONNX 时不要填与静态冲突的 min/opt/max。 |
| INT8 精度异常 | 无标定缓存 | 准备 `--calib=` 缓存或改用 FP16。 |
| 构建很慢 | Tactic 搜索 | 增大工作空间、使用 timing cache（待程序接入或使用 trtexec）。 |

---

## 10. 版本与维护

- 本文档与 **TRT Forge 源码** 同步；若增加新开关，请同时更新 `BuildOptions.h`、`parseExtraArgsInto` 与本文件。
- TensorRT 版本以运行时 `NvInferVersion.h` 为准（界面顶部状态条也会显示主版本信息）。
