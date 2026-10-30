# DefectVision AI Studio — Industrial AI Defect Inspection System

DefectVision AI Studio（中文名：AI 缺陷检测视觉工作站）是面向工业视觉的 AI 缺陷检测桌面平台，
覆盖数据标注、标准数据集导出、YOLO/Anomalib 训练导出、ONNX 转 TensorRT 引擎构建和 C++ 推理验证闭环。
其中引擎构建页直接调用 TensorRT C++ API（**不**派生 `trtexec.exe` 子进程），全程在工作线程内执行，
UI 可实时显示分级日志、进度条、阶段步进、已用时 / 预计剩余时间 / 显存占用等。

> 位置：`4-qt-studio/`
> 不会修改仓库根目录下的 `bin/`、`include/`、`lib/`，也不会改动 `3-cpp-tensorrt/trtexec/` 现有文件。

**图文并茂操作手册（推荐用户）**：见仓库根目录 [`docs/TRTForge_操作手册.md`](../../docs/TRTForge_操作手册.md)（含流程图、示意图与截图占位说明）。

**参数完整说明（界面 + 额外 token + 静态/动态输入）**：见 [`docs/TRT_Forge_参数说明.md`](docs/TRT_Forge_参数说明.md)。

**软件结构优化计划**：见仓库根目录 [`docs/DefectVision_AI_Studio_软件结构优化计划.md`](../../docs/DefectVision_AI_Studio_软件结构优化计划.md)。当前阶段只修改产品显示名和文档定位，不修改 `TRTForge.exe`、CMake target 或推理 DLL 名称。

---

## 1. 环境

| 组件 | 版本 / 路径 |
| --- | --- |
| Qt | **6.8.3 msvc2022_64** — 默认 `D:/QT/6.8.3/msvc2022_64`（CMakePresets 可改） |
| CUDA + TensorRT | **CUDA 11.8 + TensorRT 8.6.1.6** — 默认 `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8`<br/>本机 TRT 头/lib/dll 全部安装在 CUDA 目录里 |
| MSVC | Visual Studio 2022 (v143) |
| CMake | ≥ 3.21 |

## 2. 编译

### 方式 A：命令行（推荐）

```powershell
cd 4-qt-studio
cmake --preset vs2022-x64
cmake --build --preset vs2022-x64 -j 8
```

输出仍为 `4-qt-studio/bin/TRTForge.exe`、`yolo_infer.dll`、`anomalib_infer.dll`、`ocr_infer.dll` 以及 `windeployqt` 拷贝好的全部 Qt 依赖 dll；当前阶段只修改产品显示名，不修改 CMake target 或可执行文件名。

### 方式 B：Visual Studio "打开文件夹"

1. VS2022 → 文件 → 打开 → CMake → 选择 `4-qt-studio/CMakeLists.txt`
2. 解决方案配置选 `vs2022-x64`，执行 "全部生成"

### 自定义路径

通过命令行覆盖：

```powershell
cmake --preset vs2022-x64 ^
  -DQT_DIR_HINT=D:/QT/6.8.3/msvc2022_64 ^
  -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
```

### 制作 Windows 安装包（Inno Setup）

CMake 在链接 `TRTForge` 后会向 `4-qt-studio/bin` 写入可执行文件、`yolo_infer.dll`、`anomalib_infer.dll`、`ocr_infer.dll`、**windeployqt** 拉取的 Qt 运行库、**OpenCV**、**TensorRT/CUDA/cuDNN** 侧运行时 DLL（以本机 `CUDA_TOOLKIT_ROOT_DIR` 下实际存在为准）以及 `configs/` 等。OCR 模式还需要 `bin/paddle_inference` 下的 Paddle Inference 运行库和模型目录。安装包即**原样递归打包该 `bin` 目录**（不含 `.pdb` 等），用户安装后即可运行 `TRTForge.exe`。

1. 安装 [Inno Setup 6](https://jrsoftware.org/isdl.php)（需带 `ISCC.exe`）。
2. 先完成一次 **Release** 构建，确认 `4-qt-studio/bin/TRTForge.exe` 存在且可本地双击运行。
3. 在 PowerShell 中执行（UTF-8）：

```powershell
cd 4-qt-studio\installer
.\build_installer.ps1
```

生成的安装程序位于 `4-qt-studio/dist/`，文件名形如 `TRTForge_Setup_1.0.0_x64.exe`。版本号可在 `installer/TRTForge_setup.iss` 顶部的 `#define MyAppVersion` 修改。

**分发说明**：目标电脑需安装 **Visual C++ 2015–2022 x64 可再发行组件**（`windeployqt` 使用了 `--no-compiler-runtime`）。若缺少 `VCRUNTIME140.dll` 等，请用户安装微软官方 VC++ Redistributable；安装向导开始前会展示 `installer/README_INSTALL_zh.txt` 中的简要说明。

### 制作二次开发包（推理只给 DLL，训练工程一起带走）

给其他开发同事的包和上面的用户安装包不同：

- **包含**：`4-qt-studio` Qt 工程源码、`4-qt-studio/bin/training/scripts` 启动脚本、`1-python-yolo` 与 `2-python-anomaly` 训练工程、`4-qt-studio/bin` 里的预编译 `*_infer.dll` / `*_infer.lib`、`3-cpp-tensorrt` 与 `ocr/ocr_infer` 的推理 C API 头文件
- **不包含**：`yolo_infer` / `anomalib_infer` / `ocr_infer` 的实现源码（`src`、`examples`、`third_party`）

```powershell
cd <仓库根目录>
powershell -NoProfile -ExecutionPolicy Bypass -File .\3-cpp-tensorrt\trtexec\package_devkit.ps1
```

默认输出到仓库根目录 `dist/trtforge_devkit_时间戳/`。解压后必须保持 `4-qt-studio`、`1-python-yolo`、`2-python-anomaly` 在同一级目录，训练脚本才能自动找到训练工程。开发包里的 `CMakeLists.txt` 已改成导入预编译推理 DLL，不再 `add_subdirectory` 编译推理源码。

## 3. 运行

确保 `bin/` 目录里 TensorRT 的 dll 能被加载——本机做法是把
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib`（包含 `nvinfer*.dll`、
`nvonnxparser.dll`、`nvinfer_builder_resource.dll` 等）加入 PATH，或在启动前
拷贝相关 dll 到 `4-qt-studio/bin/`。

```powershell
$env:Path = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib;' + $env:Path
.\4-qt-studio\bin\TRTForge.exe
```

## 4. UI 与功能一览

```
┌─ AI │ DEFECTVISION AI                           GPU 就绪 | NVIDIA ... | CUDA 11.8 | TRT 8.6.1 ┐
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 01  模型输入输出     ONNX 路径 [...] 浏览     输出 .trt [...] 另存为                       │
│ 02  精度与运行时     精度 [FP32/FP16/INT8/BEST/TF32]    运行时 [full/lean/dispatch]        │
│ 03  构建参数         工作空间MiB / 最大Batch / DLA核心 / 设备 + 五个 chip 复选框           │
│ 04  动态形状         输入名 / min / opt / max  (例 1x3x640x640)                             │
│     额外参数         多行文本框，trtexec 风格 token (--useCudaGraph --calib=... ...)        │
│     [重置]   [▶ 构建引擎]                                                                  │
├──────────────────────── 实时日志（按级别上色） ─────────────────── 构建进度 ───────────────┤
│ [信息]/[警告]/[错误]/[阶段]/[成功]/[详细]                              ●━●━○━○━○ 解析→保存│
│                                                                        已用时/剩余/层/显存 │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 精度选项

| 选项 | 含义 |
| --- | --- |
| FP32 | 不开任何低精度标志 |
| FP16 | `BuilderFlag::kFP16` |
| INT8 | `BuilderFlag::kINT8`（未提供标定缓存时会发警告） |
| BEST | 同时 `kFP16 + kINT8` |
| TF32 | 由独立 chip 控制 `kTF32`，可与上面任意组合 |

### 4.2 额外参数文本框（与 trtexec 兼容子集）

可识别并自动合入 `BuildOptions`：
`--fp16` `--int8` `--best` `--tf32` `--sparsity` `--refit`
`--versionCompatible` `--allowGpuFallback` `--workspace=N` `--device=N`
`--useDLACore=N` `--minShapes=in:1x3x640x640` `--optShapes=...` `--maxShapes=...`
`--shapes=...` `--calib=path` `--timingCacheFile=path`
`--onnx=path` `--saveEngine=path`

未识别的 token 会以 `[警告]` 输出在日志中并被忽略，**不会**导致构建失败。

### 4.3 输入名称与动态形状

选择 ONNX 后，ENGINE BUILDER 会弹出“ONNX 加载中”进度提示，并在后台线程读取第一个 ONNX 输入 tensor，避免大模型解析期间主界面无响应：

- 自动填入输入名称，例如常见的 `images`、`input`。
- 对 4D 输入自动读取或默认使用 `C/H/W`，再生成 `min / opt / max` 三组动态 shape。
- 用户只需要修改“统一 H / 统一 W”，即可同步覆盖三组 shape 的 H/W，例如 `768 / 1024` 会生成 `1x3x768x1024`、`4x3x768x1024`、`8x3x768x1024`。
- 原始 `min / opt / max` 文本框仍保留，适合多 batch、特殊 channel 或非标准输入手动微调。
- 如果识别失败或用户取消，界面会关闭进度提示并保留手动填写入口，不影响后续手动配置。

### 4.4 引擎自检（Smoke Test）

构建完成后，会按"05 / 引擎自检"卡片设置自动跑一次：

- 反序列化生成的 `.trt`，遍历所有 IO 张量；
- 用**随机数据**填输入（FP32/FP16/INT8/UINT8/INT32/BOOL 自动适配）；
- Warmup N 次后用 `cudaEvent` 计时 M 次，得到 min/avg/max ms 与 QPS；
- 显存占用、张量形状/dtype、第一个输出前 8 个数值预览；
- 同时可对**已有 `.trt` 文件**单独执行（按钮"对已有 .trt 自检"）。

> 仅性能与维度冒烟测试，**不**做精度对比；不需要任何外部数据。

### 4.5 进度估算

TensorRT 8.6 的公开头文件中尚未提供 `IProgressMonitor`，
本工具在 GUI 端实现了一个**前端进度估算器** `TRTProgressMonitor`：

- 0–10%   解析 ONNX
- 10–25%  Builder 配置/优化
- 25–85%  Tactic 选择/调优（最长阶段，按经验时间平滑插值）
- 85–97%  序列化
- 97–100% 写盘

阶段切换由 `ILogger` 输出中的关键字（`Parsing`、`Successfully parsed`、`Tactic`、
`Engine generation completed`、`Saved engine` …）触发，结合 `QTimer` 平滑驱动百分比。

## 5. 工程结构

```
4-qt-studio/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── include/
│   ├── BuildOptions.h
│   ├── BuilderWorker.h
│   ├── MainWindow.h
│   ├── engine/
│   │   └── EngineBuilderTab.h
│   ├── common/
│   │   ├── AppEventFormatter.h
│   │   ├── AppEvents.h
│   │   └── AppSettings.h
│   ├── dataset/
│   │   ├── DatasetSplitPolicy.h
│   │   └── DatasetYamlWriter.h
│   ├── training/
│   │   ├── TrainEventParser.h
│   │   └── TrainingArtifactModel.h
│   ├── TRTLogger.h
│   └── TRTProgressMonitor.h
├── src/
│   ├── BuilderWorker.cpp
│   ├── MainWindow.cpp
│   ├── engine/
│   │   └── EngineBuilderTab.cpp
│   ├── common/
│   │   ├── AppEventFormatter.cpp
│   │   └── AppSettings.cpp
│   ├── dataset/
│   │   ├── DatasetSplitPolicy.cpp
│   │   └── DatasetYamlWriter.cpp
│   ├── training/
│   │   ├── TrainEventParser.cpp
│   │   └── TrainingArtifactModel.cpp
│   ├── TRTLogger.cpp
│   ├── TRTProgressMonitor.cpp
│   └── main.cpp
├── resources/
│   ├── app.qrc
│   └── industrial_dark.qss
└── bin/                ← 编译产物（windeployqt 已就绪）
```

### 关键类

| 类 | 线程 | 职责 |
| --- | --- | --- |
| `MainWindow` | UI | 应用外壳；创建顶部栏和 ENGINE / ANNOTATION / TRAINING / INFERENCE 四个主页签 |
| `EngineBuilderTab` | UI | 引擎构建页；收集 `BuildOptions`；管理 `BuilderWorker`、`TRTProgressMonitor`、日志、进度和自检报告 |
| `DatasetSplitPolicy` | UI | 计算并应用 train/val/test 自动划分结果 |
| `DatasetYamlWriter` | UI | 写出 Ultralytics `dataset.yaml`，保持 YOLO 导出格式稳定 |
| `TrainEventParser` | UI | 解析 Python launcher 输出的 `TRTFORGE_JSON` 结构化事件 |
| `TrainingArtifactModel` | UI | 记录训练产物和 labels JSON 状态 |
| `AppLogEvent` / `AppProgressEvent` | UI | 构建页与训练页共用的日志/进度事件模型 |
| `AppEventFormatter` | UI | 统一日志颜色、前缀和 TensorRT 日志级别转换 |
| `YoloInferenceAdapter` / `AnomalyInferenceAdapter` / `OcrInferenceAdapter` | UI | 封装推理页到 YOLO / Anomalib / OCR worker 的异步加载调用，不改变 DLL C API |
| `InferenceResultPresenter` | UI | 统一推理耗时、检测数量和异常 OK/NG 状态的显示文本与颜色 |
| `BuildOptions` | — | 普通结构体，已注册为 `Q_DECLARE_METATYPE` 以跨线程传递 |
| `TRTLogger` | UI / Worker | 同时是 `nvinfer1::ILogger` 和 `QObject`；把 TRT 日志转成 `logArrived(level,text)` 信号 |
| `BuilderWorker` | Worker `QThread` | `start(BuildOptions)` 槽内执行 ONNX→TRT 全流程并 `emit finished` |
| `TRTProgressMonitor` | UI | 监听日志驱动进度条与阶段步进 |
| `InferWorker` / `AnomalyInferWorker` / `OcrInferWorker` | Worker `QThread` | 推理页加载 YOLO / Anomalib TensorRT 引擎或 PaddleOCR 模型并执行帧推理；模型加载通过 queued connection 投递到 worker 线程，加载期间 UI 显示“正在加载模型”进度提示 |

## 6. 与旧版 / `trtexec.exe` 对比

| 维度 | `trtexec.exe` | DefectVision AI Studio |
| --- | --- | --- |
| 入口 | 命令行 | Qt GUI（中文，工业青蓝霓虹风，覆盖标注/训练/部署/推理） |
| 进程模型 | 独立可执行 | 直接调用 TRT C++ API（无子进程） |
| 日志 | 只能看 stdout | 分级上色 + 时间戳 + 自动滚动，并按日期写入 `logs/` |
| 进度 | 仅最终时间 | 阶段进度条 + 已用 / 剩余 / 显存 |
| 参数 | 全部命令行 | 常用参数图形化 + "额外参数"逃生口 |
| 取消 | Ctrl+C | `requestCancel()`（本版仅在阶段切换处响应） |

## 7. 已知限制 / 后续

- TRT 8.6 尚未公开 `IProgressMonitor`，构建中段无法精确百分比，已用平滑估算补足；升级到 TRT 10.x 后可直接接入官方接口。
- `requestCancel()` 在 `buildSerializedNetwork` 内部无法中断，需等当前 Tactic 跑完。
- INT8 标定器未集成（仅识别 `--calib=` 缓存路径），后续可加入 `IInt8EntropyCalibrator2` 实现。
- 多输入 / 多 profile 暂只支持单输入单 profile，需求来了再扩展。

### 7.1 运行日志落盘

TRTForge 会把 ENGINE、TRAINING、INFERENCE 的关键日志按日期保存到工程目录：

```text
4-qt-studio/logs/YYYY-MM-DD/trtforge_YYYY-MM-DD.log
```

日志使用 UTF-8 编码，每行包含完整时间、来源、级别和消息。界面日志仍照常显示，文件日志用于软件关闭后排查构建失败、训练依赖缺失、模型加载失败等现场问题。

## 8. 标注工作台（Annotation Dataset）

TRT Forge 已新增 `ANNOTATION DATASET · 标注数据` 页签，用于在 TensorRT 构建与推理验证前完成基础数据标注和标准格式导入导出。

### 8.1 支持任务

每个 `.tfanno` 项目只选择一种主任务类型：

| 任务 | 用途 | 主要图形 |
| --- | --- | --- |
| `detect` | YOLO 水平框检测 | 矩形 |
| `segment` | YOLO 实例分割 | 多边形 |
| `pose` | YOLO 姿态识别 | COCO 17 点 |
| `obb` | YOLO 旋转框检测 | 四角点旋转矩形 |

项目初始化时不会预置 `00 object` 标签；第一次绘制前若还没有标签，界面会要求先输入一个标签名。画完对象后会弹出标注确认对话框，确认标签、描述、Group、train/val/test 归属和置信度；确认后右侧标签、对象、属性面板会同步刷新。

对象列表与画布支持双向选中：点击右侧对象列表会在画布中高亮对应框，点击画布中的标注对象也会同步高亮右侧对象行。左侧 `选` 工具可拖动已选对象完成移动，右侧 `编辑` / `删除` 可修改或删除当前对象，左侧 `撤` 可撤销最近一次对象新增、编辑、删除或移动。

标签列表支持新增和删除。删除标签前会检查该标签是否已被标注对象使用；如果已使用，需要先编辑或删除相关对象，避免类别编号和导出结果混乱。多边形绘制方式为：左键逐个增加点位，右键将最后一个点与第一个点连线闭合并弹出确认对话框。

### 8.2 项目文件

原生项目文件后缀为 `.tfanno`，保存内容包括：

- 任务类型：`detect / segment / pose / obb`
- 类别列表与颜色
- 图片路径、尺寸和 `train / val / test` 归属
- 每张图片的标注对象、描述和分组

类别可以在项目内新增，也可以导入现有 `labels_*.json`。

### 8.3 数据集划分

标注工作台支持：

- 自动划分：在底部输入 `train / val / test` 百分比后一键划分，默认 `80 / 10 / 10`
- 比例保存：最近一次有效比例会通过 Qt `QSettings` 保存，下次打开 TRTForge 自动恢复
- 手动调整：选中当前图片后点击底部 `train / val / test` 按钮修改归属

导出训练集时会复制图片到标准输出目录，避免训练依赖原始散乱路径。

### 8.4 标准格式导入导出

| 格式 | 支持范围 | 说明 |
| --- | --- | --- |
| YOLO | `detect / segment / pose / obb` | 导出 `images/*`、`labels/*` 和 Ultralytics 可用 `dataset.yaml` |
| COCO | `detect / segment / pose` | 导出标准 COCO JSON；OBB 不生成非标准 COCO |
| VOC | 仅 `detect` | 只导出标准水平矩形 XML，不扩展 polygon/keypoints/rotated box |

YOLO OBB 使用 Ultralytics 官方四角点归一化格式：

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

其中四个角点坐标均相对原图宽高归一化到 `[0, 1]`。

## 9. YOLO Training Lab（训练 / 验证 / ONNX 导出）

TRT Forge 新增 `YOLO TRAINING LAB · 训练导出` 页签，用于从标注工作台导出的 `dataset.yaml` 继续完成 YOLO 训练、模型验证和 ONNX 导出。该页签通过 `QProcess` 调用 Python 子进程，不把 Python 嵌入 Qt 主进程，因此缺库或训练失败时只会弹窗和输出日志，不会拖崩主程序。

### 9.1 Python 统一入口

Qt 调用的脚本为：

```powershell
1-python-yolo\launcher\tf_yolo_launcher.py
```

支持模式：

| mode | 作用 |
| --- | --- |
| `check-env` | 检测 Python、torch、CUDA、ultralytics、onnx、onnxruntime、polars |
| `probe-model` | 读取模型任务、类别和基础信息 |
| `train` | 训练 YOLO 模型 |
| `val` | 验证 `best.pt` 的 precision、recall、mAP 和速度 |
| `export` | 导出原始 `.onnx`，并可生成 `.transd.onnx` |

支持任务头：`detect`、`segment`、`pose`、`obb`。

### 9.2 训练初始化方式

训练页区分三种初始化方式：

| 方式 | 参数含义 | 场景 |
| --- | --- | --- |
| 加载预训练权重微调 | `model=.pt` 或 `model=.yaml + pretrained=权重路径` | 默认推荐，适合多数工业小中型数据集 |
| 从零训练 | `model=.yaml` 且 `pretrained=false` | 数据量很大或完全自定义模型结构 |
| 断点恢复 | `model=last.pt` 且 `resume=true` | 上次训练中断后继续 |

微调时推荐把本地 `.pt` 直接填到「模型入口 model / weights」。如果用户误把本地 `.pt` 填到 `pretrained`，且 `model` 仍是 `yolov8n.pt` 等默认别名，训练页会在启动前自动把该 `.pt` 同步为模型入口，避免 Ultralytics 自动下载默认模型。

### 9.3 日志与错误处理

Python 普通 `print`、Ultralytics 日志、stderr 会原样显示在日志面板中。结构化事件使用如下前缀，Qt 会解析后刷新进度、指标和产物路径：

```text
TRTFORGE_JSON {"type":"progress","epoch":3,"total":100,"pct":3.0}
TRTFORGE_JSON {"type":"metric","name":"metrics/mAP50","value":0.812,"step":3}
TRTFORGE_JSON {"type":"epoch_summary","epoch":3,"total":100,"metrics":{"metrics/mAP50(B)":0.812}}
TRTFORGE_JSON {"type":"plot","name":"results","path":"runs/.../results.png"}
TRTFORGE_JSON {"type":"gpu_status","name":"NVIDIA GeForce RTX 4070 Laptop GPU","memory_used_mb":4096}
TRTFORGE_JSON {"type":"artifact","kind":"labels_json","name":"labels_exp_1.json","path":"runs/.../labels_exp_1.json"}
TRTFORGE_JSON {"type":"artifact","kind":"onnx","path":"runs/.../best.onnx"}
TRTFORGE_JSON {"type":"error","code":"MISSING_DEP","message":"缺少 onnx"}
```

如果本机缺少 Python 或依赖库，界面会提示用户选择 `python.exe` 或安装依赖；脚本不会自动安装库，避免修改现场环境。检测环境会显示当前 `python.exe`、虚拟环境名、缺失库和对应 `pip install` 命令。当前 Ultralytics 保存 checkpoint 时会读取 `results.csv`，需要 `polars`，缺失时可能表现为第 1 个 epoch 验证结束后 `exit=1`。

训练成功结束后，Python 会返回本次 `run_dir`、`best.pt`、`last.pt`、`args.yaml`、`results.csv`、`results.png`、混淆矩阵和常见曲线图。Qt 会在日志与产物区域显示这些路径，并弹窗汇总当前训练模型及附属文件，方便用户直接找到结果。

训练页包含「训练配置 / 日志」和「训练监控」两个页签。「训练监控」页会在每个 epoch 结束后追加深色指标表行，并用带坐标轴、刻度数字和网格线的曲线控件实时更新 F1、P、R、mAP、GPU 显存等小图；双击单个小图可在「最新训练图表」区域内放大，再双击恢复多图。`confusion_matrix` 和 `confusion_matrix_normalized` 使用最新 PNG 缩略图控件显示，适合直接查看矩阵内容。右侧 GPU 区会实时显示显存占用和利用率进度条；标签 JSON 区会显示等待生成、生成成功路径或未生成/写入失败原因。Python 返回的 PNG 曲线图仍作为产物路径记录，但界面主要以实时控件展示当前训练趋势；每次开始训练会先清空上一轮数表、图表、GPU 状态和标签路径。

训练结束会按 TRTForge 推理标签格式生成：

```json
{
  "name": "exp_1",
  "classes": ["missing_hole", "mouse_bite"]
}
```

该文件会保存到训练目录和 `bin/configs`，命名为 `labels_<实验名称>.json`；不会覆盖 `labels_coco.json`，同名时自动追加时间戳。如果训练结束后无法从 `trainer.names` 或 `dataset.yaml` 的 `names` 字段读取类别，界面会显示未生成原因。

训练页会使用 Qt `QSettings` 自动保存当前解释器路径。保存策略为：

- 保存最近一次使用的全局 `python.exe`。
- 按任务头分别保存 `detect / segment / pose / obb` 对应的解释器路径。
- 下次打开训练页时自动恢复；切换任务头时优先恢复该任务头上次使用的解释器。

### 9.5 Anomalib 训练导出设计

TRTForge 训练导出页新增 `Anomalib 异常检测` 模式。该模式不替代现有 YOLO 训练，而是在同一页签内通过训练类型切换，显示 PatchCore、PaDiM、EfficientAd 专用参数区。

设计阶段产物位于：

```text
openspec/changes/add-anomalib-training-export
```

Python 入口为：

```text
2-python-anomaly/launcher/tf_anomalib_launcher.py
```

第一版模式：

| mode | 作用 |
| --- | --- |
| `check-env` | 检查 Python、torch、CUDA、anomalib、lightning、onnx、onnxruntime、PIL、numpy |
| `train-export` | 训练 Anomalib 模型、测试、导出 `model.onnx`、生成 `metadata.json` |

Anomalib 模式下按钮为 `检测环境`、`开始训练并导出`、`停止`；YOLO 专用 `验证模型` 和单独 `导出 ONNX` 会隐藏，避免把异常检测闭环拆散。metadata 规则与异常检测推理页保持一致：PatchCore / PaDiM 的后处理 0-1 ONNX 使用 `output_contract=anomalib_post_processed_0_1` 和默认阈值 `0.5`；EfficientAd 优先使用 normal/abnormal 校准样本分布生成阈值。

Anomalib 训练导出页必须保留 `QTextBrowser` 实时日志窗口，用来打印运行 `tf_anomalib_launcher.py` 时产生的 stdout/stderr。用户应能直接看到环境检测、训练、验证/测试、ONNX 导出、metadata 生成和报错文本；结构化事件只负责刷新进度条、产物列表和弹窗，不能替代原始运行日志。

## 10. 中文路径兼容

TRTForge GUI 在 Windows 上对中文路径做了专项兼容：

- 推理演示加载 `.trt` 引擎时，GUI 仍传 UTF-8 路径，`yolo_infer` DLL 内部会转为 Windows 宽字符路径读取，避免 `std::ifstream` 窄字符串打不开中文路径。
- 推理演示加载图片时，不再把中文路径直接传给 OpenCV，而是用 Qt `QFile` 读入字节后通过 `cv::imdecode` 解码。
- 推理演示加载视频时，若路径包含中文，会先复制到系统临时目录下的 ASCII 文件名，再交给 OpenCV `VideoCapture`，播放结束后删除临时文件。
- 标签 JSON 使用 Qt `QFile` 读取，中文路径正常支持。
- ONNX 构建输入若包含中文路径，会先复制到系统临时目录下的 ASCII 文件名供 TensorRT ONNX Parser 解析，构建输出 `.trt` 仍保存到用户选择的原始路径。

## 11. 推理演示画布交互

点击“开始推理”后，TRTForge 会先判断模型参数是否变化。第一次启动，或更换任务类型、`.trt/.engine`、阈值、异常检测 `metadata.json` 等参数时，会显示“正在加载模型，请稍候...”的进度提示，并在推理 worker 线程中加载模型；如果模型参数未变化，只更换图片 / 视频 / 相机输入源，则复用已加载 worker，直接启动采集与推理。

推理演示右侧显示窗口支持查看大图细节：

- 鼠标滚轮：围绕鼠标当前位置放大或缩小画面。
- 鼠标左键按住拖动：平移已放大的图像或视频帧。
- 鼠标左键双击：恢复到适配窗口大小并居中显示。
- 异常检测三画布：左上角显示画布名称，右下角显示缩放提示，避免标题与缩放信息重叠。

## 12. Anomalib TRT 异常检测推理

推理演示页新增 `异常检测 ANOMALY` 任务模式，用于验证 Anomalib 已导出的 TensorRT engine。该模式只做推理，不做训练、不做 ONNX 导出、不做 engine 构建。Anomalib 推理核心封装在独立 `anomalib_infer.dll` 中，TRTForge GUI 通过 `AnomalyInferWorker` 薄适配层调用；同一 DLL 也可供外部 C++ 程序和 Python `ctypes` 程序调用。

### 12.1 使用步骤

1. 在「任务类型」选择 `异常检测 ANOMALY`。
2. 在「异常检测模型」选择 `EfficientAd`、`PaDiM` 或 `PatchCore`。
3. 选择已有 `.engine` 或 `.trt` 文件。
4. 在 metadata 区选择 ONNX 导出同目录生成的 `metadata.json`。注意：Anomalib 2.x 训练默认只保存 `model.ckpt`，不会自动生成 TRTForge 可读取的部署 metadata，本项目的 `anomalib-main/train.py` 会在导出 ONNX 后补写该文件。
5. 选择图像、视频或相机输入，点击「开始推理」。

### 12.2 metadata 参数说明

| 参数 | 说明 |
| --- | --- |
| `transform.height / transform.width` | 异常检测模型输入尺寸，动态 shape engine 会按该尺寸设置输入 |
| `input_size` | 新增兼容字段，格式为 `[height, width]`，与 `transform.height / transform.width` 等价 |
| `image_threshold` | 图像级异常阈值，用于判断 OK/NG；界面会以 metadata 值作为默认值，运行前可通过滑条微调 |
| `pixel_threshold` | 像素级阈值，用于生成二值 Mask；界面会以 metadata 值作为默认值，运行前可通过滑条微调 |
| `pred_scores.min/max` 或 `pred_scores.min` / `pred_scores.max` | 图像分数归一化范围 |
| `anomaly_maps.min/max` 或 `anomaly_maps.min` / `anomaly_maps.max` | 热力图归一化范围 |
| `output_contract` | 当值为 `anomalib_post_processed_0_1` 时，表示 ONNX/TRT 已输出 0~1 的 `pred_score/anomaly_map`，阈值应使用 `0.5` 而不是训练阶段的 raw 阈值 |
| `normalization` | TRTForge 输入前处理策略，`none` 表示只做 `0~1` 缩放，不再额外做 ImageNet 归一化；`imagenet` 表示外部做 ImageNet 归一化 |
| `min/max` | 旧版 metadata 的扁平字段，会作为 score/map 的兼容范围 |

如果 metadata 没有保存 `transform.height/width`，TRTForge 会优先使用 engine 的静态输入尺寸；如果 engine 是动态输入且 metadata 也没有尺寸，默认按 `256×256` 作为兜底尺寸。

异常检测模型类型以下拉框作为人工提示，但推理时以 `metadata.json` 的 `model_type`、`normalization` 和 `output_contract` 为准。对于 Anomalib 2.x 导出的 PatchCore，如果 ONNX/TRT 已经输出 `pred_score/anomaly_map/pred_mask`，metadata 会写入 `normalization=none`；此时即使界面选择 `PatchCore`，TRTForge 也不会再额外做 ImageNet 归一化。

### 12.3 输出说明

异常检测模式右侧画布显示三个子窗口：`Image` 原图、`Image + Anomaly Map` 热力图叠加图、`Image + Pred Mask` 原图加预测异常轮廓。双击任一子窗口可临时全屏，再次双击同一窗口恢复三窗口并排。右侧结果列表显示 normalized score、raw score、阈值和 OK/NG 状态。`image_threshold` 滑条使用 `pred_scores` 范围，`pixel_threshold` 滑条使用 `anomaly_maps` 范围，点击「开始推理」时会把当前滑条值传入后处理；红色异常轮廓来自 `pixel_threshold` 二值化后的外轮廓。当前输出解析规则为：

| 输出数量 | 解析方式 |
| --- | --- |
| 1 个输出 | 输出视为 anomaly map，score 取 map 最大值 |
| 2 个输出 | 优先把标量输出视为 score，把元素数量最大的输出视为 anomaly map |
| 3 个及以上输出 | 优先把标量输出视为 score，把元素数量最大的输出视为 anomaly map |

### 12.4 路径兼容

异常检测模式的 engine、metadata 路径以 UTF-8 传给 `anomalib_infer.dll`，DLL 内部负责文件读取并兼容中文路径；图像输入由 Qt 解码为 RGB 帧后传给 DLL；视频输入沿用推理演示的临时 ASCII 转存策略。

### 12.5 外部 C++ / Python 调用

`anomalib_infer.dll` 的公开头文件位于：

```text
3-cpp-tensorrt/anomalib_infer/include/anomalib_infer_c.h
```

调用顺序为 `ai_create` → `ai_get_metadata` → `ai_infer_rgb` → `ai_destroy`。`ai_infer_rgb` 返回的图像和 map 指针由 DLL 句柄持有，只保证到下一次 `ai_infer_rgb` 或 `ai_destroy` 前有效，外部程序必须立即深拷贝。

示例位置：

| 示例 | 路径 |
| --- | --- |
| C++ | `3-cpp-tensorrt/anomalib_infer/examples/cpp_basic` |
| Python ctypes | `3-cpp-tensorrt/anomalib_infer/examples/python_ctypes/infer_image.py` |

### 12.6 ckpt / onnx 推理对比

`2-python-anomaly/anomalib-main/compare_ckpt_onnx_inference.py` 用于在构建 TensorRT 前确认 `model.onnx` 与原始 `model.ckpt` 的结果是否接近。脚本放在 `train.py` 同级，典型用法：

```powershell
cd 2-python-anomaly\anomalib-main
python compare_ckpt_onnx_inference.py --model patchcore --ckpt weights\lightning\model.ckpt --onnx weights\lightning\model.onnx --metadata weights\lightning\metadata.json --input <仓库根>\2-python-anomaly\anomalib-main\datasets\bottle\test\good
```

## 13. PaddleOCR C++ OCR 推理

推理演示页新增 `OCR 文本识别 OCR` 任务模式，用于验证 PaddleOCR C++ / Paddle Inference 模型。OCR 推理核心封装在独立 `ocr_infer.dll` 中，TRTForge GUI 通过 `OcrInferWorker` 调用；同一 DLL 也可供外部 C++ 程序通过 `ocr_infer_c.h` 复用。

### 13.1 使用步骤

1. 在「任务类型」选择 `OCR 文本识别 OCR`。
2. 在原 `.trt 引擎路径` 位置选择 OCR 检测模型目录，例如 `bin/paddle_inference/ch_PP-OCRv4_det_infer`。
3. 在 `OCR 识别模型目录` 选择识别模型目录，例如 `bin/paddle_inference/ch_PP-OCRv4_rec_infer`。
4. `OCR 文本行方向模型目录` 默认建议留空；只有拿到带 `inference.yml` / `inference.yaml` 的新版 `TextLineOrientation` 模型目录时再填写。`ch_ppocr_mobile_v2.0_cls_infer` / `ch_ppocr_mobile_v2.0_cls_slim_infer` 是老版方向分类模型包，不适合作为该输入。
5. 选择图像、视频或相机输入，点击「开始推理」。
6. 推理完成后，可拖动主画布与右侧结果列表之间的分隔条调整显示比例；点击右侧任意 OCR 文本行，主画布会高亮对应文本框。

### 13.1.1 OCR 独立编译

TRTForge 主工程默认不再直接编译 `ocr_infer`，避免 Abseil、yaml-cpp、Clipper 等 OCR 第三方目标进入主解决方案。主工程默认通过 `TRTFORGE_BUILD_OCR_INFER=OFF` 导入 `bin/ocr_infer.dll` 和 `bin/ocr_infer.lib`。

如需单独重编 OCR：

```powershell
cd ocr\ocr_infer
cmake --preset vs2022-x64
cmake --build --preset vs2022-x64
```

生成的 `ocr_infer.dll`、`ocr_infer.lib` 和 `ocr_basic.exe` 仍输出到 `4-qt-studio/bin`，之后再重新配置或编译 TRTForge 主工程即可。

### 13.2 模型目录要求

每个启用模型目录必须至少包含：

- `inference.pdmodel` 或 `inference.json`
- `inference.pdiparams`

每个启用的 PaddleOCR 模型目录都需要 `inference.yml` 或 `inference.yaml`。缺少该配置时 DLL 会提前返回加载失败，避免 PaddleOCR 内部退出导致 TRTForge 崩溃。

默认 Paddle Inference 根目录为 `4-qt-studio/bin/paddle_inference`，链接库为 `4-qt-studio/bin/paddle_inference/paddle/lib/paddle_inference.lib`。测试图片建议使用 `ocr/PaddleOCR-main/docs/datasets/images`。

OCR 模式提供运行后端诊断：设备 `CPU / GPU 0`、精度 `FP32 / FP16`、`MKLDNN` 开关、CPU 线程数和识别 batch size。推理后顶部状态条与右侧结果列表会显示 `device`、`run_mode`、`mkldnn`、`threads`、`rec_batch` 以及 `io / pipeline / result / total` 分阶段耗时。当前 `oi_infer_bgr` 已走内存 `cv::Mat` 输入，不再写临时 PNG；如果选择 `GPU 0` 后日志提示 `Please use PaddlePaddle with GPU version`，说明当前 Paddle Inference 包不是可用 GPU 版，需要替换 GPU 版包后再测试。

当前 `ch_PP-OCRv4_det_infer + ch_PP-OCRv4_rec_infer` 模型在 `CDLA_demo` 上的耗时记录：`val_0633.jpg` 使用 `CPU + MKLDNN + FP32` 识别 55 行约 13.2 秒，`val_0941.jpg` 识别 83 行约 17.4 秒；同一张 `val_0633.jpg` 关闭 MKLDNN 后约 100.6 秒。GPU 0 模式当前会提示 `Please use PaddlePaddle with GPU version`，因此现有 Paddle Inference 包不能作为 GPU 性能数据。

### 13.3 输出说明

OCR 模式右侧主画布显示 OCR 多边形框和文本行序号；右侧结果列表逐行显示识别文本、置信度、方向角和多边形坐标。DLL 原始结果契约如下：

```text
oi_result.lines[i].index
oi_result.lines[i].xs / ys
oi_result.lines[i].text_utf8
oi_result.lines[i].rec_score
oi_result.lines[i].textline_angle
```

以上字段按 `i` 一一对应，调用者可直接深拷贝保存。`oi_result.raw_json_utf8` 同时保留 `rec_texts / rec_scores / rec_polys / textline_orientation_angles`，用于日志记录、二次分析或与 PaddleOCR 原始输出对齐。

### 13.4 与旧版差异

旧版推理页只支持 YOLO TensorRT 与 Anomalib TensorRT。新版新增 PaddleOCR OCR 模式，但仍保持主程序不直接依赖 PaddleOCR 预测器；PaddleOCR 逻辑全部在 `ocr_infer.dll` 内，TRTForge 只负责配置、调用、画面显示和结果列表展示。首版 `oi_infer_bgr` 内部会把帧写成临时图片后进入 PaddleOCR 文件路径推理链路，优先保证 DLL 边界和原始数据契约稳定。

脚本会生成 `compare_outputs/compare_results.csv`，包含 checkpoint 平均耗时、ONNX 单图耗时、score 差异、OK/NG 是否一致以及 anomaly map 差异摘要。
