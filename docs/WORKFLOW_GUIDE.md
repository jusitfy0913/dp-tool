# WORKFLOW_GUIDE — TRTForge / DefectVision AI Studio 全流程作业指南

> 本文回答一个核心问题：**从原始图片到产线推理，每一步该做什么、用什么命令、产出什么文件、怎么验证。**
> 面向对象：需要独立跑通全流程的工程师（不需要深入算法细节）。
> 最后更新：2026-10-30

---

## 0. 全流程图

```mermaid
flowchart TD
    A["① 原始图片"] --> B["② 标注 / 数据整理<br/>Web「数据集」页可重划分与导出"]
    B --> C{"③ 任务类型"}

    C -->|"目标检测 / 分割 / 姿态 / OBB"| D["YOLO 训练<br/>1-python-yolo"]
    C -->|"缺陷有无 / 像素级异常"| E["Anomalib 训练<br/>2-python-anomaly"]

    D --> F["best.pt"]
    E --> G["model.ckpt + model.onnx<br/>+ metadata.json"]

    F --> H["④ 导出 ONNX"]
    H --> I["⑤ 编译引擎<br/>Web「引擎构建」页 / trtexec"]
    G --> I
    I --> J["model_fp16.engine"]

    J --> K["⑥ 推理验证<br/>6-web 或 DLL"]
    K --> L{"⑦ 精度达标？"}
    L -->|否| B
    L -->|是| M["⑧ 产线部署"]

    style E fill:#e1f5ff
    style I fill:#fff4e1
    style K fill:#e8f5e9
```

---

## 1. 环境准备（只做一次）

### 1.1 必需组件

| 组件 | 版本要求 | 本机实测 |
| --- | --- | --- |
| NVIDIA 驱动 | 支持 CUDA 11.8+ | 596.71 |
| CUDA Toolkit | 11.8 | 11.8.89 |
| TensorRT | 8.5.x（与 CUDA 匹配） | 8.5.2 |
| cuDNN | 8.x | 8.7 |
| Visual Studio | 2022（v143） | 19.44 |
| Python | 3.10 ~ 3.12 | 3.12.4（`dpenv`） |
| OpenCV（C++ SDK） | 4.x | 4.12.0（vc17 / x64，路径由 `OPENCV_DIR` 指定） |

### 1.2 Python 依赖

```powershell
# YOLO 链路
python -m pip install torch torchvision ultralytics polars "numpy<3" onnx onnxruntime

# Anomalib 链路
python -m pip install anomalib lightning "numpy<3" onnx onnxruntime Pillow

# Web 工作站
python -m pip install fastapi uvicorn python-multipart
```

> **numpy 上限说明（2026-10-30 修订）**：早期要求 `numpy<2`，理由是 torch 2.3 按 numpy 1.x 编译。
> 实测 `torch 2.3.0 + numpy 2.5.4` 的 CUDA 前向、numpy↔torch 桥接均正常，`onnxruntime 1.20.1` 也支持
> numpy 2，故放宽到 `<3`。锁 `<2` 会被 Ultralytics AutoUpdate 在每次 `export` 时升回去，造成环境反复漂移。
> 唯一残留是 `scipy 1.14.0` 的 `UserWarning`（声明要求 numpy<2.3），不影响训练与推理。

### 1.3 环境自检（关键步骤，别跳过）

```powershell
cd 2-python-anomaly\launcher
python tf_anomalib_launcher.py --mode check-env

cd ..\..\1-python-yolo\launcher
python tf_yolo_launcher.py --mode check-env
```

**验收标准**：输出 JSON 中 `"missing":[]`。

补充验证 GPU：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.cudnn.version())"
# 期望：2.3.0+cu118 True 8700
```

### 1.4 已知环境坑

| 坑 | 现象 | 解决 |
| --- | --- | --- |
| CUDA 11.8 + MSVC 14.44 | `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer` | 设置 `$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"` |
| Python 3.8+ DLL 搜索 | `WinDLL` 报「找不到模块或它的依赖项」 | 必须 `os.add_dll_directory()`，`6-web/backend/infer.py` 已处理 |
| numpy 2.x | 早期认为与 torch 2.3 有 ABI 冲突 | 2026-10-30 实测 `torch 2.3.0 + numpy 2.5.4` 正常，约束已放宽到 `numpy<3`；若确实遇到 torch 导入异常，再降到 `numpy<2`（实测 1.26.4） |

---

## 2. 标注与数据集准备

### 2.1 两种数据形态

| 任务 | 需要的数据 | 目录形态 |
| --- | --- | --- |
| YOLO（检测/分割/姿态/OBB） | 图片 + 标注文件 | `dataset.yaml` + `images/` + `labels/` |
| Anomalib（异常检测） | **只要正常图**（+ 少量异常图用于验证） | `root/train/good`、`root/test/good`、`root/test/abnormal` |

### 2.2 Anomalib 标准目录

```text
2-python-anomaly/anomalib-main/datasets/bottle
├── train/good        225 张   训练用正常图（唯一的训练输入）
├── test/good          20 张   测试用正常图（算误报）
├── test/abnormal      44 张   测试用异常图（算漏检）
└── ground_truth/              像素级标注（可选）
```

**要点**：

- `train/good` 决定模型能学到什么，**必须干净**（不能混入缺陷图）
- `test/abnormal` 也参与阈值校准，**至少要 10 张以上**，否则阈值偏保守
- 目录名 `good` / `abnormal` 是 anomalib Folder 数据集的约定，不要随意改

### 2.4 EfficientAd 专用：imagenette 辅助数据

只有 EfficientAd 需要这份额外数据（用于计算教师网络通道归一化统计）。**必须注意先后顺序**：launcher 会在模型创建前检查目录是否存在，**目录不存在时直接报错，不会自动下载**。

```powershell
# 第一步：先建出空目录（关键，否则训练直接 FileNotFoundError）
New-Item -ItemType Directory -Force -Path 2-python-anomaly\anomalib-main\datasets\imagenette

# 第二步：正常跑 EfficientAd 训练，此时会自动下载 imagenette2.tgz（约 1485MB / 6~10 分钟）
```

| 情况 | 结果 |
| --- | --- |
| `datasets/imagenette` 目录不存在 | **报 `DATA_NOT_FOUND` / `exit=1`**，不下载 |
| 目录存在但缺 `imagenette2/train` | 自动下载 + 解压，解压后 9469 张 / 10 类 |
| 已就绪 | 直接使用，不会重复下载 |

### 2.3 数据集划分

YOLO 训练页支持自动划分 `train / val / test`，默认 `80 / 10 / 10`。
修改比例后点「自动划分」，会按比例分配图片归属。

### 2.5 用 Web 数据集工作台重划分与导出（推荐）

当数据已经有标注、只是需要重新划分或转格式时，用 Web 的「数据集」页，不必开 Qt：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
# 浏览器打开 http://127.0.0.1:8060 → 点顶部「数据集」
```

| 步骤 | 操作 | 说明 |
| --- | --- | --- |
| 1 | 填数据源 | YOLO 目录（含 `images/` `labels/` `data.yaml`）或 `.tfanno` |
| 2 | 任务类型 | 默认「自动识别」，按标签字段数判断 detect / segment / pose / obb |
| 3 | 填比例 | train、val 两个数，test 是余数，页面会实时显示 |
| 4 | 扫描预览 | 看各集张数、空标注数、每张图的「现有 → 新划分」 |
| 5 | 选格式与输出目录 | YOLO / COCO / VOC，输出不能放在数据源内部 |
| 6 | 开始导出 | 摘要里会列出统计、跳过项和警告 |

**划分口径（与 Qt 版一致，容易误解，务必看清）**

- `train = 总数 × train% ÷ 100` **向下取整**，`val` 同理；
- **剩余全部归 test**：692 张配 80/20 会得到 553 / 138 / **1**，不是 553 / 139 / 0；
- 想要真正的 80/10/10，请填 `80 / 10`，或 `70 / 20` 得到约 70 / 20 / 10；
- 哈希为 **FNV-1a 64 位 + `_2026` 盐值**，同一份数据在任何机器上结果逐项一致。
  Qt 用的是 `qHash`，两者哈希函数不同，所以**不能与 Qt 历史划分逐张对照**。

**导出格式限制**

| 格式 | 产物 | 限制 |
| --- | --- | --- |
| YOLO | `images/{train,val,test}` + `labels/{split}` + `dataset.yaml` | 全任务支持，`dataset.yaml` 可直接被 Ultralytics 识别 |
| COCO | `annotations.json` | 不复制图片；不支持 OBB |
| VOC | `Annotations/*.xml` + `JPEGImages/` | **仅 detect** |

验收脚本（不启动服务也能跑）：

```powershell
python 6-web\backend\tools\test_dataset_workbench.py
# 期望：通过 20 项，失败 0 项
```

### 2.6 前端单测（零依赖，只需 node）

前端是零构建的，所以单测也不装任何框架，直接 `node` 跑。这两项不需要启动服务：

```powershell
node 6-web\frontend\tests\test_viewer_math.mjs
# 期望：✓ 全部通过：37 条断言

node 6-web\frontend\tests\test_contrast.mjs
# 期望：✓ 全部通过：23 项
```

| 脚本 | 守住什么 |
| --- | --- |
| `test_viewer_math.mjs` | 滚轮缩放必须锚定光标（连续缩放 8 次后漂移 ≤ 0.5px）、屏幕与图像坐标互逆、滚轮增量按 `deltaMode` 归一化 |
| `test_contrast.mjs` | 从 `style.css` 读设计令牌按 WCAG 公式核算，正文与面板底色不低于 4.5:1，网格/扫描线不透明度 ≤ 0.05 |

改配色的正确流程：改令牌 → 跑 `test_contrast` → 不达标就再调，别靠眼睛判断。

---

## 3. 训练

### 3.1 入口一览

| 链路 | 入口 | 模式 |
| --- | --- | --- |
| YOLO | `1-python-yolo/launcher/tf_yolo_launcher.py` | `check-env` / `probe-model` / `train` / `val` / `export` |
| Anomalib | `2-python-anomaly/launcher/tf_anomalib_launcher.py` | `check-env` / `train-export` |

### 3.2 YOLO 训练

```powershell
cd 1-python-yolo\launcher
python tf_yolo_launcher.py --mode train `
  --model yolov8n.pt --data <dataset.yaml> `
  --epochs 100 --imgsz 640 --batch 16 --workers 8 `
  --project runs\trtforge --name my_yolo
```

产物：`runs/trtforge/my_yolo/weights/{best.pt, last.pt}` + `results.csv` + 各类曲线图。

#### 3.2.1 数据增强参数（v1.7 起可调）

launcher 支持 15 项增强参数，**默认值与 Ultralytics `DEFAULT_CFG` 逐项一致，不填即行为不变**。

```powershell
# 小缺陷场景（PCB 70px 划痕）：降 mosaic、延长 close_mosaic、减弱几何抖动
python tf_yolo_launcher.py --mode train `
  --model yolov8n.pt --data configs\pcb_detect.yaml `
  --epochs 100 --imgsz 640 --batch 16 `
  --mosaic 0.1 --close-mosaic 20 --scale 0.3
```

| 参数 | 默认 | 建议场景 |
| --- | --- | --- |
| `--mosaic` | 1.0 | **小缺陷降到 0~0.3**：四图拼接会把缺陷再缩小一半 |
| `--close-mosaic` | 10 | 小缺陷加大到 15~20，让最后几轮看完整图 |
| `--scale` | 0.5 | 缺陷很小可降到 0.2~0.3，避免缩到看不见 |
| `--fliplr` | 0.5 | 有丝印/文字/方向性图案时设 0 |
| `--degrees` | 0 | 拍摄位姿固定就保持 0 |
| `--hsv_v` / `--hsv_s` | 0.4 / 0.7 | 打光不稳、多工位时适度加大 |

> 完整 15 项清单见 `readme.md` 的 v1.7 版本记录，或直接在 Web「模型训练」页展开「数据增强」。
> 回归测试：`python 6-web/backend/tools/test_augment_params.py`（26 项）。


### 3.3 Anomalib 训练（三选一）

```powershell
cd 2-python-anomaly\launcher

# PatchCore —— 3分42秒，准确率高，但正常图分数饱和
python tf_anomalib_launcher.py --mode train-export --model patchcore `
  --root ..\anomalib-main\datasets\bottle `
  --project results --name my_patchcore --input-size 256

# PaDiM —— 28秒，推理最快（3.6ms），偶有漏检
python tf_anomalib_launcher.py --mode train-export --model padim `
  --root ..\anomalib-main\datasets\bottle `
  --project results --name my_padim --input-size 256

# EfficientAd —— 约1小时，准确率最高，分数有区分度
python tf_anomalib_launcher.py --mode train-export --model efficientad `
  --root ..\anomalib-main\datasets\bottle `
  --imagenet-dir ..\anomalib-main\datasets\imagenette `
  --project results --name my_efficientad --input-size 256 --max-epochs 20
```

**模型怎么选**：见 `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md`。

### 3.4 训练产物

```text
results/<ModelName>/<run_name>/v0/weights/lightning/
├── model.ckpt            Lightning 权重（可继续训练）
├── model.onnx            含后处理的 ONNX 图
└── metadata.json         阈值、归一化、输入尺寸、输出契约
```

**`metadata.json` 是推理的必要输入**，不是可选项。

### 3.5 事件协议（给界面集成用）

launcher 通过 stdout 打印结构化事件：

```text
TRTFORGE_JSON {"type":"progress","stage":"train","epoch":3,"total":20,"pct":20.0}
TRTFORGE_JSON {"type":"metric","name":"image_F1Score","value":0.94,"step":3}
TRTFORGE_JSON {"type":"artifact","kind":"onnx","path":"..."}
TRTFORGE_JSON {"type":"error","code":"DATA_NOT_FOUND","message":"..."}
```

事件类型：`env` / `progress` / `metric` / `epoch_summary` / `gpu_status` / `artifact` / `plot` / `error` / `anomaly_model` / `anomaly_export_summary` / `event`。

**退出码约定**：

| 退出码 | 含义 | 处理 |
| --- | --- | --- |
| 0 | 成功 | — |
| 1 | 一般错误 | 看 `error` 事件 |
| 2 | argparse 语法错误 | 检查参数名 |
| 130 | 用户取消 | 正常中断 |

---

## 4. 导出 ONNX

### 4.1 YOLO

```powershell
python tf_yolo_launcher.py --mode export --model runs\trtforge\my_yolo\weights\best.pt `
  --imgsz 640 --opset 12
```

> `.transd.onnx` 仅用于 TRTForge 自定义推理链路，**不要用 `YOLO()` 直接加载**
> （Ultralytics 认的是原始输出布局，转置后会解析失败）。
>
> 但反过来：**编译 TensorRT 引擎时必须用它**，见 §5.1。导出后目录里应有
> `best.onnx` 和 `best.transd.onnx` 两个文件；如果只有前者，加 `--make-transd` 重跑。

### 4.2 Anomalib

`train-export` 模式已自动完成 ONNX 导出，无需单独执行。

**注意**：导出的 ONNX **已内置 anomalib 后处理**（输出 `pred_score` / `pred_label` / `anomaly_map` / `pred_mask`，值域 0~1）。因此 metadata 里的 `normalization` 是 `none`，DLL 不会再做第二遍 ImageNet 归一化。

---

## 5. 编译 TensorRT 引擎

### 5.1 trtexec 命令

> **也可以完全不敲命令行**：Web Studio 的「引擎构建」页已支持同一套能力
> （选 ONNX → 自动探测形状 → 选精度/dynamic shape → 实时日志）。
> 命令行的价值在于批量脚本和排查；日常单个模型建议直接用网页。
> 详见本章 5.4。

```powershell
$trt = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\trtexec.exe"

# YOLO —— 必须用 .transd.onnx 编译，不能用 model.onnx
& $trt --onnx="model.transd.onnx" --saveEngine="model_fp16.engine" --fp16 `
  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 `
  --maxShapes=images:8x3x640x640 --workspace=4096

# Anomalib（输入名固定为 input）
& $trt --onnx="model.onnx" --saveEngine="model_fp16.engine" --fp16 `
  --minShapes=input:1x3x256x256 --optShapes=input:1x3x256x256 `
  --maxShapes=input:8x3x256x256 --workspace=4096
```

> ⚠️ **YOLO 引擎只能用 `.transd.onnx` 编译，这是最容易踩的坑。**
>
> | | 输出 shape | `yolo_infer.dll` 能否正确解析 |
> | --- | --- | --- |
> | `best.onnx` | `[1, 4+类别数, 候选框数]` | ❌ 不能 |
> | `best.transd.onnx` | `[1, 候选框数, 4+类别数]` | ✅ 能 |
>
> DLL 里按 `bbox_head_dims_[2] - 4` 推算类别数。用 `best.onnx` 时这个值会变成
> `18900 - 4 = 18896`，解码 kernel 越界读显存，**不报任何错误**，却吐出
> `confidence = 957.26`、坐标 `top = -725` 这类垃圾框。
>
> 实测（PCB 6 类模型 @960）：
>
> - 错误引擎：5 个框，置信度 `[957.258, 956.779, 304.125, ...]` → Web 接口返回 **HTTP 400**
> - 正确引擎：3 个框，置信度 `[0.948, 0.761, 0.822]` → 正常
>
> `--mode export` 默认就会顺带生成 `.transd.onnx`（`--make-transd` 默认开启）。
> 如果没生成，加 `--make-transd` 重跑导出即可。

### 5.2 预期耗时（T1000 实测）

| 模型 | 编译耗时 |
| --- | --- |
| EfficientAd | 约 8 分钟 |
| PatchCore | 约 15 分钟 |
| PaDiM | 约 17 分钟 |
| YOLOv8n（320） | 约 5 分钟（313 秒，`Engine built in` 值） |
| YOLOv8n（640） | 约 2~3 分钟 |
| YOLOv8n（960） | 约 8 分钟（474 秒，首次无 timing cache） |

> **必须串行编译**。并行会互相抢 GPU，实测明显变慢。
> 编译期间 GPU 利用率掉到 4% 是正常的（builder 在 CPU 侧搜索 tactic）。
> 反过来，若长时间没有日志输出但 GPU 占用接近 100%，说明仍在 tactic 搜索，
> 不是在卡死，不要中断。

**为什么 320 也要 5 分钟（重要，别按面积去推算）**

第一版 Web 估算假设「耗时 ∝ 输入面积」，于是 960 的 474 秒推到 320 只剩 53 秒。
实测 320 需要 **313.6 秒**，假设直接被推翻。真实规律是：

- 耗时主要来自「层数 × 每层候选 kernel 数」，ONNX 体积是层数的代理，
  输入尺寸只影响**每次候选试跑**的成本；
- 分辨率小 → 激活值小 → 显存里能塞下更多 tactic，搜索空间反而更大，
  两者部分抵消，所以远没有「平方」那么快。

用 960（474 s）与 320（313 s）两点反解，指数 ≈ 0.37，Web 端取 0.4：

```
秒数 ≈ 38.3 × ONNX_MB × (imgsz / 960) ^ 0.4      # fp16；fp32 × 0.7
```

> 跨显卡要重算：`38.3` 是 T1000（Turing，8 GB）上的值，换 Ampere/Ada 会明显变小。
> 不需要手调 —— Web 端每次成功构建都会把真实耗时写进
> `runtime/engine_stats.json`，同类构建下次直接取历史中位数。

**tactic 搜索期间日志是静默的，这不是卡死**

真实日志里 `Local timing cache in use.` 到 `Engine built in 474.149 sec.` 之间
是 **473 秒空白**，trtexec 一行都不打。所以：

- 命令行里只能靠「等」，别以为卡住了就 Ctrl+C；
- Web 界面会用心跳线程每 2 秒推进估算进度，即使日志不动，百分比和已用时间也在动；
- 判断是否真的在跑：`nvidia-smi` 里 trtexec 是否占着 GPU，或看日志是否终于出现新行。

### 5.3 验收

输出末尾应出现：

```text
&&&& PASSED TensorRT.trtexec [TensorRT v8502]
```

并生成 `*.engine` 文件。出现 `FAILED` 通常是算子不支持或 opset 过低。

### 5.4 用 Web 界面构建（推荐）

启动工作站后打开「引擎构建」页：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
# 浏览器打开 http://127.0.0.1:8060 → 点顶部「引擎构建」
```

| 操作 | 说明 |
| --- | --- |
| 选 ONNX | 选完自动调 `/api/engine/probe` 显示输入/输出形状与体积；未转置的 YOLO 会标红 |
| 输出引擎 | 留空自动推导为 `<onnx名>_fp16.engine` |
| 精度 | `fp16` 省显存（推荐）/ `fp32` |
| 动态 shape | 勾选后填 min/opt/max，如 `1x3x960x960` |
| workspace | 默认 4096 MB |

点「开始构建」后：

- 日志通过 `/ws/runs/{run_id}` 实时滚动（与训练页同一通道）；
- 提交未转置 ONNX 会被后端 **HTTP 400** 拒绝，日志面板写明原因，不会白等十几分钟；
- 退出码为 0 但没有生成引擎文件时，任务判为 **failed** 并发 `ARTIFACT_MISSING`，
  不会出现「显示成功、去推理却找不到文件」。

**进度面板（v1.13）**

| 元素 | 说明 |
| --- | --- |
| 阶段清单 | 启动 / 解析 ONNX / 解析完成 / 构建与 tactic 搜索 / 序列化 / 校验 / 推理基准 / 完成 |
| 百分比 | `已用 ÷ 估算`，夹在本阶段区间内（构建段封顶 94%，基准段封顶 99%），只有 `PASSED` 才 100% |
| 已用 / 预计剩余 | 前端每秒刷新；后端心跳每 2 秒推一次整数百分比 |
| 估算来源 | 有历史 → 同精度/同 dynamic/体积 ±67%/分辨率 ±20% 的中位数；无历史 → 公式外推 |

三条必须守住的行为：

1. **失败或中断的构建不补 100%**，进度停在出事的位置，避免「进度满格但失败」；
2. **统计在 `Engine built in` 出现的那一刻就落盘**：那一段是真实的构建耗时，
   即使之后的推理基准被中断，它对下次估算依然有参考价值；而根本没跑到这一步的失败
   （参数错误、早期 OOM、ONNX 解析失败）不会产生任何条目；
3. **落盘的值只认 `Engine built in X sec.`**，不拿进程总时长充数（后者含解析与基准）。

验证：`python 6-web\backend\tools\test_engine_progress.py`（26 项，含真实日志按时间戳回放）。

构建产物路径可直接填进「推理演示」页的引擎文件栏。


---

## 6. 编译 C++ 推理 DLL（仅在改了 C++ 源码时需要）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_infer_dlls.ps1
```

- 产物：`4-qt-studio/bin/{yolo_infer.dll, anomalib_infer.dll}`
- 不依赖 Qt，只需 OpenCV + CUDA
- **编译前必须先停掉 Web 服务**，否则 DLL 被占用会报 `LNK1104`

#### OpenCV 路径怎么找（v1.8 起不再写死）

所有 `CMakeLists.txt` 和 `CMakePresets.json` 里的 `D:/thirdparty/opencv` 已全部移除，
统一按下面顺序查找：

1. 命令行 `-DOPENCV_DIR=<路径>`
2. 环境变量 `OPENCV_DIR`
3. CMake 常规搜索（系统安装目录、`CMAKE_PREFIX_PATH` 等），找不到就报标准 `Could NOT find OpenCV`

```powershell
# 只做一次；之后所有 CMake 工程（含 ocr_infer）都读这个
$env:OPENCV_DIR = "E:\thirdparty\opencv"
```

> 注意 preset 曾经自己带了 `"OPENCV_DIR": "D:/thirdparty/opencv"`，它会**盖掉**环境变量。
> 已删掉，所以如果你本地 `.vs` 或 build 目录里还缓存着旧值，删掉 `build/` 重新 configure 一次。

### 6.1 编译 Qt 桌面版 `TRTForge.exe`（2026-10-31 起可用）

主工程的 `TRTForge.vcxproj` 要链接 `ocr_infer.lib`，所以**必须先编译 OCR 子工程**。
顺序不能反。

```powershell
$env:OPENCV_DIR = "E:\thirdparty\opencv"

# 第 1 步：OCR 子工程
cd ocr\ocr_infer
cmake --preset vs2022-x64 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
      "-DPADDLE_LIB=<仓库根>/ocr/ocr_infer/third_party/paddle_inference"
cmake --build --preset vs2022-x64
# 产出：4-qt-studio/bin/{ocr_infer.dll, ocr_infer.lib, ocr_basic.exe}

# 第 2 步：主工程
cd ..\..\4-qt-studio
cmake --preset vs2022-x64
cmake --build --preset vs2022-x64
# 产出：4-qt-studio/bin/{TRTForge.exe, yolo_infer.dll, anomalib_infer.dll}
```

**两个参数为什么要这么写**

| 参数 | 作用 | 不写会怎样 |
| --- | --- | --- |
| `"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"` | 让 CMake 4.x 接受 clipper 的老式 `cmake_minimum_required(VERSION 2.6.0)` | 报 `Compatibility with CMake < 3.5 has been removed` |
| `"-DPADDLE_LIB=<...>"` | 指向 Paddle Inference 包根目录 | 默认路径是 `4-qt-studio/bin/paddle_inference`，那里的布局通常不对 |

> PowerShell 里这两个参数**必须用双引号包住**。不加引号时 `=3.5` 会被当成独立参数，
> CMake 会把它误读成 preset 名，报 `Could not read presets from ...\.5`。

**Paddle 包布局要求**

`PADDLE_LIB` 指向的目录需要：

```
paddle/include/paddle_inference_api.h
paddle/lib/paddle_inference.lib
paddle/lib/paddle_inference.dll
third_party/install/{protobuf,glog,gflags,xxhash,mkldnn,mklml}/...
```

本机验证过的版本：**Paddle 2.6.2 / GPU / CUDA 11.8 / cuDNN 8.6 / TensorRT 8.5.1.7**。

> 注意是 `mkldnn` 不是 `onednn`。`ocr_infer/CMakeLists.txt` 原先只配了 `onednn/lib`，
> 导致链接期报 `LNK1181: 无法打开输入文件 mkldnn.lib`。现已两条都保留。

**运行时依赖自动复制**

`ocr_infer` 的 POST_BUILD 会把下面 6 个 DLL 自动拷到 `4-qt-studio/bin`：

```
paddle_inference.dll   common.dll   mkldnn.dll
libiomp5md.dll         yaml-cpp.dll polyclipping.dll
```

少任何一个，`TRTForge.exe` 启动即报 `0xC0000135`（找不到依赖）。
所以**不要手工复制**，清空 `build/` 重编即可。

---

## 7. 推理验证

### 7.1 启动 Web 工作站

**最快的方式**：双击仓库根目录的 `启动Web端.bat`（自动找 Python、查端口、就绪后开浏览器）。

```bat
启动Web端.bat                :: 默认 http://127.0.0.1:8060
启动Web端.bat 8070           :: 换端口
启动Web端.bat 8070 nobrowser :: 不自动开浏览器
启动Web端.bat 8060 reload    :: 改后端代码时热重载
```

等价的 PowerShell 方式（参数更全）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_web_studio.ps1
# 浏览器打开 http://127.0.0.1:8060

# 换端口
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_web_studio.ps1 -Port 8070
```

**端口被占用怎么办**：两个脚本都会在启动前检查端口，打印「占用进程名 + PID + 启动时间」。
交互式终端里会直接问「结束它并继续启动？(Y/N)」，答 Y 自动清掉旧进程；
输入被重定向（脚本/CI 调用）时不猜，直接中止并给出清理命令：

```powershell
Stop-Process -Id <提示里的 PID> -Force
```

> 为什么会有残留进程：脚本是前台拉起 uvicorn，父进程被关掉后子进程在 Windows 上不会自动退出，
> 会一直占着端口。**更隐蔽的风险**是「以为重启了，其实还在跑旧后端」——
> 页面（静态文件每次读盘）已经是新版，后端（进程内代码）还是旧版，
> 于是新加的接口参数会被静默忽略。改动后端后务必确认 PID 变了。


### 7.2 推理页操作

1. 选后端：`YOLO` 或 `Anomalib`
2. 选引擎：`*.engine`
3. **Anomalib 必须再选 `metadata.json`**
4. 上传图片 → 查看分数、判定、叠加图、热力图

#### DLL 不可用时会自动降级（v1.8）

如果 `yolo_infer.dll` / `anomalib_infer.dll` 加载失败（没编译、缺 OpenCV/CUDA 运行库），
后端**不再直接返回 503**，而是自动切到 `onnxruntime` 跑同一份 ONNX，页面照常出结果。

怎么判断这次走的是哪条路：

| 观察点 | DLL 路径 | 降级路径 |
| --- | --- | --- |
| 结果里的 `backend` | `yolo_infer.dll` / `anomalib_infer.dll` | `onnxruntime-fallback` |
| 结果里的 `warning` | 无 | 有，说明 DLL 为什么不可用 |
| 页面「后端」指标 | 显示 DLL 名 | 显示 `onnxruntime-fallback`，并多一行「降级提示」 |
| `timing` | 四阶段齐全 | 只有 `inference_ms` 有意义，前/后处理记 0 |

降级的边界（别指望它替代 DLL）：

- YOLO 只支持 `detect` / `obb`；`segment` / `pose` 会明确报错，提示改用 DLL。
- Anomalib 只吃 `.onnx`；`.engine` 必须由 DLL 加载。选的是 `.engine` 时，会按
  「同目录同名 `.onnx` → `metadata.json` 的 `onnx_path`」自动找替代模型。
- 性能不作数：降级用 onnxruntime（CUDA provider 需 CUDA 12 + cuDNN 9，本机 CUDA 11.8 会退到 CPU）。

### 7.3 接口自检

```powershell
# 健康检查
Invoke-WebRequest http://127.0.0.1:8060/api/health
# 期望：{"ok":true}

# 能力检查（infer_fallback 为 true 表示 DLL 挂了也能降级跑）
Invoke-WebRequest http://127.0.0.1:8060/api/capabilities
```

### 7.4 精度验收方法

**不要只看单张图**，至少跑完整个 `test/`：

```python
# 用 test/good 算误报率，用 test/abnormal 算漏检率
# 判定口径：normalized_score >= 0.5 为 NG
# 验收线建议：漏检 = 0，误报率 < 5%
```

---

## 8. 常见故障排查

| 现象 | 可能原因 | 排查动作 |
| --- | --- | --- |
| `missing` 非空 | 依赖缺失 | 按输出的 `pip install` 提示补装 |
| 训练第 1 个 epoch 后退出 | 缺 `polars` | `pip install polars` |
| `DATA_NOT_FOUND` | `--data` 传的是内置名而非路径 | 传绝对路径或仓库内真实路径 |
| ONNX 导出 `TypeError: dynamo` | torch < 2.5 | 已修（`export_mixin.py` 能力探测） |
| DLL 加载失败 | 缺 `os.add_dll_directory` | 确认 OpenCV / CUDA `bin` 已加入 |
| `LNK1104` | DLL 被运行中的进程占用 | 先停 Web 服务 |
| `LNK1181: 无法打开输入文件 bin\Release\ocr_infer.lib` | 主工程需要预编译的 OCR 库 | 先单独编 `ocr_infer`（见下一条） |
| `ocr_infer` 配置报 `yaml-cpp ... does not contain a CMakeLists.txt` | `ocr/ocr_infer/third_party/yaml-cpp` 是空目录，缺源码 | 补 yaml-cpp 源码后重新 configure |
| `ocr_infer` 配置报 `Compatibility with CMake < 3.5 has been removed` | clipper 里写的是 `cmake_minimum_required(VERSION 2.6.0)`，CMake 4.x 不再兼容 | 配置时加 `"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"`（PowerShell 里必须带引号） |
| `Could NOT find OpenCV` | 既没传 `-DOPENCV_DIR` 也没设环境变量 | `$env:OPENCV_DIR = "E:\thirdparty\opencv"` |
| `LNK1181: 无法打开输入文件 mkldnn.lib` | 链接搜索目录只有 `onednn/lib`，包实际用 `mkldnn` | 已修（两条路径都保留）；若用别的 Paddle 包，确认 `third_party/install` 下是哪个名字 |
| `TRTForge.exe` 启动即退，退出码 `-1073741515`（`0xC0000135`） | 缺运行时依赖 DLL | 确认 `bin/` 里 6 个 Paddle 依赖齐全；重编 OCR 会由 POST_BUILD 自动复制 |
| `Could not read presets from ...\.5` | PowerShell 把 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 拆开了 | 整个参数用双引号包住 |
| 推理页出现 `onnxruntime-fallback` | DLL 没加载成功，走了降级 | 看 `warning` 字段的原因；要真实性能必须修好 DLL |
| 异常图判成 OK | 阈值过高 | 确认 `--calibrate-metadata true`，或重跑校准 |
| `imagenet_dir 不存在` | EfficientAd 目录缺失（`DATA_NOT_FOUND`），launcher 不会自动下载 | 先 `New-Item -ItemType Directory` 建空目录（见 2.4） |
| 正常图分数全为 0 | PatchCore 分数饱和 | 属预期；要趋势监控请换 EfficientAd |
| 引擎编译卡住不动 | builder 在 CPU 侧搜索 tactic | 看 GPU 显存仍在占用即正常，耐心等 |
| 推理页图像看不清缺陷 | 窗口未放大 | 滚轮以光标为锚点缩放，或点 `1:1` 看原始像素；多个窗口可独立开关 |
| 前端改了样式但浏览器没变 | 静态资源被缓存 | 硬刷新（`Ctrl+F5`）；后端已带 `Last-Modified`/`ETag`，正常刷新即可拿到新文件 |
| 前端配色改完对比度不达标 | 只看了「好看」没看可读性 | 跑 `node 6-web\frontend\tests\test_contrast.mjs`，它会按 WCAG 公式逐对核算 |

---

## 9. 文档索引

| 想了解 | 看这个 |
| --- | --- |
| 三个异常检测模型怎么选 | `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` |
| Web 工作站怎么用 | `6-web/README.md`（含「前端视觉体系」：设计令牌、可读性护栏、查看器） |
| 项目整体结构与层级 | `readme.md` |
| DLL C API 调用与打包 | `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md` |
| YOLO TRT 推理细节 | `docs/trtexec_gui_YOLO_TRT推理流程说明.md` |
| 小白版算法与场景说明 | `docs/TRTForge_算法与应用场景说明_小白版.md` |
| 接 VLM 自动标注 / LLM 训练诊断要花多少 token | `docs/VLM_LLM接入方案与Token成本评估.md` |
