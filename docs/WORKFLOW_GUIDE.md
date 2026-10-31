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
| YOLOv8n（640） | 约 2~3 分钟 |
| YOLOv8n（960） | 约 8 分钟（474 秒，首次无 timing cache） |

> **必须串行编译**。并行会互相抢 GPU，实测明显变慢。
> 编译期间 GPU 利用率掉到 4% 是正常的（builder 在 CPU 侧搜索 tactic）。
> 反过来，若长时间没有日志输出但 GPU 占用接近 100%，说明仍在 tactic 搜索，
> 不是在卡死，不要中断。

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

构建产物路径可直接填进「推理演示」页的引擎文件栏。


---

## 6. 编译 C++ 推理 DLL（仅在改了 C++ 源码时需要）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_infer_dlls.ps1
```

- 产物：`4-qt-studio/bin/{yolo_infer.dll, anomalib_infer.dll}`
- 不依赖 Qt，只需 OpenCV + CUDA
- **编译前必须先停掉 Web 服务**，否则 DLL 被占用会报 `LNK1104`

---

## 7. 推理验证

### 7.1 启动 Web 工作站

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_web_studio.ps1
# 浏览器打开 http://127.0.0.1:8060
```

### 7.2 推理页操作

1. 选后端：`YOLO` 或 `Anomalib`
2. 选引擎：`*.engine`
3. **Anomalib 必须再选 `metadata.json`**
4. 上传图片 → 查看分数、判定、叠加图、热力图

### 7.3 接口自检

```powershell
# 健康检查
Invoke-WebRequest http://127.0.0.1:8060/api/health
# 期望：{"ok":true}

# 能力检查
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
| 异常图判成 OK | 阈值过高 | 确认 `--calibrate-metadata true`，或重跑校准 |
| `imagenet_dir 不存在` | EfficientAd 目录缺失（`DATA_NOT_FOUND`），launcher 不会自动下载 | 先 `New-Item -ItemType Directory` 建空目录（见 2.4） |
| 正常图分数全为 0 | PatchCore 分数饱和 | 属预期；要趋势监控请换 EfficientAd |
| 引擎编译卡住不动 | builder 在 CPU 侧搜索 tactic | 看 GPU 显存仍在占用即正常，耐心等 |

---

## 9. 文档索引

| 想了解 | 看这个 |
| --- | --- |
| 三个异常检测模型怎么选 | `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` |
| Web 工作站怎么用 | `6-web/README.md` |
| 项目整体结构与层级 | `readme.md` |
| DLL C API 调用与打包 | `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md` |
| YOLO TRT 推理细节 | `docs/trtexec_gui_YOLO_TRT推理流程说明.md` |
| 小白版算法与场景说明 | `docs/TRTForge_算法与应用场景说明_小白版.md` |
