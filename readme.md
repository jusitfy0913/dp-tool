# deeplearning 项目说明

本项目围绕工业视觉深度学习训练与 TensorRT 部署闭环建设，当前核心产品显示名为 **DefectVision AI Studio（AI 缺陷检测视觉工作站）**。项目核心方向是把标注、数据集导出、YOLO/Anomalib 训练、ONNX 导出、TensorRT 引擎构建和 C++ 推理验证整合为一套工业 AI 缺陷检测闭环平台。

## 主要模块

| 路径 | 说明 |
| --- | --- |
| `trtexec/gui` | DefectVision AI Studio Qt 桌面程序；当前编译产物仍为 `TRTForge.exe`，包含 ONNX 转 TensorRT、标注数据集、YOLO/Anomalib 训练页和推理演示页 |
| `trtexec/gui/installer` | Windows 安装包脚本（Inno Setup）：递归打包 `trtexec/gui/bin` 内已编译的 exe 与全部运行库，详见 `trtexec/gui/README.md` 中「制作 Windows 安装包」 |
| `trtexec/package_devkit.ps1` | 打给二次开发用的开发包：含 GUI 源码、YOLO/Anomalib 训练工程和预编译推理 DLL，**不含**封装后的推理实现源码 |
| `trtexec/gui/inference/yolo_infer` | YOLO TensorRT 推理 DLL，提供 `yi_*` C API（检测/分割/姿态/OBB） |
| `trtexec/gui/inference/anomalib_infer` | Anomalib TensorRT 推理 DLL，提供 `ai_*` C API，可供 TRTForge、外部 C++ 和 Python ctypes 调用 |
| `trtexec/gui/inference/ocr_infer` | PaddleOCR C++ 推理 DLL，提供 `oi_*` C API，返回文本行多边形、文本、分数和原始 JSON |
| `trtexec/gui/training/scripts` | 给 Qt/C++ 调用的 Python 训练脚本入口 |
| `ultralytics-main` | YOLO 训练、验证、导出的 Python 开源工程来源 |
| `docs` | 架构设计、推理流程、部署总结等中文文档 |
| `openspec` | 需求规格和变更提案 |

## 二次开发包

需要把 `trtexec` 发给其他开发同事、但**不提供封装后的推理源码**时，使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\trtexec\package_devkit.ps1
```

| 会打进去 | 不会打进去 |
| --- | --- |
| GUI 源码、训练启动脚本 | `yolo_infer` / `anomalib_infer` / `ocr_infer` 的 `src` 实现 |
| `ultralytics-main`、`anomalib/anomalib-main` | Anomalib 历史 `datasets` / `results` |
| 预编译 `*_infer.dll`、`*_infer.lib`、C API 头文件 | `gui/build`、日志、`.pdb` |

输出目录默认是仓库根下的 `dist/`。解压后 `trtexec`、`ultralytics-main`、`anomalib` 必须保持同一级，训练页才能自动找到 Python 训练工程。

## 当前结构优化进度

已完成批次 A、B、C、D、E：配置集中到 `AppSettings`，TensorRT 引擎构建页已拆为 `EngineBuilderTab`；数据集划分和 `dataset.yaml` 写入已抽到 dataset 服务；训练 JSON 事件解析和产物记录已抽到 training 模块；构建与训练先接入统一日志/进度事件模型，并通过 `AppEventFormatter` 统一日志颜色、前缀和 TensorRT 日志级别转换。编译产物名称仍保持 `TRTForge.exe`，`yolo_infer.dll` 与 `anomalib_infer.dll` 的源码和 C API 未变。

引擎构建页选择 ONNX 后会在后台线程自动识别第一个输入 tensor 的名称和维度，并弹出“ONNX 加载中”进度提示；识别完成后提供统一 `H/W` 输入，修改一次高度和宽度即可同步覆盖 `min / opt / max` 三组动态 shape。

## 标准数据集划分

TRTForge 标注工作台支持在底部设置 `train / val / test` 自动划分比例，默认 `80 / 10 / 10`。用户修改为合计 `100%` 的有效比例后，点击 `自动划分` 会按该比例一键分配图片归属，并通过 Qt `QSettings` 保存为下次打开 TRTForge 的默认比例；仍可选中单张图片后点击 `train / val / test` 手动微调。

标注工作台的项目文件、图片导入、类别导入、COCO/VOC/YOLO 标注导入、YOLO/COCO/VOC 导出目录会记住上一次选择的位置，下一次打开对话框时自动回到最近目录。

## 重要文档

| 文档 | 说明 |
| --- | --- |
| `docs/TRTForge_操作手册.md` | TRTForge 图文并茂操作手册：环境、四页签流程、ONNX→TRT、训练、推理与常见问题 |
| `docs/TRTForge_算法与应用场景说明_小白版.md` | TRTForge 算法能力与应用场景白话说明：YOLO/异常检测/OCR 适合什么人、什么产线场景 |
| `docs/机器视觉引导对位系统精度解构与验证方法.md` | 说明机器视觉引导对位系统中工装治具、执行机构、视觉测量、手眼标定和工艺过程误差的拆解、验证方法与现场问题归因 |
| `docs/DefectVision_AI_Studio_软件结构优化计划.md` | 第二阶段软件结构优化计划，说明 `MainWindow` 拆分、`AppSettings`、统一日志事件和目录整理策略 |
| `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md` | 说明 `yolo_infer` / `anomalib_infer` 的 C API、调用顺序、内存与依赖，以及 `package_inference_sdk.ps1` 离线打包源码与文档 |
| `docs/trtexec_gui_YOLO_TRT推理流程说明.md` | 说明 TRTForge YOLO TensorRT 推理库的加载、前处理、解码、坐标还原和 C API |
| `openspec/changes/add-anomalib-training-export` | Anomalib 异常检测训练导出集成设计，覆盖 PatchCore、PaDiM、EfficientAd 的 Qt UI、Python launcher、ONNX 导出和 metadata 契约 |

## YOLO 训练入口

## PaddleOCR OCR 推理诊断

TRTForge 的 `OCR 文本识别 OCR` 模式已支持显示实际 OCR 运行后端和耗时拆分：`device`、`run_mode`、`MKLDNN`、CPU 线程数、识别 batch size，以及 `io / pipeline / result / total` 耗时。默认推荐 `CPU + MKLDNN + FP32`；如需 `GPU 0`，需要确认 `trtexec/gui/bin/paddle_inference` 中的 Paddle Inference 是 GPU 版，否则会出现 `Please use PaddlePaddle with GPU version`，无法真正使用 GPU 加速。

`ocr_infer.dll` 的 `oi_infer_bgr` 已走内存 `cv::Mat` 输入路径，不再先写临时 PNG 文件，适合 TRTForge UI 的图片、视频和相机帧推理。

TRTForge 主工程默认不再把 OCR 源码和 Abseil / yaml-cpp / Clipper 等第三方目标拉进同一个 Visual Studio 解决方案；主工程默认只链接 `trtexec/gui/bin/ocr_infer.dll` 和 `trtexec/gui/bin/ocr_infer.lib`。如需修改或重编 OCR 推理库，请单独打开 `trtexec/gui/inference/ocr_infer` 的 CMake 工程，生成物仍输出到 `trtexec/gui/bin`，再重新编译 TRTForge 主工程。

OCR 推理页的画面区与右侧结果列表支持拖动分隔条调整比例。点击右侧某一条 OCR 识别结果后，左侧可视化图中对应文本框会以高亮颜色显示，方便在长文档里定位当前文本行。

当前 `ch_PP-OCRv4_det_infer + ch_PP-OCRv4_rec_infer` 测试耗时：`CDLA_demo/val_0633.jpg` 在 `CPU + MKLDNN + FP32` 下识别 55 行约 13.2 秒，`CDLA_demo/val_0941.jpg` 识别 83 行约 17.4 秒；关闭 MKLDNN 后 `val_0633.jpg` 约 100.6 秒。现有 Paddle Inference 包选择 `GPU 0` 会提示需要 GPU 版 PaddlePaddle，因此不能作为真实 GPU 耗时。

统一脚本：

```powershell
trtexec\gui\training\scripts\tf_yolo_launcher.py
```

支持模式：

| mode | 说明 |
| --- | --- |
| `check-env` | 检查 Python、torch、CUDA、ultralytics、onnx、onnxruntime、polars |
| `probe-model` | 读取模型任务、类别等基础信息 |
| `train` | 训练 YOLO 模型 |
| `val` | 验证模型性能 |
| `export` | 导出 ONNX，并可生成 `.transd.onnx` |

支持任务头：`detect`、`segment`、`pose`、`obb`。

## Anomalib 训练导出设计

TRTForge 训练导出页新增 `Anomalib 异常检测` 模式，支持 `PatchCore`、`PaDiM`、`EfficientAd` 的 Folder 数据集训练、ONNX 导出和 `metadata.json` 生成。

设计入口：

```powershell
openspec\changes\add-anomalib-training-export
```

Python 入口：

```powershell
trtexec\gui\training\scripts\tf_anomalib_launcher.py
```

规划模式：

| mode | 说明 |
| --- | --- |
| `check-env` | 检查 Python、torch、CUDA、anomalib、lightning、onnx、onnxruntime、PIL、numpy |
| `train-export` | 完成训练、测试、ONNX 导出和 metadata 生成 |

Anomalib 训练导出继续沿用 `QProcess + TRTFORGE_JSON`，不把 Python 嵌入 Qt 主进程，避免 Python 依赖或训练异常拖崩 TRTForge 主程序。

界面设计中会保留 `QTextBrowser` 实时日志窗口，用来显示运行 `tf_anomalib_launcher.py` 时 Python 打印的训练、验证/测试、ONNX 导出、metadata 写出和报错信息。进度条和产物列表来自结构化事件，但用户排查问题时主要看这个日志窗口。
当前训练配置页布局为：左侧任务与路径、中间训练参数、右侧 `QTextBrowser` 运行日志，下方用较矮的信息区显示 `03 / 环境与产物` 状态与产物汇总，避免压缩上方任务字段。
`01 / 任务与数据` 内部使用与 TRTForge 深色主题一致的滚动区承载表单，输入框、下拉框和按钮保持固定显示高度；当窗口高度不足时只在该卡片内部下拉滚动，不再压缩 label 与控件。`03 / 环境与产物` 同样使用深色滚动区显示多行环境与产物文本，确保文字完整显示，不因区域变矮而重叠。

## 训练初始化方式

| 方式 | 参数 | 适用场景 |
| --- | --- | --- |
| 加载预训练权重微调 | `model=.pt` 或 `model=.yaml + pretrained=权重路径` | 默认推荐，适合小中型工业数据集 |
| 从零训练 | `model=.yaml` 且 `pretrained=false` | 数据量很大或完全自定义模型结构 |
| 断点恢复 | `model=last.pt` 且 `resume=true` | 训练中断后继续 |

## 使用注意

- 不要修改仓库根目录下的 `bin`、`include`、`lib` 文件夹。
- 脚本和文档统一使用 UTF-8 编码。
- TRTForge 会将构建、训练和推理关键日志按日期保存到 `trtexec/gui/logs/YYYY-MM-DD/trtforge_YYYY-MM-DD.log`，用于现场问题追溯；日志文件使用 UTF-8 编码。
- 缺少 Python 或 Python 库时，TRTForge 训练页应弹窗提示，不应导致主程序崩溃。
- 检测环境会显示当前 `python.exe`、虚拟环境名、缺失库和对应 `pip install` 命令。
- `polars` 是当前 Ultralytics 保存训练 checkpoint 时会用到的依赖，缺失时可能在第 1 个 epoch 验证后退出。
- TRTForge 训练页会自动保存当前 Python 解释器路径，并按 `detect / segment / pose / obb` 任务头分别记住上次使用的解释器。
- 微调模式下如果误把本地 `.pt` 填到 `pretrained`，训练页会在开始训练前自动同步到 `model`，避免 Ultralytics 下载默认模型。
- 训练成功结束后，界面会弹窗汇总本次 `run_dir`、`best.pt`、`last.pt`、`results.csv` 和常见曲线图片等产物。
- 训练监控页会在每个 epoch 结束后追加深色指标行，并用带坐标轴、刻度数字和网格线的曲线控件实时更新 F1、P、R、mAP、GPU 显存等小图；混淆矩阵使用最新 PNG 缩略图控件显示；双击单个小图可在图表区放大，再双击恢复多图。
- GPU 区会实时显示显存占用和利用率进度条；标签 JSON 区会显示等待生成、生成成功路径或未生成/写入失败原因。每次开始训练会先清空上一轮监控数据；标签 JSON 不覆盖 `labels_coco.json`，同名时自动追加时间戳。
- `.transd.onnx` 仅用于 TRTForge/TensorRT 自定义推理链路，不要再用 Ultralytics `YOLO()` 直接加载推理。
- TRTForge 推理演示已对中文路径做兼容处理：`.trt` 引擎按 UTF-8 转宽路径读取，图像通过 Qt 读字节后解码，视频中文路径会临时转存到 ASCII 路径再交给 OpenCV，标签 JSON 继续使用 Qt `QFile` 读取。
- TRTForge 推理演示加载 YOLO / Anomalib TensorRT 引擎时会把耗时的 `yi_create` / `ai_create` 放到推理 worker 线程执行，并显示“正在加载模型”的进度提示，避免模型加载期间 Qt 主界面无响应。
- TRTForge 推理演示显示窗口支持滚轮缩放、鼠标左键按住拖动画面、左键双击恢复适配窗口，便于查看大图细节和检测框。
- TRTForge 推理演示新增 `异常检测 ANOMALY` 模式，只负责 Anomalib TensorRT 推理，不包含训练/导出。Anomalib 推理核心已封装为独立 `anomalib_infer.dll`，TRTForge 通过 `ai_create / ai_infer_rgb / ai_destroy` 调用，外部 C++ 和 Python ctypes 也可复用。使用时选择已有 `.engine/.trt`、选择模型类型 `EfficientAd / PaDiM / PatchCore`、加载 ONNX 导出同目录的 `metadata.json`，界面会读取 `image_threshold`、`pixel_threshold`、`pred_scores`、`anomaly_maps` 和输入尺寸参数，并允许在运行前通过滑条微调图像级/像素级阈值，输出原图、热力图叠加图、红色异常轮廓预测图三个子窗口和 OK/NG 状态；三个子窗口均支持双击全屏/再次双击恢复。
- TRTForge 推理演示新增 `OCR 文本识别 OCR` 模式，核心封装为独立 `ocr_infer.dll`，TRTForge 通过 `oi_create / oi_infer_bgr / oi_destroy` 调用 PaddleOCR C++ / Paddle Inference。界面选择检测模型目录和识别模型目录后，输出 OCR 可视化框、文本行序号、识别文本、置信度、方向角和多边形坐标；DLL 结果中的 `lines[i].xs/ys`、`lines[i].text_utf8`、`lines[i].rec_score`、`lines[i].textline_angle` 一一对应，`raw_json_utf8` 保留 `rec_texts / rec_scores / rec_polys` 原始数据。
- Anomalib ONNX 若已输出 `pred_score/anomaly_map/pred_mask` 等后处理结果，`metadata.json` 会记录 `output_contract=anomalib_post_processed_0_1` 与 `normalization=none`，TRTForge 使用 `0.5` 作为默认图像/像素阈值，并跳过额外 ImageNet 归一化，避免 PatchCore 因重复前处理导致只能伪装成 EfficientAd 才能正常推理。
- `anomalib/anomalib-main/train.py` 支持通过 `--model patchcore|padim|efficientad` 选择异常检测模型，默认 `patchcore`；导出 `model.onnx` 后会额外生成 TRTForge 部署契约 `metadata.json`。EfficientAd 会自动使用 `train_batch_size=1`、固定 Imagenette 辅助数据路径，并在导出后单独用 normal/abnormal 样本分布校准 `image_threshold` 与 `pixel_threshold`，避免直接套用 `0.5` 导致 good 图误判 NG；PatchCore 和 PaDiM 继续保持原有 `0.5` metadata 契约逻辑。同目录新增 `compare_ckpt_onnx_inference.py`，用于对比 `model.ckpt` 与 `model.onnx` 的推理耗时、score、OK/NG 和 anomaly map 差异。
