# deeplearning 项目说明

本项目围绕工业视觉深度学习训练与 TensorRT 部署闭环建设，当前核心产品显示名为 **DefectVision AI Studio（AI 缺陷检测视觉工作站）**。项目核心方向是把标注、数据集导出、YOLO/Anomalib 训练、ONNX 导出、TensorRT 引擎构建和 C++ 推理验证整合为一套工业 AI 缺陷检测闭环平台。

## 主要模块

| 路径 | 说明 |
| --- | --- |
| `6-web` | **浏览器版工作站（推荐）**：FastAPI 后端 + 零构建前端，功能覆盖训练导出与推理演示，不需要 Qt。详见 `6-web/README.md` |
| `4-qt-studio` | DefectVision AI Studio Qt 桌面程序；当前编译产物仍为 `TRTForge.exe`，包含 ONNX 转 TensorRT、标注数据集、YOLO/Anomalib 训练页和推理演示页。**已被 `6-web` 取代，保留作迁移参考** |
| `4-qt-studio/installer` | Windows 安装包脚本（Inno Setup）：递归打包 `4-qt-studio/bin` 内已编译的 exe 与全部运行库，详见 `4-qt-studio/README.md` 中「制作 Windows 安装包」 |
| `3-cpp-tensorrt/trtexec` | 命令行版 `trtexec` 封装（TensorRT C++ 采样工具、profiler/tracer 脚本） |
| `3-cpp-tensorrt/trtexec/package_devkit.ps1` | 打给二次开发用的开发包：含 GUI 源码、YOLO/Anomalib 训练工程和预编译推理 DLL，**不含**封装后的推理实现源码 |
| `3-cpp-tensorrt/yolo_infer` | YOLO TensorRT 推理 DLL，提供 `yi_*` C API（检测/分割/姿态/OBB） |
| `3-cpp-tensorrt/anomalib_infer` | Anomalib TensorRT 推理 DLL，提供 `ai_*` C API，可供 TRTForge、外部 C++ 和 Python ctypes 调用 |
| `ocr/ocr_infer` | PaddleOCR C++ 推理 DLL，提供 `oi_*` C API，返回文本行多边形、文本、分数和原始 JSON |
| `1-python-yolo/launcher` | 给 Qt/C++ 调用的 Python 训练脚本入口 |
| `6-web/backend` | Web 版后端：启动 launcher 子进程、转发训练事件、ctypes 调推理 DLL |
| `6-web/frontend` | Web 版前端：原生 HTML/JS，无 npm 构建步骤 |
| `1-python-yolo/ultralytics-main` | YOLO 训练、验证、导出的 Python 开源工程来源 |
| `docs` | 架构设计、推理流程、部署总结等中文文档 |
| `openspec` | 需求规格和变更提案 |

## GitHub 仓库

远程地址：[https://github.com/jusitfy0913/dp-tool](https://github.com/jusitfy0913/dp-tool)

仓库里是 **C++ / Python 训练和推理源码**，方便在别的电脑继续开发。  
**没有**把编译结果推上去，例如 `TRTForge.exe`、`*.dll`、`4-qt-studio/bin`、TensorRT 引擎、训练权重、数据集。

本地编译仍在本机的 `4-qt-studio/bin` 进行，不要把这个目录提交到 Git。

## 二次开发包

需要把开发包发给其他开发同事、但**不提供封装后的推理源码**时，使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\3-cpp-tensorrt\trtexec\package_devkit.ps1
```

| 会打进去 | 不会打进去 |
| --- | --- |
| GUI 源码、训练启动脚本 | `yolo_infer` / `anomalib_infer` / `ocr_infer` 的 `src` 实现 |
| `1-python-yolo/ultralytics-main`、`2-python-anomaly/anomalib-main` | Anomalib 历史 `datasets` / `results` |
| 预编译 `*_infer.dll`、`*_infer.lib`、C API 头文件 | `4-qt-studio/build`、日志、`.pdb` |

输出目录默认是仓库根下的 `dist/`。解压后请保持下面五级目录的相对位置，训练页和编译脚本都按这个位置找工程。

## 目录分级（2026-10-28）

老版本把训练源码、异常检测、旧 TensorRT 示例和 Qt 程序平铺在仓库根目录，TensorRT 动态库放在 `trtexec/gui/inference` 里面。新版本按用途分成五级，启动脚本和打包脚本已改成新路径。

| 级别 | 现在的路径 | 原来的路径 | 用途 |
| --- | --- | --- | --- |
| YOLO 训练 | `1-python-yolo/ultralytics-main`、`1-python-yolo/launcher/tf_yolo_launcher.py` | `ultralytics-main`、`trtexec/gui/training/scripts/tf_yolo_launcher.py` | 检测、分割、姿态、旋转框训练和导出 |
| 异常检测 | `2-python-anomaly/anomalib-main`、`2-python-anomaly/launcher/tf_anomalib_launcher.py` | `anomalib/anomalib-main`、`trtexec/gui/training/scripts/tf_anomalib_launcher.py` | PatchCore、PaDiM、EfficientAd 训练和导出 |
| TensorRT 动态库 | `3-cpp-tensorrt/yolo_infer`、`3-cpp-tensorrt/anomalib_infer` | `trtexec/gui/inference/yolo_infer`、`trtexec/gui/inference/anomalib_infer` | `yolo_infer.dll`、`anomalib_infer.dll`，产物仍输出到 `4-qt-studio/bin` |
| Qt 界面 | `4-qt-studio` | `trtexec/gui` | 标注、训练、ONNX 转 TensorRT、推理四个页面，程序名仍是 `TRTForge.exe` |
| 旧示例 | `5-reference/YOLOv8-det`、`YOLOv8-seg`、`YOLOv8_obb`、`YOLOv8-TensorRT-main` | 仓库根目录同名文件夹 | 早期对照工程，界面编译不引用 |
| 文字识别 | `ocr/PaddleOCR-main`、`ocr/ocr_infer` | `PaddleOCR-main`、`trtexec/gui/inference/ocr_infer` | Paddle 推理，不是 TensorRT |
| Web 界面（新） | `6-web/backend`、`6-web/frontend` | 无（新增层级） | 浏览器版工作站，调用同一批 launcher 与推理 DLL，不依赖 Qt |
| 构建与文档 | `3-cpp-tensorrt/trtexec`、`scripts/`、`docs/`、`openspec/` | 根目录 `trtexec/`，脚本内部引用旧 `gui/` 路径 | 命令行 trtexec、打包脚本、Paddle 构建脚本、文档与规范 |

层级依赖方向：`1-python-yolo` 与 `2-python-anomaly` 被 Qt 侧通过 `QProcess` 调用；`3-cpp-tensorrt` 与 `ocr/ocr_infer` 被 `4-qt-studio` 通过 `add_subdirectory` 引入或导入预编译 DLL；`5-reference` 不参与任何构建。所有编译产物（`TRTForge.exe`、`yolo_infer.dll`、`anomalib_infer.dll`、`ocr_infer.dll` 及对应 `.lib`）统一输出到 `4-qt-studio/bin`。

Qt 界面内部仍按四个功能分目录：`src/annotation` 标注，`src/training` 训练，`src/engine` 模型转换，`src/inference` 推理页。命令行版 trtexec 随 TensorRT 层放在 `3-cpp-tensorrt/trtexec`。

### 脚本路径推导与环境变量

脚本不再写死 `D:\test1\deeplearning\...` 这类机器绝对路径，全部按脚本自身位置推导：

| 脚本 | 默认推导结果 |
| --- | --- |
| `scripts/build_paddle_inference_win_gpu_trt.ps1` | Paddle 源码 `<仓库根>/third_party/Paddle`，构建目录 `<仓库根>/third_party/Paddle/build_infer_trt` |
| `scripts/sync_paddle_inference_to_bin.ps1` | 安装目录 `<仓库根>/third_party/Paddle/build_infer_trt/paddle_inference_install_dir`，目标 `<仓库根>/4-qt-studio/bin` |
| `scripts/paddle_infer_configure_build_sync.bat`、`paddle_infer_build_sync_only.bat` | 仓库根取 `%~dp0..`，日志落在 `<仓库根>/third_party/Paddle/build_logs` |
| `3-cpp-tensorrt/trtexec/package_devkit.ps1` | 向上探测同时含 `4-qt-studio` 与 `1-python-yolo` 的目录作为仓库根 |
| `2-python-anomaly/anomalib-main/train.py` | 数据集 `<该脚本目录>/datasets/bottle` |
| `1-python-yolo/ultralytics-main/yolo_obb.py`、`yolo_pose.py` | 测试图取 `5-reference` 下对应工程 |

需要指向别处时用环境变量覆盖，不必改代码：

| 环境变量 | 作用 |
| --- | --- |
| `PADDLE_THIRD_PARTY_ROOT` | Paddle 工作区根目录 |
| `PADDLE_SRC`、`PADDLE_BUILD` | Paddle 源码目录、构建目录 |
| `PADDLE_WORK_ROOT`、`PADDLE_LOG_DIR`、`PADDLE_INSTALL_DIR` | `.bat` 中的工作区、日志、安装目录 |
| `PYTHON_EXE` | 构建 Paddle 时使用的 Python |
| `TRTFORGE_ANOMALIB_DATASET_ROOT` | Anomalib 数据集根目录 |
| `TRTFORGE_OBB_IMAGE`、`TRTFORGE_POSE_IMAGE` | 两个导出脚本的测试图 |

`third_party/` 已加入 `.gitignore`，Paddle 源码与构建产物不会进入版本库。

### 训练环境准备（Python 解释器）

程序不内嵌 Python，训练时的解释器由训练页 `Python 环境` 指定，再对每个任务用 `QProcess` 拉起。已验证可用的环境要求：

| 链路 | 必需依赖 |
| --- | --- |
| YOLO 训练 / 导出 | `torch`、`torchvision`、`ultralytics`、`polars`、`onnx`、`onnxruntime` |
| Anomalib 训练 / 导出 | `torch`、`anomalib`、`lightning`、`onnx`、`onnxruntime`、`Pillow`、`numpy<2` |

一次性装齐（`numpy<2` 必须保留，torch 2.3 与 numpy 2 存在 ABI 冲突）：

```powershell
python -m pip install ultralytics polars "numpy<2"
```

验证方式：训练页点「检测环境」，或命令行

```powershell
cd 1-python-yolo\launcher
python tf_yolo_launcher.py --mode check-env
cd ..\..\2-python-anomaly\launcher
python tf_anomalib_launcher.py --mode check-env
```

两条都输出 `"missing":[]` 才算环境可用。注意：
- 仓库里没有 `.venv` 时，程序只能从 `VIRTUAL_ENV` / `CONDA_PREFIX` / PATH 探测，base 环境通常没有 torch，因此必须手动填 `python.exe` 路径，否则报「未找到可用 Python」。
- `.gitignore` 中旧的 `ultralytics/`、`anomalib/` 已改为锚定仓库根的 `/ultralytics/`、`/anomalib/`。旧写法会把 `1-python-yolo/ultralytics-main/ultralytics` 与 `2-python-anomaly/anomalib-main/src/anomalib` 整包忽略，导致训练源码进不了版本库。

## 当前结构优化进度

已完成批次 A、B、C、D、E：配置集中到 `AppSettings`，TensorRT 引擎构建页已拆为 `EngineBuilderTab`；数据集划分和 `dataset.yaml` 写入已抽到 dataset 服务；训练 JSON 事件解析和产物记录已抽到 training 模块；构建与训练先接入统一日志/进度事件模型，并通过 `AppEventFormatter` 统一日志颜色、前缀和 TensorRT 日志级别转换。编译产物名称仍保持 `TRTForge.exe`，`yolo_infer.dll` 与 `anomalib_infer.dll` 的源码和 C API 未变。

引擎构建页选择 ONNX 后会在后台线程自动识别第一个输入 tensor 的名称和维度，并弹出“ONNX 加载中”进度提示；识别完成后提供统一 `H/W` 输入，修改一次高度和宽度即可同步覆盖 `min / opt / max` 三组动态 shape。

## 版本更新记录

### v1.3 — 2026-10-30（PCB 真实数据集全链路打通 + 三个阻断级 bug 修复）

本次用现场采集的 `PCB data/PCB缺陷检测`（553 train / 139 val，6 类，原图 3034×1586）跑通了
「训练 → ONNX → TensorRT 引擎 → Web 推理」全流程，并修复过程中暴露的三个真实缺陷。

**实测数据（YOLOv8n @960, T1000 8GB, FP16）**

| 阶段 | 结果 |
| --- | --- |
| 训练 | 30 epoch，约 27 分钟，最终 P=0.951 / R=0.930 / mAP50=0.945 / mAP50-95=0.514 |
| ONNX | 12.39 MB，输入 `[1,3,960,960]`，输出 `[1,10,18900]`；转置版 `[1,18900,10]` |
| ONNX 复测精度 | P=0.9499 / R=0.9295 / mAP50=0.9451（与 `.pt` 一致，导出无损） |
| 引擎编译 | 515 秒，10.28 MB |
| 引擎性能 | median 8.41 ms / 125 qps（ONNX Runtime 为 47.2 ms，快 5.6 倍） |
| 端到端推理 | 4 张验证图各检出 3 个 `missing_hole`，置信度 0.76~0.95，与文件名标注吻合 |

| 缺陷 | 老版本表现 | 新版本修复 | 影响 |
| --- | --- | --- | --- |
| YOLO 引擎输出布局 | 用 `best.onnx` 编译引擎时，`yolo_infer.dll` 把类别数算成 `18900-4`，越界读显存，**静默**输出 `confidence=957`、`top=-725` 的垃圾框 | 文档明确「必须用 `.transd.onnx` 编译」；`app.py` 新增 `_check_yolo_output_sane()`，置信度越界直接返回 HTTP 400 并指明原因 | 从「看起来成功、结果全错」变为「明确报错 + 可执行提示」 |
| `--project` 相对路径 | Ultralytics 8.4 对相对路径再拼 `<runs_dir>/<task>/`，`runs/pcb` 实际落到 `runs/detect/runs/pcb` | launcher 在 `build_train_kwargs()` 里先解析成绝对路径 | 产物路径从 `runs/detect/runs/pcb/...` 回到 `runs/pcb/...` |
| Web 启动脚本选错解释器 | `Get-Command python.exe` 命中 Microsoft Store 的 0 字节存根，服务**静默起不来**（端口不监听、无报错） | `run_web_studio.ps1` 改为「已知 conda 环境优先 + 真实可用性探测」，跳过 `WindowsApps` 目录和 0 字节文件 | 按 README 一键启动即可用 |

**新增工具**

| 脚本 | 用途 |
| --- | --- |
| `6-web/backend/tools/verify_pcb_chain.py` | PCB 全链路验证：调 Web 推理接口，汇总框数/类别/置信度，输出标注图与 JSON 报告 |
| `6-web/backend/tools/diag_yolo_dll.py` | 绕过 Web 层直接调 `yolo_infer.dll`，dump `frame_stat` 与 box 原始内存，用于定位布局/结构体问题 |
| `6-web/backend/tools/check_engine_layout_guard.py` | 对比「错误引擎」与「正确引擎」在推理接口上的返回，验证防护生效 |
| `1-python-yolo/launcher/configs/pcb_detect.yaml` | PCB 数据集配置（含统计信息与 imgsz 选择建议） |

**注意**：PCB 数据集原图的缺陷只有约 70px，`imgsz=640` 时会缩到约 15px，建议 **960 及以上**。

### v1.2 — 2026-10-30（异常检测模型对比与阈值校准）

本次为功能性大版本更新，核心是三模型实测对比 + 阈值校准修复。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| ONNX 导出（torch 2.3） | 训练完成后导出直接崩溃：`TypeError: export() got an unexpected keyword argument 'dynamo'` | `export_mixin.py` 增加能力探测，torch < 2.5 不传该参数 | Anomalib 训练导出链路从「必定失败」变为可用 |
| 后处理契约模型阈值 | 固定 `0.5` | 用正常 + 异常校准样本自动标定 | PatchCore 漏检 2/44 → 0/44 |
| 阈值校准覆盖范围 | 仅 `EfficientAd` | `PatchCore` / `PaDiM` / `EfficientAd` 全覆盖 | 三个模型都不用手工调阈值 |
| 校准样本取样策略 | 按目录顺序取满即止（`train/good` 会占满配额） | 多目录轮转取样 | 校准样本来源更均衡，阈值更保守可靠 |
| `metadata.json` 的 `calibration` 字段 | 只有计数 | 增加 `sample_accuracy`、`normal_score_range`、`abnormal_score_range`、`previous_default_threshold` | 可追溯每次校准的质量 |
| 新增文档 | 无三模型横向对比 | `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` | 选型有据可依 |
| 新增文档 | 无全流程指南 | `docs/WORKFLOW_GUIDE.md` | 环境→训练→转换→推理→排障一站到底 |

改动文件清单：

| 文件 | 改动 |
| --- | --- |
| `2-python-anomaly/anomalib-main/src/anomalib/models/components/base/export_mixin.py` | `torch.onnx.export` 的 `dynamo` 参数改为能力探测 |
| `2-python-anomaly/anomalib-main/trtforge_metadata.py` | 新增 `calibrate_postprocessed_thresholds()`、`_threshold_between_distributions()`、`_calibration_percentile()` |
| `2-python-anomaly/launcher/tf_anomalib_launcher.py` | 后处理契约分支接入阈值校准；`collect_images()` 改为多目录轮转取样 |
| `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` | 新增 |
| `docs/WORKFLOW_GUIDE.md` | 新增 |
| `readme.md` | 新增模型速查表、阈值校准说明、版本更新记录 |

### v1.1 — 2026-10-28（五级目录重构）

| 项目 | 老版本 | 新版本 |
| --- | --- | --- |
| 目录结构 | 根目录平铺 | 五级分层：`1-python-yolo` / `2-python-anomaly` / `3-cpp-tensorrt` / `4-qt-studio` / `5-reference`，另加 `6-web` 与 `ocr` |
| 界面形态 | Qt 桌面程序 `TRTForge.exe` | 浏览器工作站 `6-web`（FastAPI + 零构建前端），Qt 版保留作迁移参考 |
| 脚本路径 | 硬编码 `D:\test1\deeplearning\...` | 全部按脚本自身位置推导，可用环境变量覆盖 |
| `.gitignore` | `ultralytics/`、`anomalib/` 会误伤训练源码 | 改为锚定仓库根的 `/ultralytics/`、`/anomalib/` |

## Web 版工作站（6-web，推荐）

不用 Qt、不用编译 `TRTForge.exe`，浏览器里就能跑训练和推理。

### 为什么迁

Qt 版要求每台机器先装 Qt 6.8.3 + MSVC 才能编出界面。本机长期没有 Qt，界面一直不可用。
Web 版把界面层换成浏览器，训练仍走原来的 launcher、推理仍走原来编好的 DLL，**算法与推理实现一行未改**。

### 三步启动

```powershell
# 1) 装依赖（用装了 torch 的环境）
D:\Anaconda3\envs\dpenv\python.exe -m pip install -r 6-web\backend\requirements.txt

# 2) 编推理 DLL（只需一次，不依赖 Qt，但要 OpenCV）
$env:OPENCV_DIR = "E:\thirdparty\opencv"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_infer_dlls.ps1

# 3) 启动
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
```

然后打开 <http://127.0.0.1:8060>。顶部状态灯显示「环境就绪」即可使用。

### 已实现能力

| 页面 | 能力 |
| --- | --- |
| 训练导出 | 选 YOLO / Anomalib、填参数、环境自检、实时日志与进度、指标显示、一键停止 |
| 推理演示 | 上传图片、调 `yolo_infer.dll` / `anomalib_infer.dll`、出检测框或异常热力图 |
| 环境说明 | 展示后端探测到的解释器、launcher、DLL 加载状态与环境变量 |

标注页与 ONNX→TensorRT 转换页尚未迁移，仍用 Qt 版或命令行 `trtexec`。

### 关键环境变量

| 变量 | 作用 |
| --- | --- |
| `TRTFORGE_PYTHON` | 训练用 `python.exe`，不设则自动探测 |
| `OPENCV_DIR` | OpenCV 根目录，推理 DLL 运行时必需，如 `E:\thirdparty\opencv` |
| `CUDA_PATH` | CUDA 根目录，推理 DLL 运行时必需 |
| `TRTFORGE_BROWSE_ROOTS` | 额外允许浏览的目录（分号分隔），默认只允许仓库内 |

完整说明见 `6-web/README.md`；设计决策见 `openspec/changes/add-web-studio/`。

### 踩坑记录：CUDA 11.8 + MSVC 14.44

编译 `yolo_infer` 会报：

```text
error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer.
```

MSVC 14.44 的 STL 头文件（`yvals_core.h:901`）硬性拒绝 CUDA < 12.4。CMakeLists 里虽写了
绕过宏，但 CMake 在**探测 CUDA 编译器**阶段不传 `CMAKE_CUDA_FLAGS`，所以宏没生效。
有效解法是环境变量，它能覆盖所有 nvcc 调用：

```powershell
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"
```

`scripts\build_infer_dlls.ps1` 已自动设置。根治方案是升级 CUDA 12.4+ 或改用 VS2019 工具集。

另注意：Python 3.8+ **不再用 `PATH` 解析依赖 DLL**，光把 OpenCV 的 `bin` 加进 `PATH` 仍会加载失败，
必须用 `os.add_dll_directory`。`6-web/backend/infer.py` 已处理。

## 标准数据集划分

TRTForge 标注工作台支持在底部设置 `train / val / test` 自动划分比例，默认 `80 / 10 / 10`。用户修改为合计 `100%` 的有效比例后，点击 `自动划分` 会按该比例一键分配图片归属，并通过 Qt `QSettings` 保存为下次打开 TRTForge 的默认比例；仍可选中单张图片后点击 `train / val / test` 手动微调。

标注工作台的项目文件、图片导入、类别导入、COCO/VOC/YOLO 标注导入、YOLO/COCO/VOC 导出目录会记住上一次选择的位置，下一次打开对话框时自动回到最近目录。

## 重要文档

| 文档 | 说明 |
| --- | --- |
| `6-web/README.md` | **Web 版工作站说明**：安装、启动、页面功能、后端接口、环境变量、常见问题 |
| `docs/TRTForge_操作手册.md` | TRTForge 图文并茂操作手册：环境、四页签流程、ONNX→TRT、训练、推理与常见问题 |
| `docs/TRTForge_算法与应用场景说明_小白版.md` | TRTForge 算法能力与应用场景白话说明：YOLO/异常检测/OCR 适合什么人、什么产线场景 |
| `docs/机器视觉引导对位系统精度解构与验证方法.md` | 说明机器视觉引导对位系统中工装治具、执行机构、视觉测量、手眼标定和工艺过程误差的拆解、验证方法与现场问题归因 |
| `docs/DefectVision_AI_Studio_软件结构优化计划.md` | 第二阶段软件结构优化计划，说明 `MainWindow` 拆分、`AppSettings`、统一日志事件和目录整理策略 |
| `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` | **三模型选型必读**：PatchCore / PaDiM / EfficientAd 的原理、训练耗时、ONNX 与引擎体积、编译耗时、推理延迟、实测准确率、阈值校准与踩坑记录 |
| `docs/WORKFLOW_GUIDE.md` | **全流程作业指南**：环境准备 → 标注 → 训练 → ONNX 导出 → trtexec 编译 → 推理验证 → 故障排查 |
| `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md` | 说明 `yolo_infer` / `anomalib_infer` 的 C API、调用顺序、内存与依赖，以及 `package_inference_sdk.ps1` 离线打包源码与文档 |
| `docs/trtexec_gui_YOLO_TRT推理流程说明.md` | 说明 TRTForge YOLO TensorRT 推理库的加载、前处理、解码、坐标还原和 C API |
| `openspec/changes/add-anomalib-training-export` | Anomalib 异常检测训练导出集成设计，覆盖 PatchCore、PaDiM、EfficientAd 的 Qt UI、Python launcher、ONNX 导出和 metadata 契约 |

## YOLO 训练入口

## PaddleOCR OCR 推理诊断

TRTForge 的 `OCR 文本识别 OCR` 模式已支持显示实际 OCR 运行后端和耗时拆分：`device`、`run_mode`、`MKLDNN`、CPU 线程数、识别 batch size，以及 `io / pipeline / result / total` 耗时。默认推荐 `CPU + MKLDNN + FP32`；如需 `GPU 0`，需要确认 `4-qt-studio/bin/paddle_inference` 中的 Paddle Inference 是 GPU 版，否则会出现 `Please use PaddlePaddle with GPU version`，无法真正使用 GPU 加速。

`ocr_infer.dll` 的 `oi_infer_bgr` 已走内存 `cv::Mat` 输入路径，不再先写临时 PNG 文件，适合 TRTForge UI 的图片、视频和相机帧推理。

TRTForge 主工程默认不再把 OCR 源码和 Abseil / yaml-cpp / Clipper 等第三方目标拉进同一个 Visual Studio 解决方案；主工程默认只链接 `4-qt-studio/bin/ocr_infer.dll` 和 `4-qt-studio/bin/ocr_infer.lib`。如需修改或重编 OCR 推理库，请单独打开 `ocr/ocr_infer` 的 CMake 工程，生成物仍输出到 `4-qt-studio/bin`，再重新编译 TRTForge 主工程。

OCR 推理页的画面区与右侧结果列表支持拖动分隔条调整比例。点击右侧某一条 OCR 识别结果后，左侧可视化图中对应文本框会以高亮颜色显示，方便在长文档里定位当前文本行。

当前 `ch_PP-OCRv4_det_infer + ch_PP-OCRv4_rec_infer` 测试耗时：`CDLA_demo/val_0633.jpg` 在 `CPU + MKLDNN + FP32` 下识别 55 行约 13.2 秒，`CDLA_demo/val_0941.jpg` 识别 83 行约 17.4 秒；关闭 MKLDNN 后 `val_0633.jpg` 约 100.6 秒。现有 Paddle Inference 包选择 `GPU 0` 会提示需要 GPU 版 PaddlePaddle，因此不能作为真实 GPU 耗时。

统一脚本：

```powershell
1-python-yolo\launcher\tf_yolo_launcher.py
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

### 三个异常检测模型速查

实测环境：T1000 8GB / CUDA 11.8 / TensorRT 8.5.2 / 输入 256×256 / FP16 引擎，数据集 `bottle`（训练正常图 225，测试正常 20 + 异常 44）。完整分析见 `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md`。

| 对比项 | PatchCore | PaDiM | EfficientAd |
| --- | --- | --- | --- |
| 建模方式 | 特征最近邻（memory bank） | 逐位置多元高斯 | 学生-教师知识蒸馏 |
| 是否需要梯度训练 | 否 | 否 | **是（20 epoch）** |
| 额外数据依赖 | 无 | 无 | **imagenette2（9469 张 / 1485MB）** |
| 训练阶段是否读异常图 | 否 | 否 | **是（难样本挖掘）** |
| 训练+导出耗时 | 3 分 42 秒 | **28 秒** | 59 分 35 秒 |
| ONNX 大小 | 301.55 MB | 168.46 MB | **30.78 MB** |
| TRT 引擎（FP16） | 275.9 MB | 87.2 MB | **19.4 MB** |
| trtexec 编译耗时 | 924 秒 | 1023 秒 | **480 秒** |
| TRT 推理（中位） | 24.76 ms | **3.56 ms** | 23.79 ms |
| 实测准确率 | **1.0000** | 0.9844 | **1.0000** |
| 正常图分数区分度 | **1 种（全 0，饱和）** | 4 种 | **19 种** |
| `metadata` 契约 | `anomalib_post_processed_0_1` | `anomalib_post_processed_0_1` | `efficientad_distribution_calibrated` |

选型建议：

| 诉求 | 选 | 理由 |
| --- | --- | --- |
| 产线要快、模型要小 | PaDiM | 3.56ms，引擎 87MB，训练 28 秒 |
| 最高检出率、允许等训练 | EfficientAd | 100% 准确率，分数有区分度 |
| 样本少、快速验证链路 | PatchCore / PaDiM | 分钟级内出结果 |
| 质量趋势监控 | EfficientAd | 只有它给出连续可用分数 |

**不要把 PatchCore 的正常图分数用于趋势判断**：它的后处理归一化统计量在全正常验证集上退化，正常图分数被 clip 到 0，看不出良品之间的差异。

### 阈值校准（2026-10-30 更新）

老版本 `metadata.json` 的 `image_threshold` 固定为 `0.5`，实测导致 PatchCore 在 bottle 上漏检 2/44。原因是 anomalib 的自适应阈值依赖「含异常样本的验证集」，而本工程的验证集从 `train/good` 切出，全是正常图，算不出有效阈值。

新版本在 `anomalib-main/trtforge_metadata.py` 新增 `calibrate_postprocessed_thresholds()`：用正常图 99 分位与异常图 5 分位求可分中点，结果写回 `metadata.json` 的 `image_threshold` / `pixel_threshold`，并在 `calibration` 字段记录样本准确率。**只改 metadata，不需要重新编译引擎。**

| 版本 | 阈值 | 异常检出 | 正常误判 | 准确率 |
| --- | --- | --- | --- | --- |
| 老 | 固定 0.5 | 42/44 | 0/20 | 0.969 |
| 新 | 0.1720（自动标定） | **44/44** | 0/20 | **1.000** |

设计入口：

```powershell
openspec\changes\add-anomalib-training-export
```

Python 入口：

```powershell
2-python-anomaly\launcher\tf_anomalib_launcher.py
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
- TRTForge 会将构建、训练和推理关键日志按日期保存到 `4-qt-studio/logs/YYYY-MM-DD/trtforge_YYYY-MM-DD.log`，用于现场问题追溯；日志文件使用 UTF-8 编码。
- 缺少 Python 或 Python 库时，TRTForge 训练页应弹窗提示，不应导致主程序崩溃。
- 检测环境会显示当前 `python.exe`、虚拟环境名、缺失库和对应 `pip install` 命令。
- `polars` 是当前 Ultralytics 保存训练 checkpoint 时会用到的依赖，缺失时可能在第 1 个 epoch 验证后退出。
- TRTForge 训练页会自动保存当前 Python 解释器路径，并按 `detect / segment / pose / obb` 任务头分别记住上次使用的解释器。
- 微调模式下如果误把本地 `.pt` 填到 `pretrained`，训练页会在开始训练前自动同步到 `model`，避免 Ultralytics 下载默认模型。
- 训练成功结束后，界面会弹窗汇总本次 `run_dir`、`best.pt`、`last.pt`、`results.csv` 和常见曲线图片等产物。
- 训练监控页会在每个 epoch 结束后追加深色指标行，并用带坐标轴、刻度数字和网格线的曲线控件实时更新 F1、P、R、mAP、GPU 显存等小图；混淆矩阵使用最新 PNG 缩略图控件显示；双击单个小图可在图表区放大，再双击恢复多图。
- GPU 区会实时显示显存占用和利用率进度条；标签 JSON 区会显示等待生成、生成成功路径或未生成/写入失败原因。每次开始训练会先清空上一轮监控数据；标签 JSON 不覆盖 `labels_coco.json`，同名时自动追加时间戳。
- `.transd.onnx` 仅用于 TRTForge/TensorRT 自定义推理链路，不要再用 Ultralytics `YOLO()` 直接加载推理。反过来，**编译 YOLO 的 TensorRT 引擎时必须用它**：`yolo_infer.dll` 按 `[1, 候选框数, 4+类别数]` 解析输出，直接用 `best.onnx`（`[1, 4+类别数, 候选框数]`）编译会让类别数被算成 `候选框数-4`，解码越界读显存且**不报错**，输出 `confidence=957`、`top=-725` 这类垃圾框。Web 接口已加校验，遇到这种情况返回 HTTP 400 并提示改用 `.transd.onnx`。
- TRTForge 推理演示已对中文路径做兼容处理：`.trt` 引擎按 UTF-8 转宽路径读取，图像通过 Qt 读字节后解码，视频中文路径会临时转存到 ASCII 路径再交给 OpenCV，标签 JSON 继续使用 Qt `QFile` 读取。
- TRTForge 推理演示加载 YOLO / Anomalib TensorRT 引擎时会把耗时的 `yi_create` / `ai_create` 放到推理 worker 线程执行，并显示“正在加载模型”的进度提示，避免模型加载期间 Qt 主界面无响应。
- TRTForge 推理演示显示窗口支持滚轮缩放、鼠标左键按住拖动画面、左键双击恢复适配窗口，便于查看大图细节和检测框。
- TRTForge 推理演示新增 `异常检测 ANOMALY` 模式，只负责 Anomalib TensorRT 推理，不包含训练/导出。Anomalib 推理核心已封装为独立 `anomalib_infer.dll`，TRTForge 通过 `ai_create / ai_infer_rgb / ai_destroy` 调用，外部 C++ 和 Python ctypes 也可复用。使用时选择已有 `.engine/.trt`、选择模型类型 `EfficientAd / PaDiM / PatchCore`、加载 ONNX 导出同目录的 `metadata.json`，界面会读取 `image_threshold`、`pixel_threshold`、`pred_scores`、`anomaly_maps` 和输入尺寸参数，并允许在运行前通过滑条微调图像级/像素级阈值，输出原图、热力图叠加图、红色异常轮廓预测图三个子窗口和 OK/NG 状态；三个子窗口均支持双击全屏/再次双击恢复。
- TRTForge 推理演示新增 `OCR 文本识别 OCR` 模式，核心封装为独立 `ocr_infer.dll`，TRTForge 通过 `oi_create / oi_infer_bgr / oi_destroy` 调用 PaddleOCR C++ / Paddle Inference。界面选择检测模型目录和识别模型目录后，输出 OCR 可视化框、文本行序号、识别文本、置信度、方向角和多边形坐标；DLL 结果中的 `lines[i].xs/ys`、`lines[i].text_utf8`、`lines[i].rec_score`、`lines[i].textline_angle` 一一对应，`raw_json_utf8` 保留 `rec_texts / rec_scores / rec_polys` 原始数据。
- Anomalib ONNX 若已输出 `pred_score/anomaly_map/pred_mask` 等后处理结果，`metadata.json` 会记录 `output_contract=anomalib_post_processed_0_1` 与 `normalization=none`，并跳过额外 ImageNet 归一化，避免 PatchCore 因重复前处理导致只能伪装成 EfficientAd 才能正常推理。`image_threshold` / `pixel_threshold` 的取值见上方「阈值校准」小节：老版本固定 `0.5`，现版本用正常 + 异常校准样本自动标定。
- `2-python-anomaly/anomalib-main/train.py` 支持通过 `--model patchcore|padim|efficientad` 选择异常检测模型，默认 `patchcore`；导出 `model.onnx` 后会额外生成 TRTForge 部署契约 `metadata.json`。EfficientAd 会自动使用 `train_batch_size=1`、固定 Imagenette 辅助数据路径。**PatchCore / PaDiM / EfficientAd 三个模型现在都会做阈值校准**（EfficientAd 走 `calibrate_efficientad_metadata` 独立分支，另两个走 `calibrate_postprocessed_thresholds`），不再直接套用 `0.5`。同目录新增 `compare_ckpt_onnx_inference.py`，用于对比 `model.ckpt` 与 `model.onnx` 的推理耗时、score、OK/NG 和 anomaly map 差异。
- **注意两条链路的默认 epoch 不一致**：`tf_anomalib_launcher.py` 的 `--max-epochs auto` 为 PatchCore/PaDiM = 1、EfficientAd = 20；而独立脚本 `anomalib-main/train.py` 为 PaDiM = 1、EfficientAd = 5、其他 = 50。以实际调用的那条链路为准，不要混用。
