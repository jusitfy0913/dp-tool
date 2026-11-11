# deeplearning 项目说明

本项目围绕工业视觉深度学习训练与 TensorRT 部署闭环建设，当前核心产品显示名为 **DefectVision AI Studio（AI 缺陷检测视觉工作站）**。项目核心方向是把标注、数据集导出、YOLO/Anomalib 训练、ONNX 导出、TensorRT 引擎构建和 C++ 推理验证整合为一套工业 AI 缺陷检测闭环平台。

日常操作请用浏览器版。给人看的步骤、截图和流程图在 [Web 操作手册](docs/TRTForge_Web操作手册.md)。整套网页怎么拼、用了哪些技术，看 [Web Studio 架构与功能介绍](docs/WebStudio架构与功能介绍.md)。现有缺陷、安全漏洞和商业化路线图看 [商业化差距分析与优化建议](docs/商业化差距分析与优化建议.md)。桌面版步骤仍看 [Qt 操作手册](docs/TRTForge_操作手册.md)。

要发给别人使用、又不附带源码时，在开发机执行 `packaging/package_web_user.ps1`。对方电脑需为 CUDA 11.8，先运行安装包里的 `install_train_env.bat` 配置训练环境（不必预先安装 Anaconda；无 AVX2 的机器会自动改用 Polars 兼容包），再用 `config.json` 填写数据目录。细节见 [WORKFLOW_GUIDE 7.1.3](docs/WORKFLOW_GUIDE.md)。

## 主要模块

| 路径 | 说明 |
| --- | --- |
| `6-web` | **浏览器版工作站（推荐）**：FastAPI 后端 + 零构建前端，功能覆盖训练导出、引擎构建、数据集划分导出、标注工作台与推理演示，不需要 Qt。操作步骤见 [Web 操作手册](docs/TRTForge_Web操作手册.md)，开发和版本记录见 `6-web/README.md` |
| `4-qt-studio` | DefectVision AI Studio Qt 桌面程序；当前编译产物仍为 `TRTForge.exe`，包含 ONNX 转 TensorRT、标注数据集、YOLO/Anomalib 训练页和推理演示页。**已被 `6-web` 取代，保留作迁移参考** |
| `4-qt-studio/installer` | Windows 安装包脚本（Inno Setup）：递归打包 `4-qt-studio/bin` 内已编译的 exe 与全部运行库，详见 `4-qt-studio/README.md` 中「制作 Windows 安装包」 |
| `3-cpp-tensorrt/trtexec` | 命令行版 `trtexec` 封装（TensorRT C++ 采样工具、profiler/tracer 脚本） |
| `3-cpp-tensorrt/trtexec/package_devkit.ps1` | 打给二次开发用的开发包：含 GUI 源码、YOLO/Anomalib 训练工程和预编译推理 DLL，**不含**封装后的推理实现源码 |
| `3-cpp-tensorrt/yolo_infer` | YOLO TensorRT 推理 DLL，提供 `yi_*` C API（检测/分割/姿态/OBB），参数与并发契约见 `3-cpp-tensorrt/yolo_infer/README.md` |
| `3-cpp-tensorrt/anomalib_infer` | Anomalib TensorRT 推理 DLL，提供 `ai_*` C API，可供 TRTForge、外部 C++ 和 Python ctypes 调用，见该目录 `README.md` |
| `ocr/ocr_infer` | PaddleOCR C++ 推理 DLL，提供 `oi_*` C API，返回文本行多边形、文本、分数和原始 JSON |
| `1-python-yolo/launcher` | 给 Qt/C++ 调用的 Python 训练脚本入口 |
| `6-web/backend` | Web 版后端：启动 launcher 子进程、转发训练事件、ctypes 调推理 DLL |
| `6-web/frontend` | Web 版前端：原生 HTML/JS，无 npm 构建步骤 |
| `1-python-yolo/ultralytics-main` | YOLO 训练、验证、导出的 Python 开源工程来源 |
| `7-autoresearch-yolo` | **YOLO 自主实验循环**：固定时间预算 + 单一适应度 + 台账留痕，在 PCB 数据集上做超参与增强的自动化实验。v1.33.0 起含 **LLM 决策层**（`agent` 子命令）：大模型提议下一组参数，`keep / discard` 由代码按 2σ 噪声阈值判定。**对生产 launcher 零改动**，只 import 其纯函数；产物严格收敛在该目录内。用法与参数见 `7-autoresearch-yolo/README.md` |
| `7-autoresearch-master` | Karpathy autoresearch 原始副本（LLM 预训练方法论来源），**仅供参考，不参与本工程构建** |
| `docs` | 架构设计、推理流程、部署总结等中文文档。操作入口：[Web 操作手册](docs/TRTForge_Web操作手册.md)。发给使用人员、不附源码的安装包见 [packaging/package_web_user.ps1](packaging/package_web_user.ps1) |
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
| `OPENCV_DIR` | OpenCV 根目录，CMake 构建（`4-qt-studio`、`ocr_infer`、两个推理子工程）与推理 DLL 运行时共用；`-DOPENCV_DIR` 可临时覆盖 |

`third_party/` 已加入 `.gitignore`，Paddle 源码与构建产物不会进入版本库。

### 不入库、换机器要自己准备的文件

下面几项在 `.gitignore` 里，克隆仓库后不会出现。软件日常推理不依赖前三项；缺了只影响「重新编译 ONNX 推理库」或「OCR 文字识别」。

| 路径 | 是什么 | 怎么得到 |
| --- | --- | --- |
| `3-cpp-tensorrt/onnx_infer/third_party/onnxruntime/` | ONNX Runtime 1.16.3 头文件和 DLL（约 10 MB 起）。编译 `onnx_infer.dll` 时用。安装包从 `4-qt-studio/bin` 带运行库，使用人员不用这一步 | 在仓库根目录执行 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_onnxruntime.ps1`。只要 CPU 版加 `-Variant cpu`。脚本从 NuGet 下载 `Microsoft.ML.OnnxRuntime.Gpu` 1.16.3（与 CUDA 11.8 对齐，不要换到 1.17+） |
| `9-python-paddleOCR/model/legacy_v4/` | PP-OCRv4 检测、识别、方向模型（`.pdmodel` / `.pdiparams`，合计约 500 MB）。Web 默认到这里找 OCR 模型，也可用环境变量 `TRTFORGE_OCR_MODEL_DIR` 改位置 | 从 `https://paddleocr.bj.bcebos.com/` 下载并解压到 `model/legacy_v4/`：`PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar`、`PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar`、`dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar`。官方包没有 `inference.yml`，在 `6-web/backend` 下执行 `python tools/gen_legacy_ocr_yml.py` 补上。字典用 PaddleOCR 2.x 的 `ppocr/utils/ppocr_keys_v1.txt` |
| `9-python-paddleOCR/PaddleOCR-main/` | PaddleOCR 官方源码（约 129 MB），只作对照和取字典，不参与编译、不打进安装包 | `git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git 9-python-paddleOCR/PaddleOCR-main` |
| `2-python-anomaly/launcher/resultsbottle/` | 本机一次 PatchCore（MVTec bottle）训练产物（onnx / engine / ckpt，约 780 MB）。当前测试和安装包都不读这个目录 | 不需要下载。要复现时在异常检测训练里把输出目录指到这里 |
| `runtime/`、`docs/*.rar` | 调试截图，以及和 `docs/深度学习介绍/` 重复的压缩包 | 不用准备 |

用户安装包不包含上表这些目录。文字识别要在打包时加 `-WithOcr -OcrModelDir <legacy_v4 目录>`，模型会进安装目录的 `models\ocr`。

### 训练环境准备（Python 解释器）

程序不内嵌 Python，训练时的解释器由训练页 `Python 环境` 指定，再对每个任务用 `QProcess` 拉起。已验证可用的环境要求：

| 链路 | 必需依赖 |
| --- | --- |
| YOLO 训练 / 导出 | `torch`、`torchvision`、`ultralytics`、`polars`、`onnx`、`onnxruntime` |
| Anomalib 训练 / 导出 | `torch`、`anomalib`、`lightning`、`onnx`、`onnxruntime`、`Pillow`、`numpy<3` |

一次性装齐：

```powershell
python -m pip install ultralytics polars "numpy<3"
```

关于 `numpy` 上限（2026-10-30 修订）：早期这里写的是 `numpy<2`，理由是 torch 2.3 按 numpy 1.x 编译。
实测 `torch 2.3.0 + numpy 2.5.4` 的 CUDA 前向与 numpy 桥接均正常，`onnxruntime 1.20.1` 也已支持 numpy 2，
故放宽到 `<3`。锁回 `<2` 反而会让 Ultralytics AutoUpdate 每次 `export` 都把它升上去，造成环境反复漂移。
残留告警仅 `scipy 1.14.0` 的 `UserWarning`（声明要求 numpy<2.3），不影响 anomalib 链路。

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

### v1.34.4 — 2026-11-11（良品图 + 操作点：漏检-过杀曲线、引擎工作点、自动实验过杀约束）

**一句话**：把「漏一个」和「多报一个」从只能事后感叹变成可以调的旋钮——良品图进验收集，
验证报告画出整条**漏检-过杀曲线**，选定的点存进引擎旁挂文件并在推理页自动回填；
自动实验不再允许「拿过杀换 mAP」。

**解决什么**：v1.34.3 之前的验证只回答「mAP 是多少」，而产线真正问的是两件事：
**「这个阈值下会漏几个？」** 和 **「为了不漏，要误报多少个？」**。这两件事需要两样以前没有的东西：
① 良品图（没有缺陷的 OK 图）参与评估，否则「误报」无从统计；
② 沿 conf 扫一遍的曲线，否则用户只能凭感觉填一个 0.25。

| 事项 | 变化 |
| --- | --- |
| 良品图声明 | 数据集扫描/预览/导出/体检新增 `ok_dirs`：相对数据源根的目录，多个用分号分隔（如 `ok;test/good`） |
| 良品图去向 | `ok_split` = `val`（默认）/ `test`：**只进验收划分，绝不参与训练**；导出时写**空标签文件**（不是缺文件） |
| 为什么必须显式声明 | 「没有缺陷」和「忘了标」在磁盘上都表现为没有框。只有用户声明过的目录才算良品，其余空标注/缺标注仍按漏标嫌疑处理 |
| GT 四态 | `labeled`（有标注）/ `ok`（标签文件存在且为空）/ `missing`（标签文件不存在）/ `unparsed`（有内容但解析失败）；**只有 `ok` 进过杀统计** |
| 过杀率定义 | `ok_fp_rate = 至少有一个误报框的良品图数 / 良品图总数`；**没有良品图时返回 `null`**，不是 0 |
| 曲线 | `valmetrics.operating_points()`：按 conf 从 0.05 到 0.95 扫 19 个点（可用 `--operating-confs` 指定，最多 50 个） |
| 单调性 | 每类只做一次贪心匹配，再按 conf 取前缀累积——保证 conf 升高时 TP/FP 单调不增，且与 `evaluate()` 的数字逐点对得上 |
| 推荐点 | `recommend_operating_point()`：在**满足目标漏检率**的点里取 **conf 最高**的那个（过杀最低）。取最低 conf 是反方向 |
| 无解不撒谎 | 所有点都不满足目标时返回 `feasible: false` + 最接近的点 + 原因，绝不假装达标 |
| 曲线入口 | 验证结果卡片下方：内联 SVG 折线（零依赖）+「目标漏检 → 推荐 conf / 实际漏检 / 过杀 / 图 / 良品误报」四行对照表 |
| 引擎工作点 | 新增 `POST/GET /api/engine/operating-point`，写进 `<engine>.meta.json` 的 `operating_point`（**不碰引擎文件、不改 `metadata.json`**） |
| 推理页联动 | 选中引擎时自动读回工作点并预填 `conf` 输入框，提示行显示「已回填验证操作点 conf=…，漏检 …，过杀 …」 |
| 自动实验过杀列 | `results.tsv` / `holdout.tsv` 新增 `fp_per_image`；由 Ultralytics 的 P/R 反推（`TP = R×GT`、`FP = TP×(1-P)/P`），拿不到就留空 |
| KEEP 约束 | `judge_round()`：mAP 提升超过噪声阈值，但过杀恶化超过 `max(0.01, 基线×5%)` 时判 **DISCARD**，理由写明「不拿过杀换 mAP」 |

**曲线点字段（读取方按这个格式解析）**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `conf` | number | 该点的置信度阈值 |
| `tp` / `fp` / `fn` | int | 该阈值下的命中 / 误报 / 漏检框数 |
| `recall` | number\|null | `tp / GT 框数`；没有 GT 时为 `null` |
| `miss_rate` | number\|null | `1 - recall`，**产线关心的是这个** |
| `precision` | number\|null | `tp / (tp + fp)` |
| `f1` | number\|null | P/R 调和平均 |
| `fp_per_image` | number\|null | `fp / 评估图片数` |
| `ok_images` | int | 本次评估里的良品图数 |
| `ok_fp_images` | int | 至少有一个误报的良品图数 |
| `ok_fp_rate` | number\|null | `ok_fp_images / ok_images`；**无良品图时为 `null`** |
| `per_class` | array | 逐类明细：`id, instances, tp, fp, fn, recall, miss_rate, fp_per_image` |

报告级字段：`iou`、`confs`、`points`、`num_images`、`num_gts`、`ok_images`、
`missing_label_images`、`unparsed_label_images`、`note`。

**使用方式**

```powershell
# 1) Web 数据集页：填「良品目录」（多个用分号）→ 选「良品图进哪个划分」→ 扫描/预览/导出
#    导出的良品图只出现在 val（或 test），标签是空 .txt
#    —— 验证时用的就是这份导出目录，所以良品图会自动被算进过杀率，命令行不需要额外参数

# 2) 命令行验证：对导出的 YOLO 目录跑曲线与推荐点
python 6-web/backend/tools/val_engine.py --engine runs/exp/best_fp16.engine --data out/dataset.yaml `
  --target-miss-rate 0.01 --operating-confs "0.05,0.1,0.2,0.3,0.5"

# 3) Web 验证结果页：看曲线 → 选目标漏检 → 点「保存为引擎工作点」
#    推理页再选这个引擎时 conf 会自动变成保存的值

# 4) 自动实验：过杀恶化会被拦下
python 7-autoresearch-yolo/autoexp.py run --config experiments/cur.json
```

**测试**：`test_operating_points.py` 48 项断言（曲线单调性、`ok_fp_rate` 的 null 语义、
与 `evaluate()` 逐点一致、推荐取最高 conf、无解不撒谎、过杀目标约束）；
`test_dataset_workbench.py` 新增 23 项良品图断言（导出空标签、不重复计数、良品目录有标注时报错）；
后端 **53/53** 套件全绿；`7-autoresearch-yolo/run_all_tests.py` **71/71**；前端 `node --check` 通过。

**已知边界（刻意不做）**：良品图不参与训练（只做验收，做训练要另设计半监督方案）；
`fp_per_image` 在自动实验里是**近似值**（Ultralytics 的 P/R 取自 best-F1 conf，也不是固定 conf），
只用于拦「明显恶化」，不能当产线验收数字——验收数字用验证页那条曲线；
曲线只在检测/分割/姿态/OBB 上有意义，异常检测是单阈值判定，不提供该曲线。

### v1.34.3 — 2026-11-11（数据集体检：同底板分组、跨划分泄漏、标注问题；按组划分）

**一句话**：给数据做一次「体检」——找出完全重复的图、找出**同一块底板拍出来的不同缺陷图**
（这类图会让验证指标虚高）、逐行检查标注字段；并在导出/切分时可选「同底板进同一划分」。

**解决什么**：PCB 这类工业数据集里，692 张图其实只有 10 块底板，每块底板拍了几十张
「同位置不同缺陷」的图。按图片随机划分时，同一块底板必然同时出现在 train 和 val 里，
模型记住的是底板而不是缺陷 → val 指标好看、上线就掉。以前的工具完全看不见这件事。

| 事项 | 变化 |
| --- | --- |
| 体检实现 | 新增 `6-web/backend/datasetqc.py`（算法标识 `dsqc-v1`），Web 后端与 `autoexp.py`、`labelqc.py` **共用同一份**实现 |
| 签名 | 每张图算 `sha256` + **256 位 dHash**（17×16 灰度图比较相邻像素）+ 32×32 灰度缩略图 |
| 两级判定 | **A 级 完全重复**：文件 sha256 相同（改名/复制得到的那张）；**B 级 同底板**：dHash 汉明距离 ≤ 8 **且** 32×32 灰度平均绝对差 ≤ 2.0 |
| 为什么要 MAE 复核 | dHash 只看梯度符号，大面积纯色（PCB 的空白板面）在纯哈希下**可能**让不同底板的图撞位，漏判就会给出错误的泄漏结论；MAE 用真实像素再确认一次 |
| 绝不自动删 | A 级也只是「报告 + 可选标记」，B 级只用于告警与分组。**任何情况下都不自动删除用户图片** |
| 分组算法 | 候选对按 banding（把 256 位切成 8 段做桶）生成，避免 692×692 全比；超 3000 张退回全量两两比较；分组用并查集 |
| 跨划分泄漏 | 每组的 `cross_split` 标记是否横跨 train/val/test；报告给出跨划分组数与涉及图片数 |
| 同组同标签提示 | 同底板且类别数量相同、框 IoU > 0.95 的两张图，单独列进 `same_label_pairs` 供人工确认（工业缺陷天然可能在同一位置重复出现，所以只提示不判定为错） |
| 标注质检 | 逐行检查，**永不抛异常**：字段数、数值解析、类别越界、坐标越界、零面积、极小框(<0.01%)、超大框(>60%)、贴边、同图重复框(IoU>0.9)、某类缺失、空标注、类别占比 <2%、同底板异类标注 |
| 只读 | 体检不改任何图片或标注文件；唯一写入是 `runtime/dsqc_cache.json` 的签名缓存（键 `路径｜大小｜纳秒mtime`，算法版本变化整体作废，上限 2 万条） |
| 按组划分 | 扫描/导出新增 `group_by`：`none`（默认，逐图哈希）/ `prefix`（文件名前缀，如 `01_`）/ `dir`（父目录名）。同组图片**必须进同一划分**，用「补最大缺口」的贪心近似目标比例 |
| 不拆组的代价 | 组数少于 2 或组太大时比例会明显偏离设定值，报告里会说明；这是刻意取舍——宁可比例不准，也不要泄漏 |
| 清单记录 | `manifest.json` 新增 `split_mode: {mode, groups, note}` 与 `group_signature`，事后能证明「这次是按组分的」 |
| 导出时附体检 | 导出可选 `with_qc`，把体检摘要写进 `manifest.json` 的 `qc` 字段（不额外发请求） |
| 新接口 | `POST /api/dataset/qc`（只读）：`{source, task, threshold, limit}` → 分组/泄漏/问题明细/耗时 |
| CLI | 新增 `7-autoresearch-yolo/labelqc.py <source> [--task] [--threshold] [--limit] [--json] [--fail-on none\|error\|warn]` |
| 实验门禁 | `autoexp.py run --qc-gate`：存在 **error** 级问题直接拒绝启动（退出码 6）；warn/info 只打印不拦；体检本身失败则放行（不让检查变成阻塞） |
| 按组切分 | `autoexp.py split --group-by {none,prefix,dir}` |

**体检报告字段（读取方按这个格式解析）**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `algo` | string | 当前 `dsqc-v1`；算法变了旧缓存与旧报告不可比 |
| `threshold` / `mae_limit` | number | 本次使用的汉明距离上限（默认 8）与灰度差上限（2.0） |
| `counts` | object | 图片总数、各 split 张数、空标注数 |
| `summary` | object | `{error, warn, info}` 三级问题计数 |
| `by_rule` | object | 按规则码聚合：`{code: {count, title, severity}}` |
| `groups` | object | `{total, cross_split, leak_images, items[≤50], same_label_pairs[≤50], truncated}` |
| `exact_groups` | array | A 级完全重复组（相对路径列表，≤50 组） |
| `issues` | array | `{code, severity, rel, split, line, detail}`，最多 `limit` 条 |
| `cache` | object | 命中/新增条数，回答「为什么这次快/慢」 |
| `elapsed_ms` | number | 实际耗时 |
| `version` | string | 产生报告的后端版本 |

**使用方式**

```powershell
# 1) 命令行体检（只读，人类可读格式）
D:\anaconda3\envs\dpenv\python.exe 7-autoresearch-yolo/labelqc.py "PCB data/PCB缺陷检测" --task auto

# 2) 机器可读 + 有错误时返回非 0（可接进 CI / 上线前检查）
python 7-autoresearch-yolo/labelqc.py "splits/pcb.yaml" --json --fail-on error

# 3) Web 数据集页：点「体检」，看底板组 / 跨划分标记 / 标注问题
#    导出时勾选「按组划分（prefix）」即可让同一底板只进一个划分

# 4) 自动实验：开跑前门禁 + 切分时按组
python 7-autoresearch-yolo/autoexp.py split --group-by prefix
python 7-autoresearch-yolo/autoexp.py run --config experiments/cur.json --qc-gate
```

**真机验证（T1000 8GB）**：对 `PCB data/PCB缺陷检测`（692 张）体检得到
**10 组、10 组全部跨划分、涉及 689 张**（另有 3 张自成一类未参与泄漏）——即「按图片随机划分」
在这份数据上必然泄漏；
`labelqc.py` 逐图检查未发现 error 级标注问题，耗时约 1 秒（第二次走缓存）。
接口 `POST /api/dataset/qc` 实测返回 `version=1.34.3`、`algo=dsqc-v1`、`counts.train=553`、
`summary {error:0, warn:0, info:10}`、`cache {hits:692}`、`elapsed_ms=1274`。
对照实验：同一批数据用 64 位 dHash（阈值 ≤6）也能分出这 10 组——但常规「重复图去重」会
把组内 691 张判成重复直接删掉，等于毁掉全部缺陷样本；这就是本版分两级、B 级永不删除的原因。
对 `splits/pcb`（553 train / 97 tune / 42 holdout）按 `prefix` 重新切分后，
10 组各自只落在一个划分里，比例因不拆组略有偏离（`manifest.json` 的 `split_mode` 记录了实际组数）。
泄漏对指标的影响本轮**没有做训练对比**，属于后续实验项，不在本版结论内。

**新增文件**：`6-web/backend/datasetqc.py`、`6-web/backend/tools/test_dataset_hygiene.py`、
`7-autoresearch-yolo/labelqc.py`。
**改动文件**：`6-web/backend/{dataset,app,config}.py`、
`6-web/frontend/{index.html,js/dataset.js,css/style.css}`、
`7-autoresearch-yolo/{autoexp.py,run_all_tests.py}`。

**测试**：`test_dataset_hygiene.py` 43 项断言；后端 **52/52** 套件全绿；
`7-autoresearch-yolo/run_all_tests.py` **68/68**；前端 `node --check` 通过。

**边界（刻意不做）**：不做自动删除/自动改标注（不可逆操作留给人和版本库）；
不哈希图片内容来决定去重（内容哈希只能发现字节相同的复制品，对「同一底板不同照片」无用）；
不做感知哈希的定量「相似度排名」（距离 ≤8 是工程阈值，不是物理量）。

### v1.34.2 — 2026-11-11（数据集指纹：模型/实验/引擎都能追回「用的哪批图」）

**一句话**：给数据算一个内容指纹，写进导出清单、训练目录、实验台账和引擎元数据——
以后「这个模型用的到底是哪批图」不再靠人回忆。

**解决什么**：训练目录里只有一个 `data:` 路径。图片被重标注、增删、重划分后路径不变、
内容已变，于是模型无法溯源、两次实验无法判断是否可比、线上引擎精度掉了查不到来源。
附带修掉一个隐患：`splits/_manifest.json` 以前只记文件名与 seed，holdout 的图被改过标注
后「验收通过」的结论会悄悄失效。

| 事项 | 变化 |
| --- | --- |
| 指纹实现 | 新增 `6-web/backend/fingerprint.py`，Web 后端与 `autoexp.py` **共用同一份**实现（与 `vram.py` 同样的理由：两边各写一份必然打架） |
| 指纹粒度 | `dataset_fingerprint`（整批）、`split_fingerprints`（train/val/test 各自）、`label_fingerprint`（只算标注文本） |
| 指纹输入 | 每个 split 内「相对路径 + 文件字节数 + 标注文件原始字节哈希」，排序后 sha256，取前 16 位；带 `algo: "v1"` 版本前缀 |
| 路径语义 | 相对 split 的 images 目录，所以导出到不同目录/镜像到 `splits/` 指纹相同 |
| 重命名 | **算数据变化**（路径进哈希）；重命名图片时必须同时改标签文件名，否则那条标注会丢 |
| 不做什么 | 不哈希图片内容（读全量图要几十秒，绝大多数变更靠路径+大小+标注就能发现）、不用 mtime（复制/解压会变） |
| 导出清单 | 每个导出目录新增 `manifest.json`（与 `dataset.yaml` 同级），含指纹、数量、类别、来源、划分方式、程序版本 |
| 重复导出 | 同目录内容未变时返回 `unchanged: true`；**覆盖导出前清理上次产物**，不再让同名图累积成 `xxx_2.jpg`（那会让「同一批数据再导一次」得到不同指纹） |
| 训练记录 | 开训前算一次放进 `run.meta`；launcher 在 `on_train_end` 写 `train_meta.json`（位置由 trainer 的最终 `save_dir` 决定，避免 name 被改成 `name2` 时写错目录） |
| 任务台账 | `runtime/runs.jsonl` 的 `meta` 带 `data_fp` 短码；子进程环境变量新增 `TRTFORGE_RUN_ID` |
| 自动实验 | `exp_NNNN.json` 新增 `dataset` 块；`results.tsv` 新增 `data_fp` 列（按列名迁移，旧文件备份 `.bak`）；`best.json` 的 `evidence` 记录指纹 |
| holdout 纪律 | `splits/_manifest.json` 记录 `data_fingerprint` 与 `holdout_fingerprint`；`autoexp.py holdout` 开跑前对比，数据变过会明确警告「之前的验收结论不可沿用」 |
| 引擎溯源 | 构建成功后写 `<engine>.meta.json`：ONNX 与引擎的 sha256、精度、trtexec 参数、来源训练 run、数据集指纹。**不改 DLL**，纯旁挂文件 |
| 界面 | 训练页「算力体检」多一行「数据集：`5a9749c5`（553 train / 97 val）」；数据集页导出结果显示指纹与 `unchanged`；自动实验列表按指纹着色并提示跨组不可横比 |
| 新接口 | `GET /api/dataset/fingerprint?source=<dataset.yaml>`（只读，不落盘、不缓存） |
| 兼容 | 旧模型/旧实验没有指纹时显示「未记录」，不报错、不回填（当时的原始数据已不可得，补出来的是假的） |

**指纹记录字段（读取方按这个格式解析）**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `algo` | string | 算法版本，当前 `v1`。不同版本的指纹永远不相等 |
| `dataset_fingerprint` | string(16) | 整批指纹，用于判断「能不能横比」 |
| `short` | string(8) | 前 8 位，界面与台账展示用 |
| `split_fingerprints` | object | `{train, val, test}` 各自的指纹，回答「只改了 val 吗」 |
| `label_fingerprint` | string(16) | 只算标注文本、**不含路径**，回答「标注被改过吗」 |
| `counts` | object | `{train, val, test}` 图片数 |
| `error` | string | 计算失败的原因；为空表示正常 |

**使用方式**

```powershell
# 1) 命令行直接查某个 dataset.yaml 的指纹
curl "http://127.0.0.1:8060/api/dataset/fingerprint?source=D:/data/pcb/dataset.yaml"

# 2) 导出时自动写 <输出目录>/manifest.json，响应里带 data_fp
#    （Web 数据集页点「导出」即可，无需额外操作）

# 3) 自动实验：每轮自动写进 exp_NNNN.json 的 dataset 块与 results.tsv 的 data_fp 列
python 7-autoresearch-yolo/autoexp.py run --config 7-autoresearch-yolo/experiments/cur.json
```

**真机验证（T1000 8GB）**：对 `splits/pcb_tune.yaml`（553 train / 97 val / 6 类），
Web 接口、`autoexp.py`、训练目录 `train_meta.json` 三处都得到 **`5a9749c58c8a0ea9`**；
1 epoch 最小训练（320/ batch 2）后 `train_meta.json` 落在 Ultralytics 的真实 `save_dir` 下，
`run_id` 与 `data_arg` 正确。指纹耗时 650 张 **0.14 秒**。

**新增文件**：`6-web/backend/fingerprint.py`、`6-web/backend/enginemeta.py`、
`6-web/backend/tools/test_dataset_fingerprint.py`。
**改动文件**：`6-web/backend/{dataset,app,runners,config,autoexp_api}.py`、
`1-python-yolo/launcher/tf_yolo_launcher.py`、`7-autoresearch-yolo/{autoexp.py,run_all_tests.py}`、
`6-web/frontend/{index.html,js/dataset.js,js/train.js,js/autoexp.js,css/style.css}`。

**测试**：`test_dataset_fingerprint.py` 45 项；后端 **51/51** 套件全绿；
`7-autoresearch-yolo/run_all_tests.py` **65/65**；前端 `node --check` 通过。

**边界（刻意不做）**：不做数据快照/版本库（那是 DVC 的职责，指纹只证明「是不是同一份」，
不负责把旧的那份留住）；不哈希图片内容，所以「同名同大小但内容被换」发现不了；
不回填历史产物。

### v1.34.1 — 2026-11-11（显存预检改「实测峰值表 + 安全余量 + 自校准」；AMP 状态可见可覆盖）

**一句话**：显存预检不再靠一条手拟合的公式猜——跑过一次就把真实峰值记下来，下次按实测值
+15% 余量 +400MB 保留判定；同时把「本机到底有没有用上混合精度」摆到界面上，
并允许用户在明知风险时手动放行。

| 事项 | 变化 |
| --- | --- |
| 预检判定 | 新增 `6-web/backend/vram.py`，`6-web` 训练接口与 `7-autoresearch-yolo/autoexp.py` **共用同一套**判定（此前两处各写一份，已在 960/batch=4 上给出互斥结论） |
| 实测峰值表 | `7-autoresearch-yolo/experiments/vram_samples.json`，键 `imgsz x batch x 精度`（如 `640x8xfp32`），值取历次训练的最大 `max_memory_reserved` |
| 自校准 | 训练结束（launcher 的 `on_train_end`）自动写回，越用越准，不需要人工维护 |
| 估算优先级 | 精确命中实测 → 更小 batch 线性外推 → 更大 batch 作上界 → 无样本才用公式 `2600 + 300×batch×(imgsz/640)²` |
| 余量 | 实测值 × 1.15 + 400MB（覆盖批次间波动、桌面合成与其它进程的瞬时申请） |
| AMP 体检 | 新增 `6-web/backend/ampcheck.py`，复刻 Ultralytics 8.4.192 的 AmpBlacklist 正则；T1000/T1200/GTX16xx 等会明确告警「训练实为 fp32，显存更高、速度约慢一倍」 |
| AMP 覆盖 | `POST /api/train/yolo` 新增三态 `amp`：不传＝按本机自动、`false`＝`--no-amp`、`true`＝`--amp` |
| 实际状态回显 | launcher 在训练开始时发 `amp_state` 事件，训练结束发 `vram_peak` 事件；页面据此显示「请求 / 实际」两种精度，不再靠猜 |
| 拦不住时放行 | 预检不过返回 HTTP 409，页面出现「仍然开始（显存可能不够）」，确认后带 `allow_low_vram=true` 重发 |
| 体检接口 | `GET /api/train/preflight` 实时返回 GPU 名称/总量/可用量与当前 imgsz、batch 的估算依据 |

**本机实测（T1000 8GB，fp32）**：`640/batch=8` 峰值 **2132 MB**（`allocated` 1769 MB，取
`reserved` 是因为 OOM 发生在向驱动申请那一刻），写入 `640x8xfp32`；据此
`batch=16` 外推需要 **5304 MB**、`batch=4` 取上界只需 2852 MB —— 而旧公式对
batch=4/8/16 分别给出 3800/4520/7400 MB，前两者都是错误的保守估计。

**新增文件**：`6-web/backend/vram.py`、`6-web/backend/ampcheck.py`、`6-web/backend/tools/test_vram_amp.py`。
**改动文件**：`6-web/backend/{app,config,runners}.py`、`1-python-yolo/launcher/tf_yolo_launcher.py`、
`7-autoresearch-yolo/autoexp.py`、`6-web/frontend/{index.html,js/train.js}`。

**测试**：`test_vram_amp.py` 118 项；`7-autoresearch-yolo/run_all_tests.py` 62/62；后端 5 个测试套件全绿。
真机：跑 1 epoch 得到 `amp_requested=true / amp_effective=false / precision=fp32` 与
`peak 2132 MB`，样本自动落盘，页面「算力体检」显示 5304 MB 需求并通过；
把 batch 调到 64 触发 409，页面出现覆盖按钮，服务端未创建任务。

**边界（刻意不做）**：不做 batch 插值（插值等于回到「猜系数」）；不做显存表的跨机器合并
（不同卡/驱动不可比，样本只服务本机）；AMP 无法真正强制打开（黑名单在框架内部生效，`--amp` 也压不住）。

### v1.34.0 — 2026-11-11（Web 端任务台账落盘、日志轮转、训练崩溃自动续训）

**一句话**：服务重启不再「一夜之间什么都查不到」——任务记录落到 `runtime/runs.jsonl`，
日志按体积滚动并按天清理，YOLO 训练崩了会自己续一次。

| 事项 | 变化 |
| --- | --- |
| 任务台账 | `6-web/runtime/runs.jsonl`，一个任务两行（开始 / 结束），含 `run_id / kind / status / exit_code / started_at / finished_at / command / log_file / resumed_from` |
| 重启恢复 | 启动时把仍为 `running` 的记录改判 `interrupted`（`orphan=true`），日志照样能打开 |
| 历史接口 | 新增 `GET /api/runs/history?limit=50`（上限 500）；`GET /api/runs` 语义不变，仍只返回本进程任务 |
| 顶栏「历史」 | 点开列出最近 50 条，状态区分「完成 / 失败 / 已停止 / 已中断」，可打开该次完整日志 |
| 日志轮转 | 单任务 20MB 滚动、留 3 份（`TRTFORGE_LOG_MAX_MB` / `TRTFORGE_LOG_BACKUPS`），读取接口跨分片拼接 |
| 日志清理 | 启动时整族删除 30 天未更新的任务日志（`TRTFORGE_LOG_RETENTION_DAYS`，`0` 关闭） |
| 自动续训 | `yolo-train` 失败且存在 `weights/last.pt` 时用 `--resume-from` 续跑一次；`TRTFORGE_AUTO_RESUME=0` 关闭 |

**新增文件**：`6-web/backend/runledger.py`、`6-web/backend/logstore.py`、`6-web/backend/tools/test_run_history.py`。
**改动文件**：`6-web/backend/{config,runners,app}.py`、`6-web/frontend/{index.html,css/style.css,js/api.js,js/runsbar.js}`、`6-web/frontend/tests/test_runsbar.mjs`、`6-web/backend/tools/run_all_tests.py`。

**边界（刻意不做）**：Anomalib 没有等价的命令行续训入口，自动续训只覆盖 YOLO；历史任务只读，停止接口返回 404；台账写失败不影响任务本身。

**测试**：`test_run_history.py` 48 项；回归 `test_run_log.py` 22 项、`test_exit_codes.py` 6 项、
`test_pipeline_log.py` 9 项、`test_pipeline_stages.py` 50 项、`test_concurrency.py` 全绿；
前端 `test_runsbar.mjs` 41 项。真机：运行中强杀服务后重启，历史列表出现
`anomalib-check-env / interrupted / orphan=true / 8.0s`，该任务日志仍可打开，停止接口 404。
详细说明见 [6-web/README.md](6-web/README.md) 的 v1.34.0 节。

### v1.33.3 — 2026-11-10（自动实验开始时退出推理进程，结束后按需重载）

**一句话**：自动实验一开始就让 `infer_worker` 退出并腾出显存；实验没结束前它不会自己再起来；实验结束后不马上加载，下次真正推理时才加载。

960/batch4 训练大约要 5.5GB。网页上的推理进程是懒加载的，点一次推理就占住约 1.7GB 不放，两者叠在 8GB 卡上会在第一个 batch OOM，然后 Ultralytics 把 batch 偷偷减半。现在：

| 时机 | 行为 |
| --- | --- |
| 网页点「开始」或命令行 `agent` / `run` / `noise` | 通知正在运行的 6-web：`POST /api/infer/gpu-yield`，推理进程退出 |
| 实验进行中（含每一轮之间） | 推理请求返回 503，手动「重启推理进程」也会被拒绝 |
| 整场实验结束 | 只解除限制，**不**拉起进程 |
| 下一次检测 / 分类 / OCR | 按需重新加载 |

网页启动的实验由后端直接停进程，并给子进程带上 `AUTOEXP_INFER_YIELDED=1`，避免每一轮训练结束时把显存提前还回去。命令行实验打给 `http://127.0.0.1:8060`（可用 `AUTOEXP_WEB_URL` 改地址）。网页没开时训练照常进行，日志会说明推理进程可能还占着显存。

**测试**：`test_infer_yield.py` 6 项；autoexp `run_all_tests.py` 56 项；`test_autoexp_api.py` 16 项。真机短测记录在 `docs/TEST_PLAN_AUTOEXP_ROBUSTNESS.md`。

### v1.33.2 — 2026-11-10（修「5 轮全 DISCARD」暴露的三个问题）

**一句话**：模型侧 0 失败说明 v1.33.1 的截断修复生效了，但那一场 5 轮里混着三个**与模型无关**的问题：模型以为在 960 上做实验、程序实际跑在 640；用 640 的噪声底去判 960；显存预检放行了必然 OOM 的 1152，还让训练框架把 batch 偷偷减半。

**现场**（`experiments/agent/round_0001..0005.json` + `results.tsv`）

| 轮 | 实验 | 实际跑的配置 | mAP50-95 | 与最优差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | exp_0015 | 1152 / b4 | — | — | crash（30 秒 OOM） |
| 2 | exp_0016 | 960 / b4 + degrees10 | 0.42967 | −0.029 | DISCARD |
| 3 | exp_0017 | **640 / b8** + warmup1 | 0.39062 | −0.068 | DISCARD（**整轮无效**） |
| 4 | exp_0018 | 960 / b4 + cos_lr | — | — | crash（253 秒静默死） |
| 5 | exp_0019 | 960 / b4 + freeze10 | 0.39189 | −0.067 | DISCARD |

**三个改动**

| 改动 | 老版本 | 新版本 |
| --- | --- | --- |
| 继承基准 | `cur.json` 只写本轮改动，`run` 用 `base.json` 合并 → **回落到 640** | `cur.json` 写「`best.json` 的 train + 本轮提议」；`_ask_proposal` 也按最优校验；审计记 `inherited_from` |
| KEEP 阈值 | 全库一个 σ（640 实测 0.0032） | 按 `imgsz × batch` 分组：先查该组实测，再查台账里配置相同的重复实验，都没有才借全局并标 `calibrated=false` |
| batch 降级 | 事后比较 `batch_req` / `batch_eff`，只打一行警告 | `_exec_tee` 匹配 `Reducing to batch=N` 即**终止子进程**，状态记 `batch_downgraded`，`fitness` 留空，不计 crash、不污染 `current_best` |

**顺带**：`space.json` 的 `imgsz` 枚举去掉 1152 / 1280（本机 T1000 8GB 实测必 OOM），提示词同步说明「换大卡后可加回」。

**这不是自动修复的部分**：960 的真实噪声底已补测完成：`sigma = 0.008070`、
`2σ = 0.016139`（640 是 0.003184 / 0.006368）。用新阈值重判那 5 轮，
两个 DISCARD 结论**依旧成立**——KEEP 0 次是真实的科研结果，
不是阈值冤枉了模型（`exp_0006` 的 0.45862 落在三次重复的分布内）。

**另一条环境教训**：`6-web` 的 `infer_worker` 懒加载会占约 1.7 GB 显存，
960/batch4 需要 5.5 GB，两者叠加必然 OOM。跑长实验前先 `nvidia-smi` 看一眼。

**测试**：autoexp `run_all_tests.py` **53 / 53**（新增 6 条：真实负向文案、未校准警告、分组噪声三种取值顺序、配置相同的重复实验、降 batch 提示识别、提议继承最优）；Web 后端 `tools/test_autoexp_api.py` **15 / 15**。

详细复盘见 [7-autoresearch-yolo/README.md](7-autoresearch-yolo/README.md) v0.2.2。

### v1.33.1 — 2026-11-10（修自动实验首个实测事故：推理型模型把输出预算吃光）

**一句话**：`max_tokens` 是「思考 + 正文」的**共用预算**，900 会让推理型模型把预算全花在思考上、正文一个字都不剩。默认改 4096，并在截断时自动加倍重试；同时把「模型看不到取值」和「失败被误报成没有进展」两个坑一起补上。

**事故现场（真实 30 轮计划，跑到第 2 轮就崩）**

```text
[autoexp] 返回内容不是合法 JSON：找不到 JSON 对象：'{"hypothesis": "分辨率收益在 640→800→960 上单调递增...
[autoexp] 返回内容不是合法 JSON：返回内容为空
[autoexp] 提议被拒：train.imgsz = 1024 不在允许取值 [640, 800, 960, 1152, 1280] 内
[autoexp] 连续 5 轮没有进展，停止（省下无意义的 API 花费）
```

**真凶**（`experiments/agent/round_0004.json` 里存着 `usages`，直接可查）：

| 轮次 | `completion_tokens` | `reasoning_tokens` | 正文实际拿到 |
| --- | --- | --- | --- |
| 1（成功） | 629 | 416 | 213 token → JSON 完整 |
| 3 第 1 次 | **900** | **900** | **0** → 空 |
| 4 第 1 次 | **900** | **900** | **0** → 半截 |

「返回内容为空」和「JSON 突然半截」是**同一个原因**的两种表现，而日志把它们都写成「不是合法 JSON」——这句话会让人去改提示词、换模型，方向完全错。

**四个改动**

| 改动 | 老版本 | 新版本 |
| --- | --- | --- |
| 输出预算 | `max_tokens=900` | **4096**，页面上可调；截断时自动翻倍重试（上限 8192） |
| 报错文案 | 一律「返回内容不是合法 JSON」 | 新增 `LLMTruncated`：「被 max_tokens=N 截断：其中推理占 X、正文只有 Y，请调到 4096 以上」 |
| 上下文 | 只给可改参数的**键名** | 补 `search_space`（带 `enum` / `min` / `max`）→ 模型不再提出 `imgsz=1024`、`batch=2` 这种必然被拒的值 |
| 停止判据 | 「没进展」和「没提议」共用一个计数器 | 拆成 `--max-stale`（搜索到瓶颈）与 `--max-errors`（模型侧故障，默认 3），文案不再互相冒充 |

**其他**：空正文不再写入对话历史（部分兼容接口会对空 assistant 内容返回 400）；`llm.ping()` 遇到推理型模型截断时仍报「连通正常」（自检只有 16 token，必然截断，不能因此让用户以为 Key 坏了）；上下文里的 `_comment` 元数据键被滤掉（旧版 `allowed_top_level` 里赫然写着 `_comment`）。

**测试**：autoexp `run_all_tests.py` **47 / 47**（新增 4 条直接对着这次事故：截断自动加预算、截断报错要点名 `max_tokens`、`ping` 容忍截断、上下文必须给取值、两种停止原因文案不能共用）；Web 后端 `tools/test_autoexp_api.py` **15 / 15**（新增 `max_tokens` 往返与 `--max-errors` 透传）；前端 `test_autoexp_fmt.mjs` 27 / 27、`test_runsbar.mjs` 28 / 28。

详细复盘见 [7-autoresearch-yolo/README.md](7-autoresearch-yolo/README.md) §14.9 / §14.10 / v0.2.1。

### v1.33.0 — 2026-11-10（自动实验第 4 阶段：让模型提议，让程序裁判）

**一句话**：`7-autoresearch-yolo` 从「人改配置、人跑一轮」升级为「模型提议 → 程序校验 → 训练 → 按噪声阈值判 KEEP/DISCARD → 记账」的无人值守循环；网页端新增**独立「自动实验」页签**填 API Key 与看决策，**训练页老流程零改动**。

**为什么要做这一层**：前三阶段解决了「怎么让一次实验可比」，但仍要人每轮想改什么。人连看 5 轮就会开始凭直觉加码（首轮实验里 4 个假设有 3 个落在噪声区间，靠人眼根本分不出「+0.0037 的提升」和随机波动）。

**核心设计：提议与裁判彻底分离**

| 角色 | 谁来做 | 为什么 |
| --- | --- | --- |
| 提议（`hypothesis` / `changed` / `train` 参数） | LLM | 它擅长在已知结论之上联想；而且策略写在 `prompts/experimenter.md` 一个 markdown 里，换方向不用改代码 |
| 裁判（KEEP / DISCARD） | 确定性 Python | 判据是 `fitness - best > 2σ`（σ=0.003184，阈值 0.006368）。**模型说"显著提升"不算数**——它没有噪声底这个概念 |
| 写入 | `best.json` 只在 KEEP 时更新 | DISCARD 的配置不会污染下一轮起点 |

**硬约束（模型改不动的东西）**

`data` / `task` / `budget_seconds` / `epochs` / `model` / `device` / `seed` / `deterministic` / `plots` / `workers` / `cache` / `patience` / `amp` 全部冻结；且 `data` 做路径断言——解析后若落在 `pcb_holdout.yaml` 上直接拒绝，防止用验证集调参。改动项还要过 `configs/space.json` 白名单与区间校验，越界即打回让模型重提（最多 2 次）。

**默认值**：轮数 **30**（一轮约 11 分钟 ≈ 5.5 小时）、时间上限 8 小时、连续 5 轮无 KEEP 自动停。三者先到者生效。

**网页端「自动实验」页签**（详见 [6-web/README.md](6-web/README.md) 的 v1.33.0）

| 能力 | 说明 |
| --- | --- |
| 模型接口 | `base_url` / `model` / `temperature` / `API Key`，带「测试连通性」与「预览提示词」；Key 只显示遮罩，只存 `runtime/autoexp_llm.json` |
| 运行方式 | 轮数 / 时间上限 / 连续无进展 / 等显卡时长 |
| 日志 | 按轮分块：`[模型提议]`（假设·改动·参数·理由）→ 训练进度 → `[判定] KEEP|DISCARD`（差值、阈值、连续无进展轮数） |
| 台账表 | 「模型提议」与「判定」分两列，一眼看出「提议看着合理但没超过噪声」 |

**与训练页的隔离（本次重点，逐条验证过）**

| 隔离点 | 做法 |
| --- | --- |
| 路由 | 独立 `APIRouter`（`/api/autoexp/*`），不碰 `/api/train/yolo` |
| 前端 | 独立 `js/autoexp.js`，不 import `train.js` 的任何状态 |
| 事件 | autoexp 发 `{"type": "agent"}`，训练发 `{"type": "event"}` → 训练页的 switch 没有 `agent` 分支，同开也不会串台 |
| 接管 | `train.js` 的 `TRAIN_KINDS` 不含 `autoexp-agent` → 点「开始训练」遇 409 不会误接管自动实验的 run |
| 显卡 | 共用 `gpu_lock.json`，按 **pid 委派**：后端把自己 pid 传给子进程，只在「持有者恰好是自家后端」时跳过，别的进程持锁照样退出码 5 |

**本轮修掉的两个真实故障**

1. `--events`（tee）分支下，一轮结束后状态判定用了未定义的 `Popen` 对象 → `NameError`。症状是「训练明明跑完了，台账却没这行」，极难联想到是记账代码的问题。现抽成 `derive_result_status(result, returncode)` 并加单测。
2. `llm.load_config()` 的优先级把环境变量放在**显式指定之后**，导致 shell 里遗留的 `AUTOEXP_MODEL` 会让页面上的设置静默失效。现为「内置默认 ← 入库文件 ← `llm.local.json` ← 环境变量 ← 本次显式指定」。

**测试**：autoexp 侧 `run_all_tests.py` 42 / 42；Web 后端 `tools/test_autoexp_api.py` 13 / 13；前端 `test_autoexp_fmt.mjs` 27 / 27、`test_runsbar.mjs` 28 / 28。另有一次真实 tee 链路冒烟（120 秒预算，`status=done`、mAP50-95=0.19586），确认网页端那一轮也能正确记账。

### v1.32.0 — 2026-11-10（YOLO 自主实验循环：把"调参"变成可留痕的实验）

**一句话**：新增 `7-autoresearch-yolo` 模块，用「固定时间预算 + 单一适应度 + 台账留痕」在 PCB 数据集上做 YOLO 超参与增强的自动化实验。**对 `1-python-yolo/launcher` 零文件改动**（只 `import` 它三个纯函数），补丁前的代码用 git tag `yolo-stable-before-autoexp` 与 `_backup/launcher_20261110/` 双重锁住。

**为什么不再手调**：默认 100 epoch @640 要约 35~40 分钟，一次试错半小时；而且改了 epoch 数和改了增强的实验混在一起看，分不清是哪个起的作用；跑完十几个 `exp` / `exp-2` / `exp-3` 目录后也记不清对应哪套参数。

**核心机制**

| 机制 | 做法 | 解决什么 |
| --- | --- | --- |
| 固定预算 | `budget_seconds`（默认 600 秒）转成 Ultralytics 的 `time`（小时）传入，到点即停并按首轮实测耗时反推 epoch 数 | 不同配置、不同模型规模的实验可以横比 |
| 单一适应度 | 取 `results.csv` **最后一行**的 `mAP50-95`，不取历史最佳 | 取 max 会让"早慧型"配置虚高，固定预算下比的就是到点那一刻 |
| 确定性切分 | `split` 把 139 张 val 确定性切成 tune 97 / holdout 42（固定 `seed=42`，先排序再洗牌） | `best.pt` 是按 val 选的，若搜索也盯这 139 张就会把它拟合掉 |
| holdout 纪律 | 搜索期一次不许看，只在最后验收跑一次 | 否则看到的只是"在这 139 张上很好"，不是泛化好 |
| 硬链接镜像 | 732 MB 训练集用 `os.link` 镜像进 `splits/pcb/` | 不占额外磁盘，且不改动原始数据集 |
| 台账 | `results.tsv` 一行一个实验，含请求/实际生效 batch、峰值显存、假设描述 | 可回溯、可横比 |
| 显存预检 | 开训前比对可用显存与保守需求，不够直接拒绝 | 挡住 Ultralytics 的静默降 batch |

**为什么必须挡住静默降 batch**：Ultralytics 首轮 OOM 会自动把 batch 减半重来（`16→8→4→2`，每次从第 1 个 epoch 重头再来）。batch 一变，`nbs` 归一化后的等效学习率、正则强度全跟着变，那次实验就与台账里其它实验**不是同一件事**，横比全是假的。现在预检不过就记 `preflight_blocked`，零训练时间消耗，也不污染 `crash` 统计。

**本机实测（T1000 8GB）**

- `batch=16 @640`，可用显存 5302 MB → **OOM**。根因是 `6-web` 两套常驻服务各挂一个 `infer_worker.py` 长期占用约 2.7 GB；且这张卡在 Ultralytics 的 AMP 黑名单里，训练强制 fp32，激活显存比常规高一截。
- `batch=8 @640` → 正常，峰值 1769.5 MB，约 24~26 秒/epoch。
- 完全相同的配置 600 秒跑三次，mAP50-95 = `0.40496` / `0.41109` / `0.40385`，**sigma 0.003184**，
  极差 0.00724。远低于上游要求的 0.01，说明 600 秒预算充足，keep 阈值取 2σ = **0.006368**。

**首轮 5 个假设的结论（600 秒基准）**

| 改动 | tune mAP50-95 | 判定 |
| --- | --- | --- |
| `imgsz 640→960, batch 8→4` | **0.45862** | **KEEP**（holdout 复核 0.473726，同向成立） |
| `close_mosaic 10→3` | 0.41078 | discard（噪声内） |
| `lr0 0.01→0.02` | 0.40715 | discard（噪声内） |
| `optimizer→AdamW, lr0=0.002` | 0.40407 | discard（噪声内） |
| `mosaic 1.0→0.5, scale 0.5→0.7` | 0.3892 | discard（真退步） |

只有提高分辨率是真信号（约 15 sigma）。值得注意的是 960 在 600 秒内只跑得完 12 个 epoch，
却比 640 的 24 个 epoch 更好——**"每张图看得更清楚"的收益大于"多看几遍"**。

**用法**（详见 [7-autoresearch-yolo/README.md](7-autoresearch-yolo/README.md)）

```powershell
cd 7-autoresearch-yolo
python autoexp.py split --data "..\PCB data\PCB缺陷检测\data.yaml"   # 只需一次
python autoexp.py run --config experiments/cur.json                  # 跑一次实验
python autoexp.py report                                             # 看台账
python autoexp.py noise --repeat 3                                   # 测噪声底（决定 keep 阈值）
python autoexp.py holdout --weights runs/autoexp/exp_0007/weights/best.pt
```

**范围说明**：本轮只搭地基，**不含**"谁来决定下一次改什么参数"那一层，全程不需要任何大模型 API。Anomalib 侧不做——现有 `anomaly_val_report.json` 是 `auroc 1.0 / f1 1.0 / tp44 fp0 tn20 fn0`，奖励信号已饱和，任何配置都得满分，循环没有梯度可爬。EfficientAd 也不适合：20 epoch 要 3525 秒，压进 5 分钟预算会把 LR 日程一起改掉，实验之间比的就不再是"同一套方法跑多久"。

### v1.32.1 — 2026-11-10（网页端与自动实验互斥：不再抢显存）

**一句话**：网页端训练与命令行 `autoexp.py` 现在通过 `runtime/gpu_lock.json` 互相可见，一方在跑另一方会被明确拒绝，而不是两个训练一起 OOM。

**为什么需要**：两者的产物目录本来是隔离的（网页写 `1-python-yolo/launcher/runs`，autoexp 写 `7-autoresearch-yolo/runs/autoexp`），但**共用同一块显卡**。原来的 `_reject_if_busy()` 只查本进程内存里的任务表，看不见两种真实情况：

1. 命令行在跑 `autoexp.py run`；
2. 本机同时起着两个 Web 实例（8060 与 8094 各一套），各自的任务表互不可见。

后果不只是 OOM。autoexp 侧更麻烦：Ultralytics 遇 OOM 会把 batch 静默减半重试，于是这次实验的等效学习率、正则强度都和台账里其它实验不一样，横向比较全部失真，而台账上看不出来。

**行为**

| 场景 | 结果 |
| --- | --- |
| 网页点「开始训练」，锁被 autoexp / 另一套 Web 实例持有 | **409**，文案带占用方、PID、该去哪停 |
| `autoexp.py run` / `holdout`，锁被网页端持有 | **退出码 5**，不建实验目录、不写台账、零显存消耗 |
| 持有者被强杀 | 锁残留但 PID 已死 → 视为空闲，不会锁死机器 |
| 锁文件损坏 | 视为空闲 |

**实现**：`6-web/backend/gpulock.py`，零第三方依赖。放在 backend 里是因为打包脚本会整目录冻结，放外面会漏文件；autoexp 通过 `load_gpulock()` 复用同一份，不重复实现（锁文件格式是两边共用的契约，两份实现迟早漂移）。

**修掉一个危险写法**：存活探测原本用 `os.kill(pid, 0)`。这在 POSIX 上是只读检查，但 CPython 的 Windows 实现把非控制台信号的 sig 直接传给 `TerminateProcess(handle, sig)`——**等于把对方进程以退出码 0 杀掉**。实测确认：本进程调 `os.kill(os.getpid(), 0)` 后无声退出。现改为 Windows 走 `OpenProcess` + `GetExitCodeProcess`，POSIX 仍用 `os.kill(pid, 0)`。

**测试**：`6-web/backend/tools/test_gpulock.py` 7 项（含「Web 与 autoexp 读同一套字段」的契约测试）；`tools/test_concurrency.py` 7 项全过（含「两条重任务可并存」这条既有语义未被改变）；autoexp 侧 `run_all_tests.py` 17 项全过。

### v1.31.0 — 2026-11-10（网页端推理改成进程池：多人同时用不再排队）

**一句话**：DLL 层的并发契约是「一句柄一线程」，所以网页端要并行只有多句柄或多进程两条路。这一版把 `inferproxy` 从「单个工作进程 + 一把锁」改成**进程池**：多个 `infer_worker` 各自有一套 DLL 与句柄，请求分发到空闲进程；OCR 固定留在 0 号进程（Paddle 常驻 1.7 GB，不随池扩散）；熔断分两层——单进程崩 3 次只停它自己，全池 60 秒累计崩 6 次才全池熔断。

**默认值不拍脑袋**：推理路径本身（4 客户端并发）1 个进程 88.65 QPS / P95 139.8 ms，2 个 113.31 / **19.3 ms**，4 个 115.61 / 48.7 ms —— 4 个进程吞吐不再增长、尾延迟反而变差（小卡核心数撑不住更多并发上下文）。

**但要看清端到端**：真实 HTTP 4 并发下，YOLO 检测 3034×1586 在 1 个与 2 个进程时都是约 200 ms（**无差别**）——因为一次请求 90 ms 里模型只占 11 ms，其余是 Web 进程解码/渲染/编码/base64，池化改不到这一段。而 Anomalib PatchCore 的 4 并发最慢从约 680 ms 降到约 460 ms（**1.45×**），因为它的模型与后处理占绝对多数。所以：**吞吐收益与「推理占整条请求的比例」成正比**，两个任务都吃得到的收益是稳定性（单进程崩溃不再连坐、OCR 加载不再堵住检测）。

默认值仍按显存自动定：≥6 GB 开 2，否则 1；显式设 `TRTFORGE_INFER_WORKERS` 则以它为准。详情见 `6-web/README.md` 的 v1.31.0 变更点与 `docs/WORKFLOW_GUIDE.md` §7.15。

```yaml
version_diff:
  component: web-infer-pool
  old_version: "1.30.6"
  new_version: "1.31.0"
  symptom:
    question: "接着做上层并行，你要怎么修改打算"
    measured: "同步 def 路由本可并发，但 inferproxy 一把 RLock 包住整条管道、infer_worker 单连接单循环，请求只能排队"
    tail_latency: "4 个并发请求时最后一个要等前三个跑完，P95 139.8 ms"
  root_cause:
    handle_contract: "同一个 DLL 句柄不能被两个线程同时用（显存写乱 / 0xC0000409）"
    lock_layer: "inferproxy._Worker.call 全程持 RLock；infer_worker.main 只 accept 一个连接"
  old:
    workers: "单个 infer_worker 进程"
    distribution: "无，所有请求串行"
    ocd_block: "OCR 首次加载最多 180 秒，这期间检测请求一起被堵"
    breaker: "单进程 60 秒崩 3 次 → 全部推理不可用"
    handles: "全局句柄缓存上限 4"
  new:
    workers: "进程池，TRTFORGE_INFER_WORKERS（1~8，默认按显存自动：≥6GB 开 2）"
    distribution: "空闲进程令牌队列，取一个用一次放回；等不到按 QUEUE_TIMEOUT（默认 30s）报 503"
    ocr: "固定 0 号进程；其加载期间其余进程照常服务"
    breaker: "两层：单进程 3 次只停它自己（0 号行为与 1.28 一致）；全池累计 6 次才全池熔断"
    handles: "每进程 4 个槽，合计 4×N"
    diagnostics: "worker_state 新增 pool_size / workers[] / pool_tripped / pool_error，顶层字段保持兼容"
  measured:
    qps: "T1000 8GB、4 客户端：1→88.65，2→113.31（1.28×），4→115.61（1.30×）"
    p95: "1→139.8 ms，2→19.3 ms，4→48.7 ms"
    end_to_end_http: "4 并发最慢一个：YOLO 3034×1586 约 200ms（1 与 2 进程相同）；YOLO 1280 约 67ms（相同）；anomalib 约 680→460ms（1.45×）"
    bottleneck: "YOLO 请求 90ms 里模型只占 11ms，其余是 Web 进程解码/渲染/编码/base64，池化改不到这一段"
    tests: "test_infer_pool.py 20/20；test_infer_worker.py 26/26（WORKERS=1 与 2 各一遍）"
    consistency: "WORKERS=1 与 2 的框逐位一致（5 张图，15 框）"
  not_done:
    - "infer_batch.py / infer_video.py 仍是单进程串行（第二期）"
  unchanged:
    - "所有推理路由、请求/响应字段、参数与判定口径"
    - "TRTFORGE_INFER_ISOLATION=0 直通模式的行为"
```

### v1.30.6 — 2026-11-09（分类也算进了 DLL）

**一句话**：有用户问"推理部分所有代码和算法是不是都能通过 DLL 调用"，盘点后发现**只剩分类没有**——`detect / segment / pose / obb` 与 anomalib 三模型早就走 `onnx_infer.dll` / `yolo_infer.dll` / `anomalib_infer.dll`，OCR 走 `ocr_infer.dll`，只有 `classify` 还在 Web 进程里 `import onnxruntime`。这一版给 `onnx_infer.dll` 加了 `OI_CLASSIFY`、`oi_create_classify`、`oi_infer_classify`、`oi_classify_result`，分类从此也走 C++（含 CUDA provider 自动探测）、走 `infer_worker` 进程隔离，网页单图 / 批量 / 视频三条路径统一 DLL 优先、Python 兜底。

DLL 刻意**只回原始分数和 `names` 元数据**，`softmax`、置信度门槛、中文名映射仍留在 `classifyinfer.interpret()`：那套解释逻辑已经兼容"导出的模型带 softmax 或不带"，在 C++ 里再写一份，同一个模型走两条路会得出不同结论。分类预处理（短边缩放 + 中心裁剪 + Python 同款 `round`）与 `classifyinfer.preprocess` 逐位一致，静态/动态 batch 共 16 例（含 1792×528 长图）原始分数最大绝对差 **0.0**。详情见 `6-web/README.md` 的 v1.30.6 变更点与 `docs/WORKFLOW_GUIDE.md` §7.13。

```yaml
version_diff:
  component: infer-classify-dll
  old_version: "1.30.5"
  new_version: "1.30.6"
  symptom:
    question: "现在推理部分所有代码和算法是否都能通过 DLL 调用"
    measured: "11 类推理路径里 10 类已在 DLL 内执行；classify 仍在 Web 进程内用 Python onnxruntime"
  root_cause:
    classify: "分类是 v1.30.0 新增任务，当时只接到 classifyinfer.run()，没进 C ABI"
    isolation: "不进 infer_worker，长任务会占住 Web 进程；也拿不到 CUDA execution provider"
  old:
    classify_onnx: "Python onnxruntime，进程内 CPU"
    worker_op: "无 classify 分支"
    backend_field: "分类永远显示 python"
  new:
    abi_added: "OI_CLASSIFY / oi_classify_result / oi_create_classify / oi_infer_classify（仅追加，旧调用方不受影响）"
    classify_onnx: "onnx_infer.dll，EP 自动探测（CUDA → CPU）"
    worker_op: "infer_worker op=classify，共享内存传 BGR，结果走同一条 pipe"
    backend_field: "单图 / 批量 / 视频三条路径 DLL 优先，backend 显示 onnx_infer.dll，降级时记录原因"
    contract: "DLL 只回原始分数 + names；softmax/门槛/中文名映射保留在 classifyinfer.interpret()"
    version: "onnx_infer 1.0.0 → 1.1.0（abi 仅追加）"
  measured:
    preprocessing: "静态 + 动态 batch 共 16 例（含 1792×528），DLL 与 Python 原始分数最大绝对差 0.0"
    tests: "后端 44 个脚本全绿；test_onnx_infer_dll 46 项（分类 11 项）；test_classify 45/45"
  verify:
    http: "POST /api/infer/yolo task=classify，8094/8060 均返回 backend=onnx_infer.dll，label 与降级路径一致"
    fallback: "删掉 onnx_infer.dll 后分类仍可用，backend 回落 python 并记录降级原因"
  unchanged: "分类的响应字段、判定口径与模型格式；检测/分割/姿态/OBB/anomalib/OCR 全部路径；分类仍无 TRT 引擎"
```

### v1.30.5 — 2026-11-09（OCR 往返拆解 + 识别 batch 按设备取值）

**一句话**：OCR 的「含往返 215 ms」一直被当成"传输慢"，这一版把服务端继续拆成**解码 / 进程往返 / 模型 / 响应组装**四段并显示在页面上——结论是 OCR 的 182 ms 就是模型（检测 + 逐行识别），Python 侧只有 33 ms，而且 OCR 本来就没图要传（响应体 9.5 KB）。同时顺手做了三件事：`_finish_infer` 不再为报体积序列化两次（2.5 MB 响应省约 30 ms）；识别 batch 默认值按设备分开取（**CPU 24 / GPU 6**，实测 CPU 上 24 比 6 快约 18%，GPU 上 6 才是最优）；新增 15 项计时自洽测试。详情见 `6-web/README.md` 的 v1.30.5 变更点与 `docs/WORKFLOW_GUIDE.md` §7.12。

```yaml
version_diff:
  component: web-infer-timing
  old_version: "1.30.4"
  new_version: "1.30.5"
  symptom:
    question: "OCR 推理的往返还是很高（215 ms），为什么目标检测和异常检测都好了"
    measured: "服务端 214.9 ms / DLL 182.1 ms；OCR 响应体 9.5 KB，没有可压缩的图片回传"
  root_cause:
    ocr: "OCR 的时间就是「检测 + 逐行识别」本身；v1.30.3 的 PNG→JPEG 只对回传大图的 YOLO/Anomalib 有效"
    visibility: "timing 只有 server_ms，用户无法看出服务端内部各段占比"
    batch: "识别 batch 固定默认 6，CPU 上偏小 18%；而 GPU 上 6 恰好最优（'GPU 更大更快'是单次测量的错觉）"
  old:
    timing_fields: "server_ms / payload_bytes"
    payload_bytes_impl: "填 0 序列化一次量长度、再填真值序列化一次"
    ocr_batch_default: "6"
    legend: "只有「服务端总计」与「含往返」"
  new:
    timing_fields: "新增 decode_ms / ipc_ms / encode_ms，server_ms = decode + ipc + DLL + encode"
    payload_bytes_impl: "只序列化一次，原地替换 \"payload_bytes\":0 回填真实 body 长度"
    ocr_batch_default: "CPU 24 / GPU 6，切换设备自动跟随（用户手改过的值不覆盖）"
    legend: "新增「服务端拆解 ＝ 模型 ＋ 解码 ＋ 进程往返 ＋ 响应组装」"
  measured:
    yolo: "服务端 53.6 ms = 解码 21.3 + 往返 1.3 + 模型 13.7 + 组装 17.3（响应体 1484 KB）"
    anomalib: "服务端 246.0 ms = 解码 21.7 + 往返 154.7 + 模型 64.4 + 组装 5.2（响应体 2504 KB）"
    ocr: "服务端 3083 ms = 解码 6.7 + 往返 3.2 + 模型 3070.6 + 组装 2.5（响应体 15.5 KB，75 行）"
    cpu_batch: "batch 6 → 3466 ms，24 → 2828 ms，48 → 3732 ms"
    gpu_batch: "batch 6 → 576 ms，16 → 621 ms"
  verify:
    payload_bytes: "与真实响应体逐字节一致（test_infer_timing.py 新增断言）"
    tests: "后端 44 个脚本全绿、前端 6 套全绿"
  unchanged: "三种任务的响应字段与识别结果、DLL、OCR raw_json（openspec 规定必须回传）"
```

### v1.30.4 — 2026-11-09（OCR 文字醒目化 + 异常检测回到「区域」）

**一句话**：两处判读体验修复。OCR 识别结果的文字标签原来是白色，叠在白底/浅底图上直接糊掉，现改成琥珀色加深色描边（任何底色都能读）；Anomalib 的第四个窗口原来叫「异常轮廓」，v1.30.3 为了压体积把它二值化成黑底白线，用户反馈**区域比轮廓可靠**，这一版改成真正的**填充异常区域**（RGBA 红区，前端半透明叠到本地原图上），并顺手修掉"铜/黄铜工件被整圈误判成异常"的颜色判定问题。体积仍是 20~45 KB（v1.30.2 是 7.1 MB）。详情见 `6-web/README.md` 的 v1.30.4 变更点。

```yaml
version_diff:
  component: web-infer-readability
  old_version: "1.30.3"
  new_version: "1.30.4"
  symptom:
    ocr_label: "识别到的文字是白色（#f4f7fb），浅底图上不可读"
    anomalib_mask: "异常面板从「红色区域」变成「黑底白线轮廓」，且环形铜件整圈被标成异常"
  root_cause:
    ocr_label: "文字只有 fill，没有描边/背景，深色字与浅色字都会在某类底色上失效"
    anomalib_mask: "mask_rgb 是 DLL 用 drawContours(thickness=2) 画的描边，本身只有轮廓；v1.30.3 靠偏红判定抠线，铜/黄铜天然偏红会误判"
  old:
    ocr_label: "fill=#f4f7fb"
    anomalib_mask: "抠红色 → 二值白线（单通道 PNG）"
    onnx_mask: "norm >= pixel_threshold（0~1 相对值 vs 原始分数阈值，量纲不一致）"
    window_title: "异常轮廓"
  new:
    ocr_label: "fill=#ffd54a（琥珀）+ stroke=#0b1016 + paint-order=stroke"
    anomalib_mask: "normalized_map > 0.5 生成填充区域（与 DLL 内部 mask 同口径），RGBA：异常处不透明纯红，其余透明"
    saturation_guard: "覆盖率超过 60% 视为阈值标定失效，退回只标最热一段"
    compose: "前端把红区以 0.45 透明度叠到本地原图（URL.createObjectURL）上显示"
    onnx_mask: "先按 metadata 归一化到 0..1 再判 0.5，与 DLL 口径一致"
    window_title: "异常区域"
  verify:
    region_coverage: "bottle 异常图 0.7%~1.5%；跨数据集饱和样本触发兜底 60%"
    payload: "RGBA PNG 20~45 KB"
    tests: "test_onnx_fallback 64 项、test_infer_worker 26 项、前端 6 套全绿"
  unchanged: "DLL 本体与 C API、响应字段名（仍叫 mask_png）、overlay_jpg / heatmap_png 策略、YOLO 与 OCR 计算链路"
```

### v1.30.3 — 2026-11-09（推理页往返耗时优化）

**一句话**：推理页「含往返」很长，但 DLL 的 `preprocess/inference/postprocess` 只有几毫秒——差额全在**编图、base64、JSON 和传输**。这一版把 YOLO 标注图从 PNG 改成 JPEG、Anomalib 不再回传原图并把轮廓图压成二值、OCR 提前预热句柄，本机实测 YOLO 388→约 210 ms（9.9→约 1.6 MB）、Anomalib 1077→约 180 ms（28→约 2.1 MB）。同时响应里补了 `timing.server_ms` 与 `timing.payload_bytes`，前端把往返拆成「上传 / 服务端 / 下载」三段，下次不会再误判成模型慢。详情见 `6-web/README.md` 的 v1.30.3 变更点与 `docs/WORKFLOW_GUIDE.md` §7.10。

```yaml
version_diff:
  component: web-infer-roundtrip
  old_version: "1.30.2"
  new_version: "1.30.3"
  symptom: "DLL 计时 5~74 ms，页面「含往返」却是 388~1077 ms；OCR 首次还要多等约 1.3 s"
  root_cause: "DLL 只计内核；PNG 编码、base64、JSON 序列化、传输都不在计时内，且全尺寸原图重复回传"
  old:
    yolo_image: "annotated_png（3034x1586 约 155 ms / 5.8 MB）"
    anomalib_image: "overlay_png + mask_png + source_png + heatmap_png 四张全尺寸 PNG（28 MB）"
    ocr_warm: "首次推理才 oi_create，加载飞桨 1.7 GB 约 1.3 s，不在 timing 里"
    ocr_cache: "每次 preflight 重新解析 inference.yml（character_dict 6000+ 项）"
    upload: "上传先写一份临时文件，三个调用方都没读过它"
    timing: "响应只有 DLL 三段计时，看不出服务端与传输各占多少"
  new:
    yolo_image: "annotated_jpg（JPEG q=90，约 12 ms / 0.78 MB）+ annotated_mime"
    anomalib_image: "overlay_jpg + 二值 mask_png + heatmap_png；不再回传 source_png"
    ocr_warm: "POST /api/ocr/warm 后台预热；前端模型列表加载完自动触发"
    ocr_cache: "inspect_dir 按目录 + 关键文件 mtime + 飞桨主版本缓存"
    upload: "_read_upload 只读字节不落盘，保留后缀与大小校验"
    timing: "timing.server_ms / payload_bytes，前端拆成 上传 / 服务端 / 下载 三段"
  verify:
    yolo: "本机 388 ms / 9.92 MB → 约 210 ms / 约 1.6 MB"
    anomalib: "本机 1077 ms / 28.45 MB → 约 180 ms / 约 2.1 MB"
    ocr: "首次 2446 ms 的句柄创建被预热吃掉，后续约 290 ms"
    tests: "run_all_tests.py 全绿；test_ocr_api 新增缓存与预热断言"
  unchanged: "DLL 本体与 C API、检测/异常/OCR 的结果数值、导出 CSV/JSON 结构"
```

### v1.30.2 — 2026-11-09（OCR `mklml.dll` 加载修复）

**一句话**：OCR 推理点「开始推理」就报 `mklml.dll` 错误码 126，CPU 和 GPU 都不可用。根因不是缺文件，而是**飞桨 2.6.2 用普通 `LoadLibrary("mklml.dll")` 找 MKL，只认 PATH**，网页侧只 `add_dll_directory` 对它无效。现在 `ocr_infer.py` 会把 `bin` 和 `mklml/lib` 同时前置到进程 PATH；CMake 也补上了 `mklml.dll` 的自动复制。页面实测 CPU 75 行 4529 ms、GPU 75 行 9446 ms。详情见 `6-web/README.md` 的 v1.30.2 变更点与 `docs/WORKFLOW_GUIDE.md` §7.7.1。

```yaml
version_diff:
  component: web-ocr-runtime
  old_version: "1.30.1"
  new_version: "1.30.2"
  symptom: "创建 OCR 句柄失败：PreconditionNotMetError ... (mklml.dll) ... (error code is 126)"
  root_cause: "AddDllDirectory 只影响 LoadLibraryEx 的 USER_DIRS，飞桨内部用普通 LoadLibrary，看不到"
  old:
    runtime: "ocr_infer.py 只调 os.add_dll_directory(bin, mklml/lib)"
    build: "ocr_infer 的 POST_BUILD 依赖清单漏 mklml.dll，bin 里没有它"
    qt_cmake: "只在 bin/paddle_inference 下找 mklml，源码树该目录为空"
    tests: "没有覆盖 DLL 搜索路径"
  new:
    runtime: "bin 与 mklml 目录同时前置到进程 PATH，再加载 ocr_infer.dll"
    build: "POST_BUILD 把 mklml.dll 和 libiomp5md.dll 一起拷到 4-qt-studio/bin"
    qt_cmake: "回退到 ocr/ocr_infer/third_party/paddle_inference"
    tests: "test_ocr_api.py 增加 3 项回归"
  verify:
    page_cpu_mkldnn: "75 行，4529 ms"
    api_cpu: "75 行，5209 ms"
    api_gpu0: "75 行，9446 ms"
    tests: "test_ocr_api 13 项、test_ocr_infer_dll 10 项全绿"
  unchanged: "ocr_infer.dll 本体、模型兼容策略、视频/摄像头不支持 OCR"
```

### v1.30.1 — 2026-11-09（分类真机闭环）

**一句话**：Web Studio **1.30.1** 用官方 `yolov8n-cls.pt` 和 `imagenette160` 把分类跑通了。训练不再因为传 `dataset.yaml` 报错，推理的输入和官方 `classify_transforms` 逐像素一致，类别顺序以模型自带的 `names` 为准。两套数据集逐张对比都是 20/20 一致。分类引擎仍然不做。详情见 `6-web/README.md` 的 v1.30.1 变更点。

```yaml
version_diff:
  component: web-yolo-classify
  old_version: "1.30.0"
  new_version: "1.30.1"
  old:
    train_data: "dataset.yaml 直接交给 Ultralytics，报 must be a directory"
    train_metric: "训练结束没有 top-1"
    preprocess: "cv2.resize 直接拉伸，非正方形图变形"
    class_order: "按 dataset.yaml 的顺序，中文排序不一致就错位"
    overview: "分类分辨率恒为 0"
  new:
    train_data: "yaml 解析成目录，路径不对时给中文报错"
    train_metric: "训练结束发 accuracy_top1，页面显示「top1 准确率」"
    preprocess: "短边等比缩放 + 中心裁剪，和官方逐像素一致"
    class_order: "优先用 ONNX metadata 的 names，冲突时提示"
    overview: "分类补分辨率统计；视频逐帧明细写类别和分数"
  verify:
    cn2: "10 轮 top1 0.95，逐张对比 20/20 一致"
    multi: "3 轮 top1 0.84，逐张对比 20/20 一致"
  note: 仍只支持 ONNX 分类推理，TensorRT 分类引擎未做。
```

### v1.30.0 — 2026-11-08（YOLO 整图分类）

**一句话**：Web Studio **1.30.0** 可以做整图分类。标注页点一个类别给整张图打标，训练用分类权重，推理用 ONNX 给出类别和分数。分类引擎这一版不做。检测画框流程不变。详情见 `6-web/README.md` 的 v1.30.0 变更点。

```yaml
version_diff:
  component: web-yolo-classify
  old_version: "1.29.0"
  new_version: "1.30.0"
  old:
    web_classify: false
    annotation_task: detect
    infer: "只画框"
  new:
    web_classify: true
    annotation: "整张图一个类别，导出类别文件夹"
    train: "classify 权重，准确率 + 混淆矩阵"
    infer: "ONNX 整图类别；.engine 返回 400"
    pipeline: "训练并导出 ONNX，跳过编译引擎和检测验证"
  note: 桌面版不能打开分类工程。TensorRT 分类引擎未做。
```

### v1.29.0 — 2026-11-06（检测验证报告 + YOLO INT8）

**一句话**：Web Studio **1.29.0** 的检测验证会给出混淆矩阵，以及漏检、过检、判错清单；YOLO 引擎可以选 INT8，用一批图片做校准。分类任务这次没做。详情见 `6-web/README.md` 的 v1.29.0 变更点。

### v1.28.0 — 2026-11-06（Web Studio 推理进程隔离）

**一句话**：Web Studio **1.28.0** 起，YOLO / Anomalib / ONNX / OCR 推理 DLL 只在独立子进程 **`infer_worker`** 里加载；DLL 崩溃不再拖垮整个 Web 服务，当前请求返回 503/504 并自动或手动重启工作进程。详情见 `6-web/README.md` 的 v1.28.0 变更点与 `docs/WORKFLOW_GUIDE.md` §7.8。

### v1.27.0 — 2026-11-05（推理页接入文字识别，仍走 ocr_infer.dll）

**一句话**：Web 推理页多了「OCR 文字识别」。检测、识别、文本行方向都在原来的 `ocr_infer.dll` 里完成，Python 只负责收图和把文字画回页面。桌面版程序没有改。

```yaml
version_diff:
  component: web-ocr-inference
  old_version: "1.26.0"
  new_version: "1.27.0"
  old:
    web_ocr: false
    infer_kinds: [yolo, anomalib]
    ocr_runtime: "仅 Qt 桌面版调用 ocr_infer.dll"
  new:
    web_ocr: true
    infer_kinds: [yolo, anomalib, ocr]
    backend: ocr_infer.dll
    paddle: "2.6.2"
    model_format: "inference.pdmodel + inference.pdiparams + inference.yml"
    unsupported_in_this_version:
      - "inference.json（Paddle 3.x 模型，页面置灰）"
      - "文档方向分类"
      - "UVDoc 弯曲矫正"
      - "视频"
      - "摄像头"
    packaging: "package_web_user.ps1 -WithOcr，默认关闭"
  measured_2026_11_05:
    image: "00006737.jpg（登机牌）"
    lines: 75
    contains: "登机牌"
    cpu_mkldnn_ms: 3765
    gpu0_ms: 1147
    rotated_180_with_cls: "angle=1，仍识别出登机牌与张祺伟"
```

| 项目 | 老版本（v1.26.0） | 新版本（v1.27.0） | 影响 |
| --- | --- | --- | --- |
| 推理页类型 | 只有 YOLO 和异常检测 | 增加 OCR | 选文字识别后，引擎路径换成三个模型目录 |
| 谁在算 | 文字识别只在桌面版 | Web 也调用同一份 `ocr_infer.dll` | 不新写一套 Python OCR |
| 能用的模型 | 桌面版自带目录 | 老版 `pdmodel`（本机放在 `9-python-paddleOCR/model/legacy_v4`） | 已下载的 `inference.json` 模型会显示灰色，提示需要飞桨 3.x |
| 方向 | 桌面版可选 | 文本行方向可选；正图 `angle=0`，倒图 `angle=1` | 留空就只做检测和识别 |
| 批量 | 目录 / 视频 / 摄像头 | OCR 只支持目录批量，写出 `ocr/*.ocr.json` 和 `ocr_summary.csv` | 视频和摄像头会直接提示不支持 |
| 安装包 | 不含飞桨运行库 | `-WithOcr` 才拷贝 `paddle_inference.dll`（约 1.7GB）和 `mklml.dll` | 默认安装包体积不变 |

**参数（单张与目录批量共用）**

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| 检测模型目录 / 识别模型目录 | 必填 | 各自含 `inference.pdmodel`、`inference.pdiparams`、`inference.yml` |
| 文本行方向目录 | 空 | 空 = 不旋转文本行 |
| 设备 | `cpu` | `cpu` 或 `gpu:0` |
| 线程 / 识别 batch | 8 / 6 | CPU 线程数、识别一次送几行 |
| MKLDNN | 开 | 只在 CPU 生效；关掉会从几秒变成约一百秒 |
| det_thresh / det_box_thresh / unclip | 0.3 / 0.6 / 1.5 | 检测框松紧 |
| 识别分数下限 | 0 | 低于它的行不返回 |

**使用方式**：推理页把类型改成「OCR 文字识别」，选检测和识别目录，上传图片后点开始推理。右边出现文字多边形和文本表，可复制全部文字，或导出 JSON / CSV / TXT。

**应用场景**：登机牌、单据、标签上的印刷文字。单张 CPU 大约数秒，不适合视频流。文档整页方向和弯曲矫正留到下一期。

**验证**：`run_all_tests.py` **40 个脚本全绿**（v1.26.0 是 38 个）。其中 `test_ocr_api.py` 10 项、`test_ocr_infer_dll.py` 10 项。`ocr_basic.exe` 在补上 `mklml.dll` 搜索路径后退出码 0。

### v1.26.0 — 2026-11-05（`.onnx` 走自己的动态库，与 `.engine` 结果可自动比对）

**一句话**：推理页填 `.onnx` 时不再走 Python `onnxruntime`，改走新增的 `onnx_infer.dll`；
同一张图分别用 `.engine` 与 `.onnx` 跑，结果差异可以一条命令判定通过/不通过。

| 项目 | 老版本（v1.25.3） | 新版本（v1.26.0） | 影响 |
| --- | --- | --- | --- |
| `.onnx` 推理 | Python `import onnxruntime` 兜底 | 优先 `onnx_infer.dll`（ONNX Runtime C API，CPU 必用、有 CUDA 时自动加速） | 打包只多 `onnxruntime.dll` 等 2 个文件 |
| 后端分派 | 引擎 / ONNX 两条独立代码路径 | 三级分派：`.onnx` → `onnx_infer.dll`；`.engine` → TensorRT DLL；都不可用才兜底 | 参数语义统一，换后端不改请求 |
| 结果可追溯 | 无后端信息 | 推理结果与批量/视频 JSON 都带 `backend` 字段 | 页面徽标与实际执行者一致 |
| 一致性校验 | 靠人眼看两张图 | `compare_backends.py`（结果级）+ `compare_raw_tensors.py`（张量级），超阈值非零退出 | 回归可进 CI |
| 诊断能力 | 出问题只能看最终图 | `TRTFORGE_RAW_DUMP=<dir>` 导出输入/输出原始张量（未设置零开销） | 能分清是模型差异还是后处理差异 |

**新增目录**：`3-cpp-tensorrt/onnx_infer/`（C ABI 头、ORT 会话封装、四任务解码、Anomalib 管道、README）。
**新增脚本**：`scripts/fetch_onnxruntime.ps1`；`build_infer_dlls.ps1` 增加 `-Targets onnx_infer`。
**新增工具**：`6-web/backend/tools/` 下 `compare_backends.py`、`compare_raw_tensors.py`、
`test_backend_consistency.py`、`test_onnx_infer_dll.py`。
**打包**：`packaging/package_web_user.ps1` 带上 `onnx_infer.dll` 与 ORT 运行库，并写 `bin_manifest.txt` 清单自检。

**实测结果（2026-11-05，Windows 10 / CPU EP）**

| 用例 | 指标 | 结果 |
| --- | --- | --- |
| YOLO detect（PCB 8 张） | 匹配率 / 最小 IoU / 置信度差 | 100% / > 0.92 / < 0.02，8/8 通过 |
| YOLO detect 原始张量 | 余弦相似度 / 最大相对差 | 0.999998 / ≤ 1.3% |
| Anomalib 三模型（8 张） | 分数差 / 判定一致 | ≤ 0.055 / 100%，8/8 通过 |
| Anomalib 原始张量 | 余弦相似度 / 二值 mask 一致率 | 0.9917 / 100% |
| 四任务耗时（CPU） | detect 960 / seg 320 / pose 320 / obb 320 | 70.2 / 14.6 / 11.5 / 12.6 ms |

**验证**：`run_all_tests.py` **38 个脚本全绿**（v1.25.3 是 35 个），前端 js 全部 `node --check` 通过；
`POST /api/infer/yolo`、`POST /api/infer/anomalib`、`infer_batch.py`、`infer_video.py` 四路实测
均返回 `backend: "onnx_infer.dll"`。细节见 `3-cpp-tensorrt/onnx_infer/README.md` 与 `6-web/README.md`。

标注页补充（2026-11-03）：矩形四个角拖动时，按住的角跟着鼠标走，不再跳到对角。旋转框选中后，拖上边外侧的黄点才能旋转。右栏「标注框」表列出当前图片的框，点一行，图上对应的框会青色加粗高亮。



### v1.25.0 — 2026-11-03（训练「点完就开始」，并支持拿自己的 best.pt 继续微调）

**一句话**：从点「开始训练」到第一个 epoch 的固定开销，YOLO 由 **1.56s** 降到 **0.19s**，
Anomalib 由 **4.32s** 降到 **≈0s**；页面上多了启动清单；历史训练多了一个「用它微调」按钮。

| 项目 | 老版本（v1.24.2） | 新版本（v1.25.0） | 影响 |
| --- | --- | --- | --- |
| YOLO 开训固定开销 | 每次新起进程重新 import，实测 1.56s | 预热进程池提前导入，实测 0.19s | 省 **1.37s** |
| Anomalib 开训固定开销 | 实测 4.32s | 命中预热 0.00s | 省 **4.32s** |
| 启动过程可见性 | 只有「运行中」，卡住时分不清是加载慢还是死了 | 清单 `spawned → boot → deps → model → data → train_start`，每步耗时 + 总耗时 | 慢的时候能定位 |
| 联网检查 | Ultralytics 每次开训查 PyPI | 默认 `YOLO_OFFLINE=1` | 少一次外网等待 |
| 解释器探测 | 每次开训 import torch 探一遍 | 结果缓存 30 秒（路径+mtime+大小失效） | 连续开训不再重复 |
| 训练起点 | 只有「新训」和「断点续训」 | 新增「用它微调」：载入 `best.pt`，**重置优化器与轮次**，写回原目录 | 补几张图不用重训 100 轮 |
| 覆盖旧结果 | 同目录再训静默生成 `exp2` | 微调带 `--exist-ok` 写回原目录，覆盖前备份到 `_prev/<时间戳>/` | 不误删上次结果 |
| 学习率 | 表单没有这一项 | 表单新增 `lr0`；微调自动填 0.001 | 微调不会把原权重练坏 |

**参数（环境变量）**

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `TRTFORGE_WARM` | `1` | 预热进程池开关；`0` = 回到冷启动（排障 / 内存紧） |
| `TRTFORGE_WARM_KINDS` | `yolo,anomalib` | 只预热其中一类 |
| `TRTFORGE_WARM_IDLE_SECONDS` | `1800` | 空闲多久回收预热进程 |
| `TRTFORGE_YOLO_ONLINE` | `0` | `1` 才允许 Ultralytics 联网查更新 |

**新增接口**：`GET /api/warm/status`、`POST /api/warm/prewarm?kind=`、`POST /api/train/yolo/finetune-check`。

**微调 vs 续训（别混）**

| 场景 | 请求字段 | 起点 | 优化器 / 轮次 | 超参来源 |
| --- | --- | --- | --- | --- |
| 新训 | `init_mode=preset` | 官方预训练权重 | 全新 | 表单 |
| 微调 | `init_mode=finetune` | 历史 run 的 `best.pt` | 重置 | 表单（自动 `lr0=0.001` / `epochs=30` / `mosaic=0.3` / `close_mosaic=15`） |
| 续训 | `resume_from=<last.pt>` | 历史 run 的 `last.pt` | 恢复 | 检查点（表单参数失效） |

微调前置检查会**拦住任务头不一致**（`detect` 权重拿去 `segment`），提示类别增删，
`imgsz` 变化只提醒。适用范围：同产品、同类别体系，只补了少量新样本。

**验证**：`run_all_tests.py` **33 个脚本全绿**（v1.24.2 是 30 个），新增
`test_warmpool.py`（25 项）、`test_phase_events.py`（15 项）、`test_finetune_request.py`（26 项）；
`bench_startup.py` 输出上面那组冷/热对比；页面实测 1 轮 YOLO 训练清单显示
`deps 0.07s / model 0.03s / data 18.50s`，微调确认 `_prev/` 备份出现。

### v1.24.2 — 2026-11-03（浏览按钮不再卡死，窄窗口不再挡住按钮）

| 项目 | 老版本（v1.24.1） | 新版本（v1.24.2） | 影响 |
| --- | --- | --- | --- |
| 浏览 | 系统窗口没弹出来时，按钮变成灰色「…」，要刷新才能再点 | 按钮变成可点的「取消」。窗口约 12 秒还没出现，就改用页面里的选择 | 不用再刷新页面 |
| 窄窗口 | 训练日志会盖住左栏下面的按钮 | 收起的折叠内容不再排到日志上面，表单按自己的高度往下排 | 窄屏和 1600 宽都能点到「小缺陷预设」 |
| 画布 | 矩形拖角、旋转黄点、表格点行高亮这次验证前没逐项点过 | 已在页面上点过：角跟着鼠标走，黄点能转到 90°，点表格哪一行哪一个框高亮 | 这三项可用 |

顶栏版本是 `v1.24.2`。第一轮测试里 8060 自己退出过一次，日志里没有异常，重放没有再发生，推理代码这次没改。

### v1.24.1 — 2026-11-03（权重到位后跑真图，修掉 3 个阻断性缺陷）

v1.24 交付时本机没有模型权重，SAM / GroundingDINO 只验证了"模型缺失"路径。这一版把
**214.3 MiB** 权重真的下到本地、用真图跑通，抓到 v1.24 里**单测完全覆盖不到的阻断性缺陷**。

**总结一句话：v1.24 的点框提示通道在真机上 100% 崩；文本通道能跑但分数与参考实现对不齐。**

| 项目 | 老版本（v1.24） | 新版本（v1.24.1） | 影响 |
| --- | --- | --- | --- |
| 内置模型体积 | 手估 40.1 MB / ≈700 MB（合计"约 760 MB"） | 按 `Content-Length` 校准：26.85 + 15.74 + 171.28 + 0.44 = **214.3 MiB** | 进度条分母对了，能真正走到 100% |
| 下载镜像 | `hf-mirror.com/CVHub520/...` 三条回退 | GitHub 直连 → `ghfast.top` → `gh-proxy.com` | 原来三条回退**实测全 404**，主源不通时白等超时 |
| 点框提示 | **任意一次点/框都抛 `ValueError` → 500** | 正常出多边形 | `build_prompt()` 写 `points or []`，而上游给的是 numpy 数组 |
| SAM encoder 输入 | 写死 NCHW `(1,3,H,W)`，被 onnxruntime 拒收 | 按签名判断，本项目是 **HWC rank-3 `[H,W,3]`** | 报错 `Invalid rank for input: input_image Got: 4 Expected: 3` |
| provider 选择 | `get_available_providers()` 直接信任 | 新增 `onnxenv.py`：先 `LoadLibrary` 探 provider DLL | 缺 cuDNN 的机器上不再每次建 session 都打一大段 `LoadLibrary failed with error 126` |
| 文本通道通道序 | 直接喂 BGR，注释称"顺序无关" | `preprocess_image()` 内做 `BGR→RGB` | 与参考实现 `qt_img_to_rgb_cv_img()` 一致，分数可对齐 |
| 回归脚本数 | 29 个 | **30 个**（多 `test_onnxenv.py`）；`test_prompt_label.py` 49 → **63** 项 | 两个新缺陷都有对应回归测试 |

**新增文件**：`6-web/backend/onnxenv.py`、`6-web/backend/tools/test_onnxenv.py`（11 项）。
**改动文件**：`modelzoo.py`（体积 / 镜像）、`promptlabel.py`（提示展开 / encoder 布局 / provider）、
`textlabel.py`（provider / RGB）、`app.py`（`APP_VERSION = "1.24.1"`）、`test_prompt_label.py`。

**OpenSpec**：本次走了一个正式变更 `harden-prompt-autolabel-with-real-weights`
（已归档为 `2026-11-03-harden-prompt-autolabel-with-real-weights`）。理由不是"改功能"，
而是 v1.24 归档的两个需求**验收标准没真正执行**——`--strict` 校验通过，主规格新增 4 个场景：

| 需求 | 新增场景 | 守住的是什么 |
| --- | --- | --- |
| `Prompt Based Auto Labeling` | 提示输入容忍数组类型 | 上游可能给 numpy 数组，不能写 `points or []` |
| `Prompt Based Auto Labeling` | 内置权重上真机可跑通 | 必须按模型声明的签名造输入，不能假定张量布局 |
| `Builtin Model Repository` | 体积与实际文件一致 | 体积同时是进度分母与完整性判据，必须是服务端真实值 |
| `Builtin Model Repository` | 回退地址可用 | 候选镜像必须实测能返回该文件 |

**真机实测**（Windows 10 / CPU / `onnxruntime` CPU EP）：SAM 首次 0.76 s、缓存命中 0.077 s、
框提示 0.079 s；GroundingDINO 加载 4.80 s、单图 4.42 s。明细表见下方 v1.24 的
「耗时（v1.24.1 真机实测）」。

**有一件事 v1.24.1 没能改善，写在这里免得后人白试**：PCB 图 3034×1586（比例 1.91）被硬拉伸到
模型固定输入 1200×800（比例 1.5）后，`missing hole` 会出一个占图 95.7% 的大框。
试过 letterbox 等比补边，PCB 上确实降到 75.2%，但 `bus.jpg` 上退化成**半辆车**——
这份导出权重是按拉伸验证的，不能换。用 `bus.jpg` + `bus` 自检可得到 0.91 分的紧致整框，
说明**管线本身没问题**，剩下的是模型能力与该导出权重的固有行为。

### v1.24 — 2026-11-02（提示式预标注：点一下 / 框一下 / 说一句话，外加审核状态）

v1.23 把标注变成"跑一遍 → 人只审核"，但那个"跑"有个前提：**已经训好一个 YOLO 引擎**。
新项目开工时没有引擎，第一版数据只能从零手工画；而有引擎的场景又被另一个问题拖着：
审到第 300 张去吃饭，回来不记得自己审到哪了。

这一版补的就是这两头：**没有引擎也能出草稿**（点框 / 文本两条免训练通道），
以及**审核到哪了一张看得见**（`K` 标已审核并跳下一张）。

| 项目 | 老版本（v1.23） | 新版本（v1.24） | 影响 |
| --- | --- | --- | --- |
| 预标注通道 | 只有自训 `.engine` | 引擎 / 点框（SAM）/ 文本（GroundingDINO）三选一 | 新项目没有引擎也能先出草稿，不用从空白画起 |
| 点框提示 | 无内置实现（只能接外部 HTTP 服务） | 画布上左键正点、`Shift+左键` 负点、拖动出框，`Enter` 出多边形 | 不规则目标能把边界贴着轮廓切出来 |
| 文本提示 | 无 | 填 `missing hole` 直接出框，可再细化成多边形 | 开放词表，不训也能检；类别名直接来自解出的短语 |
| 权重获取 | 用户自己找 GitHub Release | 标注页「模型仓库」：清单 + 下载 + 续传 + 进度 + 取消 | 首次使用少一次"文件放哪、要不要改名" |
| 置信度 | 引擎通道有分数，分割通道没有 | 点框草稿**固定灰档 `?`**，文本通道用真实 conf 进绿/黄/红 | 不把"点哪切哪"伪装成"机器很确定" |
| 审核状态 | 只有 `A`/`D`/空格，不知道看到哪 | `K` 标已审核并跳下一张未审核、`Shift+K` 取消、列表 `✓`/`○`、`已审核 12/64` | 中断后能接着审，不用靠记忆 |
| 草稿文件 | v1：只有 `images` | v2：多一个 `review` 块，v1 自动兼容 | 审核与草稿同一次原子写盘 |
| 批量任务 | `engine` 一种模式 | `engine` / `text`，text 模式启动前先验模型 | 不会建一个"跑 300 张全失败"的空任务 |
| 快捷键 | `K` = 关键点组 | `K` = 审核，关键点组改 `G` | **破坏性变更**，见下方 |

**三条通道怎么选**

| 通道 | 依赖 | 出什么 | 什么时候用 |
| --- | --- | --- | --- |
| 自有引擎 | 自己训的 `.transd.onnx` 编出的 `.engine` | 矩形 / 旋转框 / 多边形 / 关键点 | 已有同类缺陷模型，批量最快 |
| 点框提示（SAM） | MobileSAM，encoder 26.9 MB + decoder 15.7 MB | 多边形 | 目标不规则、边界要贴轮廓；标前几十张给训练用 |
| 文本提示（GroundingDINO） | SwinT int8 171 MB + BERT 分词表 0.44 MB | 矩形（可选细化成多边形） | 新缺陷类型、开放词表；一句话先看召回 |

三条通道产出的都是**草稿**，后面 `A` 接受 / `D` 丢弃 / `空格` 下一张 / `K` 已审核完全共用。
结果一律进旁挂 `<工程名>.drafts.json`，**`.tfanno` 一个字节都不改**（Qt 版因此仍双向兼容）。

**模型从哪来**

- 页面下载：标注页右栏「模型仓库」→ 点「下载」→ 进度条 → 可取消。落地目录 `6-web/models/`
  （已加入 `.gitignore`，权重不进版本库）。
- 换盘/换机器：设 `TRTFORGE_MODEL_DIR` 指向已下载的目录，重启服务生效。
- 被墙 / 内网：把下表文件名**原样**放进该目录，刷新仓库卡片即显示「已就绪」，
  不必走下载。文件名必须一致，改名会认不出来。

| 模型 id | 文件 | 体积（v1.24.1 实测） | 用途 |
| --- | --- | --- | --- |
| `mobile_sam` | `mobile_sam.encoder.onnx` + `sam_vit_h_4b8939.decoder.onnx` | 26.85 + 15.74 MB | 点框通道 |
| `grounding_dino_swint` | `groundingdino_swint_ogc_quant.onnx` | **171.28 MB** | 文本通道 |
| `bert_tokenizer` | `bert_base_uncased_tokenizer.json` | 0.44 MB | 文本通道必需（仓库内已有兜底副本） |

> **合计 214.3 MiB（约 215 MB，按 1024 计）。** v1.24 初版表里写的"40.1 MB / ≈700 MB"是手估的，按 700 MB 算
> 进度条只能走到 24%。现在 `modelzoo.CATALOG` 的 `size` 是对着服务端 `Content-Length`
> 逐个校准过的，它同时是进度条分母和完整性判据，**不要再手改**。

下载实现要点：64 KB 分块流式写 `.part`；已存在 `.part` 时带 `Range` 续传；每个文件按
**GitHub 直连 → `ghfast.top` 代理 → `gh-proxy.com` 代理**顺序尝试（v1.24.1 改，原来那三条
`hf-mirror.com/CVHub520/...` 实测全是 404）；分词表另配 `@local:` 兜底（仓库内有一份就直接拷，
不联网）；取消会删掉 `.part`（留着半个文件会让下次续传接在错位置）；同一模型并发下载返回 409。

**耗时（v1.24.1 真机实测，Windows 10 / CPU）**

测试图 `01_missing_hole_01.jpg`（3034×1586），`onnxruntime` 走 CPU EP：

| 操作 | 实测 | 备注 |
| --- | --- | --- |
| SAM image encode | **0.76 s**（首次，含 encode） | MobileSAM 是为 CPU 设计的轻量 encoder |
| SAM decode | **0.077 s** | encoder 缓存命中后只跑 decoder，差 10 倍 |
| SAM 框提示 | **0.079 s** | 同一 embedding，出 11 个多边形 |
| GroundingDINO 首次加载 | **4.80 s** | 171 MB int8，读盘 + 建 session，之后常驻 |
| GroundingDINO 单图推理 | **4.42 s** | 1200×800，CPU |
| 文本 + 细化成多边形 | **4.84 s** | 5 个框 → 8 个多边形（含一次 SAM encode） |

**已知限制（不是 bug，改之前先看）**

| 限制 | 实测依据 | 结论 |
| --- | --- | --- |
| 输入被硬拉伸到 1200×800，原图比例偏离 1.5 多时框偏大 | PCB 图 1.91 → `missing hole` 最大框占图 95.7% | 试过 letterbox 等比补边：PCB 上降到 75.2%，但 `bus.jpg` 上退化成**半辆车**。这份导出权重按拉伸验证，**不能改 letterbox** |
| 分数普遍偏低 | int8 量化，常见 0.3~0.7 | 默认 `box_thr=0.3` 照参考配置，别按 YOLO 的 0.5 直觉调 |
| 管线自检方法 | `bus.jpg` + `bus` → 0.91 分、紧致整框 | 出了怪框先用这张图验，能区分"模型能力不足"和"代码写错了" |

**为什么自己写代码、只用官方权重**

AnyLabeling 的推理内核只有 `onnxruntime + cv2 + numpy`，与 Qt 只耦合三个函数调用，
剥离比重写贵；但它的代码是 **GPL-3.0**，抄进仓库会把整仓传染。权重本身是 **Apache-2.0**，
与代码许可无关。所以本版**零外部源码复制**，只按 ONNX 输入输出契约重写了
预处理 / 提示展开 / 后处理三件事，其余仍是 `onnxruntime` + `cv2` + `numpy`。

**审核状态怎么存**

```json
{
  "version": 2,
  "images": { "01_missing_hole_01": [ ... ] },
  "review": { "01_missing_hole_01": { "state": "checked", "at_ns": "1762083000000000000" } }
}
```

- v1 文件（没有 `review`）直接读出空审核，不报错。
- 取消审核 = 从 `review` 里**删键**，不写 `"state":"unchecked"`，文件不会来回变长。
- `clear()` 默认**保留审核**：清草稿是"这批候选我不要了"，不等于"把审核进度忘掉"。
- 批量 `scope=pending` 现在也跳过**已审核**的图——用户确认过的结论不该被新候选换掉。

**破坏性变更**

| 键 | v1.23 | v1.24 |
| --- | --- | --- |
| `K` | 关键点组 | **标为已审核并跳下一张未审核** |
| `Shift+K` | 无 | 取消审核 |
| `G` | 无 | 关键点组 |

**验收**

- 新增 `test_prompt_label.py` **63/63**（轮廓近似 / 点数上限 400 / 退化掩码不抛异常 /
  encode 缓存命中 / 1024↔原图坐标往返 ≤1px / 提示点越界夹取 / **numpy 提示不崩** /
  **encoder 布局 HWC·NHWC·NCHW 三种**）。
- 新增 `test_text_label.py` **62/62**（tokenizer 编号与 special token / 短语还原 /
  `box_thr`·`text_thr` 边界 / 越界框夹取 / 退化框丢弃 / 模型缺失抛 `ModelMissing`）。
- 新增 `test_modelzoo.py` **53/53**（清单推导 / 目录环境变量覆盖 / `.part` 续传带 Range /
  取消删半成品 / 并发第二份被拒 / 接口层 400·404·409）。
- 新增 `test_onnxenv.py` **11/11**（v1.24.1：剔除 TensorRT / CUDA DLL 缺失即剔除 /
  真环境自检）。
- `test_annotation_drafts.py` 扩到 **87/87**（v1→v2 兼容 / review 增量合并 / 坏 review 跳过 /
  `clear` 保留审核 / 接口读写）。
- 回归 `run_all_tests.py`：**30 个脚本全绿**（约 47 秒）；前端 3 个 `.mjs`（23 / 17 / 37 条断言）
  与 13 个 `js/*.js` 的 `node --check` 全通过。
- 模型缺失路径实测：文本通道抛「需要先下载模型「grounding_dino_swint / bert_tokenizer」」，
  点框通道抛「缺少 mobile_sam.encoder.onnx, sam_vit_h_4b8939.decoder.onnx」并给出
  `TRTFORGE_MODEL_DIR` 提示——接口层统一转 400，不是 500。
- **模型到位路径实测（v1.24.1 补）**：见上方「耗时（v1.24.1 真机实测）」与
  `6-web/README.md` 的「v1.24.1 变更点」。v1.24 初版那句"权重未下载、无真图数字"已失效。

细节见 `6-web/README.md` 的「三条预标注通道」「点框提示」「文本提示」「模型仓库」「审核状态」
五节与 `docs/WORKFLOW_GUIDE.md` 的 §2.9。

### v1.23 — 2026-11-02（标注预标注变成"跑一遍、只审核"）

以前自动预标注是**一张一张点**：600 张的工程就要点 600 次，而且草稿只活在前端内存里，
刷新页面全部蒸发——所以批量这件事根本没法做。

这一版把预标注做成流水线：**先整工程跑一遍，结果全进草稿池，人只按 `A`/`D` 过一遍**。

| 项目 | 老版本（v1.22） | 新版本（v1.23） | 影响 |
| --- | --- | --- | --- |
| 预标注粒度 | 单张触发 | 「批量预标注」整工程跑一遍，带进度条与停止 | 600 次点击变成 1 次，长跑能中断 |
| 草稿存储 | 前端内存，刷新即丢 | 旁挂 `<工程名>.drafts.json`，按图增量**原子**落盘 | 刷新 / 重启 / 服务重连都不丢 |
| 工程文件 | —— | 草稿**永不**写进 `.tfanno` | Qt 版打开同一工程看不到草稿，双向兼容不变 |
| 引擎选择 | 手工翻 `runs/` 目录填路径 | 下拉直接列训练产物（`/api/models`），未转置标 `⚠` | 选错引擎（拿没转置的 ONNX 编的）在选的时候就看得见 |
| 审核 | 鼠标点「接受 / 拒绝」 | `A` 接受 / `D` 丢弃 / `空格` 下一张，按钮仍在 | 手不离开键盘，一遍过完 |
| 置信度 | 草稿统一颜色 | 高绿 / 中黄 / 低红 / 未知灰，画布与列表同色 + 徽标 | 低分框一眼挑出来重点看 |
| 断点续跑 | —— | 「跳过已标注/已有草稿」 | 补标新图不用重跑全工程 |
| 失败处理 | 单张失败只提示一次 | 单张失败记进 `failures[]` 继续下一张，整批不中断 | 一张坏图不会让 600 张白跑 |

**为什么草稿不塞进 `.tfanno` 的独立字段**

`.tfanno` 是 Qt 版也在读写的文件，Qt 保存时按自己的结构整体重写，会把不认识的键丢掉。
草稿写进去等于"Qt 一点保存就丢"。所以草稿放**旁挂文件**，`.tfanno` 一个字节都不改。

**为什么批量用线程不用子进程**

推理页那条「跑任务」是子进程 + stdout，适合跑几分钟的训练。批量预标注是"同一进程反复调
同一个引擎"：每张图起一个进程要重挂 DLL、反序列化引擎，600 张光启动就十几分钟。
`infer.run_yolo()` 本来就按引擎缓存句柄并串行化，所以用守护线程 + 内存任务表 + 轮询。
代价是服务重启会丢**进度**（内存里），但**草稿不会**——每张图跑完立刻落盘。

**验收**

- 新增 `test_annotation_drafts.py` **52/52**（往返一致、按图增量合并、坏条目跳过、
  原子写、并发写不互相清空、非法 JSON 不报 500）。
- 新增 `test_autolabel_batch.py` **42/42**（生命周期、`scope=pending` 跳过、
  单张失败不中断、停止、空清单、任务表上限）。
- 回归 `run_all_tests.py`：**26 个脚本全绿**（约 40 秒）。
- 真跑：`pcb/detect_v1/weights/best.transd_fp16.engine` 跑 6 张图 → `6/6 张，共 18 个草稿`，
  `elapsed_ms≈525`；草稿 `score` 0.75~0.88 全部落在「高」档。
- 浏览器实测：`A` 后草稿 4→3、正式标注 12→13 且自动选中下一条；`D` 后 4→3；`空格` 跳到下一张；
  刷新页面重开工程，13 条正式标注 + 2 条剩余草稿原样回来。
- 分档着色用画布像素取样核对：高 `rgb(52,211,153)` / 中 `rgb(251,191,36)` / 低 `rgb(248,113,113)`。

细节见 `6-web/README.md` 的「批量预标注」「草稿旁挂落盘」「审核效率」「训练产物下拉」四节。

### v1.22 — 2026-11-02（异常检测能说清「缺陷找没找准」了）

以前的异常检测精度只回答一个问题：**这张图有没有缺陷**（图像级 AUROC）。
但产线上更常见的失败是另一种：**判对了却说不出缺陷在哪**——分数报了 NG，热能图一片平坦，
或者好板上到处点红点。这种问题在报告里一个字都看不到，因为 `pixel` 块是写死的常量。

这一版不解掩码，用引擎本来就返回的归一化异常图算出三项诊断，让"能判不能定位"暴露出来。

| 项目 | 老版本（v1.21） | 新版本（v1.22） | 影响 |
| --- | --- | --- | --- |
| `pixel` 块 | 常量 `{"available": false, "reason": "本版只做图像级"}` | 按真实异常图计算，`mode="diagnostic"` | 报告第一次有了像素信息 |
| 好图乱指 | 无 | `normal_false_pixel_rate`：正常图上被判缺陷的像素占比 | 好板被点红多少，一眼可见 |
| 定位面积 | 无 | `abnormal_marked_rate`：异常图上超阈值像素占比 | 区分「小而集中」与「糊成一片」 |
| 指不出位置 | 无 | `abnormal_undirected`：判 NG 但异常图无一个像素达标 | 分数与位置自相矛盾，硬伤直接暴露 |
| 引擎搬运 | `run_anomalib_score` 只搬 3 个标量 | 新增 `want_map=True`，拷 `normalized_map` 副本 | 评估链路仍然不编码 PNG，8 张图 1 秒 |
| CSV | 6 列 | 多一列 `pixel_above_ratio` | 能定位到具体哪张图乱指 |
| 界面 | 10 个指标块 | 多 3 行，「指不出位置」> 0 时整行标红 | 不用翻 JSON 也能看见硬伤 |

**判定线是 0.5，不是 `pixel_threshold`**

DLL 在返回异常图之前已经做过一次归一化：

    normalizedMap = ((rawMap - pixel_threshold) / (mapMax - mapMin)) + 0.5

所以「恰好等于 `pixel_threshold` 的像素」在返回的图上就是 `0.5`，DLL 自己生成缺陷掩码时
也是 `threshold(displayMap, mask, 0.5, ...)`。照字面拿 `pixel_threshold`（例如 `0.172`）去比，
会把绝大多数像素算成异常，报告立刻变成"好板上 40% 是缺陷"——和图像级那个
`normalized >= 0.5` 的坑是同一个，只是换到了像素上。

**为什么不顺便算像素级 AUROC / PRO / AUPRO**

三者的分母都需要 ground_truth 掩码，而当前数据集只有 `good` / `bad` 两个目录
（`Folder` 数据模块没配掩码目录）。拿整图标签冒充像素标签能算出一个数，
但它在惩罚模型"把好区域判正常"，比不给数字更有害。所以 `pixel.auroc` 明确为 `null`，
`pixel.auroc_reason` 写明原因。掩码目录的两种约定（`--mask-dir` 优先、
其次 MVTec 风格 `ground_truth/<类别>/*_mask.png`）已记录，留给下一版。

**验收**

- 新增 `test_anomaly_pixel.py` **36/36**：判定线边界（`0.499` / `0.5` / `0.501`）、
  「0.2 的图不会被当成整幅异常」、占比计算、`None` / 空图 / 一维 / 三维输入、
  「没有样本」与「全是 0」必须区分、尺寸不统一要说出来、无掩码时 `auroc` 必须为 `null`。
- 回归 `run_all_tests.py`：**24 个脚本全绿**。
- 真跑：bottle 数据集（Patchcore / fp16，正常 4 张 + 异常 4 张）→ AUROC `1.0000`、
  好图乱指 `0.000%`、定位面积 `16.20%`、指不出位置 `0`，异常图 `224×224`，
  产物同时写出 `anomaly_val_report.json` 与带 `pixel_above_ratio` 列的 CSV。
- 浏览器实测：卡片多出「好图乱指像素 / 异常图定位面积 / 指不出位置」三行；
  把 `abnormal_undirected` 改成 3 后该行变成 `stat is-warn`（红底红字）；
  旧报告（`available=false`）仍只显示一行原因，不会留空数字。

细节见 `6-web/README.md` 的「像素级诊断（v1.22）」与 `docs/WORKFLOW_GUIDE.md` 的 §7.2.1。

### v1.21 — 2026-11-02（推理页把 metadata 的推荐阈值摆到台面上）

勾了「沿用 metadata 默认阈值」之后，两个阈值输入框只是被置灰，框里留着默认 `0.5`。
用户想知道"到底会用多少"只有一条路：先跑一次推理，再去看结果行里那串
`（metadata）`——推荐值被藏在了文件里。

这一版把 `metadata.json` 的内容提前读到界面上，只读文件、**不建引擎句柄**，
所以改一次路径就刷新一次，不花几秒加载模型。

| 项目 | 老版本（v1.20） | 新版本（v1.21） | 影响 |
| --- | --- | --- | --- |
| 推荐阈值可见性 | 输入框置灰，显示默认 `0.5` | 输入框被填成文件里的真实值（如 `0.172` / `0.1634`） | 不跑推理也知道会用多少 |
| 原始精度 | 卡片上只有两个数 | 卡片刻度 4 位小数，另标注模型类型与输入尺寸 | 校准过的阈值（如 `0.17198532`）不会被糊成 `0.17` |
| 可信度提示 | 无 | 有 `calibration` 区块时标「已校准 N%」，否则标「文件原值」 | 一眼分清是现场标定值还是训练集默认值 |
| 读取失败 | 静默（框里仍留着旧数字） | 卡片标「读取失败」、数值清空、hint 写明原因 | 不会拿上一次的数字冒充本次推荐值 |
| 覆盖态 | 勾掉后用界面值，但看不到原值 | hint 写「手动覆盖中：metadata 原值 图像 … ／ 像素 …」 | 过几天回来也知道自己改没改 |
| 手动精度 | 滑块步长 `0.01` | 步长 `0.0001` | 想复现 `0.172` 这种阈值不用改源码 |
| 接口 | 无 | `GET /api/infer/anomalib/metadata?path=…` | 其它页面（批量/串联）也能复用 |

**接口返回**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `path` | str | 实际读取的文件绝对路径 |
| `image_threshold` | float \| null | 图像级判定阈值（0~1，`null` = 文件里没有） |
| `pixel_threshold` | float \| null | 像素级异常区域阈值 |
| `model_type` | str | `Patchcore` / `Padim` / `EfficientAd`，缺字段时为空串 |
| `input_size` | [int, int] \| null | ONNX 的 H/W |
| `normalization` | str | 归一化方式（`none` 等） |
| `pred_scores` / `anomaly_maps` | [float, float] \| null | 分数与异常图的 `min` / `max` |
| `sample_accuracy` | float \| null | 校准时的样本判对率，仅 `calibration` 区块存在时有 |
| `error` | str | 解析失败原因（接口在失败时返回 400 + 这句原文） |

**验收**

- 新增 `test_metadata_readout.py` **37/37**：字段解析、BOM、坏 JSON、`0.0` 阈值不算"没读到"、
  非白名单 403、文件不存在 400、以及"该接口绝不建立引擎句柄"。
- 回归 `run_all_tests.py`：**23 个脚本全绿**（约 32 秒）。
- 浏览器实测：选中 Patchcore 引擎 → 卡片显示 `图像 0.1720 ／ 像素 0.1634`、`已校准 100%`、
  `Patchcore / 256×256`，两个只读框同步变成 `0.172` / `0.1634`；勾掉勾选框后 hint 变为
  「手动覆盖中：… metadata 原值 图像 0.1720 ／ 像素 0.1634」；把路径改成不存在的文件后
  卡片变「读取失败」、框被清空。

细节见 `6-web/README.md` 的「推理页：metadata 推荐阈值回显（v1.21）」与
`docs/WORKFLOW_GUIDE.md` 的 §7.1.2。

**配套的规范收尾（同一批完成）**

- **补回漏落的能力**：`openspec/specs/web-yolo-pipeline/` 此前不存在——v1.17 的 YOLO 串联
  5 个 Requirement、20+ 个场景只活在归档目录里（`openspec show web-yolo-pipeline` 直接报
  `Spec not found`）。现已落到正式 specs，`--specs` 从 13 个变成 **14 个**。
- **12 个 spec 的 `## Purpose` 不再是 `TBD`**：归档时漏写的"Update Purpose after archive"
  已全部补成真实描述，现在 14 个能力都有一句话说明"这个能力解决什么问题"。
- **本次改动有提案可查**：新增 `openspec/changes/update-infer-metadata-readout` →
  已归档为 `2026-11-02-update-infer-metadata-readout`，给 `web-infer-console` 追加了
  `Metadata Threshold Inspection` 要求，并把这个 UI 要求的 4 个新场景并入。
- **CI 注释修正**：`.github/workflows/tests.yml` 里「12 个脚本 / 444 项断言」更新为
  「23 个脚本 / 891 项断言」，并写明新增脚本会被 `run_all_tests.py` 自动收录。
- 校验：`openspec validate --all --strict` **14/14 通过**，`openspec list` 无活跃变更。

### v1.20 — 2026-11-02（路径选择：直接弹 Windows 系统对话框）

之前所有要填路径的输入框只有两条路：手打绝对路径，或用页面内那个「列目录」弹窗一层层钻。
文件类型还是前端写死的一串正则，`.pth` 之类干脆看不见。根因是**浏览器安全沙箱：
任何网页都不能直接调系统文件框**——所以这事必须绕到后端去做。

这一版让后端在**跑服务这台机器的桌面上**弹真正的 Windows 对话框，页面按钮点下去就是
系统框，能选任意盘符、按字段过滤、记住上次的目录。

| 项目 | 老版本（v1.19） | 新版本（v1.20） | 影响 |
| --- | --- | --- | --- |
| 选路径 | 只能页面内列目录 | 原生文件框 / 文件夹框 / 另存为框 | 选 `D:\数据集`、`\\NAS\共享` 不用再手打 |
| 文件过滤 | 前端硬编码正则 | 每个字段自带扩展名清单，后端生成过滤器（带「所有文件」兜底） | 少一个「为什么看不到我的文件」类问题 |
| 起始目录 | 每次从仓库根开始钻 | 按字段记忆上次目录（`runtime/dialog_state.json`） | 第二次起一步到位 |
| 远程访问 | 无差别 | 仅本机 `127.0.0.1` / `::1` 才弹，远程自动回退页面内浏览 | 不会出现「点了没反应」的鬼按钮 |
| 仓库外路径 | 选完就 403 | 选中的目录自动进**运行期放行表**（只那一级、不递归、不落盘、重启失效） | 不用为了选个数据集去改环境变量重启 |
| 并发 | — | 全局单例锁，已有框时 409 | 不会叠出两个框分不清归属 |
| 标注加图 | 只能扫整个目录 | 新增「选图片…」多选文件直接加入；另存为走原生保存框 | 挑几张图不用先归拢到一个目录 |
| 漏掉的入口 | `yoloModel` / `pythonPath` 没有浏览按钮 | 补齐；`dsSource` / `batchSource` / `annoImportSource` 拆成「文件」「目录」两个按钮 | 22 个字段全部可选 |

**交互约定（一个键就够）**

| 操作 | 结果 |
| --- | --- |
| 单击「浏览 / 目录 / 文件 / …」 | 弹系统对话框 |
| `Shift` + 单击 | 页面内目录浏览（远程、无桌面会话、或想手动切盘符） |
| 取消 | 字段保持原值，不报错、不误开页面内浏览 |

**降级**：`/api/capabilities` 新增 `native_dialog` 字段。为 `false`（无 `tkinter`、非 Windows、
或设了 `TRTFORGE_DIALOG_DISABLE=1`）时，前端直接走页面内浏览，不显示点不动的按钮。

**验收（全部实测）**

- 新增 `test_native_dialog.py` **40/40**：扩展名归一化（`.pt` / `pt` / `*.pt`）、起始目录取父目录、
  四种类型映射、单例锁、超时、子进程异常、取消不写记忆、临时目录清理、
  白名单放行与清空、接口层远程 403 / 本机放行。
- 回归 `run_all_tests.py`：**22 个脚本全绿**（约 33 秒）。
- 浏览器实测（真实 Windows 对话框）：点「dataset.yaml 浏览」→ 桌面弹框 → 输入完整路径回车 →
  字段写入 `…\PCB data\PCB缺陷检测\data.yaml`；按 Esc 取消后字段不变、按钮恢复、
  未误开页面内浏览；`Shift` + 点击打开页面内浏览；标注页「新建…」弹的是保存框。

细节见 `6-web/README.md` 的「选路径：原生对话框（v1.20）」与
`docs/WORKFLOW_GUIDE.md` 的 §7.1.1。v1.19（YOLO 标签导入 / 另存为）此前只在
`6-web/README.md` 记录，未在本文件单列。

### v1.18 — 2026-11-02（Anomalib 异常检测一键全流程）

v1.17 的串联只覆盖 YOLO。异常检测（Patchcore / PaDiM / EfficientAd）仍然是
「训练页跑 launcher → 去目录里翻 `vN` 版本号 → 手动 trtexec → 手写脚本比对分数」，
而且**没有任何精度量化**——阈值是 0.5 还是校准值、漏检多少张，全靠肉眼看几张热力图。

这一版把 Anomalib 从训练到引擎精度报告压成一条链，并首次给出图像级 AUROC /
漏检率 / 误报率这类可以直接写进验收单的数字。

| 项目 | 老版本（v1.17） | 新版本（v1.18） | 影响 |
| --- | --- | --- | --- |
| 覆盖模型 | 只有 YOLO（detect/segment/pose/obb） | 新增 Anomalib `patchcore` / `padim` / `efficientad` | 异常检测也能一次点击跑完 |
| 串联形态 | 六段（训练、导出分开） | 四段：`train_export`（训练+导出+校准）→ `probe` → `build` → `val` | 少两次模型加载，导出的一定是刚训的模型 |
| 产物定位 | 路径在训练前就能拼出来 | `StagePlan.resolve` 在阶段执行前按 `record["artifacts"]` 延迟派生命令 | `results/<类名>/<name>/vN` 的版本号训练完才知道，不再猜 `v0` |
| 动态 batch | probe 把 `-1` 判为非法形状 → 编译段直接失败 | 自动补 `--minShapes/--optShapes/--maxShapes`（opt=1、max=`max_batch`） | Anomalib 导出的 ONNX 本来就带动态 batch，现在能正常编译 |
| 精度评估 | 无（YOLO 侧才有 mAP） | `tools/val_anomaly.py` 走 `anomalib_infer.dll`，出 AUROC / F1 / 漏检 / 误报 / 最优阈值 | 引擎好坏有数字，且与 DLL 判定口径严格一致 |

**四段链**

```
train_export（训练 + ONNX 导出 + metadata 阈值校准）
  → probe（校验 metadata 字段 + ONNX 形状，几十毫秒拦掉编译白跑）
  → build（trtexec 编译 <model>_<precision>.engine）
  → val（引擎逐张推理，出 anomaly_val_report.json / .csv）
```

**参数（`POST /api/pipeline/anomalib`，前端在串联卡片选「异常检测」后填）**

| 参数 | 类型 | 默认 | 说明 | 应用场景 |
| --- | --- | --- | --- | --- |
| `model` | str | `patchcore` | `patchcore` / `padim` / `efficientad` | 小样本缺陷用 patchcore；快速验证用 padim；算力敏感用 efficientad |
| `root` | str | — | Folder 数据集根目录 | 需含 `train/good`、`test/abnormal`、`test/good` |
| `input_size` | int | 256 | 训练与导出的 H/W | 与 ONNX 的 H/W 必须一致，probe 会拦 |
| `precision` | str | `fp16` | 引擎精度 | 验收不达标时切 `fp32` 看上限 |
| `max_batch` | int | 8 | 动态 batch 的 `--maxShapes` 上限 | 以后要批量推时调大；只推单张时可设 1（编译更快） |
| `skip_train` | bool | false | 跳过训练 | 反复调编译参数、复用老模型 |
| `artifacts_dir` | str | 空 | 含 `metadata.json` 与 `.onnx` 的目录 | `skip_train=true` 时必填，填 `.../weights/lightning` |
| `run_val` | bool | true | 是否做引擎精度验证 | 只要引擎时可关 |
| `val_limit` | int | 0 | 每个目录最多评多少张，0=全部 | 冒烟时填 20 |
| `calibrate_metadata` | bool | true | 用 ONNX 推理校准 `image_threshold` | 不校准会退回 0.5，异常图基本判不出来 |

**指标口径（`6-web/backend/anomalymetrics.py`）**

| 指标 | 算法 | 为什么这样算 |
| --- | --- | --- |
| image AUROC | Mann-Whitney U 秩统计，同分取平均秩 | 与阈值无关，衡量排序能力；平均秩避免「分数全相同」时虚高 |
| TP/FP/TN/FN、P/R/F1 | `metadata.json` 的 `image_threshold` | DLL 用 `normalized = clamp01((raw-min)/(max-min))`、`>=0.5` 判异常，与 `raw >= threshold` **等价** |
| FPR（误报率） | `FP/(FP+TN)` | 产线上「好产品被误判」的比例 |
| FNR（漏检率） | `FN/(FN+TP)` | 产线上「坏产品被放过」的比例，通常比 FPR 更要紧 |
| best F1 阈值 | 扫全部唯一分数 | 只作参考（会过拟合测试集），**不回写 metadata** |
| decision_mismatch | 报告判定 vs 引擎 `is_anomaly` 不一致张数 | 非 0 说明归一化口径有偏差，必须排查 |

**像素级暂不提供**：需要 ground_truth 掩码，PRO/AUPRO 成本远高于图像级；
报告里 `pixel.available=false` 并写明原因，而不是给个空白数字。

**验收（全部实测）**

- 新增 `test_pipeline_resolve.py` **26/26**：延迟派生命令、派生失败短路、多产物去重、
  `vN` 版本定位、metadata 缺阈值拦截、动态 batch 形状参数、H/W 动态拒绝。
- 新增 `test_anomaly_metrics.py` **31/31**：归一化边界、AUROC 同分、空输入、
  混淆矩阵阈值边界（恰好等于阈值算异常）、直方图 AUROC。
- 回归 `run_all_tests.py`：**20 个脚本全绿**（约 28 秒）+ 前端 `test_contrast.mjs`、`node --check`。
- 真链路（bottle 数据集，Patchcore / 256 / fp16 / 64 张）：
  `train_export` **199.7 秒** → `probe` **0.7 秒** → `build` **1016 秒** → `val` **4.0 秒**，
  整链 success；精度卡片 image AUROC **1.000**、F1 **1.000**、FPR **0.000**、FNR **0.000**
  （20 张正常 / 44 张异常，`decision_mismatch=0`），产物
  `model_fp16.engine` + `anomaly_val_report.json` / `.csv`。
- 首轮暴露的真实缺陷：probe 把 ONNX 的动态 batch（`[-1,3,256,256]`）判成非法形状，
  训练 199 秒后在第二段失败。修复后补了 4 条单测（含「H/W 动态仍拒绝」），
  并用同一份产物 `skip_train` 重跑 `probe → build → val` 验证通过。

细节见 `6-web/README.md` 的「异常检测一键全流程（v1.18）」与
`docs/WORKFLOW_GUIDE.md` 的 §5.6。

### v1.17 — 2026-11-02（一键全流程串联 + 引擎精度验证）

v1.16 之前的链路虽然齐了，但「跑一遍」要人守着换五个页面：训练页点开始 → 等完 →
导出页选权重 → 引擎页选 ONNX → 推理页填标签 → 最后还得自己判断精度够不够。
中间任何一步忘了衔接，就得重新翻路径。这一版把这段路压成**一次点击**，
并补上了此前完全没有的**引擎精度量化**（以前的「精度」全靠肉眼看几张图）。

| 项目 | 老版本（v1.16） | 新版本（v1.17） | 影响 |
| --- | --- | --- | --- |
| 全流程 | 五个页面手工接力，人守着 | 训练页底部「全流程串联」卡片：训练→导出→校验→编译→补标签→精度验证 | 一次点击跑到精度报告 |
| 参数 | 每页重填一遍 | 直接复用「训练参数」表单，卡片只多 4 项（精度/split/是否验证/跳过训练） | 不重复填、不会填串 |
| 失败处理 | 无（靠人看日志） | 某段失败整链 `failed`，后续 `skipped`，已完成产物全部保留；不自动重试 | 不用从第一步重来 |
| 停止 | — | 只停当前阶段那一个 run（Windows 上杀父进程不会连带杀子进程） | 停了就是真的停了，不再偷偷占着 GPU |
| 阶段日志 | — | 还在跑 → 实时流；已结束/重启过 → 读 `runtime/logs/{run_id}.log` 尾部 | 重启后仍能查「那一段到底报了什么」 |
| 精度 | 只能看几张图「感觉还行」 | `val_engine.py` 走 DLL 逐张推理，COCO 口径算 mAP50 / mAP50-95 / P / R | 引擎好坏有数字，可写进验收单 |
| 重启后 | 记录丢失，页面可能永远转圈 | `runtime/pipelines/*.json` 持久化；启动时把遗留 `running` 标成「服务重启，任务已中断」 | 历史可查，不会假死 |

**四个设计决策，改之前先看这里**

1. **串联本身不 spawn 子进程，只决定「下一段跑什么」。**
   如果做成一个大进程组，在 Windows 上点「停止」只能杀父进程，训练/trtexec 会变成孤儿继续跑
   （照样占显存）。现在每段都是普通 run，停止 = 对当前那一个 `terminate()`，
   而已经完成的段产物原样留在磁盘上。
2. **元数据类步骤用可执行函数（callable），不用子进程。**
   「校验 `.transd.onnx` 的转置布局」「写 `labels_*.json`」这类操作是读一个文件、写一个文件，
   起子进程要多花 1~2 秒还多一份日志噪音，直接在编排线程里同步做完更合适。
   训练 / 导出 / 编译 / 验证这种分钟级、要日志流的才走 run。
3. **失败不自动重试。**
   训练 OOM、数据集路径写错这类问题重试一次只是再烧一遍 GPU 时间，
   而且会把「哪里出错了」淹没在第二次日志里。失败就停，把原因和已完成的产物摆出来。
4. **重启后必须清理遗留的 `running`。**
   正常退出会走 `shutdown_all()`，但断电/任务管理器强杀不会。这些记录留在磁盘上时，
   前端按 `status` 轮询会永远转圈、点「停止」只能拿到 409。所以启动时统一标成
   `cancelled`（原因写清楚），历史仍然可查。

**验收（全部实测）**

- 新增 `test_pipeline_stages.py` **50/50**：三段全成功、中途失败短路、callable 段、
  进行中停止、报告收集与缺失容错、快照隔离、重启遗留清理、产物去重（含 callable 返回值）、
  `expect_file` 必须是 `Path`。
- 新增 `test_pipeline_log.py` **9/9**：日志尾部截断、超长行、`lines` 越界夹取、
  未知阶段 / 无 `run_id` / 路径穿越 / 不存在的链 全部 404。
- 新增 `test_val_metrics.py` **37/37**：IoU 边界（0.50 恰好命中）、完美预测、多类 AP、
  P-R 曲线、无 GT 容错。
- 回归 `run_all_tests.py`：**18 个脚本全绿**，约 29 秒。
- 真链路（后端 1.17.0 + PCB `pcb_best.pt`）：跳过训练跑 `export → probe → build → labels → val`，
  整链 **1 分 24 秒 success**（fp32）/ **6 分 53 秒 success**（fp16，编译段 6 分 47 秒），
  产出 `best.transd.onnx` / `pcb_best_fp32.engine` / `labels_pipe_smoke2.json` /
  `val/val_report.json`，精度卡片显示
  mAP50 **0.125**、mAP50-95 **0.090**、P **0.500**、R **0.250**（2 张 val、12 个 GT）。
- 浏览器实测：历史记录「查看」能还原阶段/产物/精度；「日志」按钮读出 210 行 trtexec 日志；
  服务重启后遗留记录显示 `已停止` + 「服务重启，任务已中断」，「停止」按钮置灰、轮询停止。

细节见 `6-web/README.md` 的「训练完成后一键跑完全流程（v1.17）」与
`docs/WORKFLOW_GUIDE.md` 的 §5.5。

### v1.17 — 2026-11-02（一键全流程串联 + 引擎精度验证）

v1.16 之前的链路虽然齐了，但「跑一遍」要人守着换五个页面：训练页点开始 → 等完 →
导出页选权重 → 引擎页选 ONNX → 推理页填标签 → 最后还得自己判断精度够不够。
中间任何一步忘了衔接，就得重新翻路径。这一版把这段路压成**一次点击**，
并补上了此前完全没有的**引擎精度量化**（以前的「精度」全靠肉眼看几张图）。

| 项目 | 老版本（v1.16） | 新版本（v1.17） | 影响 |
| --- | --- | --- | --- |
| 全流程 | 五个页面手工接力，人守着 | 训练页底部「全流程串联」卡片：训练→导出→校验→编译→补标签→精度验证 | 一次点击跑到精度报告 |
| 参数 | 每页重填一遍 | 直接复用「训练参数」表单，卡片只多 4 项（精度/split/是否验证/跳过训练） | 不重复填、不会填串 |
| 失败处理 | 无（靠人看日志） | 某段失败整链 `failed`，后续 `skipped`，已完成产物全部保留；不自动重试 | 不用从第一步重来 |
| 停止 | — | 只停当前阶段那一个 run（Windows 上杀父进程不会连带杀子进程） | 停了就是真的停了，不再偷偷占着 GPU |
| 精度 | 只能看几张图「感觉还行」 | `val_engine.py` 走 DLL 逐张推理，COCO 口径算 mAP50 / mAP50-95 / P / R | 引擎好坏有数字，可写进验收单 |
| 重启后 | 记录丢失，页面可能永远转圈 | `runtime/pipelines/*.json` 持久化；启动时把遗留 `running` 标成「服务重启，任务已中断」 | 历史可查，不会假死 |

**四个设计决策，改之前先看这里**

1. **串联本身不 spawn 子进程，只决定「下一段跑什么」。**
   如果做成一个大进程组，在 Windows 上点「停止」只能杀父进程，训练/trtexec 会变成孤儿继续跑
   （照样占显存）。现在每段都是普通 run，停止 = 对当前那一个 `terminate()`，
   而已经完成的段产物原样留在磁盘上。
2. **元数据类步骤用可执行函数（callable），不用子进程。**
   「校验 `.transd.onnx` 的转置布局」「写 `labels_*.json`」这类操作是读一个文件、写一个文件，
   起子进程要多花 1~2 秒还多一份日志噪音，直接在编排线程里同步做完更合适。
   训练 / 导出 / 编译 / 验证这种分钟级、要日志流的才走 run。
3. **失败不自动重试。**
   训练 OOM、数据集路径写错这类问题重试一次只是再烧一遍 GPU 时间，
   而且会把「哪里出错了」淹没在第二次日志里。失败就停，把原因和已完成的产物摆出来。
4. **重启后必须清理遗留的 `running`。**
   正常退出会走 `shutdown_all()`，但断电/任务管理器强杀不会。这些记录留在磁盘上时，
   前端按 `status` 轮询会永远转圈、点「停止」只能拿到 409。所以启动时统一标成
   `cancelled`（原因写清楚），历史仍然可查。

**验收（全部实测）**

- 新增 `test_pipeline_stages.py` **49/49**：三段全成功、中途失败短路、callable 段、
  进行中停止、报告收集与缺失容错、快照隔离、重启遗留清理、产物去重、`expect_file` 必须是 `Path`。
- 新增 `test_val_metrics.py` **37/37**：IoU 边界（0.50 恰好命中）、完美预测、多类 AP、
  P-R 曲线、无 GT 容错。
- 回归 `run_all_tests.py`：**17 个脚本全绿**，约 29 秒。
- 真链路（后端 1.17.0 + PCB `pcb_best.pt`）：跳过训练跑 `export → probe → build → labels → val`，
  整链 **1 分 24 秒 success**，产出 `best.transd.onnx` / `pcb_best_fp32.engine` /
  `labels_pipe_smoke2.json` / `val/val_report.json`，精度卡片显示
  mAP50 **0.125**、mAP50-95 **0.090**、P **0.500**、R **0.250**（2 张 val、12 个 GT）。

细节见 `6-web/README.md` 的「训练完成后一键跑完全流程（v1.17）」与
`docs/WORKFLOW_GUIDE.md` 的 §5.5。

### v1.16 — 2026-11-02（目录批量、视频、摄像头三种推理形态）

v1.8 到 v1.15 把「训练 → 导出 → 编译 → 单张推理」接成了闭环，但闭环的出口只有**一张图**。
真实产线的三个问题它都答不了：一个文件夹里的 500 张怎么跑、一段录像怎么跑、
现场相机的实时画面怎么看。这一版补的就是出口。

| 项目 | 老版本（v1.15） | 新版本（v1.16） | 影响 |
| --- | --- | --- | --- |
| 输入形态 | 单张图片 | 单张 / 目录 / 视频 / 摄像头实时 | 不用再写脚本包一层 |
| 执行方式 | 同步单次 HTTP | 目录与视频走**子进程任务**（可停止、有进度、有日志），摄像头走**单帧同步接口** | 长任务不卡页面，实时画面不排队 |
| 批量产物 | 无 | `annotated/` + `results.csv`(utf-8-sig) + `results.json` + `annotated.zip` + `job.json` | 一次跑完就能交付 |
| 坏图 | 整个请求 500 | 单行 `status=error` 记原因，整批继续 | 一张坏图不再让 500 张白跑 |
| 视频编码 | 无 | `avc1` → `H264` → `mp4v` 逐级探测（实写一帧验证） | 浏览器能直接内联播放 |
| 抽帧 | 无 | 支持步长；输出帧率写 `源帧率 ÷ 步长` | 抽帧不会变成"快进" |
| 摄像头 | 无 | 浏览器采集 + 每帧只回 JSON 框，前端 canvas 叠框 | 帧不出本机，不传 base64 大图 |
| Anomalib 批量 | 无 | 支持目录批量（视频 / 摄像头仍限 YOLO） | 异常检测也能出报表 |

**三个设计决策，改之前先看这里**

1. **为什么批量/视频走子进程，摄像头走 HTTP。**
   批量与视频是长任务，需要"随时能停、关掉页面继续跑"，而这正是 `runners.py` 已经做好的事
   （`terminate` 即取消、事件经 WebSocket 广播、日志落盘）。摄像头的诉求相反：它是闭环实时的，
   每帧过任务队列必然延迟越堆越大，所以用"请求-应答"的单帧接口，一次推理的时间就是端到端延迟。
2. **摄像头接口不回标注图。**
   回 base64 图会让每帧带宽翻几十倍，还要在浏览器里解码整图，8~10 fps 会直接掉到个位数。
   接口只回框、多边形、关键点与类别色，画由前端做——这一步换来的是"能实时看"，而不只是"能看到"。
3. **编码器可用性必须实写一帧来验证。**
   没有 `libopenh264` 的机器上，`cv2.VideoWriter.isOpened()` 对 `avc1` 也会返回 `True`，
   写出的却是 0 字节文件。只查 `isOpened()` 会把"完全跑不通"伪装成成功，
   所以改成写一帧黑帧再看文件大小。

**验收（全部实测）**

- 新增 `test_infer_jobs.py` **34/34**：图片收集与排序、文件名净化（中文/括号/空格）、
  CSV 必须是 `utf-8-sig` + `\n`、zip 原子生成、`runtime/infer` 之外的路径必须拒绝。
- 新增 `test_infer_batch.py` **75/75**：命令拼装与开关透传、坏图容错、Anomalib 批量、
  单帧接口、产物下载。
- 新增 `test_infer_video.py` **43/43**：编码器探测（假 `isOpened` 场景）、抽帧后时长不变、
  帧数上限、失败路径。
- 回归 `run_all_tests.py`：**15 个脚本全绿**，约 27 秒。
- 真链路（后端 1.16.0 + PCB `best.transd_fp16.engine`）：目录批量 3 张 0.7 s 产出
  `annotated.zip`(713 KB)；视频 30 帧 0.62 s / 46.2 fps 产出可内联播放的 mp4
  （`readyState=4`、`duration=3`）；Anomalib 4 张 1.1 s 出 OK/NG 报表；
  摄像头单帧 327 ms 返回 8 框且响应不含图片；无摄像头机器提示
  `无法打开摄像头：Requested device not found` 并正确复位按钮。

细节见 `6-web/README.md` 的「目录批量 · 视频 · 摄像头（v1.16）」与
`docs/WORKFLOW_GUIDE.md` 的 §7.5。

### v1.15 — 2026-11-02（导出 ONNX 进页面 + 四种任务头的 `.transd.onnx` 全部修通）

上一版把「开跑前」和「中断后」补上了，这一版补的是**链路最后一环**：训练完的 `best.pt`
以前只能回命令行才变得成引擎，而且分割/姿态/旋转框的 `.transd.onnx` 实际上从未跑通过。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 导出入口 | 页面叫「训练导出」但只有训练；导出要敲 `--mode export --make-transd true` | 训练页新增「导出 ONNX」折叠块：选权重（可一键取当前实验 `best.pt`）、imgsz/opset/dynamic/half/simplify、勾选是否生成 `.transd.onnx` | 训练 → 导出 → 编译 → 推理全程不用离开浏览器 |
| 导出后衔接 | 手填 ONNX 路径到引擎页 | 结果表 `.transd.onnx` 行有「去编译引擎」，点击自动切页 + 填路径 + 触发形状探测 | 少一次「我该填哪个文件」的困惑 |
| 转置定位 | 取 `graph.node[-1]`，即「图里最后一个节点」 | 按形状识别预测头（3 维输出；多个时用通道<anchors 裁决） | 图末尾多一个 Cast/Concat 不会再静默转错 |
| 任务校验 | 不校验，错了留到推理时报 `expects channels=56` | 转置前按任务校验：pose=56、obb≥6、segment 需 proto 且通道 > 4+proto | 导出阶段就能看出「任务选错了」 |
| **分割转置** | **未实现**（只转一个输出，且 proto 在 binding 2） | **proto 提到 binding 1**，检测头转成 `[1, anchors, 4+nc+32]` | 以前分割引擎**编译能过、推理必报** `channel layout invalid`；现在可用 |
| 任务不匹配 | 用 `--task detect` 导 seg 权重会产出通道 116 的「检测头」 | 以权重自带 task 为准，打印 `task_overridden` | 少一类最难查的错 |
| 转置失败 | 抛异常，整个导出判失败 | 只报 `TRANSD_FAILED`，**保留已导出的原始 ONNX** | 至少还有原始 ONNX 可用 |

**必须记住的一条**：C++ 侧对 `V8Seg` 直接读 `static_dims(1)` 当 proto、`static_dims(2)` 当检测头，
而 Ultralytics 导出的顺序是「检测头在前、proto 在后」。**必须在生成 `.transd.onnx` 时重排**，
否则 trtexec 编译成功、推理时才炸——这是本次唯一一处「不修就等于功能不存在」的地方。

**两处刻意的取舍**

1. **导出仍写成任务（子进程 + WebSocket），不做同步接口**。导出要加载 torch 与权重，实测 2~30 秒，
   同步会卡住整个事件循环；复用训练那套机制，日志与「停止」按钮都不用重写。
2. **产物不做「另存为」**。与 Ultralytics 保持一致：`best.pt` → 同目录 `best.onnx` / `best.transd.onnx`。
   少一个输出目录参数，就少一类「ONNX 在哪」的问题；需要保留旧产物就先复制一份权重。

验收（全部实测）：

- 新增 `test_transd_onnx.py` **17/17**：四任务头转置、预测头不在末尾、动态维保留 `dim_param`、
  分割输出重排、通道数不符/缺 proto 时报错。
- 新增 `test_export_request.py` **24/24**：开关透传、`--simplify` 不能跟值（踩过）、非法 imgsz/task 提前拒绝。
- 回归 12 个后端脚本全绿：`test_annotation_project`(138)、`test_infer_params`(71)、
  `test_onnx_fallback`(36)、`test_train_info`(34)、`test_augment_params`(26)、`test_engine_progress`(26)、
  `test_aug_preview`(24)、`test_checkpoints`(22)、`test_dataset_workbench`(20)、`test_exit_codes`(6)；
  前端 `test_contrast`(23)、`test_viewer_math`(37)。
- **四任务头真链路**（导出 → `.transd.onnx` → trtexec → DLL 推理）：

| 任务 | `.transd.onnx` 输出 | 引擎编译 | 推理结果 |
| --- | --- | --- | --- |
| detect | `[1, 2100, 84]` | 340 s | `pcb_best.transd.onnx`（10 类）→ 313 s 编译，PCB 图片出 1 框（conf 0.585） |
| segment | proto `[1,32,80,80]` + 头 `[1,2100,116]` | 340 s | `bus.jpg` 4 框，全部 `has_mask=true` |
| pose | `[1, 2100, 56]` | 370 s | `zidane.jpg` 2 人，各 17 个关键点（置信度 0.87） |
| obb | `[1, 2100, 20]` | 368 s | `plane2.jpg` 11 个旋转框，带角度 |

- 浏览器实测：真实 PCB `best.pt` → 导出 **6.5 秒**出 `pcb_best.onnx`(12.1 MB) 与
  `pcb_best.transd.onnx`(输出 `[1,2100,10]`)，点「去编译引擎」自动切页并探测出
  「布局正确，可以编译」。

细节见 `6-web/README.md` 的 v1.15 小节与 `docs/WORKFLOW_GUIDE.md` 的 §4.1.1。

**配套的流程改进（同一批完成）**

- **测试有了单一入口和 CI**：新增 `6-web/backend/tools/run_all_tests.py`（扫描目录自动收录
  `test_*.py`，以后新增脚本不用改流水线）+ `.github/workflows/tests.yml`（Windows + py3.11
  跑后端 12 个脚本，Node 20 跑两个前端单测并逐个 `node --check`）。以前 444 项断言全靠人肉跑，
  `--simplify true` 那类错误正是这么漏出去的。
- **openspec 清零**：6 个已完成变更全部归档，`openspec/specs/` 补上 `web-studio`、
  `web-training-console`、`yolo-training-platform`、`annotation-dataset-workbench`、
  `anomalib-trt-inference`、`project-layout`，`openspec validate --specs` 11/11 通过。

### v1.14 — 2026-11-01（训练前体检：数据集概览、增强预览、断点续训）

上一版解决了「训练中看得见」，这一版解决「开跑前看得见」。三个功能都指向同一类返工：
**跑到一半才发现数据集配错、增强配错、或者白重跑了 30 轮。**

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 数据集体检 | 只能靠训练时 YOLO 自己打印的几行摘要，跑起来才知道少标注 | 训练页「数据集概览」：三 split 图片/标注/空标/漏标/孤儿标、每类框数条形图、抽样分辨率 | 开跑前 1 秒看清数据长什么样，不用等训练报错 |
| 扫描成本 | — | 只读 `dataset.yaml` + 标签文本首列，**不解码图像**；分辨率只读图像头 | 692 张图的真实 PCB 数据集 225 ms 出结果 |
| 重复扫描 | — | 按 `yaml 路径 + mtime + 大小` 缓存 60 秒，可 `refresh` 强制重扫 | 上万张图点第二次秒回，不会以为卡死 |
| 增强效果 | 改完 `mosaic`/`scale` 只能等训练出的 `train_batch*.jpg` | 训练页「增强预览」：按当前表单参数出「原图 vs 增强后」对照图 | 参数填错立刻看见，不用等一轮训完 |
| 预览可复现 | — | `seed` 决定样本与全部随机量，同参数同 seed 必出同图 | 逐个参数对比「到底变了什么」 |
| 续训入口 | `--resume` 靠 `get_latest_run()` **猜**最近一次 run | 表格列出所有可续训 run（轮次 / 最佳 mAP / last.pt 大小 / 不可续训原因），点一下就填好表单 | 多项目并存时不会续错到别的实验 |
| 续训实现 | — | launcher 新增 `--resume-from <last.pt>`，`model.train(resume=该路径)` | 显式指定检查点，不再依赖「最近一次」这个脆弱假设 |

**两处刻意的取舍**

1. **预览用 OpenCV 近似，不是跑 Ultralytics 本体**。后端环境没有 torch 也能用（不需要 GPU、不占显存），
   代价是概率采样顺序、缓存这些无法一比一复刻。所以预览是「确认方向和量级」，不是验收依据；
   `copy_paste`（需要分割掩码）与 `close_mosaic`（只在最后 N 轮生效的调度）明确不预览，
   并在响应 `ignored` 里列出来，避免用户以为「预览里有就是训练里也有」。
2. **扫 run 不加载 `.pt`**。只读 `args.yaml`（纯文本）与 `results.csv`（行数 = 已完成轮次，
   最大 `mAP50-95` = 最佳指标），所以 Web 后端不需要 torch 就能列出全部历史实验，
   也不会有「扫一下就吃了 30GB 显存」的问题。

**必须记住的一条**：续训会**写回原 run 目录**。Ultralytics 的 `check_resume` 用检查点里记录的
`project/name`，所以 `results.csv` 是继续往后追加行，而不是另开目录。想开新实验就别点「继续训练」，
改 `--name` 正常训练。

验收：`test_train_info.py` **34/34**、`test_aug_preview.py` **24/24**、`test_checkpoints.py` **22/22** 通过；
回归 `test_engine_progress`(26)、`test_exit_codes`(6)、`test_augment_params`(26)、
`test_dataset_workbench`(20)、`test_infer_params`(71)、`test_onnx_fallback`(36)、
`test_annotation_project`(138) 全通过。浏览器实测：扫描真实 PCB 数据集（692 图 / 2950 框 / 6 类）225 ms、
定位到孤儿标注 `04_mouse_bite_17.txt` 并给告警；增强预览出原图与增强对照并列出生效项与忽略项；
从中断的 `base_interrupt`（1/2 轮）续训，命令行带 `--resume-from ...\weights\last.pt`，
从 `epoch 2/2` 接着跑完，产物列表按路径去重。细节见 `6-web/README.md` 的 v1.14 小节与
`docs/WORKFLOW_GUIDE.md` 的 3.2.2。

### v1.13 — 2026-11-01（训练与引擎构建的进度可视化：终于知道它在干什么、还要多久）

问题很具体：**引擎构建页在 960×960 上会静默 473 秒**，训练页只有一条不会动的进度条。
用户的原话是「一直不动用户不知道在干什么」。本次只解决「看得见」，不改任何算法。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 引擎构建进度 | 只有「构建中/已完成」徽标，tactic 搜索期间 8 分钟无反馈 | 8 阶段清单 + 估算百分比 + 已用/预计剩余 | 静默期进度仍在动（心跳 2 秒一次），不再像卡死 |
| 构建耗时 | 无记录，每次都靠猜 | 成功即落盘 `runtime/engine_stats.json`，下次取历史中位数 | 越用越准，换模型也不用改代码 |
| 训练进度 | 一条宽度 | 阶段 + `epoch n/N` + 百分比 + 已用 + 每轮耗时 + 预计剩余 | 剩余时间按最近 3 轮中位数推算，首轮扫数据不会把估算拉偏 |
| 训练指标 | 一排最新值的 chip | 「每轮指标」表（动态列，最近 20 轮）+ 两张零依赖内联 SVG 曲线 | 趋势一眼可见，不用等训练完去看 `results.png` |
| 参数可核对 | 命令行只在日志里打一行 | 「实际执行的参数」折叠块，可复制 | 能确认自己填的参数真的传进去了 |
| 主动停止 | 读线程与 stop 抢状态，可能显示「构建失败」 | 先标记 `cancelled` 再 terminate | 稳定显示「已停止」，进度停在中断处 |

**必须记住的一条结论（它推翻了原先的直觉）**

第一版估算假设「耗时 ∝ 输入面积」，把 960 的 474 秒折算到 320 只剩 **53 秒**；
实测 320 需要 **313.6 秒**。真实规律是「层数 × 每层候选 kernel 数」占主导，
输入尺寸只影响每次候选的试跑成本，而且分辨率变小后显存能塞下更多 tactic、搜索空间反而更大。
用 960 / 320 两个实测点反解出指数 ≈ 0.37，公式定为：

```
秒数 ≈ 38.3 × ONNX_MB × (imgsz / 960) ^ 0.4      # fp16；fp32 × 0.7
```

另外实测确认：`--heuristic`（`BuilderFlag::kENABLE_TACTIC_HEURISTIC`）在 T1000（Turing）
上**被 TensorRT 直接忽略**，日志原文 `This feature is only supported on Ampere and beyond`。
所以「换 heuristic 提速」这条路在这台机器上不成立，别浪费时间。

验收：`python 6-web/backend/tools/test_engine_progress.py` → **26 / 26 通过**
（含真实日志按原始时间戳回放、473 秒静默期心跳产出 86 条进度、失败不补 100%）；
回归 `test_exit_codes`(6)、`test_augment_params`(25)、`test_dataset_workbench`(20)、
`test_infer_params`(71)、`test_onnx_fallback`(36)、`test_annotation_project`(138) 全通过。
浏览器实测：960 构建进度 8%→11%→25% 且日志静默期仍在动；320 构建收尾显示
「实际 5 分 17 秒，估算 53 秒」并落盘 313.55 s；训练 3 轮显示 `epoch 3/3`、每轮 ≈ 0:06、
最终验证行标 `final`、loss 与 mAP 曲线正常。细节见 `6-web/README.md` 的 v1.13 小节。

> v1.12 是一次纯代码审查修复（标注页的并发保存、旋转框拖边、撤销切图等 10 项），
> 只记在 `6-web/README.md`，此处不重复。

### v1.11 — 2026-10-31（Web 标注工作台：不用装 Qt 也能画框、存盘、导出）

以前「标注」这件事只能在 `4-qt-studio` 里做，而 Qt 版长期编不出来，等于整条数据链路
（采集 → 标注 → 训练）在标注这一环断着。本次把标注搬到浏览器，格式与 Qt 版**双向互通**。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 标注入口 | 只有 Qt 桌面程序 | 浏览器「标注」页签 | 局域网任意机器可标注，不用装 Qt / 不用编译 |
| `.tfanno` 读写 | 仅 Qt | `annotation.py` 逐字段对齐 `AnnotationProject.cpp` | Web 存的 Qt 能开，Qt 存的 Web 能开 |
| 形状 | — | 矩形 / 旋转矩形 / 多边形 / 线 / 点 / 关键点组，六种全支持 | 检测、分割、姿态、OBB 四类任务都能标 |
| 撤销 | — | 双栈各 50 步，拖动未位移不入栈 | 手滑不会把撤销记录塞满 |
| 并发保存 | — | `rev` 乐观锁 + `mtime_ns` 文件指纹 | 两个人改同一个工程，后保存的人看到冲突弹窗，不会静默覆盖 |
| 保存 | — | 停止编辑 3 秒自动保存 + `Ctrl+S` | 忘了存也不会丢 |
| 自动标注 | — | 本地引擎出草稿（可选外部 HTTP 服务） | 先机器粗标再人修，效率翻倍 |
| 草稿处理 | — | 单条 / 全部接受与拒绝，全部接受算一次撤销 | 不满意一键扔掉 |
| 导出 | 数据集页只能导已有文件 | 标注页把**内存中**工程直接导出 YOLO/COCO/VOC | 标完立刻能训，不用先存再切页 |
| 版本可见性 | 无版本概念 | 顶栏徽标 + `/api/health` 返回 `version` | 一眼看出页面连的是不是最新后端，不再被旧进程骗 |

三个必须知道的坑（都已修，记下来免得再踩）：

1. **`st_mtime_ns` 不能当 JSON 数字传给浏览器**。它约 `1.79e18`，超过 JS 的
   `Number.MAX_SAFE_INTEGER`（`9e15`），`JSON.parse` 会把低位舍入，
   于是"刚打开就报文件被外部修改"。改成字符串传输 + `4096ns` 容差比较。
2. **`snapPoint()` 的返回契约必须统一成 `[x, y]` 数组**。有一个分支返回了 `{x, y}` 对象，
   调用方按数组取 `point[0]` 就拿到 `undefined`，表现是"画出来的框没有坐标"并且拖拽收不了尾。
3. **引擎输出维度和 `yolo_infer.dll` 不匹配时 DLL 不报错**，只返回 `confidence=957` 这类垃圾框。
   自动标注侧加了 `sanitize_drafts()`：置信度不在 `[0,1]`、宽高非正、出图的草稿直接丢掉并给出警告，
   不至于把垃圾框灌进标注文件。

验收：`python 6-web/backend/tools/test_annotation_project.py` → **131 / 131 通过**；
回归 `test_infer_params.py`(71)、`test_augment_params.py`(26)、`test_dataset_workbench.py`(20)、
`test_exit_codes.py`(6)、`test_onnx_fallback.py`(36) 全通过。
浏览器实测：六种形状各画一次并重载后几何与类别一致、撤销 50 步后重做 50 步回到初态、
外部改文件触发冲突弹窗的两个选项都生效、真实 `best.transd_fp16.engine` 生成草稿并接受、
中文与含空格路径的图片正常显示、三种格式导出产物与界面一致。细节见 `6-web/README.md`。

**同版本附带的一处小修**：排查过程中发现 8060 端口被几小时前的旧进程占着，
导致「改了代码没效果」白查很久。因此加了版本号可见性——`app.py` 定义 `APP_VERSION`，
`/api/health` 返回 `{"ok":true,"version":"1.11.0"}`，前端顶栏显示 `v1.11.0` 徽标。
以后徽标不涨就说明连的是旧进程。版本号规则：**改后端就改 `APP_VERSION`**。

### v1.10 — 2026-10-31（Web 推理控制台：阈值可覆盖、框改矢量叠加、显存不再涨）

以前的推理页只能"填路径 → 看图"，参数是死的，图上的框是后端烧死的。本次把它做成真正的控制台。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 类别名 | 只有 `cls0`/`cls1` | 解析 `labels_*.json` 显示 `missing_hole` 等真实名 | 现场排错不用再对着类别号查表 |
| 检出框来源 | 后端画死在 PNG 上 | 后端出干净原图，前端 SVG 画框 | 缩放不糊、可单独关框、标签字号恒定 |
| Anomalib 阈值 | 只能吃 `metadata.json` | 可覆盖图像/像素阈值，也可一键沿用默认 | 同一个引擎能现场试不同灵敏度 |
| 阈值展示 | 无 | 同时显示「生效值 / 原值 / 是否覆盖」 | 不会被 `ai_get_metadata` 的原地改写骗到 |
| 热能图 | 无 | `cv2.applyColorMap` 可选色带，独立窗口 | 异常区域比红色轮廓更直观 |
| 引擎句柄 | 无上限累积 | LRU 上限 4，淘汰立即 destroy | 连续换 8 组参数，常驻句柄恒为 4 |
| 模型发现 | 手工敲路径 | `GET /api/models` 扫 `runs/`、`results/` | 选完自动带出 ONNX / 标签 / metadata |
| 结果明细 | 一段文字 | 可排序表格 + 点行高亮 + 置信度直方图 + 计时分解 | 能直接看出是哪一类、哪一步慢 |
| 导出 | 右键存图 | JSON / CSV 一键导出，数值与屏幕一致 | 出报告不用手抄 |

**必须知道的一个坑**：`ai_create` 的阈值参数是**原地写进**内部 `meta_` 的，
所以之后 `ai_get_metadata` 返回的已经是覆盖后的值。想拿文件原值只能自己读 `metadata.json`。
`infer.read_metadata_thresholds()` 就是为此单独加的，响应里 `effective` / `metadata` 两个字段分得很清。

验收：`python 6-web/backend/tools/test_infer_params.py` → **71 / 71 通过**；
另跑 `test_augment_params.py`(26)、`test_dataset_workbench.py`(20)、`test_exit_codes.py`(6)、
`test_onnx_fallback.py`(36) 全通过。真实产物在浏览器里联调过 PCB 与 `bottle.png`，
细节见 `6-web/README.md`。

### v1.9 — 2026-10-31（Qt 版首次完整编译成功：TRTForge.exe 可启动）

这条不是重构，是**把长期挂着的「本机没有 Qt，界面一直不可用」彻底解决了**。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| `TRTForge.exe` | 从未编译成功 | **编译成功并实测可启动**（361 MB 常驻） | Qt 桌面版恢复可用 |
| CMake 配置 | 失败 | `Configuring done` / `Generating done` | 四个目标全部产出 |
| `ocr_infer` | 从未编译（缺依赖） | `ocr_infer.dll` + `.lib` + `ocr_basic.exe` | OCR 链路解锁 |
| `yaml-cpp` | 空目录（坏的 gitlink） | 源码就位，随工程正常编译 | 见下方"为什么它一直是空目录" |
| 运行时依赖 | 需手工复制，漏一个就 `0xC0000135` | POST_BUILD 自动复制 6 个依赖 | 清空 `build/` 重编也不会缺文件 |
| 编译阻塞记录 | 记为"本机工具链阻塞" | 实为 OpenCV 死路径 + 三处真实缺陷 | 误判已纠正 |

**完整构建命令**

```powershell
$env:OPENCV_DIR = "E:\thirdparty\opencv"

# 1) OCR 子工程（必须先编，主工程要链接它的 .lib）
cd ocr\ocr_infer
cmake --preset vs2022-x64 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
      "-DPADDLE_LIB=<仓库>/ocr/ocr_infer/third_party/paddle_inference"
cmake --build --preset vs2022-x64

# 2) 主工程
cd ..\..\4-qt-studio
cmake --preset vs2022-x64
cmake --build --preset vs2022-x64
```

**本次修掉的四个真实缺陷**

1. **OpenCV 死路径**：5 个 `CMakeLists.txt` + 2 个 `CMakePresets.json` 写死 `D:/thirdparty/opencv`，本机实际在 `E:\thirdparty\opencv`。见 v1.8。
2. **`mkldnn/lib` 漏配**：`ocr_infer/CMakeLists.txt` 的链接搜索目录只列了 `onednn/lib`，而官方 GPU 包用的是 `mkldnn`。链接期报 `LNK1181: 无法打开输入文件 mkldnn.lib`。
3. **CMake 4.2 与 clipper 不兼容**：clipper 写的是 `cmake_minimum_required(VERSION 2.6.0)`，CMake 4.x 已移除 <3.5 兼容。用 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 绕过。**PowerShell 里必须加双引号**，否则 `=3.5` 会被拆成独立参数，报 `Could not read presets from ...\.5`。
4. **运行时依赖不落地**：`ocr_infer.dll` 依赖 `paddle_inference.dll`、`common.dll`、`mkldnn.dll`、`libiomp5md.dll`、`yaml-cpp.dll`、`polyclipping.dll`，分散在 Paddle 包与 build 目录里。已加 POST_BUILD 自动复制到 `4-qt-studio/bin`。

**为什么 `yaml-cpp` 一直是空目录**

它在 git 索引里登记成 **gitlink（模式 `160000`，指向提交 `0579ae3`）**，但仓库**缺少 `.gitmodules`**。这是一种损坏的子模块登记：克隆下来就是个空目录，构建必然失败。

对比同目录另三个第三方库，全部是普通文件管理：

| 库 | git 形态 | 文件数 |
| --- | --- | --- |
| `abseil-cpp` | 普通文件 | 1365 |
| `clipper_ver6.4.2` | 普通文件 | 262 |
| `nlohmann` | 普通文件 | 3 |
| `yaml-cpp` | **gitlink（坏）** | — |

补齐源码后确认其 HEAD 恰为 `0579ae3`（tag `yaml-cpp-0.7.0`，远端 `jbeder/yaml-cpp`），版本正确。

### v1.8 — 2026-10-31（推理降级保底 + OpenCV 路径去硬编码 + ckpt/ONNX 对比打通）

三件事凑到一起，都是"环境一换就不工作"的老毛病，这次从根上拆掉。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| DLL 不可用时的推理 | 直接 503，推理页整页不可用 | 自动降级到 `onnxruntime` 跑同源 ONNX | 没编译 DLL 也能验证模型链路 |
| 降级结果可辨识 | — | 返回 `backend="onnxruntime-fallback"` + `warning` 说明原因 | 不会把降级结果误当成 TensorRT 性能 |
| 能力探测 | 只有 `infer_yolo`/`infer_anomalib` | 增加 `infer_fallback`/`infer_fallback_detail` | 前端可提前提示"仍可跑通" |
| OpenCV 路径 | `CMakeLists.txt` 写死 `D:/thirdparty/opencv` | `-DOPENCV_DIR` > 环境变量 > CMake 常规搜索 | 换机器/换盘符不再需要改源码 |
| 受影响文件 | 2 处（任务原记录） | **5 个 `CMakeLists.txt` + 2 个 `CMakePresets.json`** | preset 里的 `OPENCV_DIR` 会盖掉环境变量，必须一起清 |
| TRTForge 配置 | 本机失败，记录为"工具链阻塞" | **`Configuring done` / `Generating done`（exit 0）** | 真因是 OpenCV 死路径，不是工具链 |
| 推理 DLL 编译 | — | `yolo_infer.dll`、`anomalib_infer.dll` 均编译成功 | 产物 `4-qt-studio/bin/*.dll` |
| ckpt vs ONNX 对比 | 脚本能跑但分数恒 0.5，结论无意义 | 三模型验证通过（PatchCore 81.1%、PaDiM 89.0%、EfficientAd 82.7% 判定一致） | 见下方"对比脚本修了什么" |

**降级推理的行为边界**

- YOLO 只支持 `detect` / `obb`。`segment` / `pose` 需要解析额外输出张量，Python 复刻容易与 C++ 产生细微差异，遇到时明确报错并提示走 DLL。
- Anomalib 只能加载 `.onnx`。TensorRT 引擎（`.engine`）由 DLL 专用，onnxruntime 不识该格式。
- 引擎选的是 `.engine` 时，会按「同目录同名 `.onnx` → `metadata.json` 里的 `onnx_path`」顺序自动找替代模型。
- 降级路径不做 C++ 侧的分阶段计时，`preprocess_ms`/`postprocess_ms` 记 0，只有 `inference_ms` 有效。

**对比脚本修了什么**（`compare_ckpt_onnx_inference.py`，三个都是"能跑但结果错"的隐形缺陷）

1. `import anomalib` 落到了 user-site 的 **1.2.0**，而 ckpt 是仓库内 **2.0.0b3** 训练的。版本反了导致 `state_dict` 报 unexpected keys，实际跑出来分数恒为 `0.5`。修法：优先把 `anomalib-main/src` 注入 `sys.path`，与 `tf_anomalib_launcher.py` 保持一致。
2. 新版 `Engine.predict` 返回 `list[dict]`，键是**复数**（`pred_scores` / `anomaly_maps` / `pred_labels`）；旧代码用 `getattr` 取单数属性，全部得到 `None`，于是判定永远"OK"。
3. `numpy 2.x` 下 `imgaug` 因 `np.sctypes` 被移除而 import 失败。加运行时垫片，不必降级 numpy（降级会波及 ultralytics / opencv / onnxruntime）。

**关于"两侧分数差 ~0.33 是不是 bug"**：不是。ckpt 侧分数由模型 `post_processor` 用自己的阈值归一化，ONNX 侧用 `metadata.json` 里**校准后**的阈值归一化。两个基准不同，同一原始分数会落到不同位置，于是差异呈近常数偏移。判定结论以 metadata 阈值为准——那是 `generate_trtforge_metadata.ps1` 在正常/异常样本上验证过的。

### v1.7 — 2026-10-31（YOLO 数据增强参数开放：15 项全链路可调）

以前 launcher 只有 `lr0/lrf/momentum/weight_decay/freeze/imgsz/batch` 这几个旋钮，
**数据增强一项都调不了**——想降 mosaic 只能改 Ultralytics 源码，改完下次升级就丢。
本次把 15 项增强参数开到命令行与 Web 表单，默认值与 Ultralytics 逐项对齐。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 增强可调项 | 0 项（只能用 Ultralytics 默认） | 15 项（见下表） | 小缺陷场景终于能关掉 mosaic |
| 默认值一致性 | 直接吃 Ultralytics 默认 | launcher 默认值逐项等于 `DEFAULT_CFG`，并加测试守住 | 不填参数 = 行为与升级前完全相同，历史实验可对比 |
| Web 表单 | 无增强字段 | 「数据增强」折叠区，留空即用默认 | 表单不会因为多 15 个字段而过长 |
| 参数传递语义 | — | 留空 = 不拼进命令行；填 `0` = 显式传 0 | 0 不会被误判成「没填」而丢掉 |
| 越界防护 | 训练启动后由 Ultralytics 报错 | 后端在启动前返回 400（概率类 0~1、`close_mosaic` ≥ 0） | 不用等几十秒才知道填错了 |
| 预设 | 无 | Web 提供「小缺陷预设」「清空」两个按钮 | PCB 这类 70px 缺陷一键套用合理值 |

**新增的 15 个参数**（默认值 = Ultralytics 8.4 `DEFAULT_CFG`，命令行用短横线命名）

| 参数 | 命令行 | 默认 | 范围 | 作用 |
| --- | --- | --- | --- | --- |
| `hsv_h` | `--hsv-h` | 0.015 | 0~1 | 色调抖动 |
| `hsv_s` | `--hsv-s` | 0.7 | 0~1 | 饱和度抖动 |
| `hsv_v` | `--hsv-v` | 0.4 | 0~1 | 亮度抖动（应对打光差异） |
| `degrees` | `--degrees` | 0 | ±度 | 随机旋转 |
| `translate` | `--translate` | 0.1 | 0~1 | 随机平移 |
| `scale` | `--scale` | 0.5 | 0~1 | 随机缩放增益 |
| `shear` | `--shear` | 0 | ±度 | 随机错切 |
| `perspective` | `--perspective` | 0 | 0~0.001 | 透视变换 |
| `flipud` | `--flipud` | 0 | 0~1 | 上下翻转 |
| `fliplr` | `--fliplr` | 0.5 | 0~1 | 左右翻转 |
| `bgr` | `--bgr` | 0 | 0~1 | RGB/BGR 通道互换 |
| `mosaic` | `--mosaic` | 1.0 | 0~1 | 马赛克四图拼接 |
| `mixup` | `--mixup` | 0 | 0~1 | 混类叠加 |
| `copy_paste` | `--copy-paste` | 0 | 0~1 | 复制粘贴（仅分割生效） |
| `close_mosaic` | `--close-mosaic` | 10 | ≥0 | 最后 N 个 epoch 关闭 mosaic |

**使用方式**

```bash
# 命令行：只写出想改的项，其余自动用默认值
python tf_yolo_launcher.py --mode train --task detect --model yolov8n.pt \
  --data configs/pcb_detect.yaml --epochs 100 --imgsz 640 --batch 16 \
  --mosaic 0.1 --close-mosaic 20 --scale 0.3
```

Web 端：「模型训练」页 → 展开「数据增强」→ 填想改的项（留空即默认）
→ 或直接点「小缺陷预设」。

**应用场景**

| 场景 | 建议 | 原因 |
| --- | --- | --- |
| 小目标缺陷（PCB 划痕约 70px） | `mosaic 0~0.3`、`close_mosaic 15~20` | mosaic 把 4 张图拼成 1 张，缺陷再缩小一半容易消失 |
| 拍摄角度固定、正对工件 | `degrees 0`、`perspective 0` | 引入不真实视角反而拖慢收敛 |
| 打光不稳定、多工位 | `hsv_v 0.4~0.6`、`hsv_s 0.7` | 提升对亮度/色彩变化的鲁棒性 |
| 有丝印/文字/方向性图案 | `fliplr 0`、`flipud 0` | 镜像后文字反了，属于不真实样本 |
| 样本极少（< 500 张） | `mixup 0.1~0.2`、`copy_paste 0.1` | 增加样本组合多样性 |

**验证记录**

| 项 | 结果 |
| --- | --- |
| 参数透传 | 15 项逐个验证 → `build_train_kwargs()` 全部正确 |
| 默认值零漂移 | 逐项对比 `ultralytics.cfg.DEFAULT_CFG` → 无一项不一致 |
| 后端命令拼装 | 不填 → 命令行零增强 flag；填 `0.0` → 仍然拼入 |
| 越界拦截 | `mosaic=1.5`、`close_mosaic=-1` → 400 且不启动进程 |
| 真实短训练 | 2 epoch @320px，`mosaic=0.1/close_mosaic=1/scale=0.3/fliplr=0` |
| `args.yaml` 核对 | 四项全部为传入值，未传项保持默认，exit 0 |
| 测试脚本 | `6-web/backend/tools/test_augment_params.py` → 26 项全通过 |

### v1.6 — 2026-10-31（Web 数据集工作台：划分与导出进浏览器）

以前要给已有图片重新划分 train/val、生成 `dataset.yaml`、或把数据转成 COCO/VOC，
只能在 Qt 版里点，或手工写脚本。本次把这套能力搬进 Web，第 5 个页签「数据集」。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 划分入口 | 只在 Qt 标注页 | Web「数据集」页：填比例 → 扫描预览 → 导出 | 无 Qt 环境也能划分 |
| 划分可复现性 | `qHash(i + "_2026")`，依赖 Qt 哈希实现 | FNV-1a 64 位 + 同盐值 `_2026` | 同数据同比例，任何机器结果逐项一致 |
| 比例语义 | train/val 向下取整，余数归 test | 完全照搬（`n*ratio//100`） | 与历史实验口径一致，不会悄悄改变含义 |
| `dataset.yaml` | 仅 Qt 导出时生成 | Web 同样生成，含 `path/train/val/test/nc/names`，pose 另加 `kpt_shape` 与 `flip_idx` | 导出即可直接训练 |
| 导出格式 | 仅 YOLO | YOLO / COCO / VOC | 一份数据多种下游可用 |
| 输入源 | Qt 项目文件 `.tfanno` | `.tfanno` 或现成 YOLO 目录（含 `data.yaml`） | 拿到别人的 YOLO 数据集也能直接重划分 |
| 误操作防护 | 无 | 输出不得在源内、非空目录需显式覆盖、OBB 拒绝 COCO、VOC 仅 detect、比例和超 100 直接拒绝 | 不会把数据覆盖坏、不会产出非标准格式 |
| 空集问题 | 无提示 | 比例向下取整为 0 时导出摘要明确告警 | 不会再出现「训练集 0 张」却没人发现 |

**实测数据（PCB 6 类 / 692 张 / 3034×1586）**

| 项 | 结果 |
| --- | --- |
| 扫描 | 692 张、6 类、2950 个标注对象、0 张空标注 |
| 划分 70/20/10 | train 484 / val 138 / test 70，两次调用逐项一致 |
| 划分 80/20 | train 553 / val 138 / test 1（余数归 test，与 Qt 一致） |
| YOLO 导出 | 484+138+70 张，标签逐值往返无损 |
| Ultralytics 识别 | `check_det_dataset()` → nc=6、6 个类名、三个划分路径正确 |
| COCO 导出 | 692 images / 2950 annotations / 6 categories |
| VOC 导出 | 692 XML（仅 detect） |
| 验收脚本 | `tools/test_dataset_workbench.py` → 20 项全通过 |

新增接口：`POST /api/dataset/scan`（预览，不落盘）、`POST /api/dataset/export`（导出）。

### v1.5 — 2026-10-31（Web 引擎构建页上线，补齐「训练 → 引擎」最后一环）

此前 Web Studio 只能训练导出 ONNX，再转 TensorRT 得回到命令行手工敲 `trtexec`。
本次把这一步收进 Web，链路在浏览器内闭环。

| 项目 | 老版本 | 新版本 | 影响 |
| --- | --- | --- | --- |
| 引擎编译入口 | 只能在终端手工执行 `trtexec` | 「引擎构建」页：选 ONNX → 填精度 → 点按钮 | 不用记 trtexec 参数 |
| ONNX 预检 | 无，编译完跑推理才发现布局错 | `GET /api/engine/probe`：选完文件即显示输入/输出形状、体积、可用性判定 | 错误在编译前暴露 |
| 未转置 YOLO 防护 | 后端只在「推理」时用 `_check_yolo_output_sane()` 兜底 | 构建前前端标红 + 后端直接 400 拒绝，错误信息说明为何要用 `.transd.onnx` | 不再产出静默出错的引擎 |
| 产物校验 | 只看退出码，`trtexec` 正常退出但没落盘也算成功 | `runners.py` 新增 `expect_file`，产物缺失判为 failed 并发 `ARTIFACT_MISSING` | 避免「显示成功、点推理找不到文件」 |
| trtexec 定位 | 依赖环境变量 | `config.resolve_trtexec()` 四级探测：`TRTFORGE_TRTEXEC` → `CUDA_PATH\bin` → CUDA 根 `v*/bin` → PATH | 装完 CUDA 即可用 |
| 环境状态灯 | 只查 DLL 与 launcher | 增加「缺少 trtexec（引擎构建不可用）」提示 | 缺失时一眼可见 |

**实测数据（PCB 6 类模型，960×960，FP16，T1000 8GB）**

| 项 | 结果 |
| --- | --- |
| 构建耗时 | 474.1 秒（首次，无 timing cache 命中） |
| 引擎大小 | 10.14 MB |
| 吞吐 | 135.4 qps |
| 退出码 | 0，产物落盘校验通过 |
| 错误路径 | 提交 `best.onnx` 前端红字提示 + 后端 400，日志面板显示完整原因 |

> 注意：960×960 这类分辨率下 trtexec 的 tactic 搜索可达数分钟，期间日志可能长时间不刷新。
> 只要 GPU 占用率接近 100% 就是在正常编译，不是卡死。

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
| 标注 | 打开 / 新建 `.tfanno`、六种形状绘制（矩形/旋转矩形/多边形/线/点/关键点组）、缩放平移吸附、撤销重做、3 秒自动保存、修订号冲突弹窗、类别与属性面板、自动标注草稿层（**v1.23**：整工程批量预标注 + 草稿旁挂落盘 + `A`/`D`/空格审核 + 置信度分档着色；**v1.24**：点框提示（SAM）与文本提示（GroundingDINO）两条免训练通道 + 模型仓库页内下载 + `K` 已审核跳下一张）、导出 YOLO / COCO / VOC |
| `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` | 新增 |
| 全流程串联 | **v1.17 YOLO**：训练→导出→校验→编译→补标签→精度验证（mAP）；**v1.18 Anomalib**：训练+导出+校准→校验 metadata→编译→引擎精度验证（AUROC / 漏检 / 误报 / 最优阈值），支持跳过训练复用已有产物 |
| `docs/WORKFLOW_GUIDE.md` | 新增 |
| `readme.md` | 新增模型速查表、阈值校准说明、版本更新记录 |
**路径选择（v1.20）**：所有填文件/文件夹的地方点按钮就弹 Windows 系统对话框
（文件 / 多选 / 文件夹 / 另存为），`Shift` + 点击退回页面内目录浏览。
浏览器自身不能弹系统框，所以框由后端在本机桌面弹出，因此只对本机访问启用。
细节见 `6-web/README.md` 的「选路径：原生对话框（v1.20）」。


### v1.1 — 2026-10-28（五级目录重构）

| 项目 | 老版本 | 新版本 |
| --- | --- | --- |
| 目录结构 | 根目录平铺 | 五级分层：`1-python-yolo` / `2-python-anomaly` / `3-cpp-tensorrt` / `4-qt-studio` / `5-reference`，另加 `6-web` 与 `ocr` |
| 界面形态 | Qt 桌面程序 `TRTForge.exe` | 浏览器工作站 `6-web`（FastAPI + 零构建前端）为**推荐入口**；Qt 版仍可完整编译运行（见 v1.9），作桌面备选 |
| 脚本路径 | 硬编码 `D:\test1\deeplearning\...` | 全部按脚本自身位置推导，可用环境变量覆盖 |
| `.gitignore` | `ultralytics/`、`anomalib/` 会误伤训练源码 | 改为锚定仓库根的 `/ultralytics/`、`/anomalib/` |

## Web 版工作站（6-web，推荐）

| `TRTFORGE_DIALOG_TIMEOUT` | 原生对话框最长等待秒数，超时按取消处理（默认 300） |
| `TRTFORGE_DIALOG_DISABLE` | 设为 `1` 关闭原生对话框，全部走页面内目录浏览 |
| `TRTFORGE_MODEL_DIR` | **v1.24**：内置模型仓库（SAM / GroundingDINO / BERT 分词表）目录，默认 `6-web/models` |
不用 Qt、不用编译 `TRTForge.exe`，浏览器里就能跑训练和推理。

### 为什么迁

Qt 版要求每台机器先装 Qt 6.8.3 + MSVC 才能编出界面。Web 版把界面层换成浏览器，训练仍走原来的 launcher、推理仍走原来编好的 DLL，**算法与推理实现一行未改**。
这样就绕开了「换一台机器就要重新配 Qt 工具链」的问题，也让 Linux / macOS 侧有机会用同一套界面。

> **2026-10-31 更正**：此前这里写"本机长期没有 Qt，界面一直不可用"。经排查那是误判——Qt 6.8.3 与 MSVC 其实都在，真正原因是 OpenCV 被硬编码到不存在的 `D:/thirdparty/opencv`。
> 现已修复并完成全量编译，`TRTForge.exe` 可正常启动（见 v1.9）。所以现在两条路都通：**Web 版免编译、适合日常**；**Qt 版适合离线单机、无浏览器环境**。

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
| 训练导出 | 选 YOLO / Anomalib、填参数（含优化器/学习率/workers/patience/amp/resume）、环境自检、实时日志、参数回显、阶段与 `epoch n/N` 进度、每轮耗时与预计剩余、每轮指标表、loss/mAP 内联 SVG 曲线、一键停止；v1.14 新增数据集概览（漏标/孤儿标/类别分布/分辨率）、增强预览（原图 vs 增强后）、断点续训（扫历史 run → 一键续训）；**v1.15 新增导出 ONNX**（四任务头 / `.transd.onnx` / 一键送去编译）；**v1.25 新增启动阶段清单与预热进程池（YOLO 1.56s→0.19s、Anomalib 4.32s→≈0s）、`lr0` 表单项、「用它微调 best.pt」（写回原目录 + `_prev/` 自动备份）** |
| 引擎构建 | 选 ONNX 自动探测形状、fp16/fp32、动态 shape、实时编译日志、8 阶段进度条 + 预计剩余（历史耗时自学习）、产物落盘校验 |
| 数据集 | 扫描 YOLO 目录 / `.tfanno`、固定哈希划分预览、导出 YOLO / COCO / VOC |
| 标注 | 打开 / 新建 `.tfanno`、六种形状绘制（矩形/旋转矩形/多边形/线/点/关键点组）、缩放平移吸附、撤销重做、3 秒自动保存、修订号冲突弹窗、类别与属性面板、自动标注草稿层、导出 YOLO / COCO / VOC |
| 推理演示 | 模型下拉自动发现产物、`conf`/`nms` 与 Anomalib 双阈值滑块、SVG 矢量框叠加、热能图色带、结果明细表（可排序/点行高亮）、置信度直方图、计时分解、JSON/CSV 导出；**v1.16 新增目录批量、视频与摄像头**（批量出 CSV/JSON/ZIP，视频出可内联播放的 mp4，摄像头逐帧只回 JSON 框） |
| 环境说明 | 展示后端探测到的解释器、launcher、DLL、trtexec 状态与环境变量 |

从 v1.11 起，**标注全流程都可以在浏览器完成**（画、改、存、导出），不再需要 Qt 版；
`.tfanno` 格式与 Qt 双向互通，历史工程可以直接打开继续标。

### 关键环境变量

| 变量 | 作用 |
| --- | --- |
| `TRTFORGE_PYTHON` | 训练用 `python.exe`，不设则自动探测 |
| `OPENCV_DIR` | OpenCV 根目录，如 `E:\thirdparty\opencv`。**推理 DLL 运行时必需，CMake 构建时也读它**（`-DOPENCV_DIR` 优先，其次本环境变量，都没有才交给 CMake 常规搜索） |
| `CUDA_PATH` | CUDA 根目录，推理 DLL 运行时必需 |
| `TRTFORGE_TRTEXEC` | `trtexec.exe` 路径，不设则从 `CUDA_PATH\bin` / CUDA 根 / PATH 自动找 |
| `TRTFORGE_BROWSE_ROOTS` | 额外允许浏览的目录（分号分隔），默认只允许仓库内 |
不用 Qt、不用编译 `TRTForge.exe`，浏览器里就能跑训练和推理。

### 为什么迁

Qt 版要求每台机器先装 Qt 6.8.3 + MSVC 才能编出界面。Web 版把界面层换成浏览器，训练仍走原来的 launcher、推理仍走原来编好的 DLL，**算法与推理实现一行未改**。
这样就绕开了「换一台机器就要重新配 Qt 工具链」的问题，也让 Linux / macOS 侧有机会用同一套界面。

> **2026-10-31 更正**：此前这里写"本机长期没有 Qt，界面一直不可用"。经排查那是误判——Qt 6.8.3 与 MSVC 其实都在，真正原因是 OpenCV 被硬编码到不存在的 `D:/thirdparty/opencv`。
> 现已修复并完成全量编译，`TRTForge.exe` 可正常启动（见 v1.9）。所以现在两条路都通：**Web 版免编译、适合日常**；**Qt 版适合离线单机、无浏览器环境**。

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
| 训练导出 | 选 YOLO / Anomalib、填参数（含优化器/学习率/workers/patience/amp/resume）、环境自检、实时日志、参数回显、阶段与 `epoch n/N` 进度、每轮耗时与预计剩余、每轮指标表、loss/mAP 内联 SVG 曲线、一键停止；v1.14 新增数据集概览（漏标/孤儿标/类别分布/分辨率）、增强预览（原图 vs 增强后）、断点续训（扫历史 run → 一键续训）；**v1.15 新增导出 ONNX**（四任务头 / `.transd.onnx` / 一键送去编译）；**v1.25 新增启动阶段清单与预热进程池（YOLO 1.56s→0.19s、Anomalib 4.32s→≈0s）、`lr0` 表单项、「用它微调 best.pt」（写回原目录 + `_prev/` 自动备份）** |
| 引擎构建 | 选 ONNX 自动探测形状、fp16/fp32、动态 shape、实时编译日志、8 阶段进度条 + 预计剩余（历史耗时自学习）、产物落盘校验 |
| 数据集 | 扫描 YOLO 目录 / `.tfanno`、固定哈希划分预览、导出 YOLO / COCO / VOC |
| 标注 | 打开 / 新建 `.tfanno`、六种形状绘制（矩形/旋转矩形/多边形/线/点/关键点组）、缩放平移吸附、撤销重做、3 秒自动保存、修订号冲突弹窗、类别与属性面板、自动标注草稿层、导出 YOLO / COCO / VOC |
| 推理演示 | 模型下拉自动发现产物、`conf`/`nms` 与 Anomalib 双阈值滑块、SVG 矢量框叠加、热能图色带、结果明细表（可排序/点行高亮）、置信度直方图、计时分解、JSON/CSV 导出；**v1.16 新增目录批量、视频与摄像头**（批量出 CSV/JSON/ZIP，视频出可内联播放的 mp4，摄像头逐帧只回 JSON 框） |
| 环境说明 | 展示后端探测到的解释器、launcher、DLL、trtexec 状态与环境变量 |

从 v1.11 起，**标注全流程都可以在浏览器完成**（画、改、存、导出），不再需要 Qt 版；
`.tfanno` 格式与 Qt 双向互通，历史工程可以直接打开继续标。

### 关键环境变量

| 变量 | 作用 |
| --- | --- |
| `TRTFORGE_PYTHON` | 训练用 `python.exe`，不设则自动探测 |
| `OPENCV_DIR` | OpenCV 根目录，如 `E:\thirdparty\opencv`。**推理 DLL 运行时必需，CMake 构建时也读它**（`-DOPENCV_DIR` 优先，其次本环境变量，都没有才交给 CMake 常规搜索） |
| `CUDA_PATH` | CUDA 根目录，推理 DLL 运行时必需 |
| `TRTFORGE_TRTEXEC` | `trtexec.exe` 路径，不设则从 `CUDA_PATH\bin` / CUDA 根 / PATH 自动找 |
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

### 训练超参速查（v1.7 起增强参数可调）

| 类别 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 基础 | `--epochs` / `--imgsz` / `--batch` / `--device` | 100 / 640 / 16 / 自动 | 训练规模 |
| 优化器 | `--optimizer` `--lr0` `--lrf` `--momentum` `--weight-decay` | auto / 0.01 / 0.01 / 0.937 / 0.0005 | 学习率与正则 |
| 冻结 | `--freeze` | 不冻结 | 冻结前 N 层，小数据集常用 |
| 颜色增强 | `--hsv-h` `--hsv-s` `--hsv-v` | 0.015 / 0.7 / 0.4 | 对应打光与色温差异 |
| 几何增强 | `--degrees` `--translate` `--scale` `--shear` `--perspective` | 0 / 0.1 / 0.5 / 0 / 0 | 对应拍摄位姿变化 |
| 翻转 | `--flipud` `--fliplr` `--bgr` | 0 / 0.5 / 0 | 左右翻转默认开启，有丝印时建议关掉 |
| 组合增强 | `--mosaic` `--mixup` `--copy-paste` `--close-mosaic` | 1.0 / 0 / 0 / 10 | **小缺陷必须降 mosaic** |

> 15 项增强参数的默认值与 Ultralytics `DEFAULT_CFG` 逐项一致；不填即行为不变。
> 小缺陷（如 PCB 70px）推荐组合：`--mosaic 0.1 --close-mosaic 20 --scale 0.3`。

### 断点续训（v1.14 新增 `--resume-from`）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--resume` | 关 | 靠 `get_latest_run()` **猜**「最近一次 run」，机器上并存多个项目时容易猜错 |
| `--resume-from` | 空 | **显式指定** `weights/last.pt` 的绝对路径；launcher 执行 `model.train(resume=该路径)`，同时把它当初始模型加载 |

```powershell
python tf_yolo_launcher.py --mode train `
  --model  runs\trtforge\my_yolo\weights\last.pt `
  --data   configs\pcb_detect.yaml `
  --resume-from runs\trtforge\my_yolo\weights\last.pt `
  --project runs\trtforge --name my_yolo
```

- 续训**写回原 run 目录**（Ultralytics `check_resume` 用检查点里记录的 `project/name`），`results.csv` 继续追加行。
  想开新实验就别续训，直接改 `--name`。
- Web 端只用 `--resume-from`：训练页「断点续训」会列出所有可续训的 run（已完成 / 计划轮次、最佳 mAP、
  `last.pt` 大小、不可续训原因），点「继续训练」自动填好表单。
- 扫描只读 `args.yaml` 与 `results.csv`，**不加载 `.pt`**，所以后端不需要 torch 也能列出历史实验。

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
