# deeplearning 项目说明

本项目围绕工业视觉深度学习训练与 TensorRT 部署闭环建设，当前核心产品显示名为 **DefectVision AI Studio（AI 缺陷检测视觉工作站）**。项目核心方向是把标注、数据集导出、YOLO/Anomalib 训练、ONNX 导出、TensorRT 引擎构建和 C++ 推理验证整合为一套工业 AI 缺陷检测闭环平台。

## 主要模块

| 路径 | 说明 |
| --- | --- |
| `6-web` | **浏览器版工作站（推荐）**：FastAPI 后端 + 零构建前端，功能覆盖训练导出、引擎构建、数据集划分导出、标注工作台与推理演示，不需要 Qt。详见 `6-web/README.md` |
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
| `OPENCV_DIR` | OpenCV 根目录，CMake 构建（`4-qt-studio`、`ocr_infer`、两个推理子工程）与推理 DLL 运行时共用；`-DOPENCV_DIR` 可临时覆盖 |

`third_party/` 已加入 `.gitignore`，Paddle 源码与构建产物不会进入版本库。

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
| `2-python-anomaly/launcher/tf_anomalib_launcher.py` | 后处理契约分支接入阈值校准；`collect_images()` 改为多目录轮转取样 |
| `docs/异常检测模型对比与选型_PatchCore_PaDiM_EfficientAd.md` | 新增 |
| `docs/WORKFLOW_GUIDE.md` | 新增 |
| `readme.md` | 新增模型速查表、阈值校准说明、版本更新记录 |

### v1.1 — 2026-10-28（五级目录重构）

| 项目 | 老版本 | 新版本 |
| --- | --- | --- |
| 目录结构 | 根目录平铺 | 五级分层：`1-python-yolo` / `2-python-anomaly` / `3-cpp-tensorrt` / `4-qt-studio` / `5-reference`，另加 `6-web` 与 `ocr` |
| 界面形态 | Qt 桌面程序 `TRTForge.exe` | 浏览器工作站 `6-web`（FastAPI + 零构建前端）为**推荐入口**；Qt 版仍可完整编译运行（见 v1.9），作桌面备选 |
| 脚本路径 | 硬编码 `D:\test1\deeplearning\...` | 全部按脚本自身位置推导，可用环境变量覆盖 |
| `.gitignore` | `ultralytics/`、`anomalib/` 会误伤训练源码 | 改为锚定仓库根的 `/ultralytics/`、`/anomalib/` |

## Web 版工作站（6-web，推荐）

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
| 训练导出 | 选 YOLO / Anomalib、填参数（含优化器/学习率/workers/patience/amp/resume）、环境自检、实时日志、参数回显、阶段与 `epoch n/N` 进度、每轮耗时与预计剩余、每轮指标表、loss/mAP 内联 SVG 曲线、一键停止；v1.14 新增数据集概览（漏标/孤儿标/类别分布/分辨率）、增强预览（原图 vs 增强后）、断点续训（扫历史 run → 一键续训）；**v1.15 新增导出 ONNX**（四任务头 / `.transd.onnx` / 一键送去编译） |
| 引擎构建 | 选 ONNX 自动探测形状、fp16/fp32、动态 shape、实时编译日志、8 阶段进度条 + 预计剩余（历史耗时自学习）、产物落盘校验 |
| 数据集 | 扫描 YOLO 目录 / `.tfanno`、固定哈希划分预览、导出 YOLO / COCO / VOC |
| 标注 | 打开 / 新建 `.tfanno`、六种形状绘制（矩形/旋转矩形/多边形/线/点/关键点组）、缩放平移吸附、撤销重做、3 秒自动保存、修订号冲突弹窗、类别与属性面板、自动标注草稿层、导出 YOLO / COCO / VOC |
| 推理演示 | 模型下拉自动发现产物、`conf`/`nms` 与 Anomalib 双阈值滑块、SVG 矢量框叠加、热能图色带、结果明细表（可排序/点行高亮）、置信度直方图、计时分解、JSON/CSV 导出；**v1.16 新增目录批量、视频与摄像头**（批量出 CSV/JSON/ZIP，视频出可内联播放的 mp4，摄像头逐帧只回 JSON 框） |
| 全流程串联 | **v1.17 YOLO**：训练→导出→校验→编译→补标签→精度验证（mAP）；**v1.18 Anomalib**：训练+导出+校准→校验 metadata→编译→引擎精度验证（AUROC / 漏检 / 误报 / 最优阈值），支持跳过训练复用已有产物 |
| 环境说明 | 展示后端探测到的解释器、launcher、DLL、trtexec 状态与环境变量 |

**路径选择（v1.20）**：所有填文件/文件夹的地方点按钮就弹 Windows 系统对话框
（文件 / 多选 / 文件夹 / 另存为），`Shift` + 点击退回页面内目录浏览。
浏览器自身不能弹系统框，所以框由后端在本机桌面弹出，因此只对本机访问启用。
细节见 `6-web/README.md` 的「选路径：原生对话框（v1.20）」。

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
| `TRTFORGE_DIALOG_TIMEOUT` | 原生对话框最长等待秒数，超时按取消处理（默认 300） |
| `TRTFORGE_DIALOG_DISABLE` | 设为 `1` 关闭原生对话框，全部走页面内目录浏览 |

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
