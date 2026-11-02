# Change: 按用途五级重构仓库目录

## Why
仓库当前把 YOLO 训练源码、Anomalib 异常检测、TensorRT 推理 DLL、Qt 桌面程序、PaddleOCR 和早期 YOLOv8 对照工程混放在根目录，TensorRT 推理子工程还嵌在 `trtexec/gui/inference` 里。结果是：

- 看一眼根目录无法判断哪些目录参与编译、哪些只是参考代码。
- 训练启动脚本、CMake、打包脚本、文档各自维护一套相对路径，移动一次就要全仓搜索。
- CMake 的 `add_subdirectory` 和 DLL 输出目录没有单一出口，产物散落。
- 新同事（或接手的人）无法判断改动某个目录会不会影响 `TRTForge.exe`。

本次变更把「按用途五级分层」固化为项目结构约定，并把散落的路径引用收敛到一套规则，后续新增模块按层归位即可。

## What Changes
- 新增五级目录约定，层级名固定：
  - `1-python-yolo/`：`ultralytics-main/` 训练源码 + `launcher/` 启动脚本
  - `2-python-anomaly/`：`anomalib-main/` 训练源码 + `launcher/` 启动脚本
  - `3-cpp-tensorrt/`：`yolo_infer/`、`anomalib_infer/` 两个 TensorRT 推理 DLL 子工程，以及命令行工具 `trtexec/`
  - `4-qt-studio/`：Qt 6 桌面程序 `TRTForge.exe`，同时是全部编译产物的唯一出口 `4-qt-studio/bin`
  - `5-reference/`：`YOLOv8-det`、`YOLOv8-seg`、`YOLOv8_obb`、`YOLOv8-TensorRT-main` 早期对照工程
- 旁挂 `ocr/`：`PaddleOCR-main/` 上游源码 + `ocr_infer/` Paddle 推理 DLL 子工程，第五级之外单独归类，避免把 Paddle 依赖混进 TensorRT 层。
- 构建与打包脚本（`scripts/`、`3-cpp-tensorrt/trtexec/package_devkit.ps1`）不再写死某台机器的绝对路径，统一按脚本位置推导仓库根。
- **BREAKING**：根目录不再提供 `ultralytics-main`、`anomalib/`、`PaddleOCR-main`、`trtexec/`、`trtexec/gui` 等旧路径。外部脚本若硬编码这些路径必须改为五级路径。
- Python 启动脚本按「安装形态」与「源码形态」双候选解析，不再依赖单一相对深度。
- CMake 用 `TRTFORGE_REPO_ROOT` 统一推导同级目录，推理 DLL 只向 `4-qt-studio/bin` 输出。
- 清理 `.gitignore`、打包脚本、README、源码注释中残留的 `trtexec/gui/...` 旧路径。
- 命令行 `trtexec` 从仓库根并入 `3-cpp-tensorrt/trtexec`，与 TensorRT 推理层归位。
- `scripts/` 下的 Paddle 构建/同步脚本与 `anomalib-main/train.py`、`generate_trtforge_metadata.ps1` 去除 `D:\test1\deeplearning\...` 绝对路径，改为按脚本位置推导仓库根。
- 补充 `readme.md` 的新旧路径对照表，让使用者能按旧记忆找到新位置。

## Impact
- Affected specs: `project-layout`（新增能力）
- Affected code:
  - `1-python-yolo/launcher/tf_yolo_launcher.py`
  - `2-python-anomaly/launcher/tf_anomalib_launcher.py`
  - `3-cpp-tensorrt/yolo_infer/CMakeLists.txt`
  - `3-cpp-tensorrt/anomalib_infer/CMakeLists.txt`
  - `4-qt-studio/CMakeLists.txt`
  - `4-qt-studio/CMakePresets.json`
  - `4-qt-studio/src/training/YoloTrainController.cpp`
  - `4-qt-studio/tests/dll_api_smoke_test.py`
  - `4-qt-studio/test/01_asset_check.py`
  - `ocr/ocr_infer/CMakeLists.txt`
  - `3-cpp-tensorrt/trtexec/package_devkit.ps1`
  - `scripts/build_paddle_inference_win_gpu_trt.ps1`
  - `scripts/paddle_infer_build_sync_only.bat`
  - `scripts/paddle_infer_configure_build_sync.bat`
  - `scripts/sync_paddle_inference_to_bin.ps1`
  - `2-python-anomaly/anomalib-main/train.py`
  - `2-python-anomaly/anomalib-main/generate_trtforge_metadata.ps1`
  - `.gitignore`
- Affected docs:
  - `readme.md`
  - `4-qt-studio/README.md`
  - `3-cpp-tensorrt/trtexec/README.md`
  - `3-cpp-tensorrt/trtexec/README_DEVKIT.md`
  - `docs/TRTForge_操作手册.md`
  - `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`
  - `docs/YOLO推理_行步长与预处理修复说明.md`
  - `docs/Paddle_Inference_Windows_GPU_TRT_编译指南.md`
  - `docs/Qt_Cpp_Python训练平台_架构设计.md`
  - `docs/YOLOv8_TensorRT_部署工程总结.md`
  - `docs/DefectVision_AI_Studio_软件结构优化计划.md`
