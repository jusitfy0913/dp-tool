## 1. 规范与基线
- [x] 1.1 创建 `openspec/changes/refactor-five-tier-layout/` 目录，写入 `proposal.md`、`design.md`、`tasks.md`、`specs/project-layout/spec.md`
- [x] 1.2 运行 `openspec validate refactor-five-tier-layout --strict` 并修正到通过
- [x] 1.3 记录改造前 `git status` 与 `git log -1` 基线（基线提交 `23a6358`），确认搬迁可用 `git mv` 保留历史
- [x] 1.4 记录编译环境前置：Qt `D:/QT/6.8.3/msvc2022_64`（不存在）、OpenCV `D:/thirdparty/opencv`（不存在）、CUDA 11.8（存在，但与 MSVC 14.44 触发 STL1002 不兼容）

## 2. 一级 1-python-yolo
- [x] 2.1 `git mv ultralytics-main 1-python-yolo/ultralytics-main`
- [x] 2.2 新建 `1-python-yolo/launcher/` 并把 `tf_yolo_launcher.py` 迁入
- [x] 2.3 核对 `tf_yolo_launcher.py:129-142` 的 `add_ultralytics_source_path()` 候选包含 `1-python-yolo/ultralytics-main` 与安装形态 `ultralytics-main`
- [x] 2.4 用 `python 1-python-yolo/launcher/tf_yolo_launcher.py --mode check-env` 验证 `sys.path` 命中源码树
- [x] 2.5 确认 `1-python-yolo/ultralytics-main` 内的自定义脚本（`yolo_obb.py`、`yolo_pose_export_transd.py`）相对引用未被破坏

## 3. 二级 2-python-anomaly
- [x] 3.1 `git mv anomalib/anomalib-main 2-python-anomaly/anomalib-main`
- [x] 3.2 新建 `2-python-anomaly/launcher/` 并把 `tf_anomalib_launcher.py` 迁入
- [x] 3.3 核对 `tf_anomalib_launcher.py:58-74` 的 `add_anomalib_source_path()` 候选包含 `2-python-anomaly/anomalib-main` 及其 `src`
- [x] 3.4 验证 `trtforge_metadata.py` 与 anomalib `src` 可被 launcher 导入
- [x] 3.5 清理搬迁后残留的空目录 `anomalib/`

## 4. 三级 3-cpp-tensorrt
- [x] 4.1 `git mv trtexec/gui/inference/yolo_infer 3-cpp-tensorrt/yolo_infer`
- [x] 4.2 `git mv trtexec/gui/inference/anomalib_infer 3-cpp-tensorrt/anomalib_infer`
- [x] 4.3 核对 `3-cpp-tensorrt/yolo_infer/CMakeLists.txt:80` 与 `anomalib_infer/CMakeLists.txt:70` 的 `TRTFORGE_BIN_DIR` 指向 `../../4-qt-studio/bin`
- [x] 4.4 核对两个子工程的 `RUNTIME/ARCHIVE_OUTPUT_DIRECTORY` 均已指向该目录
- [x] 4.5 单独配置并编译 `yolo_infer`、`anomalib_infer`，确认产物落在 `4-qt-studio/bin`（已用 `build_infer_dlls.ps1` 绕过 Qt 依赖单独编译出 DLL）

## 5. 四级 4-qt-studio
- [x] 5.1 `git mv trtexec/gui 4-qt-studio`（保留 `bin` 产物目录与 `src/include` 结构不变）
- [x] 5.2 核对 `4-qt-studio/CMakeLists.txt:121-122` 的 `TRTFORGE_REPO_ROOT`、`TRTFORGE_BIN_DIR` 定义
- [x] 5.3 核对 `CMakeLists.txt:124-149` 的 `TRTFORGE_INFER_BLOCK`，确认引用 `3-cpp-tensorrt/*` 与 `ocr/ocr_infer`，且 `TRTFORGE_BUILD_OCR_INFER=OFF` 时走导入分支
- [x] 5.4 核对 `CMakeLists.txt:237-239` 的 include 列表指向 `3-cpp-tensorrt/*/include` 与 `ocr/ocr_infer/include`
- [x] 5.5 核对 `CMakeLists.txt:331-338` 的 launcher 拷贝源为 `1-python-yolo/launcher`、`2-python-anomaly/launcher`，目标为 `bin/training/scripts`
- [x] 5.6 修正 `src/training/YoloTrainController.cpp` 的 `launcherPath()` 解析顺序与候选路径（去掉重复项，补 `appDir/../../` 源码形态）
- [x] 5.7 修正 `tests/dll_api_smoke_test.py` 与 `test/01_asset_check.py` 的 `5-reference` 与 anomalib 数据集资产路径
- [x] 5.8 **完整编译通过**：`cmake --preset vs2022-x64` → `Configuring done` / `Generating done`；`yolo_infer`、`anomalib_infer`、`ocr_infer`、`TRTForge` 四个目标全部产出到 `4-qt-studio/bin`，`TRTForge.exe` 实测可启动（361 MB 常驻）。此前记录的「本机工具链阻塞」实为**误判**，真因是 OpenCV 硬编码到不存在的 `D:/thirdparty/opencv`（本机实际在 `E:\thirdparty\opencv`），且 preset 里的 `OPENCV_DIR` 覆盖了环境变量；已在 7.5 一并修复。OCR 缺失的 `yaml-cpp`/Paddle 依赖见 7.4。

## 6. 五级 5-reference
- [x] 6.1 `git mv YOLOv8-det YOLOv8-seg YOLOv8_obb YOLOv8-TensorRT-main 5-reference/`
- [x] 6.2 全仓确认 `5-reference` 未被任何 `CMakeLists.txt` 的 `add_subdirectory` 引用
- [x] 6.3 核对 `3-cpp-tensorrt/yolo_infer/src/yolo.cu:5`、`yolo.hpp:5` 与 `4-qt-studio/src/InferenceTab.cpp:302,1528` 的对照注释指向 `5-reference/...` 新路径

## 6b. 命令行 trtexec 并入三级
- [x] 6b.1 `git mv trtexec 3-cpp-tensorrt/trtexec`
- [x] 6b.2 把 `package_devkit.ps1` 的仓库根推导从 `Split-Path -Parent $scriptDir` 改为向上探测 `4-qt-studio` + `1-python-yolo`，不再依赖固定层级
- [x] 6b.3 更新 `package_devkit.ps1` 头部用法示例与 `3-cpp-tensorrt/trtexec/README_DEVKIT.md` 的运行命令
- [x] 6b.4 更新 `3-cpp-tensorrt/trtexec/README.md` 的 GUI 入口链接为 `../../4-qt-studio`
- [x] 6b.5 更新全仓对 `.\trtexec\package_devkit.ps1` 的引用（`readme.md`、`4-qt-studio/README.md`、`docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`）

## 7. 旁挂 ocr
- [x] 7.1 `git mv PaddleOCR-main ocr/PaddleOCR-main`、`git mv trtexec/gui/inference/ocr_infer ocr/ocr_infer`
- [x] 7.2 核对 `ocr/ocr_infer/CMakeLists.txt:34` 的 `PADDLEOCR_CPP_INFER_DIR` 指向 `${REPO_ROOT}/ocr/PaddleOCR-main/deploy/cpp_infer`
- [x] 7.3 确认 `ocr_infer` 输出目录仍为 `4-qt-studio/bin`
- [x] 7.4 **OCR 全链路编译通过**（2026-10-31）。实测命令：`cmake --preset vs2022-x64 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" "-DPADDLE_LIB=<...>/third_party/paddle_inference"`。成功产出 `4-qt-studio/bin/{ocr_infer.dll, ocr_infer.lib, ocr_basic.exe}`，并由此解锁 `TRTForge.exe` 链接。过程中修掉三处真实缺陷：
  - **`third_party/yaml-cpp` 是空目录**：它在 git 索引里登记为 gitlink（`160000 0579ae3`）但仓库缺 `.gitmodules`，属于损坏的子模块登记。已确认内部仓库 HEAD 恰为 `0579ae3`（tag `yaml-cpp-0.7.0`，远端 `jbeder/yaml-cpp`），源码就位后配置即通过。
  - **CMake 4.2 与 clipper 不兼容**：clipper 写的是 `cmake_minimum_required(VERSION 2.6.0)`，用 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 绕过（PowerShell 里必须加引号，否则 `=3.5` 会被拆成独立参数）。
  - **链接搜索目录漏了 `mkldnn/lib`**：`CMakeLists.txt` 只列了 `onednn/lib`，而官方 GPU 包实际是 `mkldnn`。已补上 include 与 lib 两条路径。
  - 另新增 POST_BUILD，把 `paddle_inference.dll` / `common.dll` / `mkldnn.dll` / `libiomp5md.dll` / `yaml-cpp.dll` / `polyclipping.dll` 自动拷到输出目录，避免 `TRTForge.exe` 启动报 `0xC0000135`（缺依赖）。

## 8. 横切与验收
- [x] 8.1 清理 `.gitignore` 的 `trtexec/gui/*` 旧条目，只保留 `4-qt-studio/build|dist|logs|bin`
- [x] 8.2 核对 `3-cpp-tensorrt/trtexec/package_devkit.ps1` 的五级拷贝清单（`1-python-yolo`、`2-python-anomaly`、`3-cpp-tensorrt`、`4-qt-studio`、`ocr`）与 devkit 目标结构一致
- [x] 8.3 更新 `scripts/paddle_infer_build_sync_only.bat`、`paddle_infer_configure_build_sync.bat` 的日志文案与目标目录
- [x] 8.4 重命名 `scripts/sync_paddle_inference_to_gui_bin.ps1` 为 `sync_paddle_inference_to_bin.ps1`，`$GuiBin` 改为 `$BinDir`
- [x] 8.4b 把 `scripts/build_paddle_inference_win_gpu_trt.ps1`、`scripts/sync_paddle_inference_to_bin.ps1`、两个 `.bat` 的 `D:\test1\deeplearning\...` 绝对路径改为按脚本位置推导，并支持 `PADDLE_THIRD_PARTY_ROOT` / `PADDLE_WORK_ROOT` / `PADDLE_SRC` / `PADDLE_BUILD` 环境变量覆盖
- [x] 8.4c 新增 `-SkipConfigure` 开关，让 `paddle_infer_build_sync_only.bat` 真正做到「只编译不配置」
- [x] 8.4d 把 `2-python-anomaly/anomalib-main/train.py` 与 `generate_trtforge_metadata.ps1` 的 `D:\test1\deeplearning\...` 改为脚本同级 `datasets/bottle` 推导，并支持 `TRTFORGE_ANOMALIB_DATASET_ROOT` 覆盖
- [x] 8.4e 把两个 `.bat` 的注释改为纯 ASCII（内容仍是 UTF-8 兼容），避免 cmd.exe 按 ANSI 解析中文注释乱码；`generate_trtforge_metadata.ps1` 统一为 UTF-8 带 BOM，兼容 Windows PowerShell 5.1
- [x] 8.4f `.gitignore` 增加 `third_party/`，覆盖 Paddle 源码与构建工作区
- [x] 8.5 更新 `3-cpp-tensorrt/trtexec/README_DEVKIT.md` 的 devkit 目录说明为 `4-qt-studio` 结构
- [x] 8.6 更新 `4-qt-studio/README.md` 与 `4-qt-studio/CMakeLists.txt` 注释中的 `gui/` 旧表述
- [x] 8.7 更新 `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`、`docs/YOLO推理_行步长与预处理修复说明.md`、`docs/trtexec_gui_YOLO_TRT推理流程说明.md`、`docs/Paddle_Inference_Windows_GPU_TRT_编译指南.md` 中的现行路径
- [x] 8.8 更新 `readme.md` 的新旧路径对照表，覆盖五级目录、`ocr`、`trtexec`、`scripts`、`docs`
- [x] 8.9 全仓校验：旧路径仅剩 `readme.md` 中明确标注为「原来的路径」的历史对照
- [x] 8.10 运行 `openspec validate refactor-five-tier-layout --strict` 通过
- [ ] 8.11 用 `git add -A` 与 `git status` 确认重命名被识别，提交信息说明为目录结构重构（提交前需用户确认）

## 9. 验证记录（2026-10-28）

| 项 | 结果 | 证据 |
| --- | --- | --- |
| OpenSpec 严格校验 | 通过 | `Change 'refactor-five-tier-layout' is valid` |
| 旧路径扫描 | 通过 | `rg -n "trtexec/gui\|gui/bin\|gui/training"` 仅命中 `readme.md` 的新旧对照表 |
| 参考工程未入构建 | 通过 | `rg -n "5-reference" --glob CMakeLists.txt` 零命中 |
| CMake 引用路径存在性 | 通过 | `3-cpp-tensorrt/*`、`ocr/*`、两个 launcher、两个训练源码树、`5-reference` 测试图均 `Test-Path = True` |
| YOLO launcher 源码形态 | 通过 | `check-env` 输出 `sys_path0 = dp-tool\1-python-yolo\ultralytics-main` |
| Anomalib launcher 源码形态 | 通过 | `check-env` 输出 `anomalib: ok 2.0.0b3` |
| Python 语法 | 通过 | 4 个改动脚本 `py_compile` 通过；`train.py` 改后仍通过 |
| 绝对路径清除 | 通过 | `rg -n "test1" scripts 2-python-anomaly/anomalib-main` 仅命中上游测试文件的 `test111` 字符串 |
| 同步脚本推导 | 通过 | 实跑报错信息为 `dp-tool\third_party\Paddle\build_infer_trt\paddle_inference_install_dir`，证明仓库根推导正确 |
| 构建脚本推导 | 通过 | 实跑越过 cmake/git/python 检查后停在 `cuDNN not found`（本机无 cuDNN），说明工具链与路径解析均生效 |
| 打包脚本推导 | 通过 | 移动后实跑报错定位到 `dp-tool\4-qt-studio\bin\yolo_infer.dll`，证明 `Find-RepoRoot` 正确 |
| PowerShell 语法 | 通过 | 3 个 `.ps1` 解析器校验全部 OK |
| Python 环境自检（Anomalib） | 通过 | `tf_anomalib_launcher.py --mode check-env` 输出 `"missing":[]`（torch 2.3.0 / anomalib 2.0.0b3 / lightning 2.5.0 / CUDA 可用） |
| Python 环境自检（YOLO） | 通过 | 补装 `ultralytics 8.4.192`、`polars 2.1.0` 后 `--mode check-env` 输出 `"missing":[]`，`numpy 1.26.4` 与 `torch 2.3.0` 未被升级 |
| `.gitignore` 源码误伤 | 已修复 | `ultralytics/`、`anomalib/` 改为 `/ultralytics/`、`/anomalib/`，`git check-ignore` 确认训练源码不再被忽略 |
| CMake 配置 / 编译 | 未验证 | 本机无 Qt 6.8.3 与 OpenCV；CUDA 11.8 与 MSVC 14.44 在 `project()` 阶段报 `STL1002` |
| 提交 | 待确认 | 未执行 `git commit` |
