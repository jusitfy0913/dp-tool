## ADDED Requirements

### Requirement: Five-Tier Repository Layout
仓库 SHALL 使用按用途划分的五级目录表达工程结构，层级名称固定为 `1-python-yolo`、`2-python-anomaly`、`3-cpp-tensorrt`、`4-qt-studio`、`5-reference`，并 SHALL 额外保留旁挂目录 `ocr`。

#### Scenario: 根目录层级可辨识
- **WHEN** 开发者列出仓库根目录的一级子目录
- **THEN** 每个参与构建的目录 SHALL 带有 `1-` 到 `4-` 的数字前缀，`5-reference` SHALL 明确标注为不参与构建

#### Scenario: 训练源码归位
- **WHEN** 查找 YOLO 训练源码与启动脚本
- **THEN** 它们 SHALL 位于 `1-python-yolo/ultralytics-main` 与 `1-python-yolo/launcher`

#### Scenario: 异常检测源码归位
- **WHEN** 查找 Anomalib 训练源码与启动脚本
- **THEN** 它们 SHALL 位于 `2-python-anomaly/anomalib-main` 与 `2-python-anomaly/launcher`

#### Scenario: TensorRT 推理子工程归位
- **WHEN** 查找 `yolo_infer` 与 `anomalib_infer` 子工程
- **THEN** 它们 SHALL 位于 `3-cpp-tensorrt/yolo_infer` 与 `3-cpp-tensorrt/anomalib_infer`

#### Scenario: 命令行 trtexec 归位
- **WHEN** 查找命令行版 `trtexec` 源码与开发包脚本
- **THEN** 它们 SHALL 位于 `3-cpp-tensorrt/trtexec`，仓库根目录 SHALL 不再存在 `trtexec` 目录

#### Scenario: Qt 程序与参考工程分离
- **WHEN** 查找 Qt 桌面程序与早期 YOLOv8 对照工程
- **THEN** Qt 程序 SHALL 位于 `4-qt-studio`，对照工程 SHALL 位于 `5-reference` 且不被任何 CMake 目标引用

#### Scenario: OCR 独立归类
- **WHEN** 查找 PaddleOCR 上游源码与 `ocr_infer` 子工程
- **THEN** 它们 SHALL 位于 `ocr/PaddleOCR-main` 与 `ocr/ocr_infer`，不出现在 `3-cpp-tensorrt` 下

### Requirement: Launcher Path Resolution
训练启动脚本 SHALL 同时支持源码形态与安装形态两种目录布局，不得依赖单一相对深度。

#### Scenario: 源码形态解析 YOLO 源码树
- **WHEN** `tf_yolo_launcher.py` 从 `1-python-yolo/launcher` 启动
- **THEN** 它 SHALL 把 `1-python-yolo/ultralytics-main` 加入 `sys.path`

#### Scenario: 源码形态解析 Anomalib 源码树
- **WHEN** `tf_anomalib_launcher.py` 从 `2-python-anomaly/launcher` 启动
- **THEN** 它 SHALL 把 `2-python-anomaly/anomalib-main` 及其 `src` 加入 `sys.path`

#### Scenario: 安装形态解析启动脚本
- **WHEN** 启动脚本已被拷到 `4-qt-studio/bin/training/scripts`
- **THEN** Qt 侧 SHALL 能从应用目录下的 `training/scripts` 找到并启动脚本

#### Scenario: 从仓库根工作目录启动
- **WHEN** 当前工作目录是仓库根且可执行文件位于 `4-qt-studio/bin`
- **THEN** Qt 侧 SHALL 按 `1-python-yolo/launcher` 与 `2-python-anomaly/launcher` 候选路径解析启动脚本

#### Scenario: 候选路径全部缺失
- **WHEN** 所有候选路径都不存在
- **THEN** 训练启动 SHALL 失败并返回可读错误，指出期望的脚本位置

### Requirement: Single Build Output Directory
全部编译产物 SHALL 只输出到 `4-qt-studio/bin`。

#### Scenario: 生成推理 DLL
- **WHEN** CMake 配置并编译 `yolo_infer` 或 `anomalib_infer`
- **THEN** 生成的 `.dll` 与 `.lib` SHALL 落在 `4-qt-studio/bin`

#### Scenario: 生成 OCR DLL
- **WHEN** `TRTFORGE_BUILD_OCR_INFER=ON` 且编译 `ocr_infer`
- **THEN** 生成的 `ocr_infer.dll` 与 `ocr_infer.lib` SHALL 落在 `4-qt-studio/bin`

#### Scenario: 主工程引用同级子工程
- **WHEN** 配置 `4-qt-studio/CMakeLists.txt`
- **THEN** 它 SHALL 通过 `TRTFORGE_REPO_ROOT` 推导 `3-cpp-tensorrt/*` 与 `ocr/ocr_infer`，不得使用 `../../trtexec/gui/...` 形式

#### Scenario: 单独配置推理子工程
- **WHEN** 开发者只配置 `3-cpp-tensorrt/yolo_infer` 作为顶层工程
- **THEN** 输出目录变量 SHALL 仍解析到 `4-qt-studio/bin`

### Requirement: No Legacy Path Residue
参与构建的代码、配置与现行文档 SHALL 不再引用旧目录路径 `trtexec/gui`、根目录 `ultralytics-main`、`anomalib/anomalib-main`、根目录 `PaddleOCR-main`。

#### Scenario: 旧路径检索为零
- **WHEN** 在仓库内（排除 `5-reference` 与明确标注的历史说明）检索 `trtexec/gui`
- **THEN** 构建脚本、CMake、`.gitignore` 与 Qt 源码 SHALL 零命中

#### Scenario: 忽略规则收敛
- **WHEN** 查看 `.gitignore`
- **THEN** 构建目录、日志目录与 `bin` 的忽略条目 SHALL 指向 `4-qt-studio`，旧 `trtexec/gui` 条目 SHALL 被移除

#### Scenario: 打包脚本适配新层级
- **WHEN** 运行 `3-cpp-tensorrt/trtexec/package_devkit.ps1`
- **THEN** 它 SHALL 从 `1-python-yolo`、`2-python-anomaly`、`3-cpp-tensorrt`、`4-qt-studio`、`ocr` 拷贝内容，并按相同层级写入 devkit

#### Scenario: 打包脚本不依赖固定层级
- **WHEN** `package_devkit.ps1` 被移动到不同的子目录后再次运行
- **THEN** 它 SHALL 通过特征目录探测仓库根，仍能定位 `4-qt-studio` 与训练工程

#### Scenario: 测试脚本资产可解析
- **WHEN** 运行 `4-qt-studio/tests/dll_api_smoke_test.py` 或 `4-qt-studio/test/01_asset_check.py`
- **THEN** 测试图片路径 SHALL 按 `5-reference/...` 正确解析

### Requirement: No Machine-Specific Absolute Paths
构建与训练脚本 SHALL 通过脚本自身位置推导仓库内路径，不得包含只对某一台机器成立的绝对路径。

#### Scenario: Paddle 构建脚本推导路径
- **WHEN** 运行 `scripts/build_paddle_inference_win_gpu_trt.ps1` 且未传任何路径参数
- **THEN** Paddle 源码目录 SHALL 解析到 `<repo>/third_party/Paddle`，构建目录 SHALL 解析到 `<repo>/third_party/Paddle/build_infer_trt`

#### Scenario: Paddle 同步脚本推导路径
- **WHEN** 运行 `scripts/sync_paddle_inference_to_bin.ps1` 且未传任何路径参数
- **THEN** 安装目录 SHALL 从 `<repo>/third_party` 派生，目标目录 SHALL 为 `<repo>/4-qt-studio/bin`

#### Scenario: 批处理脚本推导仓库根
- **WHEN** 运行 `scripts/paddle_infer_configure_build_sync.bat` 或 `paddle_infer_build_sync_only.bat`
- **THEN** 仓库根 SHALL 由 `%~dp0..` 推导，日志目录 SHALL 落在 `<repo>/third_party/Paddle/build_logs`

#### Scenario: 训练脚本推导数据集
- **WHEN** 运行 `2-python-anomaly/anomalib-main/train.py` 且未设置覆盖变量
- **THEN** 数据集根目录 SHALL 解析到 `anomalib-main/datasets/bottle`

#### Scenario: 用环境变量覆盖默认路径
- **WHEN** 设置了 `PADDLE_THIRD_PARTY_ROOT` 或 `TRTFORGE_ANOMALIB_DATASET_ROOT`
- **THEN** 对应脚本 SHALL 使用该环境变量指定的目录，而不是默认推导结果

### Requirement: Documentation Tracks Layout
现行文档 SHALL 与五级目录保持一致，并 SHALL 为旧路径提供对照说明。

#### Scenario: 根 readme 提供新旧对照
- **WHEN** 开发者阅读 `readme.md`
- **THEN** 它 SHALL 含一张表格，逐项列出五级目录、旧路径与用途

#### Scenario: devkit 说明同步
- **WHEN** 开发者阅读 `3-cpp-tensorrt/trtexec/README_DEVKIT.md`
- **THEN** 目录说明 SHALL 使用 `4-qt-studio/bin` 与 `4-qt-studio/training/scripts`，不再出现 `gui/` 旧结构

#### Scenario: 操作手册路径一致
- **WHEN** 开发者按 `docs/TRTForge_操作手册.md` 定位安装目录内容
- **THEN** 其中引用的 DLL 与脚本路径 SHALL 与五级布局实际产物位置一致
