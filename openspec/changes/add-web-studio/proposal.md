# Change: 用 Web 界面替代 Qt 桌面程序

## Why

当前所有界面能力都锁在 `4-qt-studio`（Qt 6 桌面程序）里，带来三个实际成本：

1. **环境重**：任何一台机器想用界面，都得先装 Qt 6.8.3 + MSVC 才编得出来。本机至今没有 Qt，`TRTForge.exe` 从未编译成功，界面完全不可用。
2. **不可远程**：训练要跑几小时，只能守着那台机器；换机就得重新配环境。
3. **改动脆**：标注、训练、引擎构建、推理四个页面耦合在一个 C++ 工程里，每改一处都要重新编译整个程序。

已实测确认的关键事实（决定了迁移代价）：

- 三个推理库 `yolo_infer`、`anomalib_infer`、`ocr_infer` **都不依赖 Qt**（`find_package(Qt)` 零命中），是独立 CMake 工程。
- `anomalib_infer.dll` 与 `yolo_infer.dll` 已在无 Qt 环境下编译成功，并能被 Python `ctypes` 直接加载。
- 训练链路（两个 launcher）本就与界面无关，已通过 `--mode check-env` 验证 `missing:[]`。

因此界面层可以整体替换，**底层能力一行不用重写**。

## What Changes

- 新增 `6-web/` 一级目录：`6-web/backend`（Python/FastAPI）+ `6-web/frontend`（静态页面，无构建步骤）。
- 后端复用现有能力，不重写算法：
  - 训练：`subprocess` 拉起 `1-python-yolo/launcher`、`2-python-anomaly/launcher`，解析 `TRTFORGE_JSON` 事件流并通过 WebSocket 推给浏览器。
  - 推理：`ctypes` 加载 `4-qt-studio/bin` 下的 `yolo_infer.dll` / `anomalib_infer.dll`。
  - 引擎构建：调用 `3-cpp-tensorrt/trtexec`（命令行）完成 ONNX → `.trt`。
- 前端复刻四页：标注数据、训练导出、引擎构建、推理演示。
- `4-qt-studio` 整体移入 `5-reference/`，保留可查，不再参与任何构建；`TRTForge.exe` 停止维护。
- 编译脚本改为按环境变量定位 OpenCV（`OPENCV_DIR`），不再写死 `D:/thirdparty/opencv`；补记 CUDA 11.8 + MSVC 14.44 的 `NVCC_PREPEND_FLAGS` 绕过方式。
- **BREAKING**：界面不再有 `.exe`；所有界面操作改为浏览器访问。

## Impact

- 受影响能力：新增 `web-studio` 能力；`project-layout` 增加 `6-web` 层级。
- 受影响代码：
  - 新增 `6-web/backend/app.py`、`6-web/frontend/`
  - `3-cpp-tensorrt/yolo_infer/CMakeLists.txt`、`3-cpp-tensorrt/anomalib_infer/CMakeLists.txt`（OpenCV 路径改环境变量）
  - `4-qt-studio` → `5-reference/4-qt-studio`
  - `readme.md`、`docs/`、`.gitignore`
- 不受影响：两个 Python launcher、三个推理库的 C API 与实现源码、`trtexec`。
