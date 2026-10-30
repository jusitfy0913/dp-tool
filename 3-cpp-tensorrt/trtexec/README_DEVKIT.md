# TRTForge 开发包说明

本包面向二次开发：可以改 GUI、跑训练、链接推理 DLL。
**不包含** 封装后的 TensorRT / OCR 推理实现源码，只提供已经编译好的 DLL。

## 目录结构

```text
trtforge_devkit/
  4-qt-studio/                     Qt 界面与引擎构建工程（可编译）
    src/                           Qt 业务源码
    include/                       业务头文件
    bin/                           运行时 DLL 与 TRTForge.exe
    bin/training/scripts/          训练启动脚本（编译时自动拷贝）
  3-cpp-tensorrt/
    yolo_infer/include/            YOLO 推理 C API 头文件
    anomalib_infer/include/        Anomalib 推理 C API 头文件
  ocr/ocr_infer/include/          OCR 推理 C API 头文件
  1-python-yolo/
    ultralytics-main/              YOLO 训练工程
    launcher/                      YOLO 训练启动脚本源码
  2-python-anomaly/
    anomalib-main/                 Anomalib 训练工程
    launcher/                      Anomalib 训练启动脚本源码
  docs/                            操作与接口说明
  README_DEVKIT.md                 本说明
```

解压后请保持 `4-qt-studio` 与 `1-python-yolo`、`2-python-anomaly` 在同一级目录。
训练启动脚本会逐级向上查找这两个训练工程，目录关系不要拆开。

## 包含内容

- TRTForge / DefectVision AI Studio 的 Qt 工程源码
- YOLO / Anomalib 训练工程，以及 `tf_yolo_launcher.py`、`tf_anomalib_launcher.py`
- `yolo_infer.dll`、`anomalib_infer.dll`、`ocr_infer.dll` 及对应 `.lib`、C API 头文件
- `4-qt-studio/bin` 运行时（Qt、OpenCV、TensorRT、CUDA、Paddle Inference 等），便于直接运行或继续编译

## 不包含内容

- `3-cpp-tensorrt/*/src` 等封装后的推理实现源码
- `ocr/ocr_infer/third_party`、推理 CMake 源码工程
- `4-qt-studio/build`、`4-qt-studio/logs`、Visual Studio 缓存
- Anomalib 的 `datasets`、`results` 历史数据
- 训练过程产生的 `runs` 目录

## 如何编译 GUI

1. 安装 Visual Studio 2022、Qt 6.8.3、CUDA 11.8、TensorRT、OpenCV。
2. 打开 `4-qt-studio`，按原工程 CMakePresets / CMakeLists 配置。
3. 主工程会导入 `4-qt-studio/bin` 里的三个预编译推理库，不再编译推理源码。
4. 编译产物仍输出到 `4-qt-studio/bin/TRTForge.exe`。

## 如何使用训练

1. 准备好本机 Python 环境（torch、ultralytics/anomalib、onnx 等）。
2. 在 TRTForge 训练页选择 Python 解释器后，按原流程训练 / 导出。
3. 也可直接运行：
   - `1-python-yolo/launcher/tf_yolo_launcher.py`
   - `2-python-anomaly/launcher/tf_anomalib_launcher.py`

## 如何调用推理 DLL

头文件：

- `3-cpp-tensorrt/yolo_infer/include/yolo_infer_c.h`
- `3-cpp-tensorrt/anomalib_infer/include/anomalib_infer_c.h`
- `ocr/ocr_infer/include/ocr_infer_c.h`

接口说明见 `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`。

## 重新打包

在完整源码仓库（含推理源码）中执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\3-cpp-tensorrt\trtexec\package_devkit.ps1
```
