# TRTForge 开发包说明

本包面向二次开发：可以改 GUI、跑训练、链接推理 DLL。
**不包含** 封装后的 TensorRT / OCR 推理实现源码，只提供已经编译好的 DLL。

## 目录结构

```text
trtforge_devkit/
  trtexec/                      GUI 与引擎构建工程（可编译）
    gui/src                     Qt 界面与业务源码
    gui/training/scripts        训练启动脚本
    gui/inference/*/include     推理 C API 头文件
    gui/bin/*_infer.dll         预编译推理库
    gui/bin/*_infer.lib         链接用导入库
  ultralytics-main/             YOLO 训练工程
  anomalib/anomalib-main/       Anomalib 训练工程
  docs/                         操作与接口说明
  README_DEVKIT.md              本说明
```

解压后请保持 `trtexec`、`ultralytics-main`、`anomalib` 在同一级目录。
训练脚本会向上查找这两个训练工程，目录关系不要拆开。

## 包含内容

- TRTForge / DefectVision AI Studio 的 Qt 工程源码
- YOLO / Anomalib 训练工程，以及 `tf_yolo_launcher.py`、`tf_anomalib_launcher.py`
- `yolo_infer.dll`、`anomalib_infer.dll`、`ocr_infer.dll` 及对应 `.lib`、C API 头文件
- `gui/bin` 运行时（Qt、OpenCV、TensorRT、CUDA、Paddle Inference 等），便于直接运行或继续编译

## 不包含内容

- `trtexec/gui/inference/*/src` 等封装后的推理实现源码
- `ocr_infer/third_party`、推理 CMake 源码工程
- `gui/build`、`gui/logs`、Visual Studio 缓存
- Anomalib 的 `datasets`、`results` 历史数据
- 训练过程产生的 `runs` 目录

## 如何编译 GUI

1. 安装 Visual Studio 2022、Qt 6.8.3、CUDA 11.8、TensorRT、OpenCV。
2. 打开 `trtexec/gui`，按原工程 CMakePresets / CMakeLists 配置。
3. 主工程会导入 `gui/bin` 里的三个预编译推理库，不再编译推理源码。
4. 编译产物仍输出到 `trtexec/gui/bin/TRTForge.exe`。

## 如何使用训练

1. 准备好本机 Python 环境（torch、ultralytics/anomalib、onnx 等）。
2. 在 TRTForge 训练页选择 Python 解释器后，按原流程训练 / 导出。
3. 也可直接运行：
   - `trtexec/gui/training/scripts/tf_yolo_launcher.py`
   - `trtexec/gui/training/scripts/tf_anomalib_launcher.py`

## 如何调用推理 DLL

头文件：

- `trtexec/gui/inference/yolo_infer/include/yolo_infer_c.h`
- `trtexec/gui/inference/anomalib_infer/include/anomalib_infer_c.h`
- `trtexec/gui/inference/ocr_infer/include/ocr_infer_c.h`

接口说明见 `docs/TRTForge_yolo_infer与anomalib_infer_DLL调用与源码打包说明.md`。

## 重新打包

在完整源码仓库（含推理源码）中执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\trtexec\package_devkit.ps1
```
