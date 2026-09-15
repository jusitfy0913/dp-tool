# Windows 下编译 Paddle Inference（GPU + TensorRT）

本文说明如何在本机编译带 **GPU** 与 **TensorRT 子图** 的 Paddle Inference C++ 库，并替换 TRTForge OCR 使用的 `trtexec/gui/bin/paddle_inference`，以解决界面选择 `gpu:0` 时出现 `Please use PaddlePaddle with GPU version` 的问题。

官方文档（步骤参考，分支已过时）：[Windows 下从源码编译](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Windows.html)

## 1. 本机环境（已探测）

| 项目 | 路径 / 版本 |
|------|-------------|
| Visual Studio | 2022 Community（文档写 VS2017，本机用 2022 + vcvars64） |
| CMake | `C:\Program Files\CMake\bin\cmake.exe` 3.29 |
| Git | `C:\Program Files\Git\cmd\git.exe` |
| Python | `D:\Anaconda\python.exe` 3.12 |
| CUDA | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |
| TensorRT 开发包 | 与 CUDA 同目录（`include\NvInfer.h`、`lib\nvinfer.lib`） |
| cuDNN | `D:\cudnntensort安装包\cudnn-windows-x86_64-8.6.0.163_cuda11-archive` |

说明：`pip install paddlepaddle-gpu` **不能**替代 C++ `paddle_inference.dll`，TRTForge 必须换 **C++ 推理库**。

## 2. 与 PaddleOCR / TRTForge 的版本关系

- PaddleOCR 当前为 **3.3.x**，`cpp_infer` 的 `run_mode` 支持：`paddle`、`paddle_fp16`、`mkldnn`、`mkldnn_bf16`（见 `pp_option.h`）。
- 建议 Paddle 源码分支：**`release/3.3`**（与 OCR 大版本对齐）。
- 编译选项 **`-DWITH_TENSORRT=ON`** 会在 `paddle_inference` 中启用 TensorRT 子图；OCR pipeline 首版仍以 **GPU 原生 / FP16** 为主，TRT 加速需在 `Config` 层额外开启（后续可在 `ocr_infer` 中扩展）。

## 3. 一键脚本

在 **「适用于 VS 2022 的 x64 本机工具命令提示符」**（管理员可选）中执行：

```powershell
cd D:\test1\deeplearning
powershell -ExecutionPolicy Bypass -File scripts\build_paddle_inference_win_gpu_trt.ps1
```

英文脚本说明见同目录 `docs/Paddle_Inference_Windows_GPU_TRT_compile_guide.md`（内容与本文一致，便于 ASCII 环境）。

一键全流程（子模块 + 配置 + 编译 + 同步）：

```bat
start "" cmd /c D:\test1\deeplearning\scripts\paddle_infer_configure_build_sync.bat
```

若 **configure 已成功**（存在 `build_infer_trt\Paddle.sln`），仅后台编译并同步：

```bat
start "" cmd /c D:\test1\deeplearning\scripts\paddle_infer_build_sync_only.bat
```

日志：`third_party\Paddle\build_logs\paddle_infer_build.log`  
同步脚本：`scripts\sync_paddle_inference_to_gui_bin.ps1`（保留 `ch_PP-OCRv4_*` 模型目录）。

仅做 CMake 配置：

```bat
D:\test1\deeplearning\third_party\Paddle\configure_infer_trt.bat
```

自定义路径示例：

```powershell
$env:CUDNN_ROOT="D:\cudnntensort安装包\cudnn-windows-x86_64-8.6.0.163_cuda11-archive"
$env:TENSORRT_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PYTHON_EXE="D:\Anaconda\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\build_paddle_inference_win_gpu_trt.ps1
```

## 4. 核心 CMake 参数（GPU + TRT）

与官方页面对应关系：

| 选项 | 值 | 说明 |
|------|-----|------|
| `WITH_GPU` | ON | GPU 推理 |
| `WITH_TENSORRT` | ON | TensorRT 子图 |
| `TENSORRT_ROOT` | CUDA v11.8 根目录 | 本机 TRT 与 CUDA 合并安装 |
| `ON_INFER` | ON | 预测库优化 |
| `WITH_PYTHON` | OFF | 只编 C++ 推理库，缩短时间 |
| `WITH_TESTING` | OFF | 关闭单测 |
| `CUDA_ARCH_NAME` | Auto | 仅编本机 GPU 架构 |

## 5. 编译成功后替换 TRTForge 库

1. 在构建目录找到：`third_party\Paddle\build_infer_trt\paddle_inference_install_dir`
2. **备份** 现有 `trtexec/gui/bin/paddle_inference`（尤其 `ch_PP-OCRv4_*` 模型目录）
3. 复制 install_dir 中的：
   - `paddle/lib/paddle_inference.lib`、`paddle_inference.dll`
   - `paddle/include/**`
   - `third_party` 下 MKL/cuDNN/CUDA 相关 DLL（与官方 CPU 包布局对齐）
4. 重新编译 `ocr_infer.dll` 与 `TRTForge.exe`
5. 确认 `bin` 下 CUDA/cuDNN/TRT 运行时 DLL 完整（可与现有 TRTForge 打包逻辑共用）

## 6. 验收

```text
ocr_infer 测试或 TRTForge OCR 页：
  device=gpu:0, run_mode=paddle 或 paddle_fp16
  不再出现 Please use PaddlePaddle with GPU version
  val_0633.jpg 总耗时应明显低于 CPU+MKLDNN 的 ~13s（视 GPU 而定）
```

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| cmake 找不到 Python | 设置 `PYTHON_EXE=D:\Anaconda\python.exe` |
| 找不到 cuDNN | 设置 `CUDNN_ROOT` 为解压后的 `cudnn-windows-*` 目录 |
| 找不到 NvInfer | 设置 `TENSORRT_ROOT` 为含 `include\NvInfer.h` 的目录（本机可用 CUDA 根目录） |
| VS 版本报错 | 使用 VS2022 x64 Native Tools，脚本已调用 `vcvars64.bat` |
| 编译极慢 / 内存不足 | 使用 `-Parallel 2` 或 `-ConfigureOnly` 先确认配置通过 |
| TensorRT 版本 | 本机 **8.5.2** + CUDA **11.8**；与 Linux 预编译包 TRT 8.6 略有差异，以本机为准 |

## 8. 与 OpenSpec 任务的关系

完成 GPU 库替换后，可继续 OpenSpec `add-paddleocr-cpp-inference` 中的 **GPU 端到端验收** 与 UI 列表行高亮等多边形交互。
