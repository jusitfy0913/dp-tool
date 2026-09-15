# anomalib_infer DLL 使用说明

`anomalib_infer` 是 TRTForge 的 Anomalib TensorRT 推理动态库。它通过稳定 C API 对外提供 engine 加载、metadata 解析、RGB 图像推理、热力图叠加图和 mask 轮廓图输出能力，可被 TRTForge GUI、外部 C++ 程序和 Python `ctypes` 程序调用。

## 1. 依赖

运行时需要以下 DLL 可被系统找到：

- `anomalib_infer.dll`
- TensorRT / CUDA 运行时 DLL，例如 `nvinfer.dll`、`nvinfer_plugin.dll`、`cudart64_110.dll`
- OpenCV 运行时 DLL，例如 `opencv_world455.dll`

TRTForge 编译后会把 `anomalib_infer.dll` 输出到 `trtexec/gui/bin`，与 `TRTForge.exe` 同目录。

## 2. C API

公开头文件：

```text
trtexec/gui/inference/anomalib_infer/include/anomalib_infer_c.h
```

核心调用顺序：

```c
ai_handle h = ai_create(engine_path_utf8, metadata_path_utf8, "Auto", -1.0f, -1.0f);
ai_get_metadata(h, &meta);
ai_infer_rgb(h, rgb, width, height, stride, &result, &stat);
ai_destroy(h);
```

输入图像必须是 RGB888，`stride` 是每行字节数，允许大于 `width * 3`。

## 3. 内存生命周期

`ai_infer_rgb` 返回的 `ai_result` 以及其中的 `source_rgb`、`overlay_rgb`、`mask_rgb`、`normalized_map` 都由 DLL 内部句柄持有。

这些指针只保证在下一次 `ai_infer_rgb` 或 `ai_destroy` 前有效。外部 C++/Python 程序如果要保存结果，必须立即深拷贝。

## 4. 示例

- C++ 示例：`examples/cpp_basic`
- Python 示例：`examples/python_ctypes`

示例都演示了读取图片、调用 DLL、保存 `overlay.png` 和 `mask.png`。
