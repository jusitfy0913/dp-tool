# TRTForge：`yolo_infer`、`anomalib_infer` 与 `ocr_infer` 动态库调用说明与源码打包

本文说明三个推理 DLL 的 **C API、调用顺序、内存与线程约定、运行时依赖**，以及如何 **将头文件 + 源码 + 示例 + 本文档打成离线包**，便于在内网或给其他工程集成。

---

## 1. 概述

| 项目 | 动态库名 | 典型用途 | 输入像素格式 |
|------|----------|----------|----------------|
| YOLO TensorRT 推理 | `yolo_infer.dll` | 检测 / 分割 / 姿态 / 旋转框 | **BGR888** |
| Anomalib TensorRT 异常检测 | `anomalib_infer.dll` | PatchCore / PaDiM / EfficientAd 等 | **RGB888** |
| PaddleOCR C++ OCR 推理 | `ocr_infer.dll` | 文本检测 / 文本识别 / 文本行原始数据输出 | **BGR888** |

与 TRTForge 主程序一起构建时，DLL 通常输出到工程 **`4-qt-studio/bin`**，与 `TRTForge.exe` 同目录，运行时需能加载到 TensorRT、CUDA、OpenCV、Paddle Inference 等依赖 DLL。

公开接口均为 **C ABI**（`extern "C"`），可从 C++、C# P/Invoke、Python `ctypes` 等调用。

---

## 2. `yolo_infer.dll`（YOLO）

### 2.1 头文件与符号

- 头文件路径：`3-cpp-tensorrt/yolo_infer/include/yolo_infer_c.h`
- 导出宏：编译 DLL 时定义 `YOLO_INFER_EXPORTS`，外部工程 **不要** 定义该宏（使用 `__declspec(dllimport)`）。

### 2.2 核心 API

```c
yi_handle yi_create(const char* engine_path, yi_task task, float conf_thr, float nms_thr);
int yi_infer_bgr(yi_handle h, const unsigned char* bgr, int w, int height, int stride,
                   yi_box** out_boxes, int* out_count, yi_frame_stat* out_stat);
void yi_destroy(yi_handle h);
const char* yi_last_error(void);
```

- **`yi_task`**：`YI_DETECT`、`YI_SEGMENT`、`YI_POSE`、`YI_OBB`，需与 TensorRT 引擎任务一致。
- **`engine_path`**：建议使用 **UTF-8**。TRTForge 内部对中文路径做了宽字符打开等兼容；自行集成时请尽量与工程当前实现一致。
- **输入**：`bgr` 为 BGR 三通道，`stride` 为每行字节数（可大于 `w * 3`，兼容 `QImage`/对齐行）。
- **输出**：`yi_infer_bgr` 成功后 `*out_boxes` 指向 DLL 内部数组，长度为 `*bout_count`；每个 `yi_box` 含框、置信度、类别，分割、`POSE`、`OBB` 等扩展字段见头文件注释。

### 2.3 推荐调用顺序

```text
yi_create → （循环）yi_infer_bgr → yi_destroy
```

失败时：`yi_create` 返回 `nullptr` 或 `yi_infer_bgr` 返回非 0（以当前实现为准），可调用 **`yi_last_error()`** 取可读错误信息。

### 2.4 内存与生命周期

- `yi_infer_bgr` 返回的 **`yi_box*` 数组及其中 `mask_data`、`kpts` 等指针** 由 **句柄** 持有。
- 有效期：**下一次 `yi_infer_bgr` 或 `yi_destroy` 之前**。若需跨帧保存，请 **深拷贝** 到自己的缓冲。

### 2.5 线程

- 与 `anomalib_infer` 相同：**单个 `yi_handle` 不要多线程并发调用 `yi_infer_bgr`**；多路并行请 **每路一个 handle**。

### 2.6 源码与构建

- 目录：`3-cpp-tensorrt/yolo_infer/`
- 入口 CMake：`3-cpp-tensorrt/yolo_infer/CMakeLists.txt`
- 主程序通过 `4-qt-studio/CMakeLists.txt` 中 `add_subdirectory(${TRTFORGE_REPO_ROOT}/3-cpp-tensorrt/yolo_infer)` 链入。

---

## 3. `anomalib_infer.dll`（Anomalib 异常检测）

### 3.1 头文件与符号

- 头文件路径：`3-cpp-tensorrt/anomalib_infer/include/anomalib_infer_c.h`
- 导出宏：编译 DLL 时定义 `ANOMALIB_INFER_EXPORTS`。

### 3.2 核心 API

```c
ai_handle ai_create(const char* engine_path_utf8,
                      const char* metadata_path_utf8,
                      const char* model_type_utf8,
                      float image_threshold_override,
                      float pixel_threshold_override);

int ai_get_metadata(ai_handle h, ai_metadata* out_metadata);

int ai_infer_rgb(ai_handle h,
                 const unsigned char* rgb,
                 int width, int height, int stride,
                 const ai_result** out_result,
                 ai_frame_stat* out_stat);

void ai_destroy(ai_handle h);
const char* ai_last_error(void);
```

- **输入**：**RGB888**，`stride` 同上，允许大于 `width * 3`。
- **`metadata_path_utf8`**：`train.py` / `trtforge_metadata.py` 生成的 **`metadata.json`**，与导出 ONNX 同目录；需包含 `image_threshold`、`pixel_threshold`、`pred_scores` / `anomaly_maps` 等字段（详见 TRTForge README 与 `anomalib-main/trtforge_metadata.py`）。
- **后处理契约**：若 ONNX/TRT 已输出 `pred_score`、`anomaly_map` 等 **0~1 后处理结果**，`metadata.json` 应包含 **`output_contract`: `anomalib_post_processed_0_1`**、`normalization`: `none`、阈值默认 **0.5**，避免与训练阶段 raw 阈值混用导致「全是 OK」。

### 3.3 返回值

- `ai_infer_rgb`：成功为 **`AI_OK` (0)**，失败为 `ai_status` 中非 0，配合 **`ai_last_error()`**。
- `ai_get_metadata`：成功为 `AI_OK`。

### 3.4 内存与生命周期

`ai_result` 中的 **`source_rgb`、`overlay_rgb`、`mask_rgb`、`normalized_map`** 均由 DLL 句柄持有，仅在 **下一次 `ai_infer_rgb` 或 `ai_destroy` 前** 有效；长期保存须 **立即深拷贝**。

### 3.5 线程

- **一个 `ai_handle` 不要多线程同时 `ai_infer_rgb`**；多路请多 handle。

### 3.6 示例与说明

- 文字说明：`3-cpp-tensorrt/anomalib_infer/README.md`
- C++：`3-cpp-tensorrt/anomalib_infer/examples/cpp_basic/`
- Python：`3-cpp-tensorrt/anomalib_infer/examples/python_ctypes/infer_image.py`

### 3.7 源码与构建

- 目录：`3-cpp-tensorrt/anomalib_infer/`
- CMake：`3-cpp-tensorrt/anomalib_infer/CMakeLists.txt`

---

## 4. `ocr_infer.dll`（PaddleOCR C++ OCR）

### 4.1 头文件与符号

- 头文件路径：`ocr/ocr_infer/include/ocr_infer_c.h`
- 导出宏：编译 DLL 时定义 `OCR_INFER_EXPORTS`。

### 4.2 核心 API

```c
oi_handle oi_create(const oi_config* config);
int oi_infer_file(oi_handle h,
                  const char* image_path_utf8,
                  const oi_result** out_result,
                  oi_frame_stat* out_stat);
int oi_infer_bgr(oi_handle h,
                 const unsigned char* bgr,
                 int width, int height, int stride,
                 const oi_result** out_result,
                 oi_frame_stat* out_stat);
void oi_destroy(oi_handle h);
const char* oi_last_error(void);
```

### 4.3 参数说明

- `det_model_dir_utf8`：OCR 检测模型目录，例如 `bin/paddle_inference/ch_PP-OCRv4_det_infer`。
- `rec_model_dir_utf8`：OCR 识别模型目录，例如 `ch_PP-OCRv4_rec_infer`。
- `textline_orientation_model_dir_utf8`：可选文本行方向识别模型目录；目录必须带 `inference.yml` 且与当前 Paddle Inference 版本兼容，同时将 `use_textline_orientation` 置为 `1` 才启用。当前功能测试建议先留空，只验证 det + rec。
- `device_utf8`：首版默认推荐 `cpu`。
- `enable_mkldnn` / `cpu_threads`：CPU 推理性能参数。
- `det_thresh` / `det_box_thresh` / `det_unclip_ratio`：文本检测阈值和外扩参数。
- `rec_score_thresh`：识别分数阈值，首版默认 `0.0`，由调用者自行筛选也可以。

### 4.4 原始数据契约

`oi_result.lines[i]` 保证以下字段按同一 `i` 一一对应：

- `index`
- `xs / ys` 文本行多边形点
- `text_utf8`
- `rec_score`
- `textline_angle`

`raw_json_utf8` 额外保留 `rec_texts`、`rec_scores`、`rec_polys`、`textline_orientation_angles`，便于调用者记录完整原始结果。`oi_result` 及其中指针由 DLL handle 持有，仅在下一次 `oi_infer_*` 或 `oi_destroy` 前有效；需要长期保存必须深拷贝。

### 4.5 源码与构建

- 目录：`ocr/ocr_infer/`
- CMake：`ocr/ocr_infer/CMakeLists.txt`
- 示例：`ocr/ocr_infer/examples/cpp_basic/`
- 独立编译：`cd ocr/ocr_infer; cmake --preset vs2022-x64; cmake --build --preset vs2022-x64`
- TRTForge 主工程默认不再直接 `add_subdirectory` 编译 OCR；主工程默认导入 `4-qt-studio/bin/ocr_infer.dll` 与 `4-qt-studio/bin/ocr_infer.lib`，避免 OCR 第三方目标进入主解决方案。

---

## 5. TRTForge GUI 中的使用位置（参考）

- YOLO：`3-cpp-tensorrt/yolo_infer/` 由推理页通过 **C API** 加载引擎与推理。
- Anomalib：`4-qt-studio/src/AnomalyInferWorker.cpp` 等对 **`ai_*`** 的薄封装，任务类型选择 **异常检测 ANOMALY** 时使用。
- OCR：`4-qt-studio/src/OcrInferWorker.cpp` 等对 **`oi_*`** 的薄封装，任务类型选择 **OCR 文本识别 OCR** 时使用；GUI 结果列表点击某一行时，会用 `oi_result.lines[i].xs/ys` 对应的多边形在画布中高亮该文本框。

更完整的 YOLO 推理流水线（binding、解码、NMS 等）见：

- `docs/trtexec_gui_YOLO_TRT推理流程说明.md`

---

## 6. Python `ctypes` 调用要点

1. `LoadLibrary` / `CDLL` 加载 DLL，**工作目录** 或 `PATH` 需能找到 TensorRT/CUDA/OpenCV 等依赖。
2. 用 `argtypes` / `restype` 显式声明，注意 `yi_box**`、`const ai_result**` 等 **指针的指针**。
3. 字符串路径使用 **UTF-8** 字节（`path.encode('utf-8')`）。
4. 仅在使用期间持有 DLL 返回的指针；下一张图推理前勿保存裸指针。

示例代码以仓库内 `anomalib_infer/examples/python_ctypes/infer_image.py` 为准。

---

## 7. 离线打包：头文件 + 源码 + 文档

仓库提供脚本（UTF-8），在生成 zip 时会包含：

- `3-cpp-tensorrt/yolo_infer/`（头文件、`src`、`CMakeLists.txt`）
- `3-cpp-tensorrt/anomalib_infer/`（含 `examples`）
- `ocr/ocr_infer/`（头文件、`src`、`CMakeLists.txt`、`examples`）
- 本说明（zip 内文件名）：`TRTForge_inference_dll_SDK_zh-CN.md`（由脚本从 `docs/TRTForge_yolo_infer*.md` 复制）

**不**内含：已编译的 `.dll`、TensorRT/CUDA SDK（体量大且受许可约束）；分发运行时请单独拷贝 `bin` 下 DLL 及 NVIDIA/OpenCV 运行时。

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\4-qt-studio\inference\package_inference_sdk.ps1
```

可选参数：

```powershell
.\4-qt-studio\inference\package_inference_sdk.ps1 -OutDir D:\dist -ZipName my_inference_sdk
```

生成的 zip 默认类似：`trtforge_inference_dll_sdk_YYYYMMDD_HHmmss.zip`，解压后可直接用 CMake 将 `yolo_infer` / `anomalib_infer` / `ocr_infer` 子目录并入你的工程，或对照包内 `TRTForge_inference_dll_SDK_zh-CN.md` 做二次封装。

### 7.1 打包整个 `inference` 目录（三个工程 + 脚本 + 说明）

若需要 **`4-qt-studio/inference` 下全部内容**（含 `yolo_infer`、`anomalib_infer`、`ocr_infer`、`package_inference_sdk.ps1`、`package_gui_inference_full.ps1` 等），并附带仓库中的 **DLL 综合说明** 与 **YOLO 推理详解**（若存在），使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\4-qt-studio\inference\package_gui_inference_full.ps1
```

可选：

```powershell
.\4-qt-studio\inference\package_gui_inference_full.ps1 -OutDir D:\dist -ZipName trtforge_gui_inference_full
```

解压后典型结构：`README_PACKAGE.txt`、`inference/`（完整源码树）、`docs/`（说明文档副本）。

### 7.2 二次开发包（推理只给 DLL，训练工程一起带走）

给其他开发同事、但**不提供**封装后推理实现源码时，使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\3-cpp-tensorrt\trtexec\package_devkit.ps1
```

| 会打进去 | 不会打进去 |
| --- | --- |
| `4-qt-studio` Qt 源码、`4-qt-studio/bin/training/scripts` 启动脚本 | `yolo_infer` / `anomalib_infer` / `ocr_infer` 的 `src` |
| `1-python-yolo/ultralytics-main`、`2-python-anomaly/anomalib-main` | Anomalib 历史 `datasets` / `results` |
| 预编译 `*_infer.dll`、`*_infer.lib`、C API 头文件 | `4-qt-studio/build`、日志、`.pdb` |

默认输出到仓库根目录 `dist/trtforge_devkit_时间戳/`。解压后必须保持 `4-qt-studio`、`1-python-yolo`、`2-python-anomaly` 在同一级。开发包内 `CMakeLists.txt` 已改为导入预编译推理 DLL。说明见包内 `README_DEVKIT.md`。

这与上面的「推理 SDK 源码 zip」相反：开发包给 DLL、不给推理源码；SDK zip 给源码、不给 DLL。

---

## 8. 版本与维护

- 若 C API 有变更，以各目录下 **`include/*_infer_c.h`** 为唯一权威；本文描述与头文件冲突时，**以头文件为准**。
- Anomalib 部署侧 `metadata.json` 字段含义与 `output_contract` 约定，以 `2-python-anomaly/anomalib-main/trtforge_metadata.py` 及 TRTForge README 为准。
