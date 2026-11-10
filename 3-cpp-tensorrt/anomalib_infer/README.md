# anomalib_infer DLL 使用说明

`anomalib_infer` 是 TRTForge 的 Anomalib TensorRT 推理动态库。它通过稳定 C API 对外提供 engine 加载、metadata 解析、RGB 图像推理、热力图叠加图和 mask 轮廓图输出能力，可被 TRTForge GUI、外部 C++ 程序和 Python `ctypes` 程序调用。

## 1. 依赖

运行时需要以下 DLL 可被系统找到：

- `anomalib_infer.dll`
- TensorRT / CUDA 运行时 DLL，例如 `nvinfer.dll`、`nvinfer_plugin.dll`、`cudart64_110.dll`、`cublas64_11.dll`、`cublasLt64_11.dll`
- cuDNN 8：`cudnn64_8.dll`、`cudnn_cnn_infer64_8.dll`、`cudnn_ops_infer64_8.dll`
- OpenCV 4.12（分模块版）：`opencv_core4120.dll`、`opencv_imgproc4120.dll`
- 精简分发时若报 `Could not locate zlibwapi.dll`，把 `zlibwapi.dll` 放到同目录

TRTForge 编译后会把 `anomalib_infer.dll` 输出到 `4-qt-studio/bin`，与 `TRTForge.exe` 同目录。

## 2. C API

公开头文件：

```text
3-cpp-tensorrt/anomalib_infer/include/anomalib_infer_c.h
```

核心调用顺序：

```c
ai_handle h = ai_create(engine_path_utf8, metadata_path_utf8, "Auto", -1.0f, -1.0f);
ai_get_metadata(h, &meta);
ai_infer_rgb(h, rgb, width, height, stride, &result, &stat);
ai_destroy(h);
```

输入图像必须是 RGB888，`stride` 是每行字节数，允许大于 `width * 3`。

### 2.1 参数说明

| 函数 | 参数 | 类型 | 说明 |
| --- | --- | --- | --- |
| `ai_create` | `engine_path_utf8` | `const char*` | `.engine` 引擎路径，UTF-8 编码；Anomalib 侧需 FP16/FP32 引擎 |
| | `metadata_path_utf8` | `const char*` | 训练产物里的 `metadata.json`，UTF-8 编码；阈值与归一化参数取自此处 |
| | `model_type_utf8` | `const char*` | `"Auto"` / `"Patchcore"` / `"Padim"` / `"EfficientAd"`；`Auto` 按 metadata 判断 |
| | `image_threshold_override` | `float` | 传 `-1.0` 用 metadata 里的值，否则覆盖图像级判定阈值 |
| | `pixel_threshold_override` | `float` | 传 `-1.0` 用 metadata 里的值，否则覆盖像素级（mask）阈值 |
| `ai_infer_rgb` | `rgb` | `const unsigned char*` | RGB888，非 BGR；行首地址 |
| | `width` / `height` | `int` | 原图尺寸，不必等于模型输入尺寸，内部做 letterbox |
| | `stride` | `int` | 每行字节数，`<= 0` 视为 `width * 3` |
| | `result` | `ai_result**` | 出参，指针由句柄持有，下次推理或 `ai_destroy` 后失效 |
| | `stat` | `ai_frame_stat*` | 出参，分段耗时（毫秒）；可传 `NULL` |
| `ai_get_metadata` | `meta` | `ai_metadata*` | 出参，返回模型类型、输入尺寸、阈值等；创建句柄后随时可读 |
| `ai_last_error` | — | `const char*` | 返回**当前线程**最近一次错误；无错误时为空串 |

返回值：`0` 成功，非 `0` 失败，失败原因用 `ai_last_error()` 取。

## 3. 内存生命周期

`ai_infer_rgb` 返回的 `ai_result` 以及其中的 `source_rgb`、`overlay_rgb`、`mask_rgb`、`normalized_map` 都由 DLL 内部句柄持有。

这些指针只保证在下一次 `ai_infer_rgb` 或 `ai_destroy` 前有效。外部 C++/Python 程序如果要保存结果，必须立即深拷贝。

## 4. 示例

- C++ 示例：`examples/cpp_basic`
- Python 示例：`examples/python_ctypes`

示例都演示了读取图片、调用 DLL、保存 `overlay.png` 和 `mask.png`。

## 5. 应用场景

| 场景 | 为什么用它 |
| --- | --- |
| 工业表面缺陷检测（PCB / 织带 / 金属件） | Anomalib 的三类模型都支持，只学正常样本，不需要负样本标注 |
| 产线节拍要求 GPU 直推 | 引擎推理 + 内核预处理，单帧 39 ms 级（256×256 PatchCore，本机实测） |
| 需要"缺陷在哪"而不只是"有没有" | 同时输出 `overlay_rgb`（热力图叠加）和 `mask_rgb`（二值轮廓） |
| 多路相机同时检测 | 每路一个 `ai_handle`，并发实测 1.8×（见第 6 节） |

不适合的场景：需要精确框选缺陷的位置与类别（属于 YOLO detect 的职责）、需要 CPU-only 部署（用 `onnx_infer.dll` 跑同一个 `.onnx`）。

---

## 6. 并发与多模型（同步/异步、单/多线程）

测于 2026-11-10，Windows 10 / 24 逻辑核 / T1000 8GB，输入 810×1080。

### 6.1 结论速查

| 问题 | 答案 |
| --- | --- |
| 支持异步吗 | **不支持**。`ai_infer_rgb` 是同步阻塞，返回前已 `cudaStreamSynchronize` |
| 支持一次多张图吗 | **不支持**。C ABI 只有单图入口，多图请循环调用 |
| 支持多模型同时在内存吗 | **支持**。每模型一个句柄，句柄间无全局共享状态 |
| 支持多线程吗 | **支持，且是三个库里唯一能拿到线性加速的**——每个句柄有私有 CUDA 流 |
| 同一句柄多线程 | **禁止，会崩**（实测稳定复现 `0xC0000409`） |

### 6.2 为什么 anomalib 能真并发

每个句柄在创建时都建了自己的 CUDA 流，推理完成只同步自己那条流：

```cpp
// anomalib_infer_c.cpp:667
cudaStreamCreate(&stream_);
// :403  context_->enqueueV2(bindings.data(), stream_, nullptr);
// :407  cudaStreamSynchronize(stream_);
// :426  cudaStreamSynchronize(stream_);   // 拷回结果
```

对比 `yolo_infer.dll`：它不传流，走 CUDA 传统默认流（stream 0），所有句柄的 kernel 排在同一队列上，多线程只拿到 1.07×（见该库 README）。

实测（PatchCore 引擎，单次稳态 39 ms）：

| 配置 | 共 8 次耗时 | 加速 |
| --- | ---: | ---: |
| 单句柄串行 | 312 ms | 1.00× |
| 4 句柄 / 4 线程 | 170 ms | **1.80×** |

瓶颈在显存：每句柄一份 engine + 工作缓冲，8GB 卡上 PatchCore 建议 ≤ 4 路。

### 6.3 同一句柄多线程为什么不行

结果缓冲（`sourceRgb_` / `overlayRgb_` / `maskRgb_` / `normalizedMap_` / `result_`）都是句柄成员，`ai_infer_rgb` 会复用它们，`ai_result` 里的指针直接指向这些容器。两个线程并发调用时：轻则结果串台，重则一个线程正在读 `overlay_rgb`、另一个线程 `resize` 了同一个 vector —— **实测稳定崩溃**：

```
1 句柄 / 4 线程 → 进程退出码 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN)
```

### 6.4 正确用法

```python
import ctypes, threading

local = threading.local()

def handle_for(engine: str, meta: str):
    h = getattr(local, "h", None)
    if h is None:
        h = dll.ai_create(engine.encode("utf-8"), meta.encode("utf-8"), b"Auto", -1.0, -1.0)
        if not h:
            raise RuntimeError(dll.ai_last_error().decode("utf-8", "replace"))
        local.h = h
    return h
```

三条规则：

1. **一句柄一线程**；线程结束前 `ai_destroy`，不要跨线程共享。
2. **并发数按显存定**，不要按 CPU 核数定。
3. 上层的 `infer.py` 有全局 `_anom_lock`、`infer_worker.py` 是单连接单循环，**网页端与批量脚本目前是串行的**；要用满 1.8× 必须改上层。

---

## 7. 版本差异（AI 可识别）

```yaml
operator: anomalib_infer
version_scheme: "代码此前未标注版本号；本节建立文档基线，后续改动按 语义化版本 追加"
baseline_doc: 2026-11-10
doc_revision: 2026-11-10   # 仅文档：参数表、应用场景、第 6 节并发契约
baseline_abi:
  symbols:
    - "ai_create(const char*, const char*, const char*, float, float)"
    - "ai_infer_rgb(ai_handle, const unsigned char*, int, int, int, const ai_result**, ai_frame_stat*)"
    - "ai_get_metadata(ai_handle, ai_metadata*)"
    - "ai_destroy(ai_handle)"
    - "ai_last_error()"
  types: ["ai_handle", "ai_result", "ai_metadata", "ai_frame_stat"]
  models: ["PatchCore", "PaDiM", "EfficientAd"]
baseline_behavior:
  - "RGB888 输入，内部 letterbox + 归一化，返回 source/overlay/mask/normalized_map 四份结果"
  - "每个句柄独立 CUDA 流（cudaStreamCreate），多句柄多线程可真并发"
  - "ai_result 内的指针由句柄持有，仅在下一次推理或 destroy 前有效，须立即深拷贝"
  - "一个句柄不允许并发调用 ai_infer_rgb；实测并发会崩溃"
compatibility:
  abi: "发布后只允许在结构体末尾追加字段"
changes:
  - id: AI-1
    type: concurrency
    summary: 每句柄私有 CUDA 流，是三个推理库中唯一能拿到近线性加速的
    impact: 4 句柄 4 线程实测 1.80×（39 ms → 单次等效 21 ms）
  - id: AI-2
    type: contract
    summary: 同一句柄多线程调用会崩溃，不是性能问题而是内存安全问题
    impact: 调用方必须一线程一句柄；上层现有全局锁就是这个约束的体现
```
