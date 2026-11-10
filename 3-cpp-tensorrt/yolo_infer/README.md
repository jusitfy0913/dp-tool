# yolo_infer DLL 使用说明

`yolo_infer` 是 TRTForge 的 YOLO TensorRT 推理动态库。它把 `.engine` 引擎、CUDA 预处理、后处理（decode / NMS / mask / 关键点 / OBB）封装成一份稳定 C API，可被 TRTForge GUI、外部 C++ 程序和 Python `ctypes` 程序调用。

**一句话定位**：给一张 BGR 图，同步拿回框、掩膜、关键点或旋转框。

| 任务 | 枚举 | 典型模型 | 输出 |
| --- | --- | --- | --- |
| 检测 | `YI_DETECT` | `yolov8n.engine` | `yi_box`：`left/top/right/bottom` + `confidence` + `class_label` |
| 分割 | `YI_SEGMENT` | `yolov8n-seg.engine` | 上面的框 + `has_mask` / `mask_data`（可见 ROI 的 8 位掩膜） |
| 姿态 | `YI_POSE` | `yolov8n-pose.engine` | 上面的框 + `num_kpts` / `kpts`（`x, y, visibility` 三元组） |
| 旋转框 | `YI_OBB` | `yolov8n-obb.engine` | 上面的框 + `is_obb` / `cx, cy, ow, oh, angle`（弧度） |

## 1. 依赖

运行时需要以下 DLL 可被系统找到（TRTForge 编译后与 `TRTForge.exe` 一起放在 `4-qt-studio/bin`，本机实际文件名）：

- `yolo_infer.dll`（本库）
- TensorRT：`nvinfer.dll`、`nvinfer_plugin.dll`（构建引擎还需要 `nvinfer_builder_resource.dll`）
- CUDA 11.8：`cudart64_110.dll`、`cublas64_11.dll`、`cublasLt64_11.dll`
- cuDNN 8：`cudnn64_8.dll`、`cudnn_cnn_infer64_8.dll`、`cudnn_ops_infer64_8.dll`
- OpenCV 4.12（分模块版）：`opencv_core4120.dll`、`opencv_imgproc4120.dll`
- 精简分发时若报 `Could not locate zlibwapi.dll`，把 `zlibwapi.dll` 放到同目录（完整 bin 目录通常不需要）

引擎本身与**显卡算力**绑定：`sm_75` 的引擎不能拿到 `sm_86` 的卡上跑，A 卡建的引擎必须 A 卡用。

## 2. C API

公开头文件（唯一权威）：

```text
3-cpp-tensorrt/yolo_infer/include/yolo_infer_c.h
```

核心调用顺序：

```c
yi_handle h = yi_create(engine_path, YI_DETECT, 0.25f, 0.45f);
if (!h) { printf("%s\n", yi_last_error()); return; }

yi_box*   boxes = NULL;
int       count = 0;
yi_frame_stat stat;
int code = yi_infer_bgr(h, bgr, width, height, stride, &boxes, &count, &stat);

/* 用 boxes[0..count-1]，指针只在下次推理前有效 */
yi_destroy(h);
```

### 2.1 参数说明

| 函数 | 参数 | 类型 | 说明 |
| --- | --- | --- | --- |
| `yi_create` | `engine_path` | `const char*` | `.engine` 路径，UTF-8 编码 |
| | `task` | `yi_task` | `YI_DETECT` / `YI_SEGMENT` / `YI_POSE` / `YI_OBB`，**必须与引擎的输出结构一致** |
| | `conf_thr` | `float` | 置信度阈值，`0.25` 为常用默认 |
| | `nms_thr` | `float` | NMS 的 IoU 阈值，`0.45` 为常用默认 |
| | 返回 | `yi_handle` | 失败返回 `NULL`，原因用 `yi_last_error()` 取 |
| `yi_infer_bgr` | `h` | `yi_handle` | `yi_create` 的返回值 |
| | `bgr` | `const unsigned char*` | BGR888，行首地址（OpenCV `cv::Mat::data` 直接可用） |
| | `w` / `height` | `int` | 原图宽高，不必等于模型输入尺寸，内部做 letterbox |
| | `stride` | `int` | 每行字节数，`<= 0` 视为 `w * 3`；传 `mat.step` 可避免行对齐丢数据 |
| | `out_boxes` | `yi_box**` | 出参，数组由句柄持有；`count == 0` 时为 `NULL` |
| | `out_count` | `int*` | 出参，检测框数量 |
| | `out_stat` | `yi_frame_stat*` | 出参，分段耗时（毫秒）；可传 `NULL` |
| | 返回 | `int` | `0` 成功，`-1` 失败 |
| `yi_destroy` | `h` | `yi_handle` | 释放引擎、上下文与全部内部缓冲；传 `NULL` 安全 |
| `yi_last_error` | — | `const char*` | 返回**当前线程**最近一次错误；无错误时为空串 |

### 2.2 `yi_box` 字段

| 字段 | 何时有效 | 说明 |
| --- | --- | --- |
| `left` / `top` / `right` / `bottom` | 所有任务 | 原图坐标系下的水平外接框 |
| `confidence` | 所有任务 | 置信度 |
| `class_label` | 所有任务 | 类别索引，名字映射在调用方（`dataset.yaml` 的 `names`） |
| `has_mask` / `mask_x/y/w/h` / `mask_data` | `YI_SEGMENT` | 掩膜只覆盖**可见 ROI**（裁到画面内），尺寸已缩放到 ROI 大小，`mask_data` 为 8 位单通道 |
| `num_kpts` / `kpts` | `YI_POSE` | `kpts` 长度为 `num_kpts * 3`，依次是 `x, y, visibility` |
| `is_obb` / `cx` / `cy` / `ow` / `oh` / `angle` | `YI_OBB` | 中心点、宽高、弧度角；`left..bottom` 同时给出外接框 |

### 2.3 `yi_frame_stat` 字段

| 字段 | 说明 |
| --- | --- |
| `src_w` / `src_h` | 输入原图尺寸，回显用 |
| `count` | 结果数量 |
| `preprocess_ms` | letterbox + 归一化 + H2D 拷贝 |
| `inference_ms` | `enqueueV2` + 同步等待 |
| `postprocess_ms` | decode / NMS / mask 缩放 / 关键点还原 |
| `total_ms` | 三段之和（含计时开销） |

## 3. 内存生命周期

`yi_infer_bgr` 返回的 `yi_box*` 数组，以及数组里 `mask_data`、`kpts` 指向的内存，**全部由句柄持有**：

- 只保证在下一次 `yi_infer_bgr` 或 `yi_destroy` 前有效；
- 每次推理会先清空再重填，所以**上一次拿到的指针会失效**；
- 要长期保存必须立即深拷贝。

同理，`yi_last_error()` 返回的指针也是线程本地缓冲，下一次调用后会被覆盖。

## 4. 并发与多模型（同步/异步、单/多线程）

测于 2026-11-10，Windows 10 / 24 逻辑核 / T1000 8GB，detect 引擎 960×960，输入 810×1080，稳态计时。

### 4.1 结论速查

| 问题 | 答案 | 依据 |
| --- | --- | --- |
| 支持异步吗 | **不支持**。`yi_infer_bgr` 同步阻塞，返回前内部已 `cudaStreamSynchronize` | `yolo.cu:862`、`:909`、`:1038` |
| 支持一次多张图吗 | **不支持**。C ABI 只有单图入口；内部的 `forwards(vector<Image>)` 没有导出 | `yolo_infer_c.h` |
| 支持多模型同时在内存吗 | **支持**。每次 `yi_create` 独立反序列化一份引擎 + 独立 `IExecutionContext` + 独立缓冲 | `infer.cu:466` `load()` 无缓存 |
| 支持多线程吗 | **支持**。v1.1.0 起每个句柄有自己的 `cudaStreamNonBlocking`，多线程各用各的句柄可并行执行 | `yolo_infer_c.cpp` 变更见 4.6 |
| 多线程能加速多少 | **取决于任务是否把 GPU 吃满**：检测 1.10×、分割 1.25×（旧版都是 1.02×）。见 4.2 |
| 同一句柄多线程 | **禁止**，实测 20/20 次结果错乱，另一次直接崩溃 `0xC0000409` | 见 4.3 |

### 4.2 多句柄多线程：安全，但收益被 GPU 算力封顶

句柄之间没有共享状态，多线程各用各的句柄**不会**出错。v1.1.0 之前所有句柄共用 CUDA 传统默认流（`forward(img)` 不传流 → `cudaStream_t 0`），kernel 排在同一条队列上，多线程只有 1.02×；现在每句柄一条私有流（见 4.6），kernel 之间可以交错执行。

**但"能交错"不等于"变快"**：并行度上限是 GPU 的算力，不是流的数量。本机 A/B（同一批图、同一引擎，旧库/新库交替各跑 3 轮取最小）：

| 任务 | 独占单次 | 旧库 4 线程 | 新库 4 线程 |
| --- | ---: | ---: | ---: |
| detect（PCB 960 引擎） | 7.5 ms | 1.01–1.03× | **1.05–1.13×** |
| segment（yolov8n-seg） | 2.6–3.8 ms | 1.02× | **1.25×** |

**为什么只有 1.1×？** 并发时测每次调用自己的流内耗时：4 路并发下每个调用从 7.7 ms 涨到约 27 ms（≈3.5×）。这说明 4 条流的 kernel 确实在时间上交错，但 GPU 总算是"4 份工作分同一块算力"——这个 960×960 引擎在 T1000 上本来就把 GPU 吃满了，加速比自然被算力卡住。分割任务能到 1.25×，是因为它有一段 CPU 侧后处理（每个框的 `cv::resize` 掩膜）可以和多份 GPU 工作重叠。

对比 `anomalib_infer.dll` 的 1.80×：那个模型的 GPU 工作只占一小部分，主要时间在 CPU 侧（掩膜轮廓、结果图编码），所以多线程重叠的是 CPU。

**结论**：想要吞吐，先算清瓶颈在 GPU 还是 CPU。

- GPU 吃满（本例 detect）→ 换更快的卡，或多进程分摊到多张卡；
- CPU 侧占比大（分割、anomalib）→ 多句柄多线程直接有收益；
- 不确定 → 用 4 路并发时的 `total_ms` 与独占单次比：≈N 倍说明在交错中分算力，≈1 倍说明还有别的序列化点。

### 4.3 同一句柄多线程：禁止

`YoloContext` 里的结果缓冲是句柄成员，每次推理先清空再填充：

```cpp
// yolo_infer_c.cpp:15
struct YoloContext {
    std::vector<yi_box>             boxes_buf;   // ← 每次 clear 再填
    std::vector<cv::Mat>            mask_hold;
    std::vector<std::vector<float>> kpt_hold;
};
// :71  ctx->boxes_buf.clear();
```

两个线程打同一个句柄，A 刚拿到的 `boxes` 会被 B 的 `clear()` 抹掉。实测（两张不同图片交错各 10 次）：

| 运行 | 结果 |
| --- | --- |
| 第一次 | **20/20 次结果与基准不一致**（框数、坐标都对不上） |
| 第二次 | 进程直接退出，`0xC0000409`（STATUS_STACK_BUFFER_OVERRUN） |

不是性能问题，是**内存安全**问题。

### 4.4 多模型的真实成本

每多一个句柄，就多一份：引擎权重（显存）+ 执行上下文 workspace + 预处理/输出缓冲。8GB 卡上同时挂 detect + segment + pose 三个句柄通常仍有余量，但**不要为每帧新建句柄**——`yi_create` 要读盘 + `deserializeCudaEngine`，成本是推理的几十倍。正确做法是进程内缓存句柄（`infer.py` 就是这么做的，并配 LRU 上限）。

### 4.5 正确用法（一线程一句柄）

```python
import ctypes, threading

local = threading.local()

def handle_for(engine_path: str, task: int = 0):
    h = getattr(local, "h", None)
    if h is None:
        h = dll.yi_create(engine_path.encode("utf-8"), task, 0.25, 0.45)
        if not h:
            raise RuntimeError(dll.yi_last_error().decode("utf-8", "replace"))
        local.h = h
    return h
```

三条规则：

1. **一句柄一线程**，不共享、不交叉、不在别的线程 `yi_destroy`。
2. **并发按显存定**，不按 CPU 核数定。
3. 想靠线程拿吞吐，先看 4.2 判断瓶颈在 GPU 还是 CPU；GPU 吃满时多进程/多卡才是有效手段。

### 4.6 私有 CUDA 流（v1.1.0 已实施）

`yolo.cu` 的 `forward(images, stream)` 本来就支持传流，之前只是 C 包装没传。v1.1.0 补上了这层，全部改动都在 `yolo_infer_c.cpp`，**`yolo.cu` / `infer.cu` 一行都没动**：

```cpp
// 1) 句柄里存一条私有流
struct YoloContext {
    std::shared_ptr<yolo::Infer> infer;
    cudaStream_t                 stream = nullptr;   // 新增
    ...
};

// 2) 建句柄时开流（NonBlocking：不与 CUDA 默认流互相等待）
cudaError_t cerr = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
if (cerr != cudaSuccess) { set_error(...); delete ctx; return nullptr; }

// 3) 推理时用这条流
yolo::BoxArray results = ctx->infer->forward(img, ctx->stream);

// 4) 销毁时关流
if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
```

**效果（本机实测）**：4 路并发从 1.02× 提到 1.10×（detect）/ 1.25×（segment）；单线程耗时与结果均无变化（新旧库交替 A/B，独占单次都是 7.5 ms，三张图逐框结果完全一致）。收益受限的原因见 4.2——是 GPU 算力封顶，不是流没生效。

**为什么不是 1.8×**：`anomalib_infer` 的 1.80× 来自它能在多线程间重叠 CPU 侧处理，而不是 GPU 并行度更高。检测引擎在 T1000 上已经把 GPU 吃满，4 条流只是让 4 份工作分同一块算力。

### 4.7 与上层的关系

`infer.py` 用全局 `_yolo_lock` 串行化所有 `yi_*` 调用，`infer_worker.py` 是单连接单循环，`inferproxy.py` 再包一把 `RLock`。所以**网页端、`infer_batch.py`、`infer_video.py` 的推理都是排队的**；DLL 的多句柄能力上层暂未使用。也就是说：v1.1.0 让 DLL「具备」并发能力，但页面吞吐不会因此变化。

## 5. 使用方式

### 5.1 Python ctypes（最短可用）

```python
import ctypes, os

BIN = r"d:\test\deeplearning\dp-tool\4-qt-studio\bin"
os.add_dll_directory(BIN)                      # 让 yolo_infer.dll 找到 nvinfer / cudnn
dll = ctypes.WinDLL(os.path.join(BIN, "yolo_infer.dll"))
dll.yi_create.restype = ctypes.c_void_p
dll.yi_last_error.restype = ctypes.c_char_p

h = dll.yi_create(r"...\best.engine".encode("utf-8"), 0, 0.25, 0.45)
if not h:
    raise RuntimeError(dll.yi_last_error().decode("utf-8", "replace"))
# 完整结构体定义见 dp-tool/6-web/backend/infer.py 的 YiBox / YiFrameStat
```

### 5.2 网页端

不用直接调 DLL：模型填 `.engine` 路径即可，后端按后缀自动分派到本库，结果里的 `backend` 字段会返回 `yolo_infer.dll`。

### 5.3 批量与视频

```powershell
python dp-tool\6-web\backend\tools\infer_batch.py --model xxx.engine --images <dir> --task detect
python dp-tool\6-web\backend\tools\infer_video.py --engine xxx.engine --source clip.mp4 --task detect
```

### 5.4 引擎转换与量化

- 引擎构建走 Qt 的 Engine Builder 页，或 `3-cpp-tensorrt/trtexec`；
- INT8 标定工具：`int8_calib`（与 `yolo_infer.dll` 一起输出到 `4-qt-studio/bin`）；
- 诊断导出：设 `TRTFORGE_RAW_DUMP=<目录>` 可把喂给引擎的输入张量和后处理前输出落盘，与 `onnx_infer` 做张量级比对（未设置时零开销）。

## 6. 应用场景

| 场景 | 为什么用它 |
| --- | --- |
| 产线实时检测（检测/分割/姿态/旋转框） | GPU 预处理 + 内核 decode，检测单帧 7.5 ms 级（本机实测，960 输入 / T1000） |
| 与 Qt 桌面程序集成 | 纯 C ABI，`TRTForge.exe` 同目录，无 Python 依赖 |
| 需要掩膜做面积/形状判定 | `YI_SEGMENT` 直接给可见 ROI 的 8 位掩膜，省一次后处理 |
| 位姿/关键点测量、料框角度校正 | `YI_POSE` / `YI_OBB` 原图坐标系输出，无需自己反算仿射 |
| 需要与 ONNX 路径对拍 | 与 `onnx_infer.dll` 逐字段对齐的 `yi_box` 定义，同一套 ctypes 通用 |

不适合的场景：无 NVIDIA GPU、没有可用的 `.engine`、需要 CPU-only 部署（用 `onnx_infer.dll` 跑 `.onnx`）、需要多模型高并发吞吐（见 4.2）。

## 7. 排错表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `yi_create` 返回 `NULL`，`Failed to load engine` | 路径错、不是引擎文件、引擎 CUDA 版本不匹配 | 用 `trtexec --loadEngine=` 先验证引擎本身 |
| 报 `serialization header` / 版本不匹配 | 引擎由别的 TensorRT 版本构建 | 用本机同版本 TRT 重新构建 |
| 报 `Could not locate zlibwapi.dll` | 缺 zlib | 把 `zlibwapi.dll` 放到 DLL 同目录 |
| 报 `cublasLt64_11.dll` / `cudnn64_8.dll` 缺失 | 依赖没随包 | 按第 1 节补全，或用安装包 |
| 框全错、类别乱、数量离谱 | `task` 与引擎输出结构不匹配 | 按模型改 `task`；引擎输出通道数需与任务一致 |
| 分割掩膜位置偏 | 调用方自己又按 `width` 算 stride | 直接把 `mat.step` 传给 `stride` |
| 多线程后结果错乱/偶发崩溃 | 同一句柄被多线程调用 | 见 4.3，一线程一句柄 |
| 多线程吞吐没提升 | ① GPU 算力已满（检测类常见）；② 旧版共用默认流 | 用 4.2 的「并发时 total_ms ÷ 独占单次」判断瓶颈；旧版升到 v1.1.0 |

## 8. 版本差异（AI 可识别）

```yaml
operator: yolo_infer
version_scheme: "语义化版本；1.0.0 为 2026-11-05 的既有 DLL（当时无版本标注），本文档建立其基线"
current_version: 1.1.0
release: 2026-11-10
doc_revision: 2026-11-10
version_history:
  - version: 1.0.0
    release: 2026-11-05
    note: "基线：四任务 C ABI；多句柄共用 CUDA 传统默认流"
  - version: 1.1.0
    release: 2026-11-10
    note: "每句柄私有 cudaStreamNonBlocking，多句柄多线程可并行"
abi:
  symbols:
    - "yi_create(const char* engine_path, yi_task task, float conf_thr, float nms_thr)"
    - "yi_infer_bgr(yi_handle, const unsigned char* bgr, int w, int h, int stride, yi_box**, int*, yi_frame_stat*)"
    - "yi_destroy(yi_handle)"
    - "yi_last_error()"
  types: ["yi_handle", "yi_task", "yi_box", "yi_frame_stat"]
  tasks: ["YI_DETECT", "YI_SEGMENT", "YI_POSE", "YI_OBB"]
diff_1_0_0_to_1_1_0:
  abi:
    added_symbols: none
    added_types: none
    changed: none
    removed: none
  implementation:
    files_touched: ["3-cpp-tensorrt/yolo_infer/src/yolo_infer_c.cpp"]
    files_untouched: ["src/yolo.cu", "src/infer.cu", "include/yolo_infer_c.h"]
    summary:
      - "YoloContext 增加 cudaStream_t stream 成员"
      - "yi_create 用 cudaStreamCreateWithFlags(cudaStreamNonBlocking) 建流，失败则返回 NULL 并写 last_error"
      - "yi_infer_bgr 改为 forward(img, ctx->stream)"
      - "yi_destroy 先 cudaStreamDestroy 再释放句柄"
  behavior:
    - "单线程推理耗时不变（新旧交替 A/B：独占单次均 7.5 ms）"
    - "单线程结果逐框完全一致（三张图 A/B，坐标/置信度/类别全等）"
    - "4 路并发加速比：旧版 1.01–1.03× → 新版 1.05–1.13×（detect，GPU 算力已满）"
    - "4 路并发加速比：旧版 1.02× → 新版 1.25×（segment，CPU 后处理可重叠）"
  compatibility:
    backward: "ABI 未变，调用方无需改动；行为对单线程完全一致"
    abi: "发布后只允许在结构体末尾追加字段"
known_gaps:
  - id: YI-GAP-1
    status: resolved_in_1_1_0
    gap: "每句柄没有私有 CUDA 流，多线程无法并行"
    resolution: "v1.1.0 已改为每句柄 cudaStreamNonBlocking，见 4.6"
    residual: "检测类任务在 T1000 上 GPU 已吃满，加速比仍只有 1.1×——瓶颈是算力而非流"
  - id: YI-GAP-2
    status: open
    gap: "C ABI 没有批量入口，一次只能一张图"
    workaround: "调用方循环；上层脚本已如此实现"
    planned: "可复用内部 forwards(vector<Image>)，未实施"
```
