# YOLO 推理：行步长（stride）与 GUI 预处理修复说明

本文档记录为修复 **TRT Forge 与 YOLOv8-seg 同图同模型推理结果严重不一致** 问题所做的源码级修改，包括：**新增字段、注释说明、删除/替换逻辑、行为变化** 与 **验证要点**。

---

## 1. 背景与根因（摘要）

- **现象**：同一 `bus.jpg`、同一 `best.engine`，TRT Forge 侧检测/分割质量极差（漏检、低置信度、mask 异常），而 `D:\YOLOv8-seg` 工程正常。
- **结论**：**不是模型或 TensorRT 引擎问题**，主要是 **CPU 侧图像缓冲的「每行字节数」（stride / line_size）与推理预处理假设不一致**。
- **机制**：
  - `QImage::Format_RGB888` 下，每行常按 **4 字节对齐**，当 `width × 3` 不能被 4 整除时，每行末尾存在 **padding**（例如宽 810：`810×3=2430`，对齐后 `bytesPerLine()=2432`）。
  - 旧代码将 `yolo::Image` 视为 **紧凑** `width×height×3`，`memcpy` 与 GPU letterbox 内核使用 **`width×3` 作为行步长**，导致 **逐行错位读取**，整幅图在视觉上被「撕裂」，网络输入失真。

---

## 2. 修改文件一览

| 文件 | 变更类型 |
|------|----------|
| `inference/yolo_infer/src/yolo.hpp` | 结构体扩展 + Doxygen 注释 |
| `inference/yolo_infer/src/yolo.cu` | 预处理逻辑 + **英文**行内注释（见 §6） |
| `inference/yolo_infer/src/yolo_infer_c.cpp` | 去掉中间 `cv::Mat` 包装，stride 透传 + 中文注释 |
| `src/InferWorker.cpp` | 头文件引入 + RGB→BGR 路径改为 `cv::cvtColor` + 中文块注释 |

**未改动的对外 C API**：`yolo_infer_c.h` 中 `yi_infer_bgr(..., int stride, ...)` 签名未变，仅实现与下游 `yolo::Image` 语义对齐。

---

## 3. `yolo.hpp` — `yolo::Image` 增加 `line_size`

### 3.1 新增内容

- **新增成员**：`int line_size = 0;`  
  - 含义：**每行字节数**（与 OpenCV `Mat::step`、Qt `bytesPerLine()` 同量级概念）。
  - **约定**：`line_size == 0` 时，在构造函数中 **自动设为 `width * 3`**，与历史「紧凑 BGR」调用完全兼容。

- **构造函数变更**：由三参数改为四参数（第四参数带默认值）：

```cpp
Image(const void *bgrptr, int width, int height, int line_size = 0)
```

初始化列表：`line_size(line_size > 0 ? line_size : width * 3)`。

### 3.2 新增文档注释（Doxygen）

在 `struct Image` 上方增加 **`/** ... */`** 块，说明：

- `bgrptr`：BGR、uint8×3；
- `width` / `height`：像素尺寸；
- `line_size`：行步长；紧凑缓冲传 `width*3` 或 `0`；带 padding 的缓冲传真实 `bytesPerLine`；
- 举例：`QImage::Format_RGB888` 行对齐、`cv::Mat` 可能带 padding；
- **默认 `0` ⇒ `width*3`**，保证旧代码 `Image(ptr, w, h)` 行为不变。

### 3.3 兼容性

- **旧调用**：`yolo::Image(image.data, cols, rows)` — 仍合法，`line_size` 自动为 `width*3`。
- **新调用**：`yolo::Image(ptr, w, h, actual_stride)` — 用于 Qt / 带 step 的 `cv::Mat` 等。

---

## 4. `yolo.cu` — 预处理按真实行步长上传与采样

### 4.1 修改位置

在 `InferImpl::preprocess(...)`（或等价成员函数）中，原先大致为：

- `size_image = image.width * image.height * 3`
- `warp_affine_bilinear_and_normalize_plane(..., image.width * 3, image.width, image.height, ...)`

### 4.2 修改后逻辑

1. **计算源图行字节数**：

```cpp
const int src_line_size_bytes =
    (image.line_size > 0) ? image.line_size : (image.width * 3);
```

2. **按行步长 × 高度分配/拷贝字节数**：

```cpp
size_t size_image = static_cast<size_t>(src_line_size_bytes) * image.height;
memcpy(image_host, image.bgrptr, size_image);
cudaMemcpyAsync(image_device, image_host, size_image, ...);
```

含义：**整幅缓冲按「真实内存布局」原样拷到 GPU**（含每行末尾 padding）；GPU 内核仍只用 `x < width` 的有效像素，padding 不会被双线性采样读到（坐标出界已由内核处理）。

3. **letterbox 内核行步长参数**：

```cpp
warp_affine_bilinear_and_normalize_plane(
    image_device, src_line_size_bytes, image.width, image.height, ...);
```

第二个参数由固定的 `image.width * 3` 改为 **`src_line_size_bytes`**，与主机侧缓冲一致。

### 4.3 新增源代码注释（英文）

在 `src_line_size_bytes` 上一段 **4 行英文注释**，说明：

- 为何要用 `line_size`（QImage RGB888 行对齐示例）；
- 若误用紧凑 `width*3` 会怎样（每行错位 padding、沿行方向撕裂）。

**为何不用中文**：在 **未对 `.cu` 统一加 `/utf-8`** 的 MSVC + nvcc 环境下，**UTF-8 中文注释可能被按系统代码页误读**，曾导致 **声明行被截断、编译失败**；故此处 **强制使用 ASCII/英文注释**。

### 4.4 保留的注释

- `// speed up`（原有，未改语义）：表示沿用异步 `cudaMemcpyAsync` 与流。

---

## 5. `yolo_infer_c.cpp` — `yi_infer_bgr` 透传 stride

### 5.1 删除/替换的逻辑

**删除**（概念上）：

- 先构造 `cv::Mat(height, w, CV_8UC3, ptr)` 或 `cv::Mat(..., stride)` 再取 `frame.data` 传给 `yolo::Image` 的路径。  
  **问题**：即使 `cv::Mat` 带 step，旧 `yolo::Image(data, w, h)` **仍丢弃 stride**。

**替换为**：

```cpp
const int line_size = (stride > 0) ? stride : (w * 3);
yolo::Image img(bgr, w, height, line_size);
yolo::BoxArray results = ctx->infer->forward(img);
```

### 5.2 新增注释（中文）

- 「不再包装 `cv::Mat`：stride 直接透传给 `yolo::Image`，避免行对齐丢失。」
- 「`stride <= 0` 视为紧凑（`width * 3`）。」

### 5.3 计时语义

- `t0` → `t1`：仍表示进入推理前极短准备（现主要为 stride 计算 + `Image` 构造）。
- `t1` → `t2`：`forward` 推理。

---

## 6. `InferWorker.cpp` — Qt 帧 → BGR 紧凑 `cv::Mat`

### 6.1 新增头文件

```cpp
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
```

用于 `cv::Mat`、`cv::cvtColor`。

### 6.2 删除/替换的逻辑

**删除**（概念上）：

- 手写双重 `for` 循环，逐像素 `R/B` 交换写入 **另一张 `QImage`**，再 `yi_infer_bgr(..., bgrImg.bits(), ..., bgrImg.bytesPerLine())`。

**问题**：

- 性能差（全图 Python 级逐像素访问）；
- 仍依赖 `QImage` 的 **4 字节对齐** `bytesPerLine()`，若下游某处仍假设紧凑，易再次踩坑。

**替换为**：

1. `QImage` 仍统一为 `Format_RGB888`（与原先一致）。
2. **零拷贝视图**：`cv::Mat rgbView(h, w, CV_8UC3, rgb.bits(), rgb.bytesPerLine())` — 尊重 Qt 行对齐。
3. **`cv::cvtColor(rgbView, bgr, cv::COLOR_RGB2BGR)`** — OpenCV 输出 **`bgr` 默认 `step == w*3`（紧凑）**（常规构建下），同时完成通道交换。
4. 调用：

```cpp
yi_infer_bgr(m_handle, bgr.data, w, h, static_cast<int>(bgr.step), ...);
```

### 6.3 新增块注释（中文）

说明：

- 旧手写双 for + `QImage` 对齐与 `width*3` 不整除时的 stride 风险；
- `810×1080` 示例（2430 → 2432）；
- `cv::cvtColor` 的性能与紧凑 `step` 收益。

---

## 7. 行为与性能影响

| 项目 | 说明 |
|------|------|
| 正确性 | 带行 padding 的输入与紧凑输入均与 GPU letterbox 内核一致，消除「撕裂」伪影。 |
| 兼容旧工程 | `yolo::Image(ptr,w,h)` 未传第 4 参时等价于过去。 |
| GUI 路径 | 双保险：`yolo::Image` 支持 stride + `InferWorker` 输出紧凑 BGR。 |
| 预处理耗时 | `cvtColor` 通常显著快于手写双循环（具体依赖 CPU）。 |

---

## 8. 验证建议

1. 使用 **宽度满足 `width×3 % 4 != 0`** 的图（如 810 宽）复测；修复前后差异应最明显。
2. 同图同 `engine` 对比 `D:\YOLOv8-seg` 与 TRT Forge：框数量、类别、置信度、mask 轮廓应收敛。
3. 编译前若 **`yolo_infer.dll` 被占用**（`TRTForge.exe` 正在运行），链接器可能报 `LNK1104`；需先关闭进程再生成。

---

## 9. 附录：同仓库内其它相关改动（若需完整追溯）

以下条目来自 **同一 TRT Forge 工程** 的早期配置/环境类修改，**与 stride 修复不同文件**，但常与本地编译联调一起出现：

| 文件 | 内容摘要 |
|------|----------|
| `gui/CMakeLists.txt` | `QT_DIR_HINT`、`OPENCV_DIR` 默认路径；CUDA 11.8 + 新 MSVC 的 `STL1002` 旁路（`NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS`、`CUDAFLAGS`、`add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)` 等） |
| `gui/CMakePresets.json` | 与上一致的缓存变量与环境变量 |
| `gui/README.md` | 环境路径与示例命令同步 |
| `gui/inference/yolo_infer/CMakeLists.txt` | `OPENCV_DIR` 默认路径；子工程 CUDA flag 与父工程协调 |

**stride 修复的核心交付**仍以 **§2 表中四个文件** 为准。

---

## 10. 文档版本信息

- **主题**：YOLO 推理输入行步长（`line_size` / stride）与 TRT Forge GUI 预处理。
- **适用目录**：`trtexec/gui/` 下 `yolo_infer` 动态库与 Qt 推理工作线程。
- **维护建议**：若未来在 `.cu` 中增加注释，**优先 ASCII**，或为 CUDA 目标单独启用 `/utf-8` 后再写中文注释。
