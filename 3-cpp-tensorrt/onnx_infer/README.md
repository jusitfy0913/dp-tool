# onnx_infer —— 用 ONNX Runtime 直接跑 `.onnx` 的推理动态库

- **版本**：1.1.0（随 Web Studio v1.30.6 一起发布；1.0.0 见第 12 节）
- **产物**：`4-qt-studio/bin/onnx_infer.dll`（配套 `onnxruntime.dll`、`onnxruntime_providers_shared.dll`）
- **依赖**：OpenCV（图像读写与预处理）、ONNX Runtime C API（运行期按路径显式加载，不写进导入表）
- **语言**：C++17，导出纯 C ABI，Python / C# / C++ 都能直接调用

---

## 1. 它解决什么问题

在它出现之前，网页端跑 `.onnx` 只有一条路：Python 里 `import onnxruntime`。这条路的三个现实问题：

| 场景 | 只有 Python 兜底时 | 有了 onnx_infer.dll |
| --- | --- | --- |
| 打包发到现场 | 要带一整个 `onnxruntime` Python 包，且必须和 torch 的 numpy 版本兼容 | 只多两个 DLL |
| 与 `.engine` 结果对比 | 预处理/后处理在 Python 里重写了一遍，差异来源分不清 | 与 `yolo_infer.dll` 共用同一套 letterbox 与解码规则 |
| `.onnx` 与 `.engine` 切换 | 两个后端两套代码路径，参数语义容易漂移 | 同一组 `conf`/`nms`/`metadata` 参数，三条分派共用 |

现在的四级分派（Python 层实现，见 `6-web/backend/app.py`）：

1. 模型是 `.onnx` 且 `onnx_infer.dll` 可用 → **本库**（detect / segment / pose / obb / classify）；
2. 模型是 `.engine` / `.trt` 且 `yolo_infer.dll` / `anomalib_infer.dll` 可用 → TensorRT DLL；
3. 分类无法走引擎（v1.1.0 起 DLL 也只支持 `.onnx`），引擎文件会被明确拒绝；
4. 以上都不可用 → Python `onnxruntime` 兜底（`6-web/backend/onnx_fallback.py` / `classifyinfer.py`）。

---

## 2. 目录结构

```
3-cpp-tensorrt/onnx_infer/
├─ CMakeLists.txt              # 单目标；缺 ORT 头文件时只 warning 并跳过，不影响另外两个 DLL
├─ include/onnx_infer_c.h      # 唯一对外头文件（oi_* C ABI）
├─ src/
│  ├─ ort_session.{hpp,cpp}    # ORT 显式加载、EP 探测回退、会话与张量封装、metadata 读取
│  ├─ yolo_onnx.{hpp,cpp}      # detect / segment / pose / obb 四个任务的预处理与解码
│  ├─ anomalib_onnx.{hpp,cpp}  # PatchCore / PaDiM / EfficientAd 的 ONNX 管道
│  ├─ classify_onnx.{hpp,cpp}  # v1.1.0：整图分类（预处理与 classifyinfer.preprocess 逐行对齐）
│  ├─ onnx_infer_c.cpp         # 句柄生命周期、thread_local 错误串、异常不跨 C 边界
│  └─ raw_dump.hpp             # 诊断用原始张量导出（实际位于 ../anomalib_infer/src，通过 include 路径共用）
└─ third_party/onnxruntime/    # fetch_onnxruntime.ps1 下载到此，不入库
```

---

## 3. 构建与部署

### 3.1 一次性拉取 ONNX Runtime

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File dp-tool\scripts\fetch_onnxruntime.ps1
# 只要 CPU 版（安装包默认就是 CPU 版，体积小 10 倍）
powershell ... -File dp-tool\scripts\fetch_onnxruntime.ps1 -CpuOnly
```

落点：`3-cpp-tensorrt/onnx_infer/third_party/onnxruntime/`，版本记录在 `VERSION.txt`。

### 3.2 编译

```powershell
# 只编这一个库
powershell -NoProfile -ExecutionPolicy Bypass -File dp-tool\scripts\build_infer_dlls.ps1 -Targets onnx_infer
# 三个库一起编
powershell ... -File dp-tool\scripts\build_infer_dlls.ps1
```

产物与 ORT 运行库由 POST_BUILD 自动复制到 `4-qt-studio/bin`。
缺头文件时 CMake 只打印 WARNING 并 `return()`，不会让 `yolo_infer` / `anomalib_infer` 编不过。

桌面 Qt 工程默认**不纳入**这个目标（`TRTFORGE_BUILD_ONNX_INFER=OFF`），因为桌面端只用引擎；
需要一起调试时加 `-DTRTFORGE_BUILD_ONNX_INFER=ON`。

### 3.3 部署形态

```
bin\
├─ onnx_infer.dll
├─ onnxruntime.dll                    # 必须与 onnx_infer.dll 同目录
├─ onnxruntime_providers_shared.dll   # ORT 1.16+ 必需
├─ onnxruntime_providers_cuda.dll     # 可选；没有就退回 CPU
├─ opencv_core4120.dll                # 预处理 / 解码依赖（分模块版 OpenCV 4.12）
└─ opencv_imgproc4120.dll
```

查找顺序（`ort_session.cpp`，任一命中即用）：

1. 环境变量 `TRTFORGE_ORT_DIR` 指向的目录；
2. `onnx_infer.dll` 自身所在目录（正常部署形态）；
3. 系统默认搜索路径（开发机 PATH 里已有 ORT 时）。

> 为什么用 `LoadLibraryExW` + `GetProcAddress("OrtGetApiBase")` 而不是链 `onnxruntime.lib`：
> Windows 按模块名去重。后端进程里可能已经加载了 Python 的 `onnxruntime.dll`（1.20.x），
> 若导入表写死模块名，本 DLL 会复用那一份，版本不符时行为不可控。显式加载才能保证「我加载的是我旁边这一份」。

---

## 4. C API 参数说明

全部定义在 `include/onnx_infer_c.h`，结构体刻意与 `yi_*` / `ai_*` 逐字段对齐，Python 侧可复用同一份 ctypes。

### 4.1 函数表

| 函数 | 关键参数 | 返回 | 说明 |
| --- | --- | --- | --- |
| `oi_create_yolo(path, task, conf, nms)` | `path`：`.onnx` 的 **UTF-8** 路径；`task`：`0` detect / `1` segment / `2` pose / `3` obb；`conf`、`nms`：`(0,1]` | 句柄 / `nullptr` | 阈值进句柄，后续推理沿用；任务与模型结构不匹配在**创建阶段**失败 |
| `oi_infer_yolo(h, bgr, w, h, stride, &boxes, &count, &stat)` | 输入 **BGR888**；`stat` 可为 `nullptr` | `OI_OK` / 错误码 | `boxes` 指向句柄内部数组，下一次推理或销毁前有效 |
| `oi_create_anomalib(onnx, metadata, model_type, img_thr, px_thr)` | `metadata`：TRTForge `metadata.json`；两个阈值 `>=0` 覆盖、`<0` 沿用 metadata | 句柄 / `nullptr` | `model_type` 仅用于诊断（"PatchCore"/"PaDiM"/"EfficientAd"/"Auto"） |
| `oi_infer_anomalib(h, rgb, w, h, stride, &result, &stat)` | 输入 **RGB888**（注意与 YOLO 相反） | `OI_OK` / 错误码 | `result` 内所有指针由句柄持有 |
| `oi_create_classify(path)` | `path`：Ultralytics 分类 `.onnx` 的 UTF-8 路径 | 句柄 / `nullptr` | **v1.1.0 新增**。batch 维动态时自动固定为 1；宽高动态时按 224×224；不接收置信度阈值 |
| `oi_infer_classify(h, bgr, w, h, stride, &result, &stat)` | 输入 **BGR888**（与 YOLO 相同） | `OI_OK` / 错误码 | **v1.1.0 新增**。只回原始分数，不做 softmax/top-k 判定 |
| `oi_get_info(h, &info)` | `info` 不可为空 | `OI_OK` / 错误码 | 返回 `provider`（`CUDA`/`CPU`）、`ort_version`、输入尺寸、类别数、关键点数；分类句柄的 `task=4` |
| `oi_destroy(h)` | 可为 `nullptr` | — | 空操作安全 |
| `oi_last_error()` | — | `const char*` | 线程本地（`thread_local`），DLL 持有，调用方不要释放 |

返回码：`OI_OK=0`、`OI_ERR_INVALID_ARGUMENT=1`、`OI_ERR_MODEL=2`、`OI_ERR_RUNTIME=3`、`OI_ERR_INFERENCE=4`。

### 4.2 `oi_box` 字段（与 `yi_box` 对齐）

| 字段 | 含义 |
| --- | --- |
| `left/top/right/bottom` | 原图坐标系轴对齐外接框（像素） |
| `confidence` | 置信度 |
| `class_label` | 类别下标（不是 id，映射由 Python 侧 `labels.py` 完成） |
| `has_mask`, `mask_x/y/w/h`, `mask_data` | 仅 `segment` 有效；`mask_data` 为 `mask_w*mask_h` 单通道，已裁剪到框内、已缩放回原图 |
| `num_kpts`, `kpts` | 仅 `pose` 有效；`num_kpts` 从模型通道数推导（`5+3n`），不写死 17 |
| `is_obb`, `cx/cy/ow/oh/angle` | 仅 `obb` 有效；角度单位弧度，已折到 `[-π/2, π/2)` |

### 4.3 `oi_anomaly_result` 字段（与 `ai_result` 对齐）

| 字段 | 含义 |
| --- | --- |
| `source_rgb/w/h/stride` | 原图 RGB888 |
| `overlay_rgb/...` | 热力图叠加图 RGB888 |
| `mask_rgb/...` | 红色异常轮廓图 RGB888 |
| `normalized_map`, `map_w/h` | 归一化异常图 float32，值域已裁剪到 `[0,1]` |
| `raw_score` | 模型原始异常分（与 metadata 阈值同量纲） |
| `normalized_score` | 显示用归一化分 `[0,1]` |
| `is_anomaly` | `raw_score > image_threshold` |

### 4.4 `oi_frame_stat`

`src_w/src_h`、`count`、`preprocess_ms`、`inference_ms`、`postprocess_ms`、`total_ms`。
分段耗时是给性能对比表用的，单位毫秒。分类句柄的 `count` 是**类别数**（不是框数）。

### 4.5 `oi_classify_result` 字段（v1.1.0 新增）

| 字段 | 含义 |
| --- | --- |
| `num_classes` | 类别数，等于 `scores` 的长度 |
| `top1` | 原始分数最大的下标（**只是 argmax，不是判决**） |
| `top1_score` | 对应的原始分数 |
| `scores` | `num_classes` 个 float32 **原始分数**，指针由句柄持有 |
| `input_w` / `input_h` | 实际喂给网络的分辨率 |
| `names_utf8` | 模型 metadata 里 `names` 的原文（Ultralytics 写的是 Python dict 字面量）；没有则为 `nullptr` |

> 为什么不在 DLL 里算概率：`classifyinfer.interpret` 已经有一套「先判断是不是概率、
> 不是才 softmax」的口径，降级路径必须用它。C++ 再写一份，两条路径会在
> "输出已经是概率"和"输出是 logits"的模型上给出不同结论，而页面上完全看不出是哪边错了。
> 分数是事实，概率是解释，解释只保留一份。

---

## 5. 四个 YOLO 任务的实现约定

| 任务 | 输入 | 输出解析 | 易踩的坑 |
| --- | --- | --- | --- |
| `detect` | letterbox 到模型静态输入（padding 114、RGB、`/255`、NCHW） | 自动识别 `[1, 4+nc, n]` 与 `[1, n, 4+nc]` 两种布局；cxcywh→xyxy；逐类 NMS | 转置过的模型（`*.transd.onnx`）与非转置模型都能跑，不需要用户选择 |
| `segment` | 同上 | 检测头 + proto 点积 + sigmoid + 阈值 0.5，裁剪后缩放回原图 | 若把分割模型当 detect 用，会明确报「模型带有分割原型输出，请切换为 segment」 |
| `pose` | 同上 | 关键点按 `d2i` 还原到原图 | 通道数 `5+3n` 推导关键点数；把姿态模型当 detect 用会提示切 `pose` |
| `obb` | 同上 | 角度通道解析、旋转框 NMS，回填外接 AABB | 角度用弧度、已归一化；`is_obb=1` 时 `cx/cy/ow/oh/angle` 才有效 |

**输入尺寸必须是静态导出**。Ultralytics 默认就是固定尺寸；如果是动态输入，创建阶段直接报错并给出实际形状，
而不是猜一个尺寸——猜错的结果是越界读或整幅图错位，现场很难查。

### 5.5 分类（v1.1.0 新增）

分类不走 letterbox，预处理对齐 Ultralytics `classify_transforms` / `classifyinfer.preprocess`：

1. **短边等比缩放**：`scale = target / min(w, h)`，长边可能超出 `target`；
2. **中心裁剪** `target × target`，裁剪偏移用 Python 同款 `round`（0.5 时取偶数，不是 C 的 `std::round`）；
3. BGR → RGB → `/255.0` → NCHW。

以上三点任何一处偏差都会让两条路径的 `top1` 在边界样本上分叉，所以 C++ 里显式实现了 `py_round()`，
而不是用 `cvRound`。

| 项目 | 行为 |
| --- | --- |
| 输入形状 | 接受 `[1,3,H,W]` 与 `[N,3,H,W]`（动态 batch 时固定为 1）；动态宽高按 224×224 |
| 输出形状 | 接受 `[1,C]` / `[C]` / `[1,1,C]`；按输出张量名优先找 `logits` / `output0`，再按元素数兜底 |
| 分数语义 | **原始输出张量原样返回**，不 softmax、不做阈值判定 |
| 类别名 | `names_utf8` 透传模型 metadata 的 `names` 字典原文，由 Python `classifyinfer.parse_names` 解析 |
| 判定 | 全在 Python：`classifyinfer.interpret()` 负责「是不是概率 → 要不要 softmax → 置信度门槛 → 映射到中文名」 |

```c
oi_handle* h = oi_create_classify(u8"runs\\pcb\\classify_v2\\weights\\best.onnx");
const oi_classify_result* r = NULL;
oi_frame_stat st;
if (oi_infer_classify(h, bgr.data, w, h0, stride, &r, &st) == OI_OK && r) {
    for (int i = 0; i < r->num_classes; ++i) printf("%d: %f\n", i, r->scores[i]);
}
oi_destroy(h);
```

---

## 6. Anomalib 的三个模型

| 项目 | 行为 |
| --- | --- |
| metadata | 读取 `image_threshold` / `pixel_threshold` / `input_size`；缺字段直接报错并给出路径 |
| 阈值覆盖 | `>= 0` 覆盖 metadata；`< 0`（如 `-1`）沿用 metadata，与 `ai_create` 完全一致 |
| 输入尺寸 | 静态维直接用；动态维按 metadata `input_size` 固定，二者都没有则报错 |
| 预处理 | resize 到输入尺寸；归一化按 metadata 的 `normalize` 开关（imagenet mean/std） |
| 输出识别 | 先按名字找 `pred_score` / `anomaly_map`，再按元素数兜底；兼容 `[1]` 与 `[1,1]` |
| 判定 | `raw_score > image_threshold`；像素级用 `pixel_threshold` |

三个模型共用同一套 `anomalib_post.hpp`（从 `anomalib_infer_c.cpp` 抽出），所以 `.engine` 与 `.onnx`
的叠加图、轮廓、状态文字是同一份代码画的，视觉差异只可能来自张量本身。

---

## 7. 环境变量

| 变量 | 作用 | 未设置时 |
| --- | --- | --- |
| `TRTFORGE_ORT_DIR` | 指定 ONNX Runtime 目录（与 `onnx_infer.dll` 不同目录时用） | 用 DLL 同目录 → 系统搜索路径 |
| `TRTFORGE_RAW_DUMP` | 指定目录后，每次推理把输入/输出张量落盘为 `.f32` + `manifest.jsonl`，供一致性比对 | **完全不检查、不建目录、不写文件**，零运行时开销 |

---

## 8. 使用方式

### 8.1 Python ctypes（最短可用示例）

```python
import ctypes, cv2, numpy as np

dll = ctypes.WinDLL(r"4-qt-studio\bin\onnx_infer.dll")
dll.oi_create_yolo.restype = ctypes.c_void_p
dll.oi_create_yolo.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]

handle = dll.oi_create_yolo(
    r"runs\pcb\detect_v1\weights\best.transd.onnx".encode("utf-8"),
    0,      # OI_DETECT
    0.25,   # conf
    0.45,   # nms
)
if not handle:
    raise RuntimeError(dll.oi_last_error().decode("utf-8", "replace"))
# ... 声明 oi_infer_yolo / oi_destroy 原型后调用（完整结构体见 6-web/backend/infer.py）
```

> 结构体定义不要手抄，直接复用 `6-web/backend/infer.py` 里的 `OiBox` / `OiFrameStat` / `OiAnomalyResult`。

分类（v1.1.0，同样复用 `infer.py` 的 `OiClassifyResult`）：

```python
import ctypes
from infer import OiClassifyResult   # 或照抄 infer.py 的结构体定义

dll.oi_create_classify.restype = ctypes.c_void_p
dll.oi_create_classify.argtypes = [ctypes.c_char_p]
dll.oi_infer_classify.restype = ctypes.c_int
dll.oi_infer_classify.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(OiClassifyResult)),
    ctypes.POINTER(OiFrameStat),
]

h = dll.oi_create_classify(r"runs\pcb\classify_v2\weights\best.onnx".encode("utf-8"))
res, stat = ctypes.POINTER(OiClassifyResult)(), OiFrameStat()
rc = dll.oi_infer_classify(h, img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                           img.shape[1], img.shape[0], img.strides[0],
                           ctypes.byref(res), ctypes.byref(stat))
if rc == 0 and res:
    scores = [res.contents.scores[i] for i in range(res.contents.num_classes)]
    names  = res.contents.names_utf8   # bytes 或 None
dll.oi_destroy(h)
```

### 8.2 网页端

不用直接调 DLL，把模型路径填成 `.onnx` 即可。后端 `infer.py` 按后缀自动分派，
结果里的 `backend` 字段会如实返回 `onnx_infer.dll`，页面徽标显示同一个值。
详细操作见 `docs/TRTForge_Web操作手册.md` 的推理页章节。

### 8.3 批量与视频

```powershell
python dp-tool\6-web\backend\tools\infer_batch.py  --model xxx.onnx --images <dir> --task detect
python dp-tool\6-web\backend\tools\infer_video.py  --engine xxx.onnx --source clip.mp4 --task detect
```

两个脚本与网页端走同一套四级分派，输出 JSON 里同样带 `backend` 字段。
`--task classify` 时同样优先 `onnx_infer.dll`，失败才退回 `classifyinfer.run()`。

---

## 9. 应用场景

| 场景 | 为什么用它 |
| --- | --- |
| 现场机器没有 TensorRT / CUDA | 只用 CPU 也能跑，`onnxruntime.dll` 一个文件搞定 |
| 模型还没转成引擎，先验证效果 | 训练产物通常只有 `.onnx` 或 `.pt`，先直推再决定要不要转引擎 |
| 需要跨平台交付 | ONNX 是通用格式；本库是 Windows DLL，Linux 侧可用 Chromium 同款 ORT（需另编） |
| 引擎结果可疑，想确认是引擎还是模型的问题 | 同一份 `.onnx` 跑本库作为参考，见第 10 节 |
| 边缘设备显存紧张 | CPU EP 不占显存，可与训练任务共存 |
| 整图分类（良品/不良品、字符/普通零件） | `oi_create_classify` 直推 `.onnx`，免去 Python onnxruntime 进程内开销，见 5.5 |

不适合的场景：追求最低延迟（引擎仍快 3–10 倍）、需要 FP16/INT8（本库只跑模型自带精度）。

---

## 10. 一致性校验（ONNX vs TensorRT）

三件工具，全部在 `6-web/backend/tools/`：

| 工具 | 粒度 | 用法 | 退出码 |
| --- | --- | --- | --- |
| `compare_backends.py` | **结果级**：框匹配率、最小 IoU、置信度差、mask IoU、关键点距离、OBB 角度差；异常分数差、判定一致、mask 一致率、热能图差 | `--engine x.engine --onnx x.onnx --images <dir> --task detect` | `0` 通过 / `2` 超阈值 / `3` 无法比对 |
| `compare_raw_tensors.py` | **张量级**：输入与输出逐张量余弦相似度、相对最大差、二值一致率 | `--engine x.engine --onnx x.onnx --images <dir> --task detect` | 同上 |
| `test_backend_consistency.py` | 匹配/判定/对齐/配对/编码的回归用例 + 可选实测分派 | `python ... --engine x.engine --onnx x.onnx --images <dir>` | `0` 全通过 |
| `test_onnx_infer_dll.py` | C ABI 本身：错误路径、四任务、三模型、classify、provider | `python ...` | `0` 全通过 |

`compare_raw_tensors.py` 依赖 `TRTFORGE_RAW_DUMP`：它自己给子进程设这个变量，跑完再读 `manifest.jsonl`。
**张量按名字配对，不按序号**——实测 PatchCore 的 TRT binding 顺序是 `score/map/label/mask`，
ONNX 图输出是 `score/label/map/mask`，按序号会把 map 和 label 相减，得出毫无意义的巨大差异。

> `test_onnx_infer_dll.py` 支持用 `TRTFORGE_ONNX_DLL_DIR` 指向临时构建目录（例如
> `build\tmp_onnx_infer\$tmp`）。Web 服务占用 `4-qt-studio\bin\onnx_infer.dll` 时，
> 可以先在新目录验证构建，不必停服务。

### 10.1 本机实测结果（2026-11-05，Windows 10 / CPU EP）

| 用例 | 指标 | 结果 |
| --- | --- | --- |
| YOLO detect（PCB 8 张） | 匹配率 / 最小 IoU / 置信度差 | 100% / > 0.92 / < 0.02，8/8 通过 |
| YOLO detect 原始张量 | 输出余弦相似度 / 最大相对差 | 0.999998 / ≤ 1.3% |
| Anomalib（PatchCore/PaDiM/EfficientAd，8 张） | 分数差 / 判定一致 | ≤ 0.055 / 100%，8/8 通过 |
| Anomalib 原始张量 | 余弦相似度 / 二值 mask 一致率 | 0.9917 / 100% |
| Classify（静态 / 动态 batch，共 16 例；含 1792×528 长图） | 原始分数最大绝对差 / 预处理张量最大差 | **0.0 / 0.0**（逐位一致） |

### 10.2 耗时参考（CPU EP，本机实测）

| 任务 | 输入 | 预处理 | 推理 | 后处理 | 合计 |
| --- | --- | --- | --- | --- | --- |
| detect（PCB 10 类） | 960×960 | 13.4 ms | 56.5 ms | 0.2 ms | 70.2 ms |
| segment（yolov8n-seg） | 320×320 | 1.1 ms | 12.8 ms | 0.7 ms | 14.6 ms |
| pose（yolov8n-pose） | 320×320 | 1.1 ms | 10.4 ms | 0.0 ms | 11.5 ms |
| obb（yolov8n-obb） | 320×320 | 1.0 ms | 9.6 ms | 2.0 ms | 12.6 ms |
| PatchCore | 256×256 | — | — | — | 87.3 ms |
| PaDiM | 256×256 | — | — | — | 27.2 ms |
| EfficientAd | 256×256 | — | — | — | 262.0 ms |

有 CUDA provider 时 `provider` 会返回 `CUDA`；没有则 `CPU`，不会静默切换而不告知。

---

## 11. 排错表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `无法加载 onnxruntime.dll（错误码 126）` | 只拷了 `onnx_infer.dll` | 把 `onnxruntime.dll`、`onnxruntime_providers_shared.dll` 放到同一目录 |
| `onnxruntime.dll 的 API 版本低于…` | ORT 太旧或不是官方包 | 用 `fetch_onnxruntime.ps1` 拉的版本，别拿别处的 DLL |
| `找不到 OrtGetApiBase` | 那个 DLL 不是 ONNX Runtime | 同上 |
| provider 一直是 `CPU` | 安装包是 CPU 版，或缺 cuDNN | 属预期；要 GPU 就换带 `providers_cuda.dll` 的 ORT 包 |
| `该模型通道数为 56…请切换为 pose` | 任务选错 | 按提示改任务，不要改模型 |
| `YOLO ONNX 的输入尺寸是动态的` | 模型不是静态导出 | 用 Ultralytics 重新导出（默认静态），或加 `dynamic=False` |
| `metadata 缺少 image_threshold…` | metadata 不是 TRTForge 生成的 | 用训练产物里的 `metadata.json`，不要手写 |
| 与引擎结果差很大，但张量一致 | 渲染/标签/阈值在 Python 层 | 先跑 `compare_raw_tensors.py` 定位是张量还是后处理 |
| 进程里同时有 Python onnxruntime | 模块名冲突 | 本库已用显式加载规避；若仍异常，设 `TRTFORGE_ORT_DIR` 指向干净目录 |

---

## 12. 版本差异（AI 可识别）

```yaml
operator: onnx_infer
current_version: 1.1.0
release: 2026-11-09
doc_revision: 2026-11-10   # 仅文档：第 14 节并发契约，无 ABI/行为变化
version_history:
  - version: 1.0.0
    release: 2026-11-05
    note: "首个版本：detect/segment/pose/obb + anomalib 三模型；分类仍由 Python onnxruntime 执行"
diff_1_0_0_to_1_1_0:
  abi:
    added_symbols:
      - "oi_create_classify(const char* onnx_path_utf8)"
      - "oi_infer_classify(oi_handle*, const unsigned char* bgr, int w, int h, int stride, const oi_classify_result**, oi_frame_stat*)"
    added_types: ["oi_classify_result"]
    added_enum_values: ["OI_CLASSIFY = 4"]
    changed: none
    removed: none
  behavior:
    - "classify 从 Python onnxruntime 进程内 CPU 执行，改为 onnx_infer.dll（EP 自动探测，含 CUDA）"
    - "新增可选调用链：infer_worker 的 op=classify + inferproxy.run_classify"
    - "网页单图/批量/视频三条 classify 路径统一 DLL 优先、Python 兜底"
  compatibility:
    backward: "1.0.0 调用方无需改动；新增符号与枚举值均为追加"
    abi: "发布后只允许在结构体末尾追加字段"
changes:
  - id: OI-1
    version: 1.0.0
    type: new_artifact
    summary: 新增 onnx_infer.dll（C ABI），.onnx 进入动态库分派的第一级
    impact: 打包只多 2 个 DLL；不再依赖 Python onnxruntime
  - id: OI-2
    type: alignment
    summary: 结构体与 yi_box / ai_result / ai_frame_stat 逐字段对齐
    impact: Python 侧复用同一份 ctypes，不需要第二套定义
  - id: OI-3
    type: behavior
    summary: 任务与模型不匹配在创建阶段报错，动态输入尺寸直接拒绝
    impact: 错误提前到加载期，现场不再出现"跑完才发现图全错"
  - id: OI-4
    type: diagnostics
    summary: TRTFORGE_RAW_DUMP 导出原始张量，与 yolo_infer/anomalib_infer 共用 raw_dump.hpp
    impact: 未设置环境变量时零开销；设置后可做张量级比对
  - id: OI-5
    version: 1.0.0
    type: verification
    summary: 新增结果级与张量级一致性工具，不一致时以退出码失败
    impact: CI/验收可自动拦下回归，不靠人眼看报告
  - id: OI-6
    version: 1.1.0
    type: new_task
    summary: 新增整图分类任务（OI_CLASSIFY + oi_create_classify + oi_infer_classify + oi_classify_result）
    impact: 分类首次进入 C++ DLL 执行，享受 EP 自动探测与 worker 隔离
  - id: OI-7
    version: 1.1.0
    type: alignment
    summary: 分类预处理逐行对齐 classifyinfer.preprocess（短边缩放 + 中心裁剪 + py_round 偶数舍入）
    impact: 与 Python 兜底路径 16/16 用例原始分数最大差 0.0，页面不会因走哪条路径而变结论
  - id: OI-8
    version: 1.1.0
    type: contract
    summary: DLL 只回原始分数与 metadata names，softmax/门槛/中文名映射保留在 Python interpret()
    impact: 解释口径只有一份；C++ 不复制置信度逻辑，避免两条路径对 logits 模型给出不同结论
  - id: OI-9
    version: 1.1.0
    type: integration
    summary: infer.py 增加 run_classify 与句柄缓存，infer_worker/inferproxy 增加 op=classify；单图/批量/视频全部 DLL 优先
    impact: 模型是 .onnx 时页面与脚本的 backend 一律显示 onnx_infer.dll；DLL 缺失时自动降级并在日志给出原因
  - id: OI-10
    version: 1.1.0
    type: docs
    summary: 新增第 14 节「并发与多模型」，明确同步阻塞、单图、一句柄一线程三条契约与本机实测数据
    impact: 调用方不再把同一句柄塞进线程池；并发前需设 TRTFORGE_ORT_THREADS，否则线程超订会让吞吐低于串行
```

---

## 13. 已知限制

1. **YOLO 只支持静态输入尺寸**；动态输入会在创建阶段报错。Anomalib 的动态宽高可由 metadata 固定，分类的动态宽高按 224×224。
2. **只跑模型自带精度**，不做 FP16/INT8 转换；需要更低延迟请用 TRT 引擎。
3. **一个句柄不允许多线程同时推理**；并发请创建多个句柄（Python 层已做句柄缓存与 LRU）。完整契约与实测数据见 **第 14 节**。
4. **仅 Windows**：依赖 `LoadLibraryExW` 与 `__declspec(dllexport)`。
5. OBB 的角度与框尺寸直接来自模型输出，未做后处理平滑；与 `yolo_infer.dll` 的差异已由第 10 节工具覆盖。
6. **分类只有 ONNX 路径**，没有对应的 TRT 分类 DLL；`.engine` 分类模型会被明确拒绝而不是悄悄跑 Python。
7. 分类的 `top1` 只是 argmax，不代表"合格/不合格"；业务判定（置信度门槛、多类映射）在 Python 侧，见 5.5。

---

## 14. 并发与多模型（同步/异步、单/多线程）

本节是**调用契约**，不涉及 ABI 变化。测于 2026-11-10，Windows 10 / 24 逻辑核 / T1000 8GB，图 810×1080。

### 14.1 结论速查

| 问题 | 答案 | 依据 |
| --- | --- | --- |
| 支持异步（callback / future / 异步句柄）吗 | **不支持**。`oi_infer_*` 全是同步阻塞，`OrtSession::run` 内部 `Run` 返回才赋值，函数返回时结果已就绪 | `ort_session.cpp` |
| 支持一次多张图（batch）吗 | **不支持**。C ABI 只有单图入口；模型 batch 维动态时被固定成 1 | `onnx_infer_c.h` |
| 支持多模型同时在内存吗 | **支持**。每模型 `oi_create_*` 一个句柄，句柄间无全局共享状态 | `onnx_infer_c.cpp` `Handle` |
| 支持多线程吗 | **支持，但一线程一句柄** | 见 14.2 |
| 同一句柄多线程 | **禁止**，属未定义行为 | 见 14.3 |
| 错误信息是否线程安全 | 是，`last_error` 为 `thread_local`，每线程读自己的 | `ort_session.cpp` |

### 14.2 多句柄多线程：支持，但要注意 ORT 线程超订

每个 `oi_create_*` 会建**独立的 `OrtSession`**（独立 session、独立 arena），因此不存在 DLL 级锁，多线程各用各的句柄可以真并行。**但 ORT 默认给每个 session 开满 intra-op 线程池**（本机 24 核 → 每 session 24 线程），并发数一多就会互相抢核。

实测（detect ONNX，CPU EP）：

| 配置 | 单次 | 4 句柄 4 线程共 8 次 | 串行 8 次 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| 默认（每 session ≈24 线程） | 58 ms | 833 ms | 464 ms | **0.56×（更慢）** |
| `TRTFORGE_ORT_THREADS=2` | 109 ms | 421 ms | 872 ms | **2.07×** |

**用并发前先设 `TRTFORGE_ORT_THREADS`**，取值建议 `逻辑核数 ÷ 并发句柄数`：

```powershell
$env:TRTFORGE_ORT_THREADS = "6"   # 24 核 / 4 路并发
```

不设也能跑，只是并发数一多吞吐反而不如串行。

### 14.3 同一句柄多线程：禁止

结果缓冲是句柄成员，`infer()` 里先 `clear()` 再填充，两个线程打同一个句柄会互相覆盖：

```cpp
// yolo_onnx.cpp:374（decode 入口）
boxes_.clear();
masks_.clear();
kpts_.clear();
```

`oi_infer_*` 返回的 `oi_box*` / `oi_classify_result*` 指针就指向这些容器，所以并发调用后**指针内容不再对应自己那次输入**。本机 20 次并发未复现串台（窗口很短），但同结构的 `yolo_infer.dll` 实测 20/20 次结果错乱、`anomalib_infer.dll` 直接崩溃 `0xC0000409`。**不要依赖运气。**

### 14.4 正确用法（一线程一句柄）

```python
import ctypes, threading

local = threading.local()          # 每线程一份句柄

def handle_for(path: str):
    h = getattr(local, "h", None)
    if h is None:
        h = dll.oi_create_yolo(path.encode("utf-8"), 0, 0.25, 0.45)
        if not h:
            raise RuntimeError(dll.oi_last_error().decode("utf-8", "replace"))
        local.h = h
    return h

# 每个线程调用自己的 handle_for(...)，线程结束前 dll.oi_destroy(local.h)
```

三条规则：

1. **一句柄一线程**，句柄不可跨线程复用，也不可销毁后在线程内继续用。
2. **显存/内存按句柄线性增长**：每个句柄一份 session 权重，N 路并发 = N 份。
3. **同一进程内句柄总数要封顶**（Python 层 `infer.py` 已用 LRU 缓存），否则会反复创建/销毁，创建成本远大于推理。

### 14.5 与 Python/网页层的关系

`infer.py` 对 `oi_*` 调用加了全局锁（`_yolo_lock`、`_anom_lock`），`infer_worker.py` 是单连接单循环，`inferproxy.py` 再包一把 `RLock`。**所以网页端与批量脚本的推理是严格串行排队的**，DLL 的多句柄能力目前没被上层使用。要并行必须改上层（多 worker 进程或多句柄线程池），不要试图在现有锁外面直接开线程打同一个句柄。
