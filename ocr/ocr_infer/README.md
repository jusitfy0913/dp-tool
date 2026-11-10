# ocr_infer.dll

`ocr_infer.dll` 是 TRTForge 的 PaddleOCR C++ 推理动态库。TRTForge 主程序只通过 `include/ocr_infer_c.h` 中的 C API 调用，不直接持有 Paddle Inference 预测器。

## 依赖

- Paddle Inference 根目录：`4-qt-studio/bin/paddle_inference`
- 链接库：`4-qt-studio/bin/paddle_inference/paddle/lib/paddle_inference.lib`
- 检测模型示例：`4-qt-studio/bin/paddle_inference/ch_PP-OCRv4_det_infer`
- 识别模型：需要单独配置，例如 `ch_PP-OCRv4_rec_infer`
- 文本行方向识别模型：可选，例如 `ch_ppocr_mobile_v2.0_cls_slim_infer`

每个启用的模型目录至少需要包含：

- `inference.pdmodel` 或 `inference.json`
- `inference.pdiparams`
- `inference.yml` 或 `inference.yaml`

注意：`ch_ppocr_mobile_v2.0_cls_infer` / `ch_ppocr_mobile_v2.0_cls_slim_infer` 是老版方向分类模型包，通常没有 `inference.yml`，不能直接作为新版 C++ pipeline 的 `TextLineOrientation` 模型目录。UI 中方向模型目录留空即可关闭该模块。

## C API 调用顺序

```text
oi_create -> oi_infer_file / oi_infer_bgr -> oi_destroy
```

`oi_result` 中的指针由 DLL 句柄持有，只保证到下一次 `oi_infer_*` 或 `oi_destroy` 前有效。调用者需要长期保存结果时，应立即深拷贝。

## 原始数据契约

`oi_result.lines[i]` 保证以下字段一一对应：

- `index`
- `xs / ys` 多边形点
- `text_utf8`
- `rec_score`
- `textline_angle`

`raw_json_utf8` 同步包含 `rec_texts`、`rec_scores`、`rec_polys`，用于调用者做完整结果记录或二次处理。

## 运行后端与耗时诊断

`oi_frame_stat` 会返回 OCR 实际请求的运行配置和分阶段耗时：

- `device_utf8`：`cpu` 或 `gpu:0`
- `precision_utf8`：`fp32` 或 `fp16`
- `run_mode_utf8`：`mkldnn`、`paddle` 或 `paddle_fp16`
- `mkldnn_enabled`：CPU 模式下是否请求 MKLDNN
- `cpu_threads` / `rec_batch_size`：CPU 线程数与识别 batch
- `io_ms` / `pipeline_ms` / `result_ms` / `total_ms`：输入、PaddleOCR pipeline、结果转换与总耗时

当前 `oi_infer_bgr` 已改为直接走 `cv::Mat` 内存输入，不再先写临时 PNG 文件；因此 `io_ms` 正常应接近 0。若选择 `gpu:0` 后日志出现 `Please use PaddlePaddle with GPU version`，说明当前 `paddle_inference.dll` 不是可用 GPU 版，需要替换 GPU 版 Paddle Inference 包后再验证。

注意：MKLDNN 只在 CPU 模式下生效。GPU 模式即使 UI 或 C API 传入 `enable_mkldnn=1`，实际运行仍会禁用 MKLDNN，`oi_frame_stat.mkldnn_enabled` 返回 `0`，`run_mode` 为 `paddle` 或 `paddle_fp16`。

### 当前模型耗时记录

测试模型：

- 检测：`4-qt-studio/bin/paddle_inference/ch_PP-OCRv4_det_infer`
- 识别：`4-qt-studio/bin/paddle_inference/ch_PP-OCRv4_rec_infer`
- 方向：留空（不启用）
- 输入：`ocr/PaddleOCR-main/docs/datasets/images/CDLA_demo`

| 图片 | 模式 | 行数 | 关键状态 | 总耗时 |
| --- | --- | ---: | --- | ---: |
| `val_0633.jpg` | CPU + MKLDNN + FP32 | 55 | `device=cpu, run_mode=mkldnn, mkldnn=1, threads=8, rec_batch=6, io_ms=0` | 约 13.2 s |
| `val_0941.jpg` | CPU + MKLDNN + FP32 | 83 | `device=cpu, run_mode=mkldnn, mkldnn=1, threads=8, rec_batch=6, io_ms=0` | 约 17.4 s |
| `val_0633.jpg` | CPU + Paddle + FP32（关闭 MKLDNN） | 55 | `device=cpu, run_mode=paddle, mkldnn=0, threads=8, rec_batch=6, io_ms=0` | 约 100.6 s |
| `val_0633.jpg` | CPU + MKLDNN + FP32（GPU 包重编后复测） | 55 | `device=cpu, run_mode=mkldnn, mkldnn=1, threads=8, rec_batch=6, io_ms=0, pipeline_ms=14545.6` | 约 14.6 s |
| `val_0633.jpg` | GPU 0 + FP32（请求 MKLDNN） | 55 | `device=gpu:0, run_mode=paddle, mkldnn=0, threads=8, rec_batch=6, io_ms=0, pipeline_ms=1985.1` | 约 2.0 s |

结论：CPU 可用加速是 MKLDNN，关闭 MKLDNN 会从十几秒退化到约 100 秒；GPU 模式依赖 GPU 版 Paddle Inference，当前实测 `val_0633.jpg` 约 2 秒。

### Paddle Inference 2.6.2 GPU 包兼容说明

当前 GPU 包版本为 Paddle Inference 2.6.2（CUDA 11.8、cuDNN 8.6、TensorRT 8.5）。为兼容该包：

- `ocr_infer` 使用 `/MD` 运行库，避免与 GPU 包的 `paddle_inference.dll` 运行库冲突。
- `yaml-cpp` 改为随 `ocr_infer` 本地源码编译，不再链接 Paddle 包中 `/MT` 编译的 `yaml-cpp.lib`。
- PaddleOCR 新版 `cpp_infer` 中 `EnableNewIR`、`EnableCINN`、`SetOptimizationLevel`、`DeletePass` 等新 API 在 2.6.2 包中不可用，当前已在推理创建路径跳过这些调用。
- GPU 推理需要 `zlibwapi.dll`，已放入 `4-qt-studio/bin`，必须是 x64 版本。

## Web ctypes 调用（v1.27.0）

浏览器推理页不直接链接这个 DLL。`6-web/backend/ocr_infer.py` 用 ctypes 按 `ocr_infer_c.h` 声明 `oi_create`、`oi_infer_bgr`、`oi_destroy`，和 `onnx_infer.dll` 分开放，因为两边都有 `oi_` 前缀。

加载是懒的：第一次文字识别请求才 `WinDLL`。Windows 上必须先准备两套搜索路径：

| 路径 | 加到哪里 | 为什么 |
| --- | --- | --- |
| `4-qt-studio/bin` | `os.add_dll_directory` + `PATH` | `ocr_infer.dll`、`paddle_inference.dll` 所在目录 |
| `ocr/ocr_infer/third_party/paddle_inference/third_party/install/mklml/lib`（或安装包的 `bin`） | **必须同时进 `PATH`** | 飞桨 2.6.2 的 `dynamic_loader` 自己调 `LoadLibrary("mklml.dll")`，只认 PATH |

**这里有个坑（v1.30.2 修）**：只 `add_dll_directory` 是不够的。`AddDllDirectory` 只对 `LoadLibraryEx(... LOAD_LIBRARY_SEARCH_USER_DIRS)` 生效，
而飞桨内部用的是普通 `LoadLibrary`——实测只加 `add_dll_directory` 时 `oi_create` 直接报
`PreconditionNotMetError: The third-party dynamic library (mklml.dll) ... (error code is 126)`，
**CPU 和 GPU 都起不来**（MKL 是 CPU 算子依赖，`device=gpu:0` 也要先加载）。

`6-web/backend/ocr_infer.py` 现在会把 `bin` 和 `mklml` 目录**同时** `add_dll_directory` 并前置到进程 `PATH`，
顺序是「先加目录、再加载 DLL」。安装包用 `package_web_user.ps1 -WithOcr` 时会把 `mklml.dll` 拷进安装目录的 `bin`，
并把 `version.txt` 写成 `bin/paddle_inference_version.txt`，那是双保险（`bin` 本来就在启动脚本的 PATH 里）。

编译产物也要自带 `mklml.dll`：`ocr/ocr_infer/CMakeLists.txt` 的 POST_BUILD 会把它和 `libiomp5md.dll`
一起拷到 `4-qt-studio/bin`，`TRTForge.exe` 因此不用改任何环境变量。

`oi_result` 的指针在下一次推理或销毁后失效，Python 侧在返回前把文字、分数、角度和多边形都拷进普通字典。句柄最多缓存 2 个；正在推理的句柄不会被淘汰。

```yaml
version_diff:
  component: ocr_infer
  old_version: "桌面版首版，仅 TRTForge.exe 调用"
  new_version: "1.27.0 Web 也可调用，DLL 本体未改"
  unchanged:
    - "ocr_infer.dll / ocr_infer_c.h / 4-qt-studio 桌面源码"
    - "Paddle Inference 2.6.2，CUDA 11.8，cuDNN 8.6"
  added_callers:
    - "6-web/backend/ocr_infer.py"
    - "POST /api/infer/ocr"
    - "infer_batch.py kind=ocr"
  parameters:
    det_dir: "必填，含 inference.pdmodel、inference.pdiparams、inference.yml"
    rec_dir: "必填，同上；字典写在 yml 的 PostProcess.character_dict"
    textline_ori_dir: "可空。老版 cls 需自备 yml，类别 0=0 度，1=180 度"
    device: "cpu 或 gpu:0，默认 cpu"
    enable_mkldnn: "默认开，只在 CPU 生效"
    cpu_threads: 8
    rec_batch_size: 6
    det_thresh: 0.3
    det_box_thresh: 0.6
    det_unclip_ratio: 1.5
    rec_score_thresh: 0
  usage:
    single_image: "推理页选 OCR，或 POST /api/infer/ocr"
    directory_batch: "kind=ocr，产出 ocr/*.ocr.json 与 ocr_summary.csv"
    cli_check: "4-qt-studio/bin/ocr_basic.exe <det_dir> <rec_dir> <image> [cls_dir]"
  scenarios:
    - "登机牌、单据、标签上的印刷中文和英文"
    - "倒着的文本行：加上 ch_ppocr_mobile_v2.0_cls，并使用补好的 inference.yml"
  not_in_this_version:
    - "inference.json（Paddle 3.x）"
    - "文档方向、UVDoc"
    - "视频和摄像头"
  measured_2026_11_05:
    image: "docs/version2.x/ppocr/images/ch_ppocr_mobile_v2.0/00006737.jpg"
    det: "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_det_infer"
    rec: "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_rec_infer"
    cls: "9-python-paddleOCR/model/legacy_v4/ch_ppocr_mobile_v2.0_cls_infer"
    lines: 75
    sample_text: "登机牌"
    cpu_mkldnn_ms: 3765
    gpu0_ms: 1147
    upright_angle: 0
    rotated_180_angle: 1
```

官方 `ch_PP-OCRv4_*` 与 `ch_ppocr_mobile_v2.0_cls` 压缩包没有 `inference.yml`。Web 用 `6-web/backend/tools/gen_legacy_ocr_yml.py` 生成，已有文件默认不覆盖。方向模型的 yml 把输入写成宽 192、高 48，类别列表是 `0_degree` 与 `180_degree`。

```yaml
version_diff:
  component: ocr_infer_runtime
  old_version: "1.30.1"
  new_version: "1.30.2"
  symptom:
    - "推理页点开始推理，CPU 和 GPU 都失败"
    - "错误：PreconditionNotMetError: The third-party dynamic library (mklml.dll) ... (error code is 126)"
    - "模型下拉正常，说明 ocr_infer.dll 本身加载成功，挂在 oi_create"
  root_cause: "飞桨 2.6.2 的 dynamic_loader 用普通 LoadLibrary(\"mklml.dll\") 找 MKL；AddDllDirectory 只影响 LoadLibraryEx 的 USER_DIRS，对它无效"
  old_behaviour:
    - "6-web/backend/ocr_infer.py 只对 mklml/lib 调 os.add_dll_directory"
    - "4-qt-studio/bin 里只有 libiomp5md.dll 和 mkldnn.dll，没有 mklml.dll"
    - "ocr/ocr_infer/CMakeLists.txt 的 POST_BUILD 依赖清单漏了 mklml.dll"
  new_behaviour:
    - "ocr_infer.py 把 bin 与 mklml 目录同时 add_dll_directory 并前置到进程 PATH"
    - "ocr/ocr_infer/CMakeLists.txt 的 POST_BUILD 复制 mklml.dll 到 TRTFORGE_BIN_DIR"
    - "4-qt-studio/CMakeLists.txt 找 mklml 时兼容 bin/paddle_inference 与 ocr/ 下的完整飞桨包"
    - "test_ocr_api.py 增加 3 项回归：能定位 mklml、mklml 目录在 PATH、bin 在 PATH"
  verification_2026_11_09:
    image: "ocr/PaddleOCR-main/docs/version2.x/ppocr/images/ch_ppocr_mobile_v2.0/00006737.jpg"
    models: "9-python-paddleOCR/model/legacy_v4（ch_PP-OCRv4_det/rec）"
    cpu_mkldnn: "75 行，4529 ms（页面）/ 5209 ms（接口首次）"
    gpu0: "75 行，9446 ms（接口首次）"
    tests: "test_ocr_api 13 项、test_ocr_infer_dll 10 项全绿"
```

## 首版说明

TRTForge UI 默认使用 `cpu + MKLDNN`。如需 GPU，可在 OCR 模式中选择 `GPU 0`，但必须确认 Paddle Inference 包为 GPU 版且 CUDA/cuDNN 运行库完整。
