# ocr_infer.dll

`ocr_infer.dll` 是 TRTForge 的 PaddleOCR C++ 推理动态库。TRTForge 主程序只通过 `include/ocr_infer_c.h` 中的 C API 调用，不直接持有 Paddle Inference 预测器。

## 依赖

- Paddle Inference 根目录：`trtexec/gui/bin/paddle_inference`
- 链接库：`trtexec/gui/bin/paddle_inference/paddle/lib/paddle_inference.lib`
- 检测模型示例：`trtexec/gui/bin/paddle_inference/ch_PP-OCRv4_det_infer`
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

- 检测：`trtexec/gui/bin/paddle_inference/ch_PP-OCRv4_det_infer`
- 识别：`trtexec/gui/bin/paddle_inference/ch_PP-OCRv4_rec_infer`
- 方向：留空（不启用）
- 输入：`PaddleOCR-main/docs/datasets/images/CDLA_demo`

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
- GPU 推理需要 `zlibwapi.dll`，已放入 `trtexec/gui/bin`，必须是 x64 版本。

## 首版说明

TRTForge UI 默认使用 `cpu + MKLDNN`。如需 GPU，可在 OCR 模式中选择 `GPU 0`，但必须确认 Paddle Inference 包为 GPU 版且 CUDA/cuDNN 运行库完整。
