# Change: 推理页新增目录批量、视频与摄像头三种形态

## Why

v1.8~v1.15 把「训练 → 导出 → 编译 → 单张推理」接成了闭环，但出口只有**一张图**。
现场三个最常见的问题它都答不了：

- 一个文件夹里 500 张图怎么跑？现在只能写脚本循环调单张接口，进度、停止、产物打包全要自己实现。
- 一段录像怎么跑？OpenCV 读写、编码器在不在、抽帧后时长会不会变，全部靠人踩坑。
- 相机实时画面怎么看？单张接口传一张 3 MB 的图、还回一张 base64 标注图，8 fps 都撑不住。

## What Changes

- **目录批量**：`POST /api/infer/batch` 起一个子进程任务跑 `tools/infer_batch.py`，
  复用 `runners.py` 的停止 / 事件流 / 日志能力；产物为 `annotated/` + `results.csv` +
  `results.json` + `annotated.zip` + `job.json`。坏图不中断整批。
- **视频**：`POST /api/infer/video` 跑 `tools/infer_video.py`，编码器按 `avc1 → H264 → mp4v`
  逐级**实写一帧验证**；抽帧时输出帧率写 `源帧率 ÷ 步长`；产出可内联播放的 mp4 与 `detections.json`。
- **摄像头**：`POST /api/infer/frame` 同步处理单帧，**只回 JSON 坐标与类别色，不回图片**；
  前端 canvas 叠框，上一帧未返回则跳过该帧，保证掉帧而不是延迟累积。
- **产物下载**：`GET /api/infer/artifact` 限定在 `runtime/infer` 内，越界一律 404。
- **公共产物逻辑**：新增 `backend/inferjobs.py`（文件名净化 / CSV(utf-8-sig) / 原子 zip / 路径沙箱）。
- 版本升至 `1.16.0`。

## Impact

- Affected specs: 新增 `web-infer-batch`；`web-infer-console` 不变（单张链路行为未改）
- Affected code:
  - 新增 `6-web/backend/inferjobs.py`、`6-web/backend/tools/infer_batch.py`、`6-web/backend/tools/infer_video.py`
  - 修改 `6-web/backend/app.py`（4 个接口 + `_run_yolo_once` 抽取）、`infer.py`（`render_yolo` / `encode_image`）、
    `labels.py`（按 mtime/size 缓存）、`onnx_fallback.py`（统一 ONNX 解析）
  - 前端：`index.html` 新增批量卡片、`js/inferbatch.js`、`css/style.css`
  - 测试：`test_infer_jobs.py` / `test_infer_batch.py` / `test_infer_video.py`

## Non-Goals

- 不做"边训边推"：批量/视频只消费已编译好的引擎。
- 摄像头不支持 Anomalib（异常检测需要整图归一化与阈值判定，逐帧实时意义有限，且热力图无法在 canvas 上低成本复现）。
- 不做断点续跑批量任务：一次任务对应一个产物目录，失败重跑整批（`limit` 可用来快速定位）。
- 不在浏览器里做视频转码：编码全部在后端，前端只负责播放。

## 验收

- `run_all_tests.py` 15 个脚本全绿（新增 34 + 75 + 43 断言）。
- 批量 3 张 < 2 s、CSV 3 行、zip 可解压；视频 30 帧 < 3 s、产物在浏览器 `readyState=4`；
  单帧接口 < 500 ms 且响应不含 base64 图；越权下载返回 404。
- 无摄像头机器上点「开启摄像头」应回到「未启动」并给出原因，不卡在「启动中」。
