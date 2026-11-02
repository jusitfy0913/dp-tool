# Tasks

## 1. 后端公共层

- [x] 新增 `backend/inferjobs.py`：`safe_stem` / `annotated_name` / `collect_images` /
      `write_csv`(utf-8-sig, `\n`) / `build_zip`(原子) / `resolve_artifact`(沙箱) / `new_job_dir`
- [x] `infer.py` 新增 `render_yolo()` 返回 BGR ndarray（不再强制 PNG 往返）与 `encode_image(quality)`
- [x] `labels.py` 增加 `(path, mtime, size)` 缓存（上限 8）与 `cache_clear()`，为 8 fps 逐帧调用去 IO
- [x] `onnx_fallback.py` 抽出 `resolve_onnx_path()`，四条链路共用同一套"该提示哪个 ONNX"逻辑

## 2. 批量脚本

- [x] `tools/infer_batch.py`：YOLO + Anomalib，支持 `--limit` / `--no-recursive` / `--jpeg-quality` /
      `--no-zip` / `--no-boxes`，事件走 `TRTFORGE_JSON`
- [x] 单图失败记 `status=error` + `error` 原因，不抛全局异常
- [x] 每张图产出净化命名的标注图；结束时写 `results.csv` / `results.json` / `job.json` / `annotated.zip`

## 3. 视频脚本

- [x] `tools/infer_video.py`：`VideoCapture` 读、`VideoWriter` 写，编码器逐级探测（写一帧验证非空）
- [x] `--stride` 抽帧时输出帧率改为 `源帧率 ÷ stride`
- [x] `--max-frames` / `--long-side` / `--save-detections`
- [x] 进度事件按 0.5 s 节流，避免长视频刷爆事件缓冲

## 4. 接口与前端

- [x] `POST /api/infer/batch`、`POST /api/infer/video`、`POST /api/infer/frame`、`GET /api/infer/artifact`
- [x] 抽取 `_run_yolo_once()` 供单张与单帧共用（误差只在入口参数，不在推理逻辑）
- [x] `index.html` 新增「批量 · 视频 · 摄像头」卡片（三个 mode tab + 共用进度条 + 结果表）
- [x] `js/inferbatch.js`：WebSocket 事件流、进度与 ETA、结果表、视频内联播放、canvas 叠框
- [x] 摄像头：`getUserMedia` + 在途标志位 + 失败时徽标回「未启动」并给出原因
- [x] 前端过滤引擎内部刷屏日志（`num_image` 每帧一行，其余按内容去重）

## 5. 测试

- [x] `test_infer_jobs.py` 34 项：收集排序、命名净化、CSV BOM 与换行、zip、路径穿越拒绝
- [x] `test_infer_batch.py` 75 项：命令拼装、坏图容错、Anomalib 批量、单帧接口、产物下载
- [x] `test_infer_video.py` 43 项：命令拼装、编码器探测、抽帧时长不变、帧数上限、失败路径
- [x] `run_all_tests.py` 收录并通过 15 个脚本（约 27 s）

## 6. 文档与验收

- [x] `6-web/README.md`：新增 v1.16 小节、4 个接口与请求体、代码结构、测试与实测结果
- [x] 根 `readme.md`：新增 v1.16 版本记录（老/新对比、三个设计决策、验收数据）
- [x] `docs/WORKFLOW_GUIDE.md`：§7.5 三种形态的操作与验收线，§0 流程图补上出口形态
- [x] 浏览器实测：批量 3 张、视频 30 帧内联播放、单帧接口、无摄像头错误路径
