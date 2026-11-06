# Change: 推理动态库移到独立工作进程（崩溃不拖垮 Web 服务）

## Why

训练、建引擎、目录批量、视频推理早已在子进程里运行，但下面三条链路仍在 **Web 服务进程内** 用 ctypes 直接调用 C++ 动态库：

| 链路 | 入口 | 调用 |
|------|------|------|
| 单图推理（YOLO / Anomalib / ONNX） | `POST /api/infer/yolo`、`/api/infer/anomalib` | `infer.run_yolo` / `infer.run_anomalib` |
| OCR 单图 | `POST /api/infer/ocr` | `ocr_infer.run` |
| 摄像头单帧 | `POST /api/infer/frame` | 同上 |
| YOLO 自动标注 | 标注页批量自动标注（`autolabeljobs.py` 线程） | `autolabel.run` → `infer.run_yolo` |

C++ 里任何一次访问违例（坏引擎、显存异常、Paddle 内部 `exit(-1)`）都会让 Windows 直接结束整个 Python 进程：
- 网页服务消失，所有用户断开；
- 正在跑的训练虽然在子进程里，但 Web 端的任务记录与实时日志推送一起丢失；
- 产线长期运行时，一张坏图就能让整条线停下。

这是上产线前必须解决的稳定性问题，列为 P0（见 `docs/商业化差距分析与优化建议.md` 第 3 节 R1）。

## What Changes

- **新增常驻推理工作进程** `6-web/backend/infer_worker.py`：加载 `infer` / `ocr_infer` 模块并持有所有动态库句柄与缓存；Web 进程不再加载任何推理 DLL。
- **新增代理模块** `6-web/backend/inferproxy.py`：对外提供与现有函数同名同参数的 `run_yolo`、`run_anomalib`、`run_ocr`、`release_handles`、`cache_stats`、`status`；`app.py` 与 `autolabel.py` 改为调用代理。
- **通信**：
  - 控制消息与结果走 `multiprocessing.connection`（本机命名管道，带随机 authkey）。
  - 输入图像走共享内存，避免大图在管道里复制。
- **看门狗**：
  - 工作进程异常退出时，当前请求返回 503「推理进程异常退出，已自动重启」，并附模型路径与图片名，便于定位坏样本。
  - 随后自动拉起新进程。
  - 60 秒内连续崩溃 3 次则停止自动重启并在 `/api/capabilities` 标记推理不可用，避免崩溃循环。
- **超时**：单次推理超过上限（默认 60 秒；OCR 首次加载 180 秒）视为卡死，强制结束并重启工作进程。
- **开关**：`TRTFORGE_INFER_ISOLATION=1`（默认）；设为 `0` 回到进程内调用，仅用于排障。
- **打包**：冻结版通过 `TRTForgeWeb.exe --infer-worker` 启动工作进程，不额外打一个 exe。
- **版本**：`APP_VERSION` 1.27.0 → 1.28.0（安全加固提案暂不排期，本变更先合入）。

## 不做的事

- 不改任何 DLL 与 C API，不改桌面版，不动根目录 `bin` / `include` / `lib`。
- 不改 `tools/` 下批量、视频、验证脚本，它们本来就是独立子进程。
- 不做多个工作进程并行，GPU 显存有限，并行推理放到 P1 产线模式提案里评估。
- SAM / GroundingDINO 提示标注走 Python onnxruntime，不在本次范围。

## Impact

- 修改规范：`web-infer-console`（新增「推理在独立工作进程执行」要求）
- 涉及代码：
  - 新增 `6-web/backend/infer_worker.py`、`6-web/backend/inferproxy.py`
  - 修改 `6-web/backend/app.py`、`autolabel.py`、`frozen_entry.py`
- 涉及测试：新增 `test_infer_worker.py`（正常推理、结果与进程内一致、模拟崩溃后自动恢复、崩溃循环熔断、超时重启、关闭开关回退）。
- 性能：每次推理增加一次进程间往返，目标 1080p 图片额外开销 ≤ 5 ms，实测写入设计文档。
- 内存：Web 进程不再加载 CUDA / TensorRT / Paddle，常驻内存下降；工作进程承担这部分。
