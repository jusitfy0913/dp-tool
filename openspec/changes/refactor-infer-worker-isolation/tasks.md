## 1. 工作进程与代理

- [x] 1.1 新增 `infer_worker.py`：命名管道连接、authkey 校验、消息循环、调用原 `infer` / `ocr_infer` / `onnx_fallback`、异常回传
- [x] 1.2 新增 `inferproxy.py`：同名函数、请求锁、共享内存按需扩容、懒启动、看门狗、超时、熔断、`TRTFORGE_INFER_ISOLATION=0` 直通
- [x] 1.3 `frozen_entry.py` 支持 `--infer-worker`
- [x] 1.4 `app.py` 推理调用点、`/api/infer/cache`、`/api/capabilities`、关闭钩子改走代理
- [x] 1.5 `autolabel.py` 改走代理
- [x] 1.6 推理页：熔断时顶部提示 +「重试推理进程」按钮（新增 `POST /api/infer/worker/restart`）

## 2. 测试

- [x] 2.1 `test_infer_worker.py`：一致性（开关 0/1 对比）、崩溃恢复、熔断、超时、关闭开关回退
- [x] 2.2 既有 `test_infer_*`、`test_onnx_fallback.py`、`test_backend_consistency.py` 全部通过
- [x] 2.3 `run_all_tests.py` 全部通过
- [x] 2.4 性能实测（1080p / 4K 各 100 次）写入 `design.md` 第 6 节
- [ ] 2.5 浏览器实测：单图、摄像头、OCR、自动标注；模拟崩溃后页面提示并恢复（已测：YOLO 单图、强杀工作进程后自动恢复；OCR/自动标注已由 test_infer_worker 与 test_autolabel_batch 覆盖，摄像头待实机）
- [ ] 2.6 打包版 `TRTForgeWeb.exe` 实测工作进程能启动、能恢复

## 3. 文档

- [x] 3.1 `APP_VERSION` → 1.28.0
- [x] 3.2 `6-web/README.md`：架构变化、环境变量、排障、YAML `version_diff`
- [x] 3.3 `docs/WORKFLOW_GUIDE.md`：推理进程说明与排障表
- [x] 3.4 `docs/WebStudio架构与功能介绍.md`：更新总架构图与请求流程图
- [x] 3.5 `docs/TRTForge_Web操作手册.md`：「推理进程异常」提示与处理
- [x] 3.6 `docs/商业化差距分析与优化建议.md` 标记 R1 状态
