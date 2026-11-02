# Tasks: YOLO 全流程串联与引擎精度验证

## 1. 后端编排

- [x] 1.1 新增 `6-web/backend/pipeline.py`
  - `StagePlan`：`name / label / command / cwd / expect_file / report_file / callable / kind`
  - `__post_init__` 把 `expect_file` 字符串转 `Path`（`runners` 会调 `.is_file()`，传 str 会让读线程崩溃）
- [x] 1.2 `PipelineManager.create()`：建记录 → 起 supervisor 线程
- [x] 1.3 `PipelineManager._supervise()`：逐段推进，失败/停止时 `_mark_remaining(skipped)`
- [x] 1.4 `PipelineManager.stop()`：只停当前段；已结束的链返回冲突
- [x] 1.5 `PipelineManager.list()/snapshot()`：快照深拷贝，避免调用方改到内部状态
- [x] 1.6 持久化：`.json.part` → `.json` 原子改名
- [x] 1.7 启动清理：遗留 `running` 记录标 `cancelled`（原因「服务重启，任务已中断」）
- [x] 1.8 历史列表合并磁盘记录（带 `live=false`）
- [x] 1.9 产物去重：同一路径不再按阶段名重复记一行

## 2. 精度验证

- [x] 2.1 新增 `6-web/backend/valmetrics.py`（纯计算，无 IO）
  - IoU / 贪心匹配 / 全点插值 AP / 多类 mAP / P-R 曲线
  - 阈值比较带 `1e-9` 容差（`8/16` 在浮点里是 0.4999999999999999）
- [x] 2.2 新增 `6-web/backend/tools/val_engine.py`：走 `ctypes` 调 `yolo_infer.dll` 逐张推理
  - 只支持 `detect`，其余任务直接返回 `unsupported_task`
  - 产出 `val_report.json` + `val_report.csv`（`utf-8-sig`）
  - 每张图打一行 `TRTFORGE_JSON` 进度事件

## 3. 接口

- [x] 3.1 `POST /api/pipeline/yolo`：校验参数（`skip_train` 必须给 `.pt`，`data` 必填）
- [x] 3.2 `GET /api/pipeline`、`GET /api/pipeline/{id}`（带 `live`）、`POST /api/pipeline/{id}/stop`
- [x] 3.3 `runners.create()` 容忍字符串 `expect_file`
- [x] 3.4 `/static/` 加 `Cache-Control: no-cache, must-revalidate`
- [x] 3.5 版本号 → `1.17.0`

## 4. 前端

- [x] 4.1 `index.html`：`#pipeCard`（`.card.wide`）——精度 / split / 是否验证 / 跳过训练 + 权重选择
- [x] 4.2 `js/pipeline.js`：提交、轮询、阶段渲染、产物表、精度卡片、历史表
- [x] 4.3 阶段行「日志」按钮：还在跑 → `focusRun(runId)` 实时流；已结束/重启过 → 读日志文件尾巴
- [x] 4.4 产物行按钮：去推理引擎 / 带到推理页 / 复制路径（http 非安全上下文走 `execCommand` 兜底）
- [x] 4.5 历史记录 `live=false` 时停止轮询、置灰「停止」并说明原因
- [x] 4.6 `css/style.css`：`.pipe-*` 阶段态样式、精度统计卡

## 5. 测试

- [x] 5.1 `test_pipeline_stages.py`（50 项，假 run 管理器）
- [x] 5.2 `test_pipeline_log.py`（9 项，日志尾部截断 / 长行 / 路径穿越 / 404）
- [x] 5.3 `test_val_metrics.py`（37 项，纯计算）
- [x] 5.4 `run_all_tests.py` 18 个脚本全绿

## 6. 文档

- [x] 6.1 `docs/WORKFLOW_GUIDE.md`：§5.5 一键全流程串联 + 全流程图加入精度验证
- [x] 6.2 `6-web/README.md`：串联卡片说明、接口表、代码结构、验证命令
- [x] 6.3 根 `README.md`：v1.17 版本记录（新旧对比 + 设计决策 + 验收数据）

