## 1. 台账与配置

- [x] 1.1 新增 `runledger.py`：追加写 JSONL、按 run_id 折叠成最新状态、损坏行跳过
- [x] 1.2 `config.py` 增加 `RUNS_LEDGER`、`LOG_MAX_BYTES`、`LOG_BACKUPS`、`LOG_RETENTION_DAYS`、`AUTO_RESUME`

## 2. 任务生命周期

- [x] 2.1 `runners.RunManager.create()` 成功后写 `started` 记录
- [x] 2.2 `_finalize()` 写 `finished` / `failed` / `cancelled` 记录（`stop()` 先改状态，由收尾统一落账）
- [x] 2.3 启动时 `recover_interrupted()`：running → interrupted，并返回只读历史 Run

## 3. 接口

- [x] 3.1 `GET /api/runs/history?limit=`（默认 50、上限 500）
- [x] 3.2 `/api/runs/{id}/log` 支持已滚动分片，`lines=N` 从「最新分片」往前取

## 4. 日志轮转与清理

- [x] 4.1 单次任务日志按 `LOG_MAX_BYTES` 滚动，保留 `LOG_BACKUPS` 份
- [x] 4.2 启动时按 `LOG_RETENTION_DAYS` 整族清理过期任务日志

## 5. 崩溃自动续训

- [x] 5.1 `yolo-train` 失败且存在 `weights/last.pt` 时自动用 `--resume-from` 续训一次
- [x] 5.2 台账记录 `resumed_from`；续训衍生的任务不再触发续训

## 6. 前端

- [x] 6.1 顶栏「历史」按钮 + 浮层列表，可打开任意历史日志
- [x] 6.2 中断任务在列表中标出「已中断」（与「已停止」区分）

## 7. 验证

- [x] 7.1 新增 `tools/test_run_history.py`：折叠、损坏行、恢复、轮转、清理、历史接口、续训闸门（48 项）
- [x] 7.2 回归 `test_run_log.py`(22)、`test_exit_codes.py`(6)、`test_concurrency.py`、`test_pipeline_stages.py`(50)、`test_pipeline_log.py`(9)
- [x] 7.3 真机：运行中强杀服务后重启，历史可见、旧 running 变「已中断」、日志可读、停止 404
- [x] 7.4 更新 `6-web/README.md`、`readme.md`、`docs/WORKFLOW_GUIDE.md`
