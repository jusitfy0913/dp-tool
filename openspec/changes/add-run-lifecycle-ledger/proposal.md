# Change: 任务运行记录落盘、日志轮转与崩溃自动续训

## Why

`6-web/backend/runners.py` 的 `RunManager._runs` 是一个纯内存字典：

- 服务一重启，页面上所有历史任务（训练、引擎构建、批量推理、自动实验）全部消失。
  `runtime/logs/{run_id}.log` 还在磁盘上，但前端再也拿不到 `run_id`，等于日志找不回来。
- 长跑任务被系统杀掉或进程崩溃后，状态停在 `running` 或直接消失，无法事后判断
  「它到底跑到第几轮、是不是被谁停掉的」。
- `runtime/logs` 没有体积上限，7×24 跑下去会把磁盘写满。
- 训练进程崩溃后只能人工点「断点续训」，13 小时的训练在第 11 小时崩掉就白等一夜。

## What Changes

- 新增**运行台账** `runtime/runs.jsonl`（追加写、按 `run_id` 最后一条为准）：
  任务创建时写 `started`，结束/停止时写 `finished`，含 `kind / status / exit_code /
  started_at / finished_at / command / log_file / expect_file / resumed_from`。
- 服务启动时加载台账，把仍是 `running` 的条目改判为 `interrupted`（`orphan=true`），
  并写回一条 `interrupted` 记录。内存里同时建出「只读历史 Run」，让前端能打开日志。
- 新增 `GET /api/runs/history`（默认最近 50 条，可 `limit`），返回台账里的历史任务；
  `GET /api/runs` 语义不变，仍只返回本次进程内的任务。
- 日志轮转：`config.py` 新增 `LOG_MAX_BYTES`（默认 20 MB）与 `LOG_BACKUPS`（默认 3）。
  单次任务日志超限滚动为 `{run_id}.log.1`…，`/api/runs/{id}/log` 读「当前文件 + 已滚动分片」。
- 过期日志清理：启动时按 `LOG_RETENTION_DAYS`（默认 30 天，0 = 不清理）整族删除长期不动
  的任务日志。单任务有上限但「任务数 × 上限」仍会无限增长，需要这一层兜底。
- YOLO 训练任务（`yolo-train`）非零退出且存在 `weights/last.pt` 时，自动以
  `--resume-from` 续跑一次（默认开，`TRTFORGE_AUTO_RESUME=0` 关）；台账记 `resumed_from`，
  第二次失败才判 `failed`。
  Anomalib 侧没有等价的命令行续训入口，本轮不含（重跑不等于续训）。
- `APP_VERSION` 1.33.3 → 1.34.0。
- **BREAKING**：无。`Run.snapshot()` 只增字段，不删字段；`/api/runs` 行为不变。

## Impact

- Affected specs:
  - `web-run-lifecycle`（新增能力：台账、恢复、历史接口、日志轮转、自动续训）
  - `web-studio`（任务历史可见，重启不丢）
- Affected code:
  - `6-web/backend/runledger.py`（新增）
  - `6-web/backend/config.py`、`runners.py`、`app.py`
  - `6-web/frontend/js/logs.js`、`js/runsbar.js`、`index.html`
  - `6-web/backend/tools/test_run_history.py`（新增测试）
