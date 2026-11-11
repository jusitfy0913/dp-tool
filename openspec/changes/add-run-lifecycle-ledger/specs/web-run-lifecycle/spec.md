## ADDED Requirements

### Requirement: 任务运行台账

系统 SHALL 把每次任务的生命周期写入追加式台账 `runtime/runs.jsonl`，重启后仍可查询。

台账记录 MUST 包含 `run_id`、`kind`、`status`、`started_at`、`finished_at`、`exit_code`、
`command`、`log_file`；`status` 取值 MUST 为 `running`、`finished`、`failed`、`cancelled`、
`interrupted` 之一。

写入失败 MUST NOT 改变任务本身的运行状态，只允许打一行诊断信息。

#### Scenario: 任务开始与结束各写一条
- **WHEN** 用户启动一次 YOLO 训练并正常跑完
- **THEN** 台账中出现同一 `run_id` 的两条记录，先 `running` 后 `finished`，且带退出码 0

#### Scenario: 台账行损坏时不影响读取
- **WHEN** 台账最后一行只写了半截 JSON（进程被强杀）
- **THEN** 读取时跳过该行，前面所有任务仍能正常列出

#### Scenario: 台账不可写
- **WHEN** `runtime` 目录没有写权限
- **THEN** 任务照常运行与结束，接口仍能返回任务状态，只是历史列表为空

### Requirement: 重启后恢复中断任务

系统 SHALL 在启动时把台账中仍为 `running` 的记录改判为 `interrupted`（`orphan=true`），
并向台账追加一条 `interrupted` 记录。

被改判的历史任务 MUST 只读：停止接口对它返回 404，日志接口仍可读取。

#### Scenario: 服务被强杀后重启
- **WHEN** 训练运行中直接把 web 进程结束，再重新启动
- **THEN** 该任务的最终状态是 `interrupted`，历史列表可见，日志仍能打开

#### Scenario: 停止已结束的历史任务
- **WHEN** 用户对一条 `interrupted` 历史任务调用 `POST /api/runs/{id}/stop`
- **THEN** 返回 404，不产生任何副作用

### Requirement: 任务历史接口

系统 SHALL 提供 `GET /api/runs/history` 返回台账中的历史任务，按开始时间倒序，
默认最近 50 条，支持 `limit` 参数（上限 500）。

`GET /api/runs` 的语义 MUST 保持不变：只返回当前进程内的任务。

#### Scenario: 默认返回最近 50 条
- **WHEN** 台账里有 200 条历史任务，客户端请求 `/api/runs/history` 不带参数
- **THEN** 返回 50 条，按 `started_at` 倒序

#### Scenario: limit 超上限被夹紧
- **WHEN** 客户端请求 `/api/runs/history?limit=9999`
- **THEN** 最多返回 500 条，不报错

### Requirement: 任务日志轮转与清理

单次任务日志 SHALL 在超过 `LOG_MAX_BYTES`（默认 20 MB）时滚动为 `{run_id}.log.1`、
`{run_id}.log.2`…，最多保留 `LOG_BACKUPS`（默认 3）份。

`GET /api/runs/{id}/log` MUST 能读到已滚动的内容；`lines=N` 时 MUST 从最新分片向前取满 N 行。

系统 SHALL 在启动时按 `LOG_RETENTION_DAYS`（默认 30 天，设为 0 表示关闭）整族删除
长期未更新的任务日志，且 MUST NOT 删除文件名不是 `{12位十六进制}.log[N]` 的文件。

#### Scenario: 日志超过上限后滚动
- **WHEN** 一次任务的输出超过 20 MB
- **THEN** 磁盘上出现 `{run_id}.log` 与至少一个 `{run_id}.log.1`，且总量不超过 4 × 20 MB

#### Scenario: 读取尾部跨越滚动分片
- **WHEN** 日志已滚动，客户端请求 `lines=100`
- **THEN** 返回的最后一行等于最新写入的那一行，且行数不超过 100

#### Scenario: 清理过期日志不误伤其它文件
- **WHEN** 日志目录里同时有 100 天前的任务日志和 `infer_worker.log`
- **THEN** 任务日志的所有分片被删除，`infer_worker.log` 保持不动

### Requirement: 训练崩溃自动续训

`yolo-train` 任务以非零退出码结束时，若训练产物目录存在 `weights/last.pt`，系统 SHALL
自动以 `--resume-from` 续跑一次。

自动续训 MUST 最多发生一次；第二次仍失败 MUST 判为 `failed`。台账 MUST 记录 `resumed_from`。

#### Scenario: 训练进程被杀后自动恢复
- **WHEN** 训练进行到第 5 个 epoch 时进程被外部结束，且 `weights/last.pt` 存在
- **THEN** 系统自动以续训方式重跑，最终产物完整，台账能看到该任务续过一次

#### Scenario: 关闭自动续训
- **WHEN** 设置 `TRTFORGE_AUTO_RESUME=0` 后训练失败
- **THEN** 任务直接判为 `failed`，不重跑

#### Scenario: 没有权重时不重跑
- **WHEN** 训练因参数错误立即退出，目录里没有 `weights/last.pt`
- **THEN** 任务判为 `failed`，不触发续训

#### Scenario: 续训本身失败不再续训
- **WHEN** 自动续训拉起的任务再次失败
- **THEN** 不再启动第三次，任务停在 `failed`
