# web-yolo-pipeline Specification

## Purpose
把「训练 → 导出 `.transd.onnx` → 校验转置布局 → 编译引擎 → 补 `labels_*.json` →
引擎精度验证」串成一条可停止、可观测、可回溯的链，并给出引擎在验证集上的量化精度。
每个阶段都是独立任务（有自己的 `run_id` 与日志），整链记录原子落盘、服务重启后可查。

## Requirements
### Requirement: Pipeline Orchestration
后端 SHALL 提供 YOLO 全流程串联接口，按固定顺序推进各阶段，每阶段是一个独立任务。

#### Scenario: 启动串联
- **WHEN** 客户端提交 `data`（dataset.yaml）与串联参数
- **THEN** 后端 SHALL 创建整链记录并返回 `id`，随后在后台线程逐段推进
- **AND** 每个子进程阶段 SHALL 有独立的 `run_id`，日志与事件走既有 `WS /ws/runs/{id}`

#### Scenario: 阶段顺序固定
- **WHEN** 串联正常执行
- **THEN** 阶段顺序 SHALL 为 `train → export → probe → build → labels → val`，
  且 `probe`（校验转置布局）必须排在 `build` 之前

#### Scenario: 跳过训练
- **WHEN** 客户端设置 `skip_train=true`
- **THEN** 后端 SHALL 跳过 `train` 并从 `export` 开始
- **AND** 未提供 `.pt` 绝对路径时 SHALL 直接拒绝，不得进入导出阶段

#### Scenario: 关闭精度验证
- **WHEN** 客户端设置 `run_val=false`
- **THEN** 整链 SHALL 在 `labels` 完成后结束，不启动 `val`

#### Scenario: 某段失败
- **WHEN** 任一阶段以失败结束
- **THEN** 整链 SHALL 标记为 `failed` 并记录失败阶段与原因
- **AND** 该阶段之后的阶段 SHALL 标记为 `skipped`，不得继续执行
- **AND** 已完成阶段的产物 SHALL 保留，不得删除

#### Scenario: 停止串联
- **WHEN** 客户端请求停止一条进行中的链
- **THEN** 后端 SHALL 只终止**当前**阶段的子进程
- **AND** 当前段标记 `cancelled`、后续段标记 `skipped`、整链标记 `cancelled`
- **AND** 对已结束的链请求停止 SHALL 返回冲突状态码

#### Scenario: 元数据阶段不起子进程
- **WHEN** 某阶段只做文件读写（校验 ONNX 布局、写 `labels_*.json`）
- **THEN** 该阶段 SHALL 以可执行函数形式在编排线程内同步完成，不创建 run

### Requirement: Pipeline Persistence
整链记录 SHALL 原子落盘，并在服务重启后可查。

#### Scenario: 原子写入
- **WHEN** 记录状态变化
- **THEN** 后端 SHALL 先写 `<id>.json.part` 再改名为 `<id>.json`，不得出现半个 JSON

#### Scenario: 重启清理遗留记录
- **WHEN** 服务启动且磁盘上存在 `status ∈ {pending, running}` 的记录
- **THEN** 后端 SHALL 把整链标为 `cancelled`、原因写「服务重启，任务已中断」
- **AND** 进行中的阶段标 `cancelled`、未开始的阶段标 `skipped`、已完成的阶段保持原状

#### Scenario: 历史列表包含磁盘记录
- **WHEN** 客户端查询历史
- **THEN** 返回结果 SHALL 合并内存与磁盘记录，历史项带 `live=false`
- **AND** 按创建时间倒序并受 `limit` 限制

#### Scenario: 单条详情标记是否存活
- **WHEN** 客户端查询一条只在磁盘上的记录
- **THEN** 响应 SHALL 含 `live=false`，前端据此停止轮询并隐藏可停止状态

### Requirement: Stage Log Retrieval
后端 SHALL 允许按阶段取回该段任务的日志尾部，即使该段 run 已不在内存中。

#### Scenario: 已结束阶段的日志
- **WHEN** 客户端请求 `GET /api/pipeline/{id}/log?stage=export&lines=400`
- **THEN** 后端 SHALL 读取该阶段对应 run 的 `runtime/logs/{run_id}.log` 尾部并返回纯文本
- **AND** `lines` SHALL 被夹取到合法范围，超长单行不得导致超大响应

#### Scenario: 非法请求
- **WHEN** 阶段名未知、该阶段没有 `run_id`、链不存在，或阶段名包含路径穿越片段（如 `../`）
- **THEN** 后端 SHALL 返回 404，且不得读取 `runtime/logs` 之外的任何文件

### Requirement: Engine Validation
后端 SHALL 能在验证集上量化引擎精度。

#### Scenario: 精度指标
- **WHEN** 精度验证完成
- **THEN** 报告 SHALL 包含 mAP50、mAP50-95、precision、recall、评估图片数与 GT 实例数
- **AND** AP SHALL 按 COCO 口径计算（IoU 0.50:0.95 步长 0.05、全点插值、按置信度降序贪心匹配）

#### Scenario: IoU 阈值边界
- **WHEN** 某框与 GT 的 IoU 恰好等于阈值（浮点表示可能略小）
- **THEN** 该框 SHALL 视为命中（阈值比较带容差）

#### Scenario: 非检测任务
- **WHEN** 任务类型不是 `detect`
- **THEN** 后端 SHALL 明确拒绝精度验证并说明原因，不得套用检测口径给出数字

#### Scenario: 报告产物
- **WHEN** 精度验证成功
- **THEN** 产物 SHALL 包含 `val_report.json` 与 `val_report.csv`（`utf-8-sig`，Excel 直接打开不乱码）

### Requirement: Pipeline UI
训练页 SHALL 提供串联卡片，展示阶段进度、产物与精度。

#### Scenario: 参数复用
- **WHEN** 用户填写串联参数
- **THEN** dataset.yaml / epochs / imgsz / project / name SHALL 直接取「训练参数」表单，不重复输入

#### Scenario: 阶段可视化
- **WHEN** 串联运行中
- **THEN** 卡片 SHALL 显示每段状态（等待 / 进行中 / 完成 / 失败 / 停止 / 跳过）与耗时
- **AND** 每段 SHALL 提供查看该段日志的入口：该段仍在跑时切到实时日志面板，
  已结束或服务重启过时改为在卡片内直接展示日志文件尾部，且不得打断正在查看的其他任务日志

#### Scenario: 产物可衔接
- **WHEN** 串联结束
- **THEN** 产物表 SHALL 为引擎提供「去推理引擎」、为标签文件提供「带到推理页」、
  并为任意行提供「复制路径」

#### Scenario: 精度汇总
- **WHEN** 报告存在
- **THEN** 卡片 SHALL 显示 mAP50 / mAP50-95 / P / R / 评估图片数 / GT 实例数

#### Scenario: 历史记录
- **WHEN** 用户查看历史
- **THEN** 卡片 SHALL 列出最近记录并可切换查看其阶段、产物与精度
