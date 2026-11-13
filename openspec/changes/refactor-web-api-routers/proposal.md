# Change: 把 app.py 的 87 条路由按功能域拆成 APIRouter 模块

## Why

`6-web/backend/app.py` 现在 4532 行、128 个函数、87 条路由。
只有自动实验一块用了 `APIRouter`（`app.include_router(autoexp_api.router)`，`app.py:4517`），
其余 77 条全挤在一个文件里。

具体代价：

1. **改动回归面大**。加一个推理参数要在 4500 行里找 `_finish_infer` 附近；
   改训练预检要跨过引擎、流水线、数据集几千行。评审时 diff 上下文永远看不清。
2. **多人不能并行**。同一个文件同一段区域，两个改动必然冲突。
3. **定位靠行号记忆**。文档里已经出现「见 `app.py:1044`」这类引用（商业化分析文档 S3 项），
   行号一漂就失效。
4. **路由与实现混在一起**。`@app.post("/api/engine/build")` 与 `build_engine()` 的实现
   和 `probe_onnx()`、`_shape_side()` 等辅助函数交错，看不出模块边界。

好消息是：这些函数在文件里**已经按域连续排布**（train 约 417–1150、engine 1152–1510、
pipeline 1510–2340、dataset 2340–2433、runs 2433 起、infer 约 2900–3340、
annotation 约 3700–4400）。拆分可以按文件顺序机械搬运，不需要重新组织逻辑。

## What Changes

- 新增 `6-web/backend/web_state.py`：把目前散在 `app.py` 模块级的共享状态集中到一处
  （`manager`、`pipeline_manager`、`RUN_KIND_LABELS`、`_reject_if_busy`、
  `_START_LOCK`、`VIDEO_SUFFIXES` / `ENGINE_SUFFIXES` 等常量），供路由模块与
  `app.py` 共同 import，避免循环依赖。
- 新增 9 个路由模块，URL 与响应结构**逐字不变**：

| 模块 | 路径前缀 | 约含路由数 |
| --- | --- | --- |
| `routers/system.py` | `/`、`/api/health`、`/api/capabilities`、`/api/browse*`、`/api/warm/*` | 9 |
| `routers/train.py` | `/api/train/*`、`/api/export/yolo` | 8 |
| `routers/engine.py` | `/api/engine/*` | 4 |
| `routers/pipeline.py` | `/api/pipeline/*` | 6 |
| `routers/dataset.py` | `/api/dataset/*` | 4 |
| `routers/runs.py` | `/api/runs/*`、`/ws/runs/{run_id}` | 6 |
| `routers/infer.py` | `/api/infer/*`、`/api/ocr/*` | 15 |
| `routers/annotation.py` | `/api/annotation/*` | 13 |
| `routers/models.py` | `/api/models`、`/api/modelzoo/*` | 5 |

- `app.py` 只保留：常量与版本号、lifespan、异常处理器、静态文件挂载、
  `include_router` 调用。目标 ≤ 400 行。
- 新增 `6-web/backend/tools/route_table.json` 与 `tools/test_route_table.py`：
  把 `METHOD + PATH` 全量快照成 JSON，测试断言运行时路由表与快照逐条一致。
  **这是本次重构唯一的护栏**，任何 URL 写错、漏注册、重复注册都会立刻失败。
- 前端零改动：URL 不变，`js/api.js` 不用碰。

**BREAKING**：无。URL、请求体、响应体、WebSocket 地址全部不变；
`/openapi.json` 的路径顺序可能变化，但内容等价。

## Impact

- Affected specs: `web-studio`（新增一条「路由按域拆分且对外契约不变」要求）
- Affected code:
  - 新增 `6-web/backend/web_state.py`
  - 新增 `6-web/backend/routers/{__init__,system,train,engine,pipeline,dataset,runs,infer,annotation,models}.py`
  - `6-web/backend/app.py` 大幅缩减
  - 新增 `6-web/backend/tools/test_route_table.py`、`tools/route_table.json`
  - 打包脚本 `--include-module` 清单需增加 `routers.*`（否则安装包里导入失败）
- 不涉及：前端、C++、DLL、训练 launcher。
