## 1. 护栏先行（不改任何路由）

- [ ] 1.1 新增 `tools/route_table.py`：收集 `app.routes` 的 `METHOD /path`，
      排除 `/docs`、`/openapi.json`、`/redoc`、`/docs/oauth2-redirect` 与静态挂载
- [ ] 1.2 生成 `tools/route_table.json` 初始快照（当前 77 条业务路由 + `/`）
- [ ] 1.3 新增 `tools/test_route_table.py`：逐条比对，报告缺失/新增，
      支持 `--update` 重建
- [ ] 1.4 `run_all_tests.py` 收录该测试并确认全绿

## 2. 共享状态集中

- [ ] 2.1 新增 `web_state.py`：`manager`、`pipeline_manager`、`RUN_KIND_LABELS`、
      `_START_LOCK`、`reject_if_busy`、`VIDEO_SUFFIXES`、`ENGINE_SUFFIXES`
- [ ] 2.2 `app.py` 改为从 `web_state` 引用，删除本地定义
- [ ] 2.3 用 `rg` 列出所有 `app\.(manager|pipeline_manager|_reject_if_busy|_START_LOCK)`
      的测试引用点，改成 `web_state` 引用或加同名转发
- [ ] 2.4 全量测试 + 路由快照通过后单独提交

## 3. 搬小域（system / engine / dataset / models）

- [ ] 3.1 `routers/__init__.py`
- [ ] 3.2 `routers/system.py`：`/`、`/api/health`、`/api/capabilities`、
      `/api/browse`、`/api/browse/dialog`、`/api/browse/cancel`、`/api/warm/status`、`/api/warm/prewarm`
- [ ] 3.3 `routers/engine.py`：`/api/engine/probe`、`/api/engine/build`、
      `/api/engine/operating-point`（GET/POST）
- [ ] 3.4 `routers/dataset.py`：`/api/dataset/scan`、`/qc`、`/export`、`/fingerprint`
- [ ] 3.5 `routers/models.py`：`/api/models`、`/api/modelzoo` 与下载相关 3 条
- [ ] 3.6 路由快照 + 全量测试通过后提交

## 4. 搬中域（train / runs / pipeline）

- [ ] 4.1 `routers/train.py`：`/api/train/*` 7 条 + `/api/export/yolo`
- [ ] 4.2 `routers/runs.py`：`/api/runs*` 5 条 + `WS /ws/runs/{run_id}`（独立无前缀 router）
- [ ] 4.3 `routers/pipeline.py`：`/api/pipeline/*` 6 条
- [ ] 4.4 路由快照 + 全量测试通过后提交

## 5. 搬大域（infer / annotation）与收尾

- [ ] 5.1 `routers/infer.py`：`/api/infer/*` 13 条 + `/api/ocr/models`、`/api/ocr/warm`
- [ ] 5.2 `routers/annotation.py`：`/api/annotation/*` 13 条
- [ ] 5.3 `app.py` 收敛为装配层，清理无用 import，目标 ≤ 400 行
- [ ] 5.4 路由快照 + 全量测试通过后提交

## 6. 打包与验收

- [ ] 6.1 `package_web_user.ps1` 增加 `--include-package=routers`
- [ ] 6.2 出一次内部测试包，双击启动，逐个页面点开验证接口可用
- [ ] 6.3 真机确认 `/api/health`、`/api/capabilities`、`/api/models`、
      `/api/infer/yolo` 四个接口可用

## 7. 文档

- [ ] 7.1 `docs/WebStudio架构与功能介绍.md` 第 8.2 节模块表补 `routers/` 与 `web_state.py`
- [ ] 7.2 `6-web/README.md` 增加本次重构的变更点与「路由快照」使用说明
- [ ] 7.3 `docs/WORKFLOW_GUIDE.md` 补一条「新增接口要同步 route_table.json」
- [ ] 7.4 `readme.md` 如涉及目录说明则同步
