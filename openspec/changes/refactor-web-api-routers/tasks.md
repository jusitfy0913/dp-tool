## 1. 护栏先行（不改任何路由）

- [x] 1.1 新增 `tools/route_table.py`：收集 `app.routes` 的 `METHOD /path`，
      排除 `/docs`、`/openapi.json`、`/redoc`、`/docs/oauth2-redirect` 与静态挂载
- [x] 1.2 生成 `tools/route_table.json` 初始快照（实测 **86 条**：77 条业务路由
      + `/`、`/api/health`、`/api/capabilities` 等系统路由 + 9 条 `autoexp`)
- [x] 1.3 新增 `tools/test_route_table.py`：逐条比对，报告缺失/新增，
      支持 `--update` 重建
- [x] 1.4 `run_all_tests.py` 收录该测试并确认全绿

## 2. 共享状态集中

- [x] 2.1 新增 `web_state.py`：`manager`、`pipeline_manager`、`RUN_KIND_LABELS`、
      `_START_LOCK`、`_reject_if_busy`、`_ensure_allowed_path`、`VIDEO_SUFFIXES`、`ENGINE_SUFFIXES`
- [x] 2.2 `app.py` 改为从 `web_state` 引用，删除本地定义
- [x] 2.3 用 `rg` 列出所有 `app\.(manager|pipeline_manager|_reject_if_busy|_START_LOCK)`
      的测试引用点，改成 `web_state` 引用或加同名转发
- [x] 2.4 全量测试 + 路由快照通过后单独提交（本批与 3~5 合并为一次提交，见「实现记录」）

## 3. 搬小域（system / engine / dataset / models）

- [x] 3.1 `routers/__init__.py`
- [x] 3.2 `routers/system.py`：`/`、`/api/health`、`/api/capabilities`、
      `/api/browse`、`/api/browse/dialog`、`/api/browse/cancel`、`/api/warm/status`、`/api/warm/prewarm`（8 条）
- [x] 3.3 `routers/engine.py`：`/api/engine/probe`、`/api/engine/build`、
      `/api/engine/operating-point`（GET/POST）（4 条）
- [x] 3.4 `routers/dataset.py`：`/api/dataset/scan`、`/qc`、`/export`、`/fingerprint`（4 条）
- [x] 3.5 `routers/models.py`：`/api/models`、`/api/bundles`、`/api/bundles/build`、
      `/api/bundles/verify`、`/api/bundles/{id}/manifest`、`/api/modelzoo` 与下载相关 3 条（9 条）
- [x] 3.6 路由快照 + 全量测试通过后提交（同上）

## 4. 搬中域（train / runs / pipeline）

- [x] 4.1 `routers/train.py`：`/api/train/*` 8 条 + `/api/export/yolo` + `/api/runs/checkpoints`（9 条）
- [x] 4.2 `routers/runs.py`：`/api/runs*` 5 条 + `WS /ws/runs/{run_id}`（独立无前缀 router，6 条）
- [x] 4.3 `routers/pipeline.py`：`/api/pipeline/*` 6 条
- [x] 4.4 路由快照 + 全量测试通过后提交（同上）

## 5. 搬大域（infer / annotation）与收尾

- [x] 5.1 `routers/infer.py`：`/api/infer/*` 12 条 + `/api/ocr/models`、`/api/ocr/warm`（14 条）
- [x] 5.2 `routers/annotation.py`：`/api/annotation/*` 17 条
- [x] 5.3 `app.py` 收敛为装配层，清理无用 import，目标 ≤ 400 行（**实测 195 行**）
- [x] 5.4 路由快照 + 全量测试通过后提交（同上）

## 6. 打包与验收

- [x] 6.1 `package_web_user.ps1` 增加 `--include-package=routers`，
      并在编译后检查 `frozen_entry.dist\routers` 存在，缺了中止打包
- [ ] 6.2 出一次内部测试包，双击启动，逐个页面点开验证接口可用
- [ ] 6.3 真机确认 `/api/health`、`/api/capabilities`、`/api/models`、
      `/api/infer/yolo` 四个接口可用

## 7. 文档

- [x] 7.1 `docs/WebStudio架构与功能介绍.md` 第 8.2 节模块表补 `routers/` 与 `web_state.py`
- [x] 7.2 `6-web/README.md` 增加本次重构的变更点与「路由快照」使用说明
- [x] 7.3 `docs/WORKFLOW_GUIDE.md` 补 §7.22「后端代码怎么组织、改完怎么自检」
- [x] 7.4 `readme.md` 目录结构同步（`6-web/README.md` 代码结构树）

## 实现记录（与原设计的偏差，评审看这里）

| 项 | 原设计 | 实际 | 原因 |
| --- | --- | --- | --- |
| 快照条数 | 77 条业务路由 | 86 条（排除文档与静态后的全部） | 把 `/`、`/api/health`、`/api/capabilities` 与 `autoexp` 一并纳入，才会在「漏 include 一整域」时报警 |
| 提交粒度 | 2/3/4/5 四步各一次提交 | 合并为一次提交 | 用户要求本次按顺序一次做完并推送；每步快照都已单独跑过 |
| `/api/bundles*` 落点 | `routers/models.py` 只写 `/api/models` 与 modelzoo | 5 条 bundle 路由也放 `models.py` | 机械拆分时先落到了 `infer.py`，复查后按域归位（模型域=模型资产：散装产物 + 交付包 + 模型仓库） |
| 共享管理器引用方式 | 路由 `from web_state import manager` | 路由统一写 `web_state.manager.xxx` | 静态导入会把单例复制进各模块命名空间，测试替换 `web_state.manager` 时不生效，出现「两个管理器」 |
| 兼容转发 | 同名转发（赋值或 import） | 模块级 `__getattr__`（只读） | 40+ 脚本按 `app.<名字>` 取值；转发只保证「读得到」，`monkeypatch` 需改到 `routers.<域>` |
| 新增测试 | 仅 `test_route_table.py` | 追加 `test_app_assembly.py` | 快照只保证接口集合，管不住「业务重新长回 app.py」，用行数上限、禁止路由装饰器、禁止反向 import 三条断言补上 |
| 测试脚本改动 | 未列入 | 4 个脚本的 patch 目标改到 `routers.*` / `web_state` | 拆分后 `app.<名字>` 只是转发，赋值不影响路由（`test_native_dialog` / `test_infer_batch` / `test_pipeline_resolve` / `test_augment_params`） |
