## 1. 访问令牌与来源校验

- [x] 1.1 新增 `6-web/backend/auth.py`：令牌加载/生成、`hmac.compare_digest` 比较、内存会话表、登录失败限流
- [x] 1.2 `app.py` 注册中间件：Host 校验 → Origin 校验 → 令牌/会话校验，放行登录页、登录接口、`/api/health`
- [x] 1.3 新增 `POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/auth/status`
- [x] 1.4 WebSocket 握手校验会话，不通过关闭码 4401
- [x] 1.5 `frozen_entry.py` / `config.py`：解析 `TRTFORGE_AUTH`、`TRTFORGE_ALLOWED_HOSTS`；非回环且 `off` 时启动告警；启动日志打印令牌文件位置
- [x] 1.6 新增 `frontend/login.html`（沿用现有主题样式）；`api.js` 401 跳转；`ws.js` 4401 跳转且停止重连

> 实现备注（v1.42.0）：认证中间件落在 `auth.install(app)` 里，由 `app.py` 装配；
> 路由按域拆进 `routers/` 之后，认证仍在最外层 HTTP 中间件，不依赖各路由自查。
> WebSocket 例外：`/api/runs/{id}/events` 等端点在握手时调 `auth.ws_authorized()`。

## 2. 入口收口

- [x] 2.1 `_resolve_python`：白名单 + 文件名校验；配置来源的解释器自动加入白名单
- [x] 2.2 trtexec 额外参数白名单（单一常量），核对 TensorRT 8.5 `trtexec --help` 与文档推荐参数后定稿
- [x] 2.3 列出全部路径字段清单，逐个接入 `_ensure_allowed_path`（读）/父目录校验（写）
- [x] 2.4 `_save_upload` 与 `/api/infer/frame` 增加文件头校验

> 实现备注（v1.42.0）：路径白名单全覆盖 —— 训练的 `model`/`data`/`project`/`name`、
> 引擎的五个入口、推理的 `engine`/`labels`/`bundle`、全流程的 `name`（另过 `_sanitize_part`）。
> 额外参数白名单定义在 `routers/engine.py` 的单一常量里。

## 3. 测试

- [x] 3.1 `test_auth.py`：auto/on/off 三种模式；Cookie 与 Bearer；错误令牌；限流；WebSocket 4401；Host/Origin 拒绝
- [x] 3.2 `test_path_guard.py`：遍历全部路由的路径字段，白名单外路径必须 400/403
- [x] 3.3 `test_trtexec_args.py`：放行项、拒绝项、带路径值、错误信息含允许清单
- [x] 3.4 `test_upload_magic.py`：五种格式通过；改扩展名的 exe / txt 被拒
- [x] 3.5 `run_all_tests.py` 全部通过；前端 `node --check` 通过
- [x] 3.6 浏览器实测：本机无令牌直接用；`0.0.0.0` 时从另一地址访问先进登录页；登录后图片、日志下载、批量结果下载、训练实时日志都正常

> 3.5 备注：v1.42.0 起 `run_all_tests.py` 同时跑后端 `test_*.py` 与前端 `test_*.mjs`，
> 当前 84 个脚本全绿（前端单测从「手工 node」改为由总入口执行）。

## 4. 文档与打包

- [x] 4.1 `APP_VERSION` → 1.42.0（提案写的是 1.29.0，实际落地版本已是 v1.41，按现实版本号走）
- [x] 4.2 `6-web/README.md`：变更日志、环境变量、接口、YAML `version_diff`
- [x] 4.3 `docs/WORKFLOW_GUIDE.md`：多机访问与令牌步骤、排障表
- [x] 4.4 `docs/TRTForge_Web操作手册.md`：登录页说明、忘记令牌怎么办
- [x] 4.5 `packaging/使用说明.md` 与启动脚本模板：说明令牌文件位置
- [x] 4.6 根 `readme.md` 版本说明；`docs/商业化差距分析与优化建议.md` 标记 S1–S5 状态
