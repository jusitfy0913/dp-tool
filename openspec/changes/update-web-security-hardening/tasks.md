## 1. 访问令牌与来源校验

- [ ] 1.1 新增 `6-web/backend/auth.py`：令牌加载/生成、`hmac.compare_digest` 比较、内存会话表、登录失败限流
- [ ] 1.2 `app.py` 注册中间件：Host 校验 → Origin 校验 → 令牌/会话校验，放行登录页、登录接口、`/api/health`
- [ ] 1.3 新增 `POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/auth/status`
- [ ] 1.4 WebSocket 握手校验会话，不通过关闭码 4401
- [ ] 1.5 `frozen_entry.py` / `config.py`：解析 `TRTFORGE_AUTH`、`TRTFORGE_ALLOWED_HOSTS`；非回环且 `off` 时启动告警；启动日志打印令牌文件位置
- [ ] 1.6 新增 `frontend/login.html`（沿用现有主题样式）；`api.js` 401 跳转；`ws.js` 4401 跳转且停止重连

## 2. 入口收口

- [ ] 2.1 `_resolve_python`：白名单 + 文件名校验；配置来源的解释器自动加入白名单
- [ ] 2.2 trtexec 额外参数白名单（单一常量），核对 TensorRT 8.5 `trtexec --help` 与文档推荐参数后定稿
- [ ] 2.3 列出全部路径字段清单，逐个接入 `_ensure_allowed_path`（读）/父目录校验（写）
- [ ] 2.4 `_save_upload` 与 `/api/infer/frame` 增加文件头校验

## 3. 测试

- [ ] 3.1 `test_auth.py`：auto/on/off 三种模式；Cookie 与 Bearer；错误令牌；限流；WebSocket 4401；Host/Origin 拒绝
- [ ] 3.2 `test_path_guard.py`：遍历全部路由的路径字段，白名单外路径必须 400/403
- [ ] 3.3 `test_trtexec_args.py`：放行项、拒绝项、带路径值、错误信息含允许清单
- [ ] 3.4 `test_upload_magic.py`：五种格式通过；改扩展名的 exe / txt 被拒
- [ ] 3.5 `run_all_tests.py` 全部通过；前端 `node --check` 通过
- [ ] 3.6 浏览器实测：本机无令牌直接用；`0.0.0.0` 时从另一地址访问先进登录页；登录后图片、日志下载、批量结果下载、训练实时日志都正常

## 4. 文档与打包

- [ ] 4.1 `APP_VERSION` → 1.29.0
- [ ] 4.2 `6-web/README.md`：变更日志、环境变量、接口、YAML `version_diff`
- [ ] 4.3 `docs/WORKFLOW_GUIDE.md`：多机访问与令牌步骤、排障表
- [ ] 4.4 `docs/TRTForge_Web操作手册.md`：登录页截图、忘记令牌怎么办
- [ ] 4.5 `packaging/使用说明.md` 与启动脚本模板：说明令牌文件位置
- [ ] 4.6 根 `readme.md` 版本说明；`docs/商业化差距分析与优化建议.md` 标记 S1–S5 状态
