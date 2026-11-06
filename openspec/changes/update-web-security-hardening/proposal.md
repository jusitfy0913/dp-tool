# Change: Web Studio 安全加固（访问令牌 + 入口收口）

## Why

Web Studio 默认只监听 `127.0.0.1`，但 `packaging/使用说明.md` 建议车间多机访问时把 `host` 改成 `0.0.0.0`。
一旦这样做，现有代码有以下可被直接利用的问题（均已对照代码确认）：

1. **没有任何身份校验**：HTTP 与 WebSocket 路由都不检查请求来自谁，局域网任何人都能训练、建引擎、读日志、改标注。
2. **任意程序执行**：训练页「Python 解释器」字段只检查文件存在（`app.py` `_resolve_python`），填任意 exe 都会被当成解释器启动。
3. **trtexec 参数注入**：引擎页「额外参数」经 `shlex.split` 原样追加（`app.py` 约 1044 行），可用 `--plugins=` 加载任意 DLL、`--saveEngine=` 写任意位置。
4. **白名单不完整**：引擎构建的 ONNX/输出路径、YOLO 训练 `data`、Anomalib 训练 `root`、单图推理的 `engine`/`labels` 等字段没走 `_ensure_allowed_path`。
5. **上传只看扩展名**：不校验文件头；摄像头单帧接口不经过统一上传校验。
6. **本机也有隐患**：恶意网页可以对 `http://127.0.0.1:8060` 发请求（CSRF / DNS 重绑定），当前没有 `Host`/`Origin` 校验。

这些是商业交付的硬门槛，列为 P0（见 `docs/商业化差距分析与优化建议.md` 第 2 节）。

## What Changes

- **访问令牌**（新增）：
  - 模式由 `TRTFORGE_AUTH=auto|on|off` 控制，默认 `auto`：监听回环地址时不要求令牌（保持本机零配置），监听其他地址时强制要求。
  - 令牌来自 `TRTFORGE_WEB_TOKEN`；未设置时首次启动随机生成并写入 `runtime/web_token.txt`，启动日志打印一次。
  - 浏览器首次访问显示「输入访问令牌」页，校验通过后由后端下发 `HttpOnly + SameSite=Strict` 会话 Cookie。图片、下载链接、WebSocket 自动携带 Cookie，**前端现有 URL 拼接不用改**。
  - 脚本/第三方调用可用 `Authorization: Bearer <令牌>`。
  - `host` 非回环且 `TRTFORGE_AUTH=off` 时，启动日志醒目警告。
- **请求来源校验**（新增）：所有请求校验 `Host` 头属于本机地址或配置的主机名（防 DNS 重绑定）；非 GET 请求若带 `Origin`，必须与 `Host` 一致（防跨站伪造）。
- **解释器收口**：网页传入的 Python 路径必须在路径白名单内，且文件名为 `python.exe` / `pythonw.exe`（Windows）或 `python`、`python3*`（其他平台）；否则 400。自动发现与 `TRTFORGE_PYTHON` 配置的路径不受影响。
- **trtexec 额外参数白名单**：只放行与构建调优有关、不涉及文件读写和外部代码加载的参数；其余（含 `--plugins`、`--saveEngine`、`--loadEngine`、`--onnx`、`--export*`、`--timingCacheFile`、`--int8`、`--calib`、`--best`）一律 400 并说明原因。
- **路径白名单全覆盖**：所有接收文件/目录路径的接口字段统一经过 `_ensure_allowed_path`；新增测试遍历全部路由，确保今后新增接口不会漏。
- **上传文件头校验**：`_save_upload` 与 `/api/infer/frame` 都校验文件头（JPEG/PNG/BMP/TIFF/WebP），扩展名与内容不符时 400。
- **版本**：`APP_VERSION` 1.28.0 → 1.29.0。

## 不做的事

- 不做多账号、角色、审计日志（P1-6，单独提案）。
- 不做 HTTPS（车间内网为主；需要时用反向代理）。
- 不改桌面版 Qt 程序、不改任何 DLL、不动根目录 `bin` / `include` / `lib`。

## Impact

- 新增规范：`web-access-control`
- 涉及代码：
  - `6-web/backend/app.py`
  - `6-web/backend/config.py`
  - `6-web/backend/frozen_entry.py`
  - 新增 `6-web/backend/auth.py`
  - `6-web/frontend/js/api.js`（401 跳转）
  - `6-web/frontend/js/ws.js`（被拒时提示）
  - 新增 `6-web/frontend/login.html`
- 涉及测试：新增 `test_auth.py`、`test_path_guard.py`、`test_trtexec_args.py`、`test_upload_magic.py`；`run_all_tests.py` 自动纳入。
- 涉及打包：`packaging/templates` 启动脚本说明令牌位置；`packaging/使用说明.md` 写明多机访问步骤。
- 兼容性：
  - 本机访问（默认配置）行为不变。
  - 以前填写过非白名单 Python 路径或使用过被禁 trtexec 参数的用户会收到 400 提示，需按提示调整。
