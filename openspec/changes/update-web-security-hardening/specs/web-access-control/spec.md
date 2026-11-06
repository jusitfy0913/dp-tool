## ADDED Requirements

### Requirement: 访问令牌按监听地址自动启用

Web 后端 SHALL 支持 `TRTFORGE_AUTH=auto|on|off`，默认 `auto`。`auto` 模式下监听回环地址时不要求令牌，监听其他地址时所有 HTTP 与 WebSocket 请求（登录页、登录接口与 `/api/health` 除外）必须携带有效会话 Cookie 或 `Authorization: Bearer` 令牌。

#### Scenario: 本机默认无需令牌

- **WHEN** 服务以默认配置监听 `127.0.0.1`，浏览器访问任意页面或接口
- **THEN** 请求必须正常处理，不出现登录页

#### Scenario: 局域网访问必须先登录

- **WHEN** 服务监听 `0.0.0.0` 且 `TRTFORGE_AUTH=auto`，未登录的浏览器访问首页
- **THEN** 后端必须跳转到 `/login.html`
- **AND** 未登录的接口请求必须返回 401

#### Scenario: 登录后图片与下载可用

- **WHEN** 用户输入正确令牌登录
- **THEN** 后端必须下发 `HttpOnly`、`SameSite=Strict` 的会话 Cookie
- **AND** 标注图片、运行日志下载、批量结果下载与 WebSocket 必须无需修改 URL 即可访问

#### Scenario: WebSocket 未授权

- **WHEN** 需要令牌时未登录的客户端连接 `/ws/runs/{run_id}`
- **THEN** 后端必须以关闭码 4401 关闭连接

#### Scenario: 令牌自动生成

- **WHEN** 未设置 `TRTFORGE_WEB_TOKEN` 且需要令牌
- **THEN** 后端必须生成随机令牌写入 `runtime/web_token.txt`，并在启动日志中提示该文件位置

#### Scenario: 连续猜错被限流

- **WHEN** 同一 IP 连续 10 次登录失败
- **THEN** 该 IP 的登录请求必须在 5 分钟内被拒绝

### Requirement: 请求来源校验

Web 后端 SHALL 拒绝 `Host` 头不属于本机地址或 `TRTFORGE_ALLOWED_HOSTS` 的请求，并拒绝 `Origin` 与 `Host` 不一致的非 GET 请求。

#### Scenario: DNS 重绑定被拒绝

- **WHEN** 请求的 `Host` 头为未登记的域名
- **THEN** 后端必须返回 421，不执行任何业务逻辑

#### Scenario: 跨站提交被拒绝

- **WHEN** 来自其他站点的页面向本服务发送 POST 请求，`Origin` 与 `Host` 不一致
- **THEN** 后端必须返回 403

### Requirement: 训练解释器只接受白名单内的 Python

训练、导出、环境检查等接口的 `python` 字段 SHALL 只接受位于路径白名单内、文件名为 Python 解释器的文件；自动发现、`TRTFORGE_PYTHON` 与配置文件指定的解释器视为可信。

#### Scenario: 填写任意程序被拒绝

- **WHEN** 请求的 `python` 字段指向 `C:\Windows\System32\cmd.exe`
- **THEN** 后端必须返回 400，且不得启动任何子进程

#### Scenario: 训练环境解释器可用

- **WHEN** 请求的 `python` 字段指向白名单内的 `train-env\python.exe`
- **THEN** 后端必须正常启动训练

### Requirement: trtexec 额外参数白名单

引擎构建接口的 `extra_args` SHALL 只放行与构建调优相关、不读写文件且不加载外部代码的 trtexec 参数；其他参数及值中含路径分隔符的参数必须被拒绝。

#### Scenario: 加载外部插件被拒绝

- **WHEN** `extra_args` 包含 `--plugins=evil.dll`
- **THEN** 后端必须返回 400，错误信息列出允许的参数名，且不得启动 trtexec

#### Scenario: 写任意文件被拒绝

- **WHEN** `extra_args` 包含 `--saveEngine=C:\x.engine` 或 `--exportTimes=...`
- **THEN** 后端必须返回 400

#### Scenario: 调优参数可用

- **WHEN** `extra_args` 为 `--builderOptimizationLevel=5 --memPoolSize=workspace:2048`
- **THEN** 后端必须把这些参数追加到 trtexec 命令行并正常构建

### Requirement: 所有路径字段经过白名单

所有接收文件或目录路径的接口字段 SHALL 经过路径白名单校验；读取的路径必须在白名单内，写入的新文件其父目录必须在白名单内。仓库 SHALL 包含遍历全部路由的测试，防止新增接口遗漏校验。

#### Scenario: 引擎输出到白名单外被拒绝

- **WHEN** 引擎构建请求的 `engine` 字段为 `C:\Windows\x.engine`
- **THEN** 后端必须返回 403，且不得启动 trtexec

#### Scenario: 新增接口漏校验时测试失败

- **WHEN** 开发者新增一个带 `path` 字段的接口但未接入白名单
- **THEN** `test_path_guard.py` 必须失败并指出该路由与字段

### Requirement: 上传图片校验文件头

图片上传与摄像头单帧接口 SHALL 校验文件头属于 JPEG、PNG、BMP、TIFF 或 WebP，且与扩展名一致。

#### Scenario: 改扩展名的文件被拒绝

- **WHEN** 上传一个扩展名为 `.jpg`、内容为可执行文件的文件
- **THEN** 后端必须返回 400，不得把数据交给图像解码或推理动态库
