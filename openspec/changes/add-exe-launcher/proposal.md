# Change: 安装包双击 exe 启动，不再依赖 bat 与 PowerShell

## Why

拿到安装包的现场人员，第一步是双击 `start_web.bat`。这个 bat 只有 4 行，
真正干活的是 `start_user.ps1`：读 `config.json`、设置约 20 个环境变量、
起一个后台 `Start-Job` 轮询健康检查、再启动 `app\TRTForgeWeb.exe`。

这条链路有三个现场问题：

1. **看起来不像一个软件**。用户要记住双击哪个 bat，杀毒软件会把
   `powershell -ExecutionPolicy Bypass -File` 当成可疑行为拦下来，
   而提示信息是英文的 PowerShell 报错；
2. **一台机器上已经有一个同类脚本 `start_web_cmd.bat`**，用于没有 PowerShell 的机器，
   但那份脚本不读 `config.json`、端口写在文件顶部，两条路行为不一致；
3. **出错时用户拿不到可读信息**。PowerShell 报错滚一下就没了，
   环境变量拼错、`config.json` 格式坏、端口被占用，最后都表现为「窗口一闪就没了」。

另外 `TRTForgeWeb.exe` 本身已经是编好的程序（`--windows-console-mode=force`），
只是没人直接双击它——因为环境变量没设，它找不到前端目录也找不到 `bin`。

## What Changes

- 新增 `6-web/backend/launcher.py`：把 `start_user.ps1` 里的环境变量逻辑
  原样搬到 Python，成为唯一实现。
- `frozen_entry.py` 在无参数启动时先走 launcher：
  定位安装根 → 读 `config.json` → 设环境变量 → 单实例互斥 → 端口处理 →
  启动 uvicorn → 轮询健康检查 → 打开浏览器。
- 失败时用 Windows 原生消息框给出**中文**原因与下一步动作，
  同时写 `runtime/logs/launcher.log`（UTF-8）。
- Nuitka 控制台模式从 `force` 改为 `attach`：双击时完全不出黑框；
  从 cmd 或 `start_web.bat` 启动时仍能在原窗口看到日志。
- 无控制台时 `sys.stdout` / `sys.stderr` 可能为 `None`，launcher 负责兜底重定向，
  避免任何 `print` 抛 `AttributeError`。
- `start_web.bat` 保留为**排障入口**，内容简化为直接调用 exe；
  `start_web_cmd.bat` 合并进 `start_web.bat`，不再维护第二份端口配置。
- 新增 `GET /api/launcher/status` 不需要——健康检查已足够，不新增接口。

**BREAKING**：无。`config.json` 字段与含义不变，环境变量名不变，
用 `start_web.bat` 启动的行为不变。

## Impact

- Affected specs: `web-studio`（新增一条「安装包单文件启动器」要求）
- Affected code:
  - 新增 `6-web/backend/launcher.py`
  - `6-web/backend/frozen_entry.py`（无参数分支改走 launcher）
  - `packaging/package_web_user.ps1`（`--windows-console-mode=attach`）
  - `packaging/templates/start_web.bat`、`start_web_cmd.bat`、`start_user.ps1`（收敛）
  - `packaging/使用说明.md`、`docs/无PowerShell启动说明.md`（改写启动章节）
  - 新增 `6-web/backend/tools/test_launcher.py`
- 不涉及：uvicorn 本身、任何推理与训练路径、C++ 与 DLL。
