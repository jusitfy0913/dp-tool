## 1. 启动器核心

- [x] 1.1 新增 `6-web/backend/launcher.py`：
      `install_root()`、`load_config()`、`apply_environment()`、`ensure_stdio()`
- [x] 1.2 `apply_environment()` 输出的环境变量清单必须与
      `packaging/templates/start_user.ps1` 逐项一致（含 `TRTFORGE_HOME`、`OPENCV_BIN_DIR`、
      `CUDA_BIN_DIR`、`TRTFORGE_TOOL_EXE`、`TCL_LIBRARY`、`TK_LIBRARY`、`PATH` 前置）
      —— 实际做法改成「只有这一份实现」：`start_user.ps1` 降级为排障脚本，
      由 `test_launcher.py` 断言它不再设置业务环境变量
- [x] 1.3 单实例：`CreateMutexW` + `Local\TRTForgeWebStudio`，
      区分新建与已存在（`ERROR_ALREADY_EXISTS`）
- [x] 1.4 端口判定：`/api/health` 通且版本匹配视为已有实例；否则视为被占用
- [x] 1.5 后台线程轮询健康检查后 `webbrowser.open`
- [x] 1.6 `MessageBoxW` 中文错误框，文案三段式
- [x] 1.7 日志写 `runtime/logs/launcher.log`（UTF-8，含时间/安装根/端口/结果）

## 2. 入口接入

- [x] 2.1 `frozen_entry.main()` 无参数分支改为 `launcher.run()`
- [x] 2.2 `--tool` 分支保持原样，加一条测试断言其行为未变
- [x] 2.3 保留 `TRTFORGE_LEGACY_START=1` 回退到直接 `uvicorn.run`

## 3. 打包

- [x] 3.1 `package_web_user.ps1` 控制台模式 `force` → `attach`
- [x] 3.2 确认 `--include-module` 清单含 `launcher`
- [x] 3.3 `start_web.bat` 改为只设 `TRTFORGE_HOME` 后调用 exe，去掉 PowerShell
- [x] 3.4 `start_web_cmd.bat` 与 `start_web.bat` 合并为同一行为，避免两份端口配置
- [x] 3.5 `start_user.ps1` 保留但标注为「排障用，正常启动不再调用」

## 4. 测试

- [x] 4.1 新增 `tools/test_launcher.py`：
      配置解析与默认值、环境变量清单与 ps1 对齐、互斥量语义（用不同名字模拟）、
      端口判定三态、`ensure_stdio()` 在 `sys.stdout is None` 时不抛异常、
      `--tool` 分派不受影响
- [x] 4.2 现有 57 个脚本保持全绿（实跑 59 个，含新增的 `test_launcher`）
- [ ] 4.3 手工验收：双击 exe 无黑框并自动开浏览器；重复双击只开一个实例；
      占用 8060 时弹中文框；cmd 启动能看到日志
      —— 已用源码进程验证后三项（首次启动开浏览器、二次启动只开浏览器、日志内容）；
      「双击无黑框」依赖 `--windows-console-mode=attach`，需在 Nuitka 出包后确认

## 5. 文档

- [x] 5.1 `packaging/使用说明.md` 启动章节改为「双击 `app\TRTForgeWeb.exe`」
- [x] 5.2 `docs/无PowerShell启动说明.md` 重写为 `docs/启动与排障说明.md`（旧文件删除，引用全部改指新文件）
- [x] 5.3 `6-web/README.md` 增加 v1.37.0 变更点
- [x] 5.4 `docs/WORKFLOW_GUIDE.md` 打包章节同步控制台模式与启动方式
- [x] 5.5 `docs/TRTForge_Web操作手册.md` 首次启动步骤同步
- [x] 5.6 `readme.md` 启动说明同步
