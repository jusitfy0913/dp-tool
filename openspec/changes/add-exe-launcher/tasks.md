## 1. 启动器核心

- [ ] 1.1 新增 `6-web/backend/launcher.py`：
      `install_root()`、`load_config()`、`apply_environment()`、`ensure_stdio()`
- [ ] 1.2 `apply_environment()` 输出的环境变量清单必须与
      `packaging/templates/start_user.ps1` 逐项一致（含 `TRTFORGE_HOME`、`OPENCV_BIN_DIR`、
      `CUDA_BIN_DIR`、`TRTFORGE_TOOL_EXE`、`TCL_LIBRARY`、`TK_LIBRARY`、`PATH` 前置）
- [ ] 1.3 单实例：`CreateMutexW` + `Local\TRTForgeWebStudio`，
      区分新建与已存在（`ERROR_ALREADY_EXISTS`）
- [ ] 1.4 端口判定：`/api/health` 通且版本匹配视为已有实例；否则视为被占用
- [ ] 1.5 后台线程轮询健康检查后 `webbrowser.open`
- [ ] 1.6 `MessageBoxW` 中文错误框，文案三段式
- [ ] 1.7 日志写 `runtime/logs/launcher.log`（UTF-8，含时间/安装根/端口/结果）

## 2. 入口接入

- [ ] 2.1 `frozen_entry.main()` 无参数分支改为 `launcher.run()`
- [ ] 2.2 `--tool` 分支保持原样，加一条测试断言其行为未变
- [ ] 2.3 保留 `TRTFORGE_LEGACY_START=1` 回退到直接 `uvicorn.run`

## 3. 打包

- [ ] 3.1 `package_web_user.ps1` 控制台模式 `force` → `attach`
- [ ] 3.2 确认 `--include-module` 清单含 `launcher`
- [ ] 3.3 `start_web.bat` 改为只设 `TRTFORGE_HOME` 后调用 exe，去掉 PowerShell
- [ ] 3.4 `start_web_cmd.bat` 与 `start_web.bat` 合并为同一行为，避免两份端口配置
- [ ] 3.5 `start_user.ps1` 保留但标注为「排障用，正常启动不再调用」

## 4. 测试

- [ ] 4.1 新增 `tools/test_launcher.py`：
      配置解析与默认值、环境变量清单与 ps1 对齐、互斥量语义（用不同名字模拟）、
      端口判定三态、`ensure_stdio()` 在 `sys.stdout is None` 时不抛异常、
      `--tool` 分派不受影响
- [ ] 4.2 现有 57 个脚本保持全绿
- [ ] 4.3 手工验收：双击 exe 无黑框并自动开浏览器；重复双击只开一个实例；
      占用 8060 时弹中文框；cmd 启动能看到日志

## 5. 文档

- [ ] 5.1 `packaging/使用说明.md` 启动章节改为「双击 `app\TRTForgeWeb.exe`」
- [ ] 5.2 `docs/无PowerShell启动说明.md` 重写为 `docs/启动与排障说明.md` 或改写内容
- [ ] 5.3 `6-web/README.md` 增加 v1.37.0 变更点
- [ ] 5.4 `docs/WORKFLOW_GUIDE.md` 打包章节同步控制台模式与启动方式
- [ ] 5.5 `docs/TRTForge_Web操作手册.md` 首次启动步骤同步
- [ ] 5.6 `readme.md` 启动说明同步
