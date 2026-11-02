# tasks

## 1. 后端弹框能力

- [x] 1.1 新增 `nativedialog.py`：`available()` / `interpreter()` / `build_filetypes()` / `select()`
- [x] 1.2 子进程脚本支持 `file` / `files` / `dir` / `save` 四种 `filedialog` 调用，窗口置顶
- [x] 1.3 传参与结果走临时 JSON，跑完清理，避免中文/引号/编码问题
- [x] 1.4 全局单例锁（`DialogBusy`）与超时 kill（`TRTFORGE_DIALOG_TIMEOUT`，默认 300s）
- [x] 1.5 按字段记住上次目录（`runtime/dialog_state.json`），传文件时取父目录
- [x] 1.6 回传路径规范成 Windows 分隔符，与 Qt 写法一致

## 2. 白名单与接口

- [x] 2.1 `config.allow_path()` / `clear_runtime_allowed()` / `ALLOWED_RUNTIME`，`is_path_allowed` 一并考虑
- [x] 2.2 新增 `POST /api/browse/dialog`，校验 `kind`、本机来源、单例锁
- [x] 2.3 选中路径自动放行，响应带 `allowed`
- [x] 2.4 `/api/capabilities` 增加 `native_dialog`；`APP_VERSION` → 1.20.0

## 3. 前端接入

- [x] 3.1 `api.js` 增加 `nativeDialog()`；`main.js` 统一 `pickNative()` + `writeValue()` + 提示条
- [x] 3.2 22 个按钮补 `data-dialog` / `data-patterns` / `data-title`；`Shift` 强制页面内浏览
- [x] 3.3 失败/不可用自动回退页面内浏览并说明原因；409 只提示不叠框
- [x] 3.4 补齐 `yoloModel`、`pythonPath` 入口；三个「文件或目录」字段拆成双按钮
- [x] 3.5 标注页「选图片…」多选加图（`insertImages` 抽出复用）；「另存为」走原生保存框
- [x] 3.6 页面内浏览的扩展名过滤改读 `data-patterns`，无配置时退回旧正则

## 4. 测试

- [x] 4.1 新增 `test_native_dialog.py`（假子进程，不用真弹窗）：40 项
- [x] 4.2 覆盖扩展名归一化 / 起始目录 / 四种类型 / 单例锁 / 超时 / 异常 / 无结果 / 记忆 / 清理 / 白名单 / 接口来源
- [x] 4.3 `run_all_tests.py` 全量回归：22 个脚本全绿
- [x] 4.4 浏览器实测：真实弹框选中写回、Esc 取消、`Shift` 页面内浏览、保存框

## 5. 文档

- [x] 5.1 `6-web/README.md`：v1.20 变更点、能力说明、接口字段、环境变量、代码结构、验证记录、FAQ
- [x] 5.2 `docs/WORKFLOW_GUIDE.md`：§7.1.1 路径怎么选
- [x] 5.3 根 `readme.md`：v1.20 版本记录 + 能力表 + 环境变量
