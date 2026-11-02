# Change: 路径选择改成服务端原生对话框

## Why

Web Studio 里所有需要填路径的地方（22 个按钮）都只有两条路：手打绝对路径，或用页面内那个
「列目录」弹窗一层层钻。文件类型还是前端写死的一串正则，`.pth` 之类干脆看不见。

根因不是没做，是**浏览器安全沙箱不允许网页直接调系统文件框**。所以选路径这件事必须由后端
代劳：后端在它自己那台机器的桌面上弹框，把选中的绝对路径回给页面。

不做这件事的代价很直接：用户为了选 `D:\数据集` 得手抄完整路径；选仓库外目录还会 403，
因为白名单只放行仓库根。

## What Changes

- 新增 `6-web/backend/nativedialog.py`：用短命子进程跑 `tkinter` 弹框
  （file / files / dir / save 四种），传参与结果都走临时 JSON 文件，避开中文与引号问题。
  - 全局单例锁：已有框时抛 `DialogBusy`，接口转 409；
  - 超时（默认 300 秒）kill 子进程并按取消处理；
  - 按字段记住上次目录，存 `runtime/dialog_state.json`；
  - 回传路径统一成 Windows 分隔符，与 Qt 写法一致。
- `6-web/backend/app.py` 新增 `POST /api/browse/dialog`：
  - 只接受本机来源（127.0.0.1 / ::1 / localhost），远程一律 403；
  - 选中的路径加入运行期放行表（`config.allow_path`）；
  - `/api/capabilities` 增加 `native_dialog` 字段供前端判断。
- `6-web/backend/config.py` 增加运行期放行表：只放行选中那一级目录、不递归、不落盘、
  重启失效；`is_path_allowed` 一并考虑它。
- 前端（`js/main.js`、`js/api.js`、`index.html`、`css/style.css`）：
  - 单击按钮走原生框，`Shift` + 单击走页面内浏览；
  - 原生框不可用 / 远程 / 出错时自动回退页面内浏览，并用顶部提示条说明原因；
  - 22 个按钮补 `data-dialog` / `data-patterns` / `data-title`；
  - 补齐 `yoloModel`、`pythonPath` 两个漏掉的入口；
  - `dsSource` / `batchSource` / `annoImportSource` 拆成「文件」「目录」两个按钮。
- 标注页（`js/annotate.js`）：新增「选图片…」多选文件直接加入工程；「另存为」改走原生保存框。

## Impact

- Affected specs: `web-studio`（新增 Requirement: Native Path Picker）
- Affected code:
  - 新增 `6-web/backend/nativedialog.py`、`6-web/backend/tools/test_native_dialog.py`
  - 修改 `6-web/backend/app.py`、`6-web/backend/config.py`
  - 修改 `6-web/frontend/js/main.js`、`api.js`、`annotate.js`、`index.html`、`css/style.css`
  - 文档：`6-web/README.md`、`docs/WORKFLOW_GUIDE.md`、根 `readme.md`
- 版本：`APP_VERSION` 1.19.0 → 1.20.0

## Non-Goals

- 不用 `<input type="file">` 传文件内容：它拿不到真实绝对路径（只有 `fakepath`），
  而后端需要的是服务端路径。
- 不做跨机器远程选路径：原生框弹在服务端屏幕，远程用户看不见。远程保持页面内浏览。
- 不替代白名单机制：放行表只管本次运行选中的那一级目录，不会变成「读任意磁盘」。
- 不在 CI 里真弹窗：模态框无法自动点击，测试用假子进程验证自己的逻辑。

## 验收

- `D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\test_native_dialog.py` → **40 / 40 通过**
- `run_all_tests.py` → **22 个脚本全绿**
- 浏览器实测：点「dataset.yaml 浏览」→ 桌面弹框 → 输入路径回车 → 字段写入；
  Esc 取消后字段不变且不误开页面内浏览；`Shift` + 点击打开页面内浏览；
  标注页「新建…」弹保存框。
