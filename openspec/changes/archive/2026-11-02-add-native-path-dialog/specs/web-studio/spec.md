# web-studio delta

## ADDED Requirements

### Requirement: Native Path Picker
系统 SHALL 允许用户在浏览器里点一个按钮就选中**服务端机器**上的文件或文件夹，
且 SHALL 在原生对话框不可用时自动回退到页面内目录浏览。

#### Scenario: 本机访问时弹原生框
- **WHEN** 客户端从 `127.0.0.1` / `::1` 发起 `POST /api/browse/dialog`，且环境支持原生对话框
- **THEN** 后端 SHALL 在服务端桌面弹出对应类型（`file` / `files` / `dir` / `save`）的系统对话框
- **AND** 用户确认后 SHALL 返回选中的绝对路径，路径分隔符统一为平台原生写法

#### Scenario: 扩展名过滤
- **WHEN** 请求携带 `patterns`（如 `[".pt", ".pth"]`）
- **THEN** 对话框过滤器 SHALL 由后端按该清单生成
- **AND** SHALL 始终附带一条「所有文件」，避免用户文件后缀不在清单里时看不到任何文件

#### Scenario: 用户取消
- **WHEN** 用户在对话框里点了取消或关闭
- **THEN** 响应 SHALL 为 `cancelled=true` 且 `paths` 为空
- **AND** 前端 SHALL 保持输入框原值，不得视为错误、不得自动打开页面内浏览

#### Scenario: 远程来源
- **WHEN** 请求来源不是本机（原生框会弹在服务端屏幕上，用户看不见）
- **THEN** 后端 SHALL 返回 **403**
- **AND** 前端 SHALL 提示原因并回退到页面内目录浏览

#### Scenario: 并发
- **WHEN** 已有一个对话框处于打开状态，又收到新的弹框请求
- **THEN** 后端 SHALL 返回 **409**，不得弹出第二个对话框

#### Scenario: 超时
- **WHEN** 对话框超过 `TRTFORGE_DIALOG_TIMEOUT`（默认 300 秒）仍未结束
- **THEN** 后端 SHALL 终止该子进程并按取消处理，且不得留下临时文件

#### Scenario: 仓库外路径
- **WHEN** 用户选中的路径不在现有白名单内
- **THEN** 该目录 SHALL 被加入**运行期**放行表，后续接口可正常访问
- **AND** 放行范围 SHALL 仅限该目录（不递归整盘）、SHALL NOT 落盘、服务重启后失效
- **AND** 只有在来源为本机时才能触发放行

#### Scenario: 能力探测
- **WHEN** 客户端请求 `GET /api/capabilities`
- **THEN** 响应 SHALL 包含 `native_dialog` 布尔字段
- **AND** 为 `false` 时前端 SHALL 直接使用页面内目录浏览，不显示点不动的按钮

#### Scenario: 记住上次目录
- **WHEN** 某个字段成功选中过一次路径
- **THEN** 下次打开该字段的对话框 SHALL 从该路径所在目录开始
- **AND** 用户取消时 SHALL NOT 覆盖已记住的目录

#### Scenario: 强制使用页面内浏览
- **WHEN** 用户按住 `Shift` 点击浏览按钮
- **THEN** 前端 SHALL 直接打开页面内目录浏览，不发起原生对话框请求
