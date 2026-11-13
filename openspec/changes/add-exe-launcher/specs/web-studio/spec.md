## ADDED Requirements

### Requirement: Single-File Packaged Launcher

安装包中的主程序 SHALL 在无参数启动时自行完成「定位安装目录、读取配置、设置运行环境、启动网页服务、打开浏览器」全过程，MUST NOT 要求使用者先执行批处理或 PowerShell 脚本。

#### Scenario: 双击即用

- **WHEN** 使用者在安装目录内双击 `app\TRTForgeWeb.exe`
- **THEN** 程序 SHALL 按 `config.json` 中的地址与端口启动网页服务，并在健康检查通过后打开浏览器

#### Scenario: 不出现控制台窗口

- **WHEN** 使用者双击主程序
- **THEN** 系统 SHALL NOT 出现常驻的命令行窗口

#### Scenario: 从命令行启动仍可见日志

- **WHEN** 使用者在已有控制台的命令行里启动主程序
- **THEN** 服务日志 SHALL 输出到该控制台

#### Scenario: 无控制台时不因标准输出为空而崩溃

- **WHEN** 程序在没有控制台的条件下启动且标准输出不可用
- **THEN** 程序 SHALL 把标准输出与错误重定向到 `runtime/logs/launcher.log` 后继续启动，MUST NOT 抛出异常

#### Scenario: 工具参数行为不变

- **WHEN** 程序以 `--tool <名字>` 启动
- **THEN** 它 SHALL 按原有分派执行对应工具，启动流程的改造 MUST NOT 影响该分支

#### Scenario: 安装目录定位

- **WHEN** 程序启动时环境变量 `TRTFORGE_HOME` 未设置
- **THEN** 它 SHALL 以可执行文件所在目录的上一级作为安装根，并能找到 `config.json` 与 `frontend` 目录

#### Scenario: 配置文件缺失

- **WHEN** 安装根下不存在 `config.json`
- **THEN** 程序 SHALL 使用 `127.0.0.1:8060` 的默认值继续启动，并把该情况写入启动日志，MUST NOT 阻止启动

### Requirement: Launcher Single Instance And Port Handling

启动器 SHALL 保证同一 Windows 会话内只有一个服务实例，并在端口被占用时区分「已有实例」与「被其它程序占用」两种情形。

#### Scenario: 已有实例时不再启动第二个

- **WHEN** 使用者重复双击主程序且前一个实例仍在运行
- **THEN** 程序 SHALL NOT 启动第二个服务进程，且 SHALL 打开浏览器指向已在运行的地址

#### Scenario: 端口被其它程序占用

- **WHEN** 目标端口被不属于本软件的程序占用
- **THEN** 程序 SHALL 以消息框给出该端口号、`config.json` 中需要修改的字段名与可选处理方式，然后退出

#### Scenario: 上一个实例未退干净

- **WHEN** 互斥量显示已有实例但健康检查不通过
- **THEN** 程序 SHALL 提示使用者稍等或结束残留进程，MUST NOT 强行抢占端口

#### Scenario: 不同 Windows 会话互不影响

- **WHEN** 两个 Windows 用户会话各自启动一次
- **THEN** 二者 SHALL 各自运行一个实例，不得互相阻塞

### Requirement: Launcher Failure Reporting

启动失败时 SHALL 以中文给出「发生了什么、为什么、该怎么办」三段信息，并 SHALL 把同样的内容写入启动日志。

#### Scenario: 失败信息可读

- **WHEN** 启动过程中出现端口占用、目录缺失或服务启动异常
- **THEN** 界面 SHALL 弹出中文消息框，包含原因与至少一条可执行的下一步动作

#### Scenario: 日志留痕

- **WHEN** 任意一次启动结束（成功或失败）
- **THEN** `runtime/logs/launcher.log` SHALL 含本次启动的时间、安装根、端口、结果

#### Scenario: 编码一致

- **WHEN** 安装路径或日志内容含中文
- **THEN** 日志 SHALL 以 UTF-8 写入，MUST NOT 出现乱码

#### Scenario: 批处理入口仍可用

- **WHEN** 使用者双击 `start_web.bat`
- **THEN** 它 SHALL 设置 `TRTFORGE_HOME` 后直接调用主程序，MUST NOT 依赖 PowerShell
