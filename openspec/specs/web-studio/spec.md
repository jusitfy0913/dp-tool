# web-studio Specification

## Purpose
定义 Web 工作站的总体约束：训练/导出/编译/推理全部由本地 FastAPI 后端驱动并复用既有 C++ DLL，
浏览器只做展示（前端无构建步骤），事件通过 WebSocket 回传；路径选择走服务端原生对话框，
Qt 版程序归档保留作为参考实现。
## Requirements
### Requirement: Web Backend Exposes Training API
系统 SHALL 提供 HTTP 接口启动 YOLO 与 Anomalib 训练，且 SHALL 通过子进程调用现有 launcher 脚本，不得在服务进程内直接训练。

#### Scenario: 启动 YOLO 训练
- **WHEN** 客户端向训练接口提交 YOLO 训练参数
- **THEN** 后端 SHALL 以 `--mode train` 启动 `1-python-yolo/launcher/tf_yolo_launcher.py`，并返回可追踪的 `run_id`

#### Scenario: 启动 Anomalib 训练
- **WHEN** 客户端提交 Anomalib 训练参数且包含数据 root
- **THEN** 后端 SHALL 以 `--mode train-export` 启动 `2-python-anomaly/launcher/tf_anomalib_launcher.py`

#### Scenario: 缺少必填参数
- **WHEN** Anomalib 训练请求未提供数据 root
- **THEN** 后端 SHALL 返回参数错误，且不得启动子进程

#### Scenario: 复用已配置的解释器
- **WHEN** 后端需要 Python 解释器且服务配置中已指定 `python.exe` 路径
- **THEN** 后端 SHALL 使用该解释器启动 launcher，不得依赖当前服务自身的解释器

### Requirement: Training Events Stream to Browser
系统 SHALL 将 launcher 输出的 `TRTFORGE_JSON` 事件实时转发给浏览器，且 SHALL 保持原有事件类型语义。

#### Scenario: 逐行转发训练日志
- **WHEN** launcher 向 stdout 打印一行带 `TRTFORGE_JSON` 前缀的记录
- **THEN** 后端 SHALL 剥离前缀后把 JSON 原样推送到该 `run_id` 的 WebSocket 连接

#### Scenario: 训练结束通知
- **WHEN** launcher 子进程退出
- **THEN** 后端 SHALL 推送包含退出码的结束事件，并关闭该事件流

#### Scenario: 主动停止训练
- **WHEN** 客户端请求停止某个 `run_id`
- **THEN** 后端 SHALL 终止对应子进程并推送取消事件

#### Scenario: 训练失败可诊断
- **WHEN** launcher 输出 `type` 为 `error` 的事件
- **THEN** 浏览器 SHALL 能看到该事件的错误码与错误文案

### Requirement: Inference Reuses Existing C++ Libraries
系统 SHALL 通过 `ctypes` 调用 `yolo_infer.dll` 与 `anomalib_infer.dll` 完成推理；库不可用时 SHALL 返回可诊断的错误而不是静默降级。

#### Scenario: 加载推理库
- **WHEN** 后端启动推理适配器
- **THEN** 它 SHALL 先把 OpenCV 与 CUDA 的 bin 目录加入 DLL 搜索路径，再加载目标 DLL

#### Scenario: YOLO 推理
- **WHEN** 客户端提交图片与 YOLO 引擎路径
- **THEN** 后端 SHALL 通过 `yi_create` / `yi_infer_bgr` / `yi_destroy` 完成一次推理并返回检测框

#### Scenario: Anomalib 推理
- **WHEN** 客户端提交图片与 Anomalib 引擎路径
- **THEN** 后端 SHALL 通过 `ai_create` / `ai_infer_rgb` / `ai_destroy` 返回图像分数与异常图

#### Scenario: 推理库不可用时给出原因
- **WHEN** 目标 DLL 加载失败
- **THEN** 后端 SHALL 返回 503 并说明缺失的 DLL 或依赖目录，不得返回成功假象

#### Scenario: 结果内存立即深拷贝
- **WHEN** 一次推理返回指向 DLL 内部缓冲的指针
- **THEN** 后端 SHALL 在函数内立即拷贝成 Python 对象，不得把指针留到下次推理之后使用

#### Scenario: 资源释放
- **WHEN** 一次推理结束或抛出异常
- **THEN** 后端 SHALL 保证句柄在服务退出时通过 `*_destroy` 释放

### Requirement: Engine Build Uses trtexec
系统 SHALL 通过 `3-cpp-tensorrt/trtexec` 把 ONNX 转换为 TensorRT 引擎，且 SHALL 把构建日志回流到浏览器。

#### Scenario: 提交转换任务
- **WHEN** 客户端提交 ONNX 路径、输出路径、精度与动态 shape
- **THEN** 后端 SHALL 组装 `trtexec` 参数并启动子进程

#### Scenario: 转换进度可见
- **WHEN** `trtexec` 持续输出日志
- **THEN** 后端 SHALL 把日志推送到对应 `run_id` 的事件流

#### Scenario: 输出缺失时报错
- **WHEN** `trtexec` 退出码为 0 但输出引擎文件不存在
- **THEN** 后端 SHALL 返回失败并附上完整日志尾部

### Requirement: Annotation Workflow Preserved
系统 SHALL 提供数据集标注与导出能力，且 SHALL 保持 `train / val / test` 比例划分规则与 `dataset.yaml` 字段格式。

#### Scenario: 自动划分
- **WHEN** 用户设置合计为 100% 的划分比例并触发自动划分
- **THEN** 系统 SHALL 按该比例分配图片归属

#### Scenario: 比例非法
- **WHEN** 用户设置的划分比例合计不等于 100%
- **THEN** 系统 SHALL 拒绝执行并提示具体合计值

#### Scenario: 导出 YOLO 数据集
- **WHEN** 用户导出 YOLO 数据集
- **THEN** 系统 SHALL 生成图片目录、标签目录与 `dataset.yaml`

### Requirement: Frontend Requires No Build Step
前端 SHALL 为可直接由静态服务器提供的文件，SHALL NOT 依赖 npm、node 或任何打包步骤。

#### Scenario: 直接修改即生效
- **WHEN** 开发者修改前端某个 `.js` 或 `.html` 文件
- **THEN** 刷新浏览器即可看到变化，无需执行构建命令

#### Scenario: 无 node 环境可运行
- **WHEN** 机器上未安装 node
- **THEN** 服务启动后浏览器 SHALL 能正常加载全部页面

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

### Requirement: Toolchain Paths Resolve From Environment
C++ 子工程的第三方依赖路径 SHALL 从环境变量读取，SHALL NOT 写死某台机器的绝对路径。

#### Scenario: 用环境变量指定 OpenCV
- **WHEN** 设置了 `OPENCV_DIR` 后配置推理子工程
- **THEN** CMake SHALL 使用该目录，并在配置日志中回显找到的 OpenCV 版本

#### Scenario: 未设置时不猜路径
- **WHEN** `OPENCV_DIR` 未设置且缓存中也没有
- **THEN** CMake SHALL 报错并给出设置示例，不得回退到硬编码路径

#### Scenario: CUDA 与 MSVC 版本不匹配
- **WHEN** 使用 CUDA 11.8 与 MSVC 14.44 编译 `.cu` 源文件
- **THEN** 构建说明 SHALL 记录通过 `NVCC_PREPEND_FLAGS` 设置 `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` 的方式

### Requirement: Qt Program Archived
`4-qt-studio` SHALL 移入 `5-reference` 并停止参与构建，其源码 SHALL 保留供查阅。

#### Scenario: 目录归档
- **WHEN** 查看仓库目录
- **THEN** 仓库根 SHALL 不再有 `4-qt-studio`，对应源码 SHALL 位于 `5-reference/4-qt-studio`

#### Scenario: 不参与构建
- **WHEN** 执行任何构建脚本或 CMake 配置
- **THEN** 归档后的 Qt 工程 SHALL NOT 被引用

#### Scenario: 文档标注
- **WHEN** 开发者阅读 `readme.md`
- **THEN** 其中 SHALL 说明 Qt 版本仅供查阅、界面已由 Web 替代

