# web-studio Specification

## Purpose
TBD - created by archiving change add-web-studio. Update Purpose after archive.
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

