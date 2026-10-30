# 6-web —— TRTForge Web Studio

浏览器版工作站，用来替代 `4-qt-studio` 的 Qt 桌面程序。

## 为什么有这个目录

Qt 版本的问题是：**每台机器都要先装 Qt 6.8.3 + MSVC 才能编出界面**。本机长期没有 Qt，
`TRTForge.exe` 一直编不出来，界面等于不可用。

Web 版把界面换成浏览器，换来三件事：

| 事项 | Qt 版 | Web 版 |
| --- | --- | --- |
| 装环境 | Qt 6.8.3 + MSVC，编译几十分钟 | 装 Python 包，秒级启动 |
| 远程使用 | 只能坐那台机器前 | 局域网内任何浏览器 |
| 改界面 | 重新编译整个程序 | 改完刷新 |

**底层能力一行没重写**：训练仍走原来的 launcher，推理仍走原来编好的 DLL。

## 快速开始

### 1. 装依赖

```powershell
# 用装了 torch 的那个环境（训练需要）
D:\Anaconda3\envs\dpenv\python.exe -m pip install -r 6-web\backend\requirements.txt
```

注意不要顺手升 `numpy`：`pip install -r` 里已锁 `numpy<2`，因为 torch 2.3.0 是拿 numpy 1.x 编的。

### 2. 编推理 DLL（只需一次）

Web 版的推理依赖 `yolo_infer.dll` 与 `anomalib_infer.dll`。它们**不依赖 Qt**，但要 OpenCV。

```powershell
$env:OPENCV_DIR = "E:\thirdparty\opencv"
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"

powershell -File scripts\build_infer_dlls.ps1
```

第二个环境变量是给 **CUDA 11.8 + MSVC 14.44** 这组搭配用的，原因见「常见问题」。
产物落在 `4-qt-studio/bin/`。

### 3. 启动

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
```

打开 <http://127.0.0.1:8060>。顶部状态灯变绿写「环境就绪」就说明一切正常。

常用参数：

```powershell
# 换端口
powershell -File scripts\run_web_studio.ps1 -Port 8070

# 指定训练解释器
powershell -File scripts\run_web_studio.ps1 -PythonExe D:\Anaconda3\envs\dpenv\python.exe

# 改前端代码时热重载
powershell -File scripts\run_web_studio.ps1 -Reload

# 让局域网别的机器也能访问
powershell -File scripts\run_web_studio.ps1 -HostName 0.0.0.0
```

## 页面说明

### 训练导出

1. 选「训练类型」：YOLO 或 Anomalib，表单会自动切字段。
2. 填 `dataset.yaml`（YOLO）或数据集 root（Anomalib），可用「浏览」按钮选。
3. 点「检测环境」先确认 Python / torch / CUDA 是否就绪（会列出缺哪个包和修复命令）。
4. 点「开始训练」，右侧实时出日志、进度条、指标。

日志是流式的，不是训练完才一次性吐出来。中间关掉浏览器训练**不会停**，重新进来点任务还能继续看。

### 推理演示

1. 选 YOLO 或 Anomalib。
2. 填 `.trt` 引擎路径（Anomalib 还要填 `metadata.json`）。
3. 选图片，点「开始推理」。

YOLO 返回画好框的图与每个目标的类别/置信度；Anomalib 返回异常分数、OK/NG 判定、
热力图叠加图和红色异常轮廓图。

> 推理需要引擎文件。仓库里没有现成的 `.trt`，得先用训练页导出 ONNX，再转 TensorRT。

### 环境说明

显示后端探测到的全部路径、Python 解释器、DLL 加载状态，以及可用的环境变量清单。
出问题时先看这一页。

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `TRTFORGE_PYTHON` | 训练用解释器 | 自动探测（当前解释器 / dpenv / pytorch） |
| `OPENCV_DIR` | OpenCV 根目录 | 未设则推理 DLL 加载失败 |
| `CUDA_PATH` | CUDA 根目录 | 未设则推理 DLL 加载失败 |
| `TRTFORGE_REPO_ROOT` | 仓库根 | 从 `config.py` 位置向上推导 |
| `TRTFORGE_BIN_DIR` | DLL 所在目录 | `<仓库>/4-qt-studio/bin` |
| `TRTFORGE_YOLO_LAUNCHER_DIR` | YOLO launcher 目录 | `<仓库>/1-python-yolo/launcher` |
| `TRTFORGE_ANOMALIB_LAUNCHER_DIR` | Anomalib launcher 目录 | `<仓库>/2-python-anomaly/launcher` |
| `TRTFORGE_ANOMALIB_DATASET_ROOT` | Anomalib 数据集根 | `<仓库>/2-python-anomaly/anomalib-main/datasets/bottle` |
| `TRTFORGE_BROWSE_ROOTS` | 额外允许浏览的目录，分号分隔 | 仅仓库根 |
| `TRTFORGE_MAX_UPLOAD_MB` | 上传图片上限 | 32 |
| `TRTFORGE_WEB_PORT` / `TRTFORGE_WEB_HOST` | 监听端口 / 地址 | 8060 / 127.0.0.1 |

## 后端接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 存活探针 |
| GET | `/api/capabilities` | 环境探测（前端状态灯用） |
| POST | `/api/train/yolo` | 启动 YOLO 训练 |
| POST | `/api/train/anomalib` | 启动 Anomalib 训练 |
| POST | `/api/train/{yolo\|anomalib}/check-env` | 环境自检（开训前查缺依赖） |
| GET | `/api/runs` | 任务列表 |
| GET | `/api/runs/{run_id}` | 单任务状态 |
| POST | `/api/runs/{run_id}/stop` | 停止任务 |
| WS | `/ws/runs/{run_id}` | 事件流（训练日志 / 进度） |
| GET | `/api/browse?path=` | 目录浏览（Web 无法弹系统选择框） |
| POST | `/api/infer/yolo` | YOLO 推理 |
| POST | `/api/infer/anomalib` | Anomalib 推理 |

## 代码结构

```
6-web/
├── backend/
│   ├── app.py            FastAPI 路由
│   ├── config.py         路径推导与环境变量
│   ├── runners.py        子进程管理 + TRTFORGE_JSON 事件解析 + 广播
│   ├── infer.py          ctypes 调 DLL（结构体定义与深拷贝）
│   └── tools/
│       ├── abi_probe.cpp       打印 C 侧结构体布局，用于和 Python 对齐比对
│       ├── smoke_test.py       端到端冒烟测试（HTTP + WebSocket + 真实子进程）
│       └── test_exit_codes.py  退出码语义化测试（exit 2/130 的补事件逻辑）
├── frontend/
│   ├── index.html        单页，三个页签
│   ├── css/style.css
│   └── js/
│       ├── api.js        HTTP 封装
│       ├── ws.js         WebSocket（含断线重连）
│       ├── train.js      训练页
│       ├── infer.js      推理页
│       └── main.js       页签 / 环境状态 / 目录浏览
└── runtime/              上传临时文件与任务日志（已 gitignore）
```

## 验证

```powershell
# 服务已启动后
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\smoke_test.py

# 退出码处理（不需要服务，独立运行）
D:\Anaconda3\envs\dpenv\python.exe 6-web\backend\tools\test_exit_codes.py
```

2026-10-29 实测结果：`smoke_test` **13 项全通过**（能力探测、参数校验、越权拦截、
真实子进程自检、WebSocket 事件流）；`test_exit_codes` **6 项全通过**。

### ctypes 结构体对齐

跨语言传结构体如果对齐不一致会静默读错内存。`tools/abi_probe.cpp` 用 MSVC 打印
C 侧 `sizeof` 与 `offsetof`，与 Python `ctypes.sizeof` / `.offset` 比对，当前**逐项完全一致**：

| 结构体 | 大小 | 关键偏移 |
| --- | --- | --- |
| `yi_box` | 96 | `mask_data=48`、`kpts=64`、`is_obb=72`、`angle=92` |
| `yi_frame_stat` | 48 | — |
| `ai_metadata` | 192 | `image_threshold=160`、`input_h=184` |
| `ai_result` | 104 | `overlay_rgb=24`、`mask_rgb=48`、`raw_score=88` |
| `ai_frame_stat` | 40 | — |

## 常见问题

**顶部写「N 项待处理」**

看「环境说明」页的 JSON。常见原因：没装 `python-multipart`、launcher 路径不对、
DLL 没编、`OPENCV_DIR` 没设。

**推理接口返回 503**

DLL 没加载成功。两个必查点：

1. `OPENCV_DIR` 与 `CUDA_PATH` 是否指向真实目录；
2. Python 3.8+ **不用 PATH 找依赖 DLL**。本项目在 `infer.py` 里用
   `os.add_dll_directory` 显式加目录，如果你绕过后端直接写脚本，也要自己加。

**编译 yolo_infer 报 `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`**

MSVC 14.44 的 STL 头文件硬性拒绝 CUDA < 12.4。仓库里的 CMakeLists 虽然写了绕过宏，
但 CMake 在**探测 CUDA 编译器**那一步不会传 `CMAKE_CUDA_FLAGS`，所以宏没生效。

有效解法是环境变量（能覆盖所有 nvcc 调用，包括探测阶段）：

```powershell
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"
```

根治方案是升级到 CUDA 12.4+ 或换 VS2019 工具集。

**训练日志不出来**

事件只在 launcher 打印 `TRTFORGE_JSON {...}` 前缀的行时被识别，其他行走普通日志。
若完全没输出，检查 launcher 是否以无缓冲方式运行（后端已设 `PYTHONUNBUFFERED=1`）。

**只看到「退出码 2」，没有具体原因**

这是 argparse 的行为：参数不合法时直接打 usage 并 `exit(2)`，**不会**发 `error` 事件。
后端已补一层转换（`runners.py` 的 `_synthetic_error`），会额外推送一条
`ARGUMENT_ERROR` 事件并从日志尾部附上 usage 片段。若仍看不出，去
`6-web/runtime/logs/<run_id>.log` 看完整输出。
退出码 130 同理会被翻译成 `CANCELLED`。

**指标区不显示损失 / GPU 利用率**

已修正。原因是字段名与 launcher 实际结构不一致：`epoch_summary` 的
`losses`/`metrics`/`lr` 都是**字典**而非标量，GPU 利用率字段是
`utilization_gpu_pct` 而非 `utilization`。若你自己加事件渲染，请以
`1-python-yolo/launcher/tf_yolo_launcher.py` 里的 `emit` 调用为准。

**关闭浏览器后训练还在跑？**

这是设计行为。任务是后端子进程，与浏览器无关。要停就点页面「停止」，或调
`POST /api/runs/{run_id}/stop`。

## 与 Qt 版的关系

`4-qt-studio` 目前仍保留，作为**迁移参考实现**（参数拼装规则、数据集划分算法、
事件解析逻辑都在里面）。确认 Web 版功能覆盖后再归档到 `5-reference`。

详见提案：`openspec/changes/add-web-studio/`。
