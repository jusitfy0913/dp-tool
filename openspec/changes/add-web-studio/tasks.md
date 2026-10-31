# 任务清单：Web 界面替代 Qt

> 状态说明：已确认方案。训练页、推理页、引擎构建页均已完成；标注页（划分/导出工作台）仍延后。

## 1. 前置环境（已完成，留作记录）

- [x] 1.1 确认 OpenCV 位于 `E:\thirdparty\opencv`（4.12.0，vc17，x64）
- [x] 1.2 确认 CMake `E:\3rdlib\CMake\bin\cmake.exe`（4.2.1）与 MSVC 2022 Community 可用
- [x] 1.3 无 Qt 环境下编译 `anomalib_infer.dll` 成功
- [x] 1.4 用 `NVCC_PREPEND_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler` 编译 `yolo_infer.dll` 成功
- [x] 1.5 用 `ctypes.WinDLL` + `os.add_dll_directory` 加载两个 DLL 成功
- [x] 1.6 两个 launcher `--mode check-env` 均输出 `missing:[]`

## 2. 后端骨架

- [x] 2.1 建 `6-web/backend/`，引入 FastAPI + uvicorn（实测 fastapi 0.115.8 / uvicorn 0.54.0）
- [x] 2.2 实现任务注册表（`run_id` ↔ 子进程 / 状态 / 开始时间）
- [x] 2.3 实现 `/api/health` 与 `/api/capabilities`（探测 DLL、trtexec、解释器是否就绪）
- [x] 2.4 实现 WebSocket `/ws/runs/{run_id}`，支持断线重连补发最近 2000 条
- [x] 2.5 服务退出时统一 terminate 所有子进程

## 3. 训练页

- [x] 3.1 实现 `POST /api/train/yolo` 与 `/api/train/anomalib`，参数校验对齐 Qt 版
- [x] 3.2 实现训练适配器：拼参数 → `Popen` → 逐行解析 `TRTFORGE_JSON`
- [x] 3.3 实现 `POST /api/runs/{run_id}/stop`
- [x] 3.4 前端训练表单 + 实时日志面板 + 进度条 + 指标表
- [x] 3.5 训练事件覆盖 env / model / progress / metric / epoch_summary / gpu_status / artifact / error
- [x] 3.6 额外：`POST /api/train/{kind}/check-env` 环境自检（开训前查缺依赖）

## 4. 推理页

- [x] 4.1 实现 DLL 搜索路径注入（OpenCV bin、CUDA bin，均可环境变量覆盖）
- [x] 4.2 实现 YOLO 适配器（`yi_create` / `yi_infer_bgr` / `yi_destroy`）
- [x] 4.3 实现 Anomalib 适配器（`ai_create` / `ai_infer_rgb` / `ai_destroy`）
- [x] 4.4 前端上传图片 → 展示框选结果与异常热力图/轮廓图
- [x] 4.5 验证 ctypes 结构体 ABI 与 C 侧逐字节一致（`tools/abi_probe.cpp`）
- [x] 4.6 句柄按参数缓存复用，服务退出时统一 `*_destroy`
- [ ] 4.7 `onnxruntime` 降级回退（当前 DLL 不可用时返回 503 并说明原因，未实现自动降级）

## 5. 引擎构建页（已完成）

> 注意迁移陷阱 T2：Qt 版的 `runtime`（full/lean/dispatch）与「最大 Batch」
> 在 `BuilderWorker.cpp` 中**无任何引用**，属未生效的占位 UI。
> Web 版暂不提供这两项，避免做出假开关。

- [x] 5.1 实现 `POST /api/engine/build`，组装 trtexec 参数
    - 参数：`--onnx` / `--saveEngine` / `--fp16|--fp32` / `--workspace` / 动态 shape 三组 / 透传额外参数
    - `config.resolve_trtexec()` 依次探测 `TRTFORGE_TRTEXEC` → `CUDA_PATH\bin` → CUDA 根 `v*/bin` → PATH
- [x] 5.2 日志流复用训练事件通道（`/ws/runs/{run_id}`，未新增通道）
- [x] 5.3 前端 ONNX 选择、精度、动态 shape 三组输入
    - 另加 `GET /api/engine/probe` 预检：选完 ONNX 立即显示输入/输出形状与文件大小
    - 「未转置」YOLO 输出（`[1, 4+C, N]`）在前端标红、后端直接 400 拒绝，避免编译出静默出错的引擎
- [x] 5.4 构建结果校验（退出码为 0 但无输出文件要报错）
    - `runners.py` 新增 `expect_file`，任务正常退出后校验产物是否落盘，缺失则判为 failed 并发 `ARTIFACT_MISSING` 事件
- [x] 5.5 实测：PCB 模型 960×960 FP16 构建耗时 474.1 s，产物 10.14 MB，吞吐 135.4 qps，退出码 0

## 6. 标注页（延后）

> 注意迁移陷阱 T1：Qt 版用 `qHash(i + "_2026")` 排序，受哈希种子随机化影响，
> 跨进程结果可能不同。Web 版必须改成固定哈希（FNV-1a）并保留 `_2026` 盐值，
> 否则与历史实验无法对齐。

- [ ] 6.1 用固定哈希复刻划分算法（含除不尽、单类、已标注图片的边界处理）
- [ ] 6.2 实现 `dataset.yaml` 生成
- [ ] 6.3 实现导出 YOLO / COCO / VOC
- [ ] 6.4 前端图片列表 + 划分比例 + 导出按钮

## 7. 构建脚本与环境

- [x] 7.1 新增 `scripts/build_infer_dlls.ps1`：按 `OPENCV_DIR` + `NVCC_PREPEND_FLAGS` 一键编两个 DLL
- [x] 7.2 脚本自动探测 CMake 与 VS，OpenCV 路径不写死（环境变量 > 常见位置 > 报错）
- [x] 7.3 脚本预检 DLL 占用，给出「Web 服务还开着」的可读提示（替代 LNK1104）
- [x] 7.4 `.gitignore` 增加 `6-web/runtime/`、`__pycache__`、`build/infer_dlls/`
- [ ] 7.5 两个推理子工程的 `CMakeLists.txt` 仍默认 `D:/thirdparty/opencv`，待改为纯环境变量驱动

## 8. 归档与文档

- [ ] 8.1 `git mv 4-qt-studio 5-reference/4-qt-studio`
- [ ] 8.2 全仓确认无脚本引用旧 `4-qt-studio` 路径
- [x] 8.3 更新 `readme.md`：Web 启动方式、四页功能、环境变量、Qt 改为仅供查阅
- [ ] 8.4 更新 `docs/` 中涉及界面位置的说明（`WORKFLOW_GUIDE.md` 与模型对比文档已按 Web 链路重写，历史文档未逐份校对）
- [x] 8.5 写 `6-web/README.md`：安装、启动、端口、常见问题

## 9. 验收

- [x] 9.1 浏览器打开页面，状态灯显示「环境就绪」（实测通过）
- [x] 9.2 训练接口参数校验返回 400 而非 500（实测通过）
- [x] 9.3 真实 launcher 自检子进程 + WebSocket 事件流端到端跑通（实测通过）
- [x] 9.4 目录越权访问被拦截（实测通过）
- [x] 9.5 未装 node 的机器上能正常打开全部页面（前端零构建，已确认）
- [x] 9.6 完整训练一轮 YOLO 并导出 ONNX（已用 `PCB data/PCB缺陷检测` 真实数据集完成：30 epoch，mAP50=0.945，ONNX 复测精度一致，并编译出可用引擎）
- [x] 9.7 用真实 `.trt` 引擎跑一次推理（PatchCore / PaDiM / EfficientAd 三模型均已跑通，准确率 1.0 / 0.9844 / 1.0）
- [x] 9.8 关闭浏览器后训练不被杀掉；服务退出后无残留进程（设计如此并已验证子进程隔离）
- [x] 9.9 用真实 launcher 验证后端拼的参数能被 argparse 接受（YOLO / Anomalib 均产出规范 `DATA_NOT_FOUND`，非 exit 2）
- [x] 9.10 退出码语义化：exit 2 → `ARGUMENT_ERROR`，exit 130 → `CANCELLED`（`tools/test_exit_codes.py` 4 场景通过）

## 10. 验证状态

| 项 | 状态 | 说明 |
| --- | --- | --- |
| OpenSpec 校验 | 通过 | `openspec validate --all --strict` → 8 passed, 0 failed |
| DLL 编译（无 Qt） | 通过 | 一键脚本实测：`anomalib_infer.dll` 148.5 KB、`yolo_infer.dll` 148 KB |
| DLL ctypes 加载 | 通过 | 两个均返回有效 handle，`infer_yolo/infer_anomalib` 均为 true |
| ctypes 结构体 ABI | 通过 | MSVC 实测 sizeof/offsetof 与 ctypes 逐项一致（5 个结构体） |
| 训练环境自检 | 通过 | 通过 Web 接口启动自检，事件流返回 `missing=[]` |
| 前端页面 | 通过 | 四个页签渲染正常，目录浏览弹窗可用 |
| 端到端冒烟测试 | 通过 | `tools/smoke_test.py` → 13 项全通过 |
| 退出码语义化 | 通过 | `tools/test_exit_codes.py` → 6 项全通过（exit 2/0/7 + 事件解析） |
| launcher 参数真实校验 | 通过 | 用后端拼的命令调真实 launcher，两个都进入业务逻辑而非 exit 2 |
| 事件字段对齐 | 通过 | 按 Qt 源码核对，修正 `epoch_summary`/`gpu_status`/`artifact` 三处字段名 |
| 真实训练 / 真实推理 | 通过 | PCB 数据集 30 epoch（mAP50=0.945）+ ctypes 推理验证；Anomalib 三模型均跑通 |
| 引擎构建（Web UI） | 通过 | 浏览器实测：探测 → 构建 → 日志流 → 完成徽标；960×960 FP16 用时 474.1 s |
| 引擎构建防护 | 通过 | 未转置 ONNX 前端标红 + 后端 400；退出码 0 但产物缺失判为失败 |
