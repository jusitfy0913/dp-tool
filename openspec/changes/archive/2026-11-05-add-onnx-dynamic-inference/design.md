# Design: ONNX 动态库推理

## Context

推理页的两条既有链路已稳定运行：

- `yolo_infer.dll`：TensorRT 引擎 + CUDA 预处理/解码/NMS，C API 为 `yi_*`。
- `anomalib_infer.dll`：TensorRT 引擎 + OpenCV 后处理渲染，C API 为 `ai_*`。

两者都由 `6-web/backend/infer.py` 用 ctypes 调用，句柄按参数缓存并串行化，
Python 只负责搬数据与转结构。`.onnx` 目前落在 `onnx_fallback.py`，能力与表现都不完整。

约束：

- 工具链固定为 CUDA 11.8 + TensorRT 8.5 + OpenCV 4.12 + MSVC 2022 x64，
  ONNX Runtime 版本必须与 CUDA 11.8 / cuDNN 8 匹配。
- 后端进程里已经可能加载 Python 的 `onnxruntime`（`_probe_onnx` 与降级路径都在用），
  DLL 不能假定自己独占 ORT 运行库。
- 用户分发形态要求「不给源码」，所以 ONNX 推理不能依赖用户自装 Python 包。
- 阈值语义、判定口径、渲染配色必须与引擎路径完全一致，否则同一模型换路径会换结论。

## Goals / Non-Goals

- Goals:
  1. `.onnx` 与 `.engine` 一样走动态库，四条推理链路（单图/帧/批量/视频）行为一致。
  2. 覆盖 YOLO 四项任务与 Anomalib 三模型，输出结构与既有 DLL 同构。
  3. 无 NVIDIA GPU 或 CUDA 不匹配时仍能跑（CPU），且实际后端可查询。
  4. ONNX 与 TRT 的等价性可被一条命令反复验证，并作为回归门禁。
- Non-Goals:
  1. 不在 C++ 里做训练或 ONNX 导出。
  2. 不改写既有两个 DLL 的 C API 与结构体布局。
  3. 不把 ONNX 转 TensorRT 的编译流程并入本 DLL（继续用 `trtexec`）。
  4. 不承诺 fp16 引擎与 fp32 ONNX 数值完全一致。

## Decisions

### D1：一个 DLL 而不是两个

`onnx_infer.dll` 内部提供 `oi_create_yolo` 与 `oi_create_anomalib`。

- 理由：ORT 运行库、OpenCV 依赖、错误处理与句柄生命周期是共用的；拆成两个 DLL 会让
  `onnxruntime.dll` 的打包与版本校验出现两份口径。
- 代价：C 头文件略大。用「一类对象一套结构体 + 两个 create」控制复杂度。

### D2：结构体与既有 DLL 二进制对齐，Python 侧复用

`oi_box` 与 `yi_box`、`oi_anomaly_result` 与 `ai_result`、`oi_frame_stat` 与两者逐字段同序同类型。

- 理由：`infer.py` 已有 `YiBox`、`AiResult`、`AiFrameStat` 三个 `ctypes.Structure`，
  复用可避免「两份结构体定义迟早漂移」这个典型故障点。
- 约束：字段顺序与类型一旦发布即视为 ABI，只允许在末尾追加且需同步 Python。

### D3：不链接 `onnxruntime.lib`，改为显式加载

初始化时执行：

1. 以 `LoadLibraryExW(<bin>/onnxruntime.dll, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)` 加载；
2. `GetProcAddress(..., "OrtGetApiBase")` 取 `OrtApiBase`，再 `GetApi(ORT_API_VERSION)`；
3. 之后所有调用都走这张函数表。

- 理由：Windows 的 DLL 加载按模块名去重。若本 DLL 的导入表里写死 `onnxruntime.dll`，
  进程里已存在的 Python 版 ORT 会被直接复用，版本不同时可能缺符号或行为不同。
- 代价：所有 ORT 调用都要经函数表指针，写法比直接链接啰嗦；用 `src/ort_session` 封装掉。
- 头文件 `onnxruntime_c_api.h` 只用于类型与枚举声明，不产生链接依赖。

### D4：EP 选择「探测后才用」，失败即降级

- CUDA EP 的 DLL（`onnxruntime_providers_cuda.dll`）先 `LoadLibraryW` 探测，能加载才 append。
  这与 `6-web/backend/onnxenv.py` 的结论一致：`get_available_providers()` 报的是编译期支持，
  不是运行时可用。
- 若带 CUDA EP 建会话抛异常（cuDNN 版本不符、显存不足），捕获后只用 CPU 重建，不把错误抛给页面。
- 实际 provider 通过 `oi_get_info` 返回，接口层透传到 `/api/capabilities` 与推理结果。

### D5：预处理对齐 TRT 路径，后处理复用同一份代码

- YOLO：letterbox（padding 114）+ RGB + `/255` + NCHW，与 `yolo.cu` 的
  `warp_affine_bilinear_and_normalize_plane` 参数一致；缩放比与 pad 用于坐标还原。
- Anomalib：`normalizeWithRange`、normalized map 计算、JET 叠加、红色轮廓、
  状态文字这些纯函数从 `anomalib_infer_c.cpp` 抽到 `anomalib_post.hpp`，两个 DLL 共同 include。
- 理由：阈值口径与渲染是「看起来对不对」的主要来源，复制粘贴一份就是埋雷。

### D6：输出解析按形状判定 + 名字兜底

- YOLO：`[1, 4+nc, n]` 与 `[1, n, 4+nc]` 都认；通道维按 `4+nc(+1)` 反推，
  推不出就报出实际形状。
- Anomalib：先按输出名匹配 `score` / `anomaly_map|map`，再按元素数兜底（单元素当分数、
  最大元素数当异常图），与既有 DLL 的规则一致；`[1]` 与 `[1,1]` 分数都接受。
- 理由：仓库里同时存在转置与未转置导出、三个模型分数维度不同，写死任何一条都会崩。

### D7：一致性校验分两级

- Tier 1「结果级」：`tools/compare_backends.py`，同图、同参数、同标签，
  分别调 `infer.run_*` 的引擎分支与 ONNX 分支，比对最终结果。这是默认门禁。
- Tier 2「张量级」：`TRTFORGE_RAW_DUMP=<dir>` 触发三个 DLL 把预处理输入与后处理前输出落盘，
  `tools/compare_raw_tensors.py` 自动对齐转置后算差值。未设置环境变量时不产生任何文件。
- 理由：结果级能回答「能不能上线」，张量级才能回答「差在哪一层」。只有结果级时，
  一次不通过就要靠人肉二分；只有张量级时，又容易被无害的数值噪声牵着走。

### D8：阈值默认值来自实测而非想象

- YOLO：匹配率 100%、最小 IoU ≥ 0.99、置信度差 ≤ 1e-3；segment mask IoU ≥ 0.99；
  pose 关键点 ≤ 1 px；obb 角度 ≤ 0.5°。
- Anomalib：分数差 ≤ 1e-4、判定 100% 一致、anomaly map mean ≤ 1e-3、max ≤ 1e-2。
- 若实测发现 fp16 引擎在某个任务上系统性超出，先记录在报告里，再决定是放宽阈值
  （并同步修改 spec 的 Scenario）还是修实现；不允许「悄悄跳过」。

## Risks / Trade-offs

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| ORT 与 Python 同名 DLL 冲突 | 会话创建失败或行为异常 | D3 显式加载；启动自检里打印实际 ORT 版本 |
| CUDA/cuDNN 版本不匹配 | 页面报错 | D4 探测 + 回退 + `oi_get_info` 明示 |
| ORT 包体大（GPU 包数百 MB） | 发包体积 | 默认 GPU 包，提供 `-CpuOnly`；打包脚本按需复制 |
| 结构体与 Python 定义不一致 | 结果错位、崩溃 | 只复用已存在的三个结构体；新增字段只追加 |
| 解码口径与 TRT 漂移 | 同一模型换路径换结论 | D5 共享后处理 + D7 Tier 1 门禁 |
| fp16 引擎与 fp32 ONNX 存在系统差 | 阈值误报 | D8 阈值写进 spec，差异先在 Tier 2 定位 |
| 编译时 DLL 被占用 | `LNK1104` | 沿用 `build_infer_dlls.ps1` 的占用预检 |

## Migration Plan

1. 先合入 OpenSpec 提案并通过 `openspec validate --strict`。
2. 新增子工程与 ORT 拉取脚本；`build_infer_dlls.ps1` 增加目标但默认不阻塞既有两个 DLL。
3. 按任务粒度实现 YOLO → Anomalib，每步都先跑一致性工具再接 Web。
4. `infer.py` 分派改造保持「后缀决定分支」，`.engine` 行为零变化，可独立回滚。
5. 前端与能力接口最后接入；即使 ONNX DLL 缺失，页面仍按原样工作。
6. 文档、版本号、OpenSpec 归档一次完成。

回滚：删除或重命名 `onnx_infer.dll` 即可让所有 `.onnx` 请求退回 Python 兜底，
前端不需要任何改动（后端仍返回 `onnxruntime-fallback`）。

## Open Questions

- ORT GPU 包与 Python 侧 `onnxruntime-gpu` 版本是否需要统一到同一小版本？
  当前设计允许共存，但两条链路的 EP 可用性结论可能不同，需要在 README 里写清。
- `.onnx` 直推是否需要在页面上显示「这是 ONNX 而非 TensorRT」的显式提示文案，
  还是仅靠后端徽标即可？倾向于徽标 + 鼠标悬停说明。
