# Change: 为推理页新增 ONNX 动态库推理后端

## Why

推理页目前只有一条「真·动态库」路径：`.engine/.trt` 走 `yolo_infer.dll` 与 `anomalib_infer.dll`。
`.onnx` 只能落到 Python 的 `onnx_fallback.py`，带来四个实际问题：

1. **能力缺口**：兜底路径只支持 `detect` / `obb`，`segment` / `pose` 直接报错要求改用引擎。
2. **表现不一致**：异常检测在兜底路径下没有 C++ 侧渲染，叠加图、红色异常轮廓、热能图
   要么缺失、要么与引擎路径的阈值口径和配色不同，同一个模型换条路径结果看起来是两套东西。
3. **性能与部署**：Python 侧每次会话都要经过 numpy 拷贝与 GIL 串行，批量/视频比 C++ 慢；
   且用户必须装 `onnxruntime` 这个 Python 包，与「只发运行库、不发源码」的交付形态冲突。
4. **无法验证等价性**：现在没有任何工具能回答「同一个模型，ONNX 与 TRT 引擎跑出来差多少」。

工业缺陷检测场景里，模型能不能上线取决于判定是否稳定，而不是「能不能跑通」。
所以这次不只是补一条推理通道，还要把「ONNX 与 TRT 输出是否一致」变成可反复执行的验收项。

## What Changes

- 新增 C++ 子工程 `3-cpp-tensorrt/onnx_infer`，产出 `onnx_infer.dll`，导出 `oi_*` C ABI。
  - 一个 DLL 同时承载 YOLO 与 Anomalib 两类推理，内部两个 create 入口。
  - 运行后端为 ONNX Runtime：CPU 必可用；探测到可加载的 CUDA EP 时自动使用，否则回退 CPU，
    实际使用的 provider 可通过 `oi_get_info` 查询。
  - YOLO 覆盖 `detect` / `segment` / `pose` / `obb` 四项任务。
  - Anomalib 覆盖 PatchCore / PaDiM / EfficientAd，输出结构与 `ai_*` 一致。
- `6-web/backend/infer.py` 增加 `oi_*` ctypes 绑定，并按模型后缀分派：
  `.onnx` 走新 DLL，其余走既有 DLL；单图、摄像头帧、批量、视频四条调用链自动获得该能力。
- 后端接口改为三级选择：ONNX 动态库 → 引擎动态库 → Python 兜底，并在
  `/api/capabilities`、`/api/models` 与推理响应里如实回显实际使用的后端。
- 推理页支持直接选择 `.onnx` 文件，并在模型下拉中提供「ONNX 直推」条目。
- 新增一致性校验工具 `6-web/backend/tools/compare_backends.py`：同一张图分别走
  `.engine` 与 `.onnx`，按框/掩码/关键点/角度/分数/异常图逐项比对，给出报告与退出码。
- 新增诊断级原始张量导出（仅环境变量开启时生效，默认零行为差异）与
  `tools/compare_raw_tensors.py`，用于把差异定位到预处理、模型本身或后处理。
- 构建与打包：新增 ORT 拉取脚本、`build_infer_dlls.ps1` 增加目标、
  用户发包脚本补齐 `onnx_infer.dll` 与 ORT 运行库。
- **BREAKING**：无。`.engine` 路径行为、现有 C API 与结构体布局都保持不变。

## Impact

- Affected specs:
  - `onnx-inference-dll`（新增 capability）
  - `web-infer-console`（新增 ONNX 直推需求）
  - `web-infer-batch`（批量/视频/摄像头在 `.onnx` 下的后端一致性）
- Affected code:
  - 新增 `3-cpp-tensorrt/onnx_infer/`（`include/onnx_infer_c.h`、`src/*.cpp`、
    `CMakeLists.txt`、`README.md`、`examples/python_ctypes/`）
  - 新增 `scripts/fetch_onnxruntime.ps1`
  - `3-cpp-tensorrt/anomalib_infer/src/anomalib_post.hpp`（从现有实现抽出的共享后处理）
  - `6-web/backend/infer.py`、`app.py`、`config.py`、`models.py`
  - `6-web/backend/tools/infer_batch.py`、`tools/infer_video.py`
  - 新增 `6-web/backend/tools/compare_backends.py`、`compare_raw_tensors.py`、
    `test_onnx_infer_dll.py`、`test_backend_consistency.py`
  - `6-web/frontend/index.html`、`js/infer.js`
  - `scripts/build_infer_dlls.ps1`、`packaging/package_web_user.ps1`
- Affected docs:
  - `docs/WORKFLOW_GUIDE.md`、`docs/TRTForge_Web操作手册.md`
  - 根 `readme.md`、`6-web/README.md`
- 非目标（Non-Goals）:
  - 不把 Anomalib 训练/导出搬到 C++；仍然只消费已导出的 `.onnx` 与 `metadata.json`。
  - 不替换既有 `yolo_infer.dll` / `anomalib_infer.dll`，两者继续服务 `.engine`。
  - 不把标注页的 MobileSAM / GroundingDINO 纳入本 DLL（它们走独立 ONNX 会话，另行评估）。
  - 不追求 ONNX 与 fp16 引擎逐位相同；等价性由可配置阈值判定。
