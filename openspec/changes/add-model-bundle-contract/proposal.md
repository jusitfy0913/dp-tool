# Change: 模型包契约（manifest + 校验 + 推理页接入）

## Why

现在「一个可以交付的模型」是散落在训练目录里的五六个文件：`.engine`、`.onnx`、
`labels_*.json`、`metadata.json`、`<engine>.meta.json`。它们之间没有显式登记关系，
只能靠文件名后缀和目录层级猜配对（见 `6-web/backend/models.py` 的 `_find_onnx`、
`_run_dir`、`_find_labels`）。这带来四个已经发生过的问题：

1. **选错产物**：引擎与 ONNX 名字永远不同名（`best.transd_fp16.engine` 对
   `best.transd.onnx`），用未转置的 `best.onnx` 编出的引擎会让解码越界，
   目前只能靠页面上的文字警告；
2. **换机跑不起来**：`.engine` 绑定 GPU 架构与 TensorRT 版本，
   `<engine>.meta.json` 里没有记录这两项，用户只看到底层英文报错；
3. **任务类型对不上**：`detect` 的引擎配了 `segment` 的模式，结果是空的或乱框；
4. **交付说不清**：把模型给客户或做成 SDK 时，没有一份「这个模型是什么、
   在什么环境下能跑、类别叫什么」的清单。

这四个问题的共同根因是**产物之间没有契约**。本提案把「一个模型」升级成一个自描述的包。

## What Changes

- 新增模型包骨架 `<名字>.tfbundle/`：`manifest.json` + 模型文件 + 标签 +
  metadata + 来源元数据，全部放在一个目录里，复制/改名/打包不会走散。
- 新增 `manifest.json` schema v1（`trtforge.bundle.v1`），字段见
  `specs/model-bundle/spec.md`，覆盖：种类、任务、运行时环境、文件清单与哈希、
  类别表、异常阈值、工作点、来源追溯。
- 新增三条使用路径：
  - **构建**：从现有散装产物一键造包（引擎构建成功时自动造）；
  - **校验**：加载前检查 schema、哈希、任务一致性与运行时兼容性，
    结论分 `ok` / `warn` / `block` 三档，每条都带中文原因与修复建议；
  - **兼容**：散装文件继续可用，只是走「兼容」入口，不再作为默认路径。
- `enginemeta.write_meta` 增加采集 TensorRT 版本、CUDA 版本、GPU 名称、
  compute capability、输入尺寸、是否动态 shape、ONNX 是否转置。
- 推理接口新增 `bundle_id` 入参，与现有的引擎/ONNX 路径入参二选一；
  使用模型包时标签文件、metadata、阈值、工作点全部从包里读，页面不再要求重复填写。
- `/api/models` 响应新增 `bundle` 字段（包的 id、manifest 路径、校验结论），
  原有字段全部保留，前端不升级也能用。
- 新增 `GET /api/bundles`、`POST /api/bundles/build`、`POST /api/bundles/verify`、
  `GET /api/bundles/{bundle_id}/manifest`。

**BREAKING**：无。对外 C ABI（`yi_` / `ai_` / `onnx_` / `oi_`）完全不动，
模型包是 DLL 之外的旁挂约定，DLL 不知道自己被打包。

## Impact

- Affected specs: `model-bundle`（新增能力）、`web-infer-console`（新增一条模型选择要求）
- Affected code:
  - 新增 `6-web/backend/bundle.py`（构建、校验、枚举）
  - `6-web/backend/enginemeta.py`（采集运行时字段）
  - `6-web/backend/models.py`（扫描结果附带所属模型包）
  - `6-web/backend/app.py`（4 个新接口 + 推理接口 `bundle_id` 支持）
  - `6-web/frontend/js/infer.js`、`6-web/frontend/index.html`（模型下拉分组）
  - 新增 `6-web/backend/tools/test_model_bundle.py`
- 不涉及：`3-cpp-tensorrt/*`、`ocr/ocr_infer`、训练 launcher、任何 CMake 目标。
