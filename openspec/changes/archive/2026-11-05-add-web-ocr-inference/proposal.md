# Change: Web 推理页接入 PaddleOCR（C++ 动态库 + Paddle Inference 2.6.2）

## Why

TRTForge 桌面版已经通过 `ocr_infer.dll`（C++，Paddle Inference）提供 OCR，
但 Web Studio 推理页只支持 YOLO 与 Anomalib，用户在浏览器里无法做文字检测与识别。
这带来三个实际问题：

1. **能力缺口**：只装了 Web 安装包的现场机器（含无桌面环境的 Windows Server）用不了 OCR。
2. **交付形态冲突**：如果走 Python 版 PaddleOCR，就要用户装 `paddlepaddle` / `paddleocr`
   Python 包，与「只发运行库、不发源码」的交付方式冲突；推理也会离开 C++。
3. **模型格式不匹配无提示**：用户手上的新模型（PP-OCRv6/v5，`inference.json`，飞桨 3.x 格式）
   当前 2.6.2 运行时加载不了，而 C++ 管线遇到配置缺字段会直接 `exit(-1)`，
   放进 Web 进程会把整个服务一起带走。

所以本次要做的是：推理全部留在 C++ 动态库里，Web 只做薄桥接；
同时在进 DLL 之前把「模型格式 / Paddle 版本 / yml 字段」拦一道，保证服务不崩。

## What Changes

- **推理方式**：检测、识别、方向分类、DB 后处理、CTC 解码全部在既有 `ocr_infer.dll`
  （Paddle Inference 2.6.2，`oi_create` / `oi_infer_bgr` / `oi_destroy`）里完成。
  Web 后端新增 `6-web/backend/ocr_infer.py`，用 ctypes 只做参数填充与结果转 JSON，
  与 `yolo_infer.dll` / `anomalib_infer.dll` / `onnx_infer.dll` 的接入方式一致。
- **DLL 懒加载与故障隔离**：首次 OCR 请求才加载 `ocr_infer.dll`（及 1.7GB 的
  `paddle_inference.dll`）；加载失败只让 OCR 不可用，YOLO / Anomalib / ONNX 不受影响。
- **模型管理**：新增 `6-web/backend/ocr_models.py`，扫描 OCR 模型根目录，
  按 `inference.yml` 的 `Global.model_name` 归类（det / rec / textline_ori / doc_ori / unwarp），
  标注格式（`pdmodel` / `json`）与是否兼容当前 Paddle 版本。
- **进 DLL 前预检**：模型格式与 Paddle 主版本不匹配、或 yml 缺少 C++ 管线必需字段时，
  接口返回 400 并说明原因，绝不调用 `oi_create`。
- **接口**：新增 `GET /api/ocr/models`、`POST /api/infer/ocr`；
  `/api/capabilities` 增加 `infer_ocr`、`infer_ocr_detail`、`paddle_version`；
  `/api/infer/cache` 增加 OCR 句柄诊断。
- **推理页**：`#inferKind` 新增「OCR 文字识别」；参数组含检测/识别/方向模型目录、
  设备、线程、识别 batch、MKLDNN、阈值；结果为原图 + 多边形叠加 + 文本列表联动高亮，
  支持复制全部文字与导出 JSON / CSV / TXT。
- **目录批量**：`kind=ocr` 走同一 DLL，每张图一份 JSON + 汇总 CSV；视频与摄像头对 OCR 禁用。
- **老版本模型**：下载 2.6.2 可加载的 PP-OCRv4 检测/识别与 `ch_ppocr_mobile_v2.0_cls`
  方向模型到 `9-python-paddleOCR/model/legacy_v4/`，新增
  `6-web/backend/tools/gen_legacy_ocr_yml.py` 补齐 `inference.yml`。
- **打包**：`package_web_user.ps1` 新增 `-WithOcr`（默认关，Paddle 运行库体积大）与
  `-OcrModelDir`；启动脚本设置 `TRTFORGE_OCR_MODEL_DIR`。
- `APP_VERSION` 1.26.0 → 1.27.0。
- **BREAKING**：无。`ocr_infer.dll` 二进制、C API、桌面版源码全部不变，不重新编译。

## Impact

- Affected specs:
  - `web-ocr-inference`（新增 capability）
  - `web-infer-batch`（新增「目录批量支持 OCR」需求）
- Affected code:
  - 新增 `6-web/backend/ocr_infer.py`、`6-web/backend/ocr_models.py`
  - `6-web/backend/app.py`、`config.py`
  - `6-web/backend/tools/infer_batch.py`
  - 新增 `6-web/backend/tools/gen_legacy_ocr_yml.py`、`test_ocr_infer_dll.py`、`test_ocr_api.py`
  - `6-web/backend/tools/test_infer_batch.py`
  - `6-web/frontend/index.html`、`js/infer.js`、`js/inferbatch.js`，新增 `js/ocr.js`、`css/style.css`
  - `packaging/package_web_user.ps1`、`packaging/templates/start_user.ps1`、`start_web_cmd.bat`
- Affected docs:
  - 根 `readme.md`、`6-web/README.md`
  - `docs/WORKFLOW_GUIDE.md`、`docs/TRTForge_Web操作手册.md`
  - `ocr/ocr_infer/README.md`（新增「Web ctypes 调用」一节，DLL 本身不变）
- 不改动:
  - `4-qt-studio` 源码与桌面版行为、`4-qt-studio/bin/ocr_infer.dll` 二进制
  - 仓库根目录 `bin` / `include` / `lib`
- 非目标（Non-Goals）:
  - 不升级 Paddle Inference 到 3.x，不加载 `inference.json` 模型（留给第二期变更
    `update-ocr-paddle3-runtime`）。
  - 不启用文档方向分类与 UVDoc 矫正（无 2.6.2 可用的老版本模型，第二期一并处理）。
  - 不做 OCR 视频 / 摄像头实时推理（CPU 单张 10 秒级，不具备实时性）。
  - 不引入 Python 版 PaddleOCR / paddlepaddle。
