## ADDED Requirements

### Requirement: OCR 推理在 C++ 动态库内执行

Web 后端 SHALL 通过 ctypes 调用 `ocr_infer.dll` 的 C API（`oi_create` / `oi_infer_bgr` / `oi_destroy`）完成 OCR，检测、识别、方向分类及其后处理全部在动态库内执行；Python 侧只负责参数填充与结果转换，不得引入 Python 版 PaddleOCR 或 paddlepaddle。

#### Scenario: 单图 OCR 走动态库

- **WHEN** 客户端以兼容的检测与识别模型目录调用 `POST /api/infer/ocr`
- **THEN** 后端必须通过 `ocr_infer.dll` 完成推理，响应中 `backend` 字段为 `ocr_infer.dll`

#### Scenario: 结果指针不跨请求持有

- **WHEN** 动态库返回 `oi_result`
- **THEN** 后端必须在同一次调用内深拷贝多边形、文字、置信度、角度与原始 JSON，之后不得再访问任何动态库返回的指针

#### Scenario: 桌面版不受影响

- **WHEN** 本能力上线
- **THEN** `ocr_infer.dll` 二进制、`ocr_infer_c.h` 接口与桌面版源码必须保持不变

### Requirement: OCR 动态库懒加载与故障隔离

后端 SHALL 在首次 OCR 请求时才加载 `ocr_infer.dll`，加载失败只影响 OCR，不得影响 YOLO、Anomalib 与 ONNX 推理。

#### Scenario: 启动时不加载

- **WHEN** Web 服务启动且尚未收到 OCR 请求
- **THEN** 进程中不得加载 `ocr_infer.dll` 与 `paddle_inference.dll`

#### Scenario: 运行库缺失

- **WHEN** `ocr_infer.dll` 或其依赖缺失导致加载失败
- **THEN** OCR 接口必须返回 503 并说明缺失项，`/api/capabilities` 的 `infer_ocr` 为 false 且 `infer_ocr_detail` 记录原因，YOLO 与 Anomalib 接口必须照常工作

### Requirement: OCR 模型扫描与格式标记

后端 SHALL 提供 `GET /api/ocr/models`，扫描 OCR 模型根目录（`TRTFORGE_OCR_MODEL_DIR`），按 `inference.yml` 的 `Global.model_name` 将模型归类为 det、rec、textline_ori、doc_ori、unwarp 或 unknown，并标注模型格式与是否兼容当前 Paddle 运行时。

#### Scenario: 区分新旧格式

- **WHEN** 模型根目录同时包含 `inference.pdmodel` 格式与 `inference.json` 格式的模型
- **THEN** 每个条目必须给出 `format`（`pdmodel` 或 `json`），且在 Paddle 主版本小于 3 时 `json` 条目的 `compatible` 为 false 并附原因

#### Scenario: 无法归类的目录

- **WHEN** 某目录含 `inference.yml` 但 `model_name` 无法识别类型
- **THEN** 该条目的 `kind` 必须为 `unknown`，且不得出现在检测或识别的默认候选中

### Requirement: 进入动态库前的模型预检

后端 SHALL 在调用 `oi_create` 前校验模型目录的文件完整性、`inference.yml` 必需字段以及格式与 Paddle 版本的兼容性；任一项不通过时必须返回 400 与可读原因，不得调用动态库。

#### Scenario: 新格式模型配旧运行时

- **WHEN** 请求的模型目录为 `inference.json` 格式且当前 Paddle 运行时主版本小于 3
- **THEN** 接口必须返回 400，提示该模型需要 Paddle 3.x 运行时，且不得调用 `oi_create`

#### Scenario: 识别模型缺字典

- **WHEN** 识别模型的 `inference.yml` 缺少 `PostProcess.character_dict` 或为空
- **THEN** 接口必须返回 400 并指明缺少的字段，Web 服务进程必须继续存活

#### Scenario: 方向模型缺预处理字段

- **WHEN** 文本行方向模型的 yml 缺少 `ResizeImage.size`、`NormalizeImage` 参数或 `Topk.label_list` / `topk` 之一
- **THEN** 接口必须返回 400 并指明缺少的字段

#### Scenario: 模型文件缺失

- **WHEN** 模型目录缺少模型文件、`inference.pdiparams` 或 `inference.yml`
- **THEN** 接口必须返回 400 并指明缺少的文件与目录

### Requirement: OCR 推理接口契约

`POST /api/infer/ocr` SHALL 接受图片与 det / rec（必填）、textline_ori（可选）模型目录及运行参数（device、enable_mkldnn、cpu_threads、rec_batch_size、det_thresh、det_box_thresh、det_unclip_ratio、rec_score_thresh），返回行级结果与分段耗时。

#### Scenario: 正常返回

- **WHEN** 推理成功
- **THEN** 响应必须包含 `backend`、`device`、`run_mode`、`src_w`、`src_h`、`count`、`lines`（每行含 `index`、`poly`、`text`、`score`、`angle`）、`timing` 与 `raw_json`，且 `lines` 的数量等于 `count`

#### Scenario: 未检出文字

- **WHEN** 图片中没有检出任何文字
- **THEN** 接口必须返回成功，`count` 为 0、`lines` 为空列表

#### Scenario: 参数越界

- **WHEN** 阈值不在 0–1、线程数或 batch 不在 1–64、或 device 不是 `cpu` / `gpu:0`
- **THEN** 接口必须返回 400 并指明字段

#### Scenario: 动态库推理失败

- **WHEN** `oi_infer_bgr` 返回非 `OI_OK`
- **THEN** 接口必须返回 500，并附带 `oi_last_error` 的内容

### Requirement: OCR 句柄缓存

后端 SHALL 按模型目录与运行参数缓存 OCR 句柄，容量上限为 2，淘汰时调用 `oi_destroy`；同一句柄的推理必须串行。

#### Scenario: 相同参数复用句柄

- **WHEN** 连续两次请求使用相同模型目录与运行参数
- **THEN** 第二次请求不得重新调用 `oi_create`

#### Scenario: 超出容量淘汰

- **WHEN** 第三组不同参数的请求到达
- **THEN** 最久未使用的句柄必须被 `oi_destroy` 释放，`/api/infer/cache` 中 OCR 段反映当前缓存

### Requirement: 推理页 OCR 类型与结果展示

推理页 SHALL 在推理类型中提供「OCR 文字识别」，提供模型目录与运行参数控件，并以原图叠加多边形、文本列表联动高亮的方式展示结果，支持复制与导出。

#### Scenario: 不兼容模型置灰

- **WHEN** 模型下拉中存在 `compatible=false` 的条目
- **THEN** 该条目必须置灰且显示不兼容原因，用户不能将其作为默认选择

#### Scenario: 行框联动

- **WHEN** 用户点击文本列表中的某一行
- **THEN** 原图上对应序号的多边形必须高亮；点击多边形时列表对应行必须高亮

#### Scenario: 导出结果

- **WHEN** 用户点击导出
- **THEN** 必须能下载 JSON、CSV（UTF-8 BOM）与 TXT 三种格式，TXT 按行序输出识别文字

#### Scenario: 推理耗时提示

- **WHEN** OCR 推理进行中
- **THEN** 页面必须显示推理中提示并禁止重复提交，结束后显示后端、设备 / run_mode、行数与分段耗时
