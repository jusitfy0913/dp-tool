## 1. 规格与提案
- [x] 1.1 创建 `proposal.md`、`design.md`、`tasks.md`
- [x] 1.2 新增 `specs/web-ocr-inference/spec.md`（新 capability 的 ADDED 需求）
- [x] 1.3 新增 `specs/web-infer-batch/spec.md` 的「目录批量支持 OCR」增量需求
- [x] 1.4 `openspec validate add-web-ocr-inference --strict` 通过
- [x] 1.5 用户确认提案后才开始编码

## 2. 老版本模型准备（2.6.2 可加载）
- [x] 2.1 下载 `ch_PP-OCRv4_det_infer`、`ch_PP-OCRv4_rec_infer`、`ch_ppocr_mobile_v2.0_cls_infer`
  到 `9-python-paddleOCR/model/legacy_v4/`（可选 server 版 det / rec）
- [x] 2.2 新增 `6-web/backend/tools/gen_legacy_ocr_yml.py`（UTF-8），缺 yml / 缺字段时生成，不覆盖已有字段
- [x] 2.3 `ocr_basic.exe` 实测 det+rec、det+rec+方向、`--device=gpu:0`，倒置图验证方向
- [x] 2.4 实测结果写入 `design.md`「实测记录」；不通过时用 `export_model.py` 重新导出
- [x] 2.5 用户确认实测结果

## 3. 后端 DLL 桥接
- [x] 3.1 `ocr_infer.py`：ctypes 结构体与 `ocr_infer_c.h` 逐字段对齐，函数签名声明
- [x] 3.2 懒加载：首次请求加载，状态缓存，失败原因可查询，不影响 `infer.py`
- [x] 3.3 句柄缓存 LRU=2 + 每句柄锁 + 全局创建锁，淘汰时 `oi_destroy`
- [x] 3.4 `cv2.imdecode` → `oi_infer_bgr`，`oi_result` 立即深拷贝，`oi_last_error` 转可读错误

## 4. 模型扫描与预检
- [x] 4.1 `ocr_models.py`：按 `TRTFORGE_OCR_MODEL_DIR` 扫描，按 `model_name` 归类，标注格式
- [x] 4.2 读取 Paddle 版本（开发 `version.txt` / 安装包 `paddle_inference_version.txt`）
- [x] 4.3 预检：文件齐全、yml 必需键、`json` 与 Paddle 主版本兼容性
- [x] 4.4 命令行 `python -m ocr_models --check <dir>`

## 5. 接口
- [x] 5.1 `GET /api/ocr/models`
- [x] 5.2 `POST /api/infer/ocr`（参数范围校验、400 / 503 / 500 语义、0 行成功）
- [x] 5.3 `/api/capabilities` 增加 `infer_ocr`、`infer_ocr_detail`、`paddle_version`
- [x] 5.4 `/api/infer/cache` 增加 OCR 段
- [x] 5.5 `APP_VERSION` → 1.27.0

## 6. 前端推理页
- [x] 6.1 `index.html`：`#inferKind` 增加 OCR，新增 `#inferOcrGroup`
- [x] 6.2 模型下拉接 `/api/ocr/models`，不兼容项置灰并显示原因；目录浏览按钮
- [x] 6.3 `js/ocr.js`：多边形叠加 + 序号、文本列表、行框双向高亮
- [x] 6.4 复制全部文字、导出 JSON / CSV（UTF-8 BOM）/ TXT
- [x] 6.5 `infer.js`：`syncKind` / `onSubmit` / `renderResult` / `currentParamKey`，指标与耗时条
- [x] 6.6 推理中提示与防重复提交

## 7. 目录批量
- [x] 7.1 `infer_batch.py` 增加 `kind=ocr`，产出 `<name>.ocr.json` 与 `ocr_summary.csv`
- [x] 7.2 `inferbatch.js`：OCR 参数透传；视频 / 摄像头对 OCR 禁用并提示
- [x] 7.3 后端对 `kind=ocr` 的视频请求返回 400

## 8. 测试（UTF-8，接入 `run_all_tests.py`）
- [x] 8.1 `tools/test_ocr_infer_dll.py`：错误路径、v4 实图 ≥1 行且含预期文字、字段一一对应、两次结果一致、耗时 > 0
- [x] 8.2 `tools/test_ocr_api.py`：参数校验、扫描分类（用户 3.x 模型标不兼容）、json 模型 400、yml 缺字段 400、capabilities、DLL 缺失 503 且 YOLO 不受影响
- [x] 8.3 `tools/test_infer_batch.py` 增加 OCR 目录批量用例
- [x] 8.4 `run_all_tests.py` 全绿；`node --check` 前端脚本
- [x] 8.5 手工验证：推理页中文文档图，框 / 文字 / 高亮联动；GPU 0 耗时明显低于 CPU

## 9. 打包与启动
- [x] 9.1 `package_web_user.ps1`：`-WithOcr`、`-OcrModelDir`，登记 `bin_manifest.txt`
- [x] 9.2 `start_user.ps1`、`start_web_cmd.bat` 设置 `TRTFORGE_OCR_MODEL_DIR`

## 10. 文档与归档
- [x] 10.1 根 `readme.md`、`6-web/README.md`：v1.27.0 变更（新旧版本差异，AI 可读格式）
- [x] 10.2 `docs/WORKFLOW_GUIDE.md`：OCR 推理章节、故障排查行
- [x] 10.3 `docs/TRTForge_Web操作手册.md`：OCR 操作说明
- [x] 10.4 `ocr/ocr_infer/README.md`：「Web ctypes 调用」一节与版本差异 YAML 块
- [x] 10.5 `openspec archive add-web-ocr-inference` 并 `openspec validate --specs --strict`
