# Tasks: 提示式预标注（点/框/文本）+ 审核状态

## 1. 模型仓库（`modelzoo.py`）

- [x] 1.1 内置清单：MobileSAM（encoder + decoder）、GroundingDINO SwinT 量化、BERT tokenizer
- [x] 1.2 模型根目录：`TRTFORGE_MODEL_DIR` 优先，否则 `6-web/models`（已加 `.gitignore`；落地时没再套一层 `zoo/`，少一级目录少一类"放错位置"）
- [x] 1.3 `list_models()`：返回 `id / name / use / size / present / path / url / license`，磁盘上已有的自动认
- [x] 1.4 下载任务表：守护线程 + 进度字节 + 取消事件，保留最近若干条
- [x] 1.5 流式写 `.part`，完成后 `os.replace`；已存在 `.part` 时按 `Range` 续传
- [x] 1.6 取消时删 `.part` 并标记 `cancelled`；网络失败标记 `failed` 并带原因
- [x] 1.7 同一模型不允许并发下载两份
- [x] 1.8 `require(id)`：给推理通道用，缺失时抛可读错误（提示去下载）

## 2. SAM 提示通道（`promptlabel.py`）

- [x] 2.1 `SamSession`：encoder / decoder 句柄常驻，按 `路径 + mtime + size` 缓存 embedding（LRU 2）
- [x] 2.2 预处理：等比缩放到 1024 长边 + 严格按契约做 BGR float32（**不归一化**）
- [x] 2.3 提示展开：正点 `label=1`、负点 `label=0`、框 `label=2/3`，补 padding 点与 batch 维
- [x] 2.4 decode：取最高分 mask，`warpAffine` 逆变换回原图
- [x] 2.5 `mask → polygon`：`findContours` + `approxPolyDP(epsilon * arcLength)` + 面积过滤 + 点数上限 400
- [x] 2.6 退化掩码（全 0 / 全 1 / 单像素）返回空列表并给警告，不抛异常
- [x] 2.7 草稿 `score = -1`、`_note` 写明通道；`class_id` 由调用方指定

## 3. 文本提示通道（`textlabel.py`）

- [x] 3.1 预处理：resize 到 `800x1200` + ImageNet 归一化 + CHW
- [x] 3.2 文本编码：`tokenizers` 读 BERT tokenizer → `input_ids / attention_mask / token_type_ids`
- [x] 3.3 `position_ids` 与 `text_token_mask` 按 special token 分组生成
- [x] 3.4 后处理：`sigmoid` → 框阈值 → 短语阈值 → token span 还原短语 → 框映射回原图
- [x] 3.5 越界框夹到图内；退化框（宽或高为 0）丢弃并计数
- [x] 3.6 `refine=True`：对每个框调 SAM，**一次 encode 多次 decode**
- [x] 3.7 模型缺失时抛 `ModelMissing`，接口层转 400 并带下载指引

## 4. 草稿文件升级（`annotation_drafts.py`）

- [x] 4.1 `CURRENT_VERSION = 2`，新增 `review: {图片 id: {state, at_ns}}`
- [x] 4.2 `load()` 返回 `(images, review, meta)`；v1 文件 `review` 视为空
- [x] 4.3 `normalize_review()`：只接受 `state in {"checked", "unchecked"}`，坏条目跳过并警告
- [x] 4.4 `merge()` 支持 `review` 增量合并（只覆盖传入的图片 id），草稿与审核同一次写盘
- [x] 4.5 `clear()` 只删草稿、保留 review；`save(replace=True)` 整份替换
- [x] 4.6 统计接口返回 `reviewed` / `unreviewed` 张数

## 5. 后端接口（`app.py`）

- [x] 5.1 `POST /api/annotation/prompt-label`：`{image, points, labels, box, epsilon, label}` → 多边形草稿
- [x] 5.2 `POST /api/annotation/text-label`：`{image, prompt, box_thr, text_thr, refine, epsilon}` → 草稿
- [x] 5.3 `GET /api/modelzoo`：模型清单
- [x] 5.4 `POST /api/modelzoo/download` + `GET /api/modelzoo/download/{job_id}`：下载与进度
- [x] 5.5 草稿接口读写 `review`（GET 返回、POST 增量合并）
- [x] 5.6 全部走 `_ensure_allowed_path` 白名单；模型缺失转 400 而不是 500

## 6. 批量任务（`autolabeljobs.py`）

- [x] 6.1 `mode: "engine" | "text"`，`text` 模式带 `prompt`
- [x] 6.2 `text` 模式先做一次模型就绪检查，缺失时启动即失败（不建空任务）
- [x] 6.3 逐图落盘语义不变；单张失败继续

## 7. 前端：提示交互与模型仓库

- [x] 7.1 预标注参数区加「通道」切换（引擎 / 文本提示），切换时显示对应控件
- [x] 7.2 文本提示输入框 + 框阈值 / 文本阈值 + 「细化为多边形」开关 + 掩码精细度
- [x] 7.3 模型仓库卡片：已下载 / 未下载 / 下载中三态，带进度与取消
- [x] 7.4 画布提示模式：左键正点、`Shift+左键` 负点、拖动出框、`Enter` 执行、`Esc` 清空
- [x] 7.5 画布叠加：正点绿实心、负点红叉、框虚线；提示模式与绘制工具互斥
- [x] 7.6 `runPromptLabel()` / `runTextLabel()`：结果进草稿池并标脏
- [x] 7.7 缺模型时按钮禁用并给「去下载」指引

## 8. 前端：审核状态

- [x] 8.1 `K` 标已审核并跳下一张未审核；`Shift+K` 取消已审核
- [x] 8.2 左侧列表已审核项带 ✓ 标识（绿）/ 未审核灰
- [x] 8.3 审核区显示 `已审核 n/total`
- [x] 8.4 审核状态随草稿防抖写盘；切工程与刷新后恢复
- [x] 8.5 快捷键只在标注页激活、焦点不在输入框时生效

## 9. 测试

- [x] 9.1 `test_prompt_label.py`：轮廓近似 / 点数上限 / 退化掩码 / encode 缓存命中 / 坐标往返 ≤ 1px
- [x] 9.2 `test_text_label.py`：tokenizer 编号 / 短语还原 / 双阈值边界 / 越界夹取 / 模型缺失报错
- [x] 9.3 `test_modelzoo.py`：清单推导 / `.part` 续传 / 取消清理 / 并发保护 / 目录覆盖
- [x] 9.4 `test_annotation_drafts.py` 扩：v1→v2 兼容 / review 增量合并 / 坏 review 跳过 / clear 保留审核
- [x] 9.5 `run_all_tests.py` 全绿

## 10. 文档与规范

- [x] 10.1 更新 `dp-tool/readme.md`（v1.24 版本记录 + 新旧对照）
- [x] 10.2 更新 `6-web/README.md`（三通道对比、参数表、模型下载、故障排查）
- [x] 10.3 更新 `docs/WORKFLOW_GUIDE.md`（§2.9 提示式预标注与审核）
- [x] 10.4 `openspec validate add-prompt-autolabel-review --strict` 通过后归档

