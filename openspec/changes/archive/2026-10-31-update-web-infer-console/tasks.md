# 实施任务：Web 推理控制台

> 状态：全部完成（71/71 后端用例通过，真实产物联调通过）

## 1. 后端：类别名与配色

- [x] 1.1 新增 `6-web/backend/labels.py`：解析 `labels_*.json` 的 `{name, classes[]}`，对缺失字段容忍并回退
- [x] 1.2 实现唯一的类别配色生成函数（按类别 id 取模固定色相），输出 `{id: [r,g,b]}`
- [x] 1.3 `infer.py` 的 `run_yolo()` 增加标签映射，每个框返回 `label_name`
- [x] 1.4 响应增加 `class_colors`，供前端与后端绘制共用
- [x] 1.5 `draw_yolo()` 改为绘制类别名，颜色取 `class_colors`
- [x] 1.6 `onnx_fallback.py` 的降级路径同步类别名，保持两条路径输出结构一致

## 2. 后端：阈值与句柄

- [x] 2.1 `infer.py` 的 `run_anomalib()` 增加 `image_threshold` / `pixel_threshold` 参数，透传给 `ai_create`
- [x] 2.2 `_anom_handle()` 缓存键加入两个阈值
- [x] 2.3 `app.py` 的 `/api/infer/anomalib` 增加两个 Form 参数，默认 `-1.0`
- [x] 2.4 响应同时返回生效阈值与 metadata 原阈值
- [x] 2.5 句柄缓存改为 LRU：命中时刷新访问序，上限 4，淘汰时立即 `yi_destroy` / `ai_destroy`

## 3. 后端：模型扫描

- [x] 3.1 新增 `GET /api/models`：扫描 YOLO `runs/**/weights/*.engine` 与 Anomalib `results/**/*.engine`，限制遍历深度
- [x] 3.2 每个条目附带同名 ONNX（标注是否 `.transd.onnx`）、标签文件、metadata、推断的任务类型、体积、修改时间
- [x] 3.3 结果按修改时间倒序，限制返回条数
- [x] 3.4 复用 `_ensure_allowed` 白名单校验根目录，越界时返回说明而非静默空列表

## 4. 后端：热能图

- [x] 4.1 `infer.py` 读取 `result.normalized_map`（含 `map_w` / `map_h`），按 metadata 分数范围或图自身最值归一化到 0~255
- [x] 4.2 用 `cv2.applyColorMap` 渲染，默认 `COLORMAP_TURBO`，`INTER_LINEAR` 放大到原图尺寸
- [x] 4.3 指针为空或尺寸为 0 时返回 `None`，不抛错
- [x] 4.4 响应增加 `heatmap_png` 与 `source_png`
- [x] 4.5 `/api/infer/anomalib` 增加 `heatmap_colormap` Form 参数，非法名称回退默认

## 5. 前端：参数控件

- [x] 5.1 `conf` / `nms` 改为滑块 + 数字框双向联动
- [x] 5.2 Anomalib 增加图像阈值、像素阈值两个滑块 + "沿用 metadata 默认值"开关
- [x] 5.3 显示实际生效阈值（区分"用户设定"与"metadata 原值"）
- [x] 5.4 参数改动时给出"会重建引擎、首次较慢"的提示
- [x] 5.5 参数组合命中缓存时给出视觉反馈

## 6. 前端：模型与标签选择

- [x] 6.1 增加模型下拉，数据来自 `/api/models`
- [x] 6.2 选中模型后自动填充引擎、标签文件、metadata、任务类型
- [x] 6.3 对未转置的 ONNX 给出明确提示
- [x] 6.4 增加标签文件选择框，复用现有浏览弹窗
- [x] 6.5 保留手工输入路径的能力，不被下拉强制

## 7. 前端：结果可视化

- [x] 7.1 结果区接入查看器组件（`refactor-web-studio-theme` 交付）：原图、检出图、热能图、异常轮廓四窗口，可独立开关并持久化
- [x] 7.2 检出框改为 SVG 叠加层，`viewBox` 取图像原始像素尺寸
- [x] 7.3 标签显示类别名 + 置信度，接近上边缘时翻转到框内侧
- [x] 7.4 「显示框」与「显示标签」独立开关
- [x] 7.5 低置信度框按本次结果中位数降低不透明度
- [x] 7.6 明细表：类别名、置信度、坐标、面积，支持排序，点击行高亮对应框
- [x] 7.7 置信度分布展示
- [x] 7.8 计时分解条：预处理 / 推理 / 后处理
- [x] 7.9 异常检测增加叠加图透明度滑块与像素阈值参考线
- [x] 7.10 导出 JSON / CSV，内容与屏幕一致

## 8. 测试与验收

- [x] 8.1 新增 `6-web/backend/tools/test_infer_params.py`
- [x] 8.2 覆盖阈值透传：`-1` 时不覆盖、`>=0` 时覆盖，且进入缓存键
- [x] 8.3 覆盖标签映射：有标签文件、无标签文件、标签文件缺字段三种情况
- [x] 8.4 覆盖配色一致性：后端绘制与下发的 `class_colors` 取值相同
- [x] 8.5 覆盖 LRU：上限触发淘汰、命中刷新顺序、上限内不重建
- [x] 8.6 覆盖模型扫描：任务类型推断、配套文件附带、越界根目录被拒
- [x] 8.7 覆盖热能图：正常渲染、缺图不报错、非法色带回退
- [x] 8.8 用真实产物实测：PCB 图 + `best.transd_fp16.engine` 跑通，确认类别名显示为实际缺陷名而非 `cls0`
- [x] 8.9 用真实产物实测：PatchCore 引擎 + `metadata.json` 跑通，确认热能图与异常定位一致
- [x] 8.10 显存实测：连续用 8 组不同 `conf` 推理，确认常驻句柄数不超过上限

## 9. 回归

- [x] 9.1 `test_augment_params.py` 26/26 通过
- [x] 9.2 `test_dataset_workbench.py` 20/20 通过
- [x] 9.3 `test_exit_codes.py` 6/6 通过
- [x] 9.4 `test_onnx_fallback.py` 36/36 通过
- [x] 9.5 `test_infer_params.py` 71/71 通过
