## 1. 指标层

- [ ] 1.1 `valmetrics.py` 新增 `operating_points()`：一次匹配 + 前缀累加，输出 19 点曲线
- [ ] 1.2 `valmetrics.py` 新增 `recommend_operating_point()`：目标漏检反查最小 conf
- [ ] 1.3 `valmetrics.py` 新增良品/漏标/无法解析的图片状态统计
- [ ] 1.4 无良品图时 `ok_fp_rate` 返回 `null`，不返回 0

## 2. 引擎验证

- [ ] 2.1 `tools/val_engine.py` 逐图判定 `labeled / ok / missing / unparsed`
- [ ] 2.2 报告新增 `operating_points` 与 `ok`/`missing_label_images` 统计
- [ ] 2.3 终端摘要打印「漏检 1% → conf、过杀/图、良品误报」
- [ ] 2.4 无标签文件与无法解析的图单独警告，不计入良品

## 3. 良品目录

- [ ] 3.1 `dataset.py` 扫描支持 `ok_dirs`，标记 `DatasetImage.is_ok`
- [ ] 3.2 良品目录内发现标注时报错并指明文件
- [ ] 3.3 `preview()` 与 `run_export()` 强制良品图进 `ok_split`（默认 val）
- [ ] 3.4 导出 manifest 记 `ok` 块（目录、张数、目标划分）
- [ ] 3.5 `datasetqc.inspect()` 接受 `ok_dirs`，良品图不报 `LBL_MISSING`

## 4. 接口

- [ ] 4.1 `DatasetScanRequest` / `DatasetExportRequest` / `DatasetQcRequest` 增 `ok_dirs`
- [ ] 4.2 新增 `POST /api/engine/operating-point`（写 `<engine>.meta.json`）
- [ ] 4.3 新增 `GET /api/engine/operating-point?engine=`（读回，供推理页预填）

## 5. 界面

- [ ] 5.1 数据集页增「良品目录」输入与「良品进哪个划分」
- [ ] 5.2 扫描/导出结果显示良品张数与漏标张数
- [ ] 5.3 精度卡片显示漏检率、每图过杀、良品误报率
- [ ] 5.4 新增「漏检-过杀」零依赖 SVG 曲线
- [ ] 5.5 「目标漏检率」输入 + 推荐 conf + 「保存为引擎工作点」按钮
- [ ] 5.6 推理页选中引擎时预填 conf 并标注来源

## 6. 自动实验

- [ ] 6.1 `results.tsv` 增 `fp_per_image`（按列名迁移）
- [ ] 6.2 `autoexp.py` 由 P/R 与 GT 数反推过杀率
- [ ] 6.3 `judge_round()` 增加过杀不劣化约束（含容差与理由文案）
- [ ] 6.4 holdout 台账增 `fp_per_image`

## 7. 测试与文档

- [ ] 7.1 新增 `tools/test_operating_points.py`（曲线自洽、反查、无良品图、空输入、前缀匹配等价性）
- [ ] 7.2 `tools/test_dataset_workbench.py` 补良品目录与漏标区分
- [ ] 7.3 `7-autoresearch-yolo/run_all_tests.py` 补过杀列与 KEEP 约束用例
- [ ] 7.4 回归后端全部套件与 `openspec validate --strict`
- [ ] 7.5 更新 `readme.md`、`6-web/README.md`、`7-autoresearch-yolo/README.md`、
      `docs/WORKFLOW_GUIDE.md`、`docs/TRTForge_Web操作手册.md`
