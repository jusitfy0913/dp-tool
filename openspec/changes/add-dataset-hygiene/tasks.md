## 1. 体检模块

- [x] 1.1 新增 `6-web/backend/datasetqc.py`：dHash(256 位)、32×32 缩略图、sha256、缓存读写
- [x] 1.2 缓存键 `绝对路径|大小|mtime_ns`，写失败只记日志
- [x] 1.3 分组：并查集 + n>3000 时 banding 候选过滤；输出 A/B 两级与「同组同标注」提示
- [x] 1.4 标注逐行质检，输出稳定规则码、级别、行号、原始行
- [x] 1.5 跨划分泄漏统计与组内标注不一致提示

## 2. 按组切分

- [x] 2.1 `dataset.py` 新增 `group_key()` 与 `assign_splits_grouped()`
- [x] 2.2 `preview()` 支持 `group_by`，返回实际张数与比例偏差
- [x] 2.3 `run_export()` 支持 `group_by`，`split_mode` 记录 `group:<mode>`
- [x] 2.4 manifest 增 `group_signature` 与 `qc` 块（不参与 `unchanged` 判定）
- [x] 2.5 `group_by` 默认 `none`，既有行为与指纹不变

## 3. 接口

- [x] 3.1 新增 `POST /api/dataset/qc`（只读，支持 limit 与阈值）
- [x] 3.2 `DatasetScanRequest` / `DatasetExportRequest` 增 `group_by`
- [x] 3.3 导出结果带 `qc` 摘要（有体检时）

## 4. 界面

- [x] 4.1 数据集页增「体检」按钮与结果区：分组数、跨划分组数、错误/警告数
- [x] 4.2 跨划分泄漏红色警告，措辞说明指标不可视为泛化能力
- [x] 4.3 分组列表：每组各 split 张数，可展开成员
- [x] 4.4 问题列表按规则码折叠，显示路径、行号、原始行
- [x] 4.5 未体检时显示「未体检」，不报错
- [x] 4.6 扫描/导出面板增「按组划分」下拉

## 5. 命令行与自动实验

- [x] 5.1 新增 `7-autoresearch-yolo/labelqc.py`：`--json`、`--fail-on`
- [x] 5.2 `autoexp.py` 增 `--qc-gate`，默认关闭
- [x] 5.3 `autoexp split` 增 `--group-by`，默认 `none`

## 6. 测试与文档

- [x] 6.1 新增 `tools/test_dataset_hygiene.py`：分组、不塌缩、缓存、质检规则、按组切分、默认不变
- [x] 6.2 `7-autoresearch-yolo/run_all_tests.py` 补 CLI 退出码与分组切分用例
- [x] 6.3 回归后端全部测试套件（`dpenv` 解释器）
- [x] 6.4 真机：对 `splits/pcb` 体检，核对分组数 = 10、跨划分组 = 10
- [x] 6.5 更新 `readme.md`、`6-web/README.md`、`7-autoresearch-yolo/README.md`、
      `docs/WORKFLOW_GUIDE.md`

## 7. 验证记录（2026-11-11）

| 项 | 结果 |
| --- | --- |
| `6-web/backend/tools/test_dataset_hygiene.py` | 43 / 43 通过 |
| `7-autoresearch-yolo/run_all_tests.py` | 68 / 68 通过（新增 3 条） |
| 后端全量套件 | 52 / 52 套件全绿（`dpenv` 解释器） |
| 真机：`PCB data/PCB缺陷检测`（692 张） | 10 组、10 组全部跨划分、涉及 689 张（另 3 张自成一类） |
| 真机：`splits/pcb` 按 `prefix` 重切 | 10 组各自只落在一个划分内 |
| 标注质检 | 无 error 级问题；首次约 1 秒，二次走缓存 |

**未验证（刻意留在本版结论之外）**：泄漏对 mAP 的实际影响需要重训对比，本轮没做；
`group_by` 带来的比例偏差只在单元测试与真机组数上核对，未做大规模数据集压测。
