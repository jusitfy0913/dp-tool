## 1. 纯逻辑模块抽取

- [ ] 1.1 新增 `js/annotate/tiers.js`：分档常量、`tierOf`、`tierColor`、`tierText`、`countByTier`
- [ ] 1.2 新增 `js/annotate/history.js`：`pushSnapshot`、`undo`、`redo`、`canUndo`、`canRedo`
- [ ] 1.3 新增 `js/annotate/geometry.js`：`hitTest`、`snapPoint`、`bboxOf`、`rotatedCorners`、
      `objectAt`（复用 `viewer.js` 的坐标变换，不重写）
- [ ] 1.4 新增 `js/annotate/importplan.js`：`planImport`、`importSummaryText`
- [ ] 1.5 `annotate.js` 删除被抽走的实现并改为 import；不改变任何 DOM 结构
- [ ] 1.6 静态检查：`annotate.js` 中不再出现被抽走函数的 `function` 定义

## 2. 前端单测

- [ ] 2.1 `tests/test_tiers.mjs`：阈值边界（0.7 / 0.4 落点）、颜色一致性、计数
- [ ] 2.2 `tests/test_history.mjs`：深度上限、undo/redo 往返、空栈、drop 计数
- [ ] 2.3 `tests/test_geometry.mjs`：命中测试边界、吸附最近点、旋转框顶点顺序、缩放不变量
- [ ] 2.4 `tests/test_importplan.mjs`：类别对不上计入跳过、空计划、汇总文案
- [ ] 2.5 现有 `tests/test_annotate_flow.mjs`（78 项）保持全绿
- [ ] 2.6 全部 `js/*.js` 与 `js/annotate/*.js` 通过 `node --check`

## 3. 全链路自检

- [ ] 3.1 新增 `tools/test_full_chain.py`：ASGI 传输 + 假引擎桩
- [ ] 3.2 覆盖 8 个环节：建工程 → 加图 → 写标注 → 导出 → 数据集扫描 →
      产物枚举 → 模型包校验 → 推理
- [ ] 3.3 支持 `--step <名字>` 单步执行
- [ ] 3.4 失败时打印该环节请求与响应体
- [ ] 3.5 `run_all_tests.py` 收录并全绿

## 4. 可选浏览器 e2e

- [ ] 4.1 新增 `tools/e2e_browser.py`：Playwright，走完建工程 → 标注 → 保存 → 导出 → 推理页
- [ ] 4.2 失败时保存截图与控制台摘要到 `runtime/e2e/`
- [ ] 4.3 文档写明运行前置（`pip install playwright` + `playwright install chromium`）
- [ ] 4.4 `.github/workflows/tests.yml` **不**新增该脚本调用

## 5. CI 与文档

- [ ] 5.1 `.github/workflows/tests.yml` 增加 4 个新 `.mjs`
- [ ] 5.2 `6-web/README.md` 增加变更点：模块划分、全链路自检、e2e 用法
- [ ] 5.3 `docs/WebStudio架构与功能介绍.md` 第 8.1 节脚本表补 4 个新模块
- [ ] 5.4 `docs/WORKFLOW_GUIDE.md` 增加「全链路自检」与「浏览器 e2e（可选）」两条命令
- [ ] 5.5 `openspec/project.md` 的 Testing Strategy 补一句「纯逻辑必须抽模块并有 `.mjs` 单测」
