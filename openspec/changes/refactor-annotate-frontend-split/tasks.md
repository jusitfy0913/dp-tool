## 1. 纯逻辑模块抽取

- [x] 1.1 新增 `js/annotate/tiers.js`：分档常量、`tierOf`、`tierColor`、`tierText`、
      `tierClass`、`countByTier`、`tierCountText`
- [x] 1.2 新增 `js/annotate/history.js`：`pushSnapshot`、`undoStep`、`redoStep`、
      `canUndo`、`canRedo`、`historyButtons`
- [x] 1.3 新增 `js/annotate/geometry.js`：`pointInPolygon`、`distanceToSegment`、
      `shapeCorners`、`bboxOf`、`hitTest`、`hitHandle`、`hitEdge`、`rotateHandlePoint`、
      `snapPoint`、`constrainOrthogonal`（自带图像坐标，不读全局缩放状态）
- [x] 1.4 新增 `js/annotate/importplan.js`：`planImport`、`importSummaryText`、
      `summarizeReport`、`blockingReasons`
- [x] 1.5 `annotate.js` 删除被抽走的实现并改为 import；不改变任何 DOM 结构
- [x] 1.6 静态检查：`annotate.js` 中不再出现被抽走函数的 `function` 定义

## 2. 前端单测

- [x] 2.1 `tests/test_tiers.mjs`（33 项）：阈值边界（0.7 / 0.4 落点）、`null` 与 `""`
      归 `unknown`、颜色一致性、计数与文案
- [x] 2.2 `tests/test_history.mjs`（40 项）：深度上限 50、undo/redo 往返、空栈、
      drop 计数、入参不被就地修改
- [x] 2.3 `tests/test_geometry.mjs`（76 项）：命中测试边界、吸附最近点、旋转框顶点顺序、
      `{x, y}` 与 `[x, y]` 双坐标兼容、正交约束不产生 `NaN`
- [x] 2.4 `tests/test_importplan.mjs`（53 项）：类别对不上计入跳过、空扫描阻塞导入、
      单例计数、汇总文案
- [x] 2.5 现有 `tests/test_annotate_flow.mjs`（78 项）保持全绿
- [x] 2.6 全部 `js/*.js` 与 `js/annotate/*.js` 通过 `node --check`

## 3. 全链路自检

- [x] 3.1 新增 `tools/test_full_chain.py`：ASGI 传输 + 假引擎桩
- [x] 3.2 覆盖 8 个环节：建工程 → 加图 → 写标注 → 导出 → 数据集扫描 →
      产物枚举 → 模型包校验 → 推理
- [x] 3.3 支持 `--step <名字>` 单步执行（`--list` 列出环节，`--keep` 保留临时目录）
- [x] 3.4 失败时打印该环节请求与响应体
- [x] 3.5 `run_all_tests.py` 收录并全绿（62 套，0 失败）

## 4. 可选浏览器 e2e

- [x] 4.1 新增 `tools/e2e_browser.py`：Playwright 自启临时服务，校验前端脚本无报错、
      三栏渲染、关键控件齐全、标注与推理页可切换（业务全链路由第 3 节覆盖）
- [x] 4.2 失败时保存截图与控制台摘要到 `runtime/e2e/`
- [x] 4.3 文档写明运行前置（`pip install playwright` + `playwright install chromium`）
- [x] 4.4 `.github/workflows/tests.yml` **不**新增该脚本调用

## 5. CI 与文档

- [x] 5.1 `.github/workflows/tests.yml` 增加 4 个新 `.mjs`（并补 `httpx` 依赖）
- [x] 5.2 `6-web/README.md` 增加变更点：模块划分、全链路自检、e2e 用法
- [x] 5.3 `docs/WebStudio架构与功能介绍.md` 第 8.1 节脚本表补 4 个新模块
- [x] 5.4 `docs/WORKFLOW_GUIDE.md` 增加「全链路自检」与「浏览器 e2e（可选）」两条命令
- [x] 5.5 `openspec/project.md` 的 Testing Strategy 补一句「纯逻辑必须抽模块并有 `.mjs` 单测」
