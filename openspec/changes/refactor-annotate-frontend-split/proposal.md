# Change: 标注页大文件按纯逻辑抽模块 + 新增全链路自检

## Why

`6-web/frontend/js/annotate.js` 现在 **4983 行 / 182 KB**，文件里已经有 31 个
用 `// ----` 划出的自然段落（常量、状态、草稿落盘、审核状态、历史、读写工程、
命中测试、吸附、指针交互、渲染列表、类别管理、导入 YOLO、快捷键、初始化……），
说明作者一直在按域组织，只是没有物理拆开。

后果：

1. **改一处要通读上下文**。例如调草稿置信度分档，要确认没有任何渲染函数把阈值写死；
2. **纯逻辑无法单测**。`annotateflow.js` 已经把「五步判定」抽成纯函数并有 78 项单测，
   但分档、历史栈、命中测试、吸附、几何计算这些同样是纯计算的代码还埋在 DOM 代码里，
   只能靠人工点；
3. **回归靠手点**。CI 只跑前端纯函数单测和 `node --check`，
   没有人真的把「建工程 → 加图 → 标框 → 保存 → 导出 → 训练产物 → 推理」走一遍，
   改坏了要等用户点出来。

## What Changes

分两件事，都能独立验收。

### 一、抽出 4 个纯逻辑模块（不动 DOM 代码）

| 新模块 | 从 `annotate.js` 抽出什么 | 为什么是纯的 |
| --- | --- | --- |
| `annotate/tiers.js` | 草稿置信度分档、分档颜色/文案、数量统计 | 只接受 `confidence` 与阈值常量，返回分档名 |
| `annotate/history.js` | 双栈撤销/重做（深度 50） | 只接受快照与栈，返回新栈与「能否撤销/重做」 |
| `annotate/geometry.js` | 命中测试、吸附、包围盒、旋转框顶点、缩放入侵 | 只接受对象数组与 `{scale, offset}`，返回索引与坐标 |
| `annotate/importplan.js` | YOLO 标签导入的「预览 → 差异 → 待写入清单」 | 只接受扫描结果与类别映射，返回计划对象 |

`annotate.js` 改为 `import` 这些模块，自己的职责缩到「DOM 事件 + 调纯函数 + 渲染」。
目标：从 4983 行降到 3,300 行以内（净减少约 1,700 行，其余转到 4 个模块）。

**不拆的部分**：渲染函数、指针交互、初始化。它们闭包着 `state` 与 200 多个 DOM 节点，
搬出去要先把 `state` 改造成显式传递的上下文对象，那是独立提案，不在本批混做。

### 二、新增全链路自检

| 脚本 | 覆盖 | 何时跑 |
| --- | --- | --- |
| `tools/test_full_chain.py` | 后端进程内（ASGI）走完整链：建工程 → 加图 → 标注落盘 → 导出 → 数据集扫描 → 训练产物枚举 → 模型包校验 → 推理（假引擎）→ 路由表 | 每次 CI |
| `frontend/tests/test_annotate_flow.mjs`（已有） | 五步判定 | 每次 CI |
| 新增 4 个 `.mjs` | 上面 4 个纯模块 | 每次 CI |
| `tools/e2e_browser.py`（可选，不进 CI） | 真实浏览器点完整流程 | 发版前手工 |

`test_full_chain.py` 用假引擎桩（仓库已有 `abi_probe.cpp` 与 `test_infer_pool` 的桩思路），
不依赖 GPU 与真实 DLL，符合现有 CI 的「不需要 GPU / DLL / torch」原则。

**BREAKING**：无。`annotate.js` 仍以同一路径被 `index.html` 加载，
对外暴露的初始化函数名不变，其他页面不引用它的内部函数。

## Impact

- Affected specs: `web-studio`（新增「前端大文件按纯逻辑拆分」）、
  `web-e2e-regression`（新增能力：全链路自检）
- Affected code:
  - 新增 `6-web/frontend/js/annotate/{tiers,history,geometry,importplan}.js`
  - `6-web/frontend/js/annotate.js` 删掉被抽走的实现，改为 import
  - 新增 `6-web/frontend/tests/test_{tiers,history,geometry,importplan}.mjs`
  - 新增 `6-web/backend/tools/test_full_chain.py`
  - 新增 `6-web/backend/tools/e2e_browser.py`（可选）
  - `.github/workflows/tests.yml` 增加 4 个 `.mjs` 与 `test_full_chain.py`
- 不涉及：`index.html` 结构、CSS、后端接口、C++ 与 DLL。
