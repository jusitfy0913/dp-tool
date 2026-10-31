# Change: 重构 Web 推理控制台，暴露完整参数与结果可视化

## Why

推理是这套系统最终交付价值的出口，但当前推理页只能算"能跑通的最小演示"。实测确诊的五个问题：

1. **Anomalib 阈值完全无法调整**。DLL 明确支持覆盖（`anomalib_infer_c.h:95-99`：`>=0` 覆盖、`<0` 用 metadata），后端却写死传 `-1.0, -1.0`（`infer.py:506-512`），API 层（`app.py:841-848`）连字段都没有。现场换批次、换光照时阈值必须能调，否则只能重训。
2. **类别名丢失**。`labels_*.json` 在 6-web 后端零引用（只有 Qt 的 `LabelStore.cpp:13-50` 读），推理结果图的文字与前端胶囊都硬编码成 `cls{数字}`（`infer.py:467`、`infer.js:121`）。用户看到 `cls0` 并不知道那是 `missing_hole`。
3. **模型要手工粘贴路径**。`/api/runs`（`app.py:575-578`）是内存子进程任务表，与训练产物目录无关，全后端没有扫描 `runs/` 与 `results/` 的能力。用户每次都得点目录树找 `.engine`，还必须自己知道要用 `transd` 那个而不是另一个。
4. **调参会漏显存**。句柄缓存键是 `(路径, task, conf, nms)`（`infer.py:330-357`），每个不同组合都新建并常驻一个 TensorRT 引擎，只有进程退出才释放（`app.py:52`）。一旦把参数做成滑块反复试，显存会被吃光。
5. **没有热能图，且结果图放大即糊**。DLL 一直返回 `normalized_map`（`infer.py:106-108`，float 指针 + `map_w`/`map_h`），Python 侧从未读取，所以异常检测只有叠加图与轮廓图。同时检出框与文字是被烧进 PNG 的（`infer.py:467`），放大后是马赛克。

## What Changes

**后端**

- 推理接口暴露全部可调参数：Anomalib 增加图像阈值与像素阈值覆盖；YOLO 增加标签文件参数。
- 新增标签文件解析模块，读取 `labels_*.json` 的 `classes` 数组，把整数类别号映射为类别名与固定配色，并下发 `class_colors` 供前后端共用。
- 新增训练产物扫描接口：遍历 YOLO 的 `runs/**/weights/*.engine` 与 Anomalib 的 `results/**/*.engine`，返回引擎及其同名 ONNX、标签文件、metadata、任务类型、体积、修改时间与是否已转置。
- 句柄缓存改为 LRU，设定上限，超限时销毁最久未用句柄。
- 新增热能图输出：读取 `normalized_map`，用假彩色渲染成热能图返回；同时回传原图，避免前端重复解码。
- 推理结果的每个检测框增加类别名与配色字段。

**前端**

- `conf` / `nms` 改为滑块与数字框双向联动，并提示改动会重建引擎。
- Anomalib 增加两个阈值滑块与"沿用 metadata 默认值"开关，实时显示当前生效阈值。
- 增加标签文件选择与模型下拉，选中模型后自动填充引擎、标签、metadata 与任务类型。
- 结果区接入 `refactor-web-studio-theme` 的查看器：原图、检出图、热能图、异常轮廓四窗口可独立开关、同排显示、各自滚轮缩放与拖动平移。
- 检出框改为前端矢量叠加层重绘，任意倍率下框线与文字保持锐利，标签同时显示类别名与置信度。
- 新增检测明细表、置信度分布、计时分解条，以及结果导出。

## Impact

- 受影响能力：新增 `web-infer-console` 能力。
- 受影响代码：
  - `6-web/backend/app.py`（推理接口参数、模型扫描接口、生命周期）
  - `6-web/backend/infer.py`（阈值透传、类别名、热能图、LRU）
  - `6-web/backend/labels.py`（新增，标签文件解析）
  - `6-web/frontend/index.html`、`6-web/frontend/js/infer.js`（推理页重写）
- 不受影响：`yolo_infer.dll` 与 `anomalib_infer.dll` 的 C API 与实现、两个训练 launcher、`onnx_fallback.py` 的降级路径（仅同步类别名）。
- 依赖：本变更依赖 `refactor-web-studio-theme` 提供的查看器组件与设计令牌。
