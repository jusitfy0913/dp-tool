# Change: 新增 Web 标注工作台与自动标注

## Why

`6-web` 目前**完全没有标注能力**，这是它相对 Qt 版最致命的功能缺口。

实测确认的事实：

- 全前端 9 个文件里没有任何 `<canvas>`、SVG 绘图或鼠标绘制逻辑（`index.html:409` 只加载 `main.js`）。
- 后端无任何标注写入接口；`.tfanno` 只能读（`dataset.py:393` 的 `load_tfanno`），全仓没有 `save_tfanno`。
- 已归档为 Complete 的 `add-annotation-dataset-workbench` 其标注画布**全部落在 Qt**（`4-qt-studio/src/annotation/AnnotationCanvas.cpp`，640 行），`6-web` 只承接了其中的「扫描 → 划分 → 导出」。
- 结果是：Web 版现在只能消费别人标好的数据，不能生产标注；想标注就得回去开 Qt 版。

同时，Qt 版标注本身也只是一个"能标出结果"的最小可用集。逐项 grep `4-qt-studio/src/annotation/*.cpp`，以下能力命中数**全部为 0**：键盘快捷键、重做、复制粘贴、顶点/边拖动编辑、多选、对象显隐、填充透明度、图上显示分数、难例标记、对象级 flags/attributes、自动保存、导航缩略图、十字线、吸附辅助线。导出格式也只有 3 种。

因此本变更的目标不是"把 Qt 标注搬到 Web"，而是**在保持 `.tfanno` 双向互通的前提下，补齐 Qt 缺失的编辑手感与显示控制**。

## What Changes

**数据格式**

- 新增 Python 侧的 `.tfanno` 读写，根结构与标注对象字段严格对齐 `AnnotationProject.cpp:123-151` 与 `:277-291`。
- 矩形统一存 2 点（左上、右下），与 Qt 一致；写入时把 4 点折叠为 2 点。
- 关键点保持 Qt 语义：一个对象持有全部关键点与可见性数组，不采用"每点一个对象"的表达。
- 在 Qt 字段基础上**扩展三个字段**：`difficult`、`flags`、`attributes`。已确认 Qt 读取端只读已知键、未知键静默忽略，因此扩展后的工程文件 Qt 仍可正常打开。

**后端**

- 新增标注工程的读取、保存（带版本号乐观锁）、图片流、自动标注、导出五类接口。
- 图片流受现有路径白名单约束。

**前端**

- 新增「标注」页签与独立脚本，实现 Canvas 2D 标注画布。
- 支持六种形状：矩形、旋转矩形、多边形、线、点、关键点组。
- 支持缩放平移（复用查看器的光标锚点变换）、撤销与重做、最小快捷键集、防抖自动保存。
- 补齐 Qt 缺失的五项能力：顶点/边拖动编辑、重做、对象显隐与填充透明度与图上显示分数、导航缩略图与十字线与吸附辅助线、光标锚点缩放。

**自动标注**

- 主通道：调用自有 `yolo_infer.dll`，把检测结果转为标注草稿。
- 次通道：通过 HTTP 调用独立进程形式的外部标注服务，用于点/框提示分割与文字提示检测。
- 外部服务不随本产品分发，用户自行安装；不可达时明确提示并回退主通道。
- 预标注结果以草稿层呈现，支持逐个接受、拒绝、全部接受。

## Impact

- 受影响能力：新增 `web-annotation-workbench` 能力。
- 受影响代码：
  - `6-web/backend/annotation.py`（新增，`.tfanno` 读写）
  - `6-web/backend/autolabel.py`（新增，自动标注编排）
  - `6-web/backend/app.py`（新增标注相关路由）
  - `6-web/frontend/index.html`、`6-web/frontend/js/annotate.js`（新增页签与画布）
- 不受影响：Qt 版标注实现、`dataset.py` 的扫描与导出逻辑（仅复用）、两个推理 DLL 的 C API。
- 依赖：本变更依赖 `refactor-web-studio-theme` 的设计令牌与查看器变换逻辑，依赖 `update-web-infer-console` 的模型枚举能力。
- 合规约束：外部标注服务为 GPL-3.0 授权，**不得**以 import 或打包方式并入本产品，只能通过 HTTP 独立进程调用。
