# Project Context

## Purpose

DefectVision AI Studio（AI 缺陷检测视觉工作站）围绕工业视觉的深度学习闭环建设：
把数据标注、数据集导出、YOLO / Anomalib 训练、ONNX 导出、TensorRT 引擎构建与 C++ 推理验证
串成一条可交付的产线链路。当前日常入口是浏览器版 `6-web`，Qt 桌面版 `4-qt-studio` 保留作迁移参考。

## Tech Stack

| 层 | 技术 |
| --- | --- |
| Web 后端 | Python 3.10+、FastAPI、uvicorn、ctypes、numpy、OpenCV |
| Web 前端 | 原生 ES Module JavaScript、零构建、单页多面板 |
| 训练 | Ultralytics YOLO、Anomalib（Lightning）、PyTorch |
| 推理运行时 | CUDA 11.8、TensorRT 8.5、cuDNN 8、OpenCV 4.12、ONNX Runtime |
| 原生扩展 | C++17、CMake 3.21+、MSVC 2022 x64、CUDA C++ |
| 旧桌面端 | Qt 6 Widgets（`4-qt-studio`） |
| OCR | Paddle Inference（`ocr/ocr_infer`） |

## Project Conventions

### Code Style

- 文档、注释、提交信息一律使用**中文**；代码标识符使用英文。
- 所有脚本与文本文件使用 **UTF-8**，禁止 GBK，避免跨平台乱码。
- Python 使用 `from __future__ import annotations`、类型注解、`pathlib`，模块顶部写「为什么这样设计」的块注释。
- C++ 使用 C++17，导出 C ABI 的接口必须放在 `extern "C"` 中，异常不得跨越 C 边界。
- 前端不使用打包器，保持「打开即用」的零构建形态。

### Architecture Patterns

五层目录 + 一个 Web 端，依赖方向单向：

```
1-python-yolo / 2-python-anomaly   （训练与导出）
        ↓
3-cpp-tensorrt                     （C++ 推理动态库 + trtexec）
        ↓
4-qt-studio                        （Qt 界面，已被 6-web 取代）
6-web                              （浏览器版工作站，当前主线）
5-reference                        （早期对照工程，不参与构建）
ocr/                               （PaddleOCR 推理，独立子工程）
```

- 推理一律封装为**动态库 + C ABI**，界面与 Python 只做适配，不直接持有 TensorRT / CUDA 资源。
- 产物（`.engine`、`.onnx`、`labels_*.json`、`metadata.json`）统一由训练链路产出，推理端只读。
- 所有编译产物输出到 `4-qt-studio/bin`，打包脚本从该目录取运行库。

### Testing Strategy

- 后端自测脚本集中在 `6-web/backend/tools/test_*.py`，使用 `python tools/xxx.py` 直接运行，
  不依赖 pytest 也能跑；每个脚本自觉打印通过数。
- 前端改动必须通过 `node --check` 语法检查（`js/**/*.js`，含 `js/annotate/` 子目录）。
- **纯逻辑必须抽模块并有 `.mjs` 单测**：凡是不碰 DOM、不读全局 `state` 的判定
  （阈值分档、撤销栈、几何命中、导入计划、比例换算……）都放到独立 ES 模块，
  并在 `6-web/frontend/tests/test_<模块>.mjs` 里覆盖边界值。留在页面脚本里的判定
  等于永远不测——那正是标注页 4700 行时的状态。
- 跨模块契约用 `tools/test_full_chain.py`（真实 HTTP、假引擎桩、8 个环节）守住，
  接口之间字段改名这类断链由它拦；页面是否真能加载用 `tools/e2e_browser.py`（可选，CI 不跑）。
- 精度与等价性用专门的对比脚本产出 JSON + CSV 报告，不靠肉眼判断：
  `tools/val_engine.py`、`tools/val_anomaly.py`、`2-python-anomaly/anomalib-main/compare_ckpt_onnx_inference.py`。
- 新增推理后端必须附带「与既有后端的一致性对比」用例，缺样例时明确失败而不是跳过。

### Git Workflow

- 变更提案走 OpenSpec：先 `changes/<change-id>/`，实施完成后归档到 `changes/archive/YYYY-MM-DD-<id>/`。
- 提交信息使用中文，一行说清「改了什么、为什么」。
- 不提交模型权重、构建中间产物与大体积运行库。

## Domain Context

- 目标场景是 PCB / 工业零件缺陷检测：小目标多、背景纹理复杂、漏检与误报的代价不对称。
- YOLO 侧有四项任务：`detect`、`segment`、`pose`、`obb`；Anomalib 侧有 PatchCore、PaDiM、EfficientAd。
- YOLO 引擎**必须**用 `.transd.onnx`（转置输出）编译，未转置的 `best.onnx` 编出的引擎解码会越界。
- Anomalib 的阈值来自 `metadata.json`，可被运行时覆盖；判定统一按归一化分数的 0.5 边界。
- 用户可能不懂代码，界面文案要说清「为什么」与「怎么修」，错误信息必须可执行。

## Important Constraints

- CUDA 固定 11.8、TensorRT 8.5、cuDNN 8；推理与训练环境保持同一版本，避免 EP 与引擎不匹配。
- 不得修改仓库根目录下的 `bin`、`include`、`lib`。
- 中文与非 ASCII 路径必须可用（模型、图片、工程文件都可能放在中文目录）。
- Windows 10 + Python 3.8+ 的 DLL 搜索规则：依赖目录必须 `os.add_dll_directory` 显式加入。
- 对外交付可以只给运行库与可执行文件，不暴露 Python/C++ 源码。

## External Dependencies

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| Ultralytics | YOLO 训练与 ONNX 导出 | 通过 `1-python-yolo/launcher` 调用 |
| Anomalib | 异常检测训练与导出 | 通过 `2-python-anomaly/launcher` 调用 |
| TensorRT / cuDNN / CUDA | 引擎推理 | 版本必须与编译期一致 |
| OpenCV | 图像预处理与渲染 | 4.12，动态库随包分发 |
| ONNX Runtime | ONNX 推理 | 需与 CUDA 11.8 / cuDNN 8 匹配 |
| Paddle Inference | OCR | 独立子工程，版本由 `ocr_infer` 锁定 |
| trtexec | ONNX → 引擎编译 | 封装在 `3-cpp-tensorrt/trtexec` |
