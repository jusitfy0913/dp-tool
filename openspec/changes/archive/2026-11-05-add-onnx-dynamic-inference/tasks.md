## 1. 规格与提案
- [x] 1.1 创建 `proposal.md`、`design.md`、`tasks.md` 三个基础文件
- [x] 1.2 新增 `specs/onnx-inference-dll/spec.md`（新 capability 的 ADDED 需求）
- [x] 1.3 新增 `specs/web-infer-console/spec.md` 的 ONNX 直推增量需求
- [x] 1.4 新增 `specs/web-infer-batch/spec.md` 的批量/视频/摄像头后端一致性需求
- [x] 1.5 补全 `openspec/project.md` 的项目上下文、技术栈、测试策略与约束
- [x] 1.6 `openspec validate add-onnx-dynamic-inference --strict` 通过后才开始编码

## 2. ONNX Runtime 子工程骨架
- [x] 2.1 `scripts/fetch_onnxruntime.ps1`：拉取固定版本（支持 `-CpuOnly`），落到 `third_party/onnxruntime`，写版本文件
- [x] 2.2 `3-cpp-tensorrt/onnx_infer/CMakeLists.txt`：`ONNXRUNTIME_ROOT` 未配置时 warning 并跳过，不阻塞其他目标
- [x] 2.3 `include/onnx_infer_c.h`：`oi_*` C ABI，结构体与 `yi_box` / `ai_result` / `ai_frame_stat` 二进制对齐
- [x] 2.4 `src/ort_session.{hpp,cpp}`：`LoadLibraryExW` + `OrtGetApiBase` 显式加载，会话与张量封装
- [x] 2.5 EP 探测与回退：先探 `onnxruntime_providers_cuda.dll`，失败或建会话异常时退回 CPU
- [x] 2.6 `src/onnx_infer_c.cpp`：句柄生命周期、`thread_local` 错误串、异常不跨 C 边界
- [x] 2.7 POST_BUILD 复制 `onnxruntime.dll` 与 provider DLL 到 `4-qt-studio/bin`

## 3. YOLO ONNX 推理
- [x] 3.1 letterbox 预处理（padding 114、RGB、`/255`、NCHW）与坐标还原所需的缩放信息
- [x] 3.2 输出布局自适应：`[1, 4+nc, n]` 与 `[1, n, 4+nc]` 都识别，失败时报实际形状
- [x] 3.3 `detect`：cxcywh→xyxy、置信度过滤、逐类 NMS
- [x] 3.4 `obb`：角度通道解析、RotatedRect NMS、外接 AABB 回填
- [x] 3.5 `segment`：proto 与系数点积 + sigmoid + 阈值 0.5，裁剪后缩放回原图
- [x] 3.6 `pose`：关键点按 d2i 还原，关键点数从通道数推导而非写死 17
- [x] 3.7 任务与模型不匹配时在 create 阶段报出可读错误，指明该切哪个任务

## 4. Anomalib ONNX 推理
- [x] 4.1 从 `anomalib_infer_c.cpp` 抽出 `anomalib_post.hpp`（归一化、叠加、轮廓、状态文字），既有 DLL 改为 include
- [x] 4.2 `metadata.json` 读取与阈值覆盖语义（`>=0` 覆盖，`<0` 沿用）与现有实现一致
- [x] 4.3 预处理：resize 到 ONNX 输入尺寸，动态维度回退 `input_size` / 256，imagenet 归一化可选
- [x] 4.4 输出识别：按名字匹配 score / anomaly_map，再按元素数兜底，兼容 `[1]` 与 `[1,1]`
- [x] 4.5 产出 source / overlay / mask 三张 RGB 与 `normalized_map`，判定与 `ai_*` 同口径

## 5. 一致性校验工具
- [x] 5.1 `tools/compare_backends.py`：同图同参数分别走 `.engine` 与 `.onnx`，输出 JSON + CSV（`utf-8-sig`）
- [x] 5.2 指标实现：框匹配率与最小 IoU、置信度差、mask IoU、关键点距离、OBB 角度差、分数差、判定一致、map 差值
- [x] 5.3 阈值参数化 + 退出码约定（0 通过 / 2 超阈值 / 3 无法比对）
- [x] 5.4 `TRTFORGE_RAW_DUMP` 触发原始张量导出（未设置时零行为差异），落盘 manifest
- [x] 5.5 `tools/compare_raw_tensors.py`：自动对齐转置，输出每个张量的差值报告
- [x] 5.6 `tools/test_backend_consistency.py`：把 Tier 1 包成回归用例，缺样例时明确失败而非跳过（19 项，含实测分派）

## 6. Python 接入
- [x] 6.1 `infer.py` 增加 `_onnx_lib` 与 `_bind_onnx()`，复用既有三个结构体
- [x] 6.2 `run_yolo()` / `run_anomalib()` 按后缀分派，调用点签名不变
- [x] 6.3 `BackendStatus` 增加 `onnx_ok` / `onnx_detail`，新增 `onnx_available()`
- [x] 6.4 句柄缓存键加入后端类型，避免 `.engine` 与 `.onnx` 互相命中
- [x] 6.5 `app.py` 三级后端选择（ONNX DLL → 引擎 DLL → Python 兜底），错误文案更新
- [x] 6.6 `/api/capabilities` 与 `config.describe()` 暴露 ONNX DLL 与 ORT provider
- [x] 6.7 `tools/infer_batch.py` 与 `tools/infer_video.py` 同步三级分派与 `backend` 字段

## 7. 前端
- [x] 7.1 两个模型路径输入支持浏览并接受 `.onnx`
- [x] 7.2 模型下拉新增「ONNX 直推」条目，选中填充 `.onnx`（实现为 `#inferWeight` 权重格式下拉，与模型下拉联动）
- [x] 7.3 未转置警告只在选择引擎时给出（ONNX 直推两种布局都能解）
- [x] 7.4 结果区后端徽标如实显示 `onnx_infer.dll` / `onnxruntime-fallback`

## 8. 构建、打包与验收
- [x] 8.1 `scripts/build_infer_dlls.ps1` 增加 `onnx_infer` 目标与 `-OnnxRuntimeRoot` 透传
- [x] 8.2 `4-qt-studio/CMakeLists.txt` 用 `TRTFORGE_BUILD_ONNX_INFER` 控制是否纳入解决方案，默认关闭
- [x] 8.3 `packaging/package_web_user.ps1` 增加 `onnx_infer.dll` 与 ORT 运行库，清单自检（`bin_manifest.txt`）
- [x] 8.4 `tools/test_onnx_infer_dll.py`：C API 错误路径、四任务、三模型、provider 与输入尺寸（36 项全通过）
- [x] 8.5 跑通 Tier 1 一致性门禁：YOLO detect（8/8，余弦 0.999998）与 Anomalib 三模型（8/8，余弦 0.9917）达标；
      seg / pose / obb 本机无对应引擎产物，改由 `test_onnx_infer_dll.py` 的四任务用例覆盖（含掩码 / 关键点 / 旋转框字段校验）
- [x] 8.6 性能对比表：ONNX CPU 三段耗时已实测并写入 `onnx_infer/README.md`、`6-web/README.md`、`docs/WORKFLOW_GUIDE.md`
- [x] 8.7 现有 `6-web/backend/tools/` 自测脚本全量回归（38 个脚本全绿），前端 js `node --check` 全通过
- [x] 8.8 端到端：单图（HTTP `/api/infer/yolo`、`/api/infer/anomalib`）、批量（`infer_batch.py`）、视频（`infer_video.py`）
      在 `.onnx` 下各跑一遍，`backend` 字段一致为 `onnx_infer.dll`

## 9. 文档与归档
- [x] 9.1 新增 `3-cpp-tensorrt/onnx_infer/README.md`（C API 参数、调用示例、应用场景、新旧差异、一致性校验、排错）
- [x] 9.2 更新 `docs/WORKFLOW_GUIDE.md`（§6.0 编译、EP、§7.6 一致性校验、排错表）
- [x] 9.3 更新 `docs/TRTForge_Web操作手册.md` 推理页章节（权重格式下拉、ONNX 场景、排错三行）
- [x] 9.4 更新根 `readme.md` 与 `6-web/README.md`，`APP_VERSION` 1.25.3 → 1.26.0
- [x] 9.5 `openspec archive add-onnx-dynamic-inference` 并同步 `openspec/specs/`（已归档为 `2026-11-05-add-onnx-dynamic-inference`，`openspec/specs/` 已合并）
