# Change: YOLO 全流程串联与引擎精度验证

## Why

v1.15 / v1.16 把「训练 → 导出 → 编译 → 推理」的**能力**补齐了，但**操作**还是接力赛：
训练页点开始 → 等完 → 翻到导出折叠块选权重 → 翻到引擎页选 ONNX → 编译完 →
去推理页手填标签 → 最后靠肉眼判断「精度还行」。

三个具体问题：

- **每一步都要人守着换页**。跑一次完整流程要盯五个地方，中途接个电话就可能漏掉一步，
  半天后才发现引擎是用上次的 ONNX 编译的。
- **精度从未量化**。全项目没有一处计算 mAP / P / R，验收口径是「打开几张图看看框准不准」，
  既说不清好坏，也无法判断换了精度（fp16 → fp32）到底损失多少。
- **服务重启后记录假死**。编排记录只存在内存，重启后页面还对着一条永不变化的记录轮询，
  点「停止」只能拿到 409。

## What Changes

- **新增串联编排 `backend/pipeline.py`**：把「训练 → 导出 `.transd.onnx` → 校验转置布局 →
  编译引擎 → 补 `labels_*.json` → 引擎精度验证」六段串成一条链。每段仍是独立的 run，
  串联只决定「下一段跑什么」，因此停止只停当前那一个、每段日志独立、失败不影响已完成产物。
- **新增引擎精度验证**：`val_engine.py` 走 `yolo_infer.dll` 逐张推理，
  `valmetrics.py` 按 COCO 口径（IoU 0.50:0.95、全点插值）算 AP，产出 `val_report.json` / `.csv`。
- **接口**：`POST /api/pipeline/yolo`、`GET /api/pipeline`、`GET /api/pipeline/{id}`、
  `POST /api/pipeline/{id}/stop`、`GET /api/pipeline/{id}/log?stage=&lines=`
  （读 `runtime/logs/{run_id}.log` 尾部；run 已不在内存时排障的唯一入口）。
- **前端**：训练页底部新增占满整行的「全流程串联」卡片——阶段清单（含每段耗时与日志入口）、
  产物表（带「去推理引擎」/「带到推理页」/「复制路径」）、精度汇总卡、历史记录表。
  阶段「日志」按钮双路：还在跑就切右侧实时流，已结束/重启过就读日志文件尾巴。
- **持久化与重启清理**：记录原子写入 `runtime/pipelines/*.json`；启动时把遗留的
  `running` 记录标成 `cancelled`（原因「服务重启，任务已中断」），历史仍可查看。
- **静态资源禁缓存**：`/static/` 响应带 `Cache-Control: no-cache, must-revalidate`，
  避免升级后浏览器仍跑旧 JS（模块图缓存导致「导出缺 `focusRun`」这类假错）。
- 版本升至 `1.17.0`。

## Impact

- Affected specs: 新增 `web-yolo-pipeline`
- Affected code:
  - 新增 `6-web/backend/pipeline.py`、`6-web/backend/valmetrics.py`、
    `6-web/backend/tools/val_engine.py`
  - 修改 `6-web/backend/app.py`（5 个接口 + 静态资源中间件 + 版本号）、
    `6-web/backend/runners.py`（`expect_file` 容忍字符串，避免读线程崩溃）
  - 前端：`index.html`（`#pipeCard` + `#pipeLog`）、`js/pipeline.js`、`js/train.js`（导出 `focusRun`）、
    `js/main.js`、`css/style.css`
  - 测试：`test_pipeline_stages.py`（50 项）、`test_pipeline_log.py`（9 项）、
    `test_val_metrics.py`（37 项）

## Non-Goals

- 不做 Anomalib 的串联：异常检测的「精度」是图像级 OK/NG + 像素级阈值，口径与检测的 mAP
  完全不同，需要单独设计，这一版只做 YOLO。
- 不支持 `segment` / `pose` / `obb` 的 mAP：GT 解码口径不同（多边形 IoU、关键点 OKS、
  旋转框角度），硬套检测口径会给出误导性的数字，宁可先不算。
- 不自动重试失败阶段：OOM、数据配错重试只是再烧一遍 GPU 时间，还会把原因埋进第二次日志。
- 不做断点续跑串联：失败后从已有产物手工接着跑（产物都保留着），比维护一套「接着上次跑」的状态机简单可靠。
- 不在浏览器里画 P-R 曲线：这一版先给数字，曲线等有真实验收需求再加。

## 验收

- `run_all_tests.py` 18 个脚本全绿（新增 50 + 9 + 37 断言）。
- 真链路：跳过训练、用 PCB `pcb_best.pt` 跑完整链，`export → probe → build → labels → val`
  五段全 `success`（实测 fp32 1 分 24 秒、fp16 6 分 53 秒），产物 `best.transd.onnx` /
  `pcb_best_fp32.engine` / `labels_pipe_smoke2.json` / `val/val_report.json` 全部落盘，
  页面精度卡片显示 mAP50 0.125 / mAP50-95 0.090 / P 0.500 / R 0.250。
- 历史阶段排障：重启后点「日志」仍能读出该段 `runtime/logs/*.log`（实测 210 行 trtexec 输出），
  接口对 `../` 路径穿越与未知阶段一律 404。
- 中途失败：后续段 `skipped`、整链 `failed`、已完成段产物保留（单测覆盖）。
- 停止：当前段 `cancelled`、后续 `skipped`、真的调用了 `run_manager.stop`（单测覆盖）。
- 重启：遗留 `running` 记录被标成 `cancelled`，「停止」按钮置灰、轮询停止（单测 + 浏览器实测）。

