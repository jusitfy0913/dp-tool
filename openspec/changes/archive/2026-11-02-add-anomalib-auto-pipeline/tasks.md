## 1. 编排层

- [x] 1.1 `pipeline.py`：`StagePlan.resolve` 延迟派生命令（入参整条记录，返回 command / expect_file / report_file）
- [x] 1.2 `pipeline.py`：callable 阶段支持 `{"artifact": ..., "artifacts": {...}}` 多产物记录
- [x] 1.3 `pipeline.py`：resolve 抛异常时该段失败、整链 failed、不启动子进程

## 2. 后端

- [x] 2.1 `anomalymetrics.py`：clamp01 / 归一化 / AUROC（平均秩）/ 混淆矩阵 / 最优 F1 / 直方图 AUROC
- [x] 2.2 `infer.py`：`run_anomalib_score` 轻量打分（不生成四张 PNG）
- [x] 2.3 `tools/val_anomaly.py`：引擎级评估 CLI，输出 JSON + CSV（utf-8-sig）与 `TRTFORGE_JSON` 进度
- [x] 2.4 `app.py`：`POST /api/pipeline/anomalib`、产物定位（按 `vN` 与 mtime 取最新）、
      metadata 校验（阈值缺失 / bool 拒绝）、动态 batch 形状参数、`max_batch` 字段
- [x] 2.5 `app.py`：版本升至 1.18.0

## 3. 前端

- [x] 3.1 `index.html`：`#pipeKind` 类型下拉 + `#pipeAnomalyOptions` 参数区
- [x] 3.2 `pipeline.js`：`syncKind()` / `buildAnomalyPayload()` / 双模式 `onStart()`
- [x] 3.3 `pipeline.js`：异常检测精度卡片（AUROC / F1 / FNR / FPR / P / R / 最优阈值 / 判定不一致告警）

## 4. 测试

- [x] 4.1 `test_pipeline_resolve.py`：26 项（延迟派生 / 失败短路 / 产物去重 / vN 定位 / metadata 校验 / 动态 batch）
- [x] 4.2 `test_anomaly_metrics.py`：31 项（归一化边界 / AUROC 同分 / 空输入 / 混淆矩阵边界 / 直方图）
- [x] 4.3 `run_all_tests.py` 全量回归：20 个脚本全绿
- [x] 4.4 前端：`test_contrast.mjs` + 所有 `js/*.js` 的 `node --check`

## 5. 真链路与文档

- [x] 5.1 bottle 数据集实跑训练（199.7 秒），暴露「动态 batch 被判非法」缺陷
- [x] 5.2 修复动态 batch 后 `skip_train` 重跑 `probe → build → val`，整链 success
- [x] 5.3 精度核对：AUROC 1.000 / F1 1.000 / FPR 0 / FNR 0 / decision_mismatch 0
- [x] 5.4 更新 `README.md`（v1.18 变更记录 + 参数 / 指标 / 验收）
- [x] 5.5 更新 `6-web/README.md`（异常检测一键全流程 + 接口字段 + 文件树 + 测试清单）
- [x] 5.6 更新 `docs/WORKFLOW_GUIDE.md`（§4.2 动态 batch 说明、§5.6 四段链与指标口径）
