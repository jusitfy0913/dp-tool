# Tasks: 异常检测加免掩码的像素级诊断

> 实施记录：本轮只做「甲」方案（免掩码诊断）。掩码（乙）的目录约定已定下，
> 但代码不落地——没有掩码数据时它会是一条永远不执行的死路径。

## 1. 指标函数（纯 numpy，无 GPU / 无 DLL）

- [x] 1.1 `anomalymetrics.pixel_ratio_above()`：算一张归一化异常图上被判异常的像素占比
- [x] 1.2 判定线固定 `0.5`（DLL 已以 `pixel_threshold` 为中心归一化过，见 `anomalib_infer_c.cpp:479`），
      并由 `PIXEL_MAP_CUTOFF` 常量集中表达，避免各处各写一个数
- [x] 1.3 `anomalymetrics.pixel_diagnostics()`：汇总 `normal_false_pixel_rate`、
      `abnormal_marked_rate`、`abnormal_undirected`、`pixel_skipped`、`map_size`
- [x] 1.4 空输入 / 尺寸为 0 / 一维 / 三维一律返回 `None` 与明确 `reason`，不抛异常
- [x] 1.5 「没有样本」与「全是 0」区分：前者 `None`，后者 `0.0`（`_mean_or_none`）

## 2. 从引擎搬异常图

- [x] 2.1 `infer.run_anomalib_score(..., want_map=False)`：开启时把 `normalized_map`
      按 `map_w × map_h` 拷成 numpy `float32`，放在 `anomaly_map`
- [x] 2.2 指针为空或尺寸为 0 时该字段为 `None`，其余字段照常返回
- [x] 2.3 默认关闭，`run_anomalib`（单张推理）行为完全不变

## 3. 评估脚本

- [x] 3.1 `val_anomaly.py` 逐图 `want_map=True`，把占比与尺寸收进 `pixel_samples`
- [x] 3.2 `pixel` 块改为真实计算：`mode="diagnostic"`、`available=true`
- [x] 3.3 异常图缺失 / 尺寸为 0 计入 `pixel_skipped` 并在 `reason` 说明
- [x] 3.4 一张异常图都拿不到时 `available=false` 且给出原因（保持旧契约）
- [x] 3.5 `pixel.auroc` 为 `null` + `auroc_reason` 说明需要掩码，不给伪值
- [x] 3.6 CSV 增加 `pixel_above_ratio` 列，可定位到具体哪张图乱指
- [x] 3.7 终端摘要与 `TRTFORGE_JSON` summary 带上三项诊断

## 4. 界面

- [x] 4.1 `pipeline.js` 新增 `pixelMetricItems()`，卡片增加 3 行
- [x] 4.2 `abnormal_undirected > 0` 时该行加 `is-warn`（红底红字），其余保持常态
- [x] 4.3 `pixel.available=false` 时只显示一行原因，不留空数字
- [x] 4.4 旧报告（无 `pixel` 字段）实测仍只渲染一行原因，不报错

## 5. 验收

- [x] 5.1 新增 `test_anomaly_pixel.py` **36 / 36 通过**（判定线边界 `0.499/0.5/0.501`、
      「0.2 的图不会被当成整幅异常」、占比、无效输入、缺失样本、尺寸混杂）
- [x] 5.2 好图恒为 0 → `normal_false_pixel_rate == 0`、`undirected == 0`
- [x] 5.3 「判 NG 但异常图全 0」→ `abnormal_undirected == 1`
- [x] 5.4 空数组 / 缺样本 / 只有单类时返回 `None` 与原因，不抛异常
- [x] 5.5 `run_all_tests.py` **24 个脚本全绿**
- [x] 5.6 真跑（bottle / Patchcore-fp16 / 正常 4 + 异常 4）：AUROC `1.0`、好图乱指 `0%`、
      定位面积 `16.2%`、指不出位置 `0`、异常图 `224×224`
- [x] 5.7 浏览器实测：3 行正常渲染；把 `undirected` 改成 3 后该行变 `stat is-warn`

## 6. 文档

- [x] 6.1 根 `readme.md` 新增 v1.22 小节（版本对比表 + 判定线说明 + 为什么不算 PRO/AUPRO）
- [x] 6.2 `6-web/README.md` 新增「像素级诊断（v1.22）」+ 报告字段表 + 测试清单一行
- [x] 6.3 `docs/WORKFLOW_GUIDE.md` 新增 §7.4.1（三项指标怎么看、先查什么）
- [x] 6.4 版本号 → `1.22.0`
- [x] 6.5 归档本变更并 `openspec validate --all --strict`
