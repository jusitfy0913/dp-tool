# Tasks: 批量预标注 + 草稿落盘 + 审核快捷键

## 1. 草稿旁挂落盘（`annotation_drafts.py`）

- [x] 1.1 `<工程名>.drafts.json` 路径推导：`foo.tfanno` → `foo.drafts.json`（同目录）
- [x] 1.2 `load()`：不存在返回空表 + `exists=False`，损坏的条目逐条跳过并记警告
- [x] 1.3 `save()`：原子写（临时文件 + `replace`），带 `version` / `project` / `updated_ns`
- [x] 1.4 `merge(images)`：按图片 id 增量覆盖，其余图片不受影响（批量任务与前端并发写不互相清空）
- [x] 1.5 草稿规范化：`annotation.normalize_object` + `_draft` 标记 + `score`，坏形状丢弃并计数
- [x] 1.6 同一路径的写操作用模块级锁串行化，避免"读-改-写"丢更新

## 2. 批量预标注后台任务（`autolabeljobs.py`）

- [x] 2.1 `autolabel.run_for_image()`：单图执行（原有 `run()` 的薄包装，供任务与接口共用）
- [x] 2.2 任务表：内存字典 + 锁，保留最近 8 个，超过后丢弃最早的已结束任务
- [x] 2.3 `start()`：接收图片清单与参数，起守护线程逐图执行，每张图**原子**写进旁挂文件
- [x] 2.4 `scope="pending"`：跳过已有正式标注或已有草稿的图片，并计入 `skipped`
- [x] 2.5 单张失败不中断整批：记录 `error` 继续下一张，最终汇总 `failed`
- [x] 2.6 `stop()`：置停止事件，当前图片跑完后退出，状态置 `stopped`
- [x] 2.7 `snapshot()`：返回 `status/done/total/skipped/failed/drafts/current/results`
- [x] 2.8 单批图片数上限与空清单校验（空清单直接报错，不建空任务）

## 3. 后端接口

- [x] 3.1 `GET /api/annotation/drafts?path=`：返回旁挂草稿与统计
- [x] 3.2 `POST /api/annotation/drafts`：`images` 增量写、`replace=true` 整份替换
- [x] 3.3 `POST /api/annotation/autolabel/batch`：校验引擎/路径后启动任务，返回 `job_id`
- [x] 3.4 `GET /api/annotation/autolabel/batch/{job_id}`：进度快照
- [x] 3.5 `POST /api/annotation/autolabel/batch/{job_id}/stop`：停止
- [x] 3.6 `GET /api/annotation/autolabel/batch`：最近任务列表（刷新页面后能找回进度）
- [x] 3.7 全部走既有 `_ensure_allowed_path` 白名单，越界一律 403

## 4. 前端：闭环与落盘

- [x] 4.1 `api.js` 新增 `listModels` / 草稿读写 / 批量任务三个接口
- [x] 4.2 预标注参数区加「训练产物」下拉：只列 `kind="yolo"`，选中回填引擎与 `labels`
- [x] 4.3 下拉首次展开时懒加载，另有「刷新」按钮重新扫描
- [x] 4.4 打开工程时读回旁挂草稿；新建工程时清空并覆盖旧旁挂文件
- [x] 4.5 草稿变更按图片标记脏，防抖后**只提交改过的图片**
- [x] 4.6 切换工程 / 离开页面前 flush 草稿写入

## 5. 前端：审核效率与分档着色

- [x] 5.1 `A` 接受当前草稿（未选中取置信度最高的一条），随后自动选中下一条
- [x] 5.2 `D` 丢弃当前草稿（同一套"当前"规则）
- [x] 5.3 `空格` 下一张图片，最后一张给出提示且不循环
- [x] 5.4 快捷键只在标注页激活、且焦点不在输入框时生效
- [x] 5.5 草稿按 `score` 取色：`>=0.70` 绿、`>=0.40` 黄、其余红（画布 + 列表同色）
- [x] 5.6 草稿列表加「高/中/低」徽标，无置信度（`score<0`）按"未知"处理
- [x] 5.7 批量进度条 + 当前图片 + 停止按钮；结束后自动读回草稿并重绘

## 6. 测试与文档

- [x] 6.1 `test_annotation_drafts.py`：往返一致 / 增量合并 / 坏数据跳过 / 原子写
- [x] 6.2 `test_autolabel_batch.py`：任务生命周期、pending 跳过、单张失败不中断、停止
- [x] 6.3 跑 `run_all_tests.py` 全量回归
- [x] 6.4 更新 `dp-tool/readme.md`、`6-web/README.md`、`docs/WORKFLOW_GUIDE.md`
- [x] 6.5 `openspec validate add-batch-autolabel-review --strict` 通过后归档
