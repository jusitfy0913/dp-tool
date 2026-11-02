# Change: 训练页补四块面板（数据集概览 / 增强预览 / 断点续训 / 导出 ONNX）

## Why

v1.13 之后训练页能看见「进度、剩余时间、每轮指标、曲线」了，但开训**之前**和**中断之后**仍然是盲区：

- 填了 `dataset.yaml` 却不知道里面有多少张图、每类多少个框、有没有空标签和孤儿标签。数据集有问题时，唯一的发现途径是训练跑到一半看 loss 不降。
- 15 个数据增强参数全靠想象。`mosaic=1.0` 对小缺陷到底有害到什么程度，用户只能训完一轮才知道，一次试错成本是几分钟到几十分钟。
- 训练中断（关机、显存不足、主动停止）后想接着训，界面上只有 `--resume` 这个开关，而它依赖 Ultralytics 的 `get_latest_run()` 去猜「最近一次 run」，在 Web 场景下 cwd 是 launcher 目录，猜错就等于静默换了数据集。

三件事的共同点：**都是「看一眼就能省下一次训练」的信息**，实现成本远低于重训一次。

## What Changes

- **数据集概览**：新增 `POST /api/train/dataset-overview`，解析 `dataset.yaml` 后按 `train/val/test` 统计图片数、标签数、空标签、孤儿标签、每类实例数、分辨率分布，返回警告清单（类别未出现、空标签比例过高、`nc` 与实际类别数不一致、图片集与标签集不匹配）。
- **增强预览**：新增 `POST /api/train/augment-preview`，用当前 15 项参数（其中 11 项参与像素变换）对训练集真实样本做一次增强，返回「原图 / 增强后」两张 JPEG（base64）+ 本次抽样实际触发了哪些项。相同 `seed` 必然得到相同像素，便于对比不同参数。
- **断点续训**：新增 `GET /api/runs/checkpoints`，扫 `runs/**/args.yaml` + `results.csv` 而不加载 `.pt`（后端解释器没有 torch），返回每次训练的已完成轮数、最佳 mAP、`last.pt` 路径与大小；前端一键回填参数并走新的 `--resume-from <last.pt>`。
- **修正 resume 语义**：`tf_yolo_launcher.py` 新增 `--resume-from <path>`，把显式路径传给 `model.train(resume=path)`，不再依赖 `get_latest_run()` 猜；旧的 `--resume` 开关保留兼容。
- **导出 ONNX（v1.15 追加）**：新增 `POST /api/export/yolo`，把训练好的 `.pt` 导出为 ONNX 并可选生成 `.transd.onnx`，日志与产物复用训练那套子进程 + WebSocket。页面上的「训练导出」终于名副其实：训练 → 导出 → 编译 → 推理全程不用命令行。
- **修正 `.transd.onnx` 的生成逻辑（v1.15 追加）**：不再假设「图里最后一个节点就是检测头」，改为按形状识别预测头；转置前按任务校验通道数；`segment` 额外把 proto 输出提到第一位——C++ 侧对 V8Seg 硬性要求 binding1=proto，而 Ultralytics 导出的顺序正好相反，不重排就是「编译能过、推理报错」。
- 版本号 `APP_VERSION` 1.13.0 → **1.14.0** → **1.15.0**。

## Impact

- Affected specs: `web-training-console`（新建能力）
- Affected code:
  - 新增 `6-web/backend/traininfo.py`（数据集统计）、`6-web/backend/augpreview.py`（增强预览）
  - 修改 `6-web/backend/app.py`（三个接口 + `resume_from` 字段）、`6-web/backend/config.py`（如果需要新目录常量）
  - 修改 `1-python-yolo/launcher/tf_yolo_launcher.py`（`--resume-from`）
  - 修改 `6-web/frontend/index.html`、`js/train.js`、`css/style.css`
  - 新增 `6-web/backend/tools/test_train_info.py`、`test_aug_preview.py`
  - **v1.15 追加**：修改 `1-python-yolo/launcher/tf_yolo_launcher.py`（`make_transd_onnx` 按形状识别预测头 + 任务通道校验 + seg 输出重排；`mode_export` 以权重自带任务为准）
  - **v1.15 追加**：修改 `6-web/backend/app.py`（`POST /api/export/yolo` + 抽出的 `build_export_command`）、新增 `6-web/frontend/js/export.js`、修改 `js/main.js`（browse 支持 `.pt`）、`index.html`
  - **v1.15 追加**：新增 `6-web/backend/tools/test_transd_onnx.py`、`test_export_request.py`
  - 文档：`6-web/README.md`、`docs/WORKFLOW_GUIDE.md`、根 `readme.md`

## Non-Goals

- 不改训练算法、不改 Ultralytics 源码、不动 `bin/include/lib`。
- 不做「自动调参 / 自动推荐增强」，预览只回答「现在这套参数长什么样」。
- 不做训练中的批次级增强可视化（每轮几千张图，传输成本不划算）。
- 不解析 `.pt` 二进制（后端无 torch）；续训只依赖 `args.yaml` + `results.csv`。
- 不做任务队列、不做远程训练机调度。

## Approval Gate

三处技术取舍需要用户先确认（详见 `design.md` 的「Open Questions」）：
1. 增强预览做成「cv2 像素级近似」还是「在训练解释器里跑真实 pipeline」；
2. 数据集概览触发方式：手动按钮还是选完文件自动扫；
3. 是否允许本次一并改 launcher 的 `--resume-from`（跨模块改动）。
