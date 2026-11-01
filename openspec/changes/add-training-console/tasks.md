# Tasks: 训练页三块面板

## 1. 数据集概览

- [x] 1.1 新增 `6-web/backend/traininfo.py`：解析 `dataset.yaml`（`path` 相对/绝对、`train/val/test`、`nc`、`names`）
- [x] 1.2 `traininfo.scan()`：逐 split 统计图片/标签/空标签/孤儿标签/缺标签，实例按类计数
- [x] 1.3 分辨率统计：PIL 读文件头，> 2000 张等距抽样并标 `sampled`
- [x] 1.4 警告生成：类别零实例、空标签比例 > 10%、`nc` 与 `names` 不一致、画像集与标签集不匹配
- [x] 1.5 新增 `POST /api/train/dataset-overview`，带 60 秒进程内缓存
- [x] 1.6 前端：训练参数卡新增「数据集概览」折叠块（按钮 + 计数表 + 每类实例条形 + 警告列表）

## 2. 增强预览

- [x] 2.1 新增 `6-web/backend/augpreview.py`：hsv / bgr / fliplr / flipud / degrees / translate+scale+shear+perspective / mosaic / mixup
- [x] 2.2 letterbox 到 `imgsz` + JPEG q=85，`seed` 决定一切随机（`np.random.default_rng`）
- [x] 2.3 新增 `POST /api/train/augment-preview`，从 `train` split 稳定排序取第 `index` 张
- [x] 2.4 前端：增强折叠块内加「生成预览 / 换个种子 / 换一张」与并排两张图 + 本次触发项清单 + 「近似」声明

## 3. 断点续训

- [x] 3.1 `tf_yolo_launcher.py` 新增 `--resume-from <path>`，`mode_train` 从该 ckpt 加载并 `resume=str(path)`
- [x] 3.2 新增 `GET /api/runs/checkpoints`：扫 `runs/**/args.yaml` + `results.csv`，返回已完成轮数/最佳 mAP/last.pt 是否存在
- [x] 3.3 `YoloTrainRequest` 增加 `resume_from`，拼 `--resume-from`，响应回显续训标记
- [x] 3.4 前端：「扫描历史训练」表格 + 「继续训练」回填 + 按钮文案与「参数以 ckpt 为准」提示

## 4. 测试与文档

- [x] 4.1 `tools/test_train_info.py`：临时数据集覆盖空标签/孤儿标签/缺标签/类别零实例/`nc` 不一致/抽样
- [x] 4.2 `tools/test_aug_preview.py`：同 seed 像素一致、各参数开关生效、`copy_paste` 不参与、letterbox 尺寸正确
- [x] 4.3 断点续训：`args.yaml` + `results.csv` 解析单测（含缺 `results.csv`、`data` 指向已删除目录）
- [x] 4.4 回归：`test_engine_progress`、`test_exit_codes`、`test_augment_params`、`test_dataset_workbench`、`test_infer_params`、`test_onnx_fallback`、`test_annotation_project`
- [x] 4.5 浏览器实测：扫描真实 PCB 数据集、生成预览截图核对、真实续训 2 轮
- [x] 4.6 更新 `6-web/README.md`、`docs/WORKFLOW_GUIDE.md`、根 `readme.md`，`APP_VERSION` → 1.14.0
