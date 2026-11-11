## 1. 共享实现

- [x] 1.1 新增 `6-web/backend/vram.py`：GPU 探测（torch → nvidia-smi 兜底）、实测表、估算、预检、记录
- [x] 1.2 新增 `6-web/backend/ampcheck.py`：AMP 黑名单判定与精度预测
- [x] 1.3 `config.py` 增加 `VRAM_SAMPLES`（可用 `TRTFORGE_VRAM_SAMPLES` 覆盖）

## 2. launcher 观测

- [x] 2.1 `on_train_start` 上报 `amp_state`（请求值 / 实际值 / GPU 名）
- [x] 2.2 `on_train_end` 上报 `vram_peak`（reserved / allocated / 样本键）

## 3. autoexp

- [x] 3.1 `preflight_vram()` 委托给 `vram.preflight`（签名与返回结构不变）
- [x] 3.2 训练成功后读取 `trainer.amp` 并按实际精度记录样本
- [x] 3.3 `results.tsv` 新增 `amp` 列，`exp_NNNN.json` 记 `amp_effective`

## 4. Web 后端

- [x] 4.1 `GET /api/train/preflight?imgsz=&batch=&amp=`
      （精度用 `amp` 三态表达，不引入单机无意义的 `device`）
- [x] 4.2 `POST /api/train/yolo` 启动前显存闸门（409 + 建议），支持 `allow_low_vram`
- [x] 4.3 `YoloTrainRequest.amp` 三态 → `--amp` / `--no-amp`
- [x] 4.4 `runners` 收到 `vram_peak` 时记录样本，收到 `amp_state` 时留痕

## 5. 前端

- [x] 5.1 训练页「算力体检」块：GPU / 可用与需求显存 / 依据 / AMP 状态
- [x] 5.2 AMP 三态选择与显式警告文案
- [x] 5.3 409 时给「仍然开始」入口
- [x] 5.4 训练结束在日志里写实际精度与峰值显存

## 6. 验证

- [x] 6.1 新增 `tools/test_vram_amp.py`（118 项）
- [x] 6.2 `7-autoresearch-yolo/run_all_tests.py` 补 vram 委托与 tsv 加列用例（62/62）
- [x] 6.3 回归 `test_run_history.py`、`test_autoexp_api.py`、`test_export_request.py`、`test_finetune_request.py`
- [x] 6.4 真机：`640/batch=8` 1 epoch 得 `amp_requested=true / amp_effective=false / precision=fp32`
      与 `reserved 2132.0 MB`，样本自动写入 `640x8xfp32`；`640/batch=16` 按外推判 5304 MB 放行；
      `batch=64` 触发 409 且页面出现覆盖按钮、服务端未创建任务
      （`960/batch=4` 一致性验证随批次 2.5 的 holdout 一起做，本机当前没有 960 样本）
- [x] 6.5 更新 `6-web/README.md`、`7-autoresearch-yolo/README.md`、`docs/WORKFLOW_GUIDE.md`、`readme.md`
