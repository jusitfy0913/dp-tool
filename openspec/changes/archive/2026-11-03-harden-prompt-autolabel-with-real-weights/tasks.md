# 实施任务

## 1. 定位并修复真机缺陷

- [x] 1.1 下载四个内置权重并核对 `Content-Length`（26.85 / 15.74 / 171.28 / 0.44 MiB，合计 214.3 MiB）
- [x] 1.2 按实际字节校准 `modelzoo.CATALOG` 的 `size`（原 40.1 MB / 700 MB 会导致进度条卡在 24%）
- [x] 1.3 探活并替换全部镜像回退地址（原 `hf-mirror.com/CVHub520/...` 三条实测 404）
- [x] 1.4 修 `build_prompt()` 对 numpy 数组的判断（`points or []` → `_point_list()`）
- [x] 1.5 修 encoder 输入布局（按签名判断 HWC / NHWC / NCHW）
- [x] 1.6 新增 `onnxenv.py`，先探 provider DLL 再交给 onnxruntime
- [x] 1.7 文本通道预处理改为 BGR→RGB，对齐参考实现

## 2. 补回归测试

- [x] 2.1 `test_prompt_label.py` 增加 numpy 提示回归（49→63 项）
- [x] 2.2 `test_prompt_label.py` 增加 encoder 布局三种分支
- [x] 2.3 新增 `test_onnxenv.py`（11 项）
- [x] 2.4 全量回归 30 个脚本全绿

## 3. 验证

- [x] 3.1 真图跑 SAM 点提示（0.76 s）、缓存命中（0.077 s）、框提示（0.079 s）
- [x] 3.2 真图跑 GroundingDINO（加载 4.80 s、单图 4.42 s）
- [x] 3.3 用 `bus.jpg` + `bus` 做管线自检（0.91 分紧致框），区分"模型能力"与"代码缺陷"
- [x] 3.4 HTTP 端到端：点/框/文本/细化 全 200，空提示 400
- [x] 3.5 前端 3 个 `.mjs` 与 13 个 `js/*.js` 语法检查

## 4. 文档

- [x] 4.1 `readme.md`、`docs/WORKFLOW_GUIDE.md`、`6-web/README.md` 同步 v1.24.1 差异表与实测数字
- [x] 4.2 记录"不能改 letterbox"这条限制及其实测依据
