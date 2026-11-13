## 1. 契约与核心模块

- [x] 1.1 新增 `6-web/backend/bundle.py`：常量（`SCHEMA = "trtforge.bundle.v1"`、结论档位）、
      `manifest_path()`、`read_manifest()`、`enumerate_bundles()`
- [x] 1.2 实现 `build_bundle(engine_or_onnx, labels, metadata, out_dir, bundle_id, overwrite)`
      —— 采集运行时、算 sha256、硬链接或复制、写 manifest 与 `README.txt`
- [x] 1.3 实现 `verify_bundle(path, kind=None, task=None)` 返回
      `{"level": "ok|warn|block", "checks": [{"name","level","detail","fix"}]}`
- [x] 1.4 实现兼容性判定：schema、路径越界、哈希、必需文件、kind/task、
      trt 主/小版本、compute capability、imgsz、标签数

## 2. 运行时字段采集

- [x] 2.1 `enginemeta.write_meta` 增加 `runtime` 区块：`trt_version`、`cuda_version`、
      `gpu_name`、`compute_cap`、`imgsz`、`dynamic`、`onnx_layout`
- [x] 2.2 TensorRT 版本读取走 `nvinfer.dll` 文件版本（`version.dll` API，毫秒级），取不到留空
- [x] 2.3 GPU 信息读取复用现有 nvidia-smi 调用路径（`name,compute_cap`，老驱动降级到只取 name），取不到留空

## 3. 接口

- [x] 3.1 `GET /api/bundles`：枚举，返回 id、路径、manifest 摘要、轻量校验结论
- [x] 3.2 `POST /api/bundles/build`：造包，返回 manifest 与新包路径
- [x] 3.3 `POST /api/bundles/verify`：按可选 kind/task 完整校验（含 sha256）
- [x] 3.4 `GET /api/bundles/{bundle_id}/manifest`：原文返回
- [x] 3.5 `/api/models` 增加 `bundles` 段；散装条目带 `origin="loose"` / `bundle=null`，
      模型包条目带 `origin="bundle"`
- [x] 3.6 推理接口支持 `bundle`：与 `engine` 互斥，使用包内标签与 metadata
      （`infer_yolo` / `infer_anomalib` 的 `engine`、`metadata` 由必填改为可选）
- [x] 3.7 引擎构建成功后自动造包（`auto_bundle` 默认开），失败只记日志不影响构建结果

## 4. 前端

- [x] 4.1 `js/infer.js`：模型下拉分「模型包（推荐）」与「训练产物（散装）」
- [x] 4.2 选中模型包后填充引擎、标签、metadata 并标为随包只读（绿色左边条）
- [x] 4.3 显示校验结论与修复建议；`block` 时提交前拦下并说明
- [x] 4.4 手工修改路径（`input` 事件）后自动退回散装提交方式
- [x] 4.5 引擎构建页增加「同时生成模型包」勾选项（`engAutoBundle`，默认勾选）

## 5. 测试

- [x] 5.1 新增 `tools/test_model_bundle.py`：29 项，覆盖造包两条路径、原始产物不变、
      硬链接、覆盖守卫、包名收敛、schema/哈希/越界/缺文件/种类/任务/trt 主小版本/显卡架构、
      异常检测 metadata、枚举去重、散装兼容、路由注册、推理签名
- [x] 5.2 造包后 `manifest.json` 能被 `json.loads` 读回并逐字段断言
- [x] 5.3 `run_all_tests.py` 按目录自动收录（58 个脚本全通过）
- [x] 5.4 前端 `node --check` 全部通过，8 个 `.mjs` 测试全通过

## 6. 文档

- [x] 6.1 新增 `docs/模型包说明.md`：目录结构、manifest 字段表、三档结论含义、
      造包与排查步骤
- [x] 6.2 `6-web/README.md` 增加 v1.36.0 变更点
- [x] 6.3 `docs/WORKFLOW_GUIDE.md` 增加「7.5 交付模型包」一节
- [x] 6.4 `docs/TRTForge_Web操作手册.md` 推理页补模型包分组说明与三条 FAQ
- [x] 6.5 `readme.md` 重要文档表增加新文档链接
