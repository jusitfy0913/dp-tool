## 1. 契约与核心模块

- [ ] 1.1 新增 `6-web/backend/bundle.py`：常量（`SCHEMA = "trtforge.bundle.v1"`、结论档位）、
      `manifest_path()`、`read_manifest()`、`enumerate_bundles()`
- [ ] 1.2 实现 `build_bundle(engine_or_onnx, labels, metadata, out_dir, bundle_id, overwrite)`
      —— 采集运行时、算 sha256、硬链接或复制、写 manifest 与 `README.txt`
- [ ] 1.3 实现 `verify_bundle(path, kind=None, task=None)` 返回
      `{"level": "ok|warn|block", "checks": [{"name","level","detail","fix"}]}`
- [ ] 1.4 实现兼容性判定：schema、路径越界、哈希、必需文件、kind/task、
      trt 主/小版本、compute capability、imgsz、标签数

## 2. 运行时字段采集

- [ ] 2.1 `enginemeta.write_meta` 增加 `runtime` 区块：`trt_version`、`cuda_version`、
      `gpu_name`、`compute_cap`、`imgsz`、`dynamic`、`onnx_layout`
- [ ] 2.2 TensorRT 版本读取走 `nvinfer.dll` 文件版本或 `trtexec --version`，取不到留空
- [ ] 2.3 GPU 信息读取复用现有 nvidia-smi 调用路径，取不到留空

## 3. 接口

- [ ] 3.1 `GET /api/bundles`：枚举，返回 id、路径、manifest 摘要、校验结论
- [ ] 3.2 `POST /api/bundles/build`：造包，返回 manifest 与新包路径
- [ ] 3.3 `POST /api/bundles/verify`：按可选 kind/task 校验
- [ ] 3.4 `GET /api/bundles/{bundle_id}/manifest`：原文返回
- [ ] 3.5 `/api/models` 每条结果增加 `bundle` 字段（id、路径、结论），旧字段不变
- [ ] 3.6 推理接口支持 `bundle_id`：与引擎/ONNX 路径互斥，使用包内标签与 metadata
- [ ] 3.7 引擎构建成功后自动造包（开关默认开），失败只记日志不影响构建结果

## 4. 前端

- [ ] 4.1 `js/infer.js`：模型下拉分「模型包 / 训练产物（散装）」两组
- [ ] 4.2 选中模型包后填充三个路径并置为随包只读
- [ ] 4.3 显示校验结论与修复建议；`block` 时禁用推理按钮
- [ ] 4.4 手工修改路径后自动退回散装提交方式
- [ ] 4.5 引擎构建页增加「同时生成模型包」勾选项

## 5. 测试

- [ ] 5.1 新增 `tools/test_model_bundle.py`：
      schema 校验、哈希不符、路径越界、kind/task 冲突、trt 主/小版本、
      硬链接与复制两条路径、原始产物不被修改、散装兼容
- [ ] 5.2 造包后 `manifest.json` 必须能被 `json.loads` 读回并逐字段断言
- [ ] 5.3 `run_all_tests.py` 收录新脚本并全绿
- [ ] 5.4 前端 `node --check` 全部通过

## 6. 文档

- [ ] 6.1 新增 `docs/模型包说明.md`：目录结构、manifest 字段表、三档结论含义、
      如何把已有散装产物造成包、如何排查 `block`
- [ ] 6.2 `6-web/README.md` 增加 v1.36.0 变更点
- [ ] 6.3 `docs/WORKFLOW_GUIDE.md` 增加「交付模型包」一节
- [ ] 6.4 `docs/TRTForge_Web操作手册.md` 推理页章节补模型包分组说明
- [ ] 6.5 `readme.md` 重要文档表增加新文档链接
