## 1. 指纹模块

- [x] 1.1 新增 `6-web/backend/fingerprint.py`：`compute(yaml_path)`、`short()`、`algo` 版本、
      异常降级（返回 `error` 不抛）
- [x] 1.2 支持 `dataset.yaml` 的 `path:` 相对/绝对两种写法；分类任务按目录布局处理
- [x] 1.3 复用 `traininfo.resolve_splits` 解析 yaml，不重复写一份解析器

## 2. 导出 manifest

- [x] 2.1 `dataset.py` 的 `export_yolo` / `export_yolo_classify` / `export_coco` / `export_voc`
      统一在收尾写 `<output>/manifest.json`
- [x] 2.2 manifest 同目录已存在且指纹相同时返回 `unchanged=true`
- [x] 2.3 `run_export` 与标注页导出接口的返回值带上指纹
- [x] 2.4 新增 `GET /api/dataset/fingerprint?source=`（只读，不落盘）

## 3. 训练侧

- [x] 3.1 `POST /api/train/yolo` 启动前算指纹写进 `run.meta["dataset"]`
- [x] 3.2 训练目录写 `train_meta.json`（数据路径、指纹、各 split 数量、计算时间）
- [x] 3.3 `runtime/runs.jsonl` 任务台账带 `data_fp` 短码
- [x] 3.4 训练页显示「本次将使用 `<短码>`（N train / M val）」

## 4. autoexp

- [x] 4.1 `exp_NNNN.json` 增 `dataset` 块（含 `algo` 与三个指纹）
- [x] 4.2 `RESULTS_HEADER` 增 `data_fp` 列，走 `ensure_tsv()` 迁移
- [x] 4.3 `best.json` 的 `evidence` 记录数据指纹
- [x] 4.4 `splits/_manifest.json` 增 `data_fingerprint`
- [x] 4.5 找不到 `fingerprint.py` 时标 `unavailable`，不阻断实验

## 5. 引擎元数据

- [x] 5.1 `POST /api/engine/build` 成功后写 `<engine>.meta.json`
- [x] 5.2 内容含 onnx 路径 + sha256、weights、训练 run、数据集指纹、trtexec 参数、时间
- [x] 5.3 写失败只记日志，不影响构建结果

## 6. 界面

- [x] 6.1 数据集页：导出成功显示指纹短码，可展开看完整值
- [x] 6.2 自动实验列表：按指纹分组着色，跨指纹提示「数据不同，不可直接横比」
- [x] 6.3 旧产物（无指纹）显示「未记录」，不报错

## 7. 测试与文档

- [x] 7.1 新增 `tools/test_dataset_fingerprint.py`：同目录两次导出相同、跨目录相同、
      改标注变化、重命名变化、只改 val 只动 val、分类任务可用、yaml 缺失降级
- [x] 7.2 `7-autoresearch-yolo/run_all_tests.py` 补 data_fp 列与 dataset 块用例
- [x] 7.3 回归后端全部测试套件（用 `dpenv` 解释器）
- [x] 7.4 真机：导出一次、训练一次，核对 manifest / train_meta / exp 记录 / engine.meta 四处指纹一致
- [x] 7.5 更新 `readme.md`、`6-web/README.md`、`7-autoresearch-yolo/README.md`、
      `docs/WORKFLOW_GUIDE.md`
