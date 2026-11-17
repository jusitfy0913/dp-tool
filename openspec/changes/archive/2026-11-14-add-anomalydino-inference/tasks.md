## 1. 提案与契约

- [x] 1.1 写 `proposal.md` / `design.md` / `tasks.md` 与三份 spec delta，`openspec validate --strict` 通过
- [x] 1.2 定死 ONNX 输入输出契约（静态 `[1,3,H,W]`、`anomaly_map` 最大、`pred_score` 元素数为 1）
- [x] 1.3 定死 `metadata.json` 字段表（含点号平铺键与嵌套块，见 `design.md` D9）

## 2. P0 技术验证（数值不达标就停在这里）

- [x] 2.1 新增 `2-python-anomaly/launcher/anomalydino_core.py`：
      `build_pos_embed(ckpt, grid)` 按上游 `scale_factor=((grid+0.1)/37)` 插值、
      `load_backbone(ckpt, resolution)`、`build_bank(features)`、`cosine_distance()`
- [x] 2.2 新增 `tools/export_spike.py`：跑通「权重 → timm 主干 → 建 16-shot 库 → 导出 ONNX」
- [x] 2.3 验证 timm 与上游 `torch.hub` 在真实 bottle 图上逐 token 余弦 = 1.0
- [x] 2.4 验证 ONNX 与 PyTorch 的 `anomaly_map` / `pred_score` 一致（相关 ≥ 0.99、分数差 ≤ 1e-3）
- [x] 2.5 用 `trtexec` 编 FP16 静态引擎，验证 TRT 与 ONNX 热力图逐像素相关 ≥ 0.99
- [x] 2.6 记录 448 分辨率下实测单张延迟、显存峰值、ONNX 大小、编译耗时，写回 `design.md`
- [x] 2.7 ~~建 `results/_staging/bottle` junction，跑上游参考实现拿 bottle 的 AUROC 基线~~
      → **改成不做**：上游要另装 faiss-gpu、改它写死在 `/remote-home` 的路径，且它按 15 个对象
      写死目录名（见 D8）。收益只是「和论文数字并排」；「实现是否正确」已经由 2.3 的
      逐元素一致（余弦 1.000000、最大差 0）与 2.4 的 ONNX 对齐（最大差 2.9e-6）证明，
      本机指标另有 6.6 的全流程数据。留着这条只会变成「为了对齐论文去维护第二套环境」。

## 3. P1 启动器

- [x] 3.1 新增 `2-python-anomaly/launcher/tf_anomalydino_launcher.py`，三模式
      `check-env` / `build-export` / `warm`，复用 `TRTFORGE_JSON` 事件协议
- [x] 3.2 `check-env` 检查 torch / timm / onnx / onnxruntime / 权重文件 / CUDA，缺项给 `MISSING_DEP`
- [x] 3.3 `build-export` 流程：校验目录 → 轮转取样 → （可选旋转增强）→ 建库 →
      导出 ONNX → 标定阈值 → 写 metadata
- [x] 3.4 正常图少于下限（建议 8 张）时明确报错，不静默降级
- [x] 3.5 标定用与 DLL 完全一致的「拉伸 + `/255` + ImageNet」口径，算出
      `image_threshold` / `pixel_threshold` / `pred_scores.min|max` / `anomaly_maps.min|max`
- [x] 3.6 产物落到 `results/AnomalyDINO/<name>/v<N>/weights/lightning/`，与现有扫描一致
- [x] 3.7 不 import anomalib / lightning；`import torch` 只在需要时延迟到函数内

## 4. P2 Web 接入（只加分支）

- [x] 4.1 `train.py`：`AnomalibTrainRequest` 增 `shots`/`resolution`/`rotation`/`seed`，
      新增 `build_anomalydino_command`，`/api/train/anomalib` 按 model 分派
- [x] 4.2 `pipeline.py`：白名单加 `anomalydino`，新增分支（建库导出 → probe → build → val）
- [x] 4.3 `pipeline.py` 的 probe 支持静态 batch（不加 `--minShapes`），动态 batch 行为不变
- [x] 4.4 训练页 `anomModel` 与全流程 `pipeAnomalyModel` 各加一项，新字段只在该模型下显示
- [x] 4.5 推理页 `anomModelType` 加一项，`infer.js` 的 `normalizeModelType` 加映射
- [x] 4.6 新增命令快照测试：`patchcore` / `padim` / `efficientad` 三个旧值的命令行逐字符不变

## 5. P3 打包与离线

- [x] 5.1 `Compile-Launcher` 增可选参数，只给新启动器带 `timm`，不改另外两个的编译参数
- [x] 5.2 编译 `tf_anomalydino_launcher_bin.pyd` 并生成 stub
      （**实测踩坑**：`--module` 模式不会跟随「编译目录里的同级 .py」，只 `Copy-Item`
      `anomalydino_core.py` 编得出来、运行时才 `ModuleNotFoundError`。已改成
      `--include-module` + `Push-Location`，并用 `--report` 的 XML 逐个核对辅助模块在不在）
- [x] 5.3 `安装训练环境.ps1` 在 torch 之后单独装 `timm==1.0.14`，离线路径一次装齐
- [x] 5.4 DINOv2 权重（88.3 MB）进 `modelzoo`，带 sha256，缓存到 `models/dinov2/`
- [x] 5.5 打包后检查：`requiredWheels` 必须含 `timm` 等 12 个库，缺一个就 throw 不出 zip；
      安装脚本自检逐个 `import` 8 个库（torch / timm / polars / onnx / onnxruntime /
      cv2 / ultralytics / anomalib），缺一个就打印 `MISSING:` 并非零退出。
      权重不进 zip（走 modelzoo），改由启动器 `check-env` 校验 sha256 并报 `MISSING_DEP`

## 6. P4 测试与文档

- [x] 6.1 `test_anomalydino_launcher.py`：库行数 = 图数 × 1024、输出名契约、阈值单调、
      空目录 / 缺权重 / 正常图不足三条错误路径、metadata 字段齐全（80 项）
- [x] 6.2 `test_anomaly_contract.py`：两个假 ONNX（合规 / 多导出一个大 mask），
      断言 DLL 的挑选规则与 `trtforge_metadata.parse_anomaly_outputs` 一致（18 项）
- [x] 6.3 `run_all_tests.py` 收录两套新测试（共 66 个脚本），另加
      `test_packaging_wheels.py`（69 项）把「安装包必须自带依赖库」变成会红的测试
- [x] 6.3.1 打真包时暴露的安装包缺陷一并修掉：`.pyd` 的 DLL 传递依赖原来靠手写清单，
      漏了 conda `Library\bin` 里的 `tiff.dll` / `openjp2.dll` / `ffi.dll` 等（打包后双击
      即 `0xC0000409` 闪退且无日志）。改成用 `dumpbin` 扫 PE 导入表做闭包补齐，
      `api-ms-win-*` 等系统转发 DLL 一律不搬；出 zip 前在安装目录跑
      `TRTForgeWeb.exe --tool selfcheck`（9 个关键模块逐个真 import），不过就 throw。
      `frozen_entry.py` 新增 `--tool selfcheck` 分支，排障也可直接用
- [x] 6.4 文档四处：`docs/异常检测模型对照_AnomalyCLIP与Anomalib.md` 增第 3 节、
      `docs/WORKFLOW_GUIDE.md` 增 3.3.1、`6-web/README.md` 增「AnomalyDINO 零样本异常检测」、
      `readme.md` 增 v1.40.0
- [x] 6.5 版本号 `v1.40.0`，`APP_VERSION` 已改；打包侧的版本由
      `package_demo_models.ps1` 从 `config.py` 读取，无需另改
- [x] 6.6 用 bottle 跑一遍全流程（建库 → 导出 → 编译 FP16 引擎 → 经真实 DLL 推理）：
      broken_large AUROC 1.0000 / broken_small 1.0000 / contamination 0.9976，
      合计 F1 0.992、漏检 1.6%、误报 0%，8/8 张判定与 ONNX 一致；
      与上游参考实现的并排对比按 2.7 的决定不做

## 7. 归档

- [x] 7.1 全部任务完成后把所有复选框置 `- [x]`
- [x] 7.2 `openspec archive add-anomalydino-inference --yes` 并 `validate --strict`
