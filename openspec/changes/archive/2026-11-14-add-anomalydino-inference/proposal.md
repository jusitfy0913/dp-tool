# Change: 把 AnomalyDINO 作为第四种异常检测模型接入（零标注、零梯度训练）

## Why

现有三种 Anomalib 模型（PatchCore / PaDiM / EfficientAd）在产线落地时各有硬伤：

- **都要标注或调参**：EfficientAd 需要 ImageNet 目录做蒸馏，20 轮起步；PatchCore / PaDiM
  虽然只 fit 一轮，但仍依赖 anomalib + Lightning 整套栈，冷启动 import 就要 5 秒以上。
- **换产品等于重来一趟**：每个产品都要准备异常样本才能定阈值，而现场最常见的诉求是
  「我只有一堆好品图，先给我一个能筛出坏品的模型」。
- **小缺陷召回不够**：PatchCore 的 patch 粒度受 CNN 感受野限制，PCB 上的丝印断线、
  细小异物经常只报出很淡的热力图。

AnomalyDINO（WACV 2025）给出了第三条路：**DINOv2 冻结特征 + 1-NN 记忆库**。
它不需要标注、不需要梯度训练、不需要 anomalib / Lightning，所谓「训练」只是对 N 张
正常图做一次前向把 patch 特征存成库。本仓已实测 ViT-S/14@448 在 T1000 8GB 上
显存峰值 118 MB、单张前向 81 ms，16-shot 建库 1.3 秒。

## What Changes

- 新增 `2-python-anomaly/launcher/tf_anomalydino_launcher.py`，与 `tf_anomalib_launcher.py`
  并列的三模式启动器（`check-env` / `build-export` / `warm`），**不 import anomalib、不 import lightning**。
- 主干改用 **timm 的 `vit_small_patch14_dinov2.lvd142m` + 官方 DINOv2 权重**，
  并把 `pos_embed` 按上游 `interpolate_offset=0.1` 的公式预插值到目标网格。
  已实测量化验证：与上游 `torch.hub` 模型在 448×448 输入下 **逐元素差 0.0、逐 token 余弦 1.0**，
  即特征与上游完全一致。这样运行期不再需要 `torch.hub`（不依赖 GitHub 可达）。
- 记忆库**固化为 ONNX 图里的常量**：正常图的 patch 特征拼成 `[rows, 384]` 常量张量，
  图内做分块余弦相似度取最大、`1 - max` 得距离；`pred_score` 用上游同款
  「top 1% 均值」（`mean_top1p`），网格固定为 32×32，`k=10` 为常量，可导出。
- ONNX 契约与 PatchCore 同构：输入 `input [1,3,H,W]` 静态、RGB、`/255` + ImageNet 均值方差；
  输出 `anomaly_map [1,1,gh,gw]` 与 `pred_score [1,1]`，复用 `anomalib_infer.dll` **零改动**。
- `metadata.json` 新增 `model_type: "AnomalyDINO"`、`output_contract: "trtforge_anomalydino_v1"`，
  以及 `backbone / patch_size / grid / shots / bank_rows / bank_sha256` 溯源字段。
- Web 侧**只加分支、不动旧路径**：训练页与全流程各加一个模型选项，推理页模型类型下拉与
  `normalizeModelType` 补一项映射；`pipeline.py` 的模型白名单加 `elif` 分支；
  `train.py` 新增 `build_anomalydino_command` 并按 model 分派。
- 打包：`Compile-Launcher` 增加可选参数按需带 `timm`，安装脚本补 `timm==1.0.14`，
  DINOv2 权重（88 MB）进 `modelzoo` 走流式下载与清单校验，打包后检查 `import timm` 失败即停。
- 版本号 `v1.40.0`。

## Impact

- Affected specs:
  - `anomalydino-export`（**新增能力**）
  - `web-anomalib-pipeline`（MODIFIED：模型集合、静态 batch 的 preflight）
  - `anomalib-trt-inference`（MODIFIED：输出解析改为「先按名字、再按最大元素数」）
- Affected code:
  - 新增 `2-python-anomaly/launcher/tf_anomalydino_launcher.py`、
    `2-python-anomaly/launcher/anomalydino_core.py`（纯逻辑，便于单测）、
    `2-python-anomaly/launcher/tools/export_spike.py`（P0 验证脚本）
  - 新增 `6-web/backend/tools/test_anomalydino_launcher.py`、`test_anomaly_contract.py`
  - `6-web/backend/routers/train.py`、`routers/pipeline.py`、`config.py`
  - `6-web/frontend/index.html`、`js/infer.js`、`js/train.js`（如需新字段）
  - `packaging/package_web_user.ps1`、`packaging/templates/安装训练环境.ps1`
  - `6-web/backend/modelzoo.py`（新增 dinov2 权重条目）
- **BREAKING**：无。`anomalib_infer.dll` 二进制与 `anomalib_infer_c.h` 不改，
  已有的 PatchCore / PaDiM / EfficientAd 引擎与 metadata 不改、不重编，
  三个旧模型的命令拼装函数保留原样（由命令快照测试守住）。
