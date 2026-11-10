# 7-autoresearch-yolo — YOLO 自主实验循环

在 **PCB 缺陷检测任务**上，用「固定时间预算 + 单一适应度 + 台账留痕」的方式做 YOLO
超参/增强的自动化实验。方法论移植自 [Karpathy 的 autoresearch](../7-autoresearch-master/)，
但**不复用它的任何代码**——那边是 LLM 预训练（`prepare.py` 是 BPE 语料与 `val_bpb`），
与目标检测无关。

## 一、它解决什么问题

手动调 YOLO 有三个实际痛点：

1. **单次实验太久**。默认 100 epoch @640 约 35~40 分钟，一次试错半小时就没了。
2. **结果不可比**。改了 epoch 数的实验和改了增强的实验混在一起看，分不清是哪个起的作用。
3. **没有留痕**。跑完十几个 `exp`、`exp-2`、`exp-3` 目录后，记不清哪个对应哪套参数。

本模块的对策：

- **固定 `budget_seconds` 硬截断**。不论你把模型改大改小，每次实验都是同样的墙钟时间，
  于是不同配置的 mAP 可以直接横比。
- **适应度取 `results.csv` 的最后一行**，不是历史上的最佳 epoch。固定预算下比的就是
  「到点那一刻的水平」，取 max 会让早慧型配置虚高。
- **台账 `results.tsv` 一行一个实验**，含请求参数、实际生效 batch、峰值显存、假设描述。
- **显存预检**。宁可提前拒绝，也不让 Ultralytics 静默把 batch 减半——batch 一变，
  等效学习率和正则强度全跟着变，那次实验就与其它实验不等价了。

## 二、核心设计：与上游的三处有意偏离

| 维度 | 上游 autoresearch | 本模块 | 原因 |
|---|---|---|---|
| 回滚机制 | `git commit` / `git reset` | 配置 JSON 快照 | 上游 agent 改的是**代码**，必须靠 git 精确回滚；这里 agent 改的是**数值参数**，回滚就是"不采用这份 JSON"，主仓库 git 保持干净 |
| 适应度取值 | 预算结束时那一刻的 `val_bpb` | `results.csv` 最后一行的 mAP50-95 | 同上，保持一致 |
| 可改文件数 | 1 个（`train.py`） | 1 个（`experiments/cur.json`） | 上游靠"单文件"控制 review 成本，这里靠"单 JSON + 白名单校验"达到同样效果 |

## 三、目录结构

```
7-autoresearch-yolo/
├── autoexp.py                驱动器（单文件，5 个子命令）
├── configs/
│   ├── base.json             基线配置（所有实验的底色）
│   └── space.json            搜索空间白名单，越界直接拒绝
├── experiments/
│   ├── exp_NNNN.json         每次实验的完整参数快照（可回溯）
│   └── cur.json              下一次要跑的参数（只写改动的字段）
├── splits/                   由 split 子命令生成
│   ├── pcb_tune.yaml         训练用 val = tune（97 张）
│   ├── pcb_holdout.yaml      验收用 val = holdout（42 张）
│   ├── pcb/images/{train,tune,holdout}    硬链接镜像，不占额外磁盘
│   └── _manifest.json        切分签名，保证幂等
├── runs/autoexp/exp_NNNN/    训练产物（args.yaml / results.csv / weights/）
├── weights/                  本地权重（避免 Ultralytics 隐式下载）
├── results.tsv               实验台账（搜索期）
├── holdout.tsv               验收台账（搜索期禁止写入）
└── _backup/launcher_<日期>/  老版本 launcher 代码快照
```

**什么不入库**（`.gitignore` 已排除，约 914 MB）

| 路径 | 为什么 |
|---|---|
| `splits/` | 硬链接镜像的原始数据集（1390 个文件，914 MB），`autoexp.py split` 可原样重建 |
| `runs/` | 训练产物，可重跑 |
| `weights/` | 本地权重 |
| `_backup/` | launcher 快照，可由 tag `yolo-stable-before-autoexp` 复原 |

即：**clone 下来只有 21 个小文本文件**，跑一次 `split` 就能恢复完整数据集视图。

## 四、为什么要有 `splits/`：holdout 纪律

原始数据集只有一个 `val` 目录（139 张），而 `best.pt` 也是按它选出来的。如果自动搜索
也盯着这 139 张反复优化，很快就把验证集拟合掉了——最后看到的是「在这 139 张上很好」，
而不是「泛化好」。

于是 `split` 把它**确定性**切成两半：

| 用途 | 张数 | 谁看 |
|---|---|---|
| `tune` | 97 | 搜索期每次实验都看，用来决定 keep / discard |
| `holdout` | 42 | **搜索期一次都不许看**，只在最后验收时跑一次 |

切分是确定性的（固定 `seed=42`，先排序再洗牌），重复执行不会得到不同结果，也不会覆盖
已有文件（`--force` 才重建）。

镜像用**硬链接**（`os.link`），不复制文件。所以 732 MB 训练集只占一份磁盘，改动原始
数据集里的图片会同步体现（因为本来就是同一个 inode）。

## 五、使用方式

### 5.0 前置：确认老版本安全

```powershell
cd dp-tool
git tag -l "yolo-stable*"                               # 应看到 yolo-stable-before-autoexp
git diff yolo-stable-before-autoexp -- 1-python-yolo/launcher   # 应该为空（零改动）
```

### 5.1 切分（只需一次）

```powershell
cd 7-autoresearch-yolo
python autoexp.py split --data "..\PCB data\PCB缺陷检测\data.yaml" --holdout-frac 0.3 --seed 42
```

输出：`train=553 tune=97 holdout=42`。再次执行会检测签名一致并跳过。

### 5.2 跑一次实验

先写 `experiments/cur.json`，**只写本次改动的字段**，其余自动继承 `configs/base.json`：

```json
{
  "hypothesis": "把分辨率提到 960，小缺陷更容易检出",
  "changed": "imgsz 640 -> 960, batch 8 -> 4",
  "train": { "imgsz": 960, "batch": "4" }
}
```

然后：

```powershell
python autoexp.py run --config experiments/cur.json
```

### 5.3 看台账

```powershell
python autoexp.py report
python autoexp.py report --sort best_map50_95
```

### 5.4 测噪声底（决定 keep 阈值，只做一次）

```powershell
python autoexp.py noise --config configs/base.json --repeat 3
```

同一配置跑 3 次，输出 mAP50-95 的均值、sigma 与极差，并给结论：

- `sigma <= 0.01` → 预算够用，keep 阈值取 2 sigma
- `0.01 < sigma <= 0.02` → 预算偏紧，阈值放大
- `sigma > 0.02` → **不可用于自动搜索**，必须先加大 `budget_seconds`

### 5.5 最后验收（搜索结束后只跑一次）

```powershell
python autoexp.py holdout --config experiments/cur.json
# 或直接指定权重
python autoexp.py holdout --weights runs/autoexp/exp_0007/weights/best.pt
```

## 六、参数说明

### 6.1 顶层字段（`configs/base.json`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `task` | `detect` | 透传给 launcher 的 `--task` |
| `data` | `splits/pcb_tune.yaml` | 数据集 yaml，相对本目录解析 |
| `model` | `yolov8n.pt` | 权重。相对路径会先在 `weights/`、再在 launcher 目录找，找到就用绝对路径，避免隐式下载 |
| `budget_seconds` | `600` | **硬预算**。转成 Ultralytics 的 `time`（小时）传入，到点即停并按首轮耗时反推 epoch 数 |
| `epochs` | `100` | 仅在 `budget_seconds` 为 null 时生效 |
| `hypothesis` | — | 这次实验的猜想，写进台账 |
| `changed` | — | 相对上一版改了什么，写进台账 |

### 6.2 训练参数（`base.json` 的 `train` 段）

全部经生产 launcher 的 `build_train_kwargs()` 映射，与界面训练**完全一致**。

| 参数 | 默认 | 说明 |
|---|---|---|
| `imgsz` | `640` | 输入分辨率。PCB 小缺陷实测 960 比 640 高约 0.09 mAP |
| `batch` | `"8"` | 字符串。8 GB 卡上 16 会 OOM（见附录 A） |
| `device` | `"0"` | GPU 编号 |
| `workers` | `0` | Windows 上派生 worker 会吃提交内存，0 最稳 |
| `optimizer` | `auto` | Ultralytics 自动选 |
| `lr0` / `lrf` | `0.01` / `0.01` | 初始学习率 / 末端比例 |
| `momentum` | `0.937` | — |
| `weight_decay` | `0.0005` | — |
| `patience` | `0` | **0 = 关闭早停**，预算模式下由时间截断控制 |
| `seed` | `0` | 固定随机种子，配合 `deterministic` 保证可复现 |
| `deterministic` | `true` | 强制确定性算法 |
| `cache` | `"ram"` | 732 MB 图一次性读进内存，`workers=0` 时避免数据加载成为瓶颈 |
| `mosaic` / `close_mosaic` | `1.0` / `10` | 短预算下建议调小 `close_mosaic`，否则后段 epoch 全无 mosaic |
| `plots` | `false` | 搜索期不画图，省时间 |

只有 `space.json` 里声明过的键才允许修改，否则 `run` 直接拒绝并列出越界项。

### 6.3 子命令参数

| 子命令 | 关键参数 | 说明 |
|---|---|---|
| `split` | `--data` `--holdout-frac` `--seed` `--force` `--copy` | `--copy` 用复制代替硬链接（跨卷时用） |
| `run` | `--config` `--exp-id` `--no-validate` | `--exp-id` 用于复现某次编号 |
| `holdout` | `--config` `--weights` `--exp-id` `--split-yaml` `--imgsz` `--batch` | 搜索期禁止调用 |
| `noise` | `--config` `--repeat` | 默认重复 3 次 |
| `report` | `--sort` | 默认按 `fitness_map50_95` 降序 |

## 七、应用场景

**适合**
- YOLO 超参搜索（`imgsz` / `lr0` / `lrf` / `optimizer` / `freeze`）
- 数据增强策略搜索（`mosaic` / `scale` / `mixup` / HSV / 翻转）
- 模型规模选型（n / s / m）在同一时间预算下的公平对比
- 小缺陷场景的分辨率-精度权衡（PCB 已验证 960 优于 640）

**不适合**
- Anomalib 的 PatchCore / PaDiM：奖励信号已饱和（现有 `anomaly_val_report.json` 是
  `auroc 1.0 / f1 1.0 / tp44 fp0 tn20 fn0`），任何配置都得满分，循环没有梯度可爬
- EfficientAd：20 epoch 要 3525 秒，压进 5 分钟预算会把 LR 日程一起改掉，
  不同实验之间比的就不再是"同一套方法跑多久"，而是"不同的训练配方"

## 八、台账格式

`results.tsv`（Tab 分隔，UTF-8）：

| 列 | 含义 |
|---|---|
| `exp_id` | `exp_0001` 递增编号 |
| `fitness_map50_95` | **适应度**，`results.csv` 最后一行的 mAP50-95 |
| `best_map50_95` | 历史最佳（仅供参考，不用于判定） |
| `map50` / `precision` / `recall` | 同一行其它指标 |
| `epochs_done` | 预算内实际跑完的 epoch 数 |
| `train_seconds` | Ultralytics 统计的训练秒数（不含启动与收尾验证） |
| `peak_vram_mb` | PyTorch 统计的峰值显存 |
| `imgsz` | 输入分辨率 |
| `batch_req` / `batch_eff` | 请求的 batch / 实际生效的 batch。**两者不等说明被自动降过，该行不可与其它行横比** |
| `model` | 权重文件名 |
| `budget_seconds` | 本次预算 |
| `status` | `done` / `crash` / `preflight_blocked` |
| `hypothesis` / `changed` | 假设与改动描述 |

改表头时会按**列名**迁移旧数据并留 `.bak` 备份，不会让历史行错位。

## 九、安全性：为什么不会写坏生产环境

1. **零改动生产代码**。`autoexp.py` 只 `import` `tf_yolo_launcher.py` 的
   `build_parser()` / `build_train_kwargs()` / `load_yolo()` 三个纯函数，不修改它。
   `git diff yolo-stable-before-autoexp -- 1-python-yolo/launcher` 可随时验证。
2. **写入守卫**。`assert_sandbox()` 拒绝任何落在 `7-autoresearch-yolo/` 之外的写入，
   并显式列出黑名单：`1-python-yolo/launcher/runs`、`2-python-anomaly/launcher/results`、
   `6-web/runtime`、`4-qt-studio`、`3-cpp-tensorrt`。
3. **数据只读**。数据集用硬链接镜像，训练读的是 `splits/pcb/images/*`；
   `cache: "ram"` 让标注缓存也待在内存里，不往原始 `labels/` 写 `.cache`。
4. **物理备份**。`_backup/launcher_<日期>/` 存了 launcher 的 `.py` 与 `configs/`
   （不含 `runs/`，约 6 MB），不依赖 git 状态。

## 十、附录 A：这台机器的实测数据

**硬件**：NVIDIA T1000 8GB（8192 MiB），在 Ultralytics 的 AMP 黑名单里，训练强制走 fp32。

| 配置 | 结果 |
|---|---|
| `batch=16 @640`，可用显存 5302 MB | **OOM**。Ultralytics 自动降 16→8→4→2，三次后抛错 |
| `batch=8 @640`，可用显存 5302 MB | 正常，峰值 1769.5 MB |
| `batch=8 @640`，600 秒预算 | 正常，峰值 1769.5 MB，23~24 个 epoch |

OOM 的根因：`6-web` 的两套常驻服务（端口 8060 / 8094）各挂一个 `infer_worker.py`，
长期占用约 2.7 GB 显存。跑实验前若不需要 Web 端，建议停掉一套。

**单轮耗时**：`batch=8 @640` 约 24 秒/epoch，600 秒预算约 24~25 个 epoch。

## 十一、噪声测试结果

**结论：预算够用，可用于自动搜索。**

2026-11-10 实测，`configs/base.json` 原样跑 3 次（600 秒预算，`batch=8 @640`）：

| 实验 | mAP50-95 | epoch 数 | 训练秒数 |
|---|---|---|---|
| `exp_0003` | 0.40496 | 24 | 602.51 |
| `exp_0004` | 0.41109 | 24 | 602.88 |
| `exp_0005` | 0.40385 | 23 | 599.93 |

```
均值   0.406633
sigma  0.003184
极差   0.007240
keep 阈值（2 sigma）= 0.006368
```

**怎么用这个数**：新实验的 `fitness_map50_95` 必须比当前最佳高 **0.0064 以上**才算赢。
小于这个幅度的提升，无法与随机波动区分，一律 discard。

**三点值得注意**

1. **sigma 0.0032 远低于上游要求的 0.01**，说明 600 秒预算在这张卡上是充足的，
   不需要为了降噪而加预算（加预算会直接砍掉单位时间内的实验次数）。
2. **600 秒 vs 120 秒差距巨大**：同样配置，120 秒（5 epoch）只有 0.197，600 秒（23~24 epoch）
   到 0.407。短预算下模型还在快速爬坡期，测出来的差异多半是"随机种子运气"，
   这也反过来证明当初没选 120 秒是对的。
3. **`exp_0005` 的 `best_map50_95`（0.40925）比 `fitness_map50_95`（0.40385）高 0.0054**——
   已经接近 keep 阈值了。这正是"适应度取最后一行、不取历史最佳"的原因：如果取 max，
   这一行会虚高 0.0054，恰好是噪声量级，足以骗过一个本该被丢弃的配置。

## 十二、首轮实验结果（2026-11-10）

基线 `configs/base.json`（`imgsz=640 batch=8`，600 秒）连跑 5 个假设。
噪声底 sigma = 0.003184，**keep 阈值 2σ = 0.006368**，即只有比当前最佳高 0.0064 以上才算赢。

| 实验 | 改动 | tune mAP50-95 | epoch | 差值 | 判定 |
|---|---|---|---|---|---|
| `exp_0006` | `imgsz 640→960, batch 8→4` | **0.45862** | 12 | **+0.0475** | **KEEP** |
| `exp_0007` | `close_mosaic 10→3` | 0.41078 | 24 | +0.0037 | discard（在噪声内） |
| `exp_0008` | `lr0 0.01→0.02` | 0.40715 | 24 | +0.0001 | discard（在噪声内） |
| `exp_0010` | `optimizer→AdamW, lr0=0.002` | 0.40407 | 24 | -0.0031 | discard（在噪声内） |
| `exp_0009` | `mosaic 1.0→0.5, scale 0.5→0.7` | 0.3892 | 24 | **-0.0164** | discard（真退步） |

（差值基准取基线三次的均值 0.406633）

**holdout 复核**

| 模型 | tune | holdout |
|---|---|---|
| `exp_0005`（640，基线） | 0.40385 | 0.438055 |
| `exp_0006`（960） | 0.45862 | **0.473726** |

提升在 holdout 上同向成立（+0.0357），不是对 tune 那 97 张的过拟合。结论已写入
`experiments/best.json`。

**三点值得记下的**

1. **分辨率是这一轮唯一的真信号，而且信号极强**。`0.0475 / 0.003184 ≈ 15 sigma`。
   PCB 缺陷尺寸小，640 下的下采样把细节抹掉了。
2. **960 只跑得完 12 个 epoch，却比 640 的 24 个 epoch 好**。这说明在这个任务上
   「每张图看得更清楚」的收益，远大于「多看几遍」的收益。别被 epoch 数骗了。
3. **三条"看起来有道理"的短预算调整全在噪声内**（换优化器、加学习率、提早关 mosaic），
   **一条减弱增强的直接退步**。如果没有先测噪声底，这四条里至少两条会被误判为改进——
   这正是 Phase 0 必须先跑的原因。

**一个必须记住的坑**：第一次用 `holdout` 验 `exp_0006` 时得 0.328896，看起来像灾难性过拟合。
真实原因是 `holdout` 默认从 `configs/base.json` 读 `imgsz=640`，等于拿 640 的尺子去量
960 训出来的模型。已改成**自动从该次训练的 `args.yaml` 读回 `imgsz` 与 `batch`**，
命令行显式传 `--imgsz` 时才覆盖。教训：验收时必须用训练时的同一把尺子。

## 十三、版本记录

### v0.1.0 (2026-11-10)

首版。

- `autoexp.py`：`split` / `run` / `holdout` / `noise` / `report` 五个子命令
- `configs/base.json` 基线（600 秒预算，batch 8）
- `configs/space.json` 搜索空间白名单
- 显存预检（拒绝静默降 batch）
- 台账表头按列名自动迁移
- 老版本保护：git tag `yolo-stable-before-autoexp` + `_backup/launcher_20261110/`

**已知限制**

- 分类任务的 `results.csv` 列名（`metrics/accuracy_top1`）未经实测验证
- 本轮只搭地基，**不含**"谁来决定下一次改什么参数"这一层（第 4 阶段再做）
- 搜索期的人工判定（keep / discard）目前靠人看台账，尚未自动化

**已验证**

```powershell
python run_all_tests.py     # 17 项，不跑训练，约 3 秒
```

`splits/` 不存在时（新机器 clone 后），两条依赖切分的用例会**跳过**而不是失败，
其余 15 项照常通过——这样不会因为「还没跑 split」就把整轮测试标红。
- `split`：`train=553 / tune=97 / holdout=42`，幂等
- `run`：`exp_0003` / `exp_0004` / `exp_0005` 三次 600 秒实验全部 `done`
- `noise`：sigma 0.003184，keep 阈值 0.006368
- `holdout`：`exp_0005` 权重在 42 张 holdout 上得 mAP50-95 = 0.438056（185 个实例，耗时 8.5 秒）
- `report`：按 `fitness_map50_95` 降序输出
- 生产 launcher 零改动：`git diff yolo-stable-before-autoexp -- 1-python-yolo/launcher` 为空
