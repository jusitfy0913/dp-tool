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
- **台账 `results.tsv` 一行一个实验**，含请求参数、实际生效 batch、峰值显存（分配 + 预留两列）、
  实际精度、假设描述。
- **显存预检用实测峰值表，不用手拟合公式**。宁可提前拒绝，也不让 Ultralytics 静默把 batch
  减半——batch 一变，等效学习率和正则强度全跟着变，那次实验就与其它实验不等价了。
  表在 `experiments/vram_samples.json`，每次训练结束自动写回（自校准），详见 §8.1。

## 二、核心设计：与上游的三处有意偏离

| 维度 | 上游 autoresearch | 本模块 | 原因 |
|---|---|---|---|
| 回滚机制 | `git commit` / `git reset` | 配置 JSON 快照 | 上游 agent 改的是**代码**，必须靠 git 精确回滚；这里 agent 改的是**数值参数**，回滚就是"不采用这份 JSON"，主仓库 git 保持干净 |
| 适应度取值 | 预算结束时那一刻的 `val_bpb` | `results.csv` 最后一行的 mAP50-95 | 同上，保持一致 |
| 可改文件数 | 1 个（`train.py`） | 1 个（`experiments/cur.json`） | 上游靠"单文件"控制 review 成本，这里靠"单 JSON + 白名单校验"达到同样效果 |

## 三、目录结构

```
7-autoresearch-yolo/
├── autoexp.py                驱动器（单文件，7 个子命令）
├── llm.py                    OpenAI 兼容客户端（零第三方依赖，纯标准库）
├── prompts/
│   └── experimenter.md       系统提示词：角色 / 机器事实 / 已证伪方向（改策略只改这里）
├── configs/
│   ├── base.json             基线配置（所有实验的底色）
│   ├── space.json            搜索空间白名单，越界直接拒绝
│   ├── llm.json              模型接口默认值（不含密钥）
│   └── llm.local.json        本地覆盖（gitignore），临时换厂商用
├── experiments/
│   ├── exp_NNNN.json         每次实验的完整参数快照（可回溯）
│   ├── cur.json              下一次要跑的参数（只写改动的字段）
│   ├── best.json             当前最佳配置 + 证据（agent 只在 KEEP 时改它）
│   └── agent/round_NNNN.json 每轮审计：提议原文、tokens、判定理由（gitignore）
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
| `configs/llm.local.json` | 本地接口覆盖，可能含内网地址 |
| `experiments/agent/` | 每轮审计含完整 prompt 与模型原文，体积随轮数线性增长 |

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

**数据变过的检测（v0.2.4）**：`_manifest.json` 除 `signature`（文件名 + seed）外，
还记录 `data_fingerprint` 与 `holdout_fingerprint`。`holdout` 子命令开跑前会对比当前
实际指纹，不一致就明确警告「之前的验收结论不可沿用」。原因：`signature` 对「把某张
holdout 图的标注改掉」完全无感，而硬链接意味着这种改动会悄悄生效。

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

> **注意「继承」两套口径**（2026-11-10 踩过坑）：
> - 手动 `run` 合并的是 `configs/base.json`。上面那个 `cur.json` 没写的键全取基线。
> - `agent` 自动跑时，`cur.json` 由程序写，**没写的键继承的是 `experiments/best.json`
>   （当前最优）+ 本轮改动**，不是基线。否则「在 960 上改 warmup」会悄悄变成
>   「在 640 上改 warmup」，整轮结论作废（真实事故：exp_0017）。
> - 程序还会在每轮审计里记 `inherited_from`，事后能查到那一轮的起点是谁。

### 5.3 看台账

```powershell
python autoexp.py report
python autoexp.py report --sort best_map50_95
```

### 5.4 测噪声底（决定 keep 阈值）

```powershell
# 基线（640）
python autoexp.py noise --config configs/base.json --repeat 3
# 换了分辨率/批次就再测一次，两种配置的抖动不是一个量级
python autoexp.py noise --config experiments/cur.json --repeat 3
```

同一配置跑 3 次，输出 mAP50-95 的均值、sigma 与极差，并给结论：

- `sigma <= 0.01` → 预算够用，keep 阈值取 2 sigma
- `0.01 < sigma <= 0.02` → 预算偏紧，阈值放大
- `sigma > 0.02` → **不可用于自动搜索**，必须先加大 `budget_seconds`

结果写进 `experiments/_noise_summary.json`，同时按 `imgsz × batch` 分组存进
`groups` 字段。`agent` 判定 KEEP 时**优先用当前最优那一组的 sigma**：

| 取值顺序 | 来源 | `calibrated` |
|---|---|---|
| 1 | `groups["imgsz=960\|batch=4"]`，即在该配置上真实重复测过 | `true` |
| 2 | 台账里**配置完全相同**的多次实验，用它们的离散度 | `true` |
| 3 | 全局 sigma | **`false`** |

`calibrated=false` 时，判定理由和网页事件里都会明写「这组参数的噪声底没测过，
借用了别的配置的阈值」。**看到这句话就说明该去补一次 `noise`**——不然
「+0.03 到底是真提升还是抖动」根本无从判断。

> 为什么必须分组：640 能跑 24 个 epoch，960 只能跑 12 个。实测（2026-11-10）
> 640/batch8 的 sigma = 0.003184，而 960/batch4 是 **0.008070**，差 2.5 倍。
> 用 640 的 2σ=0.006368 去判 960，会把「960 本来就有 0.008 的抖动」当成真实提升
> （误 KEEP），也会把一次 ±0.01 的正常起伏读成「明显退步」。两个方向都会错。
>
> 顺带一条实测结论：960 的三次重复是 0.45984 / 0.45789 / 0.44183，
> 而 exp_0006 的 0.45862 正落在这个分布里（+0.005 < 1σ）。
> 也就是说**它不是幸运高点**，后续实验追不上它，是因为模型提的方向确实不好。

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
| `agent` | `--rounds` `--hours` `--max-stale` `--lock-wait` `--dry-run` `--events` | 见 §十五。`--rounds` 默认 **30**（≈5.5 小时） |
| `llm-test` | `--base-url` `--model` `--temperature` | 连通性自检。**Key 只从环境变量读，没有 `--api-key` 参数**（见 §十五 的安全边界） |

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
| `peak_vram_mb` | PyTorch 统计的峰值显存（`max_memory_allocated`，**分配**量） |
| `peak_vram_reserved_mb` | **v0.2.3 新增**。峰值显存（`max_memory_reserved`，**预留**量）。显存预检与自校准都用这一列：OOM 发生在向驱动申请的那一刻，`allocated` 会低估 30% 以上 |
| `imgsz` | 输入分辨率 |
| `batch_req` / `batch_eff` | 请求的 batch / 实际生效的 batch。**两者不等说明被自动降过，该行不可与其它行横比** |
| `amp` | **v0.2.3 新增**。本次训练实际生效的精度（`fp16` / `fp32` / 空=未知）。本机 T1000 命中 Ultralytics 的 AmpBlacklist，即使请求 AMP 也是 `fp32` |
| `data_fp` | **v0.2.4 新增**。本次训练所用数据集指纹的前 8 位（算法见 `6-web/backend/fingerprint.py`）。**跨这一列不同的行不可直接横比**：参数没变，数据变了 |
| `fp_per_image` | **v0.2.5 新增**。每张验证图的平均误报框数，由 Ultralytics 的 P/R 反推（近似值）。**只用于拦「拿过杀换 mAP」**，不是产线验收数字；空=算不出 |
| `model` | 权重文件名 |
| `budget_seconds` | 本次预算 |
| `status` | `done` / `crash` / `preflight_blocked` / `batch_downgraded` |
| `hypothesis` / `changed` | 假设与改动描述 |

`batch_downgraded` 是**主动作废**而不是崩溃：训练中一旦出现 Ultralytics 的
`Reducing to batch=N`，说明首轮 OOM 已经触发自动减半，等效学习率与正则强度
都变了，这次实验与台账其它行不是同一件事。程序会立刻中止子进程、把
`fitness_map50_95` 留空、把状态记为 `batch_downgraded`，并在该 run 目录写
`batch_downgraded.json` 留证。它**不计入 crash 统计**，也不会污染 `current_best`。

### 8.1 显存预检：实测峰值表 + 安全余量 + 自校准（v0.2.3）

**旧做法错在哪**。曾经这里有一条手拟合公式 `2600 + 240 × batch × (imgsz/640)²`：
台账里 `640/batch=8` 的峰值是 1769 MB，公式估 4520 MB（多估一倍半）；
`960/batch=4` 公式估 4760 MB 就放行，**实际仍然 OOM**。同一套判定在
`6-web` 里还有另一份实现，两处已经在 960/batch=4 上给出互斥结论。
显存占用与 `(batch, imgsz, 精度)` 的关系不是一条固定系数能覆盖的。

**现在的判定**：委托 `6-web/backend/vram.py`，与网页端训练页共用同一份实现。

| 依据 | 触发条件 | 需求 |
| --- | --- | --- |
| `measured` | 同 `(imgsz, batch, 精度)` 有实测样本 | `峰值 × 1.15 + 400 MB` |
| `scaled` | 同 `imgsz`/精度，只有**更小** batch 的样本 | `峰值 × b/b₀ × 1.15 + 400 MB` |
| `capped` | 同 `imgsz`/精度，只有**更大** batch 的样本 | 它的实测值作上界（显存随 batch 不减） |
| `formula` | 什么都没跑过 | `2600 + 300 × batch × (imgsz/640)²` |

**自校准**：`experiments/vram_samples.json`，键 `imgsz x batch x 精度`（如 `640x8xfp32`），
每次训练正常结束由 `exec` 子进程把 `max_memory_reserved` 写回（取历次最大），
并把 `amp_effective / precision / peak_vram_reserved_mb` 补进 `exp_NNNN.json`。
用 `reserved` 而不是 `allocated`：OOM 发生在向驱动申请的那一刻，后者低估 30% 以上。

**AMP 也是实测出来的**：本机 T1000 命中 Ultralytics 8.4.192 的 AmpBlacklist，
请求 `amp=True` 实际仍是 `fp32`。台账的 `amp` 列与 `exp_NNNN.json` 的
`amp_effective` 记的是**实际生效值**（来自 `trainer.amp`），不是请求值。

**本机实测（T1000 8GB，fp32）**

| 配置 | 依据 | 需求 | 旧公式 |
| --- | --- | --- | --- |
| `640/batch=8` | `measured`（2132 MB） | 2852 MB | 4520 MB |
| `640/batch=16` | `scaled`（由 batch=8 外推） | 5304 MB | 7400 MB |
| `640/batch=4` | `capped`（batch=8 作上界） | 2852 MB | 3800 MB |
| `960/batch=8` | `formula`（没跑过 960） | 8000 MB | 8000 MB |

### 8.2 表头迁移

改表头时会按**列名**迁移旧数据并留 `.bak` 备份，不会让历史行错位。

## 九、与网页端的关系：跨进程 GPU 互斥

网页端训练和 autoexp **各写各的产物目录**，从不互相覆盖：

```
网页端训练  →  1-python-yolo/launcher/runs/{pcb, detect, classify}
autoexp     →  7-autoresearch-yolo/runs/autoexp/exp_NNNN
```

但它们**共用同一块显卡**。所以有一条硬规则：**同一时刻只允许一方占 GPU**。

### 为什么必须互斥

一台 8 GB 卡上跑两个训练，结果通常是双双 OOM。更麻烦的是 autoexp 侧的表现：
Ultralytics 遇到 OOM 会把 batch 静默减半重试，于是「这次实验」用的 batch 和台账里
其它实验不一样——等效学习率、正则强度全变了，横向比较全部失真，而台账上看不出来。

### 怎么实现的

一个 JSON 锁文件 `runtime/gpu_lock.json`，同一时刻最多一个持有者：

```json
{"owner": "autoexp", "pid": 12345, "kind": "autoexp-run",
 "label": "exp_0011 提高分辨率", "since": "2026-11-10T13:20:00+08:00"}
```

| 场景 | 行为 |
|---|---|
| 网页端点「开始训练」，锁被 autoexp 持有 | 返回 **409**，文案带上占用方与 PID，提示去哪停止 |
| 网页端点「开始训练」，锁被另一套 Web 实例持有 | 同上（8060 与 8094 两套服务也能互相看见） |
| `autoexp.py run` / `holdout`，锁被网页端持有 | **拒绝启动，退出码 5**，零显存消耗、不写台账、不建实验目录 |
| 持有者进程被强杀 | 锁文件残留，但 PID 已死 → 视为空闲，不会永久锁死机器 |
| 锁文件损坏（写了一半） | 视为空闲（宁可漏判一次，也不让所有人用不了） |

实现放在 `6-web/backend/gpulock.py`（零第三方依赖，只用标准库）。
autoexp 通过 `load_gpulock()` 复用**同一份**，而不是自己再写一遍——
锁文件格式是两边共用的契约，两份实现迟早漂移，而漂移的表现正好是
「两个训练都以为自己是唯一持有者」。

### 手动查看/排查

```powershell
python 6-web\backend\gpulock.py status
# 锁文件：D:\test\deeplearning\dp-tool\runtime\gpu_lock.json
# 状态：空闲
```

真被卡住时（比如锁指向一个你以为已经退出的进程），直接删 `runtime/gpu_lock.json`
即可；下次 acquire 会重建。

### 已知取舍

- **不排队**。拿不到锁就立刻拒绝并说明原因，不让用户干等——排队会让界面卡住，
  而用户真正需要的是马上知道「谁在占着 GPU、去哪停」。
- **同进程可重入**。Web 内部同时挂多个独占任务时（既有并发测试依赖这一行为）
  锁不拦自己，拦的是**其它进程**。
- **锁保护不到非本工具的进程**。别人手工跑一个 `yolo train` 是看不见的，
  那种情况只能靠显存预检兜底。

## 十、安全性：为什么不会写坏生产环境

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

## 十一、附录 A：这台机器的实测数据

**硬件**：NVIDIA T1000 8GB（8192 MiB），在 Ultralytics 的 AMP 黑名单里，训练强制走 fp32。

| 配置 | 结果 |
|---|---|
| `batch=16 @640`，可用显存 5302 MB | **OOM**。Ultralytics 自动降 16→8→4→2，三次后抛错 |
| `batch=8 @640`，可用显存 5302 MB | 正常，峰值 `allocated` 1769.5 MB |
| `batch=8 @640`，600 秒预算 | 正常，峰值 1769.5 MB，23~24 个 epoch |
| `batch=8 @640`，1 epoch（2026-11-11 自校准实测） | 正常，`reserved` **2132.0 MB** / `allocated` 1769.2 MB；`amp_requested=true` 但 `amp_effective=false`，精度 `fp32`；已写入 `vram_samples.json` 的 `640x8xfp32` |

OOM 的根因：`6-web` 的两套常驻服务（端口 8060 / 8094）各挂一个 `infer_worker.py`，
长期占用约 2.7 GB 显存。跑实验前若不需要 Web 端，建议停掉一套。

**单轮耗时**：`batch=8 @640` 约 24 秒/epoch，600 秒预算约 24~25 个 epoch。

## 十二、噪声测试结果

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

这组数字写在 `experiments/_noise_summary.json` 的 `groups["imgsz=640|batch=8"]`。顶层字段后来被 960 的测量覆盖，分组必须单独留着，否则回到 640 时会借到 960 的尺子。

**怎么用这个数**：新实验的 `fitness_map50_95` 必须比当前最佳高 **0.0064 以上**才算赢。
小于这个幅度的提升，无法与随机波动区分，一律 discard。

**但这是 640 的尺子。** 换了分辨率/批次就必须重测——2026-11-10 补测 960/batch4：

| 实验 | mAP50-95 | epoch 数 | 训练秒数 |
|---|---|---|---|
| `exp_0023` | 0.45984 | 11 | 619.9 |
| `exp_0024` | 0.45789 | 11 | 623.6 |
| `exp_0025` | 0.44183 | 12 | 622.2 |

```
均值   0.453187
sigma  0.008070
极差   0.018010
keep 阈值（2 sigma）= 0.016139
```

| 配置 | sigma | 2σ 阈值 | 相对 640 |
|---|---|---|---|
| `640 / batch8` | 0.003184 | 0.006368 | 1× |
| `960 / batch4` | **0.008070** | **0.016139** | **2.5×** |

结果写在 `experiments/_noise_summary.json` 的 `groups["imgsz=960|batch=4"]`，
`agent` 判 KEEP 时会自动用这一组。

**一个重要的推论**：`exp_0006` 的 0.45862 落在这三次的分布里（均值 0.4532，+0.005 < 1σ）。
所以它是**可信的**，不是「运气好跑出来的高点」。后续实验追不上它，是因为
模型提的方向（`degrees 10`、`freeze 10`）确实有害 —— 不是阈值冤枉了它们。

**三点值得注意**

1. **sigma 0.0032 远低于上游要求的 0.01**，说明 600 秒预算在这张卡上是充足的，
   不需要为了降噪而加预算（加预算会直接砍掉单位时间内的实验次数）。
2. **600 秒 vs 120 秒差距巨大**：同样配置，120 秒（5 epoch）只有 0.197，600 秒（23~24 epoch）
   到 0.407。短预算下模型还在快速爬坡期，测出来的差异多半是"随机种子运气"，
   这也反过来证明当初没选 120 秒是对的。
3. **`exp_0005` 的 `best_map50_95`（0.40925）比 `fitness_map50_95`（0.40385）高 0.0054**——
   已经接近 keep 阈值了。这正是"适应度取最后一行、不取历史最佳"的原因：如果取 max，
   这一行会虚高 0.0054，恰好是噪声量级，足以骗过一个本该被丢弃的配置。

## 十三、首轮实验结果（2026-11-10）

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

## 十三之二、第二轮：LLM 自动跑 5 轮（2026-11-10 16:12~16:48）

**结果：完成 5 轮，KEEP 0 次，失败 0 轮，用时 36:53。**
「失败 0 轮」是那次的主要目标（v0.2.1 修了截断），已达成。但结果里混着三个工程问题，
详见 v0.2.2 的记录。

| 轮 | 实验 | 模型声称的改动 | **实际跑的配置** | mAP50-95 | 判定 |
|---|---|---|---|---|---|
| 1 | `exp_0015` | `imgsz 960→1152` | 1152 / b4 | — | crash（30 秒 OOM） |
| 2 | `exp_0016` | `degrees 0→10` | 960 / b4 | 0.42967 | DISCARD |
| 3 | `exp_0017` | `warmup 3→1`（称 960 不变） | **640 / b8** | 0.39062 | DISCARD（**轮次作废**） |
| 4 | `exp_0018` | `cos_lr true` | 960 / b4 | — | crash（253 秒静默死） |
| 5 | `exp_0019` | `freeze 0→10` | 960 / b4 | 0.39189 | DISCARD |

**用 960 自己的阈值（0.016139）重判，结论不变**：

- `+degrees 10`：−0.02895 → 超过 1.8σ，**真实负向**。旋转增强在这个数据集上有害。
- `+freeze 10`：−0.06673 → 超过 4σ，**真实负向**。冻前 10 层效果最差。
- `warmup 3→1`：那一轮实际跑在 640 上，**结论作废**，值得在 960 上重试。

所以「KEEP 0 次」是**真实的科研结果**：模型提的这两个方向都不好。
这一轮真正的问题不在判定，而在「继承错了配置」和「环境把显存抢光了」。

**环境教训（比代码更重要）**

960/batch4 训练时整卡占用约 5.5 GB。`6-web` 的 `infer_worker` 是懒加载的：
浏览网页、点一次推理，它就会把模型加载进显存并常驻。训练期间被触发 =
5.5 + 1.7 > 8 GB，表现为「预检通过 → 第一个 batch OOM → 降 batch → 崩在主机内存」。
本轮 `exp_0020~0022` 连续三次崩就是这个原因，停掉 worker 后同样配置一次跑通。

从 v1.33.3 起不用手停：自动实验开始时会让推理进程退出，整场结束前不许它自己再起来，
结束后也不立刻加载，下次在网页上推理才按需重载。命令行连的是
`http://127.0.0.1:8060`，地址不同时设 `AUTOEXP_WEB_URL`。

```powershell
# 跑长实验前可以看一眼，确认推理进程已经退出
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# 空闲应约 800 MB used；> 1.5 GB 说明让出没生效（网页没开，或地址不对）
```

## 十四、第 4 阶段：LLM 决策层（`agent` 子命令）

前十三节是「人能自动跑实验」。这一节解决最后一个问题：**谁来决定下一次改什么参数**。

### 14.1 职责切分（核心，改代码前先读）

| 角色 | 由谁做 | 为什么 |
|---|---|---|
| 提出下一组参数 | 大模型 | 它的强项：读台账、找模式、提假设 |
| 判定 keep / discard | **代码，按 2σ 噪声阈值** | 它的弱项且危害最大。sigma=0.003184，而首轮 `+0.0037` / `+0.0001` / `-0.0031` 全在噪声内。把数字交给模型，它几乎必然把噪声讲成趋势，然后沿错误方向持续加码 |
| 能改哪些字段 | **代码，冻结白名单** | 见 14.2 |
| 写台账 / 更新 best.json | 代码 | 保证留痕格式与人手工跑时完全一致 |

**结论**：模型是「提议者」，不是「裁判」。这条边界不是保守，而是由实测数据逼出来的。

### 14.2 冻结字段

即使 `space.json` 声明了，`validate_proposal()` 也会拒绝修改：

| 字段 | 冻结原因 |
|---|---|
| `data` | **硬红线**。`space.json` 根本不校验它。允许改的话，理论上能把 `splits/pcb_tune.yaml` 换成 `pcb_holdout.yaml`，holdout 纪律当场作废，而台账上一切正常、看不出异常。代码里另有一道独立断言：`data` 必须落在 `splits/` 内**且**文件名含 `tune` |
| `budget_seconds` / `epochs` / `task` / `name` | 改了就不再是「同一套方法跑同样的时间」，横比作废 |
| `model` | 第一版刻意冻结：`n→s→m` 会显著改变 600 秒内跑得完的 epoch 数（m 也许只跑 3~4 轮），很容易整晚试大模型而毫无进展。想放开删掉 `FROZEN_TOP` 里的 `"model"` 即可，其它代码不用动 |
| `device` / `seed` / `deterministic` / `cache` / `workers` / `amp` / `patience` / `plots` | 影响可复现性或进程稳定性，不属于研究变量 |

另有两层校验：顶层字段白名单（只允许 `hypothesis` / `changed` / `reasoning` / `train`），
以及沿用 `space.json` 的枚举与区间校验。

> 实测中模型最容易犯的错不是改错参数，而是**把 `batch` 写成数字 `4`**。
> `space.json` 里它是字符串枚举 `["4","8","16","-1"]`，所以会被拒绝并重问一次。
> 这类「格式性错误」正是自动校验存在的意义。

### 14.3 提示词即策略

`prompts/experimenter.md` 里写的是：

- 角色（你是研究员，不负责判断成败）
- 输出 JSON 的严格格式
- **这台机器的硬事实**（T1000 8GB 强制 fp32、`batch=16@640` 必 OOM、960 只能配 batch=4、640 能跑 24 epoch / 960 只跑 12）
- **已证伪的方向**（`lr0 0.02` / `AdamW` / `close_mosaic 3` / 减弱增强）
- 方法论约束（单变量对照、连续无进展就换维度、不要重复试过的方向）

改策略**只改这个 md**，不碰 Python。这是上游 autoresearch 最重要的一课：
研究策略应该是一等公民，不是埋在代码里的常量。

### 14.4 轮数与时间上限

一轮 = 600 秒训练 + 约 60~90 秒启动与验证开销 ≈ **11 分钟**。

| 轮数 | 墙钟时间 |
|---|---|
| 10 | ≈ 1.8 小时 |
| **30（默认）** | **≈ 5.5 小时** |
| 50 | ≈ 9 小时，会先撞 8 小时时间上限 |

两个上限（`--rounds` / `--hours`）**先到者生效**，都必须有：
只限轮数时，某几轮卡在等显卡上会拖到天亮；只限时间时，模型调用连续失败会白烧一晚 API 费。

`--max-stale 5`：连续 5 轮没有 KEEP 就停。整晚都在噪声里打转说明搜索空间已到瓶颈。

### 14.5 命令

```powershell
cd 7-autoresearch-yolo

# 1) 先看提示词：不调 API、不跑训练、不花钱
python autoexp.py agent --dry-run

# 2) 测连通性（Key 只从环境变量读）
$env:AUTOEXP_API_KEY = "sk-..."
python autoexp.py llm-test

# 3) 开跑
python autoexp.py agent --rounds 30 --hours 8 --events

# 4) 换厂商：改 configs/llm.local.json（已 gitignore），不动入库文件
#    {"base_url": "http://localhost:11434/v1", "model": "qwen2.5"}
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rounds` | **30** | 最多跑几轮 |
| `--hours` | 8 | 时间上限（小时），0 表示不限 |
| `--max-stale` | 5 | 连续多少轮没有 KEEP 就停（**搜索到瓶颈**） |
| `--max-errors` | 3 | 连续多少轮拿不到有效提议就停（**模型侧故障**，不是瓶颈） |
| `--lock-wait` | 3600 | 显卡被网页端占用时最多等多少秒，0 表示不等 |
| `--dry-run` | — | 只打印第一轮完整提示词 |
| `--events` | — | 向 stdout 输出 `TRTFORGE_JSON` 事件（网页端需要） |

### 14.6 支持的模型接口

只支持 OpenAI 兼容的 `/chat/completions`：

| 服务 | `base_url` | 备注 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | 默认。中文任务、价格低 |
| OpenAI | `https://api.openai.com/v1` | |
| 本地 Ollama | `http://localhost:11434/v1` | 模型名写 `qwen2.5` 这类 |
| 本地 vLLM | `http://localhost:8000/v1` | |

兼容性处理：若上游不认 `response_format={"type":"json_object"}`（某些中转/vLLM 版本），
`llm.chat()` 会收到 400 后**自动去掉该字段重试一次**，不会因为「兼容接口不认这个开关」
就让整个功能不可用。`llm-test` 则完全不发这个字段——自检只验证「能不能通」，
不引入格式约束，避免把「模型不听话」误判成「配置错」。

### 14.7 API Key 的三条安全边界

1. 只存在 `6-web/runtime/autoexp_llm.json`（`6-web/runtime/` 已 gitignore）。
2. GET 接口**永不返回原文**，只给 `has_key` 与遮罩 `sk-****1a2b`；再保存时留空 = 不修改。
3. 传给子进程**只走环境变量**，绝不进 `argv`。命令行会出现在「运行中」条 tooltip、
   `runtime/logs/*.log` 和 WebSocket 事件里，Key 一旦进 argv 就等于贴到界面上。
   `llm.py` 另有 `mask_key()`，所有对外文本都过它；`load_config()` 会主动剔除
   `api_key` 字段，防止 Key 随配置字典流进日志与审计快照。

### 14.8 审计与成本

每轮写 `experiments/agent/round_NNNN.json`，含：模型提议原文、校验结果、`usage`
（prompt / completion tokens）、实际写入的参数、判定与理由、退出码、等锁次数。
出问题时第一件事就是拿它和 `results.tsv` 对照——「模型当时看到了什么」和
「实际改了什么」必须能对上。

**实测 token 量（2026-11-10，写这份文档时量的，不是估的）**

| 项 | 值 | 说明 |
| --- | --- | --- |
| system prompt | 2 354 字符（≈1.2 k token） | 来自 `prompts/experimenter.md`，**每轮完全相同** |
| user 消息（12 条历史） | 5 292 字符（≈2.0 k token） | 基线 + 搜索空间 + 台账摘要 |
| user 消息（跑满 30 条历史） | ≈9 800 字符（≈3.5 k token） | `history_limit=30` 封顶，**不会再涨** |
| **单轮输入** | **≈3.2 k → 5.5 k token** | 随台账变长而增，上限固定；v0.2.1 起含 `search_space` 取值表 |
| 单轮输出 | 推理型模型 600~1 800 token（含思考） | `max_tokens=4096`，截断时自动加倍到 8192 |
| **每轮 API 调用次数** | **1 次** | 提议越界/JSON 不合法时带理由再问 1 次，最多 2 次 |

30 轮的账：输入约 **13 万 token**、输出约 **2 万 token**（推理型模型按 §14.9 计价）。

| 供应商档位（举例，按你实际价目表为准） | 30 轮总花费 |
| --- | --- |
| DeepSeek `deepseek-chat`（≈¥2/M 入、¥8/M 出） | **≈ ¥0.25 ~ 0.35** |
| DeepSeek `deepseek-flash` 等推理型（思考 token 也计费） | **≈ ¥0.5 ~ 1.0** |
| OpenAI `gpt-4o-mini`（$0.15/$0.60 每 M） | ≈ $0.025（¥0.18） |
| OpenAI `gpt-4o`（$2.5/$10 每 M） | ≈ $0.42（¥3.0） |

所以准确说法是**几毛钱**，不是「几分钱」（旧版这句写乐观了）。

**真正的约束不是钱，是电和墙钟时间**：一轮 600 秒训练，整机按 150 W 算约 25 Wh，
30 轮 ≈ 0.75 kWh、电费 ¥0.45 左右——**API 费与电费在同一量级**，别指望靠省 API 省出什么。

**烧钱的刹车有三个**，都在代码里：

1. `--rounds` / `--hours` 先到者生效 → 最多 30 次提问。
2. `--max-stale`（默认 5）：连续 5 轮**跑完但没赢**才停（搜索到瓶颈）。
3. `--max-errors`（默认 3）：连续 3 轮**拿不到有效提议**就停（模型侧故障）。
   这两件事的含义完全不同，v0.2.1 之前共用一个计数器，会把「模型输出一直被截断」
   报成「连续 5 轮没有进展」——文案一错，排查方向就跟着错。
4. `llm.chat` 内建重试：`max_retries=3` 表示**最多 3 次尝试**（含首次），只针对 429 / 5xx
   这类可重试状态码，退避 2→6→18 秒。网络抖动不会变成新的一轮，也不额外提问。

### 14.9 推理型模型：`max_tokens` 是「思考 + 正文」的共用预算（2026-11-10 实测事故）

这是本功能上线后遇到的第一个真事故，写在这里是因为**它看起来完全不像它本身**。

**现场**：`max_tokens=900`，模型 `deepseek-flash`。30 轮里只有第 1 轮跑成，之后连续
5 轮报错并停机，日志长这样：

```text
[autoexp] 返回内容不是合法 JSON：找不到 JSON 对象：'{"hypothesis": "分辨率收益在 640→800→960 上单调递增...
[autoexp] 返回内容不是合法 JSON：返回内容为空
[autoexp] 提议被拒：train.imgsz = 1024 不在允许取值 [640, 800, 960, 1152, 1280] 内
[autoexp] 连续 5 轮没有进展，停止（省下无意义的 API 花费）
```

**真凶**（`experiments/agent/round_0004.json` 里存着，直接可查）：

| 轮次 | `completion_tokens` | `reasoning_tokens` | 正文实际拿到 |
|---|---|---|---|
| 1（成功） | 629 | 416 | 213 token → JSON 完整 |
| 3 第 1 次 | **900** | **900** | **0** → 空 |
| 4 第 1 次 | **900** | **900** | **0** → 半截 |

推理型模型把「思考」和「正文」算进**同一个** `max_tokens`。900 被思考吃干净，正文
一个字都不剩。所以「返回内容为空」和「JSON 突然半截」是同一个原因的两种表现。

**修法（v0.2.1）**

| 修的是 | 怎么修 | 为什么这样修 |
|---|---|---|
| 预算 | 默认 `max_tokens` 900 → **4096**，页面上可调 | 一轮 11 分钟，省 3000 token（约 0.002 元）而废掉整轮是最差的交易 |
| 截断 | `llm.chat()` 认出 `finish_reason == "length"` → **自动把预算翻倍重试**（上限 8192），仍不够才抛 `LLMTruncated` | 截断重发同样的请求只会得到同样的半截；这不是网络抖动，不该退避等待 |
| 报错文案 | `LLMTruncated` 单独一个异常类型，消息直接写「被 max_tokens=N 截断：其中推理占 X，正文只有 Y，把预算调到 4096 以上」 | 旧文案「不是合法 JSON」会让人去改提示词、换模型，方向完全错 |
| 空正文 | 空正文不再塞进对话历史（部分兼容接口会对空 assistant 内容返回 400） | 否则「重问一次」变成「必然失败一次」 |
| 判据 | `stale`（没赢）与 `errors`（没提议）分开计数，`--max-errors` 默认 3 | 5 轮全是模型侧故障，却报「连续 5 轮没有进展」，等于把配置问题伪装成科研结论 |
| 自检 | `llm.ping()` 捕获 `LLMTruncated` 并仍报「连通正常」 | 自检用 16 token，推理型模型必然截断；若因此报失败，用户会以为自己 Key 坏了 |

**怎么一眼看出是这个问题**：看 `experiments/agent/round_NNNN.json` 的 `usages`，
若 `completion_tokens == max_tokens` 且 `reasoning_tokens` 接近 `completion_tokens`，
就是预算不够，别去怀疑提示词。

### 14.10 模型看不到取值 = 白烧轮数

同一场事故里还有第二个坑：`build_agent_context()` 早期只给 `allowed_train`
（一串**键名**），没给取值。模型只能凭经验猜，于是提出 `imgsz=1024`、`batch=2`——
这两个值不在 `space.json` 的 enum 里，必然被拒，重问一次又被拒，整轮作废。

现在 user 消息里带完整的 `search_space`（只含可改的键，冻结项不出现）：

```json
"search_space": {
  "train": {
    "imgsz": {"enum": [640, 800, 960, 1152, 1280]},
    "lr0":   {"min": 0.0005, "max": 0.02},
    "batch": {"enum": ["4", "8", "16", "-1"]}
  }
}
```

代价是每轮多约 1.5 k 输入 token（≈¥0.003）；收益是少废掉一整轮（11 分钟 + 电费）。
顺带把 `_comment` 这类元数据键滤掉了——旧版 `allowed_top_level` 里赫然写着一个
`_comment`，等于告诉模型「你可以设一个 `_comment` 字段」。

## 十五、版本记录

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
- 跨进程互斥：网页端持锁时 `autoexp.py run` 返回退出码 5 且不建实验目录；
  autoexp 持锁时 Web 的 `_reject_if_busy()` 返回 409（见 §9）
- 生产 launcher 零改动：`git diff yolo-stable-before-autoexp -- 1-python-yolo/launcher` 为空

### v0.1.1 (2026-11-10)

跨进程 GPU 互斥（配合 `6-web/backend/gpulock.py`）。

| 变化 | 说明 |
|---|---|
| 新增 | `load_gpulock()`：复用 Web 那份锁模块，取不到则降级为不做互斥 |
| 新增 | `run` / `holdout` 开跑前查锁；被占用时退出码 5，不写台账、不建实验目录 |
| 变化 | `run` 的锁检查前移到「分配 exp_id」之前，避免被拒还留下空目录 |
| 修复 | 存活探测不再用 `os.kill(pid, 0)`：该调用在 Windows 上会**真的终止对方进程** |

**踩到的坑（重要）**：`os.kill(pid, 0)` 在 POSIX 上是只读探测，但 CPython 的
Windows 实现把非控制台信号的 sig 直接交给 `TerminateProcess(handle, sig)`。
写成 `os.kill(pid, 0)` 等于「以退出码 0 杀掉那个进程」。实测确认：本进程调
`os.kill(os.getpid(), 0)` 后无声退出。现改为 Windows 走
`OpenProcess` + `GetExitCodeProcess`，POSIX 仍用 `os.kill(pid, 0)`。

### v0.2.0 (2026-11-10) — 第 4 阶段：LLM 决策层

一句话：**从「我告诉你改什么」变成「模型提议、程序裁判、连跑一夜」**。

| 分类 | 变化 | 老版本行为 → 新版本行为 |
|---|---|---|
| 新增 | `agent` 子命令：提议 → 校验 → 训练 → 判定 → 记账，全程无人值守 | v0.1.x 每轮都得人改 `configs/*.json` 再手动 `run` → 现在 `agent` 自己循环 |
| 新增 | `llm.py`（零依赖，`urllib` 直连 OpenAI 兼容接口） | 无 → 支持 DeepSeek / OpenAI / vLLM / Ollama，`response_format` 不被支持时自动重试纯文本 |
| 新增 | `prompts/experimenter.md` 当选「策略文件」 | 策略埋在代码里 → 改 markdown 就能换研究方向，不动代码 |
| 新增 | `configs/llm.json`（无密钥的默认端点） | 无 → 入库文件只放 `base_url` / `model`，Key 走环境变量或 `6-web/runtime/` |
| 新增 | `--rounds` 默认 **30**（≈5.5 小时）、`--hours` 默认 8、`--max-stale` 默认 5 | 无轮数概念 → 三个上限先到者生效，避免跑到天亮 |
| 新增 | 判定权在代码：`fitness - best > 2σ` 才 KEEP | 无自动判定 → 模型说「显著提升」不算数，只有差值说话 |
| 新增 | 冻结字段 + `data` 断言（禁止指向 `pcb_holdout.yaml`） | 无 → 模型改不动 `data` / `epochs` / `budget_seconds` / `seed` 等，也不可能偷看 holdout |
| 新增 | `--events` 输出 `TRTFORGE_JSON {"type": "agent"}` | 无 → 网页端能实时看到「提议 / 判定 / 进度」，且与训练页事件（`type: "event"`）互不干扰 |
| 变化 | `run` 的锁检查支持**锁委派** | Web 后端持锁时，子进程一律看不懂、退出码 5 → 现在后端把自己 pid 通过 `AUTOEXP_GPU_LOCK_PID` 传下去，只有「pid 恰好等于持有者」才跳过；别的进程持锁照样退出码 5 |
| 修复 | `run` 的结果状态改由 `derive_result_status()` 统一判定 | `--events`（tee）分支下 `result.json` 缺失时用了未定义的 `proc` → `NameError`，表现为「训练跑完了台账却没记录」；现只依赖 `returncode`，并抽成可单测的函数 |
| 修复 | `llm.load_config()` 优先级重排 | 环境变量压过 `configs/llm.json` 而 **显式指定被忽略** → shell 里遗留的 `AUTOEXP_MODEL` 会让页面设置静默失效；现为「内置默认 ← 入库文件 ← `llm.local.json` ← 环境变量 ← 本次显式指定」 |
| 修复 | 锁委派时的日志文案 | 被委派时也打印「未找到 gpulock.py，本次不做互斥」→ 会让人去排查一个不存在的故障；现区分「后端持有」与「真的没装模块」 |

**安全边界（新版本才有）**

- `data` 字段做路径断言：解析后若落在 `pcb_holdout.yaml` 上直接拒绝，防止用验证集调参。
- Key 只走环境变量传入子进程，不进 `argv`；`load_config()` 主动剔除 `api_key` 字段。
- 进程被停时必须杀整棵树（`autoexp.py agent` → `_exec` → launcher → PyTorch），
  只杀父进程会让界面显示「已停止」而显卡仍被占着。

**已验证**

```powershell
python run_all_tests.py       # 42 项，不跑训练，约 3 秒
```

- `derive_result_status()`：`ok→done` / `done` 不变 / 非零退出 `crash` / 保留子进程自报的 `oom`。
- `_exec_tee()`：真实子进程的事件行同时进 stdout 与 `autoexp.log`，退出码原样返回。
- 真实 tee 链路：`cmd_run(tee_events=True)` + 120 秒预算跑完 `exp_0013`，`status=done`、
  mAP50-95=0.19586，证明「网页端那一轮也能正确记账」（该冒烟行事后已从台账移除）。
- 网页端：`/api/autoexp/*` 13 项接口测试通过；`agent` 页签渲染、轮数默认 30、
  阈值显示 0.006368（σ=0.003184）、台账 12 条均正确。

**已知限制**

- `--rounds` 一轮约 11 分钟，30 轮要 5.5 小时，中途想看结果只能靠网页端日志与台账。
- 主机的**内存**（不是显存）也会成为瓶颈：`exp_0012` 在 imgsz 640 / batch 8 下跑了
  624 秒后因 OpenCV 分配失败崩掉，台账如实记为 `crash`。这不是 autoexp 的 bug，
  但多轮无人值守时应该把 `workers` 留小一些。

### v0.2.1 (2026-11-10) — 修实测事故：推理型模型把输出预算吃光

一句话：**`max_tokens` 是「思考 + 正文」的共用预算，900 会让正文一个字都不剩。**

首个真实 `agent` 长跑（30 轮计划）跑到第 2 轮就开始全体报错，5 轮后停机：
第 1 轮成功（`reasoning_tokens=416`），第 3、4 轮 `completion_tokens=900`
且 `reasoning_tokens=900`——思考独占预算，正文剩 0，也就是「返回内容为空」或 JSON 半截。

| 分类 | 变化 | 老版本行为 → 新版本行为 |
|---|---|---|
| 修复 | `max_tokens` 默认 900 → **4096**（`configs/llm.json` + `llm.DEFAULTS`） | 900 被思考全部吃掉、正文 0 token → 4096 留出足够正文空间 |
| 新增 | `llm.chat()` 识别 `finish_reason == "length"` 并**自动把预算翻倍重试**（上限 `MAX_TOKEN_CAP=8192`） | 截断的半截 JSON 直接丢给上层 → 先加预算重来，加满仍不行才报错 |
| 新增 | `LLMTruncated` 异常类型 + `_completion_stats()` / `_truncation_message()` | 截断和「模型不会写 JSON」共用「返回内容不是合法 JSON」→ 现在直接说「被 max_tokens=N 截断，其中推理占 X、正文 Y，请调到 4096 以上」 |
| 变化 | `chat()` 返回值新增 `finish_reason` / `reasoning_tokens` / `max_tokens` | 只返回 `content`，事后无从判断为什么短 → 审计文件能记下真实原因 |
| 修复 | 空正文不再写入对话历史 | 空 assistant 消息会让部分兼容接口返回 400，使「重问一次」变成「必然失败一次」 |
| 修复 | `stale` 与 `errors` 分开计数，新增 `--max-errors`（默认 3） | 5 轮模型侧故障被报成「连续 5 轮没有进展」→ 现在明说「不是搜索瓶颈，是模型侧问题」并点名 max_tokens |
| 新增 | `evaluate_stop()` 纯函数（可单测） | 三个刹车挤在循环里的 `if` → 抽出来，`deadline` / `errors` / `stale` 优先级可断言 |
| 修复 | 上下文补 `search_space`（带 enum / min / max） | 只给 `allowed_train` 键名，模型只能猜 → 提出 `imgsz=1024`、`batch=2` 必然被拒 |
| 修复 | `_comment` 等元数据键不再进 `allowed_top_level` / `search_space` | `allowed_top_level` 里写着 `_comment`，等于邀请模型去设一个注释字段 |
| 修复 | `ping()` 捕获 `LLMTruncated` 仍报「连通正常」 | 自检只有 16 token，推理型模型必然截断；若报失败，用户会以为 Key 坏了 |
| 新增 | 网页端「模型接口」增加 **max_tokens** 输入框 + 「连续失败就停」输入框 | 只能改配置文件 → 页面上就能调，且开跑时把实际值打进日志与事件 |
| 变化 | 前端日志新增 `OUTPUT_TRUNCATED` 专属渲染与 `agent_stop` 事件 | 所有失败都显示 `[调用失败]`，看不出该改什么 → 截断直接提示「把 max_tokens 调到 4096 以上」 |

**已验证**

```powershell
python run_all_tests.py                     # 47 项（v0.2.0 是 42 项），约 3 秒
python 6-web\backend\tools\test_autoexp_api.py   # 15 项（原 13 项）
node   6-web\frontend\tests\test_autoexp_fmt.mjs # 27 项
node   6-web\frontend\tests\test_runsbar.mjs     # 28 项
```

新增用例直接对着这次事故：

- `llm.chat 在 finish_reason=length 时自动加预算`：假 transport 第一次返回 `length`，
  断言第二次请求的 `max_tokens` 更大、最终内容完整。
- `预算加到上限仍被截断时，报「被截断」而不是「不是合法 JSON」`：断言消息含
  `max_tokens` / `截断` / `4096`，且**不含**「不是合法 JSON」。
- `llm.ping 遇到推理型模型截断时仍报「连通」`：防止修好截断后把自检弄坏。
- `上下文必须给出允许的取值，不只是键名`：断言 `imgsz` 的 enum 含 960、不含 1024。
- `停止判据：errors 与 stale 必须分开报`：断言模型侧故障的文案里不出现「没有进展」。
- `max_tokens 可保存、会回显，并且太小/太大都被拒绝`（后端）。

**已知限制**

- 4096 是按 `deepseek-flash` 实测的（思考约 400~900 token）。换成思考更长的模型时，
  可能每次都要走一次「翻倍重试」，即单轮花两次请求的钱。看 `round_NNNN.json` 的
  `usages` 若经常出现 `completion_tokens == max_tokens`，就该把默认值直接调大。
- 截断无法「抢救」半截 JSON：缺 `}` 的提议在语义上可能也缺 `train` 字段，
  强行补括号会得到一条「看起来改了什么、其实没改」的提案，比失败更危险。

### v0.2.2 (2026-11-10) — 修「5 轮全 DISCARD」暴露的三个问题

**背景**：v0.2.1 修完截断后跑了一场 5 轮，模型侧 0 失败（目标达成），
但结果里混着三个与模型无关的问题，全部修掉。

**1. 继承 bug：模型以为在 960 上做实验，程序实际跑在 640（真 bug）**

- 现象：`round_0003` 的 `changed` 写「imgsz 960 / batch 4 不变」，
  `resolved_train` 却是 `imgsz=640, batch=8`。
- 原因：`cur.json` 只写 `proposal.train`，而 `run` 是用 `base.json` 合并的。
  模型没写的键一律落回基线。
- 修法：
  - `cmd_agent` 写 `cur.json` 时，`train = {**best.json 的 train, **本轮提议}`；
  - `_ask_proposal` 新增 `inherit_train`，校验与 `resolved` 都基于「最优 + 本轮改动」；
  - 每轮审计记 `inherited_from`，事后能查到起点是哪个 exp。
- 影响：`exp_0017`（640 + warmup1 = 0.39062）的结论作废，需要在 960 基座上重试。

**2. 噪声底按分辨率分组（这一条最后被实测部分推翻，见下）**

- 现象：960 上三次实测 0.3919 / 0.4297 / 0.4586，跨度 0.067；
  而判定用的 σ=0.0032 是**在 640 上**测的（640 跑 24 epoch，960 只跑 12）。
- 当时的推断：960 的抖动远大于 640，0.4586 可能是幸运高点。
- 修法：
  - `noise` 结果按 `groups[imgsz|batch]` 分列保存；
  - 新增 `noise_threshold_for()`，优先用该组实测 σ，其次用台账里**配置完全相同**
    的重复实验的离散度，都没有才退回全局；
  - 退回全局时 `calibrated=false`，`judge_round` 会把这句写进判定理由，
    `agent_verdict` 事件和审计里也有 `threshold_source` / `threshold_calibrated`。

**实测结论（2026-11-10 18:32，960/batch4 连跑 3 次）**

| 次 | exp | mAP50-95 | 说明 |
| --- | --- | --- | --- |
| 1 | exp_0023 | 0.45984 | 11 epoch |
| 2 | exp_0024 | 0.45789 | 11 epoch |
| 3 | exp_0025 | 0.44183 | 12 epoch |

**sigma = 0.008070，keep 阈值 2σ = 0.016139**（640 的对应值是 0.003184 / 0.006368）。

所以 960 的抖动是 640 的 **2.5 倍**，不是当初推断的 6 倍。更重要的是：
0.45862（exp_0006）落在这三次的分布里（均值 0.4532，+0.005 < 1σ），
**它是可信的，不是幸运高点**。用正确的阈值重判那 5 轮：

| 实验 | mAP50-95 | 与 0.45862 差 | 960 阈值 0.0161 下的判定 |
| --- | --- | --- | --- |
| exp_0016（+degrees 10） | 0.42967 | −0.02895 | DISCARD，**真实负向** |
| exp_0019（+freeze 10） | 0.39189 | −0.06673 | DISCARD，**真实负向** |
| exp_0017（实跑 640） | 0.39062 | — | 配置继承错误，整轮作废 |

**结论修正：KEEP 0 次在正确阈值下依然成立。** 模型提的两个方向（旋转增强、
冻结层）确实有害，判定逻辑没有冤枉它们。分组阈值的价值不在于「让 KEEP 变得可能」，
而在于**两个方向都可能出错**：用 640 的 0.0064 判 960，既会把 −0.029 判对，
也会把一个 +0.01 的真实噪声误判成 KEEP（960 的 1σ 就有 0.008）。

**4. 唯一实测失败：降 batch 拦截漏了命令行路径**

- 现象：测 960 噪声时日志出现 `Reducing to batch=2`，但程序继续跑。
- 原因：`_exec_tee` 只在 `--events`（网页端）路径被调用；命令行 `run` / `noise`
  走 `subprocess.run`，根本不经过这个拦截。
- 修法：`_run_locked` 无条件走 `_exec_tee`。拦截必须是确定的，
  不能取决于调用方有没有传 `--events`。

**仍然需要人工注意的环境问题**

960/batch4 训练时整卡占用约 5.5 GB。`6-web` 的 `infer_worker` 是**懒加载**的：
浏览网页、点一下推理，它就会把模型加载进显存并常驻。训练期间一旦它被触发，
5.5 GB + 1.7 GB 就会把 8 GB 撑爆，表现为「预检通过、跑到第一个 batch OOM，
然后 Ultralytics 降 batch、再崩在主机内存上」（exp_0020~0022 连续三次就是这样）。

跑长实验前先确认：

```powershell
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# 空闲应接近 800 MB used / 7.2 GB free；若 used > 1.5 GB，先停掉网页端的推理 worker
```

**3. 硬拦截 Ultralytics 静默降 batch**

- 现象：`exp_0015`（1152/batch4）30 秒内 OOM，日志里出现
  `Reducing to batch=2 and retrying`，然后主机内存被吃光崩溃。
  以前只靠事后比较 `batch_req` / `batch_eff`，等发现时已经白跑一整轮。
- 修法：
  - `_exec_tee` 逐行匹配 `Reducing to batch=N`，命中即写
    `runs/autoexp/<exp>/batch_downgraded.json` 并**终止子进程**；
  - 状态记为 `batch_downgraded`（新增），`fitness` 留空，不计 crash，
    不污染 `current_best`；
  - `space.json` 的 `imgsz` 枚举去掉 1152 / 1280（本机实测必 OOM），
    提示词同步：「1152/1280 已移除，换大卡后可加回」。

**新增测试（6 个）**

- `判定：真实负向要说「下降」，不能一律说「无法与随机波动区分」`
- `判定：阈值未校准时必须在理由里说清楚`
- `噪声底分组：优先用实测 groups，其次用配置相同的重复实验，最后才借全局`
- `噪声底分组：台账里配置完全相同的重复实验可作为校准依据`
- `Ultralytics 的降 batch 提示必须能被识别（这是中止训练的触发条件）`
- `提议继承：未写的键要取「当前最优」，不是「基线」`

**验证结果**：`7-autoresearch-yolo` 53 / 53 通过，`test_autoexp_api.py` 15 / 15 通过。
实测噪声数据见本节「实测结论」：960 的 σ = 0.008070（2σ = 0.016139），
比 640 的 0.003184 大 2.5 倍，且 exp_0006 的 0.45862 落在其分布内 ——
**KEEP 0 次是真实的科研结果，不是阈值冤枉了模型**。

### v0.2.3 (2026-11-11) — 显存预检改实测峰值表并自校准，AMP 记录实际生效值

**一句话**：预检不再用手拟合的公式猜，改成「跑过就记下来」；台账新增
`peak_vram_reserved_mb` 与 `amp` 两列，记的是**实际**吃到的显存和**实际**生效的精度。

**为什么改**

- 旧公式 `2600 + 240 × batch × (imgsz/640)²` 与台账脱节：`640/batch=8` 实测峰值
  1769 MB，公式估 4520 MB；`960/batch=4` 公式估 4760 MB 就放行，实际仍然 OOM。
- 同一套判定在 `6-web` 里还有一份实现，两处已经给出互斥结论。

**改了什么**

| 事项 | 变化 |
| --- | --- |
| 判定实现 | `preflight_vram()` 委托 `6-web/backend/vram.py`（找不到时退回同一套公式并说明「按公式估算」）；`need_mb` 参数保留旧语义供测试用 |
| 实测表 | `experiments/vram_samples.json`，键 `imgsz x batch x 精度`，值取历次最大 `max_memory_reserved` |
| 自校准 | 训练结束自动写回；样本键带精度，避免「别处 fp16 的样本」放行「本机 fp32 的组合」 |
| 台账 | `RESULTS_HEADER` 新增 `peak_vram_reserved_mb`、`amp`（按列名迁移旧行，留 `.bak`） |
| 实验记录 | `exp_NNNN.json` 追加 `amp_effective / precision / peak_vram_reserved_mb`（开跑前不知道的事实） |

**本机实测（T1000 8GB，fp32）**：`640/batch=8` → `measured` 2852 MB；
`640/batch=16` → `scaled` 5304 MB；`640/batch=4` → `capped` 2852 MB；
`960/batch=8` → `formula` 8000 MB。四档意义见 §8.1。

**验证**：`run_all_tests.py` 62 / 62 通过；真机跑 1 epoch，`amp_requested=true`
但 `amp_effective=false`（T1000 在 AmpBlacklist 里），`reserved` 2132.0 MB
自动写入 `640x8xfp32`。

### v0.2.5 (2026-11-11) — 数据体检门禁与按组切分：不让同一块底板同时进训练与验证

**一句话**：`split` 新增 `--group-by`，让同一块底板（同一前缀 / 同一目录）只能落进一个划分；
`run` 新增 `--qc-gate`，开跑前体检标注，存在 error 级问题时拒绝启动；另提供独立 CLI `labelqc.py`。

**为什么改**

- PCB 数据 692 张其实只有 10 块底板，每块拍了「同位置不同缺陷」的几十张。按图片随机切分
  时同一块底板必然横跨 train 和 val，val 指标反映的是「记住了底板」，不是「认出了缺陷」。
- 标注坏掉（字段数不对、类别越界、坐标越界）时 Ultralytics 往往只打一行摘要就继续训练，
  或者直接崩在很后面，一轮几分钟到几十分钟白花。

**改了什么**

| 事项 | 变化 |
| --- | --- |
| 共享实现 | `load_web_module("datasetqc")` 加载 `6-web/backend/datasetqc.py`（与 Web 端同一份，理由同 `vram` / `fingerprint`） |
| 按组切分 | `split --group-by {none,prefix,dir}`：`prefix` 取文件名首个 `_` 前的前缀，`dir` 取父目录名；同组整组进同一侧，用「补最大缺口」贪心近似目标比例 |
| 不拆组的代价 | 组数 <2 直接退出码 2 并提示可退回 `--group-by none`；组过大时比例会偏离，日志写明实际组数 |
| 切分清单 | `_manifest.json` 新增 `group_by`、`group_signature`、`groups`，事后可证「这批切分是按组做的」 |
| 数据门禁 | `run --qc-gate`：只拦 `error` 级；`warn`/`info` 只打印；体检本身异常则放行（不让检查变成阻塞），退出码 **6** |
| 体检耗时 | 默认不检查：解码整份数据 692 张约 1~10 秒（有缓存更快），而一轮实验只有几分钟 |
| CLI | 新增 `labelqc.py`，人类可读 / `--json` 两种输出，`--fail-on {none,error,warn}` 控制退出码 |

**退出码（接 CI 时按这个判断）**

| 码 | 含义 |
| --- | --- |
| 0 | 体检完成且未触发 `--fail-on` 阈值 |
| 1 | 触发 `--fail-on`（存在对应级别的问题） |
| 2 | 输入无效（路径不存在、无法识别为数据集） |
| 6 | `run --qc-gate` 拦下：存在 error 级标注问题，本次未启动 |

**命令**

```powershell
# 体检（只读），有 error 就返回非 0
D:\anaconda3\envs\dpenv\python.exe labelqc.py "D:/test/deeplearning/dp-tool/PCB data/PCB缺陷检测" --fail-on error
# 机器可读
D:\anaconda3\envs\dpenv\python.exe labelqc.py splits/pcb.yaml --json

# 按底板前缀重新切分（同前缀只进一侧）
D:\anaconda3\envs\dpenv\python.exe autoexp.py split --group-by prefix

# 开跑前门禁
D:\anaconda3\envs\dpenv\python.exe autoexp.py run --config experiments/cur.json --qc-gate
```

**验证**：`run_all_tests.py` **68 / 68** 通过（新增 3 条：按组切分不拆组、
`labelqc.py` 退出码、`--qc-gate` 拦住坏数据）。真机：`PCB data/PCB缺陷检测` 体检得到
10 组、全部跨划分、涉及 689 张（另 3 张自成一类）；按 `prefix` 切分后 10 组各自只在一个划分里。

### v0.2.5 (2026-11-11) — 过杀进台账，KEEP 不再允许拿过杀换 mAP

**一句话**：`results.tsv` / `holdout.tsv` 新增 `fp_per_image`（每图误报框数），
`judge_round()` 在 mAP 提升超过噪声阈值、但过杀明显恶化时判 `DISCARD`。

**为什么改**

- 只看 mAP 的自动搜索会稳定地往「多报框」方向漂：多报一个高分框会抬高 recall，
  mAP 可能涨，但产线上每多报一个就要停一次线。
- `judge_round` 是全流程唯一有权说「这个改动算成功」的地方。如果它不知道过杀，
  等于把成本项完全排除在决策之外。

**改了什么**

| 事项 | 变化 |
| --- | --- |
| 台账列 | `RESULTS_HEADER` / `HOLDOUT_HEADER` 新增 `fp_per_image`（按列名迁移，旧行补空，留 `.bak`） |
| 近似换算 | `fp_per_image_from_pr(P, R, instances, images)`：`TP = R×GT`、`FP = TP×(1-P)/P`、`fp = FP/图片数` |
| 缺项语义 | P/R/GT/图片数任一缺失、或 P≤0、R≤0 时返回 **`None`**（不知道），不是 0 |
| KEEP 拦截 | `judge_round(..., fp_per_image, incumbent_fp_per_image)`：只在**两边都有值**时生效 |
| 容差 | `max(0.01, 基线×5%)`：低于这个量级的差异在近似算法下不可信，不当成「变差」 |
| 理由文案 | 被拦时写明「过杀从 x 升到 y 框/图（容差 z）——不拿过杀换 mAP」，进台账和界面 |
| 不误伤 | 任一侧缺数时退回原 2σ 规则；没有超过噪声阈值的提升也不会被过杀理由误导 |

**字段说明**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `fp_per_image` | float\|null | 每张验证图平均误报框数（**近似值**，见下方边界） |
| `incumbent_fp_per_image` | float\|null | 上一条 KEEP 记录的过杀，用于比较 |
| `val_instances` / `val_images` | int | 换算用的分母，台账里单独留一份便于复核 |

**边界（写清楚以免被当成验收数字）**

- Ultralytics 的 P/R 取自 **best-F1 conf**，不是固定 conf，也不区分良品图。
- 因此这里的 `fp_per_image` 只用于自动搜索里拦「明显恶化」，**不能当产线验收数字**；
  验收数字要用 Web 验证页的 `operating_points` 曲线（`tools/val_engine.py`）。
- 没有良品图时不做良品误报统计；曲线里的 `ok_fp_rate` 为 `null` 表示「没测过」。

**验证**：`run_all_tests.py` **71 / 71** 通过（新增 3 条：P/R 反推与缺项语义、
过杀拦截与容差不误伤、两张台账都有该列）。

### v0.2.4 (2026-11-11) — 台账记录数据集指纹，holdout 数据变了会被发现

**一句话**：每轮实验开跑前算一次数据指纹，写进 `exp_NNNN.json` 与 `results.tsv`
的 `data_fp` 列；`splits/_manifest.json` 也记下切分时的指纹，holdout 验收前自动核对。

**为什么改**

- 台账只记参数与指标。换了数据集之后，「`0.4109` → `0.4150`」到底是参数赢了
  还是数据变了，无法判断，`best.json` 的「当前最佳」也不再可比。
- `splits/_manifest.json` 的 `signature` 只记**文件名列表 + seed**。holdout 的图
  被改过标注后它一点不变，于是「验收通过」的结论悄悄失效——这是 holdout 纪律
  上最后一个没堵住的口子。

**改了什么**

| 事项 | 变化 |
| --- | --- |
| 共享实现 | 新增 `load_fingerprint()` / `dataset_fingerprint()`，加载 `6-web/backend/fingerprint.py`；找不到时标 `unavailable`，不阻断实验 |
| 实验记录 | `exp_NNNN.json` 新增 `dataset` 块（`algo` / 三个指纹 / `counts` / `error`），开跑前写 |
| 台账 | `RESULTS_HEADER` 新增 `data_fp`（紧跟 `amp`；按列名迁移，旧行补空，留 `.bak`） |
| 最佳配置 | `best.json` 的 `evidence.dataset_fingerprint` 记录这次 KEEP 依据的数据短码 |
| 切分清单 | `_manifest.json` 新增 `data_fingerprint` 与 `holdout_fingerprint`（holdout 的图不在 tune 的 train/val 里，必须单独记） |
| 漂移检查 | `holdout` 子命令开跑前调 `check_split_drift()`；数据变过时明确警告「之前的验收结论不可沿用」 |
| 内存占用 | 没有。指纹只读 `stat` 与标注文本，650 张约 0.14 秒 |

**指纹字段（读台账时按这个解析）**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `algo` | string | `v1`；不同版本永不相等 |
| `dataset_fingerprint` | string(16) | 整批指纹，跨这一列不同的行不可直接横比 |
| `short` | string(8) | `results.tsv` 的 `data_fp` 就是它 |
| `split_fingerprints` | object | `{train, val, test}`，用于判断「只改了 val 吗」 |
| `label_fingerprint` | string(16) | 标注文本指纹（不含路径） |
| `counts` | object | 各 split 图片数 |

**验证**：`run_all_tests.py` **65 / 65** 通过（新增 3 条：`data_fp` 列迁移、
与 Web 端指纹一致、无记录时漂移检查放行）。真机：`splits/pcb_tune.yaml`
在 autoexp、Web 接口、训练目录三处一致得到 `5a9749c58c8a0ea9`。

