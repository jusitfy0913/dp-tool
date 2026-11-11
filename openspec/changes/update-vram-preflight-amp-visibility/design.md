## Context

显存预检是「别让 Ultralytics 静默把 batch 减半」的最后一道闸门。它出错有两种后果：
放行必然 OOM 的组合（这一轮实验白烧，且 batch 被偷偷改成 8，与台账其它行不可比），
或者拦掉其实能跑的组合（用户以为软件坏了）。

当前的公式 `2600 + 240*batch*(imgsz/640)^2` 是 v1.33 手工拟合的，实测已经证明它在
960/batch=4 上放行了会 OOM 的组合。与此同时，本机 T1000 在 Ultralytics 的 AMP 黑名单里，
训练实际走 fp32——显存占用比常规高得多，而公式里没有这个变量。

约束：

- Web 服务进程通常**没有装 torch**（它跑在系统 Python 上），所以显存探测不能依赖 `import torch`。
- autoexp 进程有 torch，能拿到 `max_memory_allocated / max_memory_reserved`。
- 不允许为了让 fp16 生效去 monkeypatch Ultralytics 的 `check_amp()`：那正是上游用来
  避免 NaN loss / zero-mAP 的保护。
- `results.tsv` 已有 `ensure_tsv()` 的按列名迁移能力，加列是安全的。

## Goals / Non-Goals

- Goals
  - 同一台机器上「上次 batch=8 能跑、这次同样组合被拦」这种自相矛盾不再出现。
  - 每次真实训练都自动积累一条实测样本，越用越准（自校准）。
  - 页面上能直接看出这次是 fp16 还是 fp32，以及为什么。
- Non-Goals
  - 不做精确的显存预测器（不是仿真器），只要「不误放行、少误拦」。
  - 不做 batch 自动搜索（那是 autoexp 的职责，不在本批）。
  - 不改 Ultralytics 的 AMP 行为，不强行开启黑名单显卡的 fp16。

## Decisions

- **Decision**: 实测样本以 `torch.cuda.max_memory_reserved()` 为主，`allocated` 仅作参考。
  - 理由：OOM 发生在**向驱动申请**的那一刻，PyTorch 的 allocator 已经预留的块
    （reserved）才是它真正占住的显存；`allocated` 会低估 30%~50%。
  - 备选：训练中轮询 `mem_get_info()` 取峰值——会引入一个后台线程，而且采样间隔
    错过瞬时峰值，成本高于收益。
- **Decision**: 样本键包含 `precision`（`fp32` / `fp16`）。
  - 理由：T1000 的 fp32 与 fp16 激活显存差近一倍。若共用键，一张卡上的 fp16 样本
    会把另一台机器的 fp32 组合放行。代价是表更稀疏，但无样本时回落到保守公式，安全。
- **Decision**: 只在精确命中时不外推；不命中时用「线性外推 + 单调上界」两条规则兜住。
  - 理由：纯精确命中在真机上立刻暴露问题——本机 640/batch=8 实测只吃 2132 MB，
    而公式说 batch=16（表单默认值）要 7400 MB，页面一打开就是「显存不够」。
    只做「向下调小」的建议更糟：公式在 batch=7 给 4700 MB > batch=8 的 2852 MB，
    照着建议调小反而越调越红。因此：
    - 有更小 batch 的样本 → 按 batch 线性外推。偏差方向**偏保守**：实测峰值里含
      固定的 CUDA context / workspace，按比例放大会把这部分重复计一遍；
    - 有更大 batch 的样本 → 用它作**上界**（显存占用随 batch 不减，这是物理事实，
      不是拟合），这样「batch 调小」永远不会让需求变大。
  - 备选：拟合固定项 + 每样本项（两个未知数、一个样本解不出来）；或直接信任公式
    （已经在真机上证明会误拦默认参数）。
- **Decision**: Web 训练页对预检**不放行时返回 409 而不是硬拦死**，前端给「仍然开始」。
  - 理由：用户可能刚手动释放了显存；也可能这张表里没有的组合被公式保守拦掉。
    留一个显式越权入口，比让人以为软件坏了强。autoexp 侧保持**硬拦**（它无人值守）。
- **Decision**: `amp` 覆盖只影响「传不传 `--amp`」，不试图绕过 Ultralytics 的检测。
  - 理由：`trainer.amp = torch.tensor(check_amp(model)) if args.amp else torch.tensor(False)`。
    黑名单显卡即使传 `--amp` 也会被 `check_amp()` 关掉，真要强制只能改上游代码，
    而那会把上游刻意规避的 NaN 风险放进来。页面上必须如实写「无法强制开启」。
  - 可覆盖的实际价值：已知不支持的机器传 `amp=false`，跳过开训前那次 3 次前向的检测，
    并把「本次是 fp32」写死进 args 快照，台账不再需要猜。

## Risks / Trade-offs

- 样本表被一次异常运行（例如同时有别的进程抢显存）污染成很大的峰值 → 之后一直误拦。
  - 缓解：样本只是「放行阈值」而不是唯一依据；拦住时会给出建议文案；可用环境变量
    `TRTFORGE_VRAM_SAMPLES` 指到临时文件重测，或手工删掉该键。
- 表里没有的组合回落到系数 300 的公式，比之前更容易拦 → 可能会多拦一些本来能跑的组合。
  - 缓解：409 带「仍然开始」；公式只在无样本时生效，跑一次就进表。
- 在 Web 侧记录样本意味着 `runners.py` 要认识 `vram` → 多一条模块依赖。
  - 缓解：只在一个 try/except 里调用，失败只打一行日志，不影响任务状态。

## Migration Plan

- `vram_samples.json` 不存在 → 视为空表，全部走公式，不报错。
- `results.tsv` 表头新增 `amp` 列 → 首次写入时由 `ensure_tsv()` 自动迁移旧行（补空），
  旧文件备份为 `results.tsv.bak`；`read_tsv()` 用 `DictReader`，缺列的行读出 `None`，由调用方容忍。
- 不需要回滚动作：删掉新文件即回到公式行为（`preflight_vram` 的签名与返回结构不变）。

## Open Questions

- 无。若将来要支持 bf16（`args.amp="bf16"` 会跳过 `check_amp`），把 `precision` 扩一个值即可。
