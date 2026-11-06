# 变更提案：v1.24.1 用内置权重验收提示式预标注

## Why

v1.24 把「提示式预标注」和「内置模型仓库」归档时，本机没有下载权重，因此这两条
需求只验证到"模型缺失时报错"这一半，**真正带上权重跑真图的路径一次都没走**。

v1.24.1 把 214.3 MiB 权重下载到位后用真图跑通，发现已归档需求在两处被证伪：

| 被证伪的承诺 | 实际情况 |
| --- | --- |
| 模型仓库「列出每个模型的体积」，用户据此看进度 | 清单里的体积是手估的（GroundingDINO 写 700 MB，实际 171.28 MB），进度条最高只能走到 24% 就"完成" |
| 模型仓库「多镜像自动切换」，保证 GitHub 不通时还能下 | 三条回退地址实测全部 404，主源超时后只是白等一次 |
| 「点提示生成多边形」「框提示生成多边形」 | 任意一次点/框提示都抛 `ValueError` → 500，因为提示点从 `clamp_points()` 出来是 numpy 数组，而 `build_prompt()` 写了 `points or []` |
| 同上（换机可移植性） | SAM encoder 输入布局写死 NCHW，本项目权重是 HWC rank-3，被 onnxruntime 拒收 |

这四条都不是"新功能没做"，而是**已归档的验收标准没被真正执行**。所以本次变更
不改任何行为设计，只把验收标准补严到"带内置权重跑通"这一层，并给每条加上可执行的判据。

## What Changes

- `Builtin Model Repository` 需求 **MODIFIED**：增加"体积必须与服务端实际字节一致""回退地址必须实测可达"两个场景，并要求进度能真正走到 100%。
- `Prompt Based Auto Labeling` 需求 **MODIFIED**：增加"必须能在内置权重上跑通"与"提示点输入容忍数组类型"两个场景，把"单测通过"和"真机能跑"区分开。

## Impact

- 受影响规格：`specs/web-annotation-workbench/spec.md`（两个需求）
- 受影响代码：`backend/modelzoo.py`、`backend/promptlabel.py`、`backend/textlabel.py`、`backend/onnxenv.py`（新增）、`backend/app.py`
- 受影响测试：`test_prompt_label.py` 49→63 项、新增 `test_onnxenv.py` 11 项；全量 29→30 个脚本
- 不改变对外接口：`/api/annotation/prompt-label` 的 `points`/`labels`/`box` 从"拒 `null` 回 422"放宽为"接受 `null` 并按空处理"，属于放宽而非破坏
