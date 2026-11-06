# Change: 提示式预标注（点/框/文本）+ 审核状态

## Why

v1.23 把预标注做成了"整工程跑一遍、人只做审核"，但**只做审核**这句话还不成立：

- **只有一种模型通道**：必须先训练出 `.engine` 才能预标注。新类别、少量样本、
  "先看看这类缺陷长什么样"的场景，只能回去标一批再训练，闭环转不起来；
- **只能出框，不能出轮廓**：检测框对"缺陷面积 / 有无破孔"这类判断不够用，
  多边形得人手点；SAM 这类提示式模型本来就是为"点一下出掩码"设计的；
- **审核没有"做过"的记号**：`A` 接受完，这张图和没审过的图在列表里长得一样，
  几十张来回切之后用户会忘记哪张已经过了一遍。`A`/`D` 只解决了"单张怎么快点过"，
  没解决"整个工程走到哪了"。

## What Changes

- **点框提示出掩码（SAM 通道）**：画布上加正点/负点、或拉一个框，一次 encode 多次
  decode，出多边形草稿。没有置信度（`score < 0`），按"未知"档呈现。
- **文本提示开放词表（GroundingDINO 通道）**：输入 `missing hole` 直接出框，
  **不需要训练**；可选再接 SAM 把框细化成多边形。置信度是真实分数，直接进分档着色。
- **内置模型仓库**：在标注页直接下载 SAM / GroundingDINO / BERT tokenizer，
  流式 + `.part` 续传 + 进度 + 取消，落在 `6-web/models/`（不进 git）。
- **审核状态**：`K` 标为已审核并跳到下一张未审核，`Shift+K` 取消；文件列表显示
  ✓（已审核绿 / 未审核灰），审核区显示 `已审核 12/64`。状态写进旁挂草稿文件的
  `review` 块（文件升到 `version: 2`，读写兼容 v1）。
- **草稿结构升级**：`annotation_drafts.py` 增加 `review` 块，`merge()` 只覆盖传入的
  图片草稿与 review 条目，其余不动。

**为什么审核状态也放旁挂文件**：`.tfanno` 由 Qt 版整体重写，任何未知键都会丢；
"审到哪了"和"草稿"一样是过程数据，生命周期与正式标注不同，必须住在同一个文件里
才能保证一次写入的原子性（否则会出现"草稿写了、审核状态没写"的半截状态）。

**为什么不复制 X-AnyLabeling 的代码**：它是 GPL-3.0，推理层与 PyQt 有耦合；
本项目只需要它的**思路与权重**（Apache-2.0 / MIT），代码按本项目风格重写，
不产生许可证传染。

## Impact

- 影响能力：`web-annotation-workbench`
- 影响代码：
  - `6-web/backend/modelzoo.py`（新增：内置模型清单、下载与进度）
  - `6-web/backend/promptlabel.py`（新增：SAM 点/框提示 → 多边形草稿）
  - `6-web/backend/textlabel.py`（新增：GroundingDINO 文本提示 → 框，可选 SAM 细化）
  - `6-web/backend/annotation_drafts.py`（改：`review` 块，version 2）
  - `6-web/backend/autolabeljobs.py`（改：`mode=engine|text`）
  - `6-web/backend/app.py`（新增 5 个接口，草稿接口带 `review`）
  - `6-web/frontend/index.html`、`css/style.css`、`js/api.js`、`js/annotate.js`
  - `6-web/backend/tools/test_prompt_label.py`、`test_text_label.py`、`test_modelzoo.py`（新增）
  - `6-web/backend/tools/test_annotation_drafts.py`（扩：v1→v2 兼容、review 合并）
- 不涉及：`.tfanno` 字段结构、Qt 版代码、导出格式、推理 DLL 契约、五级目录结构
- 向后兼容：没有新模型的工程行为与 v1.23 完全一致；旧 `*.drafts.json`（v1）自动读成
  「无审核状态」；新增接口全部是追加
- 已知取舍：SAM 的 mask 不提供可解释置信度，草稿 `score` 固定为 `-1`（未知档），
  避免把 `iou_pred`（常年 >0.8）染成绿色误导审核
