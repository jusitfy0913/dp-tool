## Context

「这个模型用的是哪批数据」目前靠人记。`args.yaml` 里的 `data:` 是一个绝对路径，而路径
指向的内容会变（重新标注、增删图、重划分），所以它不能作为身份。同时项目里已经有三个
各自独立的产物体系需要互相核对：

- Web 导出（`6-web/backend/dataset.py`）→ `dataset.yaml` + 图片/标签目录
- 训练 run（`1-python-yolo/launcher`）→ `args.yaml` + `results.csv` + 权重
- 自动实验（`7-autoresearch-yolo`）→ `results.tsv` + `exp_NNNN.json` + `best.json`
- 引擎（`trtexec`）→ 一个 `.engine` 二进制

约束：

- Ultralytics 只认 `dataset.yaml`，不能往里塞自定义字段（会被当未知键）。
  所以 manifest 必须是**旁挂文件**。
- `6-web` 后端进程跑在系统 Python 上，不能假设有 torch/ultralytics；指纹只能用标准库。
- 导出是同步接口（前端等结果），指纹必须够快，否则页面会转圈。5000 张的目标是 ≤ 2 秒。
- C++ DLL 的加载路径不能改，引擎元数据只能是它旁边的只读文件。
- 生产上存在大量没有指纹的历史产物，不能让它们变成报错。

## Goals / Non-Goals

- Goals
  - 任何两次数据「内容不同」的情况，指纹一定不同（改标注、增删图、重命名、重划分）。
  - 任何两份数据「内容相同」时（包括导出到不同目录、镜像到 splits/），指纹相同。
  - 从引擎一路能查回训练 run，再查回数据集指纹。
  - 界面上一眼能看出「这两行是不是同一批数据」。
- Non-Goals
  - 不做数据快照/版本库（DVC 的职责）。指纹只证明「是不是同一份」，不负责把旧的那份留住。
  - 不哈希图片内容（用户已确认）：读全量图片要几十秒，而绝大多数变更靠
    「路径 + 大小 + 标注」就能发现。
  - 不回填历史产物。当时的原始数据已不可得，补出来的指纹是假的。
  - 不改 `dataset.yaml` 的字段，不改 DLL 加载逻辑。

## Decisions

- **Decision**: 指纹算法 `v1`，以「split 内相对路径 + 文件字节数 + 标注文本哈希」为输入。

  ```text
  split_fp(split) = sha256("v1\n" + split + "\n"
                           + "".join(sorted(f"{rel}\t{size}\t{label_digest}\n")))
  dataset_fp      = sha256("v1\ndataset\n" + f"train:{fp_train}\nval:{fp_val}\ntest:{fp_test}\n"
                           + f"task:{task}\nnc:{nc}\n")[:16]
  label_fp        = sha256("v1\nlabel\n" + "".join(sorted(f"{rel}\t{label_digest}\n")))[:16]
  ```

  - 相对路径以**该 split 的 images 目录**为基准（不是绝对路径，也不是 yaml 目录），
    这样同一批数据导出到 `D:\out_a` 与 `E:\out_b` 指纹相同，镜像到 `splits/` 也相同。
  - `label_digest = sha256(标注文件原始字节)[:16]`，标注不存在记 `none`。
    用原始字节而不是解析后的结构：解析会吞掉格式错误，而格式错误本身就是数据变化。
  - 版本号进哈希前缀（`v1`）：以后换算法只会让新指纹与旧指纹不同，不会误判为同一份。

- **Decision**: 三个粒度都要，缺一不可。
  - `dataset_fingerprint` 回答「能不能横比」；
  - `split_fingerprints` 回答「只改 val 的标注，之前的指标还可信吗」；
  - `label_fingerprint` 回答「模型变了还是标注变了」。只有两个总量没法区分这两件事。

- **Decision**: 用文件大小而不是修改时间。
  - 理由：`mtime` 在复制、解压、git checkout 之后会变，同一份数据会算出不同指纹，
    这是比「漏检同名同大小换内容」更常见的误报。而漏检那种情况用户已确认可以接受。

- **Decision**: 导出 manifest 与 `dataset.yaml` 同目录，文件名 `manifest.json`。
  - 理由：跟着数据走，谁拿到目录都能自查；不用记「manifest 存在哪」。
  - 同目录已有 manifest 且指纹相同 → 返回 `unchanged: true`，导出仍然照做（用户可能就是要
    重写标签），只是让前端能提示「内容未变」。

- **Decision**: Web 与 autoexp 共用 `fingerprint.py`，autoexp 用「找不到就跳过」的方式引入。
  - 理由：与 `vram.py` 同样的坑——两边各写一份，必然出现「页面说 A、台账说 B」。
    autoexp 找不到模块时把 `dataset` 块标成 `unavailable`，不阻断实验。

- **Decision**: 引擎元数据文件名 `<engine>.meta.json`（如 `model_fp16.engine.meta.json`）。
  - 理由：与引擎同名，复制/改名时不会走散；不污染目录里其它 JSON。
  - ONNX 的 sha256 要算：引擎与 ONNX 的对应关系是溯源链上最容易断的一环，
    而 ONNX 通常只有几十 MB，算一次不到一秒。

## Risks / Trade-offs

- **指纹对「同名同大小、内容被换」的图片不敏感** → 用户已确认接受。界面在指纹旁标注
  `算法 v1（路径+大小+标注）`，看到的人知道它的边界。
- **每类产物都多一个文件**（manifest / train_meta / engine.meta）→ 都是旁挂只读文件，
  删掉不影响任何既有流程；读取方一律「缺失即当作未记录」。
- **5000 张算不满 2 秒**时（机械硬盘、标注在慢速网络盘）→ 指纹计算失败不能阻断导出：
  捕获异常后写 `{"algo":"v1","error":"..."}`，导出照常完成，界面显示「指纹计算失败」。
- **autoexp 每轮都算一次指纹**（600 秒预算里占 2 秒）→ 可接受；且能及时发现「这一轮用的
  数据和上一轮不同」，这本身就是它该拦的事。

## Migration Plan

- 新增文件，不改既有字段语义。
- `results.tsv` 加列走 `ensure_tsv()` 的按列名迁移，旧文件自动备份 `.bak`，旧行补空。
- `splits/_manifest.json` 只增 `data_fingerprint` 字段；`signature` 判断逻辑不变，
  所以不会因为加了字段就要求重建切分。
- 回滚：删除 `fingerprint.py` 的调用点即可，所有产物里的 `dataset` 块都按「可选」读取。

## Open Questions

- 无。
