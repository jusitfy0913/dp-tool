# -*- coding: utf-8 -*-
"""数据集指纹：回答「这个模型/这个引擎到底用的哪批图」。

为什么需要它
------------
训练目录里只有一个 ``data: D:\\...\\dataset.yaml`` 路径。图片被重新标注、增删、
重划分之后，路径没变但内容已经不同，于是：

- 两个模型的指标没法判断是不是同一批数据下测出来的；
- 自动实验台账里「0.4109 → 0.4150」到底是参数赢了还是数据变了，无从分辨；
- holdout 的图被改了标注，之前「验收通过」的结论其实已经失效。

指纹不复制数据、不存快照，只给内容一个可核对的摘要：**同样内容 → 同样指纹，
任何一处变化 → 不同指纹**。

算法（v1）
----------
对每个 split，把该 split 下每张图的「相对路径 + 文件字节数 + 标注文本哈希」排好序
后做 sha256::

    split_fp = sha256("v1\\n" + split + "\\n" + 排序后的若干行)
    dataset_fp = sha256("v1\\ndataset\\n" + 三个分片 + 任务/类别)[:16]
    label_fp = sha256("v1\\nlabel\\n" + 路径与标注哈希)[:16]

三个粒度各有用途（少一个就分不清「模型变了」还是「标注变了」）：

- ``dataset_fingerprint``：能不能和另一次结果直接横比；
- ``split_fingerprints``：只改 val 的标注，之前训练的指标还可不可信；
- ``label_fingerprint``：排除图片增删，只看标注有没有动过。

刻意不做的事
------------
- **不哈希图片内容**（用户已确认）。5000 张读一遍要几十秒，而绝大多数变更靠
  「路径 + 大小 + 标注」就能发现。代价是「同名同大小但内容被换」发现不了，
  这一点在界面文案里会写明，别让人把它当成图片级校验。
- **不依赖修改时间**。复制、解压、git checkout 都会改 mtime，那会让同一份数据
  算出不同指纹——比漏检更常见的误报。
- **相对路径以各 split 的 images 目录为基准**（而不是绝对路径），所以同一批数据
  导出到 ``D:\\out_a`` 与 ``E:\\out_b``、或镜像到 ``splits/``，指纹都一样。
- **路径计入指纹**：重命名一张图 = 数据变了（用户确认的语义）。

容错
----
指纹是「附加信息」，绝不能阻断导出或训练。任何异常都收敛成带 ``error`` 的结果。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

#: 算法版本。换了算法要改它：不同版本的指纹永远不可能相等，不会把旧数据当新数据。
ALGO = "v1"

#: 三个 split 的固定顺序。顺序进哈希，避免「只调换 key 顺序就换指纹」的歧义。
SPLIT_NAMES = ("train", "val", "test")

#: 短码长度：界面与台账里显示用（16 位太长，8 位足够区分几百次实验）。
SHORT_LEN = 8

#: 指纹里「没有标注文件」的占位。用固定串而不是空串，避免与「空标注文件」混淆。
_NO_LABEL = "none"


# ---------------------------------------------------------------------------
# 基础哈希
# ---------------------------------------------------------------------------
def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def label_digest(path: Path) -> str:
    """标注文件的摘要。用**原始字节**而不是解析后的结构。

    解析会吞掉格式错误（多余的列、越界坐标），而格式错误本身就是数据变化——
    把它们归一化掉，指纹就失去了「发现标注被改坏」的能力。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return _NO_LABEL


def _image_files(root: Path) -> list[Path]:
    """递归收集图片并按相对路径排序。

    排序是必须的：``rglob`` 的顺序依赖文件系统，不排序会让同一份数据在不同机器上
    得到不同指纹。这与 ``dataset.py`` 的划分要用固定哈希是同一个道理。
    """
    if not root.is_dir():
        return []
    # 后缀集合与 dataset.IMAGE_EXTS 一致；这里不 import dataset，避免形成循环依赖
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    files.sort(key=lambda p: p.relative_to(root).as_posix().lower())
    return files


def _labels_dir(images_dir: Path) -> Path:
    """按 Ultralytics 约定推导标签目录：``.../images/train`` → ``.../labels/train``。

    分类数据集没有 labels 目录（类别就是文件夹名），这里会返回一个不存在的路径，
    ``label_digest`` 因而统一给出 ``none``——而类别信息已经在相对路径里了。
    """
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :])
    return images_dir.parent / "labels" / images_dir.name


# ---------------------------------------------------------------------------
# 核心
# ---------------------------------------------------------------------------
def _scan_split(dirs: list[Path]) -> tuple[list[str], dict[str, str], int]:
    """扫一个 split 的所有图片目录。

    返回 ``(路径与大小组成的行, 路径与标注哈希组成的行, 图片数)``。

    多个目录时统一加 ``d0/`` / ``d1/`` 前缀：同一份数据用 ``train: a`` 还是
    ``train: [a]`` 写出来，指纹必须相同，所以单目录也走同一套前缀。
    """
    sized: list[str] = []
    labels: dict[str, str] = {}
    count = 0
    for index, images_dir in enumerate(dirs):
        prefix = f"d{index}/"
        labels_dir = _labels_dir(images_dir)
        for image in _image_files(images_dir):
            rel = image.relative_to(images_dir).with_suffix("").as_posix().lower()
            key = f"{prefix}{rel}"
            try:
                size = image.stat().st_size
            except OSError:
                # 统计不到大小（权限/竞态）时用 0，仍然把路径记进去——
                # 路径变了指纹照样会变，比整张跳过更安全
                size = 0
            digest = label_digest(labels_dir / f"{rel}.txt")
            sized.append(f"{key}\t{size}\t{digest}")
            labels[key] = digest
            count += 1
    sized.sort()
    return sized, labels, count


def compute(yaml_path: str | Path) -> dict[str, Any]:
    """对 ``dataset.yaml`` 指向的数据算指纹。

    永远不抛异常：失败时返回 ``{"algo": "v1", "error": "..."}``，调用方按「未记录」处理。
    """
    started = time.perf_counter()
    result: dict[str, Any] = {
        "algo": ALGO,
        "dataset_fingerprint": None,
        "short": "",
        "label_fingerprint": None,
        "split_fingerprints": {name: None for name in SPLIT_NAMES},
        "counts": {name: 0 for name in SPLIT_NAMES},
        "task": "",
        "nc": None,
        "classes": [],
        "yaml": str(yaml_path),
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": 0.0,
        "error": "",
    }

    try:
        import traininfo  # noqa: PLC0415 - 延迟导入：只做哈希时不拖 dataset 解析器

        resolved = traininfo.resolve_splits(yaml_path)
    except Exception as exc:  # noqa: BLE001 - 指纹失败不许阻断主流程
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = round(time.perf_counter() - started, 3)
        return result

    try:
        split_rows: dict[str, list[str]] = {}
        label_rows: list[str] = []
        for name in SPLIT_NAMES:
            dirs = list(resolved.get("splits", {}).get(name) or [])
            sized, labels, count = _scan_split(dirs)
            split_rows[name] = sized
            result["counts"][name] = count
            result["split_fingerprints"][name] = _sha16(f"{ALGO}\n{name}\n" + "\n".join(sized))
            # 标注指纹**不带路径**：它回答的是「标注文本本身有没有被改过」。
            # 把路径放进来会让「只给图改个名字」也表现为标注变化，那正是
            # dataset_fingerprint 该管的事；这里混进来就分不清两件事了。
            label_rows.extend(f"{name}\t{digest}" for digest in sorted(labels.values()))

        classes = resolved.get("classes") or []
        result["classes"] = classes
        result["nc"] = resolved.get("nc") if resolved.get("nc") is not None else len(classes)
        result["task"] = str(resolved.get("task") or "")

        # 任务与类别名称也进总量指纹：同样的图，标注成不同类别体系就是不同的数据集
        class_text = ",".join(f"{item.get('id')}:{item.get('name')}" for item in classes)
        dataset_text = (
            f"{ALGO}\ndataset\n"
            f"task:{result['task']}\nnc:{result['nc']}\nclasses:{class_text}\n"
            + "".join(f"{name}:{result['split_fingerprints'][name]}\n" for name in SPLIT_NAMES)
        )
        result["dataset_fingerprint"] = _sha16(dataset_text)
        result["short"] = result["dataset_fingerprint"][:SHORT_LEN]
        result["label_fingerprint"] = _sha16(f"{ALGO}\nlabel\n" + "\n".join(label_rows))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def short(value: str | None) -> str:
    """取短码。空值返回空串（界面据此显示「未记录」）。"""
    return (value or "")[:SHORT_LEN]


def is_recorded(payload: Any) -> bool:
    """判断一份指纹记录是否可用（有算法版本且在合理长度内）。"""
    if not isinstance(payload, dict):
        return False
    fingerprint = payload.get("dataset_fingerprint")
    return isinstance(fingerprint, str) and len(fingerprint) == 16 and bool(payload.get("algo"))


def describe(payload: dict[str, Any] | None) -> str:
    """给日志与界面用的一句话。"""
    if not is_recorded(payload):
        return "数据集指纹未记录"
    assert payload is not None
    counts = payload.get("counts") or {}
    return (
        f"数据集指纹 {short(payload['dataset_fingerprint'])}"
        f"（{counts.get('train', 0)} train / {counts.get('val', 0)} val，"
        f"算法 {payload.get('algo')}：路径+大小+标注）"
    )


# ---------------------------------------------------------------------------
# 没有 dataset.yaml 的格式（COCO / VOC）
# ---------------------------------------------------------------------------
#: 文本类文件才算内容哈希。图片只取大小——与主算法一致，也避免把几 MB 的图读一遍。
_TEXT_SUFFIXES = {".json", ".xml", ".txt", ".yaml", ".yml"}


def compute_tree(
    root: str | Path,
    *,
    task: str = "",
    nc: int | None = None,
    classes: list[dict] | None = None,
    skip: tuple[str, ...] = ("manifest.json",),
) -> dict[str, Any]:
    """对「导出目录里的产物清单」算指纹。

    COCO / VOC 没有 ``dataset.yaml``（训练也不用它们），但导出同样需要一份身份记录。
    这里按文件清单算：文本文件取内容哈希，图片只取大小。

    为什么不复用 ``compute``：那套算法的语义是「split → 图片 → 标注」，而 COCO 只有
    一个 ``annotations.json``、VOC 是 ``JPEGImages + Annotations``，硬套会得出一个
    看起来像数据集指纹、实际不受图片变化影响的值。这里如实说明它算的是**产物**。
    """
    started = time.perf_counter()
    result: dict[str, Any] = {
        "algo": ALGO,
        "scope": "export",
        "dataset_fingerprint": None,
        "short": "",
        "label_fingerprint": None,
        "split_fingerprints": {name: None for name in SPLIT_NAMES},
        "counts": {name: 0 for name in SPLIT_NAMES},
        "task": task,
        "nc": nc,
        "classes": list(classes or []),
        "yaml": "",
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": 0.0,
        "error": "",
    }
    base = Path(root)
    try:
        if not base.is_dir():
            raise FileNotFoundError(f"导出目录不存在：{base}")
        rows: list[str] = []
        files = [p for p in base.rglob("*") if p.is_file() and p.name not in skip]
        files.sort(key=lambda p: p.relative_to(base).as_posix().lower())
        for item in files:
            rel = item.relative_to(base).as_posix().lower()
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            digest = _NO_LABEL
            if item.suffix.lower() in _TEXT_SUFFIXES:
                try:
                    digest = hashlib.sha256(item.read_bytes()).hexdigest()[:16]
                except OSError:
                    digest = _NO_LABEL
            rows.append(f"{rel}\t{size}\t{digest}")
        class_text = ",".join(f"{item.get('id')}:{item.get('name')}" for item in result["classes"])
        text = f"{ALGO}\ntree\ntask:{task}\nnc:{nc}\nclasses:{class_text}\n" + "\n".join(rows)
        result["dataset_fingerprint"] = _sha16(text)
        result["short"] = result["dataset_fingerprint"][:SHORT_LEN]
        # 产物里能反映标注变化的只有文本类文件；这个粒度对 COCO/VOC 意义有限，
        # 但保留字段让四处记录的读取方不用分叉
        result["label_fingerprint"] = _sha16(
            f"{ALGO}\nlabel\n" + "\n".join(row for row in rows if row.rsplit("\t", 1)[-1] != _NO_LABEL)
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result
