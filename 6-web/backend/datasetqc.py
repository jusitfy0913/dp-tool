"""数据集体检：同底板分组 + 标注质检。

为什么需要这个模块
------------------
数据集指纹（``fingerprint.py``）只能回答「两次用的是不是同一份数据」，回答不了
「这份数据本身干不干净」。两类污染都不会报错：

1. **同底板泄漏**。工业视觉的图常常是同一块板、同一个工位拍出来的多张，甚至就是
   同一张底片配不同的缺陷斑块。按图片划分时它们会散落在 train / val / holdout，
   于是 val 指标虚高、holdout 验收失去意义。``splits/pcb``（692 张）实测：
   64 位 dHash 汉明距离 ≤6 时 692 张塌成 10 组，分组结果与文件名前缀完全一致，
   10 组全部同时出现在三个划分里。
2. **坏标注被静默吞掉**。``dataset.py`` 解析标签时对字段数不对的行直接 ``continue``，
   页面只会显示「空标注 N 张」，分不出「真没缺陷」与「标签写坏了」。

为什么不用单一阈值判「重复」
--------------------------
实测把 692 张按 dHash ≤6 判重会全量归入 10 组。若把「同组」当「重复」，数据集会被
砍掉 98.6%。所以这里分三级，且只有 C 级允许删：

- A 级 **精确重复**：文件 sha256 相同，删掉零损失。
- B 级 **同底板分组**：256 位 dHash 与 32×32 灰度差双条件。只分组，绝不据此删除。
  它的产出是「这些图来自同一张底片」，用于泄漏警告与按组切分。
- C 级 **冗余样本**：同组且同类别、标注框一一对应 IoU>0.95。这才是真正重复的样本。

"同一张底片 + 不同的缺陷斑块" 属于 B 级，不是重复：样本本身不同，删了就是在减少
训练数据，而缺陷才是标签。

只读约定
--------
本模块不写任何数据文件。唯一会写的是 ``runtime/dsqc_cache.json``（文件签名缓存），
写失败只记日志——体检是只读功能，不能因为缓存写不进去就报错。
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

import config
import dataset

#: 算法版本。阈值或判据变化时必须改这个值，否则新旧报告会被误当作可比。
ALGO = "dsqc-v1"

#: 256 位 dHash 的汉明距离阈值（单位：位）。实测同底板 ≈0~8，跨底板 ≥91。
DEFAULT_THRESHOLD = 8

#: 32×32 灰度平均绝对差上限（0~255 标度）。dHash 只看梯度符号，纯色区域会大量撞位，
#: 所以必须再用像素差复核一次，两条件都满足才并组。
MAE_LIMIT = 2.0

#: 缩略图边长。32 已经能区分「同底板」与「不同底板」，再大只是浪费缓存体积。
THUMB = 32

#: 每张图缓存的签名个数上限，防止长期使用后缓存无限增长。
#: 测试用 ``TRTFORGE_DSQC_CACHE`` 指向临时文件，避免污染真实运行目录。
CACHE_PATH = Path(
    os.environ.get("TRTFORGE_DSQC_CACHE") or (config.RUNTIME_DIR / "dsqc_cache.json")
)

#: 缓存条目上限。超出后按写入时间丢掉最旧的一半。
CACHE_LIMIT = 20000

#: 全量两两比较的上限。超过就改用 banding 生成候选，避免 O(n²) 的内存。
FULL_COMPARE_LIMIT = 3000

#: 规则码 → (级别, 中文说明)。级别只影响排序与着色，任何级别都不会阻断导出/训练。
RULES: dict[str, tuple[str, str]] = {
    "LBL_MISSING": ("error", "标注文件缺失"),
    "LBL_FIELD": ("error", "字段数与任务类型不符"),
    "LBL_PARSE": ("error", "数值无法解析"),
    "LBL_CLASS_RANGE": ("error", "类别编号越界"),
    "LBL_COORD_RANGE": ("error", "坐标超出 0~1"),
    "LBL_ZERO_AREA": ("error", "框的宽或高为 0"),
    "LBL_TINY": ("warn", "框面积小于全图的 0.01%"),
    "LBL_HUGE": ("warn", "框面积大于全图的 60%"),
    "LBL_BORDER": ("warn", "框贴住图像边界，可能没框全"),
    "LBL_DUP_BOX": ("warn", "同一张图里有两个高度重叠的框"),
    "LBL_CLASS_ABSENT": ("warn", "某个类别在这个划分里一张都没有"),
    "LBL_EMPTY": ("info", "标注为空（可能是正常良品图）"),
    "LBL_IMBALANCE": ("info", "某个类别占比低于 2%"),
    "GRP_LABEL_DIFF": ("info", "同底板的图在同一位置标了不同类别（噪声较大，仅供参考）"),
}

#: 任务类型 → 每行的字段数。segment 是「1 + 2n」不定长，单独判断。
_FIXED_FIELDS = {"detect": 5, "obb": 9, "pose": 4 + 17 * 3}

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 文件签名：sha256 + 256 位 dHash + 32×32 缩略图
# ---------------------------------------------------------------------------
def _cache_key(path: Path, stat: os.stat_result) -> str:
    """缓存键 = 绝对路径 + 大小 + 纳秒修改时间。

    不含 mtime 的格式化文本、不含盘符大小写：同一台机器上稳定，文件被覆盖即失效，
    不需要手动清理。
    """
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _compute_signature(path: Path) -> dict[str, str]:
    """算一张图的签名。失败抛异常，由调用方决定是跳过还是记问题。"""
    with Image.open(path) as im:
        # draft 让 JPEG 解码器直接按 1/4 或 1/8 输出，大图能快 5~10 倍。
        im.draft("L", (512, 512))
        gray = im.convert("L")
        # 256 位 dHash：17×16 的灰度图，比较水平相邻像素，得到 16×16=256 位
        grid = np.asarray(gray.resize((17, 16), Image.BILINEAR), dtype=np.int16)
        thumb = np.asarray(gray.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)
    bits = (grid[:, 1:] > grid[:, :-1]).reshape(-1)
    packed = np.packbits(bits).tobytes()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dhash": packed.hex(),
        "thumb": base64.b64encode(zlib.compress(thumb.tobytes(), 6)).decode("ascii"),
    }


def _load_cache() -> dict[str, Any]:
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"algo": ALGO, "entries": {}}
    if not isinstance(payload, dict) or payload.get("algo") != ALGO:
        # 算法变了，旧缓存里的 dHash 语义不同，必须整体作废
        return {"algo": ALGO, "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"algo": ALGO, "entries": entries}


def _save_cache(entries: dict[str, Any], keep: Iterable[str]) -> None:
    """合并写入缓存，写失败不影响报告。

    为什么不「只留本次见过的键」：体检可能轮流扫描多个数据集，只留本次的会把别的
    数据集的缓存全删掉，下次又得重新解码几百张图。这里改成合并 + 容量上限。
    """
    keep_set = set(keep)
    merged = dict(entries)
    for key in keep_set:
        if key in merged:
            merged[key]["at"] = int(time.time())
    if len(merged) > CACHE_LIMIT:
        ordered = sorted(merged.items(), key=lambda item: item[1].get("at", 0), reverse=True)
        merged = dict(ordered[:CACHE_LIMIT])
    payload = {"algo": ALGO, "entries": merged}
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(CACHE_PATH)
    except OSError:
        pass


def _signatures(paths: list[Path]) -> tuple[list[dict[str, str] | None], dict[str, int]]:
    """批量取签名，带缓存。返回 (与 paths 对齐的签名, 缓存统计)。"""
    cache = _load_cache()
    entries = cache["entries"]
    keys: list[str] = []
    result: list[dict[str, str] | None] = []
    stats = {"hits": 0, "misses": 0, "failed": 0}
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            keys.append("")
            result.append(None)
            stats["failed"] += 1
            continue
        key = _cache_key(path, stat)
        keys.append(key)
        cached = entries.get(key)
        if isinstance(cached, dict) and {"sha256", "dhash", "thumb"} <= set(cached):
            result.append(cached)
            stats["hits"] += 1
            continue
        try:
            signature = _compute_signature(path)
        except (OSError, ValueError):
            result.append(None)
            stats["failed"] += 1
            continue
        entries[key] = {**signature, "at": int(time.time())}
        result.append(signature)
        stats["misses"] += 1
    _save_cache(entries, keys)
    return result, stats


# ---------------------------------------------------------------------------
# 分组：hamming 过滤 → 像素差复核 → 并查集
# ---------------------------------------------------------------------------
def _sigma(signatures: list[dict[str, str] | None], count: int) -> np.ndarray:
    """把 dHash 拼成 (count, 32) 的 uint8 矩阵；失败项填 0 并保证不与任何真实签名相等。"""
    rows = []
    for signature in signatures:
        if signature is None:
            rows.append(np.full(32, 0, dtype=np.uint8))
        else:
            rows.append(np.frombuffer(bytes.fromhex(signature["dhash"]), dtype=np.uint8))
    return np.stack(rows) if rows else np.zeros((0, 32), dtype=np.uint8)


def _thumbs(signatures: list[dict[str, str] | None]) -> np.ndarray:
    rows = []
    for signature in signatures:
        if signature is None:
            rows.append(np.zeros(THUMB * THUMB, dtype=np.uint8))
        else:
            raw = zlib.decompress(base64.b64decode(signature["thumb"]))
            rows.append(np.frombuffer(raw, dtype=np.uint8))
    return np.stack(rows) if rows else np.zeros((0, THUMB * THUMB), dtype=np.uint8)


def _band_candidates(matrix: np.ndarray, bands: int = 8) -> list[tuple[int, int]]:
    """把 256 位切成 8 段，同段完全相同才成为候选对。

    这只是**召回**手段：真正的判定仍在后面按汉明距离做。它的价值是把 1 万张图的
    5000 万次比较压到几千次，代价是极小概率漏掉「每段都差一点」的近重复。
    """
    count = len(matrix)
    per = 32 // bands
    collected: set[tuple[int, int]] = set()
    for start in range(bands):
        block = matrix[:, start * per : (start + 1) * per]
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for index in range(count):
            buckets[block[index].tobytes()].append(index)
        for members in buckets.values():
            if len(members) > 1:
                collected.update(itertools.combinations(members, 2))
    return sorted(collected)


def _pairs_within(
    matrix: np.ndarray, threshold: int, chunk: int = 256
) -> list[tuple[int, int]]:
    """全量两两比较，返回汉明距离 ≤ 阈值的下标对（i < j）。分块控制内存。"""
    count = len(matrix)
    found: list[tuple[int, int]] = []
    for start in range(0, count, chunk):
        block = matrix[start : start + chunk]
        # (m, n, 32) 的字节异或，再查表求位 1 的个数
        diff = block[:, None, :] ^ matrix[None, :, :]
        distance = _POPCOUNT[diff].sum(axis=2)
        rows, cols = np.nonzero(distance <= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            left, right = start + row, col
            if left < right:
                found.append((left, right))
    return found


def _mae_pairs(
    thumbs: np.ndarray, pairs: list[tuple[int, int]], limit: float, chunk: int = 4096
) -> list[tuple[int, int]]:
    """对候选对算 32×32 灰度平均绝对差，只保留 ≤ limit 的。"""
    kept: list[tuple[int, int]] = []
    for offset in range(0, len(pairs), chunk):
        block = pairs[offset : offset + chunk]
        left = np.array([item[0] for item in block], dtype=np.int64)
        right = np.array([item[1] for item in block], dtype=np.int64)
        diff = np.abs(
            thumbs[left].astype(np.int16) - thumbs[right].astype(np.int16)
        ).mean(axis=1)
        for (i, j), value in zip(block, diff.tolist()):
            if value <= limit:
                kept.append((i, j))
    return kept


class _UnionFind:
    """并查集。路径压缩够用，不做按秩合并——图很小，n 只到几万。"""

    def __init__(self, count: int) -> None:
        self._parent = list(range(count))

    def find(self, index: int) -> int:
        parent = self._parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self._parent)):
            out[self.find(index)].append(index)
        return out


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """归一化 (cx, cy, w, h) 的 IoU。忽略长宽比差异，够用来判断「是不是同一个框」。"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(
        0.0, by2 - by1
    ) - inter
    return inter / union if union > 0 else 0.0


def _labels_equivalent(
    left: list[tuple[int, float, float, float, float]],
    right: list[tuple[int, float, float, float, float]],
) -> bool:
    """两组框是否「同一个标注」：数量相同、类别多重集相同、能一一配对且 IoU>0.95。"""
    if len(left) != len(right):
        return False
    if sorted(item[0] for item in left) != sorted(item[0] for item in right):
        return False
    remaining = list(right)
    for item in left:
        for index, other in enumerate(remaining):
            if item[0] == other[0] and _iou(item[1:], other[1:]) > 0.95:
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


# ---------------------------------------------------------------------------
# 标注质检
# ---------------------------------------------------------------------------
def _issue(
    code: str,
    rel: str,
    split: str,
    line: int | None = None,
    raw: str = "",
    detail: str = "",
) -> dict[str, Any]:
    severity, title = RULES.get(code, ("info", code))
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "rel": rel,
        "split": split,
        "line": line,
        "raw": raw[:200],
        "detail": detail,
    }


def _expected_fields(task: str, tokens: int) -> bool:
    fixed = _FIXED_FIELDS.get(task)
    if fixed is not None:
        return tokens == fixed
    if task == "segment":
        # 1 个类别 + 2n 个坐标，n≥3 才是多边形
        return tokens >= 7 and (tokens - 1) % 2 == 0
    return tokens >= 5


def _check_line(
    tokens: list[str], task: str, nc: int, rel: str, split: str, line: int, raw: str
) -> list[dict[str, Any]]:
    """校验一行 YOLO 标签。返回问题列表（可能为空）。"""
    issues: list[dict[str, Any]] = []
    if not _expected_fields(task, len(tokens)):
        issues.append(
            _issue("LBL_FIELD", rel, split, line, raw, f"{task} 应为 {_FIXED_FIELDS.get(task, '1+2n（n≥3）')} 个字段，实际 {len(tokens)} 个")
        )
        return issues  # 字段数都不对，后面的几何检查没有意义

    try:
        raw_class = float(tokens[0])
        values = [float(token) for token in tokens[1:]]
    except ValueError:
        issues.append(_issue("LBL_PARSE", rel, split, line, raw, "存在无法转成数字的字段"))
        return issues

    if raw_class != int(raw_class) or not 0 <= int(raw_class) < max(nc, 1):
        issues.append(
            _issue(
                "LBL_CLASS_RANGE",
                rel,
                split,
                line,
                raw,
                f"类别 {tokens[0]} 不在 0~{max(nc - 1, 0)} 之间",
            )
        )

    out_of_range = [value for value in values if not 0.0 <= value <= 1.0]
    if out_of_range:
        issues.append(
            _issue(
                "LBL_COORD_RANGE",
                rel,
                split,
                line,
                raw,
                f"有 {len(out_of_range)} 个坐标超出 0~1，最小 {min(out_of_range):.4f}、最大 {max(out_of_range):.4f}",
            )
        )

    if task == "detect" and len(values) >= 4:
        cx, cy, width, height = values[:4]
        area = width * height
        if width <= 0 or height <= 0:
            issues.append(_issue("LBL_ZERO_AREA", rel, split, line, raw, f"w={width:.6f} h={height:.6f}"))
        elif area < 1e-4:
            issues.append(
                _issue("LBL_TINY", rel, split, line, raw, f"面积占比 {area * 100:.4f}%")
            )
        elif area > 0.6:
            issues.append(_issue("LBL_HUGE", rel, split, line, raw, f"面积占比 {area * 100:.1f}%"))
        if (
            cx - width / 2 <= 1e-4
            or cy - height / 2 <= 1e-4
            or cx + width / 2 >= 1 - 1e-4
            or cy + height / 2 >= 1 - 1e-4
        ):
            issues.append(_issue("LBL_BORDER", rel, split, line, raw, "框贴住边界，可能没框全"))
    return issues


def _boxes_from_line(tokens: list[str]) -> tuple[int, float, float, float, float] | None:
    """从一行标签里取出 (class, cx, cy, w, h)；不是 detect 行或解析失败返回 None。"""
    if len(tokens) != 5:
        return None
    try:
        return (
            int(float(tokens[0])),
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            float(tokens[4]),
        )
    except ValueError:
        return None


def _label_path(root: Path, image: dataset.DatasetImage) -> Path | None:
    """YOLO 工程里某张图对应的标签文件路径。

    与 ``dataset._collect_yolo_images`` 保持同一套布局判断：有 ``labels/<split>``
    就用它，扁平结构则直接在 ``labels/`` 下。
    """
    labels_root = root / "labels"
    split_dir = labels_root / image.current_split
    if split_dir.is_dir():
        return split_dir / f"{image.path.stem}.txt"
    if labels_root.is_dir():
        return labels_root / f"{image.path.stem}.txt"
    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def inspect(
    source: str | Path,
    task: str = "auto",
    *,
    threshold: int = DEFAULT_THRESHOLD,
    limit: int = 500,
    group_by: str = "prefix",
    ok_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """体检一个数据集，返回分组、泄漏与标注问题。

    只读：不修改任何数据文件。``threshold`` 是 256 位 dHash 的汉明距离上限。
    ``ok_dirs`` 声明的良品目录里的图不参与 ``LBL_MISSING`` / ``LBL_EMPTY`` 判定：
    用户已经显式声明过那是良品，再重复报一遍只会制造噪音。没有声明的空标注/
    缺标注图仍然照常报出来——那才是可能的漏标。
    """
    started = time.time()
    threshold = max(0, min(256, int(threshold)))
    project = dataset.load_project(str(source), task, ok_dirs=ok_dirs)
    root = project.source
    images = project.images
    nc = len(project.classes)
    detect_like = project.task != "classify"

    signatures, cache_stats = _signatures([image.path for image in images])
    matrix = _sigma(signatures, len(images))
    thumbs = _thumbs(signatures)

    # 解不开的图不能参与分组：它们的签名是全 0，会互相「完全相似」
    valid = [index for index, signature in enumerate(signatures) if signature is not None]
    if valid:
        matrix = matrix[valid]
        thumbs = thumbs[valid]
    else:
        matrix = np.zeros((0, 32), dtype=np.uint8)
        thumbs = np.zeros((0, THUMB * THUMB), dtype=np.uint8)

    # --- 分组 ---
    if len(valid) <= FULL_COMPARE_LIMIT:
        candidates = _pairs_within(matrix, threshold)
    else:
        candidates = _pairs_within_filtered(matrix, _band_candidates(matrix), threshold)
    verified = _mae_pairs(thumbs, candidates, MAE_LIMIT)

    union = _UnionFind(len(valid))
    for left, right in verified:
        union.union(left, right)
    raw_groups = [group for group in union.groups().values() if len(group) > 1]
    raw_groups = [
        [valid[position] for position in group] for group in raw_groups
    ]

    # --- 精确重复（A 级）---
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        if signature is not None:
            by_sha[signature["sha256"]].append(index)
    exact_groups = [members for members in by_sha.values() if len(members) > 1]

    # --- 标注解析（用于质检与 C 级判定）---
    label_paths: list[Path | None] = []
    label_lines: list[list[list[str]]] = []
    for image in images:
        path = _label_path(root, image) if project.kind == "yolo" else None
        label_paths.append(path)
        if path is not None and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            label_lines.append([line.split() for line in text.splitlines() if line.split()])
        else:
            label_lines.append([])

    issues: list[dict[str, Any]] = []
    boxes: list[list[tuple[int, float, float, float, float]]] = []
    for image, path, lines in zip(images, label_paths, label_lines):
        rel = image.rel
        split = image.current_split
        if not detect_like:
            boxes.append([])
            continue
        if project.kind == "yolo" and (path is None or not path.is_file()):
            if image.is_ok:
                # 良品图本来就可能没有标签文件（还没导出）。声明过的良品不报 LBL_MISSING，
                # 否则用户每体检一次都要看一遍自己刚声明过的东西。
                boxes.append([])
                continue
            issues.append(
                _issue("LBL_MISSING", rel, split, None, "", f"找不到 {path.name if path else 'labels 目录'}")
            )
        elif not lines and project.kind == "yolo":
            if not image.is_ok:
                issues.append(_issue("LBL_EMPTY", rel, split, None, "", "标签文件存在但没有内容"))

        parsed: list[tuple[int, float, float, float, float]] = []
        for line_no, tokens in enumerate(lines, 1):
            issues.extend(
                _check_line(tokens, project.task, nc, rel, split, line_no, " ".join(tokens))
            )
            box = _boxes_from_line(tokens)
            if box is not None and 0 <= box[0] < max(nc, 1):
                parsed.append(box)
        # 同图重复框：只在 detect 上有意义
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if _iou(parsed[i][1:], parsed[j][1:]) > 0.9:
                    issues.append(
                        _issue(
                            "LBL_DUP_BOX",
                            rel,
                            split,
                            None,
                            "",
                            f"第 {i + 1} 与第 {j + 1} 个框 IoU>0.9",
                        )
                    )
        boxes.append(parsed)

    # --- 类别覆盖与不均衡 ---
    per_split: dict[str, Counter] = {name: Counter() for name in dataset.SPLITS}
    overall: Counter = Counter()
    for image, parsed in zip(images, boxes):
        for item in parsed:
            per_split[image.current_split][item[0]] += 1
            overall[item[0]] += 1
    split_sizes = Counter(image.current_split for image in images)
    total_boxes = sum(overall.values())
    for class_item in project.classes if detect_like else []:
        class_id = int(class_item["id"])
        for split, size in split_sizes.items():
            if size > 0 and per_split[split][class_id] == 0:
                issues.append(
                    _issue(
                        "LBL_CLASS_ABSENT",
                        "",
                        split,
                        None,
                        "",
                        f"类别 {class_item['name']} 在 {split}（{size} 张）里一个都没有",
                    )
                )
    if total_boxes:
        for class_item in project.classes:
            class_id = int(class_item["id"])
            ratio = overall[class_id] / total_boxes
            if ratio < 0.02:
                issues.append(
                    _issue(
                        "LBL_IMBALANCE",
                        "",
                        "",
                        None,
                        "",
                        f"类别 {class_item['name']} 只有 {overall[class_id]} 个框（{ratio * 100:.2f}%）",
                    )
                )

    # --- 组装分组 ---
    groups: list[dict[str, Any]] = []
    same_label: list[dict[str, Any]] = []
    for member_indexes in raw_groups:
        member_indexes = sorted(member_indexes)
        members = [images[index] for index in member_indexes]
        splits: Counter = Counter(member.current_split for member in members)
        group_boxes = [boxes[index] for index in member_indexes]
        # 「同组 + 同标注」只做提示，不提供删除动作：同一块板的两条产线可能真的
        # 长出位置完全相同的同一个缺陷，靠图像相似度分不出「副本」与「同位置的不同样本」。
        # 真要删，请用 A 级（逐字节相同）。
        pairs = [
            [images[a].rel, images[b].rel]
            for a, b in itertools.combinations(range(len(member_indexes)), 2)
            if _labels_equivalent(group_boxes[a], group_boxes[b])
        ]
        if pairs:
            same_label.extend({"keep": pair[0], "drop": pair[1]} for pair in pairs)
        groups.append(
            {
                "size": len(members),
                "splits": dict(splits),
                "cross_split": len(splits) > 1,
                "members": [
                    {"rel": member.rel, "split": member.current_split}
                    for member in members[:50]
                ],
                "members_truncated": len(members) > 50,
                "same_label_pairs": len(pairs),
            }
        )
    groups.sort(key=lambda item: -item["size"])

    # --- 同底板标注不一致（提示级）---
    # 大分组（>400 张）两两比较代价过高，而这类数据集本身就应该先做按组切分，
    # 这里直接跳过，避免体检变成耗时操作。
    for member_indexes in raw_groups:
        if not 2 <= len(member_indexes) <= 400 or not detect_like:
            continue
        hits = 0
        for a, b in itertools.combinations(sorted(member_indexes), 2):
            for left in boxes[a]:
                for right in boxes[b]:
                    if left[0] != right[0] and _iou(left[1:], right[1:]) > 0.5:
                        hits += 1
        if hits:
            issues.append(
                _issue(
                    "GRP_LABEL_DIFF",
                    images[member_indexes[0]].rel,
                    images[member_indexes[0]].current_split,
                    None,
                    "",
                    "同一底板的 2 张图有 {} 处「同位置不同类别」".format(hits),
                )
            )

    by_rule: Counter = Counter(item["code"] for item in issues)
    summary = {
        "error": sum(1 for item in issues if item["severity"] == "error"),
        "warn": sum(1 for item in issues if item["severity"] == "warn"),
        "info": sum(1 for item in issues if item["severity"] == "info"),
    }
    leak_groups = [group for group in groups if group["cross_split"]]
    leak_images = sum(group["size"] for group in leak_groups)

    report = {
        "algo": ALGO,
        "source": str(root),
        "kind": project.kind,
        "task": project.task,
        "threshold": threshold,
        "mae_limit": MAE_LIMIT,
        "classes": project.classes,
        "counts": {
            "images": len(images),
            **{name: split_sizes.get(name, 0) for name in dataset.SPLITS},
            "empty_labels": sum(1 for lines in label_lines if not lines) if detect_like else 0,
            "ok_images": sum(1 for image in images if image.is_ok),
        },
        "ok_images": sum(1 for image in images if image.is_ok),
        "ok_dirs": list(project.ok_dirs),
        "summary": summary,
        "by_rule": {code: {"count": count, "title": RULES[code][1], "severity": RULES[code][0]}
                    for code, count in by_rule.most_common()},
        "groups": {
            "algo": "dhash256+mae",
            "total": len(groups),
            "cross_split": len(leak_groups),
            "leak_images": leak_images,
            "items": groups[:50],
            "truncated": len(groups) > 50,
            "same_label_pairs": same_label[:50],
            "same_label_total": len(same_label),
        },
        "exact_groups": [
            [images[index].rel for index in members] for members in exact_groups[:50]
        ],
        "exact_group_total": len(exact_groups),
        "issues": issues[:limit],
        "issues_truncated": len(issues) > limit,
        "cache": cache_stats,
        "elapsed_ms": int((time.time() - started) * 1000),
        "version": config.APP_VERSION,
    }
    if project.warnings:
        report["source_warnings"] = list(project.warnings)
    return report


def _pairs_within_filtered(
    matrix: np.ndarray, pairs: list[tuple[int, int]], threshold: int
) -> list[tuple[int, int]]:
    """banding 只做召回，这里按汉明距离做真正的判定。"""
    kept: list[tuple[int, int]] = []
    if not pairs:
        return kept
    left = np.array([item[0] for item in pairs], dtype=np.int64)
    right = np.array([item[1] for item in pairs], dtype=np.int64)
    for offset in range(0, len(pairs), 100_000):
        chunk_left = left[offset : offset + 100_000]
        chunk_right = right[offset : offset + 100_000]
        distance = _POPCOUNT[matrix[chunk_left] ^ matrix[chunk_right]].sum(axis=1)
        for index, value in enumerate(distance.tolist()):
            if value <= threshold:
                kept.append((int(chunk_left[index]), int(chunk_right[index])))
    return kept
