# -*- coding: utf-8 -*-
"""预标注草稿的旁挂持久化：`<工程名>.drafts.json`。

为什么草稿不写进 `.tfanno`：
- Qt 版保存时**整体重写**工程文件，未知键不会被回写。草稿如果住在 `.tfanno` 里，
  用户用 Qt 打开再存一次，Web 端刚跑出来的草稿就静默消失了；
- 草稿是过程数据（模型给的候选），正式标注是结论（人确认过的）。两者生命周期不同：
  导出训练集只能带后者，但刷新页面时前者必须还在。

文件结构（`version` 用于以后加字段，不做无意义的前向兼容猜测）::

    {
      "version": 2,
      "project": "demo.tfanno",
      "updated_ns": "1700000000000000000",
      "count": 12,
      "images": { "<图片 id>": [ {草稿对象}, ... ] },
      "review": { "<图片 id>": { "state": "checked", "at_ns": "..." } }
    }

v2 相对 v1 只加了 `review`：
- **审核状态和草稿同住一个文件**，因为它们是同一次"审核动作"的两半——
  接受草稿（草稿减少）和标为已审核（review 增加）必须一起落盘，
  分两个文件就会出现"草稿清了、状态没写"的半截状态，用户看到图片又变回未审核。
- v1 文件读进来 `review` 视为空；写出去一律升到 v2。**不做降级**：
  降级会丢掉审核状态，而 v1 的读取方（旧版本 Web 端）本来就不认识 `review`，
  多一个键不会让它崩。

并发约定：**同一工程路径的读-改-写必须串行**。批量预标注线程与前端自动保存会同时
写同一个文件，不串行就会互相覆盖（典型表现：批量跑完只剩最后几张图的草稿）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import annotation

#: 旁挂文件后缀。`foo.tfanno` → `foo.drafts.json`（同目录、同主名）。
DRAFT_SUFFIX = ".drafts.json"
CURRENT_VERSION = 2
#: 单张图片的草稿上限。防御性上限：坏文件不该把服务进程的内存打爆。
MAX_DRAFTS_PER_IMAGE = 5000
#: 合法的审核状态。只有两态：审核这件事对用户来说只有"看过"和"没看过"，
# 中间态（比如"待复核"）会让 K 键到底该跳到哪张变得不可解释。
REVIEW_STATES = ("checked", "unchecked")

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def drafts_path(project_path: str | Path) -> Path:
    """由工程路径推出旁挂草稿路径。"""
    project = Path(project_path)
    # 用 with_suffix 会把 `.tfanno` 换掉；这里要的是"再挂一层后缀"，
    # 所以直接拼名字，`demo.tfanno` → `demo.tfanno.drafts.json` 太丑，取主名。
    stem = project.stem if project.suffix.lower() == ".tfanno" else project.name
    return project.with_name(f"{stem}{DRAFT_SUFFIX}")


def _lock_for(target: Path) -> threading.Lock:
    key = str(target.resolve()) if target.is_absolute() else str(target)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _empty_meta(target: Path) -> dict:
    return {
        "path": str(target),
        "exists": False,
        "updated_ns": "0",
        "count": 0,
        "image_count": 0,
        "version": CURRENT_VERSION,
        "reviewed": 0,
        "review": {},
        "warnings": [],
    }


def count_drafts(images: dict) -> int:
    """统计草稿总数。容忍任何非列表值（坏数据不该让统计接口崩）。"""
    total = 0
    for value in (images or {}).values():
        if isinstance(value, list):
            total += len(value)
    return total


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------


def normalize_review(raw) -> tuple[dict, list[str]]:
    """把任意来源的审核块规范成 ``{图片 id: {state, at_ns}}``。

    只保留 `checked`：`unchecked` 是"没有记录"的同义表达，写进文件只会让文件
    随时间无限增长（用户每取消一次就多一条）。读的时候把 `unchecked` 也丢掉，
    等价于删除。
    """
    warnings: list[str] = []
    result: dict[str, dict] = {}
    if raw is None:
        return result, warnings
    if not isinstance(raw, dict):
        warnings.append("审核块不是对象，已按「全部未审核」处理")
        return result, warnings

    for image_id, entry in raw.items():
        key = str(image_id or "").strip()
        if not key:
            continue
        if isinstance(entry, str):
            # 兼容"只写一个字符串"的写法
            entry = {"state": entry}
        if not isinstance(entry, dict):
            warnings.append(f"图片「{key}」的审核记录不是对象，已跳过")
            continue
        state = str(entry.get("state") or "").strip().lower()
        if state not in REVIEW_STATES:
            warnings.append(f"图片「{key}」的审核状态「{state}」不认识，已跳过")
            continue
        if state == "unchecked":
            continue
        at_ns = str(entry.get("at_ns") or "")
        result[key] = {"state": "checked", "at_ns": at_ns if at_ns.isdigit() else "0"}
    return result, warnings


def normalize(raw) -> tuple[dict, dict, list[str]]:
    """把任意来源的旁挂内容规范成 ``(images, meta, warnings)``。

    单条草稿走 `annotation.normalize_object`，与正式标注共用同一套形状回退规则——
    否则草稿能显示、接受后却变成非法标注，问题会拖到导出阶段才暴露。
    """
    warnings: list[str] = []
    if not isinstance(raw, dict):
        if raw is not None:
            warnings.append("草稿文件根结构不是 JSON 对象，已按空草稿处理")
        return {}, {"version": CURRENT_VERSION, "updated_ns": "0", "count": 0, "review": {}, "reviewed": 0}, warnings

    images: dict[str, list[dict]] = {}
    raw_images = raw.get("images")
    if not isinstance(raw_images, dict):
        if raw_images is not None:
            warnings.append("草稿文件的 images 字段不是对象，已按空草稿处理")
        raw_images = {}

    for image_id, items in raw_images.items():
        key = str(image_id or "").strip()
        if not key:
            warnings.append("有一条草稿记录的图片 id 为空，已跳过")
            continue
        if not isinstance(items, list):
            warnings.append(f"图片「{key}」的草稿不是数组，已跳过")
            continue
        kept: list[dict] = []
        for item in items:
            draft = _normalize_draft(item, key, warnings)
            if draft is not None:
                kept.append(draft)
            if len(kept) >= MAX_DRAFTS_PER_IMAGE:
                warnings.append(f"图片「{key}」的草稿超过 {MAX_DRAFTS_PER_IMAGE} 条，其余已忽略")
                break
        if kept:
            images[key] = kept

    review, review_warnings = normalize_review(raw.get("review"))
    warnings.extend(review_warnings)

    updated_ns = str(raw.get("updated_ns") or "0")
    meta = {
        "version": annotation.as_int(raw.get("version"), CURRENT_VERSION),
        "updated_ns": updated_ns if updated_ns.isdigit() else "0",
        "count": count_drafts(images),
        "review": review,
        "reviewed": len(review),
    }
    return images, meta, warnings


def _normalize_draft(item, image_id: str, warnings: list[str]) -> dict | None:
    """规范化单条草稿；不可用时返回 ``None`` 并记一条警告。"""
    if not isinstance(item, dict):
        warnings.append(f"图片「{image_id}」有一条草稿不是对象，已跳过")
        return None
    try:
        draft = annotation.normalize_object(item, warnings)
    except annotation.AnnotationError as exc:
        warnings.append(f"图片「{image_id}」有一条草稿被跳过：{exc}")
        return None
    if not draft.get("points"):
        warnings.append(f"图片「{image_id}」有一条草稿没有点，已跳过")
        return None
    if not draft.get("id"):
        draft["id"] = f"draft_{uuid.uuid4().hex[:10]}"
    # 置信度是分档着色的依据，但只接受概率区间；越界值按"未知"处理，
    # 否则 UI 会把它画成红色，看起来像"模型很不确定"，误导审核。
    score = float(draft.get("score", -1.0))
    if score > 1.0:
        warnings.append(f"图片「{image_id}」有一条草稿的置信度 {score:g} 超出 0~1，已按未知处理")
        score = -1.0
    draft["score"] = score
    draft["_draft"] = True
    note = item.get("_note")
    if isinstance(note, str) and note.strip():
        draft["_note"] = note.strip()[:200]
    extra = item.get("_warnings")
    if isinstance(extra, list):
        cleaned = [str(text)[:200] for text in extra if str(text).strip()]
        if cleaned:
            draft["_warnings"] = cleaned[:5]
    return draft


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------


def load(project_path: str | Path) -> tuple[dict, dict]:
    """读取旁挂草稿。

    返回 ``(images, meta)``，审核状态在 ``meta["review"]``（图片 id → 记录）。

    为什么审核状态塞在 meta 里而不是返回三元组：调用点有 app、批量任务、测试、
    前端接口四处，改签名会让每个调用点都要改一遍，而它们**几乎都只用 images**。
    meta 本来就是这个文件的"描述信息"，审核状态属于描述的一部分。

    文件不存在时返回空表且 ``exists=False``——「没有草稿」是正常状态，
    不该让打开工程的流程失败。
    """
    target = drafts_path(project_path)
    meta = _empty_meta(target)
    if not target.is_file():
        return {}, meta
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        meta.update(exists=True, warnings=[f"读取草稿文件失败：{exc}"])
        return {}, meta
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        meta.update(
            exists=True,
            warnings=[
                f"草稿文件不是合法 JSON（{exc}），已按空草稿处理；"
                "下次保存会覆盖它，需要保留请先备份"
            ],
        )
        return {}, meta

    images, inner, warnings = normalize(raw)
    meta.update(
        exists=True,
        updated_ns=inner.get("updated_ns", "0"),
        count=count_drafts(images),
        image_count=len(images),
        version=inner.get("version", CURRENT_VERSION),
        review=inner.get("review", {}),
        reviewed=inner.get("reviewed", 0),
        warnings=warnings,
    )
    return images, meta


def review_of(project_path: str | Path) -> dict:
    """只取审核状态（批量任务的 `scope=pending` 判断用）。"""
    _images, meta = load(project_path)
    return meta.get("review") or {}


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------


def _write_locked(target: Path, images: dict, review: dict, project_path: str | Path) -> dict:
    """``target`` 的写锁已持有时调用：原子写整份文件。"""
    payload = {
        "version": CURRENT_VERSION,
        "project": Path(project_path).name,
        "updated_ns": str(time.time_ns()),
        "count": count_drafts(images),
        "images": images,
    }
    # 空审核块也写出来：字段固定存在，读的时候不用到处 `if "review" in raw`
    payload["review"] = review or {}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    annotation._atomic_write(target, text)
    return {
        "path": str(target),
        "exists": True,
        "updated_ns": payload["updated_ns"],
        "count": payload["count"],
        "image_count": len(images),
        "version": CURRENT_VERSION,
        "review": payload["review"],
        "reviewed": len(payload["review"]),
        "warnings": [],
    }


def save(
    project_path: str | Path,
    images: dict | None = None,
    review: dict | None = None,
    replace: bool = False,
) -> dict:
    """整份替换。

    `replace=False` 且只传 `review` 时，草稿保持磁盘上的原样——这是前端
    "只切审核状态"的路径，不能顺手把草稿清空。
    """
    target = drafts_path(project_path)
    cleaned_images, _meta, _warnings = normalize({"images": images or {}})
    cleaned_review, _ = normalize_review(review)
    with _lock_for(target):
        if not replace and images is None:
            disk_images, disk_meta = load(project_path)
            cleaned_images = disk_images
            if review is None:
                cleaned_review = disk_meta.get("review") or {}
        elif not replace and review is None:
            _disk_images, disk_meta = load(project_path)
            cleaned_review = disk_meta.get("review") or {}
        return _write_locked(target, cleaned_images, cleaned_review, project_path)


def merge(project_path: str | Path, images: dict | None = None, review: dict | None = None) -> dict:
    """按图片 id 增量覆盖，**其余图片原样保留**。

    这是前端自动保存与批量任务共用的写入口：
    - 前端只提交被改过的图片，不会把批量刚写进去的其他图片清掉；
    - 传空数组表示"这张图没有草稿了"，对应记录会被删除；
    - `review` 同样是增量：只覆盖传入的图片 id，`state="unchecked"` 表示删除。
    """
    target = drafts_path(project_path)
    if images is not None and not isinstance(images, dict):
        raise ValueError("images 必须是 {图片 id: 草稿数组} 的对象")
    if review is not None and not isinstance(review, dict):
        raise ValueError("review 必须是 {图片 id: 审核记录} 的对象")

    # 键统一成字符串：JSON 里数字键会被转成字符串，直接比较会永远不相等。
    source = {str(key): value for key, value in (images or {}).items()}
    updates, _meta, _warnings = normalize({"images": source})
    review_updates, _ = normalize_review(
        {str(key): value for key, value in (review or {}).items()}
    )

    with _lock_for(target):
        current, disk_meta = load(project_path)  # load 不取写锁，此处已持有，读是安全的
        current_review = dict(disk_meta.get("review") or {})
        for image_id, drafts in updates.items():
            current[image_id] = drafts
        for image_id in source:
            if image_id not in updates:
                # 传了键但规范化后是空（比如用户全部接受/丢弃）：删掉记录，
                # 否则文件里留一堆空数组，读回来还要再过滤一遍。
                current.pop(image_id, None)
        for image_id, record in review_updates.items():
            current_review[image_id] = record
        for image_id, raw in (review or {}).items():
            key = str(image_id)
            state = raw.get("state") if isinstance(raw, dict) else raw
            if str(state or "").strip().lower() == "unchecked":
                current_review.pop(key, None)
        return _write_locked(target, current, current_review, project_path)


def set_review(project_path: str | Path, image_id: str, state: str = "checked") -> dict:
    """单张图片的审核状态。返回最新的 meta。"""
    key = str(image_id or "").strip()
    if not key:
        raise ValueError("图片 id 不能为空")
    state = str(state or "checked").strip().lower()
    if state not in REVIEW_STATES:
        raise ValueError(f"审核状态只支持 {'/'.join(REVIEW_STATES)}，收到「{state}」")
    return merge(project_path, review={key: {"state": state, "at_ns": str(time.time_ns())}})


def mark_reviewed(project_path: str | Path, image_id: str) -> dict:
    return set_review(project_path, image_id, "checked")


def unmark_reviewed(project_path: str | Path, image_id: str) -> dict:
    return set_review(project_path, image_id, "unchecked")


def clear(project_path: str | Path, include_review: bool = False) -> dict:
    """清空草稿。

    **默认保留审核状态**：用户点"清空草稿"的意思是"这批候选我不要了，重新跑一次"，
    而不是"把这个工程的审核进度也忘掉"。要连审核一起清，必须显式传
    `include_review=True`，对应"整份替换"。
    """
    target = drafts_path(project_path)
    with _lock_for(target):
        if include_review:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return {**_empty_meta(target), "warnings": [f"删除草稿文件失败：{exc}"]}
            return _empty_meta(target)
        _images, meta = load(project_path)
        review = meta.get("review") or {}
        # 没有审核状态要保 → 直接删文件。留一个"空草稿池"的 JSON 会让
        # "这个工程有没有跑过预标注"变得看不出来（文件存在但内容全空）。
        if not review or not target.exists():
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return {**_empty_meta(target), "warnings": [f"删除草稿文件失败：{exc}"]}
            return {**_empty_meta(target), "review": review, "reviewed": len(review)}
        return _write_locked(target, {}, review, project_path)
