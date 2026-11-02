# -*- coding: utf-8 -*-
"""预标注草稿的旁挂持久化：`<工程名>.drafts.json`。

为什么草稿不写进 `.tfanno`：
- Qt 版保存时**整体重写**工程文件，未知键不会被回写。草稿如果住在 `.tfanno` 里，
  用户用 Qt 打开再存一次，Web 端刚跑出来的草稿就静默消失了；
- 草稿是过程数据（模型给的候选），正式标注是结论（人确认过的）。两者生命周期不同：
  导出训练集只能带后者，但刷新页面时前者必须还在。

文件结构（`version` 用于以后加字段，不做无意义的前向兼容猜测）::

    {
      "version": 1,
      "project": "demo.tfanno",
      "updated_ns": "1700000000000000000",
      "count": 12,
      "images": { "<图片 id>": [ {草稿对象}, ... ] }
    }

并发约定：**同一工程路径的读-改-写必须串行**。批量预标注线程与前端自动保存会同时
写同一个文件，不串行就会互相覆盖（典型表现：批量跑完只剩最后几张图的草稿）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

import annotation

#: 旁挂文件后缀。`foo.tfanno` → `foo.drafts.json`（同目录、同主名）。
DRAFT_SUFFIX = ".drafts.json"
CURRENT_VERSION = 1
#: 单张图片的草稿上限。防御性上限：坏文件不该把服务进程的内存打爆。
MAX_DRAFTS_PER_IMAGE = 5000

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
        "warnings": [],
    }


def count_drafts(images: dict) -> int:
    """统计草稿总数。容忍任何非列表值（坏数据不该让统计接口崩）。"""
    total = 0
    for value in (images or {}).values():
        if isinstance(value, list):
            total += len(value)
    return total


def normalize(raw) -> tuple[dict, dict, list[str]]:
    """把任意来源的旁挂内容规范成 ``(images, meta, warnings)``。

    单条草稿走 `annotation.normalize_object`，与正式标注共用同一套形状回退规则——
    否则草稿能显示、接受后却变成非法标注，问题会拖到导出阶段才暴露。
    """
    warnings: list[str] = []
    if not isinstance(raw, dict):
        if raw is not None:
            warnings.append("草稿文件根结构不是 JSON 对象，已按空草稿处理")
        return {}, {"version": CURRENT_VERSION, "updated_ns": "0", "count": 0}, warnings

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
                warnings.append(
                    f"图片「{key}」的草稿超过 {MAX_DRAFTS_PER_IMAGE} 条，其余已忽略"
                )
                break
        if kept:
            images[key] = kept

    updated_ns = str(raw.get("updated_ns") or "0")
    meta = {
        "version": annotation.as_int(raw.get("version"), CURRENT_VERSION),
        "updated_ns": updated_ns if updated_ns.isdigit() else "0",
        "count": count_drafts(images),
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


def load(project_path: str | Path) -> tuple[dict, dict]:
    """读取旁挂草稿。

    返回 ``(images, meta)``。文件不存在时返回空表且 ``exists=False``——
    「没有草稿」是正常状态，不该让打开工程的流程失败。
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
        warnings=warnings,
    )
    return images, meta


def _write_locked(target: Path, images: dict, project_path: str | Path) -> dict:
    """``target`` 的写锁已持有时调用：原子写整份文件。"""
    payload = {
        "version": CURRENT_VERSION,
        "project": Path(project_path).name,
        "updated_ns": str(time.time_ns()),
        "count": count_drafts(images),
        "images": images,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    annotation._atomic_write(target, text)
    return {
        "path": str(target),
        "exists": True,
        "updated_ns": payload["updated_ns"],
        "count": payload["count"],
        "image_count": len(images),
        "warnings": [],
    }


def save(project_path: str | Path, images: dict) -> dict:
    """整份替换（前端"清空全部草稿"与测试用）。"""
    target = drafts_path(project_path)
    cleaned, _meta, _warnings = normalize({"images": images or {}})
    with _lock_for(target):
        return _write_locked(target, cleaned, project_path)


def merge(project_path: str | Path, images: dict) -> dict:
    """按图片 id 增量覆盖，**其余图片原样保留**。

    这是前端自动保存与批量任务共用的写入口：
    - 前端只提交被改过的图片，不会把批量刚写进去的其他图片清掉；
    - 传空数组表示"这张图没有草稿了"，对应记录会被删除。
    """
    target = drafts_path(project_path)
    if not isinstance(images, dict):
        raise ValueError("images 必须是 {图片 id: 草稿数组} 的对象")
    # 键统一成字符串：JSON 里数字键会被转成字符串，直接比较会永远不相等。
    source = {str(key): value for key, value in images.items()}
    updates, _meta, _warnings = normalize({"images": source})
    with _lock_for(target):
        current, _disk = load(project_path)  # load 不取写锁，此处已持有，读是安全的
        for image_id, drafts in updates.items():
            current[image_id] = drafts
        for image_id in source:
            if image_id not in updates:
                # 传了键但规范化后是空（比如用户全部接受/丢弃）：删掉记录，
                # 否则文件里留一堆空数组，读回来还要再过滤一遍。
                current.pop(image_id, None)
        return _write_locked(target, current, project_path)


def clear(project_path: str | Path) -> dict:
    """删除旁挂文件（新建工程覆盖旧工程时用）。"""
    target = drafts_path(project_path)
    with _lock_for(target):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            return {**_empty_meta(target), "warnings": [f"删除草稿文件失败：{exc}"]}
        return _empty_meta(target)
