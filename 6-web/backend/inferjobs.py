# -*- coding: utf-8 -*-
"""批量 / 视频推理的任务目录、产物命名与落盘工具。

为什么要单独一个模块：
    目录批量与视频是两条独立的子进程脚本，但它们对「产物放哪、叫什么名字、
    CSV 用什么编码、下载接口怎么校验路径」的答案必须完全一致。各写一遍的结果
    是 zip 里少一层目录、CSV 在 Excel 里乱码、下载接口漏校验这类问题。

产物布局（每个任务一个目录，互不覆盖）：
    runtime/infer/<时间戳>-<类型>-<随机6位>/
        job.json          本次请求参数 + 汇总（用来复现）
        annotated/        逐图标注结果（批量）
        results.json      完整结果（含每个框）
        results.csv       供 Excel 直接打开
        annotated.zip     打包下载
        <name>.mp4        视频结果

CSV 一律用 ``utf-8-sig``：Windows Excel 读不带 BOM 的 UTF-8 会把中文类别名
显示成乱码，这是现场最常被投诉的一条。
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import config

#: 推理产物的根目录。下载接口只允许读这里的文件——白名单到 RUNTIME_DIR 太宽，
#: 里面还有上传临时文件与日志。
INFER_ROOT = config.RUNTIME_DIR / "infer"


def ensure_root() -> Path:
    """确保产物根目录存在，返回它。"""
    INFER_ROOT.mkdir(parents=True, exist_ok=True)
    return INFER_ROOT


def new_job_dir(kind: str) -> Path:
    """为一次批量/视频任务创建独立目录。

    名字里带时间戳与随机后缀：同一秒连点两次「开始」也不会写进同一个目录，
    否则第二次的产物会悄悄覆盖第一次，用户拿到的 zip 说不清是哪一次。
    """
    ensure_root()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kind_part = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in kind)[:24]
    for _ in range(8):
        job = INFER_ROOT / f"{stamp}-{kind_part}-{uuid.uuid4().hex[:6]}"
        try:
            (job / "annotated").mkdir(parents=True, exist_ok=False)
            return job
        except FileExistsError:
            continue
    raise OSError(f"无法创建推理产物目录：{INFER_ROOT}")


def safe_stem(name: str, limit: int = 60) -> str:
    """把源文件名压成可安全落盘的前缀。

    只保留字母/数字/``-_.``：中文名保留原样（Windows + UTF-8 没问题），
    但空格、括号、``#`` 这类字符在某些解压工具里会把路径截断。
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    cleaned = cleaned.strip("._") or "image"
    return cleaned[:limit]


def annotated_name(index: int, source: Path, suffix: str = ".jpg") -> str:
    """标注图文件名：序号打头，保证同目录内唯一且排序与原目录一致。"""
    return f"{index:05d}_{safe_stem(Path(source).stem)}{suffix}"


def write_json(path: str | Path, payload) -> Path:
    """写 UTF-8 JSON（不转义中文），返回路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def write_csv(path: str | Path, columns: list[str], rows: list[dict]) -> Path:
    """写 CSV，编码为 ``utf-8-sig`` 以便 Excel 直接打开。

    ``newline=""`` 是 csv 模块的硬性要求，缺了会在 Windows 上每行多一个空行。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8-sig", newline="") as handle:
        # 显式指定 \n：默认是 \r\n，在 Windows 上 git diff 会显示整文件改动，
        # 而且用户拿 csv 做脚本处理时还得多洗一次行尾。
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})
    return target


def _cell(value) -> object:
    """把 None / 浮点收成 CSV 友好的值。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def make_zip(zip_path: str | Path, files: list[Path], base_dir: str | Path) -> Path:
    """把 ``files`` 打包成 zip，归档内路径相对 ``base_dir``。

    写之前先删旧文件：中途失败留下的半截 zip 会让用户以为「下载下来是坏的」。
    先写 ``.part`` 再改名，保证任何时刻看到的 zip 都是完整的。
    """
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    base = Path(base_dir)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in files:
            file = Path(item)
            if not file.is_file():
                continue
            try:
                arcname = file.relative_to(base).as_posix()
            except ValueError:
                arcname = file.name
            archive.write(file, arcname)
    os.replace(tmp, target)
    return target


def resolve_artifact(raw: str) -> Path:
    """把下载请求里的路径收敛到 `runtime/infer` 之内。

    Web 端下载接口最容易变成「读任意磁盘」的后门，所以不信任传入路径：
    先 resolve（会展开 ``..``），再要求它真在产物根目录下面。
    """
    if not str(raw or "").strip():
        raise ValueError("缺少产物路径")
    root = INFER_ROOT.resolve()
    target = Path(str(raw)).expanduser().resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"只允许下载推理产物目录内的文件：{root}")
    return target


def guess_media_type(path: str | Path) -> str:
    """下载时的 Content-Type。按扩展名给，未知一律 octet-stream。"""
    suffix = Path(path).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".zip": "application/zip",
        ".json": "application/json; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
    }.get(suffix, "application/octet-stream")


def collect_images(
    source: str | Path,
    recursive: bool = True,
    limit: int = 0,
    suffixes: set[str] | None = None,
) -> list[Path]:
    """收集待推理的图片，按路径排序。

    排序是必须的：``iterdir`` 的顺序依赖文件系统，同一目录两次跑出来的
    ``results.csv`` 行序不同，用户没法用 diff 比较两次结果。

    ``limit`` > 0 时只取前 N 张（按排序后的顺序），用于先试跑一小批。
    """
    root = Path(str(source)).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"路径不存在：{root}")
    wanted = suffixes or set(config.IMAGE_SUFFIXES)

    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        iterator = root.rglob("*") if recursive else root.glob("*")
        candidates = [p for p in iterator if p.is_file()]
    else:
        raise FileNotFoundError(f"不是文件也不是目录：{root}")

    images = sorted(
        (p for p in candidates if p.suffix.lower() in wanted),
        key=lambda p: str(p).lower(),
    )
    if limit and limit > 0:
        images = images[:limit]
    return images
