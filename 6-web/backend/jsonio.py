# -*- coding: utf-8 -*-
"""原子写文件（JSON / 文本），一个进程内只保留一份实现。

## 为什么必须原子

这些文件里有「用户几个小时的标注工程」「自动实验一整晚的台账」「显存采样」。
直接 ``write_text`` 时如果进程被杀、盘满、杀毒软件锁住句柄，留下的是**半截
JSON**：下次读到 ``json.loads`` 直接抛，用户的工程看起来就「坏了」，而实际上
数据只丢最后一次保存。做法统一成：写临时文件 → ``os.replace``（同分区原子）。

## 临时文件名的两条要求

1. **带 pid**：两个进程同时写同一个目标时不能互相覆盖；
2. **带 uuid8**：同一进程内的两个线程/请求也不能互相覆盖。

只带 pid 不够——``annotation.py`` 里两个并发保存请求曾用同一个临时名，后一个
把前一个的半截内容 replace 成正式文件，结果是「保存成功但内容缺了一半」。
这类问题复现不了，只能靠命名规避。

## 为什么要 ``finally`` 清理

失败时临时文件必须删掉。``test_annotation_project.py`` 有一条断言专门扫工程
目录里的 ``.tmp*`` 残留——那些文件会让「另存为」把垃圾一起带走，用户看到的
是工程里多出一堆看不懂的文件。

## 与 :func:`os.replace` 的关系

``os.replace`` 在 Windows 上对「目标正被别的进程打开」会抛 ``PermissionError``。
本模块**不重试**：调用方（台账、缓存、工程保存）本来就把写盘失败当作可降级的
情况处理，静默重试只会掩盖问题、把请求拖慢。
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

__all__ = ["temp_path", "write_text", "write_json", "read_json"]


def temp_path(target: Path | str) -> Path:
    """目标文件对应的临时路径（同目录，保证 ``os.replace`` 不跨盘）。"""
    target = Path(target)
    return target.with_name(f"{target.name}.tmp{os.getpid()}.{uuid.uuid4().hex[:8]}")


def write_text(target: Path | str, text: str, *, newline: str | None = None) -> None:
    """原子写文本。

    ``newline`` 传 ``"\\n"`` 可以把 Windows 的 ``\\r\\n`` 关掉——标注工程的
    ``project.json`` 要求如此（跨机器协作时 CRLF 会让 diff 全是噪音）。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = temp_path(target)
    try:
        temp.write_text(text, encoding="utf-8", newline=newline)
        os.replace(temp, target)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def write_json(
    target: Path | str,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    newline: str | None = None,
) -> None:
    """原子写 JSON。默认 ``indent=2`` + 中文不转义（人能直接读、diff 也小）。

    需要紧凑格式（缓存文件）时传 ``indent=None``。
    """
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    write_text(target, text, newline=newline)


def read_json(target: Path | str, default: Any = None) -> Any:
    """读 JSON；文件不存在、内容损坏、编码不对时返回 ``default``。

    这个宽容度是刻意保留的：调用方读的都是「缓存 / 上次状态 / 台账」，
    坏文件不该让整个页面打不开（例如显存采样坏了不能拦住训练）。
    ``default`` 默认 ``None``，需要空字典/空列表时自己传。
    """
    try:
        raw = Path(target).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default
