# -*- coding: utf-8 -*-
"""任务运行台账：把 RunManager 的内存状态落成一份可查的历史。

## 为什么不是「把 Run 对象 pickle 到磁盘」

要做的事只有两件：**服务重启后还能列出历史任务**、**能看出某次任务是正常结束
还是被停掉/被系统杀掉的**。这不需要恢复进程句柄、事件缓冲、订阅队列，只需要
一小份元数据。把整个 Run 序列化下来既不可靠（里面有 Popen、asyncio.Queue、
threading.Lock），也会因为字段变动而读不出来。

## 为什么是追加式 JSONL 而不是每任务一个 JSON

- 追加写是 O(1)，不需要「临时文件 + 替换」，任务量大时也不会产生大量小文件；
- 崩溃安全：进程被强杀时最多坏掉最后半行，``json.loads`` 失败就跳过这一行，
  前面的记录完好；
- 与 ``7-autoresearch-yolo/results.tsv`` 的「一行一个状态」思路一致，读起来直观。

每条记录都是**完整快照**（不是增量），所以读取时「后出现的覆盖先出现的」即可，
不必做字段合并——合并逻辑是这类系统最容易出错的地方。

## 写入失败怎么办

磁盘满、目录只读时 ``append`` 返回 False 并往 stderr 打一行。**台账只是运维信息**，
它丢了不该把正在跑的训练判成失败：调用方一律忽略返回值。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import config

#: 台账超过这个体积就折叠一次（只保留最近若干任务的最终状态）。
#: 单条约 600 字节、每任务两条，8MB 大约够 6000 次任务，正常实验室用量很难摸到。
MAX_BYTES = 8 * 1024 * 1024
#: 折叠后保留的任务数。
KEEP_ENTRIES = 2000

_LOCK = threading.Lock()


def ledger_path() -> Path:
    """台账文件路径。测试用 ``config.RUNS_LEDGER`` 覆盖即可换位置。"""
    return Path(config.RUNS_LEDGER)


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def append(record: dict[str, Any]) -> bool:
    """追加一条记录。任何异常都不外抛，只返回是否写成功。"""
    if not isinstance(record, dict) or not record.get("run_id"):
        return False
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return False
    try:
        with _LOCK:
            path = ledger_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:
        # 台账不可写不能影响任务本身，只在服务日志里留个线索
        print(f"[runledger] 台账写入失败：{type(exc).__name__}: {exc}", flush=True)
        return False
    return True


def record_started(
    run_id: str,
    kind: str,
    command: list[str],
    log_file: Path | str | None,
    started_at: float | None = None,
    meta: dict[str, Any] | None = None,
    resumed_from: str | None = None,
) -> bool:
    """任务启动时的那条记录。"""
    return append(
        {
            "event": "started",
            "run_id": run_id,
            "kind": kind,
            "status": "running",
            "exit_code": None,
            "started_at": float(started_at if started_at is not None else time.time()),
            "finished_at": None,
            "command": list(command),
            "log_file": str(log_file) if log_file else None,
            "meta": dict(meta or {}),
            "resumed_from": resumed_from,
        }
    )


def record_finished(
    run_id: str,
    kind: str,
    status: str,
    exit_code: int | None,
    started_at: float,
    finished_at: float | None = None,
    command: list[str] | None = None,
    log_file: Path | str | None = None,
    meta: dict[str, Any] | None = None,
    error: str | None = None,
    resumed_from: str | None = None,
) -> bool:
    """任务结束时的那条记录（完整快照，覆盖之前的状态）。"""
    end = float(finished_at if finished_at is not None else time.time())
    return append(
        {
            "event": "finished",
            "run_id": run_id,
            "kind": kind,
            "status": status,
            "exit_code": exit_code,
            "started_at": float(started_at),
            "finished_at": end,
            "command": list(command or []),
            "log_file": str(log_file) if log_file else None,
            "meta": dict(meta or {}),
            "error": error,
            "resumed_from": resumed_from,
        }
    )


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def load_records() -> list[dict[str, Any]]:
    """按写入顺序返回全部记录，坏行跳过。"""
    path = ledger_path()
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            # 进程被强杀导致最后一行半截：跳过，不影响前面的记录
            continue
        if isinstance(data, dict) and data.get("run_id"):
            records.append(data)
    return records


def fold(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 run_id 折叠成「最后一条为准」。"""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["run_id"])] = record
    return latest


def latest_states() -> dict[str, dict[str, Any]]:
    return fold(load_records())


def to_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """把台账记录转成与 ``Run.snapshot()`` 同构的字典，前端可以复用渲染逻辑。

    ``history=True`` 是给前端的标记：这条只能看日志，不能停止。
    """
    started = float(record.get("started_at") or 0.0)
    finished = record.get("finished_at")
    end = float(finished) if finished else started
    return {
        "run_id": str(record.get("run_id") or ""),
        "kind": str(record.get("kind") or ""),
        "status": str(record.get("status") or "unknown"),
        "exit_code": record.get("exit_code"),
        "started_at": started,
        "finished_at": finished,
        "elapsed": round(max(0.0, end - started), 1),
        "meta": dict(record.get("meta") or {}),
        "warm": bool(record.get("warm", False)),
        "command": list(record.get("command") or []),
        "log_file": record.get("log_file"),
        "history": True,
        "orphan": bool(record.get("orphan", False)),
        "resumed_from": record.get("resumed_from"),
        "error": record.get("error"),
    }


def history(limit: int = 50) -> list[dict[str, Any]]:
    """历史任务（新的在前），最多 ``limit`` 条。"""
    latest = latest_states()
    items = sorted(latest.values(), key=lambda r: float(r.get("started_at") or 0.0), reverse=True)
    if limit > 0:
        items = items[:limit]
    return [to_snapshot(record) for record in items]


# ---------------------------------------------------------------------------
# 启动恢复与折叠
# ---------------------------------------------------------------------------
def recover_interrupted(now: float | None = None) -> list[dict[str, Any]]:
    """把台账里仍为 ``running`` 的记录改判为 ``interrupted``。

    只有在「确认这些进程已经不在了」时调用（web 服务启动时）。子进程可能确实
    还在跑（父进程被强杀），但那时候前端也控制不了它，如实标成「已中断」比留着
    一个假 running 更不容易误导人。
    """
    stamp = float(now if now is not None else time.time())
    recovered: list[dict[str, Any]] = []
    for record in latest_states().values():
        if str(record.get("status")) != "running":
            continue
        updated = dict(record)
        updated.update(
            {
                "event": "interrupted",
                "status": "interrupted",
                "orphan": True,
                "finished_at": stamp,
                "error": "服务重启，任务已中断（无法确认子进程是否仍在运行）",
            }
        )
        if append(updated):
            recovered.append(updated)
    return recovered


def compact(force: bool = False) -> bool:
    """台账过大时按「每任务只留最后一条」重写，保留最近 KEEP_ENTRIES 个任务。"""
    path = ledger_path()
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if not force and size < MAX_BYTES:
        return False
    latest = latest_states()
    kept = sorted(latest.values(), key=lambda r: float(r.get("started_at") or 0.0))[-KEEP_ENTRIES:]
    lines = [json.dumps(record, ensure_ascii=False, default=str) for record in kept]
    temp = path.with_suffix(".jsonl.part")
    try:
        temp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(temp, path)
    except OSError as exc:
        print(f"[runledger] 台帐折叠失败：{type(exc).__name__}: {exc}", flush=True)
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True
