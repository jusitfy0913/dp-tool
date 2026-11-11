# -*- coding: utf-8 -*-
"""任务日志的滚动写入与跨分片读取。

为什么要有这一层：

1. **磁盘会被写满**。训练日志十几到几十 MB 很常见，引擎构建更啰嗦；老版本
   ``runtime/logs/{run_id}.log`` 只增不减，7×24 跑下去迟早撑爆磁盘。
2. **滚动不能让「看日志」失效**。日志一旦分成 ``.log`` / ``.log.1`` / ``.log.2``，
   读取方（``/api/runs/{id}/log``）必须知道怎么把几份拼回去，否则用户拿到的
   只是最后一段，前面的报错照样看不到。

滚动规则（刻意按字节、不按时间）：
    当前文件写满 ``max_bytes`` 就把 ``.log`` 改名为 ``.log.1``，``.log.1``→``.log.2``…
    超出 ``backups`` 份的最老一份删掉（``.log.N`` 的数字越大越旧）。
    之所以不按时间：训练日志的写入速率跟 epoch 强相关，按小时滚动可能在某次
    长 epoch 里一次长出几百 MB，也可能整个任务都没到一小时而不落盘。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from config import LOG_BACKUPS, LOG_MAX_BYTES

import config

#: 分片读取时的单块大小。64KB 与原先 app._read_log_tail 的取值保持一致。
_READ_BLOCK = 64 * 1024


def _log_dir(log_dir: Path | None) -> Path:
    """日志根目录。

    刻意在调用时读 ``config.LOG_DIR`` 而不是 ``from config import LOG_DIR``：
    后者在 import 那一刻就固定了，测试把它指到临时目录时不生效，
    于是「测试写临时目录、代码读真实目录」——这类错误很难看出来。
    """
    return Path(log_dir) if log_dir is not None else Path(config.LOG_DIR)


def shard_paths(run_id: str, log_dir: Path | None = None) -> list[Path]:
    """该任务的日志分片，**从新到旧**排列。

    只返回真实存在的文件：首次启动时可能一份都没有（任务还没写第一行）。
    """
    base = _log_dir(log_dir) / f"{run_id}.log"
    paths = [base]
    for index in range(1, max(0, LOG_BACKUPS) + 1):
        paths.append(base.with_name(f"{base.name}.{index}"))
    return [path for path in paths if path.is_file()]


def any_shard(run_id: str, log_dir: Path | None = None) -> bool:
    """是否至少存在一份日志（含已滚动的分片）。"""
    return bool(shard_paths(run_id, log_dir))


class RunLogSink:
    """一个任务的日志写入器：按体积滚动，写失败不影响任务。

    这个类是**线程私有**的（每个读线程一个），内部不加锁：同一任务的日志只由
    一个读线程写入，加锁只会增加一次无谓的开销。
    """

    def __init__(
        self,
        path: Path,
        max_bytes: int | None = None,
        backups: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = int(max_bytes if max_bytes is not None else LOG_MAX_BYTES)
        self.backups = max(0, int(backups if backups is not None else LOG_BACKUPS))
        self.written = 0
        self._handle = None
        self._open()

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8", errors="replace")
            try:
                self.written = self.path.stat().st_size
            except OSError:
                self.written = 0
        except OSError:
            # 目录不可写（安装包只读、盘满）：日志丢了也不能让训练挂掉
            self._handle = None
            self.written = 0

    def write(self, line: str) -> None:
        """写一行（调用方已去掉换行符）。"""
        if self._handle is None:
            return
        payload = line + "\n"
        size = len(payload.encode("utf-8", errors="replace"))
        # 单行就超过上限时不要滚动：滚完照样写不进去，只会把历史分片全挤掉
        if self.written > 0 and self.written + size > self.max_bytes and size < self.max_bytes:
            self._rotate()
        try:
            self._handle.write(payload)
            self._handle.flush()
            self.written += size
        except OSError:
            self._handle = None

    def _rotate(self) -> None:
        assert self._handle is not None
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None
        # 从最旧的开始挪：.N 丢弃，.N-1 → .N，…，.log → .log.1
        try:
            if self.backups <= 0:
                self.path.unlink(missing_ok=True)
            else:
                oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
                oldest.unlink(missing_ok=True)
                for index in range(self.backups - 1, 0, -1):
                    src = self.path.with_name(f"{self.path.name}.{index}")
                    if src.is_file():
                        os.replace(src, self.path.with_name(f"{self.path.name}.{index + 1}"))
                if self.path.is_file():
                    os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        except OSError:
            # 改名失败就继续往原文件追加：宁可超出上限，也不能丢日志
            pass
        self._open()

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    def __enter__(self) -> "RunLogSink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_tail(run_id: str, limit: int, log_dir: Path | None = None) -> str:
    """跨分片取最后 ``limit`` 行。

    分片从新到旧读：先把最新那份的尾部取够，不够再往前一份补。这样即使日志
    已经滚了几十 MB，接口也只读真正需要的字节数。
    """
    if limit <= 0:
        return ""
    remaining = limit
    chunks: list[list[str]] = []
    for path in shard_paths(run_id, log_dir):
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                data = b""
                while size > 0 and data.count(b"\n") <= remaining:
                    step = min(_READ_BLOCK, size)
                    size -= step
                    handle.seek(size)
                    data = handle.read(step) + data
        except OSError:
            continue
        lines = data.decode("utf-8", errors="replace").splitlines()
        if not lines:
            continue
        take = lines[-remaining:]
        chunks.append(take)
        remaining -= len(take)
        if remaining <= 0:
            break
    # chunks 是新→旧的顺序，拼回去要从旧到新
    ordered: list[str] = []
    for part in reversed(chunks):
        ordered.extend(part)
    return "\n".join(ordered[-limit:])


def read_all(run_id: str, log_dir: Path | None = None) -> str:
    """跨分片取全文（旧→新）。"""
    parts: list[str] = []
    for path in reversed(shard_paths(run_id, log_dir)):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "".join(parts)


#: 任务日志的文件名格式：``{12 位十六进制}.log`` 或 ``{...}.log.N``。
#: 只认这个格式，是为了不误删同目录里的其它文件（例如推理进程自己的日志）。
_RUN_LOG_NAME = re.compile(r"^([0-9a-f]{12})\.log(?:\.(\d+))?$")


def prune_logs(
    max_age_days: int | None = None,
    log_dir: Path | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """删除过期的任务日志族（一次任务的所有分片一起删）。

    为什么按「整个任务」判断而不是按单个文件：分片是被滚动挪出来的，
    只删旧分片会留下最新的 ``.log``，任务看起来还在，日志却是断头的。

    ``max_age_days <= 0`` 时直接返回零，表示用户选择永久留档。
    """
    days = int(max_age_days if max_age_days is not None else config.LOG_RETENTION_DAYS)
    if days <= 0:
        return {"families": 0, "files": 0, "freed_bytes": 0}

    root = _log_dir(log_dir)
    if not root.is_dir():
        return {"families": 0, "files": 0, "freed_bytes": 0}

    stamp = float(now if now is not None else time.time())
    cutoff = stamp - days * 86400.0

    families: dict[str, list[Path]] = {}
    try:
        entries = list(root.iterdir())
    except OSError:
        return {"families": 0, "files": 0, "freed_bytes": 0}
    for entry in entries:
        if not entry.is_file():
            continue
        match = _RUN_LOG_NAME.match(entry.name)
        if not match:
            continue
        families.setdefault(match.group(1), []).append(entry)

    removed_families = 0
    removed_files = 0
    freed = 0
    for paths in families.values():
        newest = 0.0
        for path in paths:
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
        if newest >= cutoff:
            continue
        deleted_any = False
        for path in paths:
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
                removed_files += 1
                deleted_any = True
            except OSError:
                continue
        if deleted_any:
            removed_families += 1
    return {"families": removed_families, "files": removed_files, "freed_bytes": freed}
