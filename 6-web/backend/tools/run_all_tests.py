# -*- coding: utf-8 -*-
"""后端测试总入口：一条命令跑完全部 `test_*.py`，CI 与本地共用。

为什么要有它：
- 以前 12 个脚本靠人肉逐个执行，改错一行没人拦；
- CI 里逐个写步骤会漏掉「以后新增的脚本」，这里用目录扫描自动收录。

约定：脚本退出码 0 = 通过，非 0 = 失败；输出里的「通过 N 项」只用于展示，
判定只看退出码（避免各脚本输出格式不一导致误判）。

用法：
    python 6-web/backend/tools/run_all_tests.py            # 全部
    python 6-web/backend/tools/run_all_tests.py transd     # 只跑名字含 transd 的
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
#: 单个脚本超时。最慢的 test_annotation_project 约 3 秒，留足余量给慢机器
TIMEOUT_SECONDS = 600


def discover(filters: list[str]) -> list[Path]:
    files = sorted(TOOLS.glob("test_*.py"))
    if filters:
        files = [f for f in files if any(part in f.name for part in filters)]
    return files


def run_one(path: Path) -> tuple[bool, str, float]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(TOOLS),
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"超时（>{TIMEOUT_SECONDS}s）", time.time() - started

    output = (proc.stdout or b"").decode("utf-8", errors="replace")
    err = (proc.stderr or b"").decode("utf-8", errors="replace")

    # 只取最后一行有信息量的内容作为摘要，失败时附上尾部的错误输出
    summary = ""
    for line in reversed([ln for ln in output.splitlines() if ln.strip()]):
        if "通过" in line or "PASS" in line or "FAIL" in line:
            summary = line.strip()
            break
    if not summary:
        tail = (output + err).strip().splitlines()
        summary = tail[-1].strip() if tail else "(无输出)"

    if proc.returncode != 0:
        # 失败时把 stdout 尾部也带上，CI 日志里能直接看到哪条断言挂了
        tail_lines = (output + err).strip().splitlines()[-12:]
        summary = summary + "\n      " + "\n      ".join(tail_lines)
    return proc.returncode == 0, summary, time.time() - started


def main(argv: list[str]) -> int:
    files = discover(argv[1:])
    if not files:
        print("没有匹配到任何 test_*.py")
        return 1

    print(f"共 {len(files)} 个测试脚本，解释器：{sys.executable}\n")
    failed: list[str] = []
    for path in files:
        ok, summary, elapsed = run_one(path)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {path.name:<34} {elapsed:5.1f}s  {summary}")
        if not ok:
            failed.append(path.name)

    print(f"\n结果：{len(files) - len(failed)} 个通过，{len(failed)} 个失败")
    if failed:
        print("失败脚本：" + "、".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
