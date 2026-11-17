# -*- coding: utf-8 -*-
"""测试总入口：一条命令跑完全部后端 `test_*.py` 与前端 `test_*.mjs`。

为什么要有它：
- 以前 12 个脚本靠人肉逐个执行，改错一行没人拦；
- CI 里逐个写步骤会漏掉「以后新增的脚本」，这里按 `TEST_SOURCES` 清单收录；
- 前端 15 个 `.mjs` 单测以前只能手工 `node` 跑，重构时等于没有护栏，
  现在一并纳入，缺 node 会明确报错而不是静默少跑一层。

约定：脚本退出码 0 = 通过，非 0 = 失败；输出里的「通过 N 项」只用于展示，
判定只看退出码（避免各脚本输出格式不一导致误判）。

用法：
    python 6-web/backend/tools/run_all_tests.py            # 全部
    python 6-web/backend/tools/run_all_tests.py transd     # 只跑名字含 transd 的
    set TRTFORGE_SKIP_NODE_TESTS=1 && python ...run_all_tests.py   # 显式跳过前端
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
#: 单个脚本超时。最慢的 test_annotation_project 约 3 秒，留足余量给慢机器
TIMEOUT_SECONDS = 600

#: 测试来源清单（目录 + 通配）。**新增测试目录必须登记到这里**，
#: 否则「一条命令全跑」就有了盲区：以前只扫 `tools/test_*.py`，
#: 前端 15 个 `.mjs` 单测只能靠人记得手工 node 跑，等于没护栏。
#:
#: 用目录而不是文件清单，是为了「同目录新增文件自动被收录」。
TEST_SOURCES: list[tuple[Path, str]] = [
    (TOOLS, "test_*.py"),
    # AnomalyDINO 的算法与导出契约测试跟着启动器源码走
    # （2-python-anomaly/launcher/tools），只跑 CPU 纯逻辑、几秒结束。
    (TOOLS.parents[2] / "2-python-anomaly" / "launcher" / "tools", "test_*.py"),
    # 前端纯逻辑单测：零依赖、直接用 node 跑（DOM 相关的用手写桩，见各文件头注释）。
    (TOOLS.parents[1] / "frontend" / "tests", "test_*.mjs"),
]


def discover(filters: list[str]) -> list[Path]:
    files: list[Path] = []
    for directory, pattern in TEST_SOURCES:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern)):
            if path not in files:
                files.append(path)
    if filters:
        files = [f for f in files if any(part in f.name for part in filters)]
    return files


def _summary_of(stdout: bytes, stderr: bytes) -> str:
    output = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    # 只取最后一行有信息量的内容作为摘要，失败时附上尾部的错误输出
    summary = ""
    for line in reversed([ln for ln in output.splitlines() if ln.strip()]):
        if "通过" in line or "PASS" in line or "FAIL" in line:
            summary = line.strip()
            break
    if not summary:
        tail = (output + err).strip().splitlines()
        summary = tail[-1].strip() if tail else "(无输出)"
    return summary, output, err


def _test_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # 测试会真的起子进程（退出码、并发、日志轮转那些用例），它们现在会把任务写进
    # 生产台账 runtime/runs.jsonl。指到临时文件后，跑完一轮测试不会把开发机上的
    # 「历史任务」列表塞满 kind=test 的假记录。
    env.setdefault("TRTFORGE_RUNS_LEDGER", str(TOOLS / ".." / ".." / ".." / "runtime" / "runs.test.jsonl"))
    # 体检缓存同理：测试产生的临时路径签名不该写进开发机的正式缓存，否则真实数据集
    # 第一次体检时那份缓存会被挤掉，白等十秒。
    env.setdefault("TRTFORGE_DSQC_CACHE", str(TOOLS / ".." / ".." / ".." / "runtime" / "dsqc_cache.test.json"))
    return env


def run_one(path: Path, command: list[str] | None = None) -> tuple[bool, str, float]:
    """跑一个测试文件。`command` 为空时用当前解释器跑 python 脚本。"""
    started = time.time()
    argv = command if command is not None else [sys.executable, str(path)]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(TOOLS),
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            env=_test_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"超时（>{TIMEOUT_SECONDS}s）", time.time() - started

    summary, output, err = _summary_of(proc.stdout, proc.stderr)

    if proc.returncode != 0:
        # 失败时把 stdout 尾部也带上，CI 日志里能直接看到哪条断言挂了
        tail_lines = (output + err).strip().splitlines()[-12:]
        summary = summary + "\n      " + "\n      ".join(tail_lines)
    return proc.returncode == 0, summary, time.time() - started


def _node_command(path: Path) -> tuple[list[str] | None, str]:
    """前端单测的启动命令。返回 (命令, 不可用原因)。"""
    node = shutil.which("node")
    if node:
        return [node, str(path)], ""
    if os.environ.get("TRTFORGE_SKIP_NODE_TESTS") == "1":
        return None, "skip"
    # 不静默跳过：少跑一层测试等于把护栏拆了却不告警。
    # 真要跳过请显式设 TRTFORGE_SKIP_NODE_TESTS=1。
    return None, "未找到 node（装 Node.js，或显式设 TRTFORGE_SKIP_NODE_TESTS=1 跳过前端单测）"


def main(argv: list[str]) -> int:
    files = discover(argv[1:])
    if not files:
        print("没有匹配到任何测试脚本（检查 TEST_SOURCES 清单与过滤词）")
        return 1

    print(f"共 {len(files)} 个测试脚本，解释器：{sys.executable}\n")
    failed: list[str] = []
    skipped: list[str] = []
    for path in files:
        if path.suffix == ".mjs":
            command, reason = _node_command(path)
            if command is None:
                if reason == "skip":
                    print(f"[SKIP] {path.name:<34}       前端单测按 TRTFORGE_SKIP_NODE_TESTS 跳过")
                    skipped.append(path.name)
                else:
                    print(f"[FAIL] {path.name:<34}       {reason}")
                    failed.append(path.name)
                continue
            ok, summary, elapsed = run_one(path, command)
        else:
            ok, summary, elapsed = run_one(path)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {path.name:<34} {elapsed:5.1f}s  {summary}")
        if not ok:
            failed.append(path.name)

    print(f"\n结果：{len(files) - len(failed) - len(skipped)} 个通过，{len(failed)} 个失败")
    if skipped:
        print("跳过：" + "、".join(skipped) + "（TRTFORGE_SKIP_NODE_TESTS=1）")
    if failed:
        print("失败脚本：" + "、".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
