# -*- coding: utf-8 -*-
"""验证 RunManager 对特殊退出码的语义化处理。

为什么单独测这件事：
launcher 用 argparse 解析参数，参数不合法时 argparse 会直接往 stderr 打 usage 并
exit(2)，**不会**发 TRTFORGE_JSON error 事件。如果不补转换，浏览器只能看到
"退出码 2"，无法区分"参数写错"和"训练崩溃"。

用法：python 6-web/backend/tools/test_exit_codes.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import runners  # noqa: E402


def _fake_script(directory: Path, body: str) -> Path:
    """生成一个临时 py 脚本，用来模拟特定退出码。"""
    script = directory / "fake_launcher.py"
    script.write_text(body, encoding="utf-8")
    return script


def run_and_collect(command: list[str], cwd: str) -> list[dict]:
    """启动一次任务并同步收集全部事件。"""
    loop = asyncio.new_event_loop()
    manager = runners.RunManager()
    manager.bind_loop(loop)
    run = manager.create("test", command, cwd)
    queue, _history = manager.subscribe(run.run_id)

    events: list[dict] = []

    async def collect() -> None:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                break
            events.append(event)
            if event.get("type") == "finished":
                break

    try:
        loop.run_until_complete(collect())
    finally:
        loop.close()
    return events


def main() -> int:
    config.ensure_dirs()
    python = sys.executable
    passed, failed = [], []

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)

        # 场景 1：argparse 参数错误 -> exit 2，应补 ARGUMENT_ERROR 事件
        script = _fake_script(
            directory,
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--epochs', type=int)\n"
            "p.parse_args()\n",
        )
        events = run_and_collect([python, str(script), "--epochs", "not-a-number"], tmp)
        types = [e.get("type") for e in events]
        codes = [e.get("code") for e in events if e.get("type") == "error"]
        if "ARGUMENT_ERROR" in codes:
            passed.append("exit 2 补 ARGUMENT_ERROR 事件")
        else:
            failed.append(f"exit 2 未补 ARGUMENT_ERROR，事件类型={types}")
        if types and types[-1] == "finished":
            passed.append("exit 2 仍以 finished 收尾")
        else:
            failed.append(f"exit 2 收尾事件异常：{types[-1] if types else '无'}")

        # 场景 2：正常退出 0 -> 不应有额外 error 事件
        script_ok = _fake_script(directory, "print('hello ok')\n")
        events = run_and_collect([python, str(script_ok)], tmp)
        codes = [e.get("code") for e in events if e.get("type") == "error"]
        if not codes:
            passed.append("exit 0 无多余 error 事件")
        else:
            failed.append(f"exit 0 出现多余 error：{codes}")
        finished = [e for e in events if e.get("type") == "finished"]
        if finished and finished[0].get("exit") == 0:
            passed.append("exit 0 正常收尾")
        else:
            failed.append(f"exit 0 收尾异常：{finished}")

        # 场景 3：非零但非 2/130 -> 只标记 failed，不伪造错误码
        script_crash = _fake_script(directory, "raise SystemExit(7)\n")
        events = run_and_collect([python, str(script_crash)], tmp)
        codes = [e.get("code") for e in events if e.get("type") == "error"]
        finished = [e for e in events if e.get("type") == "finished"]
        if not codes and finished and finished[0].get("status") == "failed":
            passed.append("exit 7 标记 failed 且不伪造错误码")
        else:
            failed.append(f"exit 7 处理异常：codes={codes}, finished={finished}")

        # 场景 4：结构化事件被正确解析（前缀 + JSON）
        script_json = _fake_script(
            directory,
            "print('TRTFORGE_JSON {\"type\": \"progress\", \"epoch\": 1, \"total\": 10, \"pct\": 10.0}')\n"
            "print('plain log line')\n",
        )
        events = run_and_collect([python, str(script_json)], tmp)
        has_progress = any(e.get("type") == "progress" and e.get("epoch") == 1 for e in events)
        has_log = any(e.get("type") == "log" and "plain log line" in (e.get("line") or "") for e in events)
        if has_progress and has_log:
            passed.append("TRTFORGE_JSON 前缀解析正确，普通日志不受影响")
        else:
            failed.append(f"事件解析异常：progress={has_progress}, log={has_log}")

    print()
    for item in passed:
        print(f"[PASS] {item}")
    for item in failed:
        print(f"[FAIL] {item}")
    print(f"\n通过 {len(passed)} 项，失败 {len(failed)} 项")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
