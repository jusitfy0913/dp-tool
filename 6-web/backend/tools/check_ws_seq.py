# -*- coding: utf-8 -*-
"""对着**正在运行**的服务验一次事件序号：连两次同一个 run，历史里的 seq 必须一致。

用法（服务已启动）：
    python 6-web/backend/tools/check_ws_seq.py <run_id> [host:port]

只读，不启动任何任务：用来证明「断线重连补发的历史带 seq，前端能据此去重」。
"""

from __future__ import annotations

import asyncio
import json
import sys


async def collect_until_finished(url: str, limit: int = 200) -> list[dict]:
    import websockets

    events: list[dict] = []
    async with websockets.connect(url, open_timeout=10) as ws:
        try:
            while len(events) < limit:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                event = json.loads(raw)
                events.append(event)
                if event.get("type") in ("finished", "error"):
                    break
        except asyncio.TimeoutError:
            pass
    return events


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：check_ws_seq.py <run_id> [host:port]")
        return 2
    run_id = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:8060"
    url = f"ws://{host}/ws/runs/{run_id}"

    first = asyncio.run(collect_until_finished(url))
    second = asyncio.run(collect_until_finished(url))

    def seqs(events: list[dict]) -> list:
        return [event.get("seq") for event in events]

    print(f"第一次连接：{len(first)} 条，seq={seqs(first)[:8]}…（末条 {seqs(first)[-1:]}）")
    print(f"第二次连接：{len(second)} 条，seq={seqs(second)[:8]}…（末条 {seqs(second)[-1:]}）")

    ok = True
    if not all(isinstance(s, int) for s in seqs(first) + seqs(second)):
        print("[FAIL] 有事件没有 seq —— 前端无法去重")
        ok = False
    if seqs(first) != seqs(second):
        print("[FAIL] 两次连接拿到的 seq 序列不一致（重连会把历史补两遍）")
        ok = False
    if any(b <= a for a, b in zip(seqs(first), seqs(first)[1:])):
        print("[FAIL] seq 不是严格递增")
        ok = False
    print("[PASS] 历史事件带 seq 且严格递增，重连拿到的序列一致" if ok else "[FAIL] 见上面的原因")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
