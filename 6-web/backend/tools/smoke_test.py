# -*- coding: utf-8 -*-
"""Web Studio 端到端冒烟测试。

覆盖：
1. HTTP 能力探测
2. 启动 launcher 环境自检（真实子进程）
3. WebSocket 事件流（真实连通）
4. 训练接口的参数校验（应 400 而不是 500）
5. 目录越权访问拦截

用法：
    python 6-web/backend/tools/smoke_test.py
前置：服务已在 http://127.0.0.1:8060 运行。
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8060"
WS_BASE = "ws://127.0.0.1:8060"

passed: list[str] = []
failed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        passed.append(name)
        print(f"[PASS] {name}")
    else:
        failed.append(f"{name} {detail}".strip())
        print(f"[FAIL] {name} {detail}")


def test_capabilities(client: httpx.Client) -> None:
    data = client.get(f"{BASE}/api/capabilities").json()
    check("能力探测返回 python", bool(data.get("python")), str(data.get("python")))
    check("两个 launcher 存在", bool(data.get("yolo_launcher") and data.get("anomalib_launcher")))
    check("两个推理库已加载", bool(data.get("infer_yolo") and data.get("infer_anomalib")), data.get("infer_detail", ""))


def test_validation(client: httpx.Client) -> None:
    # dataset.yaml 不存在 -> 400，而不是 500
    response = client.post(
        f"{BASE}/api/train/yolo",
        json={"data": "D:/definitely/not/exists.yaml"},
    )
    check("非法 dataset.yaml 返回 400", response.status_code == 400, f"got {response.status_code}")

    # 数据集 root 不存在 -> 400
    response = client.post(
        f"{BASE}/api/train/anomalib",
        json={"root": "D:/definitely/not/exists"},
    )
    check("非法数据集 root 返回 400", response.status_code == 400, f"got {response.status_code}")

    # 未知任务
    response = client.get(f"{BASE}/api/runs/nonexistent123")
    check("未知任务返回 404", response.status_code == 404, f"got {response.status_code}")

    # 越权浏览
    response = client.get(f"{BASE}/api/browse", params={"path": "C:/Windows"})
    check("越权目录被拦截", response.status_code == 403, f"got {response.status_code}")


def test_check_env_event_stream(client: httpx.Client) -> None:
    """启动真实自检子进程，并通过 WebSocket 收完整事件流。"""
    response = client.post(f"{BASE}/api/train/yolo/check-env")
    if response.status_code != 200:
        check("自检任务启动", False, response.text[:200])
        return
    run = response.json()
    run_id = run["run_id"]
    check("自检任务启动", bool(run_id), run_id)

    events: list[dict] = []
    finished = False

    with httpx.Client(timeout=120) as _:
        import websockets
        import asyncio

        async def consume() -> None:
            nonlocal finished
            async with websockets.connect(f"{WS_BASE}/ws/runs/{run_id}", max_size=None) as ws:
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        break
                    except Exception:
                        break
                    event = json.loads(raw)
                    events.append(event)
                    if event.get("type") == "finished":
                        finished = True
                        break

        asyncio.run(consume())

    check("收到 finished 事件", finished, f"共 {len(events)} 个事件")
    check("收到结构化事件", any(e.get("type") in {"env", "model"} or "type" in e for e in events), f"types={sorted({e.get('type') for e in events})}")

    env_events = [e for e in events if e.get("type") == "env"]
    if env_events:
        missing = env_events[0].get("missing", [])
        check("自检结果无缺失依赖", not missing, f"missing={missing}")


def test_run_listing(client: httpx.Client) -> None:
    data = client.get(f"{BASE}/api/runs").json()
    check("任务列表可查询", isinstance(data.get("runs"), list), str(data)[:120])


def main() -> int:
    try:
        with httpx.Client(timeout=30) as client:
            health = client.get(f"{BASE}/api/health")
            check("服务存活", health.status_code == 200)
            if health.status_code != 200:
                return 1

            test_capabilities(client)
            test_validation(client)
            test_check_env_event_stream(client)
            test_run_listing(client)
    except httpx.ConnectError:
        print(f"无法连接 {BASE}，请先启动服务。")
        return 1

    print()
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    for item in failed:
        print(f"  - {item}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
