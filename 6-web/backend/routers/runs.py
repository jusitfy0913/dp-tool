# -*- coding: utf-8 -*-
"""任务台账域路由：运行中任务、历史记录、单任务日志与 WebSocket 事件流。"""

from __future__ import annotations

import contextlib
import re

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import logstore
import runledger
import web_state

router = APIRouter()
# ---------------------------------------------------------------------------
# 任务查询 / 停止 / 事件流
# ---------------------------------------------------------------------------
@router.get("/api/runs")
def list_runs(limit: int = 0) -> dict:
    """列出本次服务生命周期内的任务。

    ``limit`` 只取最近的 N 条：顶栏「运行中」条每 3 秒拉一次，长期运行的服务里
    ``_runs`` 会越攒越多（每条还带完整命令行），不截断就是白白浪费带宽。
    """
    runs = web_state.manager.list_runs()
    if limit > 0:
        runs = runs[:limit]
    return {"runs": runs}


@router.get("/api/runs/history")
def list_run_history(limit: int = 50) -> dict:
    """历史任务（跨服务重启）。

    与 ``/api/runs`` 的区别：那条只返回**本进程**创建的任务，服务一重启就空；
    这条读 ``runtime/runs.jsonl`` 台账，能回溯「上周那次训练到底怎么结束的」。

    ``limit`` 上限 500：台账单条约 600 字节，一次吐几千条对前端也没意义。
    """
    capped = max(1, min(int(limit), 500))
    try:
        items = runledger.history(capped)
    except Exception as exc:  # noqa: BLE001 - 台账坏了不该 500
        raise HTTPException(status_code=500, detail=f"读取任务台账失败：{exc}") from exc
    return {"runs": items, "limit": capped}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """查询单个任务。"""
    run = web_state.manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"未知任务：{run_id}")
    return run.snapshot()


@router.get("/api/runs/{run_id}/log")
def get_run_log(run_id: str, lines: int = 0) -> Response:
    """下载/查看某个任务的完整日志（纯文本）。

    ``lines=0``（默认）返回全文——这是「点开日志文件」背后的接口，用户要的是能
    自己 Ctrl+S 存档的那一份；``lines=N`` 只取最后 N 行，给面板做「补一段」用。

    为什么不能只靠 WebSocket：那条流是给**活着**的 run 用的，事件只存在内存里，
    任务一结束或者服务一重启就没了，而 ``runtime/logs/{run_id}.log`` 一直躺在磁盘上。
    训练失败要复盘的恰恰是这条已经结束的日志。
    """
    # run_id 在 URL 里，直接拼路径就能跳出 runtime/logs。这里只放行 12 位十六进制
    # （uuid4().hex[:12] 的产物），顺带挡住 ../ 之类的东西。
    if not re.fullmatch(r"[0-9a-f]{12}", run_id or ""):
        raise HTTPException(status_code=400, detail=f"非法的任务号：{run_id}")

    if not logstore.any_shard(run_id):
        raise HTTPException(status_code=404, detail=f"任务 {run_id} 的日志文件不存在")

    if lines > 0:
        # 跨滚动分片取尾部：日志一旦超过 LOG_MAX_BYTES 就会被切成 .log / .log.1…
        # 只读当前文件会丢掉前面的报错，而用户要的恰恰是最早那几行。
        text = logstore.read_tail(run_id, max(1, min(int(lines), 5000)))
    else:
        try:
            text = logstore.read_all(run_id)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"读取日志失败：{exc}") from exc
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={
            # inline 让浏览器直接显示文本（方便全选复制），文件名带上 run_id 方便另存
            "Content-Disposition": f'inline; filename="{run_id}.log"',
            "Cache-Control": "no-store",
        },
    )


@router.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """停止任务。"""
    run = web_state.manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"未知任务：{run_id}")
    stopped = web_state.manager.stop(run_id)
    return {"stopped": stopped, "run": run.snapshot()}


@router.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    """把任务事件实时推给浏览器。

    连接建立后先补发历史事件，保证刷新页面不丢日志。
    """
    await websocket.accept()

    subscribed = web_state.manager.subscribe(run_id)
    if subscribed is None:
        await websocket.send_json({"type": "error", "message": f"未知任务：{run_id}"})
        await websocket.close()
        return

    queue, history = subscribed
    try:
        for event in history:
            await websocket.send_json(event)

        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event.get("type") == "finished":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - 连接异常不应拖垮服务
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        web_state.manager.unsubscribe(run_id, queue)
        with contextlib.suppress(Exception):
            await websocket.close()
