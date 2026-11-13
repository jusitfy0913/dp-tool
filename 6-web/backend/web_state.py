# -*- coding: utf-8 -*-
"""共享状态：任务管理器、并发闸门、路径白名单、预热池配置。

路由模块都从这里拿 ``manager`` / ``_reject_if_busy`` / ``_ensure_allowed_path``，
这样它们之间不需要互相 import，也就不会形成环。"""

from __future__ import annotations

from __future__ import annotations
import os
import threading
from pathlib import Path
from fastapi import HTTPException
import config
import gpulock
import pipeline as pipeline_mod
import runners
import warmpool
from runners import manager
#: 全流程串联的注册表。每段仍是一个普通 run，所以这里的职责只有「决定下一段跑什么」。
pipeline_manager = pipeline_mod.PipelineManager(
    manager, config.RUNTIME_DIR / "pipelines"
)


# ---------------------------------------------------------------------------
# 并发闸门
# ---------------------------------------------------------------------------
#: 「检查 + 创建」必须在同一把锁里完成，否则双击按钮的两个请求会同时通过检查。
#: 进程创建本身很快（subprocess.Popen 返回就打第一条事件），持锁时间可以忽略。
_START_LOCK = threading.Lock()


#: kind -> 给用户看的名字。错误信息里写中文，别让用户自己猜 "infer-batch" 是什么。
RUN_KIND_LABELS = {
    "yolo-train": "YOLO 训练",
    "anomalib-train": "Anomalib 训练",
    "engine-build": "引擎构建",
    "infer-batch": "目录批量推理",
    "infer-video": "视频推理",
    "autoexp-agent": "自动实验循环",
}


def _describe_run(run: dict) -> str:
    """把一条 run 摘要写成「名字（run xxx，已运行 N 分钟）」。"""
    kind = str(run.get("kind") or "")
    if kind.startswith(runners.EXCLUSIVE_RUN_PREFIX):
        label = "全流程串联"
    else:
        label = RUN_KIND_LABELS.get(kind, kind)
    elapsed = float(run.get("elapsed") or 0)
    if elapsed >= 60:
        spent = f"已运行 {int(elapsed // 60)} 分钟"
    else:
        spent = f"已运行 {int(elapsed)} 秒"
    return f"{label}（run {run.get('run_id')}，{spent}）"


def _reject_if_busy() -> None:
    """已有独占任务时拒绝启动新任务，返回 **409**。

    为什么必须由后端拦而不是前端：前端刷新一次就「失忆」，页面会退回
    「未开始」状态、按钮变成可点（v1.25.2 之前的行为）。用户点一下就会
    起第二个训练进程：两个进程抢同一块显卡，通常双双 OOM，还会写进
    同一个 ``runs/<name>`` 目录把产物盖掉。

    409 而不是 400：前端要据此区分「参数写错」和「机器正忙着」，
    前者该拦住用户，后者该引导他去停止或接续正在跑的那个任务。
    """
    busy = manager.active_exclusive()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(
                f"已有任务正在运行：{_describe_run(busy[0])}。"
                "同一块 GPU 上并发会抢显存，还可能覆盖同一个产物目录。"
                "请先在顶部「运行中」条里停止它，或刷新页面接续该任务。"
            ),
        )

    # 串联刚创建时它的第一段还没起来（先在后台线程里注册，再启动子进程），
    # 所以只看 run 会漏掉「刚点了两次开始全流程」这种双击。这里再查一次链本身。
    for record in pipeline_manager.list(limit=20):
        if record.get("live", True) and record.get("status") in (pipeline_mod.PENDING, pipeline_mod.CHAIN_RUNNING):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"已有全流程串联在运行：{record.get('title') or record.get('id')}。"
                    "请先在「全流程串联」卡片里停止它，再启动新的任务。"
                ),
            )

    # 跨进程 GPU 锁：上面查的都是「本进程内存里的任务」，看不见两种真实情况——
    #   1) 命令行在跑 autoexp.py run（自动实验循环）；
    #   2) 本机还起着另一个 Web 实例（比如 8060 与 8094 各一套）。
    # 两者都会和本进程的任务抢同一块显卡，通常双双 OOM。锁由 gpulock 维护，
    # 自己持有的锁要跳过（pid == 本进程），否则会把自己拦住。
    holder = gpulock.read()
    if holder is not None and int(holder.get("pid") or 0) != os.getpid():
        raise HTTPException(
            status_code=409,
            detail=(
                f"显卡正在被其他进程占用：{gpulock.describe(holder)}。"
                "命令行任务请等它结束或在对应终端里停止；"
                "另一个 Web 实例的任务请到那个页面停止。"
                "现在就开新的训练会和它抢显存，通常两个都会 OOM。"
            ),
        )


# ---------------------------------------------------------------------------
# 常驻预热进程
# ---------------------------------------------------------------------------
def configure_warm_pool() -> None:
    """把「解释器 + launcher + 工作目录」注册给预热池，并立刻开始预热。

    找不到训练解释器、或关闭了 TRTFORGE_WARM 时什么都不做：
    预热只是提速，不是训练的前置条件。
    """
    if not config.WARM_ENABLED:
        warmpool.pool.configure({}, enabled=False)
        return
    python = config.resolve_python()
    if not python:
        warmpool.pool.configure({}, enabled=False)
        return

    specs: dict[str, dict[str, str]] = {}
    if "yolo" in config.WARM_KINDS and config.YOLO_LAUNCHER.is_file():
        specs["yolo"] = {
            "python": python,
            "launcher": str(config.YOLO_LAUNCHER),
            "cwd": str(config.YOLO_LAUNCHER_DIR),
        }
    if "anomalib" in config.WARM_KINDS and config.ANOMALIB_LAUNCHER.is_file():
        specs["anomalib"] = {
            "python": python,
            "launcher": str(config.ANOMALIB_LAUNCHER),
            "cwd": str(config.ANOMALIB_LAUNCHER_DIR),
        }
    warmpool.pool.configure(specs, enabled=True, idle_seconds=config.WARM_IDLE_SECONDS)
    for kind in specs:
        warmpool.pool.ensure_async(kind)


def _ensure_allowed_path(path: str) -> Path:
    """把路径解析成绝对路径并做白名单校验，返回解析结果。

    目录浏览、数据集、标注三条链路共用同一份白名单（`config.is_path_allowed`）。
    集中在这里是为了错误信息只维护一份——标注页最常见的困惑是
    「工程能打开，但图片全是裂的」，根因就是图片所在盘不在白名单里。
    """
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"路径无法解析：{path}（{exc}）") from exc
    if not config.is_path_allowed(resolved):
        raise HTTPException(
            status_code=403,
            detail=(
                f"路径不在允许范围内：{resolved}。"
                "如需访问仓库外的目录，请设置 TRTFORGE_BROWSE_ROOTS 并重启服务。"
            ),
        )
    return resolved
