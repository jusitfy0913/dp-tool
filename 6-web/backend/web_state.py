# -*- coding: utf-8 -*-
"""共享状态：任务管理器、并发闸门、路径白名单、预热池配置。

路由模块都从这里拿 ``manager`` / ``_reject_if_busy`` / ``_ensure_allowed_path``，
这样它们之间不需要互相 import，也就不会形成环。"""

from __future__ import annotations

import os
import re
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
    # 自己持有的锁要排除（pid == 本进程），否则会把自己拦住。
    # 用 foreign_holders() 而不是 read()：锁支持多持有者后 read() 只返回一个，
    # 可能正好是本进程自己那条，于是漏掉真正冲突的那个进程。
    foreign = gpulock.foreign_holders()
    if foreign:
        raise HTTPException(
            status_code=409,
            detail=(
                f"显卡正在被其他进程占用：{gpulock.describe_all(foreign)}。"
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
    # AnomalyDINO 默认不预热：它会再多驻留一个 torch 进程（约 500 MB），
    # 而它的冷启动只有 2.4 秒（不像 anomalib 要 5.7 秒）。
    # 内存宽裕的机器可以显式开：TRTFORGE_WARM_KINDS=yolo,anomalib,anomalydino
    if "anomalydino" in config.WARM_KINDS and config.ANOMALYDINO_LAUNCHER.is_file():
        specs["anomalydino"] = {
            "python": python,
            "launcher": str(config.ANOMALYDINO_LAUNCHER),
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


#: 合法的输出目录名。除字符表外还额外禁止 ``..``：
#: ``name`` 会直接拼进 ``project/name``，一个 ``../`` 就能把产物写到白名单外。
_SAFE_NAME = re.compile(r'^[^\\/:*?"<>|\x00-\x1f]+$')


def _looks_like_path(value: str) -> bool:
    """判断一个字符串是「路径」还是「模型名」。

    ``model`` 字段默认是 ``yolov8n.pt`` 这种由 Ultralytics 自己下载的**名字**，
    对它做文件存在性检查会把正常用法判成错误。只有写成路径形式（绝对路径或带
    分隔符）时才该按路径校验。
    """
    text = str(value or "").strip()
    if not text:
        return False
    return Path(text).is_absolute() or "/" in text or "\\" in text


def _ensure_safe_name(name: str, field: str = "名称") -> str:
    """校验「会拼进输出路径的那一段名字」，返回去掉首尾空白后的值。

    为什么单独一个函数：``project`` / ``name`` 是**写入**目录，不是读取，
    白名单只能保证父目录在仓库里；真正把文件写到仓库外的是名字里的 ``..``
    或路径分隔符。两道检查缺一不可。
    """
    value = str(name or "").strip()
    if not value or value in {".", ".."} or ".." in value:
        raise HTTPException(status_code=400, detail=f"{field}不合法：{name!r}")
    if not _SAFE_NAME.match(value):
        raise HTTPException(
            status_code=400,
            detail=f'{field}不能包含路径分隔符或 \\ / : * ? " < > | 等字符：{value}',
        )
    return value


def _ensure_allowed_dir(path: str, base: Path | None = None, field: str = "目录") -> Path:
    """校验「目录参数」：相对路径按 ``base`` 解析（启动器的 cwd），再走白名单。

    ``project`` 这类字段允许写相对路径（``runs/trtforge``），它由子进程的 cwd
    解析。后端要判断白名单就必须用同一个基准，否则会把合法路径判成越界。
    """
    raw = str(path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field}不能为空")
    target = Path(raw).expanduser()
    if not target.is_absolute() and base is not None:
        target = base / target
    return _ensure_allowed_path(str(target))


def _resolve_python(override: str | None) -> str:
    """确定训练用解释器，找不到时给出可执行建议。

    显式给出时必须同时满足两条：**在路径白名单内**、**文件名是 python 解释器**。
    只判 ``is_file`` 会让这个字段变成「执行任意程序」——传 ``\\evil.exe`` 就能
    以服务账号跑起来。``TRTFORGE_PYTHON`` 与 config 里写的是管理员配置，
    自动发现时不受这两条约束（见 ``config.resolve_python``）。
    """
    if override:
        candidate = _ensure_allowed_path(override)
        if candidate.name.lower() not in {"python.exe", "pythonw.exe", "python", "python3"}:
            raise HTTPException(
                status_code=400,
                detail=f"解释器必须是 python.exe / pythonw.exe，实际为 {candidate.name}",
            )
        if not candidate.is_file():
            raise HTTPException(status_code=400, detail=f"指定的 Python 不存在：{override}")
        return str(candidate)

    resolved = config.resolve_python()
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "未找到可用的训练解释器。请在页面上填写 python.exe 路径，"
                "或设置环境变量 TRTFORGE_PYTHON。"
            ),
        )
    return resolved
