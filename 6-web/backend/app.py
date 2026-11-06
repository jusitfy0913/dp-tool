# -*- coding: utf-8 -*-
"""TRTForge Web Studio 后端入口。

职责边界（有意保持窄）：
- 训练：拼参数 -> 启动 launcher 子进程 -> 转发事件，不碰算法。
- 推理：调封装好的 DLL，不重写前处理与解码。
- 静态托管：frontend 目录直接由本服务提供，无前端构建步骤。

启动：
    cd 6-web/backend
    python -m uvicorn app:app --host 127.0.0.1 --port 8060

或用仓库脚本：scripts/run_web_studio.ps1
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import annotation
import annotation_drafts
import annotationimport
import augpreview
import autolabel
import autolabeljobs
import checkpoints
import config
import dataset
import infer
import inferjobs
import inferproxy
import labels as labels_mod
import models as models_mod
import modelzoo
import nativedialog
import ocr_infer
import ocr_models
import onnx_fallback
import pipeline as pipeline_mod
import progress
import promptlabel
import runners
import textlabel
import traininfo
import warmpool
from runners import manager

#: 应用版本号。前端顶栏与 /api/health 都读它，方便一眼看出「页面连的是哪个后端」。
#: 起因：改了代码但 8060 上跑的是几小时前的旧进程，排查很久才发现。
APP_VERSION = "1.28.0"

#: 本文件所在目录（6-web/backend）。子进程脚本路径都以它为基准，
#: 不依赖启动时的工作目录——从仓库根启动和从 backend 启动必须都能跑。
BACKEND_DIR = Path(__file__).resolve().parent

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


# ---------------------------------------------------------------------------
# 应用与生命周期
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时准备目录与事件循环，退出时清理子进程与句柄。"""
    config.ensure_dirs()
    manager.bind_loop(asyncio.get_running_loop())
    # 拉起推理工作进程并探测推理库，让 /api/capabilities 立刻反映真实状态
    inferproxy.startup()
    # 预热训练进程：失败只记一笔，不影响服务启动（用户仍可走冷启动）
    try:
        configure_warm_pool()
    except Exception as exc:  # noqa: BLE001
        print(f"预热进程初始化失败（不影响训练）：{type(exc).__name__}: {exc}", flush=True)
    try:
        yield
    finally:
        # 先停串联（它可能正握着某个 run），再停所有子进程
        pipeline_manager.shutdown_all()
        manager.shutdown_all()
        warmpool.pool.shutdown()
        inferproxy.shutdown()


app = FastAPI(title="TRTForge Web Studio", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(inferproxy.WorkerError)
async def _worker_error(_request, exc: inferproxy.WorkerError) -> JSONResponse:
    """推理进程崩溃 503 / 超时 504，所有推理路由共用。"""
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# 预热状态（前端切到训练页时查询 / 手动触发）
# ---------------------------------------------------------------------------
@app.get("/api/warm/status")
def warm_status() -> dict:
    """预热状态：ready 表示下一次点击训练能立刻开训。"""
    return warmpool.pool.status()


@app.post("/api/warm/prewarm")
def warm_prewarm(kind: str | None = None) -> dict:
    """手动触发预热（前端切到训练页时也会调）。"""
    status = warmpool.pool.status()
    if not status["enabled"]:
        return {"ok": False, "reason": "预热未启用：TRTFORGE_WARM=0 或未找到训练解释器", **status}
    kinds = [kind] if kind else warmpool.pool.kinds()
    for item in kinds:
        warmpool.pool.ensure_async(item)
    return {"ok": True, **warmpool.pool.status()}


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """存活探针，同时回报版本号。

    ``version`` 是判断「页面连的后端是不是最新的」的唯一可靠依据：
    端口被旧进程占着时，页面一切正常，只是版本号停在旧值。
    """
    return {"ok": True, "version": APP_VERSION}


@app.get("/api/capabilities")
def capabilities() -> dict:
    """返回当前环境具备哪些能力，前端据此决定按钮可用性。"""
    data = config.capabilities()
    backend = inferproxy.status()
    data["infer_yolo"] = backend.yolo_ok
    data["infer_anomalib"] = backend.anomalib_ok
    data["infer_onnx"] = backend.onnx_ok
    data["infer_onnx_provider"] = backend.onnx_provider
    data["infer_onnx_detail"] = backend.onnx_detail
    data["infer_detail"] = backend.detail
    # 降级能力：DLL 不可用时，前端可提示「仍可用 ONNX 跑通链路」
    fallback_ok, fallback_detail = onnx_fallback.available()
    data["infer_fallback"] = fallback_ok
    data["infer_fallback_detail"] = fallback_detail
    ocr_state = inferproxy.ocr_status()
    _paddle_major, paddle_text = ocr_models.paddle_version()
    data["infer_ocr"] = ocr_state["ok"]
    data["infer_ocr_detail"] = ocr_state["detail"]
    data["paddle_version"] = paddle_text
    data["infer_worker"] = inferproxy.worker_state()
    # 原生对话框：不可用时前端自动改走页面内目录浏览，不显示一个点不动的按钮
    data["native_dialog"] = nativedialog.available()
    return data


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
class YoloTrainRequest(BaseModel):
    """YOLO 训练参数。字段名与 launcher 的 --参数 一一对应。"""

    task: str = Field(default="detect", description="detect/segment/pose/obb")
    model: str = Field(default="yolov8n.pt", description="预训练权重或模型名")
    data: str = Field(..., description="dataset.yaml 路径")
    epochs: int = 100
    imgsz: int = 640
    batch: str = "16"
    device: str = "0"
    project: str = "runs/trtforge"
    name: str = "exp"
    patience: int = 100
    workers: int = 8
    pretrained: str = "true"
    resume: bool = False
    # 断点续训：显式给出 last.pt 的绝对路径。非空时后端只拼 --resume-from，
    # 不再依赖 Ultralytics 的 get_latest_run() 去猜「最近一次 run」。
    resume_from: str = Field(default="", description="从指定 last.pt 续训")
    # 训练起点语义：preset=官方预训练/新实验；finetune=用自己的 best.pt 微调。
    # 与 resume 的区别：resume 恢复优化器与轮次（参数以检查点为准），
    # finetune 只载入权重，优化器与轮次重置（参数按表单）。
    init_mode: str = Field(default="preset", description="preset / finetune")
    lr0: float | None = Field(default=None, ge=0, description="初始学习率；留空用 Ultralytics 默认 0.01")

    exist_ok: bool = Field(default=False, description="写回已有实验目录时覆盖，而不是新建 exp2")
    python: str | None = Field(default=None, description="覆盖训练解释器")
    # 数据增强：默认值与 Ultralytics 8.4 默认逐项一致，不填即行为不变。
    # 用 None 表示「用户没填」，这样拼命令行时可以整体跳过，避免把默认值
    # 重复写进命令行、也方便以后改 Ultralytics 默认值时不被这里钉死。
    hsv_h: float | None = Field(default=None, description="色调抖动，默认 0.015")
    hsv_s: float | None = Field(default=None, description="饱和度抖动，默认 0.7")
    hsv_v: float | None = Field(default=None, description="亮度抖动，默认 0.4")
    degrees: float | None = Field(default=None, description="随机旋转 ±度，默认 0")
    translate: float | None = Field(default=None, description="平移比例，默认 0.1")
    scale: float | None = Field(default=None, description="缩放增益，默认 0.5")
    shear: float | None = Field(default=None, description="错切 ±度，默认 0")
    perspective: float | None = Field(default=None, description="透视变换，默认 0")
    flipud: float | None = Field(default=None, description="上下翻转概率，默认 0")
    fliplr: float | None = Field(default=None, description="左右翻转概率，默认 0.5")
    bgr: float | None = Field(default=None, description="RGB/BGR 互换概率，默认 0")
    mosaic: float | None = Field(default=None, description="马赛克拼接概率，默认 1.0")
    mixup: float | None = Field(default=None, description="MixUp 概率，默认 0")
    copy_paste: float | None = Field(default=None, description="Copy-Paste 概率，默认 0")
    close_mosaic: int | None = Field(default=None, description="最后 N 个 epoch 关 mosaic，默认 10")


class AnomalibTrainRequest(BaseModel):
    """Anomalib 训练参数。"""

    model: str = Field(default="patchcore", description="patchcore/padim/efficientad")
    root: str = Field(..., description="Folder 数据集根目录")
    normal_dir: str = "train/good"
    abnormal_dir: str = "test/abnormal"
    normal_test_dir: str = "test/good"
    project: str = "results"
    name: str = "latest"
    input_size: int = 256
    max_epochs: str = "auto"
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0
    val_split_ratio: float = 0.2
    calibrate_metadata: bool = True
    calib_limit: int = 64
    # EfficientAd 需要 ImageNet 目录做特征蒸馏；留空则由 launcher 用默认路径
    imagenet_dir: str = ""
    # 用独立数据校准 metadata 阈值；留空表示不校准，沿用训练集统计
    calib_normal_dir: str = ""
    calib_abnormal_dir: str = ""
    python: str | None = Field(default=None, description="覆盖训练解释器")


def _resolve_python(override: str | None) -> str:
    """确定训练用解释器，找不到时给出可执行建议。"""
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        raise HTTPException(status_code=400, detail=f"指定的 Python 不存在：{override}")

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


def build_train_command(request: "YoloTrainRequest", python: str) -> list[str]:
    """拼 YOLO 训练命令行。

    抽成纯函数的原因和 `build_export_command` 一样：全流程串联要复用同一份拼装逻辑。
    两边各写一份的后果是"页面上训练用的增强参数，串联时被悄悄丢掉"——
    跑出来的模型和用户以为的不是同一个，而日志里一切正常。
    """
    command = [
        python, str(config.YOLO_LAUNCHER),
        "--mode", "train",
        "--task", request.task,
        "--model", request.model,
        "--data", str(Path(request.data)),
        "--epochs", str(request.epochs),
        "--imgsz", str(request.imgsz),
        "--batch", request.batch,
        "--device", request.device,
        "--project", request.project,
        "--name", request.name,
        "--patience", str(request.patience),
        "--workers", str(request.workers),
        "--pretrained", request.pretrained,
    ]
    if request.resume:
        command.append("--resume")
    if request.resume_from.strip():
        checkpoint = Path(request.resume_from).expanduser()
        if not checkpoint.is_file():
            raise ValueError(f"续训检查点不存在：{checkpoint}")
        command += ["--resume-from", str(checkpoint.resolve())]

    # 起点语义：微调只载入权重、重置优化器；续训必须走 --resume-from。
    # 两者不能同时生效，否则 Ultralytics 会以 resume 为准，表单上的 epochs / lr0
    # 全部失效，而用户以为自己在微调。
    init_mode = request.init_mode if request.init_mode in {"preset", "finetune"} else "preset"
    if init_mode == "finetune":
        if request.resume or request.resume_from.strip():
            raise ValueError("微调与断点续训只能选一个：微调请先取消续训")
        weights = Path(request.model).expanduser()
        if not weights.is_file():
            raise ValueError(f"微调起点权重不存在：{weights}")
        # 微调默认写回原实验目录，必须 exist_ok=True；否则 Ultralytics 会静默生成
        # exp2/exp3，用户以为覆盖了原实验，实际多出几个目录。
        command += ["--init-mode", "finetune", "--exist-ok"]
    else:
        command += ["--init-mode", "preset"]
        if request.exist_ok:
            command.append("--exist-ok")

    # 学习率：留空 = 不传，用 Ultralytics 默认（0.01）；微调时前端会填 0.001 这种小步长。
    if request.lr0 is not None:
        command += ["--lr0", str(request.lr0)]

    # 微调时即使表单没填，也给一个保守的小步长：沿用新训的 0.01 会明显冲掉
    # 原权重已有的能力，用户看到的是一次「练坏了」的微调。
    elif init_mode == "finetune":
        command += ["--lr0", "0.001"]

    # 增强参数只在用户填了才拼进命令行：留空 = 用 launcher 默认 = 用 Ultralytics 默认
    augment_fields = (
        ("hsv_h", "--hsv-h"),
        ("hsv_s", "--hsv-s"),
        ("hsv_v", "--hsv-v"),
        ("degrees", "--degrees"),
        ("translate", "--translate"),
        ("scale", "--scale"),
        ("shear", "--shear"),
        ("perspective", "--perspective"),
        ("flipud", "--flipud"),
        ("fliplr", "--fliplr"),
        ("bgr", "--bgr"),
        ("mosaic", "--mosaic"),
        ("mixup", "--mixup"),
        ("copy_paste", "--copy-paste"),
        ("close_mosaic", "--close-mosaic"),
    )
    for field_name, flag in augment_fields:
        value = getattr(request, field_name)
        if value is not None:
            command += [flag, str(value)]
    return command


@app.post("/api/train/yolo")
def train_yolo(request: YoloTrainRequest) -> dict:
    """启动 YOLO 训练。"""
    if not config.YOLO_LAUNCHER.is_file():
        raise HTTPException(status_code=500, detail=f"找不到 launcher：{config.YOLO_LAUNCHER}")

    data_yaml = Path(request.data)
    if not data_yaml.is_file():
        raise HTTPException(status_code=400, detail=f"dataset.yaml 不存在：{request.data}")

    python = _resolve_python(request.python)

    # 概率类增强值必须在 [0,1]，超范围 Ultralytics 会在训练启动后才报错，
    # 这里提前拦住，省得等几十秒才看到 ValueError。
    for field_name in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale",
                       "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste"):
        value = getattr(request, field_name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} 必须在 0~1 之间，当前为 {value}",
            )
    if request.close_mosaic is not None and request.close_mosaic < 0:
        raise HTTPException(status_code=400, detail="close_mosaic 不能为负数")

    # 续训检查点必须存在且在允许的根目录内。命令行拼装里也会再查一次存在性，
    # 但白名单校验属于接口职责，不能挪进纯函数。
    if request.resume_from.strip():
        checkpoint = Path(request.resume_from).expanduser()
        if not checkpoint.is_file():
            raise HTTPException(status_code=400, detail=f"续训检查点不存在：{checkpoint}")
        _ensure_allowed_path(str(checkpoint))

    backup_dir: Path | None = None
    if request.init_mode == "finetune":
        weights = Path(request.model).expanduser()
        if not weights.is_file():
            raise HTTPException(status_code=400, detail=f"微调起点权重不存在：{weights}")
        _ensure_allowed_path(str(weights))
        info = checkpoints.describe_weights(weights)
        if info:
            # 写回原实验目录：project/name 由权重路径推导，不信任前端传来的值。
            # 万一前端状态不同步（比如刚刷新过页面），也不会把结果写到别处。
            original = Path(info["run_dir"])
            request.project = str(original.parent)
            request.name = original.name
            backup_dir = _backup_run_dir(original)

    try:
        command = build_train_command(request, python)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 检查与创建同锁：双击按钮的两个请求不能都通过检查
    with _START_LOCK:
        _reject_if_busy()
        try:
            run = manager.create(
                "yolo-train",
                command,
                str(config.YOLO_LAUNCHER_DIR),
                meta={"backup_dir": str(backup_dir)} if backup_dir else None,
                warm_kind="yolo",
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动训练进程失败：{exc}") from exc
    return run.snapshot()


def _backup_run_dir(run_dir: Path) -> Path | None:
    """微调写回同一实验目录前，把旧结果复制到 ``_prev/<时间戳>/``。

    为什么必须备份：用户选了「写回同一实验目录」，新训练会覆盖 results.csv 与
    weights/*.pt。不备份的话，一次误操作就换掉了之前几个小时的训练结果。
    备份失败只返回 None（前端给警告），不阻塞训练——宁可少一份备份，
    也不能因为这个让训练起不来。
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = run_dir / "_prev" / stamp
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("args.yaml", "results.csv"):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
        weights_dir = run_dir / "weights"
        if weights_dir.is_dir():
            dst = target / "weights"
            dst.mkdir(parents=True, exist_ok=True)
            for item in weights_dir.glob("*.pt"):
                shutil.copy2(item, dst / item.name)
    except OSError:
        return None
    return target


def build_anomalib_train_command(request: "AnomalibTrainRequest", python: str) -> list[str]:
    """拼 Anomalib 训练 + 导出命令行。

    抽成纯函数的原因与 `build_train_command` 一样：全流程串联要复用同一份拼装逻辑。
    两处各写一份的后果是「页面上打开的两个校准目录，串联时被悄悄丢掉」——
    阈值校准结果与手工训练的不一样，而日志里一切正常。
    """
    command = [
        python, str(config.ANOMALIB_LAUNCHER),
        "--mode", "train-export",
        "--model", request.model,
        "--root", str(Path(request.root).expanduser()),
        "--normal-dir", request.normal_dir,
        "--abnormal-dir", request.abnormal_dir,
        "--normal-test-dir", request.normal_test_dir,
        "--project", request.project,
        "--name", request.name,
        "--input-size", str(request.input_size),
        "--max-epochs", str(request.max_epochs),
        "--train-batch-size", str(request.train_batch_size),
        "--eval-batch-size", str(request.eval_batch_size),
        "--num-workers", str(request.num_workers),
        "--val-split-ratio", str(request.val_split_ratio),
        "--calibrate-metadata", "true" if request.calibrate_metadata else "false",
        "--calib-limit", str(request.calib_limit),
    ]
    # 这三个只在非空时传，避免用空串覆盖 launcher 的默认值
    if request.imagenet_dir.strip():
        command += ["--imagenet-dir", request.imagenet_dir.strip()]
    if request.calib_normal_dir.strip():
        command += ["--calib-normal-dir", request.calib_normal_dir.strip()]
    if request.calib_abnormal_dir.strip():
        command += ["--calib-abnormal-dir", request.calib_abnormal_dir.strip()]
    return command


@app.post("/api/train/anomalib")
def train_anomalib(request: AnomalibTrainRequest) -> dict:
    """启动 Anomalib 训练并导出。"""
    if not config.ANOMALIB_LAUNCHER.is_file():
        raise HTTPException(
            status_code=500, detail=f"找不到 launcher：{config.ANOMALIB_LAUNCHER}"
        )

    root = Path(request.root)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集根目录不存在：{request.root}")

    python = _resolve_python(request.python)
    command = build_anomalib_train_command(request, python)

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = manager.create(
                "anomalib-train",
                command,
                str(config.ANOMALIB_LAUNCHER_DIR),
                warm_kind="anomalib",
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动训练进程失败：{exc}") from exc
    return run.snapshot()


# ---------------------------------------------------------------------------
# 导出 ONNX（训练 -> 导出 -> 编译 的中间一环）
# ---------------------------------------------------------------------------
class YoloExportRequest(BaseModel):
    """导出参数。字段名与 launcher 的 `--mode export` 一一对应。"""

    weights: str = Field(..., description="训练得到的 best.pt / last.pt 绝对路径")
    task: str = Field(default="detect", description="detect/segment/pose/obb；与权重不符时以权重为准")
    imgsz: int = Field(default=640, description="导出输入尺寸，必须是 32 的倍数")
    batch: int = Field(default=1, ge=1, le=32, description="导出 batch，TensorRT 侧通常为 1")
    opset: int | None = Field(default=None, ge=9, le=20, description="ONNX opset，留空用 Ultralytics 默认")
    dynamic: bool = Field(default=False, description="动态 batch/尺寸")
    half: bool = Field(default=False, description="半精度导出（需要 GPU）")
    simplify: bool = Field(default=True, description="onnx-simplifier 化简")
    make_transd: bool = Field(default=True, description="同时生成 .transd.onnx（TensorRT 必须）")
    python: str | None = Field(default=None, description="覆盖训练解释器")


def build_export_command(request: "YoloExportRequest", python: str) -> list[str]:
    """拼导出命令行。

    抽成纯函数是为了能单测：参数拼错一位，用户要等 30 秒才在日志里看到 argparse
    的 usage，而 usage 里根本看不出「少了 --export-batch」。
    """
    # Ultralytics 会把尺寸取整到 stride 的倍数，静默改数值会让用户以为填错了
    if request.imgsz % 32 != 0 or not 32 <= request.imgsz <= 2048:
        raise ValueError(
            f"imgsz 必须是 32 的倍数且在 32~2048 之间，当前为 {request.imgsz}"
        )
    if request.task not in {"detect", "segment", "pose", "obb"}:
        raise ValueError(f"不支持的任务类型：{request.task}")

    command = [
        python, str(config.YOLO_LAUNCHER),
        "--mode", "export",
        "--task", request.task,
        "--weights", request.weights,
        "--format", "onnx",
        "--imgsz", str(request.imgsz),
        "--export-batch", str(request.batch),
        # 注意：--simplify 是 BooleanOptionalAction，只接受 --simplify / --no-simplify，
        # 写成 `--simplify true` 会被 argparse 判成「多余的参数」直接退出 2。
        "--simplify" if request.simplify else "--no-simplify",
        "--make-transd", "true" if request.make_transd else "false",
    ]
    if request.opset is not None:
        command += ["--opset", str(request.opset)]
    if request.dynamic:
        command.append("--dynamic")
    if request.half:
        command.append("--half")
    return command


@app.post("/api/export/yolo")
def export_yolo(request: YoloExportRequest) -> dict:
    """把训练好的 `.pt` 导出成 ONNX（可选顺带生成 `.transd.onnx`）。

    为什么做成任务而不是同步接口：导出要加载 torch 与权重，实测 5~30 秒，
    同步会卡住整个事件循环。复用训练那套子进程 + WebSocket，日志和产物一条路。
    """
    if not config.YOLO_LAUNCHER.is_file():
        raise HTTPException(status_code=500, detail=f"找不到 launcher：{config.YOLO_LAUNCHER}")

    weights = _ensure_allowed_path(request.weights)
    if not weights.is_file():
        raise HTTPException(status_code=400, detail=f"权重文件不存在：{request.weights}")
    if weights.suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail=f"只能导出 .pt 权重，当前为 {weights.suffix}")

    python = _resolve_python(request.python)
    try:
        command = build_export_command(request, python)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 预期产物先算出来：Ultralytics 的 ONNX 与权重同目录同名不同后缀。
    # dynamic / half 都不改文件名，所以这个推断是稳的；真没落盘时由
    # runners 的 ARTIFACT_MISSING 事件报出来，不用等用户去点下一步才发现。
    expect_onnx = weights.with_suffix(".onnx")

    try:
        run = manager.create(
            "yolo-export",
            command,
            str(config.YOLO_LAUNCHER_DIR),
            expect_file=expect_onnx,
            meta={
                "weights": str(weights),
                "task": request.task,
                "imgsz": request.imgsz,
                "make_transd": request.make_transd,
                "expect_onnx": str(expect_onnx),
                "expect_transd": str(weights.with_suffix(".transd.onnx")),
            },
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"启动导出进程失败：{exc}") from exc
    return run.snapshot()


# ---------------------------------------------------------------------------
# 开训前体检：数据集概览 + 增强预览
# ---------------------------------------------------------------------------
class DatasetOverviewRequest(BaseModel):
    """只读扫描 dataset.yaml，不写任何文件。"""

    source: str = Field(..., description="dataset.yaml 路径")
    refresh: bool = Field(default=False, description="跳过多秒内的缓存")


@app.post("/api/train/dataset-overview")
def train_dataset_overview(request: DatasetOverviewRequest) -> dict:
    """统计数据集规模、类别分布与结构问题。

    独立于 /api/dataset/scan：那个接口要把整个数据集读成对象列表用于划分导出，
    这里只要「已经导出的 YOLO 目录健康报告」，读标签文件名与首列类别号即可。
    """
    _ensure_allowed_path(request.source)
    try:
        return traininfo.scan(request.source, use_cache=not request.refresh)
    except traininfo.OverviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"扫描数据集失败：{exc}") from exc


class AugmentPreviewRequest(BaseModel):
    """增强预览请求。params 由前端把 15 个表单值原样传上来，空串表示用默认值。"""

    data: str = Field(..., description="dataset.yaml 路径，用于定位训练集样本")
    params: dict[str, Any] = Field(default_factory=dict, description="增强参数字典")
    index: int = Field(default=0, ge=0, description="取第几张训练图")
    seed: int | None = Field(default=None, description="随机种子；相同种子结果完全一致")
    imgsz: int = Field(default=640, ge=64, le=4096, description="与训练一致的输入尺寸")
    split: str = Field(default="train", description="从哪个 split 取样本")


@app.post("/api/train/augment-preview")
def train_augment_preview(request: AugmentPreviewRequest) -> dict:
    """用当前参数对一张真实样本做一次增强，返回原图与增强图。

    返回的两张图都是 base64 JPEG，前端直接塞进 <img>，不落临时文件。
    """
    _ensure_allowed_path(request.data)
    try:
        return augpreview.render(
            request.data,
            request.params,
            index=request.index,
            seed=request.seed,
            imgsz=request.imgsz,
            split=request.split,
        )
    except traininfo.OverviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except augpreview.PreviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/checkpoints")
def list_run_checkpoints(root: str | None = None, project: str | None = None, limit: int = 50) -> dict:
    """列出历史训练目录，供断点续训挑选。

    扫描根：``root``（显式指定）> 表单里的 ``project``（按 launcher 的 cwd 解析）
    > launcher 目录下的 ``runs`` 与仓库根下的 ``runs``。
    只读 ``args.yaml`` / ``results.csv``，不加载 ``.pt``——后端解释器没有 torch。
    """
    roots: list[Path] = []
    if root and root.strip():
        _ensure_allowed_path(root)
        roots.append(Path(root).expanduser().resolve())
    if project and project.strip():
        candidate = Path(project).expanduser()
        if not candidate.is_absolute():
            # launcher 的 cwd 就是 YOLO_LAUNCHER_DIR，project 相对路径按它解析
            candidate = config.YOLO_LAUNCHER_DIR / candidate
        if config.is_path_allowed(candidate):
            roots.append(candidate.resolve())
    roots.append(config.YOLO_LAUNCHER_DIR / "runs")
    roots.append(config.REPO_ROOT / "runs")

    try:
        return checkpoints.list_checkpoints(roots, limit=max(1, min(limit, 200)))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"扫描训练目录失败：{exc}") from exc


class FinetuneCheckRequest(BaseModel):
    """微调前的兼容性检查入参。"""

    weights: str = Field(..., description="作为起点的 best.pt 绝对路径")
    data: str = Field(..., description="本次训练用的 dataset.yaml")
    task: str = Field(default="detect", description="detect/segment/pose/obb")
    imgsz: int | None = Field(default=None, description="本次训练的 imgsz，用于尺寸差异提醒")


@app.post("/api/train/yolo/finetune-check")
def yolo_finetune_check(request: FinetuneCheckRequest) -> dict:
    """开训前告诉用户「用自己的 best.pt 继续训练」会发生什么。

    分两级返回：
    - blockers：硬性问题，必须拦住（权重不存在、任务头不一致）；
    - warnings：会被静默改变行为的情况（类别对不上、改尺寸、写回原目录），
      让用户先看到再决定，而不是训练完才发现有一半权重没生效。
    """
    weights = Path(request.weights).expanduser()
    if not weights.is_file():
        raise HTTPException(status_code=400, detail=f"权重文件不存在：{weights}")
    _ensure_allowed_path(str(weights))

    data_yaml = Path(request.data).expanduser()
    if not data_yaml.is_file():
        raise HTTPException(status_code=400, detail=f"dataset.yaml 不存在：{data_yaml}")
    _ensure_allowed_path(str(data_yaml))

    return checkpoints.finetune_check(
        weights=weights,
        data_yaml=data_yaml,
        task=request.task,
        imgsz=request.imgsz,
    )


# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------
@app.post("/api/train/{kind}/check-env")
def check_env(kind: str, python: str | None = None) -> dict:
    """跑一次 launcher 的环境自检，产出与训练同样的事件流。

    这是页面「检测环境」按钮的后端：用户能在开训前就看到缺哪个包，
    而不是训练跑一半报错。
    """
    if kind == "yolo":
        launcher, cwd = config.YOLO_LAUNCHER, config.YOLO_LAUNCHER_DIR
    elif kind == "anomalib":
        launcher, cwd = config.ANOMALIB_LAUNCHER, config.ANOMALIB_LAUNCHER_DIR
    else:
        raise HTTPException(status_code=400, detail=f"未知类型：{kind}，可选 yolo / anomalib")

    if not launcher.is_file():
        raise HTTPException(status_code=500, detail=f"找不到 launcher：{launcher}")

    python_exe = _resolve_python(python)
    command = [python_exe, str(launcher), "--mode", "check-env"]
    try:
        run = manager.create(f"{kind}-check-env", command, str(cwd))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"启动自检进程失败：{exc}") from exc
    return run.snapshot()


# ---------------------------------------------------------------------------
# 引擎构建（ONNX -> TensorRT）
# ---------------------------------------------------------------------------
class EngineBuildRequest(BaseModel):
    """trtexec 转换参数。字段刻意贴近 trtexec 官方参数，便于对照文档排查。"""

    onnx: str = Field(..., description="待转换的 ONNX 绝对路径")
    engine: str = Field(default="", description="输出引擎路径，留空按 ONNX 同名推导")
    precision: str = Field(default="fp16", description="fp32 / fp16")
    dynamic: bool = Field(default=False, description="是否编译动态 shape")
    min_shape: str = Field(default="", description="如 1x3x640x640")
    opt_shape: str = Field(default="", description="如 1x3x640x640")
    max_shape: str = Field(default="", description="如 8x3x640x640")
    workspace_mb: int = Field(default=4096, ge=256, le=32768)
    extra_args: str = Field(default="", description="追加的原始 trtexec 参数")


def _probe_onnx(onnx_path: Path) -> dict:
    """读取 ONNX 的输入名与输出形状。

    用 onnxruntime 而不是 onnx 库：requirements 里已经有 onnxruntime，
    不必额外引入依赖。探测失败不阻断构建，只是少一层预检。
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        return {
            "input_name": inp.name,
            "input_shape": [d if isinstance(d, int) else -1 for d in inp.shape],
            "output_name": out.name,
            "output_shape": [d if isinstance(d, int) else -1 for d in out.shape],
        }
    except Exception as exc:  # noqa: BLE001 - 预检失败不影响构建
        return {"error": f"{type(exc).__name__}: {exc}"}


def _looks_like_untransposed_yolo(output_shape: list) -> bool:
    """判断输出是否为「未转置的 YOLO」布局 `[1, 4+类别数, 候选框数]`。

    `yolo_infer.dll` 按 `[1, 候选框数, 4+类别数]` 解析输出。用未转置的 ONNX
    编译引擎时，DLL 会把类别数推算成 `候选框数 - 4`，解码越界读显存，
    **不返回错误码**却输出 `confidence=957`、`top=-725` 这类垃圾框。
    与其让用户以为成功了，不如在构建前直接拦下。

    判据：三维、第 2 维是小值（类别维度，<=100）、第 3 维很大（候选框数 >1000）。
    对 anomalib 的 `[1, 3, 256, 256]` 不成立（256 不 >1000），不会误伤。
    """
    if len(output_shape) != 3:
        return False
    _, dim1, dim2 = output_shape
    if not isinstance(dim1, int) or not isinstance(dim2, int):
        return False
    return 0 < dim1 < 100 and dim2 > 1000


def _shape_side(spec: str) -> int | None:
    """从 `1x3x960x960` 这类 shape 串里取出边长（最后一维）。

    构建耗时随输入面积的平方增长，所以估算只需要这一个数。解析不出来返回 None，
    由调用方回退到默认 640。
    """
    parts = [p.strip() for p in spec.replace("X", "x").split("x") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        side = int(parts[-1])
    except ValueError:
        return None
    return side if side > 0 else None


@app.get("/api/engine/probe")
def probe_onnx(onnx: str) -> dict:
    """探测 ONNX 的输入/输出形状，并判断布局是否适合当前推理链路。

    前端选完文件即可调用，让用户在点「开始构建」之前就知道选对没有。
    """
    onnx_path = Path(onnx).expanduser()
    if not onnx_path.is_file():
        raise HTTPException(status_code=400, detail=f"ONNX 文件不存在：{onnx}")

    probe = _probe_onnx(onnx_path)
    output_shape = probe.get("output_shape") or []
    untransposed = _looks_like_untransposed_yolo(output_shape)

    if probe.get("error"):
        verdict = f"无法解析：{probe['error']}"
    elif untransposed:
        verdict = (
            f"输出 {output_shape} 是「未转置」布局，yolo_infer.dll 无法正确解析。"
            "请改用 .transd.onnx。"
        )
    elif len(output_shape) == 3 and output_shape[2] < 100:
        verdict = f"输出 {output_shape}，布局正确，可以编译。"
    else:
        verdict = f"输出 {output_shape}，非 YOLO 布局，按普通模型处理。"

    return {
        "path": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        "probe": probe,
        "untransposed_yolo": untransposed,
        "verdict": verdict,
    }


@app.post("/api/engine/build")
def build_engine(request: EngineBuildRequest) -> dict:
    """用 trtexec 把 ONNX 转成 TensorRT 引擎，日志复用通用事件流。"""
    trtexec = config.resolve_trtexec()
    if not trtexec:
        raise HTTPException(
            status_code=500,
            detail=(
                "找不到 trtexec.exe。请设置环境变量 TRTFORGE_TRTEXEC 指向它，"
                "或确认 TensorRT 安装在 CUDA_PATH\\bin 下。"
            ),
        )

    onnx_path = Path(request.onnx).expanduser()
    if not onnx_path.is_file():
        raise HTTPException(status_code=400, detail=f"ONNX 文件不存在：{request.onnx}")
    if onnx_path.suffix.lower() != ".onnx":
        raise HTTPException(status_code=400, detail="只接受 .onnx 文件")

    # 输出路径：留空则按 ONNX 同名 + 精度后缀推导，减少填错
    if request.engine.strip():
        engine_path = Path(request.engine.strip()).expanduser()
    else:
        tag = "fp16" if request.precision == "fp16" else "fp32"
        engine_path = onnx_path.with_name(f"{onnx_path.stem}_{tag}.engine")

    if not engine_path.parent.is_dir():
        raise HTTPException(status_code=400, detail=f"输出目录不存在：{engine_path.parent}")
    if engine_path.suffix.lower() not in {".engine", ".trt", ".plan"}:
        raise HTTPException(
            status_code=400, detail="输出文件名需以 .engine / .trt / .plan 结尾"
        )

    # 预检：拦住「未转置」的 YOLO 输出
    probe = _probe_onnx(onnx_path)
    output_shape = probe.get("output_shape") or []
    if _looks_like_untransposed_yolo(output_shape):
        raise HTTPException(
            status_code=400,
            detail=(
                f"该 ONNX 的输出形状是 {output_shape}，属于「未转置」布局。"
                "用它编译出的引擎会被 yolo_infer.dll 误解析：结果全错却不报错"
                "（置信度上千、坐标为负）。请改用 launcher 导出时生成的 "
                ".transd.onnx（输出形如 [1, 候选框数, 4+类别数]）。"
            ),
        )

    # 组装 trtexec 参数。fp32 是 trtexec 默认行为，它没有 --fp32 参数，不传。
    command = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if request.precision == "fp16":
        command.append("--fp16")
    elif request.precision != "fp32":
        raise HTTPException(
            status_code=400, detail=f"不支持的精度：{request.precision}，可选 fp32 / fp16"
        )
    command.append(f"--workspace={request.workspace_mb}")

    if request.dynamic:
        shapes = (
            request.min_shape.strip(),
            request.opt_shape.strip(),
            request.max_shape.strip(),
        )
        if not all(shapes):
            raise HTTPException(
                status_code=400,
                detail="启用动态 shape 时必须同时填 min / opt / max，例如 1x3x640x640",
            )
        input_name = probe.get("input_name") or "images"
        command += [
            f"--minShapes={input_name}:{shapes[0]}",
            f"--optShapes={input_name}:{shapes[1]}",
            f"--maxShapes={input_name}:{shapes[2]}",
        ]

    if request.extra_args.strip():
        command += shlex.split(request.extra_args.strip(), posix=False)

    # 进度估算的信息：trtexec 的耗时几乎全在 tactic 搜索，而它主要由
    # 「ONNX 体积 × 输入面积」决定。把这三项交给 progress 模块，
    # 有历史就取中位数，没历史按实测基线外推。
    imgsz = 640
    if request.dynamic and request.opt_shape.strip():
        imgsz = _shape_side(request.opt_shape) or 640
    else:
        input_shape = probe.get("input_shape") or []
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
            imgsz = int(input_shape[2])
    meta = {
        "onnx_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        "precision": request.precision,
        "dynamic": request.dynamic,
        "imgsz": imgsz,
    }
    estimate = progress.estimate_seconds(meta, progress.default_store())
    # 刷新后前端只能拿到 snapshot（GET /api/runs），拿不到创建响应里的
    # engine / estimate_seconds。放进 meta 才能在「接管」时把进度条和输出路径恢复出来。
    meta["onnx"] = str(onnx_path)
    meta["engine"] = str(engine_path)
    meta["estimate_seconds"] = round(estimate, 1)

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = manager.create(
                "engine-build",
                command,
                cwd=str(config.REPO_ROOT),
                expect_file=engine_path,
                meta=meta,
                estimate=estimate,
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动 trtexec 失败：{exc}") from exc

    return {
        **run.snapshot(),
        "engine": str(engine_path),
        "onnx_probe": probe,
        "estimate_seconds": round(estimate, 1),
    }


# ---------------------------------------------------------------------------
# 全流程串联：训练 → 导出 ONNX → 校验转置 → 编译引擎 → 补标签 → 引擎验证
# ---------------------------------------------------------------------------
class PipelineYoloRequest(BaseModel):
    """一键全流程的参数。

    刻意与 `YoloTrainRequest` 分开：串联多了精度、跳过训练、验证这几项，
    塞进训练表单会让「只想训一次」的人也看到一堆用不到的字段。
    """

    data: str = Field(..., description="dataset.yaml 绝对路径")
    model: str = Field(default="yolov8n.pt", description="初始权重（.pt 或模型名）")
    task: str = Field(default="detect", description="检测任务类型；精度评估目前只支持 detect")
    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=32, le=2048)
    batch: str = Field(default="16")
    device: str = Field(default="0")
    project: str = Field(default="runs/trtforge", description="相对路径按 launcher 目录解析")
    name: str = Field(default="", description="留空自动生成 exp_<时间戳>")
    patience: int = Field(default=50, ge=0)
    workers: int = Field(default=8, ge=0)
    pretrained: str = Field(default="true")

    precision: str = Field(default="fp16", description="引擎精度 fp16 / fp32")
    workspace_mb: int = Field(default=4096, ge=256, le=65536)

    skip_train: bool = Field(default=False, description="跳到已有 .pt，只跑导出之后的部分")
    weights: str = Field(default="", description="skip_train 时的 .pt 绝对路径")

    run_val: bool = Field(default=True, description="编译完成后用引擎跑精度评估")
    val_split: str = Field(default="val", description="train / val / test")
    val_conf: float = Field(default=0.25, ge=0.01, le=1.0)
    val_nms: float = Field(default=0.45, ge=0.01, le=1.0)
    val_limit: int = Field(default=0, ge=0, description="最多评估多少张，0 = 全部")

    python: str | None = Field(default=None, description="覆盖训练解释器")


def _sanitize_part(text: str) -> str:
    """实验名 → 可安全做文件名的片段（与 launcher 的 labels 命名规则对齐）。"""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    return cleaned.strip("_") or "exp"


def _pipeline_run_dir(project: str, name: str) -> Path:
    """训练产物目录。相对 project 是相对 launcher 目录（launcher 的 cwd 就是那里）。"""
    root = Path(project).expanduser()
    if not root.is_absolute():
        root = config.YOLO_LAUNCHER_DIR / root
    return (root / name).resolve()


def _labels_json_path(run_dir: Path, name: str) -> Path:
    return run_dir / f"labels_{_sanitize_part(name)}.json"


def _ensure_labels_json(data_yaml: str, run_dir: Path, name: str) -> dict:
    """从 dataset.yaml 的 names 生成 labels_*.json。

    训练阶段 launcher 自己会写一份，这里只处理「跳过训练」或「launcher 没写」的情况。
    先查存在再写：不覆盖已有文件，避免把训练时记下的类别顺序改掉。
    """
    target = _labels_json_path(run_dir, name)
    if target.is_file():
        return {"artifact": str(target), "created": False}

    text = Path(data_yaml).read_text(encoding="utf-8", errors="replace")
    # parse_names_from_yaml 返回 [{"id": 0, "name": "..."}]，标签文件里要的是纯字符串数组
    entries = dataset.parse_names_from_yaml(text)
    classes = [
        entry["name"] if isinstance(entry, dict) else str(entry) for entry in entries
    ]
    if not classes:
        raise ValueError(
            f"dataset.yaml 里读不到类别名（names 字段）：{data_yaml}，无法生成标签文件"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": _sanitize_part(name), "classes": list(classes)}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"artifact": str(target), "created": True, "classes": len(classes)}


def _pipeline_probe_transd(transd_onnx: Path) -> dict:
    """编译前确认输出是「已转置」布局。

    放到独立阶段而不是塞进 export：导出成功不代表布局正确（旧版 launcher 只转一个
    输出也能"导出成功"），等到编译完、推理出垃圾框才发现就太晚了。
    """
    probe = _probe_onnx(transd_onnx)
    output_shape = probe.get("output_shape") or []
    if _looks_like_untransposed_yolo(output_shape):
        raise ValueError(
            f"{transd_onnx.name} 的输出形状是 {output_shape}，仍是「未转置」布局。"
            "用它编译的引擎会被 DLL 误解析（结果全错但不报错），已中止。"
        )
    return {"artifact": str(transd_onnx), "output_shape": output_shape}


@app.post("/api/pipeline/yolo")
def pipeline_yolo(request: PipelineYoloRequest) -> dict:
    """一键跑完「训练 → 导出 → 校验 → 编译 → 标签 → 引擎验证」。"""
    data_yaml = Path(request.data).expanduser()
    if not data_yaml.is_file():
        raise HTTPException(status_code=400, detail=f"dataset.yaml 不存在：{request.data}")
    _ensure_allowed_path(str(data_yaml))
    if request.task != "detect" and request.run_val:
        raise HTTPException(
            status_code=400,
            detail="精度评估目前只支持 detect；请关闭「完成后验证」，或把任务类型改为 detect",
        )
    if request.precision not in ("fp16", "fp32"):
        raise HTTPException(status_code=400, detail="precision 只能是 fp16 / fp32")

    trtexec = config.resolve_trtexec()
    if not trtexec:
        raise HTTPException(
            status_code=500,
            detail="找不到 trtexec.exe（设置 TRTFORGE_TRTEXEC 或确认 CUDA\\bin 下有它）",
        )

    python = _resolve_python(request.python)
    name = request.name.strip() or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = _pipeline_run_dir(request.project, name)

    weights = run_dir / "weights" / "best.pt"
    if request.skip_train:
        if not request.weights.strip():
            raise HTTPException(status_code=400, detail="跳过训练时必须给出已有 .pt 权重路径")
        weights = Path(request.weights).expanduser()
        if not weights.is_file():
            raise HTTPException(status_code=400, detail=f"权重不存在：{request.weights}")
        if weights.suffix.lower() != ".pt":
            raise HTTPException(status_code=400, detail="权重必须是 .pt")
        _ensure_allowed_path(str(weights))
        run_dir = weights.parent.parent  # weights/best.pt → run 目录

    transd_onnx = weights.with_name(f"{weights.stem}.transd.onnx")
    onnx_file = weights.with_suffix(".onnx")
    tag = "fp16" if request.precision == "fp16" else "fp32"
    engine_path = weights.with_name(f"{weights.stem}_{tag}.engine")
    labels_path = _labels_json_path(run_dir, name if not request.skip_train else run_dir.name)

    stages: list[pipeline_mod.StagePlan] = []
    if not request.skip_train:
        train_request = YoloTrainRequest(
            data=str(data_yaml),
            model=request.model,
            task=request.task,
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            device=request.device,
            project=request.project,
            name=name,
            patience=request.patience,
            workers=request.workers,
            pretrained=request.pretrained,
        )
        stages.append(
            pipeline_mod.StagePlan(
                name="train",
                label="训练",
                command=build_train_command(train_request, python),
                cwd=str(config.YOLO_LAUNCHER_DIR),
                expect_file=str(weights),
                meta={"epochs": request.epochs, "imgsz": request.imgsz},
            )
        )

    export_request = YoloExportRequest(
        weights=str(weights),
        task=request.task,
        imgsz=request.imgsz,
        make_transd=True,
        simplify=True,
    )
    stages.append(
        pipeline_mod.StagePlan(
            name="export",
            label="导出 ONNX",
            command=build_export_command(export_request, python),
            cwd=str(config.YOLO_LAUNCHER_DIR),
            expect_file=str(transd_onnx),
            meta={"weights": str(weights)},
        )
    )
    stages.append(
        pipeline_mod.StagePlan(
            name="probe",
            label="校验转置布局",
            callable=lambda: _pipeline_probe_transd(transd_onnx),
        )
    )
    build_command = [
        trtexec,
        f"--onnx={transd_onnx}",
        f"--saveEngine={engine_path}",
    ]
    if request.precision == "fp16":
        build_command.append("--fp16")
    build_command.append(f"--workspace={request.workspace_mb}")

    stages.append(
        pipeline_mod.StagePlan(
            name="build",
            label=f"编译引擎（{request.precision}）",
            command=build_command,
            cwd=str(config.REPO_ROOT),
            expect_file=str(engine_path),
            meta={"precision": request.precision, "imgsz": request.imgsz},
        )
    )
    stages.append(
        pipeline_mod.StagePlan(
            name="labels",
            label="生成标签文件",
            callable=lambda: _ensure_labels_json(str(data_yaml), run_dir, name),
        )
    )
    if request.run_val:
        report_dir = run_dir / "val"
        stages.append(
            pipeline_mod.StagePlan(
                name="val",
                label="引擎精度验证",
                command=config.tool_command(python, "val_engine.py") + [
                    "--engine", str(engine_path),
                    "--data", str(data_yaml),
                    "--labels", str(labels_path),
                    "--split", request.val_split,
                    "--task", request.task,
                    "--conf", str(request.val_conf),
                    "--nms", str(request.val_nms),
                    "--limit", str(request.val_limit),
                    "--out", str(report_dir),
                ],
                cwd=str(BACKEND_DIR),
                expect_file=str(report_dir / "val_report.json"),
                report_file=str(report_dir / "val_report.json"),
                meta={"split": request.val_split},
            )
        )

    # 串联的每一段都是独占任务，所以整条链也按独占处理：
    # 不拦的话，串联跑到「训练」那一段时用户还能从训练页再开一个训练。
    with _START_LOCK:
        _reject_if_busy()
        pipeline = pipeline_manager.create(
            kind="yolo",
            title=f"{name}（{request.task} / {request.precision}）",
            stages=stages,
            artifacts={
                "run_dir": str(run_dir),
                "pt": str(weights),
                "onnx": str(onnx_file),
                "transd_onnx": str(transd_onnx),
                "engine": str(engine_path),
                "labels": str(labels_path),
            },
            meta={
                "data": str(data_yaml),
                "name": name,
                "project": request.project,
                "precision": request.precision,
                "skip_train": request.skip_train,
                "run_val": request.run_val,
            },
        )
    return pipeline


# ---------------------------------------------------------------------------
# 全流程串联（Anomalib 异常检测）
# ---------------------------------------------------------------------------
class AnomalibPipelineRequest(BaseModel):
    """异常检测的一键全流程参数。

    与 `AnomalibTrainRequest` 分开的理由和 YOLO 那边一样：串联多了精度、跳过训练、
    验证目录这几项，塞进训练表单会让「只想训一次」的人看到一堆用不到的字段。
    """

    model: str = Field(default="patchcore", description="patchcore / padim / efficientad")
    root: str = Field(default="", description="Folder 数据集根目录；skip_train 时可留空")
    normal_dir: str = "train/good"
    abnormal_dir: str = "test/abnormal"
    normal_test_dir: str = "test/good"
    project: str = Field(default="results", description="相对路径按 launcher 目录解析")
    name: str = Field(default="", description="留空自动生成 pipe_<时间戳>")
    input_size: int = Field(default=256, ge=64, le=1024)
    max_epochs: str = "auto"
    train_batch_size: int = Field(default=4, ge=1)
    eval_batch_size: int = Field(default=4, ge=1)
    num_workers: int = Field(default=0, ge=0)
    val_split_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    calibrate_metadata: bool = True
    calib_limit: int = Field(default=64, ge=1, le=1024)
    imagenet_dir: str = ""
    calib_normal_dir: str = ""
    calib_abnormal_dir: str = ""

    precision: str = Field(default="fp16", description="引擎精度 fp16 / fp32")
    workspace_mb: int = Field(default=4096, ge=256, le=65536)
    max_batch: int = Field(
        default=8,
        ge=1,
        le=64,
        description="导出 ONNX 的 batch 恒为动态轴，编译时 maxShapes 的上限批次",
    )

    skip_train: bool = Field(default=False, description="跳过训练，直接用已有产物目录")
    artifacts_dir: str = Field(
        default="", description="skip_train 时给出包含 metadata.json 与 .onnx 的目录"
    )

    run_val: bool = Field(default=True)
    val_limit: int = Field(default=0, ge=0, description="每个类别最多评估多少张，0 = 全部")
    val_normal_dir: str = Field(default="", description="留空则用 root/normal_test_dir")
    val_abnormal_dir: str = Field(default="", description="留空则用 root/abnormal_dir")

    python: str | None = Field(default=None, description="覆盖训练解释器")


def _is_number(value: Any) -> bool:
    """排除 bool：`True` 也是 int，当成阈值会静默变成 1.0。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _anomaly_project_root(project: str) -> Path:
    """Anomalib 的 `--project` 是相对 launcher 目录解析的（launcher 的 cwd 就是那里）。"""
    root = Path(project).expanduser()
    if not root.is_absolute():
        root = config.ANOMALIB_LAUNCHER_DIR / root
    return root.resolve()


def _anomaly_version_number(directory: Path) -> int:
    """`v3` → 3；不是版本目录返回 -1。"""
    match = re.match(r"^v(\d+)$", directory.name)
    return int(match.group(1)) if match else -1


def _anomaly_artifacts_in(directory: Path) -> dict[str, Any]:
    """从一个 `weights/lightning` 目录里收齐产物。"""
    metadata = directory / "metadata.json"
    if not metadata.is_file():
        raise ValueError(f"目录里没有 metadata.json：{directory}")
    onnx_files = sorted(directory.glob("*.onnx"), key=lambda path: path.stat().st_mtime)
    if not onnx_files:
        raise ValueError(f"目录里没有 .onnx：{directory}")
    ckpt = directory / "model.ckpt"
    return {
        "run_dir": str(directory),
        "metadata": str(metadata),
        "onnx": str(onnx_files[-1]),
        "ckpt": str(ckpt) if ckpt.is_file() else "",
    }


def _locate_anomaly_artifacts(project_root: Path, name: str) -> dict[str, Any]:
    """在 `<project>/<模型类名>/<name>/v<N>/weights/lightning` 里找产物。

    为什么用 glob 而不是拼路径：中间那层目录名是 Anomalib 的模型类名
    （`Patchcore` / `EfficientAd` / ...），版本目录 `vN` 每次训练递增。
    硬拼 `<project>/<name>/v0` 在第二次训练时就会指向上一次的旧模型——
    旧模型同样能编译、同样能出分数，错得毫无提示。
    """
    if not name:
        raise ValueError("训练名称为空，无法定位产物目录")
    candidates: list[tuple[int, float, Path]] = []
    if project_root.is_dir():
        for model_dir in project_root.iterdir():
            if not model_dir.is_dir():
                continue
            run_root = model_dir / name
            if not run_root.is_dir():
                continue
            for version_dir in run_root.iterdir():
                lightning = version_dir / "weights" / "lightning"
                metadata = lightning / "metadata.json"
                if version_dir.is_dir() and metadata.is_file():
                    candidates.append(
                        (_anomaly_version_number(version_dir), metadata.stat().st_mtime, lightning)
                    )
    if not candidates:
        raise ValueError(
            f"没有在 {project_root} 下找到 {name} 的 metadata.json（应位于 "
            f"<模型类名>/{name}/v<N>/weights/lightning/）。训练是否真的导出了 ONNX？"
        )
    # 先按版本号，再按修改时间：同一版本号被覆盖时取最新那份
    version, _, lightning_dir = max(candidates, key=lambda item: (item[0], item[1]))
    artifacts = _anomaly_artifacts_in(lightning_dir)
    artifacts["version"] = f"v{version}" if version >= 0 else lightning_dir.parent.name
    return artifacts


def _probe_anomaly_metadata(artifacts: dict[str, Any]) -> dict[str, Any]:
    """编译前确认 metadata.json 与 ONNX 能撑起推理和精度评估。

    DLL 在缺 `image_threshold` / `pixel_threshold` 时会在 `ai_create` 阶段直接
    失败（`anomalib_infer_c.cpp:633`），但那时引擎已经编译完、几分钟白等。
    这里提前拦：既省时间，也避免「编译成功 = 可以推理」的误解。
    """
    metadata_path = Path(artifacts["metadata"])
    onnx_path = Path(artifacts["onnx"])
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"metadata.json 无法解析：{exc}") from exc

    missing = [
        key for key in ("image_threshold", "pixel_threshold") if not _is_number(raw.get(key))
    ]
    if missing:
        raise ValueError(
            f"metadata.json 缺字段 {missing}（anomalib_infer.dll 会拒绝加载），"
            "请重新导出，或打开阈值校准后重跑。"
        )

    probe = _probe_onnx(onnx_path)
    if probe.get("error"):
        raise ValueError(f"ONNX 无法解析：{probe['error']}")
    shape = probe.get("input_shape") or []
    # Anomalib 的 `to_onnx` 固定把 batch 导成动态轴（`dynamic_axes={"input": {0: ...}}`），
    # 所以 batch 维度读出来是符号维度 → `_probe_onnx` 记为 -1。这是**预期**的：
    # trtexec 只要给出 min/opt/max 就能编译动态 batch，DLL 侧也会
    # `setBindingDimensions(1x3xHxW)`（anomalib_infer_c.cpp:701）。
    # 真正不能接受的是 H/W 动态——DLL 拿 metadata 里的 input_size 去设形状，
    # 与导出时的动态范围对不上就会静默算错。
    if len(shape) != 4 or any(not isinstance(dim, int) or dim <= 0 for dim in shape[1:]):
        raise ValueError(
            f"ONNX 输入形状 {shape} 不是 4 维 NCHW，或高/宽是动态的。"
            "请带固定 input_size（如 256）重新导出后再编译。"
        )
    dynamic_batch = not isinstance(shape[0], int) or shape[0] <= 0
    resolved_shape = [1, *shape[1:]] if dynamic_batch else list(shape)

    recorded = {
        key: value
        for key, value in artifacts.items()
        if key in ("run_dir", "metadata", "onnx", "ckpt") and value
    }
    return {
        "artifact": str(metadata_path),
        "artifacts": recorded,
        "image_threshold": float(raw["image_threshold"]),
        "pixel_threshold": float(raw["pixel_threshold"]),
        "input_shape": resolved_shape,
        "input_name": probe.get("input_name") or "input",
        "dynamic_batch": dynamic_batch,
        "output_contract": str(raw.get("output_contract", "")),
    }


def _anomaly_engine_path(onnx_path: Path, precision: str) -> Path:
    """引擎与 ONNX 同目录同名 + 精度后缀，和 YOLO 侧的约定一致。"""
    tag = "fp16" if precision == "fp16" else "fp32"
    return onnx_path.with_name(f"{onnx_path.stem}_{tag}.engine")


def _anomaly_shape_args(probe: dict[str, Any], max_batch: int) -> list[str]:
    """给动态 batch 的 ONNX 补 trtexec 的 min/opt/max 形状参数。

    `optShapes` 取 batch=1：DLL 每次只推一张图（`anomalib_infer_c.cpp:701`
    固定 `setBindingDimensions(1x3xHxW)`），tactic 按 batch 1 调优才是实际
    推理最快的选择；`maxShapes` 留出余量给以后批量调用。静态 batch 的
    ONNX 不需要这些参数，直接返回空列表。
    """
    shape = probe.get("input_shape") or []
    if len(shape) != 4:
        return []
    batch = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else 1
    if all(isinstance(dim, int) and dim > 0 for dim in shape):
        return []
    name = probe.get("input_name") or "input"
    hw = "x".join(str(int(dim)) for dim in shape[1:])

    def spec(value: int) -> str:
        return f"{value}x{hw}"

    return [
        f"--minShapes={name}:{spec(1)}",
        f"--optShapes={name}:{spec(batch)}",
        f"--maxShapes={name}:{spec(max(max_batch, batch))}",
    ]


@app.post("/api/pipeline/anomalib")
def pipeline_anomalib(request: AnomalibPipelineRequest) -> dict:
    """一键跑完「训练+导出 → 校验 metadata → 编译 → 引擎精度验证」。

    刻意只有四段：Anomalib 的 launcher 在 `train-export` 一个进程里就把
    训练、ONNX 导出、`metadata.json` 阈值校准做完了。为了阶段列表好看而
    拆成三个进程，只会多两次模型加载，并把「导出拿到的不是刚训的模型」
    这类错埋进编排层。
    """
    if not config.ANOMALIB_LAUNCHER.is_file():
        raise HTTPException(
            status_code=500, detail=f"找不到 launcher：{config.ANOMALIB_LAUNCHER}"
        )
    if request.model not in ("patchcore", "padim", "efficientad"):
        raise HTTPException(
            status_code=400,
            detail=f"model 只能是 patchcore / padim / efficientad，实际 {request.model}",
        )
    if request.precision not in ("fp16", "fp32"):
        raise HTTPException(status_code=400, detail="precision 只能是 fp16 / fp32")

    python = _resolve_python(request.python)
    trtexec = config.resolve_trtexec()
    if not trtexec:
        raise HTTPException(
            status_code=500,
            detail="找不到 trtexec.exe（设置 TRTFORGE_TRTEXEC 或确认 CUDA\\bin 下有它）",
        )

    name = request.name.strip() or f"pipe_{time.strftime('%Y%m%d_%H%M%S')}"
    project_root = _anomaly_project_root(request.project)

    root: Path | None = None
    if request.root.strip():
        root = Path(request.root).expanduser()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"数据集根目录不存在：{request.root}")

    # 跳过训练时必须是「已经能拿到 metadata.json 与 .onnx」的目录：
    # 让用户自己挑目录，比让后端去猜上一次训练的版本号可靠。
    seeded: dict[str, Any] = {}
    if request.skip_train:
        if not request.artifacts_dir.strip():
            raise HTTPException(
                status_code=400,
                detail="跳过训练时必须给出产物目录（包含 metadata.json 与 .onnx）",
            )
        seed_dir = Path(request.artifacts_dir).expanduser()
        if not seed_dir.is_dir():
            raise HTTPException(
                status_code=400, detail=f"产物目录不存在：{request.artifacts_dir}"
            )
        try:
            seeded = _anomaly_artifacts_in(seed_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if root is None:
            raise HTTPException(status_code=400, detail="没有跳过训练时必须给出数据集根目录")
        for label, sub in (
            ("正常训练目录", request.normal_dir),
            ("异常测试目录", request.abnormal_dir),
        ):
            target = root / sub
            if not target.is_dir():
                raise HTTPException(status_code=400, detail=f"{label}不存在：{target}")

    # 精度验证的图从哪来：优先显式目录，其次按 root 下的标准子目录推
    val_normal: Path | None = None
    val_abnormal: Path | None = None
    if request.run_val:
        if request.val_normal_dir.strip():
            val_normal = Path(request.val_normal_dir).expanduser()
        elif root is not None and (root / request.normal_test_dir).is_dir():
            val_normal = root / request.normal_test_dir
        if request.val_abnormal_dir.strip():
            val_abnormal = Path(request.val_abnormal_dir).expanduser()
        elif root is not None and (root / request.abnormal_dir).is_dir():
            val_abnormal = root / request.abnormal_dir
        if val_normal is None and val_abnormal is None:
            raise HTTPException(
                status_code=400,
                detail="精度验证需要至少一个测试目录：请填「正常测试目录」或「异常测试目录」",
            )

    stages: list[pipeline_mod.StagePlan] = []

    if not request.skip_train:
        train_request = AnomalibTrainRequest(
            model=request.model,
            root=str(root),
            normal_dir=request.normal_dir,
            abnormal_dir=request.abnormal_dir,
            normal_test_dir=request.normal_test_dir,
            project=request.project,
            name=name,
            input_size=request.input_size,
            max_epochs=request.max_epochs,
            train_batch_size=request.train_batch_size,
            eval_batch_size=request.eval_batch_size,
            num_workers=request.num_workers,
            val_split_ratio=request.val_split_ratio,
            calibrate_metadata=request.calibrate_metadata,
            calib_limit=request.calib_limit,
            imagenet_dir=request.imagenet_dir,
            calib_normal_dir=request.calib_normal_dir,
            calib_abnormal_dir=request.calib_abnormal_dir,
        )
        stages.append(
            pipeline_mod.StagePlan(
                name="train_export",
                label="训练 + 导出 ONNX + 校准 metadata",
                command=build_anomalib_train_command(train_request, python),
                cwd=str(config.ANOMALIB_LAUNCHER_DIR),
                # 产物落在哪个 vN 目录训练完才知道，不在这里预言；
                # 下一段 probe 找不到会直接失败并说明找过哪里。
                meta={"model": request.model, "input_size": request.input_size},
            )
        )

    def probe_artifacts() -> dict[str, Any]:
        if seeded:
            return _probe_anomaly_metadata(seeded)
        return _probe_anomaly_metadata(_locate_anomaly_artifacts(project_root, name))

    stages.append(
        pipeline_mod.StagePlan(
            name="probe",
            label="校验 metadata 与 ONNX 形状",
            callable=probe_artifacts,
        )
    )

    def resolve_build(record: dict[str, Any]) -> dict[str, Any]:
        onnx_path = Path(record["artifacts"]["onnx"])
        engine_path = _anomaly_engine_path(onnx_path, request.precision)
        # 这里重新探一次 ONNX：probe 阶段算出的形状/输入名只留在该段返回值里，
        # 没有进 `record["artifacts"]`（那里只存路径）。读 ONNX 头只需几十毫秒，
        # 比为了传三个值去改持久化结构划算。
        probe = _probe_onnx(onnx_path)
        if probe.get("error"):
            raise ValueError(f"ONNX 无法解析：{probe['error']}")
        command = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]
        if request.precision == "fp16":
            command.append("--fp16")
        command.append(f"--workspace={request.workspace_mb}")
        command += _anomaly_shape_args(probe, request.max_batch)
        return {"command": command, "expect_file": engine_path}

    stages.append(
        pipeline_mod.StagePlan(
            name="build",
            label=f"编译引擎（{request.precision}）",
            cwd=str(config.REPO_ROOT),
            resolve=resolve_build,
            meta={"precision": request.precision, "input_size": request.input_size},
        )
    )

    if request.run_val:

        def resolve_val(record: dict[str, Any]) -> dict[str, Any]:
            onnx_path = Path(record["artifacts"]["onnx"])
            metadata_path = Path(record["artifacts"]["metadata"])
            engine_path = _anomaly_engine_path(onnx_path, request.precision)
            report_dir = (
                Path(record["artifacts"].get("run_dir") or metadata_path.parent) / "val"
            )
            command = config.tool_command(python, "val_anomaly.py") + [
                "--engine", str(engine_path),
                "--metadata", str(metadata_path),
                "--out", str(report_dir),
                "--limit", str(request.val_limit),
            ]
            if val_normal is not None:
                command += ["--normal-dir", str(val_normal)]
            if val_abnormal is not None:
                command += ["--abnormal-dir", str(val_abnormal)]
            report = report_dir / "anomaly_val_report.json"
            return {"command": command, "expect_file": report, "report_file": report}

        stages.append(
            pipeline_mod.StagePlan(
                name="val",
                label="引擎精度验证（AUROC / 漏检 / 误报）",
                cwd=str(BACKEND_DIR),
                resolve=resolve_val,
                meta={
                    "normal_dir": str(val_normal) if val_normal else "",
                    "abnormal_dir": str(val_abnormal) if val_abnormal else "",
                },
            )
        )

    artifacts: dict[str, Any] = {"project": str(project_root)}
    if seeded:
        artifacts.update({key: value for key, value in seeded.items() if value})

    with _START_LOCK:
        _reject_if_busy()
        pipeline = pipeline_manager.create(
            kind="anomalib",
            title=f"{name}（{request.model} / {request.precision}）",
            stages=stages,
            artifacts=artifacts,
            meta={
                "root": str(root) if root is not None else "",
                "name": name,
                "project": request.project,
                "model": request.model,
                "precision": request.precision,
                "skip_train": request.skip_train,
                "run_val": request.run_val,
            },
        )
    return pipeline


@app.get("/api/pipeline")
def pipeline_list(limit: int = 10) -> dict:
    """最近若干条串联记录。刷新页面后还能看到上次跑到哪。"""
    return {"items": pipeline_manager.list(max(1, min(limit, 50)))}


@app.get("/api/pipeline/{pipeline_id}/log")
def pipeline_stage_log(pipeline_id: str, stage: str, lines: int = 400) -> Response:
    """读某一段的日志尾巴。

    为什么不复用训练页的日志面板：那是 WebSocket 流，只能看**内存里还活着**的 run。
    服务重启后 run 已经没了，但 `runtime/logs/{run_id}.log` 还在——排障要的恰恰是
    这条已经结束的日志，所以这里直接读文件。
    """
    record = pipeline_manager.get(pipeline_id)
    if record is None:
        record = pipeline_mod.read_persisted(
            config.RUNTIME_DIR / "pipelines", pipeline_id
        )
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有这条串联记录：{pipeline_id}")

    stage_state = next(
        (item for item in record.get("stages") or [] if item.get("name") == stage), None
    )
    if stage_state is None:
        raise HTTPException(status_code=404, detail=f"没有这个阶段：{stage}")
    run_id = stage_state.get("run_id")
    if not run_id:
        raise HTTPException(status_code=404, detail=f"阶段 {stage} 没有日志（未启动或被跳过）")

    log_file = (config.LOG_DIR / f"{run_id}.log").resolve()
    # 路径只认 runtime/logs 下的 {run_id}.log，run_id 里带 .. 也不能跳出去
    if log_file.parent != config.LOG_DIR.resolve() or not log_file.is_file():
        raise HTTPException(status_code=404, detail=f"阶段 {stage} 的日志文件不存在")

    limit = max(1, min(int(lines), 5000))
    text = _read_log_tail(log_file, limit)
    return Response(content=text, media_type="text/plain; charset=utf-8")


def _read_log_tail(path: Path, limit: int) -> str:
    """取最后 N 行。整个文件读进来在几百 MB 的构建日志上会很难受。"""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = 64 * 1024
            data = b""
            while size > 0 and data.count(b"\n") <= limit:
                step = min(block, size)
                size -= step
                handle.seek(size)
                data = handle.read(step) + data
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"读取日志失败：{exc}") from exc
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


@app.get("/api/pipeline/{pipeline_id}")
def pipeline_detail(pipeline_id: str) -> dict:
    record = pipeline_manager.get(pipeline_id)
    live = record is not None
    if record is None:
        record = pipeline_mod.read_persisted(
            config.RUNTIME_DIR / "pipelines", pipeline_id
        )
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有这条串联记录：{pipeline_id}")
    # live=False 表示这条只能从磁盘读到：前端不该再按它轮询，也不该给「停止」按钮
    record["live"] = live
    return record


@app.post("/api/pipeline/{pipeline_id}/stop")
def pipeline_stop(pipeline_id: str) -> dict:
    """停止整条链：只停当前阶段，已完成的阶段产物保留。"""
    if not pipeline_manager.stop(pipeline_id):
        raise HTTPException(status_code=409, detail="该串联已结束，无法停止")
    record = pipeline_manager.get(pipeline_id) or {}
    return {"ok": True, "pipeline": record}


# ---------------------------------------------------------------------------
# 数据集工作台：扫描 / 划分预览 / 导出
# ---------------------------------------------------------------------------
class DatasetScanRequest(BaseModel):
    """扫描请求。ratios 只影响预览，不落盘。"""

    source: str = Field(..., description="YOLO 数据集目录、data.yaml 或 .tfanno 文件")
    task: str = Field(default="auto", description="auto / detect / segment / pose / obb")
    train: int = Field(default=80, ge=0, le=100)
    val: int = Field(default=20, ge=0, le=100)
    limit: int = Field(default=200, ge=1, le=2000, description="预览列表最多返回多少张")


class DatasetExportRequest(DatasetScanRequest):
    """导出请求。额外带上格式、输出目录与覆盖开关。"""

    format: str = Field(default="yolo", description="yolo / coco / voc")
    output: str = Field(..., description="导出目标目录")
    overwrite: bool = Field(default=False, description="目标目录非空时是否允许覆盖")


def _ensure_dataset_path(path: str) -> None:
    """数据集路径沿用目录浏览的白名单，避免接口被用来读任意磁盘。"""
    _ensure_allowed_path(path)


@app.post("/api/dataset/scan")
def dataset_scan(request: DatasetScanRequest) -> dict:
    """扫描数据集并返回按比例划分后的预览，不写任何文件。"""
    _ensure_dataset_path(request.source)
    try:
        project = dataset.load_project(request.source, request.task)
        return dataset.preview(project, request.train, request.val, request.limit)
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"读取数据集失败：{exc}") from exc


@app.post("/api/dataset/export")
def dataset_export(request: DatasetExportRequest) -> dict:
    """扫描 → 划分 → 导出，一次请求完成，不需要前端保存划分状态。"""
    _ensure_dataset_path(request.source)
    _ensure_dataset_path(request.output)
    try:
        return dataset.run_export(
            source=request.source,
            task=request.task,
            train=request.train,
            val=request.val,
            fmt=request.format,
            output=request.output,
            overwrite=request.overwrite,
        )
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"导出失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 任务查询 / 停止 / 事件流
# ---------------------------------------------------------------------------
@app.get("/api/runs")
def list_runs(limit: int = 0) -> dict:
    """列出本次服务生命周期内的任务。

    ``limit`` 只取最近的 N 条：顶栏「运行中」条每 3 秒拉一次，长期运行的服务里
    ``_runs`` 会越攒越多（每条还带完整命令行），不截断就是白白浪费带宽。
    """
    runs = manager.list_runs()
    if limit > 0:
        runs = runs[:limit]
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """查询单个任务。"""
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"未知任务：{run_id}")
    return run.snapshot()


@app.get("/api/runs/{run_id}/log")
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

    log_file = (config.LOG_DIR / f"{run_id}.log").resolve()
    if log_file.parent != config.LOG_DIR.resolve() or not log_file.is_file():
        raise HTTPException(status_code=404, detail=f"任务 {run_id} 的日志文件不存在")

    if lines > 0:
        text = _read_log_tail(log_file, max(1, min(int(lines), 5000)))
    else:
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
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


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """停止任务。"""
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"未知任务：{run_id}")
    stopped = manager.stop(run_id)
    return {"stopped": stopped, "run": run.snapshot()}


@app.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    """把任务事件实时推给浏览器。

    连接建立后先补发历史事件，保证刷新页面不丢日志。
    """
    await websocket.accept()

    subscribed = manager.subscribe(run_id)
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
        manager.unsubscribe(run_id, queue)
        with contextlib.suppress(Exception):
            await websocket.close()


# ---------------------------------------------------------------------------
# 文件浏览（Web 无法像桌面端那样弹系统选择框，提供受限于仓库内的浏览）
# ---------------------------------------------------------------------------
@app.get("/api/browse")
def browse(path: str = "") -> dict:
    """列出目录内容，供页面选择引擎/模型/metadata 路径。

    出于安全考虑，只允许浏览仓库根目录以及显式配置的额外根。
    """
    target = Path(path).resolve() if path.strip() else config.REPO_ROOT
    if not config.is_path_allowed(target):
        raise HTTPException(status_code=403, detail="该目录不在允许浏览的范围内")

    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在：{target}")

    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
            })
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"无权限读取：{target}") from exc

    return {"path": str(target), "parent": str(target.parent), "entries": entries}


class NativeDialogRequest(BaseModel):
    """原生对话框参数。

    字段名与 ``nativedialog.select`` 一一对应；``field`` 是目标输入框的 id，
    用于记住「这个字段上次从哪个目录开始选」。
    """

    kind: str = Field(default="file", description="file / files / dir / save")
    field: str = Field(default="", description="目标输入框 id，用于记住上次目录")
    title: str = Field(default="", description="对话框标题")
    initial: str = Field(default="", description="起始目录或文件路径")
    patterns: list[str] = Field(default_factory=list, description="扩展名，如 .pt / .yaml")
    label: str = Field(default="", description="扩展名组的可读描述")
    defaultextension: str = Field(default="", description="另存为时的默认后缀，如 .tfanno")
    must_exist: bool = Field(default=True, description="文件夹是否存在校验")


def _is_local_request(request: Request) -> bool:
    """请求是否来自本机。

    原生框弹在**服务端**的桌面上。远程用户点按钮时框弹在他看不见的屏幕上，
    表现为「点了没反应」，比没有这个功能更糟。所以只服务本机来源，
    远程一律拒绝并让前端回退到页面内浏览。
    """
    client = request.client
    if client is None:
        return False
    host = (client.host or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


@app.post("/api/browse/dialog")
def browse_dialog(payload: NativeDialogRequest, request: Request) -> dict:
    """在本机桌面上弹原生文件/文件夹对话框，返回选中的绝对路径。

    为什么是 POST + 同步返回：对话框是模态的，用户可能几分钟后才点确定，
    这期间请求一直挂着。前端据此显示「请到桌面完成选择」。
    取消不算错误——返回 ``cancelled=True`` 且不改字段。
    """
    if not nativedialog.available():
        raise HTTPException(
            status_code=400,
            detail="当前环境不支持原生对话框，请改用页面内目录浏览",
        )
    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="原生对话框只对本机访问可用（框会弹在服务端屏幕上），请改用页面内目录浏览",
        )

    try:
        result = nativedialog.select(
            payload.kind,
            title=payload.title,
            field=payload.field,
            initial=payload.initial,
            patterns=payload.patterns,
            label=payload.label,
            defaultextension=payload.defaultextension,
            must_exist=payload.must_exist,
        )
    except nativedialog.DialogBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 用户亲手选的路径视为可信：加入运行期放行表，否则后续接口会 403。
    # 只放行「选中的那个目录」，不递归、不落盘，服务重启即失效。
    granted = []
    for item in result.get("paths") or []:
        allowed = config.allow_path(item)
        if allowed is not None:
            granted.append(str(allowed))

    return {"ok": True, **result, "allowed": granted}


@app.post("/api/browse/cancel")
def browse_cancel(request: Request) -> dict:
    """让当前原生对话框自己关掉。

    窗口没出现时，网页上的按钮会改成「取消」。点它不能只把浏览器请求掐断，
    服务端那次选择还占着锁，下一次点「浏览」会一直得到「已经有一个选择框」。
    """
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="只能取消本机打开的选择框")
    return {"ok": True, "cancelled": nativedialog.request_cancel()}


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def _save_upload(upload: UploadFile) -> tuple[bytes, Path]:
    """读取上传文件到临时目录，并限制大小。"""
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    if suffix.lower() not in config.IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式：{suffix}")

    data = upload.file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"图片超过 {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制",
        )

    config.ensure_dirs()
    handle = tempfile.NamedTemporaryFile(
        dir=config.UPLOAD_DIR, suffix=suffix, delete=False
    )
    handle.write(data)
    handle.close()
    return data, Path(handle.name)


def _check_yolo_output_sane(result: dict, img_w: int, img_h: int) -> None:
    """拦截「引擎输出布局不对」这类静默错误。

    背景：`yolo_infer.dll` 按 `[1, num_bboxes, 4 + num_classes]` 解析输出。
    如果引擎是用**未转置**的 ONNX（`[1, 4 + num_classes, num_bboxes]`）编译的，
    DLL 会把类别数推算成 `num_bboxes - 4`（PCB 模型上是 18896）并越界读内存，
    产出 `confidence > 1`、坐标为负的垃圾框，而且**不返回任何错误码**。

    这里把这种"看起来成功、实际全错"的情况显式变成 400，
    并在提示里指明正确做法（用 `.transd.onnx` 编译）。
    """
    boxes = result.get("boxes") or []
    if not boxes:
        return

    bad_conf = [b for b in boxes if not (0.0 <= float(b.get("confidence", 0.0)) <= 1.0)]
    if bad_conf:
        raise HTTPException(
            status_code=400,
            detail=(
                "检测结果不可信（置信度不在 0~1 之间），通常说明引擎的 ONNX 输出布局不对。"
                "YOLO 引擎必须用 launcher 导出的 `.transd.onnx` 编译"
                "（输出为 [1, 候选框数, 4+类别数]）；直接用 `model.onnx` 编译会让解码越界。"
            ),
        )

    # 框贴边小幅越界是 letterbox 还原的舍入误差，超过图像 1% 才算异常
    pad = max(img_w, img_h) * 0.01
    outside = [
        b
        for b in boxes
        if float(b.get("right", 0)) < -pad
        or float(b.get("bottom", 0)) < -pad
        or float(b.get("left", 0)) > img_w + pad
        or float(b.get("top", 0)) > img_h + pad
    ]
    if outside:
        raise HTTPException(
            status_code=400,
            detail=(
                "检测框全部落在图像之外，引擎输出布局可能不对。"
                "请确认是用 `.transd.onnx` 编译的引擎，而不是 `model.onnx`。"
            ),
        )


# ---------------------------------------------------------------------------
# 推理（DLL 优先，DLL 不可用时降级到 onnxruntime）
# ---------------------------------------------------------------------------
def _resolve_fallback_onnx(engine_path: str, metadata_path: str) -> tuple[str | None, str]:
    """为降级路径找出可用的 .onnx 模型，返回 (路径, 说明)。

    实现已下沉到 `onnx_fallback.resolve_onnx_path`：批量/视频子进程要用同一套
    查找规则，留在路由层就会出现两份互相漂移的实现。
    """
    return onnx_fallback.resolve_onnx_path(engine_path, metadata_path)


def _fallback_reason(primary: str, detail: str) -> str:
    return f"{primary} 不可用（{detail}），已自动降级为 onnxruntime"


def _select_yolo_backend(engine: str) -> tuple[bool, str, str | None]:
    """决定 YOLO 这次走哪个后端，返回 (是否走 DLL, 兜底 ONNX 路径, 降级说明)。

    三级顺序：
    1. 传的是 `.onnx` 且 `onnx_infer.dll` 可用 → 走 ONNX DLL；
    2. 传的是引擎且 `yolo_infer.dll` 可用 → 走 TRT DLL；
    3. 以上都不可用 → onnxruntime 降级（只在找不到模型时抛 503）。
    """
    backend = inferproxy.status()

    if infer.is_onnx_model(engine):
        if backend.onnx_ok:
            return True, "", None
        path = Path(engine)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"ONNX 模型不存在：{engine}")
        return False, str(path), _fallback_reason(
            "onnx_infer.dll", backend.onnx_detail or "未知原因"
        )

    if backend.yolo_ok:
        return True, "", None

    onnx_path, reason = _resolve_fallback_onnx(engine, "")
    if onnx_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "yolo_infer.dll 不可用："
                f"{backend.detail}。请确认已编译 DLL，并设置 OPENCV_DIR / CUDA_PATH。"
                f"（也可改用 ONNX 降级，但{reason}）"
            ),
        )
    return False, onnx_path, _fallback_reason("yolo_infer.dll", backend.detail or "未知原因")


def _run_yolo_once(
    engine: str,
    task: str,
    array,
    conf: float,
    nms: float,
    labels: str = "",
    source: str = "",
) -> tuple[dict, str | None]:
    """执行一次 YOLO 推理并补齐类别名与配色，返回 (结果, 降级说明)。

    抽出来的原因：单图接口与摄像头接口必须走**完全同一套**后端选择与
    输出结构。摄像头每秒调 10 次，如果它是"另一条实现"，两边框的顺序、
    字段名、降级行为迟早对不上，而屏幕上看起来都"能出框"，极难发现。
    """
    use_dll, onnx_path, warning = _select_yolo_backend(engine)

    label_info = labels_mod.load_labels(labels)
    try:
        if use_dll:
            result = inferproxy.run_yolo(engine, task, array, conf=conf, nms=nms, source=source)
        else:
            result = onnx_fallback.run_yolo(
                onnx_path, task, array, conf=conf, nms=nms,
                num_classes=len(label_info["classes"]) or None,
            )
    except onnx_fallback.FallbackError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _check_yolo_output_sane(result, int(array.shape[1]), int(array.shape[0]))
    infer.attach_labels(result, label_info["classes"])
    result["labels"] = {
        "path": labels or None,
        "ok": label_info["ok"],
        "name": label_info["name"],
        "count": len(label_info["classes"]),
        "error": label_info["error"],
    }
    return result, warning


@app.post("/api/infer/yolo")
def infer_yolo(
    image: UploadFile = File(...),
    engine: str = Form(...),
    task: str = Form("detect"),
    conf: float = Form(0.25),
    nms: float = Form(0.45),
    labels: str = Form(""),
) -> JSONResponse:
    """YOLO 推理：返回检测框、类别名与标注后的结果图（base64）。

    DLL 优先；DLL 不可用时自动降级到 onnxruntime，返回结构完全一致。
    `labels` 是标签文件路径，缺失时框上只有 `cls{id}`，不报错。
    """
    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels, source=image.filename or "")
        # 先补标签再画图：draw_yolo 要读 label_name 才能把类别名烧进去
        annotated = infer.draw_yolo(array, result["boxes"], result["class_colors"])
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()

    import base64

    result["annotated_png"] = base64.b64encode(annotated).decode("ascii")
    if warning:
        result["warning"] = warning
    return JSONResponse(result)


def _require_model_dir(raw: str, field: str) -> Path:
    """OCR 模型是目录，不是单个权重文件。"""
    if not str(raw or "").strip():
        raise HTTPException(status_code=400, detail=f"{field} 不能为空")
    path = _ensure_allowed_path(raw)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"{field} 不是目录：{path}")
    return path


@app.get("/api/ocr/models")
def list_ocr_models() -> dict:
    """扫描 OCR 模型根目录，标出格式和能不能在当前 Paddle 版本上用。"""
    try:
        return ocr_models.scan_models()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"扫描 OCR 模型失败：{exc}") from exc


@app.post("/api/infer/ocr")
def infer_ocr(
    image: UploadFile = File(...),
    det_dir: str = Form(...),
    rec_dir: str = Form(...),
    textline_ori_dir: str = Form(""),
    device: str = Form("cpu"),
    enable_mkldnn: int = Form(1),
    cpu_threads: int = Form(8),
    rec_batch_size: int = Form(6),
    det_thresh: float = Form(0.3),
    det_box_thresh: float = Form(0.6),
    det_unclip_ratio: float = Form(1.5),
    rec_score_thresh: float = Form(0.0),
) -> JSONResponse:
    """OCR：检测、识别、方向分类都在 ocr_infer.dll 里完成。"""
    det = _require_model_dir(det_dir, "检测模型目录")
    rec = _require_model_dir(rec_dir, "识别模型目录")
    textline = ""
    if str(textline_ori_dir or "").strip():
        textline = str(_require_model_dir(textline_ori_dir, "方向模型目录"))

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        result = inferproxy.run_ocr(array, source=image.filename or "", options={
            "det_dir": str(det),
            "rec_dir": str(rec),
            "textline_dir": textline,
            "device": device,
            "enable_mkldnn": 1 if enable_mkldnn else 0,
            "cpu_threads": cpu_threads,
            "rec_batch_size": rec_batch_size,
            "det_thresh": det_thresh,
            "det_box_thresh": det_box_thresh,
            "det_unclip_ratio": det_unclip_ratio,
            "rec_score_thresh": rec_score_thresh,
        })
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ocr_infer.OcrError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()
    return JSONResponse(result)


def _select_anomalib_backend(engine: str, metadata: str) -> tuple[bool, str, str | None]:
    """与 `_select_yolo_backend` 同样的三级顺序，只是主库换成 anomalib_infer。"""
    backend = inferproxy.status()

    if infer.is_onnx_model(engine):
        if backend.onnx_ok:
            return True, "", None
        path = Path(engine)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"ONNX 模型不存在：{engine}")
        return False, str(path), _fallback_reason(
            "onnx_infer.dll", backend.onnx_detail or "未知原因"
        )

    if backend.anomalib_ok:
        return True, "", None

    onnx_path, reason = _resolve_fallback_onnx(engine, metadata)
    if onnx_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "anomalib_infer.dll 不可用："
                f"{backend.detail}。请确认已编译 DLL，并设置 OPENCV_DIR / CUDA_PATH。"
                f"（也可改用 ONNX 降级，但{reason}）"
            ),
        )
    return False, onnx_path, _fallback_reason("anomalib_infer.dll", backend.detail or "未知原因")


@app.post("/api/infer/anomalib")
def infer_anomalib(
    image: UploadFile = File(...),
    engine: str = Form(...),
    metadata: str = Form(...),
    model_type: str = Form("Auto"),
    image_threshold: float = Form(-1.0),
    pixel_threshold: float = Form(-1.0),
    heatmap_colormap: str = Form("TURBO"),
) -> JSONResponse:
    """Anomalib 推理：返回分数、判定、叠加图、轮廓图、热能图与原图（base64）。

    阈值语义与 DLL 一致：`>= 0` 覆盖 metadata，`< 0` 沿用 metadata。
    响应里 `thresholds` 同时给出"本次生效值"与"metadata 文件原值"，
    因为 `ai_get_metadata` 拿到的已经是覆盖后的值，原值只能自己读文件。
    """
    use_dll, onnx_path, warning = _select_anomalib_backend(engine, metadata)

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        if use_dll:
            result = inferproxy.run_anomalib(
                engine, metadata, array,
                model_type=model_type,
                image_threshold=image_threshold,
                pixel_threshold=pixel_threshold,
                colormap=heatmap_colormap,
                source=image.filename or "",
            )
        else:
            try:
                result = onnx_fallback.run_anomalib(
                    onnx_path, metadata, array, model_type=model_type
                )
            except onnx_fallback.FallbackError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # 降级路径没有 C++ 侧的热能图渲染，这里补齐字段，保证前端只有一条渲染逻辑
            result.setdefault("heatmap_png", None)
            result.setdefault("source_png", None)
            result.setdefault("heatmap_colormap", None)
            result.setdefault("thresholds", {
                "image": {"effective": None, "metadata": None, "overridden": False},
                "pixel": {"effective": None, "metadata": None, "overridden": False},
            })
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()

    import base64

    for key in ("overlay_png", "mask_png", "heatmap_png", "source_png"):
        value = result.get(key)
        result[key] = base64.b64encode(value).decode("ascii") if value else None
    if warning:
        result["warning"] = warning
    return JSONResponse(result)


class BatchInferRequest(BaseModel):
    """目录批量推理请求。只接受服务端可访问的路径，不做文件上传。

    为什么不做上传：批量动辄几千张图、上百 MB，走 HTTP 上传等于把数据
    先拷一遍到临时目录再读一遍；前后端本来就在同一台机器上，路径更直接。
    """

    kind: str = Field(default="yolo", description="yolo / anomalib / ocr")
    engine: str = Field(default="", description=".engine / .trt / .onnx；OCR 留空")
    det_dir: str = Field(default="", description="OCR 检测模型目录")
    rec_dir: str = Field(default="", description="OCR 识别模型目录")
    textline_ori_dir: str = Field(default="", description="OCR 文本行方向模型目录，可留空")
    device: str = Field(default="cpu", description="cpu 或 gpu:0")
    enable_mkldnn: int = Field(default=1)
    cpu_threads: int = Field(default=8, ge=1, le=64)
    rec_batch_size: int = Field(default=6, ge=1, le=64)
    det_thresh: float = Field(default=0.3, ge=0.0, le=1.0)
    det_box_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    det_unclip_ratio: float = Field(default=1.5, ge=0.5, le=5.0)
    rec_score_thresh: float = Field(default=0.0, ge=0.0, le=1.0)
    task: str = Field(default="detect", description="detect / segment / pose / obb")
    labels: str = Field(default="", description="labels_*.json，可留空")
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    nms: float = Field(default=0.45, gt=0.0, le=1.0)
    source: str = Field(..., description="图片目录或单张图片")
    limit: int = Field(default=0, ge=0, description="只跑前 N 张，0 表示全部")
    recursive: bool = Field(default=True)
    jpeg_quality: int = Field(default=92, ge=60, le=100)
    make_zip: bool = Field(default=True)
    full_json: bool = Field(default=True, description="results.json 是否包含每个框")
    # Anomalib 专用
    metadata: str = Field(default="", description="anomalib metadata.json")
    model_type: str = Field(default="Auto")
    image_threshold: float = Field(default=-1.0)
    pixel_threshold: float = Field(default=-1.0)
    colormap: str = Field(default="TURBO")
    heatmap: bool = Field(default=False, description="额外保存热能图")


class VideoInferRequest(BaseModel):
    """视频推理请求。输出 mp4 落在产物目录里。"""

    kind: str = Field(default="yolo", description="只支持 yolo；ocr 会被拒绝")
    engine: str = Field(..., description=".engine / .trt / .onnx")
    task: str = Field(default="detect")
    labels: str = Field(default="")
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    nms: float = Field(default=0.45, gt=0.0, le=1.0)
    source: str = Field(..., description="视频文件")
    stride: int = Field(default=1, ge=1, le=600, description="每 N 帧处理一次")
    max_frames: int = Field(default=0, ge=0)
    max_side: int = Field(default=0, ge=0, description="输出长边上限，0 保持原尺寸")
    codec: str = Field(default="auto", description="auto / avc1 / H264 / mp4v")
    save_detections: bool = Field(default=False)


#: 视频容器扩展名。只挡明显选错文件的输入（比如把图片当视频提交）。
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".ts", ".m4v", ".wmv"}
ENGINE_SUFFIXES = {".engine", ".trt", ".plan", ".onnx"}


def _require_file(raw: str, field: str) -> Path:
    """校验「必须是已存在的文件且在允许目录内」，返回解析后的路径。"""
    if not str(raw or "").strip():
        raise HTTPException(status_code=400, detail=f"{field} 不能为空")
    path = _ensure_allowed_path(raw)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"{field} 不存在：{path}")
    return path


@app.get("/api/infer/anomalib/metadata")
def inspect_anomaly_metadata(path: str = "") -> dict:
    """只读 `metadata.json`，把里面的推荐阈值提前暴露给推理页。

    为什么需要这个接口：勾选「沿用 metadata 默认阈值」时两个阈值输入框是
    只读的，值来自 metadata 文件——但文件内容在界面上没有任何出口，用户只能
    先跑一次推理、再去看结果行里的"（metadata）"才间接看到数字，等于把推荐值
    藏起来了。这个接口只读文件、不建引擎句柄，可以随路径改动随时刷新。
    """
    target = _require_file(path, "metadata.json")
    info = infer.read_metadata_thresholds(str(target))
    if info.get("error"):
        raise HTTPException(status_code=400, detail=info["error"])
    return {"path": str(target), **info}


def build_batch_command(
    request: "BatchInferRequest",
    python: str,
    job_dir: str | Path,
) -> list[str]:
    """组装批量推理命令。

    单独成函数是为了能被测试直接断言：参数在表单里填了却没进命令行，
    用户只能从日志里翻完整命令才能发现（`--simplify true` 就是这么漏的）。
    """
    if request.kind not in ("yolo", "anomalib", "ocr"):
        raise HTTPException(status_code=400, detail=f"kind 只能是 yolo / anomalib / ocr，实际 {request.kind}")

    if request.kind == "ocr":
        det = _require_model_dir(request.det_dir, "检测模型目录")
        rec = _require_model_dir(request.rec_dir, "识别模型目录")
        source = _ensure_allowed_path(request.source)
        if not source.exists():
            raise HTTPException(status_code=400, detail=f"数据源不存在：{source}")
        command = config.tool_command(python, "infer_batch.py") + [
            "--kind", "ocr",
            "--det-dir", str(det),
            "--rec-dir", str(rec),
            "--device", request.device,
            "--cpu-threads", str(int(request.cpu_threads)),
            "--rec-batch-size", str(int(request.rec_batch_size)),
            "--det-thresh", f"{request.det_thresh:g}",
            "--det-box-thresh", f"{request.det_box_thresh:g}",
            "--det-unclip-ratio", f"{request.det_unclip_ratio:g}",
            "--rec-score-thresh", f"{request.rec_score_thresh:g}",
            "--source", str(source),
            "--output", str(job_dir),
            "--limit", str(int(request.limit)),
            "--jpeg-quality", str(int(request.jpeg_quality)),
        ]
        if str(request.textline_ori_dir or "").strip():
            textline = _require_model_dir(request.textline_ori_dir, "方向模型目录")
            command += ["--textline-dir", str(textline)]
        if not request.enable_mkldnn:
            command.append("--no-mkldnn")
        if not request.recursive:
            command.append("--no-recursive")
        if not request.make_zip:
            command.append("--no-zip")
        if not request.full_json:
            command.append("--no-full-json")
        return command

    engine = _require_file(request.engine, "引擎文件")
    if engine.suffix.lower() not in ENGINE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"引擎需为 {', '.join(sorted(ENGINE_SUFFIXES))} 之一，实际 {engine.suffix}",
        )
    source = _ensure_allowed_path(request.source)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"数据源不存在：{source}")
    if request.labels.strip():
        _require_file(request.labels, "标签文件")

    command = config.tool_command(python, "infer_batch.py") + [
        "--kind", request.kind,
        "--engine", str(engine),
        "--source", str(source),
        "--output", str(job_dir),
        "--limit", str(int(request.limit)),
        "--jpeg-quality", str(int(request.jpeg_quality)),
    ]

    if request.kind == "yolo":
        if request.task not in ("detect", "segment", "pose", "obb"):
            raise HTTPException(status_code=400, detail=f"不支持的任务类型：{request.task}")
        command += [
            "--task", request.task,
            "--conf", f"{request.conf:g}",
            "--nms", f"{request.nms:g}",
        ]
    else:
        metadata = _require_file(request.metadata, "metadata.json")
        command += [
            "--metadata", str(metadata),
            "--model-type", request.model_type,
            "--image-threshold", f"{request.image_threshold:g}",
            "--pixel-threshold", f"{request.pixel_threshold:g}",
            "--colormap", request.colormap,
        ]
        if request.heatmap:
            command.append("--heatmap")

    if request.labels.strip():
        command += ["--labels", str(Path(request.labels).expanduser())]
    if not request.recursive:
        command.append("--no-recursive")
    if not request.make_zip:
        command.append("--no-zip")
    if not request.full_json:
        command.append("--no-full-json")
    return command


def build_video_command(
    request: "VideoInferRequest",
    python: str,
    output_path: str | Path,
) -> list[str]:
    """组装视频推理命令。校验规则与 bulk 接口保持一致。"""
    if request.kind == "ocr":
        raise HTTPException(status_code=400, detail="OCR 不支持视频推理：CPU 单张约 10 秒以上，请改用目录批量")
    engine = _require_file(request.engine, "引擎文件")
    if engine.suffix.lower() not in ENGINE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"引擎格式不支持：{engine.suffix}")
    if request.task not in ("detect", "segment", "pose", "obb"):
        raise HTTPException(status_code=400, detail=f"不支持的任务类型：{request.task}")

    source = _require_file(request.source, "视频文件")
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"视频需为 {', '.join(sorted(VIDEO_SUFFIXES))} 之一，实际 {source.suffix}",
        )
    if request.labels.strip():
        _require_file(request.labels, "标签文件")
    if request.codec not in ("auto", "avc1", "H264", "mp4v"):
        raise HTTPException(status_code=400, detail=f"不支持的编码器：{request.codec}")

    command = config.tool_command(python, "infer_video.py") + [
        "--engine", str(engine),
        "--task", request.task,
        "--source", str(source),
        "--output", str(output_path),
        "--conf", f"{request.conf:g}",
        "--nms", f"{request.nms:g}",
        "--stride", str(int(request.stride)),
        "--max-frames", str(int(request.max_frames)),
        "--max-side", str(int(request.max_side)),
        "--codec", request.codec,
    ]
    if request.labels.strip():
        command += ["--labels", str(Path(request.labels).expanduser())]
    if request.save_detections:
        command.append("--save-detections")
    return command


@app.post("/api/infer/batch")
def infer_batch(request: BatchInferRequest) -> dict:
    """启动目录批量推理，返回 run_id；进度经 `/ws/runs/{run_id}` 推送。"""
    config.ensure_dirs()
    try:
        job = inferjobs.new_job_dir("batch")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"创建产物目录失败：{exc}") from exc

    command = build_batch_command(request, sys.executable, job)
    expect = (job / "annotated.zip") if request.make_zip else (job / "results.csv")

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = manager.create(
                "infer-batch",
                command,
                cwd=str(BACKEND_DIR),
                expect_file=expect,
                meta={
                    "kind": request.kind,
                    "source": str(Path(request.source).expanduser()),
                    "engine": str(Path(request.engine).expanduser()),
                    "output": str(job),
                },
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动批量推理失败：{exc}") from exc

    return {
        **run.snapshot(),
        "output": str(job),
        "expect": str(expect),
        "artifact_url": f"/api/infer/artifact?path={expect.as_posix()}&download=1",
    }


@app.post("/api/infer/video")
def infer_video(request: VideoInferRequest) -> dict:
    """启动视频推理，产物是一个带框的 mp4。"""
    config.ensure_dirs()
    try:
        job = inferjobs.new_job_dir("video")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"创建产物目录失败：{exc}") from exc

    source = Path(request.source).expanduser()
    output = job / f"{inferjobs.safe_stem(source.stem)}_annotated.mp4"
    command = build_video_command(request, sys.executable, output)

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = manager.create(
                "infer-video",
                command,
                cwd=str(BACKEND_DIR),
                expect_file=output,
                meta={
                    "source": str(source),
                    "engine": str(Path(request.engine).expanduser()),
                    "stride": int(request.stride),
                    "output": str(job),
                },
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动视频推理失败：{exc}") from exc

    return {
        **run.snapshot(),
        "output": str(job),
        "video": str(output),
        "artifact_url": f"/api/infer/artifact?path={output.as_posix()}",
        "download_url": f"/api/infer/artifact?path={output.as_posix()}&download=1",
    }


@app.post("/api/infer/frame")
def infer_frame(
    image: UploadFile = File(...),
    engine: str = Form(...),
    task: str = Form("detect"),
    conf: float = Form(0.25),
    nms: float = Form(0.45),
    labels: str = Form(""),
) -> JSONResponse:
    """摄像头单帧推理：只回检测框，不回图片。

    与 `/api/infer/yolo` 的差别就是「不烧图」：摄像头每帧把标注图 base64
    回传，1080p 一帧约 400KB，10fps 就是 4MB/s，浏览器解码也会卡。
    前端拿到框自己在 canvas 上画，既省带宽也不掉帧。
    """
    data = image.file.read()
    frame_limit = 8 * 1024 * 1024
    if len(data) > frame_limit:
        raise HTTPException(
            status_code=413,
            detail=f"单帧超过 {frame_limit // (1024 * 1024)} MB，请调小采集分辨率",
        )

    try:
        array = infer.decode_image(data)
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels, source="摄像头帧")
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.pop("annotated_png", None)
    if warning:
        result["warning"] = warning
    return JSONResponse(result)


@app.get("/api/infer/artifact")
def infer_artifact(path: str, download: bool = False) -> FileResponse:
    """下载/播放大批量或视频任务的产物。

    只允许读 `runtime/infer` 下的文件：这是 Web 端最容易变成
    「读任意磁盘」后门的地方，白名单必须收到产物目录，而不是仓库根。
    `download=True` 时带 Content-Disposition，方便另存为；
    默认不带，`<video>` 标签才能直接在页面里播放。
    """
    try:
        target = inferjobs.resolve_artifact(path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"产物不存在：{target}")

    return FileResponse(
        target,
        media_type=inferjobs.guess_media_type(target),
        filename=target.name if download else None,
    )


@app.get("/api/models")
def list_models(limit: int = models_mod.MAX_ENTRIES) -> dict:
    """枚举训练产物目录里的可用引擎，供推理页的模型下拉使用。

    只读目录、只取文件属性，不读取引擎或 ONNX 内容——
    一个 PatchCore 目录就有 275MB 的引擎加 300MB 的 ONNX，读一遍会很慢。
    """
    try:
        return models_mod.scan_models(limit=max(1, min(int(limit), models_mod.MAX_ENTRIES)))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"扫描训练产物失败：{exc}") from exc


@app.get("/api/infer/cache")
def infer_cache() -> dict:
    """查看当前推理句柄缓存（只读诊断）。

    存在的理由：句柄常驻显存，用户改阈值改到显存告急时需要能看见"现在到底留了几个"。
    没有这个窗口，用户只能靠猜。
    """
    stats = inferproxy.cache_stats()
    return {"ok": True, **stats, "ocr": stats.get("ocr", {}), "worker": inferproxy.worker_state()}


@app.post("/api/infer/worker/restart")
def infer_worker_restart() -> dict:
    """熔断后由用户手动重启推理进程。"""
    return {"ok": True, **inferproxy.restart_worker()}


# ---------------------------------------------------------------------------
# 标注工作台
# ---------------------------------------------------------------------------
class AnnotationProjectRequest(BaseModel):
    """保存标注工程的请求。

    ``rev`` 是客户端读到的修订号，服务端拿它做乐观锁；
    ``stamp`` 是文件指纹（mtime + size），用来发现"有人用 Qt 或记事本改过文件"。
    两者都为 ``None`` 且 ``force=False`` 时按"新建或强制写"处理。
    """

    path: str = Field(..., description=".tfanno 工程文件路径")
    project: dict = Field(..., description="完整工程内容")
    rev: int | None = Field(default=None, description="客户端读到的修订号")
    stamp: dict | None = Field(default=None, description="客户端读到的文件指纹")
    force: bool = Field(default=False, description="跳过修订号校验，强制覆盖")


class AnnotationImportRequest(BaseModel):
    """把 YOLO 数据集（data.yaml + images/ + labels/）导入标注工程。

    ``dry_run=True`` 只返回报告，用来让用户先核对「几张图、几个框、有哪些警告」；
    ``mode="merge"`` 时必须带上 ``project``（当前内存中的工程，可能还没保存），
    否则会读到磁盘上的旧版本，把用户刚画的框覆盖掉。
    """

    source: str = Field(..., description="data.yaml 或数据集根目录")
    project_path: str = Field(default="", description="目标 .tfanno 路径，用于算相对路径")
    project: dict | None = Field(default=None, description="合并模式下的当前工程内容")
    splits: list[str] | None = Field(default=None, description="要导入的 split，留空表示全部")
    mode: str = Field(default="new", description="new 新建工程 / merge 合并到当前工程")
    duplicate: str = Field(default="skip", description="同名图片：skip / replace / append")
    task: str = Field(default="auto", description="auto 或 detect/segment/pose/obb")
    limit: int = Field(default=5000, ge=1, le=20000, description="单次导入图片上限")
    dry_run: bool = Field(default=False, description="只出报告，不返回工程内容")


class AnnotationExportRequest(BaseModel):
    """从标注工程导出训练数据集。"""

    project: dict = Field(..., description="内存中的完整工程内容")
    source: str = Field(default="", description="工程文件路径，用于输出目录防呆")
    format: str = Field(default="yolo", description="yolo / coco / voc")
    output: str = Field(..., description="导出目标目录")
    overwrite: bool = Field(default=False, description="目标目录非空时是否允许覆盖")


class AnnotationAutoLabelRequest(BaseModel):
    """预标注请求。``service`` 留空则只走自有推理通道。"""

    image: str = Field(..., description="当前图片的绝对路径")
    task: str = Field(default="detect")
    engine: str = Field(..., description=".engine 路径")
    labels: str = Field(default="", description="labels_*.json 路径")
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    nms: float = Field(default=0.45, ge=0.0, le=1.0)
    service: str = Field(default="", description="外部标注服务基址，如 http://127.0.0.1:8500")
    prompts: dict | None = Field(default=None, description="外部服务的提示（文本/点/框）")
    timeout: float = Field(default=20.0, ge=1.0, le=300.0)


class AnnotationDraftsRequest(BaseModel):
    """写旁挂草稿。

    ``images`` 是**增量**语义：只覆盖传进来的图片，其余图片在磁盘上的草稿保持不变。
    这是刻意的——前端自动保存只会带上被改过的那几张图，批量任务又可能同时在写，
    整份替换会把对方刚写的内容抹掉。

    ``review`` 同样是增量：`{图片 id: {"state": "checked"|"unchecked"}}`。
    审核和草稿一起提交是常态（按 K 的同时草稿也在变），分成两个请求就会出现
    "草稿写了、状态丢了"的中间态。
    """

    path: str = Field(..., description=".tfanno 工程路径（草稿写在其旁的 *.drafts.json）")
    images: dict[str, list] | None = Field(default=None, description="图片 id → 草稿数组（省略表示不碰草稿）")
    review: dict[str, dict | str] | None = Field(
        default=None, description="图片 id → {state, at_ns}；state=unchecked 表示取消"
    )
    replace: bool = Field(default=False, description="true 时整份替换（用于清空全部草稿）")


class AnnotationPromptLabelRequest(BaseModel):
    """点 / 框提示式预标注（SAM 通道）。

    `points` / `labels` / `box` 都允许传 `null`：`promptlabel.annotate()` 本来就把
    `None` 当空处理，而 pydantic 默认会拒掉 `null` 并回 422——那样用户看到的是
    `Input should be a valid list`，而不是后端精心写的那句
    「请先在画布上点一个点或拉一个框」。当前前端发的是 `[]`，但契约不应该靠这个巧合。
    """

    image: str = Field(..., description="当前图片绝对路径")
    points: list[list[float]] | None = Field(default=None, description="提示点 [[x, y], ...]")
    labels: list[int] | None = Field(default=None, description="每点的标签：1 正 / 0 负；缺省全按正点")
    box: list[list[float]] | None = Field(default=None, description="框提示 [[x1,y1],[x2,y2]]")
    epsilon: float = Field(default=0.001, ge=0.0, le=0.1, description="多边形近似系数（相对周长）")
    label: str = Field(default="", description="草稿类别名")
    class_id: int = Field(default=0, ge=0)


class AnnotationTextLabelRequest(BaseModel):
    """文本提示预标注（GroundingDINO 通道，可选 SAM 细化）。"""

    image: str = Field(..., description="当前图片绝对路径")
    prompt: str = Field(..., description="文本提示，如 missing hole")
    box_thr: float = Field(default=0.3, ge=0.0, le=1.0)
    text_thr: float = Field(default=0.25, ge=0.0, le=1.0)
    refine: bool = Field(default=False, description="是否再用 SAM 把框细化成多边形")
    epsilon: float = Field(default=0.001, ge=0.0, le=0.1)
    label: str = Field(default="", description="类别名（留空用模型解出的短语）")
    class_id: int = Field(default=0, ge=0)


class ModelDownloadRequest(BaseModel):
    """触发一次模型下载。"""

    id: str = Field(..., description="模型 id，见 GET /api/modelzoo")


class AnnotationAutoLabelBatchRequest(BaseModel):
    """整工程批量预标注。

    ``images`` 由前端按当前内存工程整理（含 ``annotated`` 标记），
    过滤与白名单校验在服务端做：前端不该成为"哪些图片允许跑"的判据。
    """

    path: str = Field(..., description=".tfanno 工程路径，草稿写入其旁挂文件")
    images: list[dict] = Field(..., description="[{id, path, annotated?}] 待处理图片")
    mode: str = Field(default="engine", description="engine 自有引擎 / text 文本提示")
    task: str = Field(default="detect")
    engine: str = Field(default="", description=".engine 路径（mode=engine 必填）")
    labels: str = Field(default="", description="labels_*.json 路径")
    prompt: str = Field(default="", description="mode=text 时的文本提示")
    box_thr: float = Field(default=0.3, ge=0.0, le=1.0)
    text_thr: float = Field(default=0.25, ge=0.0, le=1.0)
    refine: bool = Field(default=False, description="text 模式下再用 SAM 细化成多边形")
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    nms: float = Field(default=0.45, ge=0.0, le=1.0)
    service: str = Field(default="")
    timeout: float = Field(default=20.0, ge=1.0, le=300.0)
    scope: str = Field(default="all", description="all 全部 / pending 跳过已标注或已有草稿")
    limit: int = Field(default=20000, ge=1, le=50000, description="单批图片上限")


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


@app.get("/api/annotation/project")
def annotation_load(path: str) -> dict:
    """读取 `.tfanno` 工程。

    返回三块内容：规范化后的工程、文件指纹（供保存时回传）、图片列表统计。
    ``exists=False`` 时工程字段是一份可直接用的空白骨架，前端据此进入"新建"流程。
    """
    target = _ensure_allowed_path(path)
    if not target.is_file():
        # 文件不存在不是错误：前端要拿到一份可直接编辑的骨架，点「新建」再落盘。
        blank = annotation.new_project()
        return {
            "ok": True,
            "exists": False,
            "path": str(target),
            "project": blank,
            "meta": {"rev": -1, "mtime_ns": "0", "size": 0, "warnings": []},
            "stats": annotation.stats(blank),
        }

    try:
        project, meta = annotation.load(target)
    except annotation.AnnotationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "exists": True,
        "path": str(target),
        # resolved：相对路径（Qt 写法）解析成绝对路径给浏览器用；
        # 保存时会被 normalize_project 丢掉，不会写进 .tfanno。
        "project": annotation.with_resolved_paths(project, target),
        "meta": meta,
        "stats": annotation.stats(project),
    }


@app.post("/api/annotation/project")
def annotation_save(request: AnnotationProjectRequest) -> dict:
    """保存工程。修订号不一致返回 **409** 并附上磁盘真实状态。

    409 而不是 400：前端要据此弹"冲突"对话框，而不是显示一条普通错误。
    """
    target = _ensure_allowed_path(request.path)
    if target.suffix.lower() != ".tfanno":
        raise HTTPException(status_code=400, detail="标注工程文件扩展名必须是 .tfanno")

    try:
        meta = annotation.save(
            target,
            request.project,
            expected_rev=request.rev,
            expected_stamp=request.stamp,
            force=request.force,
        )
    except annotation.RevisionConflict as exc:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": "revision-conflict",
                "reason": exc.reason,
                "message": str(exc),
                "disk": exc.disk,
                "expected": exc.expected,
            },
        )
    except annotation.AnnotationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入工程失败：{exc}") from exc

    project, _ = annotation.load(target)
    return {
        "ok": True,
        "path": str(target),
        "rev": meta["rev"],
        "meta": meta,
        "stats": annotation.stats(project),
        "warnings": meta.get("warnings", []),
    }


@app.get("/api/annotation/image")
def annotation_image(path: str) -> FileResponse:
    """图片流。

    单独开接口而不是挂 `/static`：数据集在仓库任意位置，挂载点无法覆盖，
    而且挂载目录等于把整个目录树暴露成静态资源。
    """
    target = _ensure_allowed_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在：{target}")

    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"不是图片文件：{target}")
    return FileResponse(target, media_type=media_type)


@app.get("/api/annotation/image-meta")
def annotation_image_meta(path: str) -> dict:
    """探测单张图片的尺寸，供"添加图片到工程"时填 width/height。"""
    target = _ensure_allowed_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在：{target}")
    width, height = dataset._image_size(target)
    return {"ok": True, "path": str(target), "width": width, "height": height}


@app.post("/api/annotation/scan-images")
def annotation_scan_images(root: str, recursive: bool = True, limit: int = 5000) -> dict:
    """扫描目录里的图片，返回路径与尺寸，供前端批量加入工程。

    只认常见图片后缀，且跳过 `.tfanno` 的临时文件，避免把导出产物也收进来。
    """
    base = _ensure_allowed_path(root)
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录：{base}")

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    pattern = "**/*" if recursive else "*"
    found: list[dict] = []
    truncated = False
    for item in base.glob(pattern):
        if len(found) >= limit:
            # 命中上限就停：truncated 必须由「还有没被收录的文件」推出，
            # 不能用 len(found) >= limit 判断，否则正好 limit 张时会误报截断。
            truncated = True
            break
        if not item.is_file() or item.suffix.lower() not in suffixes:
            continue
        width, height = dataset._image_size(item)
        found.append({"path": str(item), "width": width, "height": height})

    found.sort(key=lambda entry: entry["path"].lower())
    return {"ok": True, "root": str(base), "count": len(found), "images": found, "truncated": truncated}


@app.post("/api/annotation/import")
def annotation_import(request: AnnotationImportRequest) -> dict:
    """把已有 YOLO 标签导入标注工程（不落盘）。

    导入只是把 txt 变成工程里的普通标注对象：返回的 ``project`` 交给前端接管后，
    用户照常拖框、改类、按 Ctrl+S 存成 .tfanno。之所以不在后端直接写盘：
    合并模式要拿到「内存里还没保存的工程」，后端写盘会覆盖用户刚画的框。
    """
    source = _ensure_allowed_path(request.source)
    project_path = str(_ensure_allowed_path(request.project_path)) if request.project_path.strip() else ""

    try:
        project, report = annotationimport.build(
            source,
            project_path=project_path,
            base_project=request.project,
            splits=request.splits,
            mode=request.mode,
            duplicate=request.duplicate,
            task=request.task,
            limit=request.limit,
        )
    except annotation.AnnotationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"读取数据源失败：{exc}") from exc

    payload: dict = {"ok": True, "path": project_path, "report": report}
    if request.dry_run:
        # 预览也要跑一次校验：像「图片 id 重复」这种问题必须在导入前暴露，
        # 否则用户要等到导出时才看见红字。
        payload["issues"] = annotation.validate(project)
        return payload

    normalized, warnings = annotation.normalize_project(project)
    payload["project"] = annotation.with_resolved_paths(normalized, project_path)
    payload["issues"] = annotation.validate(normalized)
    payload["warnings"] = warnings
    return payload


@app.post("/api/annotation/export")
def annotation_export(request: AnnotationExportRequest) -> dict:
    """把内存中的工程导出成 YOLO / COCO / VOC。

    走 `annotation.to_dataset_project` + `dataset.export_*`，
    既不另写一套导出，也不需要先把工程落盘再读回来（那样会丢掉未保存的编辑）。
    """
    target = _ensure_allowed_path(request.output)
    if request.format not in dataset.FORMATS:
        raise HTTPException(status_code=400, detail=f"未知导出格式：{request.format}")

    issues = annotation.validate(request.project)
    project = annotation.to_dataset_project(request.project, source=request.source or None)
    # 导出沿用工程里已经标好的集合划分，不重新按比例分配——
    # 标注页的划分是用户逐个设过的，重算等于把他的选择抹掉。
    splits = [image.current_split for image in project.images]

    try:
        if request.format == "yolo":
            result = dataset.export_yolo(project, target, splits, request.overwrite)
        elif request.format == "coco":
            result = dataset.export_coco(project, target, splits, request.overwrite)
        else:
            result = dataset.export_voc(project, target, splits, request.overwrite)
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"导出失败：{exc}") from exc

    result.update(
        {
            "task": project.task,
            "source_kind": "tfanno",
            "split_counts": dataset.split_counts(splits),
            "issues": issues,
            "warnings": project.warnings,
        }
    )
    return result


@app.post("/api/annotation/autolabel")
def annotation_autolabel(request: AnnotationAutoLabelRequest) -> dict:
    """用自有引擎（必要时加外部服务）为当前图片生成预标注草稿。

    草稿只存在会话内，绝不自动写盘——误检直接落盘等于把错误固化进工程。
    """
    image = _ensure_allowed_path(request.image)
    engine = _ensure_allowed_path(request.engine)
    if not image.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在：{image}")
    if not engine.is_file():
        raise HTTPException(status_code=404, detail=f"引擎不存在：{engine}")

    labels_path = ""
    if request.labels.strip():
        labels_path = str(_ensure_allowed_path(request.labels))

    try:
        return autolabel.run(
            image_path=str(image),
            task=request.task,
            engine=str(engine),
            labels_path=labels_path,
            conf=request.conf,
            nms=request.nms,
            service=request.service,
            prompts=request.prompts,
            timeout=request.timeout,
        )
    except autolabel.AutoLabelError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": exc.kind, "message": str(exc)})
    except infer.InferError as exc:
        raise HTTPException(status_code=503, detail=f"推理失败：{exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"预标注失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 草稿旁挂持久化
# ---------------------------------------------------------------------------
def _drafts_target(path: str) -> Path:
    """校验草稿归属的工程路径。

    只接受 `.tfanno`：草稿路径是从工程路径推导出来的，允许任意路径等于让
    这个接口变成"往磁盘任意位置写 JSON"的入口。
    """
    target = _ensure_allowed_path(path)
    if target.suffix.lower() != ".tfanno":
        raise HTTPException(status_code=400, detail="草稿只跟随 .tfanno 工程文件")
    return target


@app.get("/api/annotation/drafts")
def annotation_load_drafts(path: str) -> dict:
    """读取预标注草稿（`<工程名>.drafts.json`）。

    文件不存在返回空草稿池而不是 404：前端打开老工程时会无条件拉一次，
    没有草稿是正常状态，不该在控制台刷一片红。
    """
    target = _drafts_target(path)
    try:
        images, meta = annotation_drafts.load(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"读取草稿失败：{exc}") from exc
    return {
        "ok": True,
        "project": str(target),
        "drafts_path": meta.pop("path", ""),
        # 字段名用 drafts 而不是 images：`meta.image_count` 是图片张数，
        # 这里装的是「图片 id → 草稿数组」，同名会读错。
        "drafts": images,
        **meta,
    }


@app.post("/api/annotation/drafts")
def annotation_save_drafts(request: AnnotationDraftsRequest) -> dict:
    """写草稿与审核状态（默认增量，`replace=true` 整份替换）。

    `images` 与 `review` 都省略时是"只读一次返回"的语义，等价于一次空合并——
    不报错，让前端可以放心地"总是提交一次"而不用先判断有没有改动。
    """
    target = _drafts_target(request.path)
    try:
        if request.replace:
            meta = annotation_drafts.save(
                target, request.images, review=request.review, replace=True
            )
        else:
            meta = annotation_drafts.merge(target, request.images, review=request.review)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入草稿失败：{exc}") from exc
    return {"ok": True, "project": str(target), "drafts_path": meta.pop("path", ""), **meta}


@app.post("/api/annotation/drafts/review")
def annotation_set_review(request: AnnotationDraftsRequest) -> dict:
    """只改审核状态（`K` / `Shift+K` 的专用入口）。

    单独一个路径而不是复用 POST drafts：审核是高频操作（连续按 K），
    请求体只有 `{path, review}`，不用带上草稿，也就不存在"自动保存还没跑完，
    审核的请求先把草稿覆盖成空"的竞态。
    """
    if not request.review:
        raise HTTPException(status_code=400, detail="review 不能为空")
    target = _drafts_target(request.path)
    try:
        meta = annotation_drafts.merge(target, review=request.review)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入审核状态失败：{exc}") from exc
    return {"ok": True, "project": str(target), "drafts_path": meta.pop("path", ""), **meta}


# ---------------------------------------------------------------------------
# 提示式预标注（点 / 框 / 文本）
# ---------------------------------------------------------------------------
def _prompt_image(path: str) -> Path:
    """校验图片路径。只接受配置里允许的图片后缀：这条路会读图做推理，
    放开任意文件等于给了一个"让服务解码任意文件"的入口。"""
    target = _ensure_allowed_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在：{target}")
    if target.suffix.lower() not in config.IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式：{target.suffix}")
    return target


def _prompt_error(exc: Exception) -> HTTPException:
    """把三条通道的业务错误统一映射成 400，并带上"该怎么办"。

    模型缺失是最常见的失败（首次使用没下载），它的错误文本里已经带了下载指引，
    所以直接透传 message；其余异常也照抄，不做二次措辞——现场排查时
    原始信息比包装过的话有用。
    """
    return HTTPException(status_code=400, detail=str(exc))


@app.post("/api/annotation/prompt-label")
def annotation_prompt_label(request: AnnotationPromptLabelRequest) -> dict:
    """点 / 框 → SAM → 多边形草稿。

    **不落盘**：草稿由前端接收后进内存草稿池，再由前端的防抖自动保存统一写盘。
    接口自己写盘会让"用户反悔、不想要这批草稿"变成一次磁盘删除操作。
    """
    target = _prompt_image(request.image)
    if not request.points and not request.box:
        raise HTTPException(status_code=400, detail="请先在画布上点一个点或拉一个框")
    try:
        drafts = promptlabel.annotate(
            target,
            points=request.points or [],
            labels=request.labels or [],
            box=request.box or [],
            epsilon=request.epsilon,
            label=request.label,
            class_id=request.class_id,
        )
    except promptlabel.PromptLabelError as exc:
        raise _prompt_error(exc) from exc
    except modelzoo.ModelMissing as exc:
        raise _prompt_error(exc) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"提示式预标注失败：{exc}") from exc

    warnings = []
    if not drafts:
        warnings.append("没有生成掩码：可以换一个位置再点一次，或把框拉大一些把目标整个包住")
    return {
        "ok": True,
        "channel": "sam",
        "drafts": drafts,
        "count": len(drafts),
        "warnings": warnings,
    }


@app.post("/api/annotation/text-label")
def annotation_text_label(request: AnnotationTextLabelRequest) -> dict:
    """文本提示 → 检出框（可选细化成多边形）。"""
    target = _prompt_image(request.image)
    if not str(request.prompt or "").strip():
        raise HTTPException(status_code=400, detail="文本提示不能为空")
    try:
        result = textlabel.annotate(
            target,
            request.prompt,
            box_thr=request.box_thr,
            text_thr=request.text_thr,
            refine=request.refine,
            epsilon=request.epsilon,
            label=request.label,
            class_id=request.class_id,
        )
    except textlabel.TextLabelError as exc:
        raise _prompt_error(exc) from exc
    except modelzoo.ModelMissing as exc:
        raise _prompt_error(exc) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"文本预标注失败：{exc}") from exc

    return {"ok": True, "channel": "text", **result}


# ---------------------------------------------------------------------------
# 模型仓库
# ---------------------------------------------------------------------------
@app.get("/api/modelzoo")
def modelzoo_list() -> dict:
    """内置模型清单与本地状态（含正在进行的下载任务）。"""
    return {"ok": True, **modelzoo.list_models(), "jobs": modelzoo.recent()}


@app.post("/api/modelzoo/download")
def modelzoo_download(request: ModelDownloadRequest) -> dict:
    """开始下载（幂等：已就绪返回 `already=True`，正在下返回 400 说明原因）。"""
    try:
        job = modelzoo.start_download(request.id)
    except modelzoo.DownloadBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except modelzoo.ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "job": job}


@app.get("/api/modelzoo/download/{job_id}")
def modelzoo_progress(job_id: str) -> dict:
    job = modelzoo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知下载任务：{job_id}（服务可能重启过）")
    return {"ok": True, "job": job}


@app.post("/api/modelzoo/download/{job_id}/cancel")
def modelzoo_cancel(job_id: str) -> dict:
    job = modelzoo.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知下载任务：{job_id}")
    return {"ok": True, "job": job}


# ---------------------------------------------------------------------------
# 批量预标注
# ---------------------------------------------------------------------------
def _batch_select_images(
    request: AnnotationAutoLabelBatchRequest,
    known_drafts: dict,
    reviewed: set[str] | None = None,
) -> tuple[list[dict], int, list[str]]:
    """把请求里的图片清单过滤成"真正要跑的"。

    返回 ``(items, skipped, reasons)``。跳过原因要带回去：用户看到"跳过 12 张"
    却不知道为什么，第一反应是程序坏了。

    `scope=pending` 跳过"已标注 / 已有草稿 / 已审核"三类：审核过的图片用户已经
    确认过结论，再跑一遍会把人工确认的结果换成新的候选，等于把做过的工作抹掉。
    """
    items: list[dict] = []
    skipped = 0
    reasons: list[str] = []
    seen: set[str] = set()
    reviewed = reviewed or set()

    for raw in request.images:
        if len(items) >= request.limit:
            reasons.append(f"超过单批上限 {request.limit} 张，其余图片未处理")
            break
        if not isinstance(raw, dict):
            skipped += 1
            continue
        image_id = str(raw.get("id") or "").strip()
        image_path = str(raw.get("path") or "").strip()
        if not image_id or not image_path:
            skipped += 1
            continue
        if image_id in seen:
            skipped += 1
            continue
        seen.add(image_id)

        if request.scope == "pending" and (
            raw.get("annotated") or known_drafts.get(image_id) or image_id in reviewed
        ):
            skipped += 1
            continue

        try:
            resolved = _ensure_allowed_path(image_path)
        except HTTPException as exc:
            skipped += 1
            if len(reasons) < 10:
                reasons.append(f"{image_id}：{exc.detail}")
            continue
        if not resolved.is_file():
            skipped += 1
            if len(reasons) < 10:
                reasons.append(f"{image_id}：图片不存在（{resolved}）")
            continue
        items.append({"id": image_id, "path": str(resolved)})

    return items, skipped, reasons


@app.post("/api/annotation/autolabel/batch")
def annotation_autolabel_batch(request: AnnotationAutoLabelBatchRequest) -> dict:
    """启动整工程批量预标注，返回任务快照；进度经 `GET .../batch/{job_id}` 轮询。

    结果直接写进旁挂草稿文件，**不经过响应体**：几百张图的草稿走 HTTP 传一遍
    既慢又容易在超时处丢一半，写盘还能让"中途崩溃"最多损失当前这一张。

    两条通道（`mode`）：
    - `engine`：逐个跑自训 `.engine`，必须给引擎路径；
    - `text`：逐个跑文本提示，必须给 prompt，且**启动前**先确认模型就绪——
      否则会建一个"跑 300 张全失败"的空任务，用户等到最后才看到 300 条错误。
    """
    target = _drafts_target(request.path)
    mode = str(request.mode or "engine").strip().lower()
    if mode not in ("engine", "text"):
        raise HTTPException(status_code=400, detail=f"不支持的批量模式：{mode}（engine / text）")

    engine = ""
    labels_path = ""
    prompt = ""
    if mode == "engine":
        if not str(request.engine or "").strip():
            raise HTTPException(status_code=400, detail="引擎通道需要先选择 .engine 文件")
        engine_path = _ensure_allowed_path(request.engine)
        if not engine_path.is_file():
            raise HTTPException(status_code=404, detail=f"引擎不存在：{engine_path}")
        engine = str(engine_path)
        if request.labels.strip():
            labels_path = str(_ensure_allowed_path(request.labels))
    else:
        prompt = str(request.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="文本通道需要先填写提示词")
        # 提前探一次模型：缺文件在这里就报 400，而不是等 300 张图跑完
        try:
            textlabel.get_session()
        except textlabel.ModelMissing as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"文本模型无法加载：{exc}") from exc

    known_drafts, known_meta = annotation_drafts.load(target)
    reviewed = set((known_meta.get("review") or {}).keys())
    items, skipped, reasons = _batch_select_images(request, known_drafts, reviewed)
    if not items:
        detail = "没有需要处理的图片"
        if reasons:
            detail += "：" + "；".join(reasons[:5])
        raise HTTPException(status_code=400, detail=detail)

    options = {
        "mode": mode,
        "task": request.task,
        "engine": engine,
        "labels": labels_path,
        "prompt": prompt,
        "box_thr": request.box_thr,
        "text_thr": request.text_thr,
        "refine": request.refine,
        "conf": request.conf,
        "nms": request.nms,
        "service": request.service.strip(),
        "timeout": request.timeout,
        "scope": request.scope,
    }
    try:
        job = autolabeljobs.start(str(target), items, options)
    except autolabeljobs.BatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "job": job, "total": job["total"], "skipped": skipped, "reasons": reasons}


@app.get("/api/annotation/autolabel/batch")
def annotation_autolabel_batch_list(limit: int = autolabeljobs.MAX_JOBS) -> dict:
    """最近的批量任务列表：刷新页面后前端靠它找回"刚才那批跑到哪了"。"""
    return {"ok": True, "jobs": autolabeljobs.recent(max(1, min(int(limit), autolabeljobs.MAX_JOBS)))}


@app.get("/api/annotation/autolabel/batch/{job_id}")
def annotation_autolabel_batch_status(job_id: str) -> dict:
    job = autolabeljobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知批量任务：{job_id}（服务可能重启过）")
    return {"ok": True, "job": job}


@app.post("/api/annotation/autolabel/batch/{job_id}/stop")
def annotation_autolabel_batch_stop(job_id: str) -> dict:
    job = autolabeljobs.stop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知批量任务：{job_id}")
    return {"ok": True, "job": job}



# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------
frontend_index = config.FRONTEND_DIR / "index.html"


@app.get("/")
def index() -> FileResponse:
    """返回单页入口。"""
    if not frontend_index.is_file():
        raise HTTPException(status_code=500, detail="前端文件缺失：frontend/index.html")
    return FileResponse(frontend_index)


# 挂在最后，避免覆盖 /api 路由
if config.FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态资源一律要求重新校验。

    不带这个头时浏览器会按启发式规则把 .js 缓存住，改了前端、重启了后端，
    刷新页面跑的仍是旧模块。排查「按钮点了没反应」时遇到过：HTML 是新的、
    ES module 却是几小时前的。这些文件都在本机、体积很小，每次 revalidate
    的代价可以忽略，换来的是"刷新即最新"。
    """
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response
