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
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import annotation
import augpreview
import autolabel
import checkpoints
import config
import dataset
import infer
import inferjobs
import labels as labels_mod
import models as models_mod
import onnx_fallback
import pipeline as pipeline_mod
import progress
import traininfo
from runners import manager

#: 应用版本号。前端顶栏与 /api/health 都读它，方便一眼看出「页面连的是哪个后端」。
#: 起因：改了代码但 8060 上跑的是几小时前的旧进程，排查很久才发现。
APP_VERSION = "1.17.0"

#: 本文件所在目录（6-web/backend）。子进程脚本路径都以它为基准，
#: 不依赖启动时的工作目录——从仓库根启动和从 backend 启动必须都能跑。
BACKEND_DIR = Path(__file__).resolve().parent

#: 全流程串联的注册表。每段仍是一个普通 run，所以这里的职责只有「决定下一段跑什么」。
pipeline_manager = pipeline_mod.PipelineManager(
    manager, config.RUNTIME_DIR / "pipelines"
)

# ---------------------------------------------------------------------------
# 应用与生命周期
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时准备目录与事件循环，退出时清理子进程与句柄。"""
    config.ensure_dirs()
    manager.bind_loop(asyncio.get_running_loop())
    # 提前探测推理库，让 /api/capabilities 立刻反映真实状态
    infer.load_libraries()
    try:
        yield
    finally:
        # 先停串联（它可能正握着某个 run），再停所有子进程
        pipeline_manager.shutdown_all()
        manager.shutdown_all()
        infer.release_handles()


app = FastAPI(title="TRTForge Web Studio", version=APP_VERSION, lifespan=lifespan)

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
    backend = infer.status()
    data["infer_yolo"] = backend.yolo_ok
    data["infer_anomalib"] = backend.anomalib_ok
    data["infer_detail"] = backend.detail
    # 降级能力：DLL 不可用时，前端可提示「仍可用 ONNX 跑通链路」
    fallback_ok, fallback_detail = onnx_fallback.available()
    data["infer_fallback"] = fallback_ok
    data["infer_fallback_detail"] = fallback_detail
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

    try:
        command = build_train_command(request, python)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        run = manager.create("yolo-train", command, str(config.YOLO_LAUNCHER_DIR))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"启动训练进程失败：{exc}") from exc
    return run.snapshot()


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
    command = [
        python, str(config.ANOMALIB_LAUNCHER),
        "--mode", "train-export",
        "--model", request.model,
        "--root", str(root),
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

    try:
        run = manager.create("anomalib-train", command, str(config.ANOMALIB_LAUNCHER_DIR))
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
                command=[
                    python,
                    str(BACKEND_DIR / "tools" / "val_engine.py"),
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
def list_runs() -> dict:
    """列出本次服务生命周期内的所有任务。"""
    return {"runs": manager.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """查询单个任务。"""
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"未知任务：{run_id}")
    return run.snapshot()


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


def _run_yolo_once(
    engine: str,
    task: str,
    array,
    conf: float,
    nms: float,
    labels: str = "",
) -> tuple[dict, str | None]:
    """执行一次 YOLO 推理并补齐类别名与配色，返回 (结果, 降级说明)。

    抽出来的原因：单图接口与摄像头接口必须走**完全同一套**后端选择与
    输出结构。摄像头每秒调 10 次，如果它是"另一条实现"，两边框的顺序、
    字段名、降级行为迟早对不上，而屏幕上看起来都"能出框"，极难发现。
    """
    backend = infer.status()
    warning: str | None = None
    onnx_path = ""
    if not backend.yolo_ok:
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
        warning = _fallback_reason("yolo_infer.dll", backend.detail or "未知原因")

    label_info = labels_mod.load_labels(labels)
    try:
        if backend.yolo_ok:
            result = infer.run_yolo(engine, task, array, conf=conf, nms=nms)
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
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels)
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
    backend = infer.status()
    warning: str | None = None
    if not backend.anomalib_ok:
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
        warning = _fallback_reason("anomalib_infer.dll", backend.detail or "未知原因")

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        if backend.anomalib_ok:
            result = infer.run_anomalib(
                engine, metadata, array,
                model_type=model_type,
                image_threshold=image_threshold,
                pixel_threshold=pixel_threshold,
                colormap=heatmap_colormap,
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

    kind: str = Field(default="yolo", description="yolo / anomalib")
    engine: str = Field(..., description=".engine / .trt / .onnx")
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


def build_batch_command(
    request: "BatchInferRequest",
    python: str,
    job_dir: str | Path,
) -> list[str]:
    """组装批量推理命令。

    单独成函数是为了能被测试直接断言：参数在表单里填了却没进命令行，
    用户只能从日志里翻完整命令才能发现（`--simplify true` 就是这么漏的）。
    """
    if request.kind not in ("yolo", "anomalib"):
        raise HTTPException(status_code=400, detail=f"kind 只能是 yolo / anomalib，实际 {request.kind}")

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

    command = [
        python,
        str(BACKEND_DIR / "tools" / "infer_batch.py"),
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

    command = [
        python,
        str(BACKEND_DIR / "tools" / "infer_video.py"),
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
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels)
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
    return {"ok": True, **infer.cache_stats()}


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
        "project": project,
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
    for item in base.glob(pattern):
        if len(found) >= limit:
            break
        if not item.is_file() or item.suffix.lower() not in suffixes:
            continue
        width, height = dataset._image_size(item)
        found.append({"path": str(item), "width": width, "height": height})

    found.sort(key=lambda entry: entry["path"].lower())
    return {"ok": True, "root": str(base), "count": len(found), "images": found, "truncated": len(found) >= limit}


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
