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
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import annotation
import augpreview
import autolabel
import checkpoints
import config
import dataset
import infer
import labels as labels_mod
import models as models_mod
import onnx_fallback
import progress
import traininfo
from runners import manager

#: 应用版本号。前端顶栏与 /api/health 都读它，方便一眼看出「页面连的是哪个后端」。
#: 起因：改了代码但 8060 上跑的是几小时前的旧进程，排查很久才发现。
APP_VERSION = "1.15.0"

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

    command = [
        python, str(config.YOLO_LAUNCHER),
        "--mode", "train",
        "--task", request.task,
        "--model", request.model,
        "--data", str(data_yaml),
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
            raise HTTPException(status_code=400, detail=f"续训检查点不存在：{checkpoint}")
        _ensure_allowed_path(str(checkpoint))
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

    用户选的是 TensorRT 引擎（.engine/.trt），onnxruntime 读不了它，
    所以必须找回同源的 ONNX。按可信度依次尝试：

    1. 用户直接选的就是 .onnx —— 直接用；
    2. 同目录同名 .onnx（`best.engine` -> `best.onnx`，`best.transd.engine` -> `best.transd.onnx`）；
    3. metadata.json 里记录的 `onnx_path`（训练时写下的原始 ONNX 位置。

    第 2 条优先于第 3 条：metadata 里的 onnx_path 往往是训练机上的绝对路径，
    模型拷到别的机器后就失效了，而同目录同名文件通常跟着一起拷过来。
    """
    path = Path(engine_path)
    if path.suffix.lower() == ".onnx":
        return (str(path), "") if path.is_file() else (None, f"ONNX 文件不存在：{engine_path}")

    sibling = path.with_suffix(".onnx")
    if sibling.is_file():
        return str(sibling), ""

    if metadata_path and Path(metadata_path).is_file():
        try:
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        recorded = str(meta.get("onnx_path", "") or "").strip()
        if recorded and Path(recorded).is_file():
            return recorded, ""

    return None, (
        f"降级推理需要 .onnx，但没找到与 {path.name} 对应的 ONNX。"
        f"请把同名的 {path.stem}.onnx 放到同一目录，或直接选择 .onnx 模型。"
    )


def _fallback_reason(primary: str, detail: str) -> str:
    return f"{primary} 不可用（{detail}），已自动降级为 onnxruntime"


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
    backend = infer.status()
    warning: str | None = None
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

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        if backend.yolo_ok:
            result = infer.run_yolo(engine, task, array, conf=conf, nms=nms)
        else:
            try:
                result = onnx_fallback.run_yolo(onnx_path, task, array, conf=conf, nms=nms)
            except onnx_fallback.FallbackError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        _check_yolo_output_sane(result, array.shape[1], array.shape[0])

        # 类别名与配色在这里补齐，两条后端路径共用同一份逻辑
        label_info = labels_mod.load_labels(labels)
        infer.attach_labels(result, label_info["classes"])
        result["labels"] = {
            "path": labels or None,
            "ok": label_info["ok"],
            "name": label_info["name"],
            "count": len(label_info["classes"]),
            "error": label_info["error"],
        }
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
