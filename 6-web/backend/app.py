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
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import dataset
import infer
from runners import manager

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


app = FastAPI(title="TRTForge Web Studio", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """存活探针。"""
    return {"ok": True}


@app.get("/api/capabilities")
def capabilities() -> dict:
    """返回当前环境具备哪些能力，前端据此决定按钮可用性。"""
    data = config.capabilities()
    backend = infer.status()
    data["infer_yolo"] = backend.yolo_ok
    data["infer_anomalib"] = backend.anomalib_ok
    data["infer_detail"] = backend.detail
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
    python: str | None = Field(default=None, description="覆盖训练解释器")


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

    try:
        run = manager.create(
            "engine-build",
            command,
            cwd=str(config.REPO_ROOT),
            expect_file=engine_path,
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"启动 trtexec 失败：{exc}") from exc

    return {**run.snapshot(), "engine": str(engine_path), "onnx_probe": probe}


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
    allowed_roots = [config.REPO_ROOT]
    for extra in os.environ.get("TRTFORGE_BROWSE_ROOTS", "").split(os.pathsep):
        if extra.strip():
            allowed_roots.append(Path(extra.strip()).resolve())

    target = Path(path).expanduser().resolve()
    if not any(target == root or root in target.parents for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail=f"路径不在允许范围内：{target}。可设置 TRTFORGE_BROWSE_ROOTS 追加允许的根目录。",
        )


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
    allowed_roots = [config.REPO_ROOT]
    for extra in os.environ.get("TRTFORGE_BROWSE_ROOTS", "").split(os.pathsep):
        if extra.strip():
            allowed_roots.append(Path(extra.strip()).resolve())

    target = Path(path).resolve() if path.strip() else config.REPO_ROOT
    if not any(target == root or root in target.parents for root in allowed_roots):
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


@app.post("/api/infer/yolo")
def infer_yolo(
    image: UploadFile = File(...),
    engine: str = Form(...),
    task: str = Form("detect"),
    conf: float = Form(0.25),
    nms: float = Form(0.45),
) -> JSONResponse:
    """YOLO 推理：返回检测框与标注后的结果图（base64）。"""
    backend = infer.status()
    if not backend.yolo_ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "yolo_infer.dll 不可用："
                f"{backend.detail}。请确认已编译 DLL，并设置 OPENCV_DIR / CUDA_PATH。"
            ),
        )

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        result = infer.run_yolo(engine, task, array, conf=conf, nms=nms)
        _check_yolo_output_sane(result, array.shape[1], array.shape[0])
        annotated = infer.draw_yolo(array, result["boxes"])
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()

    import base64

    result["annotated_png"] = base64.b64encode(annotated).decode("ascii")
    return JSONResponse(result)


@app.post("/api/infer/anomalib")
def infer_anomalib(
    image: UploadFile = File(...),
    engine: str = Form(...),
    metadata: str = Form(...),
    model_type: str = Form("Auto"),
) -> JSONResponse:
    """Anomalib 推理：返回分数、判定、叠加图与异常轮廓图（base64）。"""
    backend = infer.status()
    if not backend.anomalib_ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "anomalib_infer.dll 不可用："
                f"{backend.detail}。请确认已编译 DLL，并设置 OPENCV_DIR / CUDA_PATH。"
            ),
        )

    data, temp_path = _save_upload(image)
    try:
        array = infer.decode_image(data)
        result = infer.run_anomalib(engine, metadata, array, model_type=model_type)
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink()

    import base64

    for key in ("overlay_png", "mask_png"):
        value = result.get(key)
        result[key] = base64.b64encode(value).decode("ascii") if value else None
    return JSONResponse(result)


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
