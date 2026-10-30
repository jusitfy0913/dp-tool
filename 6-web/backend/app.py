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
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
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
