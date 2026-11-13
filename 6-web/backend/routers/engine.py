# -*- coding: utf-8 -*-
"""引擎域路由：ONNX 探测、TensorRT 引擎构建、工作点存取。"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import bundle as bundle_mod
import config
import enginemeta
import models as models_mod
import progress
import web_state

# 共享状态：并发闸门与路径白名单（见 web_state.py）
from web_state import _START_LOCK, _ensure_allowed_path, _reject_if_busy

router = APIRouter()
# ---------------------------------------------------------------------------
# 引擎构建（ONNX -> TensorRT）
# ---------------------------------------------------------------------------
class EngineBuildRequest(BaseModel):
    """trtexec 转换参数。字段刻意贴近 trtexec 官方参数，便于对照文档排查。"""

    onnx: str = Field(..., description="待转换的 ONNX 绝对路径")
    engine: str = Field(default="", description="输出引擎路径，留空按 ONNX 同名推导")
    precision: str = Field(default="fp16", description="fp32 / fp16 / int8")
    calib_dir: str = Field(default="", description="INT8 校准图片文件夹")
    calib_count: int = Field(default=100, ge=1, le=500, description="从文件夹均匀抽取的张数")
    dynamic: bool = Field(default=False, description="是否编译动态 shape")
    min_shape: str = Field(default="", description="如 1x3x640x640")
    opt_shape: str = Field(default="", description="如 1x3x640x640")
    max_shape: str = Field(default="", description="如 8x3x640x640")
    workspace_mb: int = Field(default=4096, ge=256, le=32768)
    extra_args: str = Field(default="", description="追加的原始 trtexec 参数")
    auto_bundle: bool = Field(default=True, description="构建成功后同时生成模型包")
    bundle_id: str = Field(default="", description="模型包名，留空按引擎名推导")


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


@router.get("/api/engine/probe")
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


class OperatingPointRequest(BaseModel):
    """把「漏检-过杀」曲线上选定的点保存为引擎默认工作点。"""

    engine: str = Field(..., description="引擎文件绝对路径")
    conf: float = Field(..., ge=0.0, le=1.0)
    iou: float = Field(default=0.5, ge=0.01, le=1.0)
    source: str = Field(default="", description="哪个验证报告选的（便于追溯）")
    target_miss_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="当点的漏检率/过杀/良品误报，原样存进旁挂文件"
    )


@router.post("/api/engine/operating-point")
def save_operating_point(request: OperatingPointRequest) -> dict:
    """写 ``<engine>.meta.json`` 的 ``operating_point``。

    只改旁挂文件：不碰引擎、不碰 metadata.json，所以不会改变已有部署的判定行为。
    """
    engine = Path(request.engine).expanduser()
    if not engine.is_file():
        raise HTTPException(status_code=400, detail=f"引擎文件不存在：{request.engine}")
    payload = enginemeta.write_operating_point(
        engine,
        {
            "conf": float(request.conf),
            "iou": float(request.iou),
            "source": request.source,
            "target_miss_rate": request.target_miss_rate,
            **request.metrics,
        },
    )
    if payload.get("error"):
        raise HTTPException(status_code=500, detail=payload["error"])
    return {
        "engine": str(engine),
        "path": payload.get("path"),
        "operating_point": payload.get("operating_point"),
    }


@router.get("/api/engine/operating-point")
def get_operating_point(engine: str) -> dict:
    """读回引擎工作点；没有就返回空对象，让前端沿用默认值。"""
    path = Path(engine).expanduser()
    point = enginemeta.read_operating_point(path)
    return {"engine": str(path), "path": str(enginemeta.metadata_path(path)), "operating_point": point or None}


@router.post("/api/engine/build")
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
        tag = request.precision if request.precision in ("fp16", "int8") else "fp32"
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
    # INT8 改由 int8calib 决定：缓存还对就走 trtexec，否则走 int8_calib.exe。
    shapes = None
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
    if request.precision == "int8":
        import int8calib

        folder = Path(request.calib_dir.strip()).expanduser() if request.calib_dir.strip() else None
        if folder is None or not folder.is_dir():
            raise HTTPException(status_code=400, detail="INT8 需要一个放校准图片的文件夹")
        _ensure_allowed_path(str(folder))
        try:
            command = int8calib.prepare_command(
                trtexec=trtexec,
                onnx_path=onnx_path,
                engine_path=engine_path,
                images=int8calib.list_images(folder),
                count=request.calib_count,
                workspace_mb=request.workspace_mb,
                input_name=probe.get("input_name") or "images",
                shapes=shapes,
            )
        except int8calib.Int8Error as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        command = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
        if request.precision == "fp16":
            command.append("--fp16")
        elif request.precision != "fp32":
            raise HTTPException(
                status_code=400, detail=f"不支持的精度：{request.precision}，可选 fp32 / fp16 / int8"
            )
        command.append(f"--workspace={request.workspace_mb}")

    if request.dynamic and request.precision != "int8":
        input_name = probe.get("input_name") or "images"
        command += [
            f"--minShapes={input_name}:{shapes[0]}",
            f"--optShapes={input_name}:{shapes[1]}",
            f"--maxShapes={input_name}:{shapes[2]}",
        ]

    if request.extra_args.strip():
        if request.precision == "int8" and not str(command[0]).lower().endswith("trtexec.exe"):
            raise HTTPException(
                status_code=400,
                detail="第一次做 INT8 校准不能加额外参数。同一批图再次编译会走 trtexec，那时可以加。",
            )
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

    # 引擎构建是异步任务，结束后再写来源说明。用 list 做 holder：on_finished 是
    # 无参回调，而 run_id 要等 create() 返回才知道。回调里绝不抛异常——
    # 引擎已经编译好了，为了一份说明把它判成失败是本末倒置。
    created: list[Any] = []

    def _write_engine_meta() -> None:
        if not engine_path.is_file():
            return
        try:
            report = enginemeta.write_meta(
                engine_path,
                onnx_path=onnx_path,
                precision=request.precision,
                command=list(command),
                run_id=created[0].run_id if created else "",
                dynamic=request.dynamic,
                app_version=config.APP_VERSION,
                imgsz=imgsz,
            )
        except Exception as exc:  # noqa: BLE001 - 元数据失败不影响构建结果
            print(f"[enginemeta] 元数据写入异常（不影响构建）：{type(exc).__name__}: {exc}", flush=True)
            return
        if report.get("error"):
            print(f"[enginemeta] 元数据写入失败（不影响构建）：{report['error']}", flush=True)
            return
        _auto_build_bundle(engine_path, onnx_path, created[0].run_id if created else "")

    def _auto_build_bundle(target: Path, source_onnx: Path, run_id: str) -> None:
        """引擎构建成功后顺手造一个模型包。

        为什么默认做：这是唯一「刚好知道全部字段」的时刻——ONNX 布局、精度、
        run_id、输入尺寸都在手上。用户之后要拿模型去交付时，直接有包可用。

        失败一律只打日志：引擎已经编译好了，为了一份清单把它判成失败是本末倒置。
        用配置开关 ``auto_bundle`` 可以关掉（现场想只留散装产物时用）。
        """
        if not request.auto_bundle:
            return
        try:
            kind = models_mod.guess_kind(target)
            companions = models_mod.companion_files(target, kind)
            built = bundle_mod.build_bundle(
                target,
                bundle_mod.default_root(),
                request.bundle_id.strip() or f"{target.parent.parent.name}_{target.stem}",
                labels=companions.get("labels"),
                metadata=companions.get("metadata"),
                kind=kind,
                task=None,
                precision=request.precision,
                imgsz=imgsz,
                dynamic=request.dynamic,
                origin_extra={
                    "title": target.stem,
                    "created_by": f"TRTForge Web Studio {config.APP_VERSION}",
                    "app_version": config.APP_VERSION,
                    "run_id": run_id,
                    "onnx_layout": "transposed" if ".transd." in source_onnx.name.lower() else "raw",
                    "onnx_sha256": enginemeta.file_sha256(source_onnx),
                },
                overwrite=True,
            )
            if built.get("ok"):
                print(f"[bundle] 已生成模型包：{built['path']}", flush=True)
            else:
                print(f"[bundle] 造包失败（不影响构建）：{built.get('error')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[bundle] 造包异常（不影响构建）：{type(exc).__name__}: {exc}", flush=True)

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = web_state.manager.create(
                "engine-build",
                command,
                cwd=str(config.REPO_ROOT),
                expect_file=engine_path,
                meta=meta,
                estimate=estimate,
                on_finished=_write_engine_meta,
            )
            created.append(run)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动 trtexec 失败：{exc}") from exc

    return {        **run.snapshot(),
        "engine": str(engine_path),
        "onnx_probe": probe,
        "estimate_seconds": round(estimate, 1),
    }
