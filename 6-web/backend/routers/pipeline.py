# -*- coding: utf-8 -*-
"""全流程串联路由：YOLO 与 Anomalib 两条链路、串联记录与日志。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import config
import dataset
import logstore
import pipeline as pipeline_mod
import traininfo

# 共享状态：任务管理器、并发闸门、路径白名单（见 web_state.py）
from web_state import _START_LOCK, _ensure_allowed_path, _reject_if_busy, pipeline_manager

# 串联要复用训练域的拼命令逻辑与引擎域的 ONNX 探测，避免两份实现漂移
from routers.engine import _looks_like_untransposed_yolo, _probe_onnx
from routers.train import (
    AnomalibTrainRequest,
    YoloExportRequest,
    YoloTrainRequest,
    _resolve_python,
    build_anomalib_train_command,
    build_export_command,
    build_train_command,
)

router = APIRouter()
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

    precision: str = Field(default="fp16", description="引擎精度 fp16 / fp32 / int8")
    calib_dir: str = Field(default="", description="INT8 校准图片文件夹，留空则用本次验证 split")
    calib_count: int = Field(default=100, ge=1, le=500)
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


@router.post("/api/pipeline/yolo")
def pipeline_yolo(request: PipelineYoloRequest) -> dict:
    """一键跑完「训练 → 导出 → 校验 → 编译 → 标签 → 引擎验证」。"""
    data_yaml = Path(request.data).expanduser()
    if not data_yaml.is_file():
        raise HTTPException(status_code=400, detail=f"dataset.yaml 不存在：{request.data}")
    _ensure_allowed_path(str(data_yaml))
    is_classify = request.task == "classify"
    if not is_classify and request.task != "detect" and request.run_val:
        raise HTTPException(
            status_code=400,
            detail="精度评估目前只支持 detect；请关闭「完成后验证」，或把任务类型改为 detect",
        )
    if request.precision not in ("fp16", "fp32", "int8"):
        raise HTTPException(status_code=400, detail="precision 只能是 fp16 / fp32 / int8")

    trtexec = ""
    if not is_classify:
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
    tag = request.precision if request.precision in ("fp16", "int8") else "fp32"
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
        make_transd=not is_classify,
        simplify=True,
    )
    stages.append(
        pipeline_mod.StagePlan(
            name="export",
            label="导出 ONNX",
            command=build_export_command(export_request, python),
            cwd=str(config.YOLO_LAUNCHER_DIR),
            expect_file=str(onnx_file if is_classify else transd_onnx),
            meta={"weights": str(weights)},
        )
    )
    if is_classify:
        reason = "分类这一版不编译引擎，也不做检测精度验证"
        stages.append(pipeline_mod.StagePlan(name="probe", label="校验转置布局", skip_reason=reason))
        stages.append(pipeline_mod.StagePlan(name="build", label="编译引擎", skip_reason=reason))
        stages.append(
            pipeline_mod.StagePlan(
                name="labels",
                label="生成标签文件",
                callable=lambda: _ensure_labels_json(str(data_yaml), run_dir, name),
            )
        )
        stages.append(pipeline_mod.StagePlan(name="val", label="引擎精度验证", skip_reason=reason))
    if not is_classify:
        stages.append(
            pipeline_mod.StagePlan(
                name="probe",
                label="校验转置布局",
                callable=lambda: _pipeline_probe_transd(transd_onnx),
            )
        )
    if not is_classify and request.precision == "int8":
        import int8calib

        if request.calib_dir.strip():
            folder = Path(request.calib_dir.strip()).expanduser()
            if not folder.is_dir():
                raise HTTPException(status_code=400, detail=f"校准图片文件夹不存在：{folder}")
            _ensure_allowed_path(str(folder))
            calib_images = int8calib.list_images(folder)
        else:
            try:
                calib_images = traininfo.list_split_images(data_yaml, request.val_split)
            except traininfo.OverviewError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            calib_images = [
                path for path in calib_images if path.suffix.lower() in int8calib.IMAGE_SUFFIXES
            ]
        if not calib_images:
            raise HTTPException(status_code=400, detail="INT8 校准没有可用图片。请指定文件夹，或确认验证集里有图。")
        if int8calib.find_calib_exe() is None:
            raise HTTPException(
                status_code=400,
                detail="找不到 int8_calib.exe。请先运行 scripts\\build_infer_dlls.ps1。",
            )
        list_path = run_dir / "int8_images.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        int8calib.write_image_list(list_path, calib_images)
        build_command = config.tool_command(python, "int8_launch.py") + [
            "--trtexec", trtexec,
            "--onnx", str(transd_onnx),
            "--engine", str(engine_path),
            "--list", str(list_path),
            "--count", str(request.calib_count),
            "--workspace", str(request.workspace_mb),
            "--input-name", "images",
        ]
    elif not is_classify:
        build_command = [
            trtexec,
            f"--onnx={transd_onnx}",
            f"--saveEngine={engine_path}",
        ]
        if request.precision == "fp16":
            build_command.append("--fp16")
        build_command.append(f"--workspace={request.workspace_mb}")

    if not is_classify:
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
    if not is_classify and request.run_val:
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
                cwd=str(config.BACKEND_DIR),
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


@router.post("/api/pipeline/anomalib")
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
                cwd=str(config.BACKEND_DIR),
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


@router.get("/api/pipeline")
def pipeline_list(limit: int = 10) -> dict:
    """最近若干条串联记录。刷新页面后还能看到上次跑到哪。"""
    return {"items": pipeline_manager.list(max(1, min(limit, 50)))}


@router.get("/api/pipeline/{pipeline_id}/log")
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

    if not logstore.any_shard(run_id):
        raise HTTPException(status_code=404, detail=f"阶段 {stage} 的日志文件不存在")

    limit = max(1, min(int(lines), 5000))
    text = logstore.read_tail(run_id, limit)
    return Response(content=text, media_type="text/plain; charset=utf-8")


@router.get("/api/pipeline/{pipeline_id}")
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


@router.post("/api/pipeline/{pipeline_id}/stop")
def pipeline_stop(pipeline_id: str) -> dict:
    """停止整条链：只停当前阶段，已完成的阶段产物保留。"""
    if not pipeline_manager.stop(pipeline_id):
        raise HTTPException(status_code=409, detail="该串联已结束，无法停止")
    record = pipeline_manager.get(pipeline_id) or {}
    return {"ok": True, "pipeline": record}
