# -*- coding: utf-8 -*-
"""训练域路由：YOLO / Anomalib 训练、导出、环境检查、数据集预览。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import ampcheck
import augpreview
import checkpoints
import config
import fingerprint
import traininfo
import vram
import web_state

# 共享状态：并发闸门与路径白名单（见 web_state.py）
from web_state import _START_LOCK, _ensure_allowed_path, _reject_if_busy

router = APIRouter()
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

    # 混合精度三态：None = 跟随检测（不传参数，行为与升级前完全一致）；
    # False = 显式关闭（拼 --no-amp，跳过开训前那次前向自检）；True = 显式尝试。
    # 注意：True 不等于「一定用上 fp16」——黑名单显卡上框架仍会关掉（见 ampcheck.py）。
    amp: bool | None = Field(default=None, description="None=跟随检测 / False=强制关闭 / True=显式尝试")
    # 显存预检不放行时是否仍然开跑。默认 False：宁可让用户看到一次明确的拦截，
    # 也不要让 Ultralytics 静默把 batch 减半（那会让本次实验与台账其它行不可比）。
    allow_low_vram: bool = Field(default=False, description="显存预检不通过时仍然启动")


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
    if request.task not in {"detect", "segment", "pose", "obb", "classify"}:
        raise ValueError(f"不支持的任务类型：{request.task}")
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

    # 混合精度：三态。None 时**一个参数都不传**，走 launcher 默认（True），
    # 由 Ultralytics 自己在前向对比后决定——这样升级前后的命令行完全一致。
    # 之所以要显式开关：黑名单显卡（如 T1000）永远拿不到 fp16，传 --amp 只是多花
    # 几秒做自检；已知本机不支持的场景可以 --no-amp 把这个过程也省掉。
    if request.amp is True:
        command.append("--amp")
    elif request.amp is False:
        command.append("--no-amp")

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


def _batch_int(request: "YoloTrainRequest") -> int | None:
    """把表单里的 batch 变成单个整数。

    ``auto`` / ``8,16`` / ``-1`` 这类写法无法用单点估算（Ultralytics 会自己搜），
    返回 ``None`` 由调用方跳过预检，而不是猜一个数把用户拦在门外。
    """
    raw = str(request.batch or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _amp_from_mode(mode: str | None) -> bool | None:
    """``auto`` / ``off`` / ``on`` → 三态 ``None`` / ``False`` / ``True``。"""
    text = (mode or "auto").strip().lower()
    if text in {"off", "no-amp", "false", "0", "fp32"}:
        return False
    if text in {"on", "amp", "true", "1", "fp16"}:
        return True
    return None


def dataset_block(data: str) -> dict[str, Any]:
    """算数据集指纹，并压成适合放进 ``meta`` 与环境变量的精简块。

    为什么在开训**之前**算：训练要跑几百秒，源数据随时可能被重标或增删。
    先算一次，记录的才是「按下开始那一刻用的那批图」；训练后再算会把两件事混在一起。

    为什么精简：这个块会进 ``run.meta``（随接口返回给前端）和子进程环境变量
    （Windows 环境变量总量有限）。完整报告（含类别表）由 launcher 写进
    ``train_meta.json`` 时从同一份块生成，不会缺关键字段。
    """
    report = fingerprint.compute(data)
    return {
        "algo": report.get("algo"),
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "short": report.get("short") or fingerprint.short(report.get("dataset_fingerprint")),
        "label_fingerprint": report.get("label_fingerprint"),
        "split_fingerprints": report.get("split_fingerprints"),
        "counts": report.get("counts"),
        "task": report.get("task"),
        "nc": report.get("nc"),
        "yaml": report.get("yaml"),
        "computed_at": report.get("computed_at"),
        "error": report.get("error", ""),
    }


def vram_guard(request: "YoloTrainRequest") -> dict[str, Any]:
    """开训前的显存闸门。

    放行时返回预检详情（调用方写进台账）；不通过时抛 409，``detail`` 以
    ``VRAM_PREFLIGHT:`` 开头——前端靠这个前缀把「仍然开始」按钮显示出来，
    不必解析文案。
    """
    batch = _batch_int(request)
    if batch is None:
        return {"skipped": f"batch={request.batch!r} 不是单个正整数（auto/多卡列表），跳过显存预检"}
    # 只有「显式关闭混合精度」才是确定事实；其余一律按本机情况预测（vram 内部解析）。
    precision = "fp32" if request.amp is False else None
    report = vram.preflight(request.imgsz, batch, precision=precision)
    report["amp_requested"] = request.amp
    if report.get("ok") or request.allow_low_vram:
        return report
    raise HTTPException(
        status_code=409,
        detail=(
            f"VRAM_PREFLIGHT: {report.get('reason', '显存不足')}。{report.get('hint', '')}"
        ),
    )


@router.get("/api/train/preflight")
def train_preflight(
    imgsz: int = 640,
    batch: str = "16",
    amp: str = "auto",
) -> dict:
    """算力体检（只读，不启动训练）。

    前端在训练页按需调用：GPU、可用/需求显存、估算依据（实测还是公式）、
    混合精度是否可用。**不缓存**：可用显存随时在变，读旧值只会误导。
    """
    requested = _amp_from_mode(amp)
    # 只探测一次 GPU：free 与 name 都从这里取，避免 nvidia-smi 被调用两遍。
    info = vram.gpu_info()
    gpu = (info or {}).get("name")
    status = ampcheck.amp_status(gpu, requested=requested)
    precision = "fp32" if requested is False else status["expected_precision"]

    raw_batch = str(batch or "").strip()
    try:
        batch_int = int(raw_batch)
    except (TypeError, ValueError):
        batch_int = 0
    if batch_int > 0:
        report = vram.preflight(
            imgsz,
            batch_int,
            precision=precision,
            free_mb=(info or {}).get("free_mb"),
        )
    else:
        report = {
            "ok": True,
            "skipped": f"batch={raw_batch!r} 不是单个正整数（auto/多卡列表），无法估算",
            "precision": precision,
            "imgsz": imgsz,
            "batch": None,
            "free_mb": (info or {}).get("free_mb"),
            "total_mb": (info or {}).get("total_mb"),
            "gpu": gpu,
            "reason": "batch 交给 Ultralytics 自动决定，已跳过显存预检",
            "hint": "",
        }
    # 预检里 free_mb 是显式传入的，所以它自己不会再探测一次 GPU；
    # 把端点已经探测到的名称/总量补进去，避免响应里一半有一半没有。
    if gpu and not report.get("gpu"):
        report["gpu"] = gpu
    if not report.get("total_mb"):
        report["total_mb"] = (info or {}).get("total_mb")
    return {
        "ok": bool(report.get("ok")),
        "vram": report,
        "amp": status,
        "amp_requested": requested,
        "precision": precision,
        "amp_text": ampcheck.describe(status),
        "sample_count": len(vram.load_samples()),
        "store": str(vram.store_path()),
        "version": config.APP_VERSION,
    }


@router.post("/api/train/yolo")
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
        # 显存闸门放在「忙」检查之后：已有任务在跑时，那是更根本的原因，
        # 先报「显存不够」会把人引到错误的方向去释放显存。
        preflight = vram_guard(request)
        # 自动续训需要知道两件事：训练产物目录在哪（找 weights/last.pt）、
        # 失败后用什么命令接着跑。两条都在这里交给 RunManager，崩溃后由它补跑。
        run_dir = Path(request.project) / request.name
        # 数据集指纹：与显存预检同处一次请求内算好。训练跑几百秒，期间源数据
        # 随时可能被重标，事后再算记录的就不是「这次训练用的那批图」了。
        data_meta = dataset_block(str(data_yaml))
        meta: dict[str, Any] = {
            "vram_preflight": preflight,
            "dataset": data_meta,
            "data_fp": data_meta["short"],
        }
        if backup_dir:
            meta["backup_dir"] = str(backup_dir)
        try:
            run = web_state.manager.create(
                "yolo-train",
                command,
                str(config.YOLO_LAUNCHER_DIR),
                # 走环境变量而不是命令行：argv 会出现在进程列表、日志与前端 tooltip 里。
                # launcher 在 on_train_end 用它生成 train_meta.json（只有它知道最终 save_dir）。
                env={"TRTFORGE_DATASET_FP": json.dumps(data_meta, ensure_ascii=False)},
                meta=meta,
                warm_kind="yolo",
                resume_dir=run_dir,
                resume_command=lambda: build_train_command(
                    _resume_request(request, run_dir / "weights" / "last.pt"), python
                ),
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"启动训练进程失败：{exc}") from exc
    return run.snapshot()


def _resume_request(request: "YoloTrainRequest", last_weights: Path) -> "YoloTrainRequest":
    """把原始请求改成「从 last.pt 续训」的那一份。

    为什么不在原请求上改：``request`` 还要用来拼那一次的 resume_command，
    就地修改会让「成功后再点一次」这种操作拿到一份被污染的参数。
    ``exist_ok=True`` 是必须的——续训要写回同一个 ``project/name`` 目录，
    Ultralytics 默认会在目录已存在时新建 ``exp2``，用户会以为续训没生效。
    """
    patch = {
        "resume": False,
        "resume_from": str(last_weights),
        "init_mode": "preset",
        "exist_ok": True,
    }
    if hasattr(request, "model_copy"):
        return request.model_copy(update=patch)
    return request.copy(update=patch)  # pydantic v1 兜底


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


@router.post("/api/train/anomalib")
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
            run = web_state.manager.create(
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
    if request.task not in {"detect", "segment", "pose", "obb", "classify"}:
        raise ValueError(f"不支持的任务类型：{request.task}")
    # 分类输出不是检测框，不能再生成给 TensorRT 检测解码用的转置文件。
    make_transd = bool(request.make_transd) and request.task != "classify"

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
        "--make-transd", "true" if make_transd else "false",
    ]
    if request.opset is not None:
        command += ["--opset", str(request.opset)]
    if request.dynamic:
        command.append("--dynamic")
    if request.half:
        command.append("--half")
    return command


@router.post("/api/export/yolo")
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
        run = web_state.manager.create(
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


@router.post("/api/train/dataset-overview")
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


@router.post("/api/train/augment-preview")
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


@router.get("/api/runs/checkpoints")
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


@router.post("/api/train/yolo/finetune-check")
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
@router.post("/api/train/{kind}/check-env")
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
        run = web_state.manager.create(f"{kind}-check-env", command, str(cwd))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"启动自检进程失败：{exc}") from exc
    return run.snapshot()
