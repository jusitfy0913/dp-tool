# -*- coding: utf-8 -*-
"""模型域路由：散装产物列表、``.tfbundle`` 模型包、内置模型仓库下载。

三组接口的区分是有意的：
- ``/api/models`` 面向**过程产物**（训练目录里的 engine/onnx），只扫目录属性；
- ``/api/bundles*`` 面向**交付物**（打包好的模型包），带清单与哈希校验；
- ``/api/modelzoo*`` 面向**内置模型仓库**，需要联网下载。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import bundle as bundle_mod
import config
import models as models_mod
import modelzoo

# 共享状态：路径白名单（见 web_state.py）
from web_state import _ensure_allowed_path

router = APIRouter()


class BundleBuildRequest(BaseModel):
    """造模型包的参数。

    ``kind`` 留空时按模型文件所在目录推断（`runs` → yolo，`results` → anomalib）；
    推断不出就记 ``unknown``，校验时只提示不阻拦——不能因为猜不到类型就不许用户造包。
    """

    model: str = Field(..., description="引擎或 ONNX 的绝对路径")
    bundle_id: str = Field(..., description="包名，也是目录名（不含 .tfbundle）")
    title: str = Field(default="", description="给人看的名字，留空用包名")
    out_root: str = Field(default="", description="存放目录，留空用 runtime/bundles")
    kind: str = Field(default="", description="yolo / anomalib，留空自动推断")
    task: str = Field(default="", description="detect / segment / pose / obb / classify / anomaly")
    labels: str = Field(default="", description="标签文件，留空自动查找")
    metadata: str = Field(default="", description="异常检测 metadata.json，留空自动查找")
    precision: str = Field(default="", description="fp32 / fp16 / int8，留空按文件推断")
    imgsz: int = Field(default=0, ge=0, le=16384, description="输入边长，留空不记")
    overwrite: bool = Field(default=False, description="目标已存在时是否覆盖")


class ModelDownloadRequest(BaseModel):
    """触发一次模型下载。"""

    id: str = Field(..., description="模型 id，见 GET /api/modelzoo")


# ---------------------------------------------------------------------------
# 内置模型仓库
# ---------------------------------------------------------------------------
@router.get("/api/modelzoo")
def modelzoo_list() -> dict:
    """内置模型清单与本地状态（含正在进行的下载任务）。"""
    return {"ok": True, **modelzoo.list_models(), "jobs": modelzoo.recent()}


@router.post("/api/modelzoo/download")
def modelzoo_download(request: ModelDownloadRequest) -> dict:
    """开始下载（幂等：已就绪返回 `already=True`，正在下返回 400 说明原因）。"""
    try:
        job = modelzoo.start_download(request.id)
    except modelzoo.DownloadBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except modelzoo.ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "job": job}


@router.get("/api/modelzoo/download/{job_id}")
def modelzoo_progress(job_id: str) -> dict:
    job = modelzoo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知下载任务：{job_id}（服务可能重启过）")
    return {"ok": True, "job": job}


@router.post("/api/modelzoo/download/{job_id}/cancel")
def modelzoo_cancel(job_id: str) -> dict:
    job = modelzoo.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知下载任务：{job_id}")
    return {"ok": True, "job": job}


# ---------------------------------------------------------------------------
# 训练产物（散装 engine / onnx）
# ---------------------------------------------------------------------------
@router.get("/api/models")
def list_models(limit: int = models_mod.MAX_ENTRIES) -> dict:
    """枚举训练产物目录里的可用引擎，供推理页的模型下拉使用。

    只读目录、只取文件属性，不读取引擎或 ONNX 内容——
    一个 PatchCore 目录就有 275MB 的引擎加 300MB 的 ONNX，读一遍会很慢。
    """
    try:
        return models_mod.scan_models(limit=max(1, min(int(limit), models_mod.MAX_ENTRIES)))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"扫描训练产物失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 模型包（.tfbundle 交付物）
# ---------------------------------------------------------------------------
@router.get("/api/bundles")
def list_bundles() -> dict:
    """列出全部模型包（只做轻量校验，不算哈希）。

    与 ``/api/models`` 分开的理由：模型包是**交付物**，模型文件是**过程产物**。
    前端模型下拉把两者分组显示，后端也就分两个接口，各自可以独立演进。
    """
    try:
        items = bundle_mod.enumerate_bundles()
    except Exception as exc:  # noqa: BLE001 - 枚举失败要给出原因而不是 500
        raise HTTPException(status_code=500, detail=f"枚举模型包失败：{exc}") from exc

    payload = []
    for item in items:
        verdict = bundle_mod.quick_check(item["path"])
        payload.append({**item, "verify": verdict})
    return {
        "bundles": payload,
        "total": len(payload),
        "root": str(bundle_mod.default_root()),
    }


@router.post("/api/bundles/build")
def build_bundle(request: BundleBuildRequest) -> dict:
    """把散装产物造成模型包。

    原始产物一个字节都不改：模型文件优先用硬链接，跨分区才复制。
    目标已存在时默认拒绝——静默覆盖用户手里的包是不可接受的。
    """
    model_path = _ensure_allowed_path(request.model)
    if not model_path.is_file():
        raise HTTPException(status_code=400, detail=f"模型文件不存在：{model_path}")

    if request.out_root.strip():
        out_root = _ensure_allowed_path(request.out_root)
    else:
        out_root = bundle_mod.default_root()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"无法创建存放目录：{out_root}（{exc}）") from exc

    kind = request.kind.strip().lower() or models_mod.guess_kind(model_path)
    labels_path: Path | None = None
    metadata_path: Path | None = None

    # 留空时自动找配套文件，规则与推理页完全一致（models.companion_files）
    if model_path.suffix.lower() in {".engine", ".trt", ".plan"} and kind in {"yolo", "anomalib"}:
        companions = models_mod.companion_files(model_path, kind)
        labels_path = companions.get("labels")
        metadata_path = companions.get("metadata")
    if request.labels.strip():
        labels_path = _ensure_allowed_path(request.labels)
    if request.metadata.strip():
        metadata_path = _ensure_allowed_path(request.metadata)

    imgsz = request.imgsz or None
    if imgsz is None:
        # 引擎旁边有 ONNX 时用它的输入尺寸；异常检测用 metadata 里的 input_size
        if metadata_path is not None:
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                size = meta.get("input_size") if isinstance(meta, dict) else None
                if isinstance(size, list) and len(size) == 2:
                    imgsz = int(size[0])
            except (OSError, ValueError, TypeError):
                imgsz = None

    onnx_layout = "unknown"
    if model_path.suffix.lower() == ".onnx":
        onnx_layout = bundle_mod._layout_of(model_path)

    origin_extra: dict[str, Any] = {
        "title": request.title.strip() or request.bundle_id,
        "created_by": f"TRTForge Web Studio {config.APP_VERSION}",
        "app_version": config.APP_VERSION,
        "onnx_layout": onnx_layout,
    }

    built = bundle_mod.build_bundle(
        model_path,
        out_root,
        request.bundle_id,
        labels=labels_path,
        metadata=metadata_path,
        kind=kind,
        task=request.task or None,
        precision=request.precision or None,
        imgsz=imgsz,
        dynamic=False,
        origin_extra=origin_extra,
        overwrite=bool(request.overwrite),
    )
    if not built.get("ok"):
        raise HTTPException(status_code=400, detail=str(built.get("error") or "造包失败"))

    return {
        "ok": True,
        "path": built["path"],
        "manifest": built["manifest"],
        "verify": bundle_mod.quick_check(built["path"]),
    }


@router.post("/api/bundles/verify")
def verify_bundle(payload: dict) -> dict:
    """完整校验（含哈希）。由用户点「校验」时调用，不在列表接口里跑。"""
    raw = str(payload.get("bundle") or payload.get("path") or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="bundle 不能为空")

    target = bundle_mod.find_bundle(raw)
    if target is None:
        raise HTTPException(status_code=404, detail=f"找不到模型包：{raw}")

    return bundle_mod.verify_bundle(
        target,
        kind=(str(payload.get("kind") or "").strip() or None),
        task=(str(payload.get("task") or "").strip() or None),
    )


@router.get("/api/bundles/{bundle_id}/manifest")
def bundle_manifest(bundle_id: str) -> dict:
    """原文返回清单，供页面显示全部字段与排障。"""
    target = bundle_mod.find_bundle(bundle_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"找不到模型包：{bundle_id}")
    manifest = bundle_mod.read_manifest(target)
    if manifest is None:
        raise HTTPException(status_code=400, detail=f"清单不可读：{target}")
    return {"path": str(target), "manifest": manifest}
