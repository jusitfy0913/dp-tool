# -*- coding: utf-8 -*-
"""标注域路由：工程读写、图片扫描、导入导出、自动标注与草稿审核。"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import annotation
import annotation_drafts
import annotationimport
import autolabel
import autolabeljobs
import config
import dataset
import infer
import modelzoo
import promptlabel
import textlabel

# 共享状态：路径白名单（见 web_state.py）
from web_state import _ensure_allowed_path

router = APIRouter()
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


@router.get("/api/annotation/project")
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


@router.post("/api/annotation/project")
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


@router.get("/api/annotation/image")
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


@router.get("/api/annotation/image-meta")
def annotation_image_meta(path: str) -> dict:
    """探测单张图片的尺寸，供"添加图片到工程"时填 width/height。"""
    target = _ensure_allowed_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在：{target}")
    width, height = dataset._image_size(target)
    return {"ok": True, "path": str(target), "width": width, "height": height}


@router.post("/api/annotation/scan-images")
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


@router.post("/api/annotation/import")
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


@router.post("/api/annotation/export")
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
        # 划分来自工程里逐个设过的 current_split，所以 manifest 里如实写 manual，
        # 而不是伪造一个比例——后人看到「80/20」会以为可以按比例复现。
        split_mode = {"mode": "manual", "source": "annotation"}
        if request.format == "yolo":
            result = dataset.export_yolo(
                project, target, splits, request.overwrite, split_mode=split_mode
            )
        elif request.format == "coco":
            result = dataset.export_coco(
                project, target, splits, request.overwrite, split_mode=split_mode
            )
        else:
            result = dataset.export_voc(
                project, target, splits, request.overwrite, split_mode=split_mode
            )
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


@router.post("/api/annotation/autolabel")
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


@router.get("/api/annotation/drafts")
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


@router.post("/api/annotation/drafts")
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


@router.post("/api/annotation/drafts/review")
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


@router.post("/api/annotation/prompt-label")
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


@router.post("/api/annotation/text-label")
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


@router.post("/api/annotation/autolabel/batch")
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


@router.get("/api/annotation/autolabel/batch")
def annotation_autolabel_batch_list(limit: int = autolabeljobs.MAX_JOBS) -> dict:
    """最近的批量任务列表：刷新页面后前端靠它找回"刚才那批跑到哪了"。"""
    return {"ok": True, "jobs": autolabeljobs.recent(max(1, min(int(limit), autolabeljobs.MAX_JOBS)))}


@router.get("/api/annotation/autolabel/batch/{job_id}")
def annotation_autolabel_batch_status(job_id: str) -> dict:
    job = autolabeljobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知批量任务：{job_id}（服务可能重启过）")
    return {"ok": True, "job": job}


@router.post("/api/annotation/autolabel/batch/{job_id}/stop")
def annotation_autolabel_batch_stop(job_id: str) -> dict:
    job = autolabeljobs.stop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知批量任务：{job_id}")
    return {"ok": True, "job": job}
