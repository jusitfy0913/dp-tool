# -*- coding: utf-8 -*-
"""系统域路由：根页面、健康检查、能力探测、目录浏览、原生对话框、预热。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
import inferproxy
import nativedialog
import ocr_models
import onnx_fallback
import warmpool

router = APIRouter()
# ---------------------------------------------------------------------------
# 预热状态（前端切到训练页时查询 / 手动触发）
# ---------------------------------------------------------------------------
@router.get("/api/warm/status")
def warm_status() -> dict:
    """预热状态：ready 表示下一次点击训练能立刻开训。"""
    return warmpool.pool.status()


@router.post("/api/warm/prewarm")
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
@router.get("/api/health")
def health() -> dict:
    """存活探针，同时回报版本号。

    ``version`` 是判断「页面连的后端是不是最新的」的唯一可靠依据：
    端口被旧进程占着时，页面一切正常，只是版本号停在旧值。
    """
    return {"ok": True, "version": config.APP_VERSION}


@router.get("/api/capabilities")
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
# 文件浏览（Web 无法像桌面端那样弹系统选择框，提供受限于仓库内的浏览）
# ---------------------------------------------------------------------------
@router.get("/api/browse")
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


@router.post("/api/browse/dialog")
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


@router.post("/api/browse/cancel")
def browse_cancel(request: Request) -> dict:
    """让当前原生对话框自己关掉。

    窗口没出现时，网页上的按钮会改成「取消」。点它不能只把浏览器请求掐断，
    服务端那次选择还占着锁，下一次点「浏览」会一直得到「已经有一个选择框」。
    """
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="只能取消本机打开的选择框")
    return {"ok": True, "cancelled": nativedialog.request_cancel()}


# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------
frontend_index = config.FRONTEND_DIR / "index.html"


@router.get("/")
def index() -> FileResponse:
    """返回单页入口。"""
    if not frontend_index.is_file():
        raise HTTPException(status_code=500, detail="前端文件缺失：frontend/index.html")
    return FileResponse(frontend_index)
