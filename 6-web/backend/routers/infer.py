# -*- coding: utf-8 -*-
"""推理域路由：YOLO / Anomalib / OCR 单图与批量、视频、帧、产物下载、工作进程管理。"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import bundle as bundle_mod
import config
import infer
import inferjobs
import inferproxy
import labels as labels_mod
import ocr_infer
import ocr_models
import onnx_fallback
import web_state

# 共享状态：并发闸门与路径白名单（见 web_state.py）
from web_state import _START_LOCK, _ensure_allowed_path, _reject_if_busy

router = APIRouter()
#: 单图推理返回的标注图 JPEG 质量。
#: 这张图只用于导出（查看时前端用本地原图 + SVG 叠加层），所以让体积优先：
#: 3034x1586 的 PCB 图上 q=90 是 0.78 MB / 12 ms，q=100 也只到 2.4 MB 却慢一倍。
ANNOTATED_JPEG_QUALITY = 90


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def _read_upload(upload: UploadFile) -> bytes:
    """把上传的图片读进内存，并校验后缀与大小。

    v1.30.3 起不再落临时文件：以前 `_save_upload` 会写一份到 `UPLOAD_DIR`，
    但三个调用方都只是拿它的字节去 `decode_image`，文件从头到尾没被读过，
    最后在 `finally` 里删掉。白写一次盘（大图几 MB）还多一套清理逻辑。

    为什么要自己读 `upload.file` 而不是 `await upload.read()`：
    这几个端点都是同步 `def`，FastAPI 会丢到线程池里跑，读 `file` 可以直接读。
    """
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    if suffix.lower() not in config.IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式：{suffix}")

    data = upload.file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"图片超过 {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制",
        )
    if not data:
        raise HTTPException(status_code=400, detail="上传的图片是空的")
    return data


def _note_server_stages(result: dict, decode_ms: float, call_ms: float) -> None:
    """把服务端的阶段耗时写进 `timing`，让「服务端 xx ms」可以继续往下拆。

    为什么需要这三个数：DLL 只上报 pre/inference/post，用户看到的「服务端」里
    还包含图片解码、工作进程往返、base64 与 JSON 组装。不拆开就只能猜，
    「OCR 往返 215 ms 是不是传图慢」这类问题每次都要重新量一遍。

    - `decode_ms`：读上传 + 图像解码（Web 进程内，任何任务都要付）
    - `ipc_ms`：工作进程往返中除掉 DLL 的部分（共享内存拷贝 + 管道 + 回包反序列化）
    - `encode_ms`：由 `_finish_infer` 反推（base64 + JSON + 响应组装）
    """
    timing = result.get("timing")
    if not isinstance(timing, dict):
        return
    timing["decode_ms"] = round(float(decode_ms), 1)
    dll_ms = float(timing.get("total_ms") or 0.0)
    timing["ipc_ms"] = round(max(0.0, float(call_ms) - dll_ms), 1)


def _finish_infer(result: dict, started: float) -> Response:
    """补齐 `timing.server_ms` / `payload_bytes` / `encode_ms` 后返回响应。

    为什么要有这些字段：DLL 只回 pre/inference/post 三个内核计时，用户在页面上
    看到的却是「含往返」。编图、base64、JSON 序列化和传输都不在那三个数里，
    于是出现「推理 5 ms、往返 388 ms」这种看起来像 bug 的对比。这里把服务端那段
    和响应体体积明确标出来，一眼能看出时间花在模型还是传输。

    `payload_bytes` 是**响应体的真实字节数（含本字段自身）**。只序列化一次：
    先把字段填 0 编码，再把 `"payload_bytes":0` 原地替换成真实长度。
    以前的做法是「填 0 编一次量长度、填真值再编一次」，对 2.5 MB 的异常检测响应
    等于白花一次 JSON 序列化（约 30 ms）。
    """
    timing = result.get("timing")
    if not isinstance(timing, dict):
        return JSONResponse(result)

    timing["server_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    # 响应组装 = 服务端总计 - 解码 - 工作进程往返；base64 与 JSON 都在这一段里
    decode_ms = float(timing.get("decode_ms") or 0.0)
    ipc_ms = float(timing.get("ipc_ms") or 0.0)
    dll_ms = float(timing.get("total_ms") or 0.0)
    timing["encode_ms"] = round(max(0.0, timing["server_ms"] - decode_ms - ipc_ms - dll_ms), 1)

    timing["payload_bytes"] = 0
    body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    placeholder = b'"payload_bytes":0'
    if placeholder in body:
        # 真实长度 = 占位体长度 - 1（'0'）+ 数字位数；位数变了要再算一次才收敛
        size = len(body)
        for _ in range(2):
            size = len(body) - 1 + len(str(size))
        body = body.replace(placeholder, b'"payload_bytes":' + str(size).encode(), 1)
        timing["payload_bytes"] = size
    else:
        # 理论上不会走到这里（字段是自己填的）；保底给个近似值，不要让字段空着
        timing["payload_bytes"] = len(body)
        body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return Response(content=body, media_type="application/json")


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


def _infer_worker_down_but_allowed() -> bool:
    """进程没在跑，但不是让出、也不是熔断。

    训练结束后进程是故意停着的，等下一次真正推理再拉起。
    这时状态里的 yolo_ok 是 False，只表示「还没问过 DLL」，
    不能当成 DLL 坏了去走降级，否则按需重载永远不会发生。
    """
    if not inferproxy.isolation_enabled():
        return False
    state = inferproxy.worker_state()
    if state.get("gpu_hold") or state.get("pool_tripped") or state.get("state") == "tripped":
        return False
    return state.get("state") != "running"


def _select_yolo_backend(engine: str) -> tuple[bool, str, str | None]:
    """决定 YOLO 这次走哪个后端，返回 (是否走 DLL, 兜底 ONNX 路径, 降级说明)。

    三级顺序：
    1. 传的是 `.onnx` 且 `onnx_infer.dll` 可用 → 走 ONNX DLL；
    2. 传的是引擎且 `yolo_infer.dll` 可用 → 走 TRT DLL；
    3. 以上都不可用 → onnxruntime 降级（只在找不到模型时抛 503）。

    让出期间直接 503，不走降级：降级会在网页进程里加载模型，把刚让出的显存又占回去。
    进程停着但没有让出时，仍选推理进程，由真正的推理调用把它拉起来。
    """
    inferproxy._refuse_if_yielded()
    if _infer_worker_down_but_allowed():
        return True, "", None
    backend = inferproxy.status()

    if infer.is_onnx_model(engine):
        if backend.onnx_ok:
            return True, "", None
        path = Path(engine)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"ONNX 模型不存在：{engine}")
        return False, str(path), _fallback_reason(
            "onnx_infer.dll", backend.onnx_detail or "未知原因"
        )

    if backend.yolo_ok:
        return True, "", None

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
    return False, onnx_path, _fallback_reason("yolo_infer.dll", backend.detail or "未知原因")


def _run_yolo_once(
    engine: str,
    task: str,
    array,
    conf: float,
    nms: float,
    labels: str = "",
    source: str = "",
) -> tuple[dict, str | None]:
    """执行一次 YOLO 推理并补齐类别名与配色，返回 (结果, 降级说明)。

    抽出来的原因：单图接口与摄像头接口必须走**完全同一套**后端选择与
    输出结构。摄像头每秒调 10 次，如果它是"另一条实现"，两边框的顺序、
    字段名、降级行为迟早对不上，而屏幕上看起来都"能出框"，极难发现。
    """
    label_info = labels_mod.load_labels(labels)
    if task == "classify":
        import classifyinfer

        # 分类优先走 onnx_infer.dll（v1.30.6），失败或旧 DLL 才退回 onnxruntime。
        # 两条路径共用 classifyinfer.interpret 做「分数 → 类别」的判定，
        # 差别只在算分数的地方，不在解释分数的规则。
        result = None
        warning = None
        if infer.is_onnx_model(engine) and infer.classify_available():
            try:
                raw = inferproxy.run_classify(engine, array, source=source)
                result, warning = classifyinfer.result_from_raw(
                    raw, label_info["classes"], conf
                )
            except (infer.InferError, inferproxy.WorkerError) as exc:
                # DLL 不可用不能让分类直接失败：退回 Python 路径，并说明原因，
                # 否则用户只会看到"分类用不了"，不知道是 DLL 的问题。
                warning = f"onnx_infer.dll 分类不可用（{exc}），已退回 onnxruntime"
                result = None

        if result is None:
            try:
                result = classifyinfer.run(engine, array, label_info["classes"], conf)
            except classifyinfer.ClassifyError as exc:
                raise infer.InferError(str(exc)) from exc
            # 分类自己也会产生提示（标签顺序和模型 metadata 不一致），
            # 这里要原样交给调用方，否则前端只能看到「结果对了但不知道为什么」。
            python_warning = result.pop("warning", None)
            warning = warning or python_warning

        result["labels"] = {
            "path": labels or None,
            "ok": label_info["ok"],
            "name": label_info["name"],
            "count": len(label_info["classes"]),
            "error": label_info["error"],
        }
        return result, warning

    use_dll, onnx_path, warning = _select_yolo_backend(engine)
    try:
        if use_dll:
            result = inferproxy.run_yolo(engine, task, array, conf=conf, nms=nms, source=source)
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


def _apply_bundle(
    bundle_id: str,
    *,
    kind: str,
    task: str = "",
    engine: str = "",
    labels: str = "",
    metadata: str = "",
) -> dict[str, Any]:
    """把「选了模型包」翻译成推理接口要的一组路径。

    只有显式给了 ``bundle`` 才会走到这里，散装用法完全不受影响。
    校验为 ``block`` 时直接 400：这类错误是确定性的（哈希不符、种类冲突），
    放过去只会得到空结果或越界解码，比拒绝更难排查。
    """
    resolved = bundle_mod.resolve_bundle(bundle_id)
    if not resolved.get("ok"):
        raise HTTPException(status_code=400, detail=str(resolved.get("error") or "模型包不可用"))

    verdict = bundle_mod.verify_bundle(
        resolved["path"], kind=kind, task=task or None
    )
    if verdict["level"] == bundle_mod.LEVEL_BLOCK:
        reasons = "；".join(
            item["detail"] for item in verdict["checks"] if item["level"] == bundle_mod.LEVEL_BLOCK
        )
        raise HTTPException(status_code=400, detail=f"模型包校验不通过：{reasons}")

    return {
        "engine": resolved["model"],
        "task": task or str(resolved.get("task") or ""),
        "labels": labels or resolved.get("labels") or "",
        "metadata": metadata or resolved.get("metadata") or "",
        "bundle": {
            "id": Path(resolved["path"]).name,
            "path": resolved["path"],
            "kind": resolved.get("kind"),
            "level": verdict["level"],
            "warnings": [
                item["detail"] for item in verdict["checks"] if item["level"] == bundle_mod.LEVEL_WARN
            ],
        },
    }


@router.post("/api/infer/yolo")
def infer_yolo(
    image: UploadFile = File(...),
    engine: str = Form(""),
    task: str = Form("detect"),
    conf: float = Form(0.25),
    nms: float = Form(0.45),
    labels: str = Form(""),
    bundle: str = Form(""),
) -> JSONResponse:
    """YOLO 推理：返回检测框、类别名与标注后的结果图（base64 JPEG）。

    DLL 优先；DLL 不可用时自动降级到 onnxruntime，返回结构完全一致。
    `labels` 是标签文件路径，缺失时框上只有 `cls{id}`，不报错。

    `bundle` 与 `engine` 二选一：给了模型包就从包里取模型、标签与任务，
    页面不需要再填三个输入框，也不会出现「引擎换了标签没换」。

    标注图用 JPEG 而不是 PNG：一张 3034x1586 的 PCB 图，PNG 编码 155 ms、5.8 MB，
    JPEG(q=90) 只要 12 ms、0.78 MB，画面差异肉眼看不出来。这张图只用于导出，
    查看时前端用的是本地原图 + SVG 叠加层。
    """
    bundle_info: dict[str, Any] | None = None
    if bundle.strip():
        applied = _apply_bundle(
            bundle, kind="yolo", task=task, engine=engine, labels=labels,
        )
        engine = applied["engine"]
        task = applied["task"] or task
        labels = applied["labels"]
        bundle_info = applied["bundle"]
    if not engine.strip():
        raise HTTPException(status_code=400, detail="engine 与 bundle 必须提供其中一个")

    started = time.perf_counter()
    data = _read_upload(image)
    try:
        decoded_at = time.perf_counter()
        array = infer.decode_image(data)
        decode_ms = (time.perf_counter() - decoded_at) * 1000.0
        called_at = time.perf_counter()
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels, source=image.filename or "")
        call_ms = (time.perf_counter() - called_at) * 1000.0
        _note_server_stages(result, decode_ms, call_ms)
        if result.get("classify"):
            import classifyinfer
            # 直接拿 BGR 数组再编 JPEG，不要先用 overlay_png 编成 PNG 再解码
            canvas = classifyinfer.draw_label(array, result["classify"])
        else:
            # 先补标签再画图：render_yolo 要读 label_name 才能把类别名烧进去
            # 用 render_yolo（返回 BGR 数组）而不是 draw_yolo（返回 PNG 字节），
            # 省掉「编 PNG 再解码成数组编 JPEG」的往返。
            canvas = infer.render_yolo(array, result["boxes"], result["class_colors"])
        annotated = infer.encode_image(canvas, ".jpg", ANNOTATED_JPEG_QUALITY)
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import base64

    result["annotated_jpg"] = base64.b64encode(annotated).decode("ascii")
    result["annotated_mime"] = "image/jpeg"
    if warning:
        result["warning"] = warning
    if bundle_info:
        result["bundle"] = bundle_info
    return _finish_infer(result, started)


def _require_model_dir(raw: str, field: str) -> Path:
    """OCR 模型是目录，不是单个权重文件。"""
    if not str(raw or "").strip():
        raise HTTPException(status_code=400, detail=f"{field} 不能为空")
    path = _ensure_allowed_path(raw)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"{field} 不是目录：{path}")
    return path


@router.get("/api/ocr/models")
def list_ocr_models() -> dict:
    """扫描 OCR 模型根目录，标出格式和能不能在当前 Paddle 版本上用。"""
    try:
        return ocr_models.scan_models()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"扫描 OCR 模型失败：{exc}") from exc


class OcrWarmRequest(BaseModel):
    """OCR 预热参数。都可以省略：省略时自动挑根目录下第一套可用的 det/rec。"""

    det_dir: str = ""
    rec_dir: str = ""
    textline_ori_dir: str = ""
    device: str = "cpu"
    enable_mkldnn: int = 1
    cpu_threads: int = 8


#: 预热状态。前端不需要轮询它，留着方便排查"到底热没热上"。
_OCR_WARM_LOCK = threading.Lock()


_OCR_WARM_STATE: dict[str, Any] = {"running": False, "ok": None, "ms": None, "error": "", "source": ""}


def _pick_ocr_models() -> tuple[str, str, str]:
    """从扫描结果里挑一套可用的 det/rec（有 textline_ori 就一起带上）。"""
    scan = ocr_models.scan_models()
    det = rec = textline = ""
    for entry in scan.get("entries") or []:
        if not entry.get("compatible"):
            continue
        kind = entry.get("kind")
        if kind == "det" and not det:
            det = entry["path"]
        elif kind == "rec" and not rec:
            rec = entry["path"]
        elif kind == "textline_ori" and not textline:
            textline = entry["path"]
    return det, rec, textline


def _ocr_warm_worker(options: dict) -> None:
    """后台跑一次 32x32 空图，把 oi_create（加载飞桨 + 解析模型）提前做掉。

    为什么值得单独加个接口：`oi_create` 首次要 1.3 s 左右，且不计入 DLL 返回的
    timing。用户看到的是「点推理 -> 卡一秒多 -> 出结果」，会误判成模型慢。
    """
    import numpy as np

    started = time.perf_counter()
    try:
        inferproxy.run_ocr(
            np.zeros((32, 32, 3), dtype=np.uint8),
            options=options,
            source="预热",
        )
        state = {"running": False, "ok": True, "ms": round((time.perf_counter() - started) * 1000.0, 1),
                 "error": "", "source": "warm"}
    except Exception as exc:  # noqa: BLE001 - 预热失败不影响正常推理，只记状态
        state = {"running": False, "ok": False, "ms": round((time.perf_counter() - started) * 1000.0, 1),
                 "error": f"{type(exc).__name__}: {exc}", "source": "warm"}
    with _OCR_WARM_LOCK:
        _OCR_WARM_STATE.update(state)


@router.post("/api/ocr/warm")
def warm_ocr(body: OcrWarmRequest | None = None) -> dict:
    """预热 OCR 句柄：立刻返回，真正加载放到后台线程。

    幂等：已经在预热或已经热过时直接返回当前状态，不会重复建句柄。
    """
    body = body or OcrWarmRequest()
    with _OCR_WARM_LOCK:
        if _OCR_WARM_STATE["running"]:
            return {"started": False, "reason": "正在预热", **_OCR_WARM_STATE}
        if _OCR_WARM_STATE["ok"]:
            return {"started": False, "reason": "已经预热过", **_OCR_WARM_STATE}

    det = str(body.det_dir or "").strip()
    rec = str(body.rec_dir or "").strip()
    textline = str(body.textline_ori_dir or "").strip()
    if not det or not rec:
        picked_det, picked_rec, picked_textline = _pick_ocr_models()
        det = det or picked_det
        rec = rec or picked_rec
        textline = textline or picked_textline
    if not det or not rec:
        return {"started": False, "reason": "没有找到可用的检测/识别模型，先下载模型再预热"}

    options = {
        "det_dir": det,
        "rec_dir": rec,
        "textline_dir": textline,
        "device": body.device or "cpu",
        "enable_mkldnn": 1 if body.enable_mkldnn else 0,
        "cpu_threads": body.cpu_threads,
        "rec_batch_size": 6,
        "det_thresh": 0.3,
        "det_box_thresh": 0.6,
        "det_unclip_ratio": 1.5,
        "rec_score_thresh": 0.0,
    }
    with _OCR_WARM_LOCK:
        _OCR_WARM_STATE.update({"running": True, "ok": None, "ms": None, "error": "", "source": "warm"})
    threading.Thread(target=_ocr_warm_worker, args=(options,), name="ocr-warm", daemon=True).start()
    return {"started": True, "det_dir": det, "rec_dir": rec, "textline_dir": textline}


@router.post("/api/infer/ocr")
def infer_ocr(
    image: UploadFile = File(...),
    det_dir: str = Form(...),
    rec_dir: str = Form(...),
    textline_ori_dir: str = Form(""),
    device: str = Form("cpu"),
    enable_mkldnn: int = Form(1),
    cpu_threads: int = Form(8),
    rec_batch_size: int = Form(6),
    det_thresh: float = Form(0.3),
    det_box_thresh: float = Form(0.6),
    det_unclip_ratio: float = Form(1.5),
    rec_score_thresh: float = Form(0.0),
) -> JSONResponse:
    """OCR：检测、识别、方向分类都在 ocr_infer.dll 里完成。"""
    started = time.perf_counter()
    det = _require_model_dir(det_dir, "检测模型目录")
    rec = _require_model_dir(rec_dir, "识别模型目录")
    textline = ""
    if str(textline_ori_dir or "").strip():
        textline = str(_require_model_dir(textline_ori_dir, "方向模型目录"))

    data = _read_upload(image)
    try:
        decoded_at = time.perf_counter()
        array = infer.decode_image(data)
        decode_ms = (time.perf_counter() - decoded_at) * 1000.0
        called_at = time.perf_counter()
        result = inferproxy.run_ocr(array, source=image.filename or "", options={
            "det_dir": str(det),
            "rec_dir": str(rec),
            "textline_dir": textline,
            "device": device,
            "enable_mkldnn": 1 if enable_mkldnn else 0,
            "cpu_threads": cpu_threads,
            "rec_batch_size": rec_batch_size,
            "det_thresh": det_thresh,
            "det_box_thresh": det_box_thresh,
            "det_unclip_ratio": det_unclip_ratio,
            "rec_score_thresh": rec_score_thresh,
        })
        call_ms = (time.perf_counter() - called_at) * 1000.0
        _note_server_stages(result, decode_ms, call_ms)
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ocr_infer.OcrError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return _finish_infer(result, started)


def _select_anomalib_backend(engine: str, metadata: str) -> tuple[bool, str, str | None]:
    """与 `_select_yolo_backend` 同样的三级顺序，只是主库换成 anomalib_infer。"""
    backend = inferproxy.status()

    if infer.is_onnx_model(engine):
        if backend.onnx_ok:
            return True, "", None
        path = Path(engine)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"ONNX 模型不存在：{engine}")
        return False, str(path), _fallback_reason(
            "onnx_infer.dll", backend.onnx_detail or "未知原因"
        )

    if backend.anomalib_ok:
        return True, "", None

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
    return False, onnx_path, _fallback_reason("anomalib_infer.dll", backend.detail or "未知原因")


@router.post("/api/infer/anomalib")
def infer_anomalib(
    image: UploadFile = File(...),
    engine: str = Form(""),
    metadata: str = Form(""),
    model_type: str = Form("Auto"),
    image_threshold: float = Form(-1.0),
    pixel_threshold: float = Form(-1.0),
    heatmap_colormap: str = Form("TURBO"),
    bundle: str = Form(""),
) -> JSONResponse:
    """Anomalib 推理：返回分数、判定、叠加图、轮廓图与热能图（base64）。

    不回传原图：前端切图时已经显示本地原图，后端再编一张全尺寸 PNG 是纯重复
    （3034x1586 上要 155 ms + 7.1 MB）。宽高仍在 `image_width/height` 里返回。

    阈值语义与 DLL 一致：`>= 0` 覆盖 metadata，`< 0` 沿用 metadata。
    响应里 `thresholds` 同时给出"本次生效值"与"metadata 文件原值"，
    因为 `ai_get_metadata` 拿到的已经是覆盖后的值，原值只能自己读文件。

    `bundle` 与 `engine` 二选一：给了模型包就用包里的引擎与 metadata。
    """
    bundle_info: dict[str, Any] | None = None
    if bundle.strip():
        applied = _apply_bundle(bundle, kind="anomalib", engine=engine, metadata=metadata)
        engine = applied["engine"]
        metadata = applied["metadata"]
        bundle_info = applied["bundle"]
    if not engine.strip():
        raise HTTPException(status_code=400, detail="engine 与 bundle 必须提供其中一个")
    if not metadata.strip():
        raise HTTPException(status_code=400, detail="metadata 与 bundle 必须提供其中一个")

    use_dll, onnx_path, warning = _select_anomalib_backend(engine, metadata)
    started = time.perf_counter()

    data = _read_upload(image)
    try:
        decoded_at = time.perf_counter()
        array = infer.decode_image(data)
        decode_ms = (time.perf_counter() - decoded_at) * 1000.0
        called_at = time.perf_counter()
        if use_dll:
            result = inferproxy.run_anomalib(
                engine, metadata, array,
                model_type=model_type,
                image_threshold=image_threshold,
                pixel_threshold=pixel_threshold,
                colormap=heatmap_colormap,
                source=image.filename or "",
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
            result.setdefault("heatmap_colormap", None)
            result.setdefault("thresholds", {
                "image": {"effective": None, "metadata": None, "overridden": False},
                "pixel": {"effective": None, "metadata": None, "overridden": False},
            })
        call_ms = (time.perf_counter() - called_at) * 1000.0
        _note_server_stages(result, decode_ms, call_ms)
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import base64

    for key in ("overlay_jpg", "mask_png", "heatmap_png"):
        value = result.get(key)
        result[key] = base64.b64encode(value).decode("ascii") if value else None
    # MIME 决定前端用哪种方式挂图；overlay 已经改成 JPEG，其他两张仍是 PNG
    result.setdefault("overlay_mime", "image/jpeg" if result.get("overlay_jpg") else None)
    if warning:
        result["warning"] = warning
    if bundle_info:
        result["bundle"] = bundle_info
    return _finish_infer(result, started)


class BatchInferRequest(BaseModel):
    """目录批量推理请求。只接受服务端可访问的路径，不做文件上传。

    为什么不做上传：批量动辄几千张图、上百 MB，走 HTTP 上传等于把数据
    先拷一遍到临时目录再读一遍；前后端本来就在同一台机器上，路径更直接。
    """

    kind: str = Field(default="yolo", description="yolo / anomalib / ocr")
    engine: str = Field(default="", description=".engine / .trt / .onnx；OCR 留空")
    det_dir: str = Field(default="", description="OCR 检测模型目录")
    rec_dir: str = Field(default="", description="OCR 识别模型目录")
    textline_ori_dir: str = Field(default="", description="OCR 文本行方向模型目录，可留空")
    device: str = Field(default="cpu", description="cpu 或 gpu:0")
    enable_mkldnn: int = Field(default=1)
    cpu_threads: int = Field(default=8, ge=1, le=64)
    rec_batch_size: int = Field(default=6, ge=1, le=64)
    det_thresh: float = Field(default=0.3, ge=0.0, le=1.0)
    det_box_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    det_unclip_ratio: float = Field(default=1.5, ge=0.5, le=5.0)
    rec_score_thresh: float = Field(default=0.0, ge=0.0, le=1.0)
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

    kind: str = Field(default="yolo", description="只支持 yolo；ocr 会被拒绝")
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


@router.get("/api/infer/anomalib/metadata")
def inspect_anomaly_metadata(path: str = "") -> dict:
    """只读 `metadata.json`，把里面的推荐阈值提前暴露给推理页。

    为什么需要这个接口：勾选「沿用 metadata 默认阈值」时两个阈值输入框是
    只读的，值来自 metadata 文件——但文件内容在界面上没有任何出口，用户只能
    先跑一次推理、再去看结果行里的"（metadata）"才间接看到数字，等于把推荐值
    藏起来了。这个接口只读文件、不建引擎句柄，可以随路径改动随时刷新。
    """
    target = _require_file(path, "metadata.json")
    info = infer.read_metadata_thresholds(str(target))
    if info.get("error"):
        raise HTTPException(status_code=400, detail=info["error"])
    return {"path": str(target), **info}


def build_batch_command(
    request: "BatchInferRequest",
    python: str,
    job_dir: str | Path,
) -> list[str]:
    """组装批量推理命令。

    单独成函数是为了能被测试直接断言：参数在表单里填了却没进命令行，
    用户只能从日志里翻完整命令才能发现（`--simplify true` 就是这么漏的）。
    """
    if request.kind not in ("yolo", "anomalib", "ocr"):
        raise HTTPException(status_code=400, detail=f"kind 只能是 yolo / anomalib / ocr，实际 {request.kind}")

    if request.kind == "ocr":
        det = _require_model_dir(request.det_dir, "检测模型目录")
        rec = _require_model_dir(request.rec_dir, "识别模型目录")
        source = _ensure_allowed_path(request.source)
        if not source.exists():
            raise HTTPException(status_code=400, detail=f"数据源不存在：{source}")
        command = config.tool_command(python, "infer_batch.py") + [
            "--kind", "ocr",
            "--det-dir", str(det),
            "--rec-dir", str(rec),
            "--device", request.device,
            "--cpu-threads", str(int(request.cpu_threads)),
            "--rec-batch-size", str(int(request.rec_batch_size)),
            "--det-thresh", f"{request.det_thresh:g}",
            "--det-box-thresh", f"{request.det_box_thresh:g}",
            "--det-unclip-ratio", f"{request.det_unclip_ratio:g}",
            "--rec-score-thresh", f"{request.rec_score_thresh:g}",
            "--source", str(source),
            "--output", str(job_dir),
            "--limit", str(int(request.limit)),
            "--jpeg-quality", str(int(request.jpeg_quality)),
        ]
        if str(request.textline_ori_dir or "").strip():
            textline = _require_model_dir(request.textline_ori_dir, "方向模型目录")
            command += ["--textline-dir", str(textline)]
        if not request.enable_mkldnn:
            command.append("--no-mkldnn")
        if not request.recursive:
            command.append("--no-recursive")
        if not request.make_zip:
            command.append("--no-zip")
        if not request.full_json:
            command.append("--no-full-json")
        return command

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

    command = config.tool_command(python, "infer_batch.py") + [
        "--kind", request.kind,
        "--engine", str(engine),
        "--source", str(source),
        "--output", str(job_dir),
        "--limit", str(int(request.limit)),
        "--jpeg-quality", str(int(request.jpeg_quality)),
    ]

    if request.kind == "yolo":
        if request.task == "classify" and engine.suffix.lower() in {".engine", ".trt", ".plan"}:
            raise HTTPException(status_code=400, detail="分类引擎还没做，请改用 ONNX。")
        if request.task not in ("detect", "segment", "pose", "obb", "classify"):
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
    if request.kind == "ocr":
        raise HTTPException(status_code=400, detail="OCR 不支持视频推理：CPU 单张约 10 秒以上，请改用目录批量")
    engine = _require_file(request.engine, "引擎文件")
    if engine.suffix.lower() not in ENGINE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"引擎格式不支持：{engine.suffix}")
    if request.task == "classify" and engine.suffix.lower() in {".engine", ".trt", ".plan"}:
        raise HTTPException(status_code=400, detail="分类引擎还没做，请改用 ONNX。")
    if request.task not in ("detect", "segment", "pose", "obb", "classify"):
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

    command = config.tool_command(python, "infer_video.py") + [
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


@router.post("/api/infer/batch")
def infer_batch(request: BatchInferRequest) -> dict:
    """启动目录批量推理，返回 run_id；进度经 `/ws/runs/{run_id}` 推送。"""
    config.ensure_dirs()
    try:
        job = inferjobs.new_job_dir("batch")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"创建产物目录失败：{exc}") from exc

    command = build_batch_command(request, sys.executable, job)
    expect = (job / "annotated.zip") if request.make_zip else (job / "results.csv")

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = web_state.manager.create(
                "infer-batch",
                command,
                cwd=str(config.BACKEND_DIR),
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


@router.post("/api/infer/video")
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

    with _START_LOCK:
        _reject_if_busy()
        try:
            run = web_state.manager.create(
                "infer-video",
                command,
                cwd=str(config.BACKEND_DIR),
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


@router.post("/api/infer/frame")
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
        result, warning = _run_yolo_once(engine, task, array, conf, nms, labels, source="摄像头帧")
    except infer.InferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.pop("annotated_png", None)
    result.pop("annotated_jpg", None)
    if warning:
        result["warning"] = warning
    return JSONResponse(result)


@router.get("/api/infer/artifact")
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


@router.get("/api/infer/cache")
def infer_cache() -> dict:
    """查看当前推理句柄缓存（只读诊断）。

    存在的理由：句柄常驻显存，用户改阈值改到显存告急时需要能看见"现在到底留了几个"。
    没有这个窗口，用户只能靠猜。
    """
    stats = inferproxy.cache_stats()
    return {"ok": True, **stats, "ocr": stats.get("ocr", {}), "worker": inferproxy.worker_state()}


@router.post("/api/infer/worker/restart")
def infer_worker_restart() -> dict:
    """熔断后由用户手动重启推理进程。训练占着显存时会拒绝，避免把显存抢回去。"""
    return {"ok": True, **inferproxy.restart_worker()}


@router.post("/api/infer/gpu-yield")
def infer_gpu_yield(reason: str = "训练") -> dict:
    """停掉推理进程，把显存让给训练。自动实验开始时调用。"""
    return inferproxy.yield_gpu(reason)


@router.post("/api/infer/gpu-release")
def infer_gpu_release() -> dict:
    """训练结束：允许下次推理按需重新加载，此刻不拉起进程。"""
    return inferproxy.release_gpu_hold()
