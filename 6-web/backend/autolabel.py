"""自动标注：双通道编排。

主通道用自有的 ``yolo_infer.dll``（或 onnxruntime 降级路径）出检测框，零新增依赖；
次通道通过 HTTP 调一个**独立进程**的外部标注服务，用来接 SAM / 文字提示这类重模型。

为什么次通道只能走 HTTP 而不是 import：`8-X-AnyLabeling-main` 是 GPL-3.0，
import 进来就等于把它变成产品的一部分，分发时整个程序都要开源。走进程边界
只构成"通过网络使用服务"，不派生作品。所以本模块只定义协议、只发请求，
**不包含也不打包任何外部服务代码**。

外部服务的接口契约（最小集）：

    POST {service}/predict        multipart/form-data
        file      : 图片二进制
        prompts   : JSON 字符串，形如 {"text": "hole", "points": [[x,y]], "model": "sam2"}
    ->  200 {"shapes": [ {"label", "shape_type", "points", "group_id", "score", "rotation"} ]}

`shape_type` 允许 ``rectangle`` / ``polygon`` / ``point`` / ``keypoints`` 等，
与 `.tfanno` 的字符串映射一致（``annotation.shape_from_str`` 会做别名兼容）。
"""

from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import annotation
import labels as labels_mod

#: 外部服务请求超时（秒）。SAM 推理在 CPU 上可能十几秒，给宽松一点。
DEFAULT_TIMEOUT = 20.0


class AutoLabelError(Exception):
    """自动标注失败。``kind`` 用于前端区分提示文案。"""

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind  # unreachable / timeout / http-error / bad-response / no-service


# ---------------------------------------------------------------------------
# 主通道：自有 YOLO 推理
# ---------------------------------------------------------------------------

def drafts_from_yolo(result: dict, task: str = "detect") -> list[dict]:
    """把 ``infer.run_yolo()`` 的返回转成标注草稿。

    输入结构以 `infer.run_yolo` 的实际输出为准：每个框有
    ``left/top/right/bottom`` / ``class_label`` / ``confidence`` /
    ``label_name``（由 ``attach_labels`` 补）/ 可选 ``is_obb``+``obb`` / ``kpts``。

    - ``detect``：每个框一个矩形草稿（2 点）。
    - ``obb``：用 4 个角点算包围盒当矩形，并把 ``angle`` 写进 ``rotation``，
      这样接受后仍是可编辑的旋转矩形。
    - ``pose``：全部关键点聚合成**一个** keypoints 草稿。
    - ``segment``：DLL 只给 ``has_mask`` 标志不给掩码顶点，因此退回矩形，
      并在草稿上打 ``_note`` 说明原因，避免用户以为"分割没生效"。
    """
    task = annotation.task_to_str(task)
    boxes = result.get("boxes") or []
    drafts: list[dict] = []

    if task == "pose":
        drafts.extend(_pose_drafts(boxes))
        return drafts

    for index, box in enumerate(boxes):
        left = float(box.get("left", 0.0) or 0.0)
        top = float(box.get("top", 0.0) or 0.0)
        right = float(box.get("right", 0.0) or 0.0)
        bottom = float(box.get("bottom", 0.0) or 0.0)
        # 有的实现会给反向坐标，这里统一成正序，否则矩形会"负面积"
        x1, x2 = (left, right) if left <= right else (right, left)
        y1, y2 = (top, bottom) if top <= bottom else (bottom, top)

        shape = "rectangle"
        rotation = 0.0
        if box.get("is_obb") and isinstance(box.get("obb"), dict):
            obb = box["obb"]
            cx, cy = float(obb.get("cx", (x1 + x2) / 2)), float(obb.get("cy", (y1 + y2) / 2))
            half_w = abs(float(obb.get("w", x2 - x1))) / 2.0
            half_h = abs(float(obb.get("h", y2 - y1))) / 2.0
            shape = "rotated_rect"
            rotation = float(obb.get("angle", 0.0) or 0.0)
            points = [[cx - half_w, cy - half_h], [cx + half_w, cy + half_h]]
        else:
            points = [[x1, y1], [x2, y2]]

        draft = _draft(
            shape=shape,
            points=points,
            class_id=int(box.get("class_label", 0) or 0),
            label=str(box.get("label_name") or box.get("label") or ""),
            score=float(box.get("confidence", -1.0) or -1.0),
            rotation=rotation,
            group_id=index + 1,
        )
        if shape != "rotated_rect":
            draft["rotation"] = 0.0
        if task == "segment" and box.get("has_mask"):
            draft["_note"] = "引擎只返回掩码标志，未给出顶点，已按包围盒转为矩形"
        drafts.append(draft)
    return drafts


def _pose_drafts(boxes: list[dict]) -> list[dict]:
    """姿态结果聚合：同一 frame 的全部关键点放进一个对象。

    Qt 的模型是"一个对象装全部关键点 + keypoint_visibility 数组"，
    拆成多个 ``point`` 对象的话 Qt 打开后就没法再编辑姿态了。
    """
    points: list[list[float]] = []
    visibility: list[int] = []
    class_id = 0
    label = ""
    for box in boxes:
        kpts = box.get("kpts")
        if not isinstance(kpts, list):
            continue
        for kpt in kpts:
            try:
                # DLL 的 kpts 是 [x, y, v] 三元组
                x, y = float(kpt[0]), float(kpt[1])
                v = int(kpt[2]) if len(kpt) > 2 else 2
            except (IndexError, TypeError, ValueError):
                continue
            points.append([x, y])
            visibility.append(v)
        if not class_id:
            class_id = int(box.get("class_label", 0) or 0)
        if not label:
            label = str(box.get("label_name") or "")

    if not points:
        return []
    draft = _draft(shape="keypoints", points=points, class_id=class_id, label=label, score=-1.0, group_id=1)
    draft["keypoint_visibility"] = visibility
    return [draft]


# ---------------------------------------------------------------------------
# 次通道：外部标注服务
# ---------------------------------------------------------------------------

def service_configured(service: str | None) -> bool:
    return bool((service or "").strip())


def _endpoint(service: str) -> str:
    base = service.strip().rstrip("/")
    if base.endswith("/predict"):
        return base
    return f"{base}/predict"


def external_predict(
    service: str,
    image_path: str | Path,
    prompts: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict]:
    """向外部服务提交一张图，取回形状列表。

    错误分类是刻意做细的：用户只有知道"是没启动 / 网络不通 / 服务返回了坏数据"，
    才知道该去修哪一步。全部失败都会抛 :class:`AutoLabelError`，由上层决定是否回退。
    """
    if not service_configured(service):
        raise AutoLabelError("未配置外部标注服务地址", "no-service")

    path = Path(image_path)
    if not path.is_file():
        raise AutoLabelError(f"图片不存在：{path}", "bad-response")

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AutoLabelError(f"读取图片失败：{exc}", "bad-response") from exc

    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    boundary = f"----trtforge{uuid.uuid4().hex}"
    body = _multipart(boundary, path.name, payload, mime, prompts or {})

    request = urllib.request.Request(
        _endpoint(service),
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise AutoLabelError(f"外部服务返回 HTTP {exc.code}：{exc.reason}", "http-error") from exc
    except urllib.error.URLError as exc:
        raise AutoLabelError(
            f"外部标注服务不可达（{service}）：{exc.reason}。请先启动服务，或关闭外部通道。",
            "unreachable",
        ) from exc
    except TimeoutError as exc:
        raise AutoLabelError(f"外部标注服务超时（{timeout:.0f}s）", "timeout") from exc
    except OSError as exc:
        raise AutoLabelError(f"调用外部标注服务失败：{exc}", "unreachable") from exc

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise AutoLabelError(f"外部服务返回的不是 JSON：{raw[:160]!r}", "bad-response") from exc

    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        raise AutoLabelError("外部服务返回里没有 shapes 数组", "bad-response")
    return shapes


def _multipart(boundary: str, filename: str, payload: bytes, mime: str, prompts: dict) -> bytes:
    """手工拼 multipart。用 stdlib 而不引 requests，与项目"零新增依赖"的基调一致。"""
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
    )
    parts.append(payload)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="prompts"\r\n')
    parts.append(b"Content-Type: application/json\r\n\r\n")
    parts.append(json.dumps(prompts, ensure_ascii=False).encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


def drafts_from_shapes(shapes: list[dict], task: str = "detect") -> list[dict]:
    """把外部服务返回的形状列表转成草稿。

    先按 ``group_id`` 聚合姿态散点（外部服务是"一点一个对象"），
    再逐个走 ``annotation.normalize_object`` 做形状回退与矩形折叠，
    保证两条通道产出的草稿结构完全一致。
    """
    task = annotation.task_to_str(task)
    warnings: list[str] = []
    drafts: list[dict] = []

    groups: dict[int, list[dict]] = {}
    singles: list[dict] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        shape_type = str(shape.get("shape_type") or shape.get("type") or "rectangle").lower()
        group_id = int(shape.get("group_id", 0) or 0)
        if task == "pose" and shape_type in ("point", "keypoint") and group_id:
            groups.setdefault(group_id, []).append(shape)
            continue
        if group_id:
            groups.setdefault(group_id, []).append(shape)
        else:
            singles.append(shape)

    for group_id, members in groups.items():
        if task == "pose" and all(
            str(m.get("shape_type") or m.get("type") or "").lower() in ("point", "keypoint") for m in members
        ):
            points = []
            visibility = []
            for member in members:
                pair = member.get("points") or []
                if isinstance(pair, list) and len(pair) == 1 and isinstance(pair[0], (list, tuple)):
                    pair = pair[0]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    points.append([float(pair[0]), float(pair[1])])
                    visibility.append(int(member.get("visibility", 2)))
            if points:
                draft = _draft(
                    shape="keypoints",
                    points=points,
                    class_id=int(members[0].get("class_id", 0) or 0),
                    label=str(members[0].get("label") or ""),
                    score=-1.0,
                    group_id=group_id,
                )
                draft["keypoint_visibility"] = visibility
                drafts.append(draft)
            continue
        # 非姿态的组：按组内第一个对象处理（外部服务通常一个形状一个 group，不应出现多成员）
        singles.extend(members)

    for shape in singles:
        try:
            obj = annotation.normalize_object(shape, warnings)
        except annotation.AnnotationError:
            continue
        obj["id"] = obj.get("id") or _draft_id()
        obj["class_id"] = int(shape.get("class_id", 0) or 0)
        drafts.append(obj)

    for draft in drafts:
        draft.setdefault("_draft", True)
    if warnings:
        drafts[0].setdefault("_warnings", warnings) if drafts else None
    return drafts


# ---------------------------------------------------------------------------
# 统一草稿构造
# ---------------------------------------------------------------------------

def _draft(
    shape: str,
    points: list[list[float]],
    class_id: int = 0,
    label: str = "",
    score: float = -1.0,
    rotation: float = 0.0,
    group_id: int = 0,
) -> dict:
    return {
        "id": _draft_id(),
        "class_id": int(class_id),
        "label": label,
        "description": "",
        "group_id": int(group_id),
        "shape_type": annotation.shape_to_str(shape),
        "points": [[float(p[0]), float(p[1])] for p in points],
        "rotation": float(rotation),
        "keypoint_visibility": [],
        "score": float(score),
        "_draft": True,
    }


def _draft_id() -> str:
    return f"draft_{uuid.uuid4().hex[:10]}"


def sanitize_drafts(
    drafts: list[dict],
    width: int,
    height: int,
    warnings: list[str] | None = None,
) -> list[dict]:
    """过滤明显不合理的草稿，并记一条警告说明原因。

    典型场景：``.engine`` 的输出布局与当前 ``yolo_infer.dll`` 不匹配时，
    解出来的框会落在图像外、置信度也不在 ``[0,1]``。这种草稿直接进标注层
    等于把错数据写进工程，所以宁可丢掉并提示，也不要"看起来有结果"。

    判定规则（宁可少放行，也不放脏数据）：
    - 置信度 > 1：正则不可能是概率，说明解码错了；
    - 所有点都落在图像外的半个平面内（完全不可见）。
    """
    if not drafts:
        return drafts

    width = max(0, int(width or 0))
    height = max(0, int(height or 0))
    kept: list[dict] = []
    bad_score = 0
    out_of_image = 0

    for draft in drafts:
        try:
            score = float(draft.get("score", -1.0))
        except (TypeError, ValueError):
            score = -1.0
        points = draft.get("points") or []
        if score > 1.0 or not points:
            bad_score += 1
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if width and height:
            if max(xs) < 0 or min(xs) > width or max(ys) < 0 or min(ys) > height:
                out_of_image += 1
                continue
        kept.append(draft)

    if warnings is not None and (bad_score or out_of_image):
        parts = []
        if bad_score:
            parts.append(f"{bad_score} 个置信度不在 0~1")
        if out_of_image:
            parts.append(f"{out_of_image} 个框完全落在图像外")
        warnings.append(
            "已丢弃 " + "、".join(parts) + " 的草稿："
            "通常是 .engine 的输出布局与当前 yolo_infer.dll 不匹配，建议先用「推理演示」核对该引擎"
        )
    return kept


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def run(
    image_path: str,
    task: str,
    engine: str,
    labels_path: str | None = None,
    conf: float = 0.25,
    nms: float = 0.45,
    service: str | None = None,
    prompts: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """执行一次预标注，返回草稿与过程说明。

    次通道失败时**回退主通道并把原因原样带回**（字段 ``external_error``），
    绝不静默——否则用户会以为"模型只找到 2 个"，实际是服务没启动。
    """
    import cv2
    import infer  # 延迟导入：让不装 DLL 的环境也能 import 本模块做单测

    warnings: list[str] = []
    external_error: dict | None = None
    channel = "yolo"

    if service_configured(service):
        try:
            shapes = external_predict(service, image_path, prompts, timeout)
            drafts = drafts_from_shapes(shapes, task)
            if drafts:
                return {
                    "channel": "external",
                    "drafts": drafts,
                    "warnings": warnings,
                    "external_error": None,
                    "count": len(drafts),
                }
            warnings.append("外部服务返回了空结果，已回退到自有推理")
        except AutoLabelError as exc:
            external_error = {"kind": exc.kind, "message": str(exc)}
            warnings.append(f"外部服务不可用，已回退到自有推理：{exc}")

    try:
        image = cv2.imdecode(_read_bytes(image_path), cv2.IMREAD_COLOR)
    except cv2.error as exc:  # 解码失败要按"业务错误"返回，而不是抛成 500
        raise AutoLabelError(f"图片解码失败：{exc}", "bad-response") from exc
    if image is None:
        raise AutoLabelError(f"图片无法解码：{image_path}", "bad-response")

    result = infer.run_yolo(engine_path=engine, task=annotation.task_to_str(task), image=image, conf=conf, nms=nms)
    if labels_path:
        loaded = labels_mod.load_labels(labels_path)
        if loaded["ok"]:
            result = infer.attach_labels(result, loaded["classes"])
        else:
            # 标签文件坏了不该让预标注整条挂掉：退回类别号，并明确告知。
            warnings.append(f"标签文件不可用，类别名将显示为类别号：{loaded['error']}")
            result = infer.attach_labels(result, None)
    else:
        warnings.append("未提供标签文件，类别名将显示为类别号")

    drafts = drafts_from_yolo(result, task)
    drafts = sanitize_drafts(drafts, image.shape[1], image.shape[0], warnings)
    return {
        "channel": channel,
        "drafts": drafts,
        "warnings": warnings,
        "external_error": external_error,
        "count": len(drafts),
        "inference": {
            "backend": result.get("backend"),
            "inference_ms": (result.get("timing") or {}).get("inference_ms", 0),
        },
    }


def _read_bytes(path: str | Path):
    """读图片原始字节（uint8 ndarray）。路径含中文时用 ``np.fromfile`` 更稳
    （``cv2.imread`` 在部分 Windows 盘符下会因为编码失败）。

    注意返回的是 ndarray 而不是 bytes：``cv2.imdecode`` 只接受 numpy 数组，
    传 bytes 会报 "buf is not a numpy array"。
    """
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise AutoLabelError(f"读取图片失败：{exc}", "bad-response") from exc
    if data.size == 0:
        raise AutoLabelError(f"图片是空文件：{path}", "bad-response")
    return data
