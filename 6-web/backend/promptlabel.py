# -*- coding: utf-8 -*-
"""提示式预标注的 SAM 通道：点 / 框 → 多边形草稿。

为什么单独一个模块：
    检测通道（`autolabel.py`）是"喂图 → 出框"，一路无状态；SAM 是"**编码一次、
    多次解码**"——同一张图上用户会连着点好几下，每次只改一个点。如果每次点击都
    重跑 encoder，1080P 图在 CPU 上要等好几秒，交互直接不可用。

    所以这里的 `SamSession` 负责把 embedding 缓存起来，`annotate()` 只跑 decoder。

坐标链路（三套坐标系，搞错就是"框永远偏一半"）：
    原图 (w0,h0)
      │  等比缩放 scale = min(in_h/h0, in_w/w0)，贴左上角，余下补 0
      ▼
    编码输入 (in_w, in_h)          ← encoder 吃这个尺寸
      │  decoder 的 orig_im_size 传 (in_h, in_w)
      ▼
    掩码 (in_w, in_h)
      │  warpAffine(inv(scale))   → 原图
      ▼
    原图坐标的多边形 → 草稿

模型输入尺寸为什么是**探测**出来的而不是写死：
    官方 SAM 导出是 1024×1024，而这里用的 MobileSAM 权重是 684×1024 的变体。
    写死任何一个，换一份权重就报 shape mismatch，而且报错点离真正原因很远
    （在 encoder.run 里，看不出是尺寸配置错了）。所以读取 ONNX 的静态输入维度，
    动态维才回退到 1024。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

#: 掩码转多边形时的默认近似系数（相对轮廓周长）。
DEFAULT_EPSILON = 0.001
#: 单条草稿的点数上限。`.tfanno` 里多边形是纯 JSON，一个 6 万点的轮廓能让
#: 工程文件膨胀到几十 MB，而画布上肉眼根本分不出 400 点和 4000 点。
MAX_POINTS = 400
#: 面积过滤：小于整图 0.01% 的轮廓基本是噪点。
MIN_AREA_RATIO = 0.0001
#: embedding 缓存条数。用户通常在同一张图上连续点，2 条足够覆盖"点完 A 图再回 B 图"。
CACHE_SIZE = 2


class PromptLabelError(Exception):
    """提示式预标注的可预期错误（接口层转 400）。"""


# ---------------------------------------------------------------------------
# 纯几何 / 纯 numpy 的工具（不需要模型，全部可单测）
# ---------------------------------------------------------------------------


def clamp_points(points, width: int, height: int) -> np.ndarray:
    """把点夹到图内。越界的提示点会让 SAM 的解码器输出垃圾掩码。"""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if array.size == 0:
        return array
    array[:, 0] = np.clip(array[:, 0], 0, max(0, width - 1))
    array[:, 1] = np.clip(array[:, 1], 0, max(0, height - 1))
    return array


def _point_list(value) -> list:
    """把"点集"统一成 ``[[x, y], ...]``。

    上游 `clamp_points()` 传进来的是 **numpy 数组**，这里不能再写 `value or []`：
    ndarray 的 `__bool__` 在元素多于一个时直接抛
    `ValueError: truth value of an array is ambiguous`。
    v1.24.1 就是这么被抓到的——单测都喂 list，集成路径全走数组，一跑真图就崩。
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return np.asarray(value, dtype=np.float64).reshape(-1, 2).tolist()
    return [[float(p[0]), float(p[1])] for p in value]


def _label_list(value) -> list:
    """把标签统一成 ``[int, ...]``。同样不能用 `value or []`。"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [] if value.size == 0 else np.asarray(value).reshape(-1).tolist()
    return list(value)


def build_prompt(points, labels, box) -> tuple[np.ndarray, np.ndarray]:
    """把"点 + 框"展开成 decoder 要的 ``(coords, labels)``。

    label 约定（与 SAM 官方导出脚本一致）：
        1  = 正点（这里要）
        0  = 负点（不是这里）
        2/3 = 框的左上 / 右下

    框用两个点表示，**不补成四个角**：decoder 认的就是 2/3 这组标签，
    补成四个角会让它按"四个正点"理解，出的是把框内四角包起来的怪形状。
    """
    coords: list[list[float]] = []
    marks: list[int] = []
    point_list = _point_list(points)
    mark_list = _label_list(labels)
    for index, point in enumerate(point_list):
        coords.append([float(point[0]), float(point[1])])
        # labels 可以比 points 短（前端省字段），缺的按正点处理
        marks.append(int(mark_list[index]) if index < len(mark_list) else 1)
    for index, point in enumerate(_point_list(box)):
        coords.append([float(point[0]), float(point[1])])
        marks.append(2 if index == 0 else 3)
    if not coords:
        raise PromptLabelError("没有可用的提示：请先在画布上点一个点或拉一个框")
    return np.asarray(coords, dtype=np.float64), np.asarray(marks, dtype=np.float64)


def preprocess_shape(image_h: int, image_w: int, in_h: int, in_w: int) -> tuple[float, tuple[int, int]]:
    """返回等比缩放系数与缩放后的画布尺寸 ``(h, w)``。"""
    if image_h <= 0 or image_w <= 0:
        raise PromptLabelError("图片尺寸非法")
    scale = min(in_h / image_h, in_w / image_w)
    new_h = max(1, int(round(image_h * scale)))
    new_w = max(1, int(round(image_w * scale)))
    return scale, (new_h, new_w)


def scale_points(points: np.ndarray, scale: float) -> np.ndarray:
    """原图坐标 → 编码输入坐标（纯等比缩放，无平移）。"""
    return np.asarray(points, dtype=np.float64).reshape(-1, 2) * float(scale)


def unscale_points(points: np.ndarray, scale: float) -> np.ndarray:
    """编码输入坐标 → 原图坐标。"""
    if scale <= 0:
        raise PromptLabelError("缩放系数非法")
    return np.asarray(points, dtype=np.float64).reshape(-1, 2) / float(scale)


def simplify_contour(contour: np.ndarray, epsilon: float, max_points: int = MAX_POINTS) -> list[list[float]]:
    """把一条轮廓近似成多边形点列。

    `epsilon` 是**相对**系数（乘周长）：绝对值在不同分辨率下效果完全不同，
    1080P 上合适的 2.0 在 200×200 的小图上会把整个缺陷抹平。
    """
    import cv2

    if contour is None or len(contour) < 3:
        return []
    arc = cv2.arcLength(contour, True)
    if arc <= 0:
        return []
    approx = cv2.approxPolyDP(contour, max(1e-6, float(epsilon)) * arc, True)
    points = [[float(p[0][0]), float(p[0][1])] for p in approx]

    # 近似后仍超上限：按等距抽稀，但要保证仍是闭合多边形（>=3 点）
    while len(points) > max_points:
        keep = max(3, int(len(points) * 0.7))
        step = len(points) / keep
        points = [points[int(i * step)] for i in range(keep)]
    return points


def mask_to_polygons(
    mask: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
    max_points: int = MAX_POINTS,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> list[list[list[float]]]:
    """二值掩码 → 多边形列表（每个多边形是一串 ``[x, y]``）。

    取**外轮廓**（`RETR_EXTERNAL`）：SAM 常在缺陷内部给出同心的"空洞"内轮廓，
    把它们也当成独立多边形会产生一堆套娃框，审核时分不清哪个才是缺陷本体。
    """
    import cv2

    if mask is None:
        return []
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.squeeze()
    if array.ndim != 2 or array.size == 0:
        raise PromptLabelError(f"掩码维度不支持：{array.shape}")

    binary = (array > 0.5).astype(np.uint8) if array.dtype != np.uint8 else (array > 0).astype(np.uint8)
    if binary.max() == 0:
        return []
    height, width = binary.shape[:2]
    min_area = max(1.0, float(min_area_ratio) * height * width)

    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    results: list[list[list[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = simplify_contour(contour, epsilon, max_points)
        if len(points) >= 3:
            results.append(points)
    # 面积大的先出：审核时最关心大缺陷，列表顺序与视觉重要性一致
    results.sort(key=lambda item: _polygon_area(item), reverse=True)
    return results


def _polygon_area(points: list[list[float]]) -> float:
    """鞋带公式。有它才能排序，也能在测试里断言"面积没变太多"。"""
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def mask_to_boxes(mask: np.ndarray, min_area_ratio: float = MIN_AREA_RATIO) -> list[list[float]]:
    """掩码 → 外接矩形 ``[x1, y1, x2, y2]``（文本通道细化失败时的降级用）。"""
    import cv2

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.squeeze()
    if array.ndim != 2 or array.size == 0:
        return []
    binary = (array > 0.5).astype(np.uint8) if array.dtype != np.uint8 else (array > 0).astype(np.uint8)
    if binary.max() == 0:
        return []
    height, width = binary.shape[:2]
    min_area = max(1.0, float(min_area_ratio) * height * width)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


# ---------------------------------------------------------------------------
# ONNX 会话
# ---------------------------------------------------------------------------


def _static_hw(shape) -> tuple[int, int] | None:
    """从 ONNX 输入维度里取静态 (H, W)；任一维为动态则返回 None。"""
    dims = list(shape or [])
    if len(dims) != 4:
        return None
    height, width = dims[2], dims[3]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


#: encoder 输入布局。MobileSAM 官方 ONNX 导出是 **HWC（rank 3）**，
#: 与网上教程里常见的 NCHW（rank 4）不同——见 `encoder_layout()`。
_ENCODER_HWC = "hwc"
_ENCODER_NCHW = "nchw"
_ENCODER_NHWC = "nhwc"


def encoder_layout(input_shape) -> str:
    """按模型的输入签名判断 encoder 要吃哪种布局。

    为什么必须判断而不是写死：
        `mobile_sam.encoder.onnx` 的输入声明是 ``['image_height', 'image_width', 3]``
        ——rank 3、通道在最后、H/W 还是动态维。照着常见的 ``[1, 3, H, W]`` 喂，
        onnxruntime 会直接拒收：
        ``Invalid rank for input: input_image Got: 4 Expected: 3``。
        而换一份别的 SAM 导出权重时，它可能真就是 NCHW，所以按签名走。
    """
    shape = list(input_shape or [])
    if len(shape) == 4:
        if shape[1] == 3:
            return _ENCODER_NCHW
        if shape[-1] == 3:
            return _ENCODER_NHWC
    # rank 3（HWC）与"动态维无法判断"都按 HWC 处理——本项目锁定的就是这个导出。
    return _ENCODER_HWC


class SamSession:
    """一个 SAM 权重对（encoder + decoder）的常驻会话。

    线程安全：`annotate()` 用 `lock` 串行化。批量任务与前端点击可能同时进来，
    encoder 内部有 session 状态，并发跑会得到错的 embedding（且不会报错）。
    """

    def __init__(self, encoder_path: str, decoder_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        import onnxenv

        self.encoder_path = str(encoder_path)
        self.decoder_path = str(decoder_path)
        # 用 onnxenv 探过的列表：缺 cuDNN 的机器上不会把 CUDA 交给 onnxruntime，
        # 否则每次建 session 都会打一段 LoadLibrary 失败的错误再退回 CPU。
        self.providers = providers or onnxenv.usable_providers(ort)
        self.encoder = ort.InferenceSession(self.encoder_path, providers=self.providers)
        self.decoder = ort.InferenceSession(self.decoder_path, providers=self.providers)
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.encoder_shape = self.encoder.get_inputs()[0].shape
        # 输入布局按签名判断：本项目的 MobileSAM 导出是 HWC rank-3，不是 NCHW。
        self.input_layout = encoder_layout(self.encoder_shape)
        # 探测编码器输入尺寸；动态维回退到官方 SAM 的 1024
        detected = _static_hw(self.encoder.get_inputs()[0].shape)
        self.input_h, self.input_w = detected or (1024, 1024)
        self.lock = threading.RLock()
        self._cache: list[tuple[tuple, dict]] = []

    @staticmethod
    def _default_providers(ort) -> list[str]:
        """兼容旧调用点：等价于 ``onnxenv.usable_providers()``。"""
        import onnxenv

        return onnxenv.usable_providers(ort)

    # -- embedding 缓存 -------------------------------------------------
    @staticmethod
    def _cache_key(path: str | Path) -> tuple:
        target = Path(path)
        try:
            stat = target.stat()
            return (str(target.resolve()), stat.st_mtime_ns, stat.st_size)
        except OSError:
            return (str(target), 0, 0)

    def embed(self, image_bgr: np.ndarray, key: tuple) -> dict:
        """编码一张图（带缓存）。返回 `{embedding, original_size, scale}`。"""
        with self.lock:
            for cached_key, value in self._cache:
                if cached_key == key:
                    self._cache.remove((cached_key, value))
                    self._cache.insert(0, (cached_key, value))  # LRU 提升
                    return value

        original_h, original_w = image_bgr.shape[:2]
        scale, (new_h, new_w) = preprocess_shape(original_h, original_w, self.input_h, self.input_w)
        canvas = np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
        resized = _resize(image_bgr, new_w, new_h).astype(np.float32)
        canvas[:new_h, :new_w] = resized
        # 注意：**不做归一化**。官方 SAM 导出的 encoder 内部已含 mean/std，
        # 外面再除 255 会让输出变成"一片空白"，表现为 mask 永远全 0。
        blob = self._to_blob(canvas)
        embedding = self.encoder.run(None, {self.encoder_input: blob})[0]
        value = {"embedding": embedding, "original_size": (original_h, original_w), "scale": scale}
        with self.lock:
            self._cache.insert(0, (key, value))
            del self._cache[CACHE_SIZE:]
        return value

    def _to_blob(self, canvas: np.ndarray) -> np.ndarray:
        """把 ``(H, W, 3)`` 画布整理成 encoder 期望的布局（含 batch 维与否）。"""
        if self.input_layout == _ENCODER_NCHW:
            return np.transpose(canvas, (2, 0, 1))[None, ...]
        if self.input_layout == _ENCODER_NHWC:
            return canvas[None, ...]
        return canvas

    # -- 解码 -----------------------------------------------------------
    def decode(self, state: dict, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """跑 decoder，返回原图尺寸的 2D 掩码。"""
        original_h, original_w = state["original_size"]
        scaled = scale_points(coords, state["scale"])

        # 补齐一个 padding 点（label=-1）是官方导出脚本的约定：
        # decoder 的输入点数是动态的，导出时按固定上限补位。
        coords_with_pad = np.concatenate([scaled, np.zeros((1, 2))], axis=0)[None, :, :]
        labels_with_pad = np.concatenate([labels, np.array([-1.0])], axis=0)[None, :].astype(np.float32)

        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask = np.zeros(1, dtype=np.float32)
        inputs = {
            "image_embeddings": state["embedding"],
            "point_coords": coords_with_pad.astype(np.float32),
            "point_labels": labels_with_pad,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
            "orig_im_size": np.array([self.input_h, self.input_w], dtype=np.float32),
        }
        outputs = self.decoder.run(None, inputs)
        masks, scores = outputs[0], (outputs[1] if len(outputs) > 1 else None)
        best = _pick_mask(masks, scores)
        return _to_original(best, state["scale"], original_h, original_w)

    def annotate(
        self,
        image_bgr: np.ndarray,
        image_key,
        points,
        labels,
        box,
        epsilon: float = DEFAULT_EPSILON,
        max_points: int = MAX_POINTS,
    ) -> list[list[list[float]]]:
        """点/框 → 原图坐标的多边形列表。"""
        coords, marks = build_prompt(points, labels, box)
        with self.lock:
            state = self.embed(image_bgr, image_key if isinstance(image_key, tuple) else self._cache_key(image_key))
            mask = self.decode(state, coords, marks)
        return mask_to_polygons(mask, epsilon=epsilon, max_points=max_points)

    def clear_cache(self) -> None:
        with self.lock:
            self._cache.clear()


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _pick_mask(masks: np.ndarray, scores) -> np.ndarray:
    """多个候选掩码里挑一个：有打分按打分，没有就取第一个。"""
    array = np.asarray(masks)
    if array.ndim == 4:  # (1, N, H, W)
        array = array[0]
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise PromptLabelError(f"掩码维度不支持：{array.shape}")
    if scores is not None:
        flat = np.asarray(scores).reshape(-1)
        if flat.size >= array.shape[0]:
            return array[int(np.argmax(flat[: array.shape[0]]))]
    return array[0]


def _to_original(mask: np.ndarray, scale: float, original_h: int, original_w: int) -> np.ndarray:
    """把编码尺寸的掩码缩放回原图。

    **必须先裁掉补边再缩放**：编码画布是 (in_h, in_w)，原图等比缩放后只占左上角
    (new_h, new_w)，其余是补 0 的黑边。直接按整块画布缩放会把内容压小
    （1920×1080 的图在 684×1024 画布上只占 1024×576，整块缩放会让结果缩水 44%），
    表现为"掩码永远偏小一圈"。
    """
    import cv2

    if scale <= 0:
        raise PromptLabelError("缩放系数非法")
    array = np.asarray(mask, dtype=np.float32)
    crop_h = min(array.shape[0], max(1, int(round(original_h * scale))))
    crop_w = min(array.shape[1], max(1, int(round(original_w * scale))))
    return cv2.resize(
        array[:crop_h, :crop_w],
        (int(original_w), int(original_h)),
        interpolation=cv2.INTER_LINEAR,
    )


# ---------------------------------------------------------------------------
# 进程级会话缓存（与 `infer.py` 的引擎句柄同思路）
# ---------------------------------------------------------------------------

_SESSIONS: dict[tuple, SamSession] = {}
_SESSION_LOCK = threading.Lock()


def get_session(encoder_path: str, decoder_path: str) -> SamSession:
    """按路径取常驻会话。同一对权重只加载一次——加载 40 MB 模型也要几百毫秒。"""
    key = (str(Path(encoder_path).resolve()), str(Path(decoder_path).resolve()))
    with _SESSION_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = SamSession(key[0], key[1])
            _SESSIONS[key] = session
        return session


def release() -> None:
    """释放全部会话（测试与"换模型"时用）。"""
    with _SESSION_LOCK:
        for session in _SESSIONS.values():
            session.clear_cache()
        _SESSIONS.clear()


# ---------------------------------------------------------------------------
# 面向接口的入口
# ---------------------------------------------------------------------------


def read_image(path: str | Path) -> np.ndarray:
    import cv2

    target = Path(path)
    if not target.is_file():
        raise PromptLabelError(f"图片不存在：{target}")
    # 用 imdecode 读，绕开 cv2.imread 不支持中文路径的老问题
    data = np.fromfile(str(target), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise PromptLabelError(f"图片无法解码：{target}")
    return image


def annotate(
    image_path: str | Path,
    points=None,
    labels=None,
    box=None,
    epsilon: float = DEFAULT_EPSILON,
    max_points: int = MAX_POINTS,
    label: str = "",
    class_id: int = 0,
) -> list[dict]:
    """一站式：读图 → SAM → 多边形草稿列表（直接能进草稿池）。

    返回的草稿 `score = -1`：SAM 的 `iou_predictions` 常年 > 0.8，拿它当置信度
    会把所有草稿染成绿色，用户以为"机器很确定"。分割任务是"你点哪儿它框哪儿"，
    本来就没有"有多确定"这个概念，按未知档呈现才是诚实的。
    """
    import modelzoo
    import autolabel  # 复用 _draft / sanitize 那套结构，避免两处草稿格式不一致

    files = modelzoo.require("mobile_sam")
    encoder = files["mobile_sam.encoder.onnx"]
    decoder = files["sam_vit_h_4b8939.decoder.onnx"]

    image = read_image(image_path)
    height, width = image.shape[:2]
    coords = clamp_points(points or [], width, height)
    box_points = clamp_points(box or [], width, height) if box else None
    session = get_session(encoder, decoder)
    polygons = session.annotate(
        image,
        modelzoo_placeholder_key(image_path),
        coords,
        labels,
        box_points,
        epsilon=epsilon,
        max_points=max_points,
    )

    drafts = []
    for polygon in polygons:
        drafts.append(
            autolabel._draft(
                shape="polygon",
                points=polygon,
                class_id=class_id,
                label=label,
                score=-1.0,
            )
        )
    return drafts


def modelzoo_placeholder_key(image_path) -> tuple:
    """用文件指纹做缓存键：同名的图被替换后必须重算 embedding。"""
    return SamSession._cache_key(image_path)
