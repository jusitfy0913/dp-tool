# -*- coding: utf-8 -*-
"""文本提示通道：说一句话 → 检出目标（GroundingDINO）。

和检测通道的差别：
    检测通道的类别表在训练时就定死了，遇到新缺陷只有"重训"一条路。
    GroundingDINO 是**开放词表**：把 `missing hole` 编成文本特征，与图像特征做
    跨模态对齐，输出"哪些框对应这句话"。所以它能在没标过任何样本时先出个草稿，
    让用户看着结果决定"这个类别值不值得训一个专用引擎"。

    输入契约（ONNX 导出时定死，改不了）：
    img                (1, 3, H, W) float32，ImageNet 归一化，**RGB 顺序**
                       （与参考实现 `qt_img_to_rgb_cv_img()` 一致，见 `preprocess_image`）
    input_ids          (1, T) int64
    attention_mask     (1, T) bool
    token_type_ids     (1, T) int64
    position_ids       (1, T) int64
    text_token_mask    (1, T, T) bool
输出：
    logits             (1, 900, T)   每个 query 对每个 token 的得分（sigmoid 前）
    boxes              (1, 900, 4)   归一化 cxcywh

`position_ids` / `text_token_mask` 为什么要自己造：
    BERT 本身不需要这两个输入，但 GroundingDINO 的文本增强器（self-attention
    enhancer）要用它们把"同一句话内的 token 互相可见，跨句不可见"表达出来。
    导出脚本没把它们固化进模型，只能在前处理里按特殊 token 的位置分段生成。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

import onnxenv

#: BERT 特殊 token：`[CLS]=101`、`[SEP]=102`、`"."=1012`、`","? =1029`。
#: 后两个是句读——GroundingDINO 把标点也当分句边界，一逗号一句。
SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029)
#: 文本最长 token 数。模型侧固定 256，超出会截断（截断丢的是后半句，前端要提示）。
MAX_TEXT_LEN = 256
#: 默认输入尺寸。800×1200 是 GroundingDINO 官方配置的比例（短边 800）。
DEFAULT_INPUT_SIZE = (1200, 800)  # (w, h)
#: 单图最多保留的框。900 个 query 全过阈值时前端会被刷屏，也没人审得完。
MAX_BOXES = 300

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TextLabelError(Exception):
    """文本通道的可预期错误（接口层转 400）。"""


class ModelMissing(TextLabelError):
    """缺少模型文件。带上"去下载"的指引，避免用户看到裸 FileNotFoundError。"""

    def __init__(self, which: str):
        self.which = which
        super().__init__(f"文本通道需要先下载模型「{which}」，请在标注页的模型仓库里点击下载")


# ---------------------------------------------------------------------------
# 纯函数（无模型依赖，全部可单测）
# ---------------------------------------------------------------------------


def sigmoid(values: np.ndarray) -> np.ndarray:
    """数值稳定版 sigmoid。

    直接 ``1/(1+exp(-x))`` 在 x=-1000 时溢出成 inf-inf=nan，
    而量化模型确实会吐出这种极端 logit。
    """
    array = np.asarray(values, dtype=np.float64)
    out = np.empty_like(array)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_x = np.exp(array[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def make_caption(prompt: str) -> str:
    """规范成"小写 + 以句点结尾"。训练时 caption 都这么写，少了句点分布会偏。"""
    text = str(prompt or "").strip().lower()
    if not text:
        raise TextLabelError("文本提示不能为空")
    if not text.endswith("."):
        text += "."
    return text


def phrase_ranges(input_ids, special_ids=SPECIAL_TOKEN_IDS) -> list[tuple[int, int]]:
    """按特殊 token 切出每段短语的 token 下标区间 ``[start, end)``。"""
    ids = np.asarray(input_ids).reshape(-1)
    ranges: list[tuple[int, int]] = []
    start = None
    for index, token_id in enumerate(ids):
        if int(token_id) in special_ids:
            if start is not None:
                ranges.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        ranges.append((start, len(ids)))
    return ranges


def build_text_inputs(input_ids, special_ids=SPECIAL_TOKEN_IDS):
    """由 token id 序列生成 `(position_ids, text_token_mask)`。

    规则：以特殊 token 为界分段，段内位置从 0 递增、段内两两可见；
    界外（CLS/SEP 自身）只看自己。这正是 BERT 里 "sentence A / sentence B"
    的可见性语义，GroundingDINO 用它把多个短语隔离开。
    """
    ids = np.asarray(input_ids).reshape(-1)
    length = ids.shape[0]
    position_ids = np.zeros((1, length), dtype=np.int64)
    token_mask = np.eye(length, dtype=bool)[None, ...]

    boundaries = [index for index, token_id in enumerate(ids) if int(token_id) in special_ids]
    previous = 0
    for boundary in boundaries:
        if boundary == 0 or boundary == length - 1:
            # 首尾特殊 token 不参与任何跨 token 注意力
            token_mask[0, boundary, boundary] = True
            continue
        token_mask[0, previous + 1 : boundary + 1, previous + 1 : boundary + 1] = True
        position_ids[0, previous + 1 : boundary + 1] = np.arange(0, boundary - previous)
        previous = boundary
    return position_ids, token_mask


def pad_text(length: int, max_len: int = MAX_TEXT_LEN) -> bool:
    """返回"是否发生了截断"。前端据此提示"提示词太长"。"""
    return length > max_len


def preprocess_image(image: np.ndarray, size: tuple[int, int] = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """BGR uint8 → `(1, 3, H, W)` float32（模型要吃 **RGB**）。

    v1.24.1 更正：这里原来直接喂 BGR，并注释说"RGB 顺序无关"。那条结论没实测过，
    与参考实现不符——X-AnyLabeling 的 `qt_img_to_rgb_cv_img()` 明确做了
    `cv2.COLOR_BGR2RGB` 才送进模型。实测（bus.jpg + "bus"）BGR 0.91 分 / RGB 0.91 分，
    zidane 的 person 0.85 vs 0.87，**差异不大但方向一致**，所以按参考改成 RGB，
    让同一张图在这两套实现下分数可对齐。
    """
    import cv2

    width, height = size
    # BGR → RGB：调用方（app / 批量任务 / 草稿回放）都从 cv2 读图，一律是 BGR。
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0
    blob = (blob - IMAGE_MEAN) / IMAGE_STD
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, 0).astype(np.float32)


def rescale_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """归一化 cxcywh → 像素 xyxy。"""
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    scaled = array * np.array([width, height, width, height], dtype=np.float64)
    result = np.empty_like(scaled)
    result[:, 0] = scaled[:, 0] - scaled[:, 2] / 2.0
    result[:, 1] = scaled[:, 1] - scaled[:, 3] / 2.0
    result[:, 2] = scaled[:, 0] + scaled[:, 2] / 2.0
    result[:, 3] = scaled[:, 1] + scaled[:, 3] / 2.0
    return result


def clamp_box(box, width: int, height: int) -> list[float] | None:
    """夹到图内；夹完宽或高不足 1px 就返回 ``None``（退化框没有标注价值）。"""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def decode_caption_tokens(input_ids, tokenizer) -> str:
    """整句解码（调试与"短语为空"时的兜底显示用）。"""
    return tokenizer.decode([int(v) for v in np.asarray(input_ids).reshape(-1)])


def decode_phrase(
    posmap: np.ndarray,
    input_ids,
    tokenizer,
    left_idx: int = 0,
    right_idx: int = MAX_TEXT_LEN,
) -> str:
    """把"哪些 token 超过文本阈值"还原成可读短语。

    `left/right` 是对 posmap 的硬裁剪：模型给的 token 维长度可能大于实际序列，
    多出来的位置全 0，不裁会解出空串。
    """
    mask = np.array(posmap, dtype=bool).reshape(-1).copy()
    if mask.size == 0:
        return ""
    mask[: left_idx + 1] = False
    mask[right_idx:] = False
    ids = np.asarray(input_ids).reshape(-1)
    # 特殊 token 一律丢掉：句点 `[SEP]` 如果没被裁掉，解出来会变成标签的一部分
    # （`missing hole [SEP]`），既不像类别名，也没法当导出时的文件名。
    picked = [
        int(ids[index])
        for index in np.where(mask)[0]
        if index < ids.size and int(ids[index]) not in SPECIAL_TOKEN_IDS
    ]
    if not picked:
        return ""
    return str(tokenizer.decode(picked)).strip()


def select_candidates(
    logits: np.ndarray,
    boxes: np.ndarray,
    box_thr: float,
    text_thr: float,
    max_boxes: int = MAX_BOXES,
) -> tuple[np.ndarray, np.ndarray]:
    """按框阈值筛 query，返回 ``(保留的 logits, 保留的 boxes)``，按得分降序。"""
    scores_all = np.asarray(logits, dtype=np.float64)
    boxes_all = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if scores_all.ndim == 3:
        scores_all = scores_all[0]
    if scores_all.size == 0:
        width = scores_all.shape[-1] if scores_all.ndim == 2 else 0
        return np.zeros((0, width)), np.zeros((0, 4))
    if boxes_all.shape[0] != scores_all.shape[0]:
        # 有的导出会把 batch 维也压掉，这里对齐一下，别让后面按索引取值崩
        boxes_all = boxes_all.reshape(-1, 4)

    # **无条件** sigmoid：导出的是原始 logit，不 sigmoid 的话阈值 0.3 完全没意义。
    # 不要"看最大值判断是否已归一化"——logit 恰好落在 (0,1) 区间时会被误判。
    scores_all = sigmoid(scores_all)

    best = scores_all.max(axis=1)
    keep = np.where(best > float(box_thr))[0]
    if keep.size == 0:
        return scores_all[:0], boxes_all[:0]
    order = keep[np.argsort(-best[keep])]
    order = order[: max(1, int(max_boxes))]
    return scores_all[order], boxes_all[order]


def refine_boxes_to_polygons(
    image: np.ndarray,
    boxes: list[list[float]],
    epsilon: float,
    max_points: int,
    image_key,
    session=None,
) -> list[list[list[float]]]:
    """把一批框送 SAM 细化成多边形。

    **一次 encode 多次 decode**：每个框单独 encode 的话，10 个框要多花 10 倍时间，
    而编码结果与提示无关——这正是 SAM "分离 encoder/decoder" 的设计意图。
    细化失败（掩码退化 / 模型报错）时**保留原框**，不能因为细化挂了就丢掉检出结果。
    """
    import promptlabel

    if not boxes:
        return []
    try:
        if session is None:
            import modelzoo

            files = modelzoo.require("mobile_sam")
            session = promptlabel.get_session(
                files["mobile_sam.encoder.onnx"], files["sam_vit_h_4b8939.decoder.onnx"]
            )
        state = session.embed(image, image_key)
    except Exception:  # noqa: BLE001 - 细化是增强，任何原因都退回矩形
        return []

    polygons: list[list[list[float]]] = []
    for box in boxes:
        try:
            coords, marks = promptlabel.build_prompt([], [], [[box[0], box[1]], [box[2], box[3]]])
            mask = session.decode(state, coords, marks)
            found = promptlabel.mask_to_polygons(mask, epsilon=epsilon, max_points=max_points)
        except Exception:  # noqa: BLE001
            found = []
        if found:
            polygons.append(found[0])
        else:
            polygons.append(_box_as_polygon(box))
    return polygons


def _box_as_polygon(box) -> list[list[float]]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------

_SESSION: "TextSession | None" = None
_SESSION_LOCK = threading.Lock()


class TextSession:
    """GroundingDINO 的常驻会话（模型近 1 GB，加载一次就该一直用它）。"""

    def __init__(self, model_path: str, tokenizer_path: str, input_size: tuple[int, int] = DEFAULT_INPUT_SIZE):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 环境缺包时的明确提示
            raise TextLabelError(
                f"文本通道缺少依赖：{exc}。请执行 pip install onnxruntime tokenizers"
            ) from exc

        self.model_path = str(model_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # 与 SAM 通道一致：只在 provider DLL 真正能加载时才启用 CUDA，
        # 否则缺 cuDNN 的机器每次建 session 都会打一段 LoadLibrary 错误。
        self.providers = onnxenv.usable_providers(ort)
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_size = input_size
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.lock = threading.RLock()

    def encode_text(self, prompt: str) -> dict:
        caption = make_caption(prompt)
        encoded = self.tokenizer.encode(caption)
        ids = np.asarray([encoded.ids], dtype=np.int64)[:, :MAX_TEXT_LEN]
        types = np.asarray([encoded.type_ids], dtype=np.int64)[:, :MAX_TEXT_LEN]
        attention = np.asarray([encoded.attention_mask], dtype=np.int64)[:, :MAX_TEXT_LEN]
        position_ids, token_mask = build_text_inputs(ids[0])
        return {
            "caption": caption,
            "truncated": bool(pad_text(len(encoded.ids))),
            "inputs": {
                "input_ids": ids,
                "attention_mask": attention.astype(bool),
                "token_type_ids": types,
                "position_ids": position_ids,
                "text_token_mask": token_mask,
            },
            "input_ids": ids[0],
        }

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_thr: float = 0.3,
        text_thr: float = 0.25,
        refine: bool = False,
        epsilon: float = 0.001,
        max_points: int = 400,
        image_key=None,
        label: str = "",
        class_id: int = 0,
    ) -> dict:
        """跑一次文本提示，返回 `{drafts, warnings, count, truncated}`。"""
        import autolabel

        text = self.encode_text(prompt)
        blob = preprocess_image(image, self.input_size)
        feed = {"img": blob, **text["inputs"]}
        feed = {key: value for key, value in feed.items() if key in self.input_names}

        with self.lock:
            outputs = self.session.run(None, feed)
        if len(outputs) < 2:
            raise TextLabelError(f"模型输出不足（得到 {len(outputs)} 个），期望 logits + boxes")
        logits, boxes = outputs[0], outputs[1]

        scores, picked = select_candidates(logits, boxes, box_thr, text_thr)
        height, width = image.shape[:2]
        pixels = rescale_boxes(picked, height, width)

        warnings: list[str] = []
        if text["truncated"]:
            warnings.append(f"提示词超过 {MAX_TEXT_LEN} 个 token，超出部分被截断")

        kept_boxes: list[list[float]] = []
        kept_labels: list[str] = []
        kept_scores: list[float] = []
        dropped = 0
        for row, box in zip(scores, pixels):
            phrase = decode_phrase(row > float(text_thr), text["input_ids"], self.tokenizer)
            if not phrase or "##" in phrase:
                phrase = self._fallback_phrase(row, text)
            clamped = clamp_box(box, width, height)
            if clamped is None:
                dropped += 1
                continue
            kept_boxes.append(clamped)
            kept_labels.append(phrase or label)
            kept_scores.append(float(np.max(row)))
        if dropped:
            warnings.append(f"已丢弃 {dropped} 个退化框（宽或高不足 1px）")
        if not kept_boxes:
            warnings.append(f"没有超过阈值的检出（框阈值 {box_thr}，文本阈值 {text_thr}）")

        polygons: list[list[list[float]]] = []
        if refine and kept_boxes:
            polygons = refine_boxes_to_polygons(
                image, kept_boxes, epsilon, max_points, image_key if image_key is not None else None
            )
            if not polygons:
                warnings.append("细化为多边形失败（分割模型未就绪或掩码退化），已保留矩形")

        drafts = []
        for index, box in enumerate(kept_boxes):
            score = kept_scores[index]
            name = kept_labels[index] or label
            if refine and index < len(polygons) and polygons[index]:
                drafts.append(
                    autolabel._draft(
                        shape="polygon", points=polygons[index], class_id=class_id, label=name, score=score
                    )
                )
            else:
                drafts.append(
                    autolabel._draft(
                        shape="rectangle",
                        points=[[box[0], box[1]], [box[2], box[3]]],
                        class_id=class_id,
                        label=name,
                        score=score,
                    )
                )
        return {"drafts": drafts, "warnings": warnings, "count": len(drafts), "truncated": text["truncated"]}

    def _fallback_phrase(self, row: np.ndarray, text: dict) -> str:
        """阈值切出来的短语太碎（含 `##` 子词）时，用得分最高的整段短语顶上。

        出现 `##` 说明切在了词中间，这样的标签用户看不懂，也不适合当类别名。
        """
        ranges = phrase_ranges(text["input_ids"])
        if not ranges:
            return ""
        best_score = -np.inf
        best_range = None
        for start, end in ranges:
            window = row[start:end]
            if window.size == 0:
                continue
            score = float(np.max(window))
            if score > best_score:
                best_score, best_range = score, (start, end)
        if best_range is None:
            return ""
        mask = np.zeros(row.shape[0], dtype=bool)
        mask[best_range[0] : best_range[1]] = True
        phrase = decode_phrase(mask, text["input_ids"], self.tokenizer)
        return phrase.replace("##", "")


def get_session(model_path: str | None = None, tokenizer_path: str | None = None) -> TextSession:
    """按路径取常驻会话。缺模型时抛 :class:`ModelMissing` 而不是 OSError。"""
    global _SESSION
    import modelzoo

    if model_path is None or tokenizer_path is None:
        try:
            dino = modelzoo.require("grounding_dino_swint")
            bert = modelzoo.require("bert_tokenizer")
        except modelzoo.ModelMissing as exc:
            raise ModelMissing("grounding_dino_swint / bert_tokenizer") from exc
        model_path = dino["groundingdino_swint_ogc_quant.onnx"]
        tokenizer_path = bert["bert_base_uncased_tokenizer.json"]

    key = (str(Path(model_path).resolve()), str(Path(tokenizer_path).resolve()))
    with _SESSION_LOCK:
        # 路径变了就要重新加载：用户可能在两个模型之间切换。
        if _SESSION is None or getattr(_SESSION, "_key", None) != key:
            _SESSION = TextSession(key[0], key[1])
            _SESSION._key = key
        return _SESSION


def release() -> None:
    global _SESSION
    with _SESSION_LOCK:
        _SESSION = None


# ---------------------------------------------------------------------------
# 面向接口的入口
# ---------------------------------------------------------------------------


def annotate(
    image_path: str | Path,
    prompt: str,
    box_thr: float = 0.3,
    text_thr: float = 0.25,
    refine: bool = False,
    epsilon: float = 0.001,
    max_points: int = 400,
    label: str = "",
    class_id: int = 0,
) -> dict:
    """一站式：读图 → 文本提示 → 草稿。"""
    import promptlabel

    if not str(prompt or "").strip():
        raise TextLabelError("文本提示不能为空")
    # 先确认模型就绪再读图：模型没下载时"图片打不开"会把人引到错误的方向，
    # 真正该看到的是"去下载模型"。
    session = get_session()
    image = promptlabel.read_image(image_path)
    return session.predict(
        image,
        prompt,
        box_thr=box_thr,
        text_thr=text_thr,
        refine=refine,
        epsilon=epsilon,
        max_points=max_points,
        image_key=promptlabel.modelzoo_placeholder_key(image_path),
        label=label,
        class_id=class_id,
    )
