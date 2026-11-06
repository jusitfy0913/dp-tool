# -*- coding: utf-8 -*-
"""文本提示通道（GroundingDINO）验收脚本（不需要真模型、不需要 GPU）。

覆盖变更 `add-prompt-autolabel-review` 里"说一句话出框"这一块最容易出错的点：

1. **文本编码**：句点补齐、小写化、分句边界、`position_ids` / `text_token_mask`
   的可见性语义（错了的表现是"框出来了但 label 乱码"，很难定位到是 mask 写错）；
2. **双阈值**：框阈值与文本阈值分别生效，低于阈值的候选必须被过滤；
3. **短语还原**：`##` 子词、空短语的兜底，标签必须是能当类别名的词；
4. **坐标**：归一化 cxcywh → 像素 xyxy，越界夹取，退化框丢弃；
5. **模型缺失**：要报可读的业务错误，不能是裸 FileNotFoundError。

用法：
    python 6-web/backend/tools/test_text_label.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import textlabel  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


class FakeTokenizer:
    """只实现 decode 的假分词器：把 id 直接映射成预设的词。

    用真 BERT tokenizer 会把测试绑死在"某个 JSON 文件在哪"上，而这里要验的
    是"给定 token 序列，短语还原对不对"，与词表内容无关。
    """

    def __init__(self, vocab: dict[int, str] | None = None):
        self.vocab = vocab or {101: "[CLS]", 102: "[SEP]", 9999: "missing", 9998: "hole"}

    def decode(self, ids) -> str:
        return " ".join(self.vocab.get(int(i), f"tok{int(i)}") for i in ids)


# ---------------------------------------------------------------------------
# 1. sigmoid / caption
# ---------------------------------------------------------------------------
def test_sigmoid() -> None:
    values = textlabel.sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]))
    check("sigmoid：极端负数不产生 nan/inf", bool(np.all(np.isfinite(values))), str(values))
    check("sigmoid：极端负数趋近 0", abs(float(values[0])) < 1e-9, str(values[0]))
    check("sigmoid：极端正数趋近 1", abs(float(values[4]) - 1.0) < 1e-9, str(values[4]))
    check("sigmoid：0 → 0.5", abs(float(values[2]) - 0.5) < 1e-12, str(values[2]))
    check("sigmoid：单调递增", bool(np.all(np.diff(values) > 0)), str(values))


def test_caption() -> None:
    check("caption：小写化", textlabel.make_caption("Missing Hole") == "missing hole.", textlabel.make_caption("Missing Hole"))
    check("caption：已有句点不重复加", textlabel.make_caption("hole.") == "hole.", textlabel.make_caption("hole."))
    check("caption：去首尾空白", textlabel.make_caption("  hole  ") == "hole.", textlabel.make_caption("  hole  "))
    try:
        textlabel.make_caption("   ")
        check("caption：空提示报业务错误", False, "没有抛错")
    except textlabel.TextLabelError:
        check("caption：空提示报业务错误", True)


# ---------------------------------------------------------------------------
# 2. 文本编码
# ---------------------------------------------------------------------------
def test_phrase_ranges() -> None:
    ids = [101, 2054, 9999, 9998, 1012, 3000, 102]
    ranges = textlabel.phrase_ranges(ids)
    check("分句：按标点切成两段", len(ranges) == 2, str(ranges))
    check("分句：第一段边界", ranges[0] == (1, 4), str(ranges))
    check("分句：第二段边界", ranges[1] == (5, 6), str(ranges))

    ids2 = [101, 2054, 102]
    check("分句：单段", textlabel.phrase_ranges(ids2) == [(1, 2)], str(textlabel.phrase_ranges(ids2)))
    check("分句：全是特殊 token 时为空", textlabel.phrase_ranges([101, 102]) == [], "")


def test_build_text_inputs() -> None:
    ids = np.array([101, 2054, 9999, 9998, 1012, 3000, 102])
    position_ids, token_mask = textlabel.build_text_inputs(ids)
    check("position_ids：形状 1×T", position_ids.shape == (1, 7), str(position_ids.shape))
    check("text_token_mask：形状 1×T×T", token_mask.shape == (1, 7, 7), str(token_mask.shape))
    check("text_token_mask：对称", bool(np.array_equal(token_mask[0], token_mask[0].T)), "")

    # 段内两两可见
    check(
        "可见性：同段两个普通 token 互相可见",
        bool(token_mask[0, 1, 3]) and bool(token_mask[0, 3, 1]),
        str(token_mask[0, 1:5, 1:5]),
    )
    # 跨段不可见（索引 1..4 是段一，5 是段二）
    check("可见性：跨段不可见", not bool(token_mask[0, 1, 5]) and not bool(token_mask[0, 5, 1]), str(token_mask[0, 1, 5]))
    # 首尾特殊 token 只看自己
    check("可见性：CLS 只看自己", bool(token_mask[0, 0, 0]) and int(token_mask[0, 0].sum()) == 1, str(token_mask[0, 0]))
    check("可见性：SEP 只看自己", bool(token_mask[0, 6, 6]) and int(token_mask[0, 6].sum()) == 1, str(token_mask[0, 6]))
    # 段内位置从 0 递增
    check("position_ids：段内从 0 开始", position_ids[0, 1] == 0 and position_ids[0, 4] == 3, str(position_ids.tolist()))


def test_preprocess_image() -> None:
    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    blob = textlabel.preprocess_image(image, (1200, 800))
    check("预处理：形状 (1,3,H,W)", blob.shape == (1, 3, 800, 1200), str(blob.shape))
    check("预处理：dtype float32", blob.dtype == np.float32, str(blob.dtype))
    # 128/255 ≈ 0.502，ImageNet 归一化后 (0.502-0.485)/0.229 ≈ 0.07
    channel_mean = float(blob[0, 0].mean())
    check("预处理：归一化后量级合理", -0.5 < channel_mean < 0.5, str(channel_mean))
    check("预处理：三个通道因 mean 不同而不同", float(blob[0, 0].mean()) != float(blob[0, 1].mean()), "")


# ---------------------------------------------------------------------------
# 3. 框解码
# ---------------------------------------------------------------------------
def test_rescale_boxes() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64)
    pixels = textlabel.rescale_boxes(boxes, 1000, 2000)
    expected = [800.0, 300.0, 1200.0, 700.0]  # cx*W ± w*W/2
    check("坐标：归一化 → 像素 xyxy", bool(np.allclose(pixels[0], expected)), str(pixels[0].tolist()))
    check("坐标：空输入返回 0×4", textlabel.rescale_boxes(np.zeros((0, 4)), 10, 10).shape == (0, 4), "")


def test_clamp_box() -> None:
    check(
        "夹取：越界被裁到图内",
        textlabel.clamp_box([-10, -10, 120, 120], 100, 80) == [0.0, 0.0, 100.0, 80.0],
        str(textlabel.clamp_box([-10, -10, 120, 120], 100, 80)),
    )
    check(
        "夹取：反向坐标被摆正",
        textlabel.clamp_box([80, 60, 20, 10], 100, 80) == [20.0, 10.0, 80.0, 60.0],
        str(textlabel.clamp_box([80, 60, 20, 10], 100, 80)),
    )
    check(
        "夹取：宽不足 1px 丢弃",
        textlabel.clamp_box([10, 10, 10.4, 50], 100, 80) is None,
        str(textlabel.clamp_box([10, 10, 10.4, 50], 100, 80)),
    )
    check("夹取：完全在图外丢弃", textlabel.clamp_box([200, 200, 300, 300], 100, 80) is None, "")


def test_select_candidates() -> None:
    # 3 个 query：两个超阈值，一个不超
    logits = np.array([[[0.9, 0.1], [0.2, 0.2], [0.1, 0.8]]], dtype=np.float64)  # sigmoid 前
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.3, 0.3]]], dtype=np.float64
    )

    scores, picked = textlabel.select_candidates(logits, boxes, box_thr=0.6, text_thr=0.2)
    check("筛选：只剩超框阈值的 query", len(scores) == 2, str(len(scores)))
    check("筛选：按得分降序", float(scores[0].max()) >= float(scores[1].max()), str(scores.max(axis=1)))
    check("筛选：框跟着得分走", bool(np.allclose(picked[0], boxes[0][0])), str(picked.tolist()))

    # 框阈值调到 0.99 → 全被滤掉
    none_scores, none_boxes = textlabel.select_candidates(logits, boxes, box_thr=0.99, text_thr=0.2)
    check(
        "筛选：阈值过高时为空",
        none_scores.shape[0] == 0 and none_boxes.shape[0] == 0,
        f"{none_scores.shape} {none_boxes.shape}",
    )

    # max_boxes 上限
    many = np.tile(logits, (1, 50, 1)).reshape(1, 150, 2)
    many_boxes = np.tile(boxes, (1, 50, 1)).reshape(1, 150, 4)
    capped_scores, _ = textlabel.select_candidates(many, many_boxes, 0.5, 0.2, max_boxes=10)
    check("筛选：受 max_boxes 限制", capped_scores.shape[0] <= 10, str(capped_scores.shape))

    # 二维输入（导出压掉了 batch 维）也要能处理
    scores2, boxes2 = textlabel.select_candidates(logits[0], boxes[0], 0.6, 0.2)
    check("筛选：二维 logits 也能处理", scores2.shape[0] == 2 and boxes2.shape[0] == 2, f"{scores2.shape} {boxes2.shape}")


def test_decode_phrase() -> None:
    tokenizer = FakeTokenizer()
    input_ids = np.array([101, 9999, 9998, 102])
    posmap = np.array([False, True, True, False])
    phrase = textlabel.decode_phrase(posmap, input_ids, tokenizer)
    check("短语：拼接为可读词", phrase == "missing hole", phrase)

    # 左裁剪：CLS 位置即使超阈值也不能被解出来
    posmap_all = np.ones(4, dtype=bool)
    phrase2 = textlabel.decode_phrase(posmap_all, input_ids, tokenizer)
    check("短语：左裁剪掉 CLS/SEP", "[CLS]" not in phrase2 and "[SEP]" not in phrase2, phrase2)

    check(
        "短语：全 False 返回空",
        textlabel.decode_phrase(np.zeros(4, dtype=bool), input_ids, tokenizer) == "",
        "",
    )
    check("短语：空 mask 返回空", textlabel.decode_phrase(np.array([], dtype=bool), input_ids, tokenizer) == "", "")


def test_fallback_phrase() -> None:
    """阈值切出来的短语含 `##` 时，要退到得分最高的整段短语。"""
    tokenizer = FakeTokenizer({101: "[CLS]", 102: "[SEP]", 9999: "missing", 9998: "##hole"})
    session = textlabel.TextSession.__new__(textlabel.TextSession)
    session.tokenizer = tokenizer
    input_ids = np.array([101, 9999, 9998, 102])
    row = np.zeros(4, dtype=np.float64)
    row[1] = 0.7  # 只有 missing 超阈值
    phrase = session._fallback_phrase(row, {"input_ids": input_ids})
    check("兜底：取整段短语", phrase in ("missing ##hole", "missing hole"), phrase)
    check("兜底：不残留 ##", "##" not in phrase, phrase)

    # 整段都低于任何阈值也要能给出一个（取最大者的整段）
    row_zero = np.zeros(4, dtype=np.float64)
    phrase2 = session._fallback_phrase(row_zero, {"input_ids": input_ids})
    check("兜底：全 0 也能给出整段", bool(phrase2), phrase2)


# ---------------------------------------------------------------------------
# 4. 模型缺失
# ---------------------------------------------------------------------------
def test_model_missing(tmp: Path) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "empty-zoo")
    try:
        textlabel.release()
        try:
            textlabel.get_session()
            check("缺失：没有模型时抛 ModelMissing", False, "没有抛错")
        except textlabel.ModelMissing as exc:
            check("缺失：没有模型时抛 ModelMissing", True)
            check("缺失：错误里带下载指引", "下载" in str(exc), str(exc))
        except textlabel.TextLabelError as exc:
            check("缺失：没有模型时抛 ModelMissing", False, f"抛的是 TextLabelError：{exc}")

        try:
            textlabel.annotate(str(tmp / "nothing.png"), "hole")
            check("缺失：annotate 也抛业务错误", False, "没有抛错")
        except textlabel.TextLabelError:
            check("缺失：annotate 也抛业务错误", True)
    finally:
        textlabel.release()
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_refine_degrades(tmp: Path) -> None:
    """细化失败时必须退回矩形，不能把检出结果整体丢掉。"""
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "no-sam")
    try:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = textlabel.refine_boxes_to_polygons(image, [[10, 10, 50, 50]], 0.001, 400, None)
        check("细化：SAM 不可用时返回空表（调用方保留矩形）", result == [], str(result))
        empty = textlabel.refine_boxes_to_polygons(image, [], 0.001, 400, None)
        check("细化：没有框时直接返回空", empty == [], str(empty))
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


# ---------------------------------------------------------------------------
# 5. HTTP 接口（模型缺失 → 400 而不是 500）
# ---------------------------------------------------------------------------
def test_api(tmp: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        check(f"接口：跳过（未安装 httpx：{exc}）", True)
        return

    import cv2

    import app as web_app

    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "api-zoo")
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(web_app.config.REPO_ROOT)
    try:
        client = TestClient(web_app.app)
        image_path = tmp / "frame.png"
        cv2.imwrite(str(image_path), np.full((60, 80, 3), 127, dtype=np.uint8))

        empty_prompt = client.post(
            "/api/annotation/text-label",
            json={"image": str(image_path), "prompt": "   "},
        )
        check("接口：空提示被拒 400", empty_prompt.status_code == 400, f"{empty_prompt.status_code} {empty_prompt.text[:200]}")

        missing = client.post(
            "/api/annotation/text-label",
            json={"image": str(image_path), "prompt": "hole"},
        )
        check("接口：模型缺失返回 400（不是 500）", missing.status_code == 400, f"{missing.status_code} {missing.text[:300]}")
        check("接口：错误里提到下载", "下载" in missing.text, missing.text[:300])

        no_image = client.post(
            "/api/annotation/text-label",
            json={"image": str(tmp / "nope.png"), "prompt": "hole"},
        )
        check("接口：图片不存在返回 404", no_image.status_code == 404, f"{no_image.status_code} {no_image.text[:200]}")

        no_prompt_point = client.post(
            "/api/annotation/prompt-label",
            json={"image": str(image_path), "points": [], "box": []},
        )
        check(
            "接口：点框通道无提示被拒 400",
            no_prompt_point.status_code == 400,
            f"{no_prompt_point.status_code} {no_prompt_point.text[:200]}",
        )

        missing_sam = client.post(
            "/api/annotation/prompt-label",
            json={"image": str(image_path), "points": [[10, 10]], "labels": [1]},
        )
        check("接口：SAM 模型缺失返回 400", missing_sam.status_code == 400, f"{missing_sam.status_code} {missing_sam.text[:300]}")

        # 路径白名单仍然生效
        blocked = client.post(
            "/api/annotation/text-label",
            json={"image": str(Path(tempfile.gettempdir()).parent / "outside.png"), "prompt": "hole"},
        )
        check("接口：白名单外被拒 403", blocked.status_code == 403, f"{blocked.status_code} {blocked.text[:200]}")
    finally:
        textlabel.release()
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


# ---------------------------------------------------------------------------
# 6. 真 tokenizer（有就用，没有就跳过）
# ---------------------------------------------------------------------------
def test_real_tokenizer() -> None:
    tokenizer_path = (
        BACKEND.parents[1]
        / "8-X-AnyLabeling-main"
        / "anylabeling"
        / "services"
        / "auto_labeling"
        / "configs"
        / "bert"
        / "bert_base_uncased_tokenizer.json"
    )
    if not tokenizer_path.is_file():
        check("真分词器：跳过（本地没有 bert tokenizer）", True)
        return
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        check(f"真分词器：跳过（未安装 tokenizers：{exc}）", True)
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode("missing hole.")
    ids = list(encoded.ids)
    check("真分词器：CLS/SEP 包住", ids[0] == 101 and ids[-1] == 102, str(ids))
    check("真分词器：句点被单独切出", 1012 in ids, str(ids))

    ranges = textlabel.phrase_ranges(np.array(ids))
    check("真分词器：切出至少一段短语", len(ranges) >= 1, str(ranges))

    position_ids, token_mask = textlabel.build_text_inputs(np.array(ids))
    check("真分词器：position_ids 长度一致", position_ids.shape[1] == len(ids), f"{position_ids.shape} vs {len(ids)}")
    check("真分词器：mask 是方阵", token_mask.shape[1] == token_mask.shape[2] == len(ids), str(token_mask.shape))


def main() -> int:
    test_sigmoid()
    test_caption()
    test_phrase_ranges()
    test_build_text_inputs()
    test_preprocess_image()
    test_rescale_boxes()
    test_clamp_box()
    test_select_candidates()
    test_decode_phrase()
    test_fallback_phrase()
    test_real_tokenizer()
    with tempfile.TemporaryDirectory(prefix="text-label-") as temp:
        tmp = Path(temp)
        test_model_missing(tmp)
        test_refine_degrades(tmp)
        test_api(tmp)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
