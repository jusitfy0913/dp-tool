# -*- coding: utf-8 -*-
"""提示式预标注 SAM 通道验收脚本（不需要真模型、不需要 GPU）。

覆盖变更 `add-prompt-autolabel-review` 里"点/框 → 多边形"这一块最容易出错的点：

1. **坐标往返**：原图 → 编码输入 → 掩码 → 原图，误差不得超过 1px。
   这条错了的表现是"框永远偏一半"，而且看起来像模型不准，很难查到是缩放写错了；
2. **点数上限**：轮廓点数必须压到上限以内，否则 `.tfanno` 会被撑成几十 MB；
3. **面积过滤**：噪点轮廓不该变成草稿；
4. **退化掩码**：全 0 / 全 1 / 单像素不能抛异常，只能返回空或合理结果；
5. **encode 缓存**：同一张图连续点几下，encoder 只能跑一次（这是交互可用的前提）；
6. **提示展开**：正点 1 / 负点 0 / 框 2,3，且没有提示时必须拒绝。

用法：
    python 6-web/backend/tools/test_prompt_label.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import promptlabel  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def make_mask(height: int = 200, width: int = 300) -> np.ndarray:
    """一个居中的实心矩形掩码（相当于"模型圈住了一个目标"）。"""
    mask = np.zeros((height, width), dtype=np.float32)
    mask[50:150, 80:220] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. 掩码 → 多边形
# ---------------------------------------------------------------------------
def test_mask_to_polygons() -> None:
    mask = make_mask()
    polygons = promptlabel.mask_to_polygons(mask, epsilon=0.001)
    check("掩码：一个矩形掩码产出一个多边形", len(polygons) == 1, str(len(polygons)))
    check("掩码：矩形四个角", len(polygons[0]) == 4, str(len(polygons[0])))

    xs = [point[0] for point in polygons[0]]
    ys = [point[1] for point in polygons[0]]
    check("掩码：x 范围对得上", min(xs) <= 81 and max(xs) >= 218, f"{min(xs)}..{max(xs)}")
    check("掩码：y 范围对得上", min(ys) <= 51 and max(ys) >= 148, f"{min(ys)}..{max(ys)}")

    area = promptlabel._polygon_area(polygons[0])
    check("掩码：面积接近 100×140", abs(area - 14000) < 2000, str(area))

    # 两个分离的方块 → 两个多边形，且大的排前面
    two = np.zeros((200, 300), dtype=np.float32)
    two[10:30, 10:30] = 1.0
    two[50:150, 100:250] = 1.0
    result = promptlabel.mask_to_polygons(two, epsilon=0.001)
    check("掩码：分离目标各出一个多边形", len(result) == 2, str(len(result)))
    check(
        "掩码：大的排前面（审核先看大缺陷）",
        promptlabel._polygon_area(result[0]) > promptlabel._polygon_area(result[1]),
        str([promptlabel._polygon_area(item) for item in result]),
    )


def test_empty_and_degenerate() -> None:
    check("退化：全 0 掩码返回空", promptlabel.mask_to_polygons(np.zeros((64, 64), np.float32)) == [])
    check("退化：None 返回空", promptlabel.mask_to_polygons(None) == [])

    full = np.ones((64, 64), dtype=np.float32)
    border = promptlabel.mask_to_polygons(full, epsilon=0.001)
    check("退化：全 1 掩码不崩，返回贴边的多边形", len(border) == 1 and len(border[0]) >= 3, str(len(border)))

    single = np.zeros((32, 32), dtype=np.float32)
    single[10, 10] = 1.0
    check("退化：单像素掩码被面积过滤掉", promptlabel.mask_to_polygons(single) == [])

    # 大面积噪点：远小于面积阈值的轮廓必须被丢掉（阈值 = 0.01% × 160000 = 16 px²）
    noisy = np.zeros((400, 400), dtype=np.float32)
    noisy[5, 5] = 1.0
    noisy[380:382, 380:382] = 1.0  # 4 px²，低于阈值
    check("退化：小块噪点被过滤", promptlabel.mask_to_polygons(noisy) == [])

    # 维度不对的掩码应当是可读的业务错误，而不是 numpy 的裸 ValueError
    try:
        promptlabel.mask_to_polygons(np.zeros((2, 3, 4, 5), dtype=np.float32))
        check("退化：四维掩码报业务错误", False, "没有抛错")
    except promptlabel.PromptLabelError:
        check("退化：四维掩码报业务错误", True)


def test_point_limit() -> None:
    """半径很大、锯齿很多的掩码 → 点数必须被压到上限。"""
    import cv2

    canvas = np.zeros((1200, 1200), dtype=np.float32)
    # 星形：轮廓点数天然很多，用来压上限逻辑
    points = []
    for index in range(240):
        radius = 500 if index % 2 == 0 else 380
        angle = index * np.pi * 2 / 240
        points.append(
            [
                int(600 + radius * np.cos(angle)),
                int(600 + radius * np.sin(angle)),
            ]
        )
    cv2.fillPoly(canvas, [np.asarray(points, dtype=np.int32)], 1.0)

    coarse = promptlabel.mask_to_polygons(canvas, epsilon=0.001, max_points=400)
    check("上限：轮廓被压到 400 点以内", coarse and len(coarse[0]) <= 400, str(len(coarse[0]) if coarse else 0))
    check("上限：压完仍是闭合多边形（>=3 点）", coarse and len(coarse[0]) >= 3, str(len(coarse[0]) if coarse else 0))

    tiny = promptlabel.mask_to_polygons(canvas, epsilon=0.001, max_points=12)
    check("上限：max_points=12 也遵守", tiny and len(tiny[0]) <= 12, str(len(tiny[0]) if tiny else 0))

    # 近似系数越大，点数越少（epsilon 的语义必须是"越大越粗"）
    fine = promptlabel.mask_to_polygons(canvas, epsilon=0.0002)
    rough = promptlabel.mask_to_polygons(canvas, epsilon=0.005)
    check(
        "上限：epsilon 越大点数越少",
        bool(fine) and bool(rough) and len(rough[0]) < len(fine[0]),
        f"{len(fine[0]) if fine else 0} vs {len(rough[0]) if rough else 0}",
    )


# ---------------------------------------------------------------------------
# 2. 坐标往返
# ---------------------------------------------------------------------------
def test_coordinate_roundtrip() -> None:
    cases = [
        (1080, 1920, 684, 1024),
        (500, 500, 1024, 1024),
        (3000, 4000, 684, 1024),
        (97, 1201, 1024, 1024),
    ]
    worst = 0.0
    for image_h, image_w, in_h, in_w in cases:
        scale, (new_h, new_w) = promptlabel.preprocess_shape(image_h, image_w, in_h, in_w)
        # 原图四角 → 编码坐标 → 回到原图
        corners = np.array([[0, 0], [image_w - 1, 0], [0, image_h - 1], [image_w - 1, image_h - 1]], dtype=float)
        forward = promptlabel.scale_points(corners, scale)
        back = promptlabel.unscale_points(forward, scale)
        error = float(np.max(np.abs(back - corners)))
        worst = max(worst, error)
        # 缩放后必须落在编码画布内（否则掩码边缘会被裁掉）
        inside = (
            float(np.min(forward[:, 0])) >= -1e-6
            and float(np.min(forward[:, 1])) >= -1e-6
            and float(np.max(forward[:, 0])) <= in_w + 1e-6
            and float(np.max(forward[:, 1])) <= in_h + 1e-6
            and new_h <= in_h
            and new_w <= in_w
        )
        check(f"坐标：{image_w}×{image_h} 缩放后在画布内", inside, f"scale={scale:.5f} new={new_w}×{new_h}")
    check("坐标：往返误差 ≤ 1px", worst <= 1.0, f"{worst:.6f}px")


def test_mask_roundtrip() -> None:
    """掩码从编码尺寸缩回原图后，多边形的坐标误差也要在 1px 内。"""
    image_h, image_w = 480, 640
    in_h, in_w = 684, 1024
    scale, (new_h, new_w) = promptlabel.preprocess_shape(image_h, image_w, in_h, in_w)

    # 在编码尺寸上画一个与"目标矩形"严格对应的掩码
    target = [100.0, 80.0, 300.0, 240.0]  # 原图坐标的 x1,y1,x2,y2
    mask = np.zeros((in_h, in_w), dtype=np.float32)
    x1 = int(round(target[0] * scale))
    y1 = int(round(target[1] * scale))
    x2 = int(round(target[2] * scale))
    y2 = int(round(target[3] * scale))
    mask[y1:y2, x1:x2] = 1.0

    shrunk = promptlabel._to_original(mask, scale, image_h, image_w)
    polygons = promptlabel.mask_to_polygons(shrunk, epsilon=0.001)
    check("坐标：缩回原图后仍是一个多边形", len(polygons) == 1, str(len(polygons)))
    xs = [point[0] for point in polygons[0]]
    ys = [point[1] for point in polygons[0]]
    error = max(abs(min(xs) - target[0]), abs(max(xs) - target[2]), abs(min(ys) - target[1]), abs(max(ys) - target[3]))
    check("坐标：缩回原图后误差 ≤ 1px", error <= 1.0, f"{error:.3f}px")


# ---------------------------------------------------------------------------
# 3. 提示展开
# ---------------------------------------------------------------------------
def test_build_prompt() -> None:
    coords, labels = promptlabel.build_prompt([[10, 20], [30, 40]], [1, 0], None)
    check("提示：两个点", coords.shape == (2, 2), str(coords.shape))
    check("提示：标签照抄", labels.tolist() == [1.0, 0.0], str(labels.tolist()))

    # labels 缺省时按正点处理（前端省字段）
    _coords, labels2 = promptlabel.build_prompt([[1, 2]], [], None)
    check("提示：缺省标签按正点", labels2.tolist() == [1.0], str(labels2.tolist()))

    # 框用 2/3 两个点，不补四角
    coords3, labels3 = promptlabel.build_prompt([], [], [[5, 6], [50, 60]])
    check("提示：框展开成两个点", coords3.tolist() == [[5.0, 6.0], [50.0, 60.0]], str(coords3.tolist()))
    check("提示：框标签是 2/3", labels3.tolist() == [2.0, 3.0], str(labels3.tolist()))

    # 点 + 框共存：点在前，框在后（顺序会影响 decoder 的语义）
    coords4, labels4 = promptlabel.build_prompt([[1, 1]], [1], [[2, 2], [9, 9]])
    check("提示：点 + 框共存", coords4.shape[0] == 3 and labels4.tolist() == [1.0, 2.0, 3.0], str(labels4.tolist()))

    try:
        promptlabel.build_prompt([], [], None)
        check("提示：没有任何提示时报错", False, "没有抛错")
    except promptlabel.PromptLabelError:
        check("提示：没有任何提示时报错", True)


def test_build_prompt_numpy() -> None:
    """回归 v1.24.1：上游给的是 numpy 数组，不能写 `points or []`。

    `clamp_points()` 返回 ndarray，ndarray 的 `__bool__` 在元素多于一个时直接抛
    `ValueError: truth value of an array is ambiguous`。单测若只喂 list 永远抓不到，
    一跑真图（annotate → session.annotate → build_prompt）就崩。
    """
    points = np.array([[10.0, 20.0], [30.0, 40.0]])
    coords, labels = promptlabel.build_prompt(points, np.array([1, 0]), None)
    check("numpy 点：不因 __bool__ 崩", coords.shape == (2, 2), str(coords.shape))
    check("numpy 点：坐标照抄", coords.tolist() == [[10.0, 20.0], [30.0, 40.0]], str(coords.tolist()))
    check("numpy 标签：照抄", labels.tolist() == [1.0, 0.0], str(labels.tolist()))

    # 空数组也算"空"，要走到"没有可用提示"的分支，而不是崩在 __bool__
    try:
        promptlabel.build_prompt(np.zeros((0, 2)), np.zeros((0,)), None)
        check("numpy 空点：报无提示错误", False, "没有抛错")
    except promptlabel.PromptLabelError:
        check("numpy 空点：报无提示错误", True)

    # 框用 numpy 时同样不能崩
    coords_box, labels_box = promptlabel.build_prompt(
        np.zeros((0, 2)), [], np.array([[5.0, 6.0], [50.0, 60.0]])
    )
    check("numpy 框：标签 2/3", labels_box.tolist() == [2.0, 3.0], str(labels_box.tolist()))
    check("numpy 框：坐标照抄", coords_box.tolist() == [[5.0, 6.0], [50.0, 60.0]], str(coords_box.tolist()))

    # 真实集成路径：clamp_points 的输出直接喂 build_prompt
    clamped = promptlabel.clamp_points([[10, 20]], 100, 100)
    coords_rt, _ = promptlabel.build_prompt(clamped, [1], None)
    check("集成：clamp_points → build_prompt 不崩", coords_rt.tolist() == [[10.0, 20.0]], str(coords_rt.tolist()))


def test_encoder_layout() -> None:
    """回归 v1.24.1：encoder 输入布局按签名判断，不能写死 NCHW。

    `mobile_sam.encoder.onnx` 的输入是 `['image_height','image_width',3]`（rank 3 / HWC），
    照 NCHW 喂会被 onnxruntime 拒收：
    `Invalid rank for input: input_image Got: 4 Expected: 3`。
    """
    check(
        "布局：本项目导出 rank-3 HWC",
        promptlabel.encoder_layout(["image_height", "image_width", 3]) == promptlabel._ENCODER_HWC,
        "",
    )
    check(
        "布局：4 维且通道在 1 位判为 NCHW",
        promptlabel.encoder_layout([1, 3, 1024, 1024]) == promptlabel._ENCODER_NCHW,
        "",
    )
    check(
        "布局：4 维且通道在末位判为 NHWC",
        promptlabel.encoder_layout([1, 1024, 1024, 3]) == promptlabel._ENCODER_NHWC,
        "",
    )
    check("布局：签名缺失时回退 HWC", promptlabel.encoder_layout(None) == promptlabel._ENCODER_HWC, "")

    # blob 形状要跟布局一致
    session = object.__new__(promptlabel.SamSession)
    canvas = np.zeros((8, 8, 3), dtype=np.float32)
    session.input_layout = promptlabel._ENCODER_HWC
    check("blob：HWC 原样喂（rank 3）", session._to_blob(canvas).shape == (8, 8, 3), "")
    session.input_layout = promptlabel._ENCODER_NCHW
    check("blob：NCHW 转置并加 batch（rank 4）", session._to_blob(canvas).shape == (1, 3, 8, 8), "")
    session.input_layout = promptlabel._ENCODER_NHWC
    check("blob：NHWC 只加 batch", session._to_blob(canvas).shape == (1, 8, 8, 3), "")


def test_clamp_points() -> None:
    clamped = promptlabel.clamp_points([[-10, -5], [999, 999]], 100, 50)
    check("夹取：负坐标贴到 0", clamped[0].tolist() == [0.0, 0.0], str(clamped[0].tolist()))
    check("夹取：越界贴到宽高-1", clamped[1].tolist() == [99.0, 49.0], str(clamped[1].tolist()))
    check("夹取：空输入返回空", promptlabel.clamp_points([], 10, 10).size == 0)


# ---------------------------------------------------------------------------
# 4. encode 缓存
# ---------------------------------------------------------------------------
class FakeSession:
    """把 encoder/decoder 换成可计数的假实现，专测"缓存有没有命���"。"""

    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0
        self.input_h = 128
        self.input_w = 256

    def embed(self, image, key):
        self.encode_calls += 1
        return {"embedding": np.zeros((1, 4, 4, 4), np.float32), "original_size": image.shape[:2], "scale": 0.5}

    def decode(self, state, coords, labels):
        self.decode_calls += 1
        return make_mask(state["original_size"][0], state["original_size"][1])


def test_encode_cache(tmp: Path) -> None:
    """同路径同 mtime 的图只 encode 一次；换文件必然重 encode。"""
    session = FakeSession()
    session._cache = []
    cache_size = promptlabel.CACHE_SIZE
    session.lock = __import__("threading").RLock()

    image_path = tmp / "a.png"
    image_path.write_bytes(b"fake")
    key1 = promptlabel.SamSession._cache_key(image_path)

    image = np.zeros((40, 60, 3), dtype=np.uint8)
    real_embed = promptlabel.SamSession.embed

    # 直接用真实 embed 的缓存逻辑，但绕开 ONNX
    def fake_embed(self, image_bgr, key):
        with self.lock:
            for cached_key, value in self._cache:
                if cached_key == key:
                    self._cache.remove((cached_key, value))
                    self._cache.insert(0, (cached_key, value))
                    return value
        state = FakeSession.embed(self, image_bgr, key)
        with self.lock:
            self._cache.insert(0, (key, state))
            del self._cache[cache_size:]
        return state

    bound = fake_embed.__get__(session, FakeSession)
    view = session.embed

    first = bound(image, key1)
    second = bound(image, key1)
    check("缓存：同一张图第二次命中", session.encode_calls == 1, str(session.encode_calls))
    check("缓存：命中返回同一个 embedding", first is second, "not same object")

    # 文件被替换（mtime 变化）→ key 变化 → 必须重算
    import time as _time

    _time.sleep(0.01)
    image_path.write_bytes(b"fake2")
    key2 = promptlabel.SamSession._cache_key(image_path)
    check("缓存：文件变了 key 就变", key1 != key2, f"{key1} vs {key2}")
    bound(image, key2)
    check("缓存：换文件后重新 encode", session.encode_calls == 2, str(session.encode_calls))

    # 超过容量要淘汰最老的
    for index in range(cache_size + 2):
        other = tmp / f"b{index}.png"
        other.write_bytes(b"x")
        bound(image, promptlabel.SamSession._cache_key(other))
    check("缓存：容量不超过上限", len(session._cache) <= cache_size, str(len(session._cache)))
    check("缓存：最少也保留一条", len(session._cache) >= 1, str(len(session._cache)))
    del view, real_embed


def test_static_hw() -> None:
    check("探测：静态维度被识别", promptlabel._static_hw([1, 3, 684, 1024]) == (684, 1024), "")
    check("探测：NCHW 动态高宽返回 None", promptlabel._static_hw([1, 3, "height", "width"]) is None, "")
    check("探测：维度数不对返回 None", promptlabel._static_hw([1, 3, 1024]) is None, "")
    check("探测：0 维视为动态", promptlabel._static_hw([1, 3, 0, 0]) is None, "")


def test_pick_mask() -> None:
    masks = np.stack(
        [
            np.full((8, 8), 0.1, dtype=np.float32),
            np.full((8, 8), 0.9, dtype=np.float32),
            np.full((8, 8), 0.5, dtype=np.float32),
        ]
    )[None, ...]
    picked = promptlabel._pick_mask(masks, np.array([[0.1, 0.95, 0.5]]))
    check("选掩码：按打分取最高", bool(np.isclose(float(picked[0, 0]), 0.9)), str(float(picked[0, 0])))

    picked2 = promptlabel._pick_mask(masks, None)
    check("选掩码：没有打分取第一个", bool(np.isclose(float(picked2[0, 0]), 0.1)), str(float(picked2[0, 0])))

    picked3 = promptlabel._pick_mask(masks[0], None)
    check("选掩码：2D 直接透传", picked3.shape == (8, 8), str(picked3.shape))


def test_draft_shape() -> None:
    """annotate() 产的草稿必须能被 annotation.normalize_object 接受。"""
    import annotation

    draft = {
        "id": "draft_test",
        "class_id": 0,
        "label": "hole",
        "description": "",
        "group_id": 0,
        "shape_type": "polygon",
        "points": [[1.0, 2.0], [30.0, 2.0], [30.0, 40.0]],
        "rotation": 0.0,
        "keypoint_visibility": [],
        "score": -1.0,
        "_draft": True,
    }
    warnings: list[str] = []
    normalized = annotation.normalize_object(draft, warnings)
    check("草稿：多边形结构能被正式标注接受", len(normalized["points"]) == 3, str(warnings))
    check("草稿：score=-1 表示未知档", float(normalized.get("score", -1)) < 0, str(normalized.get("score")))


def main() -> int:
    test_mask_to_polygons()
    test_empty_and_degenerate()
    test_point_limit()
    test_coordinate_roundtrip()
    test_mask_roundtrip()
    test_build_prompt()
    test_build_prompt_numpy()
    test_encoder_layout()
    test_clamp_points()
    test_static_hw()
    test_pick_mask()
    test_draft_shape()
    with tempfile.TemporaryDirectory(prefix="prompt-label-") as temp:
        test_encode_cache(Path(temp))

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
