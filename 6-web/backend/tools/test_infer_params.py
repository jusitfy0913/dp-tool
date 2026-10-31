# -*- coding: utf-8 -*-
"""推理控制台后端验收脚本（不需要 GPU，也不需要启动 Web 服务）。

覆盖变更 2 的六类风险点：
1. 阈值透传语义：`-1` 不覆盖、`>=0` 覆盖，且不同阈值不得共用句柄；
2. 标签映射：有标签文件 / 无标签文件 / 标签缺字段三种情况都不报错；
3. 配色一致性：后端绘制与下发的 `class_colors` 取同一个值；
4. 句柄缓存 LRU：上限触发淘汰、命中刷新顺序、容量内不重建；
5. 模型扫描：任务类型推断、配套文件附带、越界根目录被拒；
6. 热能图：正常渲染、缺图不报错、非法色带回退。

用法：
    python 6-web/backend/tools/test_infer_params.py

设计说明（为什么用假 DLL 而不是真引擎）：
    真引擎要 GPU、要几十秒加载、要上百 MB 显存，放进每次都能跑的验收脚本里
    会让人不愿意跑。这里用最小可用的假对象替换 DLL 边界，只验证"我们这一层的
    数据搬运与判定逻辑"，DLL 自身的行为由 C++ 侧测试覆盖。
"""

from __future__ import annotations

import ctypes
import json
import sys
import tempfile
from pathlib import Path

# 允许直接以脚本方式运行（把 backend 加入 sys.path）
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import infer  # noqa: E402
import labels  # noqa: E402
import models  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def close(actual, expected, tol: float = 1e-5) -> bool:
    """浮点比较用容差。

    阈值会经过 `c_float`（32 位），0.8 存进去再读出来是 0.800000011920929，
    直接用 `==` 断言会误报失败。这是真实的类型事实，不是实现缺陷。
    """
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeMetadata:
    """模拟 ai_metadata：字段与 C 结构体同名，只填本层会用到的。"""

    def __init__(self, image_threshold: float, pixel_threshold: float,
                 map_min: float = 0.0, map_max: float = 1.0):
        self.model_type = b"PatchCore"
        self.normalization = b"none"
        self.output_contract = b"anomalib_post_processed_0_1"
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.score_min = 0.0
        self.score_max = 1.0
        self.map_min = map_min
        self.map_max = map_max
        self.input_h = 256
        self.input_w = 256


def _num(value) -> float:
    """把 ctypes 标量或不带 argtypes 时的裸值统一转成 float。

    真实 DLL 有 argtypes，ctypes 会自动把 `c_float` 拆成 float；
    假对象没有 argtypes，收到的仍是 `c_float` 包装对象，这里补一步拆包。
    """
    return float(getattr(value, "value", value))


class FakeAnomalibLib:
    """假 anomalib DLL：记录 ai_create 收到的阈值，模拟 override 语义。

    关键点：真实 DLL 的 override 会**原地改写内部 meta_**，
    所以 ai_get_metadata 返回的是生效值。这里必须复现这个行为，
    否则"响应里同时给出生效值与原值"这条就测不到真正的坑。
    """

    def __init__(self, file_image_thr: float = 0.5, file_pixel_thr: float = 0.5):
        self.file_image_thr = file_image_thr
        self.file_pixel_thr = file_pixel_thr
        self.created: list[tuple[float, float]] = []
        self.destroyed: list[int] = []
        self._next_handle = 1000
        self._effective: dict[int, tuple[float, float]] = {}

    def ai_create(self, engine, metadata, model_type, image_override, pixel_override):
        image_override, pixel_override = _num(image_override), _num(pixel_override)
        self.created.append((image_override, pixel_override))
        handle = self._next_handle
        self._next_handle += 1
        image = image_override if image_override >= 0 else self.file_image_thr
        pixel = pixel_override if pixel_override >= 0 else self.file_pixel_thr
        self._effective[handle] = (image, pixel)
        return handle

    def ai_get_metadata(self, handle, out):
        image, pixel = self._effective[handle]
        # byref 出来的是 CArgObject，不能直接改属性；先 cast 回结构体再逐字段写，
        # 这样才真正模拟"DLL 把生效阈值写回调用方缓冲区"。
        meta = ctypes.cast(out, ctypes.POINTER(infer.AiMetadata)).contents
        meta.image_threshold = image
        meta.pixel_threshold = pixel
        meta.map_min = 0.0
        meta.map_max = 1.0
        meta.input_h = 256
        meta.input_w = 256
        return 0

    def ai_infer_rgb(self, handle, rgb, width, height, stride, out_result, out_stat):
        # 造一份最小结果：raw_score 固定 0.42，归一化分数按"生效图像阈值"折算，
        # 这样 is_anomaly 会随阈值翻转，正好用来验证"判定依据覆盖后的阈值"。
        image, _ = self._effective[handle]
        normalized = 0.42 / image if image > 0 else 0.0
        result = infer.AiResult()
        result.raw_score = 0.42
        result.normalized_score = normalized
        result.is_anomaly = 1 if normalized >= image else 0
        self._result = result  # 保持引用，否则指针立即失效

        ptr = ctypes.cast(out_result, ctypes.POINTER(ctypes.POINTER(infer.AiResult)))
        ptr[0] = ctypes.pointer(self._result)

        stat = ctypes.cast(out_stat, ctypes.POINTER(infer.AiFrameStat)).contents
        stat.preprocess_ms = 0.5
        stat.inference_ms = 1.0
        stat.postprocess_ms = 0.2
        stat.total_ms = 1.7
        return 0

    def ai_destroy(self, handle):
        self.destroyed.append(handle)

    def ai_last_error(self):
        return b""


class FakeResult:
    """模拟 ai_result：只做最小可渲染的数据。"""

    def __init__(self):
        self.source_rgb = None
        self.source_w = 0
        self.source_h = 0
        self.source_stride = 0
        self.overlay_rgb = None
        self.overlay_w = 0
        self.overlay_h = 0
        self.overlay_stride = 0
        self.mask_rgb = None
        self.mask_w = 0
        self.mask_h = 0
        self.mask_stride = 0
        self.normalized_map = None
        self.map_w = 0
        self.map_h = 0
        self.raw_score = 0.42
        self.normalized_score = 0.42
        self.is_anomaly = 0


class FakeYoloLib:
    """假 yolo DLL：记录 yi_create 收到的 conf/nms，模拟句柄生命周期。"""

    def __init__(self):
        self.created: list[tuple[float, float]] = []
        self.destroyed: list[int] = []
        self._next_handle = 5000

    def yi_create(self, engine, task, conf, nms):
        self.created.append((_num(conf), _num(nms)))
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def yi_destroy(self, handle):
        self.destroyed.append(handle)

    def yi_last_error(self):
        return b""


# ---------------------------------------------------------------------------
# 1. 标签解析与配色
# ---------------------------------------------------------------------------
def test_labels(tmp: Path) -> None:
    good = tmp / "labels_good.json"
    good.write_text(json.dumps({"name": "pcb", "classes": ["missing_hole", "short"]}), encoding="utf-8")

    info = labels.load_labels(good)
    check("标签：正常文件解析成功", info["ok"] and info["classes"] == ["missing_hole", "short"], str(info))
    check("标签：name 被读取", info["name"] == "pcb", info["name"])
    check("标签：类别名映射", labels.name_for(1, info["classes"]) == "short")
    check("标签：越界回退 cls{id}", labels.name_for(9, info["classes"]) == "cls9")

    # 带 BOM 的文件（Windows 记事本常见）不能报错
    bom = tmp / "labels_bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps({"classes": ["a"]}).encode("utf-8"))
    check("标签：容忍 BOM", labels.load_labels(bom)["ok"])

    # 缺 classes 字段
    no_classes = tmp / "labels_no_classes.json"
    no_classes.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    info = labels.load_labels(no_classes)
    check("标签：缺 classes 时不抛错且 ok=False", (not info["ok"]) and bool(info["error"]), str(info))
    check("标签：缺 classes 时回退 cls0", labels.name_for(0, info["classes"]) == "cls0")

    # classes 里混空串与 null
    messy = tmp / "labels_messy.json"
    messy.write_text(json.dumps({"classes": ["a", "", None, "b"]}), encoding="utf-8")
    info = labels.load_labels(messy)
    check("标签：过滤空项", info["ok"] and info["classes"] == ["a", "b"], str(info))

    # 文件不存在 / 未提供
    missing = labels.load_labels(tmp / "nope.json")
    check("标签：文件不存在不抛错", (not missing["ok"]) and "不存在" in missing["error"])
    check("标签：未提供路径不抛错", (not labels.load_labels(None)["ok"]))

    # 配色：同 id 恒定、不同 id 不同色、跨越人工序列边界仍稳定
    check("配色：同 id 恒定", labels.color_for(3) == labels.color_for(3))
    check("配色：相邻 id 不同色", labels.color_for(0) != labels.color_for(1))
    check("配色：超出人工序列仍稳定", labels.color_for(40) == labels.color_for(40))
    colors = labels.class_colors(5)
    check("配色：键为字符串 id", set(colors.keys()) == {"0", "1", "2", "3", "4"}, str(colors.keys()))
    check("配色：每个色都是 3 通道", all(len(v) == 3 for v in colors.values()))
    check("配色：通道在 0~255", all(0 <= c <= 255 for v in colors.values() for c in v))


# ---------------------------------------------------------------------------
# 2. 句柄缓存 LRU
# ---------------------------------------------------------------------------
def test_lru() -> None:
    destroyed: list[int] = []
    cache = infer._HandleCache(3, lambda h: destroyed.append(h))

    for i in range(3):
        cache.put((i,), 100 + i)
    check("LRU：容量内不销毁", not destroyed and len(cache) == 3, str(destroyed))

    # 命中一次 (0)，把它刷到队尾；(1) 应该变成最久未用
    check("LRU：命中返回句柄", cache.get((0,)) == 100)
    check("LRU：命中不改容量", len(cache) == 3)

    cache.put((3,), 103)
    check("LRU：超限淘汰一个", len(cache) == 3 and destroyed == [101], str(destroyed))
    check("LRU：淘汰的是最久未用（被刷新的 (0) 幸存）",
          (0,) in cache.keys() and (1,) not in cache.keys(), str(cache.keys()))
    check("LRU：淘汰时立即销毁", 101 in destroyed)

    # 连续写入多个，保持上限
    for i in range(10, 20):
        cache.put((i,), 200 + i)
    check("LRU：连续写入后仍不超上限", len(cache) == 3, str(len(cache)))
    check("LRU：淘汰次数正确", len(destroyed) == 11, str(len(destroyed)))

    cache.clear()
    check("LRU：clear 全部销毁", len(cache) == 0 and len(destroyed) == 14, str(len(destroyed)))


# ---------------------------------------------------------------------------
# 3. 异常检测：阈值透传与生效值/原值区分
# ---------------------------------------------------------------------------
def test_threshold_passthrough(tmp: Path) -> None:
    metadata_file = tmp / "metadata.json"
    metadata_file.write_text(json.dumps({
        "image_threshold": 0.5,
        "pixel_threshold": 0.4,
        "anomaly_maps": {"min": 0.0, "max": 1.0},
    }), encoding="utf-8")
    engine_file = tmp / "model_fp16.engine"
    engine_file.write_bytes(b"\x00")

    fake = FakeAnomalibLib(file_image_thr=0.5, file_pixel_thr=0.4)
    original_lib = infer._anomalib_lib
    original_handles = infer._anom_handles
    infer._anomalib_lib = fake
    infer._anom_handles = infer._HandleCache(4, fake.ai_destroy)
    try:
        # 不覆盖：应把 -1 传给 ai_create，且生效值等于文件原值
        payload = infer.run_anomalib(
            str(engine_file), str(metadata_file), _fake_bgr(),
            want_overlay=False, want_heatmap=False,
        )
        check("阈值：未覆盖时向 DLL 传 -1", fake.created[-1] == (-1.0, -1.0), str(fake.created))
        check("阈值：未覆盖时生效值=原值",
              close(payload["thresholds"]["image"]["effective"], 0.5)
              and close(payload["thresholds"]["image"]["metadata"], 0.5), str(payload["thresholds"]))
        check("阈值：未覆盖时 overridden=false", payload["thresholds"]["image"]["overridden"] is False)

        # 覆盖：应把用户值传给 ai_create，生效值为用户值，原值仍为文件值
        payload = infer.run_anomalib(
            str(engine_file), str(metadata_file), _fake_bgr(),
            want_overlay=False, want_heatmap=False,
            image_threshold=0.8, pixel_threshold=0.2,
        )
        sent = fake.created[-1]
        check("阈值：覆盖时向 DLL 传用户值",
              close(sent[0], 0.8) and close(sent[1], 0.2), str(sent))
        check("阈值：生效值=用户值",
              close(payload["thresholds"]["image"]["effective"], 0.8),
              str(payload["thresholds"]["image"]["effective"]))
        check("阈值：原值仍为 metadata 值",
              close(payload["thresholds"]["image"]["metadata"], 0.5),
              str(payload["thresholds"]["image"]["metadata"]))
        check("阈值：overridden=true", payload["thresholds"]["image"]["overridden"] is True)
        check("阈值：像素阈值同样生效",
              close(payload["thresholds"]["pixel"]["effective"], 0.2)
              and close(payload["thresholds"]["pixel"]["metadata"], 0.4), str(payload["thresholds"]))

        # 关键回归：ai_get_metadata 返回的是"覆盖后"的值，metadata 字段不能当成原值
        check("阈值：metadata 字段反映生效值（DLL 会原地改写）",
              close(payload["metadata"]["image_threshold"], 0.8), str(payload["metadata"]))

        # 阈值必须进入缓存键：前两次用了两组不同阈值 = 两次 ai_create
        check("阈值：不同阈值建两个句柄", len(fake.created) == 2, str(fake.created))

        # 同阈值再来一次 = 命中缓存，不得再创建
        infer.run_anomalib(
            str(engine_file), str(metadata_file), _fake_bgr(),
            want_overlay=False, want_heatmap=False,
            image_threshold=0.8, pixel_threshold=0.2,
        )
        check("阈值：同阈值不重复建句柄", len(fake.created) == 2, str(fake.created))

        # 判定必须依据生效阈值：raw=0.42，阈值 0.1 时归一化分数 > 1，判定 NG
        low = infer.run_anomalib(
            str(engine_file), str(metadata_file), _fake_bgr(),
            want_overlay=False, want_heatmap=False,
            image_threshold=0.1, pixel_threshold=0.2,
        )
        check("阈值：降低阈值后判定翻转为 NG", low["is_anomaly"] is True, str(low["is_anomaly"]))
        check("阈值：第三组阈值又建一个句柄", len(fake.created) == 3, str(fake.created))
    finally:
        infer._anomalib_lib = original_lib
        infer._anom_handles = original_handles


def _fake_bgr():
    """3x3 的假图，只要求能满足 cvtColor 与指针取址。"""
    import numpy as np

    return np.zeros((3, 3, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 4. 热能图
# ---------------------------------------------------------------------------
def test_heatmap() -> None:
    import numpy as np

    # 正常渲染
    values = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    result = FakeResult()
    result.normalized_map = values.ctypes.data_as(infer.ctypes.POINTER(infer.ctypes.c_float))
    result.map_w, result.map_h = 8, 8
    result.source_w, result.source_h = 16, 16

    png, name = infer._render_heatmap(result, FakeMetadata(0.5, 0.5))
    check("热能图：正常渲染出 PNG", bool(png) and png[:4] == b"\x89PNG", str(name))
    check("热能图：默认色带为 TURBO", name == "TURBO", name)

    # 缺图（指针为空 / 尺寸为 0）不报错
    empty = FakeResult()
    png, name = infer._render_heatmap(empty, FakeMetadata(0.5, 0.5))
    check("热能图：指针为空返回 None", png is None, str(png))

    zero = FakeResult()
    zero.normalized_map = values.ctypes.data_as(infer.ctypes.POINTER(infer.ctypes.c_float))
    zero.map_w, zero.map_h = 0, 8
    png, _ = infer._render_heatmap(zero, FakeMetadata(0.5, 0.5))
    check("热能图：尺寸为 0 返回 None", png is None)

    # 全图同值不能除以 0
    flat = FakeResult()
    flat_values = np.full((4, 4), 0.3, dtype=np.float32)
    flat.normalized_map = flat_values.ctypes.data_as(infer.ctypes.POINTER(infer.ctypes.c_float))
    flat.map_w, flat.map_h = 4, 4
    flat.source_w, flat.source_h = 4, 4
    png, _ = infer._render_heatmap(flat, FakeMetadata(0.5, 0.5))
    check("热能图：全图同值不除零且仍出图", bool(png) and png[:4] == b"\x89PNG")

    # 非法色带回退
    _, name = infer.resolve_colormap("NOT_A_COLORMAP")
    check("色带：非法名称回退默认", name == "TURBO", name)
    _, name = infer.resolve_colormap("jet")
    check("色带：大小写不敏感", name == "JET", name)
    _, name = infer.resolve_colormap("COLORMAP_VIRIDIS")
    check("色带：容忍带前缀写法", name == "VIRIDIS", name)


# ---------------------------------------------------------------------------
# 5. 绘制与下发配色一致
# ---------------------------------------------------------------------------
def test_color_consistency() -> None:
    import numpy as np

    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [{
        "left": 8, "top": 8, "right": 40, "bottom": 40,
        "confidence": 0.9, "class_label": 2, "label_name": "open_circuit",
    }]
    colors = labels.class_colors(6)

    # 让 draw_yolo 生成带框图，再与"下发配色"逐通道比对：
    # 取框左上角像素，应当等于下发配色的 RGB 反序（OpenCV 是 BGR）
    png = infer.draw_yolo(canvas, boxes, colors)
    decoded = _decode_png(png)
    expected_b, expected_g, expected_r = colors["2"][2], colors["2"][1], colors["2"][0]
    hit = (
        int(decoded[8, 8, 0]) == expected_b
        and int(decoded[8, 8, 1]) == expected_g
        and int(decoded[8, 8, 2]) == expected_r
    )
    check("配色：绘制用的颜色与下发一致", hit,
          f"期望 BGR=({expected_b},{expected_g},{expected_r}) 实际={decoded[8, 8].tolist()}")

    # 不传配色时按同一纯函数现算，结果应完全相同
    png_auto = infer.draw_yolo(canvas, boxes, None)
    decoded_auto = _decode_png(png_auto)
    check("配色：不传配色时结果相同", decoded_auto[8, 8].tolist() == decoded[8, 8].tolist(),
          str(decoded_auto[8, 8].tolist()))

    # attach_labels 补的类别名与配色要和 draw_yolo 用的一致
    result = {"boxes": [{"class_label": 2, "confidence": 0.9}]}
    infer.attach_labels(result, ["a", "b", "open_circuit", "d", "e", "f"])
    check("标签：attach_labels 写入 label_name", result["boxes"][0]["label_name"] == "open_circuit")
    check("标签：attach_labels 下发配色", result["class_colors"]["2"] == colors["2"])

    # 无标签文件时也要有配色，否则前端取不到颜色会全变成同一种
    bare = {"boxes": [{"class_label": 4, "confidence": 0.5}]}
    infer.attach_labels(bare, None)
    check("标签：无标签文件仍有配色", "4" in bare["class_colors"], str(bare["class_colors"].keys()))
    check("标签：无标签文件回退 cls4", bare["boxes"][0]["label_name"] == "cls4")


def _decode_png(png: bytes):
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# 6. 模型扫描
# ---------------------------------------------------------------------------
def test_model_scan(tmp: Path) -> None:
    # 构造一个"像 YOLO"的目录：引擎名带 _fp16，ONNX 是 .transd，标签在上一级
    run = tmp / "runs" / "pcb" / "detect_v1"
    weights = run / "weights"
    weights.mkdir(parents=True)
    (weights / "best.transd_fp16.engine").write_bytes(b"\x00" * 16)
    (weights / "best_fp16.engine").write_bytes(b"\x00" * 16)
    (weights / "best.transd.onnx").write_bytes(b"\x00" * 8)
    (weights / "best.onnx").write_bytes(b"\x00" * 8)
    (run / "labels_detect_v1.json").write_text(json.dumps({"classes": ["a", "b"]}), encoding="utf-8")
    (run / "args.yaml").write_text("task: segment\n", encoding="utf-8")

    # 构造一个"像 Anomalib"的目录：多两层固定目录，metadata 与引擎同目录
    anom_dir = tmp / "results" / "Patchcore" / "exp1" / "v0" / "weights" / "lightning"
    anom_dir.mkdir(parents=True)
    (anom_dir / "model_fp16.engine").write_bytes(b"\x00" * 32)
    (anom_dir / "model.onnx").write_bytes(b"\x00" * 8)
    (anom_dir / "metadata.json").write_text(json.dumps({
        "model_type": "Patchcore", "input_size": [256, 256],
    }), encoding="utf-8")
    # 残缺 run：只有目录没有引擎，不应影响扫描
    (tmp / "runs" / "pcb" / "half_done" / "weights").mkdir(parents=True)

    # 临时把两个扫描根指向测试目录
    original_targets = models.SCAN_TARGETS
    models.SCAN_TARGETS = [
        ("yolo", lambda: tmp / "runs"),
        ("anomalib", lambda: tmp / "results"),
    ]
    # 测试目录在临时区，需要临时放行白名单
    original_repo = config.REPO_ROOT
    config.REPO_ROOT = tmp
    try:
        data = models.scan_models()
        entries = data["entries"]
        check("扫描：三个引擎全部找到", len(entries) == 3, str([e["name"] for e in entries]))
        check("扫描：根目录都标记为可用", all(r["ok"] for r in data["roots"]), str(data["roots"]))

        by_name = {e["name"]: e for e in entries}
        check("扫描：引擎名集合正确",
              set(by_name) == {"best.transd_fp16.engine", "best_fp16.engine", "model_fp16.engine"},
              str(set(by_name)))

        yolo = by_name["best.transd_fp16.engine"]
        check("扫描：任务类型优先取 args.yaml（segment）", yolo["task"] == "segment", yolo["task"])
        check("扫描：配对去精度后缀后的 ONNX",
              yolo["onnx"] is not None and yolo["onnx"].endswith("best.transd.onnx"), str(yolo["onnx"]))
        check("扫描：transd ONNX 被标记为已转置", yolo["transposed"] is True)
        check("扫描：标签文件在上级目录也能找到",
              yolo["labels"] is not None and yolo["labels"].endswith("labels_detect_v1.json"),
              str(yolo["labels"]))

        # 未转置的那一个必须被正确标注，否则前端无法提醒用户"这个引擎不能用"
        plain = by_name["best_fp16.engine"]
        check("扫描：未转置引擎被标为 transposed=False", plain["transposed"] is False)
        check("扫描：未转置引擎配对的 ONNX 是 best.onnx",
              plain["onnx"] is not None and plain["onnx"].endswith("best.onnx"), str(plain["onnx"]))

        anom = by_name["model_fp16.engine"]
        check("扫描：Anomalib 任务类型", anom["task"] == "anomalib", anom["task"])
        check("扫描：metadata 与引擎同目录", anom["metadata"] is not None
              and anom["metadata"].endswith("metadata.json"), str(anom["metadata"]))
        check("扫描：读取 model_type", anom["model_type"] == "Patchcore", str(anom["model_type"]))
        check("扫描：读取 input_size", anom["input_size"] == [256, 256], str(anom["input_size"]))
        check("扫描：Anomalib 没有标签文件", anom["labels"] is None, str(anom["labels"]))

        # 排序：修改时间倒序
        mtimes = [e["mtime"] for e in entries]
        check("扫描：按修改时间倒序", mtimes == sorted(mtimes, reverse=True), str(mtimes))

        # 上限
        limited = models.scan_models(limit=1)
        check("扫描：limit 生效", len(limited["entries"]) == 1 and limited["truncated"] is True,
              str(len(limited["entries"])))
        check("扫描：total 反映真实总数", limited["total"] == 3, str(limited["total"]))

        # 白名单外：换一个不在 tmp 下的根，应被拒绝且给出原因
        outside = Path(tempfile.gettempdir()) / "tf_outside_root_xyz"
        outside.mkdir(exist_ok=True)
        models.SCAN_TARGETS = [("yolo", lambda: outside)]
        blocked = models.scan_models()
        check("扫描：白名单外目录不扫描", blocked["entries"] == [], str(len(blocked["entries"])))
        check("扫描：白名单外给出原因",
              any("允许范围" in (r.get("reason") or "") for r in blocked["roots"]), str(blocked["roots"]))
    finally:
        models.SCAN_TARGETS = original_targets
        config.REPO_ROOT = original_repo


# ---------------------------------------------------------------------------
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tf_infer_test_"))
    try:
        test_labels(tmp)
        test_lru()
        test_threshold_passthrough(tmp)
        test_heatmap()
        test_color_consistency()
        test_model_scan(tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
