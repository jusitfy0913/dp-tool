# -*- coding: utf-8 -*-
"""标签文件解析与类别配色。

为什么单独成模块：
    类别名与配色是「后端烧录图」和「前端矢量叠加层」共用的同一份数据。
    一旦两边各自计算，导出的图和屏幕上看到的就会是两个样子。
    所以解析与配色只在这里实现一次，两边都从接口响应里取。

标签文件格式（launcher 导出）：
    {"name": "detect_v1", "classes": ["missing_hole", "short", ...]}

容忍策略：
    现场拿到的标签文件未必规范——可能只有 classes 没有 name，可能 classes 里
    混了数字，也可能是空数组。这些都不该让推理接口报错：
    解析失败就当没有标签文件，退回 `cls{id}`，并在返回值里说明原因。
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

# 前 12 个类别用人工挑选的色相：彼此在色环上拉开，且在浅色 PCB 与深色面板上都看得清。
# 之所以不用「纯按 id 取模」直接从色环等分：等分 12 份时相邻色相差 30°，
# 现场常见的 4~6 类场景里，相邻类别容易看混。人工序列把最常用的前几个拉到最大分离。
_CURATED: list[tuple[int, int, int]] = [
    (34, 211, 238),   # 青蓝
    (244, 114, 182),  # 粉
    (250, 204, 21),   # 琥珀
    (74, 222, 128),   # 绿
    (167, 139, 250),  # 紫
    (251, 146, 60),   # 橙
    (56, 189, 248),   # 天蓝
    (248, 113, 113),  # 红
    (163, 230, 53),   # 黄绿
    (232, 121, 249),  # 品红
    (45, 212, 191),   # 青绿
    (148, 163, 184),  # 灰蓝
]

_GOLDEN_ANGLE = 137.508


def _generated(index: int) -> tuple[int, int, int]:
    """超出人工序列后的兜底配色：黄金角散布色相。

    黄金角保证任意前 N 个色相都尽量均匀，不会像「按 id × 30°」那样
    在 N 趋大时出现成片相近的颜色。
    """
    hue = ((index - len(_CURATED)) * _GOLDEN_ANGLE) % 360.0 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.98)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def color_for(class_id: int) -> list[int]:
    """按类别 id 取固定颜色，返回 [r, g, b]。

    纯函数：同样的 id 永远得到同样的颜色，后端绘制与前端叠加层因此天然一致。
    """
    index = max(0, int(class_id))
    if index < len(_CURATED):
        return list(_CURATED[index])
    return list(_generated(index))


def class_colors(count: int) -> dict[str, list[int]]:
    """生成 `count` 个类别的配色表，键为字符串形式的类别 id。

    用字符串键是为了和 JSON 对齐：JSON 对象的键只能是字符串，
    这里直接用字符串，避免前端拿到数字键对象后索引不到。
    """
    total = max(0, int(count))
    return {str(i): color_for(i) for i in range(total)}


def load_labels(path: str | Path | None) -> dict:
    """读取标签文件，返回 `{ok, name, classes, error}`。

    永远不抛异常：调用方只需要看 `ok` 决定是否使用类别名。
    """
    result = {"ok": False, "name": "", "classes": [], "error": ""}
    if path is None or not str(path).strip():
        result["error"] = "未提供标签文件"
        return result

    file = Path(str(path))
    if not file.is_file():
        result["error"] = f"标签文件不存在：{file}"
        return result

    try:
        # 用 utf-8-sig 容忍 Windows 记事本保存时带的 BOM，
        # 否则 json.loads 会因为开头多一个 \ufeff 直接失败。
        raw = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["error"] = f"标签文件解析失败：{exc}"
        return result

    if not isinstance(raw, dict):
        result["error"] = "标签文件根节点不是对象"
        return result

    name = raw.get("name")
    result["name"] = str(name) if isinstance(name, (str, int, float)) else ""

    classes = raw.get("classes")
    if not isinstance(classes, list):
        result["error"] = "标签文件缺少 classes 数组"
        return result

    # 逐项转字符串并丢掉空白项与 null：
    # 现场出现过 classes 里混 null 或空串的标签文件。
    # 这里必须显式排除 None——`str(None)` 是 "None"，会被当成一个合法类别名混进来。
    cleaned = [
        str(item).strip()
        for item in classes
        if item is not None and str(item).strip()
    ]
    if not cleaned:
        result["error"] = "标签文件 classes 为空"
        return result

    result["ok"] = True
    result["classes"] = cleaned
    return result


def name_for(class_id: int, classes: list[str] | None) -> str:
    """把类别号映射成类别名；越界或没有类别表时回退为 `cls{id}`。"""
    label = int(class_id)
    if classes and 0 <= label < len(classes):
        return classes[label]
    return f"cls{label}"
