# -*- coding: utf-8 -*-
"""上传图片的文件头识别（v1.42.0）。

## 为什么光看扩展名不够

`_read_upload` 早就校验了后缀与大小，但后缀是**客户端自己填的字符串**。
往 ``a.png`` 里塞一段任意二进制，后端仍会把它交给 OpenCV/`infer.decode_image`；
换成一个畸形文件就能让解码器走进未定义行为。DLL 是 C++ 写的，那里不该接收
来历不明的字节。

这里只做一件事：读前几个字节，确认它确实是某一种支持的图片，并且与扩展名族
一致（``.jpg`` 里必须是 JPEG，``.png`` 里必须是 PNG）。不做完整解码，也不
参与业务判断——越权/解码交给下面各自那道。

## 为什么按「族」而不是精确后缀

``.jpg`` 与 ``.jpeg`` 是同一族、``.tif`` 与 ``.tiff`` 同理。按族比较既能拦住
「把可执行文件改成 .png」，又不会把合法的 ``.jpeg`` 判错。
"""

from __future__ import annotations

#: 后缀 → 族名
EXTENSION_FAMILY: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".webp": "webp",
}

#: 族名 → 人类可读的名字（错误信息里用）
FAMILY_LABEL = {
    "jpeg": "JPEG",
    "png": "PNG",
    "bmp": "BMP",
    "tiff": "TIFF",
    "webp": "WebP",
}


def sniff(data: bytes) -> str | None:
    """按文件头判断格式，返回族名；识别不了返回 ``None``。"""
    head = bytes(data[:16])
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"BM"):
        return "bmp"
    if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
        return "tiff"
    # WebP: "RIFF" + 4 字节长度 + "WEBP"
    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    return None


def family_of(suffix: str) -> str | None:
    """后缀 → 族名；不认识的后缀返回 ``None``。"""
    return EXTENSION_FAMILY.get(str(suffix or "").lower())


def check(data: bytes, suffix: str) -> None:
    """校验字节与后缀；不通过时抛 ``ValueError``（调用方转 400）。

    空数据不在这里判——调用方已有更贴切的「上传的图片是空的」提示。
    """
    if not data:
        return
    expected = family_of(suffix)
    actual = sniff(data)
    if actual is None:
        raise ValueError(
            "文件内容不是可识别的图片（支持 JPEG / PNG / BMP / TIFF / WebP）。"
            "请确认没有把其它格式的文件改名成图片后缀。"
        )
    if expected is None:
        raise ValueError(f"不支持的图片后缀：{suffix}")
    if actual != expected:
        raise ValueError(
            f"文件内容与扩展名不符：后缀是 {suffix}（{FAMILY_LABEL.get(expected, expected)}），"
            f"实际内容是 {FAMILY_LABEL.get(actual, actual)}"
        )
