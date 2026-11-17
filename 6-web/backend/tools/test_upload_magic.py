# -*- coding: utf-8 -*-
"""上传文件头校验的验收脚本（``backend/imagetype.py`` + 推理路由）。

背景：以前只校验扩展名和大小，而扩展名是客户端自己填的字符串。往 ``a.png``
里塞一段任意二进制，字节会直接进 C++ 解码器。这层只做「是不是图片、和扩展名
对不对得上」，不做完整解码。

    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_upload_magic.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402# noqa: E402

from fastapi import HTTPException, UploadFile  # noqa: E402

import imagetype  # noqa: E402
from routers import infer as infer_routes  # noqa: E402


#: 各格式的最小可信文件头（后面补几个字节，够识别即可）
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 12
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
BMP = b"BM" + b"\x00" * 14
TIFF_LE = b"II*\x00" + b"\x00" * 12
TIFF_BE = b"MM\x00*" + b"\x00" * 12
WEBP = b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"\x00" * 4
NOT_IMAGE = b"MZ\x90\x00" + b"\x00" * 12


def rejected(data: bytes, suffix: str) -> bool:
    try:
        imagetype.check(data, suffix)
    except ValueError:
        return True
    return False


def test_sniff() -> None:
    check("识别 JPEG", imagetype.sniff(JPEG) == "jpeg")
    check("识别 PNG", imagetype.sniff(PNG) == "png")
    check("识别 BMP", imagetype.sniff(BMP) == "bmp")
    check("识别 TIFF（小端）", imagetype.sniff(TIFF_LE) == "tiff")
    check("识别 TIFF（大端）", imagetype.sniff(TIFF_BE) == "tiff")
    check("识别 WebP", imagetype.sniff(WEBP) == "webp")
    check("非图片返回 None", imagetype.sniff(NOT_IMAGE) is None)
    check("空数据返回 None", imagetype.sniff(b"") is None)
    check("EXE 头不会被误判", imagetype.sniff(b"MZ" + b"\x00" * 20) is None)


def test_pairs() -> None:
    for suffix, data in (
        (".jpg", JPEG),
        (".jpeg", JPEG),
        (".png", PNG),
        (".bmp", BMP),
        (".tif", TIFF_LE),
        (".tiff", TIFF_BE),
        (".webp", WEBP),
    ):
        check(f"{suffix} 与内容一致通过", not rejected(data, suffix))

    check(".png 里放 EXE 被拒", rejected(NOT_IMAGE, ".png"))
    check(".jpg 里放 PNG 被拒（内容与扩展名不符）", rejected(PNG, ".jpg"))
    check(".png 里放 JPEG 被拒", rejected(JPEG, ".png"))
    check("未知后缀被拒", rejected(PNG, ".gif"))
    check("空数据在调用方处理（这里放行）", not rejected(b"", ".png"))


def test_messages() -> None:
    try:
        imagetype.check(NOT_IMAGE, ".png")
    except ValueError as exc:
        text = str(exc)
        check("错误信息说明不是图片", "不是可识别的图片" in text, text)
    else:
        check("错误信息说明不是图片", False, "没有抛异常")

    try:
        imagetype.check(PNG, ".jpg")
    except ValueError as exc:
        text = str(exc)
        check("错误信息指出两边的真实格式", "PNG" in text and "jpeg" in text.lower(), text)
    else:
        check("错误信息指出两边的真实格式", False, "没有抛异常")


def test_read_upload() -> None:
    def upload(data: bytes, name: str) -> UploadFile:
        return UploadFile(filename=name, file=io.BytesIO(data))

    check("_read_upload 接受合法 PNG", infer_routes._read_upload(upload(PNG, "a.png")) == PNG)

    try:
        infer_routes._read_upload(upload(PNG, "a.jpg"))
    except HTTPException as exc:
        check("_read_upload 拒绝改名后的 PNG", exc.status_code == 400, str(exc.detail))
    else:
        check("_read_upload 拒绝改名后的 PNG", False, "没有抛 HTTPException")

    try:
        infer_routes._read_upload(upload(b"", "a.png"))
    except HTTPException as exc:
        check("空文件仍是 400", exc.status_code == 400, str(exc.detail))
    else:
        check("空文件仍是 400", False, "没有抛 HTTPException")

    try:
        infer_routes._read_upload(upload(PNG + b"\x00" * (33 * 1024 * 1024), "a.png"))
    except HTTPException as exc:
        check("超大文件仍先报 413（大小优先于内容）", exc.status_code == 413, str(exc.detail))
    else:
        check("超大文件仍先报 413（大小优先于内容）", False, "没有抛 HTTPException")

    try:
        infer_routes._read_upload(upload(PNG, "a.gif"))
    except HTTPException as exc:
        check("不支持的扩展名 400", exc.status_code == 400, str(exc.detail))
    else:
        check("不支持的扩展名 400", False, "没有抛 HTTPException")


def test_frame_endpoint() -> None:
    """摄像头单帧接口也必须在解码前拦下非图片内容。"""
    upload = UploadFile(filename="frame.jpg", file=io.BytesIO(NOT_IMAGE))
    try:
        infer_routes.infer_frame(
            image=upload, engine="whatever.engine", task="detect", conf=0.25, nms=0.45, labels=""
        )
    except HTTPException as exc:
        check("单帧接口拒绝非图片内容", exc.status_code == 400, str(exc.detail))
        check("单帧接口在内容处就拒绝（不是后面报引擎不存在）", "图片" in str(exc.detail), str(exc.detail))
    else:
        check("单帧接口拒绝非图片内容", False, "没有抛 HTTPException")


def main() -> int:
    test_sniff()
    test_pairs()
    test_messages()
    test_read_upload()
    test_frame_endpoint()
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
