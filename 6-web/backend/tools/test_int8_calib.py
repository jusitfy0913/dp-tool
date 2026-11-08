#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INT8 校准的抽图、缓存判断和命令拼装。不编译引擎，也不需要显卡。"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import int8calib  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def main() -> int:
    ten = [Path(f"img{i}.jpg") for i in range(10)]
    picked = int8calib.sample_evenly(ten, 3)
    check("均匀抽 3 张含首尾", picked[0] == ten[0] and picked[-1] == ten[-1] and len(picked) == 3, str(picked))
    check("张数多于图片就全用", len(int8calib.sample_evenly(ten, 500)) == 10)
    check("空目录抽不到", int8calib.sample_evenly([], 100) == [])

    root = Path(__file__).resolve().parent / "_int8_tmp"
    folder = root / "pics"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("a.jpg", "b.png", "c.txt", "sub/d.webp"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    found = int8calib.list_images(folder)
    check("只收集图片并含子目录", len(found) == 3 and all(p.suffix.lower() != ".txt" for p in found), str(found))

    onnx = root / "best.onnx"
    onnx.write_bytes(b"onnx")
    engine = root / "best_int8.engine"
    images = int8calib.sample_evenly(found, 2)
    key = int8calib.cache_key(onnx, images, opt_shape="1x3x640x640", workspace_mb=4096)
    cache = engine.with_suffix(".cache")
    check("没有缓存时不能复用", int8calib.cache_reusable(cache, key) is False)
    cache.write_bytes(b"cache")
    cache.with_suffix(".cache.key").write_text(key + "\n", encoding="utf-8")
    check("缓存和标记一致才能复用", int8calib.cache_reusable(cache, key) is True)
    check("标记不同就不能复用", int8calib.cache_reusable(cache, "other") is False)

    command = int8calib.prepare_command(
        trtexec=r"C:\cuda\trtexec.exe",
        onnx_path=onnx,
        engine_path=engine,
        images=found,
        count=2,
        workspace_mb=4096,
        input_name="images",
        shapes=("1x3x320x320", "1x3x640x640", "1x3x960x960"),
    )
    joined = " ".join(command)
    check("能复用时走 trtexec 和 --int8", "trtexec.exe" in joined and "--int8" in joined and f"--calib={cache}" in joined, joined)
    check("动态尺寸带上校准 profile", "--optShapesCalib=images:1x3x640x640" in joined, joined)

    cache.unlink()
    old = int8calib.find_calib_exe
    int8calib.find_calib_exe = lambda: None
    try:
        try:
            int8calib.prepare_command(
                trtexec="trtexec",
                onnx_path=onnx,
                engine_path=engine,
                images=found,
                count=2,
                workspace_mb=4096,
                input_name="images",
                shapes=None,
            )
            check("没有校准程序时报错", False, "没有抛 Int8Error")
        except int8calib.Int8Error as exc:
            check("没有校准程序时报错", "int8_calib.exe" in str(exc), str(exc))
    finally:
        int8calib.find_calib_exe = old

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
