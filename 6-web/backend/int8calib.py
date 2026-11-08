# -*- coding: utf-8 -*-
"""YOLO INT8 校准：抽图、判断缓存能不能复用、拼编译命令。

不在这里编译引擎。真正读图、做 letterbox、写校准缓存的是 ``int8_calib.exe``。
本模块只决定这次该走哪条命令：

- 缓存还对得上：``trtexec --int8 --fp16 --calib=缓存``，不再读图；
- 对不上：``int8_calib.exe`` 读图、写缓存，并一次编出引擎。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


class Int8Error(Exception):
    """校准准备失败。调用方把它变成 400。"""

import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_IMAGES = 500


def list_images(folder: Path) -> list[Path]:
    """文件夹里的图片，含子目录，按路径排序，换机器顺序也稳定。"""
    if not folder.is_dir():
        return []
    found = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    found.sort(key=lambda path: str(path).lower())
    return found


def sample_evenly(paths: list[Path], count: int) -> list[Path]:
    """均匀抽 count 张。不足就全用。count 限制在 1 到 500。"""
    if not paths:
        return []
    count = max(1, min(int(count), MAX_IMAGES, len(paths)))
    if len(paths) <= count:
        return list(paths)
    if count == 1:
        return [paths[len(paths) // 2]]
    step = (len(paths) - 1) / (count - 1)
    return [paths[round(index * step)] for index in range(count)]


def find_calib_exe() -> Path | None:
    """校准程序。优先环境变量，其次和推理 DLL 放在一起。"""
    override = os.environ.get("TRTFORGE_INT8_CALIB", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.append(config.BIN_DIR / "int8_calib.exe")
    for path in candidates:
        if path.is_file():
            return path
    return None


def cache_key(
    onnx_path: Path,
    images: list[Path],
    *,
    opt_shape: str,
    workspace_mb: int,
) -> str:
    """模型和图片没变时，这个字符串也不变。变了就要重新校准。"""
    stat = onnx_path.stat()
    lines = [
        str(onnx_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        opt_shape.strip(),
        str(int(workspace_mb)),
    ]
    for image in images:
        image_stat = image.stat()
        lines.append(f"{image.resolve()}|{image_stat.st_size}|{image_stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def cache_reusable(cache_path: Path, key: str) -> bool:
    key_path = cache_path.with_suffix(cache_path.suffix + ".key")
    if not cache_path.is_file() or not key_path.is_file():
        return False
    try:
        saved = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return saved == key


def write_image_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(image.resolve()) for image in images)
    path.write_text(text + "\n", encoding="utf-8")


def write_pending_key(cache_path: Path, key: str) -> Path:
    """先写到旁边。校准程序成功后才把它换成正式的 .key。"""
    pending = cache_path.with_suffix(cache_path.suffix + ".key.pending")
    pending.write_text(key + "\n", encoding="utf-8")
    return pending


def trtexec_reuse_command(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    cache_path: Path,
    *,
    workspace_mb: int,
    input_name: str,
    shapes: tuple[str, str, str] | None,
) -> list[str]:
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--int8",
        f"--calib={cache_path}",
        f"--workspace={workspace_mb}",
    ]
    if shapes:
        opt = shapes[1]
        command += [
            f"--minShapes={input_name}:{shapes[0]}",
            f"--optShapes={input_name}:{opt}",
            f"--maxShapes={input_name}:{shapes[2]}",
            f"--minShapesCalib={input_name}:{opt}",
            f"--optShapesCalib={input_name}:{opt}",
            f"--maxShapesCalib={input_name}:{opt}",
        ]
    return command


def prepare_command(
    *,
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    images: list[Path],
    count: int,
    workspace_mb: int,
    input_name: str,
    shapes: tuple[str, str, str] | None,
) -> list[str]:
    """返回这一次真正要执行的命令。图片或模型变了就重新校准。"""
    if not onnx_path.is_file():
        raise Int8Error(f"ONNX 不存在：{onnx_path}")
    chosen = sample_evenly(images, count)
    if not chosen:
        raise Int8Error("没有可用于校准的图片")
    cache_path = engine_path.with_suffix(".cache")
    opt_shape = shapes[1] if shapes else ""
    key = cache_key(onnx_path, chosen, opt_shape=opt_shape, workspace_mb=workspace_mb)
    if cache_reusable(cache_path, key):
        return trtexec_reuse_command(
            trtexec, onnx_path, engine_path, cache_path,
            workspace_mb=workspace_mb, input_name=input_name, shapes=shapes,
        )
    exe = find_calib_exe()
    if exe is None:
        raise Int8Error(
            "找不到 int8_calib.exe。请先运行 scripts\\build_infer_dlls.ps1，"
            "编译 yolo_infer 时会一起生成它。"
        )
    list_path = cache_path.with_suffix(".images.txt")
    write_image_list(list_path, chosen)
    pending = write_pending_key(cache_path, key)
    return calib_exe_command(
        exe, onnx_path, engine_path, cache_path, list_path, pending,
        workspace_mb=workspace_mb, shapes=shapes,
    )


def calib_exe_command(
    exe: Path,
    onnx_path: Path,
    engine_path: Path,
    cache_path: Path,
    list_path: Path,
    pending_key: Path,
    *,
    workspace_mb: int,
    shapes: tuple[str, str, str] | None,
) -> list[str]:
    command = [
        str(exe),
        "--onnx", str(onnx_path),
        "--save", str(engine_path),
        "--cache", str(cache_path),
        "--list", str(list_path),
        "--key-pending", str(pending_key),
        "--workspace", str(int(workspace_mb)),
    ]
    if shapes:
        command += ["--min", shapes[0], "--opt", shapes[1], "--max", shapes[2]]
    return command
