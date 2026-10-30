# -*- coding: utf-8 -*-
"""TRTForge Web Studio 配置与路径解析。

设计原则（与五级目录重构保持一致）：
- 所有路径都从本文件位置向上推导，不写死某台机器的绝对路径。
- 需要用别的目录时，用环境变量覆盖，不改代码。

目录约定：
    <repo>/6-web/backend/config.py   <- 本文件
    <repo>                            = parents[2]
    <repo>/1-python-yolo/launcher
    <repo>/2-python-anomaly/launcher
    <repo>/4-qt-studio/bin            <- 推理 DLL 产物目录
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 仓库根目录推导
# ---------------------------------------------------------------------------
# config.py -> backend -> 6-web -> <repo>
_WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("TRTFORGE_REPO_ROOT", "").strip() or _WEB_ROOT.parent).resolve()

WEB_ROOT = _WEB_ROOT
FRONTEND_DIR = WEB_ROOT / "frontend"
RUNTIME_DIR = WEB_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
LOG_DIR = RUNTIME_DIR / "logs"

# ---------------------------------------------------------------------------
# 训练工程与 launcher
# ---------------------------------------------------------------------------
YOLO_LAUNCHER_DIR = Path(
    os.environ.get("TRTFORGE_YOLO_LAUNCHER_DIR", "").strip()
    or REPO_ROOT / "1-python-yolo" / "launcher"
).resolve()
YOLO_LAUNCHER = YOLO_LAUNCHER_DIR / "tf_yolo_launcher.py"

ANOMALIB_LAUNCHER_DIR = Path(
    os.environ.get("TRTFORGE_ANOMALIB_LAUNCHER_DIR", "").strip()
    or REPO_ROOT / "2-python-anomaly" / "launcher"
).resolve()
ANOMALIB_LAUNCHER = ANOMALIB_LAUNCHER_DIR / "tf_anomalib_launcher.py"

# Anomalib 默认数据集根：仓库自带 bottle
DEFAULT_ANOMALIB_ROOT = Path(
    os.environ.get("TRTFORGE_ANOMALIB_DATASET_ROOT", "").strip()
    or REPO_ROOT / "2-python-anomaly" / "anomalib-main" / "datasets" / "bottle"
).resolve()

# ---------------------------------------------------------------------------
# 推理库与原生依赖
# ---------------------------------------------------------------------------
BIN_DIR = Path(
    os.environ.get("TRTFORGE_BIN_DIR", "").strip()
    or REPO_ROOT / "4-qt-studio" / "bin"
).resolve()

# DLL 依赖搜索目录。Python 3.8+ 不再使用 PATH 解析依赖 DLL，必须显式添加。
_DLL_DIR_CANDIDATES = [
    os.environ.get("OPENCV_BIN_DIR", "").strip(),
    str(Path(os.environ.get("OPENCV_DIR", "").strip() or "") / "x64" / "vc17" / "bin")
    if os.environ.get("OPENCV_DIR", "").strip()
    else "",
    os.environ.get("CUDA_BIN_DIR", "").strip(),
    str(Path(os.environ.get("CUDA_PATH", "").strip() or "") / "bin")
    if os.environ.get("CUDA_PATH", "").strip()
    else "",
]

DLL_DIRS: list[Path] = [d for d in (Path(p) for p in _DLL_DIR_CANDIDATES if p) if d.is_dir()]

# ---------------------------------------------------------------------------
# Python 解释器（训练用）
# ---------------------------------------------------------------------------
# 训练解释器与「运行本服务的解释器」是两件事：服务可能跑在任意 Python 上，
# 而训练需要装了 torch/ultralytics/anomalib 的那个环境。
PYTHON_EXE = os.environ.get("TRTFORGE_PYTHON", "").strip()

TRAIN_EVENT_PREFIX = "TRTFORGE_JSON "

# 允许的图片扩展名（与 launcher 保持一致）
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# 上传大小上限（字节）
MAX_UPLOAD_BYTES = int(os.environ.get("TRTFORGE_MAX_UPLOAD_MB", "32")) * 1024 * 1024

# 保留的最近日志行数（WebSocket 断线重连时补发）
EVENT_BUFFER_SIZE = 2000


def ensure_dirs() -> None:
    """创建运行期需要的目录。"""
    for d in (RUNTIME_DIR, UPLOAD_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_python() -> str | None:
    """解析训练用 Python 解释器。

    优先级：
    1. 环境变量 TRTFORGE_PYTHON
    2. 当前服务自身解释器（若它已装 torch，直接用）
    3. 常见 conda 环境
    找不到返回 None。
    """
    import sys

    if PYTHON_EXE and Path(PYTHON_EXE).is_file():
        return PYTHON_EXE

    # 服务自身解释器能用就用，避免用户额外配置
    if _has_training_deps(sys.executable):
        return sys.executable

    # 常见位置兜底
    for candidate in (
        Path(r"D:\Anaconda3\envs\dpenv\python.exe"),
        Path(r"D:\Anaconda3\envs\pytorch\python.exe"),
    ):
        if candidate.is_file() and _has_training_deps(str(candidate)):
            return str(candidate)

    return None


def _has_training_deps(python_exe: str) -> bool:
    """检查解释器是否装了 torch（训练的最低要求）。"""
    import subprocess

    try:
        result = subprocess.run(
            [python_exe, "-c", "import torch"],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        return False


def capabilities() -> dict:
    """汇总当前环境是否具备各项能力，供前端决定显示什么。"""
    return {
        "repo_root": str(REPO_ROOT),
        "python": resolve_python(),
        "yolo_launcher": str(YOLO_LAUNCHER) if YOLO_LAUNCHER.is_file() else None,
        "anomalib_launcher": str(ANOMALIB_LAUNCHER) if ANOMALIB_LAUNCHER.is_file() else None,
        "anomalib_dataset_root": str(DEFAULT_ANOMALIB_ROOT) if DEFAULT_ANOMALIB_ROOT.is_dir() else None,
        "bin_dir": str(BIN_DIR),
        "yolo_dll": (BIN_DIR / "yolo_infer.dll").is_file(),
        "anomalib_dll": (BIN_DIR / "anomalib_infer.dll").is_file(),
        "dll_dirs": [str(d) for d in DLL_DIRS],
    }
