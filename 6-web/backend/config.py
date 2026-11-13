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
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 仓库根目录推导
# ---------------------------------------------------------------------------
# config.py -> backend -> 6-web -> <repo>
_WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("TRTFORGE_REPO_ROOT", "").strip() or _WEB_ROOT.parent).resolve()

WEB_ROOT = _WEB_ROOT
# 安装包把网页文件放在安装目录的 frontend\，不跟编译后的程序挤在一起。
_frontend_env = os.environ.get("TRTFORGE_FRONTEND_DIR", "").strip()
FRONTEND_DIR = Path(_frontend_env).resolve() if _frontend_env else (WEB_ROOT / "frontend")
RUNTIME_DIR = WEB_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
LOG_DIR = RUNTIME_DIR / "logs"

# 任务运行台账：追加式 JSONL，一个任务两行（开始一行、结束一行）。
# 为什么需要它：RunManager 的内存字典一重启就空，而日志文件还躺在 LOG_DIR 里，
# 前端拿不到 run_id 就等于把那份日志弄丢了。台账是「日志文件的索引 + 结束原因」。
# 可用 TRTFORGE_RUNS_LEDGER 指到别处：测试与 CI 不该往开发机的历史列表里塞假任务。
RUNS_LEDGER = Path(
    os.environ.get("TRTFORGE_RUNS_LEDGER", "").strip() or (RUNTIME_DIR / "runs.jsonl")
)

# 单次任务日志上限与保留份数。默认 20MB × 4 份 = 80MB / 任务。
# 训练日志一次十几个 MB 很常见，不设上限长期跑会把磁盘写满。
LOG_MAX_BYTES = max(1, int(os.environ.get("TRTFORGE_LOG_MAX_MB", "20"))) * 1024 * 1024
LOG_BACKUPS = max(0, int(os.environ.get("TRTFORGE_LOG_BACKUPS", "3")))

#: 应用版本号。放在 config 而不是 app：数据集导出（dataset.py）也要把它写进 manifest，
#: 而 dataset 反过来 import app 会形成循环。app.py 只做 `APP_VERSION = config.APP_VERSION`。
APP_VERSION = "1.37.0"

# 任务日志的保留天数（按整个任务的所有分片一起判断，最旧一份多久没更新就删）。
# 之所以还要这一层：单任务有上限，但「任务数 × 上限」仍然会无限增长。
# 设 0 表示关闭自动清理（想永久留档时用）。
LOG_RETENTION_DAYS = max(0, int(os.environ.get("TRTFORGE_LOG_RETENTION_DAYS", "30")))

# 显存实测峰值表：按 (imgsz, batch, 精度) 记录历史最大峰值，供开训前预检使用。
#
# 为什么落在 7-autoresearch-yolo/experiments/ 而不是 6-web/runtime/：
# 这张表由 autoexp（无人值守跑实验）和 Web 训练页共同积累，谁写进去都该被对方看见。
# autoexp 的 experiments/ 本来就是「实验事实」的落盘位置（best.json、exp_*.json 都在那），
# 放在这里语义一致；6-web/runtime 是纯运行期缓存，打包/清理时会被当成垃圾。
# 该目录不存在（只部署 Web 的场景）时退回 RUNTIME_DIR，功能不受影响。
_VRAM_SAMPLES_ENV = os.environ.get("TRTFORGE_VRAM_SAMPLES", "").strip()
VRAM_SAMPLES = Path(_VRAM_SAMPLES_ENV) if _VRAM_SAMPLES_ENV else (
    (REPO_ROOT / "7-autoresearch-yolo" / "experiments" / "vram_samples.json")
    if (REPO_ROOT / "7-autoresearch-yolo").is_dir()
    else (RUNTIME_DIR / "vram_samples.json")
)

# 训练类任务异常退出后是否自动续训一次（见 runners.RunManager._maybe_resume）。
AUTO_RESUME = os.environ.get("TRTFORGE_AUTO_RESUME", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

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
# 安装包把 OpenCV / CUDA / TensorRT 和推理 DLL 放在同一个 bin。
# 开发机上这些库也可能已经复制到 4-qt-studio/bin，一并加入搜索路径。
if BIN_DIR.is_dir() and BIN_DIR not in DLL_DIRS:
    DLL_DIRS.insert(0, BIN_DIR)

# ---------------------------------------------------------------------------
# Python 解释器（训练用）
# ---------------------------------------------------------------------------
# 训练解释器与「运行本服务的解释器」是两件事：服务可能跑在任意 Python 上，
# 而训练需要装了 torch/ultralytics/anomalib 的那个环境。
PYTHON_EXE = os.environ.get("TRTFORGE_PYTHON", "").strip()

#: 解释器解析结果缓存：(path|None, 缓存时间戳)。见 resolve_python()
_PYTHON_CACHE: tuple[str | None, float] | None = None
PYTHON_CACHE_SECONDS = 30.0

TRAIN_EVENT_PREFIX = "TRTFORGE_JSON "

# 允许的图片扩展名（与 launcher 保持一致）
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# 上传大小上限（字节）
MAX_UPLOAD_BYTES = int(os.environ.get("TRTFORGE_MAX_UPLOAD_MB", "32")) * 1024 * 1024

# 保留的最近日志行数（WebSocket 断线重连时补发）
EVENT_BUFFER_SIZE = 2000

# ---------------------------------------------------------------------------
# 常驻预热进程（训练启动提速）
# ---------------------------------------------------------------------------
# 训练 launcher 每次冷启动都要重新 import torch / ultralytics / anomalib，
# 实测 YOLO 约 1.3 秒、Anomalib 约 5.7 秒。预热进程在服务空闲时先把导入做掉，
# 点击训练时直接复用。
#   TRTFORGE_WARM=0        关闭（内存受限或排障时用）
#   TRTFORGE_WARM_KINDS    只预热哪几类，默认两类都预热
#   TRTFORGE_WARM_IDLE_SECONDS  空闲多久回收，默认 30 分钟
WARM_ENABLED = os.environ.get("TRTFORGE_WARM", "1").strip().lower() not in {"0", "false", "no", "off"}
WARM_KINDS = tuple(
    item.strip()
    for item in os.environ.get("TRTFORGE_WARM_KINDS", "yolo,anomalib").split(",")
    if item.strip()
)
WARM_IDLE_SECONDS = int(os.environ.get("TRTFORGE_WARM_IDLE_SECONDS", "1800"))


def tool_command(python: str, script_name: str) -> list[str]:
    """拼一条「运行仓库里的 tools 脚本」的命令。

    开发时是 ``python tools/xxx.py``。安装包里没有这些 .py，
    启动脚本会设置 ``TRTFORGE_TOOL_EXE``，改由编译后的程序执行同一功能。
    环境变量没设时，命令和以前完全一样。
    """
    tool_exe = os.environ.get("TRTFORGE_TOOL_EXE", "").strip()
    if tool_exe:
        return [str(Path(tool_exe)), "--tool", Path(script_name).stem]
    return [python, str(Path(__file__).resolve().parent / "tools" / script_name)]


def ensure_dirs() -> None:
    """创建运行期需要的目录。"""
    for d in (RUNTIME_DIR, UPLOAD_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def allowed_roots() -> list[Path]:
    """允许被接口访问的根目录列表。

    Web 端没有原生文件对话框，路径全靠后端列出，因此必须有一份白名单，
    否则接口就成了"读任意磁盘"的后门。默认只放仓库根，需要别的目录时
    用 TRTFORGE_BROWSE_ROOTS 显式追加（分号/冒号分隔，看平台）。

    收敛到这里的原因：目录浏览、数据集扫描、模型扫描三处都需要同一份判断，
    各写一遍迟早出现"某处忘了校验"的漏洞。
    """
    roots = [REPO_ROOT]
    for extra in os.environ.get("TRTFORGE_BROWSE_ROOTS", "").split(os.pathsep):
        if extra.strip():
            try:
                roots.append(Path(extra.strip()).expanduser().resolve())
            except OSError:
                continue
    return roots


# ---------------------------------------------------------------------------
# 运行期临时放行（原生对话框选中的路径）
# ---------------------------------------------------------------------------
# 原生对话框能选到白名单外的目录（比如 D:\数据集）。如果选完就 403，
# 用户会觉得"选了个路径却打不开"，比现在的目录列表还难用。
# 因此接口在确认"是本机用户亲手选的"之后，把该目录加进这份进程级放行表。
#
# 三条约束，缺一不可：
#   1. 只有 127.0.0.1 / ::1 的请求才能触发（远程访问一律回退到页面内浏览）；
#   2. 只加"用户点中的那个目录"，不递归整盘；
#   3. 不落盘，服务重启即失效——临时授权不该有持久副作用。
ALLOWED_RUNTIME: set[Path] = set()
_ALLOW_LOCK = threading.Lock()


def allow_path(path: str | Path) -> Path | None:
    """把一条路径加入本次运行的放行表（不写配置、不落盘）。"""
    try:
        target = Path(path).expanduser().resolve()
    except OSError:
        return None
    if not target.is_dir():
        target = target.parent
    with _ALLOW_LOCK:
        ALLOWED_RUNTIME.add(target)
    return target


def clear_runtime_allowed() -> None:
    """清空运行期放行表（测试用）。"""
    with _ALLOW_LOCK:
        ALLOWED_RUNTIME.clear()


def is_path_allowed(path: str | Path) -> bool:
    """判断路径是否落在白名单内（含根目录自身，含运行期临时放行）。"""
    try:
        target = Path(path).expanduser().resolve()
    except OSError:
        return False
    with _ALLOW_LOCK:
        extra = tuple(ALLOWED_RUNTIME)
    if any(target == root or root in target.parents for root in extra):
        return True
    return any(target == root or root in target.parents for root in allowed_roots())



def resolve_python() -> str | None:
    """解析训练用 Python 解释器。

    优先级：
    1. 环境变量 TRTFORGE_PYTHON
    2. 当前服务自身解释器（若它已装 torch，直接用）
    3. 常见 conda 环境
    找不到返回 None。

    结果缓存 30 秒：每次点训练都起一个子进程 `python -c "import torch"` 去探测
    要白等 1~2 秒，而这个结果在用户不换环境时不会变。缓存时间短到改配置后
    很快生效，又长到连点训练不会重复付这笔钱。
    """
    global _PYTHON_CACHE
    now = time.time()
    cached = _PYTHON_CACHE
    if cached is not None and now - cached[1] < PYTHON_CACHE_SECONDS:
        return cached[0]

    import sys

    resolved: str | None = None
    if PYTHON_EXE and Path(PYTHON_EXE).is_file():
        resolved = PYTHON_EXE
    # 服务自身解释器能用就用，避免用户额外配置
    elif _has_training_deps(sys.executable):
        resolved = sys.executable
    else:
        # 常见位置兜底
        for candidate in (
            Path(r"D:\Anaconda3\envs\dpenv\python.exe"),
            Path(r"D:\Anaconda3\envs\pytorch\python.exe"),
        ):
            if candidate.is_file() and _has_training_deps(str(candidate)):
                resolved = str(candidate)
                break

    _PYTHON_CACHE = (resolved, now)
    return resolved


def invalidate_python_cache() -> None:
    """清掉解释器缓存（测试或用户改了环境后用）。"""
    global _PYTHON_CACHE
    _PYTHON_CACHE = None


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


def resolve_trtexec() -> str | None:
    """解析 trtexec.exe 的位置，找不到返回 None。

    优先级：
    1. 环境变量 `TRTFORGE_TRTEXEC`（直接指向 trtexec.exe）
    2. `CUDA_PATH` 下的 `bin\\trtexec.exe`（TensorRT 常与 CUDA 装在一起）
    3. 扫描 `CUDA\\v*\\bin\\trtexec.exe`，取版本号最大的
    4. 常见 TensorRT 独立安装目录
    5. PATH 里的 trtexec

    之所以要扫目录而不是只认环境变量：实测这台机器根本没设 CUDA_PATH，
    trtexec 却躺在 CUDA bin 里。只认环境变量会让引擎构建页直接不可用。
    """
    explicit = os.environ.get("TRTFORGE_TRTEXEC", "").strip()
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())

    candidates: list[Path] = []

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin" / "trtexec.exe")

    # CUDA 默认安装根，按版本号倒序取最新
    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.is_dir():
        found = sorted(cuda_root.glob("v*/bin/trtexec.exe"), reverse=True)
        candidates.extend(found)

    for extra in (
        r"C:\TensorRT-8.5.2.2\bin\trtexec.exe",
        r"D:\TensorRT-8.5.2.2\bin\trtexec.exe",
        r"C:\TensorRT\bin\trtexec.exe",
        r"D:\thirdparty\TensorRT-8.5.2.2\bin\trtexec.exe",
    ):
        candidates.append(Path(extra))

    from shutil import which

    for name in ("trtexec.exe", "trtexec"):
        located = which(name)
        if located:
            candidates.append(Path(located))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def capabilities() -> dict:
    """汇总当前环境是否具备各项能力，供前端决定显示什么。"""
    trtexec = resolve_trtexec()
    return {
        "repo_root": str(REPO_ROOT),
        "python": resolve_python(),
        "yolo_launcher": str(YOLO_LAUNCHER) if YOLO_LAUNCHER.is_file() else None,
        "anomalib_launcher": str(ANOMALIB_LAUNCHER) if ANOMALIB_LAUNCHER.is_file() else None,
        "anomalib_dataset_root": str(DEFAULT_ANOMALIB_ROOT) if DEFAULT_ANOMALIB_ROOT.is_dir() else None,
        "bin_dir": str(BIN_DIR),
        "yolo_dll": (BIN_DIR / "yolo_infer.dll").is_file(),
        "anomalib_dll": (BIN_DIR / "anomalib_infer.dll").is_file(),
        "onnx_dll": (BIN_DIR / "onnx_infer.dll").is_file(),
        "onnxruntime_dll": (BIN_DIR / "onnxruntime.dll").is_file(),
        "dll_dirs": [str(d) for d in DLL_DIRS],
        "trtexec": trtexec,
        "engine_build": trtexec is not None,
    }
