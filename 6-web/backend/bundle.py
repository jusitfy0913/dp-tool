# -*- coding: utf-8 -*-
"""模型包：把「一个可交付模型」写成一个自描述目录。

## 为什么需要这个文件

现状是一个模型由 4~6 个散落文件构成：``.engine``、``.onnx``、``labels_*.json``、
``metadata.json``、``<engine>.meta.json``。它们之间没有显式登记关系，
只能靠 ``models.py`` 里的启发式规则去猜配对（该文件自己写着「实测的真实现象，
不是猜的」）。猜错的后果不是报错，而是**解码越界或空结果**，排查成本极高。

``manifest.json`` 把配对关系从「猜」变成「读」。同时它也是唯一能回答
「这个模型在眼前这台机器上能不能跑」的地方——``.engine`` 里没有 GPU 型号，
也没有 TensorRT 版本。

## 三条使用路径

1. **构建**（:func:`build_bundle`）：从散装产物造包，模型文件用硬链接避免占空间；
2. **校验**（:func:`verify_bundle`）：加载前检查，结论分 ``ok`` / ``warn`` / ``block``；
3. **解析**（:func:`resolve_bundle`）：推理时从包里取出模型、标签、metadata、种类、任务。

## 为什么结论要分三档而不是布尔

``block`` 只用于**确定性**错误（哈希不符、schema 不识别、种类冲突）。
运行时字段缺失（本提案之前产生的旧 meta）只能给 ``warn``：
用户手里已有的可用模型，不该因为「当年没记 GPU 型号」就被升级挡住。

## 与 DLL 的关系

**没有关系**。模型包是 DLL 之外的旁挂约定，``yolo_infer.dll`` /
``anomalib_infer.dll`` 完全不知道它存在。这样加这个功能不需要动 C++、
不需要重编、也不影响已经交付的 SDK。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 清单版本。改结构必须换这个字符串，否则读取方会按旧规则解析新字段。
SCHEMA = "trtforge.bundle.v1"

#: 目录后缀。用目录而不是 zip：现场要能用资源管理器打开看、用记事本读。
SUFFIX = ".tfbundle"

#: 包内模型文件的固定名。固定名让清单不必处理中文与原文件名的各种写法；
#: 原文件名记录在 ``origin.meta.json`` 里，不会丢信息。
MODEL_FILE = "model.engine"
MODEL_FILE_ONNX = "model.onnx"
LABELS_FILE = "labels.json"
METADATA_FILE = "metadata.json"
ORIGIN_FILE = "origin.meta.json"
README_FILE = "README.txt"

#: files[] 的 role 取值。
ROLE_MODEL = "model"
ROLE_LABELS = "labels"
ROLE_METADATA = "metadata"

#: 校验结论。
LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_BLOCK = "block"

#: 种类。``unknown`` 只在无法判断时使用：它不做冲突判定，只提示。
KIND_YOLO = "yolo"
KIND_ANOMALIB = "anomalib"
KIND_UNKNOWN = "unknown"

#: YOLO 侧四种任务 + 分类，供冲突判定使用。
YOLO_TASKS = ("detect", "segment", "pose", "obb", "classify")

_ENGINE_SUFFIXES = {".engine", ".trt", ".plan"}
_ONNX_SUFFIXES = {".onnx"}

_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def is_bundle(path: Path | str) -> bool:
    """判断一个路径是不是模型包目录（只看后缀与 manifest 是否存在）。"""
    candidate = Path(path)
    return candidate.is_dir() and candidate.name.lower().endswith(SUFFIX) and manifest_path(candidate).is_file()


def manifest_path(bundle_dir: Path | str) -> Path:
    return Path(bundle_dir) / "manifest.json"


def read_manifest(bundle_dir: Path | str) -> dict[str, Any] | None:
    """读清单。读不出来返回 ``None``——调用方据此报「不是模型包」而不是抛异常。"""
    try:
        raw = Path(bundle_dir, "manifest.json").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path | str) -> str:
    """文件内容哈希。读不出来返回空串（清单里如实留空，不编造）。"""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_component(text: str) -> str:
    """把用户输入的包名收敛成安全目录名。

    去掉路径分隔符、盘符与 ``..``：包名来自页面输入，不能让它把目录建到别处。
    中文与空格保留——现场就爱用「PCB检测_v1」这类名字。
    """
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(text or "").strip())
    cleaned = cleaned.replace("..", "_").strip(" .")
    return cleaned or "model"


def _kind_from_path(model_path: Path) -> str:
    """从路径猜种类，只在调用方没指定时用。

    依据是仓库既定的产物布局：YOLO 在 ``launcher/runs``，异常检测在
    ``launcher/results``。猜不出来返回 ``unknown``，不给错误结论。
    """
    lowered = str(model_path).lower().replace("\\", "/")
    if "/results/" in lowered or "anomalib" in lowered:
        return KIND_ANOMALIB
    if "/runs/" in lowered or "yolo" in lowered:
        return KIND_YOLO
    return KIND_UNKNOWN


def _layout_of(path: Path) -> str:
    """ONNX 是否已转置。判断规则与 ``models.py`` 一致，避免两处口径不同。"""
    name = path.name.lower()
    if name.endswith(".transd.onnx") or ".transd." in name:
        return "transposed"
    return "raw"


# ---------------------------------------------------------------------------
# 运行时环境采集
# ---------------------------------------------------------------------------

class _VS_FIXEDFILEINFO(ctypes.Structure):
    """Windows PE 版本资源里的固定字段。字段顺序来自 winver.h，不能调换。"""

    _fields_ = [
        ("dwSignature", ctypes.c_uint32),
        ("dwStrucVersion", ctypes.c_uint32),
        ("dwFileVersionMS", ctypes.c_uint32),
        ("dwFileVersionLS", ctypes.c_uint32),
        ("dwProductVersionMS", ctypes.c_uint32),
        ("dwProductVersionLS", ctypes.c_uint32),
        ("dwFileFlagsMask", ctypes.c_uint32),
        ("dwFileFlags", ctypes.c_uint32),
        ("dwFileOS", ctypes.c_uint32),
        ("dwFileType", ctypes.c_uint32),
        ("dwFileSubtype", ctypes.c_uint32),
        ("dwFileDateMS", ctypes.c_uint32),
        ("dwFileDateLS", ctypes.c_uint32),
    ]


def file_version(path: Path | str) -> str | None:
    """读 Windows PE 文件版本，形如 ``8.5.2.2``。

    为什么不用 ``trtexec --version``：那要起一个子进程、约 0.5~2 秒，
    而造包是「点一下就期望立刻完成」的操作。版本资源在文件里，读它是毫秒级。
    非 Windows 或文件没有版本资源时返回 ``None``——调用方记录为空，不猜。
    """
    if os.name != "nt":
        return None
    target = str(path)
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(target, None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        if not ctypes.windll.version.GetFileVersionInfoW(target, 0, size, buffer):
            return None
        pointer = ctypes.c_void_p()
        length = ctypes.c_uint32()
        if not ctypes.windll.version.VerQueryValueW(
            buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)
        ):
            return None
        info = ctypes.cast(pointer, ctypes.POINTER(_VS_FIXEDFILEINFO)).contents
        return "{}.{}.{}.{}".format(
            info.dwFileVersionMS >> 16,
            info.dwFileVersionMS & 0xFFFF,
            info.dwFileVersionLS >> 16,
            info.dwFileVersionLS & 0xFFFF,
        )
    except Exception:  # noqa: BLE001 - 读版本失败不影响造包
        return None


def _bin_dir() -> Path | None:
    """推理运行库目录。取不到 config 时按目录约定推一遍（单测/脚本场景）。"""
    try:
        import config  # noqa: PLC0415

        return Path(config.BIN_DIR)
    except Exception:  # noqa: BLE001
        backend = Path(__file__).resolve().parent
        candidate = backend.parent.parent / "4-qt-studio" / "bin"
        return candidate if candidate.is_dir() else None


def trt_version() -> str | None:
    """TensorRT 版本，从 ``nvinfer.dll`` 的文件版本读。"""
    root = _bin_dir()
    if root is None:
        return None
    dll = root / "nvinfer.dll"
    if not dll.is_file():
        return None
    return file_version(dll)


def cuda_version() -> str | None:
    """CUDA 运行库版本。

    ``cudart64_110.dll`` 的文件名只给出 11.x，具体到 11.8 要看版本资源，
    所以优先读 ``cudart64_*.dll`` 的版本；读不到再退回文件名里的主版本。
    """
    root = _bin_dir()
    if root is None:
        return None
    try:
        candidates = sorted(root.glob("cudart64_*.dll"))
    except OSError:
        return None
    for dll in candidates:
        version = file_version(dll)
        if version:
            parts = version.split(".")
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}"
    for dll in candidates:
        found = re.search(r"cudart64_(\d+)\.dll", dll.name, re.IGNORECASE)
        if found:
            digits = found.group(1)
            if len(digits) >= 3:
                return f"{digits[:-1]}.{digits[-1]}"
    return None


def _smi_field(names: str) -> list[str]:
    """按给定字段名查一次 nvidia-smi，返回第一张卡的值。

    各驱动版本支持的查询字段不同（``compute_cap`` 要 R510+），
    因此失败时返回空列表，由调用方逐字段降级，而不是整个探测失败。
    """
    try:
        import subprocess  # noqa: PLC0415

        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={names}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return []
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        return []
    return [part.strip() for part in line[0].split(",")]


def gpu_snapshot() -> dict[str, Any]:
    """当前 GPU 的名称与 compute capability。取不到的字段留 ``None``。"""
    fields = _smi_field("name,compute_cap")
    if fields:
        return {
            "name": fields[0] or None,
            "compute_cap": fields[1] if len(fields) > 1 and fields[1] not in {"", "N/A"} else None,
        }
    # 老驱动不支持 compute_cap：至少拿到型号，arch 留空（缺失只给 warn）
    name = _smi_field("name")
    return {"name": name[0] if name else None, "compute_cap": None}


def runtime_snapshot(
    model_path: Path,
    *,
    precision: str | None = None,
    imgsz: int | None = None,
    dynamic: bool = False,
) -> dict[str, Any]:
    """采集造包时的运行时环境。

    ``format=onnx`` 时不记 TensorRT 与 GPU：ONNX 与设备无关，
    记上去反而会让「换台机器能不能跑」的判定产生假冲突。
    """
    suffix = model_path.suffix.lower()
    if suffix in _ONNX_SUFFIXES:
        return {
            "format": "onnx",
            "precision": precision or "unknown",
            "dynamic": bool(dynamic),
            "imgsz": [imgsz, imgsz] if imgsz else None,
            "trt_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "compute_cap": None,
            "onnx_layout": _layout_of(model_path),
        }
    gpu = gpu_snapshot()
    return {
        "format": "engine",
        "precision": precision or "unknown",
        "dynamic": bool(dynamic),
        "imgsz": [imgsz, imgsz] if imgsz else None,
        "trt_version": trt_version(),
        "cuda_version": cuda_version(),
        "gpu_name": gpu.get("name"),
        "compute_cap": gpu.get("compute_cap"),
        "onnx_layout": "unknown",
    }


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------

def _link_or_copy(source: Path, target: Path) -> bool:
    """优先硬链接，失败退回复制。返回是否用了硬链接。

    为什么要硬链接：一个引擎动辄 100~500 MB，造包是「点一下」的操作，
    复制 500 MB 要几秒还会占双份空间。同分区硬链接是零成本的。
    跨分区时 ``os.link`` 会抛 ``OSError``，此时安静地退回复制。
    """
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return True
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(source, target)
        return False


def _read_labels(path: Path | None) -> dict[str, Any] | None:
    """读标签文件，抽成 ``{count, names}``。

    兼容两种布局：``{"names": {...}}``（Ultralytics）与 ``{"names": [...]}``
    与 ``{"classes": [...]}``。读取失败返回 ``None``，不阻止造包——
    标签缺失时框上只有 ``cls0``，仍可用来核对模型本身。
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("names", payload.get("classes"))
    names: list[str] = []
    if isinstance(raw, dict):
        # dict 的键可能是字符串化的序号，按数值排序以免顺序错乱
        try:
            ordered = sorted(raw.items(), key=lambda pair: int(pair[0]))
        except (TypeError, ValueError):
            ordered = list(raw.items())
        names = [str(value) for _, value in ordered]
    elif isinstance(raw, list):
        names = [str(item) for item in raw]
    if not names:
        return None
    return {"count": len(names), "names": names}


def _anomaly_block(metadata: Path | None) -> dict[str, Any] | None:
    """从 metadata.json 抽出异常检测阈值。缺字段就不写，不编造默认值。"""
    if metadata is None or not Path(metadata).is_file():
        return None
    try:
        payload = json.loads(Path(metadata).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    block: dict[str, Any] = {}
    for key in ("image_threshold", "pixel_threshold", "model_type", "input_size", "normalization"):
        if key in payload:
            block[key] = payload[key]
    return block or None


def build_bundle(
    model_path: Path | str,
    out_root: Path | str,
    bundle_id: str,
    *,
    labels: Path | str | None = None,
    metadata: Path | str | None = None,
    kind: str | None = None,
    task: str | None = None,
    precision: str | None = None,
    imgsz: int | None = None,
    dynamic: bool = False,
    origin_extra: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """从散装产物造一个模型包，返回 ``{"ok", "path", "manifest", "error"}``。

    不修改、不删除原始产物。目标目录已存在时默认拒绝（避免静默覆盖用户已有的包），
    需要覆盖必须由调用方显式传 ``overwrite=True``。
    """
    source = Path(model_path)
    if not source.is_file():
        return {"ok": False, "path": "", "manifest": {}, "error": f"模型文件不存在：{source}"}
    if source.suffix.lower() not in (_ENGINE_SUFFIXES | _ONNX_SUFFIXES):
        return {
            "ok": False,
            "path": "",
            "manifest": {},
            "error": f"不支持的模型后缀：{source.suffix}（只支持 .engine / .trt / .plan / .onnx）",
        }

    target = Path(out_root) / (_safe_component(bundle_id) + SUFFIX)
    if target.exists() and not overwrite:
        return {
            "ok": False,
            "path": str(target),
            "manifest": {},
            "error": f"目标已存在：{target}。确认覆盖请显式传 overwrite",
        }

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "path": str(target), "manifest": {}, "error": f"创建目录失败：{exc}"}

    resolved_kind = (kind or _kind_from_path(source)).strip().lower()
    if resolved_kind not in (KIND_YOLO, KIND_ANOMALIB, KIND_UNKNOWN):
        resolved_kind = KIND_UNKNOWN

    runtime = runtime_snapshot(source, precision=precision, imgsz=imgsz, dynamic=dynamic)

    is_onnx = source.suffix.lower() in _ONNX_SUFFIXES
    model_name = MODEL_FILE_ONNX if is_onnx else MODEL_FILE
    # 引擎的 onnx_layout 取决于配套 ONNX 是否转置；造包时能拿到标签但拿不到 ONNX，
    # 因此由调用方通过 origin_extra 传入（引擎构建路径知道）。
    if not is_onnx:
        runtime["onnx_layout"] = str((origin_extra or {}).get("onnx_layout") or "unknown")

    linked = _link_or_copy(source, target / model_name)

    files: list[dict[str, Any]] = [
        {
            "role": ROLE_MODEL,
            "path": model_name,
            "bytes": int((target / model_name).stat().st_size),
            "sha256": sha256_file(target / model_name),
        }
    ]

    if labels is not None and Path(labels).is_file():
        _link_or_copy(Path(labels), target / LABELS_FILE)
        files.append(
            {
                "role": ROLE_LABELS,
                "path": LABELS_FILE,
                "bytes": int((target / LABELS_FILE).stat().st_size),
                "sha256": sha256_file(target / LABELS_FILE),
            }
        )

    if metadata is not None and Path(metadata).is_file():
        _link_or_copy(Path(metadata), target / METADATA_FILE)
        files.append(
            {
                "role": ROLE_METADATA,
                "path": METADATA_FILE,
                "bytes": int((target / METADATA_FILE).stat().st_size),
                "sha256": sha256_file(target / METADATA_FILE),
            }
        )

    # 来源信息单独成文件：它是「可再生的说明」，坏掉时不应该让整个包不可用，
    # 所以不登记进 files[]（不参与完整性判定）。
    origin = {
        "source_model": str(source),
        "source_model_name": source.name,
        "source_labels": str(labels) if labels else "",
        "source_metadata": str(metadata) if metadata else "",
        "model_sha256": files[0]["sha256"],
        "built_at": _now(),
        "linked": linked,
        **(origin_extra or {}),
    }
    try:
        (target / ORIGIN_FILE).write_text(
            json.dumps(origin, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return {"ok": False, "path": str(target), "manifest": {}, "error": f"写来源文件失败：{exc}"}

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "id": _safe_component(bundle_id),
        "title": str((origin_extra or {}).get("title") or bundle_id),
        "created_at": _now(),
        "created_by": str((origin_extra or {}).get("created_by") or ""),
        "kind": resolved_kind,
        "kind_source": "explicit" if kind else "inferred",
        "task": (task or "").strip().lower() or None,
        "runtime": runtime,
        "files": files,
        "labels": _read_labels(Path(labels) if labels else None),
        "anomaly": _anomaly_block(Path(metadata) if metadata else None),
        "operating_point": (origin_extra or {}).get("operating_point"),
        "provenance": {
            "source_model": str(source),
            "train_run_id": str((origin_extra or {}).get("run_id") or ""),
            "dataset_fingerprint": (origin_extra or {}).get("dataset_fingerprint"),
            "engine_sha256": files[0]["sha256"] if not is_onnx else None,
            "onnx_sha256": files[0]["sha256"] if is_onnx else (origin_extra or {}).get("onnx_sha256"),
        },
        "linked": linked,
    }
    try:
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return {"ok": False, "path": str(target), "manifest": {}, "error": f"写清单失败：{exc}"}

    try:
        (target / README_FILE).write_text(_readme_text(manifest, origin), encoding="utf-8")
    except OSError:
        pass  # 说明文件写不进去不影响包本身可用

    return {"ok": True, "path": str(target), "manifest": manifest, "error": ""}


def _readme_text(manifest: dict[str, Any], origin: dict[str, Any]) -> str:
    """生成给现场人看的一页说明。用纯文本：现场可能只有记事本。"""
    runtime = manifest.get("runtime") or {}
    files = manifest.get("files") or []
    labels = manifest.get("labels") or {}
    lines = [
        f"模型包：{manifest.get('title') or manifest.get('id')}",
        "",
        "这个文件夹里装的是一个可以直接交给推理程序使用的模型。",
        "不要单独拷走某个文件——清单和模型文件必须在一起。",
        "",
        "【基本信息】",
        f"  种类：{manifest.get('kind')}",
        f"  任务：{manifest.get('task') or '未记录'}",
        f"  格式：{runtime.get('format')}    精度：{runtime.get('precision')}",
        f"  输入尺寸：{runtime.get('imgsz')}",
        f"  清单版本：{manifest.get('schema')}",
        "",
        "【运行环境】",
        f"  TensorRT：{runtime.get('trt_version') or '未记录'}",
        f"  CUDA：{runtime.get('cuda_version') or '未记录'}",
        f"  显卡：{runtime.get('gpu_name') or '未记录'}",
        "",
        "【类别】",
    ]
    names = labels.get("names") or []
    if names:
        lines.append(f"  共 {len(names)} 类：" + "、".join(names))
    else:
        lines.append("  未记录（框上只显示 cls<编号>）")
    lines += [
        "",
        "【文件】",
    ]
    for item in files:
        lines.append(f"  {item.get('path')}  （{item.get('bytes')} 字节）")
    lines += [
        "",
        "【来源】",
        f"  原始文件：{origin.get('source_model')}",
        f"  构建时间：{origin.get('built_at')}",
        "",
        "【换机器之前务必确认】",
        "  .engine 绑定显卡架构与 TensorRT 版本。换到别的机器上如果加载失败，",
        "  请在那台机器上用「引擎构建」页把同一个 ONNX 重新编译一次。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def _check(
    name: str,
    level: str,
    detail: str,
    fix: str = "",
) -> dict[str, Any]:
    return {"name": name, "level": level, "detail": detail, "fix": fix}


def _major(version: str | None) -> str | None:
    if not version:
        return None
    return str(version).split(".")[0]


def verify_bundle(
    bundle_dir: Path | str,
    *,
    kind: str | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """校验模型包，返回 ``{"level", "checks", "manifest", "path"}``。

    ``level`` 取所有检查项里最严重的一档（``block`` > ``warn`` > ``ok``）。
    ``kind`` / ``task`` 是本次请求要用的模式，用于冲突判定；不传就跳过该项。
    """
    target = Path(bundle_dir)
    checks: list[dict[str, Any]] = []
    manifest = read_manifest(target)

    if manifest is None:
        return {
            "level": LEVEL_BLOCK,
            "path": str(target),
            "manifest": None,
            "checks": [
                _check("清单可读", LEVEL_BLOCK, "读不到 manifest.json 或不是合法 JSON",
                       "确认这是完整的模型包目录，不要只拷走其中一部分")
            ],
        }

    # 1. schema
    schema = manifest.get("schema")
    if schema == SCHEMA:
        checks.append(_check("清单版本", LEVEL_OK, f"schema = {schema}"))
    else:
        checks.append(
            _check(
                "清单版本",
                LEVEL_BLOCK,
                f"不认识的清单版本：{schema!r}（本软件支持 {SCHEMA}）",
                "用当前版本的软件重新构建这个模型包",
            )
        )

    # 2. 必填字段
    missing = [key for key in ("kind", "task", "runtime", "files") if key not in manifest]
    if missing:
        checks.append(
            _check("必填字段", LEVEL_BLOCK, "清单缺少字段：" + "、".join(missing),
                   "重新构建模型包")
        )
    else:
        checks.append(_check("必填字段", LEVEL_OK, "kind / task / runtime / files 齐全"))

    # 3. 文件清单与完整性
    files = manifest.get("files") or []
    if not isinstance(files, list) or not files:
        checks.append(_check("文件清单", LEVEL_BLOCK, "files 为空", "重新构建模型包"))
        files = []

    model_entry = next(
        (item for item in files if isinstance(item, dict) and item.get("role") == ROLE_MODEL),
        None,
    )
    if model_entry is None:
        checks.append(
            _check("模型文件", LEVEL_BLOCK, "清单里没有 role=model 的条目", "重新构建模型包")
        )

    escaped = [
        str(item.get("path") or "")
        for item in files
        if isinstance(item, dict) and _is_unsafe_relative(str(item.get("path") or ""))
    ]
    if escaped:
        checks.append(
            _check(
                "包内路径",
                LEVEL_BLOCK,
                "清单里出现绝对路径或越界路径：" + "、".join(escaped),
                "这个包可能是伪造或被改坏的，不要使用",
            )
        )
    elif files:
        checks.append(_check("包内路径", LEVEL_OK, "全部为包内相对路径"))

    missing_files: list[str] = []
    bad_hash: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("path") or "")
        if _is_unsafe_relative(relative):
            continue
        file_path = target / relative
        if not file_path.is_file():
            missing_files.append(relative)
            continue
        expected = str(item.get("sha256") or "")
        if expected:
            actual = sha256_file(file_path)
            if actual and actual != expected:
                bad_hash.append(f"{relative}（登记 {expected[:12]}…，实际 {actual[:12]}…）")
    if missing_files:
        checks.append(
            _check("文件存在", LEVEL_BLOCK, "缺失文件：" + "、".join(missing_files),
                   "重新构建模型包，或把缺失的文件从原位置拷回来")
        )
    elif files:
        checks.append(_check("文件存在", LEVEL_OK, "清单登记的文件都在"))

    if bad_hash:
        checks.append(
            _check("文件完整性", LEVEL_BLOCK, "内容与清单不一致：" + "；".join(bad_hash),
                   "文件被改过或拷坏了，重新构建模型包")
        )
    elif files:
        checks.append(_check("文件完整性", LEVEL_OK, "全部哈希一致"))

    # 4. 种类与任务冲突
    manifest_kind = str(manifest.get("kind") or KIND_UNKNOWN).lower()
    if kind and manifest_kind != KIND_UNKNOWN:
        if kind.strip().lower() != manifest_kind:
            checks.append(
                _check(
                    "种类匹配",
                    LEVEL_BLOCK,
                    f"这个包是 {manifest_kind}，当前请求是 {kind}",
                    "在推理页把模式切到对应的类型，或换一个该类型下的模型包",
                )
            )
        else:
            checks.append(_check("种类匹配", LEVEL_OK, f"{manifest_kind} 与请求一致"))
    elif manifest_kind == KIND_UNKNOWN:
        checks.append(
            _check("种类匹配", LEVEL_WARN, "清单没有记录种类，无法核对",
                   "重新构建模型包时会自动判断")
        )

    manifest_task = str(manifest.get("task") or "").strip().lower()
    if task and manifest_task:
        requested = task.strip().lower()
        if manifest_kind == KIND_YOLO and requested in YOLO_TASKS and requested != manifest_task:
            checks.append(
                _check(
                    "任务匹配",
                    LEVEL_BLOCK,
                    f"这个包训练时是 {manifest_task}，当前请求是 {requested}",
                    f"把任务切换成 {manifest_task}，或改选一个 {requested} 的模型包",
                )
            )
        else:
            checks.append(_check("任务匹配", LEVEL_OK, f"{manifest_task} 与请求一致"))
    elif not manifest_task:
        checks.append(
            _check("任务匹配", LEVEL_WARN, "清单没有记录任务类型，无法核对",
                   "重新构建模型包时会补上")
        )

    # 5. 运行时兼容性（只对 engine 有意义）
    runtime = manifest.get("runtime") or {}
    fmt = str(runtime.get("format") or "")
    if fmt == "onnx":
        checks.append(_check("运行时", LEVEL_OK, "ONNX 与设备无关，不需要核对显卡与 TensorRT"))
    else:
        recorded_trt = runtime.get("trt_version")
        local_trt = trt_version()
        if not recorded_trt:
            checks.append(
                _check("TensorRT 版本", LEVEL_WARN, "清单没有记录 TensorRT 版本",
                       "重新构建模型包可补全该信息")
            )
        elif not local_trt:
            checks.append(
                _check("TensorRT 版本", LEVEL_WARN, f"清单记录 {recorded_trt}，本机读不到版本",
                       "确认 bin 目录里有 nvinfer.dll")
            )
        elif _major(recorded_trt) != _major(local_trt):
            checks.append(
                _check(
                    "TensorRT 版本",
                    LEVEL_BLOCK,
                    f"这个引擎由 TensorRT {recorded_trt} 编译，本机是 {local_trt}",
                    "在本机用「引擎构建」页把同一个 ONNX 重新编译一次",
                )
            )
        elif recorded_trt != local_trt:
            checks.append(
                _check(
                    "TensorRT 版本",
                    LEVEL_WARN,
                    f"清单记录 {recorded_trt}，本机 {local_trt}（主版本相同，可能仍可加载）",
                    "如果加载报错，在本机重新编译一次即可",
                )
            )
        else:
            checks.append(_check("TensorRT 版本", LEVEL_OK, f"{local_trt} 一致"))

        recorded_cap = runtime.get("compute_cap")
        recorded_gpu = runtime.get("gpu_name")
        local = gpu_snapshot()
        if not recorded_cap:
            checks.append(
                _check("显卡架构", LEVEL_WARN,
                       f"清单没有记录 compute capability（记录的显卡：{recorded_gpu or '未记录'}）",
                       "重新构建模型包可补全该信息")
            )
        elif not local.get("compute_cap"):
            checks.append(
                _check("显卡架构", LEVEL_WARN,
                       f"清单记录 {recorded_cap}，本机读不到 compute capability",
                       "确认 nvidia-smi 可用")
            )
        elif str(recorded_cap) != str(local.get("compute_cap")):
            checks.append(
                _check(
                    "显卡架构",
                    LEVEL_WARN,
                    f"这个引擎在 {recorded_gpu or '未知显卡'}（{recorded_cap}）上编译，"
                    f"本机是 {local.get('name') or '未知显卡'}（{local.get('compute_cap')}）",
                    "引擎通常不能跨架构加载。在本机用「引擎构建」页重新编译一次，约 1 分钟",
                )
            )
        else:
            checks.append(
                _check("显卡架构", LEVEL_OK, f"{local.get('name') or ''}（{recorded_cap}）一致")
            )

    # 6. 异常检测必须有 metadata
    if manifest_kind == KIND_ANOMALIB:
        has_metadata = any(
            isinstance(item, dict) and item.get("role") == ROLE_METADATA for item in files
        )
        if has_metadata:
            checks.append(_check("异常检测 metadata", LEVEL_OK, "包内有 metadata.json"))
        else:
            checks.append(
                _check("异常检测 metadata", LEVEL_BLOCK, "异常检测模型包缺少 metadata.json",
                       "重新构建模型包，并在构建时选中 metadata.json")
            )

    level = LEVEL_OK
    if any(item["level"] == LEVEL_BLOCK for item in checks):
        level = LEVEL_BLOCK
    elif any(item["level"] == LEVEL_WARN for item in checks):
        level = LEVEL_WARN

    return {"level": level, "path": str(target), "manifest": manifest, "checks": checks}


def quick_check(bundle_dir: Path | str) -> dict[str, Any]:
    """不读文件内容的轻量校验，供模型列表使用。

    为什么需要另一个函数：完整校验要算几百 MB 的哈希，而模型列表是
    「每次打开推理页都请求」的接口，把它拖到几秒是灾难。
    这里只做「清单能读、必填项在、文件都在、schema 认识」四项，
    真正的完整性校验留给用户点「校验」时执行。
    """
    target = Path(bundle_dir)
    manifest = read_manifest(target)
    if manifest is None:
        return {"level": LEVEL_BLOCK, "reason": "清单不可读"}
    if manifest.get("schema") != SCHEMA:
        return {"level": LEVEL_BLOCK, "reason": f"不认识的清单版本：{manifest.get('schema')!r}"}
    missing = [key for key in ("kind", "task", "runtime", "files") if key not in manifest]
    if missing:
        return {"level": LEVEL_BLOCK, "reason": "清单缺少字段：" + "、".join(missing)}
    absent = []
    for item in manifest.get("files") or []:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("path") or "")
        if _is_unsafe_relative(relative):
            return {"level": LEVEL_BLOCK, "reason": f"越界的包内路径：{relative}"}
        if not (target / relative).is_file():
            absent.append(relative)
    if absent:
        return {"level": LEVEL_BLOCK, "reason": "缺失文件：" + "、".join(absent)}
    runtime = manifest.get("runtime") or {}
    if runtime.get("format") == "engine" and not runtime.get("trt_version"):
        return {"level": LEVEL_WARN, "reason": "清单未记录 TensorRT 版本"}
    return {"level": LEVEL_OK, "reason": ""}


def _is_unsafe_relative(relative: str) -> bool:
    """包内路径是否越界。

    这条检查防的是「清单被改成指向 ``C:\\Windows\\...`` 或 ``../../``」这类情况。
    模型包会从 U 盘、网盘来，不能假设清单是善意的。
    """
    if not relative:
        return True
    text = relative.replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return True
    parts = [part for part in text.split("/") if part not in ("", ".")]
    return any(part == ".." for part in parts)


# ---------------------------------------------------------------------------
# 枚举与解析
# ---------------------------------------------------------------------------

def default_root() -> Path:
    """模型包的默认存放目录：``6-web/runtime/bundles``。

    为什么不放在训练产物旁边：``launcher/runs`` 是训练输出，里面的 ``.engine``
    会被训练侧与自动实验脚本递归扫描；模型包内也有一个叫 ``model.engine`` 的文件，
    混进去会让那些工具多看到一个「来源不明的引擎」。放在 runtime 下一个固定目录，
    既是交付物的语义（可以整个拷走），也不会污染训练目录。
    """
    try:
        import config  # noqa: PLC0415

        return Path(config.RUNTIME_DIR) / "bundles"
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parent.parent / "runtime" / "bundles"


def _scan_roots(roots: Iterable[tuple[str, Path]] | None) -> list[tuple[str, Path]]:
    if roots is not None:
        return [(kind, Path(root)) for kind, root in roots]
    collected: list[tuple[str, Path]] = []
    try:
        import models as models_mod  # noqa: PLC0415

        collected = [(kind, Path(resolver())) for kind, resolver in models_mod.SCAN_TARGETS]
    except Exception:  # noqa: BLE001
        collected = []
    collected.append(("bundle", default_root()))
    return collected


def _file_of(bundle_dir: Path, manifest: dict[str, Any], role: str) -> str:
    """清单里某个 role 对应文件的绝对路径；没有或越界时返回空串。"""
    for item in manifest.get("files") or []:
        if not isinstance(item, dict) or item.get("role") != role:
            continue
        relative = str(item.get("path") or "")
        if relative and not _is_unsafe_relative(relative):
            return str(bundle_dir / relative)
    return ""


def enumerate_bundles(roots: Iterable[tuple[str, Path]] | None = None) -> list[dict[str, Any]]:
    """在给定根目录下列出全部模型包。

    ``roots`` 是 ``[(kind, 目录), ...]``；不传时用 ``models.SCAN_TARGETS`` 加
    默认的 ``runtime/bundles``。

    允许放在根下 4 层以内：用户手工把包放在 ``runs/<实验>/xxx.tfbundle`` 也能被找到，
    但不会无限制深挖（扫描是每次打开推理页都要跑的）。进入一个 ``.tfbundle`` 后
    不再往下走，避免把包内部文件当成候选。
    """
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for kind_hint, root in _scan_roots(roots):
        if not root.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                children = list(current.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if child.name.lower().endswith(SUFFIX):
                    resolved = str(child.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    manifest = read_manifest(child)
                    if manifest is None:
                        continue
                    try:
                        mtime = float(child.stat().st_mtime)
                    except OSError:
                        mtime = 0.0
                    found.append(
                        {
                            "id": str(manifest.get("id") or child.name[: -len(SUFFIX)]),
                            "path": str(child),
                            "root": str(root),
                            "kind_hint": kind_hint,
                            "title": str(manifest.get("title") or ""),
                            "kind": str(manifest.get("kind") or KIND_UNKNOWN),
                            "task": manifest.get("task"),
                            "runtime": manifest.get("runtime") or {},
                            "labels": manifest.get("labels"),
                            "mtime": mtime,
                            "origin": "bundle",
                            # 包内三个关键文件的绝对路径：前端选中后直接填，
                            # 不必为了三个路径再发一次请求。
                            "model_path": _file_of(child, manifest, ROLE_MODEL),
                            "labels_path": _file_of(child, manifest, ROLE_LABELS),
                            "metadata_path": _file_of(child, manifest, ROLE_METADATA),
                        }
                    )
                    continue
                if child.name.startswith(".") or depth >= 4:
                    continue
                stack.append((child, depth + 1))

    found.sort(key=lambda item: item["mtime"], reverse=True)
    return found


def find_bundle(identifier: str) -> Path | None:
    """按「目录路径」或「包 id」找一个模型包，找不到返回 ``None``。

    两种写法都要支持：页面传的是 id（短），命令行/脚本习惯传路径。
    """
    raw = str(identifier or "").strip()
    if not raw:
        return None
    direct = Path(raw)
    if is_bundle(direct):
        return direct
    wanted = raw.lower()
    for entry in enumerate_bundles():
        if entry["id"].lower() == wanted or Path(entry["path"]).name.lower() == wanted:
            return Path(entry["path"])
    return None


def resolve_bundle(identifier: str) -> dict[str, Any]:
    """把模型包解析成推理需要的一组路径。

    返回 ``{"ok", "error", "path", "kind", "task", "model", "labels", "metadata", "manifest"}``。
    调用方不需要再去看文件布局——这正是模型包存在的意义。
    """
    bundle_dir = find_bundle(identifier)
    if bundle_dir is None:
        return {"ok": False, "error": f"找不到模型包：{identifier}", "path": ""}
    manifest = read_manifest(bundle_dir)
    if manifest is None:
        return {"ok": False, "error": f"模型包清单不可读：{bundle_dir}", "path": str(bundle_dir)}

    model_entry = next(
        (
            item
            for item in (manifest.get("files") or [])
            if isinstance(item, dict) and item.get("role") == ROLE_MODEL
        ),
        None,
    )
    if model_entry is None or _is_unsafe_relative(str(model_entry.get("path") or "")):
        return {
            "ok": False,
            "error": f"模型包里没有可用的模型文件：{bundle_dir}",
            "path": str(bundle_dir),
        }

    def _path_of(role: str) -> str:
        for item in manifest.get("files") or []:
            if isinstance(item, dict) and item.get("role") == role:
                relative = str(item.get("path") or "")
                if relative and not _is_unsafe_relative(relative):
                    return str(bundle_dir / relative)
        return ""

    model_path = bundle_dir / str(model_entry.get("path"))
    return {
        "ok": True,
        "error": "",
        "path": str(bundle_dir),
        "kind": str(manifest.get("kind") or KIND_UNKNOWN),
        "task": manifest.get("task"),
        "model": str(model_path),
        "labels": _path_of(ROLE_LABELS),
        "metadata": _path_of(ROLE_METADATA),
        "manifest": manifest,
    }
