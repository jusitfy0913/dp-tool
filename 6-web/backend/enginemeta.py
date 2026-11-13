# -*- coding: utf-8 -*-
"""引擎元数据：给一个 ``.engine`` 二进制留下「它是怎么来的」。

为什么需要它
------------
``.engine`` 里只有 TensorRT 序列化的网络，没有来源信息。上线后精度掉了，
只能靠人回忆它对应哪个 ONNX、哪次训练、哪批数据。而这三件事恰好都是
最容易同时存在多个版本的东西。

写在哪
------
``<engine>.meta.json``，与引擎同目录、同前缀（``model_fp16.engine`` →
``model_fp16.engine.meta.json``）。这样复制、改名、打包时它不会走散；
也不用在别处维护一张「引擎 → 来源」的索引表。

为什么不改 DLL
--------------
``yolo_infer.dll`` 只认引擎文件本身。元数据是**纯旁挂只读文件**，DLL 完全不知道
它存在，所以加这个功能不需要动 C++、不需要重编、也不会影响已有部署。

容错
----
写不进去（目录只读、磁盘满）只返回错误字符串，**绝不影响构建结果**——
引擎已经编译好了，为了一份说明把它判成失败是本末倒置。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

#: 训练目录里的指纹文件名（由 launcher 写）。用来把引擎与数据集接上。
TRAIN_META_NAME = "train_meta.json"

#: 从 ONNX 往上找几层父目录。引擎通常挨着 ONNX，ONNX 通常在 run 目录或 run/weights 下。
_MAX_PARENTS = 4

#: 大文件分块读，避免把几 GB 的 ONNX 整个读进内存。
_CHUNK = 4 * 1024 * 1024


def file_sha256(path: Path) -> str:
    """文件内容哈希。读不出来返回空串（元数据里如实留空，不编造）。"""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def find_train_meta(onnx_path: Path) -> dict[str, Any] | None:
    """从 ONNX 位置往上找 ``train_meta.json``。

    为什么是「往上找」而不是要求用户填：ONNX 通常在 ``<run>/best.onnx`` 或
    ``<run>/weights/best.onnx``，训练元数据在 ``<run>/train_meta.json``。
    让用户再填一次路径是把「本可以自动推断」的事变成新的填错来源。
    """
    current = onnx_path.parent
    for _ in range(_MAX_PARENTS + 1):
        candidate = current / TRAIN_META_NAME
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return payload if isinstance(payload, dict) else None
        if current.parent == current:
            break
        current = current.parent
    return None


def _runtime_block(
    engine_path: Path,
    onnx_path: Path,
    precision: str,
    imgsz: int | None,
    dynamic: bool,
) -> dict[str, Any]:
    """采集「这个引擎怎么来的」所需的环境信息。

    为什么放在 enginemeta 而不是 bundle：引擎构建成功后就要写这份说明，
    而造模型包可能永远不发生。记录在一处，两边都能读。

    任何异常都返回空字段——版本读不到不该让引擎构建被记成失败。
    """
    try:
        import bundle  # noqa: PLC0415 - 避免 enginemeta 在只做文件写入时也拖上它

        snapshot = bundle.runtime_snapshot(
            engine_path, precision=precision, imgsz=imgsz, dynamic=dynamic
        )
        snapshot["onnx_layout"] = "transposed" if ".transd." in onnx_path.name.lower() else "raw"
        return snapshot
    except Exception:  # noqa: BLE001
        return {
            "format": "engine",
            "precision": precision,
            "dynamic": bool(dynamic),
            "imgsz": [imgsz, imgsz] if imgsz else None,
            "trt_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "compute_cap": None,
            "onnx_layout": "transposed" if ".transd." in onnx_path.name.lower() else "raw",
        }


def write_meta(
    engine_path: Path,
    *,
    onnx_path: Path,
    precision: str,
    command: list[str] | None = None,
    run_id: str = "",
    dynamic: bool = False,
    app_version: str = "",
    imgsz: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写 ``<engine>.meta.json``，返回写进文件的 payload（含 ``error`` 字段时表示失败）。

    ``runtime`` 区块（TensorRT/CUDA 版本、显卡、compute capability、输入尺寸）
    是为模型包准备的：``.engine`` 二进制里没有这些信息，而「这个引擎在眼前这台
    机器上能不能跑」恰恰只能靠它们回答。采集失败只让字段为空，不影响构建结果。
    """
    engine_path = Path(engine_path)
    onnx_path = Path(onnx_path)
    train_meta = find_train_meta(onnx_path)
    dataset = (train_meta or {}).get("dataset") if isinstance(train_meta, dict) else None

    payload: dict[str, Any] = {
        "engine": str(engine_path),
        "engine_sha256": file_sha256(engine_path) if engine_path.is_file() else "",
        "onnx": str(onnx_path),
        "onnx_sha256": file_sha256(onnx_path),
        "precision": precision,
        "dynamic": bool(dynamic),
        "command": list(command or []),
        "run_id": run_id,
        "train_meta": str(onnx_path.parent / TRAIN_META_NAME) if train_meta else "",
        "dataset_fingerprint": (dataset or {}).get("dataset_fingerprint") if dataset else None,
        "data_fp": (dataset or {}).get("short") if dataset else "",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app_version": app_version,
        "runtime": _runtime_block(engine_path, onnx_path, precision, imgsz, dynamic),
    }
    if extra:
        payload.update(extra)

    target = engine_path.with_name(engine_path.name + ".meta.json")
    payload["path"] = str(target)
    try:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    if not onnx_path.is_file():
        payload["onnx_sha256"] = ""
        payload["error"] = payload.get("error") or "ONNX 已不存在，无法计算哈希"
    return payload


# ---------------------------------------------------------------------------
# 工作点（operating point）：漏检-过杀曲线上选定的那个 conf
#
# 为什么写在这里而不是 metadata.json：metadata.json 是 DLL 的输入契约，
# 改它会改变已有引擎的判定行为；旁挂文件改的只是「建议默认值」，用户不采用就无效。
# ---------------------------------------------------------------------------
def metadata_path(engine_path: Path | str) -> Path:
    """引擎的旁挂文件路径（与 :func:`write_meta` 用的是同一规则）。"""
    path = Path(engine_path)
    return path.with_name(path.name + ".meta.json")


def read_operating_point(engine_path: Path | str) -> dict[str, Any]:
    """读引擎工作点。文件不存在或没有该字段时返回空字典，不抛异常。"""
    path = metadata_path(engine_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    point = payload.get("operating_point") if isinstance(payload, dict) else None
    return point if isinstance(point, dict) else {}


def write_operating_point(engine_path: Path | str, point: dict[str, Any]) -> dict[str, Any]:
    """把工作点写进 ``<engine>.meta.json``，保留文件里的其它字段。

    写失败只返回 ``error``，绝不影响已经产出的引擎或验证报告——这与 ``write_meta``
    的容错口径一致：一份说明文件不该把已经成功的产物判成失败。
    """
    engine = Path(engine_path)
    target = metadata_path(engine)
    payload: dict[str, Any] = {}
    if target.is_file():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, ValueError):
            # 旧文件坏了就直接重建成只含工作点的新文件：丢的是可再生的来源信息，
            # 保的是这次用户显式保存的工作点。
            payload = {}

    saved = dict(point or {})
    saved["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload.setdefault("engine", str(engine))
    payload["operating_point"] = saved
    payload["path"] = str(target)
    try:
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload
