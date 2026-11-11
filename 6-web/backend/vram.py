# -*- coding: utf-8 -*-
"""显存预检：实测峰值表 + 安全余量 + 自校准。

为什么要单独一个模块
--------------------
这条策略现在有两个使用方：``6-web`` 的训练接口（开训前的闸门 + 页面体检）和
``7-autoresearch-yolo/autoexp.py``（无人值守跑实验）。两边必须**用同一套判定**，
否则会出现「页面上能开、autoexp 说不能开」这种自相矛盾。历史上这两处各写了一份
估算，已经在 960/batch=4 上给出了互斥的结论。

为什么不再用一条写死的公式
--------------------------
旧公式 ``2600 + 240*batch*(imgsz/640)**2`` 是手工拟合的，与实测脱节：
台账里 batch=8 @640 的峰值记的是 1769 MB，公式却估 4520 MB；到了 960/batch=4，
公式估 4760 MB 就放行，实际仍然 OOM。显存占用与 (batch, imgsz, 精度) 的关系
不是一条固定系数能覆盖的，所以改成「跑过就记下来，下次按实测判断」。

三层结构
--------
1. 实测峰值表：``(imgsz, batch, 精度) -> 历史最大峰值``，落在
   ``7-autoresearch-yolo/experiments/vram_samples.json``（见 ``config.VRAM_SAMPLES``）。
2. 放行条件（按可信度从高到低选依据）：
   - 精确命中样本：``free >= peak * 1.15 + 400MB``；
   - 只有更小 batch 的样本：按 batch 线性外推（偏保守，固定开销会被重复计一遍）；
   - 只有更大 batch 的样本：用它的实测值作**上界**（显存随 batch 不减）；
   - 什么都没跑过：公式 ``2600 + 300 * batch * (imgsz/640)^2``。
3. 自校准：每次训练正常结束把峰值写回表里，越用越准，不需要人工维护。

关键取舍（详见 openspec/changes/update-vram-preflight-amp-visibility/design.md）
--------------------------------------------------------------------------------
- 峰值优先取 ``max_memory_reserved``：OOM 发生在向驱动申请的那一刻，PyTorch 已经
  预留（reserved）的块才是它真正占住的显存；``allocated`` 会低估 30% 以上。
- 样本键包含精度（fp32/fp16）：同一张卡上两种精度的激活显存差近一倍，
  共用键会让「别处 fp16 的样本」放行「本机 fp32 的组合」。
- 只在精确命中时使用样本，不做插值：插值又回到「拟合系数」的老问题。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 参数：改这些会直接改变「放不放行」，改之前先看 design.md
# ---------------------------------------------------------------------------

#: 实测峰值的安全余量。15% 覆盖批次间波动（图片尺寸分布、cudnn 选核差异）。
MARGIN = 1.15

#: 额外预留的固定显存（MB）。给 CUDA context、cudnn workspace 之外的东西留余地：
#: 桌面合成、浏览器的 GPU 加速、其它进程的瞬时申请都不在 PyTorch 的统计里。
RESERVE_MB = 400.0

#: 无实测样本时的回退公式：固定开销 + 每样本开销 × batch × (imgsz/640)^2。
#: 这两个数来自 T1000 8GB 上的实测边界（batch=8 @640 可跑、batch=16 @640 在 5302MB 可用时 OOM），
#: 且刻意偏保守——它是闸门，不是精确预测。
FORMULA_FIXED_MB = 2600.0
FORMULA_PER_BATCH_MB = 300.0

#: 合理峰值的上下界（MB）。超出范围的读数来自驱动异常或别的进程，写进表里会长期误拦。
MIN_PEAK_MB = 200.0
MAX_PEAK_MB = 1024.0 * 1024.0

PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_UNKNOWN = "unknown"

#: 精度为「未知」时不查表（表里只有 fp32/fp16），直接走保守公式。
_KNOWN_PRECISIONS = (PRECISION_FP32, PRECISION_FP16)

_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def store_path() -> Path:
    """实测峰值表的位置。

    优先 ``config.VRAM_SAMPLES``；为了能在没有 config 的场景（单测、脚本）里用，
    再按同样的规则自己推一遍。``TRTFORGE_VRAM_SAMPLES`` 永远最优先。
    """
    raw = os.environ.get("TRTFORGE_VRAM_SAMPLES", "").strip()
    if raw:
        return Path(raw).resolve()
    try:
        import config  # type: ignore

        return Path(config.VRAM_SAMPLES)
    except Exception:  # noqa: BLE001 - 没有 config 时按目录约定推
        backend = Path(__file__).resolve().parent
        repo = backend.parent.parent
        if (repo / "7-autoresearch-yolo").is_dir():
            return repo / "7-autoresearch-yolo" / "experiments" / "vram_samples.json"
        return backend.parent / "runtime" / "vram_samples.json"


def normalize_precision(value: Any) -> str:
    """把各种写法收敛成 ``fp32`` / ``fp16`` / ``unknown``。

    launcher 的 ``args.amp`` 可能是 ``True``/``False``/``"fp32"``/``"bf16"``，
    trainer 上又是 0 维张量，所以统一在这里翻译，别让调用方各自判断。
    """
    # 0 维张量（trainer.amp）走 .item()，否则会落到最后的 unknown 分支
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return normalize_precision(value.item())
        except Exception:  # noqa: BLE001
            return PRECISION_UNKNOWN
    if isinstance(value, bool):
        return PRECISION_FP16 if value else PRECISION_FP32
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"fp16", "amp", "half", "true", "1", "16"}:
            return PRECISION_FP16
        if text in {"fp32", "false", "0", "no", "32"}:
            return PRECISION_FP32
        if text in {"bf16", "bfloat16"}:
            # bf16 的显存占用与 fp16 同量级，按 fp16 归档即可（表里不细分）
            return PRECISION_FP16
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return PRECISION_FP16 if value else PRECISION_FP32
    return PRECISION_UNKNOWN


def _resolve_precision(precision: Any) -> str:
    """把 ``None`` 解释成「按本机情况预测」，其余照字面归一。

    为什么需要这一层：显存表按精度分键，若调用方不传精度就用 ``unknown`` 查表，
    永远查不到任何样本——自校准就白做了。默认「自己判断」比默认「不知道」安全。
    """
    if precision is not None:
        return normalize_precision(precision)
    try:
        import ampcheck  # noqa: PLC0415 - 延迟导入，避免 vram 只做存储时也拖上一个依赖

        return ampcheck.expected_precision(None, gpu_name())
    except Exception:  # noqa: BLE001 - 判断失败就按未知处理（走保守公式）
        return PRECISION_UNKNOWN


def sample_key(imgsz: int, batch: int, precision: str = PRECISION_UNKNOWN) -> str:
    """表键：``960x4xfp32``。人读得懂，也方便直接手改文件删掉坏样本。"""
    return f"{int(imgsz)}x{int(batch)}x{normalize_precision(precision)}"


# ---------------------------------------------------------------------------
# GPU 探测
# ---------------------------------------------------------------------------
def _torch_gpu_info() -> dict[str, Any] | None:
    try:
        import torch  # noqa: PLC0415 - 只有训练侧进程才有 torch，Web 进程按需探测

        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return {
            "name": torch.cuda.get_device_name(index),
            "total_mb": round(total_bytes / (1024 ** 2), 1),
            "free_mb": round(free_bytes / (1024 ** 2), 1),
            "source": "torch",
        }
    except Exception:  # noqa: BLE001 - 没装 torch / 驱动异常都走 nvidia-smi
        return None


def _smi_gpu_info() -> dict[str, Any] | None:
    """零依赖兜底：Web 服务进程通常跑在没装 torch 的系统 Python 上。"""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        total_mb = float(parts[1])
        free_mb = float(parts[2])
    except ValueError:
        return None
    return {"name": parts[0], "total_mb": total_mb, "free_mb": free_mb, "source": "nvidia-smi"}


def gpu_info() -> dict[str, Any] | None:
    """当前 GPU 的名称与显存（MB）。没有任何可用探测手段时返回 ``None``。

    只做一次即时读取（free 会变），不做缓存；调用频率是「点一次按钮」级别。
    """
    return _torch_gpu_info() or _smi_gpu_info()


def gpu_name() -> str | None:
    info = gpu_info()
    return None if info is None else str(info.get("name") or "") or None


# ---------------------------------------------------------------------------
# 实测峰值表
# ---------------------------------------------------------------------------
def _empty_store() -> dict[str, Any]:
    return {"_comment": _COMMENT, "updated_at": None, "samples": {}}


_COMMENT = (
    "显存实测峰值表。键格式 imgsz x batch x 精度（如 960x4xfp32），"
    "值为该组合历次训练的最大峰值（torch.cuda.max_memory_reserved，MB）。"
    "由训练结束自动写入，不要手工改成「理论值」，否则预检会失去意义。"
)


def load_samples(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """读实测表。文件不存在、内容损坏、结构不对都视为空表，绝不抛异常。

    为什么容错到这个程度：这张表只是「更好的估算依据」，读不出来就该退回公式继续干活，
    不能因为它把训练启动流程弄挂。
    """
    target = Path(path) if path is not None else store_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for key, entry in samples.items():
        if not isinstance(entry, dict):
            continue
        try:
            peak = float(entry.get("peak_mb"))
        except (TypeError, ValueError):
            continue
        if MIN_PEAK_MB <= peak <= MAX_PEAK_MB:
            cleaned[str(key)] = {**entry, "peak_mb": peak}
    return cleaned


def save_samples(samples: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    """原子写入（临时文件 + ``os.replace``），避免读到半个文件。"""
    target = Path(path) if path is not None else store_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": _COMMENT,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": samples,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def record_sample(
    imgsz: int,
    batch: int,
    peak_mb: float,
    precision: Any = PRECISION_UNKNOWN,
    peak_allocated_mb: float | None = None,
    exp_id: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """记录一次实测峰值（自校准）。同一组合只保留最大值。

    返回值是这个键的最新条目；被拒绝的读数返回 ``{"skipped": 原因}``。

    失败一律吞掉：自校准是「顺手让下次更准」，不该让一次训练因为写不了文件而失败。
    """
    try:
        peak = float(peak_mb)
    except (TypeError, ValueError):
        return {"skipped": f"峰值不是数字：{peak_mb!r}"}
    if not (MIN_PEAK_MB <= peak <= MAX_PEAK_MB):
        return {"skipped": f"峰值 {peak} MB 超出合理范围 [{MIN_PEAK_MB}, {MAX_PEAK_MB}]"}

    key = sample_key(imgsz, batch, precision)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _WRITE_LOCK:
        samples = load_samples(path)
        entry = dict(samples.get(key) or {})
        previous = float(entry.get("peak_mb") or 0.0)
        entry.update(
            {
                "imgsz": int(imgsz),
                "batch": int(batch),
                "precision": normalize_precision(precision),
                "peak_mb": max(previous, peak),
                "count": int(entry.get("count") or 0) + 1,
                "last_at": now,
                "first_at": entry.get("first_at") or now,
            }
        )
        if peak_allocated_mb is not None:
            try:
                entry["peak_allocated_mb"] = round(float(peak_allocated_mb), 1)
            except (TypeError, ValueError):
                pass
        if exp_id:
            entry["last_exp"] = str(exp_id)
        samples[key] = entry
        try:
            save_samples(samples, path)
        except OSError as exc:  # 只报，不抛
            return {"skipped": f"写入失败：{type(exc).__name__}: {exc}"}
    return entry


# ---------------------------------------------------------------------------
# 估算与预检
# ---------------------------------------------------------------------------
def formula_mb(imgsz: int, batch: int) -> float:
    """无样本时的保守估算。"""
    return FORMULA_FIXED_MB + FORMULA_PER_BATCH_MB * max(int(batch), 1) * (int(imgsz) / 640.0) ** 2


def estimate(imgsz: int, batch: int, precision: Any = None) -> dict[str, Any]:
    """估算峰值需求。

    三档依据，优先级从高到低：

    1. ``measured``：同 (imgsz, batch, 精度) 有实测样本，直接用（最可信）；
    2. ``scaled``：同 (imgsz, 精度) 有**更小 batch** 的样本，按 batch 线性外推。
       为什么需要这一档：公式在「卡大、batch 大」时过于保守——本机 640/batch=8
       实测只吃 2132 MB，而公式说 batch=16 要 7400 MB，于是默认参数一开页面就是
       「显存不够」。线性外推的偏差方向是**安全**的：实测峰值里含固定的
       CUDA context / workspace，按 batch 等比放大会把这部分重复计一遍；
    3. ``formula``：什么都没跑过，只能用公式（保守，宁可拦也不放行）。

    只向上外推（请求 batch ≥ 样本 batch）。向下等比缩小会把固定开销也缩掉，
    得到偏小的数——那正是「放行了会 OOM 的组合」这种最坏的错。
    """
    imgsz_int = max(int(imgsz), 32)
    batch_int = max(int(batch), 1)
    prec = _resolve_precision(precision)
    base_formula = round(formula_mb(imgsz_int, batch_int), 1)
    result: dict[str, Any] = {
        "imgsz": imgsz_int,
        "batch": batch_int,
        "precision": prec,
        "key": sample_key(imgsz_int, batch_int, prec),
        "formula_mb": base_formula,
        "measured_mb": None,
        "sample_batch": None,
        "sample_count": 0,
        "source": "formula",
        "needed_mb": base_formula,
        "margin": MARGIN,
        "reserve_mb": RESERVE_MB,
    }
    if prec not in _KNOWN_PRECISIONS:
        return result

    samples = load_samples()
    entry = samples.get(result["key"])
    if entry:
        peak = float(entry["peak_mb"])
        result.update(
            {
                "measured_mb": round(peak, 1),
                "sample_batch": batch_int,
                "sample_count": int(entry.get("count") or 1),
                "source": "measured",
                "needed_mb": round(peak * MARGIN + RESERVE_MB, 1),
            }
        )
        return result

    # 同分辨率、同精度的样本分两类：比请求小的（可向上外推）和比请求大的（是需求的上界）。
    smaller: list[tuple[int, float]] = []
    larger: list[float] = []
    for item in samples.values():
        if int(item.get("imgsz") or 0) != imgsz_int:
            continue
        if normalize_precision(item.get("precision")) != prec:
            continue
        anchor_batch = int(item.get("batch") or 0)
        if anchor_batch <= 0:
            continue
        peak = float(item["peak_mb"])
        if anchor_batch <= batch_int:
            smaller.append((anchor_batch, peak))
        else:
            larger.append(peak)

    if smaller:
        anchor_batch, anchor_peak = max(smaller, key=lambda pair: pair[0])
        scaled = round(anchor_peak * batch_int / anchor_batch, 1)
        result.update(
            {
                "measured_mb": scaled,
                "sample_batch": anchor_batch,
                "sample_count": len(smaller),
                "source": "scaled",
                "needed_mb": round(scaled * MARGIN + RESERVE_MB, 1),
            }
        )

    # 单调性上界：显存占用随 batch 不减，所以「更大 batch 的实测峰值」就是当前 batch
    # 需求的上界。没有这条，公式会算出「batch 调小反而更吃显存」的荒唐结论
    # （640/batch=8 有实测 2132，batch=7 却按公式要 4700），被拦住的人照着建议
    # 往下调只会越来越红。
    if larger:
        bound = round(min(larger) * MARGIN + RESERVE_MB, 1)
        if bound < result["needed_mb"]:
            result.update(
                {
                    "measured_mb": round(min(larger), 1),
                    "source": "capped",
                    "needed_mb": bound,
                }
            )
    return result


def suggest_batch(imgsz: int, free_mb: float, precision: Any = None, limit: int = 64) -> int | None:
    """在给定可用显存下，按同一套估算反推能跑的最大 batch。

    只用于「被拦住时给出可执行建议」，不参与放行判定，所以线性递减扫描足够。
    """
    for candidate in range(max(int(limit), 1), 0, -1):
        if estimate(imgsz, candidate, precision)["needed_mb"] <= free_mb:
            return candidate
    return None


def _describe(est: dict[str, Any]) -> str:
    if est["source"] == "measured":
        return (
            f"实测峰值 {est['measured_mb']:.0f} MB × {MARGIN:.2f} + {RESERVE_MB:.0f} MB"
            f"（来自 {est['sample_count']} 次历史训练，键 {est['key']}）"
        )
    if est["source"] == "scaled":
        return (
            f"实测峰值 {est['measured_mb']:.0f} MB × {MARGIN:.2f} + {RESERVE_MB:.0f} MB"
            f"（按 batch={est['sample_batch']} 的实测样本线性外推到 batch={est['batch']}）"
        )
    if est["source"] == "capped":
        return (
            f"实测峰值 {est['measured_mb']:.0f} MB × {MARGIN:.2f} + {RESERVE_MB:.0f} MB"
            f"（取更大 batch 的实测值做上界：显存占用随 batch 不减）"
        )
    return (
        f"无实测样本，按公式 {FORMULA_FIXED_MB:.0f} + {FORMULA_PER_BATCH_MB:.0f} × batch × (imgsz/640)² 估算"
    )


def preflight(
    imgsz: int,
    batch: int,
    precision: Any = None,
    free_mb: float | None = None,
) -> dict[str, Any]:
    """开训前的显存预检。

    ``free_mb`` 显式传入时不再探测（测试与「先探测一次、多处使用」的场景用）。

    读不到显存（没有 CUDA / 没有 nvidia-smi）时**放行**：这台机器本来也训不了 CUDA 训练，
    拦在这里只会把错误信息变成误导。
    """
    info = None
    if free_mb is None:
        info = gpu_info()
        free_mb = info.get("free_mb") if info else None
    prec = _resolve_precision(precision)

    # batch 可能是 "auto" / "8,16" / "-1"（Ultralytics 自己搜）：这时没法用单点估算，
    # 只能放行并在返回里说明。**任何情况下这里都不能抛异常**——预检是闸门，
    # 不是入口；闸门自己出错会把「本来能跑的训练」拦成 500。
    try:
        imgsz_int = int(imgsz)
        batch_int = int(batch)
    except (TypeError, ValueError):
        imgsz_int = batch_int = 0
    if imgsz_int <= 0 or batch_int <= 0:
        return {
            "ok": True,
            "skipped": f"imgsz={imgsz!r} batch={batch!r} 不是正整数，无法估算",
            "imgsz": imgsz if imgsz_int <= 0 else imgsz_int,
            "batch": None,
            "precision": prec,
            "needed_mb": None,
            "source": "skipped",
            "free_mb": None if free_mb is None else round(float(free_mb), 1),
            "gpu": (info or {}).get("name"),
            "total_mb": (info or {}).get("total_mb"),
            "reason": f"imgsz={imgsz!r} batch={batch!r} 无法用单点估算，已跳过预检",
            "hint": "",
        }

    est = estimate(imgsz_int, batch_int, prec)
    payload: dict[str, Any] = {
        **est,
        "free_mb": None if free_mb is None else round(float(free_mb), 1),
        "gpu": (info or {}).get("name"),
        "total_mb": (info or {}).get("total_mb"),
        "hint": "",
    }
    if free_mb is None:
        payload["ok"] = True
        payload["reason"] = "无法读取显存（没有 CUDA 或 nvidia-smi），已跳过预检"
        return payload

    free = float(free_mb)
    if free >= est["needed_mb"]:
        payload["ok"] = True
        payload["reason"] = (
            f"可用显存 {free:.0f} MB ≥ 需求 {est['needed_mb']:.0f} MB"
            f"（{_describe(est)}；imgsz={est['imgsz']} batch={est['batch']} {est['precision']}）"
        )
        return payload

    better = suggest_batch(est["imgsz"], free, est["precision"])
    hint = "请先停掉 6-web 的常驻推理进程（infer_worker）或其它占用显存的程序"
    if better:
        hint += f"，或把 batch 降到 {better} 或更小"
    else:
        hint += "，或把 imgsz 调小"
    payload["ok"] = False
    payload["suggest_batch"] = better
    payload["reason"] = (
        f"可用显存 {free:.0f} MB < 需求 {est['needed_mb']:.0f} MB"
        f"（{_describe(est)}；imgsz={est['imgsz']} batch={est['batch']} {est['precision']}）"
    )
    payload["hint"] = hint
    return payload


def snapshot() -> dict[str, Any]:
    """给页面用的一份只读快照：GPU + 表规模。"""
    info = gpu_info()
    samples = load_samples()
    return {
        "gpu": info,
        "sample_count": len(samples),
        "store": str(store_path()),
        "samples": sorted(
            (
                {
                    "key": key,
                    "imgsz": entry.get("imgsz"),
                    "batch": entry.get("batch"),
                    "precision": entry.get("precision"),
                    "peak_mb": entry.get("peak_mb"),
                    "count": entry.get("count"),
                    "last_at": entry.get("last_at"),
                }
                for key, entry in samples.items()
            ),
            key=lambda item: (item.get("imgsz") or 0, item.get("batch") or 0),
        ),
    }
