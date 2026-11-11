#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRTForge YOLO 自主实验循环驱动器（autoexp）。

设计目标
--------
把 7-autoresearch-master 的方法论（固定评估 + 固定时间预算 + 单点可改 + keep/discard）
移植到本工程的 YOLO 训练上，同时满足两条硬约束：

1. **对生产代码零改动**。不修改 1-python-yolo/launcher/tf_yolo_launcher.py，
   只 import 它的纯函数（build_parser / build_train_kwargs / load_yolo），
   保证超参映射与生产 100% 一致。
2. **写盘严格收敛在 7-autoresearch-yolo/ 内**。数据集用硬链接镜像到 splits/，
   训练产物落 runs/autoexp/，绝不写 PCB 原始数据集目录或其它模块的产物目录。

子命令
------
split    把 data.yaml 里固定的 val 集确定性切成 tune / holdout，并建硬链接镜像沙箱
run      跑一次实验（时间预算硬截断），追加一行台账到 results.tsv
holdout  在 holdout 上验收某一份配置（搜索期禁止调用）
noise    同一配置重复 N 次，测噪声底 sigma，决定 keep/discard 阈值
report   打印台账

单次实验的 exec 阶段跑在独立子进程里（`_exec`），保证 OOM / 段错误不会带走台账。

编码：全文件 UTF-8，跨平台不出现乱码。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.util
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DP_TOOL = ROOT.parent

LAUNCHER_DIR = DP_TOOL / "1-python-yolo" / "launcher"
LAUNCHER_PY = LAUNCHER_DIR / "tf_yolo_launcher.py"

CONFIGS_DIR = ROOT / "configs"
BASE_JSON = CONFIGS_DIR / "base.json"
SPACE_JSON = CONFIGS_DIR / "space.json"
EXPERIMENTS_DIR = ROOT / "experiments"
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs" / "autoexp"
WEIGHTS_DIR = ROOT / "weights"

RESULTS_TSV = ROOT / "results.tsv"
HOLDOUT_TSV = ROOT / "holdout.tsv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# 第 4 阶段：LLM 决策层（agent 子命令）
# ---------------------------------------------------------------------------
# 这些路径单独成组，避免和上面「跑一次实验」那套混在一起看。
AGENT_DIR = EXPERIMENTS_DIR / "agent"
#: 每轮审计快照（请求、响应、tokens、判定）。排查「模型当时看到了什么」全靠它。
AGENT_AUDIT_PREFIX = "round_"
PROMPT_MD = ROOT / "prompts" / "experimenter.md"
BEST_JSON = EXPERIMENTS_DIR / "best.json"
NOISE_JSON = EXPERIMENTS_DIR / "_noise_summary.json"
#: agent 每轮写这里的改动项，与人工流程完全一致（人也能看懂、也能直接 run）
CUR_JSON = EXPERIMENTS_DIR / "cur.json"

#: 给 Web 端读的结构化事件前缀。必须与 launcher 完全一致，
#: 否则后端 `_parse_line()` 会把它当普通日志，界面上就看不到决策块。
JSON_PREFIX = "TRTFORGE_JSON "

#: 是否把事件打到 stdout。只有 agent 子命令加 --events 才打开；
#: 其它子命令保持原来的「只用 stderr 说话」行为，避免污染原有输出。
EVENTS_ENABLED = False

#: 冻结字段：即使 space.json 里声明了，也不许模型改。
#:
#: 为什么必须冻结 `data`：space.json 根本不校验它。一旦让模型动它，
#: 理论上可以把 splits/pcb_tune.yaml 换成 pcb_holdout.yaml ——
#: holdout 纪律当场作废，而台账上一切正常，没有任何迹象。
#:
#: 为什么第一版连 `model` 一起冻结：n -> s -> m 会显著改变 600 秒内能跑完的
#: epoch 数（m 可能只跑 3~4 轮），很容易整晚都在试大模型而没有进展。
#: 想放开只需把 "model" 从这个集合里删掉，其它代码不用动。
FROZEN_TOP = {"data", "task", "budget_seconds", "epochs", "model", "name"}
FROZEN_TRAIN = {"device", "seed", "deterministic", "cache", "workers",
                "amp", "patience", "plots", "val", "save", "profile", "fraction"}

#: 提议里允许出现的顶层键（其余一律拒绝）。
PROPOSAL_KEYS = {"hypothesis", "changed", "reasoning", "train"}

#: 噪声摘要缺失时的兜底阈值。等于 2026-11-10 实测的 2 sigma（0.006368）。
#: 用一个实测值兜底而不是 0：0 会让任何微小波动都被判成 KEEP。
FALLBACK_KEEP_THRESHOLD = 0.0064

#: 主动中止训练时的退出码（batch 被 Ultralytics 静默降级）。
#: 与普通崩溃区分开：这不是「训练出错」，而是「我们发现这次实验已经不可比，
#: 于是主动中止」——所以不该计入 crash 统计，也不该写 fitness。
EXIT_BATCH_DOWNGRADED = 78

#: 禁止写入的生产产物目录。命中即抛异常，不做「尽力而为」。
FORBIDDEN_ROOTS = [
    DP_TOOL / "1-python-yolo" / "launcher" / "runs",
    DP_TOOL / "2-python-anomaly" / "launcher" / "results",
    DP_TOOL / "6-web" / "runtime",
    DP_TOOL / "4-qt-studio",
    DP_TOOL / "3-cpp-tensorrt",
]

#: 这些键不在 launcher 的 CLI 白名单里，但 Ultralytics 的 trainer 接受，
#: 因此直接注入 train kwargs，绕过 CLI 限制（不需要改生产代码）。
DIRECT_TRAIN_KEYS = {
    "seed",
    "deterministic",
    "time",
    "nbs",
    "cos_lr",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "multi_scale",
    "rect",
    "fraction",
    "profile",
    "val",
    "save",
    "save_period",
}

RESULTS_HEADER = [
    "exp_id",
    "created_at",
    "fitness_map50_95",
    "best_map50_95",
    "map50",
    "precision",
    "recall",
    "fp_per_image",
    "epochs_done",
    "train_seconds",
    "peak_vram_mb",
    "peak_vram_reserved_mb",
    "imgsz",
    "batch_req",
    "batch_eff",
    "amp",
    "data_fp",
    "model",
    "budget_seconds",
    "status",
    "hypothesis",
    "changed",
]

HOLDOUT_HEADER = [
    "exp_id",
    "created_at",
    "weights",
    "split_yaml",
    "map50_95",
    "map50",
    "precision",
    "recall",
    "fp_per_image",
    "seconds",
    "num_images",
    "num_instances",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    """统一日志出口：写 stderr 并立即 flush，父进程重定向到文件时不会丢行。"""
    print(f"[autoexp] {message}", file=sys.stderr, flush=True)


def emit_event(payload: dict[str, Any]) -> None:
    """把一条结构化事件写到 stdout（Web 端据此渲染决策块与进度）。

    为什么与 :func:`log` 分开两条流：
    - `log()` 走 stderr，是**给人看的**、带 `[autoexp]` 前缀、粘在日志文件里；
    - `emit_event()` 走 stdout，是**给程序解析的**，一行一个 JSON。
    后端 `runners._parse_line()` 正是按 stdout 里的 `TRTFORGE_JSON ` 前缀分流。
    混在一起的话，`[autoexp] ...` 这种半结构化文本会让界面出现两套格式。

    `EVENTS_ENABLED` 为 False 时静默丢弃，保证 run / noise / holdout 这些
    老命令的输出与加这个功能之前逐字节一致。
    """
    if not EVENTS_ENABLED:
        return
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        body = json.dumps({"type": "log", "line": str(payload)}, ensure_ascii=False)
    print(JSON_PREFIX + body, flush=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def annotate_exp_record(exp_id: str, extra: dict[str, Any]) -> None:
    """在 ``experiments/exp_NNNN.json`` 上补记「训练后才知道」的事实。

    为什么需要这一步：``exp_NNNN.json`` 是在开跑**之前**写的快照（那时只有请求参数），
    而「实际用了 fp32 还是 fp16」「真正吃到多少显存」只有跑完才知道。
    ``args.yaml`` 记的同样是请求值——黑名单卡上请求 ``amp=True`` 实际是 fp32，
    不回写就永远查不到真相。只补键、不覆盖已有内容，失败只记一行日志。
    """
    path = EXPERIMENTS_DIR / f"{exp_id}.json"
    if not path.is_file():
        return
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    changed = False
    for key, value in extra.items():
        if value is None or value == "":
            continue
        if payload.get(key) != value:
            payload[key] = value
            changed = True
    if not changed:
        return
    try:
        save_json(path, payload)
    except OSError as exc:
        log(f"{exp_id} 实验记录补记失败（不影响结论）：{exc}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并，override 优先。用于让 cur.json 只写「本次改动的那几项」。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def assert_sandbox(target: Path, what: str) -> Path:
    """写入守卫：目标必须落在 7-autoresearch-yolo/ 内，且不得命中生产产物目录。"""
    resolved = Path(target).resolve()
    root = ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"拒绝写入：{what} = {resolved} 不在沙箱 {root} 内。"
            "自动实验循环只允许写 7-autoresearch-yolo/。"
        )
    for forbidden in FORBIDDEN_ROOTS:
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise PermissionError(f"拒绝写入：{what} = {resolved} 命中受保护的生产目录 {forbidden}。")
    return resolved


def ensure_tsv(path: Path, header: list[str]) -> None:
    """确保台账存在且表头与当前版本一致。

    表头升级时必须按**列名**迁移旧数据，不能直接覆盖：否则新增一列之后，
    所有历史行都会整体错位一格，看起来像数据被写坏了。
    """
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(header)
        return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        old_header = next(reader, [])
        old_rows = list(reader)
    if old_header == header:
        return

    # 按列名重排，缺的列补空串，多的列丢弃（列的语义由 header 定义）
    index = {name: position for position, name in enumerate(old_header)}
    migrated: list[list[str]] = []
    for row in old_rows:
        if not any(cell.strip() for cell in row):
            continue
        migrated.append([row[index[name]] if name in index and index[name] < len(row) else "" for name in header])

    backup = path.with_suffix(path.suffix + ".bak")
    path.replace(backup)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(migrated)
    log(f"台账表头已升级（旧文件备份为 {backup.name}，{len(migrated)} 行按列名迁移）")


def append_tsv(path: Path, header: list[str], row: dict[str, Any]) -> None:
    ensure_tsv(path, header)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([row.get(key, "") for key in header])


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def next_exp_id() -> str:
    """按 experiments/exp_*.json 递增编号，保证台账 ID 单调不重复。"""
    highest = 0
    if EXPERIMENTS_DIR.is_dir():
        for item in EXPERIMENTS_DIR.glob("exp_*.json"):
            stem = item.stem
            try:
                highest = max(highest, int(stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return f"exp_{highest + 1:04d}"


#: 低于这个可用内存就不建议开跑：Windows 上 dataloader worker 会各自重新 import torch，
#: 每个 worker 常驻数百 MB，8 个工作进程能把一台 32 GB 机器直接压死。
MIN_FREE_RAM_GB = 4.0
MIN_FREE_VRAM_GB = 2.5


def probe_resources() -> dict[str, Any]:
    """采集可用内存/显存，供开跑前判断和写进台账。"""
    info: dict[str, Any] = {}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info["free_ram_gb"] = round(status.ullAvailPhys / (1024 ** 3), 2)
            info["total_ram_gb"] = round(status.ullTotalPhys / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001 - 非 Windows 或调用失败都不阻塞
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["free_vram_gb"] = round(free / (1024 ** 3), 2)
            info["total_vram_gb"] = round(total / (1024 ** 3), 2)
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


def preflight(strict: bool) -> dict[str, Any]:
    """开跑前的资源体检。

    这台机器同时跑着 Web 服务（含 2 个 infer_worker 进程）和编辑器，
    32 GB 内存经常只剩 4~8 GB。经验教训：不是显存不够，而是**内存**先炸，
    表现却是 cuDNN 抛 CUDA OOM / DefaultCPUAllocator 失败，极易误判。
    """
    info = probe_resources()
    problems: list[str] = []
    free_ram = info.get("free_ram_gb")
    if free_ram is not None and free_ram < MIN_FREE_RAM_GB:
        problems.append(f"可用内存仅 {free_ram} GB（要求 >= {MIN_FREE_RAM_GB} GB）")
    free_vram = info.get("free_vram_gb")
    if free_vram is not None and free_vram < MIN_FREE_VRAM_GB:
        problems.append(f"可用显存仅 {free_vram} GB（要求 >= {MIN_FREE_VRAM_GB} GB）")

    if problems:
        for item in problems:
            log(f"资源不足：{item}")
        if strict:
            log("提示：可以先停掉 6-web 后端与其它吃显存/内存的程序，或用 --allow-low-resources 强制开跑。")
            raise RuntimeError("资源体检未通过：" + "；".join(problems))
        log("资源偏紧，按 --allow-low-resources 继续（结果可能不稳定）")
    else:
        log(
            "资源体检通过：内存 {ram} GB 可用，显存 {vram} GB 可用".format(
                ram=free_ram if free_ram is not None else "?",
                vram=free_vram if free_vram is not None else "?",
            )
        )
    return info


# ---------------------------------------------------------------------------
# launcher 复用层
# ---------------------------------------------------------------------------


def load_launcher() -> Any:
    """以模块方式载入生产 launcher（只读复用，不执行其 main）。"""
    if not LAUNCHER_PY.is_file():
        raise FileNotFoundError(f"找不到生产 launcher：{LAUNCHER_PY}")
    if "tf_yolo_launcher" in sys.modules:
        return sys.modules["tf_yolo_launcher"]
    spec = importlib.util.spec_from_file_location("tf_yolo_launcher", LAUNCHER_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法载入 launcher：{LAUNCHER_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tf_yolo_launcher"] = module
    spec.loader.exec_module(module)
    return module


def resolve_weights(model: str) -> str:
    """权重一律尽量给绝对路径，避免 Ultralytics 隐式下载到当前目录。"""
    candidate = Path(model)
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate)
    for base in (WEIGHTS_DIR, LAUNCHER_DIR):
        local = base / model
        if local.is_file():
            return str(local.resolve())
    # 本地没有：原样返回，交给 Ultralytics 决定（会触发下载，README 已注明）
    return model


def resolve_data_path(data: str) -> str:
    candidate = Path(data)
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate)
    local = ROOT / data
    if local.is_file():
        return str(local.resolve())
    raise FileNotFoundError(f"dataset.yaml 不存在：{data}（相对路径以 {ROOT} 为基准）")


def build_argv(cfg: dict[str, Any], train: dict[str, Any], project: Path, name: str) -> list[str]:
    """把 JSON 配置翻译成 launcher 的 CLI argv。

    做法：反射 launcher 自己的 argparse parser，按 action 类型决定是否需要值、
    布尔开关该发 --flag 还是 --no-flag。这样即使 launcher 以后加参数也不会错位。
    """
    launcher = load_launcher()
    parser = launcher.build_parser()
    actions = {action.dest: action for action in parser._actions}

    argv: list[str] = ["--mode", "train"]
    argv += ["--task", str(cfg.get("task", "detect"))]
    argv += ["--model", str(cfg.get("model", "yolov8n.pt"))]
    argv += ["--data", resolve_data_path(str(cfg["data"]))]
    argv += ["--project", str(project)]
    argv += ["--name", str(name)]
    argv += ["--exist-ok"]
    argv += ["--epochs", str(int(cfg.get("epochs", 100)))]

    for key, value in train.items():
        if key in DIRECT_TRAIN_KEYS:
            continue  # 这些不走 CLI，稍后直接注入 kwargs
        action = actions.get(key)
        if action is None:
            raise KeyError(f"launcher 不认识训练参数：{key}（请检查 configs/base.json）")
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        # BooleanOptionalAction：True 发 --flag，False 发 --no-flag
        if action.__class__.__name__ == "BooleanOptionalAction":
            argv.append(flag if value else f"--no-{key.replace('_', '-')}")
        elif action.nargs == 0:  # store_true / store_false
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    return argv


def validate_against_space(cfg: dict[str, Any], space: dict[str, Any]) -> list[str]:
    """按 space.json 校验配置。返回错误列表（空表示通过）。"""
    errors: list[str] = []

    def check_scalar(label: str, value: Any, rule: dict[str, Any]) -> None:
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{label} = {value!r} 不在允许取值 {rule['enum']} 内")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            low = rule.get("min")
            high = rule.get("max")
            if low is not None and value < low:
                errors.append(f"{label} = {value} 小于下限 {low}")
            if high is not None and value > high:
                errors.append(f"{label} = {value} 大于上限 {high}")
        if isinstance(value, str) and "enum" not in rule:
            pass

    for key, rule in space.items():
        if key == "train":
            continue
        if key in cfg and isinstance(rule, dict):
            check_scalar(key, cfg[key], rule)

    allowed_train = set(space.get("train", {}).keys())
    for key, value in (cfg.get("train") or {}).items():
        label = f"train.{key}"
        if allowed_train and key not in allowed_train:
            errors.append(f"{label} 不在搜索空间白名单内（space.json 未声明）")
            continue
        rule = space.get("train", {}).get(key)
        if isinstance(rule, dict):
            check_scalar(label, value, rule)
    return errors


# ---------------------------------------------------------------------------
# results.csv 解析
# ---------------------------------------------------------------------------

_METRIC_KEYS = {
    "map50_95": ("metrics/mAP50-95",),
    "map50": ("metrics/mAP50",),
    "precision": ("metrics/precision",),
    "recall": ("metrics/recall",),
}


def _pick(row: dict[str, str], candidates: tuple[str, ...]) -> float | None:
    """按候选前缀匹配列名（兼容 (B)/(M)/(P)/(O) 等任务后缀）。"""
    for name, raw in row.items():
        for prefix in candidates:
            if name.startswith(prefix + "(") or name == prefix:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None
    return None


def read_results_csv(csv_path: Path) -> dict[str, Any]:
    """读 Ultralytics 的 results.csv：返回最后一轮指标 + 历史最佳。"""
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    last = rows[-1]
    payload: dict[str, Any] = {
        "epochs_done": int(float(last.get("epoch", 0) or 0)),
        "train_seconds": round(float(last.get("time", 0) or 0), 2),
    }
    for field, candidates in _METRIC_KEYS.items():
        payload[field] = _pick(last, candidates)

    history = [v for v in (_pick(row, _METRIC_KEYS["map50_95"]) for row in rows) if v is not None]
    payload["best_map50_95"] = max(history) if history else None
    return payload


#: val 集统计缓存：键是 dataset.yaml 的绝对路径字符串。
#: autoexp 每轮都要用同一个 val 集，重复扫标签只是浪费 IO。
_VAL_STATS_CACHE: dict[str, dict[str, int]] = {}


def val_gt_stats(data_path: str) -> dict[str, int]:
    """统计 val 划分的图片数与 GT 框数。读不到返回空字典。

    这两项是用来把 Ultralytics 的 P/R 换算成「每图过杀数」的分母：
    ``TP = R × GT``，``FP = TP × (1-P)/P``，``每图过杀 = FP / 图片数``。
    拿不到就留空，绝不猜一个数填进台账。
    """
    try:
        key = str(Path(resolve_data_path(data_path)).resolve())
    except (FileNotFoundError, OSError):
        return {}
    if key in _VAL_STATS_CACHE:
        return dict(_VAL_STATS_CACHE[key])
    module = load_web_module("traininfo")
    if module is None:
        return {}
    try:
        splits = module.resolve_splits(key)
        dirs = splits["splits"].get("val") or []
        yaml_dir = Path(splits["yaml_dir"])
        images = 0
        instances = 0
        for directory in dirs:
            directory = Path(directory)
            split_images = module.collect_images(directory)
            images += len(split_images)
            label_dir = module.labels_dir_for(directory, yaml_dir, "val")
            for image_path in split_images:
                label = label_dir / f"{image_path.stem}.txt"
                if not label.is_file():
                    continue
                try:
                    text = label.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                instances += sum(1 for line in text.splitlines() if len(line.split()) >= 5)
        stats = {"images": images, "instances": instances}
    except Exception as exc:  # noqa: BLE001 - 统计失败只是少一列，不该让实验失败
        log(f"val 集统计失败（过杀列将留空）：{exc}")
        return {}
    _VAL_STATS_CACHE[key] = stats
    return dict(stats)


def fp_per_image_from_pr(
    precision: float | None,
    recall: float | None,
    instances: int | None,
    images: int | None,
) -> float | None:
    """由 P / R / GT 数反推「每图过杀框数」。

    ⚠ 这是**近似值**：Ultralytics 的 P/R 取自 best-F1 conf，不是固定 conf，
    也不区分良品图。它的用途是在自动搜索里拦住「拿过杀换 mAP」的候选，
    不能当成产线验收数字——那个要用 ``tools/val_engine.py`` 的曲线。
    """
    if precision is None or recall is None or not instances or not images:
        return None
    if precision <= 0.0 or recall <= 0.0:
        return None
    tp = recall * float(instances)
    fp = tp * (1.0 - precision) / precision
    return round(fp / float(images), 6)


# ---------------------------------------------------------------------------
# split：确定性切分 + 硬链接沙箱
# ---------------------------------------------------------------------------


def list_images(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() in {".txt", ".yaml", ".yml"}:
        if target.suffix.lower() != ".txt":
            raise ValueError(f"暂不支持把 yaml 当作 split 输入：{target}")
        with target.open("r", encoding="utf-8-sig") as handle:
            entries = [line.strip() for line in handle if line.strip()]
        return [Path(item).resolve() for item in entries]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"图像来源既不是目录也不是 txt：{target}")


def label_for(image: Path) -> Path:
    """与 Ultralytics img2label_paths 同语义：把最后一段 /images/ 换成 /labels/。"""
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def mirror(paths: list[Path], dest_dir: Path, use_links: bool) -> str:
    """把图像（及对应标签）镜像到沙箱目录。优先硬链接，退化为复制。

    返回实际使用的方式："hardlink" 或 "copy"。混用会返回 "mixed"。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(str(dest_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
    label_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    for image in paths:
        target = dest_dir / image.name
        if target.exists() and target.stat().st_size == image.stat().st_size:
            used.add("skip")
            _mirror_one(label_for(image), label_dir / label_for(image).name, used, use_links)
            continue
        _mirror_one(image, target, used, use_links)
        _mirror_one(label_for(image), label_dir / label_for(image).name, used, use_links)

    if not used or used == {"skip"}:
        return "reuse"
    real = used - {"skip"}
    if real == {"hardlink"}:
        return "hardlink"
    if real == {"copy"}:
        return "copy"
    return "mixed"


def _mirror_one(source: Path, target: Path, used: set[str], use_links: bool) -> None:
    if target.exists():
        return
    if not source.is_file():
        return  # 标签缺失是合法的（空标注图），跳过即可
    if use_links:
        try:
            os.link(source, target)
            used.add("hardlink")
            return
        except OSError:
            use_links = False
    target.write_bytes(source.read_bytes())
    used.add("copy")


def write_split_yaml(path: Path, names: list[str], train_rel: str, val_rel: str, nc: int) -> None:
    """手写 YAML，避免引入 pyyaml 以便和 ultralytics 的解析行为完全一致。

    注意：Windows 路径必须写成正斜杠。双引号 YAML 标量里反斜杠是转义符，
    `"D:\\test"` 会被当成未知转义而报 syntax error。
    """
    sandbox = (SPLITS_DIR / "pcb").as_posix()
    lines = [
        f'path: "{sandbox}"',
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"nc: {nc}",
        "names:",
    ]
    lines += [f'  - "{name}"' for name in names]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml_minimal(path: Path) -> dict[str, Any]:
    """极简 YAML 读取（只处理本工程 data.yaml 的形态），避免额外依赖。"""
    result: dict[str, Any] = {}
    names: list[str] = []
    in_names = False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if in_names:
            stripped = line.strip()
            if stripped.startswith("- "):
                names.append(stripped[2:].strip().strip("\"'"))
                continue
            in_names = False
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "names":
            inline = value.strip()
            if inline.startswith("[") and inline.endswith("]"):
                inner = inline[1:-1].strip()
                names = [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
            else:
                in_names = True
            continue
        if key == "nc":
            result["nc"] = int(value)
        else:
            result[key] = value.strip("\"'")
    if names:
        result["names"] = names
    return result


def grouped_val_split(
    images: list[Path], holdout_frac: float, group_by: str
) -> tuple[list[Path], list[Path]] | None:
    """把 val 图按「同一底板」整组切 tune / holdout，保证任何一组不跨两侧。

    为什么不能沿用按图片洗牌：``splits/pcb`` 实测 10 个底板组全部同时出现在
    train / tune / holdout 里——holdout 的 42 张每一张都能在 tune 里找到同底片的
    兄弟。那样验收出来的分数衡量的是「记住了这几块板」，不是泛化。

    返回 ``None`` 表示组数少于 2、无法在不拆组的前提下分两侧，由调用方决定是否报错。
    """
    module = load_web_module("dataset")
    if module is None:
        return None
    keys = [module.group_key(image.name, group_by) for image in images]
    groups: dict[str, list[Path]] = {}
    for image, key in zip(images, keys):
        groups.setdefault(key, []).append(image)
    if len(groups) < 2:
        return None

    total = len(images)
    holdout_target = max(1, round(total * holdout_frac))
    targets = {"holdout": holdout_target, "tune": total - holdout_target}
    got = {"holdout": 0, "tune": 0}
    buckets: dict[str, list[Path]] = {"holdout": [], "tune": []}
    order = sorted(groups, key=lambda key: (module.fnv1a64(f"{key}{module.SALT}"), key))
    for key in order:
        chosen = max(("holdout", "tune"), key=lambda name: targets[name] - got[name])
        buckets[chosen].extend(groups[key])
        got[chosen] += len(groups[key])
    if not buckets["holdout"] or not buckets["tune"]:
        return None
    return (
        sorted(buckets["holdout"], key=lambda p: str(p).lower()),
        sorted(buckets["tune"], key=lambda p: str(p).lower()),
    )


def cmd_split(args: argparse.Namespace) -> int:
    source_yaml = Path(args.data).resolve()
    if not source_yaml.is_file():
        log(f"data.yaml 不存在：{source_yaml}")
        return 2

    meta = load_yaml_minimal(source_yaml)
    root = Path(meta.get("path") or source_yaml.parent)
    if not root.is_absolute():
        root = (source_yaml.parent / root).resolve()

    def resolve_entry(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    train_images = list_images(resolve_entry(meta["train"]))
    val_images = list_images(resolve_entry(meta["val"]))
    if not train_images or not val_images:
        log(f"切分失败：train={len(train_images)} val={len(val_images)}，不能为空")
        return 2

    names = meta.get("names") or [f"class_{i}" for i in range(meta.get("nc", 0))]
    nc = meta.get("nc") or len(names)

    # 确定性切分：排序后再用固定种子洗牌，保证同样的输入永远得到同样的两组
    ordered = sorted(val_images, key=lambda p: str(p).lower())
    group_by = getattr(args, "group_by", "none")
    if group_by and group_by != "none":
        grouped = grouped_val_split(ordered, args.holdout_frac, group_by)
        if grouped is None:
            log(
                f"按组切分失败：组数少于 2（{group_by}），无法在不拆组的前提下分两侧。"
                " 加 --group-by none 可退回按图片切分。"
            )
            return 2
        holdout, tune = grouped
        log(f"按组切分（{group_by}）：holdout {len(holdout)} / tune {len(tune)}，同一组不跨两侧")
    else:
        shuffled = list(ordered)
        random.Random(args.seed).shuffle(shuffled)
        holdout_count = max(1, round(len(shuffled) * args.holdout_frac))
        holdout = sorted(shuffled[:holdout_count], key=lambda p: str(p).lower())
        tune = sorted(shuffled[holdout_count:], key=lambda p: str(p).lower())

    manifest_path = SPLITS_DIR / "_manifest.json"
    signature = {
        "source_yaml": str(source_yaml),
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "group_by": group_by,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "holdout": [p.name for p in holdout],
        "tune": [p.name for p in tune],
    }
    if manifest_path.is_file() and not args.force:
        try:
            previous = load_json(manifest_path)
        except (OSError, ValueError):
            previous = {}
        if previous.get("signature") == signature:
            log(f"切分已存在且参数一致，跳过（--force 可强制重建）：tune={len(tune)} holdout={len(holdout)}")
            return 0

    sandbox = SPLITS_DIR / "pcb"
    assert_sandbox(sandbox, "split 沙箱")
    log(f"镜像训练集 {len(train_images)} 张 → {sandbox / 'images' / 'train'}")
    train_mode = mirror(train_images, sandbox / "images" / "train", use_links=not args.copy)
    log(f"镜像 tune {len(tune)} 张")
    tune_mode = mirror(tune, sandbox / "images" / "tune", use_links=not args.copy)
    log(f"镜像 holdout {len(holdout)} 张")
    holdout_mode = mirror(holdout, sandbox / "images" / "holdout", use_links=not args.copy)

    write_split_yaml(SPLITS_DIR / "pcb_tune.yaml", names, "images/train", "images/tune", nc)
    write_split_yaml(SPLITS_DIR / "pcb_holdout.yaml", names, "images/train", "images/holdout", nc)

    save_json(
        manifest_path,
        {
            "created_at": now_iso(),
            "signature": signature,
            "mode": {"train": train_mode, "tune": tune_mode, "holdout": holdout_mode},
            "counts": {"train": len(train_images), "tune": len(tune), "holdout": len(holdout)},
            "tune_yaml": str(SPLITS_DIR / "pcb_tune.yaml"),
            "holdout_yaml": str(SPLITS_DIR / "pcb_holdout.yaml"),
            # 切分文件本身的内容身份。signature 只记文件名与 seed——holdout 的图
            # 被改过标注后它完全不变，于是「验收通过」的结论会悄悄失效。
            "data_fingerprint": dataset_fingerprint(SPLITS_DIR / "pcb_tune.yaml"),
            # holdout 的图不在 tune 切分的 train/val 里，必须单独记一份，
            # 否则改了 holdout 标注也发现不了。
            "holdout_fingerprint": dataset_fingerprint(SPLITS_DIR / "pcb_holdout.yaml"),
        },
    )
    log(f"完成：train={len(train_images)} tune={len(tune)} holdout={len(holdout)}（{train_mode}/{tune_mode}/{holdout_mode}）")
    return 0


# ---------------------------------------------------------------------------
# run / _exec
# ---------------------------------------------------------------------------


#: 从 6-web/backend 加载的模块缓存（同一个名字只导入一次）。
#: 存 None 也是有效缓存：表示「查过了，没有」，避免每轮实验都去摸磁盘。
_WEB_MODULES: dict[str, Any] = {}


def load_web_module(name: str) -> Any:
    """从 ``6-web/backend`` 加载一个模块（gpulock / vram / ampcheck）。

    为什么复用 Web 那一份而不是自己再写一遍：锁文件的格式、AMP 黑名单、显存余量
    都是两边共用的契约，两份实现迟早会漂移（一边加字段、一边改判定），而漂移的表现
    是「页面说能跑，autoexp 说不能跑」——比任何单边错误都难查。

    为什么放在 6-web/backend：Web 打包脚本会把整个 backend 目录一起冻结，
    放在那里才不会漏文件。取不到就返回 None，由调用方降级，不要让一个
    「更好的估算」成为训练跑不起来的原因。
    """
    if name in _WEB_MODULES:
        return _WEB_MODULES[name]
    module: Any = None
    backend = ROOT.parent / "6-web" / "backend"
    if (backend / f"{name}.py").is_file():
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 缺依赖/写坏了都当作「没有这个模块」
            module = None
    _WEB_MODULES[name] = module
    return module


def load_gpulock() -> Any:
    """加载 6-web 的 ``gpulock``（零第三方依赖，只用标准库）。"""
    return load_web_module("gpulock")


def load_vram() -> Any:
    """加载 6-web 的 ``vram``：显存预检与实测峰值表（两边唯一的实现）。"""
    return load_web_module("vram")


def load_fingerprint() -> Any:
    """加载 6-web 的 ``fingerprint``：数据集指纹（两边唯一的实现）。

    找不到时返回 ``None``，调用方把 ``dataset`` 块标成 ``unavailable``。
    指纹是附加信息，不该因为少了一个文件就把一整轮实验拦下来。
    """
    return load_web_module("fingerprint")


def dataset_fingerprint(data: str | Path) -> dict[str, Any]:
    """算数据集指纹的精简块，写进 ``exp_NNNN.json`` / ``results.tsv`` / ``best.json``。

    与 Web 训练页用同一份实现，四处记录的指纹才能互相对上。任何失败都收敛成
    ``unavailable`` 或 ``error`` 字段，不抛异常。
    """
    module = load_fingerprint()
    if module is None:
        return {"algo": "", "unavailable": "找不到 6-web/backend/fingerprint.py"}
    raw = str(data or "").strip()
    if not raw:
        return {"algo": "", "unavailable": "配置里没有 data 字段"}
    target = Path(raw).expanduser()
    if not target.is_absolute():
        # autoexp 的配置里写的是相对 7-autoresearch-yolo 的路径（splits/xxx.yaml）。
        # 用 __file__ 而不是 cwd：从 Web 后端拉起时 cwd 是仓库根，两者不一致。
        target = Path(__file__).resolve().parent / target
    try:
        report = module.compute(target)
    except Exception as exc:  # noqa: BLE001 - 指纹失败不阻断实验
        return {"algo": "", "unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "algo": report.get("algo"),
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "short": report.get("short") or "",
        "label_fingerprint": report.get("label_fingerprint"),
        "split_fingerprints": report.get("split_fingerprints"),
        "counts": report.get("counts"),
        "error": report.get("error", ""),
    }


def qc_gate(cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    """开跑前的数据门禁：只看 error 级问题（字段数不对、类别越界、坐标越界等）。

    为什么默认关闭：体检要解码整份数据（692 张约 1~10 秒，取决于缓存），而自动实验
    一轮只跑几分钟。默认打开会让每轮多花一段时间，用户明确要检查时再加 ``--qc-gate``。

    ``warn`` 级（极小框、贴边、跨划分）一律不拦——它们大多是「值得看一眼」，
    而不是「数据已经坏了」。
    """
    module = load_web_module("datasetqc")
    if module is None:
        return True, []
    raw = str(cfg.get("data") or "").strip()
    if not raw:
        return True, []
    target = Path(raw).expanduser()
    if not target.is_absolute():
        # 配置里的 data 是相对 7-autoresearch-yolo 的路径（splits/xxx.yaml）
        target = Path(__file__).resolve().parent / target
    try:
        report = module.inspect(target, "auto")
    except Exception as exc:  # noqa: BLE001 - 体检失败不该变成训练跑不起来
        return True, [f"数据体检失败，已跳过门禁：{type(exc).__name__}: {exc}"]

    summary = report.get("summary") or {}
    groups = report.get("groups") or {}
    lines = [
        "数据体检：错误 {} / 警告 {}，底板组 {}（跨划分 {}）".format(
            summary.get("error", 0),
            summary.get("warn", 0),
            groups.get("total", 0),
            groups.get("cross_split", 0),
        )
    ]
    if groups.get("cross_split"):
        lines.append("注意：同一底板跨划分，本次指标不能当作泛化能力。")
    if not summary.get("error"):
        return True, lines

    shown = 0
    for issue in report.get("issues") or []:
        if issue.get("severity") != "error" or shown >= 10:
            continue
        where = issue.get("rel") or issue.get("split") or "全局"
        if issue.get("line"):
            where = f"{where}:{issue['line']}"
        lines.append(f"  [{issue.get('code')}] {where}  {issue.get('detail')}")
        shown += 1
    return False, lines


def check_split_drift(yaml_path: Path) -> bool:
    """holdout 验收前核对数据是否变过。

    ``splits/_manifest.json`` 里的 ``signature`` 只记文件名与 seed：把某张 holdout
    图改个标注、换个同名文件，signature 一点都不会变，于是「验收通过」的结论
    悄悄失效。指纹补上了这一环。

    返回 True 表示一致或无法判断（升级前生成的切分没有指纹），False 表示确实变了。
    """
    manifest = SPLITS_DIR / "_manifest.json"
    if not manifest.is_file():
        return True
    try:
        payload = load_json(manifest) or {}
    except (OSError, ValueError):
        return True
    if not isinstance(payload, dict):
        return True

    # holdout 的图不在 pcb_tune 的 train/val 里，所以优先比 holdout 的那份指纹
    recorded = payload.get("holdout_fingerprint") or {}
    expected = recorded.get("dataset_fingerprint") if isinstance(recorded, dict) else None
    target = SPLITS_DIR / "pcb_holdout.yaml"
    if not expected:
        recorded = payload.get("data_fingerprint") or {}
        expected = recorded.get("dataset_fingerprint") if isinstance(recorded, dict) else None
        target = SPLITS_DIR / "pcb_tune.yaml"
    if not expected or not target.is_file():
        return True

    actual = dataset_fingerprint(target).get("dataset_fingerprint")
    if actual and actual != expected:
        log(
            f"警告：holdout 数据已变化（切分时 {str(expected)[:8]}，当前 {str(actual)[:8]}）。"
            "之前的 holdout 验收结论不可沿用，请重新执行 split 与验收。"
        )
        return False
    return True


def derive_result_status(result: dict[str, Any], returncode: int) -> str:
    """由 ``result.json`` 与子进程退出码决定这一轮的状态。

    抽成模块级函数是为了能被测试直接调用，也为了逼着调用方只用 ``returncode``：
    ``--events`` 走 tee 分支时没有 ``Popen`` 对象，早先那版直接用 ``proc.returncode``
    会在网页端每一轮训练结束时抛 ``NameError``，表现为「训练明明跑完了，台账却没有记录」。
    """
    raw = result.get("status") or ("crash" if int(returncode) != 0 else "ok")
    return "done" if raw == "ok" else str(raw)


#: Ultralytics 遇到首轮 OOM 会打印这句并把 batch 减半。
#: 一旦出现，这次实验的等效学习率与正则强度都变了，与台账里其它实验不可比。
BATCH_DOWNGRADE_PATTERN = re.compile(r"Reducing to batch=\s*(\d+)", re.IGNORECASE)


def _exec_tee(cmd: list[str], log_path: Path, exp_id: str, created_at: str,
              downgrade_flag: Path | None = None) -> int:
    """跑 ``_exec`` 子进程，并把它的输出**同时**写进日志文件和 stdout。

    agent 模式专用。需求是两头都要：
    - 后端要从 stdout 拿到 launcher 的 ``TRTFORGE_JSON`` 事件，网页日志栏才能
      显示 epoch 进度和验证指标；
    - 同时必须留一份完整日志在 ``runs/autoexp/<exp>/.autoexp.log``，否则训练
      报错时没法事后复盘（后端那 3000 行前端缓冲不是归档）。

    只逐行转发、**不做解析**：后端 ``runners._parse_line()`` 已经会区分
    JSON 事件和普通日志。在这里再判一次就是把同一套分流规则写两遍，
    以后改一处忘另一处，界面就会出现错位。

    唯一的例外是 ``BATCH_DOWNGRADE_PATTERN``：那是**必须立刻拦下**的事件。
    Ultralytics 首轮 OOM 会把 batch 减半重来，batch 一变，nbs 归一化后的等效
    学习率与正则强度全跟着变，这次实验就和台账里其它实验不是同一件事了。
    以前只靠事后比较 ``batch_requested`` 与 ``batch_effective``，等发现时
    已经白跑完整轮 600 秒；现在命中即写标记、终止子进程，省下这 10 分钟。

    返回子进程退出码，与 ``subprocess.run(...).returncode`` 语义一致；
    被本函数主动终止时返回 ``EXIT_BATCH_DOWNGRADED``。
    """
    process: subprocess.Popen[bytes] | None = None
    returncode = 1
    downgraded_to: int | None = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# {exp_id} {created_at}\n")
        log_handle.write(f"# cmd: {' '.join(cmd)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        # 记下真正的训练进程号。Web 端「停止」要杀整棵树：
        # Windows 上杀父进程不会连带子进程，没有这个 pid 就只能等它自己跑完。
        _write_child_pid(process.pid)
        try:
            assert process.stdout is not None
            for raw in iter(process.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                log_handle.write(line)
                log_handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                if downgraded_to is None:
                    hit = BATCH_DOWNGRADE_PATTERN.search(line)
                    if hit:
                        downgraded_to = int(hit.group(1))
                        message = (
                            f"{exp_id} 检测到 Ultralytics 把 batch 降到 {downgraded_to}："
                            "这次实验的等效学习率与正则强度都变了，与台账其它实验不可比，"
                            "已立即中止（不写指标，不计入 crash）"
                        )
                        log(message)
                        for handle in (log_handle, sys.stdout):
                            handle.write(message + "\n")
                            handle.flush()
                        if downgrade_flag is not None:
                            try:
                                downgrade_flag.parent.mkdir(parents=True, exist_ok=True)
                                downgrade_flag.write_text(
                                    json.dumps({"exp_id": exp_id, "batch_effective": downgraded_to,
                                                "line": line.strip()[:400]},
                                               ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                            except OSError:
                                pass
                        with contextlib.suppress(Exception):
                            process.terminate()
                        break
            returncode = process.wait()
        except BaseException:
            # 被 Ctrl+C 或父进程终止：先把子进程收掉，再让异常继续往上抛。
            # 不收的话训练会变成孤儿进程，继续占着显卡，而界面上显示「已停止」。
            with contextlib.suppress(Exception):
                process.terminate()
                process.wait(timeout=10)
            raise
        finally:
            _clear_child_pid()
    if downgraded_to is not None:
        return EXIT_BATCH_DOWNGRADED
    return returncode


def _write_child_pid(pid: int) -> None:
    """记录当前训练子进程号，供 Web 端「停止」时杀整棵树。"""
    try:
        AGENT_DIR.mkdir(parents=True, exist_ok=True)
        (AGENT_DIR / "child.pid").write_text(str(pid), encoding="utf-8")
    except OSError:
        pass


def _clear_child_pid() -> None:
    try:
        (AGENT_DIR / "child.pid").unlink(missing_ok=True)
    except OSError:
        pass


#: 父进程已经让推理进程退出时，子进程会继承这个环境变量。
#: 子进程看到它就不再通知网页，也不会在每一轮训练结束时把显存让出提前还回去。
INFER_YIELDED_ENV = "AUTOEXP_INFER_YIELDED"


def _infer_web_bases() -> list[str]:
    """网页服务地址。命令行跑实验时靠它通知正在运行的 6-web 退出推理进程。"""
    explicit = os.environ.get("AUTOEXP_WEB_URL", "").strip().rstrip("/")
    if explicit:
        return [explicit]
    return ["http://127.0.0.1:8060"]


def _post_infer_control(path: str, reason: str = "") -> dict[str, Any] | None:
    """通知网页端让出或归还推理显存。网页没开时返回 None，不当成训练失败。"""
    import urllib.parse
    import urllib.request

    last_error = "没有可用地址"
    for base in _infer_web_bases():
        url = base + path
        if reason:
            url += "?reason=" + urllib.parse.quote(reason)
        request = urllib.request.Request(
            url,
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                body = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - 网页没开是正常情况
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {"ok": True}
        return data if isinstance(data, dict) else {"ok": True}
    log(f"通知网页推理进程失败（{path}）：{last_error}")
    return None


def acquire_infer_yield(reason: str) -> bool:
    """让网页端退出 infer_worker。

    返回 True 表示**这一层**新拿到了所有权，结束时必须调用 ``release_infer_yield(True)``。
    父进程（或网页后端）已经让出时返回 False：嵌套的每一轮训练不能提前把显存还回去，
    否则下一轮一开始推理进程又被页面点醒，8GB 卡会再次 OOM。
    """
    if os.environ.get(INFER_YIELDED_ENV) == "1":
        return False
    result = _post_infer_control("/api/infer/gpu-yield", reason)
    if result is None:
        log("网页端没有响应，推理进程可能仍占着显存。若训练一开始就 OOM，先在网页上停止推理。")
        return False
    os.environ[INFER_YIELDED_ENV] = "1"
    log(
        f"已让推理进程退出（{result.get('stopped', '?')} 个），显存留给训练。"
        "全部训练结束后不会立刻重载，下次在网页上推理时才重新加载。"
    )
    return True


def release_infer_yield(owned: bool) -> None:
    """只有拿到所有权的那一层才能解除。解除后进程保持退出，等下次推理再加载。"""
    if not owned:
        return
    _post_infer_control("/api/infer/gpu-release")
    os.environ.pop(INFER_YIELDED_ENV, None)
    log("训练结束，推理进程保持退出。下次在网页上推理时会自动重新加载。")


def cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    override = load_json(config_path)
    base = load_json(BASE_JSON) if BASE_JSON.is_file() else {}
    cfg = deep_merge(base, override)

    if SPACE_JSON.is_file() and not args.no_validate:
        errors = validate_against_space(cfg, load_json(SPACE_JSON))
        if errors:
            for item in errors:
                log(f"校验失败：{item}")
            log("如确认要越界，加 --no-validate 重跑。")
            return 2

    # 跨进程 GPU 锁：网页端可能正在训练，或者还起着另一套 Web 实例。
    # 那种情况下开跑必然抢显存，而 Ultralytics 会静默减半 batch 重试，
    # 于是这次实验与台账里其它实验不再是同一件事。宁可拒绝，也不要产出不可比的数。
    #
    # 位置刻意放在分配 exp_id 之前：被拒时不该留下 experiments/exp_NNNN.json
    # 和空的 runs 目录，否则台账会多出一堆「从没跑过」的编号，看的人会以为丢数据了。
    lock = load_gpulock()
    lock_held = False
    # 锁委派：Web 后端为一个独占任务取锁时（owner=web），子进程会看到
    # 「pid 是别人且存活」而直接退出码 5，一轮都跑不了。
    #
    # 这里**只跳过「父进程自己持有的那把锁」**，不是无条件跳过：
    # 后端把自己的 pid 通过 AUTOEXP_GPU_LOCK_PID 传进来，只有持有者 pid
    # 与之相等才认为「这把锁是我家父进程的」。
    # 为什么不能只用一个布尔开关：那样连「命令行正另跑着一轮实验」也看不见，
    # 结果就是两个训练一起抢显存 —— 正是这把锁要防的事。
    parent_pid = os.environ.get("AUTOEXP_GPU_LOCK_PID", "").strip()
    delegated = False
    if lock is not None and parent_pid.isdigit():
        holder = lock.read()
        if holder is not None and int(holder.get("pid") or 0) == int(parent_pid):
            lock = None
            delegated = True
    if lock is not None:
        holder = lock.read()
        if holder is not None and int(holder.get("pid") or 0) != os.getpid():
            log(f"显卡正被占用：{lock.describe(holder)}")
            log("本次不启动。等它结束，或到那个页面/终端里停止后重试。")
            return 5
        lock_held, lock_reason = lock.acquire("autoexp", "autoexp-run", args.exp_id or "")
        if lock_held and lock_reason:
            log(f"注意：跨进程互斥未生效——{lock_reason}")
    elif delegated:
        # 两种情况都会走到「lock 为空」，但含义完全相反：
        # 一种是真找不到 gpulock.py（于是确实没有互斥），
        # 另一种是网页后端已经替我们持锁（互斥是生效的，只是持有者不是我）。
        # 混成一句「未找到 gpulock.py」会让人以为互斥失效，去排查一个不存在的故障。
        log("显卡锁由网页后端持有，本轮不再重复加锁")
    else:
        log("注意：未找到 6-web/backend/gpulock.py，本次不做跨进程互斥")

    # 锁到手之后、真正开训之前退出推理进程。放在预检前面：
    # 预检读的是「现在还剩多少显存」，推理进程还活着时这个数是假的。
    owned_yield = acquire_infer_yield("自动实验训练")
    try:
        return _run_locked(args, cfg, config_path, lock, lock_held)
    finally:
        # 训练子进程已退出（或被打断），立刻让出 GPU 位置。
        # 放在 finally 里：按 Ctrl+C 中途退出也要释放，否则锁会挂到进程死亡才清。
        if lock is not None and lock_held:
            lock.release()
        # 嵌套在 agent / noise 里时 owned_yield 为 False，不会把父进程的让出提前还回去。
        release_infer_yield(owned_yield)


def _run_locked(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    lock: Any,
    lock_held: bool,
) -> int:
    """``cmd_run`` 的主体。拆出来是为了让锁的获取/释放在一个地方看全。"""
    exp_id = args.exp_id or next_exp_id()
    project = assert_sandbox(RUNS_DIR, "project")
    exp_dir = assert_sandbox(project / exp_id, "exp_dir")

    # 拿到锁之后再补一次 label：上面的 acquire 在 exp_id 还没分配时只能用参数值，
    # 而锁里写明「哪一次实验」对排障很关键（看到 GPU 被占时能直接找到对应 run）。
    if lock is not None and lock_held:
        lock.acquire("autoexp", "autoexp-run", exp_id)

    # 开跑前体检。宁可早退，也不要跑一半因内存/显存不足产出假数据。
    try:
        resources = preflight(strict=not getattr(args, "allow_low_resources", False))
    except RuntimeError as exc:
        log(str(exc))
        return 4

    # 数据门禁：放在写 exp_NNNN.json 之前，被拦下时台账里不留「从没跑过」的编号。
    if getattr(args, "qc_gate", False):
        passed, lines = qc_gate(cfg)
        for line in lines:
            log(line)
        if not passed:
            log("数据存在错误级标注问题，本次不启动。修完再跑，或去掉 --qc-gate 跳过检查。")
            return 6

    snapshot = {
        "exp_id": exp_id,
        "created_at": now_iso(),
        "config_source": str(config_path),
        "hypothesis": cfg.get("hypothesis", ""),
        "changed": cfg.get("changed", ""),
        "budget_seconds": int(cfg.get("budget_seconds", 600)),
        "resources_at_start": resources,
        # 数据身份：开跑前算一次。跑完再算会混进「这期间源数据被改过」的情况，
        # 那样记录的就不是「这一轮实际训练用的那批图」。
        "dataset": dataset_fingerprint(cfg.get("data", "")),
        "resolved": cfg,
    }
    snapshot_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    save_json(snapshot_path, snapshot)

    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "autoexp.log"
    log(f"{exp_id} 启动：budget={snapshot['budget_seconds']}s 产物={exp_dir}")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_exec",
        "--snapshot",
        str(snapshot_path),
        "--out",
        str(exp_dir),
    ]
    started = time.time()
    # 两种模式都走 tee。以前只有 --events（网页端）走 tee，于是命令行 `run` 与
    # `noise` 完全绕过了「降 batch 就中止」的拦截——2026-11-10 测 960 噪声时
    # 亲眼看到日志里出现 `Reducing to batch=2` 而程序毫无反应，就是这个原因。
    # 拦截必须是**确定**的，不能取决于调用方有没有传 --events。
    returncode = _exec_tee(
        cmd, log_path, exp_id, snapshot["created_at"],
        downgrade_flag=exp_dir / "batch_downgraded.json",
    )
    elapsed = round(time.time() - started, 1)

    result_path = exp_dir / "result.json"
    result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            result = load_json(result_path)
        except (OSError, ValueError):
            result = {}

    status = derive_result_status(result, returncode)
    if not result:
        result = {"status": "crash", "error": f"子进程退出码 {returncode}，且未生成 result.json"}

    # batch 降级是**主动中止**，不是训练崩溃：训练本身没出错，是我们发现
    # 条件已经变了于是叫停。所以单独一个状态，既不计 crash 也不写 fitness。
    downgrade_note: dict[str, Any] = {}
    downgrade_file = exp_dir / "batch_downgraded.json"
    if downgrade_file.is_file():
        try:
            downgrade_note = load_json(downgrade_file)
        except (OSError, ValueError):
            downgrade_note = {}
    if returncode == EXIT_BATCH_DOWNGRADED or downgrade_note:
        status = "batch_downgraded"
        result = {
            "status": status,
            "error": (
                "BATCH_DOWNGRADED: 首轮 OOM 触发 Ultralytics 自动降 batch"
                f"（→ {downgrade_note.get('batch_effective', '?')}）。"
                "等效学习率与正则强度随之改变，本次结果与台账其它实验不可比，已作废"
            ),
            **{k: v for k, v in result.items() if k in ("exp_id", "elapsed_seconds")},
        }

    # 过杀率（近似）：Ultralytics 的 P/R 是 best-F1 conf 下的值，配合 val 集的
    # GT 数与图片数可反推「每图误报几个框」。拿不到统计就留空，绝不填 0。
    val_stats = val_gt_stats(str(cfg.get("data") or "")) if status == "done" else {}
    fp_per_image = fp_per_image_from_pr(
        _as_float(result.get("precision")),
        _as_float(result.get("recall")),
        val_stats.get("instances"),
        val_stats.get("images"),
    )

    row = {
        "exp_id": exp_id,
        "created_at": snapshot["created_at"],
        # 适应度取 results.csv 最后一轮，不是历史最佳：固定预算下比的就是到点那一刻的水平
        # batch 降级一律留空：它已经不是同一套参数，写进去会污染「当前最优」。
        "fitness_map50_95": "" if status == "batch_downgraded" else result.get("map50_95", ""),
        "best_map50_95": result.get("best_map50_95", ""),
        "map50": result.get("map50", ""),
        "precision": result.get("precision", ""),
        "recall": result.get("recall", ""),
        "fp_per_image": "" if fp_per_image is None else fp_per_image,
        "epochs_done": result.get("epochs_done", ""),
        "train_seconds": result.get("train_seconds", elapsed),
        "peak_vram_mb": result.get("peak_vram_mb", ""),
        "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb", ""),
        "imgsz": (cfg.get("train") or {}).get("imgsz", cfg.get("imgsz", "")),
        "batch_req": result.get("batch_requested", (cfg.get("train") or {}).get("batch", "")),
        "batch_eff": result.get("batch_effective", ""),
        # 实际生效的精度（fp32/fp16/unknown）。黑名单卡永远是 fp32——
        # 这一列存在的唯一目的就是让「这轮为什么慢」变得可回答。
        "amp": result.get("precision", ""),
        # 数据指纹短码。跨这一列不同的行不能直接横比：参数没变，数据变了。
        "data_fp": (snapshot.get("dataset") or {}).get("short", ""),
        "model": cfg.get("model", ""),
        "budget_seconds": snapshot["budget_seconds"],
        "status": status,
        "hypothesis": cfg.get("hypothesis", ""),
        "changed": cfg.get("changed", ""),
    }
    append_tsv(RESULTS_TSV, RESULTS_HEADER, row)

    # 把「跑完才知道」的事实回写进实验记录（args.yaml 里只有请求值）。
    annotate_exp_record(
        exp_id,
        {
            "amp_effective": result.get("amp_effective", ""),
            "precision": result.get("precision", ""),
            "peak_vram_reserved_mb": result.get("peak_vram_reserved_mb"),
        },
    )

    # batch 被 Ultralytics 自动降过，说明这次实验与请求的参数不是同一件事，
    # 后续和其它实验横向比较会失真，必须显式提示。
    # 这是**兜底**：正常路径上 _exec_tee 已经中止并把状态改成了 batch_downgraded；
    # 这里覆盖非 tee 分支（命令行直接 run）以及「降级信息只体现在 batch_effective」的情况。
    if status == "done":
        req, eff = str(row["batch_req"]), str(row["batch_eff"])
        if req and eff and req != eff:
            log(f"注意：batch 被自动从 {req} 降到 {eff}（首轮 OOM）。本次结果与请求参数不等价，跨实验比较需谨慎。")

    fitness = row["fitness_map50_95"]
    log(f"{exp_id} 结束：status={status} mAP50-95={fitness} epochs={row['epochs_done']} 用时={elapsed}s")
    if status == "crash":
        log(f"失败详情见 {log_path}")
        return 1
    if status == "batch_downgraded":
        log(f"已作废：{result.get('error', '')}")
        log("处理：把 batch 或 imgsz 调低到显存装得下的组合，再重跑这一个方向。")
        return 1
    if status == "preflight_blocked":
        log(f"预检拦截，未消耗训练时间：{result.get('error', '')}")
        return 1
    return 0


def free_vram_mb() -> float | None:
    """读当前可用显存（MB）。读不到返回 None，由调用方决定是否放行。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return None


def read_trainer_amp(trainer: Any) -> bool | None:
    """读 trainer 上**真正生效**的混合精度标志。

    必须是 ``trainer.amp`` 而不是 ``args.amp``：黑名单显卡（T1000 等）上
    ``args.amp`` 仍是 True，但 ``Trainer.__init__`` 里的 ``check_amp()`` 已经把它
    关掉了。args 记录的是「我想要什么」，trainer.amp 记录的是「实际用了什么」。
    """
    value = getattr(trainer, "amp", None)
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)
    except (TypeError, ValueError):
        return None


def preflight_vram(
    imgsz: int,
    batch: int,
    need_mb: float | None = None,
    precision: str | None = None,
) -> tuple[bool, str]:
    """开训前的显存预检。

    Ultralytics 的首轮 OOM 会自动把 batch 减半重来，这会**悄悄改变实验条件**：
    batch 一变，nbs 归一化后的等效学习率、正则强度全跟着变，于是这一次实验
    与台账里其它实验不再可比。所以宁可提前拒绝，也不要让它自动降级。

    判定本身**不在这里**：委托给 ``6-web/backend/vram.py``，与 Web 训练页共用一份
    实现（实测峰值表 + 1.15 倍余量 + 400MB 预留，无样本才回落到公式）。
    历史上这里有一条写死的公式 ``2600 + 240*batch*(imgsz/640)**2``，已经因为
    「960/batch=4 估算放行但实际 OOM」被废弃——两处各写一份必然出现这种矛盾。

    ``need_mb`` 显式给出时按旧语义直接比（测试与「已知需求」的场景用）。
    ``precision`` 为 None 时由 vram 按本机显卡判断（黑名单卡的 fp32 与 fp16 差近一倍）。
    """
    free = free_vram_mb()
    if free is None:
        return True, "无法读取显存（可能无 CUDA），跳过预检"

    if need_mb is not None:
        if free < need_mb:
            return False, (
                f"可用显存 {free:.0f} MB < 保守需求 {need_mb:.0f} MB"
                f"（imgsz={imgsz} batch={batch}）。"
                "请先释放显存（例如停掉 6-web 的 infer_worker 常驻进程），或把 batch 调小。"
            )
        return True, f"可用显存 {free:.0f} MB ≥ 保守需求 {need_mb:.0f} MB"

    module = load_vram()
    if module is None:
        # 降级：拿不到共享实现时用同一套公式手算，宁可保守也不要因为缺文件而放行
        est = 2600.0 + 300.0 * max(int(batch), 1) * (int(imgsz) / 640.0) ** 2
        if free < est:
            return False, (
                f"可用显存 {free:.0f} MB < 保守需求 {est:.0f} MB"
                f"（imgsz={imgsz} batch={batch}，未找到 vram.py，按公式估算）。"
                "请先释放显存或把 batch 调小。"
            )
        return True, f"可用显存 {free:.0f} MB ≥ 保守需求 {est:.0f} MB（按公式估算）"

    report = module.preflight(imgsz, max(int(batch), 1), precision=precision, free_mb=free)
    return bool(report.get("ok")), str(report.get("reason") or "")


def cmd_exec(args: argparse.Namespace) -> int:
    """子进程侧：真正跑一次训练。任何异常都写 result.json 后退出，不吞错误。"""
    snapshot_path = Path(args.snapshot).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    snapshot = load_json(snapshot_path)
    cfg = snapshot["resolved"]
    train_cfg: dict[str, Any] = dict(cfg.get("train") or {})
    budget_seconds = int(snapshot.get("budget_seconds", 600))
    exp_id = snapshot["exp_id"]

    os.chdir(ROOT)
    started = time.time()
    try:
        launcher = load_launcher()
        # 产物目录由驱动器统一决定，配置里不接受 project，避免写飞
        argv = build_argv(cfg, train_cfg, project=RUNS_DIR, name=exp_id)
        namespace = launcher.build_parser().parse_args(argv)
        kwargs = launcher.build_train_kwargs(namespace)

        # 预算与可复现性：这些键不在 launcher 的 CLI 白名单里，直接注入
        kwargs["time"] = budget_seconds / 3600.0  # Ultralytics 的 time 单位是小时
        kwargs["seed"] = int(train_cfg.get("seed", 0))
        kwargs["deterministic"] = bool(train_cfg.get("deterministic", True))
        kwargs["patience"] = int(train_cfg.get("patience", 0))
        kwargs["plots"] = False           # 搜索期不画图，省时间
        kwargs["verbose"] = False
        kwargs["exist_ok"] = True
        # cache 默认 "ram"：553 张图约 750 MB，一次性读进内存，
        # 换来 workers=0 时数据加载不再成为瓶颈（Windows 上派生 worker 会耗尽提交内存）。
        kwargs["cache"] = train_cfg.get("cache", "ram")
        for key in DIRECT_TRAIN_KEYS:
            if key in train_cfg and key not in {"time"}:
                kwargs[key] = train_cfg[key]

        weights = resolve_weights(str(cfg.get("model", "yolov8n.pt")))
        log(f"{exp_id} 权重={weights}")
        log(f"{exp_id} kwargs={json.dumps(kwargs, ensure_ascii=False, default=str)}")

        # 显存预检：宁可提前失败，也不要让 Ultralytics 静默降 batch 改变实验条件
        batch_int = int(kwargs.get("batch", 16)) if str(kwargs.get("batch", "16")).lstrip("-").isdigit() else 8
        imgsz_int = int(kwargs.get("imgsz", 640))
        # 只有「显式关闭混合精度」是确定事实，其余交给 vram 按本机显卡判断：
        # 黑名单卡（T1000/T1200/1660 等）上必然是 fp32，显存需求与 fp16 差近一倍，
        # 用错精度去查实测表等于没查。
        amp_requested = kwargs.get("amp", True)
        precision_hint = "fp32" if amp_requested is False else None
        ok, reason = preflight_vram(imgsz_int, max(batch_int, 1), precision=precision_hint)
        log(f"{exp_id} 显存预检：{reason}")
        if not ok:
            payload = {
                "status": "preflight_blocked",
                "exp_id": exp_id,
                "error": f"PREFLIGHT_VRAM: {reason}",
                "batch_requested": str(kwargs.get("batch", "")),
                "elapsed_seconds": round(time.time() - started, 1),
            }
            save_json(result_path, payload)
            log(f"{exp_id} 预检未通过，已中止（不消耗训练时间，也不计入 crash）")
            return 1

        model = launcher.load_yolo(weights)
        # 注册 launcher 自己的训练回调。
        #
        # 为什么必须有这一步：`load_yolo()` 只发 deps / model 两条 phase 事件，
        # 真正的 `progress`（epoch/total/pct）与 `epoch_summary`（验证指标）
        # 是由回调发出来的，而回调只在 launcher 的 `run()` 里注册。
        # 少了这一步，autoexp 跑起来界面上只有 Ultralytics 的原始文本流，
        # 一行 epoch 进度都看不到——看起来像卡死，实际在正常训练。
        #
        # 用 suppress 包住：万一上游改了回调签名，也只是「少几条进度事件」，
        # 不该让整轮实验因为一个观测功能而失败。
        with contextlib.suppress(Exception):
            launcher.add_train_callbacks(model, namespace)

        model.train(**kwargs)

        save_dir = Path(getattr(model.trainer, "save_dir", out_dir))
        metrics = read_results_csv(save_dir / "results.csv")
        # 实际生效的 batch：与请求值不同即说明被自动降级过
        batch_effective = None
        try:
            batch_effective = int(getattr(model.trainer, "batch_size", 0) or 0) or None
        except (TypeError, ValueError):
            batch_effective = None

        # 峰值显存与**实际**生效的混合精度。
        #
        # peak_vram_mb 保持旧语义（max_memory_allocated），历史行仍可比；
        # peak_vram_reserved_mb 是新口径：OOM 发生在向驱动申请那一刻，预留量
        # 才是真正占住的显存，实测比 allocated 高 30%~50%。显存实测表用 reserved。
        peak_vram = None
        peak_vram_reserved = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                peak_vram_reserved = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
        except Exception:  # noqa: BLE001 - 显存读数失败不影响结论
            peak_vram = None
            peak_vram_reserved = None

        amp_effective = read_trainer_amp(model.trainer)
        precision_actual = "unknown" if amp_effective is None else ("fp16" if amp_effective else "fp32")
        # 自校准：把这次实测写进峰值表。用**实际生效**的 batch 与精度做键——
        # 降级过的 batch 如果按请求值记账，下次预检会算出偏小的需求。
        sample_batch = batch_effective or max(batch_int, 1)
        if peak_vram_reserved:
            module = load_vram()
            if module is not None:
                entry = module.record_sample(
                    imgsz=imgsz_int,
                    batch=sample_batch,
                    peak_mb=peak_vram_reserved,
                    peak_allocated_mb=peak_vram,
                    precision=precision_actual,
                    exp_id=exp_id,
                )
                if entry.get("skipped"):
                    log(f"{exp_id} 显存样本未记录：{entry['skipped']}")
                else:
                    log(
                        f"{exp_id} 显存样本已记录：{imgsz_int}x{sample_batch}x{precision_actual} "
                        f"峰值 {entry.get('peak_mb')} MB（累计 {entry.get('count')} 次）"
                    )
        else:
            log(f"{exp_id} 未取到峰值显存，本次不更新实测表")

        log(f"{exp_id} 实际精度：{precision_actual}（请求 amp={amp_requested}）")

        payload = {
            "status": "done",
            "exp_id": exp_id,
            "save_dir": str(save_dir),
            "weights": str(save_dir / "weights" / "best.pt"),
            "peak_vram_mb": peak_vram,
            "peak_vram_reserved_mb": peak_vram_reserved,
            "amp_requested": amp_requested if isinstance(amp_requested, bool) else str(amp_requested),
            "amp_effective": "" if amp_effective is None else amp_effective,
            "precision": precision_actual,
            "batch_requested": str(kwargs.get("batch", "")),
            "batch_effective": "" if batch_effective is None else str(batch_effective),
            "elapsed_seconds": round(time.time() - started, 1),
            **metrics,
        }
        save_json(result_path, payload)
        log(f"{exp_id} 完成，指标已写入 {result_path}")
        return 0
    except BaseException as exc:  # noqa: BLE001 - 必须记录后退出，不能静默
        payload = {
            "status": "crash",
            "exp_id": exp_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        save_json(result_path, payload)
        log(f"{exp_id} 崩溃：{type(exc).__name__}: {exc}")
        return 1


# ---------------------------------------------------------------------------
# holdout / noise / report
# ---------------------------------------------------------------------------


def _metric_from_results_dict(results_dict: dict[str, Any], prefix: str) -> float | None:
    for name, value in results_dict.items():
        if name.startswith(prefix + "(") or name == prefix:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def read_run_args_yaml(weights: Path) -> dict[str, Any]:
    """从 `runs/autoexp/<exp>/args.yaml` 里读回该次训练实际用的关键参数。

    用途：holdout 验收时必须以训练时的分辨率评估，否则量的是「换了个尺子」，
    而不是「这个模型好不好」。读不到就返回空字典，由调用方回退到配置值。
    """
    run_dir = weights.parent.parent if weights.parent.name == "weights" else weights.parent
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return {}
    wanted = {"imgsz", "batch", "device", "task"}
    found: dict[str, Any] = {}
    try:
        for raw in args_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key in wanted:
                found[key] = value.strip().strip("\"'")
    except OSError:
        return {}
    return found


def cmd_holdout(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    override = load_json(config_path)
    base = load_json(BASE_JSON) if BASE_JSON.is_file() else {}
    cfg = deep_merge(base, override)

    exp_id = str(cfg.get("exp_id") or override.get("exp_id") or args.exp_id or "")
    if args.weights:
        weights = Path(args.weights).resolve()
    else:
        if not exp_id:
            log("holdout 需要 --weights，或在配置里写 exp_id")
            return 2
        weights = (RUNS_DIR / exp_id / "weights" / "best.pt").resolve()
    if not weights.is_file():
        log(f"权重不存在：{weights}")
        return 2

    split_yaml = Path(args.split_yaml or (SPLITS_DIR / "pcb_holdout.yaml")).resolve()
    if not split_yaml.is_file():
        log(f"holdout 切分不存在：{split_yaml}，先跑 split 子命令")
        return 2
    check_split_drift(split_yaml)

    # 分辨率必须跟随训练时用的值：拿 640 去验一个 960 训出来的模型，
    # 精度会因分辨率不匹配而暴跌，看起来像「这个配置过拟合」，其实是量错了尺子。
    # 优先级：命令行显式指定 > 该次训练的 args.yaml > 配置 > 640
    run_args = read_run_args_yaml(weights)
    train_imgsz = args.imgsz or run_args.get("imgsz") or (cfg.get("train") or {}).get("imgsz") or 640
    train_batch = args.batch or run_args.get("batch") or (cfg.get("train") or {}).get("batch") or "16"
    if run_args.get("imgsz"):
        log(f"从训练记录读到 imgsz={run_args['imgsz']} batch={run_args.get('batch')}")
    else:
        log(f"未找到训练记录，按配置取 imgsz={train_imgsz} batch={train_batch}")

    # holdout 也要占显卡。虽然只要 10 秒左右，但和正在跑的网页训练撞上
    # 一样会 OOM，把本来 10 秒的验收变成一次失败。
    lock = load_gpulock()
    lock_held = False
    if lock is not None:
        holder = lock.read()
        if holder is not None and int(holder.get("pid") or 0) != os.getpid():
            log(f"显卡正被占用：{lock.describe(holder)}；holdout 未启动")
            return 5
        lock_held, lock_reason = lock.acquire("autoexp", "holdout", exp_id or weights.stem)
        if lock_held and lock_reason:
            log(f"注意：跨进程互斥未生效——{lock_reason}")

    # holdout 验收走独立子进程，防止污染主进程状态
    out_dir = assert_sandbox(ROOT / "runs" / "holdout" / f"{exp_id or 'adhoc'}_{datetime.now():%Y%m%d_%H%M%S}", "holdout_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_holdout_exec",
        "--weights",
        str(weights),
        "--data",
        str(split_yaml),
        "--out",
        str(out_dir),
        "--imgsz",
        str(int(train_imgsz)),
        "--batch",
        str(train_batch),
        "--device",
        str(args.device or (cfg.get("train") or {}).get("device", "0")),
    ]
    log_path = out_dir / "holdout.log"
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    finally:
        if lock is not None and lock_held:
            lock.release()

    result_path = out_dir / "result.json"
    if not result_path.is_file():
        log(f"holdout 失败，退出码 {proc.returncode}，详见 {log_path}")
        return 1
    result = load_json(result_path)
    if result.get("status") != "done":
        # 失败绝不能写进验收台账：holdout 每记录一次就是一次「看过」，
        # 一行空指标会让后面的比对误以为「这个配置在 holdout 上得 0 分」。
        log(f"holdout 失败：{result.get('error', '未知错误')}，详见 {log_path}")
        return 1
    row = {
        "exp_id": exp_id or out_dir.name,
        "created_at": now_iso(),
        "weights": str(weights),
        "split_yaml": str(split_yaml),
        "map50_95": result.get("map50_95", ""),
        "map50": result.get("map50", ""),
        "precision": result.get("precision", ""),
        "recall": result.get("recall", ""),
        "fp_per_image": result.get("fp_per_image", ""),
        "seconds": result.get("seconds", ""),
        "num_images": result.get("num_images", ""),
        "num_instances": result.get("num_instances", ""),
    }
    append_tsv(HOLDOUT_TSV, HOLDOUT_HEADER, row)
    log(
        f"holdout 完成：{row['exp_id']} mAP50-95={row['map50_95']} "
        f"图片数={row['num_images']} 实例数={row['num_instances']}"
    )
    return 0


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    """指标统一保留 6 位小数，避免台账里出现 0.4380553793247416 这种噪声位。"""
    return None if value is None else round(float(value), digits)


def _sum_array(value: Any) -> int:
    """安全求和一个可能是 numpy 数组 / 列表 / 标量 / None 的值。

    不能写成 `value or []`：numpy 数组的布尔判断会抛
    「truth value of an array with more than one element is ambiguous」。
    """
    if value is None:
        return 0
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float).ravel()
        return int(arr.sum()) if arr.size else 0
    except Exception:  # noqa: BLE001
        return 0


def cmd_holdout_exec(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    os.chdir(ROOT)
    started = time.time()
    try:
        launcher = load_launcher()
        model = launcher.load_yolo(args.weights)
        metrics = model.val(
            data=args.data,
            split="val",
            imgsz=int(args.imgsz),
            batch=int(float(args.batch)) if str(args.batch).lstrip("-").replace(".", "", 1).isdigit() else 16,
            device=args.device,
            workers=0,
            plots=False,
            verbose=False,
            project=str(out_dir),
            name="val",
            exist_ok=True,
        )
        results_dict = dict(getattr(metrics, "results_dict", {}) or {})
        # seen：val 阶段实际过过的图片数；nt_per_class：每个类别的实例数
        num_images = int(getattr(metrics, "seen", 0) or 0)
        if not num_images:
            num_images = _sum_array(getattr(metrics, "nt_per_image", None))
        payload = {
            "status": "done",
            "seconds": round(time.time() - started, 2),
            "num_images": num_images,
            "num_instances": _sum_array(getattr(metrics, "nt_per_class", None)),
            "map50_95": _round_or_none(_metric_from_results_dict(results_dict, "metrics/mAP50-95")),
            "map50": _round_or_none(_metric_from_results_dict(results_dict, "metrics/mAP50")),
            "precision": _round_or_none(_metric_from_results_dict(results_dict, "metrics/precision")),
            "recall": _round_or_none(_metric_from_results_dict(results_dict, "metrics/recall")),
        }
        payload["fp_per_image"] = fp_per_image_from_pr(
            payload["precision"], payload["recall"], payload["num_instances"], payload["num_images"]
        )
        save_json(result_path, payload)
        return 0
    except BaseException as exc:  # noqa: BLE001
        save_json(
            result_path,
            {"status": "crash", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        log(f"holdout 崩溃：{type(exc).__name__}: {exc}")
        return 1


def cmd_noise(args: argparse.Namespace) -> int:
    """同一配置重复 N 次，用 mAP50-95 的离散程度判断实验预算是否够用。"""
    config_path = Path(args.config).resolve()
    override = load_json(config_path)
    base = load_json(BASE_JSON) if BASE_JSON.is_file() else {}
    cfg = deep_merge(base, override)

    # 整段重复测量只让出一次。每一轮 cmd_run 看到环境变量就不会提前归还，
    # 否则两轮之间网页一点推理，显存又被占回去。
    owned_yield = acquire_infer_yield("噪声测量")
    try:
        return _cmd_noise_body(args, cfg)
    finally:
        release_infer_yield(owned_yield)


def _cmd_noise_body(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    values: list[float] = []
    for index in range(1, args.repeat + 1):
        round_cfg = dict(cfg)
        round_cfg["hypothesis"] = f"noise-{index}/{args.repeat}"
        round_cfg["changed"] = "none（噪声底测量）"
        temp = EXPERIMENTS_DIR / f"_noise_{datetime.now():%H%M%S}_{index}.json"
        save_json(temp, round_cfg)
        log(f"噪声测试 {index}/{args.repeat}")
        code = cmd_run(argparse.Namespace(config=temp, exp_id=None, no_validate=True))
        temp.unlink(missing_ok=True)
        if code != 0:
            log(f"第 {index} 次失败，跳过该样本")
            continue
        rows = read_tsv(RESULTS_TSV)
        if rows:
            try:
                values.append(float(rows[-1]["fitness_map50_95"]))
            except (TypeError, ValueError):
                continue

    if len(values) < 2:
        log(f"有效样本不足（{len(values)} 个），无法计算 sigma")
        return 1

    mean = statistics.fmean(values)
    sigma = statistics.pstdev(values)
    spread = max(values) - min(values)
    log("=" * 60)
    log(f"噪声底：样本 {[round(v, 6) for v in values]}")
    log(f"均值 {mean:.6f}  sigma {sigma:.6f}  极差 {spread:.6f}")
    if sigma <= 0.01:
        log(f"结论：预算够用。keep 阈值建议 2 sigma = {2 * sigma:.6f}")
    elif sigma <= 0.02:
        log(f"结论：预算偏紧。keep 阈值建议 2 sigma = {2 * sigma:.6f}，并考虑加大 budget_seconds")
    else:
        log("结论：噪声过大，当前预算不可用于自动搜索，必须先提高 budget_seconds 或 batch")
    log("=" * 60)

    # 按 (imgsz, batch) 分组记录：640 能跑 24 个 epoch、960 只能跑 12，
    # 两者末行 mAP 的抖动不是一个量级。只存一个全局 sigma 等于逼着所有分辨率
    # 共用同一把尺子，而 agent 的判定正是靠这把尺子（见 noise_threshold_for）。
    train_cfg = cfg.get("train") or {}
    key = noise_group_key(train_cfg.get("imgsz"), train_cfg.get("batch"))
    summary_path = ROOT / "experiments" / "_noise_summary.json"
    previous: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            previous = load_json(summary_path)
        except (OSError, ValueError):
            previous = {}
    groups = previous.get("groups") or {}
    if not isinstance(groups, dict):
        groups = {}
    groups[key] = {
        "created_at": now_iso(),
        "values": values,
        "mean": mean,
        "sigma": sigma,
        "spread": spread,
        "keep_threshold": round(2 * sigma, 6),
    }
    save_json(summary_path, {
        "created_at": now_iso(), "values": values, "mean": mean, "sigma": sigma, "spread": spread,
        "groups": groups,
        "_comment": (
            "顶层字段是最近一次测量的结果（向后兼容）；groups 按 imgsz×batch 分列，"
            "agent 判定 KEEP 时会优先用 groups 里对应那一组，没有才退回顶层。"
        ),
    })
    log(f"已按组记录：groups[{key}] = sigma {sigma:.6f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = read_tsv(RESULTS_TSV)
    if not rows:
        log("台账为空，还没有实验记录")
        return 0
    key = args.sort
    try:
        rows.sort(key=lambda r: float(r.get(key) or -1), reverse=True)
    except (TypeError, ValueError):
        pass

    log(f"共 {len(rows)} 条记录（按 {key} 降序）")
    header = (
        f"{'exp_id':<10}{'mAP50-95':>10}{'best':>9}{'mAP50':>8}{'ep':>5}{'sec':>8}"
        f"{'vram':>8}{'batch':>7}  {'status':<16} hypothesis"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        batch = row.get("batch_eff") or row.get("batch_req") or ""
        print(
            f"{row.get('exp_id',''):<10}"
            f"{row.get('fitness_map50_95','')[:8]:>10}"
            f"{row.get('best_map50_95','')[:7]:>9}"
            f"{row.get('map50','')[:6]:>8}"
            f"{row.get('epochs_done',''):>5}"
            f"{row.get('train_seconds',''):>8}"
            f"{row.get('peak_vram_mb',''):>8}"
            f"{batch:>7}  "
            f"{row.get('status',''):<16}"
            f"{row.get('hypothesis','')}"
        )
    holdout_rows = read_tsv(HOLDOUT_TSV)
    if holdout_rows:
        print()
        log(f"holdout 验收记录 {len(holdout_rows)} 条")
        for row in holdout_rows:
            raw = row.get("map50_95", "")
            try:
                shown = f"{float(raw):.6f}"
            except (TypeError, ValueError):
                shown = raw
            print(f"  {row.get('exp_id',''):<28} mAP50-95={shown}  n={row.get('num_images','')}")
    return 0


# ---------------------------------------------------------------------------
# 第 4 阶段：LLM 决策层
# ---------------------------------------------------------------------------
# 职责切分（这是整个设计的前提，改代码前先读一遍）：
#   - 模型只负责「提议下一组参数」，它不知道、也不决定 KEEP / DISCARD；
#   - 判定由本模块按实测噪声底 2 sigma 算死。
# 为什么不能让模型自己判断成败：实测 sigma = 0.0032，而 exp_0007(+0.0037)、
# exp_0008(+0.0001)、exp_0010(-0.0031) 全在噪声内。把数字交给模型，它几乎
# 必然把噪声讲成趋势，然后沿着错误方向持续加码——这是它最典型的行为模式。


class ProposalError(ValueError):
    """模型返回的内容没法解析成提议。与网络错误区分开：这类可以「再问一次」。"""


def load_llm() -> Any:
    """加载同目录的 ``llm.py``（零第三方依赖）。

    优先按**文件路径**加载而不是 ``import llm``：``llm`` 是个太通用的名字，
    sys.path 上任何同名模块都会把它遮蔽掉，而那种错误的表现是
    「莫名其妙调用了别的东西」，极难排查。

    返回 None 表示文件缺失或加载失败——调用方据此降级，不要让
    一个可选功能成为老命令跑不起来的原因。
    """
    module_path = ROOT / "llm.py"
    if not module_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("autoexp_llm", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001 - 缺了它就退回「不做决策层」
        log(f"加载 llm.py 失败：{type(exc).__name__}: {exc}")
        return None


def _as_float(value: Any) -> float | None:
    """宽松转 float：台账里全是字符串，空串/None/「nan」都要当作没有值。"""
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def noise_threshold() -> tuple[float, float | None, str]:
    """取 keep 阈值。

    返回 ``(阈值, sigma, 来源说明)``。sigma 为 None 表示没有实测数据。

    **为什么阈值必须由代码算、不能问模型**：这是整个功能的安全底线。
    模型对「0.0037 算不算提升」没有概念，它只会讲故事。

    注意这是**全局**阈值，用它去判所有分辨率是不对的——见 :func:`noise_threshold_for`。
    """
    if NOISE_JSON.is_file():
        try:
            summary = load_json(NOISE_JSON)
        except (OSError, ValueError):
            summary = {}
        sigma = _as_float(summary.get("sigma"))
        if sigma is not None and sigma > 0:
            # 先把 sigma 收敛到 6 位再乘 2：否则「阈值」与「2 × 报告的 sigma」
            # 在 6 位精度下会差一个末位，测试和人对不上（0.008069739 → 0.016139
            # 而不是 2 × 0.00807 = 0.016140）。
            sigma = round(sigma, 6)
            return round(2 * sigma, 6), sigma, "experiments/_noise_summary.json"
    return FALLBACK_KEEP_THRESHOLD, None, (
        f"兜底常量（未找到 _noise_summary.json，先跑 noise 子命令可得到实测值）"
    )


#: 台账/快照里，「同一套配置」用这个签名判定。键排序保证可比较。
def train_signature(train: dict[str, Any] | None) -> str:
    """把 ``train`` 字典压成一个可比较的签名。

    用途是「找出配置完全相同、只是重复跑了几次」的实验，用它们的离散度
    当作那份配置的真实噪声底——这是唯一不含真实配置效应的方差来源。
    """
    items = {k: v for k, v in (train or {}).items() if not str(k).startswith("_")}
    return json.dumps(items, ensure_ascii=False, sort_keys=True, default=str)


def noise_group_key(imgsz: Any, batch: Any) -> str:
    """噪声底的分组键：分辨率 × 批次。

    为什么恰好是这两个：它们是本机上决定「一次实验能跑多少 epoch」和
    「显存能有多紧」的两个变量，而 epoch 数直接决定末行 mAP 抖多大。
    同样的 600 秒，640 能跑 24 轮、960 只能跑 12 轮，两者的抖动不可能一样。
    """
    try:
        size = int(imgsz)
    except (TypeError, ValueError):
        size = 0
    return f"imgsz={size or '?'}|batch={batch if batch not in (None, '') else '?'}"


def _recorded_group_sigma(key: str) -> tuple[float, int] | None:
    """读 ``_noise_summary.json`` 里按组记录过的 sigma。"""
    if not NOISE_JSON.is_file():
        return None
    try:
        summary = load_json(NOISE_JSON)
    except (OSError, ValueError):
        return None
    groups = summary.get("groups") or {}
    if not isinstance(groups, dict):
        return None
    entry = groups.get(key)
    if not isinstance(entry, dict):
        return None
    sigma = _as_float(entry.get("sigma"))
    if sigma is None or sigma <= 0:
        return None
    return round(sigma, 6), len(entry.get("values") or [])


def _replicate_sigma(
    rows: list[dict[str, str]],
    signature: str,
) -> tuple[float, list[str]] | None:
    """用台账里**配置完全相同**的多次实验求 sigma。

    这是最可信的一类数据：配置一样、预算一样，差异只可能来自随机性
    （数据打乱、cuDNN 非确定性、mAP 阈值抖动）。

    要求至少 2 个样本（1 个样本无从谈离散度）。
    """
    if not signature:
        return None
    values: list[float] = []
    exp_ids: list[str] = []
    for row in rows:
        if (row.get("status") or "").strip() != "done":
            continue
        value = _as_float(row.get("fitness_map50_95"))
        if value is None:
            continue
        exp_id = str(row.get("exp_id") or "")
        snapshot_path = EXPERIMENTS_DIR / f"{exp_id}.json"
        if not snapshot_path.is_file():
            continue
        try:
            resolved = load_json(snapshot_path).get("resolved") or {}
        except (OSError, ValueError):
            continue
        if train_signature(resolved.get("train") or {}) != signature:
            continue
        values.append(value)
        exp_ids.append(exp_id)
    if len(values) < 2:
        return None
    return round(statistics.pstdev(values), 6), exp_ids


def noise_threshold_for(
    rows: list[dict[str, str]],
    train_cfg: dict[str, Any] | None,
    *,
    global_threshold: float,
    global_sigma: float | None,
    global_source: str,
) -> dict[str, Any]:
    """按「这组参数」取噪声底，而不是全库一个数。

    ## 为什么必须有这一层

    2026-11-10 那场 5 轮实测里出现了一个反直觉现象：同一套 960 基座，
    三次实验拿回 0.4586 / 0.4297 / 0.3919——跨度 0.067。而判定用的
    σ=0.0032 是在 **640** 上用重复实验测出来的（640 能跑 24 个 epoch，
    末行 mAP 反而稳）。用 640 的尺子去量 960，等于把「960 本来的抖动」
    当成「真实的巨大进步/退步」。

    更糟的是这个偏差有方向：谁运气好跑出高点，谁就被记住，之后所有实验
    都追不上那个高点的随机上沿 —— 表现为「KEEP 永远是 0」，而人以为是
    「模型提的方向都不行」。

    ## 取值优先级

    1. ``_noise_summary.json`` 的 ``groups[key]``：用 ``noise --config`` 在该配置上
       真实重复测过。这是唯一「确定可信」的数据。
    2. 台账里**配置完全相同**的多次实验：配置一样，差异只可能来自随机性。
    3. 以上都没有 → 退回全局 sigma，但 ``calibrated=False``。
       调用方必须把这件事告诉用户：**这一轮的判定是拿别的分辨率的尺子量的**。

    第 3 条刻意不「猜一个更大的数」：凭空放大阈值和凭空缩小一样没有依据，
    只会让人以为门槛是校准过的。诚实地说「没测过」，用户才知道该去跑 noise。
    """
    train = train_cfg or {}
    key = noise_group_key(train.get("imgsz"), train.get("batch"))

    recorded = _recorded_group_sigma(key)
    if recorded is not None:
        sigma, count = recorded
        return {
            "threshold": round(2 * sigma, 6),
            "sigma": sigma,
            "group": key,
            "calibrated": True,
            "samples": count,
            "source": f"{NOISE_JSON.name} 的 groups[{key}]（{count} 次重复实测）",
        }

    replicate = _replicate_sigma(rows, train_signature(train))
    if replicate is not None:
        sigma, exp_ids = replicate
        return {
            "threshold": round(2 * sigma, 6),
            "sigma": sigma,
            "group": key,
            "calibrated": True,
            "samples": len(exp_ids),
            "source": f"台账里配置相同的 {len(exp_ids)} 次实验（{', '.join(exp_ids)}）",
        }

    return {
        "threshold": global_threshold,
        "sigma": global_sigma,
        "group": key,
        "calibrated": False,
        "samples": 0,
        "source": f"全局阈值（{global_source}）——注意：这是别的配置测出来的尺子",
    }


def params_diff(cfg: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """算出 cfg 相对 base 的改动，只保留改动项。

    不能把整份 40 多个参数的快照塞给模型：那会让每轮 prompt 涨到上万 token，
    同时把「这次到底改了什么」淹没在噪声里——而后者才是决策需要的信息。
    """
    skip = {"_comment", "hypothesis", "changed", "resolved", "exp_id",
            "created_at", "config_source", "resources_at_start"}
    diff: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in skip or key == "train":
            continue
        if base.get(key) != value:
            diff[key] = value
    train_diff: dict[str, Any] = {}
    for key, value in (cfg.get("train") or {}).items():
        if (base.get("train") or {}).get(key) != value:
            train_diff[key] = value
    if train_diff:
        diff["train"] = train_diff
    return diff


def current_best(rows: list[dict[str, str]], base: dict[str, Any]) -> dict[str, Any] | None:
    """台账里适应度最高的那条有效实验。

    只看 ``status == "done"``：崩溃行和预检拦截行没有可比指标，
    把它们算进来会让「当前最佳」莫名其妙地变低。
    """
    best: dict[str, Any] | None = None
    for row in rows:
        if (row.get("status") or "").strip() != "done":
            continue
        value = _as_float(row.get("fitness_map50_95"))
        if value is None:
            continue
        if best is None or value > best["fitness"]:
            best = {"exp_id": row.get("exp_id", ""), "fitness": value, "row": row}
    if best is None:
        return None
    exp_id = str(best.get("exp_id") or "")
    snapshot_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    resolved: dict[str, Any] = {}
    if snapshot_path.is_file():
        try:
            resolved = load_json(snapshot_path).get("resolved") or {}
        except (OSError, ValueError):
            resolved = {}
    best["params_diff"] = params_diff(resolved, base) if resolved else {}
    return best


def history_with_delta(
    rows: list[dict[str, str]],
    base: dict[str, Any],
    limit: int = 30,
) -> list[dict[str, Any]]:
    """把台账整理成「给模型看的历史」。

    只给三样东西：改了什么、拿到了什么、相对当时的最佳差多少。
    `delta` 用**当时的历史最佳**而非最终最佳来算——否则早期实验的 delta
    会随着后期进步而全部变成负数，看不出「当时赢没赢」。
    """
    entries: list[dict[str, Any]] = []
    running_best: float | None = None
    for row in rows:
        status = (row.get("status") or "").strip()
        value = _as_float(row.get("fitness_map50_95"))
        delta = None
        if value is not None and running_best is not None:
            delta = round(value - running_best, 6)

        exp_id = str(row.get("exp_id") or "")
        diff: dict[str, Any] = {}
        snapshot_path = EXPERIMENTS_DIR / f"{exp_id}.json"
        if snapshot_path.is_file():
            try:
                resolved = load_json(snapshot_path).get("resolved") or {}
                diff = params_diff(resolved, base) if resolved else {}
            except (OSError, ValueError):
                diff = {}

        entries.append({
            "exp_id": exp_id,
            "status": status or "unknown",
            "fitness": value,
            "delta_vs_best_then": delta,
            "changed": row.get("changed", ""),
            "hypothesis": row.get("hypothesis", ""),
            "params_diff": diff,
        })

        # 只有跑完的实验才推进「历史最佳」：崩溃行不能抬高基准
        if status == "done" and value is not None:
            running_best = value if running_best is None else max(running_best, value)

    return entries[-limit:] if limit > 0 else entries


def build_agent_context(
    rows: list[dict[str, str]],
    base: dict[str, Any],
    space: dict[str, Any],
    round_index: int,
    rounds_total: int,
    history_limit: int = 30,
) -> dict[str, Any]:
    """组装给模型看的全部信息。

    刻意**不含** holdout 的任何数据或文件路径：模型看到的东西会进日志、
    进审计 JSON、进 prompt 缓存。holdout 纪律的底线是「搜索期一次都不许看」，
    所以这条边界要从源头卡住，而不是指望模型自觉。
    """
    threshold, sigma, source = noise_threshold()
    # 下划线开头的键（_comment 等）是给人和程序看的元数据，既不是参数也不能改。
    # 不过滤的话它会混进 allowed_top_level，等于告诉模型「你可以设一个 _comment 字段」。
    def _real_keys(mapping: dict[str, Any]) -> set[str]:
        return {k for k in mapping if not str(k).startswith("_")}

    allowed_train = sorted(_real_keys(space.get("train", {}) or {}) - FROZEN_TRAIN)
    allowed_top = sorted((_real_keys(space) - {"train"}) - FROZEN_TOP)

    # 只给出「能改的那些键」的取值约束，而不是整份 space.json：
    # 冻结键写出来只会诱导模型去改它们。
    #
    # 这一项曾经是整条链路上最贵的一处疏漏：旧版只给 allowed_train（**键名**），
    # 没有给取值。模型只能猜，于是提出 imgsz=1024、batch=2 这种必然越界的值，
    # 每次都被拒、重问一次、再被拒，整轮作废——一轮就是 11 分钟。
    # 把 enum / min / max 明确摆出来，比任何「请遵守搜索空间」的措辞都有效。
    train_space = space.get("train", {}) or {}
    top_space = {k: v for k, v in space.items() if k != "train"}
    search_space: dict[str, Any] = {}
    top_rules = {k: top_space[k] for k in allowed_top if k in top_space}
    if top_rules:
        search_space.update(top_rules)
    train_rules = {k: train_space[k] for k in allowed_train if k in train_space}
    if train_rules:
        search_space["train"] = train_rules

    baseline = {k: v for k, v in base.items() if not k.startswith("_")}
    incumbent = current_best(rows, base)

    context: dict[str, Any] = {
        "round": round_index,
        "rounds_total": rounds_total,
        "noise": {
            "sigma": sigma,
            "keep_threshold": threshold,
            "source": source,
            "note": "差异绝对值小于 keep_threshold 一律视为随机波动，不要当成趋势",
        },
        "baseline": baseline,
        "allowed_train": allowed_train,
        "search_space": search_space,
        "frozen_fields": sorted(FROZEN_TOP | FROZEN_TRAIN),
        "incumbent": (
            {
                "exp_id": incumbent["exp_id"],
                "fitness_map50_95": round(incumbent["fitness"], 6),
                "params_diff_from_baseline": incumbent.get("params_diff", {}),
                "note": (
                    "下一轮里你没写的参数，继承「baseline + 上面这个 diff」，"
                    "**不是**继承 baseline 本身。"
                ),
            }
            if incumbent else None
        ),
        "history": history_with_delta(rows, base, history_limit),
    }
    # 顶层当前全被冻结（model / budget_seconds / data 都不许动），所以这个列表是空的。
    # 空数组反而是个「你还能改顶层哦」的暗示，干脆不出现。
    if allowed_top:
        context["allowed_top_level"] = allowed_top
    return context


_FALLBACK_SYSTEM = (
    "你是 YOLO 训练参数搜索助手。只输出一个 JSON 对象，字段为 "
    "hypothesis / changed / reasoning / train。"
    "train 里只放本次要改的参数。不要输出 JSON 以外的任何内容。"
)


def render_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """把上下文拼成对话消息。

    system prompt 来自 ``prompts/experimenter.md`` 而不是硬编码在 Python 里：
    上游 autoresearch 最重要的一课是「调研究策略应该改 markdown」。
    用户想换策略（比如只搜增强、或专攻学习率）时改一个 md 就行，不用碰代码。
    """
    if PROMPT_MD.is_file():
        try:
            system = PROMPT_MD.read_text(encoding="utf-8")
        except OSError:
            system = _FALLBACK_SYSTEM
    else:
        system = _FALLBACK_SYSTEM
    # 注意：system 里必须出现 "json" 字样，否则 OpenAI / DeepSeek 的
    # response_format=json_object 会直接报 400。两份内容都满足。
    user = (
        "以下是本轮实验的上下文（JSON）。请给出下一组要试的参数：\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_proposal(content: str) -> dict[str, Any]:
    """把模型回的内容解析成 dict。

    模型偶尔会包一层 ```json 代码块，或者在 JSON 前后加一句「好的，我的提议是」。
    这两种都不该让整轮废掉——它们是无害的表达习惯，不是错误决策。
    """
    text = (content or "").strip()
    if not text:
        # 空正文多半是「输出预算被思考吃光」，而不是模型不会写 JSON。
        # 把这个可能性直接写进错误里，省掉一轮瞎猜。
        raise ProposalError(
            "返回内容为空。若使用推理型模型（deepseek-flash / reasoner / o 系列），"
            "通常是 max_tokens 太小、被思考阶段耗尽，请调到 4096 以上"
        )

    if text.startswith("```"):
        # 去掉首行 ``` 或 ```json，再去掉结尾的 ```
        first_break = text.find("\n")
        if first_break >= 0:
            text = text[first_break + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except ValueError:
        # 兜底：截取第一个 { 到最后一个 }。宁可多试一次，也不要因为
        # 一句「好的」就空耗一轮 API 费。
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            # 带上总长度：160 字以内就是「模型在闲聊」，几百字以上却没闭合花括号，
            # 那基本是输出被截断——这两种情况的下一步动作完全不同。
            raise ProposalError(
                f"找不到 JSON 对象（正文 {len(text)} 字符）：{text[:160]!r}"
            ) from None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError as exc:
            raise ProposalError(
                f"JSON 解析失败：{exc}（正文 {len(text)} 字符）；原文={text[:160]!r}"
            ) from exc

    if not isinstance(data, dict):
        raise ProposalError(f"顶层必须是对象，实际是 {type(data).__name__}")
    return data


def validate_proposal(
    proposal: dict[str, Any],
    base: dict[str, Any],
    space: dict[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    """校验模型的提议。返回 ``(错误列表, 合并后的完整配置)``。

    比 ``validate_against_space()`` 多两层：
      1. 顶层字段白名单 —— 模型不许自己发明 "epochs" 之类的键；
      2. 冻结字段 —— 即使 space.json 声明了也不许改（见 FROZEN_TOP 的注释）。

    最后还独立断言一次数据路径。这一条**不能省**：`space.json` 根本不校验
    `data`，而 `data` 指向 holdout 就等于让整个搜索在验收集上过拟合，
    且台账上完全看不出来。
    """
    errors: list[str] = []

    unknown = set(proposal) - PROPOSAL_KEYS
    if unknown:
        errors.append(f"出现了不允许的顶层字段：{sorted(unknown)}（只允许 {sorted(PROPOSAL_KEYS)}）")

    frozen_top_hit = sorted(set(proposal) & FROZEN_TOP)
    if frozen_top_hit:
        errors.append(f"这些字段被冻结，不允许修改：{frozen_top_hit}")

    train = proposal.get("train", {})
    if train is None:
        train = {}
    if not isinstance(train, dict):
        errors.append("train 必须是对象")
        train = {}

    frozen_train_hit = sorted(set(train) & FROZEN_TRAIN)
    if frozen_train_hit:
        errors.append(f"这些训练参数被冻结，不允许修改：{frozen_train_hit}")

    if errors:
        return errors, None

    merged = deep_merge(base, {"train": train})
    errors.extend(validate_against_space(merged, space))
    if errors:
        return errors, None

    # 数据路径硬校验（见上面的注释：这一条是安全边界，不是洁癖）
    try:
        data_path = Path(resolve_data_path(str(merged.get("data", "")))).resolve()
        splits_root = SPLITS_DIR.resolve()
        inside = data_path == splits_root or splits_root in data_path.parents
        if not inside:
            errors.append(f"data 必须落在 {splits_root} 内，实际是 {data_path}")
        elif "tune" not in data_path.name:
            # 只允许 tune 切分。holdout 的名字里没有 "tune"，会在这里被拦下。
            errors.append(f"data 只能指向 tune 切分，实际是 {data_path.name}")
    except Exception as exc:  # noqa: BLE001 - 解析失败本身就是错误
        errors.append(f"data 路径无法解析：{type(exc).__name__}: {exc}")

    return errors, (merged if not errors else None)


def judge_round(
    fitness: float | None,
    incumbent_fitness: float | None,
    threshold: float,
    *,
    threshold_source: str | None = None,
    calibrated: bool | None = None,
    group: str | None = None,
    fp_per_image: float | None = None,
    incumbent_fp_per_image: float | None = None,
) -> tuple[str, str]:
    """按噪声阈值判定 keep / discard，返回 ``(结论, 理由)``。

    这是全流程唯一有权说「这个改动算成功」的地方，而且它只看数字。

    ``calibrated=False`` 时必须把它写进理由：那意味着阈值是**别的配置**
    测出来的尺子，用它判 KEEP/DISCARD 都可能错。让这句话出现在台账和界面上，
    用户才知道该去补一次 ``noise`` 测量，而不是怀疑模型。

    ``fp_per_image`` 是过杀率（每图误报框数，由 P/R 反推的近似值）。两边都有值时，
    即使 mAP 提升超过噪声阈值，过杀恶化也必须拦下——产线上「多报」是有成本的，
    拿过杀换 mAP 的改动不能靠 2σ 规则自动转正。
    """
    if fitness is None:
        return "DISCARD", "没有产出有效指标（崩溃或预检拦截），本轮不计入进展"
    if incumbent_fitness is None:
        return "KEEP", "首个有效实验，自动作为后续比较的基准"

    delta = fitness - incumbent_fitness
    caveat = ""
    if calibrated is False:
        caveat = (
            f"（注意：{group or '当前配置'} 的噪声底没测过，"
            f"这里借用了其它配置的阈值，结论仅供参考；建议先跑一次 noise 校准）"
        )

    if delta > threshold:
        # 过杀拦截：只在两边都有数时生效。容差取 max(0.01, 基线×5%)——
        # 低于这个量级的差异在近似算法下不可信，不能当成「过杀变差」。
        if fp_per_image is not None and incumbent_fp_per_image is not None:
            tolerance = max(0.01, float(incumbent_fp_per_image) * 0.05)
            if float(fp_per_image) - float(incumbent_fp_per_image) > tolerance:
                return (
                    "DISCARD",
                    f"mAP 提升 {delta:+.5f} 超过噪声阈值 {threshold:.5f}，"
                    f"但过杀从 {float(incumbent_fp_per_image):.3f} 升到 {float(fp_per_image):.3f} 框/图"
                    f"（容差 {tolerance:.3f}）——不拿过杀换 mAP{caveat}",
                )
        return (
            "KEEP",
            f"提升 {delta:+.5f}，超过噪声阈值 {threshold:.5f}，可判定为真实改进{caveat}",
        )
    if delta < -threshold:
        return (
            "DISCARD",
            f"下降 {delta:+.5f}，超过噪声阈值 {threshold:.5f}，是真实的负向{caveat}",
        )
    return (
        "DISCARD",
        f"差异 {delta:+.5f}，未超过噪声阈值 {threshold:.5f}，无法与随机波动区分{caveat}",
    )


def _write_agent_audit(index: int, record: dict[str, Any]) -> Path | None:
    """写单轮审计快照。

    成本核算与「模型当时到底看到了什么」全靠它：出问题时第一件事就是
    对比 prompt 与实际发生在台账上的改动。
    """
    try:
        AGENT_DIR.mkdir(parents=True, exist_ok=True)
        path = AGENT_DIR / f"{AGENT_AUDIT_PREFIX}{index:04d}.json"
        save_json(path, record)
        return path
    except OSError as exc:
        log(f"审计快照写入失败（不影响实验）：{exc}")
        return None


def update_best(
    exp_id: str,
    row: dict[str, str],
    cfg: dict[str, Any],
    threshold: float,
    sigma: float | None,
) -> None:
    """KEEP 之后刷新 ``experiments/best.json``。

    沿用原有 schema，让「人看的文件」与「自动维护的文件」是同一份，
    不新增第二处「当前最佳」的事实来源——两份真相迟早不一致。
    holdout 结论留空并注明需人工验收：搜索期禁止调用 holdout 是纪律。
    """
    previous: dict[str, Any] = {}
    if BEST_JSON.is_file():
        try:
            previous = load_json(BEST_JSON)
        except (OSError, ValueError):
            previous = {}

    base = load_json(BASE_JSON) if BASE_JSON.is_file() else {}
    diff = params_diff(cfg, base)
    noise_mean = None
    if NOISE_JSON.is_file():
        try:
            noise_mean = _as_float(load_json(NOISE_JSON).get("mean"))
        except (OSError, ValueError):
            noise_mean = None

    fitness = _as_float(row.get("fitness_map50_95"))
    payload = {
        "_comment": previous.get("_comment")
        or "当前最佳配置。由 autoexp agent 自动维护：只有超过 2 sigma 噪声阈值的实验才会写到这里。",
        "updated_at": now_iso(),
        "exp_id": exp_id,
        "source": "autoexp agent",
        "top_level": {k: v for k, v in diff.items() if k != "train"},
        "train": diff.get("train", {}),
        "evidence": {
            "tune_map50_95": fitness,
            "baseline_tune_mean": noise_mean,
            "noise_sigma": sigma,
            "keep_threshold_2sigma": threshold,
            "delta_tune": (
                None if (fitness is None or noise_mean is None)
                else round(fitness - noise_mean, 6)
            ),
            "holdout_map50_95": None,
            # 这次 KEEP 依据的是哪批数据。换了数据集之后，「比基线高」这件事
            # 可能只是数据变了，没有这一行就无法回头核对。
            "dataset_fingerprint": row.get("data_fp") or None,
        },
        "note": (
            f"holdout 尚未验收，需人工执行：python autoexp.py holdout --exp-id {exp_id}"
        ),
    }
    save_json(BEST_JSON, payload)


def _ask_proposal(
    llm: Any,
    messages: list[dict[str, str]],
    base: dict[str, Any],
    space: dict[str, Any],
    attempts: int = 2,
    inherit_train: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """问模型要一组参数；被拒绝时带着理由再问一次。

    只重问一次的理由：越界通常说明它没仔细看搜索空间，提示一次就够；
    真不听话的任务重复问只会烧钱，不如跳过这一轮。

    ``inherit_train``：当前最优相对基线的差值（``best.json`` 的 ``train``）。
    校验与合并都基于「最优 + 本轮改动」，因为 `run` 最终是用 base 合并 cur.json，
    所以模型没写的键必须在这里就补成最优值——否则「在 960 上改 warmup」
    会变成「在 640 上改 warmup」，整轮结论作废。
    """
    history = list(messages)
    meta: dict[str, Any] = {"attempts": 0, "usages": [], "finish_reasons": []}
    last_error = "未获得有效提议"
    inherited = {k: v for k, v in (inherit_train or {}).items() if not str(k).startswith("_")}

    for attempt in range(1, attempts + 1):
        meta["attempts"] = attempt
        try:
            reply = llm.chat(history)
        except llm.LLMTruncated as exc:
            # 截断单独处理：这**不是**模型不会写 JSON，是输出预算不够。
            # 以前两者共用同一句「返回内容不是合法 JSON」，害得人只会去改提示词。
            meta["error"] = str(exc)
            meta["truncated"] = True
            log(f"模型输出被截断：{exc}")
            emit_event({
                "type": "agent",
                "name": "agent_error",
                "code": "OUTPUT_TRUNCATED",
                "message": str(exc),
                "max_tokens": getattr(exc, "max_tokens", 0),
                "reasoning_tokens": getattr(exc, "reasoning_tokens", 0),
            })
            return None, meta
        except llm.LLMError as exc:
            meta["error"] = str(exc)
            log(f"模型调用失败：{exc}")
            emit_event({"type": "agent", "name": "agent_error", "code": "LLM_FAILED", "message": str(exc)})
            return None, meta

        content = reply.get("content") or ""
        meta["usages"].append(reply.get("usage") or {})
        meta["latency_ms"] = reply.get("latency_ms")
        if reply.get("finish_reason"):
            meta["finish_reasons"].append(reply.get("finish_reason"))

        try:
            proposal = parse_proposal(content)
        except ProposalError as exc:
            last_error = f"返回内容不是合法 JSON：{exc}"
            log(last_error)
            emit_event({"type": "agent", "name": "agent_error", "code": "PROPOSAL_UNPARSABLE", "message": last_error})
            # 空正文不要塞进历史。部分兼容接口会因为 assistant 的 content 为空
            # 直接 400，于是「重问一次」变成「必然失败一次」，白烧一次费用。
            if content.strip():
                history = history + [{"role": "assistant", "content": content}]
            history = history + [
                {
                    "role": "user",
                    "content": (
                        "只输出一个 JSON 对象，不要任何其它文字，也不要 markdown 代码块。"
                        "`train` 的每个值都必须落在 `search_space` 声明的 enum 或 min/max 内。请重来。"
                    ),
                },
            ]
            continue

        # 把继承的最优值并进来再校验：cur.json 最终会被 `run` 用 base 合并，
        # 所以模型没写的键在这里就得补成最优值，否则实际跑的不是它以为的配置。
        candidate = dict(proposal)
        if inherited:
            merged_train = {**inherited, **dict(proposal.get("train") or {})}
            candidate["train"] = merged_train
        errors, merged = validate_proposal(candidate, base, space)
        meta["proposal"] = proposal
        meta["candidate"] = candidate
        meta["raw"] = content
        if not errors and merged is not None:
            meta["resolved"] = merged
            return proposal, meta

        last_error = "；".join(errors) or "提议无效"
        log(f"提议被拒：{last_error}")
        emit_event({
            "type": "agent",
            "name": "agent_rejected",
            "attempt": attempt,
            "reasons": errors or [last_error],
        })
        if content.strip():
            history = history + [{"role": "assistant", "content": content}]
        history = history + [
            {
                "role": "user",
                "content": (
                    "上面的提议被程序拒绝，原因：\n- " + "\n- ".join(errors or [last_error])
                    + "\n请只使用 `search_space` 里声明过的参数名与取值"
                    "（enum 只能取列出的值，min/max 是连续区间），重新给出 JSON。"
                ),
            },
        ]

    meta["error"] = last_error
    return None, meta


def cmd_agent(args: argparse.Namespace) -> int:
    """无人值守连跑：提议 -> 校验 -> 训练 -> 判定 -> 记账。"""
    global EVENTS_ENABLED
    EVENTS_ENABLED = bool(getattr(args, "events", False))

    llm = load_llm()
    if llm is None:
        log("找不到可用的 llm.py（或加载失败），agent 无法启动")
        return 1

    base = load_json(BASE_JSON)
    space = load_json(SPACE_JSON) if SPACE_JSON.is_file() else {}
    rounds_total = max(1, int(getattr(args, "rounds", 30) or 30))

    # 时间上限：0 表示不限。两个上限取先到者，防止「轮数没到但已经天亮」。
    hours = float(getattr(args, "hours", 8.0) or 0)
    deadline = (time.time() + hours * 3600) if hours > 0 else None

    # --dry-run：把第一轮的完整 prompt 打出来，不调 API、不跑训练。
    # 这是「花钱之前先看一眼」的唯一机会，值得单独一条路径。
    if getattr(args, "dry_run", False):
        rows_now = read_tsv(RESULTS_TSV)
        context = build_agent_context(rows_now, base, space, 1, rounds_total)
        threshold, sigma, source = noise_threshold()
        # dry-run 也要反映真实的判定口径：否则预览里写着 0.0064，
        # 真跑起来却按 960 那组的标准判，人会以为程序自相矛盾。
        inc_now = current_best(rows_now, base)
        train_now = dict(base.get("train") or {})
        if inc_now is not None:
            train_now.update((inc_now.get("params_diff") or {}).get("train", {}))
        nb = noise_threshold_for(
            rows_now, train_now,
            global_threshold=threshold, global_sigma=sigma, global_source=source,
        )
        context["noise"] = {
            "sigma": nb["sigma"],
            "keep_threshold": nb["threshold"],
            "source": nb["source"],
            "group": nb["group"],
            "calibrated": nb["calibrated"],
            "note": (
                "本轮判定用的阈值。calibrated=false 表示这组参数没测过噪声底，"
                "阈值是借来的，结论仅供参考"
            ),
        }
        messages = render_messages(context)
        bar = "=" * 72
        print(bar)
        print(f"system prompt：{len(messages[0]['content'])} 字符（来自 prompts/experimenter.md）")
        print(bar)
        print(messages[0]["content"])
        print(bar)
        print("user 消息（本轮上下文）")
        print(bar)
        print(messages[1]["content"])
        print(bar)
        print(f"当前 keep 阈值：{nb['threshold']}（sigma={nb['sigma']}，来源：{nb['source']}）")
        print(f"分组：{nb['group']}  已校准：{nb['calibrated']}")
        print(f"模型配置：{llm.describe_config()}")
        return 0

    threshold, sigma, source = noise_threshold()
    log(f"agent 启动：最多 {rounds_total} 轮，时间上限 {hours or '不限'} 小时")
    log(f"keep 阈值 {threshold}（sigma={sigma}，来自 {source}）")

    # 启动就说清楚「这把尺子准不准」。等到判定完再解释已经晚了一轮：
    # 640 的 σ 拿去量 960，会让后续每一轮都被判 DISCARD 而人不知道原因。
    try:
        rows_now = read_tsv(RESULTS_TSV)
        inc_now = current_best(rows_now, base)
        if inc_now is not None:
            train_now = dict(base.get("train") or {})
            train_now.update((inc_now.get("params_diff") or {}).get("train", {}))
            nb = noise_threshold_for(
                rows_now, train_now,
                global_threshold=threshold, global_sigma=sigma, global_source=source,
            )
            if nb["calibrated"]:
                log(f"噪声底已校准：{nb['group']}（sigma={nb['sigma']}，{nb['source']}）")
            else:
                log(f"注意：{nb['group']} 的噪声底还没测过，本轮判定会借用全局阈值。")
                log(f"      {nb['source']}")
                log("      建议先补测一次（约 30 分钟）："
                    "python autoexp.py noise --config experiments/cur.json --repeat 3")
    except (OSError, ValueError):
        pass
    # 把 max_tokens 打出来：推理型模型被截断时，这一行是唯一能直接对照的数字。
    llm_cfg = llm.load_config()
    log(
        f"模型 {llm.describe_config()}；max_tokens={llm_cfg.get('max_tokens')}"
        f"；Key {llm.mask_key(llm.resolve_key()) or '未配置'}"
    )
    if int(llm_cfg.get("max_tokens") or 0) < 1500:
        log("注意：max_tokens 偏小。推理型模型会把「思考」和「正文」算进同一预算，"
            "容易被截断成半截 JSON（表现为「返回内容为空」）。建议改到 4096。")
    # 整场 agent 只让出一次，盖住每一轮训练。网页后端启动时会自己让出并打上
    # 环境变量，这里再调一次会直接返回，不会在中途把显存还回去。
    owned_yield = acquire_infer_yield("自动实验")
    try:
        return _cmd_agent_loop(
            args, llm, base, space, rounds_total, deadline, threshold, sigma, source,
            hours, llm_cfg,
        )
    finally:
        release_infer_yield(owned_yield)


def _cmd_agent_loop(
    args: argparse.Namespace,
    llm: Any,
    base: dict[str, Any],
    space: dict[str, Any],
    rounds_total: int,
    deadline: float | None,
    threshold: float,
    sigma: float | None,
    source: str,
    hours: float,
    llm_cfg: dict[str, Any],
) -> int:
    emit_event({
        "type": "agent",
        "name": "agent_start",
        "rounds": rounds_total,
        "hours": hours,
        "keep_threshold": threshold,
        "sigma": sigma,
        "model": llm.describe_config(),
        "max_tokens": llm_cfg.get("max_tokens"),
    })

    max_stale = max(1, int(getattr(args, "max_stale", 5) or 5))
    # 「没拿到有效提议」有自己的上限，且比 stale 更小：
    # 连续 3 次问不出合规 JSON，多半是配置坏了（预算太小/模型名错/限流），
    # 再问 2 轮只是把钱烧掉，还让「连续 N 轮没有进展」这句话骗人。
    max_errors = max(1, int(getattr(args, "max_errors", 3) or 3))
    stale = 0
    errors = 0
    kept = 0
    completed = 0
    started_at = time.time()

    for index in range(1, rounds_total + 1):
        stop = evaluate_stop(
            stale=stale, max_stale=max_stale,
            errors=errors, max_errors=max_errors,
            now=time.time(), deadline=deadline,
        )
        if stop is not None:
            code, message = stop
            log(message)
            emit_event({"type": "agent", "name": "agent_stop", "code": code, "message": message})
            break

        log(f"===== 第 {index}/{rounds_total} 轮 =====")
        emit_event({"type": "agent", "name": "round_start", "round": index, "rounds": rounds_total})

        rows = read_tsv(RESULTS_TSV)
        incumbent = current_best(rows, base)
        context = build_agent_context(rows, base, space, index, rounds_total)

        # 噪声底按「当前最优所在的那组参数」取，而不是全库一个数。
        # 640 能跑 24 epoch、960 只能跑 12，两者末行 mAP 的抖动差一个量级；
        # 用同一个阈值判，等于拿 640 的尺子量 960（见 noise_threshold_for 的注释）。
        incumbent_train = (incumbent.get("params_diff") or {}).get("train", {}) if incumbent else {}
        incumbent_full_train = dict(base.get("train") or {})
        incumbent_full_train.update(incumbent_train)
        noise = noise_threshold_for(
            rows, incumbent_full_train,
            global_threshold=threshold, global_sigma=sigma, global_source=source,
        )
        round_threshold = float(noise["threshold"])
        context["noise"] = {
            "sigma": noise["sigma"],
            "keep_threshold": round_threshold,
            "source": noise["source"],
            "group": noise["group"],
            "calibrated": noise["calibrated"],
            "note": (
                "本轮判定用的阈值。calibrated=false 表示这组参数没测过噪声底，"
                "阈值是借来的，结论仅供参考"
            ),
        }
        messages = render_messages(context)

        proposal, meta = _ask_proposal(llm, messages, base, space, inherit_train=incumbent_train)
        if proposal is None:
            # 注意：这里**不加** stale。这一轮根本没产生实验结果，
            # 说「没有进展」会让排查方向跑偏——真正的问题在模型侧。
            errors += 1
            log(f"第 {index} 轮无有效提议（连续失败 {errors}/{max_errors}），跳过：{meta.get('error', '')}")
            _write_agent_audit(index, {
                "round": index,
                "created_at": now_iso(),
                "outcome": "skipped",
                "error": meta.get("error", ""),
                "truncated": bool(meta.get("truncated")),
                "attempts": meta.get("attempts"),
                "finish_reasons": meta.get("finish_reasons", []),
                "usages": meta.get("usages", []),
            })
            continue

        # 拿到有效提议就清零：error 计数衡量的是「连续」，不是累计。
        errors = 0

        resolved: dict[str, Any] = meta["resolved"]
        # cur.json 写的是「从当前最优出发、本轮要显式声明的改动」。
        #
        # 为什么不能只写 proposal.train：`run` 是用 **base.json** 合并的，
        # 于是「只写 warmup_epochs」会在 **640** 上跑，而模型以为它在 960 上
        # 做单变量对照（2026-11-10 第 3 轮真实发生过：exp_0017 意图 960，
        # 实际 640，整轮结论作废）。
        # 把最优配置相对基线的差值一并写上，既有显式声明、又能直接 `run`：
        # 提议里的键排在后面，所以本轮想改的仍然会覆盖继承来的值。
        best_train = dict(incumbent_train)
        proposal_train = dict(proposal.get("train") or {})
        effective_train = {**best_train, **proposal_train}
        cur_payload = {
            "hypothesis": str(proposal.get("hypothesis") or "").strip(),
            "changed": str(proposal.get("changed") or "").strip(),
            "train": effective_train,
        }
        save_json(CUR_JSON, cur_payload)

        log(f"模型提议：{cur_payload['hypothesis']}")
        log(f"改动：{cur_payload['changed'] or '（无）'}")
        if best_train:
            inherited = ", ".join(f"{k}={v}" for k, v in best_train.items() if k not in proposal_train)
            if inherited:
                log(f"继承当前最优（{incumbent['exp_id'] if incumbent else '?'}）：{inherited}")
        emit_event({
            "type": "agent",
            "name": "agent_proposal",
            "round": index,
            "hypothesis": cur_payload["hypothesis"],
            "changed": cur_payload["changed"],
            "params": json.dumps(cur_payload["train"], ensure_ascii=False),
            "reasoning": str(proposal.get("reasoning") or "").strip()[:400],
            "inherited_from": (incumbent or {}).get("exp_id"),
        })

        rows_before = len(rows)
        run_code, run_waited = _run_with_lock_wait(args, index, deadline)
        if run_code is None:
            # 等到超时也没拿到显卡：不算一轮失败，直接收工
            log("等待显卡超时，停止")
            break

        rows_after = read_tsv(RESULTS_TSV)
        if len(rows_after) <= rows_before:
            stale += 1
            log(f"第 {index} 轮没有产生台账行（退出码 {run_code}），跳过")
            _write_agent_audit(index, {
                "round": index,
                "created_at": now_iso(),
                "outcome": "no_ledger",
                "exit_code": run_code,
                "proposal": proposal,
                "usages": meta.get("usages", []),
            })
            continue

        row = rows_after[-1]
        exp_id = str(row.get("exp_id") or "")
        fitness = _as_float(row.get("fitness_map50_95"))
        incumbent_fitness = incumbent["fitness"] if incumbent else None
        verdict, reason = judge_round(
            fitness, incumbent_fitness, round_threshold,
            threshold_source=noise["source"],
            calibrated=noise["calibrated"],
            group=noise["group"],
            fp_per_image=_as_float(row.get("fp_per_image")),
            incumbent_fp_per_image=_as_float(
                (incumbent.get("row") or {}).get("fp_per_image") if incumbent else None
            ),
        )

        completed += 1
        if verdict == "KEEP":
            kept += 1
            stale = 0
            update_best(exp_id, row, resolved, round_threshold, noise["sigma"])
            log(f"判定 KEEP：{reason}；best.json 已更新为 {exp_id}")
        else:
            stale += 1
            log(f"判定 DISCARD：{reason}")

        delta = None if (fitness is None or incumbent_fitness is None) else round(fitness - incumbent_fitness, 6)
        emit_event({
            "type": "agent",
            "name": "agent_verdict",
            "round": index,
            "exp_id": exp_id,
            "fitness": fitness,
            "delta": delta,
            "threshold": round_threshold,
            "threshold_source": noise["source"],
            "calibrated": noise["calibrated"],
            "verdict": verdict,
            "reason": reason,
            "stale": stale,
        })

        _write_agent_audit(index, {
            "round": index,
            "created_at": now_iso(),
            "exp_id": exp_id,
            "outcome": "done",
            "verdict": verdict,
            "reason": reason,
            "fitness": fitness,
            "incumbent_fitness": incumbent_fitness,
            "delta": delta,
            "threshold": round_threshold,
            "threshold_source": noise["source"],
            "threshold_calibrated": noise["calibrated"],
            "noise_group": noise["group"],
            "inherited_from": (incumbent or {}).get("exp_id"),
            "proposal": proposal,
            "resolved_train": resolved.get("train", {}),
            "run_exit_code": run_code,
            "lock_waits": run_waited,
            "usages": meta.get("usages", []),
            "attempts": meta.get("attempts"),
        })

    elapsed = round(time.time() - started_at, 1)
    log(f"agent 结束：完成 {completed} 轮，KEEP {kept} 次，失败 {errors} 轮，用时 {elapsed}s")
    emit_event({
        "type": "agent",
        "name": "agent_summary",
        "completed": completed,
        "kept": kept,
        "errors": errors,
        "stale": stale,
        "elapsed_seconds": elapsed,
    })
    # 有有效实验就算成功，别让「一个都没留住」变成非零退出码——
    # 无人值守时退出码会被上层当成故障，而全部 discard 是完全正常的科研结果。
    # 但「一轮都没跑成」必须非零：那说明配置坏了，退出码 0 会让人以为跑过了。
    return 0 if completed > 0 or rounds_total == 0 else 1


def evaluate_stop(
    *,
    stale: int,
    max_stale: int,
    errors: int,
    max_errors: int,
    now: float,
    deadline: float | None,
) -> tuple[str, str] | None:
    """该不该收工。返回 ``(原因代码, 给用户看的文案)``，None 表示继续。

    三个刹车分开判，因为含义完全不同：
    - ``deadline``：墙钟到点，正常收工；
    - ``stale``：有结果但没赢 → 搜索可能到瓶颈，该换维度或收工；
    - ``errors``：压根没拿到有效提议 → 是模型/配置的问题，继续问就是白烧钱。

    旧版把后两者塞进同一个计数器，于是「模型输出一直被 max_tokens 截断」
    会被报成「连续 5 轮没有进展」，用户以为是调参到了瓶颈——排查方向完全错。
    """
    if deadline is not None and now > deadline:
        return ("deadline", "到达时间上限，停止")
    if errors >= max_errors:
        return (
            "errors",
            f"连续 {errors} 轮没有拿到有效提议，停止。"
            "这通常不是搜索瓶颈，而是模型侧的问题：输出被 max_tokens 截断、模型名写错、"
            "被限流或余额不足。先看最后一次报错，或回到「模型接口」点一次「测试连通性」。",
        )
    if stale >= max_stale:
        return ("stale", f"连续 {stale} 轮没有进展，停止（省下无意义的 API 花费）")
    return None


def _run_with_lock_wait(
    args: argparse.Namespace,
    index: int,
    deadline: float | None,
) -> tuple[int | None, int]:
    """跑一轮实验；显卡被别人占着时等一会儿再试。

    为什么不能直接失败：无人值守跑一晚，中途在网页上手动训一次很正常。
    那种情况下 agent 应该暂停等待，而不是整晚就此结束。
    返回 ``(退出码, 等待次数)``；``(None, n)`` 表示等到超时。
    """
    wait_seconds = max(0, int(getattr(args, "lock_wait", 3600) or 0))
    waited = 0
    wait_deadline = time.time() + wait_seconds if wait_seconds else None

    while True:
        code = cmd_run(argparse.Namespace(
            config=str(CUR_JSON),
            exp_id=None,
            no_validate=False,
            allow_low_resources=False,
            tee_events=True,
        ))
        if code != 5:
            return code, waited
        waited += 1
        if wait_deadline is not None and time.time() > wait_deadline:
            return None, waited
        if deadline is not None and time.time() > deadline:
            return None, waited
        log("显卡被其它任务占用，60 秒后重试（不会消耗轮数）")
        emit_event({"type": "agent", "name": "agent_wait_gpu", "round": index, "waits": waited})
        time.sleep(60)


def cmd_llm_test(args: argparse.Namespace) -> int:
    """连通性自检：验证 base_url + Key + 模型名三者都对。

    Key 只从环境变量取（``AUTOEXP_API_KEY`` / ``DEEPSEEK_API_KEY``），
    **不提供 --api-key 参数**：命令行会出现在进程列表、日志和运行中条的
    tooltip 里，把 Key 放进 argv 等于把它贴到界面上。
    """
    llm = load_llm()
    if llm is None:
        print(json.dumps({"ok": False, "error": "找不到可用的 llm.py"}, ensure_ascii=False))
        return 1

    overrides: dict[str, Any] = {}
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "temperature", None) is not None:
        overrides["temperature"] = args.temperature

    try:
        result = llm.ping(overrides or None)
    except llm.LLMError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    except Exception as exc:  # noqa: BLE001 - 任何异常都要变成可解析的输出
        print(json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        ))
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoexp.py",
        description="TRTForge YOLO 自主实验循环驱动器（固定时间预算 + 台账 + holdout）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="把 val 集确定性切成 tune / holdout 并建沙箱镜像")
    p_split.add_argument("--data", required=True, help="原始 data.yaml 路径")
    p_split.add_argument("--holdout-frac", type=float, default=0.3, help="holdout 占比，默认 0.3")
    p_split.add_argument("--seed", type=int, default=42, help="切分随机种子，默认 42")
    p_split.add_argument("--force", action="store_true", help="忽略已存在的切分，强制重建")
    p_split.add_argument("--copy", action="store_true", help="用复制代替硬链接（跨卷时用）")
    p_split.add_argument(
        "--group-by",
        default="none",
        choices=("none", "prefix", "dir"),
        help="按组切分 val：同一底板的图整组进同一侧，默认 none（按图片洗牌，与旧版一致）",
    )
    p_split.set_defaults(func=cmd_split)

    p_run = sub.add_parser("run", help="跑一次实验并追加台账")
    p_run.add_argument("--config", required=True, help="实验配置 JSON（会叠加在 configs/base.json 之上）")
    p_run.add_argument("--exp-id", default=None, help="指定实验编号，默认自动递增")
    p_run.add_argument("--no-validate", action="store_true", help="跳过 configs/space.json 校验")
    p_run.add_argument(
        "--allow-low-resources",
        action="store_true",
        help="内存/显存体检不通过时仍然开跑（结果可能因 OOM 失败）",
    )
    p_run.add_argument(
        "--qc-gate",
        action="store_true",
        help="开跑前做数据体检，存在错误级标注问题时拒绝启动（默认不检查）",
    )
    p_run.set_defaults(func=cmd_run)

    p_exec = sub.add_parser("_exec", help="内部：子进程执行一次训练")
    p_exec.add_argument("--snapshot", required=True)
    p_exec.add_argument("--out", required=True)
    p_exec.set_defaults(func=cmd_exec)

    p_hold = sub.add_parser("holdout", help="在 holdout 上验收")
    p_hold.add_argument("--config", default="configs/base.json")
    p_hold.add_argument("--weights", default=None, help="直接指定 .pt 路径")
    p_hold.add_argument("--exp-id", default=None)
    p_hold.add_argument("--split-yaml", default=None)
    p_hold.add_argument("--imgsz", type=int, default=None)
    p_hold.add_argument("--batch", default=None)
    p_hold.add_argument("--device", default=None)
    p_hold.set_defaults(func=cmd_holdout)

    p_hold_exec = sub.add_parser("_holdout_exec", help="内部：子进程执行一次验证")
    p_hold_exec.add_argument("--weights", required=True)
    p_hold_exec.add_argument("--data", required=True)
    p_hold_exec.add_argument("--out", required=True)
    p_hold_exec.add_argument("--imgsz", type=int, default=640)
    p_hold_exec.add_argument("--batch", default="16")
    p_hold_exec.add_argument("--device", default="0")
    p_hold_exec.set_defaults(func=cmd_holdout_exec)

    p_noise = sub.add_parser("noise", help="同配置重复 N 次，测噪声底 sigma")
    p_noise.add_argument("--config", default="configs/base.json")
    p_noise.add_argument("--repeat", type=int, default=3)
    p_noise.set_defaults(func=cmd_noise)

    p_report = sub.add_parser("report", help="打印台账")
    p_report.add_argument("--sort", default="fitness_map50_95")
    p_report.set_defaults(func=cmd_report)

    # 第 4 阶段：LLM 决策层
    p_agent = sub.add_parser("agent", help="让大模型提议参数，连跑多轮（自动 keep/discard）")
    p_agent.add_argument(
        "--rounds", type=int, default=30,
        help="最多跑几轮，默认 30（600 秒预算下约 5.5 小时）",
    )
    p_agent.add_argument(
        "--hours", type=float, default=8.0,
        help="时间上限（小时），默认 8；0 表示不限。与 --rounds 先到者生效",
    )
    p_agent.add_argument(
        "--max-stale", type=int, default=5,
        help="连续多少轮没有 KEEP 就停，默认 5（省下无意义的 API 花费）",
    )
    p_agent.add_argument(
        "--max-errors", type=int, default=3,
        help="连续多少轮拿不到有效提议就停，默认 3（这属于模型侧故障，不是搜索瓶颈）",
    )
    p_agent.add_argument(
        "--lock-wait", type=int, default=3600,
        help="显卡被网页端占用时最多等多少秒，默认 3600；0 表示不等",
    )
    p_agent.add_argument(
        "--dry-run", action="store_true",
        help="只打印第一轮的完整提示词，不调 API、不跑训练",
    )
    p_agent.add_argument(
        "--events", action="store_true",
        help="向 stdout 输出 TRTFORGE_JSON 事件（Web 端据此显示决策与进度）",
    )
    p_agent.set_defaults(func=cmd_agent)

    p_llm_test = sub.add_parser("llm-test", help="测试模型接口连通性（Key 只从环境变量读）")
    p_llm_test.add_argument("--base-url", default=None, help="临时覆盖接口地址")
    p_llm_test.add_argument("--model", default=None, help="临时覆盖模型名")
    p_llm_test.add_argument("--temperature", type=float, default=None)
    p_llm_test.set_defaults(func=cmd_llm_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except PermissionError as exc:
        log(f"安全拦截：{exc}")
        return 3
    except FileNotFoundError as exc:
        log(f"文件缺失：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
