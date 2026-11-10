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
import csv
import importlib.util
import json
import os
import random
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
    "epochs_done",
    "train_seconds",
    "peak_vram_mb",
    "imgsz",
    "batch_req",
    "batch_eff",
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
        },
    )
    log(f"完成：train={len(train_images)} tune={len(tune)} holdout={len(holdout)}（{train_mode}/{tune_mode}/{holdout_mode}）")
    return 0


# ---------------------------------------------------------------------------
# run / _exec
# ---------------------------------------------------------------------------


def load_gpulock() -> Any:
    """加载 6-web 的 ``gpulock``（零第三方依赖，只用标准库）。

    为什么复用 Web 那一份而不是自己再写一遍：锁文件的格式是两边共用的契约，
    两份实现迟早会漂移（一边加字段、一边改判定），而漂移的表现是
    「两个训练都以为自己是唯一持有者」——正好是最危险的失效方式。

    为什么放在 6-web/backend：Web 打包脚本会把整个 backend 目录一起冻结，
    放在那里才不会漏文件。取不到就返回 None，由调用方降级为「不做互斥」，
    不要让一把锁成为训练跑不起来的原因。
    """
    backend = ROOT.parent / "6-web" / "backend"
    if not (backend / "gpulock.py").is_file():
        return None
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    try:
        import gpulock  # noqa: PLC0415 - 刻意延迟导入，缺文件时才能降级

        return gpulock
    except Exception:  # noqa: BLE001 - 互斥失败不该阻断训练
        return None


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
    if lock is not None:
        holder = lock.read()
        if holder is not None and int(holder.get("pid") or 0) != os.getpid():
            log(f"显卡正被占用：{lock.describe(holder)}")
            log("本次不启动。等它结束，或到那个页面/终端里停止后重试。")
            return 5
        lock_held, lock_reason = lock.acquire("autoexp", "autoexp-run", args.exp_id or "")
        if lock_held and lock_reason:
            log(f"注意：跨进程互斥未生效——{lock_reason}")
    else:
        log("注意：未找到 6-web/backend/gpulock.py，本次不做跨进程互斥")

    try:
        return _run_locked(args, cfg, config_path, lock, lock_held)
    finally:
        # 训练子进程已退出（或被打断），立刻让出 GPU 位置。
        # 放在 finally 里：按 Ctrl+C 中途退出也要释放，否则锁会挂到进程死亡才清。
        if lock is not None and lock_held:
            lock.release()


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

    snapshot = {
        "exp_id": exp_id,
        "created_at": now_iso(),
        "config_source": str(config_path),
        "hypothesis": cfg.get("hypothesis", ""),
        "changed": cfg.get("changed", ""),
        "budget_seconds": int(cfg.get("budget_seconds", 600)),
        "resources_at_start": resources,
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
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# {exp_id} {snapshot['created_at']}\n")
        log_handle.write(f"# cmd: {' '.join(cmd)}\n\n")
        log_handle.flush()
        # 锁的释放在 cmd_run 的 finally 里，这里不再重复：
        # 两处都放会让人以为「必须两层救援」，实际只需最外层那一处。
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
    elapsed = round(time.time() - started, 1)

    result_path = exp_dir / "result.json"
    result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            result = load_json(result_path)
        except (OSError, ValueError):
            result = {}

    status = result.get("status") or ("crash" if proc.returncode != 0 else "ok")
    if status == "ok":
        status = "done"
    if not result:
        result = {"status": "crash", "error": f"子进程退出码 {proc.returncode}，且未生成 result.json"}

    row = {
        "exp_id": exp_id,
        "created_at": snapshot["created_at"],
        # 适应度取 results.csv 最后一轮，不是历史最佳：固定预算下比的就是到点那一刻的水平
        "fitness_map50_95": result.get("map50_95", ""),
        "best_map50_95": result.get("best_map50_95", ""),
        "map50": result.get("map50", ""),
        "precision": result.get("precision", ""),
        "recall": result.get("recall", ""),
        "epochs_done": result.get("epochs_done", ""),
        "train_seconds": result.get("train_seconds", elapsed),
        "peak_vram_mb": result.get("peak_vram_mb", ""),
        "imgsz": (cfg.get("train") or {}).get("imgsz", cfg.get("imgsz", "")),
        "batch_req": result.get("batch_requested", (cfg.get("train") or {}).get("batch", "")),
        "batch_eff": result.get("batch_effective", ""),
        "model": cfg.get("model", ""),
        "budget_seconds": snapshot["budget_seconds"],
        "status": status,
        "hypothesis": cfg.get("hypothesis", ""),
        "changed": cfg.get("changed", ""),
    }
    append_tsv(RESULTS_TSV, RESULTS_HEADER, row)

    # batch 被 Ultralytics 自动降过，说明这次实验与请求的参数不是同一件事，
    # 后续和其它实验横向比较会失真，必须显式提示。
    if status == "done":
        req, eff = str(row["batch_req"]), str(row["batch_eff"])
        if req and eff and req != eff:
            log(f"注意：batch 被自动从 {req} 降到 {eff}（首轮 OOM）。本次结果与请求参数不等价，跨实验比较需谨慎。")

    fitness = row["fitness_map50_95"]
    log(f"{exp_id} 结束：status={status} mAP50-95={fitness} epochs={row['epochs_done']} 用时={elapsed}s")
    if status == "crash":
        log(f"失败详情见 {log_path}")
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


def preflight_vram(imgsz: int, batch: int, need_mb: float | None = None) -> tuple[bool, str]:
    """开训前的显存预检。

    Ultralytics 的首轮 OOM 会自动把 batch 减半重来，这会**悄悄改变实验条件**：
    batch 一变，nbs 归一化后的等效学习率、正则强度全跟着变，于是这一次实验
    与台账里其它实验不再可比。所以宁可提前拒绝，也不要让它自动降级。

    估算依据（本机 T1000 8GB 实测）：
      - batch=8  @640 峰值约 1769 MB（PyTorch 统计值，不含 CUDA context / cudnn workspace）
      - batch=16 @640 在 5302 MB 可用时仍然 OOM
      - 这张卡在 Ultralytics 的 AMP 黑名单里，训练强制走 fp32，激活显存显著高于常规
    因此这里的估算刻意保守：它是「显存紧时别开跑」的闸门，不是精确预测。
    """
    free = free_vram_mb()
    if free is None:
        return True, "无法读取显存（可能无 CUDA），跳过预检"
    # 2600 MB 固定开销（CUDA context + cudnn workspace + fp32 缓冲），
    # 240 MB/样本随 batch 与分辨率线性增长
    estimate = need_mb if need_mb is not None else 2600.0 + 240.0 * max(batch, 1) * (imgsz / 640.0) ** 2
    if free < estimate:
        return False, (
            f"可用显存 {free:.0f} MB < 保守需求 {estimate:.0f} MB"
            f"（imgsz={imgsz} batch={batch}）。"
            "请先释放显存（例如停掉 6-web 的 infer_worker 常驻进程），或把 batch 调小。"
        )
    return True, f"可用显存 {free:.0f} MB ≥ 保守需求 {estimate:.0f} MB"


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
        ok, reason = preflight_vram(int(kwargs.get("imgsz", 640)), max(batch_int, 1))
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
        model.train(**kwargs)

        save_dir = Path(getattr(model.trainer, "save_dir", out_dir))
        metrics = read_results_csv(save_dir / "results.csv")
        # 实际生效的 batch：与请求值不同即说明被自动降级过
        batch_effective = None
        try:
            batch_effective = int(getattr(model.trainer, "batch_size", 0) or 0) or None
        except (TypeError, ValueError):
            batch_effective = None
        peak_vram = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        except Exception:  # noqa: BLE001 - 显存读数失败不影响结论
            peak_vram = None

        payload = {
            "status": "done",
            "exp_id": exp_id,
            "save_dir": str(save_dir),
            "weights": str(save_dir / "weights" / "best.pt"),
            "peak_vram_mb": peak_vram,
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
    save_json(
        ROOT / "experiments" / "_noise_summary.json",
        {"created_at": now_iso(), "values": values, "mean": mean, "sigma": sigma, "spread": spread},
    )
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
