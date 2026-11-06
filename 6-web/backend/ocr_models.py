# -*- coding: utf-8 -*-
"""OCR 模型目录扫描与进动态库前的预检。

C++ 管线遇到 yml 缺字段时，有的路径会抛异常，有的会直接 exit(-1)。
后者会把整个网页进程关掉，所以任何模型问题都必须在调用 oi_create 之前拦住。

扫描根目录：
- 环境变量 TRTFORGE_OCR_MODEL_DIR
- 否则开发默认 <repo>/9-python-paddleOCR/model
- 安装包默认不在这里猜，由启动脚本设置环境变量
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import config

_VERSION_RE = re.compile(r"Paddle version:\s*(\d+)\.(\d+)(?:\.(\d+))?", re.IGNORECASE)


class OcrModelError(ValueError):
    """模型目录或配置不合格。调用方应把它变成 HTTP 400，不要进 DLL。"""


def model_root() -> Path:
    """OCR 模型扫描根。环境变量优先，开发环境回落到仓库里的下载目录。"""
    raw = os.environ.get("TRTFORGE_OCR_MODEL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (config.REPO_ROOT / "9-python-paddleOCR" / "model").resolve()


def _version_files() -> list[Path]:
    return [
        config.BIN_DIR / "paddle_inference_version.txt",
        config.REPO_ROOT / "ocr" / "ocr_infer" / "third_party" / "paddle_inference" / "version.txt",
    ]


def paddle_version() -> tuple[int | None, str]:
    """返回 (主版本号, 原始版本串)。读不到时主版本为 None，按不兼容处理 json 模型。"""
    for path in _version_files():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = _VERSION_RE.search(text)
        if match:
            return int(match.group(1)), match.group(0).split(":", 1)[-1].strip()
    return None, ""


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise OcrModelError("读取 inference.yml 需要 PyYAML（pip install PyYAML）") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 把解析错误变成可读原因
        raise OcrModelError(f"无法解析 {path.name}：{exc}") from exc
    if not isinstance(data, dict):
        raise OcrModelError(f"{path.name} 顶层不是映射")
    return data


def _flatten(node, prefix: str = "") -> dict:
    """把嵌套 yml 摊成「点路径 → 值」，列表同时保留整体和每个元素。"""
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child))
    elif isinstance(node, list):
        out[prefix] = node
        for index, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}[{index}]"))
    else:
        if prefix:
            out[prefix] = node
    return out


def _has(flat: dict, suffix: str) -> bool:
    return any(key == suffix or key.endswith("." + suffix) for key in flat)


def _value(flat: dict, suffix: str):
    for key, value in flat.items():
        if key == suffix or key.endswith("." + suffix):
            return value
    return None


def classify_name(model_name: str) -> str:
    """按 Global.model_name 归类。顺序不能反：textline 里也有 ori。"""
    name = (model_name or "").strip().lower()
    if not name:
        return "unknown"
    if "textline" in name:
        return "textline_ori"
    if "doc_ori" in name or "doc_text_ori" in name:
        return "doc_ori"
    if "uvdoc" in name or "unwarp" in name:
        return "unwarp"
    if name.endswith("_det") or name.endswith("-det"):
        return "det"
    if name.endswith("_rec") or name.endswith("-rec"):
        return "rec"
    return "unknown"


def model_format(directory: Path) -> str:
    """cpp_infer 优先加载 inference.json，两者并存时按 json 判定。"""
    if (directory / "inference.json").is_file():
        return "json"
    if (directory / "inference.pdmodel").is_file():
        return "pdmodel"
    return ""


def _yml_path(directory: Path) -> Path | None:
    for name in ("inference.yml", "inference.yaml"):
        path = directory / name
        if path.is_file():
            return path
    return None


def inspect_dir(directory: Path, paddle_major: int | None) -> dict:
    """检查一个模型目录，返回给下拉框和预检共用的条目。"""
    directory = directory.resolve()
    yml = _yml_path(directory)
    fmt = model_format(directory)
    entry = {
        "path": str(directory),
        "name": directory.name,
        "model_name": "",
        "kind": "unknown",
        "format": fmt or "missing",
        "compatible": False,
        "reason": "",
    }
    reasons: list[str] = []
    if not fmt:
        reasons.append("缺少 inference.pdmodel 或 inference.json")
    if not (directory / "inference.pdiparams").is_file():
        reasons.append("缺少 inference.pdiparams")
    if yml is None:
        reasons.append("缺少 inference.yml")
        entry["reason"] = "；".join(reasons)
        return entry

    try:
        data = _load_yaml(yml)
    except OcrModelError as exc:
        entry["reason"] = str(exc)
        return entry

    model_name = str((data.get("Global") or {}).get("model_name") or "").strip()
    entry["model_name"] = model_name
    entry["kind"] = classify_name(model_name)
    flat = _flatten(data)
    reasons.extend(_missing_keys(entry["kind"], flat))

    if fmt == "json" and (paddle_major is None or paddle_major < 3):
        version = "未知" if paddle_major is None else str(paddle_major)
        reasons.append(f"inference.json 需要 Paddle 3.x，当前主版本是 {version}")

    entry["reason"] = "；".join(reasons)
    entry["compatible"] = not reasons and entry["kind"] != "unknown"
    if entry["kind"] == "unknown" and not entry["reason"]:
        entry["reason"] = f"无法从 model_name={model_name or '空'} 判断模型类型"
    return entry


def _missing_keys(kind: str, flat: dict) -> list[str]:
    missing: list[str] = []
    if kind == "det":
        if not _has(flat, "DetResizeForTest.resize_long"):
            missing.append("缺少 DetResizeForTest.resize_long")
        if not _has(flat, "PostProcess.max_candidates"):
            missing.append("缺少 PostProcess.max_candidates")
    elif kind == "rec":
        chars = _value(flat, "PostProcess.character_dict")
        if not isinstance(chars, list) or not chars:
            missing.append("缺少 PostProcess.character_dict")
    elif kind == "textline_ori":
        size = _value(flat, "ResizeImage.size")
        if not isinstance(size, list) or len(size) < 2:
            missing.append("缺少 PreProcess.ResizeImage.size")
        for suffix in ("NormalizeImage.scale", "NormalizeImage.mean", "NormalizeImage.std"):
            if not _has(flat, suffix):
                missing.append(f"缺少 {suffix}")
        labels = _value(flat, "PostProcess.Topk.label_list")
        if not isinstance(labels, list) or not labels:
            missing.append("缺少 PostProcess.Topk.label_list")
        if not _has(flat, "PostProcess.Topk.topk"):
            missing.append("缺少 PostProcess.Topk.topk")
    elif kind in ("doc_ori", "unwarp"):
        missing.append("文档方向与弯曲矫正需要 Paddle 3.x，第一期不启用")
    return missing


def scan_models(root: Path | None = None) -> dict:
    """扫描根目录下两层，凡是含 inference.yml 的目录都算一个模型。"""
    root = (root or model_root()).resolve()
    major, version = paddle_version()
    entries: list[dict] = []
    if root.is_dir():
        candidates = [root]
        candidates.extend(path for path in root.iterdir() if path.is_dir())
        for child in list(candidates):
            if child == root:
                continue
            candidates.extend(path for path in child.iterdir() if path.is_dir())
        seen: set[Path] = set()
        for directory in candidates:
            resolved = directory.resolve()
            if resolved in seen or _yml_path(resolved) is None:
                continue
            seen.add(resolved)
            entries.append(inspect_dir(resolved, major))
    entries.sort(key=lambda item: (item["kind"], item["name"]))
    return {
        "root": str(root),
        "paddle_version": version,
        "paddle_major": major,
        "entries": entries,
    }


def preflight(det_dir: str, rec_dir: str, textline_dir: str = "") -> None:
    """不通过就抛 OcrModelError。通过则什么都不返回。"""
    major, _version = paddle_version()
    for label, raw, expected in (
        ("检测", det_dir, "det"),
        ("识别", rec_dir, "rec"),
    ):
        path = Path(raw or "").expanduser()
        if not str(raw or "").strip() or not path.is_dir():
            raise OcrModelError(f"{label}模型目录不存在：{raw}")
        entry = inspect_dir(path, major)
        if entry["kind"] != expected:
            raise OcrModelError(
                f"{label}模型类型是 {entry['kind']}，需要 {expected}（{entry['model_name'] or path.name}）"
            )
        if not entry["compatible"]:
            raise OcrModelError(f"{label}模型不可用：{entry['reason'] or path}")

    if str(textline_dir or "").strip():
        path = Path(textline_dir).expanduser()
        if not path.is_dir():
            raise OcrModelError(f"方向模型目录不存在：{textline_dir}")
        entry = inspect_dir(path, major)
        if entry["kind"] != "textline_ori":
            raise OcrModelError(f"方向模型类型是 {entry['kind']}，需要 textline_ori")
        if not entry["compatible"]:
            raise OcrModelError(f"方向模型不可用：{entry['reason'] or path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查一个 OCR 模型目录能不能给 ocr_infer.dll 用")
    parser.add_argument("--check", required=True, help="模型目录")
    args = parser.parse_args(argv)
    major, version = paddle_version()
    entry = inspect_dir(Path(args.check), major)
    print(f"paddle={version or '未知'} major={major}")
    for key in ("path", "model_name", "kind", "format", "compatible", "reason"):
        print(f"{key}={entry[key]}")
    return 0 if entry["compatible"] else 2


if __name__ == "__main__":
    sys.exit(main())
