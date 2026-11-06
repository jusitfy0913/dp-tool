#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 Paddle Inference 2.6.2 可加载的老版本 OCR 模型补齐 inference.yml。

官方 tar 包只有 inference.pdmodel 与 inference.pdiparams。
C++ 管线（ocr_infer.dll）还要读 inference.yml：
- 检测缺 DetResizeForTest.resize_long 或 PostProcess.max_candidates 时，创建预测器会抛异常；
- 识别缺 PostProcess.character_dict 时无法解码；
- 方向分类缺 ResizeImage.size 时会直接 exit(-1)，把调用进程一起关掉。

本脚本只在目标目录还没有 inference.yml 时写入。已有文件不覆盖。
识别字典来自仓库内的 ppocr_keys_v1.txt，不另外下载。

用法：
    python tools/gen_legacy_ocr_yml.py
    python tools/gen_legacy_ocr_yml.py --model-root D:\\models\\legacy_v4 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
DEFAULT_ROOT = REPO / "9-python-paddleOCR" / "model" / "legacy_v4"
DEFAULT_DICT = REPO / "ocr" / "PaddleOCR-main" / "ppocr" / "utils" / "ppocr_keys_v1.txt"

DET_YML = """\
Global:
  model_name: PP-OCRv4_mobile_det
PreProcess:
  transform_ops:
  - DecodeImage:
      channel_first: false
      img_mode: BGR
  - DetResizeForTest:
      resize_long: 960
  - NormalizeImage:
      scale: 0.00392156862745098
      mean:
      - 0.485
      - 0.456
      - 0.406
      std:
      - 0.229
      - 0.224
      - 0.225
      order: hwc
  - ToCHWImage: null
PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5
"""

CLS_YML = """\
Global:
  model_name: PP-LCNet_x0_25_textline_ori
PreProcess:
  transform_ops:
  - ResizeImage:
      size:
      - 192
      - 48
  - NormalizeImage:
      scale: 0.00392156862745098
      mean:
      - 0.5
      - 0.5
      - 0.5
      std:
      - 0.5
      - 0.5
      - 0.5
      order: hwc
  - ToCHWImage: null
PostProcess:
  name: Topk
  Topk:
    topk: 1
    label_list:
    - 0_degree
    - 180_degree
"""


def load_dict(path: Path) -> list[str]:
    """按行读取识别字典。只去掉换行，保留行内字符（含引号、空格）。"""
    text = path.read_text(encoding="utf-8")
    chars: list[str] = []
    for line in text.splitlines():
        if line == "":
            continue
        chars.append(line)
    if not chars:
        raise SystemExit(f"字典是空的：{path}")
    return chars


def rec_yml(chars: list[str]) -> str:
    """生成识别模型 yml。每个字符用 JSON 双引号，避免 YAML 把 yes/null 当成布尔。"""
    lines = [
        "Global:",
        "  model_name: PP-OCRv4_mobile_rec",
        "PreProcess:",
        "  transform_ops:",
        "  - DecodeImage:",
        "      channel_first: false",
        "      img_mode: BGR",
        "  - RecResizeImg:",
        "      image_shape:",
        "      - 3",
        "      - 48",
        "      - 320",
        "PostProcess:",
        "  name: CTCLabelDecode",
        "  character_dict:",
    ]
    for ch in chars:
        lines.append("  - " + json.dumps(ch, ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


def write_if_missing(path: Path, content: str, force: bool) -> str:
    if path.is_file() and not force:
        return f"保留已有 {path}"
    path.write_text(content, encoding="utf-8", newline="\n")
    return f"写入 {path}（{path.stat().st_size} 字节）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为老版本 OCR 模型补齐 inference.yml")
    parser.add_argument("--model-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--dict", dest="dict_path", default=str(DEFAULT_DICT))
    parser.add_argument("--force", action="store_true", help="覆盖已有 inference.yml")
    args = parser.parse_args(argv)

    root = Path(args.model_root)
    if not root.is_dir():
        print(f"模型根目录不存在：{root}", file=sys.stderr)
        return 1
    dict_path = Path(args.dict_path)
    if not dict_path.is_file():
        print(f"字典不存在：{dict_path}", file=sys.stderr)
        return 1

    targets = {
        "ch_PP-OCRv4_det_infer": DET_YML,
        "ch_PP-OCRv4_rec_infer": rec_yml(load_dict(dict_path)),
        "ch_ppocr_mobile_v2.0_cls_infer": CLS_YML,
    }
    missing = [name for name in targets if not (root / name).is_dir()]
    if missing:
        print("缺少模型目录：" + ", ".join(missing), file=sys.stderr)
        return 1

    for name, content in targets.items():
        print(write_if_missing(root / name / "inference.yml", content, args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
