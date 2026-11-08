#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全流程里的 INT8 编译入口。

导出 ONNX 之后才运行，所以这时模型文件已经在。由它决定沿用缓存还是重新校准，
再把 trtexec 或 int8_calib.exe 的输出原样打到自己的标准输出，进度条才能跟上。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import int8calib  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按缓存情况编译 YOLO INT8 引擎")
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--list", required=True, help="候选图片清单，一行一个路径")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workspace", type=int, default=4096)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--min", default="")
    parser.add_argument("--opt", default="")
    parser.add_argument("--max", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    images = []
    for line in Path(args.list).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            images.append(Path(text))
    shapes = None
    if args.min and args.opt and args.max:
        shapes = (args.min, args.opt, args.max)
    try:
        command = int8calib.prepare_command(
            trtexec=args.trtexec,
            onnx_path=Path(args.onnx),
            engine_path=Path(args.engine),
            images=images,
            count=args.count,
            workspace_mb=args.workspace,
            input_name=args.input_name,
            shapes=shapes,
        )
    except int8calib.Int8Error as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("$ " + subprocess.list2cmdline(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
