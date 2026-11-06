# -*- coding: utf-8 -*-
"""安装包开关：没设 TRTFORGE_TOOL_EXE 时，命令必须仍是 python + tools/*.py。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402


def main() -> int:
    old = os.environ.pop("TRTFORGE_TOOL_EXE", None)
    try:
        command = config.tool_command(r"C:\Python\python.exe", "infer_batch.py")
        assert command[0] == r"C:\Python\python.exe", command
        assert command[1].endswith("infer_batch.py"), command
        assert len(command) == 2, command

        os.environ["TRTFORGE_TOOL_EXE"] = r"D:\TRTForge\app\TRTForgeWeb.exe"
        packed = config.tool_command(r"C:\Python\python.exe", "val_engine.py")
        assert packed == [
            r"D:\TRTForge\app\TRTForgeWeb.exe",
            "--tool",
            "val_engine",
        ], packed
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_TOOL_EXE", None)
        else:
            os.environ["TRTFORGE_TOOL_EXE"] = old
    print("tool_command ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
