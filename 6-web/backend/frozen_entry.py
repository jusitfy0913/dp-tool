# -*- coding: utf-8 -*-
"""安装包里的程序入口。

开发时仍然用 uvicorn 直接跑 app.py，不经过本文件。
打包脚本用 Nuitka 把本文件编译成 TRTForgeWeb.exe：

- 不带参数：启动网页服务
- ``--tool val_engine`` 等：执行原来 tools 目录里的那个命令
- ``--tool infer_worker``：推理工作进程（由网页服务自己拉起，不要手动运行）

训练仍由单独的 Python 环境运行 launchers，不进这个程序。
"""

from __future__ import annotations

import os
import sys


def _run_tool(name: str, argv: list[str]) -> int:
    """按名字调用已经编进本程序的工具。argv 不含程序名。"""
    if name == "val_engine":
        from tools.val_engine import main
        return main(argv)
    if name == "val_anomaly":
        from tools.val_anomaly import main
        return main(argv)
    if name == "infer_batch":
        from tools.infer_batch import main
        sys.argv = [name, *argv]
        return main(argv)
    if name == "infer_video":
        from tools.infer_video import main
        sys.argv = [name, *argv]
        return main(argv)
    if name == "infer_worker":
        from infer_worker import main
        return main(argv)
    if name == "int8_launch":
        from tools.int8_launch import main
        return main(argv)
    print(f"未知工具：{name}", file=sys.stderr)
    return 2


def _serve() -> int:
    """按环境变量里的地址启动网页。地址由安装目录的启动脚本写入。"""
    import uvicorn

    import app as app_module

    host = os.environ.get("TRTFORGE_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("TRTFORGE_WEB_PORT", "8060"))
    uvicorn.run(app_module.app, host=host, port=port, log_level="info")
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--tool":
        return _run_tool(sys.argv[2], sys.argv[3:])
    return _serve()


if __name__ == "__main__":
    raise SystemExit(main())
