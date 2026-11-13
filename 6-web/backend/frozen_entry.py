# -*- coding: utf-8 -*-
"""安装包里的程序入口。

开发时仍然用 uvicorn 直接跑 app.py，不经过本文件。
打包脚本用 Nuitka 把本文件编译成 TRTForgeWeb.exe：

- 不带参数：走 :mod:`launcher`（读 config.json、设环境、单实例、开浏览器）
- ``--tool val_engine`` 等：执行原来 tools 目录里的那个命令
- ``--tool infer_worker``：推理工作进程（由网页服务自己拉起，不要手动运行）

训练仍由单独的 Python 环境运行 launchers，不进这个程序。
"""

from __future__ import annotations

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
    if name == "native_dialog":
        # 安装包没有 python.exe，不能再用「python -c」弹系统选择框。
        if len(argv) != 2:
            print("native_dialog 需要请求文件和结果文件两个参数", file=sys.stderr)
            return 2
        from nativedialog import serve_dialog
        return serve_dialog(argv[0], argv[1])
    print(f"未知工具：{name}", file=sys.stderr)
    return 2


def main() -> int:
    # 带参数（工具分支）绝不走启动流程：那些分支由服务进程自己拉起，
    # 弹消息框、抢互斥量、开浏览器都会干扰调用方。
    if len(sys.argv) >= 3 and sys.argv[1] == "--tool":
        return _run_tool(sys.argv[2], sys.argv[3:])
    import launcher

    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
