# -*- coding: utf-8 -*-
"""安装包里的程序入口。

开发时仍然用 uvicorn 直接跑 app.py，不经过本文件。
打包脚本用 Nuitka 把本文件编译成 TRTForgeWeb.exe：

- 不带参数：走 :mod:`launcher`（读 config.json、设环境、单实例、开窗口或浏览器）
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


def _selfcheck() -> int:
    """安装包自检：起服务前必须能 import 的东西，挨个真 import 一遍。

    为什么需要：Nuitka 打包只保证「模块代码进了 exe」，不保证它依赖的 **DLL** 也在。
    Conda 的 Pillow 把编译库链到 `<env>\\Library\\bin`（tiff.dll / openjp2.dll 之类），
    少一个就 `import PIL.Image` 失败，而报错信息是空的 `ImportError`，
    用户看到的就是双击闪退。v1.40.0 打包时踩过这个坑，所以让打包脚本最后跑一次本自检：
    缺库当场失败，不把半成品发给用户。
    """
    import traceback  # noqa: PLC0415

    # 顺序按「先底层、后装配」：哪一层缺库能一眼看出来，不用顺着 traceback 猜。
    modules = (
        "numpy",
        "cv2",
        "PIL.Image",
        "onnxruntime",
        "fastapi",
        "uvicorn",
        # 桌面窗口（WebView2）。它是**打包硬门槛**：缺了就不许出包——窗口模式
        # 是双击后的默认形态，运行期虽然会退回浏览器，但那等于把默认体验做没了。
        # 注意运行期可降级不代表打包可以缺，两个判断不要混。
        # 实测（2026-11-16）单独 import webview 不够：它连带 pythonnet/clr_loader，
        # 而 _ctypes 需要 ffi.dll、ssl 需要 libssl/libcrypto，PE 依赖闭包必须补齐，
        # 否则这里会报一个空 ImportError。
        "webview",
        # 这一行才是真正的门槛：它会把 pythonnet、.NET 程序集、WinForms 一起拉起来。
        # 只 import webview 不碰 winforms，缺 .NET 也照样「通过」。
        "webview.platforms.winforms",
        "desktop",
        "routers",
        "launcher",
        "app",  # 最后一步：它会把全部路由与 datasetqc 一起拉进来
    )
    failed: list[str] = []
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - 自检要把任何异常都报出来
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        else:
            print(f"OK   {name}", flush=True)
    if failed:
        print("自检未通过：" + ", ".join(failed), flush=True)
        return 1
    print("自检通过", flush=True)
    return 0


def main() -> int:
    # 带参数（工具分支）绝不走启动流程：那些分支由服务进程自己拉起，
    # 弹消息框、抢互斥量、开浏览器都会干扰调用方。
    if len(sys.argv) >= 3 and sys.argv[1] == "--tool":
        if sys.argv[2] == "selfcheck":
            return _selfcheck()
        return _run_tool(sys.argv[2], sys.argv[3:])
    import launcher

    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
