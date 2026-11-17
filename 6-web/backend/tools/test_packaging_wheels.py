# -*- coding: utf-8 -*-
"""安装包「自带依赖库」的验收脚本：证明打包脚本与安装脚本没有偷偷退回去让人联网装。

为什么要有它
------------
规矩是「发给使用人员的安装包必须自带训练库，对方机器不能靠联网 pip install」。
这条规矩有个很自然的退化路径：

1. 代码里新 `import` 了一个库，打包脚本忘了加 → 用户点「开始」才 ModuleNotFoundError；
2. 有人图省事把这些库写进 ``--nofollow-import-to`` 就从安装包里去掉了；
3. ``pip install`` 那条路是联网的，开发机有网、对方没有 → 交付现场才发现。

所以这里不跑 PowerShell，只读脚本内容做静态核对：**缺一个库就失败**。

用法：
    python 6-web/backend/tools/test_packaging_wheels.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from _harness import check, finish  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
PACKAGING = REPO / "packaging"
WEB_SCRIPT = PACKAGING / "package_web_user.ps1"
INSTALL_SCRIPT = PACKAGING / "templates" / "安装训练环境.ps1"


def read(path: Path) -> str:
    # PowerShell 5.1 读无 BOM 文件按 ANSI，所以这些脚本一律带 BOM 或纯 ASCII。
    # 这里按 utf-8-sig 读，两种都能吃。
    return path.read_text(encoding="utf-8-sig")


def array_items(text: str, name: str) -> list[str]:
    """取出 `$name = @( ... )` 里的字符串字面量。"""
    match = re.search(r"\$" + re.escape(name) + r"\s*=\s*@\((.*?)\)", text, re.S)
    if not match:
        return []
    return re.findall(r'"([^"]*)"', match.group(1))


#: 必须在安装包里出现、能被目标解释器 import 的发行版（wheel 文件名前缀）
MUST_SHIP = {
    "torch": "训练/建库全靠它",
    "torchvision": "torch 的配套",
    "ultralytics": "YOLO 训练",
    "anomalib": "异常检测训练",
    "timm": "AnomalyDINO 的 DINOv2 主干",
    "onnx": "导出 ONNX",
    "onnxruntime": "ONNX 兜底推理",
    "polars": "数据集处理",
    "polars_runtime_compat": "无 AVX2 机器上的 Polars",
}


def test_wheels_listed(text: str) -> None:
    required = array_items(text, "requiredWheels")
    check("打包脚本：找到 requiredWheels 清单", len(required) >= 5, str(required))
    for name, why in MUST_SHIP.items():
        check(f"打包脚本：requiredWheels 含 {name}（{why}）", name in required)

    reqs = array_items(text, "wheelRequirements")
    check("打包脚本：找到 wheelRequirements 清单", len(reqs) >= 5, str(reqs))
    joined = "\n".join(reqs)
    for name in MUST_SHIP:
        prefix = name.replace("_", "-")
        check(f"打包脚本：wheelRequirements 含 {prefix}", prefix in joined)

    # 版本必须钉死，否则今天打的和明天打的不是同一套
    check("打包脚本：torch 钉死 2.3.0+cu118", "torch==2.3.0+cu118" in joined)
    check("打包脚本：torchvision 钉死 0.18.0+cu118", "torchvision==0.18.0+cu118" in joined)
    check("打包脚本：timm 钉死 1.0.14", "timm==1.0.14" in joined)
    check("打包脚本：anomalib 钉死 1.2.0", "anomalib==1.2.0" in joined)
    check("打包脚本：ultralytics 钉死 8.4.192", "ultralytics==8.4.192" in joined)


def test_avx2_trap(text: str) -> None:
    """polars-runtime-32 会在没有 AVX2 的机器上直接崩，必须只带 compat 版。"""
    joined = "\n".join(array_items(text, "wheelRequirements"))
    check("打包脚本：带 polars-runtime-compat", "polars-runtime-compat==" in joined)
    check("打包脚本：不带 polars-runtime-32", "polars-runtime-32" not in joined)
    install = read(INSTALL_SCRIPT)
    check(
        "安装脚本：装完卸掉 polars-runtime-32",
        "uninstall -y polars-runtime-32" in install,
    )
    # 卸掉之后必须真的验证 import polars（compat 顶替是否生效，只能这样验）。
    # v1.40.0 起自检改成逐个 import，"polars" 出现在模块清单里即等价于旧的那行自检。
    polars_check = re.search(r"mods = \[(.*?)\]", install, re.S)
    check(
        "安装脚本：卸完 polars-runtime-32 仍 import polars",
        polars_check is not None and '"polars"' in polars_check.group(1),
    )
    # wheels 里那份 runtime-32 必须留着：pip 离线解依赖要用它（已实测：
    # 删掉它 pip 直接报「找不到 polars-runtime-32」，离线装 ultralytics 失败）
    check(
        "安装脚本：不删除 wheels 里的 runtime-32（pip 解依赖需要）",
        "polars_runtime_32-*.whl" not in install,
    )


def test_missing_wheels_stops(text: str) -> None:
    """缺库必须 throw 停止出 zip，而不是打个半成品继续。"""
    check("打包脚本：缺 wheel 直接 throw", "离线依赖包不全" in text)
    check(
        "打包脚本：出 zip 前检查 wheels/manifest.txt",
        "wheels\\manifest.txt" in text,
    )
    check(
        "打包脚本：出 zip 前检查嵌入式 Python",
        "runtime\\python-embed.zip" in text,
    )


def test_install_script_offline() -> None:
    install = read(INSTALL_SCRIPT)
    check("安装脚本：离线安装 pip 参数", "--no-index" in install and "--find-links" in install)
    check("安装脚本：离线时装 timm", "timm==1.0.14" in install)

    # 自检必须真的 import，否则「装上了」只是文件在，import 仍可能失败。
    # v1.40.0 起改成逐个 import（一长串里第一个失败就中断，后面缺什么看不见），
    # 所以这里改为核对自检脚本里的模块清单。
    match = re.search(r"mods = \[(.*?)\]", install, re.S)
    check("安装脚本：自检有模块清单", match is not None)
    modules = set(re.findall(r'"([^"]+)"', match.group(1))) if match else set()
    # 训练/推理真正会用到的全部第三方库，缺一个都该在安装时红，而不是等用户点训练
    for name in ("torch", "timm", "polars", "onnx", "onnxruntime", "cv2", "ultralytics", "anomalib"):
        check(f"安装脚本：自检 import {name}", name in modules, str(sorted(modules)))

    # 自检要能报出「具体缺哪个」，不能只留一个笼统的失败
    check("安装脚本：自检打印 FAIL 明细", '"FAIL %s:' in install)
    check("安装脚本：自检汇总 MISSING", '"MISSING: "' in install)
    # PowerShell 5.1 传 -c 会吃掉双引号（实测变成 SyntaxError），所以自检必须落临时文件再跑。
    # 只盯自检这一块：脚本别处的 `$Python -c "import importlib.metadata..."` 是单行、
    # 无嵌套引号，安全，不能一起禁掉。
    selfcheck_block = re.search(r"\$selfCheck = @'.*?Remove-Item -LiteralPath \$selfCheckPath.*?\n", install, re.S)
    check("安装脚本：自检写临时 .py 再执行（不用 python -c）", selfcheck_block is not None and "trtforge_selfcheck.py" in install)
    check(
        "安装脚本：自检执行处不带 -c",
        selfcheck_block is not None and "-c $selfCheck" not in selfcheck_block.group(0),
    )
    check("安装脚本：自检完删临时文件", "Remove-Item -LiteralPath $selfCheckPath" in install)


def test_launcher_packaging(text: str) -> None:
    """AnomalyDINO 的算法模块必须编进同一个 .pyd，且要有东西拦住「漏编」。"""
    check("打包脚本：编译 anomalydino 启动器", "tf_anomalydino_launcher_bin" in text)
    check("打包脚本：带 anomalydino_core 一起编", "anomalydino_core.py" in text)
    # --module 模式不会自动跟随同级 .py，只能 --include-module
    check("打包脚本：辅助模块用 --include-module", '"--include-module=$name"' in text)
    check("打包脚本：按 report.xml 核对辅助模块", "nuitka-report.xml" in text)
    check("打包脚本：缺辅助模块就 throw", "编译产物缺少辅助模块" in text)
    check(
        "打包脚本：三个启动器的 .pyd 都要在",
        '"tf_yolo_launcher_bin", "tf_anomalib_launcher_bin", "tf_anomalydino_launcher_bin"' in text,
    )


#: nofollow 里的每个名字，实际由哪个 wheel 提供。空字符串表示「随 anomalib 一起装进来」。
NOFOLLOW_SUPPLIER = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "anomalib": "anomalib",
    "lightning": "",          # anomalib 的必需依赖，pip 解析时一并下来
    "pytorch_lightning": "",  # 同上
    "cv2": "opencv_python",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "timm": "timm",
}


def test_nofollow_not_a_way_out(text: str) -> None:
    """nofollow 只允许跟「wheels 里确实装了」的库配套使用。

    这条不是禁止 nofollow（torch 2.6 GB 冻进 .pyd 不现实），而是要求每个被排除的库
    都能在 wheelhouse 里找到对应 wheel —— 否则就是「包里有、对方却 import 不到」。
    """
    match = re.search(r"\$nofollow\s*=\s*@\((.*?)\)", text, re.S)
    check("打包脚本：找到 nofollow 清单", match is not None)
    if not match:
        return
    names = re.findall(r'"([^"]*)"', match.group(1))
    required = array_items(text, "requiredWheels")
    for name in names:
        supplier = NOFOLLOW_SUPPLIER.get(name, "?")
        if supplier == "?":
            check(f"打包脚本：nofollow 的 {name} 有 wheel 兜底", False, "清单里没登记")
            continue
        if supplier == "":
            ok = "anomalib" in required
        else:
            ok = supplier in required
        check(f"打包脚本：nofollow 的 {name} 由 wheel 兜底", ok, supplier)
    before = text[: match.start()]
    check(
        "打包脚本：nofollow 前有解释「由 wheels 提供」",
        "由 wheels" in before[-600:] or "wheels\\ 里" in before[-600:],
    )
    # torch 不允许被冻进 .pyd
    check("打包脚本：torch 在 nofollow 里（由 wheel 提供）", "torch" in names)


def test_runtime_dlls_and_selfcheck(text: str) -> None:
    """安装包必须自带 .pyd 依赖的 DLL，并且装完要真跑一遍 import。

    为什么盯这两条：v1.40.0 打包时踩过 —— Conda 的 Pillow/_ctypes/_ssl 把编译库
    链到 ``<env>\\Library\\bin``（tiff.dll、openjp2.dll、ffi.dll……），Nuitka 不带。
    结果安装包双击即崩（0xC0000409，日志一个字都没有），开发机却一切正常，
    因为开发机的 PATH 里有 conda 的 Library\\bin。

    防线有两道，缺一不可：
    1. 扫 PE 导入表做传递闭包，缺什么从编译环境补（不能手写清单，早晚漏）；
    2. 打完在安装目录里真跑 ``--tool selfcheck``，import 不过就 throw。
    """
    check("打包脚本：有 PE 依赖扫描函数", "Get-PeImportedDlls" in text)
    check("打包脚本：用 dumpbin 读导入表", "dumpbin" in text.lower())
    check("打包脚本：按闭包继续追（补齐传递依赖）", "queue.Enqueue($depKey)" in text)
    check("打包脚本：补不齐 dumpbin 就 throw", "找不到 dumpbin.exe" in text)
    # 手写清单是最初的写法，只覆盖 libssl/tcl 几个，正是它漏掉了 Pillow 的依赖。
    check(
        "打包脚本：不再用手写 conda DLL 清单",
        '"libssl-3-x64.dll", "libcrypto-3-x64.dll", "ffi.dll", "sqlite3.dll", "zlib.dll"' not in text,
    )
    # api-ms-win-* 是 conda 的 UCRT 转发件，搬进包里会和系统 UCRT 打架。
    check("打包脚本：不搬 api-ms-win-* 系统 DLL", "api-ms-win-" in text and "Test-SystemDll -Name $dep" in text)

    check("打包脚本：打完跑安装包自检", "selfcheck" in text)
    check("打包脚本：自检拿进程退出码", "Start-Process" in text and "$selfcheckProc.ExitCode" in text)
    check("打包脚本：自检不过就 throw", "安装包自检未通过" in text)

    entry = read(BACKEND / "frozen_entry.py")
    check("入口：有 _selfcheck", "def _selfcheck()" in entry)
    check("入口：自检 import app（一次带出全部路由）", '"app"' in entry)
    check("入口：自检 import PIL.Image", '"PIL.Image"' in entry)
    check("入口：--tool selfcheck 分发", 'sys.argv[2] == "selfcheck"' in entry)


def main() -> int:
    print(f"仓库根：{REPO}")
    if not WEB_SCRIPT.is_file():
        print(f"找不到打包脚本：{WEB_SCRIPT}")
        return 1
    if not INSTALL_SCRIPT.is_file():
        print(f"找不到安装脚本：{INSTALL_SCRIPT}")
        return 1

    text = read(WEB_SCRIPT)
    test_wheels_listed(text)
    test_avx2_trap(text)
    test_missing_wheels_stops(text)
    test_install_script_offline()
    test_launcher_packaging(text)
    test_nofollow_not_a_way_out(text)
    test_runtime_dlls_and_selfcheck(text)

    return finish()


if __name__ == "__main__":
    sys.exit(main())
