# -*- coding: utf-8 -*-
"""trtexec 额外参数白名单的验收脚本（``backend/trtexecargs.py``）。

为什么单独测：这层是「网页不能执行任意操作」的最后一道门。trtexec 自带
``--saveEngine`` / ``--exportOutput`` / ``--loadInputs``（读写任意文件）与
``--plugins``（加载外部 DLL），一旦这些能通过 ``extra_args`` 拼进命令行，
路径白名单与令牌都形同虚设。

    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_trtexec_args.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402# noqa: E402

import trtexecargs  # noqa: E402


def rejected(raw: str) -> bool:
    try:
        trtexecargs.check(raw)
    except ValueError:
        return True
    return False


def test_allowed() -> None:
    check("空串返回空列表", trtexecargs.check("") == [])
    check("空白串返回空列表", trtexecargs.check("   ") == [])
    ok = trtexecargs.check("--verbose --builderOptimizationLevel=5")
    check("调优参数放行", ok == ["--verbose", "--builderOptimizationLevel=5"], str(ok))
    check("--noTF32 放行", trtexecargs.check("--noTF32") == ["--noTF32"])
    check("独立值的写法放行",
          trtexecargs.check("--iterations 100") == ["--iterations", "100"])
    check("内存池带命名空间的值放行",
          trtexecargs.check("--memPoolSize=workspace:1024") == ["--memPoolSize=workspace:1024"])
    check("层精度带冒号的值放行",
          trtexecargs.check("--layerPrecisions=conv1:fp16") == ["--layerPrecisions=conv1:fp16"])


def test_rejected() -> None:
    for raw in (
        "--plugins=evil.dll",
        "--staticPlugins=evil.dll",
        "--saveEngine=D:\\out\\x.engine",
        "--loadEngine=D:\\out\\x.engine",
        "--onnx=D:\\data\\x.onnx",
        "--exportOutput=D:\\out.json",
        "--exportProfile=D:\\p.json",
        "--exportLayerInfo=D:\\l.json",
        "--loadInputs=conv1:D:\\in.bin",
        "--timingCacheFile=D:\\t.bin",
        "--int8",
        "--best",
        "--calib=D:\\calib",
        "--workspace=8192",
        "--fp16",
        "--minShapes=images:1x3x640x640",
    ):
        check(f"拒绝 {raw}", rejected(raw), raw)

    check("参数名大小写不敏感地拒绝",
          rejected("--SaveEngine=D:\\x.engine"))
    check("未知参数一律拒绝", rejected("--totallyNewTrtexecFlag=1"))
    check("含路径分隔符的独立值被拒绝", rejected("--iterations D:/tmp/100"))
    check("反斜杠路径被拒绝", rejected("--iterations D:\\tmp\\100"))
    check("短参数被拒绝", rejected("-v"))
    check("允许项与禁止项混在一起也整串拒绝",
          rejected("--verbose --saveEngine=D:\\x.engine"))


def test_message() -> None:
    try:
        trtexecargs.check("--saveEngine=D:\\x.engine")
    except ValueError as exc:
        text = str(exc)
        check("错误信息指出被拒的项", "--saveEngine" in text, text)
        check("错误信息列出允许清单", "--memPoolSize" in text, text)
        check("错误信息提示改用页面字段", "字段" in text, text)
    else:
        check("错误信息指出被拒的项", False, "没有抛异常")

    check("describe_allowed 非空", "--memPoolSize" in trtexecargs.describe_allowed())


def main() -> int:
    test_allowed()
    test_rejected()
    test_message()
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
