# -*- coding: utf-8 -*-
"""onnxruntime 运行时环境探测：**哪些 provider 真的能用**。

为什么需要这个模块：
    `ort.get_available_providers()` 回答的是"这个 onnxruntime 编译时带了哪些 EP"，
    不是"运行时能不能起来"。pip 装的 `onnxruntime-gpu` 在缺 cuDNN 9 / CUDA 12 的
    机器上照样把 `CUDAExecutionProvider` 列出来，然后在 `InferenceSession(...)` 里
    打一大段

        LoadLibrary failed with error 126 ... onnxruntime_providers_cuda.dll
        Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*

    再默默退回 CPU。功能没坏，但两件事被搞坏了：
      1. Web 端每次新建 session 都把这段糊进日志，现场排查时真正的错误被淹没；
      2. 前端"用 GPU 了吗"的判断失去依据（列出来的 provider 不可信）。

    所以这里先**探一次 provider DLL 能不能加载**（`LoadLibrary` 级探测，毫秒级），
    加载不了就别把它交给 onnxruntime，直接只用 CPU，日志干净、行为确定。
"""

from __future__ import annotations

import os
import sys

#: 明确排除的 EP。TensorRT EP 在 SAM 解码器这类小算子上会崩（社区已知问题），
#: 而且本项目推理本来就只走 CPU 或 CUDA。
EXCLUDED = ("TensorrtExecutionProvider",)

#: provider 名字 → 对应的运行时 DLL 文件名（只有 Windows 需要探测）。
_DLL_BY_PROVIDER = {
    "CUDAExecutionProvider": "onnxruntime_providers_cuda.dll",
    "TensorrtExecutionProvider": "onnxruntime_providers_tensorrt.dll",
    "DmlExecutionProvider": "onnxruntime_providers_dml.dll",
    "ROCMExecutionProvider": "onnxruntime_providers_rocm.dll",
    "OpenVINOExecutionProvider": "onnxruntime_providers_openvino.dll",
}

_DLL_CACHE: dict[str, bool] = {}


def _dll_loadable(ort_module, filename: str) -> bool:
    """探测 onnxruntime 的 provider DLL 能否被加载（结果缓存，避免每次建 session 都试）。

    只做 `LoadLibrary`：它的失败原因（缺 cuDNN / 缺 CUDA runtime / 架构不符）
    与 onnxruntime 自己加载时完全一致，但**不会打印任何东西**，这是关键——
    要的就是"探而不叫"。
    """
    if not sys.platform.startswith("win"):
        # Linux/macOS 的 EP 是同一批 .so，且缺库时 onnxruntime 会正常报错、
        # 不做那套宏大的 fallback 日志，不需要额外探测。
        return True
    if filename in _DLL_CACHE:
        return _DLL_CACHE[filename]

    ok = False
    try:
        import ctypes

        capi = os.path.join(os.path.dirname(os.path.abspath(ort_module.__file__)), "capi")
        path = os.path.join(capi, filename)
        if os.path.isfile(path):
            ctypes.WinDLL(path)
            ok = True
    except (OSError, AttributeError):
        # OSError 涵盖 WinError 126（找不到依赖）/ 193（架构不符）；
        # AttributeError 是非 Windows 上的 WinDLL 缺失。
        ok = False
    _DLL_CACHE[filename] = ok
    return ok


def usable_providers(ort_module=None) -> list[str]:
    """返回**当前机器上真能起来**的 EP 列表，按 onnxruntime 的优先级排列。

    永远至少返回 `["CPUExecutionProvider"]`。
    """
    if ort_module is None:
        try:
            import onnxruntime as ort_module  # noqa: PLC0415 - 延迟导入，缺包时不炸模块加载
        except ImportError:  # pragma: no cover - 依赖缺失由调用方给提示
            return ["CPUExecutionProvider"]

    available = list(ort_module.get_available_providers())
    chosen: list[str] = []
    for name in available:
        if name in EXCLUDED:
            continue
        dll = _DLL_BY_PROVIDER.get(name)
        if dll and not _dll_loadable(ort_module, dll):
            continue
        chosen.append(name)
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def describe(providers: list[str]) -> str:
    """给日志/接口用的一句话说明，比如 ``CPU`` 或 ``CUDA + CPU``。"""
    names = [p.replace("ExecutionProvider", "") for p in providers]
    return " + ".join(names) if names else "CPU"


def reset_cache() -> None:
    """清掉 DLL 探测缓存（测试用）。"""
    _DLL_CACHE.clear()
