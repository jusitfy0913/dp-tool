# -*- coding: utf-8 -*-
"""onnxenv 单元测试：provider 只挑"真能起来"的。

背景（v1.24.1）：`ort.get_available_providers()` 报的是编译期支持的 EP，不是运行时
能加载的 EP。缺 cuDNN 9 的机器上它照样列出 CUDA，然后 InferenceSession 里打一大段
`LoadLibrary failed with error 126` 再退回 CPU。这里验证我们不再犯这个错。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxenv  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}  {detail}")


class FakeOrt:
    """假的 onnxruntime 模块：控制"列出哪些 EP"和"哪些 DLL 能加载"。"""

    def __init__(self, available, loadable):
        self._available = list(available)
        self._loadable = set(loadable)
        self.__file__ = str(Path(__file__).resolve().parents[1] / "onnxruntime_stub.py")

    def get_available_providers(self):
        return list(self._available)


def test_tensorrt_always_excluded() -> None:
    """TensorRT EP 在 SAM 解码器的小算子上会崩，无论能否加载都不该被选中。"""
    onnxenv.reset_cache()
    fake = FakeOrt(["TensorrtExecutionProvider", "CPUExecutionProvider"], [])
    got = onnxenv.usable_providers(fake)
    check("排除：TensorRT 不入选", "TensorrtExecutionProvider" not in got, str(got))
    check("排除：仍保留 CPU", "CPUExecutionProvider" in got, str(got))


def test_cpu_only_environment() -> None:
    onnxenv.reset_cache()
    fake = FakeOrt(["CPUExecutionProvider"], [])
    check("仅 CPU：结果就是 CPU", onnxenv.usable_providers(fake) == ["CPUExecutionProvider"], "")


def test_cuda_dropped_when_dll_missing() -> None:
    """DLL 不存在时 CUDA 必须被剔除——这是本模块存在的唯一理由。"""
    onnxenv.reset_cache()
    fake = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"], [])
    got = onnxenv.usable_providers(fake)
    check("CUDA：DLL 缺失时剔除", "CUDAExecutionProvider" not in got, str(got))
    check("CUDA：剔除后仍可用", got == ["CPUExecutionProvider"], str(got))


def test_describe() -> None:
    check("描述：单 CPU", onnxenv.describe(["CPUExecutionProvider"]) == "CPU", onnxenv.describe(["CPUExecutionProvider"]))
    check(
        "描述：CUDA + CPU",
        onnxenv.describe(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "CUDA + CPU",
        onnxenv.describe(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    check("描述：空列表兜底 CPU", onnxenv.describe([]) == "CPU", onnxenv.describe([]))


def test_real_onnxruntime() -> None:
    """真环境自检：返回的列表非空、含 CPU、且没有 TensorRT。"""
    onnxenv.reset_cache()
    got = onnxenv.usable_providers()
    check("真环境：非空", bool(got), str(got))
    check("真环境：含 CPU EP", "CPUExecutionProvider" in got, str(got))
    check("真环境：无 TensorRT", "TensorrtExecutionProvider" not in got, str(got))
    engine = "GPU" if "CUDAExecutionProvider" in got else "CPU"
    print(f"       本机推理引擎：{engine}（可用 provider 列表 {got}）")


def main() -> int:
    test_tensorrt_always_excluded()
    test_cpu_only_environment()
    test_cuda_dropped_when_dll_missing()
    test_describe()
    test_real_onnxruntime()
    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
