# -*- coding: utf-8 -*-
"""混合精度（AMP）可用性判定。

为什么要单独判定，而不是信任 ``--amp``
--------------------------------------
Ultralytics 在 ``Trainer.__init__`` 里这样决定本次训练用不用混合精度::

    self.amp = self.args.amp not in {False, "fp32"}
    if self.amp and self.args.amp != "bf16":
        self.amp = torch.tensor(check_amp(self.model), device=self.device)

``check_amp()`` 先用正则把一批已知有问题的显卡直接判死（会 NaN loss 或 zero-mAP），
再做一次 fp32/fp16 前向对比。也就是说：

- 传 ``--amp`` **不等于** 真的会用 fp16；黑名单显卡上会被静默关掉；
- **没有任何参数能强制打开** 黑名单显卡的 fp16（除非改上游代码，把上游刻意规避的
  NaN 风险放进来，本模块不做这件事）。

于是这里提供的是「**预测** + 解释」：让页面在开训前就说清「本机预计 fp32」，
而不是让用户对着慢一倍的训练速度猜原因。训练结束后以 trainer 上报的**实际值**为准。

黑名单与上游保持一致（ultralytics 8.4.192 ``utils/checks.py::check_amp``）。
上游改了这里要跟着改——这是唯一一处「复制上游常量」，所以单独成文件并写清版本。
"""

from __future__ import annotations

import re
from typing import Any

#: 与 ultralytics 8.4.192 的 check_amp() 完全一致。
AMP_BLOCKLIST_PATTERN = (
    r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)"
)
_BLOCKLIST = re.compile(AMP_BLOCKLIST_PATTERN, re.IGNORECASE)

#: 黑名单里的型号 token，用来告诉用户「命中的是哪一个」，而不是只丢一句「不可用」。
_BLOCKLIST_MODELS = ("1660", "1650", "1630", "t400", "t550", "t600", "t1000", "t1200", "t2000", "k40m")

#: 上游版本，出现在页面提示里，方便对账。
UPSTREAM_VERSION = "ultralytics 8.4.192"


def matched_model(gpu_name: str | None) -> str | None:
    """返回命中的黑名单型号（小写），没命中返回 ``None``。"""
    if not gpu_name:
        return None
    match = _BLOCKLIST.search(str(gpu_name))
    if not match:
        return None
    hit = match.group(0).lower()
    for model in _BLOCKLIST_MODELS:
        if model in hit:
            return model
    return None


def amp_status(gpu_name: str | None = None, requested: bool | None = None) -> dict[str, Any]:
    """判定本机 AMP 可用性。

    ``available`` 三态：
      - ``True``  ：不在黑名单，预计可以用 fp16（训练开始时上游还会做一次前向对比，
                    极小概率仍会关闭，所以文案写「预计」）；
      - ``False`` ：命中黑名单，上游一定关掉；
      - ``None``  ：拿不到显卡名（没卡 / 没驱动 / 没 nvidia-smi），无法判断。
    """
    name = str(gpu_name).strip() if gpu_name else ""
    hit = matched_model(name)
    if not name:
        return {
            "gpu": None,
            "available": None,
            "matched": None,
            "requested": requested,
            "expected_precision": "unknown",
            "reason": "无法读取显卡名称（没有 CUDA 或 nvidia-smi），混合精度状态未知",
            "note": "",
            "upstream": UPSTREAM_VERSION,
        }
    if hit:
        return {
            "gpu": name,
            "available": False,
            "matched": hit,
            "requested": requested,
            "expected_precision": "fp32",
            "reason": (
                f"{name} 命中 {UPSTREAM_VERSION} 的 AMP 黑名单（{hit}）："
                "这批卡用 fp16 会出现 NaN loss 或 zero-mAP，框架会在训练开始时关闭混合精度"
            ),
            "note": (
                "本次训练实际走 fp32：显存占用明显更高（同等 batch 约为 fp16 的 1.5~2 倍），"
                "速度约慢一倍。即使传 --amp 也无法强制开启——关闭发生在框架内部。"
            ),
            "upstream": UPSTREAM_VERSION,
        }
    return {
        "gpu": name,
        "available": True,
        "matched": None,
        "requested": requested,
        "expected_precision": "fp16",
        "reason": f"{name} 不在 AMP 黑名单内，预计可以使用混合精度（fp16）",
        "note": "训练开始时框架会再做一次 fp32/fp16 前向对比，极小概率仍会回退 fp32；以训练结束上报的实际值为准。",
        "upstream": UPSTREAM_VERSION,
    }


def expected_precision(amp_requested: bool | None, gpu_name: str | None = None) -> str:
    """预计的精度，用来给显存实测表选键。

    ``amp_requested is False`` 时直接 fp32，不需要看显卡：那是用户或调用方显式指定的。
    """
    if amp_requested is False:
        return "fp32"
    status = amp_status(gpu_name)
    if status["available"] is True:
        return "fp16"
    if status["available"] is False:
        return "fp32"
    return "unknown"


def describe(status: dict[str, Any]) -> str:
    """一行文案，给日志与接口响应复用。"""
    available = status.get("available")
    if available is True:
        return f"混合精度可用（{status.get('gpu') or '未知显卡'}，预计 fp16）"
    if available is False:
        return f"混合精度不可用（{status.get('gpu')}，训练将走 fp32）"
    return "混合精度状态未知（读不到显卡名称）"
