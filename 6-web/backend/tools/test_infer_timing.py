# -*- coding: utf-8 -*-
"""推理响应里的服务端阶段计时（v1.30.5）。

为什么单独测：`timing` 是判断「慢在模型还是慢在传图」的唯一依据。一旦
`payload_bytes` 与实际响应体不一致、或者 `decode_ms / ipc_ms / encode_ms`
加起来对不上 `server_ms`，用户就会再次被误导成「模型慢」——这正是 v1.30.3
要解决的问题。

覆盖：
1. `_finish_infer` 的 `payload_bytes` 等于响应体真实字节数（含字段自身）
2. `encode_ms` = server_ms - decode_ms - ipc_ms - 模型
3. `decode_ms` / `ipc_ms` 来自 `_note_server_stages`，ipc 不会出现负数
4. 同一份结果重复调用 `_finish_infer` 不会累加字段

用法：python 6-web/backend/tools/test_infer_timing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import app  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def _sample_result() -> dict:
    return {
        "backend": "ocr_infer.dll",
        "count": 2,
        "lines": [
            {"index": 0, "poly": [[1.0, 2.0], [3.0, 4.0]], "text": "登机牌", "score": 0.98, "angle": 0},
            {"index": 1, "poly": [[5.0, 6.0], [7.0, 8.0]], "text": "TRTForge", "score": 0.91, "angle": 0},
        ],
        "timing": {"io_ms": 0.0, "pipeline_ms": 180.0, "result_ms": 0.2, "total_ms": 180.2},
    }


def test_finish_infer() -> None:
    print("\n[1] payload_bytes 与实际响应体一致")
    result = _sample_result()
    app._note_server_stages(result, decode_ms=12.0, call_ms=184.0)
    # started 往前挪 200 ms，模拟真实请求已消耗的时间，server_ms 才有意义
    response = app._finish_infer(result, time.perf_counter() - 0.2)
    body = response.body
    timing = result["timing"]

    check("payload_bytes 与 body 长度相同", timing["payload_bytes"] == len(body),
          f"{timing['payload_bytes']} vs {len(body)}")
    check("payload_bytes 不是 0", timing["payload_bytes"] > 0)
    decoded = json.loads(body.decode("utf-8"))
    check("响应里 payload_bytes 就是真实长度（不是占位 0）",
          decoded["timing"]["payload_bytes"] == len(body))

    print("\n[2] 阶段计时自洽")
    check("decode_ms 来自调用方", decoded["timing"]["decode_ms"] == 12.0)
    check("ipc_ms = 往返 184 - 模型 180.2", abs(decoded["timing"]["ipc_ms"] - 3.8) < 0.15,
          str(decoded["timing"]["ipc_ms"]))
    total = (decoded["timing"]["decode_ms"] + decoded["timing"]["ipc_ms"]
             + decoded["timing"]["total_ms"] + decoded["timing"]["encode_ms"])
    check("四段之和等于 server_ms", abs(total - decoded["timing"]["server_ms"]) < 0.3,
          f"{total:.1f} vs {decoded['timing']['server_ms']}")
    check("server_ms 反映真实耗时（约 200 ms）", 190 <= decoded["timing"]["server_ms"] <= 260,
          str(decoded["timing"]["server_ms"]))
    check("encode_ms 不为负", decoded["timing"]["encode_ms"] >= 0)


def test_ipc_floor() -> None:
    print("\n[3] 计时缺失 / 异常时不出现负数")
    result = _sample_result()
    app._note_server_stages(result, decode_ms=5.0, call_ms=1.0)  # 比模型还小
    check("ipc_ms 被夹到 0", result["timing"]["ipc_ms"] == 0.0, str(result["timing"]["ipc_ms"]))

    plain = {"timing": {"inference_ms": 1.0}}  # 没有 total_ms
    response = app._finish_infer(plain, time.perf_counter())
    decoded = json.loads(response.body.decode("utf-8"))
    check("没有 total_ms 时 encode_ms 不为负", decoded["timing"]["encode_ms"] >= 0)
    check("没有 decode/ipc 时如实填 0",
          decoded["timing"].get("decode_ms", 0) == 0 and decoded["timing"].get("ipc_ms", 0) == 0)

    no_timing = {"lines": []}
    response = app._finish_infer(no_timing, time.perf_counter())
    check("没有 timing 字段时仍返回 JSON", response.status_code == 200)


def test_repeat_safe() -> None:
    print("\n[4] 重复收尾不累加")
    result = _sample_result()
    app._note_server_stages(result, decode_ms=10.0, call_ms=190.0)
    first = app._finish_infer(result, time.perf_counter())
    second = app._finish_infer(result, time.perf_counter())
    a = json.loads(first.body.decode("utf-8"))["timing"]
    b = json.loads(second.body.decode("utf-8"))["timing"]
    check("decode_ms 不被累加", a["decode_ms"] == b["decode_ms"] == 10.0)
    check("ipc_ms 不被累加", a["ipc_ms"] == b["ipc_ms"])
    check("两次 payload_bytes 相同", a["payload_bytes"] == b["payload_bytes"])


def main() -> int:
    test_finish_infer()
    test_ipc_floor()
    test_repeat_safe()
    print(f"\n结果：通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
