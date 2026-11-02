# -*- coding: utf-8 -*-
"""导出 ONNX 的命令拼装与参数校验验收脚本，不需要 GPU / torch / 真实权重。

守的是「填了但没传」这类静默问题：表单开关没进命令行时，用户只能从日志里
翻完整命令才能发现，而 argparse 的 usage 提示根本看不出少了哪个参数。

用法：
    python 6-web/backend/tools/test_export_request.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import app  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def make(**overrides):
    payload = {
        "weights": r"D:\runs\exp\weights\best.pt",
        "task": "detect",
        "imgsz": 640,
    }
    payload.update(overrides)
    return app.YoloExportRequest(**payload)


def as_pairs(command: list[str]) -> dict[str, str]:
    """把 --key value 形式解析成字典，方便断言「传了没、值对不对」。"""
    out: dict[str, str] = {}
    for idx, token in enumerate(command):
        if token.startswith("--") and idx + 1 < len(command):
            nxt = command[idx + 1]
            if not nxt.startswith("--"):
                out[token] = nxt
    return out


def main() -> int:
    # ---- 1. 默认值：make_transd 必须默认开 --------------------------------
    command = app.build_export_command(make(), "python.exe")
    pairs = as_pairs(command)
    check("走的是 export 模式", pairs.get("--mode") == "export", str(pairs.get("--mode")))
    check("默认生成 transd", pairs.get("--make-transd") == "true", str(pairs.get("--make-transd")))
    # --simplify 是 BooleanOptionalAction：只能出现 --simplify / --no-simplify，
    # 写成 `--simplify true` 会被 argparse 判成多余参数并退出 2（实测踩过）。
    check("默认带 --simplify 且不带值", "--simplify" in command and "--no-simplify" not in command, str(command))
    check("simplify 后面不跟 true/false", "true" not in command[command.index("--simplify") + 1: command.index("--simplify") + 2], str(command))
    check("默认 export-batch=1", pairs.get("--export-batch") == "1", str(pairs.get("--export-batch")))
    check("默认不传 dynamic", "--dynamic" not in command, str(command))
    check("默认不传 half", "--half" not in command, str(command))
    check("默认不传 opset（用 Ultralytics 默认）", "--opset" not in command, str(command))
    check("权重原样传", pairs.get("--weights") == make().weights, str(pairs.get("--weights")))

    # ---- 2. 开关全部打开时都要真的进命令行 --------------------------------
    full = app.build_export_command(
        make(task="segment", imgsz=320, opset=17, dynamic=True, half=True, make_transd=False, simplify=False),
        "python.exe",
    )
    pairs = as_pairs(full)
    check("task 透传", pairs.get("--task") == "segment", str(pairs.get("--task")))
    check("imgsz 透传", pairs.get("--imgsz") == "320", str(pairs.get("--imgsz")))
    check("opset 透传", pairs.get("--opset") == "17", str(pairs.get("--opset")))
    check("dynamic 开关生效", "--dynamic" in full, str(full))
    check("half 开关生效", "--half" in full, str(full))
    check("关掉 transd 时传 false 而不是省略", pairs.get("--make-transd") == "false", str(pairs.get("--make-transd")))
    check("关掉 simplify 时传 --no-simplify", "--no-simplify" in full and "--simplify" not in full, str(full))

    # ---- 3. 非法 imgsz 提前拦下，不要让用户等 30 秒 -----------------------
    for bad in (100, 0, 4096, -32):
        try:
            app.build_export_command(make(imgsz=bad), "python.exe")
            check(f"imgsz={bad} 被拒绝", False, "没有抛异常")
        except ValueError as exc:
            check(f"imgsz={bad} 被拒绝", "32" in str(exc), str(exc))

    # ---- 4. 非法 task 被拒绝 ---------------------------------------------
    try:
        app.build_export_command(make(task="classify"), "python.exe")
        check("非法 task 被拒绝", False, "没有抛异常")
    except ValueError as exc:
        check("非法 task 被拒绝", "classify" in str(exc), str(exc))

    # ---- 5. opset 越界由 pydantic 拦（不是我们自己写 if） ------------------
    try:
        make(opset=99)
        check("opset 越界被模型校验拦下", False, "没有抛异常")
    except Exception as exc:  # noqa: BLE001 - pydantic 的异常类型随版本变化
        check("opset 越界被模型校验拦下", "opset" in str(exc), str(exc))

    # ---- 6. batch 越界同样被拦 -------------------------------------------
    try:
        make(batch=0)
        check("batch 越界被模型校验拦下", False, "没有抛异常")
    except Exception as exc:  # noqa: BLE001
        check("batch 越界被模型校验拦下", "batch" in str(exc), str(exc))

    # ---- 7. 可选执行器可覆盖 ---------------------------------------------
    custom = app.build_export_command(make(), r"D:\envs\py\python.exe")
    check("解释器可覆盖", custom[0] == r"D:\envs\py\python.exe", custom[0])

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
