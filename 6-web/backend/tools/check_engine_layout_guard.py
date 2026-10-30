# -*- coding: utf-8 -*-
"""对比「错误引擎」与「正确引擎」在 Web 推理接口上的表现。

目的是确认 app.py 新增的输出合理性校验真的拦住了
用未转置 ONNX 编译出来的引擎，而不是继续吐出垃圾框。
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

API = "http://127.0.0.1:8060"
REPO = Path(r"D:/test/deeplearning/dp-tool")
IMG_DIR = REPO / "PCB data" / "PCB缺陷检测" / "images" / "val"
WEIGHTS = REPO / "1-python-yolo" / "launcher" / "runs" / "pcb" / "detect_v1" / "weights"


def main() -> int:
    img = sorted(IMG_DIR.glob("*.jpg"))[0]
    print(f"测试图片: {img.name}")

    cases = [
        ("best_fp16.engine", "未转置 ONNX 编译（应当被拦截）"),
        ("best.transd_fp16.engine", "转置 ONNX 编译（应当正常）"),
    ]

    for name, desc in cases:
        engine = WEIGHTS / name
        if not engine.is_file():
            print(f"跳过 {name}：文件不存在")
            continue
        with img.open("rb") as fh:
            resp = requests.post(
                f"{API}/api/infer/yolo",
                data={"engine": engine.as_posix(), "task": "detect", "conf": 0.25, "nms": 0.45},
                files={"image": (img.name, fh, "image/jpeg")},
                timeout=120,
            )
        body = resp.json()
        if resp.status_code == 200:
            confs = [round(float(b["confidence"]), 3) for b in body.get("boxes", [])]
            print(f"  {name}: HTTP 200 | {desc} | {len(confs)} 框 | conf={confs}")
        else:
            print(f"  {name}: HTTP {resp.status_code} | {desc}")
            print(f"      detail: {str(body.get('detail', body))[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
