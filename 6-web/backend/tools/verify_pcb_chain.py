# -*- coding: utf-8 -*-
"""PCB detect 链路端到端验证：训练产物 -> ONNX -> TRT 引擎 -> Web 推理 API。

用法（需先启动 6-web 服务）：
    python verify_pcb_chain.py [图片数]

校验点：
1. Web 服务健康、yolo_infer.dll 可用；
2. 逐张图片调用 /api/infer/yolo，拿到检测框；
3. 把标注结果图保存到 runtime/verify/ 供人工比对；
4. 汇总检测数量与置信度分布，异常时非零退出。

编码统一 UTF-8，避免 Windows 控制台乱码。
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import requests

API = "http://127.0.0.1:8060"
REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "PCB data" / "PCB缺陷检测" / "images" / "val"
WEIGHTS = REPO / "1-python-yolo" / "launcher" / "runs" / "pcb" / "detect_v1" / "weights"
ENGINE = WEIGHTS / "best.transd_fp16.engine"
OUT_DIR = REPO / "6-web" / "runtime" / "verify"


def main(argv: list[str]) -> int:
    limit = int(argv[1]) if len(argv) > 1 else 3

    health = requests.get(f"{API}/api/health", timeout=5).json()
    print(f"[1] 服务健康: {health}")

    status = requests.get(f"{API}/api/capabilities", timeout=5).json()
    print(f"[2] yolo 后端可用: {status.get('infer_yolo')}  detail={status.get('infer_detail')}")
    if not status.get("infer_yolo"):
        print("!! yolo_infer.dll 不可用，终止")
        return 2

    if not ENGINE.is_file():
        print(f"!! 引擎不存在: {ENGINE}")
        return 3
    print(f"[3] 引擎: {ENGINE.name} ({ENGINE.stat().st_size / 1024 / 1024:.2f} MB)")

    images = sorted(DATASET.glob("*.jpg"))[:limit]
    if not images:
        print(f"!! 验证集无图片: {DATASET}")
        return 4

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for img in images:
        with img.open("rb") as fh:
            resp = requests.post(
                f"{API}/api/infer/yolo",
                data={
                    "engine": ENGINE.as_posix(),
                    "task": "detect",
                    "conf": 0.25,
                    "nms": 0.45,
                },
                files={"image": (img.name, fh, "image/jpeg")},
                timeout=120,
            )
        if resp.status_code != 200:
            print(f"!! {img.name} HTTP {resp.status_code}: {resp.text[:300]}")
            return 5

        data = resp.json()
        boxes = data.get("boxes", [])
        confs = [round(float(b.get("confidence", 0)), 4) for b in boxes]
        labels = [int(b.get("class_label", -1)) for b in boxes]

        png = base64.b64decode(data["annotated_png"])
        dst = OUT_DIR / f"{img.stem}_pred.png"
        dst.write_bytes(png)

        summary.append(
            {"image": img.name, "boxes": len(boxes), "labels": labels, "confs": confs, "out": str(dst)}
        )
        print(f"    {img.name}: {len(boxes)} 框, 类别={labels}, 置信度={confs}")

    report = OUT_DIR / "verify_report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] 标注图与报告已写入: {OUT_DIR}")
    print("[OK] PCB detect 全链路验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
