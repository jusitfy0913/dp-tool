# -*- coding: utf-8 -*-
"""串联阶段日志接口的验收脚本，不需要 GPU / DLL / torch。

守的是「日志看不了」这一类问题——排障时最需要的恰恰是**已经结束**的那一段：

- 训练页的日志面板走 WebSocket，只认内存里还活着的 run；服务一重启就只会
  打印「未知任务」，而阶段日志文件其实还在磁盘上；
- 读日志尾巴时如果整文件读进来，几百 MB 的构建日志会把内存顶满；
- 日志路径必须限定在 `runtime/logs/{run_id}.log`，`run_id` 里塞 `..` 不能跳出去。

用法：
    python 6-web/backend/tools/test_pipeline_log.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import app as backend_app  # noqa: E402
import config  # noqa: E402
import pipeline  # noqa: E402
from fastapi import HTTPException  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))


def main() -> int:
    store = config.RUNTIME_DIR / "pipelines"
    store.mkdir(parents=True, exist_ok=True)
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = "logtest" + "0" * 5
    pid = "logtest-0001"
    log_file = log_dir / f"{run_id}.log"
    record_path = store / f"{pid}.json"

    try:
        # 1. 尾巴读取：只取最后 N 行，而且不能把前面的大段丢掉顺序
        log_file.write_text(
            "\n".join(f"line-{i:04d}" for i in range(1, 501)), encoding="utf-8"
        )
        record = {
            "id": pid,
            "kind": "yolo",
            "title": "日志测试",
            "status": pipeline.CHAIN_SUCCESS,
            "created_at": 1.0,
            "stages": [
                {"name": "export", "label": "导出", "status": "success", "run_id": run_id},
                {"name": "labels", "label": "补标签", "status": "success", "run_id": None},
            ],
            "artifacts": {},
            "metrics": None,
            "error": None,
            "meta": {},
        }
        record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        resp = backend_app.pipeline_stage_log(pid, "export", lines=5)
        text = resp.body.decode("utf-8")
        check("能读到阶段日志", resp.status_code == 200, str(resp.status_code))
        check("只返回最后 N 行", text.splitlines() == [f"line-{i:04d}" for i in range(496, 501)], repr(text))
        check("按行尾截断不丢最后一行", text.endswith("line-0500"), repr(text[-20:]))

        # 2. 大文件也只需要读尾部：构造 8 KB 一段的长行，确认 lines=1 只回一行
        log_file.write_text(
            "\n".join(["x" * 8000, "y" * 8000, "z" * 8000]), encoding="utf-8"
        )
        one = backend_app.pipeline_stage_log(pid, "export", lines=1)
        check("长行只回最后一行", one.body.decode("utf-8") == "z" * 8000,
              str(len(one.body)))

        # 3. 参数兜底：lines 越界被夹住，不会返回空
        clamp = backend_app.pipeline_stage_log(pid, "export", lines=0)
        check("lines=0 至少返回一行", len(clamp.body.decode("utf-8").splitlines()) >= 1)

        # 4. 错误路径
        def expect_404(stage: str, name: str) -> None:
            try:
                backend_app.pipeline_stage_log(pid, stage, lines=10)
            except HTTPException as exc:
                check(name, exc.status_code == 404, str(exc.status_code))
            else:
                check(name, False, "没有抛异常")

        expect_404("不存在的阶段", "未知阶段返回 404")
        expect_404("labels", "没有 run_id 的阶段返回 404")

        # 5. 路径穿越：run_id 里带 .. 只能 404，不能读到别的目录
        evil = dict(record)
        evil["stages"] = [
            {"name": "export", "label": "导出", "status": "success", "run_id": "../secret"}
        ]
        record_path.write_text(json.dumps(evil, ensure_ascii=False), encoding="utf-8")
        expect_404("export", "run_id 带 ../ 返回 404")

        # 6. 不存在的串联记录
        try:
            backend_app.pipeline_stage_log("nope-0000", "export", lines=10)
        except HTTPException as exc:
            check("不存在的串联返回 404", exc.status_code == 404, str(exc.status_code))
        else:
            check("不存在的串联返回 404", False, "没有抛异常")

    finally:
        log_file.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
