# -*- coding: utf-8 -*-
"""任务日志出口与事件序号的验收脚本，不需要 GPU / DLL / torch。

守的是两类「事后才发现」的问题：

1. **日志拿不到**：面板只留最近若干行（训练 3000 行），真正的报错往往在更早的
   地方；任务结束后内存里的 run 已经没了，但 `runtime/logs/{run_id}.log` 还在。
   `/api/runs/{id}/log` 就是把这份文件交出去的口子，它必须能挡住 `../` 之类的越界。
2. **重连补两遍**：断线重连时后端会整段补发历史事件。没有序号时前端只能再追加
   一遍，日志区每个文件事件都出现两次，指标表还会重复记同一轮。

用法：
    python 6-web/backend/tools/test_run_log.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import app as backend_app  # noqa: E402
import config  # noqa: E402
import runners  # noqa: E402
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


def make_run(run_id: str) -> runners.Run:
    return runners.Run(run_id=run_id, kind="yolo-train", command=["python", "train.py"], cwd=".")


def test_event_seq() -> None:
    """序号必须从 1 开始、严格递增，并且落在事件体里。"""
    manager = runners.RunManager()
    run = make_run("seqtest00001")

    manager._publish(run, {"type": "log", "line": "a"})
    manager._publish(run, {"type": "log", "line": "b"})
    manager._publish(run, {"type": "finished", "exit": 0, "status": "finished"})

    seqs = [event.get("seq") for event in run.events]
    check("事件带上 seq", all(isinstance(item, int) for item in seqs), str(seqs))
    check("seq 从 1 开始", seqs[:1] == [1], str(seqs[:1]))
    check("seq 严格递增", seqs == [1, 2, 3], str(seqs))
    check("seq 与事件体一起入缓冲", run.events[0].get("line") == "a", str(run.events[0]))

    # 重连补发的就是缓冲里的这批事件：游标（最后一条 seq）能被前端直接复用
    history = list(run.events)
    check("补发历史带序号可供去重", history[-1]["seq"] == 3, str(history[-1]))

    # 缓冲区只保留最近 N 条，序号不能因为裁剪而回退
    small = runners.RunManager()
    run2 = make_run("seqtest00002")
    for index in range(runners.EVENT_BUFFER_SIZE + 5):
        small._publish(run2, {"type": "log", "line": str(index)})
    check(
        "缓冲裁剪后 seq 不回退",
        run2.seq == runners.EVENT_BUFFER_SIZE + 5 and len(run2.events) == runners.EVENT_BUFFER_SIZE,
        f"seq={run2.seq} len={len(run2.events)}",
    )


def test_snapshot_has_log_file() -> None:
    """/api/runs 的返回值要带 log_file，前端才知道去哪取全文。"""
    run = make_run("snaptest0001")
    run.log_file = config.LOG_DIR / "snaptest0001.log"
    snap = run.snapshot()
    check("快照带 log_file", snap.get("log_file", "").endswith("snaptest0001.log"), str(snap.get("log_file")))

    run.log_file = None
    check("没有日志文件时为 null 而不是空串", run.snapshot().get("log_file") is None)


def test_log_endpoint() -> None:
    """接口本身：全文 / 尾部 / 越界 / 不存在。"""
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = "abcdef123456"
    log_file = log_dir / f"{run_id}.log"
    log_file.write_text("\n".join(f"line-{i:04d}" for i in range(1, 301)), encoding="utf-8")

    try:
        full = backend_app.get_run_log(run_id, lines=0)
        text = full.body.decode("utf-8")
        check("能取到完整日志", full.status_code == 200, str(full.status_code))
        check("默认返回全文（不是尾巴）", text.startswith("line-0001") and text.endswith("line-0300"))
        check("全文行数正确", len(text.splitlines()) == 300, str(len(text.splitlines())))
        check(
            "media_type 是纯文本",
            full.media_type == "text/plain; charset=utf-8",
            str(full.media_type),
        )
        disposition = full.headers.get("content-disposition", "")
        check("带下载文件名（方便另存）", f'filename="{run_id}.log"' in disposition, disposition)

        tail = backend_app.get_run_log(run_id, lines=5).body.decode("utf-8")
        check("lines=N 只回最后 N 行", tail.splitlines() == [f"line-{i:04d}" for i in range(296, 301)], tail)

        clamp = backend_app.get_run_log(run_id, lines=999999).body.decode("utf-8")
        check("lines 超上限被夹住而不是报错", len(clamp.splitlines()) == 300, str(len(clamp.splitlines())))

        # 路径越界：run_id 里带 .. 或分隔符一律 400，不能读到 runtime 外面
        def expect(code: int, run_id_arg: str, name: str) -> None:
            try:
                backend_app.get_run_log(run_id_arg, lines=0)
            except HTTPException as exc:
                check(name, exc.status_code == code, f"{exc.status_code} != {code}")
            else:
                check(name, False, "没有抛异常")

        expect(400, "../../etc/passwd", "run_id 带 ../ 返回 400")
        expect(400, "..", "run_id 是 .. 返回 400")
        expect(400, "ABCDEF123456", "大写/非十六进制的 run_id 返回 400")
        expect(400, "abc", "长度不足的 run_id 返回 400")
        expect(404, "0123456789ab", "日志文件不存在返回 404")
    finally:
        log_file.unlink(missing_ok=True)


def test_dedup_helper() -> None:
    """把前端的去重规则用 Python 复述一遍：同样的输入序列不能有重复。"""

    def fresh(payload: dict, cursor: int) -> bool:
        seq = payload.get("seq")
        return True if not isinstance(seq, int) else seq > cursor

    cursor = 0
    applied: list = []

    def feed(events: list[dict]) -> None:
        nonlocal cursor
        for event in events:
            if not fresh(event, cursor):
                continue
            # 只有带 seq 的事件才推进游标；旧格式事件照旧应用，但不影响去重
            if isinstance(event.get("seq"), int):
                cursor = event["seq"]
            applied.append(event.get("seq"))

    first = [{"seq": index} for index in (1, 2, 3)]
    replay = [{"seq": index} for index in (1, 2, 3, 4, 5)]
    feed(first)
    feed(replay)
    check("重连后事件连续且不重复", applied == [1, 2, 3, 4, 5], str(applied))

    # 没有 seq 的旧格式事件必须照旧放行，否则去重就变成丢日志
    applied.clear()
    cursor = 0
    feed([{"line": "no-seq"}, {"line": "no-seq-2"}])
    check("没有 seq 的事件不被丢弃", applied == [None, None] and cursor == 0, str(applied))


def main() -> int:
    test_event_seq()
    test_snapshot_has_log_file()
    test_log_endpoint()
    test_dedup_helper()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
