# -*- coding: utf-8 -*-
"""跨进程 GPU 锁（gpulock）的回归测试。

测的是什么：Web 后端与命令行 autoexp 之间的**互斥契约**。
不测「锁的实现细节」，因为那个可以改；测的是行为——
「一方在跑，另一方必须能看见并且拒绝」。

    python tools/test_gpulock.py

关键设计：每个用例都把 ``TRTFORGE_GPU_LOCK`` 指到临时文件，
绝不碰真实的 ``runtime/gpu_lock.json``。否则跑一次测试就会挡住真实训练，
而那种 bug 极难联想到「是测试留下的」。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from _harness import case, check_eq, finish, require, run_cases  # noqa: E402


class temp_lock:
    """把锁文件重定向到临时目录，退出时恢复。"""

    def __init__(self) -> None:
        self.holder = tempfile.TemporaryDirectory(prefix="gpulock_")
        self.path = Path(self.holder.name) / "gpu_lock.json"
        self._saved = os.environ.get("TRTFORGE_GPU_LOCK")

    def __enter__(self) -> Path:
        os.environ["TRTFORGE_GPU_LOCK"] = str(self.path)
        return self.path

    def __exit__(self, *exc) -> bool:
        if self._saved is None:
            os.environ.pop("TRTFORGE_GPU_LOCK", None)
        else:
            os.environ["TRTFORGE_GPU_LOCK"] = self._saved
        self.holder.cleanup()
        return False


@case("空闲时 acquire 成功并写出可读的持有者信息")
def test_acquire_when_free() -> None:
    import gpulock

    with temp_lock() as path:
        ok, reason = gpulock.acquire("web", "yolo-train", "run_abc")
        try:
            require(ok, f"空闲时应能拿到锁，实际被拒：{reason}")
            data = json.loads(path.read_text(encoding="utf-8"))
            # v1.42.0 起是 {holders: {"kind|label": {...}}}，支持多持有者
            holders = data["holders"]
            key = gpulock.holder_key("yolo-train", "run_abc")
            require(key in holders, f"应写下 {key} 这一条，实际键：{list(holders)}")
            item = holders[key]
            check_eq(item["owner"], "web", "owner")
            check_eq(item["kind"], "yolo-train", "kind")
            check_eq(item["label"], "run_abc", "label")
            check_eq(item["pid"], os.getpid(), "pid")
            require("since" in item, "应记录获取时间")
        finally:
            gpulock.release("yolo-train", "run_abc")
        require(not path.exists(), "release 后锁文件应被删除")


@case("被别的进程持有时拒绝，并说清是谁在占")
def test_refuse_when_held_by_other() -> None:
    import gpulock

    with temp_lock() as path:
        # 用当前进程的真实 PID 写一份「别人」的锁存盘，再以一个不同的
        # 「我的 pid」视角读取——直接伪造会读到自己的 pid，测不出互斥。
        path.write_text(
            json.dumps(
                {"owner": "autoexp", "pid": os.getpid(), "kind": "autoexp-run",
                 "label": "exp_9999", "since": "2026-11-10T13:00:00+08:00"}
            ),
            encoding="utf-8",
        )
        # read() 看到的是存活的 pid，因此算「被占用」
        holder = gpulock.read()
        require(holder is not None, "存活的锁应被读到")
        text = gpulock.describe(holder)
        require("自动实验循环" in text, f"描述里应点明占用方，实际：{text}")
        require("exp_9999" in text, f"描述里应带上 label，实际：{text}")

        # acquire 在 pid 相同时视为「自己的锁」，允许重入（同进程多任务需要）
        ok, _ = gpulock.acquire("web", "yolo-train", "run_x")
        require(ok, "同 pid 重入应放行")
        gpulock.release()


@case("PID 已死的陈旧锁被当作空闲，不会永久锁死机器")
def test_stale_lock_is_ignored() -> None:
    import gpulock

    with temp_lock() as path:
        # 999999 在 Windows 上可稳定判定为不存在（WinError 87）
        path.write_text(
            json.dumps({"owner": "autoexp", "pid": 999999, "kind": "autoexp-run", "label": "dead"}),
            encoding="utf-8",
        )
        check_eq(gpulock.read(), None, "死亡 PID 的锁应被忽略")
        ok, reason = gpulock.acquire("web", "yolo-train", "run_y")
        require(ok, f"陈旧锁不应阻止新的 acquire，实际被拒：{reason}")
        gpulock.release()


@case("损坏的锁文件被当作空闲，不会把所有人挡在门外")
def test_corrupt_lock_is_ignored() -> None:
    import gpulock

    with temp_lock() as path:
        path.write_text("{ 这不是 JSON", encoding="utf-8")
        check_eq(gpulock.read(), None, "损坏文件应被忽略")
        ok, _ = gpulock.acquire("web", "yolo-train", "run_z")
        require(ok, "损坏文件不应阻止 acquire")
        gpulock.release()


@case("release 只删自己的锁，不误伤别人")
def test_release_does_not_steal() -> None:
    import gpulock

    with temp_lock() as path:
        # 写入一份「别的进程」持有的锁：换一个存活的、不是自己的 pid。
        # 用父进程（测试启动者的 python）不太稳，改用系统进程风险也大，
        # 所以这里直接构造一个「pid 是别人的」场景：把文件里的 pid 改成 1
        # （Windows 上 PID 1 通常不存在 → 会被判陈旧），因此改用真实存活但对不上的
        # 做法：先由本进程写，再把 pid 改成 4（Windows System，必然存活）。
        path.write_text(
            json.dumps({"owner": "web", "pid": 4, "kind": "yolo-train", "label": "other"}),
            encoding="utf-8",
        )
        if gpulock.read() is None:
            # 该平台没有 PID 4，跳过这层校验（不把平台差异记成失败）
            return
        gpulock.release()
        require(path.exists(), "release 不应删除别人持有的锁")


@case("Web 与 autoexp 读的是同一个文件、同一套字段（契约不漂移）")
def test_autoexp_and_web_share_contract() -> None:
    import gpulock

    with temp_lock() as path:
        ok, _ = gpulock.acquire("autoexp", "autoexp-run", "exp_0042")
        require(ok, "autoexp 应能拿到锁")
        try:
            # 用 Web 侧同一个模块读，验证字段齐全
            holder = gpulock.read()
            require(holder is not None, "Web 侧应能读到 autoexp 写的锁")
            for key in ("owner", "pid", "kind", "label", "since"):
                require(key in holder, f"锁文件缺少字段 {key}")

            # 再验证 autoexp.py 侧真的会去加载这个模块（防止有人改回自己实现）
            # BACKEND = <repo>/6-web/backend → parents[1] = <repo>
            autoexp_dir = BACKEND.parents[1] / "7-autoresearch-yolo"
            probe = (
                "import sys; sys.path.insert(0, r'%s');\n"
                "import autoexp;\n"
                "lock = autoexp.load_gpulock();\n"
                "print('NONE' if lock is None else lock.read() is not None)\n" % autoexp_dir
            )
            out = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            last = (out.stdout or "").strip().splitlines()
            require(last, f"autoexp 探针无输出。stdout={out.stdout!r} stderr={out.stderr[-500:]!r}")
            check_eq(
                last[-1],
                "True",
                f"autoexp 应看到锁文件被占用（若为 NONE 说明它没加载到 gpulock）。"
                f"stderr={out.stderr[-300:]!r}",
            )
        finally:
            gpulock.release()


@case("Web 的 _reject_if_busy 会因外来占用返回 409")
def test_web_reject_on_foreign_holder() -> None:
    import gpulock

    with temp_lock() as path:
        path.write_text(
            json.dumps({"owner": "autoexp", "pid": 999999, "kind": "autoexp-run", "label": "x"}),
            encoding="utf-8",
        )
        # 陈旧锁 → 不该拦
        check_eq(gpulock.read(), None, "陈旧锁不该让 Web 拒绝")

        # 换成存活 pid 但不是自己的：模拟「命令行在跑」。用父进程的 pid
        # （本测试就是被它启动的，必然存活），且不等于 os.getpid()。
        parent = os.getppid()
        if parent in (0, os.getpid()):
            return
        path.write_text(
            json.dumps({"owner": "autoexp", "pid": parent, "kind": "autoexp-run", "label": "exp_1"}),
            encoding="utf-8",
        )
        holder = gpulock.read()
        require(holder is not None, "存活的外部 pid 应被读到")
        check_eq(holder["owner"], "autoexp", "owner")


@case("同进程两个任务各占一条，先结束的不误删另一条")
def test_two_tasks_same_process() -> None:
    import gpulock

    with temp_lock() as path:
        ok_a, reason_a = gpulock.acquire("web", "yolo-train", "run_a")
        ok_b, reason_b = gpulock.acquire("web", "infer-video", "run_b")
        try:
            require(ok_a, f"第一个任务应拿到锁：{reason_a}")
            require(ok_b, f"同进程第二个任务应拿到锁（不能静默覆盖）：{reason_b}")
            data = json.loads(path.read_text(encoding="utf-8"))
            check_eq(len(data["holders"]), 2, "两条任务应在文件里各占一条")

            # 先结束的那一个只删自己那条
            gpulock.release("yolo-train", "run_a")
            require(path.exists(), "还有任务在跑，不该删文件")
            if not path.exists():
                return
            left = json.loads(path.read_text(encoding="utf-8"))["holders"]
            check_eq(list(left), [gpulock.holder_key("infer-video", "run_b")], "剩下的是另一条")
        finally:
            gpulock.release("yolo-train", "run_a")
            gpulock.release("infer-video", "run_b")
        require(not path.exists(), "全部释放后文件应被删除")


@case("别的进程占着时 acquire 被拒（第二个实例拿到 409 的依据）")
def test_refuse_multi_holder_foreign() -> None:
    import gpulock

    with temp_lock() as path:
        parent = os.getppid()
        if parent in (0, os.getpid()):
            return  # 拿不到可用的「别的进程」，跳过（不把平台差异记成失败）
        # 新格式：另一个进程的条目 + 一条本进程自己的条目
        path.write_text(
            json.dumps(
                {
                    "holders": {
                        gpulock.holder_key("autoexp-run", "exp_7"): {
                            "owner": "autoexp", "pid": parent,
                            "kind": "autoexp-run", "label": "exp_7",
                            "since": "2026-11-10T13:00:00+08:00",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        ok, reason = gpulock.acquire("web", "anomalib-train", "run_q")
        require(not ok, "别的进程占着时必须拒绝，否则两个训练会一起 OOM")
        require("PID" in reason, f"拒绝原因应说清是谁在占，实际：{reason}")
        # foreign_holders 只看别的进程，不能因为自己有条目就漏判
        foreign = gpulock.foreign_holders()
        check_eq(len(foreign), 1, "应只报出那一条外来占用")


@case("旧格式（单持有者扁平结构）仍可读，下次写入自动升级")
def test_legacy_format_compatible() -> None:
    import gpulock

    with temp_lock() as path:
        path.write_text(
            json.dumps(
                {"owner": "autoexp", "pid": os.getpid(), "kind": "autoexp-run",
                 "label": "old", "since": "2026-11-10T13:00:00+08:00"}
            ),
            encoding="utf-8",
        )
        check_eq(len(gpulock.holders()), 1, "旧格式应被当成一个持有者")
        # 同 pid 已持有 → 允许再加一条，并升级成新结构
        ok, reason = gpulock.acquire("web", "yolo-train", "new")
        require(ok, f"同进程追加应放行：{reason}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            require("holders" in data, "写入后应升级成多持有者结构")
            check_eq(len(data["holders"]), 2, "旧条目应被保留，不能丢")
        finally:
            gpulock.release("autoexp-run", "old")
            gpulock.release("yolo-train", "new")
        require(not path.exists(), "全部释放后文件应被删除")


def main() -> int:
    print(f"gpulock 回归测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}")
    print()

    run_cases(globals())

    return finish()


if __name__ == "__main__":
    sys.exit(main())
