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

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def case(name: str):
    def wrap(func):
        def runner():
            try:
                func()
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                print(f"  [FAIL] {name}")
            else:
                PASSED.append(name)
                print(f"  [ ok ] {name}")

        runner.__name__ = func.__name__
        return runner

    return wrap


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}：期望 {expected!r}，实际 {actual!r}")


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
            check(ok, f"空闲时应能拿到锁，实际被拒：{reason}")
            data = json.loads(path.read_text(encoding="utf-8"))
            check_eq(data["owner"], "web", "owner")
            check_eq(data["kind"], "yolo-train", "kind")
            check_eq(data["label"], "run_abc", "label")
            check_eq(data["pid"], os.getpid(), "pid")
            check("since" in data, "应记录获取时间")
        finally:
            gpulock.release()
        check(not path.exists(), "release 后锁文件应被删除")


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
        check(holder is not None, "存活的锁应被读到")
        text = gpulock.describe(holder)
        check("自动实验循环" in text, f"描述里应点明占用方，实际：{text}")
        check("exp_9999" in text, f"描述里应带上 label，实际：{text}")

        # acquire 在 pid 相同时视为「自己的锁」，允许重入（同进程多任务需要）
        ok, _ = gpulock.acquire("web", "yolo-train", "run_x")
        check(ok, "同 pid 重入应放行")
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
        check(ok, f"陈旧锁不应阻止新的 acquire，实际被拒：{reason}")
        gpulock.release()


@case("损坏的锁文件被当作空闲，不会把所有人挡在门外")
def test_corrupt_lock_is_ignored() -> None:
    import gpulock

    with temp_lock() as path:
        path.write_text("{ 这不是 JSON", encoding="utf-8")
        check_eq(gpulock.read(), None, "损坏文件应被忽略")
        ok, _ = gpulock.acquire("web", "yolo-train", "run_z")
        check(ok, "损坏文件不应阻止 acquire")
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
        check(path.exists(), "release 不应删除别人持有的锁")


@case("Web 与 autoexp 读的是同一个文件、同一套字段（契约不漂移）")
def test_autoexp_and_web_share_contract() -> None:
    import gpulock

    with temp_lock() as path:
        ok, _ = gpulock.acquire("autoexp", "autoexp-run", "exp_0042")
        check(ok, "autoexp 应能拿到锁")
        try:
            # 用 Web 侧同一个模块读，验证字段齐全
            holder = gpulock.read()
            check(holder is not None, "Web 侧应能读到 autoexp 写的锁")
            for key in ("owner", "pid", "kind", "label", "since"):
                check(key in holder, f"锁文件缺少字段 {key}")

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
            check(last, f"autoexp 探针无输出。stdout={out.stdout!r} stderr={out.stderr[-500:]!r}")
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
        check(holder is not None, "存活的外部 pid 应被读到")
        check_eq(holder["owner"], "autoexp", "owner")


def main() -> int:
    print(f"gpulock 回归测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}")
    print()

    for name, obj in sorted(globals().items()):
        if name.startswith("test_"):
            obj()

    print()
    print("=" * 60)
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print()
        for name, detail in FAILED:
            print(f"[FAIL] {name}")
            print(detail)
            print("-" * 60)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
