# -*- coding: utf-8 -*-
"""自动实验让出推理显存：退出后不许自己拉起，解除后才按需重载。

不启动真实推理进程，也不占显卡。startup 被换成哨兵，用来确认
「让出期间一次都不会拉起，解除之后的下一次请求才会拉起」。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("TRTFORGE_INFER_ISOLATION", "1")

import inferproxy  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_yield_blocks_restart_until_release() -> None:
    pool = inferproxy._pool
    inferproxy.release_gpu_hold()
    started: list[str] = []
    original = pool.startup

    def sentinel() -> None:
        started.append("startup")
        raise RuntimeError("sentinel-start")

    pool.startup = sentinel  # type: ignore[method-assign]
    try:
        info = inferproxy.yield_gpu("单元测试")
        check(info.get("held") is True, f"应处于让出：{info}")
        check(pool.gpu_hold() == "单元测试", f"原因应记下：{pool.gpu_hold()!r}")

        try:
            pool.call({"op": "status"})
        except inferproxy.WorkerError as exc:
            check(exc.status == 503, f"应是 503，实际 {exc.status}")
            check("单元测试" in str(exc), f"错误应说明让给谁：{exc}")
        else:
            raise AssertionError("让出期间推理请求不该被放行")
        check(started == [], f"让出期间不该拉起进程：{started}")

        state = pool.state()
        check(state.get("state") == "yielded", f"状态应是 yielded：{state}")
        refused = inferproxy.restart_worker()
        check(refused.get("refused") is True, f"训练期间手动重启应被拒绝：{refused}")
        check(started == [], "拒绝重启时也不该拉起")

        released = inferproxy.release_gpu_hold()
        check(released.get("held") is False, f"应已解除：{released}")
        check(released.get("reloaded") is False, "解除时不能马上把进程拉起来")
        check(pool.gpu_hold() == "", "解除后标记应为空")

        try:
            pool.call({"op": "status"})
        except RuntimeError as exc:
            check("sentinel-start" in str(exc), f"解除后的下一次请求应尝试拉起：{exc}")
        else:
            raise AssertionError("解除后应走进按需拉起")
        check(started == ["startup"], f"应只拉起一次：{started}")
    finally:
        pool.startup = original  # type: ignore[method-assign]
        inferproxy.release_gpu_hold()


def test_direct_mode_also_refuses() -> None:
    previous = os.environ.get("TRTFORGE_INFER_ISOLATION")
    os.environ["TRTFORGE_INFER_ISOLATION"] = "0"
    try:
        info = inferproxy.yield_gpu("直通测试")
        check(info.get("mode") == "direct", f"直通模式：{info}")
        try:
            inferproxy.run_yolo("unused.engine", "detect", None)  # type: ignore[arg-type]
        except inferproxy.WorkerError as exc:
            check("直通测试" in str(exc), f"直通模式也要拦住：{exc}")
        else:
            raise AssertionError("直通模式让出后不该继续推理")
        inferproxy.release_gpu_hold()
        check(inferproxy._DIRECT_HOLD == "", "解除后直通标记应清空")
    finally:
        if previous is None:
            os.environ.pop("TRTFORGE_INFER_ISOLATION", None)
        else:
            os.environ["TRTFORGE_INFER_ISOLATION"] = previous
        inferproxy.release_gpu_hold()


def test_repeat_yield_and_release_are_safe() -> None:
    """连续让出、连续解除都不能把标记弄乱，也不能抛异常。"""
    pool = inferproxy._pool
    inferproxy.release_gpu_hold()
    try:
        first = inferproxy.yield_gpu("第一次")
        second = inferproxy.yield_gpu("第二次")
        check(first.get("held") is True and second.get("held") is True, "两次让出都应成功")
        check(pool.gpu_hold() == "第二次", f"后一次原因应覆盖前一次：{pool.gpu_hold()!r}")
        inferproxy.release_gpu_hold()
        again = inferproxy.release_gpu_hold()
        check(again.get("held") is False, f"第二次解除仍应是未占用：{again}")
        check(pool.gpu_hold() == "", "连续解除后标记必须为空")
    finally:
        inferproxy.release_gpu_hold()


def test_isolated_run_yolo_refuses_without_starting() -> None:
    """隔离模式下，让出期间 run_yolo 直接 503，不能走到拉起进程。"""
    pool = inferproxy._pool
    inferproxy.release_gpu_hold()
    started: list[str] = []
    original = pool.startup

    def sentinel() -> None:
        started.append("startup")
        raise RuntimeError("should-not-start")

    pool.startup = sentinel  # type: ignore[method-assign]
    previous = os.environ.get("TRTFORGE_INFER_ISOLATION")
    os.environ["TRTFORGE_INFER_ISOLATION"] = "1"
    try:
        inferproxy.yield_gpu("隔离拦截")
        try:
            inferproxy.run_yolo("unused.engine", "detect", None)  # type: ignore[arg-type]
        except inferproxy.WorkerError as exc:
            check(exc.status == 503, f"应是 503，实际 {exc.status}")
            check("隔离拦截" in str(exc), f"应说明让给谁：{exc}")
        else:
            raise AssertionError("让出期间不该进入真正推理")
        check(started == [], f"不应拉起进程：{started}")
    finally:
        pool.startup = original  # type: ignore[method-assign]
        if previous is None:
            os.environ.pop("TRTFORGE_INFER_ISOLATION", None)
        else:
            os.environ["TRTFORGE_INFER_ISOLATION"] = previous
        inferproxy.release_gpu_hold()


def test_route_refuses_while_yielded_and_reloads_after_release() -> None:
    """网页选后端时：让出中直接 503；解除后进程虽停着，仍要选推理进程，好让下次请求把它拉起。"""
    import app as web_app

    inferproxy.release_gpu_hold()
    inferproxy.yield_gpu("选择层")
    try:
        try:
            web_app._select_yolo_backend("unused.engine")
        except inferproxy.WorkerError as exc:
            check(exc.status == 503, f"应是 503，实际 {exc.status}")
            check("选择层" in str(exc), f"应说明让给谁：{exc}")
        else:
            raise AssertionError("让出期间不该去找 DLL 或降级模型")
        inferproxy.release_gpu_hold()
        use_dll, onnx_path, warning = web_app._select_yolo_backend("unused.engine")
        check(use_dll is True, f"解除后应走推理进程：{(use_dll, onnx_path, warning)}")
        check(inferproxy.worker_state().get("state") != "running", "选后端本身不能把进程拉起来")
    finally:
        inferproxy.release_gpu_hold()


def test_finalize_survives_callback_error() -> None:
    """结束回调自己崩了，任务状态仍要落成 finished，不能把收尾打断。"""
    import runners

    class FakeProc:
        def wait(self, timeout=None):
            return 0

    called: list[str] = []

    def boom() -> None:
        called.append("called")
        raise RuntimeError("回调故意失败")

    run = runners.Run(
        run_id="yield-finalize",
        kind="autoexp-agent",
        command=["autoexp"],
        cwd=".",
        process=FakeProc(),
        status="running",
        on_finished=boom,
    )
    runners.RunManager()._finalize(run)
    check(called == ["called"], f"回调应被调用：{called}")
    check(run.status == "finished", f"回调失败不能改掉任务状态：{run.status}")
    check(run.exit_code == 0, f"退出码应记下：{run.exit_code}")


def main() -> int:
    tests = [
        test_yield_blocks_restart_until_release,
        test_direct_mode_also_refuses,
        test_repeat_yield_and_release_are_safe,
        test_isolated_run_yolo_refuses_without_starting,
        test_route_refuses_while_yielded_and_reloads_after_release,
        test_finalize_survives_callback_error,
    ]
    for fn in tests:
        fn()
        print(f"  [ ok ] {fn.__name__}")
    print(f"通过 {len(tests)} / {len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
