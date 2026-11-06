"""启动阶段事件（phase）的验证。

覆盖：
1. 事件字段契约：elapsed_ms 单调递增、step_ms >= 0、带 pid / python；
2. warm 进程的阶段时钟会归零：复用预热时，耗时从「收到任务」开始算，
   否则用户会看到「启动用了 40 秒」，其实那 40 秒是服务空闲时预热掉的；
3. 真实 launcher 的预热握手：先 boot(warm=false) -> warm_ready -> 下发任务后 boot(warm=true)。

为什么值得单独测：
前端「启动清单」完全依赖这些事件。字段少一个，界面就会少一行或显示 undefined，
而这种问题在训练成功时不会暴露，只有用户盯着启动阶段时才看得见。

用法：
    python backend/tools/test_phase_events.py
"""

from __future__ import annotations

import io
import json
import queue
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BACKEND_DIR.parent
YOLO_LAUNCHER = WEB_DIR.parent / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py"
ANOMALIB_LAUNCHER = WEB_DIR.parent / "2-python-anomaly" / "launcher" / "tf_anomalib_launcher.py"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(YOLO_LAUNCHER.parent))
sys.path.insert(0, str(ANOMALIB_LAUNCHER.parent))

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


def capture_phase(module, phase: str, label: str) -> dict:
    """调用 emit_phase 并把它打印的那行 JSON 解析回来。"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        module.emit_phase(phase, label)
    line = buffer.getvalue().strip().splitlines()[-1]
    prefix = module.JSON_PREFIX
    assert line.startswith(prefix), line
    return json.loads(line[len(prefix):])


def test_contract() -> None:
    print("\n[1] 事件字段契约")
    import tf_yolo_launcher as Y

    first = capture_phase(Y, "boot", "python 进程已启动")
    time.sleep(0.02)
    second = capture_phase(Y, "deps", "训练依赖已导入")

    check("type=phase", first.get("type") == "phase")
    check("带 phase / label", first.get("phase") == "boot" and first.get("label") == "python 进程已启动")
    check("带 pid 与 python", isinstance(first.get("pid"), int) and bool(first.get("python")))
    check("elapsed_ms 单调不减", second["elapsed_ms"] >= first["elapsed_ms"], f"{first} / {second}")
    check("step_ms >= 0", second["step_ms"] >= 0)

    import tf_anomalib_launcher as A

    anomalib_phase = capture_phase(A, "boot", "python 进程已启动")
    check("两个 launcher 字段一致", set(anomalib_phase) == set(first), str(anomalib_phase))


def test_warm_clock() -> None:
    print("\n[2] 预热复用时阶段时钟归零")
    import tf_yolo_launcher as Y

    Y._TASK_T0 = None
    check("冷启动返回 False", Y.reset_phase_clock() is False)
    Y._TASK_T0 = time.perf_counter() + 100  # 人为制造一个「很久以前就开始预热」的场景
    check("复用预热返回 True", Y.reset_phase_clock() is True)
    check("标记被消费掉", Y._TASK_T0 is None)

    captured = capture_phase(Y, "boot", "python 进程已启动")
    check("归零后 elapsed_ms 很小", captured["elapsed_ms"] < 1000, str(captured))


class EventReader:
    """一个进程只起一个读线程，事件先入队再按需取。

    注意不能对同一个 stdout 起两个读线程：两个 readline 会互相抢行，
    表现为「事件明明打了却等不到」。这正是本测试第一版踩到的坑。
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._process.stdout is not None
        for raw in iter(self._process.stdout.readline, b""):
            self._lines.put(raw.decode("utf-8", errors="replace"))
        self._lines.put("")

    def read_until(self, predicate, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if not line:
                raise AssertionError("进程提前结束")
            text = line.strip()
            if not text.startswith("TRTFORGE_JSON "):
                continue
            try:
                payload = json.loads(text[len("TRTFORGE_JSON "):])
            except json.JSONDecodeError:
                continue
            if predicate(payload):
                return payload
        raise AssertionError("等待事件超时")


def test_real_warm_handshake() -> None:
    print("\n[3] 真实 launcher 的预热握手")
    if not YOLO_LAUNCHER.is_file():
        check("YOLO launcher 存在", False, str(YOLO_LAUNCHER))
        return

    process = subprocess.Popen(
        [sys.executable, str(YOLO_LAUNCHER), "--mode", "warm"],
        cwd=str(YOLO_LAUNCHER.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        reader = EventReader(process)
        boot = reader.read_until(lambda e: e.get("phase") == "boot", 30)
        check("预热进程先报 boot(warm=false)", boot.get("warm") is False)

        ready = reader.read_until(lambda e: e.get("name") == "warm_ready", 90)
        check("预热完成发 warm_ready", ready.get("name") == "warm_ready")
        check("warm_ready 带耗时", isinstance(ready.get("elapsed_ms"), int))

        assert process.stdin is not None
        process.stdin.write(json.dumps({"argv": ["--mode", "check-env"]}).encode("utf-8") + b"\n")
        process.stdin.flush()

        warm_boot = reader.read_until(lambda e: e.get("phase") == "boot", 60)
        check("命中预热后 boot(warm=true)", warm_boot.get("warm") is True)
        check("预热复用的启动耗时归零", warm_boot.get("elapsed_ms", 9999) < 1000, str(warm_boot))
    finally:
        process.kill()


def main() -> int:
    print("启动阶段事件验证")
    test_contract()
    test_warm_clock()
    test_real_warm_handshake()
    print(f"\n结果：通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
