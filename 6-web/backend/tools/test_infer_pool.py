#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理进程池（多 worker）验收。

覆盖：
1. ``TRTFORGE_INFER_WORKERS=2`` 确实起了 2 个进程，且状态字段仍与单进程时代兼容；
2. 并发请求会落在不同的进程上（真的并行，不是排队）；
3. 杀进程池里的某个进程，其余进程继续服务，被杀的那个自动拉起；
4. 单个进程 60 秒崩 3 次只停它自己，另一个照常出结果；
5. 全池 60 秒累计崩 6 次才全池熔断，状态把后端报成不可用；
6. 进程都在忙时按 ``TRTFORGE_INFER_QUEUE_TIMEOUT`` 报 503，不会无限挂住；
7. OCR 永远只在 0 号进程（不跟着池子扩散，省掉 1.7GB × N）；
8. 有模型时：``WORKERS=1`` 与 ``WORKERS=2`` 的框完全一致，并给出并发吞吐对比。

没有 DLL / 模型的环境（CI）只跑 1、3、5、6，模型相关项跳过。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ["TRTFORGE_TEST_HOOKS"] = "1"
os.environ["TRTFORGE_INFER_ISOLATION"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import inferproxy  # noqa: E402

REPO = BACKEND.parents[1]
YOLO_ENGINE = REPO / "1-python-yolo/launcher/runs/pcb/detect_v1/weights/best.transd_fp16.engine"
YOLO_IMAGES = REPO / "PCB data/PCB缺陷检测"
OCR_DET = REPO / "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_det_infer"
OCR_REC = REPO / "9-python-paddleOCR/model/legacy_v4/ch_PP-OCRv4_rec_infer"
OCR_IMAGE = REPO / "ocr/PaddleOCR-main/docs/version2.x/ppocr/images/ch_ppocr_mobile_v2.0/00006737.jpg"

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name}（{reason}）")


def configure(workers: int, queue_timeout: float | None = None) -> None:
    """按指定进程数重启整个池。环境变量在 startup 时才读，所以必须先停机再改。"""
    inferproxy.shutdown()
    os.environ[inferproxy.WORKERS_ENV] = str(workers)
    if queue_timeout is not None:
        os.environ[inferproxy.QUEUE_TIMEOUT_ENV] = str(queue_timeout)
    inferproxy.startup()


def wait_workers_ready(timeout: float = 120.0, allow_tripped: bool = False) -> bool:
    """等所有进程进入 running（后台启动的进程需要时间加载 DLL）。

    本身已熔断的进程永远不会变 running，验证「熔断」的用例要把 allow_tripped 打开。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        workers = inferproxy.worker_state().get("workers") or []
        if workers:
            def ok(worker: dict) -> bool:
                return worker["state"] == "running" or (allow_tripped and worker["state"] == "tripped")
            if all(ok(w) for w in workers):
                return True
        time.sleep(0.3)
    return False


def read_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def sorted_boxes(result: dict) -> list[tuple]:
    keys = ("class_label", "left", "top", "right", "bottom", "confidence")
    return sorted(tuple(round(float(box[k]), 3) for k in keys) for box in result["boxes"])


def kill_pid(pid: int) -> None:
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


def test_auto_workers() -> None:
    """未显式设环境变量时应按显存自动决定，而不是固定 1。"""
    saved = os.environ.pop(inferproxy.WORKERS_ENV, None)
    try:
        auto = inferproxy.worker_count()
        expected = inferproxy._auto_workers()
        memory = inferproxy._gpu_memory_mb()
        check("未设环境变量时按显存自动定进程数", auto == expected and expected in (1, 2),
              f"显存 {memory} MB → {auto} 个")
    finally:
        if saved is not None:
            os.environ[inferproxy.WORKERS_ENV] = saved


def test_pool_shape() -> None:
    configure(2)
    check("起 2 个推理进程", wait_workers_ready(), str(inferproxy.worker_state().get("workers")))
    state = inferproxy.worker_state()
    check("pool_size 反映进程数", state.get("pool_size") == 2, str(state.get("pool_size")))
    check("顶层字段仍兼容单进程时代（state/pid/restarts/last_error）",
          state.get("state") == "running" and state.get("pid") and "restarts" in state and "last_error" in state,
          str({k: state.get(k) for k in ("state", "pid", "restarts")}))
    pids = {w["pid"] for w in state["workers"]}
    check("两个进程 pid 不同", len(pids) == 2 and None not in pids, str(pids))
    logs = {p.name for p in config.LOG_DIR.glob("infer_worker*.log")}
    check("多进程日志分开写", "infer_worker.log" in logs and "infer_worker-2.log" in logs, str(sorted(logs)))
    status = inferproxy.status()
    check("状态接口在池模式下可用", isinstance(status.detail, str), status.detail)


def test_parallel_dispatch() -> None:
    if not wait_workers_ready():
        skip("并发分发落不同进程", "进程未全部就绪")
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        served = list(pool.map(lambda _: inferproxy._test_pool_call("pid"), range(8)))
    check("并发请求落在多个进程上（真的并行）", len(set(served)) >= 2, f"服务进程 {sorted(set(served))}")


def test_single_crash_isolated() -> None:
    configure(2)
    if not wait_workers_ready():
        skip("单进程崩溃隔离", "进程未全部就绪")
        return
    victim = inferproxy.worker_state()["workers"][1]["pid"]
    kill_pid(victim)
    time.sleep(0.5)
    ok, detail = True, ""
    try:
        for _ in range(4):
            inferproxy._test_pool_call("pid")
    except inferproxy.WorkerError as exc:
        ok, detail = False, f"{exc.status} {exc}"
    check("1 号进程被杀后其余进程仍能服务", ok, detail or f"被杀 pid {victim}")
    check("被杀进程自动拉起", wait_workers_ready(), str(inferproxy.worker_state().get("workers")))


def test_per_worker_trip() -> None:
    configure(2)
    if not wait_workers_ready():
        skip("单进程熔断不影响另一个", "进程未全部就绪")
        return
    inferproxy.restart_worker()
    # 固定崩 0 号进程（_test_call 只走 0 号），避免崩到随机一个进程
    for i in range(inferproxy.CRASH_LIMIT):
        try:
            inferproxy._test_call("crash")
        except inferproxy.WorkerError:
            pass
        wait_workers_ready(60, allow_tripped=True)
        if i + 1 < inferproxy.CRASH_LIMIT:
            time.sleep(0.3)
    workers = inferproxy.worker_state()["workers"]
    check("0 号进程崩 3 次后自身熔断", workers[0]["state"] == "tripped", str(workers[0]))
    check("1 号进程不受影响", workers[1]["state"] == "running", str(workers[1]))

    served = []
    ok, detail = True, ""
    try:
        for _ in range(3):
            served.append(inferproxy._test_pool_call("pid"))
    except inferproxy.WorkerError as exc:
        ok, detail = False, f"{exc.status} {exc}"
    check("熔断的进程被移出分发，其余进程继续出结果", ok and len(set(served)) == 1, detail or f"服务进程 {sorted(set(served))}")


def test_pool_trip() -> None:
    # 承接上一个用例：0 号已熔断（池累计 3 次），再把 1 号崩 3 次就凑满 6 次
    if inferproxy.worker_state()["workers"][0]["state"] != "tripped":
        configure(2)
        if not wait_workers_ready():
            skip("全池熔断", "进程未全部就绪")
            return
        inferproxy.restart_worker()
        for _ in range(inferproxy.CRASH_LIMIT):
            try:
                inferproxy._test_call("crash")
            except inferproxy.WorkerError:
                pass

    tripped, detail = False, ""
    for _ in range(inferproxy.CRASH_LIMIT + 2):
        try:
            inferproxy._test_pool_call("crash")
        except inferproxy.WorkerError as exc:
            detail = str(exc)
            if "累计崩溃" in detail:
                tripped = True
                break
        time.sleep(0.5)
        wait_workers_ready(60, allow_tripped=True)
    state = inferproxy.worker_state()
    check("全池累计崩 6 次后熔断", tripped or state.get("pool_tripped"),
          str(state.get("pool_error")) or detail)
    status = inferproxy.status()
    check("全池熔断后后端报不可用（触发 onnxruntime 降级）",
          not status.yolo_ok and not status.anomalib_ok, status.detail)
    inferproxy.restart_worker()
    check("手动重启解除全池熔断",
          inferproxy.worker_state()["state"] == "running" and not inferproxy.worker_state()["pool_tripped"],
          str(inferproxy.worker_state()["state"]))


def test_queue_timeout() -> None:
    configure(1, queue_timeout=1)
    if not wait_workers_ready():
        skip("忙时 503", "进程未就绪")
        return
    holder: list = []

    def occupy() -> None:
        try:
            holder.append(inferproxy._test_pool_call("sleep", seconds=4, timeout=30))
        except Exception as exc:  # noqa: BLE001
            holder.append(exc)

    thread = threading.Thread(target=occupy, daemon=True)
    thread.start()
    time.sleep(0.3)
    started = time.monotonic()
    try:
        inferproxy._test_pool_call("pid")
        status_code, message = 0, "没有抛异常"
    except inferproxy.WorkerError as exc:
        status_code, message = exc.status, str(exc)
    elapsed = time.monotonic() - started
    check("进程都忙时报 503", status_code == 503, f"{status_code} {message}")
    check("等待时间受 QUEUE_TIMEOUT 约束（不无限挂）", elapsed < 5.0, f"{elapsed:.2f}s")
    thread.join(timeout=30)
    os.environ.pop(inferproxy.QUEUE_TIMEOUT_ENV, None)


def test_ocr_pinned() -> None:
    if not (OCR_DET.is_dir() and OCR_REC.is_dir() and OCR_IMAGE.is_file()):
        skip("OCR 固定在 0 号进程", "缺少模型或示例图")
        return
    configure(2)
    if not wait_workers_ready():
        skip("OCR 固定在 0 号进程", "进程未全部就绪")
        return
    if not inferproxy.ocr_status()["ok"]:
        skip("OCR 固定在 0 号进程", inferproxy.ocr_status()["detail"])
        return
    options = {
        "det_dir": str(OCR_DET), "rec_dir": str(OCR_REC), "textline_dir": "",
        "device": "cpu", "enable_mkldnn": 1, "cpu_threads": 4, "rec_batch_size": 6,
        "det_thresh": 0.3, "det_box_thresh": 0.6, "det_unclip_ratio": 1.5, "rec_score_thresh": 0.0,
    }
    image = read_image(OCR_IMAGE)
    inferproxy.run_ocr(image, options, source=OCR_IMAGE.name)
    warm = [w.index for w in inferproxy._pool.workers if w._ocr_warm]
    check("OCR 只在 0 号进程上加载 Paddle", warm == [0], f"已加载的进程 {warm}")


def test_consistency() -> None:
    images = []
    if YOLO_IMAGES.is_dir():
        images = sorted(p for p in YOLO_IMAGES.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})[:5]
    if not images or not YOLO_ENGINE.is_file():
        skip("多进程一致性", "缺少模型或图片")
        return
    if not inferproxy.status().yolo_ok:
        skip("多进程一致性", inferproxy.status().detail)
        return
    arrays = [read_image(p) for p in images]

    configure(1)
    if not wait_workers_ready():
        skip("多进程一致性", "单进程未就绪")
        return
    single = [inferproxy.run_yolo(str(YOLO_ENGINE), "detect", a, source="t") for a in arrays]

    configure(2)
    if not wait_workers_ready():
        skip("多进程一致性", "双进程未就绪")
        return
    duo = [inferproxy.run_yolo(str(YOLO_ENGINE), "detect", a, source="t") for a in arrays]

    same = all(sorted_boxes(x) == sorted_boxes(y) for x, y in zip(single, duo))
    total = sum(len(x["boxes"]) for x in duo)
    check("WORKERS=1 与 WORKERS=2 的框完全一致", same and total > 0, f"{len(arrays)} 张，共 {total} 框")


def bench_concurrency(workers: int, arrays: list[np.ndarray], rounds: int, clients: int) -> tuple[float, float]:
    """返回（QPS, 单请求 P95 毫秒）。clients 个线程同时打，模拟多人同时用。"""
    configure(workers)
    if not wait_workers_ready():
        return 0.0, 0.0
    for array in arrays:  # 预热：别把引擎反序列化算进首批
        inferproxy.run_yolo(str(YOLO_ENGINE), "detect", array)

    latencies: list[float] = []
    lock = threading.Lock()

    def worker(seed: int) -> None:
        local = []
        for i in range(rounds):
            array = arrays[(seed + i) % len(arrays)]
            start = time.perf_counter()
            inferproxy.run_yolo(str(YOLO_ENGINE), "detect", array)
            local.append((time.perf_counter() - start) * 1000)
        with lock:
            latencies.extend(local)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        list(pool.map(worker, range(clients)))
    spent = time.perf_counter() - started
    if not latencies:
        return 0.0, 0.0
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    return (clients * rounds) / spent, p95


def test_bench() -> None:
    images = []
    if YOLO_IMAGES.is_dir():
        images = sorted(p for p in YOLO_IMAGES.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})[:4]
    if not images or not YOLO_ENGINE.is_file():
        skip("并发吞吐对比", "缺少模型或图片")
        return
    if not inferproxy.status().yolo_ok:
        skip("并发吞吐对比", inferproxy.status().detail)
        return
    arrays = [read_image(p) for p in images]
    print()
    print("并发吞吐（4 个客户端同时推理，每客户端 10 次）")
    print(f"{'进程数':<6}{'QPS':>10}{'P95 毫秒':>12}{'相对单进程':>14}")
    base = 0.0
    for workers in (1, 2, 4):
        qps, p95 = bench_concurrency(workers, arrays, 10, 4)
        if workers == 1:
            base = qps
        ratio = qps / base if base else 0.0
        print(f"{workers:<6}{qps:>10.2f}{p95:>12.1f}{ratio:>13.2f}x")
    print()


def main() -> int:
    print("推理进程池")
    try:
        test_auto_workers()
        test_pool_shape()
        test_parallel_dispatch()
        test_single_crash_isolated()
        test_per_worker_trip()
        test_pool_trip()
        test_queue_timeout()
        test_ocr_pinned()
        test_consistency()
        if "--bench" in sys.argv:
            test_bench()
    finally:
        os.environ[inferproxy.WORKERS_ENV] = "1"
        os.environ.pop(inferproxy.QUEUE_TIMEOUT_ENV, None)
        inferproxy.shutdown()

    print()
    print(f"通过 {len(PASSED)}，失败 {len(FAILED)}，跳过 {len(SKIPPED)}")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
