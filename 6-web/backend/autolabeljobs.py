# -*- coding: utf-8 -*-
"""整工程批量预标注的后台任务。

为什么不用 `RunManager`（`/api/runs` + WebSocket）：那条链路是 **子进程 + stdout**，
每张图起一个 python 进程要几秒（还要重新加载 DLL / 反序列化引擎），100 张图光启动开销
就是十几分钟。这里的活是"同一进程里反复调同一个引擎"，`infer.run_yolo()` 内部已经按
引擎缓存句柄并用锁串行化，所以用**线程 + 内存任务表 + 轮询**最合适。

代价要写在明处：服务重启后任务进度会丢（进程内存），但**已写进旁挂文件的草稿不会丢**——
每张图处理完立刻原子写盘，本来就是为了让"重启/崩溃"最多损失当前这一张。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import annotation_drafts

#: 保留的已完成任务数。前端刷新页面后要能找到刚才的任务，但也不能无限攒。
MAX_JOBS = 8
#: 失败明细上限：够定位问题即可，不把整批的错误堆在内存里。
MAX_FAILURES = 200
#: 警告明细上限（同类警告会在每张图上重复出现）。
MAX_WARNINGS = 40

_JOBS: dict[str, "BatchJob"] = {}
_REGISTRY_LOCK = threading.Lock()


class BatchError(Exception):
    """批量任务的参数/状态错误（交给接口转成 400）。"""


@dataclass
class BatchJob:
    """一个批量预标注任务的运行状态。

    所有可变字段都在 ``lock`` 保护下读写：工作线程在改，HTTP 轮询在读，
    不加锁时 `snapshot()` 可能读到"done=5 但 results 里只有 3 条"的中间态。
    """

    id: str
    project_path: str
    options: dict
    total: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    done: int = 0
    failed: int = 0
    drafts: int = 0
    empty: int = 0
    status: str = "running"
    current: str = ""
    error: str = ""
    started_ns: int = field(default_factory=time.time_ns)
    ended_ns: int = 0
    failures: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    saved_images: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None

    # -- 工作线程写入 ----------------------------------------------------
    def begin(self, item: dict) -> None:
        with self.lock:
            self.current = item.get("id") or item.get("path") or ""

    def succeed(self, item: dict, count: int, warnings: list[str]) -> None:
        with self.lock:
            self.done += 1
            self.drafts += count
            if count == 0:
                self.empty += 1
            image_id = item.get("id") or ""
            if image_id and len(self.saved_images) < 20000:
                self.saved_images.append(image_id)
            for text in warnings:
                text = str(text).strip()
                if text and text not in self.warnings and len(self.warnings) < MAX_WARNINGS:
                    self.warnings.append(text)

    def fail(self, item: dict, error: object) -> None:
        message = str(error).strip() or error.__class__.__name__
        with self.lock:
            self.done += 1
            self.failed += 1
            if len(self.failures) < MAX_FAILURES:
                self.failures.append(
                    {"id": item.get("id") or "", "path": item.get("path") or "", "error": message}
                )

    def stopping(self) -> bool:
        return self.stop_event.is_set()

    def finish(self) -> None:
        with self.lock:
            if self.status == "running":
                self.status = "stopped" if self.stop_event.is_set() else "done"
            self.ended_ns = time.time_ns()
            self.current = ""

    def crash(self, error: object) -> None:
        with self.lock:
            self.status = "failed"
            self.error = str(error).strip() or error.__class__.__name__

    # -- HTTP 读取 -------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            elapsed_ms = 0
            if self.ended_ns:
                elapsed_ms = int((self.ended_ns - self.started_ns) / 1_000_000)
            elif self.started_ns:
                elapsed_ms = int((time.time_ns() - self.started_ns) / 1_000_000)
            return {
                "id": self.id,
                "status": self.status,
                "stopping": self.stop_event.is_set() and self.status == "running",
                "project": self.project_path,
                "options": dict(self.options),
                "total": self.total,
                "done": self.done,
                "pending": max(0, self.total - self.done),
                "failed": self.failed,
                "drafts": self.drafts,
                "empty": self.empty,
                "current": self.current,
                "error": self.error,
                "started_ns": str(self.started_ns),
                "ended_ns": str(self.ended_ns),
                "elapsed_ms": elapsed_ms,
                "failures": list(self.failures),
                "warnings": list(self.warnings),
                "saved_images": list(self.saved_images),
            }


def _prune_locked() -> None:
    """丢掉超出上限的**已结束**任务；运行中的永远保留。"""
    if len(_JOBS) <= MAX_JOBS:
        return
    finished = [
        job for job in _JOBS.values() if job.status != "running"
    ]
    finished.sort(key=lambda job: job.started_ns)
    for job in finished[: max(0, len(_JOBS) - MAX_JOBS)]:
        _JOBS.pop(job.id, None)


def start(project_path: str, items: list[dict], options: dict) -> dict:
    """创建并启动任务，返回启动瞬间的快照。

    ``items`` 必须是已经过滤过的 ``[{"id", "path"}]``：过滤（scope/白名单/存在性）
    放在接口层，任务只负责"把给我的图片一张张跑完"，这样任务本身好测。
    """
    if not items:
        raise BatchError("没有需要处理的图片（可能都被跳过或工程是空的）")

    job = BatchJob(
        id=uuid.uuid4().hex[:12],
        project_path=project_path,
        options=dict(options),
        total=len(items),
    )
    with _REGISTRY_LOCK:
        _prune_locked()
        _JOBS[job.id] = job

    thread = threading.Thread(
        target=_run,
        args=(job, list(items)),
        name=f"autolabel-{job.id}",
        daemon=True,
    )
    job.thread = thread
    thread.start()
    return job.snapshot()


def _run(job: BatchJob, items: list[dict]) -> None:
    """工作线程主体：逐图推理、逐图落盘、单张失败不中断。

    两条通道（``options["mode"]``）：
    - ``engine``：调用 ``autolabel.run``（.engine / DLL / 外部服务三条路都在里面）；
    - ``text``：调用 ``textlabel``，文本模型只在第一张图时加载一次（模块内常驻），
      并把 ``refine`` 透传给它，让"文本出框 + SAM 细化"在批量里也能用。

    为什么不用 ``autolabel.run`` 统一包一层：文本通道不需要引擎、不需要标签文件，
    硬塞进 ``run`` 会让那个函数长出更多"这个参数在文本模式下无意义"的分支。
    """
    import autolabel  # 延迟导入：没有 DLL 的环境也能 import 本模块做单测

    options = job.options
    mode = str(options.get("mode") or "engine").strip().lower()
    try:
        if mode == "text":
            result = _run_text(job, items, options)
        else:
            result = _run_engine(job, items, options, autolabel)
        if result is not None:
            job.crash(result)
    except Exception as exc:  # noqa: BLE001 - 线程里的意外必须反映到状态里，不能静默死掉
        job.crash(exc)
    finally:
        job.finish()


def _run_engine(job: "BatchJob", items: list[dict], options: dict, autolabel) -> Exception | None:
    """引擎通道：返回 ``None`` 表示正常走完，返回异常表示整批崩了。"""
    for item in items:
        if job.stopping():
            break
        job.begin(item)
        try:
            result = autolabel.run(
                image_path=item["path"],
                task=options.get("task", "detect"),
                engine=options["engine"],
                labels_path=options.get("labels") or None,
                conf=float(options.get("conf", 0.25)),
                nms=float(options.get("nms", 0.45)),
                service=options.get("service") or None,
                prompts=None,
                timeout=float(options.get("timeout", 20.0)),
            )
            drafts = result.get("drafts") or []
        except Exception as exc:  # noqa: BLE001 - 单张失败必须被记录而不是炸掉整批
            job.fail(item, exc)
            continue
        if not _persist(job, item, drafts):
            continue
        job.succeed(item, len(drafts), result.get("warnings") or [])
    return None


def _run_text(job: "BatchJob", items: list[dict], options: dict) -> Exception | None:
    """文本通道：先加载模型，之后逐图跑。"""
    import promptlabel
    import textlabel

    prompt = str(options.get("prompt") or "").strip()
    if not prompt:
        return BatchError("文本通道需要提示词")
    try:
        session = textlabel.get_session()  # 一次加载，整批复用
    except Exception as exc:  # noqa: BLE001 - 模型加载失败整批都没意义
        return exc

    for item in items:
        if job.stopping():
            break
        job.begin(item)
        try:
            image = promptlabel.read_image(item["path"])
            result = session.predict(
                image,
                prompt,
                box_thr=float(options.get("box_thr", 0.3)),
                text_thr=float(options.get("text_thr", 0.25)),
                refine=bool(options.get("refine")),
                image_key=promptlabel.modelzoo_placeholder_key(item["path"]),
            )
            drafts = result.get("drafts") or []
        except Exception as exc:  # noqa: BLE001
            job.fail(item, exc)
            continue
        if not _persist(job, item, drafts):
            continue
        job.succeed(item, len(drafts), result.get("warnings") or [])
    return None


def _persist(job: "BatchJob", item: dict, drafts: list[dict]) -> bool:
    """把一张图的结果落盘。返回 ``False`` 表示这张图算失败（已记账）。"""
    try:
        # 先落盘再记账：写盘失败时这张图的结果本来就没保存，
        # 计成成功会让统计与磁盘对不上（用户按"已完成"去审核会看到空列表）。
        annotation_drafts.merge(job.project_path, {item["id"]: drafts})
    except (OSError, ValueError) as exc:
        job.fail(item, f"草稿写盘失败：{exc}")
        return False
    return True


def get(job_id: str) -> dict | None:
    with _REGISTRY_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
    return job.snapshot() if job else None


def stop(job_id: str) -> dict | None:
    """请求停止。**不 kill 线程**：当前那张图会跑完再退出，结果不浪费。"""
    with _REGISTRY_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
        if job is None:
            return None
        job.stop_event.set()
    return job.snapshot()


def recent(limit: int = MAX_JOBS) -> list[dict]:
    with _REGISTRY_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda job: job.started_ns, reverse=True)
    return [job.snapshot() for job in jobs[: max(1, int(limit))]]


def wait(job_id: str, timeout: float = 30.0) -> dict | None:
    """等任务结束（测试用）。超时返回当前快照，不阻塞调用方。"""
    with _REGISTRY_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
    if job is None:
        return None
    if job.thread is not None:
        job.thread.join(timeout)
    return job.snapshot()


def reset_for_tests() -> None:
    """清空任务表（只给测试用，避免用例之间互相看见对方的任务）。"""
    with _REGISTRY_LOCK:
        for job in _JOBS.values():
            job.stop_event.set()
        _JOBS.clear()
