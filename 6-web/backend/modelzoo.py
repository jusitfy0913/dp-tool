# -*- coding: utf-8 -*-
"""内置模型仓库：提示式预标注要用的权重，在标注页直接下载。

为什么需要这个模块：
    v1.23 的预标注只认自己训练出来的 `.engine`。而"点一下出掩码""说一句话出框"
    这两类模型是**通用预训练权重**，不是本项目训练产物，用户没有理由知道它们该
    从哪个 GitHub Release 下载、放在哪个目录、文件名要不要改。

设计边界：
    - **只放清单与下载**，不做模型管理/版本切换。清单写死（能用的就这三种），
      因为"下载一个模型并保证它和推理代码契约一致"这件事本身就需要人工验证，
      自动发现任意 ONNX 只会把不匹配的错误留到推理时才炸。
    - 下载走**流式 + `.part` 续传**：GroundingDINO 量化版接近 1 GB，一次下载
      断掉就从零开始是不可接受的。
    - 目录默认 `6-web/models/`，可用 `TRTFORGE_MODEL_DIR` 覆盖。**不进 git**：
      权重是二进制大文件，仓库只保存"从哪里下载"。
"""

from __future__ import annotations

import os
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------

#: 模型根目录。用一个环境变量兜住"模型放 D 盘、代码放 C 盘"这类部署方式。
_ENV_MODEL_DIR = "TRTFORGE_MODEL_DIR"
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models"


def model_dir() -> Path:
    """解析模型根目录（不自动创建，创建放到 :func:`ensure_dir`）。"""
    raw = os.environ.get(_ENV_MODEL_DIR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_DIR


def ensure_dir() -> Path:
    target = model_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------
# 每个条目是一个"模型"，可以包含多个文件（SAM 是 encoder + decoder 两个）。
#
# `urls` 是有序候选：同一个文件在不同镜像上放一份，先到先得。为什么要多份：
# 国内访问 GitHub Release 经常超时，而模型的 sha 是确定的，换镜像不影响正确性。
#
# 2026-11-03 实测校准（v1.24.1）：
#   - 体积按服务端 Content-Length 逐项核对过。原来写的 40.1 MB / 700 MB 是拍脑袋估的，
#     GroundingDINO 实际只有 171 MB，按 700 MB 算会让进度条永远走不到 100%。
#   - 原先挂的 `hf-mirror.com/CVHub520/...` 三条回退**全部 404**（那些 repo 不存在），
#     已换成实测返回 200 的 GitHub 代理前缀（ghfast.top / gh-proxy.com）。
#     代理只是把 github.com 的地址转发一遍，文件本身仍是同一个 Release 产物。
GITHUB_RELEASE = "https://github.com/CVHub520/X-AnyLabeling/releases/download"
#: 实测可用的 GitHub 加速前缀（顺序即尝试顺序，直连在 CATALOG 里排第一）。
GH_PROXIES = ("https://ghfast.top/", "https://gh-proxy.com/")


def _mirror(url: str) -> list[str]:
    """把一个 GitHub 直连地址展开成「直连 + 若干代理」的候选列表。

    代理不可用时会抛 URLError / timeout，`_download_file` 本来就会顺着候选往下试，
    所以这里不需要额外判活——多试一次的成本远低于"启动时先探测所有镜像"。
    """
    return [url] + [proxy + url for proxy in GH_PROXIES]


CATALOG: list[dict] = [
    {
        "id": "mobile_sam",
        "name": "MobileSAM (ViT-T)",
        "use": "点 / 框提示生成多边形掩码",
        "license": "Apache-2.0",
        "channel": "sam",
        "files": [
            {
                "name": "mobile_sam.encoder.onnx",
                "urls": _mirror(f"{GITHUB_RELEASE}/v0.2.0/mobile_sam.encoder.onnx"),
                "size": 28_157_093,
            },
            {
                "name": "sam_vit_h_4b8939.decoder.onnx",
                "urls": _mirror(f"{GITHUB_RELEASE}/v0.2.0/sam_vit_h_4b8939.decoder.onnx"),
                "size": 16_500_272,
            },
        ],
    },
    {
        "id": "grounding_dino_swint",
        "name": "GroundingDINO (SwinT, int8)",
        "use": "文本提示检出目标（不训练，开放词表）",
        "license": "Apache-2.0",
        "channel": "text",
        "files": [
            {
                "name": "groundingdino_swint_ogc_quant.onnx",
                "urls": _mirror(f"{GITHUB_RELEASE}/v1.0.0/groundingdino_swint_ogc_quant.onnx"),
                "size": 179_597_886,
            }
        ],
    },
    {
        "id": "bert_tokenizer",
        "name": "BERT 分词表 (bert-base-uncased)",
        "use": "文本通道的 tokenizer（GroundingDINO 必需）",
        "license": "Apache-2.0",
        "channel": "text",
        "files": [
            {
                "name": "bert_base_uncased_tokenizer.json",
                "urls": [
                    # HuggingFace 的两个域名实测都返回 200，与上面三个"模型权重"不同源，
                    # 所以不走 GitHub 代理前缀。
                    "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
                    "https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer.json",
                    # 兜底：项目内自带的 X-AnyLabeling 子工程里有一份（存在才用，且不入库）
                    "@local:8-X-AnyLabeling-main/anylabeling/services/auto_labeling/configs/bert",
                ],
                "size": 466_062,
            }
        ],
    },
]

_BY_ID = {item["id"]: item for item in CATALOG}

#: 单块大小。64 KB 足够让"取消"在 100 ms 级别生效，又不至于把 CPU 打满。
CHUNK = 64 * 1024
#: 连接/读取超时。模型服务器偶尔很慢，20 秒会让正常下载被误判失败。
TIMEOUT = 30

# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class ModelError(Exception):
    """模型相关的可预期错误（接口层转 400）。"""


class ModelMissing(ModelError):
    """模型文件不存在。"""

    def __init__(self, model_id: str, names: list[str]):
        self.model_id = model_id
        self.names = names
        super().__init__(
            f"模型「{model_id}」缺少文件：{', '.join(names)}。"
            "请在标注页的模型仓库里下载，或设置 TRTFORGE_MODEL_DIR 指向已下载的目录。"
        )


class DownloadBusy(ModelError):
    """同一模型已有下载任务在跑。"""


# ---------------------------------------------------------------------------
# 清单查询
# ---------------------------------------------------------------------------


def _file_path(name: str) -> Path:
    return model_dir() / name


def _local_source(url: str) -> Path | None:
    """把 `@local:` 解析成仓库内路径（用于"已随仓库存在"的兜底文件）。"""
    if not url.startswith("@local:"):
        return None
    rel = url[len("@local:") :].strip()
    root = Path(__file__).resolve().parents[2]
    return root / rel


def _present_files(entry: dict) -> tuple[list[dict], list[str]]:
    """返回 (每个文件的状态, 缺失文件名列表)。"""
    status = []
    missing = []
    for spec in entry["files"]:
        path = _file_path(spec["name"])
        ok = path.is_file() and path.stat().st_size > 0
        status.append(
            {
                "name": spec["name"],
                "path": str(path),
                "size": spec.get("size") or 0,
                "present": ok,
                "actual_size": path.stat().st_size if path.is_file() else 0,
            }
        )
        if not ok:
            missing.append(spec["name"])
    return status, missing


def list_models() -> dict:
    """列出全部内置模型与本地状态。

    `present` 是**整组**文件的判断：SAM 少一个 decoder 就不能用，
    按"有文件就算有"会让界面显示"已就绪"然后推理时才炸。
    """
    items = []
    total_missing = 0
    for entry in CATALOG:
        files, missing = _present_files(entry)
        if missing:
            total_missing += 1
        items.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "use": entry["use"],
                "license": entry["license"],
                "channel": entry["channel"],
                "present": not missing,
                "missing": missing,
                "files": files,
                "total_bytes": sum(item["size"] for item in files),
            }
        )
    return {
        "dir": str(model_dir()),
        "items": items,
        "total": len(items),
        "missing": total_missing,
    }


def require(model_id: str) -> dict[str, str]:
    """取某个模型的全部文件路径，缺任何一个就抛 :class:`ModelMissing`。"""
    entry = _BY_ID.get(str(model_id or "").strip())
    if entry is None:
        raise ModelError(f"未知模型：{model_id}（可选：{', '.join(_BY_ID)}）")
    paths: dict[str, str] = {}
    missing: list[str] = []
    for spec in entry["files"]:
        path = _file_path(spec["name"])
        if path.is_file() and path.stat().st_size > 0:
            paths[spec["name"]] = str(path)
        else:
            missing.append(spec["name"])
    if missing:
        raise ModelMissing(entry["id"], missing)
    return paths


def resolve(model_id: str, filename: str) -> str | None:
    """按文件名取单个已就绪文件（找不到返回 ``None``）。"""
    entry = _BY_ID.get(str(model_id or "").strip())
    if entry is None:
        return None
    for spec in entry["files"]:
        if spec["name"] == filename:
            path = _file_path(filename)
            return str(path) if path.is_file() and path.stat().st_size > 0 else None
    return None


# ---------------------------------------------------------------------------
# 下载任务
# ---------------------------------------------------------------------------
# 与 `autolabeljobs` 同样的取舍：**线程 + 内存任务表**。下载是长 IO，不需要 GPU，
# 起线程比起进程轻；任务表放内存意味着服务重启后进度清零，但 `.part` 文件还在，
# 重新点下载会从断点继续——这是用户真正在意的东西。

MAX_JOBS = 8

_JOBS: dict[str, "DownloadJob"] = {}
_JOBS_LOCK = threading.Lock()
#: model_id -> 运行中的 job id。同一模型并发下两份会把 `.part` 写坏。
_ACTIVE: dict[str, str] = {}


class DownloadJob:
    """一次下载的状态。

    `lock` 保护所有可变字段：工作线程在写，HTTP 轮询在读，
    不加锁时 `snapshot()` 可能读到"done=100 但 total=0"这种中间态。
    """

    def __init__(self, model_id: str):
        self.id = uuid.uuid4().hex[:12]
        self.model_id = model_id
        self.status = "running"  # running / done / failed / cancelled
        self.error = ""
        self.downloaded = 0
        self.total = 0
        self.current = ""
        self.started_ns = time.time_ns()
        self.ended_ns = 0
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            if self.ended_ns:
                elapsed_ms = int((self.ended_ns - self.started_ns) / 1_000_000)
            else:
                elapsed_ms = int((time.time_ns() - self.started_ns) / 1_000_000)
            total = self.total or 0
            return {
                "id": self.id,
                "model": self.model_id,
                "status": self.status,
                "error": self.error,
                "downloaded": self.downloaded,
                "total": total,
                "percent": int(self.downloaded * 100 / total) if total > 0 else 0,
                "current": self.current,
                "elapsed_ms": elapsed_ms,
            }

    def add_bytes(self, count: int) -> None:
        with self.lock:
            self.downloaded += count

    def set_total(self, total: int) -> None:
        with self.lock:
            self.total = max(0, int(total))

    def set_current(self, name: str) -> None:
        with self.lock:
            self.current = name


def _open(url: str, offset: int = 0):
    """打开一个下载连接。返回 (response, 是否需要以追加方式写)。"""
    request = urllib.request.Request(url)
    if offset > 0:
        request.add_header("Range", f"bytes={offset}-")
    # 企业网里常见自签证书；模型下载没有机密性要求，校验失败就放弃等于不可用。
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(request, timeout=TIMEOUT, context=context)
    if offset > 0 and getattr(response, "status", 200) == 200:
        # 服务端不认 Range：只能从头下，调用方要把已有字节清零。
        return response, False
    return response, True


def _download_file(job: DownloadJob, spec: dict, dest: Path) -> None:
    """下载单个文件到 `dest`，支持 `.part` 续传。"""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    offset = part.stat().st_size if part.is_file() else 0
    if offset:
        job.add_bytes(offset)

    last_error: Exception | None = None
    for url in spec["urls"]:
        local = _local_source(url)
        if local is not None:
            # 兜底：仓库内已有一份，直接拷过去，不联网
            if local.is_file():
                dest.write_bytes(local.read_bytes())
                job.add_bytes(dest.stat().st_size - offset)
                return
            continue

        attempt_offset = offset
        try:
            response, append = _open(url, attempt_offset)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            continue

        if not append:
            attempt_offset = 0
            with job.lock:
                job.downloaded = max(0, job.downloaded - offset)

        mode = "ab" if append else "wb"
        try:
            with response, open(part, mode) as handle:
                while True:
                    if job.cancel_event.is_set():
                        raise _Cancelled()
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    job.add_bytes(len(chunk))
        except _Cancelled:
            raise
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            # 已写入的部分留在 `.part` 里，下次点下载会从这里继续
            offset = part.stat().st_size if part.is_file() else 0
            continue

        if part.is_file() and part.stat().st_size > 0:
            os.replace(part, dest)
            return
        last_error = last_error or ModelError(f"{spec['name']} 下载为空文件")

    raise ModelError(f"{spec['name']} 下载失败：{last_error or '所有地址都不可用'}")


class _Cancelled(Exception):
    """内部信号：用户取消。"""


def _run(job: DownloadJob, entry: dict) -> None:
    try:
        planned = sum(spec.get("size") or 0 for spec in entry["files"])
        job.set_total(planned)
        for spec in entry["files"]:
            if job.cancel_event.is_set():
                raise _Cancelled()
            dest = _file_path(spec["name"])
            if dest.is_file() and dest.stat().st_size > 0:
                continue  # 已就绪的文件不重下
            job.set_current(spec["name"])
            _download_file(job, spec, dest)
        with job.lock:
            job.status = "done"
    except _Cancelled:
        with job.lock:
            job.status = "cancelled"
        _drop_parts(entry)
    except Exception as exc:  # noqa: BLE001 - 线程里的意外必须反映到状态
        with job.lock:
            job.status = "failed"
            job.error = str(exc).strip() or exc.__class__.__name__
    finally:
        with job.lock:
            job.ended_ns = time.time_ns()
        with _JOBS_LOCK:
            if _ACTIVE.get(entry["id"]) == job.id:
                _ACTIVE.pop(entry["id"], None)


def _drop_parts(entry: dict) -> None:
    """取消后清掉 `.part`：留着半个文件会让下次"续传"接在错误的位置上。"""
    for spec in entry["files"]:
        part = _file_path(spec["name"]).with_name(spec["name"] + ".part")
        try:
            part.unlink()
        except (FileNotFoundError, OSError):
            pass


def _prune_locked() -> None:
    if len(_JOBS) <= MAX_JOBS:
        return
    finished = [job for job in _JOBS.values() if job.status != "running"]
    finished.sort(key=lambda job: job.started_ns)
    for job in finished[: max(0, len(_JOBS) - MAX_JOBS)]:
        _JOBS.pop(job.id, None)


def start_download(model_id: str) -> dict:
    entry = _BY_ID.get(str(model_id or "").strip())
    if entry is None:
        raise ModelError(f"未知模型：{model_id}（可选：{', '.join(_BY_ID)}）")
    _files, missing = _present_files(entry)
    if not missing:
        return {"id": "", "model": entry["id"], "status": "done", "already": True, "error": "",
                "downloaded": 0, "total": 0, "percent": 100, "current": "", "elapsed_ms": 0}

    with _JOBS_LOCK:
        running_id = _ACTIVE.get(entry["id"])
        if running_id:
            running = _JOBS.get(running_id)
            if running and running.status == "running":
                raise DownloadBusy(f"模型「{entry['id']}」正在下载中")
        _prune_locked()
        job = DownloadJob(entry["id"])
        _JOBS[job.id] = job
        _ACTIVE[entry["id"]] = job.id

    thread = threading.Thread(target=_run, args=(job, entry), name=f"modelzoo-{job.id}", daemon=True)
    job.thread = thread
    thread.start()
    return job.snapshot()


def get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
    return job.snapshot() if job else None


def cancel(job_id: str) -> dict | None:
    """请求取消。**不 kill 线程**：当前这块写完就退出，`.part` 由线程自己清。"""
    with _JOBS_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
        if job is None:
            return None
        job.cancel_event.set()
    return job.snapshot()


def recent(limit: int = MAX_JOBS) -> list[dict]:
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda job: job.started_ns, reverse=True)
    return [job.snapshot() for job in jobs[: max(1, int(limit))]]


def wait(job_id: str, timeout: float = 30.0) -> dict | None:
    """等任务结束（测试用）。"""
    with _JOBS_LOCK:
        job = _JOBS.get(str(job_id or "").strip())
    if job is None:
        return None
    if job.thread is not None:
        job.thread.join(timeout)
    return job.snapshot()


def reset_for_tests() -> None:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            job.cancel_event.set()
        _JOBS.clear()
        _ACTIVE.clear()
