# -*- coding: utf-8 -*-
"""自动实验（autoexp agent）的 Web 接口。

## 为什么单独一个文件

这个功能是**外置模块**：它的存在不该让「训练导出」那条主流程多出任何风险。
所以整个实现收敛在这个 APIRouter 里，`app.py` 只做三件事：
注册路由、登记任务类型名、把并发闸门挂进来。训练接口一行不动。

## 它和训练页的关系

- 共用 ``runners.manager``：这样自动实验能得到「顶栏运行中条」「停止按钮」
  「日志文件」「WebSocket 事件流」这一整套现成能力，不必重造。
- 因此它必须在 ``EXCLUSIVE_RUN_KINDS`` 里登记：一块显卡同时跑两个训练必然 OOM。
- 任务类型 ``autoexp-agent`` 走的是同一套 GPU 锁，网页端与命令行都能看见彼此。

## API Key 的处理

存在 ``6-web/runtime/autoexp_llm.json``（该目录在 .gitignore 里）。
- GET 接口**永不返回原文**，只给 ``has_key`` 与 ``sk-****1a2b`` 形式的遮罩值；
- 启动子进程时只通过**环境变量**传递，绝不进命令行——argv 会出现在
  「运行中」条的 tooltip、``runtime/logs/*.log`` 和 WebSocket 事件里。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
import gpulock
import inferproxy
import runners
from runners import manager

#: 任务类型。必须与 ``runners.EXCLUSIVE_RUN_KINDS`` 里的登记保持一致。
RUN_KIND = "autoexp-agent"

#: autoexp 工程目录。所有路径都由它派生，不接受前端传入。
AUTOEXP_DIR = config.REPO_ROOT / "7-autoresearch-yolo"
AUTOEXP_PY = AUTOEXP_DIR / "autoexp.py"
RESULTS_TSV = AUTOEXP_DIR / "results.tsv"
BEST_JSON = AUTOEXP_DIR / "experiments" / "best.json"
NOISE_JSON = AUTOEXP_DIR / "experiments" / "_noise_summary.json"
AGENT_DIR = AUTOEXP_DIR / "experiments" / "agent"
SPLITS_DIR = AUTOEXP_DIR / "splits"
PROMPT_MD = AUTOEXP_DIR / "prompts" / "experimenter.md"

#: 模型接口配置（含 Key）的存放位置。runtime 目录已 gitignore。
LLM_STORE = config.RUNTIME_DIR / "autoexp_llm.json"

#: 「检查 + 创建」同锁：双击按钮的两个请求不能都通过检查。
_START_LOCK = threading.Lock()

#: 并发闸门由 app.py 注入（它还要连带检查全流程串联链）。
#: 用注入而不是 import app，是为了避免循环导入：app 要 include 本模块的路由。
_busy_check: Callable[[], None] | None = None

router = APIRouter(prefix="/api/autoexp", tags=["autoexp"])


def set_busy_check(func: Callable[[], None]) -> None:
    """由 ``app.py`` 在启动时注册「是否已有独占任务」的检查函数。"""
    global _busy_check
    _busy_check = func


# ---------------------------------------------------------------------------
# 惰性加载 autoexp 侧的两个纯模块
# ---------------------------------------------------------------------------
# 为什么按文件路径加载而不是 import：
#   1) 这两个文件不在 backend 的 sys.path 上；
#   2) `llm` 这种名字太通用，sys.path 上任何同名模块都会遮蔽它。
# 加载失败一律返回 None，由调用方降级成「功能不可用」，不要让 import 期异常
# 把整个后端拖垮——（那样连训练页都打不开，正是本功能最该避免的事）。

_llm_mod: Any = None


def _load_llm() -> Any:
    global _llm_mod
    if _llm_mod is not None:
        return _llm_mod
    path = AUTOEXP_DIR / "llm.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("autoexp_llm_web", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _llm_mod = module
        return module
    except Exception:  # noqa: BLE001 - 可选功能，失败即降级
        return None


def _mask(key: str) -> str:
    """遮罩 Key。优先用 llm.py 的实现，保证与命令行输出一致。"""
    module = _load_llm()
    if module is not None:
        try:
            return module.mask_key(key)
        except Exception:  # noqa: BLE001
            pass
    if not key:
        return ""
    text = str(key)
    return "*" * len(text) if len(text) <= 8 else f"{text[:3]}****{text[-4:]}"


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------

class LlmConfigRequest(BaseModel):
    """页面上「模型接口」卡片的提交体。

    ``api_key`` 的三态语义（这条容易写错，所以显式定义）：
    - 空字符串 ``""``：**保留原 Key**（用户只改了地址/模型，不该把 Key 弄丢）
    - ``"__clear__"``：清除（用户想彻底断开）
    - 其它：覆盖
    """

    base_url: str = ""
    model: str = ""
    temperature: float | None = None
    # 输出预算。推理型模型会把「思考」和「正文」算进同一个 max_tokens，
    # 给 900 会全被思考吃掉、正文剩 0（现象是「返回内容为空」或 JSON 半截）。
    # 所以这里给一个足够宽的上限，并允许用户在页面上调。
    max_tokens: int | None = Field(default=None, ge=256, le=32768)
    api_key: str = ""


class AgentStartRequest(BaseModel):
    rounds: int = Field(default=30, ge=1, le=200)
    hours: float = Field(default=8.0, ge=0.0, le=48.0)
    max_stale: int = Field(default=5, ge=1, le=50)
    max_errors: int = Field(default=3, ge=1, le=50)
    lock_wait: int = Field(default=3600, ge=0, le=86400)


def _read_store() -> dict[str, Any]:
    try:
        data = json.loads(LLM_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(payload: dict[str, Any]) -> None:
    """原子写入：先写临时文件再替换，避免并发读到半个 JSON。

    POSIX 上顺手把权限收到 0600。Windows 没有等价的简易语义，
    依赖用户目录本身的访问控制——这一点在文档里写清楚，不假装做到了。
    """
    LLM_STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LLM_STORE.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, LLM_STORE)
    if os.name != "nt":
        try:
            os.chmod(LLM_STORE, 0o600)
        except OSError:
            pass


def _effective_config() -> dict[str, Any]:
    """把「代码里的默认值」与「页面上填的」合并成最终生效值。"""
    module = _load_llm()
    base: dict[str, Any] = {}
    if module is not None:
        try:
            base = module.load_config()
        except Exception:  # noqa: BLE001
            base = {}
    store = _read_store()
    merged = dict(base)
    for key in ("base_url", "model", "temperature", "max_tokens"):
        if store.get(key) not in (None, ""):
            merged[key] = store[key]
    return merged


# ---------------------------------------------------------------------------
# 状态与历史
# ---------------------------------------------------------------------------

def _summarize_ledger() -> dict[str, Any]:
    """读台账，算出给页面用的摘要。不加载 torch、不跑训练。"""
    rows: list[dict[str, str]] = []
    if RESULTS_TSV.is_file():
        import csv

        try:
            with RESULTS_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        except OSError:
            rows = []

    def _num(value: Any) -> float | None:
        try:
            text = str(value).strip()
            return float(text) if text else None
        except (TypeError, ValueError):
            return None

    best = None
    for row in rows:
        if (row.get("status") or "").strip() != "done":
            continue
        value = _num(row.get("fitness_map50_95"))
        if value is None:
            continue
        if best is None or value > best["fitness_map50_95"]:
            best = {
                "exp_id": row.get("exp_id", ""),
                "fitness_map50_95": value,
                "changed": row.get("changed", ""),
            }
    return {"rows": rows, "count": len(rows), "best": best}


def _read_noise() -> dict[str, Any]:
    try:
        summary = json.loads(NOISE_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sigma": None, "keep_threshold": None, "source": "未测噪声底"}
    sigma = summary.get("sigma")
    try:
        sigma_value = float(sigma)
    except (TypeError, ValueError):
        return {"sigma": None, "keep_threshold": None, "source": "噪声摘要损坏"}
    return {
        "sigma": round(sigma_value, 6),
        "keep_threshold": round(2 * sigma_value, 6),
        "source": "experiments/_noise_summary.json",
        "mean": summary.get("mean"),
    }


def _active_run() -> dict[str, Any] | None:
    """当前是否已有 autoexp 任务在跑（用于页面刷新后接续）。"""
    for item in manager.active_exclusive():
        if item.get("kind") == RUN_KIND:
            return item
    return None


def _ensure_idle() -> None:
    """已有独占任务时拒绝启动，返回 409。

    优先用 app.py 注入的检查（它能连带看到全流程串联链）；
    单独 import 本模块做测试时退回到只看 ``manager``，行为一致且不依赖 app。
    """
    if _busy_check is not None:
        _busy_check()
        return
    busy = manager.active_exclusive()
    if busy:
        first = busy[0]
        raise HTTPException(
            status_code=409,
            detail=f"已有任务正在运行：{first.get('kind')}（run {first.get('run_id')}）。",
        )


@router.get("/status")
def get_status() -> dict[str, Any]:
    """页面顶部要的「这个功能现在能不能用」。"""
    ledger = _summarize_ledger()
    noise = _read_noise()
    best: dict[str, Any] = {}
    if BEST_JSON.is_file():
        try:
            raw = json.loads(BEST_JSON.read_text(encoding="utf-8"))
            best = {
                "exp_id": raw.get("exp_id"),
                "updated_at": raw.get("updated_at"),
                "train": raw.get("train") or {},
            }
        except (OSError, ValueError):
            best = {}

    return {
        "available": AUTOEXP_PY.is_file(),
        "autoexp_py": str(AUTOEXP_PY),
        "prompt_md": str(PROMPT_MD),
        "prompt_present": PROMPT_MD.is_file(),
        # 没切分就不能开跑：agent 第一轮就会因为找不到 splits/pcb_tune.yaml 失败，
        # 提前在界面上说清楚，比让它跑到一半报错好。
        "splits_ready": (SPLITS_DIR / "pcb_tune.yaml").is_file(),
        "ledger_rows": ledger["count"],
        "ledger_best": ledger["best"],
        "noise": noise,
        "best_json": best,
        "active_run": _active_run(),
        "llm": {
            "config": {k: v for k, v in _effective_config().items()
                       if k in ("base_url", "model", "temperature", "max_tokens")},
            "has_key": bool(_read_store().get("api_key")),
            "key_masked": _mask(str(_read_store().get("api_key") or "")),
        },
    }


@router.get("/history")
def get_history(limit: int = 60) -> dict[str, Any]:
    """台账 + 每轮模型决策。

    两者合并返回，是因为页面上就是一张表：左边是「模型想改什么」，
    右边是「改动怎么判」。分成两个接口会让前端还得自己对齐。
    """
    ledger = _summarize_ledger()
    audits: dict[str, dict[str, Any]] = {}
    if AGENT_DIR.is_dir():
        for path in sorted(AGENT_DIR.glob("round_*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("exp_id"):
                audits[str(record["exp_id"])] = record

    rows: list[dict[str, Any]] = []
    for row in ledger["rows"][-max(1, limit):]:
        exp_id = row.get("exp_id", "")
        audit = audits.get(exp_id) or {}
        rows.append({
            "exp_id": exp_id,
            "created_at": row.get("created_at", ""),
            "fitness_map50_95": row.get("fitness_map50_95", ""),
            "status": row.get("status", ""),
            "changed": row.get("changed", ""),
            "hypothesis": row.get("hypothesis", ""),
            "verdict": audit.get("verdict", ""),
            "reason": audit.get("reason", ""),
            "delta": audit.get("delta"),
            # 升级前的台账没有这一列，读出来是空串：前端据此显示「未记录」，
            # 而不是把所有旧行都当成同一批数据。
            "data_fp": row.get("data_fp", ""),
        })
    return {"rows": rows, "total": ledger["count"]}


@router.get("/llm-config")
def get_llm_config() -> dict[str, Any]:
    """返回生效中的接口配置。**Key 只给遮罩**，永不返回原文。"""
    store = _read_store()
    effective = _effective_config()
    module = _load_llm()
    return {
        "base_url": effective.get("base_url", ""),
        "model": effective.get("model", ""),
        "temperature": effective.get("temperature", 0.7),
        "max_tokens": effective.get("max_tokens", 4096),
        "has_key": bool(store.get("api_key")),
        "key_masked": _mask(str(store.get("api_key") or "")),
        "store_path": str(LLM_STORE),
        # 让用户知道「不填也一样能跑」的兜底顺序
        "key_envs": list(getattr(module, "KEY_ENVS", ())) if module else [],
    }


@router.post("/llm-config")
def save_llm_config(request: LlmConfigRequest) -> dict[str, Any]:
    """保存接口配置。Key 的三态语义见 :class:`LlmConfigRequest`。"""
    base_url = (request.base_url or "").strip()
    model = (request.model or "").strip()

    if base_url and not base_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="接口地址必须以 http:// 或 https:// 开头，例如 https://api.deepseek.com",
        )
    if request.temperature is not None and not 0.0 <= request.temperature <= 2.0:
        raise HTTPException(status_code=400, detail="temperature 必须在 0~2 之间")

    store = _read_store()
    if base_url:
        store["base_url"] = base_url
    if model:
        store["model"] = model
    if request.temperature is not None:
        store["temperature"] = request.temperature
    if request.max_tokens is not None:
        store["max_tokens"] = int(request.max_tokens)

    incoming = request.api_key
    if incoming == "__clear__":
        store.pop("api_key", None)
    elif incoming.strip():
        store["api_key"] = incoming.strip()

    try:
        _write_store(store)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入配置失败：{exc}") from exc

    return {
        "ok": True,
        "has_key": bool(store.get("api_key")),
        "key_masked": _mask(str(store.get("api_key") or "")),
        "base_url": store.get("base_url", ""),
        "model": store.get("model", ""),
        "max_tokens": store.get("max_tokens", ""),
    }


@router.post("/llm-test")
def test_llm() -> dict[str, Any]:
    """用当前配置发一次极短请求，验证地址 + Key + 模型名。

    为什么值得单独一个接口：一次实验 10 分钟、一晚几十轮。
    如果 Key 在开跑时就写错，代价是整晚白跑，而这里只要 2 秒。
    """
    module = _load_llm()
    if module is None:
        raise HTTPException(status_code=503, detail="找不到 7-autoresearch-yolo/llm.py")

    store = _read_store()
    effective = _effective_config()
    key = str(store.get("api_key") or "").strip()
    if not key:
        # 也允许只配环境变量的用户（命令行习惯），这里如实告知而不是硬失败
        key = ""
    try:
        result = module.ping(
            {
                "base_url": effective.get("base_url"),
                "model": effective.get("model"),
                "temperature": effective.get("temperature"),
            },
            key or None,
        )
    except module.LLMError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 一律转成可显示文本
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"ok": True, **result}


def _resolve_python() -> str:
    """训练用解释器。与训练页共用同一套探测逻辑，避免两个页面跑在两个环境里。"""
    resolved = config.resolve_python()
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "未找到可用的训练解释器。请先在「训练导出」页填写 python.exe 路径，"
                "或设置环境变量 TRTFORGE_PYTHON。"
            ),
        )
    return resolved


@router.post("/start")
def start_agent(request: AgentStartRequest) -> dict[str, Any]:
    """启动自动实验循环。"""
    if not AUTOEXP_PY.is_file():
        raise HTTPException(status_code=503, detail=f"找不到 {AUTOEXP_PY}")

    store = _read_store()
    key = str(store.get("api_key") or "").strip()
    if not key:
        # 没有 Key 时不让它起：否则第一轮就在等一个必然失败的请求，
        # 用户看到的只是一行「模型调用失败」，很难联想到「你没填 Key」。
        raise HTTPException(
            status_code=400,
            detail="尚未配置 API Key。请先在「模型接口」卡片里填写并保存。",
        )

    python = _resolve_python()
    command = [
        python,
        str(AUTOEXP_PY),
        "agent",
        "--rounds", str(int(request.rounds)),
        "--hours", str(float(request.hours)),
        "--max-stale", str(int(request.max_stale)),
        "--max-errors", str(int(request.max_errors)),
        "--lock-wait", str(int(request.lock_wait)),
        "--events",
    ]

    # Key 走环境变量，不进 argv（argv 会出现在 tooltip / 日志 / 事件流里）。
    child_env = {
        "AUTOEXP_API_KEY": key,
        # 告诉 autoexp「显卡锁由哪个进程持有」。
        # 只传 pid 而不是一个布尔开关：布尔开关会让子进程连「命令行正另跑着
        # 一轮实验」都看不见，于是两个训练一起抢显存——正是这把锁要防的事。
        # autoexp 只在「锁持有者 pid == 这个值」时才跳过检查。
        "AUTOEXP_GPU_LOCK_PID": str(os.getpid()),
        # 页面填的 base_url / model / temperature 通过同一个文件生效
        "AUTOEXP_LLM_CONFIG": str(LLM_STORE),
        # 中文模型名/提示词的编码问题在 Windows 上很常见，显式钉死
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }

    with _START_LOCK:
        _ensure_idle()
        # 再查一次跨进程锁：命令行可能在跑 autoexp.py，那条路径不经过
        # manager，`_ensure_idle()` 看不见它。这里拦下来比让 agent 起跑后
        # 一直等锁更好——用户当场就知道是谁占着显卡。
        holder = gpulock.read()
        if holder is not None:
            holder_pid = int(holder.get("pid") or 0)
            if holder_pid != os.getpid() and gpulock.pid_alive(holder_pid):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"显卡正被占用：{gpulock.describe(holder)}。"
                        "关闭那个任务或终端后重试。"
                    ),
                )
        try:
            info = inferproxy.yield_gpu("自动实验")
        except Exception as exc:  # noqa: BLE001 - 让出失败不能把训练卡在半路
            raise HTTPException(status_code=500, detail=f"退出推理进程失败：{exc}") from exc

        def _release_infer() -> None:
            try:
                inferproxy.release_gpu_hold()
            except Exception as exc:  # noqa: BLE001
                print(f"[autoexp] 解除推理让出失败：{type(exc).__name__}: {exc}", flush=True)

        # 子进程看到这个标记就不再自己去通知网页、更不会在每一轮结束时把让出提前还回去。
        # 整场实验结束由上面的 _release_infer 解除，下次推理才重新加载。
        child_env["AUTOEXP_INFER_YIELDED"] = "1"
        try:
            run = manager.create(
                RUN_KIND,
                command,
                str(AUTOEXP_DIR),
                env=child_env,
                on_finished=_release_infer,
            )
        except OSError as exc:
            _release_infer()
            raise HTTPException(status_code=500, detail=f"启动自动实验进程失败：{exc}") from exc
    snapshot = run.snapshot()
    snapshot["infer_yielded"] = info
    return snapshot


@router.post("/stop")
def stop_agent(run_id: str = "") -> dict[str, Any]:
    """停止自动实验。

    必须**杀整棵进程树**。Windows 上终止父进程不会连带子进程，而 autoexp
    的链路是：``autoexp.py agent`` → ``autoexp.py _exec`` → ``launcher`` → PyTorch。
    只停父进程的结果是：界面显示「已停止」，显卡却还在被训练占着，
    下一个任务立刻 OOM——这是最难排查的一类问题。

    所以顺序是：先 ``taskkill /T /F`` 杀树，再让 manager 收尾（定状态、发事件）。
    反过来不行：父进程一死，``/T`` 就找不到子进程了。
    """
    target = run_id.strip()
    run = manager.get(target) if target else None
    if run is None:
        active = _active_run()
        if active is None:
            raise HTTPException(status_code=404, detail="没有正在运行的自动实验任务")
        target = str(active.get("run_id") or "")
        run = manager.get(target)
        if run is None:
            raise HTTPException(status_code=404, detail=f"未知任务：{target}")

    pid = run.process.pid if run.process is not None else None
    killed_tree = False
    if pid and os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            killed_tree = True
        except (OSError, subprocess.SubprocessError):
            killed_tree = False

    # 兜底：autoexp 会把当前训练子进程号写在这里（tee 模式）。
    # 万一 taskkill 漏了（比如进程刚换了一代），这一枪补上。
    child_pid_file = AGENT_DIR / "child.pid"
    if child_pid_file.is_file():
        try:
            child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(child_pid)],
                    capture_output=True,
                    timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            elif child_pid > 0:
                os.kill(child_pid, 9)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    stopped = manager.stop(target)
    return {"ok": stopped, "run_id": target, "killed_tree": killed_tree}


@router.get("/prompt-preview")
def prompt_preview() -> dict[str, Any]:
    """生成「第一轮要发给模型的完整提示词」，**不调 API、不跑训练**。

    这是花钱之前唯一能自查的机会：用户能确认机器事实、搜索空间、
    以及「模型会不会自作主张改不该改的东西」。
    """
    if not AUTOEXP_PY.is_file():
        raise HTTPException(status_code=503, detail=f"找不到 {AUTOEXP_PY}")
    python = _resolve_python()
    env = os.environ.copy()
    env["AUTOEXP_LLM_CONFIG"] = str(LLM_STORE)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [python, str(AUTOEXP_PY), "agent", "--dry-run"],
            cwd=str(AUTOEXP_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="生成提示词超时（>180 秒）") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"执行失败：{exc}") from exc

    text = (proc.stdout or "") + (proc.stderr or "")
    return {"ok": proc.returncode == 0, "text": text, "exit_code": proc.returncode}


@router.get("/ping")
def ping() -> dict[str, Any]:
    """轻量探活：页面初始化时判断模块是否可用，不触碰任何外部服务。"""
    return {
        "ok": True,
        "available": AUTOEXP_PY.is_file(),
        "kind": RUN_KIND,
        "registered_exclusive": RUN_KIND in runners.EXCLUSIVE_RUN_KINDS,
        "server_time": time.time(),
    }
