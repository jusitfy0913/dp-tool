# -*- coding: utf-8 -*-
"""OpenAI 兼容的极简对话客户端（零第三方依赖）。

## 为什么不用 openai / requests

- `autoexp.py` 从第一版起就是**零第三方依赖**：它要能在任何装了 torch 的环境里跑，
  不该因为「想接个大模型」就多装一个包。标准库 `urllib.request` 足够。
- `requests` 虽然当前环境有，但它一旦被本模块 import，
  这段代码就不能在没装 requests 的解释器里加载（例如 Web 打包后的精简环境）。
- 后端 `6-web/backend/autoexp_api.py` 会用 `importlib` 按文件路径加载本模块，
  所以这里只允许出现标准库 import。加一个第三方依赖就会让后端启动时炸掉。

## 协议

统一按 OpenAI 的 `/chat/completions` 说话。DeepSeek、OpenAI、vLLM、
Ollama（`http://localhost:11434/v1`）都是这套，改 `base_url` 就能换。

## API Key 的处理（安全边界）

- 只从**环境变量**或**调用方显式传入的字典**里取，绝不自己落盘。
- 本模块所有对外文本都过 :func:`mask_key`，保证 Key 不会出现在日志、
  异常信息、事件流或界面里。
- 绝不把 Key 拼进命令行——`argv` 会出现在「运行中」条的 tooltip、
  `runtime/logs/*.log` 和 WebSocket 事件里。

编码：UTF-8。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径与默认值
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
#: 入库的默认配置（不含任何密钥）
CONFIG_JSON = CONFIGS_DIR / "llm.json"
#: 本地覆盖（.gitignore 排除），用于临时换地址/换模型
LOCAL_JSON = CONFIGS_DIR / "llm.local.json"

#: Key 的环境变量名。Web 端启动子进程时用第一个传入。
KEY_ENVS = ("AUTOEXP_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")

#: 配置文件缺失时的兜底。刻意把 DeepSeek 放默认：中文任务、价格低。
DEFAULTS: dict[str, Any] = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.7,
    # 4096 不是随手写的：推理型模型（deepseek-flash / deepseek-reasoner / o 系列）
    # 会把「思考」和「正文」**算进同一个 max_tokens 预算**。实测 900 时，
    # 思考阶段独占 900 token，正文剩 0，表现为「返回内容为空」或 JSON 半截。
    # 一次实验 11 分钟，省这 3000 token（约 0.002 元）而废掉一整轮，是最差的交易。
    "max_tokens": 4096,
    "timeout_seconds": 120,
    "max_retries": 3,
}

#: 截断后自动加预算的上限。给到 8192 是因为正文（一个几行的 JSON）本身很小，
#: 大头永远是思考；再往上加只会掩盖「这个模型太啰嗦」的问题，不如让用户换模型。
MAX_TOKEN_CAP = 8192

#: 这些 HTTP 状态码值得重试：429 限流、5xx 服务端抽风。
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: 退避基数（秒）。2 → 6 → 18，总等待不超过 30 秒：
#: 一次实验要 10 分钟，等太久的重试不如直接让这一轮失败。
BACKOFF_BASE = 2.0


class LLMError(RuntimeError):
    """模型调用失败。

    单独一个异常类型是为了让上层区分两种失败：
    - `LLMError`：网络/鉴权/限流——应该重试或提示用户检查配置；
    - 返回内容不是合法 JSON——是模型不听话，应该带着错误说明再问一次。
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class LLMTruncated(LLMError):
    """输出被 ``max_tokens`` 截断，正文不完整。

    单独一个类型、而且**不**标成 retryable：这不是网络抖动，等几秒再发同样
    请求只会得到同样的半截 JSON。上层要据此做的是「加预算重来」或「提示用户
    换模型」，而不是退避重试。

    这个类的存在是为了解决一个具体的排查灾难：以前截断和「模型输出不合规」
    共用一句「返回内容不是合法 JSON」，用户看到的现象是「模型好像不会写 JSON」，
    而真正的原因只是配置里一个数字太小。
    """

    def __init__(
        self,
        message: str,
        *,
        max_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        finish_reason: str = "length",
    ) -> None:
        super().__init__(message, status=None, retryable=False)
        self.max_tokens = max_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason


# ---------------------------------------------------------------------------
# 配置与密钥
# ---------------------------------------------------------------------------

def mask_key(key: str | None) -> str:
    """把 Key 打成 ``sk-****1a2b`` 这种能核对又不泄露的形式。

    短于 8 位的直接全遮：那种长度的「Key」通常是用户误贴了别的东西，
    露出前 3 后 4 就等于全露了。
    """
    if not key:
        return ""
    text = str(key).strip()
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:3]}****{text[-4:]}"


def _read_json(path: Path) -> dict[str, Any]:
    """读 JSON，文件不存在或坏掉都返回空字典（配置不该成为启动的硬门槛）。"""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并顺序：内置默认 ← llm.json ← llm.local.json ← 环境变量 ← 本次显式指定。

    「本次显式指定」排在最后，因为它是用户刚刚亲手敲下/填下的东西：
    ``--model`` 参数、以及 Web 页面写进 runtime 的那个文件。
    如果环境变量压过它们，那么 shell 里遗留一个 ``AUTOEXP_MODEL``
    就会让页面上的设置静默失效——查起来非常费劲。
    """
    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update(_read_json(CONFIG_JSON))
    merged.update(_read_json(LOCAL_JSON))

    # 环境变量只覆盖标量，且要求非空——空字符串表示「没设」，不是「设成空」。
    # 它压在入库文件之上（CI 里换端点/换模型不用改仓库），
    # 但压在下面「本次显式指定」之下。
    for env_name in ("AUTOEXP_BASE_URL", "AUTOEXP_MODEL"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            merged["base_url" if env_name.endswith("BASE_URL") else "model"] = raw

    # Web 端把「页面上填的 base_url / model / temperature」写到 runtime 下的
    # 一个 JSON 里，再通过这个环境变量指过来。
    # 为什么不直接写 configs/llm.local.json：那样会改动仓库工作树
    # （虽然是 gitignore 的），而 Web 的运行产物应该全部待在 runtime/ 里。
    external = os.environ.get("AUTOEXP_LLM_CONFIG", "").strip()
    if external:
        try:
            merged.update(_read_json(Path(external).expanduser()))
        except (OSError, ValueError):
            pass

    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None and v != ""})

    # 数值字段容错：界面上可能传字符串过来
    for key, caster in (("temperature", float), ("max_tokens", int),
                        ("timeout_seconds", int), ("max_retries", int)):
        try:
            merged[key] = caster(merged.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            merged[key] = DEFAULTS[key]

    # 注释字段与密钥字段都不参与请求。
    # 特别是 api_key：Web 端的存储文件里带着它，若不在这里剔除，
    # 它会随配置字典流进日志、审计快照和 prompt 上下文——那等于把 Key 写进台账。
    for noisy in ("_comment", "_note", "api_key", "key", "apikey"):
        merged.pop(noisy, None)
    return merged


def resolve_key(explicit: str | None = None) -> str:
    """取 API Key：显式传入 > 环境变量。

    显式传入这条路径给 Web 端用（Key 是用户刚在页面上填的，还没落盘）。
    """
    if explicit:
        return str(explicit).strip()
    for name in KEY_ENVS:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return ""


def describe_config(config: dict[str, Any] | None = None) -> str:
    """一行人类可读的配置摘要（不含 Key）。"""
    cfg = config or load_config()
    return f"{cfg.get('model')} @ {cfg.get('base_url')}"


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

def _endpoint(base_url: str) -> str:
    """把 base_url 归一成完整的 completions 地址。

    三种用户写法都要能吃下：
    - ``https://api.deepseek.com``            → 补 ``/chat/completions``
    - ``https://api.openai.com/v1``           → 补 ``/chat/completions``
    - ``http://localhost:11434/v1/chat/completions`` → 原样用
    """
    text = str(base_url or "").strip().rstrip("/")
    if not text:
        raise LLMError("base_url 为空，请先在「模型接口」里填写接口地址")
    if text.endswith("/chat/completions"):
        return text
    return f"{text}/chat/completions"


def chat(
    messages: list[dict[str, str]],
    *,
    config: dict[str, Any] | None = None,
    key: str | None = None,
    json_mode: bool = True,
) -> dict[str, Any]:
    """发一次对话请求。

    返回 ``{"content": str, "usage": dict, "model": str, "latency_ms": float}``。

    ``json_mode=True`` 时带上 ``response_format={"type": "json_object"}``。
    注意：DeepSeek 与 OpenAI 都要求 messages 里出现 "json" 字样才肯配合，
    system prompt 里已保证这一点；万一上游不支持这个字段导致 400，
    下面会自动去掉它重试一次，避免「换了个兼容接口就整个功能不可用」。
    """
    cfg = load_config(config)
    api_key = resolve_key(key)
    if not api_key:
        raise LLMError(
            "未配置 API Key。请在「模型接口」里填写，或设置环境变量 "
            f"{KEY_ENVS[0]}。"
        )

    url = _endpoint(cfg.get("base_url", ""))
    timeout = float(cfg.get("timeout_seconds") or 120)
    attempts = max(1, int(cfg.get("max_retries") or 1))
    want_json = bool(json_mode)
    # 输出预算。推理型模型会把它同时用于「思考」和「正文」，所以这里会在
    # 发现截断时逐次加倍（上限 MAX_TOKEN_CAP），而不是让这一轮白跑。
    budget = max(64, int(cfg.get("max_tokens") or DEFAULTS["max_tokens"]))
    cap = max(budget, MAX_TOKEN_CAP)

    last_error: LLMError | None = None
    for attempt in range(attempts):
        body: dict[str, Any] = {
            "model": cfg.get("model"),
            "messages": messages,
            "temperature": cfg.get("temperature", 0.7),
            "max_tokens": budget,
            "stream": False,
        }
        if want_json:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            payload = _post_json(url, body, api_key, timeout)
        except LLMError as exc:
            last_error = exc
            # json_mode 被上游拒绝：去掉该字段再来一次。
            # 兼容接口（vLLM/Ollama 的某些版本、第三方中转）常见这个问题，
            # 而它们其实照样能输出 JSON，只是不认这个开关。
            if want_json and exc.status in (400, 422):
                want_json = False
                continue
            if exc.retryable and attempt + 1 < attempts:
                time.sleep(BACKOFF_BASE * (3 ** attempt))
                continue
            raise
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"接口返回结构不符合预期：{exc}；原文={str(payload)[:300]}") from exc

        stats = _completion_stats(payload, choice, budget)

        # 截断：正文必然不完整，重发同样的请求只会得到同样的半截 JSON。
        # 所以这里不加退避等待，直接把预算翻倍重来（钱花在正事上）。
        if stats["finish_reason"] == "length":
            grown = min(budget * 2, cap)
            message = _truncation_message(stats, budget)
            if grown > budget and attempt + 1 < attempts:
                last_error = LLMTruncated(message, **stats)
                budget = grown
                continue
            raise LLMTruncated(
                message + f"（已尝试把预算加到 {cap}，仍然不够）", **stats
            )

        # 没有截断但正文是空的：多见于「思考模式」把内容留在 reasoning_content 里。
        # 这种不该说成「模型不会写 JSON」，否则用户会去改提示词，方向完全错。
        if not content.strip():
            detail = _truncation_message(stats, budget)
            raise LLMError(
                f"模型返回了空正文（finish_reason={stats['finish_reason']}）。{detail}"
            )

        return {
            "content": content,
            "usage": payload.get("usage") or {},
            "model": payload.get("model") or cfg.get("model"),
            "latency_ms": latency_ms,
            "finish_reason": stats["finish_reason"],
            "reasoning_tokens": stats["reasoning_tokens"],
            "max_tokens": budget,
        }

    # 走到这里说明每次都被「可重试」错误耗完，抛出最后一次
    raise last_error or LLMError("请求失败，原因未知")


def _completion_stats(payload: dict[str, Any], choice: dict[str, Any], budget: int) -> dict[str, Any]:
    """从响应里提取「这次为什么短」。

    ``reasoning_tokens`` 是关键字段：推理型模型（deepseek-flash / reasoner / o 系列）
    会把它和正文一起计入 ``completion_tokens``，而它不出现在 ``content`` 里。
    不看这个数，就只能看到「正文是空的」这种没法解释的现象。
    """
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        # DeepSeek 早期版本把它放在 usage 顶层
        reasoning = usage.get("reasoning_tokens")
    try:
        reasoning_tokens = int(reasoning or 0)
    except (TypeError, ValueError):
        reasoning_tokens = 0
    try:
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        completion_tokens = 0
    return {
        "finish_reason": str(choice.get("finish_reason") or ""),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "max_tokens": int(budget),
    }


def _truncation_message(stats: dict[str, Any], budget: int) -> str:
    """把截断翻译成「该改哪个数字」。

    用户看到的原话必须是可执行的，不能是「内容不是合法 JSON」——那句话会让人
    去怀疑提示词或模型能力，而实际情况是配置里一个数字不够大。
    """
    completion = stats.get("completion_tokens", 0)
    reasoning = stats.get("reasoning_tokens", 0)
    body = max(0, int(completion) - int(reasoning))
    if reasoning > 0:
        share = f"其中推理（思考）占了 {reasoning} token，正文只有 {body} token"
    else:
        share = f"推理 token 未上报；completion 共 {completion} token"
    return (
        f"输出被 max_tokens={budget} 截断：{share}。"
        "推理型模型会把「思考」和「正文」算进同一个预算，"
        "把「模型接口 → max_tokens」调到 4096 以上，或改用非推理型模型。"
    )


def _post_json(url: str, body: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    """一次 HTTP POST。把各种失败翻译成带 ``retryable`` 的 :class:`LLMError`。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
            # 有些网关会拦没有 UA 的请求
            "User-Agent": "trtforge-autoexp/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001 - 读错误体失败不能盖掉真正的状态码
            detail = ""
        raise LLMError(
            _explain_status(exc.code, detail),
            status=exc.code,
            retryable=exc.code in RETRY_STATUS,
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"连不上接口：{exc.reason}（检查 base_url 与网络/代理）", retryable=True) from exc
    except TimeoutError as exc:
        raise LLMError(f"请求超时（{timeout:.0f} 秒）", retryable=True) from exc

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise LLMError(f"返回内容不是 JSON：{raw[:300]!r}") from exc


def _explain_status(status: int, detail: str) -> str:
    """把 HTTP 状态码翻译成「用户下一步该做什么」，而不是只丢一个 401。"""
    hint = {
        400: "请求参数被拒绝（模型名写错？或该接口不支持 response_format）",
        401: "API Key 无效或已过期，请重新填写",
        402: "账户余额不足",
        403: "没有权限访问该模型",
        404: "接口路径不存在，检查 base_url（通常形如 https://api.deepseek.com）",
        429: "请求太频繁被限流，稍等后重试",
    }.get(status, "接口返回错误")
    return f"{hint}（HTTP {status}）{('：' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# 连通性自检
# ---------------------------------------------------------------------------

def ping(config: dict[str, Any] | None = None, key: str | None = None) -> dict[str, Any]:
    """发一次极短请求，验证「地址 + Key + 模型名」三者都对。

    为什么值得单独做：一次真实实验 10 分钟、一整晚几十轮。
    如果 Key 在启动时就写错，代价是整晚白跑。花 2 秒先测一次非常划算。

    刻意固定一个极小的 max_tokens 且不要 json 模式：只要接口能正常返回
    ``choices`` 就算通，不引入任何格式约束，避免把「模型不听话」误判成「配置错」。

    推理型模型在这里**必然**被截断（16~64 token 还不够它想完），这是预期行为，
    不是故障：能拿到 200 + choices 就已证明「地址、Key、模型名」三者都对。
    所以捕获 :class:`LLMTruncated` 并如实报告，而不是让「测试连通性」显示失败。
    """
    started = time.perf_counter()
    cfg = load_config(config)
    truncated = False
    try:
        result = chat(
            [{"role": "user", "content": "只回复两个字：可用"}],
            config={**cfg, "max_tokens": 16, "max_retries": 1},
            key=key,
            json_mode=False,
        )
    except LLMTruncated as exc:
        # 连通性是好的，只是预算小到装不下正文。这是「成功」的另一种形态。
        truncated = True
        result = {
            "model": cfg.get("model"),
            "content": "",
            "usage": {},
            "reasoning_tokens": getattr(exc, "reasoning_tokens", 0),
            "max_tokens": getattr(exc, "max_tokens", 0),
        }
    return {
        "ok": True,
        "model": result.get("model"),
        "reply": (result.get("content") or "").strip()[:40] or (
            "（推理型模型：自检用的极小预算内只有思考、没有正文，连通性正常）"
            if truncated else ""
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "usage": result.get("usage") or {},
        "truncated": truncated,
        "reasoning_tokens": result.get("reasoning_tokens", 0),
    }
