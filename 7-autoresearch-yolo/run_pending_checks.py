# -*- coding: utf-8 -*-
"""把还没跑完的补测按顺序排队。

1152 短测和 exp_0006 的 holdout 已在 2026-11-11 跑完，这里只接后面三项：
warmup=1、cos_lr，然后最多 30 轮自动实验。显卡一次只能跑一个。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = Path(r"D:\Anaconda3\envs\dpenv\python.exe")
LOG = ROOT / "experiments" / "_pending_checks.log"
LLM_STORE = Path(r"d:\test\deeplearning\dp-tool\6-web\runtime\autoexp_llm.json")
#: Web 端把 Key 用 Windows DPAPI 加密后存在上面那个文件里。
#: 这里直接复用它的解密实现（按文件路径加载），而不是再抄一份 DPAPI 调用——
#: 两份加密代码早晚会走偏，而走偏的代价是「命令行读不出 Key」这种难查故障。
KEYVAULT_PY = ROOT.parent / "6-web" / "backend" / "keyvault.py"


def _load_keyvault():
    import importlib.util

    if not KEYVAULT_PY.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_trtforge_keyvault", KEYVAULT_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - 只是辅助脚本，解不开就让调用方降级
        return None


def _read_saved_key() -> str:
    """从 Web 的配置文件里取 Key：兼容「DPAPI 密文」与「旧格式明文」。"""
    try:
        saved = json.loads(LLM_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(saved, dict):
        return ""

    legacy = str(saved.get("api_key") or "").strip()
    if legacy:
        return legacy

    cipher = str(saved.get("api_key_dpapi") or "").strip()
    if not cipher:
        return ""
    vault = _load_keyvault()
    if vault is None:
        print("[warn] 找不到 6-web/backend/keyvault.py，无法解开已加密的 Key")
        return ""
    try:
        return str(vault.unprotect(cipher) or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 已保存的 Key 在这台机器上解不开：{exc}")
        return ""


def _child_env() -> dict[str, str]:
    """让命令行实验用网页里已经保存的接口配置。Key 只放进子进程环境，不写日志。"""
    env = os.environ.copy()
    env["AUTOEXP_WEB_URL"] = "http://127.0.0.1:8060"
    env["PYTHONIOENCODING"] = "utf-8"
    if not LLM_STORE.is_file():
        return env
    env["AUTOEXP_LLM_CONFIG"] = str(LLM_STORE)
    key = _read_saved_key()
    if key:
        env["AUTOEXP_API_KEY"] = key
    return env


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n重启电脑后续跑 {time.strftime('%Y-%m-%d %H:%M:%S')}："
            "只补 warmup=1、cos_lr，两者至少有一个跑完才进入 30 轮。\n"
        )
    env = _child_env()
    jobs = [
        ("warmup=1 @960 重试", ["run", "--config", "experiments/_retry_warmup960.json"], 1500),
        ("cos_lr @960 重试", ["run", "--config", "experiments/_retry_coslr960.json"], 1500),
    ]
    codes: list[tuple[str, int]] = []
    for title, args, timeout in jobs:
        code = _run_logged(title, args, timeout, env)
        codes.append((title, code))
    if all(code != 0 for _title, code in codes):
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write("两次 960 训练都没跑完，不再启动 30 轮自动实验，避免空烧接口调用。\n")
    else:
        code = _run_logged("自动实验 30 轮", ["agent", "--rounds", "30", "--hours", "8"], None, env)
        codes.append(("自动实验 30 轮", code))
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n===== 队列结束 =====\n")
        for title, code in codes:
            handle.write(f"{title}: {code}\n")
    print("队列结束", flush=True)
    return 0


def _run_logged(title: str, args: list[str], timeout: float | None, env: dict[str, str]) -> int:
    stamp = time.strftime("%H:%M:%S")
    header = f"\n===== {stamp} {title} =====\n"
    print(header, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.flush()
        try:
            completed = subprocess.run(
                [str(PY), "-u", str(ROOT / "autoexp.py"), *args],
                cwd=str(ROOT),
                timeout=timeout,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"\n{title} 超过 {timeout} 秒，已终止。\n")
            if exc.stdout:
                text = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace")
                handle.write(text)
            return 124
        return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
