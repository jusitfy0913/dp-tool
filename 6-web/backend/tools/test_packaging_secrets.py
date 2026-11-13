# -*- coding: utf-8 -*-
"""打包密钥扫描的验收脚本：证明「Key 不出去」这件事有代码在拦，而不是靠运气。

背景
----
自动实验（LLM 决策层）的 API Key 在开发机上有两份来源：

1. 环境变量 ``AUTOEXP_API_KEY`` / ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``；
2. ``6-web/runtime/autoexp_llm.json`` —— 在网页上填过一次就落盘。
    v1.35.3 起 Key 以 Windows DPAPI 密文存放（``api_key_dpapi``），
    但密文在原机器原账户上仍可解开，所以同样不许出现在安装包里。

第 2 份最危险：它在仓库目录里、名字像普通配置。
一旦随安装包发出去就收不回来（只能去厂商后台吊销）。

所以 ``packaging/secret_scan.ps1`` 是打包的最后一关，本脚本验收三件事：

| 验收点 | 为什么 |
| --- | --- |
| 两个打包脚本都必须调用这道关 | 只在一个包里加，另一个包迟早漏 |
| 干净的目录必须通过 | 关卡不能误伤，否则有人会加「跳过」开关 |
| 真实 Key 文件/Key 字符串必须失败 | 拦不住等于没有 |

用法：
    python 6-web/backend/tools/test_packaging_secrets.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
PACKAGING = REPO / "packaging"
SCANNER = PACKAGING / "secret_scan.ps1"
WEB_SCRIPT = PACKAGING / "package_web_user.ps1"
DEMO_SCRIPT = PACKAGING / "package_demo_models.ps1"
RUNTIME_KEY = REPO / "6-web" / "runtime" / "autoexp_llm.json"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def powershell() -> str:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        raise RuntimeError("找不到 PowerShell")
    return exe


def write_driver(path: Path) -> None:
    # 驱动脚本只含 ASCII，所以不需要 BOM；PowerShell 5.1 读无 BOM 文件按 ANSI，
    # 纯 ASCII 内容不受影响。
    scanner = str(SCANNER).replace("'", "''")
    path.write_text(
        "param([string]$Root)\n"
        "$ErrorActionPreference = 'Stop'\n"
        f". '{scanner}'\n"
        "$findings = @(Find-PackagedSecrets -Root $Root)\n"
        "Write-Output (\"FINDINGS=\" + $findings.Count)\n"
        "foreach ($f in $findings) { Write-Output (\"HIT=\" + $f.path) }\n",
        encoding="utf-8",
    )


def scan(root: Path, driver: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver), str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def findings_count(output: str) -> int:
    for line in output.splitlines():
        if line.startswith("FINDINGS="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return -1
    return -1


# ---------------------------------------------------------------------------
# 1. 静态接线：两个包都必须调用这道关
# ---------------------------------------------------------------------------
def test_wiring() -> None:
    check("扫描器存在", SCANNER.is_file(), str(SCANNER))
    for script, label in ((WEB_SCRIPT, "软件包"), (DEMO_SCRIPT, "示例模型包")):
        check(f"{label}打包脚本存在", script.is_file(), str(script))
        if not script.is_file():
            continue
        text = script.read_text(encoding="utf-8")
        check(f"{label}加载了 secret_scan.ps1", 'secret_scan.ps1' in text)
        check(f"{label}调用了 Assert-NoPackagedSecrets",
              "Assert-NoPackagedSecrets" in text)

    if WEB_SCRIPT.is_file():
        text = WEB_SCRIPT.read_text(encoding="utf-8")
        # 软件包绝不能把 runtime 目录拷进去：Key 就躺在 6-web\runtime\autoexp_llm.json。
        copies_runtime = any(
            token in text
            for token in ("robocopy $runtime", "6-web\\runtime\"", "6-web/runtime/", "$backend\\runtime")
        )
        check("软件包没有复制 runtime 目录", not copies_runtime)

    # 扫描器本身不能带「跳过」开关，否则第一道裂缝就是它
    if SCANNER.is_file():
        text = SCANNER.read_text(encoding="utf-8")
        check("扫描器没有提供跳过开关", "SkipSecret" not in text and "skip_secret" not in text)

    check("扫描器会读环境变量", "AUTOEXP_API_KEY" in SCANNER.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 2. 干净的包必须通过（不能误伤）
# ---------------------------------------------------------------------------
def test_clean_dir_passes(driver: Path, workdir: Path) -> None:
    root = workdir / "clean"
    (root / "frontend").mkdir(parents=True)
    (root / "frontend" / "index.html").write_text("<html>模型推理</html>", encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"port": 8060, "host": "127.0.0.1"}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"models": ["pcb"]}), encoding="utf-8")
    (root / "README.txt").write_text("把 base_url 换成自己的，Key 走环境变量。", encoding="utf-8")
    (root / "model.onnx").write_bytes(b"\x00\x01binary")

    code, output = scan(root, driver)
    check("干净目录扫描不报错", code == 0, output[-300:])
    check("干净目录发现 0 处", findings_count(output) == 0, output[-300:])


# ---------------------------------------------------------------------------
# 3. 每一种真危险都必须被拦下
# ---------------------------------------------------------------------------
def test_planted_secrets(driver: Path, workdir: Path) -> None:
    cases = [
        ("runtime 深处的 Key 文件", "6-web/runtime/autoexp_llm.json",
         '{"api_key": "sk-0123456789abcdef0123456789abcdef"}'),
        ("configs 下的私有配置", "7-autoresearch-yolo/configs/llm.local.json",
         '{"model": "deepseek-chat"}'),
        ("任意 *.local.json", "docs/llm.local.json", '{"model": "x"}'),
        ("普通 json 里的 Key", "frontend/settings.json",
         '{"base_url": "https://api.deepseek.com", "api_key": "sk-abcdefghijklmnopqrstuvwxyz01"}'),
        ("yaml 里的 Key", "launchers/conf.yaml",
         "api_key: sk-abcdefghijklmnopqrstuvwxyz012345"),
        ("markdown 里的 Key", "docs/说明.md",
         "用这个 Key：sk-abcdefghijklmnopqrstuvwxyz012345"),
        ("私钥文件", "bin/server.pem", "-----BEGIN PRIVATE KEY-----"),
        ("裸 .key 文件", "bin/plain.key", "abcdefghijklmnop"),
        # DPAPI 密文也是密钥材料：原机器原账户还能解开，
        # 而且它出现在包里就说明有人把 runtime 目录拷进去了。
        ("DPAPI 加密的 Key（改过名）", "docs/ordinary-settings.json",
         '{"key_enc": "dpapi", "api_key_dpapi": "AQAAANCMnd8BFdERjHoAwE/Cl+sBAAAA9Jx2hYq9R0q1V0kQq2Wb7AAAACAAAAAAAQAAA"}'),
    ]
    for index, (label, relative, content) in enumerate(cases):
        root = workdir / f"bad{index}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        code, output = scan(root, driver)
        check(f"拦下：{label}", findings_count(output) > 0, output[-300:])

    # 公开 CA 证书必须放行：软件包里 certifi\cacert.pem 一定会出现，
    # 一律把 *.pem 当私钥会让软件包永远打不出来（这是实测撞到过的误报）。
    root = workdir / "public_cert"
    target = root / "app" / "certifi" / "cacert.pem"
    target.parent.mkdir(parents=True)
    target.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIUY2VydGlmaWNhdGU=\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    code, output = scan(root, driver)
    check("公开 CA 证书（cacert.pem）放行", findings_count(output) == 0, output[-300:])

    # 二进制不该被读成文本（会误伤，也慢）
    root = workdir / "binary"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "yolo_infer.dll").write_bytes(b"sk-" + b"A" * 40)
    code, output = scan(root, driver)
    check("二进制文件不被误判", findings_count(output) == 0, output[-300:])


# ---------------------------------------------------------------------------
# 4. 本机真实的 Key 文件必须被拦（最有说服力的一条）
# ---------------------------------------------------------------------------
def test_real_runtime_key_caught(driver: Path, workdir: Path) -> None:
    if not RUNTIME_KEY.is_file():
        print("[SKIP] 本机没有 6-web/runtime/autoexp_llm.json（没在网页上填过 Key）")
        return
    root = workdir / "real"
    target = root / "6-web" / "runtime" / "autoexp_llm.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(RUNTIME_KEY, target)
    code, output = scan(root, driver)
    check("拦下本机真实的 autoexp_llm.json", findings_count(output) > 0, output[-300:])

    # 去掉文件名那一道关，内容里也必须能扫出来（防止有人改了文件名绕过）
    renamed = root / "6-web" / "runtime" / "ordinary-settings.json"
    shutil.move(str(target), str(renamed))
    code, output = scan(root, driver)
    check("改名后仍被内容扫描拦下", findings_count(output) > 0, output[-300:])


def main() -> int:
    print(f"仓库根：{REPO}")
    test_wiring()

    if not SCANNER.is_file():
        print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项。")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="trtforge_secret_"))
    try:
        driver = workdir / "driver.ps1"
        write_driver(driver)
        test_clean_dir_passes(driver, workdir)
        test_planted_secrets(driver, workdir)
        test_real_runtime_key_caught(driver, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项。")
    if FAILED:
        for name in FAILED:
            print(f"  未通过：{name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
