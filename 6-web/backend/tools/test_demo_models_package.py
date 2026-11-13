# -*- coding: utf-8 -*-
"""示例模型包验收脚本：证明「打出来的模型包能装进软件、装完后端能认出来」。

为什么这个测试必须存在
----------------------
模型包的布局是**隐式契约**：`models.py` 只认两个根
（`launchers/runs` 与 `launchers/results`），标签文件必须优先按 run 目录名命中，
anomalib 的 `metadata.json` 必须与引擎同目录。任何一条对不上，
结果是「推理页下拉列表是空的」——用户看到的是空白，我们自己却无从下手。

所以这里不只检查文件存在，还**真的把包装进去**，再调一次
`models.scan_models()`，用后端自己的扫描器回答「装对了没有」。
测试自己造一个假的 `.engine` 占位：引擎要在目标显卡上编，测试环境没有，
但扫描器只按后缀认文件，假的足以验证「装好后会自动出现」这条链路。

跳过策略
--------
示例模型来自训练产物（不在 git 里）。CI 或别人的机器上没有这些文件时，
本脚本打印 [SKIP] 并返回 0 —— 断言的是「打包逻辑对不对」，
而不是「这台机器训过模型没有」。

用法：
    python 6-web/backend/tools/test_demo_models_package.py
"""

from __future__ import annotations

import hashlib
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
MANIFEST = PACKAGING / "models_manifest.json"
PACK_SCRIPT = PACKAGING / "package_demo_models.ps1"
INSTALL_TEMPLATE = PACKAGING / "templates" / "install_demo_models.ps1"
BAT_TEMPLATE = PACKAGING / "templates" / "install_demo_models.bat"

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name}：{reason}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_powershell(script: Path, *args: str) -> subprocess.CompletedProcess:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        raise RuntimeError("找不到 PowerShell")
    command = [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. 清单本身
# ---------------------------------------------------------------------------
def test_manifest_shape() -> dict | None:
    check("清单文件存在", MANIFEST.is_file(), str(MANIFEST))
    if not MANIFEST.is_file():
        return None
    try:
        spec = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        check("清单是合法 JSON", False, str(exc))
        return None
    check("清单是合法 JSON", True)

    models = spec.get("models") or []
    check("清单至少有一个模型", isinstance(models, list) and len(models) > 0, repr(models))
    ok_kind = all(str(m.get("kind")) in ("yolo", "anomalib") for m in models)
    check("kind 只有 yolo / anomalib", ok_kind)
    ok_run = all(str(m.get("target_run") or "") for m in models)
    check("每个模型都指定了 target_run", ok_run)
    ok_source = all(str(m.get("source_dir") or "") for m in models)
    check("每个模型都指定了 source_dir", ok_source)
    # target_run 决定 labels_<run>.json 的文件名；带空格或中文会让路径与匹配都变脆
    bad_run = [str(m.get("target_run")) for m in models
               if any(ch in str(m.get("target_run")) for ch in " \\/:*?\"<>|")]
    check("target_run 是干净的目录名", not bad_run, repr(bad_run))
    return spec


def test_templates() -> None:
    check("安装脚本模板存在", INSTALL_TEMPLATE.is_file(), str(INSTALL_TEMPLATE))
    check("安装 bat 模板存在", BAT_TEMPLATE.is_file(), str(BAT_TEMPLATE))
    if INSTALL_TEMPLATE.is_file():
        text = INSTALL_TEMPLATE.read_text(encoding="utf-8")
        check("安装脚本会校验 SHA256", "Get-FileHash" in text and "sha256" in text)
        check("安装脚本会找 app\\TRTForgeWeb.exe", "app\\TRTForgeWeb.exe" in text)
        check("安装脚本会改写 metadata 的 onnx_path", 'Add-Member -NotePropertyName "onnx_path"' in text)
    if BAT_TEMPLATE.is_file():
        raw = BAT_TEMPLATE.read_bytes()
        # cmd.exe 按系统 ANSI 读 .bat；一旦有非 ASCII 字节，中文机器以外的环境会乱码
        check("bat 模板是纯 ASCII", all(byte < 128 for byte in raw))


def sources_available(spec: dict) -> bool:
    for model in spec.get("models") or []:
        source = REPO / str(model.get("source_dir") or "")
        weight = source / str(model.get("source_weight") or "")
        if not weight.is_file():
            return False
        if model.get("kind") == "yolo":
            labels = source / str(model.get("source_labels") or "")
            if not labels.is_file():
                return False
        else:
            meta = source / str(model.get("source_metadata") or "")
            if not meta.is_file():
                return False
    return True


# ---------------------------------------------------------------------------
# 2. 真的打一次包，检查产物
# ---------------------------------------------------------------------------
def build_package(workdir: Path) -> Path | None:
    if not PACK_SCRIPT.is_file():
        check("打包脚本存在", False, str(PACK_SCRIPT))
        return None
    proc = run_powershell(PACK_SCRIPT, "-OutDir", str(workdir), "-SkipZip")
    ok = proc.returncode == 0
    check("打包脚本能跑通（只有 ONNX，不压缩）", ok, (proc.stdout + proc.stderr)[-600:])
    if not ok:
        return None
    return workdir / "TRTForgeWeb_models_demo"


def test_staged_package(staged: Path, spec: dict) -> None:
    payload = staged / "payload"
    check("产物有 payload 目录", payload.is_dir())
    check("产物有 manifest.json", (staged / "manifest.json").is_file())
    check("产物有安装脚本", (staged / "install_demo_models.ps1").is_file())
    check("产物有安装 bat", (staged / "install_demo_models.bat").is_file())
    check("产物有说明文档", (staged / "README_示例模型.md").is_file())

    engines = list(payload.rglob("*.engine")) + list(payload.rglob("*.trt"))
    check("包里不含 .engine（引擎绑显卡）", not engines, str(engines[:3]))
    programs = [p for p in payload.rglob("*") if p.suffix.lower() in (".py", ".pyc", ".pyd")]
    check("包里不含程序文件", not programs, str(programs[:3]))

    try:
        manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        check("产物的 manifest.json 可解析", False, str(exc))
        return
    check("产物的 manifest.json 可解析", True)
    check("manifest 记录 app_version", bool(manifest.get("app_version")))
    check("manifest 声明不含引擎", manifest.get("engine_included") is False)

    entries = manifest.get("models") or []
    check("manifest 的模型数与清单一致", len(entries) == len(spec.get("models") or []),
          f"{len(entries)} vs {len(spec.get('models') or [])}")

    for entry in entries:
        model_id = entry.get("id")
        files = entry.get("files") or []
        check(f"[{model_id}] 有文件条目", len(files) > 0)
        onnx_files = [f for f in files if str(f.get("path", "")).lower().endswith(".onnx")]
        check(f"[{model_id}] 带 ONNX", len(onnx_files) == 1, str(onnx_files))

        # 清单里的路径必须落在 payload 之内：越界就等于让安装脚本往软件目录外面写文件
        for item in files:
            relative = str(item.get("path") or "")
            target = (payload / relative).resolve()
            inside = str(target).startswith(str(payload.resolve()))
            check(f"[{model_id}] 目标路径没越界：{relative}", inside, str(target))
            if not target.is_file():
                check(f"[{model_id}] 文件已落位：{relative}", False, str(target))
                continue
            size_ok = target.stat().st_size == int(item.get("bytes", -1))
            check(f"[{model_id}] 大小一致：{Path(relative).name}", size_ok)
            hash_ok = sha256_of(target) == str(item.get("sha256", "")).upper()
            check(f"[{model_id}] SHA256 一致：{Path(relative).name}", hash_ok)

        if entry.get("kind") == "yolo":
            run_name = Path(str(entry.get("target", ""))).parent.parent.name
            labels = [f for f in files if Path(str(f.get("path"))).name == f"labels_{run_name}.json"]
            check(f"[{model_id}] 标签文件名与 run 目录同名（models.py 靠它精确匹配）",
                  len(labels) == 1, f"期望 labels_{run_name}.json")
            args = [f for f in files if Path(str(f.get("path"))).name == "args.yaml"]
            check(f"[{model_id}] 带 args.yaml（任务类型从它推断）", len(args) == 1)
            if args:
                text = (payload / str(args[0]["path"])).read_text(encoding="utf-8")
                check(f"[{model_id}] args.yaml 里没有开发机绝对路径",
                      "D:\\" not in text and "D:/" not in text, text[:120])
                check(f"[{model_id}] args.yaml 写了 task", "task:" in text)
        else:
            meta = [f for f in files if Path(str(f.get("path"))).name == "metadata.json"]
            check(f"[{model_id}] 带 metadata.json（DLL 靠它读阈值）", len(meta) == 1)
            if meta:
                data = json.loads((payload / str(meta[0]["path"])).read_text(encoding="utf-8-sig"))
                check(f"[{model_id}] metadata 有 input_size", bool(data.get("input_size")))
                check(f"[{model_id}] metadata 有 image/pixel 阈值",
                      data.get("image_threshold") is not None and data.get("pixel_threshold") is not None)
                check(f"[{model_id}] metadata 的 ckpt_path 已清空（不泄露开发机路径）",
                      not str(data.get("ckpt_path") or ""))
                recorded = str(data.get("onnx_path") or "")
                check(f"[{model_id}] metadata 的 onnx_path 是包内相对路径",
                      bool(recorded) and not Path(recorded).is_absolute(), recorded)


# ---------------------------------------------------------------------------
# 3. 装进一个假安装目录，再让后端扫描
# ---------------------------------------------------------------------------
def test_install_and_scan(staged: Path) -> None:
    install = staged.parent / "TRTForgeWeb"
    install.mkdir(parents=True, exist_ok=True)
    (install / "app").mkdir(exist_ok=True)
    (install / "app" / "TRTForgeWeb.exe").write_bytes(b"fake-exe")

    proc = run_powershell(staged / "install_demo_models.ps1", "-InstallDir", str(install))
    check("安装脚本能跑通", proc.returncode == 0, (proc.stdout + proc.stderr)[-800:])
    # 安装脚本任何 WARN 都意味着有一项"本该做成"的事没做成。
    # 实测踩过：判断 metadata.json 的写法有运算符优先级坑，导致 YOLO 的 args.yaml
    # 被当成 metadata 解析（报 Invalid JSON primitive: task），只留一行黄字就过去了。
    combined = (proc.stdout or "") + (proc.stderr or "")
    check("安装过程没有 WARN", "WARN" not in combined, combined[-600:])

    manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8-sig"))
    installed = 0
    for model in manifest.get("models") or []:
        for item in model.get("files") or []:
            if (install / str(item["path"])).is_file():
                installed += 1
    total = sum(len(m.get("files") or []) for m in manifest.get("models") or [])
    check("所有文件都装进了安装目录", installed == total, f"{installed}/{total}")
    check("安装脚本留下安装记录", (install / "launchers" / "demo_models_installed.json").is_file())

    # args.yaml 必须保持原样（它不是 JSON，一旦被当成 metadata 处理就会报错或被覆盖）
    for model in manifest.get("models") or []:
        for item in model.get("files") or []:
            if Path(str(item["path"])).name != "args.yaml":
                continue
            text = (install / str(item["path"])).read_text(encoding="utf-8")
            check("args.yaml 装完后内容没被改动", text.startswith("task:"), text[:80])

    # anomalib 的 metadata 在安装时必须换成这台机器上的绝对路径：
    # 降级推理（显卡不支持 / DLL 加载失败）要靠它找回同源 ONNX。
    for model in manifest.get("models") or []:
        if model.get("kind") != "anomalib":
            continue
        meta_item = next((f for f in model["files"] if Path(str(f["path"])).name == "metadata.json"), None)
        if not meta_item:
            continue
        data = json.loads((install / str(meta_item["path"])).read_text(encoding="utf-8-sig"))
        recorded = str(data.get("onnx_path") or "")
        check("安装后 metadata 的 onnx_path 变成绝对路径", Path(recorded).is_absolute(), recorded)
        check("安装后 metadata 指向的 ONNX 真实存在", Path(recorded).is_file(), recorded)

    # 造假的引擎：目标机器上要编的那一步，测试环境用占位文件代替。
    # 扫描器只按后缀认文件，这样就能验证「引擎一出现，模型自动进列表」。
    for model in manifest.get("models") or []:
        for item in model.get("files") or []:
            path = install / str(item["path"])
            if path.suffix.lower() != ".onnx":
                continue
            if model.get("kind") == "yolo":
                engine = path.with_name(path.stem.replace(".transd", "") + ".transd_fp16.engine")
            else:
                engine = path.with_name(path.stem + "_fp16.engine")
            engine.write_bytes(b"fake-engine")

    os.environ["TRTFORGE_REPO_ROOT"] = str(install)
    os.environ["TRTFORGE_YOLO_LAUNCHER_DIR"] = str(install / "launchers")
    os.environ["TRTFORGE_ANOMALIB_LAUNCHER_DIR"] = str(install / "launchers")
    sys.path.insert(0, str(BACKEND))
    import models as models_mod  # noqa: E402 - 必须在环境变量设好之后再导入

    report = models_mod.scan_models()
    entries = {Path(e["path"]).name: e for e in report.get("entries") or []}
    kinds = {e["kind"] for e in report.get("entries") or []}
    check("后端扫到了示例模型", len(entries) >= 2, f"扫到 {len(entries)} 个：{list(entries)}")
    check("检测与异常两类都被识别", {"yolo", "anomalib"} <= kinds, str(kinds))

    for entry in report.get("entries") or []:
        if entry["kind"] == "yolo":
            check("YOLO 条目的任务类型正确", entry["task"] == "detect", str(entry["task"]))
            check("YOLO 条目配上了标签文件", bool(entry.get("labels")), str(entry.get("labels")))
            if entry.get("labels"):
                labels = json.loads(Path(entry["labels"]).read_text(encoding="utf-8-sig"))
                check("标签文件里的 name 与 run 目录一致",
                      str(labels.get("name")) == Path(entry["run_dir"]).name,
                      f"{labels.get('name')} vs {Path(entry['run_dir']).name}")
                check("标签文件里有 6 个类别", len(labels.get("classes") or []) == 6,
                      str(labels.get("classes")))
            check("YOLO 条目带 ONNX（降级推理要用）", bool(entry.get("onnx")), str(entry.get("onnx")))
        else:
            check("anomalib 条目的模型类型读出来了", entry.get("model_type") == "EfficientAd",
                  str(entry.get("model_type")))
            check("anomalib 条目的输入尺寸读出来了", entry.get("input_size") == [256, 256],
                  str(entry.get("input_size")))
            check("anomalib 条目带 metadata", bool(entry.get("metadata")), str(entry.get("metadata")))


def main() -> int:
    print(f"仓库根：{REPO}")
    spec = test_manifest_shape()
    test_templates()
    if spec is None:
        return 1

    if not sources_available(spec):
        skip("打包/安装/扫描链路", "本机缺少训练产物（示例模型不在 git 里）")
    else:
        workdir = Path(tempfile.mkdtemp(prefix="trtforge_demo_"))
        try:
            staged = build_package(workdir)
            if staged is not None:
                test_staged_package(staged, spec)
                test_install_and_scan(staged)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项，跳过 {len(SKIPPED)} 项。")
    if FAILED:
        for name in FAILED:
            print(f"  未通过：{name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
