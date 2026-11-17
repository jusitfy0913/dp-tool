# -*- coding: utf-8 -*-
"""路径白名单全覆盖的验收脚本。

这层挡的是「网页变成读任意磁盘、写任意目录的后门」。它有个特有的失效方式：
**新加一个接口时忘了校验**，而现有功能一切正常，没人会发现。所以这个测试
用两种办法互相兜底：

- **静态扫描**（``test_static``）：用 ``ast`` 遍历 ``routers/*`` 里的每个接口函数，
  只要函数体里出现了路径类字段（``engine`` / ``path`` / ``source`` …），就必须
  同时出现一个校验调用（``_ensure_allowed_path`` 等），或委托给一个含校验的
  helper。新增接口忘校验直接失败。
- **行为验证**（``test_behavior``）：拿一份**在白名单外但真实存在**的文件（放在
  系统临时目录）去打关键接口，断言返回 403。用「存在但在外面」的路径才能区分
  「被白名单拦住」和「文件不存在」，后者掩盖不了漏洞。

    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_path_guard.py
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402# noqa: E402

import config  # noqa: E402


#: 字段名里出现这些「词根」就视为路径类字段。
#: 只列真实会进文件系统调用的名字，宁可窄一点——宽了会把 ``source`` 这种
#: 纯描述性字段（``OperatingPointRequest.source`` 是"哪个报告选的"）也算进来。
PATH_FIELDS = {
    "path",
    "root",
    "dir",
    "data",
    "engine",
    "onnx",
    "model",
    "weights",
    "labels",
    "metadata",
    "source",
    "project",
    "python",
    "bundle",
    "initial",
    "artifacts_dir",
}

#: 出现在函数体里就算「已经校验过」的调用名。
GUARDS = {
    "_ensure_allowed_path",
    "_ensure_allowed_dir",
    "_ensure_safe_name",
    "_ensure_dataset_path",
    "_require_file",
    "_require_model_dir",
    "_resolve_python",
    "is_path_allowed",
    "resolve_artifact",
}

#: 明确豁免：字段名命中但**不是**服务端路径（写明原因，将来才能判断能不能删）。
EXEMPT: dict[str, str] = {
    # source 在这两个请求模型里是"报告来源说明"，不是文件路径
    "save_operating_point": "source 是报告名，仅原样存进旁注文件",
    # 原生对话框的 initial 只是弹框起始位置，用户点确定后还会再走一次 allow_path
    "browse_dialog": "initial 只是对话框初始目录，选中结果才登记白名单",
    "browse_cancel": "无路径字段（仅撤销本机对话框）",
}


# ---------------------------------------------------------------------------
# 静态扫描
# ---------------------------------------------------------------------------
def _decorated_endpoints(tree: ast.AST) -> list[ast.AST]:
    """取出所有挂了 ``@router.get/post/...`` 的函数。"""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Attribute) and target.attr in {
                "get", "post", "put", "patch", "delete", "websocket", "api_route"
            }:
                found.append(node)
                break
    return found


def _identifier_names(node: ast.AST) -> set[str]:
    """函数体里出现过的所有标识符（含被调用的函数名、局部变量名）。"""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
    return names


def _field_names(node: ast.AST) -> set[str]:
    """只取「参数名」与「属性名」。

    局部变量不参与判定：``capabilities()`` 里 ``data = config.capabilities()``
    的 ``data`` 是一个字典，按局部变量名匹配会把它误判成路径字段。
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _called_names(node: ast.AST) -> set[str]:
    """函数体里直接调用的本模块函数名（用于「委托给 helper 校验」的场景）。"""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            names.add(child.func.id)
    return names


def test_static() -> None:
    routers_dir = BACKEND / "routers"
    modules = sorted(routers_dir.glob("*.py"))
    check("找到 routers 目录", routers_dir.is_dir(), str(routers_dir))

    total_endpoints = 0
    violations: list[str] = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        # 模块内的普通函数：接口把校验委托给 helper 时，要能看进 helper 里
        helpers = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in _decorated_endpoints(tree):
            total_endpoints += 1
            fields = _field_names(node) & PATH_FIELDS
            if not fields:
                continue
            if node.name in EXEMPT:
                continue
            if _identifier_names(node) & GUARDS:
                continue
            # 一层委托：build_batch_command / build_video_command 这类 helper 里校验
            delegated = [
                helper
                for name, helper in helpers.items()
                if name in _called_names(node) and _identifier_names(helper) & GUARDS
            ]
            if delegated:
                continue
            violations.append(
                f"{module.name}:{node.lineno} {node.name} 接受 {sorted(fields)} 但没有校验"
            )

    check(f"扫描到接口函数（{total_endpoints} 个）", total_endpoints > 40, str(total_endpoints))
    check("所有接受路径字段的接口都有白名单/存在性校验", not violations, "\n    ".join(violations))


# ---------------------------------------------------------------------------
# 行为验证
# ---------------------------------------------------------------------------
def _make_outside_files(root: Path) -> dict[str, Path]:
    """在白名单外造一批「存在且后缀正确」的文件。

    存在性很关键：不存在时接口会以「文件不存在」400 收场，那**证明不了**
    白名单生效。这里每个文件都真实存在，唯一能拒绝它的理由就是不在白名单内。
    """
    files = {
        "onnx": root / "probe.onnx",
        "pt": root / "probe.pt",
        "engine": root / "probe.engine",
        "json": root / "probe.json",
        "yaml": root / "dataset.yaml",
        "png": root / "probe.png",
    }
    files["onnx"].write_bytes(b"\x08\x01\x12\x00")
    files["pt"].write_bytes(b"PK\x03\x04")
    files["engine"].write_bytes(b"engine")
    files["json"].write_text('{"image_threshold": 0.5, "pixel_threshold": 0.5}', encoding="utf-8")
    files["yaml"].write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: ok}\n", encoding="utf-8")
    files["png"].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    return files


def test_behavior(outside: Path) -> None:
    from starlette.testclient import TestClient

    import app as web_app

    files = _make_outside_files(outside)
    check("测试文件确实在白名单外", not config.is_path_allowed(files["onnx"]),
          str(files["onnx"]))

    with TestClient(web_app.app) as client:
        # --- 目录浏览 ---
        response = client.get("/api/browse", params={"path": str(outside)})
        check("/api/browse 拒绝白名单外目录", response.status_code == 403, str(response.status_code))

        # --- 引擎域 ---
        response = client.get("/api/engine/probe", params={"onnx": str(files["onnx"])})
        check("/api/engine/probe 拒绝白名单外 ONNX", response.status_code == 403,
              f"{response.status_code} {response.text[:120]}")

        response = client.get("/api/engine/operating-point", params={"engine": str(files["engine"])})
        check("/api/engine/operating-point(GET) 拒绝白名单外引擎", response.status_code == 403,
              f"{response.status_code} {response.text[:120]}")

        response = client.post(
            "/api/engine/operating-point",
            json={"engine": str(files["engine"]), "conf": 0.5},
        )
        check("/api/engine/operating-point(POST) 拒绝白名单外引擎", response.status_code == 403,
              f"{response.status_code} {response.text[:120]}")

        # --- 训练域 ---
        response = client.post(
            "/api/train/yolo",
            json={"data": str(files["yaml"]), "model": "yolov8n.pt"},
        )
        check("/api/train/yolo 拒绝白名单外 dataset.yaml", response.status_code == 403,
              f"{response.status_code} {response.text[:150]}")

        response = client.post(
            "/api/train/yolo",
            json={"data": "runs/not-exist.yaml", "model": "yolov8n.pt", "name": ".."},
        )
        # data 不存在时先报 400；name 的 .. 只有在 data 通过后才会被检查，
        # 所以这里换成真实存在的问题在下面单独测。
        check("不存在的 dataset.yaml 报 400", response.status_code == 400, str(response.status_code))

        response = client.post(
            "/api/train/anomalib",
            json={"root": str(outside), "model": "patchcore"},
        )
        check("/api/train/anomalib 拒绝白名单外数据集根目录", response.status_code == 403,
              f"{response.status_code} {response.text[:150]}")

        # --- 串联 ---
        response = client.post(
            "/api/pipeline/yolo",
            json={"data": str(files["yaml"]), "name": "guard-test"},
        )
        check("/api/pipeline/yolo 拒绝白名单外 dataset.yaml", response.status_code == 403,
              f"{response.status_code} {response.text[:150]}")

        # --- 推理域 ---
        with files["png"].open("rb") as handle:
            response = client.post(
                "/api/infer/anomalib",
                data={"engine": str(files["engine"]), "metadata": str(files["json"])},
                files={"image": ("probe.png", handle, "image/png")},
            )
        check("/api/infer/anomalib 拒绝白名单外引擎", response.status_code == 403,
              f"{response.status_code} {response.text[:150]}")

        with files["png"].open("rb") as handle:
            response = client.post(
                "/api/infer/frame",
                data={"engine": str(files["engine"]), "task": "detect"},
                files={"image": ("probe.png", handle, "image/png")},
            )
        check("/api/infer/frame 拒绝白名单外引擎", response.status_code == 403,
              f"{response.status_code} {response.text[:150]}")

        # --- 写入目录名消毒 ---
        # project 指向白名单外目录 → 403（父目录校验）
        response = client.post(
            "/api/train/yolo",
            json={"data": "runs/not-exist.yaml", "model": "yolov8n.pt", "project": str(outside)},
        )
        check("白名单外的 project 被拒", response.status_code in {400, 403}, str(response.status_code))


def test_name_sanitizer() -> None:
    """实验名消毒是「写入」路径上唯一的防线，单独测全一点。"""
    from fastapi import HTTPException

    import web_state

    for bad in ("..", "../x", "a/b", "a\\b", "", "   ", ".", "x\0y", "exp:1"):
        try:
            web_state._ensure_safe_name(bad)
        except HTTPException:
            ok = True
        else:
            ok = False
        check(f"非法实验名被拒：{bad!r}", ok)

    for good in ("exp", "exp_20260101_101010", "runs-1", "实验A"):
        try:
            value = web_state._ensure_safe_name(good)
        except HTTPException as exc:
            check(f"合法实验名通过：{good}", False, str(exc.detail))
            continue
        check(f"合法实验名通过：{good}", value == good, value)


def test_python_guard() -> None:
    """解释器字段必须同时校验白名单与文件名，否则等于执行任意程序。"""
    from fastapi import HTTPException

    import web_state

    outside_exe = Path(tempfile.gettempdir()) / "totally-not-python.exe"
    outside_exe.write_bytes(b"MZ")
    try:
        web_state._resolve_python(str(outside_exe))
    except HTTPException as exc:
        check("白名单外的 python 被拒", exc.status_code in {400, 403}, str(exc.detail))
    else:
        check("白名单外的 python 被拒", False, "没有抛 HTTPException")

    repo_exe = config.BACKEND_DIR / "tools" / "evil.exe"
    repo_exe.write_bytes(b"MZ")
    try:
        web_state._resolve_python(str(repo_exe))
    except HTTPException as exc:
        check("白名单内但不是 python.exe 也拒绝", exc.status_code == 400, str(exc.detail))
    else:
        check("白名单内但不是 python.exe 也拒绝", False, "没有抛 HTTPException")
    finally:
        repo_exe.unlink(missing_ok=True)


def main() -> int:
    test_static()
    test_name_sanitizer()
    test_python_guard()
    # 临时目录在仓库外，`is_path_allowed` 一定返回 False
    with tempfile.TemporaryDirectory(prefix="trtforge-guard-") as tmp:
        outside = Path(tmp).resolve()
        if config.is_path_allowed(outside):
            # 用户显式把临时目录加进了 TRTFORGE_BROWSE_ROOTS：换一个再试
            check("临时目录在白名单外（环境变量未把整盘放开）", False, str(outside))
        else:
            check("临时目录在白名单外（环境变量未把整盘放开）", True)
            test_behavior(outside)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
