# -*- coding: utf-8 -*-
"""装配层测试：`app.py` 必须只做装配，不再承载业务实现。

为什么单独测这个（路由快照测试挡不住）：
- 快照只核对「接口集合」，把 500 行业务塞回 `app.py` 它照样绿；
- 真正的回归是「一年后又长出第二条 4800 行的 app.py」，
  所以这里把三条约束写成断言：

1. `app.py` 行数有上限（当前 195 行，上限 400）；
2. `app.py` 里不再出现 `@app.get(` / `@app.post(` 这类路由注册（够不到的功能就写不进 app.py）；
3. `routers/*` 不 import `app`（否则拆分就退化成循环依赖，`import routers.x` 会拉起整个服务）。

另外验证 `app.<旧名字>` 的兼容转发仍然可用：拆分前有 40 多个脚本按
`app._finish_infer` 这样引用，转发层删早了会一片红。

用法：
    python backend/tools/test_app_assembly.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

TOOLS = BACKEND_DIR / "tools"
APP_LIMIT = 400

#: 拆分前外部脚本会按 `app.<名字>` 取的东西，取几个有代表性的即可
COMPAT_NAMES = (
    "_finish_infer",
    "_run_yolo_once",
    "infer_yolo",
    "BatchInferRequest",
    "build_train_command",
    "train_preflight",
    "_probe_onnx",
    "pipeline_stage_log",
    "get_run_log",
    "capabilities",
    "manager",
    "_describe_run",
)

ROUTERS = (
    "system",
    "train",
    "engine",
    "pipeline",
    "dataset",
    "runs",
    "infer",
    "models",
    "annotation",
)

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


def test_app_is_thin() -> None:
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    check(f"app.py 行数 {len(lines)} <= {APP_LIMIT}", len(lines) <= APP_LIMIT)

    # 只允许装配用装饰器；路由注册一律去 routers/
    route_decorators = re.findall(r"^@app\.(get|post|put|delete|patch|websocket)\(", source, re.M)
    check("app.py 内没有路由注册", not route_decorators, f"发现 {route_decorators}")

    check("app.py 显式 include_router", "include_router" in source)


def test_compat_shim() -> None:
    import app

    missing = [name for name in COMPAT_NAMES if not hasattr(app, name)]
    check("兼容转发可用", not missing, f"取不到 {missing}")

    raised = False
    try:
        getattr(app, "_definitely_not_a_real_name")
    except AttributeError:
        raised = True
    check("未知名字抛 AttributeError", raised)


def test_routers_independent() -> None:
    # 新进程里只 import routers，验证不会把 app 一起拉起来（无循环依赖）
    code = (
        "import sys;"
        + "".join(f"import routers.{name};" for name in ROUTERS)
        + "assert 'app' not in sys.modules, 'routers 反向 import 了 app';"
        + "import routers;"
        + "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(
        "routers 可在不导入 app 的前提下导入",
        proc.returncode == 0 and "ok" in (proc.stdout or ""),
        (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "",
    )


def test_each_router_has_routes() -> None:
    import importlib

    for name in ROUTERS:
        module = importlib.import_module(f"routers.{name}")
        count = len(getattr(module.router, "routes", []))
        check(f"routers.{name} 有 {count} 条路由", count > 0)


def main() -> int:
    print("装配层：app.py 只做装配")
    test_app_is_thin()
    test_compat_shim()
    test_routers_independent()
    test_each_router_has_routes()
    print(f"\n通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
