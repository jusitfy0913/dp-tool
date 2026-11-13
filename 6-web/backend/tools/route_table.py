# -*- coding: utf-8 -*-
"""把当前后端暴露的接口拍成一张快照，供重构时对比。

    python tools/route_table.py            # 打印当前路由表
    python tools/route_table.py --update   # 用它覆盖 route_table.json

为什么要这张表
--------------
`app.py` 正在按功能域拆成 `routers/*.py`。拆分本身只搬代码，但**最怕漏注册**：
少写一个 `include_router`、把 `/api/infer/anomalib` 写成 `/api/infer/anomalib/`、
或者把 websocket 路由挂进了普通 router，页面都会在某个次要功能上静默失效，
而这些路径测试用例未必覆盖到。

快照只关心「方法 + 路径」，不关心实现、顺序、tag：
- 顺序会变（拆分后按 include 顺序），所以比较前排序；
- tag / summary 属于文档，不影响调用，不纳入比较。

被排除的只有框架自带的文档路由（`/docs`、`/openapi.json`、`/redoc`、
`/docs/oauth2-redirect`）与静态挂载（`/static`）：它们由 FastAPI / Starlette
自动生成，不属于业务契约。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

SNAPSHOT = Path(__file__).resolve().parent / "route_table.json"

#: 文档与静态资源，不属于业务契约
IGNORED_PATHS = {
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
}


def collect(application=None) -> list[str]:
    """返回 ``["GET /api/health", "WS /ws/runs/{run_id}", ...]``（已排序去重）。"""
    if application is None:
        import app as app_module  # noqa: PLC0415 - 导入 app 会拉起配置与推理代理，只在需要时做

        application = app_module.app

    from fastapi.routing import APIRoute  # noqa: PLC0415
    from starlette.routing import WebSocketRoute  # noqa: PLC0415

    entries: set[str] = set()
    for route in application.routes:
        path = getattr(route, "path", None)
        if not path or path in IGNORED_PATHS:
            continue
        if isinstance(route, WebSocketRoute):
            entries.add(f"WS {path}")
            continue
        if isinstance(route, APIRoute):
            # HEAD 由 GET 自动派生，不需要单独记账
            for method in sorted(set(route.methods) - {"HEAD"}):
                entries.add(f"{method} {path}")
    return sorted(entries)


def load() -> list[str]:
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    routes = payload.get("routes") if isinstance(payload, dict) else payload
    return sorted(str(item) for item in routes or [])


def save(routes: list[str]) -> None:
    payload = {
        "_comment": "接口快照。新增或改名接口后执行 python tools/route_table.py --update 并说明原因。",
        "routes": sorted(routes),
    }
    SNAPSHOT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def diff(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    """返回 ``(缺失, 新增)``：缺失 = 快照里有而运行时没有，新增 = 运行时多出来的。"""
    want, have = set(expected), set(actual)
    return sorted(want - have), sorted(have - want)


def update(routes: list[str]) -> None:
    """重写快照，并把相对旧快照的变化逐条打印出来。

    重建快照时最怕「顺手一起更新了，没人看改了什么」。把新增/删除打在终端上，
    提交时这段输出就是评审要看的说明。
    """
    before = load() if SNAPSHOT.is_file() else []
    save(routes)
    missing, extra = diff(before, routes)
    print(f"已写入 {SNAPSHOT}（{len(routes)} 条，原 {len(before)} 条）")
    for item in extra:
        print(f"  [新增] {item}")
    for item in missing:
        print(f"  [删除] {item}")
    if not extra and not missing:
        print("  与旧快照无差异")


def main(argv: list[str]) -> int:
    routes = collect()
    if "--update" in argv:
        update(routes)
        return 0
    for item in routes:
        print(item)
    print(f"\n共 {len(routes)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
