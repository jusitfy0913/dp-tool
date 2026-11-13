# -*- coding: utf-8 -*-
"""TRTForge Web Studio 后端入口 —— 只做装配，不放业务实现。

职责边界（有意保持窄）：
- 训练：拼参数 -> 启动 launcher 子进程 -> 转发事件，不碰算法。
- 推理：调封装好的 DLL，不重写前处理与解码。
- 静态托管：frontend 目录直接由本服务提供，无前端构建步骤。

**路由按功能域拆在 `routers/` 里**（v1.38.0）。本文件只保留：版本号、lifespan、
异常处理器、`include_router` 清单、静态挂载与静态缓存中间件。
要加接口请改对应的 `routers/<域>.py`，并同步 `tools/route_table.json`
（`python tools/route_table.py --update`，`tools/test_route_table.py` 会盯着）。

为什么拆：拆分前本文件 4800 行、86 条路由。加一个推理参数要在几千行里找位置，
评审时 diff 永远看不清上下文，两个改动必然冲突。拆分只搬代码，
**URL、请求体、响应体、WebSocket 地址一字未改**。

启动：
    cd 6-web/backend
    python -m uvicorn app:app --host 127.0.0.1 --port 8060

或用仓库脚本：scripts/run_web_studio.ps1
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import autoexp_api
import config
import inferproxy
import logstore
import web_state
import warmpool
from runners import manager
from web_state import _reject_if_busy, configure_warm_pool, pipeline_manager

# 路由模块。**顺序有意义**：FastAPI 按注册顺序匹配，`/api/runs/checkpoints`
# 必须排在 `/api/runs/{run_id}` 之前，否则会被当成 run_id=checkpoints。
# 下面这个顺序与原 app.py 里各域首次出现的行号一致。
from routers import (
    annotation as annotation_routes,
    dataset as dataset_routes,
    engine as engine_routes,
    infer as infer_routes,
    models as models_routes,
    pipeline as pipeline_routes,
    runs as runs_routes,
    system as system_routes,
    train as train_routes,
)

#: 应用版本号。前端顶栏与 /api/health 都读它，方便一眼看出「页面连的是哪个后端」。
#: 起因：改了代码但 8060 上跑的是几小时前的旧进程，排查很久才发现。
#: 常量本体在 config：dataset.py 写 manifest 时也要用，反向 import app 会循环。
APP_VERSION = config.APP_VERSION


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时准备目录与事件循环，退出时清理子进程与句柄。"""
    config.ensure_dirs()
    # 旧的明文 Key 就地改成 DPAPI 密文。启动时做一次而不是读接口里做：
    # GET 请求不该偷偷写磁盘，写失败还会让页面莫名报错。
    try:
        migrated = autoexp_api.migrate_key_store()
        print(f"[app] API Key 存储：{migrated['reason']}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 迁移失败不能挡住服务启动
        print(f"[app] API Key 迁移失败：{type(exc).__name__}: {exc}", flush=True)
    manager.bind_loop(asyncio.get_running_loop())
    # 上次进程被强杀时，台账里会留下状态为 running 的「僵尸任务」。
    # 这里改判为 interrupted：前端能看见「它没跑完」，而不是一条永远转圈的记录。
    try:
        recovered = manager.recover_interrupted()
        if recovered:
            print(f"[app] 台账恢复：{len(recovered)} 个上次未结束的任务已标记为中断", flush=True)
    except Exception as exc:  # noqa: BLE001 - 台账坏了不能挡住服务启动
        print(f"[app] 台账恢复失败：{type(exc).__name__}: {exc}", flush=True)
    # 清理过期任务日志：单次任务有上限，任务数多了总量仍会无限涨。
    # 失败只记一笔，不影响启动。
    try:
        pruned = logstore.prune_logs()
        if pruned["families"]:
            print(
                f"[app] 清理过期日志：{pruned['families']} 个任务、{pruned['files']} 个文件、"
                f"{pruned['freed_bytes'] / 1048576:.1f} MB",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[app] 日志清理失败：{type(exc).__name__}: {exc}", flush=True)
    # 拉起推理工作进程并探测推理库，让 /api/capabilities 立刻反映真实状态
    inferproxy.startup()
    # 预热训练进程：失败只记一笔，不影响服务启动（用户仍可走冷启动）
    try:
        configure_warm_pool()
    except Exception as exc:  # noqa: BLE001
        print(f"预热进程初始化失败（不影响训练）：{type(exc).__name__}: {exc}", flush=True)
    try:
        yield
    finally:
        # 先停串联（它可能正握着某个 run），再停所有子进程
        pipeline_manager.shutdown_all()
        manager.shutdown_all()
        warmpool.pool.shutdown()
        inferproxy.shutdown()


app = FastAPI(title="TRTForge Web Studio", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(inferproxy.WorkerError)
async def _worker_error(_request, exc: inferproxy.WorkerError) -> JSONResponse:
    """推理进程崩溃 503 / 超时 504，所有推理路由共用。"""
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# 路由装配
# ---------------------------------------------------------------------------
for _router in (
    system_routes.router,
    train_routes.router,
    engine_routes.router,
    pipeline_routes.router,
    dataset_routes.router,
    runs_routes.router,
    infer_routes.router,
    models_routes.router,
    annotation_routes.router,
):
    app.include_router(_router)

# 自动实验（外置模块）：整块功能收在自己的 APIRouter 里，
# 上报并发闸门复用 `_reject_if_busy`，避免「自动实验和网页训练同时开跑」。
autoexp_api.set_busy_check(_reject_if_busy)
app.include_router(autoexp_api.router)

# 挂在最后，避免覆盖 /api 路由
if config.FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态资源一律要求重新校验。

    不带这个头时浏览器会按启发式规则把 .js 缓存住，改了前端、重启了后端，
    刷新页面跑的仍是旧模块。排查「按钮点了没反应」时遇到过：HTML 是新的、
    ES module 却是几小时前的。这些文件都在本机、体积很小，每次 revalidate
    的代价可以忽略，换来的是"刷新即最新"。
    """
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# ---------------------------------------------------------------------------
# 兼容转发（module `__getattr__`）
#
# 拆分前有 40 多个测试脚本与外部脚本按 `app._finish_infer`、`app.manager`、
# `app.train_preflight` 这样引用函数。为了让拆分是「纯移动 diff」、不把 40 个
# 文件一次性卷进来，这里做一次按名字转发的兜底：查不到就报 AttributeError，
# 与普通模块行为一致。
#
# 新代码请直接从 `routers.<域>` 或 `web_state` 取，别再依赖这层。
# 这一层在下一轮清理（改用 routers.* 之后）可以整段删掉。
# ---------------------------------------------------------------------------
_COMPAT_MODULES = (
    system_routes,
    train_routes,
    engine_routes,
    pipeline_routes,
    dataset_routes,
    runs_routes,
    infer_routes,
    models_routes,
    annotation_routes,
    web_state,
    autoexp_api,
)


def __getattr__(name: str):
    """让 `from app import _xxx` 继续可用，即使 `_xxx` 已经搬去 routers/。"""
    for module in _COMPAT_MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
