# -*- coding: utf-8 -*-
"""全链路自检：一条命令从「建工程」走到「推理」，确认各环节串起来没断。

为什么要有它：
- 单元测试各自只验一段逻辑，接口之间「字段名改了但对面没改」这种断裂没人拦；
- 手工点一遍 Web 要 3~5 分钟，回归时通常会被跳过；
- 走真实 HTTP（ASGI in-process），路径是前端真正调的那些。

覆盖 8 个环节：
    1. project  建工程（保存 → 读回）
    2. images   加图（扫描目录 → 取图片流 → 取尺寸）
    3. annotate 写标注（带 rev 乐观锁再保存 → 统计对得上）
    4. export   导出 YOLO 数据集
    5. dataset  数据集扫描（读刚导出的目录）
    6. models   产物枚举（假引擎文件）
    7. bundle   模型包造包 + 完整校验（固定运行时戳）
    8. infer    推理（假引擎桩，验响应结构与出图）

用法：
    python 6-web/backend/tools/test_full_chain.py              # 全部环节
    python 6-web/backend/tools/test_full_chain.py --step infer # 只跑到指定环节
    python 6-web/backend/tools/test_full_chain.py --keep       # 保留临时目录
    python 6-web/backend/tools/test_full_chain.py --list       # 列出环节名

退出码 0 = 全绿；非 0 = 某环节失败（会打印该环节的请求与响应体）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

#: 环节顺序（--step 依赖它做前缀执行：后面的环节需要前面的产物）
STEPS = ["project", "images", "annotate", "export", "dataset", "models", "bundle", "infer"]

#: 打印请求/响应体时的截断长度，避免 CI 日志被一张 base64 图淹掉
BODY_LIMIT = 400


class ChainError(AssertionError):
    """某个环节的断言失败。带上环节名，主函数据此定位并打印报文。"""


def _brief(value) -> str:
    """把请求体/响应体压成一行可读文本。"""
    if isinstance(value, bytes):
        text = f"<{len(value)} bytes>"
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= BODY_LIMIT else text[:BODY_LIMIT] + "…"


def _describe(kwargs: dict) -> str:
    """请求参数摘要：query / json / data / files 里有什么就打什么。"""
    parts = []
    if kwargs.get("params"):
        parts.append(f"params={_brief(kwargs['params'])}")
    if kwargs.get("json") is not None:
        parts.append(f"json={_brief(kwargs['json'])}")
    if kwargs.get("data"):
        parts.append(f"data={_brief(kwargs['data'])}")
    if kwargs.get("files"):
        parts.append("files=" + _brief({k: "<upload>" for k in kwargs["files"]}))
    return " ".join(parts)


class Chain:
    """一次全链路的上下文：临时目录、HTTP 客户端、各环节产物的路径。"""

    def __init__(self, tmp: Path, client) -> None:
        self.tmp = tmp
        self.client = client
        self.last: tuple[str, str, str, dict, object] | None = None
        self.facts: dict[str, str] = {}
        self.keep = False

    # -- HTTP ---------------------------------------------------------------
    def req(self, label: str, method: str, url: str, expect: int = 200, **kwargs):
        """发一次请求并断言状态码；失败时把这次报文记在 ``self.last`` 里。"""
        response = self.client.request(method, url, **kwargs)
        self.last = (label, method, url, kwargs, response)
        if response.status_code != expect:
            raise ChainError(
                f"{label}：期望 {expect}，实际 {response.status_code}\n"
                f"    响应体：{response.text[:BODY_LIMIT]}"
            )
        return response

    def json(self, label: str, method: str, url: str, expect: int = 200, **kwargs) -> dict:
        response = self.req(label, method, url, expect, **kwargs)
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - 只有后端返回非 JSON 才会走到
            raise ChainError(f"{label}：响应不是 JSON（{exc}）：{response.text[:BODY_LIMIT]}") from exc

    # -- 断言 ---------------------------------------------------------------
    def check(self, label: str, condition, detail: str = "") -> None:
        """条件不成立就抛 ChainError；detail 用于说明"实际是什么"。"""
        if not condition:
            raise ChainError(f"{label}：{detail or '断言不成立'}")

    def eq(self, label: str, actual, expected) -> None:
        if actual != expected:
            raise ChainError(f"{label}：期望 {expected!r}，实际 {actual!r}")

    def report(self, step: str, note: str) -> None:
        self.facts[step] = note
        print(f"  ✓ {note}")


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def make_image(path: Path, width: int = 320, height: int = 240) -> None:
    """生成一张纯色 PNG：测试不能依赖仓库里的真实数据。"""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(canvas, (10, 10), (100, 80), (255, 120, 30), -1)
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:  # pragma: no cover - 编码失败只可能是 OpenCV 环境坏了
        raise ChainError("夹具：测试图片编码失败")
    buffer.tofile(str(path))


def build_project(root: Path) -> dict:
    """一份最小但字段完整的工程：1 类 / 1 图 / 1 框。"""
    import annotation

    classes = [annotation.new_class(0, "missing_hole", [31, 182, 255])]
    images = [
        annotation.new_image(
            "img_a",
            str(root / "images" / "a.png"),
            320,
            240,
            "train",
            [annotation.new_object("o1", 0, "missing_hole", "rectangle", [[10, 20], [110, 90]])],
        )
    ]
    return annotation.new_project("detect", classes, images)


# ---------------------------------------------------------------------------
# 1. 建工程
# ---------------------------------------------------------------------------
def step_project(ctx: Chain) -> None:
    path = ctx.tmp / "proj" / "chain.tfanno"
    ctx.bundle_path = path
    make_image(ctx.tmp / "proj" / "images" / "a.png")
    project = build_project(ctx.tmp / "proj")

    saved = ctx.json(
        "保存工程",
        "POST",
        "/api/annotation/project",
        json={"path": str(path), "project": project, "force": True},
    )
    ctx.eq("首次保存 rev", saved.get("rev"), 1)
    ctx.eq("首次保存对象数", (saved.get("stats") or {}).get("total_objects"), 1)
    ctx.check("工程文件落盘", path.is_file(), str(path))

    loaded = ctx.json("读回工程", "GET", "/api/annotation/project", params={"path": str(path)})
    ctx.eq("读回 exists", loaded.get("exists"), True)
    ctx.eq("读回类别名", (loaded.get("project") or {}).get("classes", [{}])[0].get("name"), "missing_hole")
    ctx.project = loaded.get("project")
    ctx.rev = saved.get("rev")
    ctx.report("project", f"工程已建立（rev={ctx.rev}，1 类 / 1 图 / 1 框）")


# ---------------------------------------------------------------------------
# 2. 加图
# ---------------------------------------------------------------------------
def step_images(ctx: Chain) -> None:
    images_dir = ctx.tmp / "proj" / "images"
    make_image(images_dir / "b.png")

    scan = ctx.json("扫描图片目录", "POST", "/api/annotation/scan-images", params={"root": str(images_dir)})
    ctx.eq("扫描到的图片数", scan.get("count"), 2)
    ctx.eq("扫描结果不截断", scan.get("truncated"), False)

    stream = ctx.req("取图片流", "GET", "/api/annotation/image", params={"path": str(images_dir / "a.png")})
    ctx.check("图片字节非空", len(stream.content) > 0, str(len(stream.content)))
    ctx.check(
        "图片 MIME 是 png",
        stream.headers.get("content-type", "").startswith("image/png"),
        stream.headers.get("content-type", ""),
    )

    meta = ctx.json("取图片尺寸", "GET", "/api/annotation/image-meta", params={"path": str(images_dir / "a.png")})
    ctx.eq("图片宽", meta.get("width"), 320)
    ctx.eq("图片高", meta.get("height"), 240)
    ctx.report("images", "扫描到 2 张图，图片流与尺寸都正常")


# ---------------------------------------------------------------------------
# 3. 写标注
# ---------------------------------------------------------------------------
def step_annotate(ctx: Chain) -> None:
    import annotation

    project = ctx.project
    image = project["images"][0]
    image["annotations"].append(
        annotation.new_object("o2", 0, "missing_hole", "rectangle", [[150, 60], [260, 180]])
    )

    saved = ctx.json(
        "带 rev 再保存",
        "POST",
        "/api/annotation/project",
        json={"path": str(ctx.bundle_path), "project": project, "rev": ctx.rev},
    )
    ctx.eq("第二次保存 rev", saved.get("rev"), ctx.rev + 1)
    ctx.eq("第二次保存对象数", (saved.get("stats") or {}).get("total_objects"), 2)

    loaded = ctx.json("复核标注", "GET", "/api/annotation/project", params={"path": str(ctx.bundle_path)})
    objects = (loaded.get("project") or {}).get("images", [{}])[0].get("annotations", [])
    ctx.eq("读回的框数", len(objects), 2)
    ctx.eq("读回的第二个框形状", objects[1].get("shape_type"), "rectangle")
    ctx.project = loaded.get("project")
    ctx.rev = saved.get("rev")
    ctx.export_project = loaded.get("project")
    ctx.report("annotate", f"标注已写入（rev={ctx.rev}，2 框）")


# ---------------------------------------------------------------------------
# 4. 导出
# ---------------------------------------------------------------------------
def step_export(ctx: Chain) -> None:
    out = ctx.tmp / "export"
    exported = ctx.json(
        "导出 YOLO",
        "POST",
        "/api/annotation/export",
        json={
            "project": ctx.export_project,
            "source": str(ctx.bundle_path),
            "format": "yolo",
            "output": str(out),
            "overwrite": True,
        },
    )
    counts = exported.get("counts") or {}
    ctx.eq("导出 train 张数", counts.get("train"), 1)
    ctx.check("导出目录存在", out.is_dir(), str(out))
    labels = sorted((out / "labels").rglob("*.txt")) if (out / "labels").is_dir() else []
    ctx.check("导出产生标签文件", len(labels) == 1, f"实际 {len(labels)} 个")
    ctx.check(
        "标签文件有两行（两个框）",
        len(labels[0].read_text(encoding="utf-8").strip().splitlines()) == 2,
        labels[0].read_text(encoding="utf-8")[:120],
    )
    ctx.check("导出带 dataset.yaml", (out / "dataset.yaml").is_file(), str(out / "dataset.yaml"))
    ctx.export_dir = out
    ctx.report("export", f"导出 train={counts.get('train')} / val={counts.get('val')}")


# ---------------------------------------------------------------------------
# 5. 数据集扫描
# ---------------------------------------------------------------------------
def step_dataset(ctx: Chain) -> None:
    preview = ctx.json("扫描导出数据集", "POST", "/api/dataset/scan", json={"source": str(ctx.export_dir)})
    counts = preview.get("counts") or {}
    ctx.eq("数据集总张数", counts.get("total"), 1)
    ctx.eq("数据集类别数", len(preview.get("classes") or []), 1)
    ctx.eq("数据集识别为检测任务", preview.get("task"), "detect")
    ctx.check("数据集没有空标注图", counts.get("empty") == 0, str(counts.get("empty")))
    ctx.report("dataset", f"数据集可读（total={counts.get('total')}，类别 1 个）")


# ---------------------------------------------------------------------------
# 6. 产物枚举
# ---------------------------------------------------------------------------
def step_models(ctx: Chain) -> None:
    import models

    runs = ctx.tmp / "runs"
    weights = runs / "proj" / "detect_v1" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    engine = weights / "best_fp16.engine"
    engine.write_bytes(b"FAKE-ENGINE" * 8)
    labels = weights.parent / "labels_detect_v1.json"
    labels.write_text(json.dumps({"names": {"0": "missing_hole"}}, ensure_ascii=False), encoding="utf-8")
    ctx.engine = engine
    ctx.labels = labels

    # 产物根是固定目录（YOLO_LAUNCHER_DIR/runs），测试要指到临时目录：
    # 换 SCAN_TARGETS 而不是改 config，避免影响同进程里其它模块的判定。
    original = models.SCAN_TARGETS
    models.SCAN_TARGETS = [("yolo", lambda: runs)]
    try:
        listed = ctx.json("枚举训练产物", "GET", "/api/models")
    finally:
        models.SCAN_TARGETS = original

    entries = listed.get("entries") or []
    ctx.check("枚举到引擎", len(entries) == 1, f"实际 {len(entries)} 条")
    ctx.eq("引擎类型识别", entries[0].get("kind"), "yolo")
    ctx.check("引擎路径一致", Path(entries[0].get("path", "")).name == engine.name, str(entries[0].get("path")))
    ctx.check("产物根可用", any((r or {}).get("ok") for r in listed.get("roots") or []), _brief(listed.get("roots")))
    ctx.report("models", "产物枚举正常（1 个 yolo 引擎）")


# ---------------------------------------------------------------------------
# 7. 模型包
# ---------------------------------------------------------------------------
class FixedRuntime:
    """把运行时探测固定住，否则断言只能在本机 TensorRT 版本上通过。"""

    def __enter__(self):
        import bundle

        self._bundle = bundle
        self._orig = (bundle.trt_version, bundle.cuda_version, bundle.gpu_snapshot)
        bundle.trt_version = lambda: "8.5.2.2"
        bundle.cuda_version = lambda: "11.8"
        bundle.gpu_snapshot = lambda: {"name": "NVIDIA T1000", "compute_cap": "7.5"}
        return self

    def __exit__(self, *exc_info):
        bundle = self._bundle
        bundle.trt_version, bundle.cuda_version, bundle.gpu_snapshot = self._orig
        return False


def step_bundle(ctx: Chain) -> None:
    with FixedRuntime():
        built = ctx.json(
            "造模型包",
            "POST",
            "/api/bundles/build",
            json={
                "model": str(ctx.engine),
                "bundle_id": "chain_v1",
                "out_root": str(ctx.tmp / "bundles"),
                "kind": "yolo",
                "task": "detect",
                "labels": str(ctx.labels),
                "precision": "fp16",
                "imgsz": 640,
            },
        )
        target = Path(built.get("path") or "")
        ctx.check("包目录存在", bool(target) and target.is_dir(), str(target))
        ctx.check("清单文件存在", (target / "manifest.json").is_file(), str(target / "manifest.json"))

        verified = ctx.json("校验模型包", "POST", "/api/bundles/verify", json={"bundle": str(target)})
        ctx.eq("模型包校验结论", verified.get("level"), "ok")
        manifest = verified.get("manifest") or {}
        ctx.eq("清单 schema", manifest.get("schema"), "trtforge.bundle.v1")
        ctx.eq("清单任务类型", manifest.get("task"), "detect")
        ctx.eq("清单精度", (manifest.get("runtime") or {}).get("precision"), "fp16")

    ctx.bundle_dir = target
    ctx.report("bundle", f"模型包可用（{target.name}，schema=1）")


# ---------------------------------------------------------------------------
# 8. 推理
# ---------------------------------------------------------------------------
def step_infer(ctx: Chain) -> None:
    """用假引擎桩跑一次推理：验的是"接口契约 + 出图"，不是模型精度。"""
    import labels as labels_mod
    from routers import infer as infer_router

    boxes = [
        {
            "left": 10,
            "top": 20,
            "right": 110,
            "bottom": 90,
            "confidence": 0.93,
            "class_label": 0,
            "label_name": "missing_hole",
        }
    ]
    colors = labels_mod.class_colors(1)

    def fake_run_once(engine, task, array, conf, nms, labels_path, source=""):
        # 必须返回 (result, warning) 二元组，且 result 里带绘制需要的配色；
        # timing 带上是为了让 _finish_infer 走真实的"补耗时"分支
        return (
            {
                "boxes": [dict(item) for item in boxes],
                "class_colors": colors,
                "backend": "stub",
                "timing": {"total_ms": 3.0, "decode_ms": 1.0, "ipc_ms": 0.5},
            },
            "",
        )

    original = infer_router._run_yolo_once
    infer_router._run_yolo_once = fake_run_once
    try:
        image_bytes = (ctx.tmp / "proj" / "images" / "a.png").read_bytes()
        result = ctx.json(
            "推理",
            "POST",
            "/api/infer/yolo",
            files={"image": ("a.png", image_bytes, "image/png")},
            data={"engine": str(ctx.engine), "task": "detect", "conf": "0.25", "nms": "0.45"},
        )
    finally:
        infer_router._run_yolo_once = original

    got = result.get("boxes") or []
    ctx.eq("返回框数", len(got), 1)
    ctx.eq("框的类别名", got[0].get("label_name"), "missing_hole")
    ctx.check("返回标注图", bool(result.get("annotated_jpg")), "annotated_jpg 为空")
    ctx.eq("标注图 MIME", result.get("annotated_mime"), "image/jpeg")
    timing = result.get("timing") or {}
    ctx.check("补齐服务端耗时", "server_ms" in timing, _brief(timing))
    ctx.check("记录响应体体积", (timing.get("payload_bytes") or 0) > 0, _brief(timing))
    ctx.report("infer", f"推理链路通（1 框，bbox={got[0].get('left')},{got[0].get('top')}）")


STEPS_MAP = {
    "project": step_project,
    "images": step_images,
    "annotate": step_annotate,
    "export": step_export,
    "dataset": step_dataset,
    "models": step_models,
    "bundle": step_bundle,
    "infer": step_infer,
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(steps: list[str], keep: bool) -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("✗ 缺少 httpx：请先 pip install httpx（TestClient 依赖它）")
        return 2

    import app as web_app

    tmp = Path(tempfile.mkdtemp(prefix="trtforge_chain_"))
    # 白名单默认只有仓库根，临时目录必须显式放行，否则每个接口都先 403
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(web_app.config.REPO_ROOT)

    client = TestClient(web_app.app)
    ctx = Chain(tmp, client)
    ctx.keep = keep
    print(f"全链路自检（临时目录：{tmp}）")
    failures = 0
    current = ""
    for name in steps:
        current = name
        print(f"\n[{name}]")
        try:
            STEPS_MAP[name](ctx)
        except ChainError as exc:
            failures += 1
            print(f"  ✗ {exc}")
            if ctx.last:
                label, method, url, kwargs, response = ctx.last
                print(f"    最近请求：{label} → {method} {url} {_describe(kwargs)}")
                print(f"    响应 {response.status_code}：{response.text[:BODY_LIMIT]}")
            break
        except Exception as exc:  # noqa: BLE001 - 任何异常都要带环节名打印出来
            failures += 1
            print(f"  ✗ {name} 环节异常：{type(exc).__name__}: {exc}")
            traceback.print_exc()
            break

    print()
    if failures:
        print(f"✗ 全链路自检失败（在 {current} 环节停下）")
    else:
        print(f"✓ 全链路自检通过：{len(steps)} 个环节")
        for name in steps:
            print(f"    {name:<9} {ctx.facts.get(name, '')}")

    if keep:
        print(f"\n临时目录已保留：{tmp}")
    else:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="TRTForge Web 全链路自检")
    parser.add_argument("--step", default="", help=f"只跑到指定环节：{', '.join(STEPS)}")
    parser.add_argument("--keep", action="store_true", help="保留临时目录，便于事后查看产物")
    parser.add_argument("--list", action="store_true", help="列出环节名")
    args = parser.parse_args(argv[1:])

    if args.list:
        print("\n".join(STEPS))
        return 0

    target = (args.step or "").strip()
    if target:
        if target not in STEPS:
            print(f"未知环节：{target}\n可选：{', '.join(STEPS)}")
            return 2
        # 前缀执行：后面的环节依赖前面生成的工程与产物
        steps = STEPS[: STEPS.index(target) + 1]
    else:
        steps = list(STEPS)
    return run(steps, args.keep)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
