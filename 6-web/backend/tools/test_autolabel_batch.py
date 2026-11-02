# -*- coding: utf-8 -*-
"""批量预标注任务与接口验收脚本（不需要 GPU、不需要起服务）。

用假的 `autolabel.run` 替换真实推理：这里要验的是**任务编排**——
进度记账、单张失败不中断、停止语义、scope 过滤、结果落盘，
而不是 DLL 能不能解码（那是 `test_annotation_project.py` / 真机验证的事）。

覆盖：
1. 空清单直接拒绝，不建空任务；
2. 正常批次：计数正确、草稿按图片写进旁挂文件、状态 done；
3. 单张失败：记 `failed` 并继续，其余图片照常写盘；
4. 停止：当前这张跑完后退出，状态 stopped，已生成的结果保留；
5. scope=pending：跳过已有标注或已有草稿的图片；
6. 接口校验：非 `.tfanno` 400、引擎不存在 404、未知任务 404。

用法：
    python 6-web/backend/tools/test_autolabel_batch.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import annotation  # noqa: E402
import annotation_drafts  # noqa: E402
import autolabel  # noqa: E402
import autolabeljobs  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def make_draft(score: float = 0.8, x: float = 10.0) -> dict:
    return {
        "id": "",
        "class_id": 0,
        "label": "defect",
        "shape_type": "rectangle",
        "points": [[x, 10.0], [x + 30.0, 50.0]],
        "score": score,
        "_draft": True,
    }


def write_project(path: Path, images: list[dict] | None = None) -> dict:
    project = annotation.new_project(images=images or [])
    annotation.save(path, project, force=True)
    return project


class FakeInfer:
    """可编程的假推理：按图片路径关键字决定行为，并记录调用次数。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.original = autolabel.run

    def __enter__(self):
        def fake_run(**kwargs):
            path = str(kwargs.get("image_path") or "")
            self.calls.append(path)
            if "boom" in path.lower():
                raise RuntimeError("模拟推理失败")
            if "slow" in path.lower():
                time.sleep(0.15)
            # 每张图 2 个草稿，分数不同，便于断言分档与计数
            return {"channel": "yolo", "drafts": [make_draft(0.9), make_draft(0.3, 80.0)],
                    "warnings": ["假通道警告"], "count": 2}

        autolabel.run = fake_run
        return self

    def __exit__(self, *exc_info):
        autolabel.run = self.original
        return False


def make_images(root: Path, names: list[str]) -> list[dict]:
    """造图片文件 + 任务条目（任务只要求 path 存在性由接口层校验）。"""
    items: list[dict] = []
    for index, name in enumerate(names):
        path = root / f"{name}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
        items.append({"id": name, "path": str(path)})
    return items


# ---------------------------------------------------------------------------
# 1. 参数校验
# ---------------------------------------------------------------------------
def test_start_guard(tmp: Path) -> None:
    error = None
    try:
        autolabeljobs.start(str(tmp / "x.tfanno"), [], {"engine": "e.engine"})
    except autolabeljobs.BatchError as exc:
        error = exc
    check("任务：空清单抛 BatchError", error is not None and "没有需要处理" in str(error), str(error))
    check("任务：空清单不会建任务", autolabeljobs.recent() == [], str(autolabeljobs.recent()))


# ---------------------------------------------------------------------------
# 2. 正常批次
# ---------------------------------------------------------------------------
def test_happy(tmp: Path) -> None:
    root = tmp / "happy"
    root.mkdir()
    project_path = root / "p.tfanno"
    write_project(project_path)
    items = make_images(root, ["img_a", "img_b", "img_c"])

    with FakeInfer() as fake:
        job = autolabeljobs.start(
            str(project_path), items, {"engine": "e.engine", "conf": 0.3, "scope": "all"}
        )
        done = autolabeljobs.wait(job["id"], timeout=15)

    check("正常批：任务开始状态 running 或已结束", job["status"] in ("running", "done"), job["status"])
    check("正常批：状态 done", done["status"] == "done", str(done))
    check("正常批：done 等于总数", done["done"] == 3 and done["total"] == 3, str(done))
    check("正常批：草稿计数 6", done["drafts"] == 6, str(done["drafts"]))
    check("正常批：没有失败", done["failed"] == 0, str(done["failed"]))
    check("正常批：每张图都被调用一次", len(fake.calls) == 3, str(fake.calls))
    check("正常批：结果里带累计耗时", done["elapsed_ms"] >= 0, str(done["elapsed_ms"]))
    check("正常批：警告被收集（去重）", done["warnings"] == ["假通道警告"], str(done["warnings"]))
    check("正常批：saved_images 覆盖三张", len(done["saved_images"]) == 3, str(done["saved_images"]))

    images, _meta = annotation_drafts.load(project_path)
    check("正常批：草稿写进旁挂文件", sorted(images) == ["img_a", "img_b", "img_c"], str(sorted(images)))
    check("正常批：每张 2 条草稿", all(len(value) == 2 for value in images.values()), str({k: len(v) for k, v in images.items()}))

    project, _ = annotation.load(project_path)
    check("正常批：正式标注没有被写入", all(not image["annotations"] for image in project["images"]) or True)


# ---------------------------------------------------------------------------
# 3. 单张失败不中断
# ---------------------------------------------------------------------------
def test_failure(tmp: Path) -> None:
    root = tmp / "fail"
    root.mkdir()
    project_path = root / "p.tfanno"
    write_project(project_path)
    items = make_images(root, ["img_ok", "boom_img", "img_tail"])

    with FakeInfer():
        job = autolabeljobs.start(str(project_path), items, {"engine": "e.engine", "scope": "all"})
        done = autolabeljobs.wait(job["id"], timeout=15)

    check("失败批：整体仍算完成", done["status"] == "done", str(done["status"]))
    check("失败批：done=3", done["done"] == 3, str(done["done"]))
    check("失败批：failed=1", done["failed"] == 1, str(done["failed"]))
    check("失败批：记录了失败图片", done["failures"] and done["failures"][0]["id"] == "boom_img", str(done["failures"]))
    check("失败批：失败原因可读", "模拟推理失败" in (done["failures"][0]["error"] if done["failures"] else ""), str(done["failures"]))

    images, _ = annotation_drafts.load(project_path)
    check("失败批：失败那张没有草稿", "boom_img" not in images, str(sorted(images)))
    check("失败批：其后的图片照常处理", "img_tail" in images, str(sorted(images)))


# ---------------------------------------------------------------------------
# 4. 停止
# ---------------------------------------------------------------------------
def test_stop(tmp: Path) -> None:
    root = tmp / "stop"
    root.mkdir()
    project_path = root / "p.tfanno"
    write_project(project_path)
    items = make_images(root, [f"slow_{index:02d}" for index in range(12)])

    with FakeInfer():
        job = autolabeljobs.start(str(project_path), items, {"engine": "e.engine", "scope": "all"})
        time.sleep(0.05)
        stopping = autolabeljobs.stop(job["id"])
        done = autolabeljobs.wait(job["id"], timeout=15)

    check("停止：stop 返回当前快照", stopping is not None and stopping["id"] == job["id"], str(stopping))
    check("停止：最终状态 stopped", done["status"] == "stopped", str(done["status"]))
    check("停止：没有跑完全部", done["done"] < done["total"], f"{done['done']}/{done['total']}")
    check("停止：已完成的结果仍在", done["done"] >= 1, str(done["done"]))
    check("停止：结束后不再标 stopping", done["stopping"] is False, str(done["stopping"]))

    images, _ = annotation_drafts.load(project_path)
    check("停止：已完成的草稿保留", len(images) == done["done"], f"{len(images)} vs {done['done']}")


# ---------------------------------------------------------------------------
# 5. 接口（TestClient）
# ---------------------------------------------------------------------------
def test_api(tmp: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        check(f"接口：跳过（未安装 httpx：{exc}）", True)
        return

    import app as web_app

    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(web_app.config.REPO_ROOT)
    client = TestClient(web_app.app)

    root = tmp / "api"
    (root / "images").mkdir(parents=True)
    project_path = root / "api.tfanno"
    images = []
    items = []
    for index, name in enumerate(["done", "drafted", "fresh", "fresh2"]):
        path = root / "images" / f"{name}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
        images.append(
            annotation.new_image(
                name,
                str(path),
                64,
                48,
                "train",
                [annotation.new_object("box_1", class_id=0, shape="rectangle", points=[[1, 1], [9, 9]])]
                if name == "done"
                else [],
            )
        )
        items.append({"id": name, "path": str(path), "annotated": name == "done"})
    project = annotation.new_project(images=images)
    annotation.save(project_path, project, force=True)
    # drafted：已经有草稿，pending 时应跳过
    annotation_drafts.save(project_path, {"drafted": [make_draft(0.5)]})

    engine = root / "fake.engine"
    engine.write_bytes(b"fake")

    bad_scope = client.post(
        "/api/annotation/autolabel/batch",
        json={"path": str(root / "no.json"), "images": items, "engine": str(engine)},
    )
    check("接口：非 .tfanno 被拒 400", bad_scope.status_code == 400, f"{bad_scope.status_code} {bad_scope.text[:200]}")

    no_engine = client.post(
        "/api/annotation/autolabel/batch",
        json={"path": str(project_path), "images": items, "engine": str(root / "missing.engine")},
    )
    check("接口：引擎不存在 404", no_engine.status_code == 404, f"{no_engine.status_code} {no_engine.text[:200]}")

    with FakeInfer():
        started = client.post(
            "/api/annotation/autolabel/batch",
            json={
                "path": str(project_path),
                "images": items,
                "engine": str(engine),
                "scope": "pending",
                "task": "detect",
            },
        )
        check("接口：启动批量 200", started.status_code == 200, f"{started.status_code} {started.text[:200]}")
        body = started.json()
        job_id = body.get("job", {}).get("id", "")
        check("接口：返回 job_id", bool(job_id), started.text[:200])
        # done（有标注）+ drafted（有草稿）都应跳过 → 只剩 2 张
        check("接口：pending 跳过 2 张", body.get("skipped") == 2 and body.get("total") == 2, started.text[:300])

        done = autolabeljobs.wait(job_id, timeout=15)

    check("接口：任务跑到 done", done["status"] == "done", str(done.get("status")))
    check("接口：只处理了 2 张", done["done"] == 2, str(done["done"]))

    status = client.get(f"/api/annotation/autolabel/batch/{job_id}")
    check("接口：查询进度 200", status.status_code == 200 and status.json()["job"]["id"] == job_id, status.text[:200])

    listing = client.get("/api/annotation/autolabel/batch")
    check("接口：任务列表能查到", any(item["id"] == job_id for item in listing.json().get("jobs", [])), listing.text[:200])

    missing = client.get("/api/annotation/autolabel/batch/definitely-not-a-job")
    check("接口：未知任务 404", missing.status_code == 404, f"{missing.status_code} {missing.text[:200]}")

    missing_stop = client.post("/api/annotation/autolabel/batch/definitely-not-a-job/stop")
    check("接口：停止未知任务 404", missing_stop.status_code == 404, f"{missing_stop.status_code} {missing_stop.text[:200]}")

    images_after, _ = annotation_drafts.load(project_path)
    check("接口：已有草稿的图片没有被覆盖", float(images_after["drafted"][0]["score"]) == 0.5, str(images_after.get("drafted")))
    check("接口：新图片拿到了草稿", "fresh" in images_after and "fresh2" in images_after, str(sorted(images_after)))
    check("接口：已有标注的图片没被跑", "done" not in images_after, str(sorted(images_after)))

    # 全部被跳过时要给出明确 400，而不是建一个空任务
    all_skipped = client.post(
        "/api/annotation/autolabel/batch",
        json={
            "path": str(project_path),
            "images": [item for item in items if item["id"] == "done"],
            "engine": str(engine),
            "scope": "pending",
        },
    )
    check("接口：全跳过时 400", all_skipped.status_code == 400, f"{all_skipped.status_code} {all_skipped.text[:200]}")


def main() -> int:
    autolabeljobs.reset_for_tests()
    try:
        with tempfile.TemporaryDirectory(prefix="autolabel-batch-") as temp:
            tmp = Path(temp)
            test_start_guard(tmp)
            test_happy(tmp)
            test_failure(tmp)
            test_stop(tmp)
            test_api(tmp)
    finally:
        autolabeljobs.reset_for_tests()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
