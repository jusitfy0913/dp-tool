# -*- coding: utf-8 -*-
"""预标注草稿旁挂持久化（`*.drafts.json`）验收脚本（不需要 GPU、不需要起服务）。

覆盖变更 `add-batch-autolabel-review` 里"草稿落盘"这一块最容易出错的点：

1. **不污染 `.tfanno`**：草稿写在旁挂文件，工程文件一个字节都不变；
2. **增量写**：只提交改过的图片，其余图片的草稿在磁盘上必须原样保留
   （批量任务与前端自动保存会同时写，整份覆盖会互相清零）；
3. **坏数据不炸**：非法 JSON、非对象条目、没有点的草稿、置信度越界，
   都要降级成"跳过并给出警告"，接口不能 500；
4. **原子写**：中途失败不会留下半截文件，也不留临时文件；
5. **接口契约**：只接受 `.tfanno` 作为归属路径（否则等于"往任意路径写 JSON"）。

用法：
    python 6-web/backend/tools/test_annotation_drafts.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import annotation  # noqa: E402
import annotation_drafts  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def make_draft(score: float = 0.9, x: float = 10.0, label: str = "hole") -> dict:
    """一条矩形草稿。id 留空，让规范化阶段补一个。"""
    return {
        "id": "",
        "class_id": 0,
        "label": label,
        "shape_type": "rectangle",
        "points": [[x, 10.0], [x + 40.0, 60.0]],
        "score": score,
        "_draft": True,
    }


def write_project(path: Path) -> None:
    """写一个最小的合法 `.tfanno`，用于断言"草稿写入不改工程"。"""
    project = annotation.new_project()
    annotation.save(path, project, force=True)


# ---------------------------------------------------------------------------
# 1. 路径推导
# ---------------------------------------------------------------------------
def test_path(tmp: Path) -> None:
    check(
        "路径：foo.tfanno → foo.drafts.json",
        annotation_drafts.drafts_path(tmp / "foo.tfanno").name == "foo.drafts.json",
        annotation_drafts.drafts_path(tmp / "foo.tfanno").name,
    )
    check(
        "路径：大小写混写也能认（.TFANNO）",
        annotation_drafts.drafts_path(tmp / "Foo.TFANNO").name == "Foo.drafts.json",
        annotation_drafts.drafts_path(tmp / "Foo.TFANNO").name,
    )
    other = annotation_drafts.drafts_path(tmp / "dataset")
    check("路径：非 tfanno 也能推导（不会抛错）", other.name == "dataset.drafts.json", other.name)


# ---------------------------------------------------------------------------
# 2. 往返一致 + 不污染工程
# ---------------------------------------------------------------------------
def test_roundtrip(tmp: Path) -> None:
    root = tmp / "roundtrip"
    root.mkdir()
    project_path = root / "demo.tfanno"
    write_project(project_path)
    before = project_path.read_bytes()

    draft = make_draft(0.87)
    draft["_note"] = "来自外部服务"
    draft["_warnings"] = ["类别名缺失，已用类别号"]
    meta = annotation_drafts.save(project_path, {"img_a": [draft], "img_b": [make_draft(0.31)]})

    check("往返：save 返回草稿总数", meta["count"] == 2, str(meta))
    check("往返：回报图片张数", meta.get("image_count") == 2, str(meta))
    check("往返：旁挂文件存在", (root / "demo.drafts.json").is_file())
    check("往返：工程文件一字未改", project_path.read_bytes() == before)

    images, meta2 = annotation_drafts.load(project_path)
    check("往返：读回两张图的草稿", sorted(images) == ["img_a", "img_b"], str(sorted(images)))
    kept = images["img_a"][0]
    check("往返：置信度保留", abs(float(kept["score"]) - 0.87) < 1e-6, str(kept.get("score")))
    check("往返：矩形折叠为 2 点", len(kept["points"]) == 2, str(kept.get("points")))
    check("往返：_draft 标记被补回", kept.get("_draft") is True, str(kept))
    check("往返：_note 保留", kept.get("_note") == "来自外部服务", str(kept.get("_note")))
    check("往返：_warnings 保留", kept.get("_warnings") == ["类别名缺失，已用类别号"], str(kept.get("_warnings")))
    check("往返：自动补了草稿 id", bool(kept.get("id")), str(kept.get("id")))
    check("往返：meta.exists 正确", meta2.get("exists") is True, str(meta2))

    # 文件结构本身也要对：Qt 不会读它，但人要看懂
    raw = json.loads((root / "demo.drafts.json").read_text(encoding="utf-8"))
    check("往返：根结构带 version", raw.get("version") == annotation_drafts.CURRENT_VERSION, str(raw.get("version")))
    check("往返：根结构记工程名", raw.get("project") == "demo.tfanno", str(raw.get("project")))
    check("往返：根结构记更新时间", str(raw.get("updated_ns", "0")).isdigit(), str(raw.get("updated_ns")))


# ---------------------------------------------------------------------------
# 3. 增量写（只提交改过的图片）
# ---------------------------------------------------------------------------
def test_merge(tmp: Path) -> None:
    root = tmp / "merge"
    root.mkdir()
    project_path = root / "m.tfanno"
    write_project(project_path)

    annotation_drafts.save(
        project_path,
        {"keep": [make_draft(0.5)], "change": [make_draft(0.2)], "drop": [make_draft(0.1)]},
    )
    annotation_drafts.merge(project_path, {"change": [make_draft(0.95)], "drop": []})

    images, _meta = annotation_drafts.load(project_path)
    check("增量：未提交的图片保留", "keep" in images, str(sorted(images)))
    check("增量：提交的图片被覆盖", abs(float(images["change"][0]["score"]) - 0.95) < 1e-6, str(images.get("change")))
    check("增量：空数组表示删除该图记录", "drop" not in images, str(sorted(images)))
    check("增量：总数随之更新", annotation_drafts.count_drafts(images) == 2, str(annotation_drafts.count_drafts(images)))

    # 只提交一张图时，另一张必须原封不动
    before_keep = images["keep"][0]["score"]
    annotation_drafts.merge(project_path, {"change": [make_draft(0.4)]})
    images2, _ = annotation_drafts.load(project_path)
    check("增量：只改一张时另一张不变", float(images2["keep"][0]["score"]) == float(before_keep), str(images2.get("keep")))


# ---------------------------------------------------------------------------
# 4. 坏数据降级
# ---------------------------------------------------------------------------
def test_bad_data(tmp: Path) -> None:
    root = tmp / "bad"
    root.mkdir()
    project_path = root / "b.tfanno"
    write_project(project_path)
    target = annotation_drafts.drafts_path(project_path)

    # 4.1 不是合法 JSON
    target.write_text("{ not json", encoding="utf-8")
    images, meta = annotation_drafts.load(project_path)
    check("坏数据：非法 JSON 返回空草稿", images == {}, str(images))
    check("坏数据：非法 JSON 有警告", any("JSON" in text for text in meta.get("warnings", [])), str(meta))
    check("坏数据：非法 JSON 时 exists=true（文件确实在）", meta.get("exists") is True, str(meta))

    # 4.2 结构不对 / 单条草稿坏
    target.write_text(
        json.dumps(
            {
                "images": {
                    "ok": [
                        make_draft(0.6),
                        "不是对象",
                        {"shape_type": "rectangle", "points": [], "score": 0.5},
                        {"shape_type": "rectangle", "points": [[0, 0], [5, 5]], "score": 1.8},
                    ],
                    "not_a_list": {"a": 1},
                    "": [make_draft(0.3)],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    images, meta = annotation_drafts.load(project_path)
    kept = images.get("ok", [])
    check("坏数据：坏条目被跳过，好的留下", len(kept) == 2, f"{len(kept)} {kept}")
    check("坏数据：空图片 id 被跳过", "" not in images, str(sorted(images)))
    check("坏数据：非数组条目被跳过", "not_a_list" not in images, str(sorted(images)))
    check(
        "坏数据：置信度越界按未知处理",
        sorted(float(item["score"]) for item in kept) == [-1.0, 0.6],
        str([item.get("score") for item in kept]),
    )
    check("坏数据：给出越界警告", any("0~1" in text for text in meta.get("warnings", [])), str(meta.get("warnings")))

    # 4.3 根结构是数组
    target.write_text("[]", encoding="utf-8")
    images, meta = annotation_drafts.load(project_path)
    check("坏数据：根是数组按空处理", images == {}, str(images))
    check("坏数据：根是数组有警告", bool(meta.get("warnings")), str(meta))

    # 4.4 文件不存在
    target.unlink()
    images, meta = annotation_drafts.load(project_path)
    check("坏数据：文件不存在返回空且 exists=false", images == {} and meta.get("exists") is False, str(meta))


# ---------------------------------------------------------------------------
# 5. 原子写
# ---------------------------------------------------------------------------
def test_atomic(tmp: Path) -> None:
    root = tmp / "atomic"
    root.mkdir()
    project_path = root / "a.tfanno"
    write_project(project_path)
    annotation_drafts.save(project_path, {"img": [make_draft(0.77)]})

    leftovers = [p.name for p in root.iterdir() if ".tmp" in p.name]
    check("原子写：没有留下临时文件", not leftovers, str(leftovers))
    text = annotation_drafts.drafts_path(project_path).read_text(encoding="utf-8")
    check("原子写：文件是完整 JSON", json.loads(text)["count"] == 1)

    # clear 删掉旁挂文件但不动工程
    annotation_drafts.clear(project_path)
    check("原子写：clear 删除旁挂文件", not annotation_drafts.drafts_path(project_path).exists())
    check("原子写：clear 不动工程文件", project_path.is_file())
    images, meta = annotation_drafts.load(project_path)
    check("原子写：clear 后读回空草稿", images == {} and meta.get("exists") is False, str(meta))


# ---------------------------------------------------------------------------
# 6. HTTP 接口
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
    write_project(project_path)

    empty = client.get("/api/annotation/drafts", params={"path": str(project_path)})
    check("接口：没有草稿时返回 200", empty.status_code == 200, empty.text[:200])
    check("接口：没有草稿时 exists=false", empty.json().get("exists") is False, empty.text[:200])
    check("接口：没有草稿时 drafts 为空对象", empty.json().get("drafts") == {}, empty.text[:200])
    check("接口：返回旁挂文件路径", empty.json().get("drafts_path", "").endswith(".drafts.json"), empty.text[:200])

    saved = client.post(
        "/api/annotation/drafts",
        json={"path": str(project_path), "images": {"img_a": [make_draft(0.88)]}},
    )
    check("接口：写入草稿 200", saved.status_code == 200, saved.text[:200])
    check("接口：写入后 count=1", saved.json().get("count") == 1, saved.text[:200])

    second = client.post(
        "/api/annotation/drafts",
        json={"path": str(project_path), "images": {"img_b": [make_draft(0.35)]}},
    )
    check("接口：第二次写入 200", second.status_code == 200, second.text[:200])

    loaded = client.get("/api/annotation/drafts", params={"path": str(project_path)})
    body = loaded.json()
    check("接口：两次增量写都在", sorted(body.get("drafts", {})) == ["img_a", "img_b"], loaded.text[:300])
    check("接口：总数为 2", body.get("count") == 2, loaded.text[:300])

    replaced = client.post(
        "/api/annotation/drafts",
        json={"path": str(project_path), "images": {"img_c": [make_draft(0.7)]}, "replace": True},
    )
    check("接口：replace 整份替换 200", replaced.status_code == 200, replaced.text[:200])
    after = client.get("/api/annotation/drafts", params={"path": str(project_path)}).json()
    check("接口：替换后只剩新图片", sorted(after.get("drafts", {})) == ["img_c"], str(sorted(after.get("drafts", {}))))

    wrong = client.get("/api/annotation/drafts", params={"path": str(root / "not-project.json")})
    check("接口：非 .tfanno 被拒 400", wrong.status_code == 400, f"{wrong.status_code} {wrong.text[:200]}")

    blocked = client.get(
        "/api/annotation/drafts",
        params={"path": str(Path(tempfile.gettempdir()).parent / "outside" / "x.tfanno")},
    )
    check("接口：白名单外被拒 403", blocked.status_code == 403, f"{blocked.status_code} {blocked.text[:200]}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="anno-drafts-") as temp:
        tmp = Path(temp)
        test_path(tmp)
        test_roundtrip(tmp)
        test_merge(tmp)
        test_bad_data(tmp)
        test_atomic(tmp)
        test_api(tmp)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
