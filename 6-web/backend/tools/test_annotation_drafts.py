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


# ---------------------------------------------------------------------------
# 7. 审核状态（v2）
# ---------------------------------------------------------------------------
def test_review(tmp: Path) -> None:
    root = tmp / "review"
    root.mkdir()
    project_path = root / "r.tfanno"
    write_project(project_path)
    target = annotation_drafts.drafts_path(project_path)

    # 7.1 v1 文件（没有 review 块）能读，审核状态视为全部未审核
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "images": {"img_a": [make_draft(0.7)]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    images, meta = annotation_drafts.load(project_path)
    check("v1→v2：旧文件能读出草稿", "img_a" in images, str(sorted(images)))
    check("v1→v2：旧文件审核为空", meta.get("review") == {}, str(meta.get("review")))
    check("v1→v2：旧文件 reviewed=0", meta.get("reviewed") == 0, str(meta.get("reviewed")))

    # 7.2 标记已审核：写在同一个文件里，草稿不丢
    annotation_drafts.mark_reviewed(project_path, "img_a")
    images, meta = annotation_drafts.load(project_path)
    check("审核：标为已审核后 review 里有该图", "img_a" in (meta.get("review") or {}), str(meta.get("review")))
    check("审核：reviewed 计数跟上", meta.get("reviewed") == 1, str(meta.get("reviewed")))
    check("审核：草稿没有被顺手清掉", "img_a" in images, str(sorted(images)))
    raw = json.loads(target.read_text(encoding="utf-8"))
    check("审核：文件版本升到 v2", raw.get("version") == 2, str(raw.get("version")))
    check("审核：根结构里有 review 块", isinstance(raw.get("review"), dict), str(type(raw.get("review"))))
    check(
        "审核：记录带时间戳",
        str((raw.get("review") or {}).get("img_a", {}).get("at_ns", "")).isdigit(),
        str((raw.get("review") or {}).get("img_a")),
    )

    # 7.3 增量合并：review 只覆盖传入的图片 id
    annotation_drafts.merge(project_path, review={"img_b": {"state": "checked"}})
    _images, meta = annotation_drafts.load(project_path)
    check("审核：增量合并保留原有条目", {"img_a", "img_b"} <= set(meta.get("review") or {}), str(meta.get("review")))

    # 7.4 unchecked 表示删除
    annotation_drafts.unmark_reviewed(project_path, "img_a")
    _images, meta = annotation_drafts.load(project_path)
    check("审核：unchecked 会删掉该条目", "img_a" not in (meta.get("review") or {}), str(meta.get("review")))
    check("审核：另一个条目还在", "img_b" in (meta.get("review") or {}), str(meta.get("review")))

    # 7.5 只传 review 时草稿不动（K 键的专用路径）
    annotation_drafts.mark_reviewed(project_path, "img_a")
    annotation_drafts.merge(project_path, review={"img_a": {"state": "checked"}})
    images, _meta = annotation_drafts.load(project_path)
    check("审核：只提交 review 不会清空草稿", "img_a" in images, str(sorted(images)))

    # 7.6 clear 默认保留审核
    annotation_drafts.clear(project_path)
    images, meta = annotation_drafts.load(project_path)
    check("审核：clear 之后草稿清空", images == {}, str(sorted(images)))
    check("审核：clear 之后审核保留", "img_a" in (meta.get("review") or {}), str(meta.get("review")))
    check("审核：clear 之后 reviewed 保持（img_a + img_b）", meta.get("reviewed") == 2, str(meta.get("reviewed")))

    # 7.7 clear(replace) 才连审核一起清
    annotation_drafts.clear(project_path, include_review=True)
    _images, meta = annotation_drafts.load(project_path)
    check("审核：include_review 时审核一起清", (meta.get("review") or {}) == {}, str(meta.get("review")))
    check("审核：include_review 时文件被删", not target.exists())


def test_review_bad_data(tmp: Path) -> None:
    root = tmp / "review-bad"
    root.mkdir()
    project_path = root / "rb.tfanno"
    write_project(project_path)
    target = annotation_drafts.drafts_path(project_path)

    target.write_text(
        json.dumps(
            {
                "version": 2,
                "images": {"img": [make_draft(0.5)]},
                "review": {
                    "good": {"state": "checked", "at_ns": "1700000000000000000"},
                    "string_state": "checked",
                    "unknown_state": {"state": "wat"},
                    "not_object": 3,
                    "": {"state": "checked"},
                    "cancelled": {"state": "unchecked"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _images, meta = annotation_drafts.load(project_path)
    review = meta.get("review") or {}
    check("坏审核：正常条目保留", "good" in review, str(review))
    check("坏审核：字符串写法被接受", "string_state" in review, str(review))
    check("坏审核：不认识的状态被跳过", "unknown_state" not in review, str(review))
    check("坏审核：非对象条目被跳过", "not_object" not in review, str(review))
    check("坏审核：空图片 id 被跳过", "" not in review, str(review))
    check("坏审核：unchecked 读时视为没有", "cancelled" not in review, str(review))
    check("坏审核：有警告", any("审核" in text for text in meta.get("warnings", [])), str(meta.get("warnings")))

    # review 整体不是对象
    target.write_text(json.dumps({"images": {}, "review": [1, 2]}), encoding="utf-8")
    _images, meta = annotation_drafts.load(project_path)
    check("坏审核：review 是数组时按空处理", (meta.get("review") or {}) == {}, str(meta.get("review")))

    # 非法状态值要报错而不是静默
    try:
        annotation_drafts.set_review(project_path, "x", "whatever")
        check("坏审核：非法状态被拒绝", False, "没有抛错")
    except ValueError:
        check("坏审核：非法状态被拒绝", True)
    try:
        annotation_drafts.set_review(project_path, "", "checked")
        check("坏审核：空图片 id 被拒绝", False, "没有抛错")
    except ValueError:
        check("坏审核：空图片 id 被拒绝", True)


def test_review_api(tmp: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        check(f"审核接口：跳过（未安装 httpx：{exc}）", True)
        return

    import app as web_app

    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(web_app.config.REPO_ROOT)
    client = TestClient(web_app.app)

    root = tmp / "review-api"
    root.mkdir()
    project_path = root / "ra.tfanno"
    write_project(project_path)

    marked = client.post(
        "/api/annotation/drafts/review",
        json={"path": str(project_path), "review": {"img_a": {"state": "checked"}}},
    )
    check("审核接口：标记 200", marked.status_code == 200, marked.text[:200])
    check("审核接口：返回 reviewed=1", marked.json().get("reviewed") == 1, marked.text[:200])

    loaded = client.get("/api/annotation/drafts", params={"path": str(project_path)}).json()
    check("审核接口：读回 review", "img_a" in (loaded.get("review") or {}), loaded.get("review"))

    empty_review = client.post(
        "/api/annotation/drafts/review",
        json={"path": str(project_path), "review": {}},
    )
    check("审核接口：空 review 被拒 400", empty_review.status_code == 400, empty_review.text[:200])

    # 通过草稿接口也能写 review（一次请求同时写草稿与状态）
    both = client.post(
        "/api/annotation/drafts",
        json={
            "path": str(project_path),
            "images": {"img_b": [make_draft(0.42)]},
            "review": {"img_b": {"state": "checked"}},
        },
    )
    check("审核接口：草稿与审核一次写入 200", both.status_code == 200, both.text[:200])
    body = both.json()
    check("审核接口：一次写入后 count=1", body.get("count") == 1, both.text[:200])
    check("审核接口：一次写入后 reviewed=2", body.get("reviewed") == 2, both.text[:200])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="anno-drafts-") as temp:
        tmp = Path(temp)
        test_path(tmp)
        test_roundtrip(tmp)
        test_merge(tmp)
        test_bad_data(tmp)
        test_atomic(tmp)
        test_api(tmp)
        test_review(tmp)
        test_review_bad_data(tmp)
        test_review_api(tmp)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
