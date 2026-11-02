# -*- coding: utf-8 -*-
"""批量推理产物工具的验收脚本，不需要 GPU / DLL / torch。

守的是「产物对但不可用」这一类问题：
- CSV 少 BOM，Excel 打开中文类别名全是乱码；
- zip 里带了绝对路径，解压后散落一地；
- 下载接口只做 ``startswith`` 判断，``runtime/infer_evil`` 这样同前缀的
  隔壁目录也能被读走；
- 图片收集顺序随文件系统变化，两次跑的结果无法 diff。

用法：
    python 6-web/backend/tools/test_infer_jobs.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import inferjobs  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def make_temp(tag: str) -> Path:
    """测试目录一律放在 runtime 下。

    为什么不用系统 temp：下载接口的白名单只放仓库根，
    系统 temp 里的路径会被正确拒绝——但那样测的就不是本意了。
    """
    config.ensure_dirs()
    return Path(tempfile.mkdtemp(prefix=f"test-{tag}-", dir=str(config.RUNTIME_DIR)))


def main() -> int:
    root = make_temp("inferjobs")
    original_root = inferjobs.INFER_ROOT
    try:
        # ---- 1. collect_images --------------------------------------------
        source = root / "images"
        (source / "sub").mkdir(parents=True)
        for name in ("b.JPG", "a.jpg", "sub/c.png"):
            (source / name).write_bytes(b"x")
        (source / "notes.txt").write_text("ignore", encoding="utf-8")

        recursive = inferjobs.collect_images(source, recursive=True)
        check("递归收集 3 张图", len(recursive) == 3, str(recursive))
        check(
            "按路径排序（a -> b -> sub/c）",
            [p.name for p in recursive] == ["a.jpg", "b.JPG", "c.png"],
            str([p.name for p in recursive]),
        )
        check("非图片被忽略", all(p.name != "notes.txt" for p in recursive))

        flat = inferjobs.collect_images(source, recursive=False)
        check("非递归只取顶层 2 张", len(flat) == 2, str([p.name for p in flat]))

        limited = inferjobs.collect_images(source, recursive=True, limit=2)
        check("limit 截断生效", len(limited) == 2 and limited[0].name == "a.jpg")

        single = inferjobs.collect_images(source / "a.jpg")
        check("传单文件也能收集", len(single) == 1 and single[0].name == "a.jpg")

        try:
            inferjobs.collect_images(root / "not-exists")
            check("不存在的路径抛错", False, "没有抛异常")
        except FileNotFoundError:
            check("不存在的路径抛错", True)

        # ---- 2. 产物命名 ---------------------------------------------------
        check(
            "文件名带 5 位序号前缀",
            inferjobs.annotated_name(7, Path("abc.jpg")) == "00007_abc.jpg",
            inferjobs.annotated_name(7, Path("abc.jpg")),
        )
        check(
            "空格与括号被替换",
            inferjobs.annotated_name(1, Path("板 1(2).png")) == "00001_板_1_2.jpg",
            inferjobs.annotated_name(1, Path("板 1(2).png")),
        )
        long_name = inferjobs.annotated_name(3, Path("x" * 200 + ".jpg"))
        check("超长名被截断", len(long_name) < 80, str(len(long_name)))
        check(
            "不同序号不会重名",
            inferjobs.annotated_name(1, Path("same.jpg")) != inferjobs.annotated_name(2, Path("same.jpg")),
        )
        check("空名字有兜底", "image" in inferjobs.safe_stem(""), inferjobs.safe_stem(""))

        # ---- 3. CSV ---------------------------------------------------------
        csv_path = root / "out" / "results.csv"
        inferjobs.write_csv(
            csv_path,
            ["file", "count", "conf", "flag", "note"],
            [
                {"file": "a.jpg", "count": 3, "conf": 0.123456789, "flag": True, "note": None},
                {"file": "b.jpg", "count": 0, "conf": 1.0, "flag": False, "note": "缺标签"},
            ],
        )
        raw = csv_path.read_bytes()
        check("CSV 带 BOM（Excel 中文不乱码）", raw.startswith(b"\xef\xbb\xbf"), raw[:6].hex())
        with open(csv_path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        check("CSV 行数正确", len(rows) == 2, str(rows))
        check("None 写成空串", rows[0]["note"] == "", repr(rows[0]["note"]))
        check("浮点被收敛位数", rows[0]["conf"] == "0.123457", rows[0]["conf"])
        check("bool 写成 true/false", rows[0]["flag"] == "true" and rows[1]["flag"] == "false", str(rows))
        check("CSV 用 \\n 换行（不留空行）", b"\r\n" not in raw, raw[:80])

        # ---- 4. JSON --------------------------------------------------------
        json_path = inferjobs.write_json(root / "out" / "results.json", {"名称": "检测", "n": 1})
        text = json_path.read_text(encoding="utf-8")
        check("JSON 中文不转义", "检测" in text, text)
        check("JSON 可被读回", json.loads(text)["n"] == 1)

        # ---- 5. zip ---------------------------------------------------------
        files = [root / "out" / "results.csv", root / "out" / "results.json", root / "out" / "missing.txt"]
        zip_path = inferjobs.make_zip(root / "out" / "pack.zip", files, root)
        check("zip 生成成功", zip_path.is_file() and zip_path.stat().st_size > 0)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        check("zip 内是相对路径", all(not n.startswith("/") and ":" not in n for n in names), str(names))
        check("zip 内保留目录层级", any(n.endswith("out/results.csv") for n in names), str(names))
        check("不存在的文件被跳过", all("missing" not in n for n in names), str(names))
        check("不留 .part 中间文件", not zip_path.with_suffix(".zip.part").exists())

        # ---- 6. 下载白名单 ---------------------------------------------------
        inferjobs.INFER_ROOT = root / "infer"
        inside = root / "infer" / "job1" / "results.csv"
        inside.parent.mkdir(parents=True)
        inside.write_text("ok", encoding="utf-8")
        check("产物目录内可下载", inferjobs.resolve_artifact(str(inside)) == inside.resolve())

        sibling = root / "infer_evil" / "secret.txt"
        sibling.parent.mkdir(parents=True)
        sibling.write_text("no", encoding="utf-8")
        try:
            inferjobs.resolve_artifact(str(sibling))
            check("同前缀的隔壁目录被拒绝", False, "竟然通过了")
        except ValueError:
            check("同前缀的隔壁目录被拒绝", True)

        try:
            inferjobs.resolve_artifact("")
            check("空路径被拒绝", False, "没有报错")
        except ValueError:
            check("空路径被拒绝", True)

        try:
            inferjobs.resolve_artifact(str(root / "infer" / ".." / "infer_evil" / "secret.txt"))
            check("../ 逃逸被拒绝", False, "竟然通过了")
        except ValueError:
            check("../ 逃逸被拒绝", True)

        # ---- 7. Content-Type -------------------------------------------------
        check("mp4 的 MIME 正确", inferjobs.guess_media_type("a.mp4") == "video/mp4")
        check("csv 带 charset", "charset=utf-8" in inferjobs.guess_media_type("a.csv"))
        check("未知扩展名兜底", inferjobs.guess_media_type("a.bin") == "application/octet-stream")

        # ---- 8. new_job_dir 不重名 -------------------------------------------
        job_a = inferjobs.new_job_dir("batch")
        job_b = inferjobs.new_job_dir("batch")
        check("两次建目录不重名", job_a != job_b, f"{job_a} / {job_b}")
        check("产物目录含 annotated", (job_a / "annotated").is_dir())
    finally:
        inferjobs.INFER_ROOT = original_root
        shutil.rmtree(root, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
