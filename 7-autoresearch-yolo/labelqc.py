# -*- coding: utf-8 -*-
"""数据集体检命令行：同底板分组、跨划分泄漏、标注问题。

为什么要有这个入口
------------------
网页端的「体检」按钮只能在有人看着的时候用。自动实验是无人值守跑的，如果数据集
在某次补图之后被写坏了（字段数不对的行会被解析器静默丢弃），那一整轮搜索都会建立在
错误的数据上，而台账里只会显示「这次指标低」。

用法
----
    python 7-autoresearch-yolo/labelqc.py splits/pcb_tune.yaml
    python 7-autoresearch-yolo/labelqc.py "PCB data/PCB缺陷检测" --json
    python 7-autoresearch-yolo/labelqc.py splits/pcb_tune.yaml --fail-on error

退出码：0 无阻断问题；1 达到 --fail-on 的阈值；2 参数或路径错误。

实现复用 ``6-web/backend/datasetqc.py``：与网页端同一份判据。两边各写一份的话，
迟早会出现「页面说没问题，命令行说有问题」——比任何单边错误都难查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "6-web" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    import datasetqc  # noqa: E402
except ImportError as exc:  # pragma: no cover - 只会在残缺部署里发生
    print(f"找不到 6-web/backend/datasetqc.py：{exc}", file=sys.stderr)
    raise SystemExit(2) from exc


def _print_human(report: dict, limit: int) -> None:
    counts = report.get("counts", {})
    groups = report.get("groups", {})
    summary = report.get("summary", {})
    print(f"数据源：{report.get('source')}")
    print(
        f"图片 {counts.get('images')} 张"
        f"（train {counts.get('train', 0)} / val {counts.get('val', 0)} / test {counts.get('test', 0)}）"
        f"，类别 {len(report.get('classes') or [])} 个"
    )
    print(
        f"底板分组：{groups.get('total', 0)} 组，跨划分 {groups.get('cross_split', 0)} 组"
        f"（涉及 {groups.get('leak_images', 0)} 张），逐字节重复 {report.get('exact_group_total', 0)} 组"
    )
    print(
        f"标注问题：错误 {summary.get('error', 0)} / 警告 {summary.get('warn', 0)}"
        f" / 提示 {summary.get('info', 0)}，耗时 {report.get('elapsed_ms', 0)} ms"
    )

    if groups.get("cross_split"):
        print()
        print("!! 同一底板同时出现在多个划分里：指标不能当作泛化能力。")
        print("   修复方向：数据集页把「划分口径」改成按前缀/按目录后重新导出，")
        print("   或把整组固定到同一个划分。体检本身不会改你的数据。")

    by_rule = report.get("by_rule") or {}
    if by_rule:
        print()
        print("按规则统计：")
        for code, item in by_rule.items():
            print(f"  [{item.get('severity')}] {code:18s} {item.get('count'):5d}  {item.get('title')}")

    issues = report.get("issues") or []
    if issues:
        print()
        print(f"前 {min(limit, len(issues))} 条明细：")
        for issue in issues[:limit]:
            where = issue.get("rel") or issue.get("split") or "全局"
            if issue.get("line"):
                where = f"{where}:{issue['line']}"
            raw = f"  「{issue['raw']}」" if issue.get("raw") else ""
            print(f"  [{issue.get('severity')}] {where}{raw}  {issue.get('detail')}")
        if report.get("issues_truncated"):
            print("  ……列表已截断，上面的计数是完整的")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="数据集体检：同底板分组、跨划分泄漏、标注问题（只读）"
    )
    parser.add_argument("source", help="数据集目录、data.yaml 或 .tfanno")
    parser.add_argument("--task", default="auto", help="auto / detect / segment / pose / obb / classify")
    parser.add_argument("--threshold", type=int, default=datasetqc.DEFAULT_THRESHOLD,
                        help=f"256 位 dHash 汉明距离上限，默认 {datasetqc.DEFAULT_THRESHOLD}")
    parser.add_argument("--limit", type=int, default=50, help="明细最多打印多少条，默认 50")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON（供其它脚本解析）")
    parser.add_argument(
        "--fail-on",
        choices=("none", "error", "warn"),
        default="none",
        help="达到该级别时以退出码 1 结束，默认 none（只报告）",
    )
    args = parser.parse_args(argv)

    target = Path(args.source).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.exists():
        print(f"路径不存在：{target}", file=sys.stderr)
        return 2

    try:
        report = datasetqc.inspect(target, args.task, threshold=args.threshold)
    except Exception as exc:  # noqa: BLE001 - 命令行要把异常转成可读信息
        print(f"体检失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report, args.limit)

    summary = report.get("summary", {})
    if args.fail_on == "error" and summary.get("error", 0) > 0:
        return 1
    if args.fail_on == "warn" and summary.get("error", 0) + summary.get("warn", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
