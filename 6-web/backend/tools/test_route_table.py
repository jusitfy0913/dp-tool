# -*- coding: utf-8 -*-
"""路由快照测试：对外接口清单必须与 `tools/route_table.json` 逐条一致。

    python tools/test_route_table.py            # 对比
    python tools/test_route_table.py --update   # 重建快照（改了接口才用）

这是 `app.py` 按域拆分唯一的护栏。拆模块时最容易犯的三个错都能被它挡住：

1. 漏写 `app.include_router(...)` —— 一整域接口消失；
2. 路径写错（多一个斜杠、复数变单数）—— 前端 404，但没人会立刻发现；
3. 方法写错（GET 写成 POST）—— 页面点了没反应。

它不是「差不多就行」的检查：缺失和新增都算失败。确实要改接口时，
用 `--update` 重建快照，并在提交信息里写明为什么改。
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))

import route_table  # noqa: E402


def main(argv: list[str]) -> int:
    actual = route_table.collect()
    if "--update" in argv:
        route_table.update(actual)
        return 0

    if not route_table.SNAPSHOT.is_file():
        print(f"缺少快照 {route_table.SNAPSHOT}，先执行 --update")
        return 1

    expected = route_table.load()
    missing, extra = route_table.diff(expected, actual)
    if not missing and not extra:
        print(f"路由表与快照一致：{len(actual)} 条")
        return 0

    print(f"路由表与快照不一致（快照 {len(expected)} 条，运行时 {len(actual)} 条）")
    for item in missing:
        print(f"  [缺失] {item}   快照里有，运行时没有")
    for item in extra:
        print(f"  [新增] {item}   运行时有，快照里没有")
    print("\n拆模块时出现上面任何一条都要先修好；确实是新增接口才用 --update。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
