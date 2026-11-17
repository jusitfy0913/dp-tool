# -*- coding: utf-8 -*-
"""前端页面结构测试：标签配平 / id 唯一 / 主题脚本位置 / 表单分区。

    python tools/test_frontend_html.py

为什么要它：
    这套前端是零构建的纯静态页，没有模板引擎也没有打包器，改一个区块只能靠手写缩进。
    最常见的三种坏法都不会让 Python 报错，也不会让页面变白，只会悄悄坏掉：

    1. **多一个 / 少一个 `</div>`** —— 浏览器会自动"猜"补在哪，于是整块内容
       被塞进上一个卡片里，看起来像布局错乱，很难反查到源头；
    2. **id 重复** —— `getElementById` 只返回第一个。数据集页的「开始导出」
       曾经和训练页的「导出 ONNX」同名 `btnExport`，结果数据集导出时禁用按钮，
       禁用的是另一页那个（v1.43 修），页面上表现为"点了没反应"；
    3. **theme.js 位置错了** —— 它必须同步写在 `<head>` 里。放到 `<body>` 末尾或
       改成 `type="module"`（defer 语义），深色主题用户每次开页都会先闪一帧白底。

检查是纯文本静态扫描，不启浏览器、不装依赖。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
FRONTEND = TOOLS.parents[1] / "frontend"

#: 自闭合元素：没有结束标签，也不进栈
VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
#: 允许省略结束标签的元素（浏览器会自动补），不当作未闭合报错
OPTIONAL_CLOSE = {"li", "p", "option", "tr", "td", "th", "thead", "tbody"}


class TagBalance(HTMLParser):
    """只做两件事：标签配平、收集 id。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        line = self.getpos()[0]
        if not self.stack:
            self.problems.append(f"第 {line} 行多出一个 </{tag}>")
            return
        # 从栈顶往下找同名开标签：中间那些就是"忘了闭合"的
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] != tag:
                continue
            for name, open_line in self.stack[index + 1:]:
                if name not in OPTIONAL_CLOSE:
                    self.problems.append(
                        f"<{name}>（第 {open_line} 行开）在 </{tag}>（第 {line} 行）之前没有闭合"
                    )
            del self.stack[index:]
            return
        self.problems.append(f"第 {line} 行 </{tag}> 没有对应的开标签")


def check_page(path: Path) -> list[str]:
    checker = TagBalance()
    checker.feed(path.read_text(encoding="utf-8"))
    for tag, line in checker.stack:
        if tag not in OPTIONAL_CLOSE:
            checker.problems.append(f"<{tag}>（第 {line} 行）直到文件末尾都没闭合")

    dupes = [name for name, count in Counter(checker.ids).items() if count > 1]
    if dupes:
        # 报出重复的 id，并指出各自行号，方便直接跳过去改名
        text = path.read_text(encoding="utf-8")
        detail = []
        for name in dupes:
            lines = [
                str(i + 1)
                for i, line in enumerate(text.splitlines())
                if re.search(rf'id="{re.escape(name)}"', line)
            ]
            detail.append(f"{name}（第 {'、'.join(lines)} 行）")
        checker.problems.append(f"id 重复：{'；'.join(detail)}")
    return checker.problems


def main() -> int:
    failures: list[str] = []
    passed = 0

    for name in ("index.html", "login.html"):
        path = FRONTEND / name
        if not path.is_file():
            failures.append(f"{name} 不存在")
            continue
        problems = check_page(path)
        if problems:
            failures.extend(f"{name}：{problem}" for problem in problems)
        else:
            passed += 1
        print(f"{'✓' if not problems else '✗'} {name} 标签配平 + id 唯一")

        # theme.js 必须同步出现在 </head> 之前：位置错了会白闪一帧
        text = path.read_text(encoding="utf-8")
        at = text.find("/static/js/theme.js")
        head = text.find("</head>")
        body = text.find("<body")
        ok = 0 <= at < head and at < body and 'type="module"' not in text[max(0, at - 120):at]
        if ok:
            passed += 1
        else:
            failures.append(f"{name}：theme.js 必须在 <body> 之前的 <head> 内同步加载")
        print(f"{'✓' if ok else '✗'} {name} theme.js 在 head 内同步加载（防白闪）")

    # 训练页 / 引擎页的分区标题：这些 h3 是用户跳读的入口，被误删会让长表单又变回"一大坨"
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    required = ["任务与运行环境", "模型与数据", "超参数", "输出与动作", "模型与输入", "精度与产物"]
    missing = [title for title in required if f"<h3>{title}</h3>" not in index]
    if missing:
        failures.append(f"表单分区标题缺失：{', '.join(missing)}")
    else:
        passed += 1
    print(f"{'✓' if not missing else '✗'} 表单分区标题齐全（{len(required)} 个）")

    # 主题切换入口：顶栏一颗，登录遮罩上一颗。两套图标都要有（显隐交给 CSS）
    toggle_ok = (
        'id="themeToggle"' in index
        and 'id="bootThemeToggle"' in index
        and index.count('class="icon-sun"') == 2
        and index.count('class="icon-moon"') == 2
    )
    if toggle_ok:
        passed += 1
    else:
        failures.append("主题按钮应有顶栏 #themeToggle 和登录遮罩 #bootThemeToggle，各带太阳/月亮图标")
    print(f"{'✓' if toggle_ok else '✗'} 顶栏与登录遮罩的主题切换按钮")

    boot_ids = ("bootGate", "bootPassword", "bootSubmit", "bootWait", "bootEnter", "operatorCard", "operatorSave")
    boot_missing = [name for name in boot_ids if f'id="{name}"' not in index]
    # 遮罩必须写在顶栏之前，首屏才不会先闪出表单
    boot_first = index.find('id="bootGate"')
    chrome_at = index.find('class="chrome"')
    about_at = index.find('id="panel-about"')
    card_at = index.find('id="operatorCard"')
    main_end = index.find("</main>")
    boot_ok = (
        not boot_missing
        and 0 <= boot_first < chrome_at
        and 0 <= about_at < card_at < main_end
    )
    if boot_ok:
        passed += 1
    else:
        failures.append(
            "登录遮罩缺失或排在顶栏后面："
            + (", ".join(boot_missing) if boot_missing else "bootGate 应在 .chrome 之前")
        )
    print(f"{'✓' if boot_ok else '✗'} 登录遮罩在顶栏之前，进入密码卡片在环境说明页")

    print("")
    if failures:
        print(f"✗ 失败 {len(failures)} 项（通过 {passed} 项）：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"✓ 全部通过：{passed} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
