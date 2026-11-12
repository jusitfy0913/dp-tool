# -*- coding: utf-8 -*-
"""标注页「五步流程 + 图片移除」的静态接线检查（不需要 GPU，也不需要启动 Web 服务）。

为什么需要这个脚本：
    v1.35.0 重排了标注页的左右两栏——大段 DOM 被搬了位置、标题从 <h2> 换成分区、
    草稿与审核塞进 <details>。这类改动的典型事故不是"逻辑写错"，而是：
    1) 搬的时候漏了一个 id，JS 里 `getElementById` 拿到 null，功能静默失效；
    2) 复制粘贴出两个同 id 的元素，`getElementById` 只返回第一个，第二处永远是死的；
    3) 样式类名改了但 CSS 没跟上，界面上"看起来还在"，其实布局已经塌了。
    这三种都不会抛异常、也不会让后端测试变红，只能靠静态接线检查兜住。

检查项：
    1. annotate.js 里所有 `getElementById("X")` 的 id 都存在于 index.html；
    2. 标注页这段 HTML 的标签配对平衡（搬 DOM 最容易漏掉一个 </div>）；
    3. 引导条的 5 个 id 存在且被 JS 引用；
    4. annotate.js 确实从 ./annotateflow.js 引入纯逻辑，并调用了移除图片的纯函数；
    5. 图片列表渲染里有移除按钮、确认文案写明"不删磁盘文件"；
    6. style.css 有引导条 / 步骤号 / 移除按钮 / 面板 flex 高度这几块样式；
    7. 引导条在 DOM 顺序上位于三栏工作台之前（结构约定，改了就不是"顶部引导条"了）。

用法：
    python 6-web/backend/tools/test_annotate_wiring.py
"""

from __future__ import annotations

import collections
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parents[2]
HTML_PATH = WEB / "frontend" / "index.html"
CSS_PATH = WEB / "frontend" / "css" / "style.css"
JS_PATH = WEB / "frontend" / "js" / "annotate.js"
FLOW_PATH = WEB / "frontend" / "js" / "annotateflow.js"
TEST_PATH = WEB / "frontend" / "tests" / "test_annotate_flow.mjs"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def read(path: Path) -> str:
    if not path.is_file():
        print(f"[FAIL] 缺少文件：{path}")
        raise SystemExit(1)
    return path.read_text(encoding="utf-8")


VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}


class _Balance(HTMLParser):
    """极简标签配对检查：只统计非空元素的开闭是否对齐。

    不做完整的 HTML 校验（那样要引第三方库），只回答一个问题：
    "重排大段 DOM 之后，有没有漏掉一个 </div> 或 </details>"。
    这类错误浏览器会自己"猜"着修，结果是后面的面板整段落进错误的容器里，
    症状是"某块内容突然跑到别的栏"，很难靠读 diff 发现。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in VOID_TAGS:
            return
        self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"多余的 </{tag}> 于 {self.getpos()}")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        self.errors.append(f"</{tag}> 与未闭合的 {self.stack[-1][0]} 不匹配，位置 {self.getpos()}")
        while self.stack and self.stack[-1][0] != tag:
            self.stack.pop()
        if self.stack:
            self.stack.pop()


def main() -> int:
    html = read(HTML_PATH)
    css = read(CSS_PATH)
    js = read(JS_PATH)
    flow = read(FLOW_PATH)

    html_ids = re.findall(r'id="([A-Za-z0-9_-]+)"', html)
    js_ids = sorted(set(re.findall(r'getElementById\("([^"]+)"\)', js)))

    # ------------------------------------------------------------------
    # 1. JS 引用的 id 必须存在
    # ------------------------------------------------------------------
    missing = [item for item in js_ids if item not in html_ids]
    check(
        f"annotate.js 引用的 {len(js_ids)} 个元素 id 都存在于 index.html",
        not missing,
        f"缺失：{missing}",
    )

    # ------------------------------------------------------------------
    # 2. id 唯一
    #     只对标注页这一段做硬性检查：index.html 是七个页签拼在一个文件里，
    #     其它页签的历史重复 id（如 btnExport / dsWarnings）不在本次改动范围，
    #     改它们会动到别的页面的行为，所以单独作为警告打印出来。
    # ------------------------------------------------------------------
    panel_start = html.find('id="panel-annotate"')
    panel_end = html.find('id="panel-autoexp"', panel_start)
    # 从包含它的 <section ...> 开始切，切到下一个 <section 之前：
    # 保证片段以完整标签开始/结束，标签配对检查才有意义
    panel_begin = html.rfind("<section", 0, panel_start)
    panel_stop = html.find("<section", panel_end) if panel_end > 0 else len(html)
    panel_html = html[panel_begin:panel_stop] if panel_begin >= 0 and panel_stop > panel_begin else ""
    panel_ids = re.findall(r'id="([A-Za-z0-9_-]+)"', panel_html)
    panel_dup = [key for key, count in collections.Counter(panel_ids).items() if count > 1]
    check(
        f"标注页 {len(panel_ids)} 个 id 无重复",
        bool(panel_html) and not panel_dup,
        f"重复：{panel_dup}",
    )
    # 标签配对：只查标注页这一段，避免其它页签的历史问题干扰
    balance = _Balance()
    balance.feed(panel_html)
    check(
        "标注页 HTML 标签配对平衡（无漏写的 </div> / </details>）",
        not balance.errors and not balance.stack,
        f"错误：{balance.errors[:3]} 未闭合：{[item[0] for item in balance.stack[:3]]}",
    )
    outside_dup = [key for key, count in collections.Counter(html_ids).items() if count > 1]
    if outside_dup:
        print(f"[WARN] index.html 其它页签存在历史重复 id（不在本次范围）：{outside_dup}")

    # ------------------------------------------------------------------
    # 3. 引导条的 id 存在且被 JS 引用
    # ------------------------------------------------------------------
    guide_ids = ["annoGuide", "annoSteps", "annoGuideTip", "annoGuideAction", "annoGuideToggle"]
    for item in guide_ids:
        check(f"引导条元素 {item} 存在于 index.html", item in html_ids)
        check(f"引导条元素 {item} 被 annotate.js 使用", item in js)

    # 进阶折叠块与它的计数徽标
    for item in ["annoAdvDraft", "annoDraftCount", "annoReviewCount", "annoImageList", "annoAddFiles"]:
        check(f"关键元素 {item} 仍存在（重排未漏搬）", item in html_ids)

    # ------------------------------------------------------------------
    # 4. 纯逻辑模块的引入与调用
    # ------------------------------------------------------------------
    check(
        "annotate.js 从 ./annotateflow.js 引入引导逻辑",
        'from "./annotateflow.js"' in js,
    )
    for symbol in ["buildAnnoSteps", "currentStep", "resolveGuideAction", "removeImageAt", "shouldExpandDrafts",
                   "displayName", "labelAfterClassChange", "planSplitChange"]:
        check(f"annotate.js 使用 {symbol}", symbol in js)
        check(f"annotateflow.js 导出 {symbol}", f"export function {symbol}" in flow)
    check(
        "annotateflow.js 不碰 DOM（无 document / getElementById）",
        "document." not in flow and "getElementById" not in flow,
    )
    check("前端单测脚本存在（引导判定 / 图片移除）", TEST_PATH.is_file(), str(TEST_PATH))

    # ------------------------------------------------------------------
    # 5. 图片移除的接线
    # ------------------------------------------------------------------
    check("图片列表行内有移除按钮 class anno-remove", "anno-remove" in js)
    check("移除按钮点击后调用 removeImage(index)", "removeImage(index)" in js)
    check("移除操作有 async function removeImage", "async function removeImage" in js)
    check("移除确认文案写明不删磁盘文件", "磁盘上的图片文件不会被删除" in js)
    check("移除后清理旁挂草稿（空数组 = 删除）", "removed.id]: []" in js)
    check("移除以整工程快照进撤销栈", "pushHistory(true)" in js)
    check("空图片列表有占位提示", 'className = "empty"' in js)

    # ------------------------------------------------------------------
    # 5b. 类别显示名同步与集合入口（v1.35.1）
    #     这两条都是"界面上看不出来"的接线：CSS/HTML 都在，逻辑没接就完全没反应。
    # ------------------------------------------------------------------
    check("改类别时同步显示名（labelAfterClassChange）", "labelAfterClassChange(obj.label" in js)
    check("画布/表格统一用 displayName 显示", "displayName(obj.label" in js)
    check("「标签」已改名为「显示名」", 'field("显示名"' in js and 'field("标签"' not in js)
    check("属性面板顶部固定图片卡片 anno-imagecard", 'className = "anno-imagecard"' in js)
    check("图片卡片里的集合下拉 id 为 annoImageSplit", 'splitSelect.id = "annoImageSplit"' in js)
    check("批量改集合调用 planSplitChange", "planSplitChange(" in js)
    check("批量条按选中数量显隐", "state.imageSelection.size" in js and "updateBatchBar" in js)
    for item in ["annoBatchBar", "annoBatchCount", "annoBatchTrain", "annoBatchVal", "annoBatchTest", "annoBatchClear"]:
        check(f"批量条元素 {item} 存在于 index.html", item in html_ids)
    check("多选不靠右键菜单（Ctrl / Shift 修饰键）", "event.ctrlKey" in js and "event.shiftKey" in js)

    # ------------------------------------------------------------------
    # 6. 样式接线
    # ------------------------------------------------------------------
    for selector in [
        "#panel-annotate.active",
        ".anno-guide",
        ".anno-steps",
        ".anno-step",
        ".anno-step-no",
        ".anno-remove",
        ".anno-adv",
        ".anno-help",
        ".anno-group-label",
        ".anno-actions",
        ".anno-batchbar",
        ".anno-imagecard",
        ".anno-imagelist li.picked",
    ]:
        check(f"style.css 定义 {selector}", selector in css)
    check("引导条收起态样式存在", ".anno-guide.is-collapsed" in css)
    check("窄屏下标注页恢复自然高度", "#panel-annotate.active" in css and "height: auto" in css)

    # ------------------------------------------------------------------
    # 7. 结构约定：引导条在三栏工作台之前
    # ------------------------------------------------------------------
    guide_pos = html.find('id="annoGuide"', panel_start)
    layout_pos = html.find('class="anno-layout"', panel_start)
    check("引导条位于 #panel-annotate 内", panel_start >= 0 and guide_pos > panel_start)
    check("引导条在 .anno-layout 之前（顶部引导条结构）", 0 < guide_pos < layout_pos)
    check("三栏面板类名保留", all(cls in html for cls in ["anno-left", "anno-center", "anno-right"]))

    # 步骤号与引导条数字必须一致：1 工程 / 2 图片 / 3 类别 / 4 标注框 / 5 导出
    for number in range(1, 6):
        check(f"侧栏有第 {number} 步的步骤号", f'<span class="anno-step-no">{number}</span>' in html)

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    total = len(PASSED) + len(FAILED)
    print(f"\n通过 {len(PASSED)} / {total} 项")
    if FAILED:
        print("失败项：")
        for item in FAILED:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
