# -*- coding: utf-8 -*-
"""后端测试脚本的公共脚手架。

## 为什么要有这个文件

`tools/` 下有 70 个 `test_*.py`，每个都抄了一遍「收集通过/失败 + 打印
`[PASS] name` + 最后打印 `通过 N / M` + 返回退出码」。抄了 70 份的后果不是
行数，而是**格式逐渐漂移**：

- 有的用 `PASSED: list[str]`，有的用 `PASSED = 0` 计数器；
- 失败项有的只记名字，有的记「名字 + 详情」；
- 汇总行有的写「通过 3 项，失败 0 项」，有的写「通过 3 / 3」；
- `check()` 参数顺序有两种互不兼容的写法。

`run_all_tests.py` 只按退出码判定，所以这些漂移不会让测试失效，但会让人读
CI 日志时要猜。这里把这一套收敛成一份实现，测试文件只保留「断言什么」。

## 三种调用姿势都支持

```python
from _harness import check, check_eq, finish            # 直接式（最多）

check("conf 去重后升序", confs == [0.0, 0.1], str(confs))
check_eq(len(rows), 3, "行数")

def main() -> int:
    ...
    return finish("数据集指纹")
```

```python
from _harness import case, run_cases                    # 装饰器式

@case("杀树会带走孙进程")
def test_kill_tree() -> None:
    check(...)          # 断言失败用 AssertionError 抛出

def main() -> int:
    run_cases(globals())
    return finish("关停杀进程树")
```

## 约定

- **`PASSED` / `FAILED` 是模块级列表**，文件名互不干扰（每个脚本独立进程）。
- `FAILED` 里只放名字（`list[str]`），保持与老脚本 `"、".join(FAILED)` 兼容；
  详情放在 `FAILED_DETAILS`。
- 退出码只看 `finish()`：有失败就 1，全过就 0。
- 这个文件**不许** import 测试对象或第三方库，只 import 标准库——
  否则「脚手架」会变成新的耦合点。
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Iterable

__all__ = [
    "PASSED",
    "FAILED",
    "FAILED_DETAILS",
    "check",
    "require",
    "check_eq",
    "case",
    "run_cases",
    "finish",
    "reset",
]

#: 通过的用例名（按执行顺序）。
PASSED: list[str] = []
#: 失败的用例名。只放名字：老脚本里有 `"、".join(FAILED)` 这种写法。
FAILED: list[str] = []
#: 失败详情（名字 → 文案），由 :func:`finish` 打印。
FAILED_DETAILS: list[tuple[str, str]] = []


def reset() -> None:
    """清空计数。只给「同一进程里跑两轮」的场景用（正常不需要）。"""
    PASSED.clear()
    FAILED.clear()
    FAILED_DETAILS.clear()


def check(name: str, ok: Any, detail: str = "") -> bool:
    """记录一条断言结果。返回 ``ok`` 的布尔值，方便继续写条件式断言。"""
    passed = bool(ok)
    if passed:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        FAILED_DETAILS.append((name, detail))
        suffix = f" {detail}" if detail else ""
        print(f"[FAIL] {name}{suffix}")
    return passed


def require(condition: Any, message: str = "") -> None:
    """断言式写法：不满足就抛 ``AssertionError``。

    存在的理由：装饰器式（``@case``）的脚本里一条用例有多条断言，
    只有「这条用例通过/失败」才有意义，中间断言直接抛异常最省事。
    它的参数顺序是 ``(条件, 说明)``，与 :func:`check` 的 ``(名字, 条件)`` 相反——
    早期两套脚本各自这么写，迁移时把顺序反过来的静默 bug 很隐蔽，
    所以这里保留两个名字而不是硬合成一个。
    """
    if not condition:
        raise AssertionError(message or "断言失败")


def check_eq(actual: Any, expected: Any, label: str) -> bool:
    """相等断言。失败时把期望值与实际值都打出来（只看 "False" 没法排查）。"""
    return check(label, actual == expected, f"期望 {expected!r}，实际 {actual!r}")


def case(name: str, detail: str = "") -> Callable[[Callable[[], None]], Callable[[], None]]:
    """把一个无参函数登记成用例；断言失败（抛异常）由 :func:`run_cases` 兜住。

    ``detail`` 只是给「失败时必须解释为什么」的一条备注，会打在第一行，
    便于在 CI 日志里判断这条挂了意味着什么。
    """

    def wrap(func: Callable[[], None]) -> Callable[[], None]:
        def runner() -> None:
            try:
                func()
            except Exception as exc:  # noqa: BLE001 - 用例内任何异常都算这条失败
                message = f"{type(exc).__name__}: {exc}"
                if detail:
                    message = f"{detail}\n{message}"
                FAILED.append(name)
                FAILED_DETAILS.append((name, message + "\n" + traceback.format_exc()))
                print(f"  [FAIL] {name}")
            else:
                PASSED.append(name)
                print(f"  [ ok ] {name}")

        runner.__name__ = func.__name__
        runner.__doc__ = func.__doc__
        return runner

    return wrap


def run_cases(namespace: dict[str, Any] | Iterable[Callable[[], None]] | None = None) -> None:
    """执行所有 ``test_*`` 函数。

    ``namespace`` 传 ``globals()`` 即可（装饰器式脚本的惯例：按函数名排序跑，
    顺序稳定，日志可比）。传可迭代对象则按给定顺序跑。
    """
    if isinstance(namespace, dict):
        funcs = [obj for name, obj in sorted(namespace.items()) if name.startswith("test_") and callable(obj)]
    elif namespace is None:
        funcs = []
    else:
        funcs = list(namespace)
    for func in funcs:
        func()


def finish(title: str = "", exit_on_success: bool = True) -> int:
    """打印汇总并返回退出码（0 = 全过，1 = 有失败）。

    ``title`` 为空时只打空行，方便老脚本的 main 里已经打过标题的情况。
    """
    total = len(PASSED) + len(FAILED)
    print()
    if title:
        print(f"== {title} ==")
    print("=" * 60)
    print(f"通过 {len(PASSED)} / {total}")

    if FAILED:
        print()
        for name in FAILED:
            print(f"[FAIL] {name}")
            for failed_name, detail in FAILED_DETAILS:
                if failed_name == name:
                    print(detail)
                    break
            print("-" * 60)
        # 失败清单单独打一行：`run_all_tests.py` 只取最后一行含「通过/FAIL」的
        # 输出做摘要，放在这里能保证 CI 摘要一定带得上失败项。
        print("失败项：" + "、".join(FAILED))
        return 1

    print("全部通过")
    return 0


if __name__ == "__main__":  # pragma: no cover - 自检
    check("真条件通过", True)
    check_eq(1 + 1, 2, "1+1")

    @case("假条件失败（预期，验证输出格式）")
    def _demo() -> None:
        check("这行应该是 FAIL", 1 == 2, "故意失败")

    run_cases(globals())
    sys.exit(finish("_harness 自检"))
