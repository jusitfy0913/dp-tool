// 撤销/重做双栈的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_history.mjs
//
// 为什么值得单测：
// - 深度上限写错会让第 51 步撤销"跳着回退"，而且看起来像丢标注；
// - push 忘记清空 redo，用户撤两步再画一笔，Ctrl+Y 会回到一条不存在的分支；
// - 纯函数必须**不改入参**，否则调用方拿着一半新一半旧的栈，按钮状态就对不上。

import {
  HISTORY_LIMIT,
  canRedo,
  canUndo,
  historyButtons,
  pushSnapshot,
  redoStep,
  undoStep,
} from "../js/annotate/history.js";

// ---------------------------------------------------------------------------
// 极简断言框架（与 test_annotate_flow.mjs 保持一致）
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function check(name, actual, expected) {
  const ok = Object.is(actual, expected);
  if (ok) {
    passed += 1;
  } else {
    failures.push(`${name}\n    期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
  }
  console.log(`${ok ? "✓" : "✗"} ${name}`);
}

function checkDeep(name, actual, expected) {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a === b) {
    passed += 1;
    console.log(`✓ ${name}`);
  } else {
    failures.push(`${name}\n    期望 ${b}，实际 ${a}`);
    console.log(`✗ ${name}`);
  }
}

const entry = (name) => ({ name });

// ---------------------------------------------------------------------------
// 1. pushSnapshot
// ---------------------------------------------------------------------------
check("历史深度是 50", HISTORY_LIMIT, 50);

const pushed = pushSnapshot({ undo: [], entry: entry("a") });
checkDeep("空栈压入一条", pushed.undo, [entry("a")]);
checkDeep("压入会清空 redo", pushed.redo, []);
check("正常压入不丢条目", pushed.dropped, 0);

const atLimit = Array.from({ length: HISTORY_LIMIT }, (_, i) => entry(`e${i}`));
const overflow = pushSnapshot({ undo: atLimit, entry: entry("new") });
check("超深压入丢一条", overflow.dropped, 1);
check("深度始终卡在上限", overflow.undo.length, HISTORY_LIMIT);
checkDeep("丢的是最老的一条", overflow.undo[0], entry("e1"));
checkDeep("新条目在栈顶", overflow.undo[overflow.undo.length - 1], entry("new"));
check("溢出不改入参", atLimit.length, HISTORY_LIMIT);

const withRedo = pushSnapshot({ undo: [entry("a")], entry: entry("b") });
checkDeep("新改动让重做分支失效", withRedo.redo, []);

const nullEntry = pushSnapshot({ undo: [entry("a")], entry: null });
checkDeep("没有工程时原样返回", nullEntry.undo, [entry("a")]);
check("没有工程时不算丢弃", nullEntry.dropped, 0);

const zeroLimit = pushSnapshot({ undo: [entry("a")], entry: entry("b"), limit: 0 });
checkDeep("上限为 0 时栈永远是空的", zeroLimit.undo, []);
check("上限为 0 时计入丢弃", zeroLimit.dropped, 2);

const negativeLimit = pushSnapshot({ undo: [], entry: entry("a"), limit: -3 });
checkDeep("负数上限按 0 处理", negativeLimit.undo, []);

// ---------------------------------------------------------------------------
// 2. undo / redo 往返
// ---------------------------------------------------------------------------
const r1 = undoStep({ undo: [entry("a"), entry("b")], redo: [], current: entry("c") });
check("撤销成功", r1.ok, true);
checkDeep("撤销弹出栈顶", r1.entry, entry("b"));
checkDeep("撤销后 undo 只剩一条", r1.undo, [entry("a")]);
checkDeep("当前状态进入 redo", r1.redo, [entry("c")]);

const r2 = redoStep({ undo: r1.undo, redo: r1.redo, current: entry("d") });
check("重做成功", r2.ok, true);
checkDeep("重做取回快照", r2.entry, entry("c"));
checkDeep("重做后 undo 补回两条", r2.undo, [entry("a"), entry("d")]);
checkDeep("重做后 redo 清空", r2.redo, []);

const emptyUndo = undoStep({ undo: [], redo: [], current: entry("x") });
check("空栈撤销返回 false", emptyUndo.ok, false);
checkDeep("空栈撤销不动 redo", emptyUndo.redo, []);
check("空栈撤销没有条目", emptyUndo.entry, null);

const emptyRedo = redoStep({ undo: [entry("a")], redo: [], current: entry("x") });
check("空栈重做返回 false", emptyRedo.ok, false);
checkDeep("空栈重做不动 undo", emptyRedo.undo, [entry("a")]);

const noCurrent = undoStep({ undo: [entry("a")], redo: [] });
check("没给当前快照也能撤销", noCurrent.ok, true);
checkDeep("没给当前快照就不往 redo 塞 null", noCurrent.redo, []);

// 连续往返 30 次后栈状态必须回到原点（防止原地 push 导致的漂移）
let stack = { undo: [entry("a"), entry("b")], redo: [] };
for (let i = 0; i < 30; i += 1) {
  const down = undoStep({ ...stack, current: entry(`c${i}`) });
  stack = { undo: down.undo, redo: down.redo };
  const up = redoStep({ ...stack, current: entry(`d${i}`) });
  stack = { undo: up.undo, redo: up.redo };
}
check("往返 30 次后 undo 深度不变", stack.undo.length, 2);
// 重做会把"重做前的当前状态"压回 undo 栈顶——这正是 Ctrl+Y 之后能再 Ctrl+Z 的原因，
// 所以栈顶是最后一轮传进去的 current，而不是最初的 b。
checkDeep("往返 30 次后栈顶是最后一轮的当前状态", stack.undo[1], entry("d29"));
checkDeep("往返 30 次后栈底还是最早的一条", stack.undo[0], entry("a"));
check("往返 30 次后 redo 为空", stack.redo.length, 0);

// 深拷贝语义：返回的数组是新数组，改它不影响入参
const originalUndo = [entry("a")];
const detached = pushSnapshot({ undo: originalUndo, entry: entry("b") });
detached.undo.push(entry("c"));
check("返回值是副本，不污染入参", originalUndo.length, 1);

// ---------------------------------------------------------------------------
// 3. 按钮状态
// ---------------------------------------------------------------------------
check("空栈不能撤销", canUndo([]), false);
check("有栈能撤销", canUndo([entry("a")]), true);
check("空栈不能重做", canRedo([]), false);
checkDeep("按钮状态取反", historyButtons([entry("a")], []), { undoDisabled: false, redoDisabled: true });
checkDeep("双空栈两个按钮都禁用", historyButtons([], []), { undoDisabled: true, redoDisabled: true });

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log("");
if (failures.length) {
  console.error(`✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  failures.forEach((text) => console.error(`\n${text}`));
  process.exit(1);
}
console.log(`✓ 全部通过（${passed} 项）`);
