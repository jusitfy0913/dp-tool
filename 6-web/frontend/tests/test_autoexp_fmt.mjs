// 自动实验页「决策块」格式化的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_autoexp_fmt.mjs
//
// 为什么这层值得单独测：
// 「模型提议了什么 / 这一轮算不算数」全靠日志里那几行文字表达。
// 一旦格式写错，用户看到的可能是：
//   - 判定成 KEEP 却显示「—」（差值丢了，看不出赢了多少）；
//   - DISCARD 被染成红色（一整晚几十轮全红，真错误反而被淹没）；
//   - 轮数换算错一倍（用户以为 30 轮要跑 11 小时，其实 5.5 小时）。
// 这些都是「看起来没错」的错误，只能靠断言兜住。

import { estimateBudget, formatDelta, verdictLine } from "../js/autoexp.js";

let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) passed += 1;
  else failures.push(message);
}

function equal(actual, expected, message) {
  assert(actual === expected, `${message}（实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}）`);
}

function contains(text, needle, message) {
  assert(String(text).includes(needle), `${message}（实际 ${JSON.stringify(text)}，应含 ${needle}）`);
}

function group(name) {
  console.log(`\n── ${name}`);
}

// ---------------------------------------------------------------------------
group("差值格式");
// ---------------------------------------------------------------------------
equal(formatDelta(0.0037), "+0.003700", "正值带 +，保留 6 位");
equal(formatDelta(-0.0164), "-0.016400", "负值保留符号");
equal(formatDelta(0), "+0.000000", "0 也算正向（不能显示 -0）");
equal(formatDelta("0.052"), "+0.052000", "字符串输入按数字解析");
equal(formatDelta(null), "—", "缺失显示破折号");
equal(formatDelta(""), "—", "空串显示破折号");
equal(formatDelta("abc"), "—", "非数字不抛异常");

// ---------------------------------------------------------------------------
group("判定行");
// ---------------------------------------------------------------------------
const keep = verdictLine({ verdict: "KEEP", exp_id: "exp_0012", delta: 0.052 });
contains(keep.text, "KEEP", "KEEP 字样");
contains(keep.text, "exp_0012", "带实验编号");
contains(keep.text, "+0.052000", "带差值");
equal(keep.cls, "ae-keep", "KEEP 用绿色样式");
equal(keep.tone, "ok", "KEEP 让状态徽章变成功色");

const discard = verdictLine({ verdict: "DISCARD", exp_id: "exp_0013", delta: 0.0031 });
contains(discard.text, "DISCARD", "DISCARD 字样");
equal(discard.cls, "ae-discard", "DISCARD 用次要色");
equal(discard.tone, "", "DISCARD 不染成错误色——它是正常的科研结果，不是故障");

const broken = verdictLine({});
contains(broken.text, "DISCARD", "缺字段时默认按 DISCARD（宁可保守）");
contains(broken.text, "—", "缺差值显示破折号");
equal(broken.cls, "ae-discard", "缺字段不抛异常");

const noDelta = verdictLine({ verdict: "KEEP", exp_id: "exp_0001" });
contains(noDelta.text, "—", "首轮没有基准时差值显示破折号");

// ---------------------------------------------------------------------------
group("轮数 → 时长估算");
// ---------------------------------------------------------------------------
// 默认 30 轮：一轮 600 秒训练 + 约 60~90 秒开销 ≈ 11 分钟
contains(estimateBudget(30, 8), "5.5 小时", "30 轮约 5.5 小时");
contains(estimateBudget(30, 8), "8 小时会强制停", "应提示时间上限");
contains(estimateBudget(10, 8), "1.8 小时", "10 轮约 1.8 小时");
contains(estimateBudget(50, 0), "9.2 小时", "50 轮约 9.2 小时");
contains(estimateBudget(50, 0), "不限时间", "0 小时表示不限");
contains(estimateBudget(0, 8), "0 分钟", "0 轮不能算出负数或 NaN");
contains(estimateBudget("30", "8"), "5.5 小时", "字符串输入也要能算");
assert(!estimateBudget(30, 8).includes("NaN"), "不能出现 NaN");

console.log();
if (failures.length) {
  for (const message of failures) console.log(`✗ ${message}`);
  console.log(`\n✗ 失败 ${failures.length} 项，通过 ${passed} 项`);
  process.exit(1);
}
console.log(`✓ 全部通过：${passed} 项`);
