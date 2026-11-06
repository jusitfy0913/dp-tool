// 日志出口与事件去重的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_log_actions.mjs
//
// 为什么这层值得测：
// 1. 去重写错的代价是「丢日志」或「日志重复两遍」，两者都比报错更难发现——
//    断线重连只发生在网络抖动时，用户很难主动复现。
// 2. 「日志文件」按钮的启用条件写错，用户点下去只会打开一个 404 页面，
//    而那一刻他往往正因为训练失败要去看日志。

import { createLogActions, copyText, runLogUrl } from "../js/logs.js";
import { isFreshEvent } from "../js/ws.js";

let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) passed += 1;
  else failures.push(message);
}

function equal(actual, expected, message) {
  assert(actual === expected, `${message}（实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}）`);
}

function group(name) {
  console.log(`\n── ${name}`);
}

// ---------------------------------------------------------------------------
group("事件去重（断线重连不能把日志补两遍）");
// ---------------------------------------------------------------------------
// 后端每次 _publish 都会打一个从 1 开始自增的 seq
assert(isFreshEvent({ seq: 1 }, 0), "第一条事件必须放行");
assert(!isFreshEvent({ seq: 3 }, 3), "已处理过的 seq 不能重复放行（重连补发的就是这些）");
assert(!isFreshEvent({ seq: 2 }, 3), "比游标更小的 seq 同样丢弃");
assert(isFreshEvent({ seq: 4 }, 3), "新的 seq 必须放行");

// 老后端没有 seq：一律放行。去重不能变成丢日志。
assert(isFreshEvent({ type: "log", line: "x" }, 99), "没有 seq 的事件放行（兼容旧后端）");
assert(isFreshEvent({ seq: "5" }, 3), "seq 不是数字时按无 seq 处理，不做隐式转换");
assert(isFreshEvent(null, 99), "空事件不抛异常");

// 模拟真实的重连：第一次连接收到 1..3，闪断后后端把 1..3 再补发一遍，再接 4..5。
// 正确的行为是「一句都不重复」，而不是「宁可重复也不丢」。
const reconnect = (() => {
  let cursor = 0;
  const received = [];
  const feed = (events) => {
    for (const event of events) {
      if (!isFreshEvent(event, cursor)) continue;
      cursor = event.seq;
      received.push(event.seq);
    }
  };
  const first = [1, 2, 3].map((seq) => ({ seq }));
  const replay = [1, 2, 3, 4, 5].map((seq) => ({ seq }));
  feed(first);
  feed(replay);
  return received;
})();
equal(reconnect.join(","), "1,2,3,4,5", "重连后事件序列连续且不重复");

// ---------------------------------------------------------------------------
group("日志文件地址");
// ---------------------------------------------------------------------------
equal(runLogUrl("abc123def456"), "/api/runs/abc123def456/log", "正常 run_id 直连");
equal(runLogUrl("../etc"), "/api/runs/..%2Fetc/log", "run_id 里的 .. 必须被编码（后端另有白名单）");

// ---------------------------------------------------------------------------
group("复制为空时不碰剪贴板");
// ---------------------------------------------------------------------------
// copyText 在文本为空时应当在接触 navigator 之前就返回 false：
// node 里没有 navigator，如果实现顺序写反了这里会直接抛 ReferenceError。
const emptyCopy = await copyText("");
equal(emptyCopy, false, "空文本直接返回 false，不抛异常");

// ---------------------------------------------------------------------------
group("按钮状态机");
// ---------------------------------------------------------------------------
function fakeButton(label) {
  return {
    textContent: label,
    disabled: true,
    title: "",
    handlers: {},
    classList: { toggle() {}, remove() {} },
    addEventListener(type, fn) {
      this.handlers[type] = fn;
    },
  };
}

// 模拟「还没开始任何任务」：两个按钮都该是灰的
{
  let runId = null;
  const copyBtn = fakeButton("复制");
  const openBtn = fakeButton("日志文件");
  const actions = createLogActions({
    copyBtn,
    openBtn,
    getText: () => "",
    getRunId: () => runId,
  });

  actions.sync();
  assert(openBtn.disabled, "没有 run 时「日志文件」不能点（点开只会是 404）");
  assert(copyBtn.disabled, "没有日志内容时「复制」不能点");

  // 任务起来后：有 run 才能打开文件；出现第一行日志才能复制
  runId = "abc123def456";
  actions.sync();
  assert(!openBtn.disabled, "有 run 后「日志文件」可点");
  assert(copyBtn.disabled, "此时还没日志，复制仍是灰的");

  actions.markText();
  assert(!copyBtn.disabled, "出现日志后「复制」可点");

  // 切到新任务：面板被清空，复制必须重新变灰（不能复制上一个任务的残留）
  actions.reset();
  assert(copyBtn.disabled, "切换任务时「复制」回到禁用");
  assert(openBtn.disabled, "reset 之后要重新 sync 才能恢复「日志文件」");

  actions.sync();
  assert(!openBtn.disabled, "sync 后「日志文件」恢复可点");
}

// 「最近一次已结束的任务」场景：state.runId 为空，但历史 run 还在磁盘上
{
  const copyBtn = fakeButton("复制");
  const openBtn = fakeButton("日志文件");
  const actions = createLogActions({
    copyBtn,
    openBtn,
    getText: () => "line",
    getRunId: () => "abc123def456",
  });
  actions.sync();
  assert(!openBtn.disabled, "刷新页面后仍能打开上一次任务的日志");
}

// ---------------------------------------------------------------------------
if (failures.length) {
  console.log(`\n✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  for (const message of failures) console.log(`  - ${message}`);
  process.exit(1);
}
console.log(`\n✓ 全部通过：${passed} 项`);
