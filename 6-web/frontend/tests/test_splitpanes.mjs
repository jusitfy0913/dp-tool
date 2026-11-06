// 分栏拖动的比例换算单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_splitpanes.mjs
//
// 只测纯函数（clampRatio / readStoredRatio）。真正的拖动、双击复位、写入
// localStorage 走浏览器实测。之所以单测这两个函数，是因为本次故障的全部
// 数据都在这里：拖到底存了 15.61%，换个窗口宽度后它只剩 240px，表单里的
// 输入框全都要横向滚动，而画面上看起来"只是有点挤"，很难联想到是存档值。

import { clampRatio, readStoredRatio } from "../js/splitpanes.js";

// ---------------------------------------------------------------------------
// 极简断言框架
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

function group(title) {
  console.log(`\n── ${title}`);
}

// ---------------------------------------------------------------------------
// readStoredRatio：存下来的记录能不能用
// ---------------------------------------------------------------------------
group("存档比例：能用的保留");
// 1600 宽的窗口拖到 33.83%，换回同一个宽度还是 519px，应该原样恢复。
check("正常比例原样恢复", readStoredRatio("33.83%", { width: 1537, minPx: 340 }), 33.83);
check("刚好等于下限可用", readStoredRatio("22.12%", { width: 1537, minPx: 340 }), 22.12);
check("表头带不带百分号都认", readStoredRatio("30", { width: 1537, minPx: 340, maxRatio: 0.72 }), 30);

group("存档比例：本次故障的元凶");
// 15.61% × 1537px = 240px，低于 340px 的可用下限 —— 必须丢掉回默认 380px。
check("拖到底留下的 15.61% 被丢弃", readStoredRatio("15.61%", { width: 1537, minPx: 340 }), null);
// 同一个存档值换到更宽的屏幕：15.61% × 1920 = 300px，还是不够，仍然丢。
check("换到 1920 宽仍然不够", readStoredRatio("15.61%", { width: 1920, minPx: 340 }), null);
// 但换到极窄窗口时 15.61% 只剩 160px，更该丢。
check("换到 1024 宽更该丢", readStoredRatio("15.61%", { width: 961, minPx: 340 }), null);

group("存档比例：明显是脏数据");
check("非数字", readStoredRatio("abc", { width: 1537, minPx: 340 }), null);
check("空串", readStoredRatio("", { width: 1537, minPx: 340 }), null);
check("太小（2%）", readStoredRatio("2%", { width: 1537, minPx: 340 }), null);
check("太大（90%）", readStoredRatio("90%", { width: 1537, minPx: 340 }), null);
check("负数", readStoredRatio("-30%", { width: 1537, minPx: 340 }), null);

group("存档比例：量不到宽度时先信它");
// 没激活的页签宽度是 0，这时判不出像素下限，先沿用记录；
// 真可见时由 CSS 的 minmax(340px, …) 兜底，不会被压没。
check("宽度为 0 时不丢记录", readStoredRatio("15.61%", { width: 0, minPx: 340 }), 15.61);

group("存档比例：手改出来的超大值收回拖动上限");
check("78% 被收回到 72%", readStoredRatio("78%", { width: 1537, minPx: 340, maxRatio: 0.72 }), 72);

// ---------------------------------------------------------------------------
// clampRatio：拖动过程中的夹取
// ---------------------------------------------------------------------------
group("拖动夹取");
check("往下限以下夹回 340px", clampRatio(0.01, { width: 1000, minPx: 340, maxRatio: 0.72 }), 0.34);
check("往上限以上夹回 72%", clampRatio(0.99, { width: 1000, minPx: 340, maxRatio: 0.72 }), 0.72);
check("区间内不动", clampRatio(0.5, { width: 1000, minPx: 340, maxRatio: 0.72 }), 0.5);
// 容器比下限还窄（窄屏还没切单列的瞬间）：最低给到 80%，再由 maxRatio 收住，
// 绝不能算出负数或 NaN。
check("容器过窄时不越过上限", clampRatio(0.1, { width: 300, minPx: 340, maxRatio: 0.72 }), 0.72);
check("量不到宽度时原样返回", clampRatio(0.5, { width: 0, minPx: 340, maxRatio: 0.72 }), 0.5);

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log("");
if (failures.length) {
  console.log(`✗ 失败 ${failures.length} 项：`);
  failures.forEach((f) => console.log(`  - ${f}`));
  process.exit(1);
}
console.log(`✓ 全部通过：${passed} 项`);
