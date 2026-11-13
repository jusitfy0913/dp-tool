// 草稿分档的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_tiers.mjs
//
// 为什么值得单测：
// - 阈值落点（0.7 / 0.4）是闭区间，改动比较符会让"边界那一条"换档而不报错；
// - 颜色与文案必须一一对应，画布绿、列表红是用户一眼能看出来的 bug；
// - countByTier 的键顺序被 UI 的 tooltip 拼接依赖，乱序会拼出"低 1 / 高 2"。

import {
  TIER_COLORS,
  TIER_HIGH,
  TIER_MID,
  TIER_ORDER,
  TIER_TEXT,
  countByTier,
  tierClass,
  tierColor,
  tierCountText,
  tierOf,
  tierText,
} from "../js/annotate/tiers.js";

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

// ---------------------------------------------------------------------------
// 1. 阈值
// ---------------------------------------------------------------------------
checkDeep("阈值就是 0.7 / 0.4", [TIER_HIGH, TIER_MID], [0.7, 0.4]);
check("1.0 归高置信度", tierOf(1), "high");
check("0.7 正好落在高位（闭区间）", tierOf(0.7), "high");
check("0.699 掉到中位", tierOf(0.699), "mid");
check("0.4 正好落在中位（闭区间）", tierOf(0.4), "mid");
check("0.399 掉到低位", tierOf(0.399), "low");
check("0 是低位而不是未知", tierOf(0), "low");
check("负数表示没给置信度", tierOf(-1), "unknown");
check("null 视为未知", tierOf(null), "unknown");
check("undefined 视为未知", tierOf(undefined), "unknown");
check("NaN 视为未知", tierOf(Number.NaN), "unknown");
check("Infinity 视为未知", tierOf(Number.POSITIVE_INFINITY), "unknown");
check("字符串数字照常分档", tierOf("0.75"), "high");

// ---------------------------------------------------------------------------
// 2. 颜色 / 文案 / 类名
// ---------------------------------------------------------------------------
checkDeep("高档颜色与 CSS 的 --ok 同值", TIER_COLORS.high, [52, 211, 153]);
checkDeep("中档颜色与 CSS 的 --amber 同值", TIER_COLORS.mid, [251, 191, 36]);
checkDeep("低档颜色与 CSS 的 --danger 同值", TIER_COLORS.low, [248, 113, 113]);
check("未知档不标红（用琥珀色，避免整批看起来像模型在猜）", JSON.stringify(tierColor(-1)), JSON.stringify(TIER_COLORS.mid));
check("高档色跟随阈值", tierColor(0.71), TIER_COLORS.high);
check("中档文案", tierText(0.5), "中");
check("低位文案", tierText(0.1), "低");
check("未知档文案是问号", tierText(-5), "?");
check("高档类名", tierClass(0.9), "tier-high");
check("低位类名", tierClass(0.05), "tier-low");
check("未知档类名", tierClass(-1), "tier-unknown");
checkDeep("文案表覆盖四个档位", Object.keys(TIER_TEXT).sort(), ["high", "low", "mid", "unknown"]);
checkDeep("档位顺序固定为高中低未知", [...TIER_ORDER], ["high", "mid", "low", "unknown"]);

// ---------------------------------------------------------------------------
// 3. 计数
// ---------------------------------------------------------------------------
const drafts = [
  { score: 0.95 },
  { score: 0.7 },
  { score: 0.55 },
  { score: 0.4 },
  { score: 0.2 },
  { score: -1 },
  { score: undefined },
];
checkDeep("七条草稿按档位计数", countByTier(drafts), { high: 2, mid: 2, low: 1, unknown: 2, total: 7 });
checkDeep("空集合全零", countByTier([]), { high: 0, mid: 0, low: 0, unknown: 0, total: 0 });
checkDeep("null 入参不炸", countByTier(null), { high: 0, mid: 0, low: 0, unknown: 0, total: 0 });
checkDeep("支持 Map.values()", countByTier(new Map([["a", { score: 0.8 }], ["b", { score: 0.1 }]]).values()), {
  high: 1,
  mid: 0,
  low: 1,
  unknown: 0,
  total: 2,
});
check("计数摘要跳过为 0 的档", tierCountText({ high: 2, mid: 0, low: 1, unknown: 0, total: 3 }), "高 2 / 低 1");
check("空计数没有摘要", tierCountText({ high: 0, mid: 0, low: 0, unknown: 0, total: 0 }), "");
check("没有入参也没有摘要", tierCountText(null), "");

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
