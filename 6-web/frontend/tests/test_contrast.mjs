// 可读性护栏的自动化核算 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_contrast.mjs
//
// 为什么要有这个脚本：
//   "正文对比度不低于 4.5:1" 是规范里写死的硬约束，但用眼睛看不出来 4.2 和 4.6 的区别。
//   改一次配色就可能悄悄跌破门槛，所以从 style.css 里直接读令牌、按 WCAG 公式算，
//   任何一处不达标就让 CI 失败。
//
// 两个主题各自独立核算（v1.43）：
//   深色主题的令牌在 :root，浅色主题在 html[data-theme="light"] 覆盖块里。
//   浅色那套是"重新推导"出来的状态色，不是简单提亮 —— 深色的 --ok #34d399
//   对白底只有 1.9:1，直接沿用会让"成功"提示在白底上几乎看不见。
//   所以两套都必须逐条过，任何一套漏算都等于没有护栏。
//
// 计入口径与规范一致：
//   · 只核算"承载信息的正文/标签/数值"，装饰性描边不在此列
//   · 渐变背景（如主按钮）按其两端点分别核算，取更差的那个
//   · 半透明前景先按底色做 alpha 合成，再算对比度

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const cssPath = resolve(here, "../css/style.css");
const css = readFileSync(cssPath, "utf8");

// ---------------------------------------------------------------------------
// 1. 从样式表里抽出两套 :root 令牌
// ---------------------------------------------------------------------------

/** 从 selector 第一次出现的位置起，取它后面那对配平的花括号里的内容。 */
function blockAfter(text, selector) {
  const at = text.indexOf(selector);
  if (at < 0) return null;
  let depth = 0;
  let i = text.indexOf("{", at);
  const open = i;
  for (; i < text.length; i += 1) {
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) break;
    }
  }
  return text.slice(open + 1, i);
}

function parseTokens(block) {
  const tokens = {};
  const re = /(--[a-z0-9-]+)\s*:\s*([^;]+);/gi;
  let m;
  while ((m = re.exec(block))) {
    tokens[m[1]] = m[2].trim();
  }
  return tokens;
}

const darkBlock = blockAfter(css, ":root");
if (!darkBlock) throw new Error("style.css 中找不到 :root");

const lightBlock = blockAfter(css, 'html[data-theme="light"]');
if (!lightBlock) throw new Error('style.css 中找不到 html[data-theme="light"]，浅色主题未实现');

const THEMES = {
  dark: parseTokens(darkBlock),
  light: parseTokens(lightBlock),
};

/** 解析 #rgb / #rrggbb / rgb() / rgba()，失败返回 null。 */
function parseColor(value) {
  if (!value) return null;
  const text = value.trim();
  if (text.startsWith("#")) {
    const hex = text.slice(1);
    if (hex.length === 3) {
      return {
        r: parseInt(hex[0] + hex[0], 16),
        g: parseInt(hex[1] + hex[1], 16),
        b: parseInt(hex[2] + hex[2], 16),
        a: 1,
      };
    }
    if (hex.length === 6 || hex.length === 8) {
      return {
        r: parseInt(hex.slice(0, 2), 16),
        g: parseInt(hex.slice(2, 4), 16),
        b: parseInt(hex.slice(4, 6), 16),
        a: hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1,
      };
    }
    return null;
  }
  const fn = text.match(/rgba?\(([^)]+)\)/i);
  if (fn) {
    const parts = fn[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some((n) => !Number.isFinite(n))) return null;
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
  }
  return null;
}

/** 令牌可能是 var(--x) 的别名，递归解开。按主题各自解析，不能跨主题混用。 */
function makeResolver(tokens) {
  return function resolveToken(name, seen = new Set()) {
    if (seen.has(name)) return null;
    seen.add(name);
    const raw = tokens[name];
    if (!raw) return null;
    const alias = raw.match(/^var\((--[a-z0-9-]+)\)$/i);
    if (alias) return resolveToken(alias[1], seen);
    return parseColor(raw);
  };
}

// ---------------------------------------------------------------------------
// 2. WCAG 对比度
// ---------------------------------------------------------------------------

function channel(v) {
  const s = v / 255;
  return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

function luminance(c) {
  return 0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b);
}

/** 把半透明前景合成到不透明底色上。 */
function composite(fg, bg) {
  if (fg.a >= 1) return fg;
  return {
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  };
}

function contrast(fg, bg) {
  const solid = composite(fg, bg);
  const l1 = luminance(solid);
  const l2 = luminance(bg);
  const hi = Math.max(l1, l2);
  const lo = Math.min(l1, l2);
  return (hi + 0.05) / (lo + 0.05);
}

// ---------------------------------------------------------------------------
// 3. 断言
// ---------------------------------------------------------------------------

const AA_NORMAL = 4.5;
// 大字号（≥18px 常规或 ≥14px 粗体）按 WCAG 只需 3:1，用于 KPI 数值这类大字
const AA_LARGE = 3;

let passed = 0;
const failures = [];

/**
 * 需要核对的「前景 / 背景」组合。两套主题跑同一张表，
 * 这样任何一边的改动都会在两边同时被检验（表里漏一个主题不会悄悄通过）。
 * 第四项是最低对比度，省略按正文 4.5:1。
 */
const PAIRS = [
  ["主文字 / 卡片", "--text", "--surface-1"],
  ["主文字 / 次级块", "--text", "--surface-2"],
  ["主文字 / 悬停块", "--text", "--surface-3"],
  ["主文字 / 页面底", "--text", "--bg"],
  ["次要文字 / 卡片", "--text-dim", "--surface-1"],
  ["次要文字 / 次级块", "--text-dim", "--surface-2"],
  ["次要文字 / 悬停块", "--text-dim", "--surface-3"],
  ["次要文字 / 页面底", "--text-dim", "--bg"],
  ["主色 / 卡片", "--cyan", "--surface-1"],
  ["主色 / 次级块", "--cyan", "--surface-2"],
  ["次级强调 / 卡片", "--magenta", "--surface-1"],
  ["警告 / 卡片", "--amber", "--surface-1"],
  ["警告 / 次级块", "--amber", "--surface-2"],
  ["错误 / 卡片", "--danger", "--surface-1"],
  ["错误 / 次级块", "--danger", "--surface-2"],
  ["成功 / 次级块", "--ok", "--surface-2"],
  // 日志正文在日志底上，长时间阅读，标准不降
  ["日志正文 / 日志底", "--text-log", "--bg-void"],
  // 占位符不承载信息，但仍需可辨；WCAG 对占位符无豁免，按 AA 要求
  ["输入占位符 / 次级块", "--text-faint", "--surface-2"],
  // 主按钮：深色（或浅色）文字压在青蓝渐变上，两端点都要过
  ["按钮文字 / 主色端点", "--ink-on-accent", "--cyan"],
  ["按钮文字 / 主色端点2", "--ink-on-accent", "--teal"],
  // KPI 大数值允许按大字号档
  ["KPI 数值 / 次级块（大字）", "--text", "--surface-2", AA_LARGE],
];

for (const [themeName, tokens] of Object.entries(THEMES)) {
  const resolveToken = makeResolver(tokens);
  console.log(`\n══════ ${themeName === "dark" ? "深色主题 :root" : '浅色主题 html[data-theme="light"]'} ══════`);

  function expectPair(label, fgName, bgName, minimum = AA_NORMAL) {
    const fg = resolveToken(fgName);
    const bg = resolveToken(bgName);
    if (!fg || !bg) {
      failures.push(`${themeName} ${label}：令牌缺失（${fgName}=${tokens[fgName]}，${bgName}=${tokens[bgName]}）`);
      return;
    }
    const ratio = contrast(fg, bg);
    const ok = ratio >= minimum;
    if (ok) passed += 1;
    else failures.push(`${themeName} ${label}：${ratio.toFixed(2)}:1 < ${minimum}:1（${fgName} on ${bgName}）`);
    console.log(
      `${ok ? "✓" : "✗"} ${label.padEnd(30)} ${ratio.toFixed(2).padStart(6)}:1   ${fgName} on ${bgName}`,
    );
  }

  for (const [label, fg, bg, min] of PAIRS) expectPair(label, fg, bg, min);

  console.log(`\n── ${themeName} 效果令牌强度`);
  // 装饰必须够弱，否则会干扰图像判读。这里给一个明确上限，而不是靠肉眼感觉。
  const gridOpacity = parseFloat(tokens["--grid-opacity"]);
  const scanOpacity = parseFloat(tokens["--scan-opacity"]);
  if (!(gridOpacity <= 0.05)) failures.push(`${themeName} --grid-opacity 应 ≤ 0.05，实际 ${gridOpacity}`);
  else passed += 1;
  if (!(scanOpacity <= 0.05)) failures.push(`${themeName} --scan-opacity 应 ≤ 0.05，实际 ${scanOpacity}`);
  else passed += 1;
  console.log(`${gridOpacity <= 0.05 ? "✓" : "✗"} 网格不透明度 ${gridOpacity}`);
  console.log(`${scanOpacity <= 0.05 ? "✓" : "✗"} 扫描线不透明度 ${scanOpacity}`);
}

// ---------------------------------------------------------------------------
// 4. 两套主题的令牌一致性
//
// 尺度组（--sp-* / --fs-* / --radius-* / --font-*）两主题共用，所以浅色块里
// 不重复定义是正常的。真正要防的是两件事：
//   a) 浅色块里写了一个深色块没有的令牌（拼写错误 / 忘了在 :root 里建）
//   b) 新加的语义令牌只在深色里定义，浅色主题下悄悄用了深色值
// ---------------------------------------------------------------------------

console.log("\n── 两套主题的令牌一致性");

const lightOnly = Object.keys(THEMES.light).filter((key) => !(key in THEMES.dark));
if (lightOnly.length) {
  failures.push(`浅色主题定义了深色主题没有的令牌：${lightOnly.join(", ")}`);
} else {
  passed += 1;
}
console.log(`${lightOnly.length ? "✗" : "✓"} 浅色块无孤立令牌`);

// 必须被浅色主题覆盖的"随主题变化"的令牌。加新令牌时如果不进这张表，
// 就等于放弃了它的浅色检查 —— 所以这张表要跟着设计令牌一起维护。
const MUST_OVERRIDE = [
  "--bg", "--bg-void", "--surface-1", "--surface-2", "--surface-3",
  "--line", "--line-strong", "--text", "--text-dim", "--text-faint", "--text-log",
  "--cyan", "--teal", "--magenta", "--amber", "--danger", "--ok", "--ink-on-accent",
  "--stage", "--chrome-bg", "--focus-ring", "--row-alt",
  "--shadow-1", "--shadow-2", "--shadow-3", "--shadow-modal", "--scrim",
  "--scan-line", "--topbar-top", "--grid-opacity", "--scan-opacity",
  "--glow-sm", "--glow-md",
];
const missing = MUST_OVERRIDE.filter((key) => !(key in THEMES.light));
if (missing.length) {
  failures.push(`浅色主题缺少必须覆盖的令牌：${missing.join(", ")}`);
} else {
  passed += 1;
}
console.log(`${missing.length ? "✗" : "✓"} 关键令牌在浅色主题中全部覆盖（${MUST_OVERRIDE.length} 项）`);

console.log("");
if (failures.length === 0) {
  console.log(`✓ 全部通过：${passed} 项`);
  process.exit(0);
} else {
  console.error(`✗ 失败 ${failures.length} 项（通过 ${passed} 项）：`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
