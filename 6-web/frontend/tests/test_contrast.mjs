// 可读性护栏的自动化核算 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_contrast.mjs
//
// 为什么要有这个脚本：
//   "正文对比度不低于 4.5:1" 是规范里写死的硬约束，但用眼睛看不出来 4.2 和 4.6 的区别。
//   改一次配色就可能悄悄跌破门槛，所以从 style.css 里直接读令牌、按 WCAG 公式算，
//   任何一处不达标就让 CI 失败。
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
// 1. 从样式表里抽出 :root 令牌
// ---------------------------------------------------------------------------

function rootBlock(text) {
  const start = text.indexOf(":root");
  if (start < 0) throw new Error("style.css 中找不到 :root");
  let depth = 0;
  let i = text.indexOf("{", start);
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

const tokens = parseTokens(rootBlock(css));

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

/** 令牌可能是 var(--x) 的别名，递归解开。 */
function resolveToken(name, seen = new Set()) {
  if (seen.has(name)) return null;
  seen.add(name);
  const raw = tokens[name];
  if (!raw) return null;
  const alias = raw.match(/^var\((--[a-z0-9-]+)\)$/i);
  if (alias) return resolveToken(alias[1], seen);
  return parseColor(raw);
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

function expectPair(label, fgName, bgName, minimum = AA_NORMAL) {
  const fg = resolveToken(fgName);
  const bg = resolveToken(bgName);
  if (!fg || !bg) {
    failures.push(`${label}：令牌缺失（${fgName}=${tokens[fgName]}，${bgName}=${tokens[bgName]}）`);
    return;
  }
  const ratio = contrast(fg, bg);
  const ok = ratio >= minimum;
  if (ok) passed += 1;
  else failures.push(`${label}：${ratio.toFixed(2)}:1 < ${minimum}:1（${fgName} on ${bgName}）`);
  console.log(
    `${ok ? "✓" : "✗"} ${label.padEnd(30)} ${ratio.toFixed(2).padStart(6)}:1   ${fgName} on ${bgName}`,
  );
}

console.log("── 正文与次要文字");
expectPair("主文字 / 卡片", "--text", "--surface-1");
expectPair("主文字 / 次级块", "--text", "--surface-2");
expectPair("主文字 / 悬停块", "--text", "--surface-3");
expectPair("主文字 / 页面底", "--text", "--bg");
expectPair("次要文字 / 卡片", "--text-dim", "--surface-1");
expectPair("次要文字 / 次级块", "--text-dim", "--surface-2");
expectPair("次要文字 / 悬停块", "--text-dim", "--surface-3");
expectPair("次要文字 / 页面底", "--text-dim", "--bg");

console.log("\n── 强调与状态色");
expectPair("主色 / 卡片", "--cyan", "--surface-1");
expectPair("主色 / 次级块", "--cyan", "--surface-2");
expectPair("次级强调 / 卡片", "--magenta", "--surface-1");
expectPair("警告 / 卡片", "--amber", "--surface-1");
expectPair("警告 / 次级块", "--amber", "--surface-2");
expectPair("错误 / 卡片", "--danger", "--surface-1");
expectPair("错误 / 次级块", "--danger", "--surface-2");
expectPair("成功 / 次级块", "--ok", "--surface-2");

console.log("\n── 特殊区域");
// 日志正文在纯黑底上，长时间阅读，标准不降
expectPair("日志正文 / 日志底", "--text-log", "--bg-void");
// 占位符不承载信息，但仍需可辨；WCAG 对占位符无豁免，按 AA 要求
expectPair("输入占位符 / 次级块", "--text-faint", "--surface-2");
// 主按钮：深色文字压在青蓝渐变上，两端点都要过
expectPair("按钮文字 / 主色端点", "--ink-on-accent", "--cyan");
expectPair("按钮文字 / 主色端点2", "--ink-on-accent", "--teal");

// KPI 大数值允许按大字号档
expectPair("KPI 数值 / 次级块（大字）", "--text", "--surface-2", AA_LARGE);

// ---------------------------------------------------------------------------
console.log("\n── 效果令牌强度");
// 装饰必须够弱，否则会干扰图像判读。这里给一个明确上限，而不是靠肉眼感觉。
const gridOpacity = parseFloat(tokens["--grid-opacity"]);
const scanOpacity = parseFloat(tokens["--scan-opacity"]);
if (!(gridOpacity <= 0.05)) failures.push(`--grid-opacity 应 ≤ 0.05，实际 ${gridOpacity}`);
else passed += 1;
if (!(scanOpacity <= 0.05)) failures.push(`--scan-opacity 应 ≤ 0.05，实际 ${scanOpacity}`);
else passed += 1;
console.log(`${gridOpacity <= 0.05 ? "✓" : "✗"} 网格不透明度 ${gridOpacity}`);
console.log(`${scanOpacity <= 0.05 ? "✓" : "✗"} 扫描线不透明度 ${scanOpacity}`);

console.log("");
if (failures.length === 0) {
  console.log(`✓ 全部通过：${passed} 项`);
  process.exit(0);
} else {
  console.error(`✗ 失败 ${failures.length} 项（通过 ${passed} 项）：`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
