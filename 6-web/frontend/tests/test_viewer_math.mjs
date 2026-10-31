// 查看器变换数学的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_viewer_math.mjs
//
// 只测纯函数。DOM 交互（滚轮、拖动、持久化）走浏览器实测，见 tasks.md 6.5。
// 之所以把数学抽出来单测，是因为"光标锚点"这类错误在浏览器里肉眼很难发现，
// 只会表现为"越缩放越跑偏"，用户会以为程序坏了。

import {
  MIN_SCALE,
  MAX_SCALE,
  clampScale,
  computeZoomTransform,
  screenToImage,
  imageToScreenRect,
  normalizeWheelDelta,
  wheelDeltaToFactor,
} from "../js/viewer.js";

// ---------------------------------------------------------------------------
// 极简断言框架
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) {
    passed += 1;
  } else {
    failures.push(message);
  }
}

function nearly(actual, expected, epsilon, message) {
  assert(
    Math.abs(actual - expected) <= epsilon,
    `${message}（实际 ${actual}，期望 ${expected}，容差 ${epsilon}）`,
  );
}

function group(name) {
  console.log(`\n── ${name}`);
}

// 正变换：图像坐标 → 屏幕坐标
function forward(point, scale, offset) {
  return { x: point.x * scale + offset.x, y: point.y * scale + offset.y };
}

// ---------------------------------------------------------------------------
group("倍率钳制");
// ---------------------------------------------------------------------------
assert(clampScale(0.05) === MIN_SCALE, "低于下界应钳制到 MIN_SCALE");
assert(clampScale(50) === MAX_SCALE, "高于上界应钳制到 MAX_SCALE");
assert(clampScale(1) === 1, "区间内应原样返回");
assert(clampScale(-3) === MIN_SCALE, "负数应钳制到 MIN_SCALE");
assert(clampScale(Number.NaN) === MIN_SCALE, "NaN 应退化为下界");
assert(clampScale(5, 2, 3) === 3, "应支持自定义上下界");
assert(clampScale(1, 2, 3) === 2, "自定义下界应生效");

// ---------------------------------------------------------------------------
group("光标锚点：单次缩放后锚点不移动");
// ---------------------------------------------------------------------------
{
  const cursor = { x: 400, y: 300 };
  const s0 = 1;
  const t0 = { x: 0, y: 0 };
  const anchorImage = screenToImage(cursor, s0, t0);

  const next = computeZoomTransform(cursor, s0, t0, 2.5);
  const anchorScreen = forward(anchorImage, next.scale, next.offset);

  nearly(anchorScreen.x, cursor.x, 0.001, "缩放后锚点 X 应停在光标处");
  nearly(anchorScreen.y, cursor.y, 0.001, "缩放后锚点 Y 应停在光标处");
}

// ---------------------------------------------------------------------------
group("光标锚点：连续多次缩放累积漂移不超过 0.5 像素");
// ---------------------------------------------------------------------------
{
  const cursor = { x: 517, y: 233 };
  let scale = 0.8;
  let offset = { x: -120, y: 46 };
  const anchorImage = screenToImage(cursor, scale, offset);

  // 模拟用户来回缩放的一串倍率
  const chain = [3.4, 1.1, 9.7, 0.42, 6.0, 2.2, 15.5, 1.0];
  for (const target of chain) {
    const next = computeZoomTransform(cursor, scale, offset, target);
    scale = next.scale;
    offset = next.offset;
  }

  const anchorScreen = forward(anchorImage, scale, offset);
  nearly(anchorScreen.x, cursor.x, 0.5, "连续缩放后锚点 X 漂移应小于 0.5px");
  nearly(anchorScreen.y, cursor.y, 0.5, "连续缩放后锚点 Y 漂移应小于 0.5px");
}

// ---------------------------------------------------------------------------
group("缩放倍率不变时平移量也不变");
// ---------------------------------------------------------------------------
{
  const cursor = { x: 100, y: 100 };
  const t = { x: 33, y: -12 };
  const next = computeZoomTransform(cursor, 2, t, 2);
  nearly(next.offset.x, t.x, 1e-9, "倍率不变时平移 X 应不变");
  nearly(next.offset.y, t.y, 1e-9, "倍率不变时平移 Y 应不变");
}

// ---------------------------------------------------------------------------
group("平移与缩放的组合顺序可交换");
// ---------------------------------------------------------------------------
{
  // 先以 c 为锚点缩放、再整体平移 d
  // 与 先整体平移 d、再以 c+d 为锚点缩放，结果必须完全相同。
  // 这条性质保证"拖动后立刻滚轮"与"滚轮后立刻拖动"不会让用户感到跳变。
  const c = { x: 260, y: 180 };
  const d = { x: 37, y: -21 };
  const scale0 = 1.4;
  const offset0 = { x: 15, y: 9 };
  const scale1 = 3.1;

  const a1 = computeZoomTransform(c, scale0, offset0, scale1);
  const a = { scale: a1.scale, offset: { x: a1.offset.x + d.x, y: a1.offset.y + d.y } };

  const offsetB = { x: offset0.x + d.x, y: offset0.y + d.y };
  const b = computeZoomTransform(
    { x: c.x + d.x, y: c.y + d.y },
    scale0,
    offsetB,
    scale1,
  );

  nearly(a.offset.x, b.offset.x, 1e-9, "两种顺序的平移 X 应一致");
  nearly(a.offset.y, b.offset.y, 1e-9, "两种顺序的平移 Y 应一致");
  nearly(a.scale, b.scale, 1e-12, "两种顺序的倍率应一致");
}

// ---------------------------------------------------------------------------
group("屏幕与图像坐标互逆");
// ---------------------------------------------------------------------------
{
  const scale = 2.75;
  const offset = { x: -40, y: 130 };
  const samples = [
    { x: 0, y: 0 },
    { x: 320, y: 240 },
    { x: 1536, y: 793 },
  ];
  for (const point of samples) {
    const image = screenToImage(point, scale, offset);
    const back = forward(image, scale, offset);
    nearly(back.x, point.x, 1e-9, `屏幕→图像→屏幕 X 应还原（${point.x}）`);
    nearly(back.y, point.y, 1e-9, `屏幕→图像→屏幕 Y 应还原（${point.y}）`);
  }
}

// ---------------------------------------------------------------------------
group("包围盒换算");
// ---------------------------------------------------------------------------
{
  const rect = imageToScreenRect(0.5, { x: 10, y: 20 }, 800, 600);
  nearly(rect.x, 10, 1e-9, "包围盒 X 等于平移量");
  nearly(rect.y, 20, 1e-9, "包围盒 Y 等于平移量");
  nearly(rect.width, 400, 1e-9, "包围盒宽度等于 图像宽 × 倍率");
  nearly(rect.height, 300, 1e-9, "包围盒高度等于 图像高 × 倍率");
}

// ---------------------------------------------------------------------------
group("滚轮增量归一化");
// ---------------------------------------------------------------------------
{
  // 像素模式：原样返回
  nearly(normalizeWheelDelta({ deltaY: 100, deltaMode: 0 }), 100, 1e-9, "像素模式应原样返回");
  // 行模式：×16
  nearly(normalizeWheelDelta({ deltaY: 3, deltaMode: 1 }), 48, 1e-9, "行模式应乘以 16");
  // 页模式：按视口高度换算
  nearly(normalizeWheelDelta({ deltaY: 1, deltaMode: 2 }, 200), 200, 1e-9, "页模式应按视口高度换算");
  // 页模式换算结果同样受上限约束：一页 900px 也不该产生 900 的缩放跃变
  nearly(normalizeWheelDelta({ deltaY: 1, deltaMode: 2 }, 900), 300, 1e-9, "页模式换算后仍应受上限钳制");
  // 上限钳制：某些设备会一次给出极大增量
  nearly(normalizeWheelDelta({ deltaY: 99999, deltaMode: 0 }), 300, 1e-9, "超大增量应钳制到上限");
  nearly(normalizeWheelDelta({ deltaY: -99999, deltaMode: 0 }), -300, 1e-9, "超小增量应钳制到下限");
  // 缺字段不应崩
  nearly(normalizeWheelDelta({}), 0, 1e-9, "缺少 deltaY 时应返回 0");
}

// ---------------------------------------------------------------------------
group("滚轮因子可逆");
// ---------------------------------------------------------------------------
{
  // 放大一格再缩小一格，必须回到原倍率；线性方案在边界会被吃了步长。
  const factorIn = wheelDeltaToFactor(-100);
  const factorOut = wheelDeltaToFactor(100);
  assert(factorIn > 1, "向上滚应放大");
  assert(factorOut < 1, "向下滚应缩小");
  nearly(factorIn * factorOut, 1, 1e-12, "一进一出应回到原倍率");

  // 方向必须与"向上滚 = 放大"一致，且增量越大放得越多
  assert(wheelDeltaToFactor(-1) < wheelDeltaToFactor(-100), "小幅上滚的放大倍数应小于大幅上滚");
}

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log("");
if (failures.length === 0) {
  console.log(`✓ 全部通过：${passed} 条断言`);
  process.exit(0);
} else {
  console.error(`✗ 失败 ${failures.length} 条（通过 ${passed} 条）：`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
