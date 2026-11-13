// 几何与命中测试的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_geometry.mjs
//
// 为什么值得单测：
// - 命中容差是"屏幕像素 / scale"，缩放后手感必须一致，写反了会变成"放大后点不中"；
// - hitTest 从后往前找，顺序错了会选中被盖住的那个框；
// - 旋转框的角点顺序被旋转手柄依赖，顺序变了手柄会跑到对角线上去；
// - snapPoint 的返回值契约是数组 `[x, y]`，一旦某个分支返回 `{x, y}`，
//   拖拽画框就永远收不了尾。

import {
  EDGE_HIT_RADIUS,
  HIT_TOLERANCE,
  POINT_SLACK,
  SNAP_RADIUS,
  bboxOf,
  constrainOrthogonal,
  distanceToSegment,
  hasRotateHandle,
  hitEdge,
  hitHandle,
  hitTest,
  pointInPolygon,
  rotateHandlePoint,
  shapeCorners,
  snapPoint,
} from "../js/annotate/geometry.js";

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

function near(name, actual, expected, eps = 1e-6) {
  const ok = Math.abs(actual - expected) <= eps;
  if (ok) {
    passed += 1;
    console.log(`✓ ${name}`);
  } else {
    failures.push(`${name}\n    期望 ≈${expected}，实际 ${actual}`);
    console.log(`✗ ${name}`);
  }
}

const rect = (id, x0, y0, x1, y1, extra = {}) => ({
  id,
  shape_type: "rectangle",
  points: [[x0, y0], [x1, y1]],
  ...extra,
});
const polygon = (id, points, extra = {}) => ({ id, shape_type: "polygon", points, ...extra });
const point = (id, x, y) => ({ id, shape_type: "point", points: [[x, y]] });
const line = (id, a, b) => ({ id, shape_type: "line", points: [a, b] });

// ---------------------------------------------------------------------------
// 1. 点在多边形内 / 点到线段
// ---------------------------------------------------------------------------
const square = [[0, 0], [10, 0], [10, 10], [0, 10]];
check("正方形内的点算命中", pointInPolygon({ x: 5, y: 5 }, square), true);
check("正方形外的点不算", pointInPolygon({ x: 15, y: 5 }, square), false);
check("负坐标区域外的点不算", pointInPolygon({ x: -1, y: 5 }, square), false);
check("顶点右侧一点点算外", pointInPolygon({ x: 10.5, y: 10.5 }, square), false);

near("垂足落在线段内取垂直距离", distanceToSegment({ x: 5, y: 3 }, [0, 0], [10, 0]), 3);
near("垂足落在线段外取端点距离", distanceToSegment({ x: 13, y: 4 }, [0, 0], [10, 0]), 5);
near("零长线段退化为点距", distanceToSegment({ x: 3, y: 4 }, [0, 0], [0, 0]), 5);
near("点在线上距离为 0", distanceToSegment({ x: 2, y: 2 }, [0, 0], [4, 4]), 0);

// ---------------------------------------------------------------------------
// 2. shapeCorners / bboxOf
// ---------------------------------------------------------------------------
checkDeep("矩形两点展开成四角（左上→右上→右下→左下）", shapeCorners(rect("r", 2, 3, 8, 9)), [
  [2, 3],
  [8, 3],
  [8, 9],
  [2, 9],
]);
checkDeep("矩形反向拖拽也能得到正序角点", shapeCorners(rect("r", 8, 9, 2, 3)), [
  [8, 9],
  [2, 9],
  [2, 3],
  [8, 3],
]);
checkDeep("多边形原样返回顶点", shapeCorners(polygon("p", [[0, 0], [4, 0], [2, 3]])), [[0, 0], [4, 0], [2, 3]]);

const rotated = {
  id: "rr",
  shape_type: "rotated_rect",
  points: [[0, 0], [10, 4]],
  rotation: Math.PI / 2,
};
const rc = shapeCorners(rotated);
check("旋转框仍然是 4 个角", rc.length, 4);
// 中心 (5, 2)，halfW=5、halfH=2；转 90° 后长边变成竖直方向
near("旋转 90° 后第一个角的 x", rc[0][0], 5 + 2, 1e-9);
near("旋转 90° 后第一个角的 y", rc[0][1], 2 - 5, 1e-9);
near("旋转 90° 后对角点中心对称", rc[0][0] + rc[2][0], 10, 1e-9);
near("旋转 90° 后对角点中心对称（y）", rc[0][1] + rc[2][1], 4, 1e-9);

checkDeep("包围盒覆盖所有点", bboxOf([[2, 3], [8, 1], [5, 9]]), { x: 2, y: 1, width: 6, height: 8 });
checkDeep("单点包围盒尺寸为 0", bboxOf([[4, 4]]), { x: 4, y: 4, width: 0, height: 0 });
check("空点集没有包围盒", bboxOf([]), null);
check("null 点集没有包围盒", bboxOf(null), null);

// ---------------------------------------------------------------------------
// 3. hitTest
// ---------------------------------------------------------------------------
const r1 = rect("r1", 0, 0, 10, 10);
const r2 = rect("r2", 2, 2, 6, 6);
check("点在框内命中", hitTest({ x: 5, y: 5 }, { objects: [r1] })?.id, "r1");
check("重叠时后画的优先", hitTest({ x: 4, y: 4 }, { objects: [r1, r2] })?.id, "r2");
check("点在框外不命中", hitTest({ x: 40, y: 40 }, { objects: [r1] }), null);
check(
  "贴着边框（容差内）也能选中",
  hitTest({ x: 10 + HIT_TOLERANCE - 0.5, y: 5 }, { objects: [r1] })?.id,
  "r1",
);
check("超出容差不命中", hitTest({ x: 10 + HIT_TOLERANCE + 1, y: 5 }, { objects: [r1] }), null);

// scale=2（放大两倍）时，图像坐标里允许的误差减半——屏幕手感不变
check(
  "放大 2 倍后容差按图像坐标减半（边界内命中）",
  hitTest({ x: 10 + HIT_TOLERANCE / 2 - 0.1, y: 5 }, { objects: [r1], scale: 2 })?.id,
  "r1",
);
check(
  "放大 2 倍后容差按图像坐标减半（超出不命中）",
  hitTest({ x: 10 + HIT_TOLERANCE / 2 + 1, y: 5 }, { objects: [r1], scale: 2 }),
  null,
);
check("scale 非法时按 1 处理", hitTest({ x: 5, y: 5 }, { objects: [r1], scale: 0 })?.id, "r1");

const hiddenSet = new Set(["r2"]);
check("隐藏的对象不能被点中，落到下层", hitTest({ x: 4, y: 4 }, { objects: [r1, r2], hidden: hiddenSet })?.id, "r1");

const p1 = point("p1", 20, 20);
check("点目标有额外放宽", hitTest({ x: 20 + HIT_TOLERANCE + POINT_SLACK - 0.1, y: 20 }, { objects: [p1] })?.id, "p1");
check("点目标超出放宽不命中", hitTest({ x: 20 + HIT_TOLERANCE + POINT_SLACK + 1, y: 20 }, { objects: [p1] }), null);

const l1 = line("l1", [0, 0], [10, 10]);
check("线段靠距离命中", hitTest({ x: 5, y: 5.5 }, { objects: [l1], scale: 1 })?.id, "l1");
check("离线段远的点不命中", hitTest({ x: 5, y: 20 }, { objects: [l1] }), null);

const poly = polygon("poly", [[0, 0], [10, 0], [10, 10]]);
check("多边形内部命中", hitTest({ x: 8, y: 2 }, { objects: [poly] })?.id, "poly");
check("多边形外部远处不命中", hitTest({ x: -40, y: 40 }, { objects: [poly] }), null);
check("空对象列表返回 null", hitTest({ x: 0, y: 0 }, { objects: [] }), null);
check("空顶点对象被跳过", hitTest({ x: 0, y: 0 }, { objects: [{ id: "e", points: [] }] }), null);

// ---------------------------------------------------------------------------
// 4. 旋转手柄与手柄命中
// ---------------------------------------------------------------------------
// 手柄容差是 (HANDLE_RADIUS + 3) 像素，只有几十像素的小框整块都在容差里，
// 所以手柄测试统一用 400×400 的大框，避免"中心点被判成角点"这种假失败。
const bigRect = rect("big", 0, 0, 400, 400);
const bigPoly = polygon("bp", [[0, 0], [400, 0], [400, 400]]);
check("普通矩形没有旋转手柄", rotateHandlePoint(bigRect), null);
const knob = rotateHandlePoint(rotated, 1);
check("旋转框有旋转手柄", Array.isArray(knob), true);
// 两点 (0,0)-(10,4) 绕中心 (5,2) 转 90° 后，角点为 (7,-3)(7,7)(3,7)(3,-3)，
// 顶边中点落在 (7,2)，手柄沿"中心 → 顶边中点"的外法线方向再外推 22 像素。
near("手柄沿顶边外法线方向（x 正方向）", knob[0], 7 + 22, 1e-9);
near("手柄与顶边中点的 y 一致", knob[1], 2, 1e-9);
const knob2 = rotateHandlePoint(rotated, 2);
near("scale=2 时手柄的图像距离减半", knob2[0], 7 + 11, 1e-9);
check("没有旋转字段也按 0 度处理", Array.isArray(rotateHandlePoint({ ...rotated, rotation: undefined })), true);

checkDeep("矩形角点手柄", hitHandle(bigRect, { x: 0, y: 0 }, { scale: 1 }), { kind: "corner", index: 0 });
checkDeep("矩形右下角手柄", hitHandle(bigRect, { x: 400, y: 400 }), { kind: "corner", index: 2 });
check("大框中心不是手柄", hitHandle(bigRect, { x: 200, y: 200 }), null);
checkDeep("旋转框顶部的圆点优先判为 rotate", hitHandle(rotated, { x: knob[0], y: knob[1] }), { kind: "rotate" });
checkDeep("多边形顶点手柄", hitHandle(bigPoly, { x: 400, y: 0 }), { kind: "vertex", index: 1 });
checkDeep("多边形边中点可以插点", hitHandle(bigPoly, { x: 200, y: 0 }), { kind: "insert", index: 0 });
check("顶点容差随缩放收紧（偏 10 像素点不中）", hitHandle(bigPoly, { x: 410, y: 0 }, { scale: 1 }), null);
const looseAtSmallScale = hitHandle(bigPoly, { x: 410, y: 0 }, { scale: 0.5 });
checkDeep("缩小一半后同样的屏幕偏移仍能点中", looseAtSmallScale, { kind: "vertex", index: 1 });

checkDeep("矩形边中点命中", hitEdge(bigRect, { x: 200, y: 0 }), { kind: "edge", index: 0 });
check("边中点超出容差不命中", hitEdge(bigRect, { x: 200, y: EDGE_HIT_RADIUS + 1 }), null);
check("多边形没有边拖拽", hitEdge(bigPoly, { x: 200, y: 0 }), null);
// 旋转框边长只有 4~10 像素，四个边中点相距不到容差，互相抢命中；
// 单独造一个 100×40 的旋转框来验底边。
const bigRot = { id: "br", shape_type: "rotated_rect", points: [[0, 0], [100, 40]], rotation: Math.PI / 2 };
checkDeep("旋转框也有边中点（底边）", hitEdge(bigRot, { x: 50, y: -30 }, { scale: 1 }), { kind: "edge", index: 3 });
checkDeep("旋转框右边中点", hitEdge(bigRot, { x: 70, y: 20 }, { scale: 1 }), { kind: "edge", index: 0 });
check("旋转框才有旋转手柄判定", hasRotateHandle(rotated), true);
check("普通矩形不报告旋转手柄", hasRotateHandle(bigRect), false);

// ---------------------------------------------------------------------------
// 5. 吸附
// ---------------------------------------------------------------------------
const anchorObj = rect("v", 100, 100, 120, 120);
checkDeep(
  "靠近顶点时吸到顶点",
  snapPoint({ x: 101, y: 101 }, { candidates: [anchorObj], scale: 1 }),
  { point: [100, 100], snap: [100, 100] },
);
checkDeep(
  "超出吸附半径不吸",
  snapPoint({ x: 130, y: 130 }, { candidates: [anchorObj], scale: 1 }),
  { point: [130, 130], snap: null },
);
checkDeep(
  "多个候选取最近的",
  snapPoint({ x: 119.4, y: 119.4 }, { candidates: [anchorObj], scale: 1 }),
  { point: [120, 120], snap: [120, 120] },
);
checkDeep(
  "关掉吸附就原样返回",
  snapPoint({ x: 100.5, y: 100.5 }, { candidates: [anchorObj], enabled: false }),
  { point: [100.5, 100.5], snap: null },
);
// scale=2 时吸附半径换算成图像坐标是 4.5
checkDeep(
  "放大 2 倍后吸附半径减半（4 以内仍吸）",
  snapPoint({ x: 104, y: 100 }, { candidates: [anchorObj], scale: 2 }),
  { point: [100, 100], snap: [100, 100] },
);
checkDeep(
  "放大 2 倍后吸附半径减半（5 以外不吸）",
  snapPoint({ x: 105, y: 100 }, { candidates: [anchorObj], scale: 2 }),
  { point: [105, 100], snap: null },
);

const pending = [[0, 0], [10, 0], [10, 10]];
checkDeep(
  "多边形闭合吸附到起点",
  snapPoint({ x: 1, y: 1 }, { pending, allowClose: true, scale: 1 }),
  { point: [0, 0], snap: [0, 0] },
);
checkDeep(
  "点不够 3 个不闭合",
  snapPoint({ x: 1, y: 1 }, { pending: [[0, 0], [5, 5]], allowClose: true, scale: 1 }),
  { point: [1, 1], snap: null },
);
checkDeep(
  "没开 allowClose 不闭合",
  snapPoint({ x: 1, y: 1 }, { pending, allowClose: false, scale: 1 }),
  { point: [1, 1], snap: null },
);
const snapResult = snapPoint({ x: 3, y: 4 }, { enabled: false });
check("返回值永远是数组契约", Array.isArray(snapResult.point), true);
check("关闭吸附时 point 是 [x, y]", snapResult.point.join(","), "3,4");
check("吸附半径常量是 9", SNAP_RADIUS, 9);

// ---------------------------------------------------------------------------
// 6. 正交约束
// ---------------------------------------------------------------------------
checkDeep("横向位移大就锁 y", constrainOrthogonal([0, 0], { x: 10, y: 3 }), [10, 0]);
checkDeep("纵向位移大就锁 x", constrainOrthogonal([0, 0], { x: 3, y: -10 }), [0, -10]);
checkDeep("位移相等时按横向处理", constrainOrthogonal([0, 0], { x: 5, y: 5 }), [5, 0]);
checkDeep("反向也能约束", constrainOrthogonal([5, 5], { x: 1, y: 6 }), [1, 5]);
// toImage 返回 {x, y}、state.pending 存 [x, y]，两种都要吃：
// 老实现只用 point.x 却返回 point[0]，结果是 X 变成 undefined（NaN 坐标）。
checkDeep("对象形式的点也能约束（toImage 的返回值）", constrainOrthogonal([0, 0], { x: 10, y: 3 }), [10, 0]);
checkDeep("数组形式的点也能约束（state.pending 里的点）", constrainOrthogonal([0, 0], [3, 10]), [0, 10]);

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
