// 标注画布的几何与命中测试：纯函数，几何输入输出一律是图像坐标。
//
// 坐标系约定（与 viewer.js 相同）：`p_screen = p_image * scale + offset`。
// 本模块**只吃 scale**，不碰 offset——命中测试全部在图像坐标里做，
// 所以容差要除以 scale 才能换算成"屏幕像素手感"。offset 只影响显示，
// 换算交给调用方（annotate.js 的 toImage）。
//
// 为什么不在这里读全局 state：这些函数被画布拖拽在每一帧调用，
// 一旦隐式依赖界面上某个开关，单测就只能靠造假 DOM 来跑。
//
// 单测：node 6-web/frontend/tests/test_geometry.mjs

/** 顶点手柄的屏幕半径（像素）。固定屏幕尺寸，缩放时不变。 */
export const HANDLE_RADIUS = 5;
/** 命中测试的容差（屏幕像素）。太小点不中，太大容易误选。 */
export const HIT_TOLERANCE = 6;
/** 吸附半径（屏幕像素）。 */
export const SNAP_RADIUS = 9;
/** 旋转手柄离矩形顶边的距离（屏幕像素）。 */
export const ROTATE_KNOB_GAP = 22;
/** 多边形边中点"加顶点"手柄的判定半径（屏幕像素）。 */
export const EDGE_INSERT_RADIUS = 4 + 3;
/** 矩形边中点"整体拖边"的判定半径（屏幕像素）。 */
export const EDGE_HIT_RADIUS = 6;
/** 点/关键点命中时额外放宽的像素，让小目标也好点。 */
export const POINT_SLACK = 4;

/** 射线法判点在多边形内。`polygon` 为 `[[x, y], ...]`。 */
export function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi || 1e-9) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** 点到线段的最短距离（同坐标系）。零长线段退化为点距。 */
export function distanceToSegment(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(point.x - a[0], point.y - a[1]);
  let t = ((point.x - a[0]) * dx + (point.y - a[1]) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(point.x - (a[0] + t * dx), point.y - (a[1] + t * dy));
}

/**
 * 形状的顶点（图像坐标）。
 *
 * 矩形类的两点表示在这里展开成 4 个角点用于绘制/命中；
 * `rotated_rect` 的角点由中心 + 长宽 + `rotation` 旋转得到，
 * 顺序固定为「左上、右上、右下、左下」（未旋转时），
 * 旋转手柄依赖 `corners[0] → corners[1]` 这条顶边算朝向，不能改顺序。
 */
export function shapeCorners(obj) {
  const points = obj.points || [];
  if (obj.shape_type === "rectangle" && points.length >= 2) {
    const [a, b] = points;
    return [
      [a[0], a[1]],
      [b[0], a[1]],
      [b[0], b[1]],
      [a[0], b[1]],
    ];
  }
  if (obj.shape_type === "rotated_rect" && points.length >= 2) {
    const [a, b] = points;
    const cx = (a[0] + b[0]) / 2;
    const cy = (a[1] + b[1]) / 2;
    const angle = Number(obj.rotation) || 0;
    const halfW = Math.abs(b[0] - a[0]) / 2;
    const halfH = Math.abs(b[1] - a[1]) / 2;
    const corners = [];
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const rx = sx * halfW;
      const ry = sy * halfH;
      corners.push([
        cx + rx * Math.cos(angle) - ry * Math.sin(angle),
        cy + rx * Math.sin(angle) + ry * Math.cos(angle),
      ]);
    }
    return corners;
  }
  return points;
}

/** 一组点的包围盒（图像坐标）。空数组返回 `null`，避免产出 0 尺寸的假框。 */
export function bboxOf(points) {
  if (!points || !points.length) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * 命中测试：返回命中的对象。
 *
 * `objects` 按绘制顺序传入（后画的在上层），函数从后往前找——
 * 用户点哪个就应该选中看到的那个。点在形状内部也算命中，
 * 线段和点只能靠距离。
 *
 * `hidden` 是隐藏对象的 id 集合（`Set` 或 null）；隐藏的不能被点中。
 */
export function hitTest(point, { objects = [], scale = 1, hidden = null, tolerance = HIT_TOLERANCE } = {}) {
  const safeScale = Number(scale) > 0 ? Number(scale) : 1;
  const tol = tolerance / safeScale;
  for (let i = objects.length - 1; i >= 0; i -= 1) {
    const obj = objects[i];
    if (!obj) continue;
    if (hidden && hidden.has(obj.id)) continue;
    const points = obj.points || [];
    if (!points.length) continue;

    if (obj.shape_type === "point") {
      if (Math.hypot(point.x - points[0][0], point.y - points[0][1]) <= tol + POINT_SLACK) return obj;
      continue;
    }
    if (obj.shape_type === "line") {
      for (let k = 1; k < points.length; k += 1) {
        if (distanceToSegment(point, points[k - 1], points[k]) <= tol) return obj;
      }
      continue;
    }
    if (obj.shape_type === "keypoints") {
      for (const p of points) {
        if (Math.hypot(point.x - p[0], point.y - p[1]) <= tol + POINT_SLACK - 1) return obj;
      }
      continue;
    }
    const corners = shapeCorners(obj);
    if (corners.length >= 3 && pointInPolygon(point, corners)) return obj;
    // 内部没点中时再判边框，这样"框挨着但没套住"也能选中
    for (let k = 1; k <= corners.length; k += 1) {
      if (distanceToSegment(point, corners[k - 1], corners[k % corners.length]) <= tol) return obj;
    }
  }
  return null;
}

/** 旋转框顶部外侧的圆点（图像坐标）。普通矩形没有这个点，所以不能转。 */
export function rotateHandlePoint(obj, scale = 1) {
  if (!obj || obj.shape_type !== "rotated_rect") return null;
  const corners = shapeCorners(obj);
  if (corners.length !== 4) return null;
  const mid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
  const [a, b] = obj.points || [];
  if (!a || !b) return null;
  const cx = (a[0] + b[0]) / 2;
  const cy = (a[1] + b[1]) / 2;
  const dx = mid[0] - cx;
  const dy = mid[1] - cy;
  const len = Math.hypot(dx, dy) || 1;
  const gap = ROTATE_KNOB_GAP / (Number(scale) > 0 ? Number(scale) : 1);
  return [mid[0] + (dx / len) * gap, mid[1] + (dy / len) * gap];
}

/**
 * 手柄命中：返回 `{ kind, index }` 或 `null`。
 *
 * kind 取值：`rotate`（旋转圆点）、`corner`（矩形四角）、`vertex`（顶点）、
 * `insert`（多边形边中点，插新顶点）。矩形先判旋转点再判角点，
 * 否则圆点贴在顶边上时永远选不中。
 */
export function hitHandle(obj, point, { scale = 1 } = {}) {
  const safeScale = Number(scale) > 0 ? Number(scale) : 1;
  const tolerance = (HANDLE_RADIUS + 3) / safeScale;
  if (obj.shape_type === "rectangle" || obj.shape_type === "rotated_rect") {
    const knob = rotateHandlePoint(obj, safeScale);
    if (knob && Math.hypot(point.x - knob[0], point.y - knob[1]) <= tolerance + 2 / safeScale) {
      return { kind: "rotate" };
    }
    const corners = shapeCorners(obj);
    for (let index = 0; index < corners.length; index += 1) {
      if (Math.hypot(point.x - corners[index][0], point.y - corners[index][1]) <= tolerance) {
        return { kind: "corner", index };
      }
    }
    return null;
  }
  const points = obj.points || [];
  for (let index = 0; index < points.length; index += 1) {
    if (Math.hypot(point.x - points[index][0], point.y - points[index][1]) <= tolerance) {
      return { kind: "vertex", index };
    }
  }
  if (obj.shape_type === "polygon") {
    for (let index = 0; index < points.length; index += 1) {
      const next = (index + 1) % points.length;
      const mid = [(points[index][0] + points[next][0]) / 2, (points[index][1] + points[next][1]) / 2];
      if (Math.hypot(point.x - mid[0], point.y - mid[1]) <= EDGE_INSERT_RADIUS / safeScale) {
        return { kind: "insert", index };
      }
    }
  }
  return null;
}

/** 矩形边中点命中（整体拖边）。非矩形恒为 `null`。 */
export function hitEdge(obj, point, { scale = 1 } = {}) {
  if (obj.shape_type !== "rectangle" && obj.shape_type !== "rotated_rect") return null;
  const corners = shapeCorners(obj);
  if (corners.length !== 4) return null;
  const safeScale = Number(scale) > 0 ? Number(scale) : 1;
  const tolerance = EDGE_HIT_RADIUS / safeScale;
  for (let index = 0; index < 4; index += 1) {
    const a = corners[index];
    const b = corners[(index + 1) % 4];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (Math.hypot(point.x - mid[0], point.y - mid[1]) <= tolerance) return { kind: "edge", index };
  }
  return null;
}

/**
 * 吸附：闭合吸附 + 顶点吸附。
 *
 * 契约：**永远**返回 `{ point: [x, y], snap }`。
 * 调用方一律按数组取 `point[0] / point[1]`；如果某个分支返回 `{x, y}` 对象，
 * 取到的就是 undefined，代价是画出"空点"标注并且拖拽画框永远收不了尾。
 *
 * - `enabled=false`：原样返回，不做任何吸附；
 * - `allowClose=true` 且 `pending` 已有 ≥3 点：靠近起点直接吸上去，
 *   否则"精确点回起点"几乎做不到；
 * - `candidates` 是参与顶点吸附的对象列表（通常是当前图的全部标注）。
 */
export function snapPoint(
  point,
  { enabled = true, candidates = [], pending = [], scale = 1, allowClose = false, radius = SNAP_RADIUS } = {},
) {
  const origin = [point.x, point.y];
  if (!enabled) return { point: origin, snap: null };
  const safeScale = Number(scale) > 0 ? Number(scale) : 1;
  const limit = radius / safeScale;

  if (allowClose && pending.length >= 3) {
    const first = pending[0];
    if (Math.hypot(point.x - first[0], point.y - first[1]) <= limit) {
      return { point: [first[0], first[1]], snap: first };
    }
  }

  let best = null;
  let bestDistance = limit;
  for (const obj of candidates) {
    for (const p of (obj && obj.points) || []) {
      const distance = Math.hypot(point.x - p[0], point.y - p[1]);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = p;
      }
    }
  }
  if (best) return { point: [best[0], best[1]], snap: best };
  return { point: origin, snap: null };
}

/** 同时兼容 `{x, y}` 与 `[x, y]` 两种点表示（调用方来源不同，混用是历史遗留）。 */
function xy(p) {
  return Array.isArray(p) ? [p[0], p[1]] : [p.x, p.y];
}

/** 线段正交吸附：按住 Shift 时把端点约束到水平或垂直，返回 `[x, y]`。 */
export function constrainOrthogonal(anchor, point) {
  const [px, py] = xy(point);
  const ax = anchor[0];
  const ay = anchor[1];
  const dx = Math.abs(px - ax);
  const dy = Math.abs(py - ay);
  return dx >= dy ? [px, ay] : [ax, py];
}

/** 点是否落在某个形状的手柄上（供拖拽起手判定，含旋转点）。 */
export function hasRotateHandle(obj) {
  return Boolean(obj && obj.shape_type === "rotated_rect" && (obj.points || []).length >= 2);
}
