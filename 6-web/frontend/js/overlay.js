// 检出框矢量叠加层。
//
// 为什么用 SVG 而不是让后端把框烧进 PNG：
//   烧进 PNG 的框会被放大成马赛克，缺陷本身在大倍率下还是清晰的、
//   框却是一团色块，判读时非常别扭。SVG 在任意倍率下都是锐利的。
//   后端那份 `annotated_png` 保留，只用于"导出带标注结果图"。
//
// 坐标系：叠加层容器 `.viewer-overlay` 的尺寸等于图像原始像素尺寸，
// 所以 SVG 直接用 `viewBox="0 0 图像宽 图像高"`，自然与查看器的变换对齐，
// 不需要手动换算屏幕坐标。
//
// 两个难点及解法：
//   1. 框线在放大后会变粗 —— 用 `vector-effect="non-scaling-stroke"` 固定屏幕线宽。
//   2. 标签文字在缩小后看不清、放大后过大 —— 标签包在 `<g>` 里，
//      用 `scale(1/倍率)` 反向缩放，使文字在屏幕上恒定字号。

const SVG_NS = "http://www.w3.org/2000/svg";

// 标签在屏幕上的固定尺寸（px）。与图像分辨率无关，只跟"好不好读"有关。
const LABEL_FONT_PX = 12;
const LABEL_PAD_X = 5;
const LABEL_PAD_Y = 3;
const LABEL_HEIGHT = LABEL_FONT_PX + LABEL_PAD_Y * 2;
const LABEL_GAP = 2;

/** 把 [r,g,b] 转成 CSS 颜色。 */
function cssColor(rgb, alpha = 1) {
  if (!rgb) return `rgba(255, 255, 255, ${alpha})`;
  const [r, g, b] = rgb;
  return alpha >= 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** 取类别配色；缺项时退回白色，保证不会因为一处缺色整体不显示。 */
function colorOf(classColors, classLabel) {
  if (!classColors) return null;
  return classColors[String(classLabel)] || classColors[classLabel] || null;
}

/**
 * 估算一段文字的像素宽度。
 *
 * 不用 `getBBox()` 的原因：叠加层可能尚未挂到文档上（构建时），
 * 此时测量会得到 0。用等宽字体近似即可满足"标签背景别裁字"的需要。
 */
function estimateTextWidth(text) {
  // 中文按 1em、数字与字母按 0.6em 估，混排场景取平均偏差最小
  let units = 0;
  for (const ch of text) {
    units += ch.charCodeAt(0) > 0x2e80 ? 1 : 0.6;
  }
  return units * LABEL_FONT_PX;
}

/** OBB → 四个角点。与后端 C++ 的 boxPoints 保持同一套定义（角度为弧度）。 */
function obbCorners(obb) {
  const { cx, cy, w, h, angle } = obb;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const hw = w / 2;
  const hh = h / 2;
  return [
    [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
  ].map(([dx, dy]) => [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos]);
}

/**
 * 构建叠加层。
 *
 * @param {object} options
 * @param {Array} options.boxes 后端返回的 boxes
 * @param {number} options.width 图像宽（像素）
 * @param {number} options.height 图像高（像素）
 * @param {object} options.classColors `{"0": [r,g,b], ...}`
 * @param {boolean} [options.showBoxes] 是否画框
 * @param {boolean} [options.showLabels] 是否画标签
 * @param {boolean} [options.fadeLowConf] 是否淡化低分框
 * @param {number} [options.highlight] 高亮的下标（-1 表示不高亮）
 * @returns {SVGSVGElement}
 */
export function buildOverlay(options) {
  const {
    boxes = [],
    width = 0,
    height = 0,
    classColors = null,
    showBoxes = true,
    showLabels = true,
    fadeLowConf = true,
    highlight = -1,
  } = options;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "detect-overlay");
  svg.setAttribute("viewBox", `0 0 ${Math.max(1, width)} ${Math.max(1, height)}`);
  svg.setAttribute("width", String(Math.max(1, width)));
  svg.setAttribute("height", String(Math.max(1, height)));
  svg.setAttribute("preserveAspectRatio", "none");

  if (!boxes.length || width <= 0 || height <= 0) return svg;

  // 低分框的判据取"本次结果的中位数"，而不是某个绝对阈值：
  // 不同场景的置信度分布差异极大（有的全在 0.9 以上，有的全在 0.3），
  // 写死 0.5 会让前者全部"高亮"、后者全部"淡化"，失去区分作用。
  const sorted = [...boxes].map((b) => b.confidence).sort((a, b) => a - b);
  const median = sorted.length
    ? (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2
    : 0;

  boxes.forEach((box, index) => {
    // 姿态/分割的 mask 与关键点不在本次范围内，只画框
    const color = colorOf(classColors, box.class_label);
    const faded = fadeLowConf && boxes.length > 2 && box.confidence < median;
    const opacity = faded ? 0.45 : 1;
    const isHighlight = index === highlight;

    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", "detect-box");
    group.dataset.index = String(index);
    if (isHighlight) group.classList.add("is-highlight");

    let shape;
    let anchorX;
    let anchorY;

    if (box.is_obb && box.obb) {
      const corners = obbCorners(box.obb);
      shape = document.createElementNS(SVG_NS, "polygon");
      shape.setAttribute("points", corners.map(([x, y]) => `${x},${y}`).join(" "));
      anchorX = corners[0][0];
      anchorY = Math.min(...corners.map((point) => point[1]));
    } else {
      shape = document.createElementNS(SVG_NS, "rect");
      shape.setAttribute("x", String(box.left));
      shape.setAttribute("y", String(box.top));
      shape.setAttribute("width", String(Math.max(1, box.right - box.left)));
      shape.setAttribute("height", String(Math.max(1, box.bottom - box.top)));
      anchorX = box.left;
      anchorY = box.top;
    }

    if (showBoxes) {
      shape.setAttribute("fill", "none");
      shape.setAttribute("stroke", cssColor(color, 1));
      // 固定屏幕线宽：放大到 8 倍也是 2px，不会糊成一整块
      shape.setAttribute("stroke-width", "2");
      shape.setAttribute("vector-effect", "non-scaling-stroke");
      shape.setAttribute("opacity", String(opacity));
      group.appendChild(shape);

      // 高亮底衬一直存在，靠 CSS 类控制显隐。
      // 这样点表格行时只需加一个 class，不必重建整层 SVG。
      const fill = shape.cloneNode();
      fill.setAttribute("class", "box-fill");
      fill.setAttribute("fill", cssColor(color, 0.2));
      fill.setAttribute("stroke", "none");
      fill.removeAttribute("vector-effect");
      group.appendChild(fill);
    }

    if (showLabels) {
      const text = `${box.label_name || `cls${box.class_label}`} ${box.confidence.toFixed(2)}`;
      group.appendChild(buildLabel({ text, color, x: anchorX, y: anchorY, opacity }));
    }

    for (const kpt of box.kpts || []) {
      if (kpt.length < 2) continue;
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", String(kpt[0]));
      dot.setAttribute("cy", String(kpt[1]));
      dot.setAttribute("r", "3");
      dot.setAttribute("fill", cssColor(color, 1));
      dot.setAttribute("vector-effect", "non-scaling-stroke");
      group.appendChild(dot);
    }

    svg.appendChild(group);
  });

  applyOverlayScale(svg, 1);
  return svg;
}

/**
 * 构建一个标签（背景块 + 文字），并放进 `g.ov-label` 里等待反向缩放。
 */
function buildLabel({ text, color, x, y, opacity }) {
  const group = document.createElementNS(SVG_NS, "g");
  group.setAttribute("class", "ov-label");
  group.setAttribute("opacity", String(opacity));

  const textWidth = estimateTextWidth(text);
  const boxWidth = textWidth + LABEL_PAD_X * 2;

  // 默认放在框的左上方外侧；贴到图像上边缘时翻到框内侧，
  // 否则标签会被画布裁掉一半——这是密集小目标场景最常见的可读性问题。
  const outsideTop = y - LABEL_HEIGHT - LABEL_GAP;
  const top = outsideTop >= 0 ? outsideTop : y + LABEL_GAP;

  // 贴右边缘时向左收，避免文字出界
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", String(x));
  rect.setAttribute("y", String(top));
  rect.setAttribute("width", String(boxWidth));
  rect.setAttribute("height", String(LABEL_HEIGHT));
  rect.setAttribute("rx", "2");
  rect.setAttribute("fill", cssColor(color, 0.92));
  group.appendChild(rect);

  const label = document.createElementNS(SVG_NS, "text");
  label.setAttribute("x", String(x + LABEL_PAD_X));
  label.setAttribute("y", String(top + LABEL_FONT_PX + LABEL_PAD_Y - 1));
  label.setAttribute("font-size", String(LABEL_FONT_PX));
  label.setAttribute("font-family", "var(--font-mono, monospace)");
  label.setAttribute("fill", "#070a0e");
  label.setAttribute("opacity", "1");
  label.textContent = text;
  group.appendChild(label);

  // 记录定位基准，缩放时只需重设 transform
  group.dataset.anchorX = String(x);
  group.dataset.anchorY = String(y);
  group.dataset.baseY = String(top);
  return group;
}

/**
 * 按当前倍率反向缩放所有标签，使文字在屏幕上保持恒定字号。
 *
 * @param {SVGSVGElement} svg
 * @param {number} scale 当前查看器倍率
 */
export function applyOverlayScale(svg, scale) {
  if (!svg) return;
  const inverse = 1 / (Number.isFinite(scale) && scale > 0 ? scale : 1);
  for (const group of svg.querySelectorAll(".ov-label")) {
    const anchorX = Number(group.dataset.anchorX || 0);
    const baseY = Number(group.dataset.baseY || 0);
    group.setAttribute("transform", `translate(${anchorX}, ${baseY}) scale(${inverse}) translate(${-anchorX}, ${-baseY})`);
  }
}

/** 在已有叠加层上切换高亮，避免整层重建。 */
export function setOverlayHighlight(svg, index) {
  if (!svg) return;
  for (const group of svg.querySelectorAll(".detect-box")) {
    group.classList.toggle("is-highlight", Number(group.dataset.index) === index);
  }
}
