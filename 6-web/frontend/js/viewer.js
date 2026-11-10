// 可复用图像查看器 —— 零依赖，浏览器与 Node 均可导入。
//
// 为什么单独抽一个模块：
//   推理页与标注页都需要「光标锚点缩放 + 左键拖动平移」这套手感。
//   若各写一遍，两处实现迟早会出现细微差异（比如一个把缩放中心算错），
//   而那种差异几乎不可能在 code review 里看出来，只有用户觉得"手感变了"。
//   因此变换数学只写在这里，标注画布（变更 3）直接复用导出的纯函数。
//
// 模块边界：
//   纯函数（可直接在 Node 里单测，不碰 DOM）
//     clampScale / computeZoomTransform / normalizeWheelDelta / screenToImage
//   组件（只有调用时才触碰 DOM，因此 Node 导入本模块不会报错）
//     createViewer / createViewerStrip
//
// 坐标系约定（全局唯一，改之前请先确认所有调用点）：
//     p_screen = p_image * scale + offset
//   即 scale 与 offset 描述的是「图像坐标 → 视口坐标」的仿射变换。

// ---------------------------------------------------------------------------
// 1. 常量
// ---------------------------------------------------------------------------

/** 用户可缩放的最小倍率。实际的每张图会与"适应窗口倍率"取更小者，见 effectiveMin。 */
export const MIN_SCALE = 0.1;
/** 用户可缩放的最大倍率。再大只是位图插值放大，没有信息增益。 */
export const MAX_SCALE = 20;
/** 一个滚轮事件最多等效的像素增量，防止某些设备一次事件产生巨大跳变。 */
const MAX_WHEEL_DELTA = 300;
/** 滚轮增量 → 缩放指数的系数。经验值：一格标准滚轮（100）约缩放 18%。 */
const WHEEL_ZOOM_K = 0.002;
/** 视口尺寸变化的重新适应窗口，带一点防抖余量（毫秒）。 */
const RESIZE_DEBOUNCE = 120;

// ---------------------------------------------------------------------------
// 2. 纯函数
// ---------------------------------------------------------------------------

/**
 * 把倍率钳制到允许区间。
 * @param {number} scale 期望倍率
 * @param {number} min 下界，默认 MIN_SCALE；查看器会传入按图像计算的动态下界
 * @param {number} max 上界
 * @returns {number} 钳制后的倍率
 */
export function clampScale(scale, min = MIN_SCALE, max = MAX_SCALE) {
  if (!Number.isFinite(scale)) return min;
  if (scale < min) return min;
  if (scale > max) return max;
  return scale;
}

/**
 * 以屏幕上的某个点为锚点做缩放：保证该点下方的图像像素在缩放后仍停留在原屏幕位置。
 *
 * 推导（p_screen = p_image * s + t）：
 *   p_image = (cursor - t0) / s0
 *   t1      = cursor - p_image * s1
 *
 * Qt 版的错误示范是只改倍率不改平移，表现为"越缩放，光标下的东西跑得越远"。
 *
 * @param {{x:number,y:number}} cursor 视口内的光标坐标（像素）
 * @param {number} scale0 缩放前倍率
 * @param {{x:number,y:number}} offset0 缩放前平移
 * @param {number} scale1 缩放后倍率
 * @returns {{scale:number, offset:{x:number,y:number}}}
 */
export function computeZoomTransform(cursor, scale0, offset0, scale1) {
  const s0 = Number.isFinite(scale0) && scale0 > 0 ? scale0 : 1;
  const s1 = Number.isFinite(scale1) && scale1 > 0 ? scale1 : s0;
  const imageX = (cursor.x - offset0.x) / s0;
  const imageY = (cursor.y - offset0.y) / s0;
  return {
    scale: s1,
    offset: {
      x: cursor.x - imageX * s1,
      y: cursor.y - imageY * s1,
    },
  };
}

/**
 * 屏幕坐标 → 图像坐标。用于命中测试、叠加层换算。
 * @param {{x:number,y:number}} point
 * @param {number} scale
 * @param {{x:number,y:number}} offset
 * @returns {{x:number,y:number}}
 */
export function screenToImage(point, scale, offset) {
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1;
  return {
    x: (point.x - offset.x) / s,
    y: (point.y - offset.y) / s,
  };
}

/**
 * 位置 + 倍率 → 图像的屏幕包围盒。用于导航缩略图的视口框。
 * @param {number} scale
 * @param {{x:number,y:number}} offset
 * @param {number} imageWidth
 * @param {number} imageHeight
 * @returns {{x:number,y:number,width:number,height:number}}
 */
export function imageToScreenRect(scale, offset, imageWidth, imageHeight) {
  return {
    x: offset.x,
    y: offset.y,
    width: imageWidth * scale,
    height: imageHeight * scale,
  };
}

/**
 * 把滚轮事件的增量归一化为像素增量。
 * 触控板、鼠标滚轮、以及把滚轮事件映射成"行/页"的设备，deltaMode 与量级都不同；
 * 不归一化会出现"同一个滚轮，换台机器缩放速度差十倍"。
 * @param {WheelEvent|{deltaY:number, deltaMode?:number}} event
 * @param {number} pageHeight 用于 page 模式换算的视口高度
 * @returns {number} 归一化后的像素增量（正数表示向下滚 → 缩小）
 */
export function normalizeWheelDelta(event, pageHeight = 800) {
  const raw = Number.isFinite(event.deltaY) ? event.deltaY : 0;
  const mode = event.deltaMode || 0;
  let pixels;
  if (mode === 1) {
    pixels = raw * 16;            // DOM_DELTA_LINE
  } else if (mode === 2) {
    pixels = raw * (pageHeight || 800); // DOM_DELTA_PAGE
  } else {
    pixels = raw;                 // DOM_DELTA_PIXEL
  }
  if (pixels > MAX_WHEEL_DELTA) return MAX_WHEEL_DELTA;
  if (pixels < -MAX_WHEEL_DELTA) return -MAX_WHEEL_DELTA;
  return pixels;
}

/**
 * 像素增量 → 缩放倍率因子。
 * 用指数映射而不是线性加减：这样"放大两格再缩小两格"能精确回到原倍率，
 * 线性方案在两端会被钳制吃掉步长，回不去。
 * @param {number} delta 归一化后的滚轮像素增量
 * @returns {number} 倍率因子
 */
export function wheelDeltaToFactor(delta) {
  return Math.exp(-delta * WHEEL_ZOOM_K);
}

// ---------------------------------------------------------------------------
// 3. 组件
// ---------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * 创建一个图像查看器窗口。
 *
 * @param {object} options
 * @param {HTMLElement} options.mount 容纳窗口的容器（通常是 .viewer-strip）
 * @param {string} options.id 窗口标识，用于状态持久化与外部引用
 * @param {string} [options.title] 标题栏文案
 * @param {string} [options.foot] 底部说明文案
 * @param {string} [options.aspect] CSS aspect-ratio，默认 4 / 3
 * @param {string} [options.emptyText] 无图时的占位文案
 * @returns {object} 查看器实例
 */
export function createViewer(options) {
  const {
    mount,
    id,
    title = "",
    foot = "",
    aspect = "",
    emptyText = "暂无图像",
  } = options;

  if (!mount) throw new Error("createViewer: 缺少 mount 容器");
  if (!id) throw new Error("createViewer: 缺少 id");

  // ---- 3.1 结构 ----
  const root = document.createElement("div");
  root.className = "viewer";
  root.dataset.viewerId = id;

  const head = document.createElement("div");
  head.className = "viewer-head";

  const titleEl = document.createElement("span");
  titleEl.className = "viewer-title";
  titleEl.textContent = title;
  titleEl.title = title;

  const tools = document.createElement("div");
  tools.className = "viewer-tools";

  const scaleEl = document.createElement("span");
  scaleEl.className = "viewer-scale";
  scaleEl.textContent = "100%";

  const stage = document.createElement("div");
  stage.className = "viewer-stage";

  const plane = document.createElement("div");
  plane.className = "viewer-plane";

  const img = document.createElement("img");
  img.alt = title || "图像";
  img.draggable = false;

  const overlay = document.createElement("div");
  overlay.className = "viewer-overlay";

  const emptyEl = document.createElement("div");
  emptyEl.className = "viewer-empty";
  emptyEl.textContent = emptyText;

  plane.appendChild(img);
  plane.appendChild(overlay);
  stage.appendChild(plane);
  stage.appendChild(emptyEl);

  head.appendChild(titleEl);
  head.appendChild(tools);

  root.appendChild(head);
  root.appendChild(stage);

  let footEl = null;
  if (foot) {
    footEl = document.createElement("div");
    footEl.className = "viewer-foot";
    footEl.textContent = foot;
    root.appendChild(footEl);
  }

  if (aspect) stage.style.setProperty("--viewer-aspect", aspect);

  // ---- 3.2 状态 ----
  let scale = 1;
  let offset = { x: 0, y: 0 };
  let imageWidth = 0;
  let imageHeight = 0;
  let autoFit = true;          // 未手动操作过时，视口尺寸变化自动重新适应
  let dragging = false;
  let dragLast = { x: 0, y: 0 };
  let activePointer = null;
  let wheelAccum = 0;
  let wheelCursor = null;
  let wheelFrame = 0;
  let resizeTimer = 0;
  let currentSrc = "";

  // ---- 3.3 变换应用 ----
  function effectiveMin() {
    // 超大图在小视口里，适应窗口所需倍率可能低于 MIN_SCALE。
    // 此时以"能看全"为优先，用适应倍率作为下界，否则用户永远看不到全图。
    if (imageWidth <= 0) return MIN_SCALE;
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return MIN_SCALE;
    const fitScale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
    return Math.min(MIN_SCALE, fitScale);
  }

  function apply() {
    plane.style.transform = `translate(${offset.x}px, ${offset.y}px) scale(${scale})`;
    scaleEl.textContent = `${Math.round(scale * 100)}%`;
    // 通知外层"倍率变了"。叠加层靠这个事件把标签尺寸反向缩放，
    // 使文字在屏幕上恒定可读——否则 24% 时标签会小到看不清，
    // 1000% 时又一个字占满半个屏幕。
    root.dispatchEvent(new CustomEvent("viewer-zoom", {
      bubbles: true,
      detail: { id, scale, offset: { x: offset.x, y: offset.y } },
    }));
  }

  function setTransform(nextScale, nextOffset) {
    scale = nextScale;
    offset = { x: nextOffset.x, y: nextOffset.y };
    apply();
  }

  // ---- 3.4 适应窗口 / 1:1 ----
  function fit() {
    if (imageWidth <= 0 || imageHeight <= 0) return;
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const raw = Math.min(rect.width / imageWidth, rect.height / imageHeight);
    const nextScale = clampScale(raw, Math.min(MIN_SCALE, raw), MAX_SCALE);
    setTransform(nextScale, {
      x: (rect.width - imageWidth * nextScale) / 2,
      y: (rect.height - imageHeight * nextScale) / 2,
    });
    autoFit = true;
  }

  function actualSize() {
    if (imageWidth <= 0 || imageHeight <= 0) return;
    const rect = stage.getBoundingClientRect();
    setTransform(1, {
      x: (rect.width - imageWidth) / 2,
      y: (rect.height - imageHeight) / 2,
    });
    autoFit = false;
  }

  function zoomAt(cursor, targetScale) {
    const nextScale = clampScale(targetScale, effectiveMin(), MAX_SCALE);
    if (nextScale === scale) return;
    const next = computeZoomTransform(cursor, scale, offset, nextScale);
    setTransform(next.scale, next.offset);
    autoFit = false;
  }

  function zoomBy(factor) {
    const rect = stage.getBoundingClientRect();
    zoomAt(
      { x: rect.width / 2, y: rect.height / 2 },
      scale * factor,
    );
  }

  // ---- 3.5 滚轮缩放 ----
  // 累积 + requestAnimationFrame 合并：一个屏幕帧（约 16.7ms）内的多个滚轮事件
  // 只应用一次，既满足"≥16ms 不重复处理"，又比直接丢弃事件更跟手。
  function flushWheel() {
    wheelFrame = 0;
    if (!wheelCursor || wheelAccum === 0) return;
    const factor = wheelDeltaToFactor(wheelAccum);
    wheelAccum = 0;
    zoomAt(wheelCursor, scale * factor);
  }

  function onWheel(event) {
    // 必须阻断默认行为：否则滚轮会连带滚动整个页面，用户以为程序出错。
    event.preventDefault();
    if (imageWidth <= 0) return;

    const rect = stage.getBoundingClientRect();
    wheelCursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    wheelAccum += normalizeWheelDelta(event, rect.height);

    if (!wheelFrame) {
      wheelFrame = requestAnimationFrame(flushWheel);
    }
  }

  // ---- 3.6 拖动平移 ----
  function onPointerDown(event) {
    if (event.button !== 0) return;
    if (imageWidth <= 0) return;
    dragging = true;
    activePointer = event.pointerId;
    dragLast = { x: event.clientX, y: event.clientY };
    stage.classList.add("dragging");
    // 捕获指针：拖出窗口边界后仍然持续收到 move，松手时一定收到 up
    if (stage.setPointerCapture) {
      try { stage.setPointerCapture(event.pointerId); } catch { /* 某些合成事件无法捕获 */ }
    }
    event.preventDefault();
  }

  function onPointerMove(event) {
    if (!dragging || event.pointerId !== activePointer) return;
    const dx = event.clientX - dragLast.x;
    const dy = event.clientY - dragLast.y;
    dragLast = { x: event.clientX, y: event.clientY };
    offset = { x: offset.x + dx, y: offset.y + dy };
    autoFit = false;
    apply();
  }

  function endDrag(event) {
    if (!dragging) return;
    if (event && event.pointerId !== activePointer) return;
    dragging = false;
    stage.classList.remove("dragging");
    if (stage.releasePointerCapture && activePointer != null) {
      try { stage.releasePointerCapture(activePointer); } catch { /* 已释放 */ }
    }
    activePointer = null;
  }

  // ---- 3.7 图与叠加层 ----
  function onImageLoad() {
    imageWidth = img.naturalWidth || 0;
    imageHeight = img.naturalHeight || 0;
    emptyEl.hidden = imageWidth > 0;
    // 叠加层尺寸与图像像素尺寸一致，这样 SVG 用 viewBox 即可免换算
    overlay.style.width = `${imageWidth}px`;
    overlay.style.height = `${imageHeight}px`;
    plane.style.width = `${imageWidth}px`;
    plane.style.height = `${imageHeight}px`;
    fit();
  }

  img.addEventListener("load", onImageLoad);

  /**
   * 设置图像。传入空值则清空并显示占位。
   * @param {string} src 图像地址（http、blob:、data: 均可）
   */
  function setImage(src) {
    if (!src) {
      clear();
      return;
    }
    if (src === currentSrc) return;
    currentSrc = src;
    // 切图前显式断掉旧图的引用，避免大图位图在内存里累积
    img.removeAttribute("src");
    plane.style.width = "0px";
    plane.style.height = "0px";
    overlay.replaceChildren();
    imageWidth = 0;
    imageHeight = 0;
    emptyEl.hidden = true;
    img.src = src;
  }

  /**
   * 设置图像的显示不透明度（0~1）。
   *
   * 用途：异常叠加图默认是"原图 + 半透明热力"的成品，用户有时想把热力压淡一点
   * 看清下面的纹理。只调图片本身，不影响叠加层（框/标签）的清晰度。
   */
  function setImageOpacity(value) {
    const opacity = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 1;
    img.style.opacity = String(opacity);
  }

  function clear() {
    currentSrc = "";
    img.removeAttribute("src");
    plane.style.width = "0px";
    plane.style.height = "0px";
    overlay.replaceChildren();
    imageWidth = 0;
    imageHeight = 0;
    scale = 1;
    offset = { x: 0, y: 0 };
    apply();
    emptyEl.hidden = false;
    autoFit = true;
  }

  /**
   * 挂载叠加层（例如检测框 SVG）。叠加层与图像共用坐标系，随缩放一起变换。
   * @param {Node|null} node
   */
  function setOverlay(node) {
    overlay.replaceChildren();
    if (node) overlay.appendChild(node);
  }

  /** 叠加层容器，交给调用方自行增删子节点。 */
  function getOverlay() {
    return overlay;
  }

  // ---- 3.8 工具条 ----
  function makeToolButton(label, titleText, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "viewer-btn";
    button.textContent = label;
    button.title = titleText;
    button.setAttribute("aria-label", titleText);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      onClick();
    });
    tools.appendChild(button);
    return button;
  }

  makeToolButton("−", "缩小", () => zoomBy(1 / 1.25));
  makeToolButton("+", "放大", () => zoomBy(1.25));
  makeToolButton("1:1", "原始像素大小", actualSize);
  const fitButton = makeToolButton("FIT", "适应窗口", fit);
  const hideButton = makeToolButton("×", "隐藏该窗口", () => {
    root.dispatchEvent(new CustomEvent("viewer-hide", { bubbles: true, detail: { id } }));
  });
  hideButton.classList.add("viewer-btn-hide");

  // 倍率读数放在最右：左边是操作、右边是当前状态，与工业 HUD 的读数区一致
  tools.appendChild(scaleEl);

  // ---- 3.9 事件绑定 ----
  stage.addEventListener("wheel", onWheel, { passive: false });
  stage.addEventListener("pointerdown", onPointerDown);
  stage.addEventListener("pointermove", onPointerMove);
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener("lostpointercapture", endDrag);
  stage.addEventListener("dblclick", () => fit());
  // 拖动时禁止浏览器做文本/图像选择
  stage.addEventListener("dragstart", (event) => event.preventDefault());

  const observer = new ResizeObserver(() => {
    if (!autoFit) return;
    clearTimeout(resizeTimer);
    // 定时器里必须再判一次 autoFit：用户可能在防抖窗口内就手动缩放了，
    // 此时不能把视图抢回去复位。
    resizeTimer = setTimeout(() => {
      if (autoFit) fit();
    }, RESIZE_DEBOUNCE);
  });
  observer.observe(stage);

  mount.appendChild(root);
  clear();

  return {
    id,
    el: root,
    stage,
    setImage,
    clear,
    setImageOpacity,
    setOverlay,
    getOverlay,
    fit,
    actualSize,
    zoomBy,
    getScale: () => scale,
    getOffset: () => ({ ...offset }),
    getImageSize: () => ({ width: imageWidth, height: imageHeight }),
    setAvailable(available) {
      root.hidden = !available;
    },
    setVisible(visible) {
      root.hidden = !visible;
    },
    isVisible: () => !root.hidden,
    setTitle(text) {
      titleEl.textContent = text;
      titleEl.title = text;
    },
    setFoot(text) {
      if (footEl) footEl.textContent = text;
    },
  };
}

// ---------------------------------------------------------------------------
// 4. 多窗口容器
// ---------------------------------------------------------------------------

const STORAGE_PREFIX = "trtforge.viewer";

function readToggle(stripId, viewerId) {
  try {
    const raw = localStorage.getItem(`${STORAGE_PREFIX}.${stripId}.${viewerId}.enabled`);
    if (raw === null) return null;
    return raw === "1";
  } catch {
    // 隐私模式或存储被禁用：退化为"总是显示"，不影响功能
    return null;
  }
}

function writeToggle(stripId, viewerId, enabled) {
  try {
    localStorage.setItem(
      `${STORAGE_PREFIX}.${stripId}.${viewerId}.enabled`,
      enabled ? "1" : "0",
    );
  } catch {
    /* 写不进去就算了，本次会话内状态仍然正确 */
  }
}

/**
 * 创建一个多窗口容器：自适应网格布局 + 一组开关胶囊 + 空状态。
 *
 * @param {object} options
 * @param {HTMLElement} options.mount 挂载点（通常是结果卡片内的一个 div）
 * @param {string} options.id 容器标识，用于隔离 localStorage 键名
 * @param {Array<{id:string,title:string,foot?:string,enabled?:boolean,available?:boolean}>} options.windows
 * @param {string} [options.emptyText] 全部窗口关闭时的提示
 * @returns {object} 容器实例
 */
export function createViewerStrip(options) {
  const {
    mount,
    id,
    windows = [],
    emptyText = "全部窗口已关闭，点击上方开关重新显示",
  } = options;

  if (!mount) throw new Error("createViewerStrip: 缺少 mount 容器");
  if (!id) throw new Error("createViewerStrip: 缺少 id");

  const root = document.createElement("div");
  root.className = "viewer-root";

  const chips = document.createElement("div");
  chips.className = "viewer-chips";

  const strip = document.createElement("div");
  strip.className = "viewer-strip";

  const emptyEl = document.createElement("div");
  emptyEl.className = "viewer-strip-empty hidden";
  emptyEl.textContent = emptyText;

  root.appendChild(chips);
  root.appendChild(strip);
  root.appendChild(emptyEl);
  mount.appendChild(root);

  const viewers = new Map();
  const chipMap = new Map();
  const availableMap = new Map();

  for (const spec of windows) {
    const viewer = createViewer({
      mount: strip,
      id: spec.id,
      title: spec.title,
      foot: spec.foot,
      aspect: spec.aspect,
      emptyText: spec.emptyText || "暂无图像",
    });
    viewers.set(spec.id, viewer);

    const persisted = readToggle(id, spec.id);
    const enabled = persisted === null ? spec.enabled !== false : persisted;
    availableMap.set(spec.id, spec.available !== false);

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip-toggle";
    chip.textContent = spec.title;
    chip.title = `显示/隐藏 ${spec.title}`;
    chip.setAttribute("aria-pressed", String(enabled));
    chip.addEventListener("click", () => setEnabled(spec.id, !isEnabled(spec.id)));
    chips.appendChild(chip);
    chipMap.set(spec.id, chip);

    viewer.el.addEventListener("viewer-hide", () => setEnabled(spec.id, false));
  }

  function isEnabled(viewerId) {
    return chipMap.get(viewerId)?.getAttribute("aria-pressed") === "true";
  }

  function setEnabled(viewerId, enabled) {
    const chip = chipMap.get(viewerId);
    const viewer = viewers.get(viewerId);
    if (!chip || !viewer) return;
    chip.setAttribute("aria-pressed", String(enabled));
    writeToggle(id, viewerId, enabled);
    syncLayout();
  }

  /** 无内容可显示的窗口（如尚未产生异常区域图）应从布局中移除，但保留其开关状态。 */
  function setAvailable(viewerId, available) {
    if (!availableMap.has(viewerId)) return;
    availableMap.set(viewerId, available);
    syncLayout();
  }

  function visibleIds() {
    return Array.from(viewers.keys()).filter(
      (viewerId) => availableMap.get(viewerId) && isEnabled(viewerId),
    );
  }

  function syncLayout() {
    const visible = new Set(visibleIds());

    for (const [viewerId, viewer] of viewers) {
      const shouldShow = visible.has(viewerId);
      viewer.el.hidden = !shouldShow;
      const chip = chipMap.get(viewerId);
      if (chip) chip.classList.toggle("hidden", !availableMap.get(viewerId));
    }

    const count = visible.size;
    if (count > 3) {
      strip.classList.add("is-many");
      strip.style.setProperty("--viewer-cols", String(count));
    } else {
      strip.classList.remove("is-many");
      strip.style.removeProperty("--viewer-cols");
    }

    emptyEl.classList.toggle("hidden", count > 0);
    strip.classList.toggle("hidden", count === 0);

    // 尺寸变化后重新适应：隐藏再显示的窗口需要重算，否则保持旧的错位变换
    for (const viewerId of visible) {
      viewers.get(viewerId)?.fit();
    }
  }

  syncLayout();

  return {
    id,
    el: root,
    strip,
    chips,
    viewer: (viewerId) => viewers.get(viewerId),
    setEnabled,
    isEnabled,
    setAvailable,
    visibleIds,
    syncLayout,
  };
}

export { SVG_NS };
