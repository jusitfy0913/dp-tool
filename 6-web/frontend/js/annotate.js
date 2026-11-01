// 标注工作台：Canvas 2D 画布、六种形状编辑、撤销重做、自动保存、预标注草稿。
//
// 坐标系约定与推理页查看器完全一致（复用 viewer.js 的纯函数）：
//   p_screen = p_image * scale + offset
// 这样"在推理页把图放大看缺陷，再来这里标"的手感是同一个，不会突然变。
//
// 数据格式的唯一权威是 `.tfanno`（Qt 版的格式），本文件不定义任何私有字段：
// 扩展字段只有 difficult / flags / attributes 三个，且默认值不写盘。

import {
  ApiError,
  annotationImageMeta,
  annotationImageUrl,
  autoLabel,
  exportAnnotationProject,
  loadAnnotationProject,
  saveAnnotationProject,
  scanAnnotationImages,
} from "./api.js";
import {
  clampScale,
  computeZoomTransform,
  normalizeWheelDelta,
  screenToImage,
  wheelDeltaToFactor,
} from "./viewer.js";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/** 撤销历史深度。对齐 Qt 版的 50，兼顾内存与"够用"。 */
const HISTORY_LIMIT = 50;
/** 自动保存防抖间隔（毫秒）。改慢会丢更多编辑，改快会把磁盘打满。 */
const SAVE_DEBOUNCE = 3000;
/** 顶点手柄的屏幕半径（像素）。固定屏幕尺寸，缩放时不变。 */
const HANDLE_RADIUS = 5;
/** 命中测试的容差（屏幕像素）。太小点不中，太大容易误选。 */
const HIT_TOLERANCE = 6;
/** 吸附半径（屏幕像素）。 */
const SNAP_RADIUS = 9;
/** 画布绝对最小倍率，防止把图缩成看不见的一点。 */
const MIN_SCALE_ABS = 0.05;

const SHAPE_LABELS = {
  rectangle: "矩形",
  rotated_rect: "旋转框",
  polygon: "多边形",
  line: "线段",
  point: "点",
  keypoints: "关键点",
};

const TOOL_SHORTCUTS = {
  v: "select",
  r: "rectangle",
  o: "rotated_rect",
  p: "polygon",
  l: "line",
  t: "point",
  k: "keypoints",
};

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const state = {
  path: "",
  exists: false,
  project: null,
  meta: { rev: 0, mtime_ns: "0", size: 0 },
  dirty: false,
  saving: false,
  currentIndex: -1,
  selectedId: "",
  tool: "select",
  /** 正在绘制中的顶点（图像坐标）。画完即清空。 */
  pending: [],
  /** 预标注草稿，按图片 id 分组；只存在会话里，不落盘。 */
  drafts: new Map(),
  undo: [],
  redo: [],
  image: null,
  /** 图片解码序号：用于丢弃「后到的旧图」，见 loadCurrentImage。 */
  imageLoadSeq: 0,
  view: { scale: 1, offset: { x: 0, y: 0 } },
  /** 是否已经成功"适应窗口"过一次。用于在面板由隐藏变可见时补算倍率。 */
  viewValid: false,
  cursor: null,
  hover: null,
  opts: {
    showShapes: true,
    showLabels: true,
    showScore: false,
    crosshair: false,
    snap: true,
    fillAlpha: 0.18,
  },
  hidden: new Set(),
  /** 拖动会话：{ kind, index, start, original, startScreen, startOffset } */
  drag: null,
  saveTimer: 0,
  /** 在飞保存请求的 promise；flushSave 用它等待落地。 */
  saveTask: null,
  /**
   * 编辑序号：每次 markDirty 递增。
   *
   * 用途是「保存期间又改了东西」的检测——保存请求发出时记下序号，
   * 回来时若序号变了，说明这次存的是旧内容，必须保持 dirty 并重新排期，
   * 否则那批编辑既没写盘、也没被标脏，关页面就静默丢失。
   */
  editSeq: 0,
  /** 保存进行中又被请求保存：结束后补跑一次，不能直接丢掉。 */
  saveQueued: false,
  conflict: null,
};

const el = {};

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
function clone(value) {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

/**
 * 解析「每行 键: 值」的文本，返回键值对象。
 *
 * 只按**第一个**冒号切分，所以值里可以继续出现冒号（URL、时间戳）。
 * 键为空、或整行没有冒号的行直接跳过——用户多敲一个空行不该变成一条脏属性。
 */
function parseKeyedLines(text) {
  const result = {};
  for (const line of text.split("\n")) {
    const position = line.indexOf(":");
    if (position <= 0) continue;
    const key = line.slice(0, position).trim();
    if (!key) continue;
    result[key] = line.slice(position + 1).trim();
  }
  return result;
}

/** 把键值对象还原成「每行 键: 值」的文本，用于回填编辑框。 */
function formatKeyedLines(source) {
  return Object.entries(source || {})
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
}

function currentImage() {
  if (!state.project) return null;
  return state.project.images[state.currentIndex] || null;
}

function currentDrafts() {
  const image = currentImage();
  if (!image) return [];
  if (!state.drafts.has(image.id)) state.drafts.set(image.id, []);
  return state.drafts.get(image.id);
}

function classById(id) {
  const classes = state.project?.classes || [];
  return classes.find((item) => item.id === id) || classes[0] || { id: 0, name: "object", color: [31, 182, 255] };
}

function rgb(color, alpha = 1) {
  const [r, g, b] = color || [31, 182, 255];
  return `rgba(${r},${g},${b},${alpha})`;
}

/** 当前选中的对象，先查正式标注再查草稿。 */
function selectedObject() {
  const image = currentImage();
  if (!image || !state.selectedId) return null;
  return (
    image.annotations.find((item) => item.id === state.selectedId) ||
    currentDrafts().find((item) => item.id === state.selectedId) ||
    null
  );
}

function isDraft(obj) {
  return currentDrafts().some((item) => item.id === obj?.id);
}

function newObjectId() {
  // 不需要全局唯一，只需要在单个工程内不撞。
  return `o${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function setStatus(text, tone = "") {
  if (el.status) {
    el.status.textContent = text;
    el.status.className = `anno-status ${tone}`;
  }
}

// ---------------------------------------------------------------------------
// 历史（双栈，深度 50）
// ---------------------------------------------------------------------------
function snapshot(full = false) {
  if (!state.project) return null;
  if (full) return { full: true, project: clone(state.project) };
  const image = currentImage();
  return {
    full: false,
    index: state.currentIndex,
    annotations: image ? clone(image.annotations) : [],
  };
}

function applySnapshot(entry) {
  if (!entry) return false;
  let indexChanged = false;
  if (entry.full) {
    state.project = entry.project;
    const next = Math.min(Math.max(0, state.currentIndex), Math.max(0, state.project.images.length - 1));
    indexChanged = next !== state.currentIndex;
    state.currentIndex = next;
  } else {
    // 撤销/重做必须切到快照所属的那张图。
    // 只改数组不切视图的话，用户会看到"标注莫名消失"——其实改的是另一张图。
    if (entry.index !== state.currentIndex) {
      state.currentIndex = entry.index;
      indexChanged = true;
    }
    const image = state.project.images[entry.index];
    if (image) image.annotations = entry.annotations;
  }
  if (state.selectedId && !selectedObject()) state.selectedId = "";
  return indexChanged;
}

/**
 * 在**改动之前**调用：把当前状态压进 undo，并清空 redo。
 *
 * 顺序很重要——必须"先快照再改"，否则撤销会退回到改动后的状态，看起来像没生效。
 * 快照只存单图标注数组，不存图像位图：50 步深度才不会吃光内存。
 */
function pushHistory(full = false) {
  const entry = snapshot(full);
  if (!entry) return;
  state.undo.push(entry);
  if (state.undo.length > HISTORY_LIMIT) state.undo.shift();
  state.redo.length = 0;
  updateHistoryButtons();
}

function undo() {
  if (!state.undo.length) return false;
  const entry = state.undo.pop();
  state.redo.push(snapshot(entry.full));
  const indexChanged = applySnapshot(entry);
  markDirty();
  renderAll();
  if (indexChanged) loadCurrentImage();
  else renderCanvas();
  updateHistoryButtons();
  setStatus("已撤销");
  return true;
}

function redo() {
  if (!state.redo.length) return false;
  const entry = state.redo.pop();
  state.undo.push(snapshot(entry.full));
  const indexChanged = applySnapshot(entry);
  markDirty();
  renderAll();
  if (indexChanged) loadCurrentImage();
  else renderCanvas();
  updateHistoryButtons();
  setStatus("已重做");
  return true;
}

function updateHistoryButtons() {
  if (el.undo) el.undo.disabled = state.undo.length === 0;
  if (el.redo) el.redo.disabled = state.redo.length === 0;
}

// ---------------------------------------------------------------------------
// 编辑后处理：标脏、防抖保存、重绘
// ---------------------------------------------------------------------------
function afterEdit() {
  markDirty();
  renderAll();
  renderCanvas();
}

function markDirty() {
  state.dirty = true;
  state.editSeq += 1;
  scheduleSave();
  updateSaveState();
}

function scheduleSave() {
  if (state.saveTimer) clearTimeout(state.saveTimer);
  state.saveTimer = window.setTimeout(() => {
    state.saveTimer = 0;
    // 保存动作本身**不进撤销栈**：否则用户按 Ctrl+Z 会"撤销掉保存"，语义荒谬。
    saveProject({ silent: true });
  }, SAVE_DEBOUNCE);
  updateSaveState();
}

async function flushSave() {
  if (state.saveTimer) {
    clearTimeout(state.saveTimer);
    state.saveTimer = 0;
  }
  // 先等在飞的请求落地：否则下面可能拿还没更新的 rev 再存一次，白撞 409。
  if (state.saveTask) await state.saveTask.catch(() => {});
  if (state.dirty && state.project && state.path && !state.conflict) {
    await saveProject({ silent: true });
  }
}

// ---------------------------------------------------------------------------
// 读写工程
// ---------------------------------------------------------------------------
async function openProject(path, { discard = false } = {}) {
  if (!path) {
    setStatus("请先填写工程路径", "bad");
    return;
  }
  if (discard) {
    // 用户在冲突弹窗里选了"重新加载（丢弃我的修改）"：先把待保存状态清干净。
    // 否则下面 flushSave 会拿刚刚判过冲突的内容再存一次，弹窗会立刻重新弹出来。
    state.dirty = false;
    if (state.saveTimer) {
      clearTimeout(state.saveTimer);
      state.saveTimer = 0;
    }
  }
  await flushSave();
  try {
    const data = await loadAnnotationProject(path);
    adoptProject(data, path);
    setStatus(data.exists ? `已打开 rev=${data.meta.rev}` : "工程不存在，可点「新建」开始", data.exists ? "" : "warn");
  } catch (error) {
    setStatus(`打开失败：${error.message}`, "bad");
  }
}

function adoptProject(data, path) {
  state.path = data.path || path;
  state.exists = Boolean(data.exists);
  state.project = data.project;
  state.meta = data.meta || { rev: 0, mtime_ns: "0", size: 0 };
  state.dirty = false;
  state.undo.length = 0;
  state.redo.length = 0;
  state.drafts.clear();
  state.hidden.clear();
  state.selectedId = "";
  state.pending = [];
  state.viewValid = false;
  state.currentIndex = state.project.images.length ? 0 : -1;
  if (el.path) el.path.value = state.path;

  updateHistoryButtons();
  renderAll();
  if (state.currentIndex >= 0) loadCurrentImage();
  else renderCanvas();
}

async function createProject() {
  const path = (el.path?.value || "").trim();
  if (!path) {
    setStatus("请先填写要创建的 .tfanno 路径", "bad");
    return;
  }
  // 1) 先把手上的修改落盘：切工程不能顺手带走当前工程的未保存内容。
  await flushSave();
  if (state.dirty) {
    setStatus("当前工程还有未保存的修改，新建已取消", "bad");
    return;
  }

  // 2) 目标已存在就必须显式确认：新建等于用空工程覆盖磁盘上的旧内容。
  let existed = false;
  try {
    existed = Boolean((await loadAnnotationProject(path)).exists);
  } catch {
    /* 读不到就交给保存阶段去报错 */
  }
  if (existed && !window.confirm(`磁盘上已存在工程：\n${path}\n\n新建会用空工程覆盖它，确定继续？`)) {
    setStatus("已取消新建", "warn");
    return;
  }

  const project = {
    version: 1,
    task: "detect",
    rev: 0,
    classes: [{ id: 0, name: "object", color: [31, 182, 255] }],
    images: [],
  };

  // 3) 先写盘、成功后再接管内存状态。
  // 反过来的话，写失败会留下一个「内存里是新工程、磁盘上是旧工程」的假象，
  // 用户接着编辑并自动保存，就把旧工程悄悄覆盖掉了。
  state.saving = true;
  try {
    const result = await saveAnnotationProject({ path, project, rev: null, stamp: null, force: true });
    state.path = result.path || path;
    state.project = project;
    state.meta = { rev: result.rev, mtime_ns: result.meta.mtime_ns, size: result.meta.size };
    state.exists = true;
    state.dirty = false;
    state.currentIndex = -1;
    state.selectedId = "";
    state.undo.length = 0;
    state.redo.length = 0;
    if (el.path) el.path.value = state.path;
    updateHistoryButtons();
    renderAll();
    renderCanvas();
    setStatus(`已新建工程 rev=${result.rev}`, "ok");
  } catch (error) {
    setStatus(`新建失败：${error.message}`, "bad");
  } finally {
    state.saving = false;
    updateSaveState();
  }
}

/**
 * 保存工程。同一时刻只允许一个请求在飞：
 * 期间的新请求记成 `saveQueued`，等当前请求结束后补跑一次。
 *
 * 外层这层包装只做一件事：把在飞请求的 promise 挂到 `state.saveTask`，
 * 让 `flushSave` 能等它落地（切图、导出、冲突重载都依赖这一点）。
 */
function saveProject(options = {}) {
  if (!state.project || !state.path) return Promise.resolve();
  if (state.saving) {
    // 已有请求在飞：记一笔，等它结束后补存一次。
    // 直接丢弃会让这批编辑永远等不到保存（防抖定时器已被 flushSave 清掉）。
    state.saveQueued = true;
    return state.saveTask || Promise.resolve();
  }
  const task = runSaveProject(options).finally(() => {
    if (state.saveTask === task) state.saveTask = null;
  });
  state.saveTask = task;
  return task;
}

async function runSaveProject({ silent = false, force = false } = {}) {
  state.saving = true;
  const sentSeq = state.editSeq;
  try {
    const result = await saveAnnotationProject({
      path: state.path,
      project: state.project,
      rev: force ? null : state.meta.rev,
      stamp: force ? null : { mtime_ns: state.meta.mtime_ns, size: state.meta.size },
      force,
    });
    state.meta = { rev: result.rev, mtime_ns: result.meta.mtime_ns, size: result.meta.size };
    state.exists = true;
    // 请求体是发出时的快照：期间若又编辑过，这批内容并没有写盘，
    // 因此必须保持 dirty 并重新排期，不能无条件清掉。
    state.dirty = state.editSeq !== sentSeq;
    state.path = result.path || state.path;
    if (el.path) el.path.value = state.path;
    if (result.warnings?.length) {
      setStatus(`已保存 rev=${result.rev}，但有问题：${result.warnings[0]}`, "warn");
    } else if (!silent) {
      setStatus(`已保存 rev=${result.rev}`, "ok");
    }
    updateSaveState();
    renderImageList();
    if (state.dirty) scheduleSave();
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      showConflict(error);
    } else {
      setStatus(`保存失败：${error.message}`, "bad");
    }
  } finally {
    state.saving = false;
    if (state.saveQueued) {
      state.saveQueued = false;
      if (state.dirty && !state.conflict) saveProject({ silent: true });
    }
  }
}

function updateSaveState() {
  if (!el.saveState) return;
  if (!state.project) {
    el.saveState.textContent = "未打开工程";
    el.saveState.className = "anno-hint";
    return;
  }
  if (!state.exists) {
    el.saveState.textContent = "尚未保存";
    el.saveState.className = "anno-hint warn";
    return;
  }
  el.saveState.textContent = state.dirty ? `有未保存的修改 · rev=${state.meta.rev}` : `已保存 · rev=${state.meta.rev}`;
  el.saveState.className = `anno-hint ${state.dirty ? "warn" : ""}`;
}

function showConflict(error) {
  state.conflict = error;
  const payload = error.payload || {};
  const disk = payload.disk || {};
  if (el.conflictText) {
    el.conflictText.textContent =
      `${payload.message || error.message}\n\n磁盘当前 rev=${disk.rev}，你基于 rev=${state.meta.rev}。` +
      "\n「重新加载」会丢弃你这次的修改；「强制覆盖」会覆盖别人的保存。";
  }
  el.conflictModal?.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// 图片
// ---------------------------------------------------------------------------
async function loadCurrentImage() {
  const image = currentImage();
  if (!image) {
    state.image = null;
    renderCanvas();
    return;
  }
  // 解码是异步的：记下本次加载的序号，回调里比对。
  // 不加这层，快速切图时旧图后到会覆盖新图，画布上就会是"
  // 上一张的位图配这一张的标注"，坐标全错。
  state.imageLoadSeq += 1;
  const seq = state.imageLoadSeq;

  if (!image.width || !image.height) {
    try {
      const meta = await annotationImageMeta(image.path);
      image.width = meta.width;
      image.height = meta.height;
    } catch {
      /* 尺寸探测失败不阻塞显示，画布按已有宽高处理 */
    }
  }
  if (seq !== state.imageLoadSeq) return;  // 等元信息期间又切图了

  const bitmap = new Image();
  bitmap.onload = () => {
    if (seq !== state.imageLoadSeq) return;  // 旧图后到，丢弃
    state.image = bitmap;
    fitView();
    renderCanvas();
    setStatus(`${image.id} · ${image.width}×${image.height} · ${image.annotations.length} 个标注`);
  };
  bitmap.onerror = () => {
    if (seq !== state.imageLoadSeq) return;
    state.image = null;
    renderCanvas();
    setStatus(`图片无法显示：${image.path}（检查是否在 TRTFORGE_BROWSE_ROOTS 内）`, "bad");
  };
  bitmap.src = annotationImageUrl(image.path);
}

function fitView() {
  const stage = el.stage;
  const image = currentImage();
  if (!stage || !image || !image.width || !image.height) return;
  // 面板隐藏时 clientWidth 为 0，算出来的倍率毫无意义（会变成最小倍率）。
  // 这种情况不把它当成"已适应窗口"，等可见后再补一次。
  if (stage.clientWidth < 40 || stage.clientHeight < 40) {
    state.viewValid = false;
    return;
  }
  const pad = 24;
  const availableW = Math.max(80, stage.clientWidth - pad);
  const availableH = Math.max(80, stage.clientHeight - pad);
  const scale = clampScale(Math.min(availableW / image.width, availableH / image.height), MIN_SCALE_ABS, 20);
  state.view = {
    scale,
    offset: {
      x: (stage.clientWidth - image.width * scale) / 2,
      y: (stage.clientHeight - image.height * scale) / 2,
    },
  };
  state.viewValid = true;
  updateZoomLabel();
}

function updateZoomLabel() {
  if (el.zoom) el.zoom.textContent = `${Math.round(state.view.scale * 100)}%`;
}

// ---------------------------------------------------------------------------
// 画布尺寸与绘制
// ---------------------------------------------------------------------------
function resizeCanvas() {
  const canvas = el.canvas;
  const stage = el.stage;
  if (!canvas || !stage) return;
  // devicePixelRatio：不加这层，高分屏上框线会糊成一团。
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, stage.clientWidth);
  const height = Math.max(1, stage.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function renderAll() {
  renderClassList();
  renderDraftList();
  renderImageList();
  renderProps();
  updateSaveState();
  updateEmptyHint();
}

/** 没打开图片时在画布上盖一层提示；有图片就藏起来，避免文字压在图层上。 */
function updateEmptyHint() {
  if (!el.empty) return;
  const hasImage = Boolean(currentImage());
  el.empty.classList.toggle("hidden", hasImage);
}

function imageToScreen(point) {
  return {
    x: point[0] * state.view.scale + state.view.offset.x,
    y: point[1] * state.view.scale + state.view.offset.y,
  };
}

function toImage(point) {
  return screenToImage(point, state.view.scale, state.view.offset);
}

/** 形状的顶点（图像坐标）；矩形类的两点表示在这里展开成 4 个角点用于绘制。 */
function shapeCorners(obj) {
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

function drawPolygon(ctx, points, close) {
  if (!points.length) return;
  ctx.beginPath();
  const first = imageToScreen(points[0]);
  ctx.moveTo(first.x, first.y);
  for (let i = 1; i < points.length; i += 1) {
    const screen = imageToScreen(points[i]);
    ctx.lineTo(screen.x, screen.y);
  }
  if (close) ctx.closePath();
}

function drawLabel(ctx, obj, anchor, color, dashed = false) {
  if (!state.opts.showLabels) return;
  const name = obj.label || classById(obj.class_id).name;
  const score = Number(obj.score);
  const text = state.opts.showScore && score >= 0 ? `${name} ${score.toFixed(2)}` : name;
  ctx.save();
  ctx.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
  const metrics = ctx.measureText(text);
  const width = metrics.width + 8;
  const height = 16;
  const x = anchor.x;
  // 贴近上边缘时翻到框内侧，否则标签会跑到画布外看不见
  let y = anchor.y - height;
  if (y < 0) y = anchor.y;
  ctx.fillStyle = dashed ? "rgba(10,12,16,0.72)" : "rgba(10,12,16,0.82)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  if (dashed) ctx.setLineDash([3, 3]);
  ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
  ctx.setLineDash([]);
  ctx.fillStyle = "#eaf6ff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 4, y + height / 2 + 0.5);
  ctx.restore();
}

function drawShape(ctx, obj, options = {}) {
  const { draft = false, selected = false } = options;
  const baseColor = draft ? [255, 196, 60] : classById(obj.class_id).color;
  const color = rgb(baseColor);
  const points = obj.points || [];
  const alpha = state.opts.fillAlpha;

  ctx.save();
  ctx.lineWidth = selected ? 2.5 : 1.6;
  ctx.strokeStyle = color;
  if (draft) ctx.setLineDash([6, 4]);

  switch (obj.shape_type) {
    case "rectangle":
    case "rotated_rect": {
      drawPolygon(ctx, shapeCorners(obj), true);
      if (alpha > 0) {
        ctx.fillStyle = rgb(baseColor, alpha);
        ctx.fill();
      }
      ctx.stroke();
      break;
    }
    case "polygon": {
      drawPolygon(ctx, points, true);
      if (alpha > 0) {
        ctx.fillStyle = rgb(baseColor, alpha);
        ctx.fill();
      }
      ctx.stroke();
      break;
    }
    case "line": {
      drawPolygon(ctx, points, false);
      ctx.stroke();
      break;
    }
    case "point": {
      const p = imageToScreen(points[0] || [0, 0]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, selected ? 6 : 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "rgba(6,10,14,0.9)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      break;
    }
    case "keypoints": {
      points.forEach((point, index) => {
        const visibility = obj.keypoint_visibility?.[index] ?? 2;
        const p = imageToScreen(point);
        if (index > 0) {
          const prev = imageToScreen(points[index - 1]);
          ctx.beginPath();
          ctx.moveTo(prev.x, prev.y);
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = rgb([255, 196, 60], 0.55);
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.beginPath();
        // 三态可见性：0=不可见、1=遮挡、2=可见，用不同填充区分
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle =
          visibility === 0 ? "rgba(120,130,140,0.5)" : visibility === 1 ? "rgba(255,150,80,0.85)" : color;
        ctx.fill();
        ctx.strokeStyle = "rgba(6,10,14,0.9)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });
      break;
    }
    default:
      break;
  }
  ctx.setLineDash([]);
  ctx.restore();

  if (state.opts.showShapes && points.length) {
    const corners = shapeCorners(obj);
    drawLabel(ctx, obj, imageToScreen(corners[0]), color, draft);
  }
}

function drawHandles(ctx, obj) {
  const points = obj.points || [];
  ctx.save();
  for (const point of points) {
    const screen = imageToScreen(point);
    ctx.beginPath();
    ctx.rect(screen.x - HANDLE_RADIUS, screen.y - HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2);
    ctx.fillStyle = "rgba(8,12,16,0.9)";
    ctx.fill();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // 多边形的边中点插点手柄：从这里拖出去就在该位置长出一个新顶点
  if (obj.shape_type === "polygon") {
    for (let index = 0; index < points.length; index += 1) {
      const next = (index + 1) % points.length;
      const mid = imageToScreen([
        (points[index][0] + points[next][0]) / 2,
        (points[index][1] + points[next][1]) / 2,
      ]);
      ctx.beginPath();
      ctx.arc(mid.x, mid.y, 3.2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(34,211,238,0.65)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  // 矩形的边中点手柄：拖动改那一条边
  if (obj.shape_type === "rectangle" || obj.shape_type === "rotated_rect") {
    const corners = shapeCorners(obj);
    if (corners.length === 4) {
      for (let index = 0; index < 4; index += 1) {
        const a = corners[index];
        const b = corners[(index + 1) % 4];
        const mid = imageToScreen([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
        ctx.beginPath();
        ctx.rect(mid.x - 3.5, mid.y - 3.5, 7, 7);
        ctx.strokeStyle = "rgba(125,255,178,0.85)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }
  }
  ctx.restore();
}

/** 正在绘制中的形状预览。 */
function pendingPreview() {
  const points = [...state.pending];
  const hover = state.hover?.image;
  if (!points.length) return null;
  if ((state.tool === "rectangle" || state.tool === "rotated_rect") && hover) {
    const [a] = points;
    return {
      shape_type: state.tool,
      points: [[Math.min(a[0], hover.x), Math.min(a[1], hover.y)], [Math.max(a[0], hover.x), Math.max(a[1], hover.y)]],
      class_id: activeClassId(),
      label: "",
      rotation: 0,
    };
  }
  if (state.tool === "line" && hover) {
    return { shape_type: "line", points: [points[0], [hover.x, hover.y]], class_id: activeClassId(), label: "" };
  }
  return { shape_type: state.tool, points, class_id: activeClassId(), label: "" };
}

function renderCanvas() {
  const canvas = el.canvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;

  ctx.save();
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, width, height);

  const image = currentImage();
  if (!image) {
    ctx.restore();
    if (el.empty) el.empty.hidden = false;
    if (el.mini) el.mini.style.display = "none";
    return;
  }
  if (el.empty) el.empty.hidden = true;

  if (state.image) {
    ctx.imageSmoothingEnabled = state.view.scale < 2;
    ctx.drawImage(
      state.image,
      state.view.offset.x,
      state.view.offset.y,
      image.width * state.view.scale,
      image.height * state.view.scale,
    );
    // 图像边框：深色图上没有这圈线就看不出图的边界在哪
    ctx.strokeStyle = "rgba(120,170,200,0.35)";
    ctx.lineWidth = 1;
    ctx.strokeRect(
      state.view.offset.x + 0.5,
      state.view.offset.y + 0.5,
      image.width * state.view.scale - 1,
      image.height * state.view.scale - 1,
    );
  }

  if (state.opts.showShapes) {
    for (const obj of image.annotations) {
      if (state.hidden.has(obj.id)) continue;
      drawShape(ctx, obj, { selected: obj.id === state.selectedId });
    }
    // 草稿层画在正式标注之上，用虚线 + 琥珀色区分
    for (const draft of currentDrafts()) {
      if (state.hidden.has(draft.id)) continue;
      drawShape(ctx, draft, { draft: true, selected: draft.id === state.selectedId });
    }
  }

  const selected = selectedObject();
  if (selected && !state.hidden.has(selected.id) && !isDraft(selected)) drawHandles(ctx, selected);
  else if (selected && isDraft(selected)) drawHandles(ctx, selected);

  const preview = pendingPreview();
  if (preview) drawShape(ctx, preview, { draft: true });

  for (const point of state.pending) {
    const screen = imageToScreen(point);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#22d3ee";
    ctx.fill();
  }

  // 吸附提示：一个圈说明"松开就落到这儿"
  if (state.hover?.snap) {
    const screen = imageToScreen(state.hover.snap);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, SNAP_RADIUS, 0, Math.PI * 2);
    ctx.strokeStyle = "#7dffb2";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (state.opts.crosshair && state.cursor) {
    ctx.save();
    ctx.strokeStyle = "rgba(34,211,238,0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, state.cursor.y);
    ctx.lineTo(width, state.cursor.y);
    ctx.moveTo(state.cursor.x, 0);
    ctx.lineTo(state.cursor.x, height);
    ctx.stroke();
    ctx.restore();
  }

  ctx.restore();
  renderMini();
}

function renderMini() {
  const mini = el.mini;
  const image = currentImage();
  if (!mini) return;
  if (!image) {
    mini.style.display = "none";
    return;
  }
  mini.style.display = "block";
  const ratio = window.devicePixelRatio || 1;
  const boxW = 180;
  const boxH = 120;
  mini.width = Math.round(boxW * ratio);
  mini.height = Math.round(boxH * ratio);
  mini.style.width = `${boxW}px`;
  mini.style.height = `${boxH}px`;

  const ctx = mini.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, boxW, boxH);
  ctx.fillStyle = "rgba(6,10,14,0.85)";
  ctx.fillRect(0, 0, boxW, boxH);

  const scale = Math.min(boxW / (image.width || 1), boxH / (image.height || 1));
  const dx = (boxW - (image.width || 1) * scale) / 2;
  const dy = (boxH - (image.height || 1) * scale) / 2;

  if (state.image) ctx.drawImage(state.image, dx, dy, image.width * scale, image.height * scale);

  ctx.strokeStyle = "rgba(34,211,238,0.7)";
  ctx.lineWidth = 1;
  for (const obj of image.annotations) {
    const pts = obj.points || [];
    if (!pts.length || obj.shape_type === "point") continue;
    ctx.beginPath();
    pts.forEach((point, index) => {
      const x = dx + point[0] * scale;
      const y = dy + point[1] * scale;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (obj.shape_type !== "line") ctx.closePath();
    ctx.stroke();
  }

  // 视口框：告诉用户"现在看的是整张图的哪一块"
  const view = state.view;
  const vx = dx + ((0 - view.offset.x) / view.scale) * scale;
  const vy = dy + ((0 - view.offset.y) / view.scale) * scale;
  const vw = ((el.stage?.clientWidth || 0) / view.scale) * scale;
  const vh = ((el.stage?.clientHeight || 0) / view.scale) * scale;
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(vx, vy, vw, vh);
}

// ---------------------------------------------------------------------------
// 命中测试与几何
// ---------------------------------------------------------------------------
function pointInPolygon(point, polygon) {
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

function distanceToSegment(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(point.x - a[0], point.y - a[1]);
  let t = ((point.x - a[0]) * dx + (point.y - a[1]) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(point.x - (a[0] + t * dx), point.y - (a[1] + t * dy));
}

/**
 * 命中测试：返回命中的对象。
 *
 * 从后往前遍历——后画的在上层，用户点哪个就该选中看到的那个。
 * 点在形状内部也算命中（填充分明就会点进去），线段和点只能靠距离。
 */
function hitTest(point) {
  const image = currentImage();
  if (!image) return null;
  const tolerance = HIT_TOLERANCE / state.view.scale;
  const ordered = [
    ...image.annotations.map((obj) => ({ obj })),
    ...currentDrafts().map((obj) => ({ obj })),
  ];

  for (let i = ordered.length - 1; i >= 0; i -= 1) {
    const obj = ordered[i].obj;
    if (state.hidden.has(obj.id)) continue;
    const points = obj.points || [];
    if (!points.length) continue;

    if (obj.shape_type === "point") {
      if (Math.hypot(point.x - points[0][0], point.y - points[0][1]) <= tolerance + 4) return obj;
      continue;
    }
    if (obj.shape_type === "line") {
      for (let k = 1; k < points.length; k += 1) {
        if (distanceToSegment(point, points[k - 1], points[k]) <= tolerance) return obj;
      }
      continue;
    }
    if (obj.shape_type === "keypoints") {
      for (const p of points) {
        if (Math.hypot(point.x - p[0], point.y - p[1]) <= tolerance + 3) return obj;
      }
      continue;
    }
    const corners = shapeCorners(obj);
    if (corners.length >= 3 && pointInPolygon(point, corners)) return obj;
    // 内部没点中时再判边框，这样"框挨着但没套住"也能选中
    for (let k = 1; k <= corners.length; k += 1) {
      if (distanceToSegment(point, corners[k - 1], corners[k % corners.length]) <= tolerance) return obj;
    }
  }
  return null;
}

function hitHandle(obj, point) {
  const tolerance = (HANDLE_RADIUS + 3) / state.view.scale;
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
      if (Math.hypot(point.x - mid[0], point.y - mid[1]) <= (4 + 3) / state.view.scale) {
        return { kind: "insert", index };
      }
    }
  }
  return null;
}

function hitEdge(obj, point) {
  if (obj.shape_type !== "rectangle" && obj.shape_type !== "rotated_rect") return null;
  const corners = shapeCorners(obj);
  if (corners.length !== 4) return null;
  const tolerance = 6 / state.view.scale;
  for (let index = 0; index < 4; index += 1) {
    const a = corners[index];
    const b = corners[(index + 1) % 4];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (Math.hypot(point.x - mid[0], point.y - mid[1]) <= tolerance) return { kind: "edge", index };
  }
  return null;
}

// ---------------------------------------------------------------------------
// 吸附
// ---------------------------------------------------------------------------
function snapPoint(point, { allowClose = false } = {}) {
  // 契约：**永远**返回 `{ point: [x, y], snap }`。
  // 调用方一律按数组取 `point[0] / point[1]`；如果这里有的分支返回 `{x, y}` 对象，
  // 取到的就是 undefined，代价是画出"空点"标注并且拖拽画框永远收不了尾。
  const origin = [point.x, point.y];
  if (!state.opts.snap) return { point: origin, snap: null };
  const image = currentImage();
  if (!image) return { point: origin, snap: null };
  const radius = SNAP_RADIUS / state.view.scale;

  // 闭合吸附：多边形靠近起点时直接吸上去，否则"精确点回起点"几乎做不到
  if (allowClose && state.pending.length >= 3) {
    const first = state.pending[0];
    if (Math.hypot(point.x - first[0], point.y - first[1]) <= radius) {
      return { point: [first[0], first[1]], snap: first };
    }
  }

  // 顶点吸附：贴着已有标注的顶点画，能画出严格对齐的数据
  let best = null;
  let bestDistance = radius;
  for (const obj of image.annotations) {
    for (const p of obj.points || []) {
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

/** 线段正交吸附：按住 Shift 时把端点约束到水平或垂直。 */
function constrainOrthogonal(anchor, point) {
  const dx = Math.abs(point.x - anchor[0]);
  const dy = Math.abs(point.y - anchor[1]);
  return dx >= dy ? [point[0], anchor[1]] : [anchor[0], point[1]];
}

// ---------------------------------------------------------------------------
// 绘制流程
// ---------------------------------------------------------------------------
function activeClassId() {
  const selected = selectedObject();
  if (selected) return selected.class_id;
  return state.project?.classes?.[0]?.id ?? 0;
}

function finishShape(points) {
  if (!points.length) return;
  const image = currentImage();
  if (!image) return;
  pushHistory();
  const shape = state.tool === "select" ? "rectangle" : state.tool;
  const obj = {
    id: newObjectId(),
    class_id: activeClassId(),
    label: "",
    description: "",
    group_id: 0,
    shape_type: shape,
    points: points.map((p) => [p[0], p[1]]),
    rotation: 0,
    keypoint_visibility: [],
    score: -1,
  };
  if (shape === "keypoints") obj.keypoint_visibility = points.map(() => 2);
  if (shape === "rectangle" || shape === "rotated_rect") {
    const [a, b] = obj.points;
    obj.points = [
      [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
      [Math.max(a[0], b[0]), Math.max(a[1], b[1])],
    ];
  }
  image.annotations.push(obj);
  state.selectedId = obj.id;
  state.pending = [];
  setTool("select");
  afterEdit();
  setStatus(`已添加${SHAPE_LABELS[shape] || "标注"}`);
}

/**
 * 去掉"几乎重合"的连续重复点。
 *
 * 双击收尾时浏览器会连发两次 click，加上双击本身的坐标，末尾常常会多出一两个
 * 与上一个顶点完全相同的点；多边形导出后就成了退化多边形，关键点也会多出无名点。
 * 容差按屏幕 0.5px 折算成图像坐标，缩放很小时也不会误删真实的密集点。
 */
function dropTrailingDuplicates(points, tolerance = null) {
  const limit = tolerance ?? Math.max(0.5, 0.5 / state.view.scale);
  const out = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (last && Math.hypot(p[0] - last[0], p[1] - last[1]) <= limit) continue;
    out.push([p[0], p[1]]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 指针交互
// ---------------------------------------------------------------------------
function stagePoint(event) {
  const rect = el.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

/**
 * 捕获指针，失败不抛。
 *
 * 正常鼠标操作下不会失败；但自动化脚本与某些触摸设备会给出非活跃 pointerId，
 * 那时 `setPointerCapture` 会抛 NotFoundError。捕获只是"拖出画布也别断"的增强，
 * 抛错会把整个按下的逻辑中断掉，所以必须容忍失败。
 */
function capturePointer(event) {
  try {
    el.canvas.setPointerCapture(event.pointerId);
  } catch {
    /* 忽略：没有捕获也能拖，只是移出画布会断 */
  }
}

function onPointerDown(event) {
  const image = currentImage();
  if (!image || !state.project) return;
  const screen = stagePoint(event);
  const point = toImage(screen);

  // 中键 = 平移画布（不改变选择，避免"想挪一下视野结果把标注改了"）
  if (event.button === 1) {
    state.drag = { kind: "pan", startScreen: screen, startOffset: { ...state.view.offset } };
    capturePointer(event);
    return;
  }
  if (event.button === 2) return;

  if (state.tool === "select") {
    const selected = selectedObject();
    if (selected) {
      // 1) 顶点手柄优先：拖动只改那一个点
      const handle = hitHandle(selected, point);
      if (handle) {
        if (handle.kind === "insert") {
          // 插点会立刻改变几何，所以这一步就得压栈
          pushHistory();
          const points = selected.points;
          const next = (handle.index + 1) % points.length;
          points.splice(handle.index + 1, 0, [
            (points[handle.index][0] + points[next][0]) / 2,
            (points[handle.index][1] + points[next][1]) / 2,
          ]);
          state.drag = { kind: "vertex", index: handle.index + 1, pushed: true };
        } else {
          // 只按下还没动，先不压栈：真的拖动了才记历史（见 ensureDragHistory）
          state.drag = { kind: "vertex", index: handle.index, pushed: false };
        }
        capturePointer(event);
        renderCanvas();
        return;
      }
      // 2) 边手柄：拖动改那一条边
      const edge = hitEdge(selected, point);
      if (edge) {
        state.drag = { kind: "edge", index: edge.index, pushed: false };
        capturePointer(event);
        return;
      }
    }

    // 3) 普通命中
    const hit = hitTest(point);
    if (hit) {
      // Ctrl+点击多边形顶点 = 删点
      if ((event.ctrlKey || event.metaKey) && hit.shape_type === "polygon") {
        const tolerance = (HANDLE_RADIUS + 3) / state.view.scale;
        const index = (hit.points || []).findIndex((p) => Math.hypot(point.x - p[0], point.y - p[1]) <= tolerance);
        if (index >= 0) {
          pushHistory();
          hit.points.splice(index, 1);
          if (hit.points.length < 3) removeObject(hit.id);
          afterEdit();
          renderCanvas();
          setStatus("已删除一个顶点");
          return;
        }
      }
      state.selectedId = hit.id;
      state.drag = { kind: "move", start: point, original: clone(hit.points), objectId: hit.id, pushed: false };
      capturePointer(event);
      renderAll();
      renderCanvas();
      setStatus(`选中 ${hit.label || classById(hit.class_id).name}`);
      return;
    }

    // 4) 空白处：取消选择并平移
    state.selectedId = "";
    state.drag = { kind: "pan", startScreen: screen, startOffset: { ...state.view.offset } };
    capturePointer(event);
    renderAll();
    renderCanvas();
    return;
  }

  // 绘制模式
  // 顺序：先吸附（snapPoint 的返回契约是 [x, y] 数组），再按 Shift 做正交约束，
  // 这样"吸附到顶点"和"保持水平/垂直"不会互相覆盖。
  const snapped = event.altKey
    ? { point: [point.x, point.y], snap: null }
    : snapPoint(point, { allowClose: state.tool === "polygon" });
  let target = snapped.point;
  if (event.shiftKey && state.pending.length) target = constrainOrthogonal(state.pending[0], target);

  if (state.tool === "keypoints") {
    state.pending.push([target[0], target[1]]);
    renderCanvas();
    return;
  }
  if (state.tool === "point") {
    finishShape([target]);
    return;
  }
  if (state.tool === "rectangle" || state.tool === "rotated_rect" || state.tool === "line") {
    if (state.pending.length === 0) {
      state.pending = [[target[0], target[1]]];
      state.drag = { kind: "draw" };
      capturePointer(event);
      renderCanvas();
    } else {
      finishShape([state.pending[0], target]);
    }
    return;
  }
  if (state.tool === "polygon") {
    if (state.pending.length >= 3) {
      const first = state.pending[0];
      if (Math.hypot(target[0] - first[0], target[1] - first[1]) <= SNAP_RADIUS / state.view.scale) {
        finishShape(state.pending);
        return;
      }
    }
    state.pending.push([target[0], target[1]]);
    renderCanvas();
  }
}

function onPointerMove(event) {
  const screen = stagePoint(event);
  state.cursor = screen;
  const image = currentImage();
  if (!image) return;
  const point = toImage(screen);

  if (state.drag) {
    // 拖动途中也要更新 hover：绘制预览、边拖动都靠它拿当前点
    state.hover = { image: point, snap: null };
    handleDrag(point);
    return;
  }

  const snapped = event.altKey ? { point, snap: null } : snapPoint(point, { allowClose: state.tool === "polygon" });
  state.hover = { image: point, snap: snapped.snap };

  if (state.opts.crosshair || state.pending.length) renderCanvas();

  // 悬停在顶点/边上时换光标，让"这里能拖"这件事可发现
  if (state.tool === "select") {
    const selected = selectedObject();
    let cursor = "default";
    if (selected && (hitHandle(selected, point) || hitEdge(selected, point))) cursor = "nwse-resize";
    else if (hitTest(point)) cursor = "move";
    el.canvas.style.cursor = cursor;
  }
}

/** 拖动类操作：只有真的改变了几何才压栈，避免"点一下手柄/框体"留下空历史。 */
function ensureDragHistory(drag) {
  if (drag.pushed) return;
  drag.pushed = true;
  pushHistory();
}

function handleDrag(point) {
  const drag = state.drag;
  if (!drag) return;

  if (drag.kind === "pan") {
    state.view.offset = {
      x: drag.startOffset.x + (state.cursor.x - drag.startScreen.x),
      y: drag.startOffset.y + (state.cursor.y - drag.startScreen.y),
    };
    renderCanvas();
    return;
  }

  const image = currentImage();
  if (!image) return;

  if (drag.kind === "move") {
    const target =
      image.annotations.find((obj) => obj.id === drag.objectId) ||
      currentDrafts().find((obj) => obj.id === drag.objectId);
    if (!target) return;
    const dx = point.x - drag.start.x;
    const dy = point.y - drag.start.y;
    // 位移为零时不写历史、也不标脏
    if (dx === 0 && dy === 0) return;
    ensureDragHistory(drag);
    target.points = drag.original.map((p) => [p[0] + dx, p[1] + dy]);
    renderCanvas();
    return;
  }

  const target = selectedObject();
  if (!target) return;
  const points = target.points;
  const tolerance = (HANDLE_RADIUS + 4) / state.view.scale;

  if (drag.kind === "vertex") {
    const index = Math.min(drag.index, points.length - 1);
    if (target.shape_type === "rectangle") {
      // 两点表示：移动哪个角，对角都不动，然后重新归一化为左上/右下。
      // 这样拖任意角都不会把矩形拖成"负面积"或整体翻转。
      const anchor = points[1 - index];
      const b = [point.x, point.y];
      if (anchor[0] === b[0] && anchor[1] === b[1]) return;
      ensureDragHistory(drag);
      target.points = [
        [Math.min(anchor[0], b[0]), Math.min(anchor[1], b[1])],
        [Math.max(anchor[0], b[0]), Math.max(anchor[1], b[1])],
      ];
    } else {
      if (points[index][0] === point.x && points[index][1] === point.y) return;
      ensureDragHistory(drag);
      // 旋转矩形维持对角线方向，只改端点（保角）
      points[index] = [point.x, point.y];
    }
    renderCanvas();
    renderProps();
    return;
  }

  if (drag.kind === "edge") {
    // 拖边：只改那条边，另三条边不动。
    // 旋转框必须**在它自己的局部坐标系里**改——先按 AABB 求出四条边再写回，
    // 等于把斜放的框摆正，角度和形状都会糊掉。
    const angle = Number(target.rotation) || 0;
    const rotated = target.shape_type === "rotated_rect" && angle !== 0;
    let minX;
    let maxX;
    let minY;
    let maxY;
    if (rotated) {
      const [a, b] = target.points;
      minX = Math.min(a[0], b[0]);
      maxX = Math.max(a[0], b[0]);
      minY = Math.min(a[1], b[1]);
      maxY = Math.max(a[1], b[1]);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      // 把光标换算到未旋转的局部坐标（R(-angle)）
      const dx = point.x - cx;
      const dy = point.y - cy;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const localX = cx + dx * cos + dy * sin;
      const localY = cy - dx * sin + dy * cos;
      // 0=上边 1=右边 2=下边 3=左边
      if (drag.index === 0) minY = localY;
      else if (drag.index === 1) maxX = localX;
      else if (drag.index === 2) maxY = localY;
      else minX = localX;
    } else {
      const corners = shapeCorners(target);
      const xs = corners.map((p) => p[0]);
      const ys = corners.map((p) => p[1]);
      minX = Math.min(...xs);
      maxX = Math.max(...xs);
      minY = Math.min(...ys);
      maxY = Math.max(...ys);
      // 0=上边 1=右边 2=下边 3=左边
      if (drag.index === 0) minY = point.y;
      else if (drag.index === 1) maxX = point.x;
      else if (drag.index === 2) maxY = point.y;
      else minX = point.x;
    }
    if (Math.abs(maxX - minX) < tolerance || Math.abs(maxY - minY) < tolerance) return;
    ensureDragHistory(drag);
    target.points = [
      [Math.min(minX, maxX), Math.min(minY, maxY)],
      [Math.max(minX, maxX), Math.max(minY, maxY)],
    ];
    renderCanvas();
    renderProps();
    return;
  }

  if (drag.kind === "draw") renderCanvas();
}

function onPointerUp() {
  const drag = state.drag;
  state.drag = null;
  if (!drag) return;
  if (drag.kind === "move" || drag.kind === "vertex" || drag.kind === "edge") {
    // 没有真正拖动过就别标脏：否则"点一下手柄"也会触发自动保存
    if (drag.pushed) afterEdit();
    else renderCanvas();
    return;
  }
  if (drag.kind === "draw") {
    // 拖拽式绘制：按下 → 拖 → 松开即成形。
    // 同时保留"点一下定起点、再点一下定终点"的两点式，两种手感都成立。
    const start = state.pending[0];
    const end = state.hover?.image;
    if (start && end) {
      const distance = Math.hypot(end.x - start[0], end.y - start[1]) * state.view.scale;
      if (distance >= 3) {
        finishShape([start, [end.x, end.y]]);
        return;
      }
    }
    renderCanvas();
    return;
  }
  renderCanvas();
}

function onWheel(event) {
  const image = currentImage();
  if (!image) return;
  event.preventDefault();
  const screen = stagePoint(event);
  const delta = normalizeWheelDelta(event, el.stage.clientHeight);
  const factor = wheelDeltaToFactor(delta);
  const next = clampScale(state.view.scale * factor, MIN_SCALE_ABS, 40);
  const transform = computeZoomTransform(screen, state.view.scale, state.view.offset, next);
  state.view = { scale: transform.scale, offset: transform.offset };
  updateZoomLabel();
  renderCanvas();
}

// ---------------------------------------------------------------------------
// 对象操作
// ---------------------------------------------------------------------------
function removeObject(id) {
  const image = currentImage();
  if (!image) return false;
  const index = image.annotations.findIndex((item) => item.id === id);
  if (index >= 0) {
    image.annotations.splice(index, 1);
    if (state.selectedId === id) state.selectedId = "";
    return true;
  }
  const drafts = currentDrafts();
  const draftIndex = drafts.findIndex((item) => item.id === id);
  if (draftIndex >= 0) {
    drafts.splice(draftIndex, 1);
    if (state.selectedId === id) state.selectedId = "";
    return true;
  }
  return false;
}

function deleteSelected() {
  if (!state.selectedId) return;
  pushHistory();
  removeObject(state.selectedId);
  afterEdit();
  renderCanvas();
  setStatus("已删除选中标注");
}

// ---------------------------------------------------------------------------
// 预标注草稿
// ---------------------------------------------------------------------------
function promoteDraft(draft) {
  const obj = clone(draft);
  delete obj._draft;
  delete obj._note;
  delete obj._warnings;
  // 保留置信度：接受后要能看出"这条是模型给的，不是我手点的"
  if (!(Number(obj.score) >= 0)) obj.score = -1;
  return obj;
}

/**
 * 把草稿的类别落到工程的类别表里。
 *
 * 外部服务/自有引擎给出的 class_id 可能超出工程现有类别范围（例如模型有 80 类、
 * 工程只建了 2 类）。这时按类别名补建一条，而不是把越界 id 直接写进工程——
 * 越界 id 在 Qt 里会退化成"无类别"，导出时也拿不到正确的类名。
 */
function ensureDraftClass(draft) {
  const classes = state.project.classes;
  const hit = classes.find((item) => item.id === draft.class_id);
  if (hit) return hit;
  const name = (draft.label || "").trim() || `cls${draft.class_id}`;
  // 同名类别已存在就复用，避免同一个类名在类别表里出现多条
  const byName = classes.find((item) => item.name === name);
  if (byName) return byName;
  const id = classes.reduce((max, item) => Math.max(max, Number(item.id) || 0), -1) + 1;
  const cls = { id, name, color: defaultColor(id) };
  classes.push(cls);
  return cls;
}

function acceptDraft(id) {
  const image = currentImage();
  if (!image) return;
  const drafts = currentDrafts();
  const index = drafts.findIndex((item) => item.id === id);
  if (index < 0) return;
  pushHistory();
  const [draft] = drafts.splice(index, 1);
  const obj = promoteDraft(draft);
  const cls = ensureDraftClass(obj);
  obj.class_id = cls.id;
  image.annotations.push(obj);
  state.selectedId = obj.id;
  afterEdit();
  renderCanvas();
  setStatus(`已接受草稿（类别：${cls.name}）`);
}

function rejectDraft(id) {
  // 丢弃草稿与删除标注是同一件事，只是对象在草稿区而不是标注区。
  if (!removeObject(id)) return;
  renderAll();
  renderCanvas();
  setStatus("已丢弃草稿");
}

function acceptAllDrafts() {
  const image = currentImage();
  const drafts = currentDrafts();
  if (!image || !drafts.length) return;
  // 一次压一条历史：用户按一次撤销就该回到"没有这些标注"，
  // 而不是要按 N 次才撤干净。
  pushHistory();
  const count = drafts.length;
  for (const draft of drafts.splice(0)) {
    const obj = promoteDraft(draft);
    obj.class_id = ensureDraftClass(obj).id;
    image.annotations.push(obj);
  }
  state.selectedId = "";
  afterEdit();
  renderCanvas();
  setStatus(`已接受 ${count} 个草稿`);
}

function rejectAllDrafts() {
  const drafts = currentDrafts();
  if (!drafts.length) return;
  const count = drafts.length;
  drafts.length = 0;
  renderAll();
  renderCanvas();
  setStatus(`已丢弃 ${count} 个草稿`);
}

async function runAutoLabel() {
  const image = currentImage();
  if (!image) {
    setStatus("先选一张图片", "bad");
    return;
  }
  const engine = (el.alEngine?.value || "").trim();
  if (!engine) {
    el.autoLabelBox?.setAttribute("open", "");
    setStatus("请先填预标注用的 .engine 路径", "bad");
    return;
  }
  setStatus("预标注中…", "warn");
  try {
    const result = await autoLabel({
      image: image.path,
      task: state.project.task || "detect",
      engine,
      labels: (el.alLabels?.value || "").trim(),
      conf: Number(el.alConf?.value || 0.25),
      nms: Number(el.alNms?.value || 0.45),
      service: (el.alService?.value || "").trim(),
    });
    const drafts = result.drafts || [];
    state.drafts.set(image.id, drafts);
    const bits = [`${drafts.length} 个草稿（通道：${result.channel}）`];
    if (result.warnings?.length) bits.push(result.warnings.join("；"));
    setStatus(bits.join(" · "), result.external_error || !drafts.length ? "warn" : "ok");
    renderAll();
    renderCanvas();
  } catch (error) {
    setStatus(`预标注失败：${error.message}`, "bad");
  }
}

// ---------------------------------------------------------------------------
// 渲染：列表与面板
// ---------------------------------------------------------------------------
function renderImageList() {
  if (!el.imageList) return;
  el.imageList.innerHTML = "";
  const images = state.project?.images || [];
  if (el.imageCount) el.imageCount.textContent = String(images.length);

  images.forEach((image, index) => {
    const li = document.createElement("li");
    li.className = index === state.currentIndex ? "active" : "";
    if (index === state.currentIndex && state.dirty) li.classList.add("dirty");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = image.id || image.path.split(/[\\/]/).pop();
    name.title = image.path;
    const meta = document.createElement("span");
    meta.className = "meta";
    const draftCount = (state.drafts.get(image.id) || []).length;
    meta.textContent = `${image.annotations.length}${draftCount ? `+${draftCount}草` : ""}`;
    const split = document.createElement("span");
    split.className = `split ${image.split}`;
    split.textContent = image.split;
    li.append(name, meta, split);
    li.addEventListener("click", () => switchImage(index));
    el.imageList.appendChild(li);
  });
}

async function switchImage(index) {
  if (index === state.currentIndex) return;
  await flushSave();
  state.currentIndex = index;
  state.selectedId = "";
  state.pending = [];
  state.hidden.clear();
  state.viewValid = false;
  renderAll();
  renderCanvas();
  loadCurrentImage();
}

function renderClassList() {
  if (!el.classList) return;
  el.classList.innerHTML = "";
  const classes = state.project?.classes || [];
  if (el.classCount) el.classCount.textContent = String(classes.length);
  const counts = new Map();
  for (const image of state.project?.images || []) {
    for (const obj of image.annotations) counts.set(obj.class_id, (counts.get(obj.class_id) || 0) + 1);
  }

  classes.forEach((cls) => {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = rgb(cls.color);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = cls.name;
    const count = document.createElement("span");
    count.className = "meta";
    count.textContent = String(counts.get(cls.id) || 0);
    const del = document.createElement("button");
    del.type = "button";
    del.className = "mini danger";
    del.textContent = "×";
    del.title = "删除类别";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteClass(cls.id);
    });
    li.append(swatch, label, count, del);
    li.addEventListener("click", () => applyClassToSelection(cls.id));
    el.classList.appendChild(li);
  });
}

function renderDraftList() {
  if (!el.draftList) return;
  el.draftList.innerHTML = "";
  const drafts = currentDrafts();
  if (el.draftCount) el.draftCount.textContent = String(drafts.length);

  drafts.forEach((draft, index) => {
    const li = document.createElement("li");
    if (draft.id === state.selectedId) li.classList.add("active");
    const name = document.createElement("span");
    name.className = "name";
    const score = Number(draft.score);
    name.textContent = `#${index + 1} ${draft.label || classById(draft.class_id).name}${
      score >= 0 ? ` ${score.toFixed(2)}` : ""
    }`;
    name.title = draft._note || "";
    const ok = document.createElement("button");
    ok.type = "button";
    ok.className = "mini";
    ok.textContent = "✓";
    ok.title = "接受";
    ok.addEventListener("click", (event) => {
      event.stopPropagation();
      acceptDraft(draft.id);
    });
    const no = document.createElement("button");
    no.type = "button";
    no.className = "mini danger";
    no.textContent = "×";
    no.title = "丢弃";
    no.addEventListener("click", (event) => {
      event.stopPropagation();
      rejectDraft(draft.id);
    });
    li.append(name, ok, no);
    li.addEventListener("click", () => {
      state.selectedId = draft.id;
      renderAll();
      renderCanvas();
    });
    el.draftList.appendChild(li);
  });
}

function renderProps() {
  if (!el.props) return;
  el.props.innerHTML = "";
  const image = currentImage();
  const obj = selectedObject();

  if (!obj) {
    const hint = document.createElement("p");
    hint.className = "anno-hint";
    hint.textContent = image
      ? `${image.id} · ${image.width}×${image.height} · ${image.annotations.length} 个标注`
      : "未选中标注";
    el.props.appendChild(hint);

    if (image) {
      const splitRow = document.createElement("label");
      splitRow.className = "anno-toggle";
      splitRow.append(document.createTextNode("集合 "));
      const splitSelect = document.createElement("select");
      ["train", "val", "test"].forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        option.selected = image.split === name;
        splitSelect.appendChild(option);
      });
      splitSelect.addEventListener("change", () => {
        pushHistory();
        image.split = splitSelect.value;
        afterEdit();
        renderCanvas();
      });
      splitRow.appendChild(splitSelect);
      el.props.appendChild(splitRow);
    }
    return;
  }

  const field = (label, input) => {
    const row = document.createElement("label");
    row.className = "anno-field";
    const span = document.createElement("span");
    span.textContent = label;
    row.append(span, input);
    el.props.appendChild(row);
  };

  const textInput = (value, onChange, placeholder = "") => {
    const input = document.createElement("input");
    input.type = "text";
    input.value = value ?? "";
    input.placeholder = placeholder;
    input.addEventListener("change", onChange);
    return input;
  };

  const numberInput = (value, onChange, step = null) => {
    const input = document.createElement("input");
    input.type = "number";
    if (step) input.step = step;
    input.value = value ?? "";
    input.addEventListener("change", onChange);
    return input;
  };

  field(
    "标签",
    textInput(obj.label, (event) => {
      pushHistory();
      obj.label = event.target.value;
      afterEdit();
      renderCanvas();
    }),
  );

  const classSelect = document.createElement("select");
  for (const cls of state.project.classes) {
    const option = document.createElement("option");
    option.value = String(cls.id);
    option.textContent = `${cls.id}: ${cls.name}`;
    option.selected = obj.class_id === cls.id;
    classSelect.appendChild(option);
  }
  classSelect.addEventListener("change", () => {
    pushHistory();
    obj.class_id = Number(classSelect.value);
    afterEdit();
    renderCanvas();
  });
  field("类别", classSelect);

  field(
    "描述",
    textInput(obj.description, (event) => {
      pushHistory();
      obj.description = event.target.value;
      // 必须与其它属性一致地走 afterEdit：否则描述改了不标脏、不排期自动保存，
      // 关页面时静默丢掉。
      afterEdit();
    }),
  );

  field(
    "分组",
    numberInput(obj.group_id ?? 0, (event) => {
      pushHistory();
      obj.group_id = Number(event.target.value) || 0;
      afterEdit();
    }),
  );

  if (obj.shape_type === "keypoints") {
    field(
      "可见性",
      textInput(
        (obj.keypoint_visibility || []).join(","),
        (event) => {
          pushHistory();
          obj.keypoint_visibility = event.target.value
            .split(",")
            .map((token) => Number(token.trim()))
            .filter((value) => Number.isFinite(value));
          afterEdit();
        },
        "2,1,0",
      ),
    );
  }

  if (obj.shape_type === "rotated_rect") {
    field(
      "旋转角",
      numberInput(obj.rotation ?? 0, (event) => {
        pushHistory();
        obj.rotation = Number(event.target.value) || 0;
        afterEdit();
        renderCanvas();
      }, "0.01"),
    );
  }

  const scoreInput = numberInput(
    Number(obj.score) >= 0 ? Number(obj.score) : "",
    () => {
      pushHistory();
      const value = Number(scoreInput.value);
      obj.score = scoreInput.value !== "" && Number.isFinite(value) ? value : -1;
      afterEdit();
      renderCanvas();
    },
    "0.01",
  );
  scoreInput.placeholder = "无";
  field("置信度", scoreInput);

  const difficultRow = document.createElement("label");
  difficultRow.className = "anno-toggle";
  const difficultBox = document.createElement("input");
  difficultBox.type = "checkbox";
  difficultBox.checked = Boolean(obj.difficult);
  difficultBox.addEventListener("change", () => {
    pushHistory();
    if (difficultBox.checked) obj.difficult = true;
    else delete obj.difficult;
    afterEdit();
  });
  difficultRow.append(difficultBox, document.createTextNode("难例"));
  el.props.appendChild(difficultRow);

  // 属性用 "key: value" 每行一条：比 JSON 好输入，也比键值表格省空间
  const keyedArea = (title, rows, source, onChange) => {
    const label = document.createElement("p");
    label.className = "anno-hint";
    label.textContent = title;
    const area = document.createElement("textarea");
    area.rows = rows;
    area.value = formatKeyedLines(source);
    area.addEventListener("change", () => {
      pushHistory();
      onChange(parseKeyedLines(area.value));
      afterEdit();
    });
    el.props.append(label, area);
  };

  keyedArea("属性（每行 键: 值）", 3, obj.attributes, (attributes) => {
    if (Object.keys(attributes).length) obj.attributes = attributes;
    else delete obj.attributes;
  });

  keyedArea("标记（每行 键: true/false）", 2, obj.flags, (rawFlags) => {
    const flags = {};
    for (const [key, value] of Object.entries(rawFlags)) {
      flags[key] = value.toLowerCase() === "true" || value === "1";
    }
    if (Object.keys(flags).length) obj.flags = flags;
    else delete obj.flags;
  });

  const hideRow = document.createElement("label");
  hideRow.className = "anno-toggle";
  const hideBox = document.createElement("input");
  hideBox.type = "checkbox";
  hideBox.checked = state.hidden.has(obj.id);
  hideBox.addEventListener("change", () => {
    if (hideBox.checked) state.hidden.add(obj.id);
    else state.hidden.delete(obj.id);
    renderCanvas();
  });
  hideRow.append(hideBox, document.createTextNode("隐藏此标注"));
  el.props.appendChild(hideRow);
}

// ---------------------------------------------------------------------------
// 类别管理
// ---------------------------------------------------------------------------
function applyClassToSelection(classId) {
  const obj = selectedObject();
  if (!obj) {
    setStatus("先在画布上选中一个标注，再点类别即可改类别");
    return;
  }
  pushHistory();
  obj.class_id = classId;
  afterEdit();
  renderCanvas();
}

function addClass() {
  const name = (el.classNameInput?.value || "").trim();
  if (!name || !state.project) return;
  pushHistory(true);
  const nextId = state.project.classes.reduce((max, item) => Math.max(max, item.id + 1), 0);
  // 配色沿用 Qt 的公式，保证同一个 id 在两端颜色相同
  state.project.classes.push({ id: nextId, name, color: defaultColor(nextId) });
  if (el.classNameInput) el.classNameInput.value = "";
  afterEdit();
  renderCanvas();
}

function deleteClass(classId) {
  if (!state.project) return;
  const used = (state.project.images || []).reduce(
    (sum, image) => sum + image.annotations.filter((obj) => obj.class_id === classId).length,
    0,
  );
  if (used > 0) {
    // 直接删会让这些标注变成"引用了不存在的类别"，导出时才炸。
    // 在入口处拒绝，并把数量说清楚。
    setStatus(`该类别正被 ${used} 个标注引用，请先改掉这些标注的类别`, "bad");
    return;
  }
  if (state.project.classes.length <= 1) {
    setStatus("至少要保留一个类别", "bad");
    return;
  }
  pushHistory(true);
  state.project.classes = state.project.classes.filter((item) => item.id !== classId);
  afterEdit();
  renderCanvas();
}

/** 与 annotation.py 的 default_color 逐位一致（Qt: fromHsv((seed*47)%360, 210, 240)）。 */
function defaultColor(seed) {
  const h = (((seed * 47) % 360) + 360) % 360;
  const s = 210;
  const v = 240;
  const sector = h / 60;
  const index = Math.floor(sector) % 6;
  const fraction = sector - Math.floor(sector);
  const p = v * (1 - s / 255);
  const q = v * (1 - (s / 255) * fraction);
  const t = v * (1 - (s / 255) * (1 - fraction));
  const table = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q],
  ];
  return table[index].map((channel) => Math.round(channel));
}

// ---------------------------------------------------------------------------
// 图片批量加入
// ---------------------------------------------------------------------------
async function addImages() {
  const root = (el.addRoot?.value || "").trim();
  if (!root || !state.project) return;
  setStatus("扫描图片…", "warn");
  try {
    const data = await scanAnnotationImages(root, true);
    if (!data.images.length) {
      setStatus("该目录下没有图片", "warn");
      return;
    }
    pushHistory(true);
    const known = new Set(state.project.images.map((image) => image.path));
    const usedIds = new Set(state.project.images.map((image) => image.id));
    let added = 0;
    for (const item of data.images) {
      if (known.has(item.path)) continue;
      // 不同子目录里可能存在同名图片；id 一旦重复，草稿会串到别的图上，
      // 导出校验也会报"图片 id 重复"，所以在入口就做成唯一。
      const stem = item.path.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
      let id = stem;
      let suffix = 2;
      while (usedIds.has(id)) {
        id = `${stem}_${suffix}`;
        suffix += 1;
      }
      usedIds.add(id);
      state.project.images.push({
        id,
        path: item.path,
        width: item.width,
        height: item.height,
        split: "train",
        annotations: [],
      });
      added += 1;
    }
    if (!added) {
      setStatus("这些图片已经在工程里了", "warn");
      return;
    }
    if (state.currentIndex < 0) {
      state.currentIndex = 0;
      loadCurrentImage();
    }
    afterEdit();
    renderCanvas();
    setStatus(`加入 ${added} 张图片`);
  } catch (error) {
    setStatus(`扫描失败：${error.message}`, "bad");
  }
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------
async function runExport() {
  if (!state.project) return;
  const output = (el.exportOut?.value || "").trim();
  if (!output) {
    setStatus("请填导出目录", "bad");
    return;
  }
  await flushSave();
  setStatus("导出中…", "warn");
  try {
    const result = await exportAnnotationProject({
      project: state.project,
      source: state.path,
      format: el.exportFormat?.value || "yolo",
      output,
      overwrite: true,
    });
    const counts = result.counts || {};
    // 各格式的 counts 结构不同：YOLO 给 train/val/test，COCO 给 images/annotations，
    // VOC 只给 images。硬套 train=0 val=0 test=0 会让人以为"没导出任何东西"。
    let countText;
    if ("train" in counts) {
      countText = `train=${counts.train ?? 0} val=${counts.val ?? 0} test=${counts.test ?? 0}`;
    } else if ("annotations" in counts) {
      countText = `图片=${counts.images ?? 0} 标注=${counts.annotations ?? 0}`;
    } else {
      countText = `图片=${counts.images ?? counts.total ?? 0}`;
    }
    if (counts.empty) countText += ` · 空图=${counts.empty}`;
    const bits = [`导出 ${result.format} 完成`, countText];
    if (result.issues?.length) bits.push(`校验发现 ${result.issues.length} 个问题：${result.issues[0]}`);
    if (result.skipped?.length) bits.push(`跳过 ${result.skipped.length} 张（图片缺失）`);
    setStatus(bits.join(" · "), result.issues?.length ? "warn" : "ok");
  } catch (error) {
    setStatus(`导出失败：${error.message}`, "bad");
  }
}

// ---------------------------------------------------------------------------
// 工具与选项
// ---------------------------------------------------------------------------
function setTool(tool) {
  state.tool = tool;
  state.pending = [];
  document.querySelectorAll("#panel-annotate .tool").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === tool);
  });
  if (el.canvas) el.canvas.style.cursor = tool === "select" ? "default" : "crosshair";
}

// ---------------------------------------------------------------------------
// 导航缩略图
// ---------------------------------------------------------------------------
function handleMini(event) {
  const image = currentImage();
  const mini = el.mini;
  if (!image || !mini) return;
  const rect = mini.getBoundingClientRect();
  const scale = Math.min(rect.width / (image.width || 1), rect.height / (image.height || 1));
  const dx = (rect.width - (image.width || 1) * scale) / 2;
  const dy = (rect.height - (image.height || 1) * scale) / 2;
  const imageX = (event.clientX - rect.left - dx) / scale;
  const imageY = (event.clientY - rect.top - dy) / scale;
  state.view.offset = {
    x: el.stage.clientWidth / 2 - imageX * state.view.scale,
    y: el.stage.clientHeight / 2 - imageY * state.view.scale,
  };
  renderCanvas();
}

// ---------------------------------------------------------------------------
// 快捷键
// ---------------------------------------------------------------------------
function onKeyDown(event) {
  const tag = event.target?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const panel = document.getElementById("panel-annotate");
  if (!panel || !panel.classList.contains("active")) return;

  const control = event.ctrlKey || event.metaKey;
  if (control && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
    return;
  }
  if (control && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redo();
    return;
  }
  if (control && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveProject({ silent: false });
    return;
  }
  if (control) return;

  if (event.key === "Escape") {
    state.pending = [];
    state.selectedId = "";
    setTool("select");
    renderAll();
    renderCanvas();
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    deleteSelected();
    return;
  }
  if (event.key === "Enter") {
    // 多边形 / 关键点用 Enter 结束，比"精确点回起点"宽容
    if (state.tool === "polygon" && state.pending.length >= 3) finishShape(state.pending);
    else if (state.tool === "keypoints" && state.pending.length) finishShape(state.pending);
    return;
  }
  const lower = event.key.toLowerCase();
  if (lower === "f") {
    fitView();
    renderCanvas();
    return;
  }
  if (TOOL_SHORTCUTS[lower]) setTool(TOOL_SHORTCUTS[lower]);
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
export function initAnnotate() {
  if (!document.getElementById("panel-annotate")) return;

  Object.assign(el, {
    path: document.getElementById("annoPath"),
    saveState: document.getElementById("annoSaveState"),
    imageList: document.getElementById("annoImageList"),
    imageCount: document.getElementById("annoImageCount"),
    addRoot: document.getElementById("annoAddRoot"),
    stage: document.getElementById("annoStage"),
    canvas: document.getElementById("annoCanvas"),
    mini: document.getElementById("annoMini"),
    empty: document.getElementById("annoEmpty"),
    status: document.getElementById("annoStatus"),
    zoom: document.getElementById("annoZoom"),
    undo: document.getElementById("annoUndo"),
    redo: document.getElementById("annoRedo"),
    classList: document.getElementById("annoClassList"),
    classCount: document.getElementById("annoClassCount"),
    classNameInput: document.getElementById("annoClassName"),
    draftList: document.getElementById("annoDraftList"),
    draftCount: document.getElementById("annoDraftCount"),
    props: document.getElementById("annoProps"),
    exportFormat: document.getElementById("annoExportFormat"),
    exportOut: document.getElementById("annoExportOut"),
    autoLabelBox: document.getElementById("annoAutoLabelBox"),
    alEngine: document.getElementById("annoAlEngine"),
    alLabels: document.getElementById("annoAlLabels"),
    alConf: document.getElementById("annoAlConf"),
    alNms: document.getElementById("annoAlNms"),
    alService: document.getElementById("annoAlService"),
    conflictModal: document.getElementById("annoConflictModal"),
    conflictText: document.getElementById("annoConflictText"),
  });

  document.getElementById("annoOpen")?.addEventListener("click", () => openProject((el.path.value || "").trim()));
  document.getElementById("annoCreate")?.addEventListener("click", createProject);
  document.getElementById("annoSaveNow")?.addEventListener("click", () => saveProject({ silent: false }));
  document.getElementById("annoAddImages")?.addEventListener("click", addImages);
  document.getElementById("annoUndo")?.addEventListener("click", undo);
  document.getElementById("annoRedo")?.addEventListener("click", redo);
  document.getElementById("annoDelete")?.addEventListener("click", deleteSelected);
  document.getElementById("annoFit")?.addEventListener("click", () => {
    fitView();
    renderCanvas();
  });
  document.getElementById("annoZoomReset")?.addEventListener("click", () => {
    const image = currentImage();
    if (!image) return;
    state.view = {
      scale: 1,
      offset: { x: (el.stage.clientWidth - image.width) / 2, y: (el.stage.clientHeight - image.height) / 2 },
    };
    updateZoomLabel();
    renderCanvas();
  });
  document.getElementById("annoAutoLabel")?.addEventListener("click", runAutoLabel);
  document.getElementById("annoExport")?.addEventListener("click", runExport);
  document.getElementById("annoClassAdd")?.addEventListener("click", addClass);
  document.getElementById("annoAcceptAll")?.addEventListener("click", acceptAllDrafts);
  document.getElementById("annoRejectAll")?.addEventListener("click", rejectAllDrafts);

  document.getElementById("annoConflictReload")?.addEventListener("click", async () => {
    el.conflictModal?.classList.add("hidden");
    await openProject(state.path, { discard: true });
  });
  document.getElementById("annoConflictForce")?.addEventListener("click", async () => {
    el.conflictModal?.classList.add("hidden");
    // 强制覆盖要以磁盘最新 rev 为基线，否则下一次自动保存又会冲突
    if (state.conflict?.payload?.disk?.rev != null) state.meta.rev = state.conflict.payload.disk.rev;
    await saveProject({ force: true, silent: false });
  });

  document.querySelectorAll("#panel-annotate .tool").forEach((button) => {
    button.addEventListener("click", () => setTool(button.dataset.tool));
  });

  const bindToggle = (id, key) => {
    const input = document.getElementById(id);
    if (!input) return;
    input.checked = state.opts[key];
    input.addEventListener("change", () => {
      state.opts[key] = input.checked;
      renderCanvas();
    });
  };
  bindToggle("annoShowShapes", "showShapes");
  bindToggle("annoShowLabels", "showLabels");
  bindToggle("annoShowScore", "showScore");
  bindToggle("annoCrosshair", "crosshair");
  bindToggle("annoSnap", "snap");

  const alpha = document.getElementById("annoFillAlpha");
  alpha?.addEventListener("input", () => {
    state.opts.fillAlpha = Number(alpha.value) / 100;
    renderCanvas();
  });

  el.canvas?.addEventListener("pointerdown", onPointerDown);
  el.canvas?.addEventListener("pointermove", onPointerMove);
  el.canvas?.addEventListener("pointerup", onPointerUp);
  el.canvas?.addEventListener("pointercancel", onPointerUp);
  el.canvas?.addEventListener("pointerleave", () => {
    state.cursor = null;
    state.hover = null;
    if (!state.drag) renderCanvas();
  });
  el.canvas?.addEventListener("wheel", onWheel, { passive: false });
  el.canvas?.addEventListener("contextmenu", (event) => event.preventDefault());
  // 双击：多边形 / 关键点收尾
  el.canvas?.addEventListener("dblclick", () => {
    const points = dropTrailingDuplicates(state.pending);
    if (state.tool === "polygon" && points.length >= 3) finishShape(points);
    else if (state.tool === "keypoints" && points.length) finishShape(points);
  });

  if (el.mini) {    let dragging = false;
    el.mini.addEventListener("pointerdown", (event) => {
      dragging = true;
      el.mini.setPointerCapture(event.pointerId);
      handleMini(event);
    });
    el.mini.addEventListener("pointermove", (event) => {
      if (dragging) handleMini(event);
    });
    el.mini.addEventListener("pointerup", () => {
      dragging = false;
    });
  }

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("resize", () => {
    resizeCanvas();
    renderCanvas();
  });
  // 面板在隐藏状态下 clientWidth 为 0，画布会被初始化成 1×1。
  // 两个兜底：页签切换时主动补一次尺寸，以及 ResizeObserver 在布局稳定后再补一次。
  document.addEventListener("tab-change", (event) => {
    if (event.detail?.name !== "annotate") return;
    resizeCanvas();
    // 只在"从没成功适应过窗口"时重算，避免把用户调好的缩放重置掉
    if (!state.viewValid && currentImage()) fitView();
    renderCanvas();
  });
  if (typeof ResizeObserver !== "undefined" && el.stage) {
    let lastWidth = 0;
    const observer = new ResizeObserver(() => {
      const width = el.stage.clientWidth;
      if (width === lastWidth) return;
      lastWidth = width;
      resizeCanvas();
      if (!state.viewValid && currentImage()) fitView();
      renderCanvas();
    });
    observer.observe(el.stage);
  }
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) {
      event.preventDefault();
      event.returnValue = "有未保存的标注，确定离开？";
    }
  });

  // 只读调试钩子：浏览器级 e2e 测试需要断言内部状态（工具、拖拽阶段、点数），
  // 没有它就只能靠像素猜测。只暴露快照，不提供任何写入入口。
  window.__annoDebug = () => ({
    path: state.path,
    tool: state.tool,
    pending: state.pending.length,
    drag: state.drag ? state.drag.kind : null,
    hover: state.hover ? { x: state.hover.image.x, y: state.hover.image.y } : null,
    view: { scale: state.view.scale, x: state.view.offset.x, y: state.view.offset.y },
    selectedId: state.selectedId,
    dirty: state.dirty,
    rev: state.meta.rev,
    exists: state.exists,
    objects: (currentImage()?.annotations || []).map((obj) => ({
      id: obj.id,
      shape_type: obj.shape_type,
      points: obj.points.map((p) => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]),
    })),
    history: state.undo.length,
    redo: state.redo.length,
  });

  resizeCanvas();
  renderAll();
  renderCanvas();
}

