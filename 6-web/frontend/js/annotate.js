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
  browse,
  cancelModelDownload,
  downloadModel,
  exportAnnotationProject,
  getBatchAutoLabel,
  getModelDownload,
  importAnnotationProject,
  listBatchAutoLabel,
  listModelzoo,
  listModels,
  loadAnnotationDrafts,
  loadAnnotationProject,
  nativeDialog,
  promptLabel,
  saveAnnotationDrafts,
  saveAnnotationProject,
  saveAnnotationReview,
  scanAnnotationImages,
  startBatchAutoLabel,
  stopBatchAutoLabel,
  textLabel,
} from "./api.js";
import {
  clampScale,
  computeZoomTransform,
  normalizeWheelDelta,
  screenToImage,
  wheelDeltaToFactor,
} from "./viewer.js";
// 五步流程的判定全是纯函数（有单测 tests/test_annotate_flow.mjs 覆盖），
// 这里只负责把结果画到界面上、把按钮动作分发出去。
import {
  buildAnnoSteps,
  countAllDrafts,
  currentStep,
  hasCustomClass,
  removeImageAt,
  resolveGuideAction,
  shouldExpandDrafts,
} from "./annotateflow.js";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/** 撤销历史深度。对齐 Qt 版的 50，兼顾内存与"够用"。 */
const HISTORY_LIMIT = 50;
/** 自动保存防抖间隔（毫秒）。改慢会丢更多编辑，改快会把磁盘打满。 */
const SAVE_DEBOUNCE = 3000;
/** 草稿落盘防抖：草稿比工程小得多，且审核是连续操作，短一点手感更好。 */
const DRAFTS_SAVE_DEBOUNCE = 800;
/**
 * 草稿置信度分档。
 *
 * 阈值不是拍脑袋：`0.70` 是 Ultralytics 默认 val 的置信度基准之上，
 * 现场审核时"基本不用看"的那一档；`0.40` 以下基本是模型在猜，需要人逐条看。
 * 两侧都取闭区间上界，避免 0.7 / 0.4 正好落在缝里。
 */
const DRAFT_TIER_HIGH = 0.7;
const DRAFT_TIER_MID = 0.4;
const DRAFT_TIER_COLORS = {
  // 与 style.css 的 --ok / --amber / --danger 取同一组值：
  // 列表用 CSS、画布用 JS，两边数值不同的话同一批草稿会有两种"绿"，看着像 bug。
  high: [52, 211, 153],
  mid: [251, 191, 36],
  low: [248, 113, 113],
  unknown: [251, 191, 36],
};
const DRAFT_TIER_TEXT = { high: "高", mid: "中", low: "低", unknown: "?" };
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
  // 关键点组从 K 改成 G：K 让给了「标为已审核」——审核是每个工程都要用的动作，
  // 而关键点是很少用的工具，占用高频键会让审核流程每次都要先想一下。
  g: "keypoints",
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
  /** 表格高亮滚动只在「选中对象变了」时做一次，避免每次重绘都把表格拉回去。 */
  boxScrollId: "",
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
  /** 最近一次「导入 YOLO 数据集」的预览报告；有值才允许点导入，避免参数改了还按旧结果落盘。 */
  importReport: null,
  /**
   * 草稿脏图集合：只有这些图片的草稿会被提交。
   *
   * 为什么要按图记脏：批量预标注在后台直接写旁挂文件，如果前端每次
   * 都"整份覆盖"，就会把批量刚写进去的其他图片清空。
   */
  draftDirty: new Map(),
  draftSeq: 0,
  draftSaveTimer: 0,
  draftSaveTask: null,
  /** 批量预标注任务快照（后端任务表的镜像，用于进度条与停止按钮）。 */
  batchJob: null,
  /** 批量任务的轮询定时器 id。 */
  batchTimer: 0,
  /** 训练产物是否已经拉过（懒加载：不展开预标注参数就不扫目录）。 */
  engineLoaded: false,
  /**
   * 预标注通道：`engine` 自有引擎 / `prompt` 点框提示（SAM）/ `text` 文本提示。
   *
   * 三选一而不是"可叠加"：提示通道要求画布的点击变成"放点"，与绘制工具
   * 抢同一套鼠标事件。做成叠加态的话，用户点一下不确定是画框还是放点，
   * 而两者都是"改数据"的操作，猜错的代价是白画一个框。
   */
  alChannel: "engine",
  /**
   * 提示点：`{points: [[x,y]], labels: [1|0]}`（图像坐标）。
   * label 1 = 正点（要这里），0 = 负点（不要这里），与 SAM 的 decoder 标签一致。
   */
  promptPoints: [],
  promptLabels: [],
  /** 拖拽中的框提示 `[[x1,y1],[x2,y2]]`（图像坐标），松手后才有值。 */
  promptBox: null,
  /** 正在拖框（`{startScreen, start, moved}`），用于区分"点一下"和"拉一个框"。 */
  promptDrag: null,
  /** 提示式推理是否在跑：防止连点 Enter 重复发请求。 */
  promptBusy: false,
  /**
   * 审核状态：图片 id → `{state: "checked", at_ns}`。
   *
   * 和草稿分开存（`state.drafts` 是 Map<id, 草稿数组>）：审核状态属于"图"，
   * 草稿属于"候选"，两者数量与生命周期都不一样，混在一起会让"清空草稿"
   * 顺手把审核进度一起清掉。
   */
  review: new Map(),
  /** 审核状态是否还在等写盘（连按 K 时合并成一次请求）。 */
  reviewDirty: false,
  reviewSaveTimer: 0,
  /** 内置模型清单与下载任务：`{items, jobs, dir}`，null 表示还没拉过。 */
  modelzoo: null,
  /** 正在轮询的下载任务 id 集合（避免同一任务起两个定时器）。 */
  modelPolls: new Set(),
  /**
   * 右栏「进阶：预标注草稿与审核」是否被用户手动动过。
   *
   * false：按内容自动展开/收起（有草稿就展开，让批量预标注的结果直接可见）；
   * true ：用户点过折叠条，之后一律听用户的，不再自动弹开或收起。
   * 切工程时重置，因为那是另一份草稿池，用户的临时偏好不该跨工程传染。
   */
  advDraftTouched: false,
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

// ---------------------------------------------------------------------------
// 草稿置信度分档
// ---------------------------------------------------------------------------
/**
 * 草稿置信度档位。
 *
 * `score < 0` 是"通道没给置信度"（外部服务常见），单独一档：
 * 把它算成低置信度会让整批红框看起来像"模型全在猜"，反而让人不敢信绿框。
 */
function draftTier(score) {
  const value = Number(score);
  if (!Number.isFinite(value) || value < 0) return "unknown";
  if (value >= DRAFT_TIER_HIGH) return "high";
  if (value >= DRAFT_TIER_MID) return "mid";
  return "low";
}

function draftTierColor(score) {
  return DRAFT_TIER_COLORS[draftTier(score)];
}

function draftTierText(score) {
  return DRAFT_TIER_TEXT[draftTier(score)];
}

/** 草稿列表项的颜色类名，CSS 与画布共用同一套分档。 */
function draftTierClass(score) {
  return `tier-${draftTier(score)}`;
}

// ---------------------------------------------------------------------------
// 草稿落盘：旁挂 `<工程名>.drafts.json`
// ---------------------------------------------------------------------------
/** 标记某张图片的草稿变了。只提交这些图片，避免整份覆盖清掉批量任务刚写的内容。 */
function markDraftsDirty(imageId) {
  const id = imageId || currentImage()?.id || "";
  if (!id) return;
  state.draftSeq += 1;
  state.draftDirty.set(id, state.draftSeq);
  scheduleDraftSave();
}

function scheduleDraftSave() {
  if (!state.path) return;
  if (state.draftSaveTimer) clearTimeout(state.draftSaveTimer);
  state.draftSaveTimer = window.setTimeout(() => {
    state.draftSaveTimer = 0;
    flushDraftChanges();
  }, DRAFTS_SAVE_DEBOUNCE);
}

/**
 * 把脏图写回旁挂文件。
 *
 * 请求按顺序串起来（`draftSaveTask` 链）：两次增量写如果并发，后发出的可能先到，
 * 磁盘上就会留下一份旧内容。链式执行最坏只是多一次往返。
 */
function flushDraftChanges({ silent = true } = {}) {
  const run = () => doFlushDrafts({ silent });
  const task = (state.draftSaveTask || Promise.resolve()).then(run, run);
  state.draftSaveTask = task;
  task.then(
    () => {
      if (state.draftSaveTask === task) state.draftSaveTask = null;
    },
    () => {
      if (state.draftSaveTask === task) state.draftSaveTask = null;
    },
  );
  return task;
}

async function doFlushDrafts({ silent }) {
  if (!state.path || state.draftDirty.size === 0) return;
  const path = state.path;
  // 快照脏集合：提交期间用户可能又改了某张图，那批改动必须留到下一轮
  const entries = [...state.draftDirty.entries()];
  const images = {};
  for (const [imageId] of entries) images[imageId] = state.drafts.get(imageId) || [];
  try {
    const result = await saveAnnotationDrafts({ path, images });
    if (path !== state.path) return; // 期间切了工程：这次结果只对旧工程有意义
    for (const [imageId, seq] of entries) {
      if (state.draftDirty.get(imageId) === seq) state.draftDirty.delete(imageId);
    }
    if (!silent) setStatus(`草稿已保存（共 ${result.count ?? 0} 条）`, "ok");
  } catch (error) {
    setStatus(`草稿保存失败：${error.message}`, "bad");
  }
}

/** 打开工程后把旁挂草稿读回内存。失败不阻塞工程本身。 */
async function loadDraftsFromDisk() {
  state.drafts.clear();
  state.draftDirty.clear();
  if (!state.path) return;
  const path = state.path;
  try {
    const data = await loadAnnotationDrafts(path);
    if (path !== state.path) return;
    for (const [imageId, drafts] of Object.entries(data.drafts || {})) {
      if (!Array.isArray(drafts) || !drafts.length) continue;
      state.drafts.set(imageId, drafts.map((draft) => ({ ...draft, _draft: true })));
    }
    // 审核状态和草稿同住一个文件：一起读回来，刷新后"审到哪了"不会倒退
    state.review = new Map(Object.entries(data.review || {}));
    if (data.warnings?.length) {
      setStatus(`草稿文件有问题：${data.warnings[0]}`, "warn");
    }
  } catch (error) {
    setStatus(`读取草稿失败：${error.message}`, "warn");
  }
}

// ---------------------------------------------------------------------------
// 审核状态（已审核 / 未审核）
// ---------------------------------------------------------------------------
/**
 * 待写盘的审核变更：图片 id → `"checked" | "unchecked"`。
 *
 * 为什么需要"待写盘"这一层：连续按 K 浏览 20 张是一秒内的事，每张都发一次请求
 * 会让浏览器排 20 个连接，而写的是同一个文件，服务端串行处理后有一半是白跑的。
 * 合并成一次写，语义完全一样（每次都是"这几张图的状态变成这样"）。
 */
let reviewPending = new Map();
/** 审核状态写盘请求的链：保证两次写不并发（并发会互相覆盖）。 */
let reviewSaveTask = null;

function isReviewed(imageId) {
  return Boolean(imageId && state.review.has(imageId));
}

function reviewedStats() {
  const total = state.project?.images?.length || 0;
  let checked = 0;
  for (const image of state.project?.images || []) {
    if (state.review.has(image.id)) checked += 1;
  }
  return { checked, total };
}

function scheduleReviewSave() {
  if (!state.path) return;
  if (state.reviewSaveTimer) clearTimeout(state.reviewSaveTimer);
  state.reviewSaveTimer = window.setTimeout(() => {
    state.reviewSaveTimer = 0;
    flushReviewChanges();
  }, 300);
}

/**
 * 把待写的审核变更落盘。
 *
 * 成功时用后端返回的 `review` 覆盖内存：后端会把 `unchecked` 从文件里删掉，
 * 直接拿服务端的结果当权威，比在前端自己模拟"删除"更不容易两边不一致。
 */
function flushReviewChanges({ silent = true } = {}) {
  const run = () => doFlushReview({ silent });
  const task = (reviewSaveTask || Promise.resolve()).then(run, run);
  reviewSaveTask = task;
  task.then(
    () => {
      if (reviewSaveTask === task) reviewSaveTask = null;
    },
    () => {
      if (reviewSaveTask === task) reviewSaveTask = null;
    },
  );
  return task;
}

async function doFlushReview({ silent }) {
  if (!state.path || reviewPending.size === 0) return;
  const path = state.path;
  const batch = reviewPending;
  reviewPending = new Map();
  const review = {};
  for (const [imageId, stateName] of batch) {
    review[imageId] = { state: stateName, at_ns: String(Date.now()) };
  }
  try {
    const result = await saveAnnotationReview({ path, review });
    if (path !== state.path) return;
    // 期间用户可能又按了几张：把它们重新并进待写集合
    state.review = new Map(Object.entries(result.review || {}));
    if (reviewPending.size) scheduleReviewSave();
    renderReviewState();
    renderImageList();
    if (!silent) setStatus(`审核状态已保存（已审核 ${state.review.size} 张）`, "ok");
  } catch (error) {
    // 失败了要把这批放回去，否则界面显示"已审核"而磁盘上没有，刷新就倒退
    for (const [imageId, stateName] of batch) {
      if (!reviewPending.has(imageId)) reviewPending.set(imageId, stateName);
    }
    setStatus(`审核状态保存失败：${error.message}`, "bad");
  }
}

/** 标为已审核（并可选跳到下一张未审核）。 */
async function markReviewed({ advance = true } = {}) {
  const image = currentImage();
  if (!image) {
    setStatus("先选一张图片", "bad");
    return;
  }
  state.review.set(image.id, { state: "checked", at_ns: String(Date.now()) });
  reviewPending.set(image.id, "checked");
  renderReviewState();
  renderImageList();
  if (advance) {
    const moved = await nextUnreviewed();
    if (!moved) setStatus(`已审核 ${image.id}（已是最后一张未审核的图片）`, "ok");
  } else {
    setStatus(`已把 ${image.id} 标为已审核`, "ok");
  }
  flushReviewChanges({ silent: true });
}

/** 取消当前图片的审核状态。 */
function unmarkReviewed() {
  const image = currentImage();
  if (!image) return;
  if (!state.review.has(image.id)) {
    setStatus("当前图片本来就没有审核状态", "warn");
    return;
  }
  state.review.delete(image.id);
  reviewPending.set(image.id, "unchecked");
  renderReviewState();
  renderImageList();
  flushReviewChanges({ silent: true });
  setStatus(`已取消 ${image.id} 的审核状态`);
}

/**
 * 跳到下一张未审核的图片。
 *
 * 只往后找，找不到再绕回开头找一遍（只绕一次）。不无限循环：
 * 全部审完时应该停下来提示"都审过了"，而不是把用户又送回第一张。
 */
async function nextUnreviewed() {
  const images = state.project?.images || [];
  if (!images.length) return false;
  const start = state.currentIndex;
  for (let step = 1; step <= images.length; step += 1) {
    const index = (start + step) % images.length;
    if (index === start) break;
    if (!state.review.has(images[index].id)) {
      await switchImage(index);
      return true;
    }
  }
  return false;
}

function renderReviewState() {
  if (el.reviewCount) {
    const { checked, total } = reviewedStats();
    el.reviewCount.textContent = `${checked}/${total}`;
    el.reviewCount.title = `已审核 ${checked} 张，共 ${total} 张`;
  }
  const image = currentImage();
  const checked = isReviewed(image?.id);
  if (el.unmarkReviewed) el.unmarkReviewed.disabled = !checked;
}

// ---------------------------------------------------------------------------
// 提示式预标注（点 / 框 / 文本）
// ---------------------------------------------------------------------------
/** 当前是否在提示模式：只有这个模式下画布点击变成"放点"。 */
function promptActive() {
  return state.alChannel === "prompt";
}

function clearPrompt() {
  state.promptPoints = [];
  state.promptLabels = [];
  state.promptBox = null;
  state.promptDrag = null;
  renderCanvas();
}

function setAlChannel(channel) {
  const next = ["engine", "prompt", "text"].includes(channel) ? channel : "engine";
  state.alChannel = next;
  if (next !== "prompt") clearPrompt();
  if (el.alChannel) el.alChannel.value = next;
  el.chEngine?.classList.toggle("hidden", next !== "engine");
  el.chPrompt?.classList.toggle("hidden", next !== "prompt");
  el.chText?.classList.toggle("hidden", next !== "text");
  // 切到提示模式就顺手把绘制工具收回 select：否则"点一下"既可能是放点
  // 也可能是画框，取决于用户上次选的工具，而界面上看不出来。
  // 工具按钮也一并禁用——按钮亮着却画不出东西，比按钮变灰更让人困惑。
  if (next === "prompt" && state.tool !== "select") setTool("select");
  document.querySelectorAll("#panel-annotate .tool").forEach((button) => {
    button.disabled = next === "prompt" && button.dataset.tool !== "select";
  });
  if (next === "text" || next === "prompt") refreshModelzoo();
  renderCanvas();
}

/** 读界面上的掩码精细度，夹到合理区间（0 会让 approxPolyDP 退化成原始轮廓）。 */
function promptEpsilon() {
  const value = Number(el.alEpsilon?.value ?? 0.001);
  if (!Number.isFinite(value) || value <= 0) return 0.001;
  return Math.min(0.05, value);
}

function promptSuggestionLabel() {
  const classes = state.project?.classes || [];
  return classes[0]?.name || "";
}

async function runPromptLabel() {
  const image = currentImage();
  if (!image) {
    setStatus("先选一张图片", "bad");
    return;
  }
  if (!state.promptPoints.length && !state.promptBox) {
    setStatus("先在画布上点一个正点、或拉一个框，再按 Enter 执行", "warn");
    return;
  }
  if (state.promptBusy) return;
  state.promptBusy = true;
  setStatus("SAM 正在生成掩码…（首次加载模型会慢一些）", "warn");
  try {
    const result = await promptLabel({
      image: imageApiPath(image),
      points: state.promptPoints,
      labels: state.promptLabels,
      box: state.promptBox || [],
      epsilon: promptEpsilon(),
      label: (el.alPrompt?.value || "").trim() || promptSuggestionLabel(),
    });
    const drafts = result.drafts || [];
    // 追加而不是替换：用户经常对同一张图连点几处分别出草稿，
    // 替换会把前一次刚点出来的东西丢掉。
    const pool = currentDrafts();
    state.drafts.set(image.id, pool.concat(drafts));
    markDraftsDirty(image.id);
    const bits = [`新增 ${drafts.length} 个多边形草稿`];
    if (result.warnings?.length) bits.push(result.warnings.join("；"));
    setStatus(bits.join(" · "), drafts.length ? "ok" : "warn");
    renderAll();
    renderCanvas();
  } catch (error) {
    setStatus(`提示式预标注失败：${error.message}`, "bad");
  } finally {
    state.promptBusy = false;
  }
}

function textPromptValue() {
  return (el.alPrompt?.value || "").trim();
}

async function runTextLabel() {
  const image = currentImage();
  if (!image) {
    setStatus("先选一张图片", "bad");
    return;
  }
  const prompt = textPromptValue();
  if (!prompt) {
    setStatus("先在「预标注参数 → 文本提示」里填一句话，例如 missing hole", "bad");
    el.autoLabelBox?.setAttribute("open", "");
    return;
  }
  if (state.promptBusy) return;
  state.promptBusy = true;
  setStatus(`文本预标注中…（${prompt}）首帧要加载模型，之后快很多`, "warn");
  try {
    const result = await textLabel({
      image: imageApiPath(image),
      prompt,
      box_thr: Number(el.alBoxThr?.value || 0.3),
      text_thr: Number(el.alTextThr?.value || 0.25),
      refine: Boolean(el.alRefine?.checked),
      epsilon: promptEpsilon(),
    });
    const drafts = result.drafts || [];
    state.drafts.set(image.id, drafts);
    markDraftsDirty(image.id);
    const bits = [`${drafts.length} 个草稿（文本：${prompt}）`];
    if (result.warnings?.length) bits.push(result.warnings.join("；"));
    setStatus(bits.join(" · "), drafts.length ? "ok" : "warn");
    renderAll();
    renderCanvas();
  } catch (error) {
    setStatus(`文本预标注失败：${error.message}`, "bad");
  } finally {
    state.promptBusy = false;
  }
}

/** 当前通道对应的单图执行函数（工具栏「预标注」按钮与 Enter 共用）。 */
function runCurrentChannel() {
  if (state.alChannel === "prompt") return runPromptLabel();
  if (state.alChannel === "text") return runTextLabel();
  return runAutoLabel();
}

// ---------------------------------------------------------------------------
// 内置模型仓库
// ---------------------------------------------------------------------------
async function refreshModelzoo({ manual = false } = {}) {
  if (manual) setStatus("正在读取模型清单…", "warn");
  try {
    const data = await listModelzoo();
    state.modelzoo = data;
    renderModelzoo();
    // 有正在跑的任务就接着轮询：刷新页面后进度条不能凭空消失
    for (const job of data.jobs || []) {
      if (job.status === "running") pollModelDownload(job.id);
    }
    if (manual) {
      const missing = Number(data.missing || 0);
      setStatus(
        missing ? `模型仓库：还缺 ${missing} 个模型（点「下载」）` : "模型仓库：全部就绪",
        missing ? "warn" : "ok",
      );
    }
  } catch (error) {
    if (manual) setStatus(`读取模型清单失败：${error.message}`, "bad");
  }
}

function renderModelzoo() {
  if (!el.modelList) return;
  const data = state.modelzoo;
  el.modelList.innerHTML = "";
  if (!data) {
    if (el.modelCount) el.modelCount.textContent = "—";
    return;
  }
  const items = data.items || [];
  const present = items.filter((item) => item.present).length;
  if (el.modelCount) el.modelCount.textContent = `${present}/${items.length}`;
  if (el.modelHint) {
    const jobs = (data.jobs || []).filter((job) => job.status === "running");
    el.modelHint.textContent = jobs.length
      ? `正在下载：${jobs.map((job) => `${job.model} ${job.percent}%`).join("；")}`
      : `模型目录：${data.dir || ""}`;
  }
  // 运行中的任务按模型索引，用来显示进度条
  const running = new Map();
  for (const job of data.jobs || []) {
    if (job.status === "running") running.set(job.model, job);
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "anno-model" + (item.present ? " present" : "");
    const head = document.createElement("div");
    head.className = "anno-model-head";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.name;
    const stateTag = document.createElement("span");
    stateTag.className = `anno-model-state ${item.present ? "ok" : ""}`;
    stateTag.textContent = item.present ? "已就绪" : "未下载";
    head.append(name, stateTag);

    const use = document.createElement("p");
    use.className = "anno-hint";
    use.textContent = `${item.use} · ${item.license} · ${formatBytes(item.total_bytes)}`;
    li.append(head, use);

    const job = running.get(item.id);
    if (job) {
      const bar = document.createElement("div");
      bar.className = "anno-progress anno-progress-inline";
      const fill = document.createElement("div");
      fill.style.width = `${Math.min(100, job.percent || 0)}%`;
      bar.appendChild(fill);
      const meta = document.createElement("p");
      meta.className = "anno-hint";
      meta.textContent = `${job.percent || 0}% · ${formatBytes(job.downloaded)} / ${formatBytes(job.total)}`;
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "mini danger";
      cancel.textContent = "取消";
      cancel.addEventListener("click", () => cancelDownload(job.id));
      li.append(bar, meta, cancel);
    } else if (!item.present) {
      const download = document.createElement("button");
      download.type = "button";
      download.className = "mini accent";
      download.textContent = `下载 · ${formatBytes(item.total_bytes)}`;
      download.addEventListener("click", () => startDownload(item.id));
      li.appendChild(download);
    }
    el.modelList.appendChild(li);
  }
}

function formatBytes(value) {
  const number = Number(value || 0);
  if (!number) return "未知大小";
  if (number < 1024 * 1024) return `${(number / 1024).toFixed(0)} KB`;
  return `${(number / (1024 * 1024)).toFixed(1)} MB`;
}

async function startDownload(id) {
  setStatus(`开始下载 ${id}…（支持断点续传）`, "warn");
  try {
    const result = await downloadModel(id);
    if (result.job?.already) {
      setStatus(`${id} 已就绪`, "ok");
      await refreshModelzoo({ manual: true });
      return;
    }
    if (result.job?.id) pollModelDownload(result.job.id);
    await refreshModelzoo();
  } catch (error) {
    setStatus(`下载启动失败：${error.message}`, "bad");
  }
}

async function cancelDownload(jobId) {
  try {
    await cancelModelDownload(jobId);
    setStatus("已请求取消，当前分片写完就停", "warn");
    await refreshModelzoo();
  } catch (error) {
    setStatus(`取消失败：${error.message}`, "bad");
  }
}

/**
 * 轮询下载进度。
 *
 * 用 `setTimeout` 自递归而不是 `setInterval`：请求本身有往返时间，
 * 固定间隔会在网络慢时堆叠请求，堆到十几个之后进度反而更卡。
 */
function pollModelDownload(jobId) {
  if (!jobId || state.modelPolls.has(jobId)) return;
  state.modelPolls.add(jobId);
  const tick = async () => {
    try {
      const result = await getModelDownload(jobId);
      const job = result.job;
      if (job.status === "running") {
        if (state.modelzoo) {
          state.modelzoo.jobs = [job, ...(state.modelzoo.jobs || []).filter((item) => item.id !== job.id)];
          renderModelzoo();
        }
        window.setTimeout(tick, 500);
        return;
      }
      state.modelPolls.delete(jobId);
      if (job.status === "done") {
        setStatus(`模型下载完成：${job.model}`, "ok");
      } else if (job.status === "cancelled") {
        setStatus(`已取消下载：${job.model}`, "warn");
      } else {
        setStatus(`下载失败：${job.model} · ${job.error || "未知原因"}`, "bad");
      }
      await refreshModelzoo();
    } catch (error) {
      state.modelPolls.delete(jobId);
      setStatus(`下载进度查询失败：${error.message}`, "bad");
    }
  };
  window.setTimeout(tick, 400);
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
  await flushDraftChanges({ silent: true });
  try {
    const data = await loadAnnotationProject(path);
    await adoptProject(data, path);
    setStatus(data.exists ? `已打开 rev=${data.meta.rev}` : "工程不存在，可点「新建」开始", data.exists ? "" : "warn");
  } catch (error) {
    setStatus(`打开失败：${error.message}`, "bad");
  }
}

async function adoptProject(data, path) {
  state.path = data.path || path;
  state.exists = Boolean(data.exists);
  state.project = data.project;
  state.meta = data.meta || { rev: 0, mtime_ns: "0", size: 0 };
  state.dirty = false;
  state.undo.length = 0;
  state.redo.length = 0;
  state.drafts.clear();
  state.draftDirty.clear();
  state.review.clear();
  // 换了工程就重置进阶块的"用户偏好"：新工程可能有草稿需要自动展开
  state.advDraftTouched = false;
  reviewPending = new Map();
  state.hidden.clear();
  state.selectedId = "";
  state.pending = [];
  clearPrompt();
  state.viewValid = false;
  state.currentIndex = state.project.images.length ? 0 : -1;
  if (el.path) el.path.value = state.path;

  // 草稿来自旁挂文件：刷新/重开工程必须能把"还没审的候选"接回来，
  // 否则批量预标注跑一次、F5 一下，几百个框就白跑了。
  await loadDraftsFromDisk();
  resumeBatchJob();

  updateHistoryButtons();
  renderAll();
  if (state.currentIndex >= 0) loadCurrentImage();
  else renderCanvas();
}

/**
 * 把「只给了目录」的工程路径补成 `<目录名>.tfanno`。
 *
 * 浏览弹窗受浏览器限制只能选目录，选中后 `annoPath` 里是个目录；
 * 直接点「新建」会被后端以「扩展名必须是 .tfanno」拒绝，用户不知道该怎么改名。
 * 这里只在**确认它是已存在的目录**时才改写，避免把用户手写的半截文件名猜错。
 */
async function resolveProjectPath(raw) {
  if (!raw || /\.tfanno$/i.test(raw)) return { path: raw, renamed: false };
  const dir = raw.replace(/[\\/]+$/, "");
  try {
    await browse(dir); // 是目录才会成功；是文件/不存在都会抛错
  } catch {
    return { path: raw, renamed: false };
  }
  const name = dir.split(/[\\/]/).pop() || "project";
  const sep = dir.includes("\\") ? "\\" : "/";
  return { path: `${dir}${sep}${name}.tfanno`, renamed: true };
}

async function createProject() {
  const raw = (el.path?.value || "").trim();
  if (!raw) {
    setStatus("请先填写要创建的 .tfanno 路径", "bad");
    return;
  }
  const { path, renamed } = await resolveProjectPath(raw);
  if (el.path) el.path.value = path;
  if (renamed) {
    setStatus(`「浏览」只能选目录，已按目录名补成文件名：${path.split(/[\\/]/).pop()}（可手工改）`, "warn");
  }
  // 1) 先把手上的修改落盘：切工程不能顺手带走当前工程的未保存内容。
  await flushSave();
  await flushDraftChanges({ silent: true });
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

  const task = document.getElementById("annoTask")?.value || "detect";
  const project = {
    version: 1,
    task,
    rev: 0,
    classes: task === "classify"
      ? [
          { id: 0, name: "良品", color: defaultColor(0) },
          { id: 1, name: "不良", color: defaultColor(1) },
        ]
      : [{ id: 0, name: "object", color: [31, 182, 255] }],
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
    state.drafts.clear();
    state.draftDirty.clear();
    state.review.clear();
    state.advDraftTouched = false;
    reviewPending = new Map();
    if (el.path) el.path.value = state.path;
    // 同名旁挂草稿属于"旧工程"：新工程里既没有那些图片，留着只会污染草稿池
    try {
      await saveAnnotationDrafts({ path: state.path, images: {}, replace: true });
    } catch {
      /* 清草稿失败不影响新建本身，下一次保存草稿会覆盖它 */
    }
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
/**
 * 传给后端的图片路径。
 *
 * `.tfanno` 里 Qt 存的是相对工程目录的路径，浏览器只能把字符串原样拼进
 * `/api/annotation/image?path=`，后端按服务进程的 cwd 解析 → 图片全裂。
 * 后端加载工程时补了 `resolved`（绝对路径），这里必须优先用它。
 */
function imageApiPath(image) {
  if (!image) return "";
  return image.resolved || image.path || "";
}

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
      const meta = await annotationImageMeta(imageApiPath(image));
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
    setStatus(`图片无法显示：${imageApiPath(image)}（检查是否在 TRTFORGE_BROWSE_ROOTS 内）`, "bad");
  };
  bitmap.src = annotationImageUrl(imageApiPath(image));
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

function isClassify() {
  return state.project?.task === "classify";
}

function applyTaskChrome() {
  const classify = isClassify();
  document.querySelectorAll("#panel-annotate .anno-detect-only").forEach((node) => {
    node.hidden = classify;
  });
  document.querySelectorAll("#panel-annotate .anno-classify-only").forEach((node) => {
    node.hidden = !classify;
  });
  const task = document.getElementById("annoTask");
  if (task && state.project?.task) task.value = state.project.task;
  if (el.exportFormat) {
    for (const option of el.exportFormat.options) {
      option.disabled = classify && option.value !== "yolo";
    }
    if (classify) el.exportFormat.value = "yolo";
  }
  document.getElementById("annoAutoLabelBox")?.toggleAttribute("hidden", classify);
}

// ---------------------------------------------------------------------------
// 五步引导条（v1.35.0）
//
// 目标只有一个：打开标注页的人不用读文档就知道"现在该做什么"。
// 判定逻辑全在 annotateflow.js（纯函数、有单测），这里只做三件事：
// 收集现场数据 → 渲染步骤条 → 把主按钮分发给已有函数。
// ---------------------------------------------------------------------------

/** 引导条隐藏状态存在 localStorage：老用户收起过一次，不该每次刷新又冒出来。 */
const GUIDE_STORAGE = "trtforge.anno.guide";

/** 当前该高亮的提示文案；没有未完成步骤时给一句收尾。 */
function guideSuggestion(steps) {
  const step = currentStep(steps);
  if (!step) return { text: "", action: "", label: "" };
  if (steps.every((item) => item.state === "done")) {
    return { text: "五步都完成了：继续标下一张，或到「数据集」页切分后开始训练", action: "", label: "" };
  }
  return { text: step.hint, action: resolveGuideAction(step), label: step.actionLabel };
}

/** 收集这一步判定需要的数据。所有输入都来自内存状态，不额外发请求。 */
function guideContext() {
  const project = state.project;
  const image = currentImage();
  let annotationCount = 0;
  let classifiedCount = 0;
  for (const item of project?.images || []) {
    annotationCount += (item.annotations || []).length;
    if (item.class_id !== undefined && item.class_id !== null) classifiedCount += 1;
  }
  return {
    task: project?.task || "detect",
    hasProject: Boolean(project && state.path),
    imageCount: project?.images?.length || 0,
    classCount: project?.classes?.length || 0,
    customClass: hasCustomClass(project?.classes || [], project?.task || "detect"),
    // 画框进度按"当前图片"判定：用户关心的是这张标完了没有
    annotationCount: image ? (image.annotations || []).length : annotationCount,
    classifiedCount,
    // 只看 `.tfanno` 本身的未保存改动；草稿有 800ms 防抖，算进来会让第 5 步
    // 在"刚保存完"的瞬间还闪一下"未保存"
    dirty: Boolean(state.dirty),
  };
}

function renderGuide() {
  if (!el.steps) return;
  const steps = buildAnnoSteps(guideContext());
  el.steps.innerHTML = "";
  for (const step of steps) {
    const li = document.createElement("li");
    li.className = `anno-step ${step.state}`;
    li.dataset.step = step.id;
    li.title = step.hint;
    const no = document.createElement("span");
    no.className = "no";
    // 完成的步骤显示 ✓，比"数字变绿"更直观
    no.textContent = step.state === "done" ? "✓" : String(step.no);
    const title = document.createElement("span");
    title.textContent = step.title;
    li.append(no, title);
    li.addEventListener("click", () => runGuideAction(step.action));
    el.steps.appendChild(li);
  }
  const suggestion = guideSuggestion(steps);
  if (el.guideTip) el.guideTip.textContent = suggestion.text;
  if (el.guideAction) {
    el.guideAction.textContent = suggestion.label || "已就绪";
    el.guideAction.disabled = !suggestion.action;
  }
}

/** 引导条主按钮 / 步骤圆片共用的动作分发。每个动作都落到已有的按钮或状态上。 */
function runGuideAction(action) {
  switch (action) {
    case "create":
      // 没有工程时先聚焦路径输入框：用户多半还没填路径，直接弹"新建"会先报错
      if (!(el.path?.value || "").trim()) {
        el.path?.focus();
        setStatus("先填 .tfanno 路径（可点「选路径…」），再点「新建工程」", "warn");
        return;
      }
      createProject();
      return;
    case "addFiles":
      addImagesByDialog();
      return;
    case "focusClass":
      el.classNameInput?.focus();
      setStatus("输入缺陷名后回车或点「加」", "");
      return;
    case "toolRect":
      setTool("rectangle");
      setStatus("已切到矩形工具：在图上按住拖出一个框", "");
      return;
    case "save":
      saveProject({ silent: false });
      return;
    default:
      return;
  }
}

/**
 * 引导条的显示/隐藏。
 *
 * 收起后只留一个「显示流程」按钮：不用 `hidden` 整条藏掉，是因为藏掉之后
 * 用户找不到恢复入口（这个页面本来就没有设置面板）。
 */
function applyGuideVisibility() {
  if (!el.guide) return;
  const collapsed = localStorage.getItem(GUIDE_STORAGE) === "0";
  el.guide.classList.toggle("is-collapsed", collapsed);
  if (el.guideToggle) {
    el.guideToggle.textContent = collapsed ? "显示流程" : "隐藏流程";
    el.guideToggle.title = collapsed ? "展开五步流程说明" : "收起后画布会变高，可从同一位置恢复";
  }
}

/**
 * 「进阶：预标注草稿与审核」的展开状态。
 *
 * 判定在 annotateflow.shouldExpandDrafts：用户手动动过就完全听用户的，
 * 没动过才按"有没有草稿/审核记录"自动展开。
 */
function syncDraftPanel() {
  if (!el.advDraft) return;
  const open = shouldExpandDrafts({
    draftCount: countAllDrafts(state.drafts),
    reviewCount: state.review.size,
    touched: state.advDraftTouched,
    open: el.advDraft.open,
  });
  if (el.advDraft.open !== open) el.advDraft.open = open;
}

function renderAll() {
  applyTaskChrome();
  renderGuide();
  syncDraftPanel();
  renderClassList();
  renderBoxTable();
  renderDraftList();
  renderImageList();
  renderProps();
  renderReviewState();
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
  let text = state.opts.showScore && score >= 0 ? `${name} ${score.toFixed(2)}` : name;
  // 草稿标签带档位字：真机上绿/黄/红的观感会受图片底色影响，
  // 有"高/中/低"三个字兜底，色弱用户也能一眼分辨。
  if (obj._draft) text = `${text} · ${draftTierText(score)}`;
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
  // 草稿按置信度分档取色：绿=可放心过、黄=看一下、红=重点看。
  // 鼠标悬停/选中靠虚线 + 线宽区分，不靠颜色，避免和分档语义打架。
  const baseColor = draft ? draftTierColor(obj.score) : classById(obj.class_id).color;
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
      // 选中再描一圈青色：表格点中和画布点中是同一种高亮，线宽不够时在深色图上会看丢。
      if (selected) {
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
        ctx.stroke();
      }
      break;
    }
    case "polygon": {
      drawPolygon(ctx, points, true);
      if (alpha > 0) {
        ctx.fillStyle = rgb(baseColor, alpha);
        ctx.fill();
      }
      ctx.stroke();
      if (selected) {
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
        ctx.stroke();
      }
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
  // 矩形的手柄画在四个可见角上，不画在存储用的两个对角点上。
  // 存储点在拖动时会被归一化成左上/右下，手柄若跟着存储点走，
  // 鼠标按住的那个角会突然跳到另一头。
  if (obj.shape_type === "rectangle" || obj.shape_type === "rotated_rect") {
    for (const corner of shapeCorners(obj)) {
      const screen = imageToScreen(corner);
      ctx.beginPath();
      ctx.rect(screen.x - HANDLE_RADIUS, screen.y - HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2);
      ctx.fillStyle = "rgba(8,12,16,0.9)";
      ctx.fill();
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    const knob = rotateHandlePoint(obj);
    if (knob) {
      const screen = imageToScreen(knob);
      const top = imageToScreen(shapeCorners(obj)[0]);
      const top2 = imageToScreen(shapeCorners(obj)[1]);
      const mid = { x: (top.x + top2.x) / 2, y: (top.y + top2.y) / 2 };
      ctx.beginPath();
      ctx.moveTo(mid.x, mid.y);
      ctx.lineTo(screen.x, screen.y);
      ctx.strokeStyle = "rgba(255,209,102,0.9)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ffd166";
      ctx.fill();
      ctx.strokeStyle = "rgba(8,12,16,0.9)";
      ctx.stroke();
    }
  }
  for (const point of points) {
    if (obj.shape_type === "rectangle" || obj.shape_type === "rotated_rect") break;
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

let canvasFrame = 0;
let canvasSkipMini = false;

function renderCanvas(options = {}) {
  if (options.coalesce) {
    canvasSkipMini = canvasSkipMini || Boolean(options.skipMini);
    if (canvasFrame) return;
    canvasFrame = requestAnimationFrame(() => {
      canvasFrame = 0;
      const skipMini = canvasSkipMini;
      canvasSkipMini = false;
      paintCanvas({ skipMini });
    });
    return;
  }
  paintCanvas(options);
}

function paintCanvas(options = {}) {
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

  if (isClassify()) {
    const unlabeled = image.class_id === undefined || image.class_id === null;
    const caption = unlabeled
      ? "未标注"
      : (state.project.classes.find((item) => item.id === image.class_id)?.name || "未标注");
    ctx.font = "700 28px sans-serif";
    const textWidth = ctx.measureText(caption).width;
    ctx.fillStyle = "rgba(8,12,16,0.72)";
    ctx.fillRect(state.view.offset.x + 12, state.view.offset.y + 12, textWidth + 28, 44);
    ctx.fillStyle = unlabeled ? "#f5c16c" : "#d7fff2";
    ctx.fillText(caption, state.view.offset.x + 24, state.view.offset.y + 44);
  }

  if (state.opts.showShapes && !isClassify()) {
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

  if (promptActive()) drawPromptOverlay(ctx);

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
  if (!options.skipMini) renderMini();
}

/**
 * 画布上的提示叠加层：正点绿实心、负点红叉、框虚线。
 *
 * 为什么不复用 `drawShape`：提示点不是标注对象，没有类别、不该被选中/隐藏/
 * 参与命中测试。走同一条渲染路径会让它出现在"标注数量"和导出里，
 * 而它只是一个临时输入。
 */
function drawPromptOverlay(ctx) {
  const image = currentImage();
  if (!image) return;

  const box = state.promptBox || (state.promptDrag?.moved ? dragBoxPoints(state.promptDrag) : null);
  if (box) {
    const a = imageToScreen(box[0]);
    const b = imageToScreen(box[1]);
    ctx.save();
    ctx.strokeStyle = "rgba(34,211,238,0.95)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
    ctx.restore();
  }

  state.promptPoints.forEach((point, index) => {
    const screen = imageToScreen(point);
    const negative = Number(state.promptLabels[index]) === 0;
    ctx.save();
    ctx.lineWidth = 2;
    if (negative) {
      // 负点用叉：圆点靠颜色区分的话，红绿色盲用户看不出区别
      ctx.strokeStyle = "#f87171";
      const r = 6;
      ctx.beginPath();
      ctx.moveTo(screen.x - r, screen.y - r);
      ctx.lineTo(screen.x + r, screen.y + r);
      ctx.moveTo(screen.x + r, screen.y - r);
      ctx.lineTo(screen.x - r, screen.y + r);
      ctx.stroke();
    } else {
      ctx.fillStyle = "#34d399";
      ctx.strokeStyle = "rgba(10,14,19,0.9)";
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    // 序号：点多的时候能看出先后，方便删掉点错的那一个
    ctx.fillStyle = "rgba(230,240,250,0.9)";
    ctx.font = "11px system-ui";
    ctx.fillText(String(index + 1), screen.x + 8, screen.y - 8);
    ctx.restore();
  });
}

/** 拖拽中的框：按起点与当前光标算出归一化（左上/右下）的两点。 */
function dragBoxPoints(drag) {
  const end = state.hover?.image;
  if (!drag || !end) return null;
  return [
    [Math.min(drag.start[0], end.x), Math.min(drag.start[1], end.y)],
    [Math.max(drag.start[0], end.x), Math.max(drag.start[1], end.y)],
  ];
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
  const nextW = Math.round(boxW * ratio);
  const nextH = Math.round(boxH * ratio);
  // 每帧改 width 会清空并重建画布，拖框时这是卡顿的主要来源。尺寸没变就别动它。
  if (mini.width !== nextW || mini.height !== nextH) {
    mini.width = nextW;
    mini.height = nextH;
  }
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

/** 旋转框顶部外侧的圆点（图像坐标）。普通矩形没有这个点，所以不能转。 */
function rotateHandlePoint(obj) {
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
  const gap = 22 / state.view.scale;
  return [mid[0] + (dx / len) * gap, mid[1] + (dy / len) * gap];
}

function hitHandle(obj, point) {
  const tolerance = (HANDLE_RADIUS + 3) / state.view.scale;
  if (obj.shape_type === "rectangle" || obj.shape_type === "rotated_rect") {
    const knob = rotateHandlePoint(obj);
    if (knob && Math.hypot(point.x - knob[0], point.y - knob[1]) <= tolerance + 2 / state.view.scale) {
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

  // 提示模式接管左键：这一模式下"点击"的含义是放提示点，不是画形状。
  // 中键平移在它之前就返回了，所以缩放/平移手感不受影响。
  if (promptActive() && event.button === 0) {
    if (event.shiftKey) {
      state.promptPoints.push([point.x, point.y]);
      state.promptLabels.push(0);
      setStatus(`已加一个负点（共 ${state.promptPoints.length} 个提示点），Enter 执行`, "warn");
      renderCanvas();
      return;
    }
    state.promptDrag = { startScreen: screen, start: [point.x, point.y], moved: false };
    capturePointer(event);
    return;
  }

  if (state.tool === "select") {
    const selected = selectedObject();
    if (selected) {
      // 1) 顶点手柄优先：拖动只改那一个点
      const handle = hitHandle(selected, point);
      if (handle) {
        if (handle.kind === "corner") {
          const corners = shapeCorners(selected);
          const opposite = corners[(handle.index + 2) % 4];
          state.drag = {
            kind: "corner",
            index: handle.index,
            opposite: [opposite[0], opposite[1]],
            pushed: false,
          };
          capturePointer(event);
          return;
        }
        if (handle.kind === "rotate") {
          state.drag = { kind: "rotate", pushed: false };
          capturePointer(event);
          return;
        }
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

  // 提示拖框：先更新"是不是真的拖了"，再交给渲染
  if (state.promptDrag) {
    const dx = screen.x - state.promptDrag.startScreen.x;
    const dy = screen.y - state.promptDrag.startScreen.y;
    if (Math.hypot(dx, dy) > 4) state.promptDrag.moved = true;
    state.hover = { image: point, snap: null };
    renderCanvas();
    return;
  }

  if (state.drag) {
    // 拖动途中也要更新 hover：绘制预览、边拖动都靠它拿当前点
    state.hover = { image: point, snap: null };
    handleDrag(point);
    return;
  }

  const snapped = event.altKey ? { point, snap: null } : snapPoint(point, { allowClose: state.tool === "polygon" });
  state.hover = { image: point, snap: snapped.snap };

  if (state.opts.crosshair || state.pending.length) renderCanvas({ coalesce: true, skipMini: true });

  // 悬停在顶点/边上时换光标，让"这里能拖"这件事可发现
  if (state.tool === "select") {
    const selected = selectedObject();
    let cursor = "default";
    const handle = selected ? hitHandle(selected, point) : null;
    if (handle?.kind === "rotate") cursor = "grab";
    else if (handle || (selected && hitEdge(selected, point))) cursor = "nwse-resize";
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
    renderCanvas({ coalesce: true, skipMini: true });
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
    renderCanvas({ coalesce: true, skipMini: true });
    return;
  }

  if (drag.kind === "rotate") {
    const target = selectedObject();
    if (!target) return;
    const [a, b] = target.points;
    const cx = (a[0] + b[0]) / 2;
    const cy = (a[1] + b[1]) / 2;
    // 圆点在框的上边外侧。光标在正上方时角度为 0，往右拖角度增大。
    const next = Math.atan2(point.y - cy, point.x - cx) + Math.PI / 2;
    if (Math.abs(next - (Number(target.rotation) || 0)) < 1e-4) return;
    ensureDragHistory(drag);
    target.rotation = next;
    renderCanvas({ coalesce: true, skipMini: true });
    return;
  }

  if (drag.kind === "corner") {
    const target = selectedObject();
    if (!target || !drag.opposite) return;
    const opposite = drag.opposite;
    const dx = point.x - opposite[0];
    const dy = point.y - opposite[1];
    if (Math.hypot(dx, dy) < 2) return;
    ensureDragHistory(drag);
    const angle = target.shape_type === "rotated_rect" ? Number(target.rotation) || 0 : 0;
    if (angle !== 0) {
      const cx = (opposite[0] + point.x) / 2;
      const cy = (opposite[1] + point.y) / 2;
      const vx = point.x - cx;
      const vy = point.y - cy;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const localX = vx * cos + vy * sin;
      const localY = -vx * sin + vy * cos;
      const halfW = Math.abs(localX);
      const halfH = Math.abs(localY);
      if (halfW < 1 || halfH < 1) return;
      target.points = [
        [cx - halfW, cy - halfH],
        [cx + halfW, cy + halfH],
      ];
    } else {
      target.points = [
        [Math.min(opposite[0], point.x), Math.min(opposite[1], point.y)],
        [Math.max(opposite[0], point.x), Math.max(opposite[1], point.y)],
      ];
    }
    renderCanvas({ coalesce: true, skipMini: true });
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
    renderCanvas({ coalesce: true, skipMini: true });
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
    renderCanvas({ coalesce: true, skipMini: true });
    return;
  }

  if (drag.kind === "draw") renderCanvas({ coalesce: true, skipMini: true });
}

function onPointerUp() {
  const drag = state.drag;
  state.drag = null;

  // 提示模式的松手：拖够距离算一个框，否则算一个正点。
  const promptDrag = state.promptDrag;
  state.promptDrag = null;
  if (promptDrag) {
    const box = dragBoxPoints(promptDrag);
    const wide = box && box[1][0] - box[0][0] >= 3 && box[1][1] - box[0][1] >= 3;
    if (promptDrag.moved && wide) {
      state.promptBox = box;
      setStatus("已放置框提示，Enter 执行（Esc 清空）", "warn");
    } else {
      state.promptPoints.push(promptDrag.start);
      state.promptLabels.push(1);
      setStatus(`已加一个正点（共 ${state.promptPoints.length} 个提示点），Enter 执行`, "warn");
    }
    renderCanvas();
    return;
  }

  if (!drag) return;
  if (drag.kind === "move" || drag.kind === "vertex" || drag.kind === "edge" || drag.kind === "corner" || drag.kind === "rotate") {
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
    markDraftsDirty(image.id);
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
  markDraftsDirty(image.id);
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

/**
 * 审核快捷键作用的目标草稿。
 *
 * 有选中就处理选中的；没选中就取**置信度最高**的一条——这符合人的直觉：
 * 先过最可信的一批，把注意力留给真正需要看的低分框。
 * `score < 0`（通道没给置信度）排在最后，但不被排除。
 */
function draftTarget() {
  const drafts = currentDrafts();
  if (!drafts.length) return null;
  const selected = drafts.find((draft) => draft.id === state.selectedId);
  if (selected) return selected;
  let best = drafts[0];
  for (const draft of drafts) {
    if (Number(draft.score) > Number(best.score)) best = draft;
  }
  return best;
}

/** 接受后自动选中下一条，`A` 连按就是一路上报，不用回头点鼠标。 */
function acceptCurrentDraft() {
  const target = draftTarget();
  if (!target) {
    setStatus("当前图片没有草稿，按空格去下一张", "warn");
    return;
  }
  acceptDraft(target.id);
  const rest = currentDrafts();
  state.selectedId = rest.length ? draftTarget()?.id || "" : "";
  renderAll();
  renderCanvas();
}

function rejectCurrentDraft() {
  const target = draftTarget();
  if (!target) {
    setStatus("当前图片没有草稿，按空格去下一张", "warn");
    return;
  }
  rejectDraft(target.id);
  state.selectedId = "";
  renderAll();
  renderCanvas();
}

/** 空格：下一张。最后一张给提示但不循环——循环会让人以为"还没看完"。 */
async function nextImage() {
  if (!state.project || state.currentIndex < 0) return;
  if (state.currentIndex + 1 >= state.project.images.length) {
    setStatus("已是最后一张", "warn");
    return;
  }
  await switchImage(state.currentIndex + 1);
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
  markDraftsDirty(image.id);
  afterEdit();
  renderCanvas();
  setStatus(`已接受 ${count} 个草稿`);
}

function rejectAllDrafts() {
  const image = currentImage();
  const drafts = currentDrafts();
  if (!image || !drafts.length) return;
  const count = drafts.length;
  drafts.length = 0;
  markDraftsDirty(image.id);
  renderAll();
  renderCanvas();
  setStatus(`已丢弃 ${count} 个草稿`);
}

/** 当前预标注参数（单张与批量共用，避免两处各读一遍控件）。 */
function autoLabelOptions() {
  return {
    task: state.project?.task || "detect",
    engine: (el.alEngine?.value || "").trim(),
    labels: (el.alLabels?.value || "").trim(),
    conf: Number(el.alConf?.value || 0.25),
    nms: Number(el.alNms?.value || 0.45),
    service: (el.alService?.value || "").trim(),
  };
}

function requireEngine() {
  const options = autoLabelOptions();
  if (!options.engine) {
    el.autoLabelBox?.setAttribute("open", "");
    setStatus("请先选一个引擎：展开「预标注参数」，从训练产物下拉里选，或手工填 .engine 路径", "bad");
    return null;
  }
  return options;
}

async function runAutoLabel() {
  const image = currentImage();
  if (!image) {
    setStatus("先选一张图片", "bad");
    return;
  }
  const options = requireEngine();
  if (!options) return;
  setStatus("预标注中…", "warn");
  try {
    const result = await autoLabel({
      image: imageApiPath(image),
      task: options.task,
      engine: options.engine,
      labels: options.labels,
      conf: options.conf,
      nms: options.nms,
      service: options.service,
    });
    const drafts = result.drafts || [];
    state.drafts.set(image.id, drafts);
    markDraftsDirty(image.id);
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
// 训练产物下拉（闭环：训练完直接选，不用去文件框里翻）
// ---------------------------------------------------------------------------
async function refreshEngineOptions({ manual = false } = {}) {
  if (!el.alEnginePick) return;
  if (manual) setStatus("正在扫描训练产物…", "warn");
  try {
    const data = await listModels();
    const entries = (data.entries || []).filter((item) => item.kind === "yolo");
    state.engineLoaded = true;
    el.alEnginePick.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = entries.length ? "— 从训练产物选择 —" : "— 没有扫到 YOLO 引擎 —";
    el.alEnginePick.appendChild(placeholder);
    for (const entry of entries) {
      const option = document.createElement("option");
      option.value = entry.path;
      const size = (Number(entry.size || 0) / (1024 * 1024)).toFixed(1);
      // "未转置"要标出来：拿 Ultralytics 原始布局的 onnx 编出来的引擎，
      // 解码会越界，跑出来是一堆离谱的框。
      const warn = entry.transposed ? "" : " ⚠ 未转置";
      option.textContent = `${entry.label} · ${size}MB · ${entry.task}${warn}`;
      option.dataset.labels = entry.labels || "";
      el.alEnginePick.appendChild(option);
    }
    // 当前输入的引擎如果正好在列表里，让下拉也停在它上面（看起来才一致）
    const current = (el.alEngine?.value || "").trim();
    if (current && entries.some((entry) => entry.path === current)) {
      el.alEnginePick.value = current;
    }
    if (manual) {
      setStatus(
        entries.length ? `已载入 ${entries.length} 个 YOLO 引擎` : "没有扫到 YOLO 引擎，可手工填路径",
        entries.length ? "ok" : "warn",
      );
    }
    const roots = (data.roots || []).filter((root) => root.kind === "yolo" && !root.ok);
    if (!entries.length && roots.length) {
      setStatus(`训练产物目录不可用：${roots[0].reason || roots[0].path}`, "warn");
    }
  } catch (error) {
    setStatus(`扫描训练产物失败：${error.message}`, "bad");
  }
}

function applyEngineOption() {
  const path = (el.alEnginePick?.value || "").trim();
  if (!path) return;
  if (el.alEngine) el.alEngine.value = path;
  // 标签文件跟着引擎走：同一个 run 目录下的 labels_*.json 才是这套类别的权威；
  // 但用户手填过的标签文件不覆盖，避免把外部标注服务的配置冲掉。
  const labels = el.alEnginePick.selectedOptions?.[0]?.dataset?.labels || "";
  if (labels && el.alLabels && !(el.alLabels.value || "").trim()) el.alLabels.value = labels;
  setStatus(`已选引擎：${path}`, "ok");
}

// ---------------------------------------------------------------------------
// 批量预标注：整个工程跑一遍，结果进草稿池
// ---------------------------------------------------------------------------
function renderBatchProgress() {
  const job = state.batchJob;
  if (!el.batchProgress) return;
  const bar = el.batchProgress.firstElementChild;
  if (!job) {
    el.batchProgress.classList.add("hidden");
    if (bar) bar.style.width = "0%";
    if (el.batchHint) el.batchHint.textContent = "";
    if (el.batchStop) el.batchStop.disabled = true;
    return;
  }
  el.batchProgress.classList.remove("hidden");
  const ratio = job.total ? Math.round((job.done / job.total) * 100) : 0;
  if (bar) bar.style.width = `${Math.min(100, ratio)}%`;
  el.batchProgress.classList.toggle("done", job.status !== "running");
  if (el.batchStop) el.batchStop.disabled = job.status !== "running";
  if (el.batchHint) {
    const parts = [];
    parts.push(`${job.status === "running" ? (job.stopping ? "停止中" : "处理中") : job.status === "done" ? "已完成" : job.status === "stopped" ? "已停止" : "失败"} ${job.done}/${job.total}`);
    if (job.current) parts.push(`当前：${job.current}`);
    parts.push(`草稿 ${job.drafts}`);
    if (job.failed) parts.push(`失败 ${job.failed}`);
    if (job.empty) parts.push(`无结果 ${job.empty}`);
    if (job.error) parts.push(job.error);
    el.batchHint.textContent = parts.join(" · ");
    el.batchHint.className = `anno-hint ${job.failed || job.status === "failed" ? "warn" : ""}`;
  }
}

async function runBatchAutoLabel() {
  if (!state.project || !state.path) {
    setStatus("先打开一个工程", "bad");
    return;
  }
  if (!state.project.images.length) {
    setStatus("工程里还没有图片", "bad");
    return;
  }
  if (state.batchJob?.status === "running") {
    setStatus("上一批还在跑，先停止或等它结束", "warn");
    return;
  }

  // 通道决定"批量到底跑什么"：引擎通道需要引擎路径，其余两个需要模型/提示。
  // 提示式（点/框）**不参与批量**：它每次都要人在画布上点，天然不是批量的活；
  // 这里按"当前通道"决定，但把点框通道落到引擎通道会让人莫名其妙跑错东西，
  // 所以显式拦一下。
  const channel = state.alChannel;
  if (channel === "prompt") {
    setStatus("点/框提示需要逐张在画布上点，不能批量。批量请选「自有引擎」或「文本提示」", "warn");
    return;
  }
  let options;
  if (channel === "text") {
    const prompt = textPromptValue();
    if (!prompt) {
      setStatus("文本通道需要先填写提示词", "bad");
      el.autoLabelBox?.setAttribute("open", "");
      return;
    }
    options = { task: state.project?.task || "detect", engine: "", labels: "" };
  } else {
    options = requireEngine();
    if (!options) return;
  }

  // 先把工程与草稿落盘：后端会读旁挂草稿判断"哪些图已有草稿"，
  // 内存里刚改的东西不先写下去，scope=pending 就会重复跑。
  await flushSave();
  await flushDraftChanges({ silent: true });
  await flushReviewChanges({ silent: true });

  const images = state.project.images.map((image) => ({
    id: image.id,
    path: imageApiPath(image),
    annotated: (image.annotations || []).length > 0,
  }));
  setStatus(`正在启动批量预标注（${images.length} 张，通道：${channel === "text" ? "文本提示" : "引擎"}）…`, "warn");
  try {
    const result = await startBatchAutoLabel({
      path: state.path,
      images,
      mode: channel === "text" ? "text" : "engine",
      task: options.task,
      engine: options.engine,
      labels: options.labels,
      prompt: channel === "text" ? textPromptValue() : "",
      box_thr: Number(el.alBoxThr?.value || 0.3),
      text_thr: Number(el.alTextThr?.value || 0.25),
      refine: Boolean(el.alRefine?.checked),
      conf: options.conf,
      nms: options.nms,
      service: options.service,
      scope: el.batchSkipDone?.checked ? "pending" : "all",
    });
    state.batchJob = result.job;
    const skipped = Number(result.skipped || 0);
    setStatus(
      `批量预标注已启动：${result.total} 张${skipped ? `，跳过 ${skipped} 张` : ""}`,
      "ok",
    );
    renderBatchProgress();
    scheduleBatchPoll();
  } catch (error) {
    state.batchJob = null;
    renderBatchProgress();
    setStatus(`批量预标注启动失败：${error.message}`, "bad");
  }
}

function scheduleBatchPoll() {
  if (state.batchTimer) clearTimeout(state.batchTimer);
  state.batchTimer = window.setTimeout(pollBatchJob, 700);
}

async function pollBatchJob() {
  const job = state.batchJob;
  if (!job?.id) return;
  state.batchTimer = 0;
  try {
    const result = await getBatchAutoLabel(job.id);
    state.batchJob = result.job;
  } catch (error) {
    state.batchJob = { ...job, status: "failed", error: error.message };
    renderBatchProgress();
    setStatus(`批量预标注状态查询失败：${error.message}`, "bad");
    return;
  }
  renderBatchProgress();
  if (state.batchJob.status === "running") {
    scheduleBatchPoll();
    return;
  }
  // 结束后读回草稿：批量结果是**后端直接写盘**的，前端内存里的那份不含新图片。
  await loadDraftsFromDisk();
  const job2 = state.batchJob;
  const bits = [`批量预标注结束：${job2.done}/${job2.total} 张，共 ${job2.drafts} 个草稿`];
  if (job2.failed) bits.push(`失败 ${job2.failed} 张`);
  setStatus(bits.join(" · "), job2.failed ? "warn" : "ok");
  renderAll();
  renderCanvas();
}

async function stopBatchRun() {
  const job = state.batchJob;
  if (!job?.id || job.status !== "running") return;
  setStatus("正在停止批量预标注（当前这张跑完就停）…", "warn");
  try {
    const result = await stopBatchAutoLabel(job.id);
    state.batchJob = result.job;
    renderBatchProgress();
  } catch (error) {
    setStatus(`停止失败：${error.message}`, "bad");
  }
}

/**
 * 刷新页面后接回"还在跑的那一批"。
 *
 * 任务在服务端进程里，刷新浏览器不会让它停；不接回来的话进度条凭空消失，
 * 用户会以为任务挂了，再点一次就是两批并发跑同一个引擎。
 */
async function resumeBatchJob() {
  if (state.batchJob?.status === "running") return;
  try {
    const data = await listBatchAutoLabel();
    const running = (data.jobs || []).find(
      (job) => job.status === "running" && job.project === state.path,
    );
    if (!running) return;
    state.batchJob = running;
    renderBatchProgress();
    scheduleBatchPoll();
    setStatus("检测到后台批量预标注仍在运行，已接上进度", "warn");
  } catch {
    /* 拿不到任务列表不影响标注本身 */
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
    if (state.review.has(image.id)) li.classList.add("reviewed");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = image.id || image.path.split(/[\\/]/).pop();
    name.title = image.path;
    // 审核标记：已审核绿勾 / 未审核灰圈。用文字符号而不是图标字体，
    // 少一个外部依赖，也不会有"图标没加载出来就分不清状态"的问题。
    const flag = document.createElement("span");
    flag.className = `review-flag ${state.review.has(image.id) ? "checked" : "pending"}`;
    flag.textContent = state.review.has(image.id) ? "✓" : "○";
    flag.title = state.review.has(image.id) ? "已审核 (K 取消用 Shift+K)" : "未审核 (K 标为已审核)";
    if (isClassify() && (image.class_id === undefined || image.class_id === null)) {
      li.classList.add("is-unlabeled");
    }
    const meta = document.createElement("span");
    meta.className = "meta";
    const draftCount = (state.drafts.get(image.id) || []).length;
    if (isClassify()) {
      const named = image.class_id === undefined || image.class_id === null
        ? "未标注"
        : (state.project.classes.find((item) => item.id === image.class_id)?.name || "未标注");
      meta.textContent = named;
    } else {
      meta.textContent = `${image.annotations.length}${draftCount ? `+${draftCount}草` : ""}`;
    }
    const split = document.createElement("span");
    split.className = `split ${image.split}`;
    split.textContent = image.split;
    // 行尾移除：只从工程里拿掉，不碰磁盘文件。默认低调，悬停才变红，
    // 避免用户把"点整行切图"和"删图"弄混。
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "mini danger anno-remove";
    remove.textContent = "×";
    remove.title = "从工程移除（不删磁盘上的图片，Ctrl+Z 可撤销）";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      removeImage(index);
    });
    li.append(flag, name, meta, split, remove);
    li.addEventListener("click", () => switchImage(index));
    el.imageList.appendChild(li);
  });

  // 空列表要给出"下一步该点哪里"，否则新工程打开后只是一片空白
  if (!images.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = state.project
      ? "还没有图片：填上面的图片目录后点「批量加入」，或点「选图片…」逐张挑。"
      : "先按第 1 步新建或打开一个 .tfanno 工程，再回来加图片。";
    el.imageList.appendChild(empty);
  }
}

/**
 * 从工程里移除一张图片（v1.35.0）。
 *
 * 只改 `.tfanno` 里的图片数组，**不删磁盘文件**——用户点「×」的意思几乎都是
 * "这张不该在这个工程里"，而不是"把这张照片毁掉"。
 *
 * 三件事必须一起做，否则会留下脏数据：
 * 1) 整工程进撤销栈：撤销要能把这行连标注一起恢复；
 * 2) 同步旁挂草稿与审核状态：图片 id 没了，草稿还留着就是"孤儿"，
 *    下次同 id 的新图会被旧草稿污染；
 * 3) 修正当前下标：删掉的正好是当前图时要切到相邻图，删空则回空状态。
 */
async function removeImage(index) {
  const project = state.project;
  const image = project?.images?.[index];
  if (!image) return;
  const name = image.id || image.path;
  const draftCount = (state.drafts.get(image.id) || []).length;
  const extra = draftCount ? `\n\n这张图还有 ${draftCount} 条预标注草稿，也会一起清掉。` : "";
  const confirmed = window.confirm(
    `从工程中移除「${name}」？\n\n` +
      `只移出工程，磁盘上的图片文件不会被删除。\n` +
      `该图片的标注会一起移除，可以用 Ctrl+Z 撤销。${extra}`,
  );
  if (!confirmed) {
    setStatus("已取消移除", "");
    return;
  }
  // 先把手上的改动落盘：移除操作会整份重写 images 数组，不能让未保存内容陪葬
  await flushSave();
  pushHistory(true);
  const wasCurrent = index === state.currentIndex;
  const { images, removed, focusIndex } = removeImageAt(project.images, index);
  if (!removed) return;
  project.images = images;
  // 下标修正：删的位置在当前之前，当前图整体前移一位；删的就是当前图，按 focusIndex 落位
  if (state.currentIndex > index) state.currentIndex -= 1;
  else if (wasCurrent) state.currentIndex = focusIndex;
  state.drafts.delete(removed.id);
  state.draftDirty.delete(removed.id);
  state.review.delete(removed.id);
  if (wasCurrent) {
    state.image = null;
    state.selectedId = "";
    state.pending = [];
    state.hidden.clear();
    state.viewValid = false;
  }
  afterEdit();
  if (wasCurrent) loadCurrentImage();
  else renderCanvas();
  // 旁挂文件增量清理：空数组 = 删掉这张图的草稿，unchecked = 删掉审核记录。
  // 失败只提示不算失败的操作——磁盘上的 `.tfanno` 已经改对了。
  if (state.path) {
    saveAnnotationDrafts({
      path: state.path,
      images: { [removed.id]: [] },
      review: { [removed.id]: { state: "unchecked" } },
    }).catch((error) => setStatus(`草稿清理失败：${error.message}`, "warn"));
  }
  setStatus(`已从工程移除 ${name}（磁盘文件未删除，Ctrl+Z 可撤销）`);
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
    if (isClassify()) {
      if (image.class_id !== undefined && image.class_id !== null) {
        counts.set(image.class_id, (counts.get(image.class_id) || 0) + 1);
      }
    } else {
      for (const obj of image.annotations) counts.set(obj.class_id, (counts.get(obj.class_id) || 0) + 1);
    }
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
    if (isClassify() && currentImage()?.class_id === cls.id) li.classList.add("is-current");
    li.append(swatch, label, count, del);
    li.addEventListener("click", () => {
      if (isClassify()) assignImageClass(cls.id);
      else applyClassToSelection(cls.id);
    });
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
    li.classList.add(draftTierClass(draft.score));
    if (draft.id === state.selectedId) li.classList.add("active");
    const score = Number(draft.score);
    const badge = document.createElement("span");
    badge.className = `tier-badge ${draftTierClass(draft.score)}`;
    badge.textContent = draftTierText(draft.score);
    badge.title = score >= 0
      ? `置信度 ${score.toFixed(3)}（高 ≥ ${DRAFT_TIER_HIGH}，中 ≥ ${DRAFT_TIER_MID}）`
      : "该通道没有返回置信度";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `#${index + 1} ${draft.label || classById(draft.class_id).name}${
      score >= 0 ? ` ${score.toFixed(2)}` : ""
    }`;
    name.title = draft._note || "";
    const ok = document.createElement("button");
    ok.type = "button";
    ok.className = "mini";
    ok.textContent = "✓";
    ok.title = "接受 (A)";
    ok.addEventListener("click", (event) => {
      event.stopPropagation();
      acceptDraft(draft.id);
    });
    const no = document.createElement("button");
    no.type = "button";
    no.className = "mini danger";
    no.textContent = "×";
    no.title = "丢弃 (D)";
    no.addEventListener("click", (event) => {
      event.stopPropagation();
      rejectDraft(draft.id);
    });
    li.append(badge, name, ok, no);
    li.addEventListener("click", () => {
      state.selectedId = draft.id;
      renderAll();
      renderCanvas();
    });
    el.draftList.appendChild(li);
  });
}

function annotationBounds(obj) {
  const corners = shapeCorners(obj);
  if (!corners.length) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of corners) {
    minX = Math.min(minX, point[0]);
    minY = Math.min(minY, point[1]);
    maxX = Math.max(maxX, point[0]);
    maxY = Math.max(maxY, point[1]);
  }
  return { x: minX, y: minY, w: Math.max(0, maxX - minX), h: Math.max(0, maxY - minY) };
}

/** 点表格时，框如果在画面外，就把视野挪到它身上，否则高亮了也看不见。 */
function revealObject(obj) {
  const stage = el.stage;
  const bounds = annotationBounds(obj);
  if (!stage || !bounds) return;
  const cx = bounds.x + bounds.w / 2;
  const cy = bounds.y + bounds.h / 2;
  const screen = imageToScreen([cx, cy]);
  const margin = 48;
  const inside =
    screen.x >= margin &&
    screen.y >= margin &&
    screen.x <= stage.clientWidth - margin &&
    screen.y <= stage.clientHeight - margin;
  if (inside) return;
  state.view.offset = {
    x: stage.clientWidth / 2 - cx * state.view.scale,
    y: stage.clientHeight / 2 - cy * state.view.scale,
  };
}

function renderBoxTable() {
  const body = el.boxBody;
  if (!body) return;
  body.replaceChildren();
  const image = currentImage();
  const rows = (image?.annotations || []).filter((obj) =>
    ["rectangle", "rotated_rect", "polygon"].includes(obj.shape_type),
  );
  if (el.boxCount) el.boxCount.textContent = String(rows.length);
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "empty";
    td.textContent = image ? "这张图还没有框" : "先打开一张图片";
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }
  rows.forEach((obj, index) => {
    const bounds = annotationBounds(obj);
    const tr = document.createElement("tr");
    if (obj.id === state.selectedId) tr.className = "active";
    const shapeName = SHAPE_LABELS[obj.shape_type] || obj.shape_type;
    const angle = obj.shape_type === "rotated_rect" ? (Number(obj.rotation) || 0) * (180 / Math.PI) : 0;
    const place = bounds
      ? `${Math.round(bounds.x)},${Math.round(bounds.y)}  ${Math.round(bounds.w)}×${Math.round(bounds.h)}` +
        (obj.shape_type === "rotated_rect" ? `  ${angle.toFixed(0)}°` : "")
      : "—";
    for (const text of [String(index + 1), obj.label || classById(obj.class_id).name, shapeName, place]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => {
      state.selectedId = obj.id;
      state.boxScrollId = obj.id;
      revealObject(obj);
      renderAll();
      renderCanvas();
      setStatus(`选中 ${obj.label || classById(obj.class_id).name}`);
    });
    body.appendChild(tr);
    if (obj.id === state.selectedId && state.boxScrollId !== obj.id) {
      state.boxScrollId = obj.id;
      tr.scrollIntoView({ block: "nearest" });
    }
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
function assignImageClass(classId) {
  const image = currentImage();
  if (!image || !state.project) {
    setStatus("先选一张图片", "bad");
    return;
  }
  pushHistory();
  if (classId === null) delete image.class_id;
  else image.class_id = classId;
  image.annotations = [];
  afterEdit();
  const next = state.project.images.findIndex(
    (item, index) => index > state.currentIndex && (item.class_id === undefined || item.class_id === null),
  );
  if (classId !== null && next >= 0) {
    switchImage(next);
    return;
  }
  renderAll();
  renderCanvas();
  setStatus(classId === null ? "已清空这张图的类别" : "后面没有未标注的图");
}

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
  const used = (state.project.images || []).reduce((sum, image) => {
    if (isClassify()) return sum + (image.class_id === classId ? 1 : 0);
    return sum + image.annotations.filter((obj) => obj.class_id === classId).length;
  }, 0);
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
  if (!state.project) {
    // 静默返回会让人以为按钮坏了：明确指向第 1 步，并把焦点送回路径输入框
    setStatus("先按第 1 步新建或打开一个 .tfanno 工程", "warn");
    el.path?.focus();
    return;
  }
  if (!root) {
    setStatus("先填图片目录（或点左侧「…」选文件夹）", "warn");
    el.addRoot?.focus();
    return;
  }
  setStatus("扫描图片…", "warn");
  try {
    const data = await scanAnnotationImages(root, true);
    if (!data.images.length) {
      setStatus("该目录下没有图片", "warn");
      return;
    }
    insertImages(data.images);
  } catch (error) {
    setStatus(`扫描失败：${error.message}`, "bad");
  }
}

/**
 * 用系统对话框一次选多张图片直接加入工程（v1.20）。
 *
 * 和「批量加入」的区别：那个是扫整个目录（还会递归子目录），这个是精确挑几张。
 * 路径来自后端原生对话框，本身就是绝对路径，不需要再探测白名单。
 */
async function addImagesByDialog() {
  if (!state.project) {
    setStatus("先按第 1 步新建或打开一个 .tfanno 工程，再加图片", "warn");
    el.path?.focus();
    return;
  }
  setStatus("请到系统窗口里选图片…", "warn");
  let picked;
  try {
    picked = await nativeDialog({
      kind: "files",
      field: "annoAddFiles",
      title: "选择要加入工程的图片",
      initial: (el.addRoot?.value || "").trim(),
      patterns: [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
      label: "图片",
    });
  } catch (error) {
    const reason = error.payload?.detail || error.message;
    setStatus(`系统对话框打不开（${reason}），请改用「批量加入」扫目录`, "bad");
    return;
  }
  const paths = picked.paths || [];
  if (!paths.length) {
    setStatus("没有选择图片", "warn");
    return;
  }

  // 尺寸要逐张问后端：画布按 width/height 做坐标换算，缺了会画到画布外面
  setStatus(`读取 ${paths.length} 张图片的尺寸…`, "warn");
  const items = [];
  for (const path of paths) {
    let width = 0;
    let height = 0;
    try {
      const meta = await annotationImageMeta(path);
      width = meta.width || 0;
      height = meta.height || 0;
    } catch {
      /* 尺寸拿不到就按 0 存，画布会等到图片加载完再校正 */
    }
    items.push({ path, width, height });
  }
  insertImages(items);
}

/**
 * 把一批图片塞进工程（两个入口共用）。
 *
 * 去重按「后端能打开的绝对路径」比较：导入进来的图存的是相对路径，
 * 拿原始 path 去比会把同一张图再加一遍。
 */
function insertImages(items) {
  pushHistory(true);
  const known = new Set(state.project.images.map((image) => imageApiPath(image).toLowerCase()));
  const usedIds = new Set(state.project.images.map((image) => image.id));
  let added = 0;
  for (const item of items) {
    if (known.has((item.path || "").toLowerCase())) continue;
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
    known.add((item.path || "").toLowerCase());
    state.project.images.push({
      id,
      path: item.path,
      // 扫描接口与原生对话框给的都是绝对路径，直接当 resolved 用，
      // 前端只认这一个字段取图
      resolved: item.path,
      width: item.width || 0,
      height: item.height || 0,
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
}

// ---------------------------------------------------------------------------
// 导入已有 YOLO 标签（v1.19）
//
// 为什么「预览」是必经一步：导入会一次性改掉几百张图的标注，
// 先给出「几张图 / 几个框 / 哪些数据被跳过」，用户核对完再落盘。
// 任何参数改动都会作废上一次预览，防止拿旧结果去导入新的数据源。
// ---------------------------------------------------------------------------
function importSplits() {
  const box = document.getElementById("annoImportSplits");
  if (!box) return [];
  return [...box.querySelectorAll('input[type="checkbox"]')]
    .filter((input) => input.checked)
    .map((input) => input.value);
}

function setImportEnabled(enabled) {
  if (el.importApply) el.importApply.disabled = !enabled;
  if (el.importApplySave) el.importApplySave.disabled = !enabled;
}

function setImportReport(text, cls = "") {
  if (!el.importReport) return;
  el.importReport.textContent = text || "";
  el.importReport.className = cls ? `anno-hint ${cls}` : "anno-hint";
}

function invalidateImport() {
  state.importReport = null;
  setImportEnabled(false);
  setImportReport("参数改了，请重新「预览」。", "warn");
}

function importPayload(dryRun) {
  const mode = el.importMode?.value || "new";
  return {
    source: (el.importSource?.value || "").trim(),
    project_path: (el.path?.value || "").trim(),
    // 合并模式必须带上内存里的工程：磁盘上的可能是旧版本，会把刚画的框覆盖掉
    project: mode === "merge" ? state.project : null,
    splits: importSplits(),
    mode,
    duplicate: el.importDuplicate?.value || "skip",
    task: state.project?.task || document.getElementById("annoTask")?.value || "detect",
    dry_run: Boolean(dryRun),
  };
}

function formatImportReport(report, issues) {
  const lines = [];
  const splitText = Object.entries(report.per_split || {})
    .map(([name, item]) => `${name} ${item.images} 图 / ${item.objects} 框`)
    .join("　");
  lines.push(
    `任务 ${report.task} ｜ ${(report.classes || []).length} 类 ｜ ` +
      `${report.images} 图 / ${report.objects} ${report.task === "classify" ? "张" : "框"}`,
  );
  if (splitText) lines.push(splitText);

  const bits = [];
  if (report.added_images) bits.push(`新增 ${report.added_images} 图 / ${report.added_objects} 框`);
  if (report.skipped_existing) bits.push(`跳过已存在 ${report.skipped_existing} 图`);
  if (report.replaced_images) bits.push(`覆盖 ${report.replaced_images} 图`);
  if (report.empty_images) bits.push(`空标注 ${report.empty_images} 图`);
  if (report.unknown_class_objects) bits.push(`未知类别 ${report.unknown_class_objects} 框`);
  if (report.orphan_labels) bits.push(`孤儿标签 ${report.orphan_labels}`);
  if (report.malformed_lines) bits.push(`畸形行 ${report.malformed_lines}`);
  if (report.truncated) bits.push("已按上限截断");
  if (bits.length) lines.push(bits.join(" ｜ "));

  lines.push(`图片路径：${report.path_mode === "relative" ? "相对工程目录（可搬移）" : "绝对路径"}`);
  (report.warnings || []).forEach((text) => lines.push(`⚠ ${text}`));
  (issues || []).forEach((text) => lines.push(`✗ ${text}`));
  return lines.join("\n");
}

async function previewImport() {
  const payload = importPayload(true);
  if (!payload.source) {
    setImportReport("请先填 data.yaml 或数据集根目录", "bad");
    return;
  }
  if (payload.mode === "merge" && !state.project) {
    setImportReport("先打开一个工程，或把导入方式改成「新建工程」", "bad");
    return;
  }
  setImportEnabled(false);
  setImportReport("解析中…（几百张图通常一两秒）", "warn");
  try {
    const data = await importAnnotationProject(payload);
    state.importReport = data.report;
    const bad = (data.issues || []).length > 0;
    setImportReport(formatImportReport(data.report, data.issues), bad ? "warn" : "ok");
    setImportEnabled(true);
  } catch (error) {
    state.importReport = null;
    setImportReport(`预览失败：${error.message}`, "bad");
  }
}

async function applyImport({ save }) {
  if (!state.importReport) {
    setImportReport("请先「预览」，确认图数与框数没问题再导入", "warn");
    return;
  }
  const mode = el.importMode?.value || "new";
  const payload = importPayload(false);

  if (mode === "new") {
    const raw = (el.path?.value || "").trim();
    if (!raw) {
      setImportReport("新建工程要先填上面的 .tfanno 路径", "bad");
      return;
    }
    const resolved = await resolveProjectPath(raw);
    el.path.value = resolved.path;
    payload.project_path = resolved.path;
    // 先把手上的修改落盘，再确认「要丢弃吗」——否则确认框说的和磁盘上的不一致
    if (state.project) await flushSave();
    if (state.project && state.dirty && !window.confirm("当前工程有未保存的修改，导入新工程会丢弃它。继续？")) {
      return;
    }
    let existed = false;
    try {
      existed = Boolean((await loadAnnotationProject(resolved.path)).exists);
    } catch {
      /* 读不到就交给保存阶段报错 */
    }
    if (
      existed &&
      !window.confirm(`磁盘上已存在工程：\n${resolved.path}\n\n导入会覆盖它，确定继续？`)
    ) {
      return;
    }
  } else if (!state.project) {
    setImportReport("合并模式需要先打开一个工程", "bad");
    return;
  }

  setImportEnabled(false);
  setImportReport("导入中…", "warn");
  try {
    const data = await importAnnotationProject(payload);
    const report = data.report;

    if (mode === "new") {
      // 换了一份文档：旧的撤销栈属于另一个工程，留着会「撤销出一条不存在的历史」
      state.project = data.project;
      state.path = payload.project_path;
      state.meta = { rev: -1, mtime_ns: "0", size: 0 };
      state.exists = false;
      state.undo.length = 0;
      state.redo.length = 0;
      state.selectedId = "";
      state.drafts.clear();
      state.currentIndex = state.project.images.length ? 0 : -1;
      el.path.value = payload.project_path;
      updateHistoryButtons();
      markDirty();
      if (state.currentIndex >= 0) await loadCurrentImage();
      renderAll();
      renderCanvas();
      if (save) await saveProject({ force: true, silent: true });
    } else {
      // 一次导入 = 一个可撤销步骤：按一下 Ctrl+Z 能整批回退
      pushHistory(true);
      state.project = data.project;
      markDirty();
      renderAll();
      if (state.currentIndex >= 0) await loadCurrentImage();
      renderCanvas();
      if (save) await saveProject({ silent: true });
    }

    const suffix = save
      ? `，已写入 ${(state.path.split(/[\\/]/).pop() || ".tfanno")}`
      : "，记得按 Ctrl+S 保存";
    setImportReport(`${formatImportReport(report, data.issues)}\n✓ 导入完成${suffix}`, "ok");
    setStatus(`导入 ${report.added_images} 图 / ${report.added_objects} 框`);
  } catch (error) {
    setImportReport(`导入失败：${error.message}`, "bad");
  } finally {
    // 工程已经变了：再点一次就会重复导入，必须重新预览
    state.importReport = null;
    setImportEnabled(false);
  }
}

// ---------------------------------------------------------------------------
// 另存为
// ---------------------------------------------------------------------------
function isAbsolutePath(value) {
  return /^[a-zA-Z]:[\\/]/.test(value) || value.startsWith("\\\\") || value.startsWith("/");
}

function dirOf(path) {
  const index = Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/"));
  return index > 0 ? path.slice(0, index) : path;
}

/**
 * 用系统「另存为」框问目标路径，并写回输入框。
 *
 * 失败或用户取消时**不动**输入框——取消的语义是"按原来的路径存"，
 * 把输入框清空会让人以为工程丢了。返回是否拿到了新路径。
 */
async function pickSaveTarget() {
  const current = (el.path?.value || "").trim() || state.path || "";
  try {
    const picked = await nativeDialog({
      kind: "save",
      field: "annoPath",
      title: "另存为 .tfanno 工程",
      initial: current,
      patterns: [".tfanno"],
      label: "标注工程",
      defaultextension: ".tfanno",
    });
    const path = (picked.paths || [])[0];
    if (!path) return false;
    // 用户可能没写后缀，Qt 与后端都认 .tfanno，这里补齐避免存出无后缀文件
    el.path.value = /\.tfanno$/i.test(path) ? path : `${path}.tfanno`;
    return true;
  } catch (error) {
    // 409（已有一个框）= 等用户先处理；其余（远程/无桌面）静默退回手填
    if (error.status === 409) setStatus(error.message, "warn");
    return false;
  }
}

async function saveAs() {
  if (!state.project) {
    setStatus("没有可保存的工程", "bad");
    return;
  }
  // 先弹系统「另存为」框：输入框里手打完整路径对非技术用户不友好。
  // 框打不开（远程访问 / 无桌面会话）就退回输入框里的值，行为与 v1.19 一致。
  await pickSaveTarget();
  const raw = (el.path?.value || "").trim();
  if (!raw) {
    setStatus("请先填另存为的 .tfanno 路径", "bad");
    return;
  }
  const resolved = await resolveProjectPath(raw);
  el.path.value = resolved.path;
  if (resolved.path === state.path) {
    setStatus("路径没变，用「保存」即可", "warn");
    return;
  }

  let existed = false;
  try {
    existed = Boolean((await loadAnnotationProject(resolved.path)).exists);
  } catch {
    /* 读不到就交给保存阶段报错 */
  }
  if (existed && !window.confirm(`目标已存在：\n${resolved.path}\n\n另存为会覆盖它，确定继续？`)) {
    el.path.value = state.path;
    return;
  }

  const previousPath = state.path;
  const previousMeta = { ...state.meta };
  const previousExists = state.exists;

  // 换了目录后相对路径会指向错的地方。这里把它们就地改成绝对路径，
  // 保证新文件打开时图还在——比起"静默裂图"，多几个绝对路径是更好的取舍。
  let rebased = 0;
  if (dirOf(previousPath).toLowerCase() !== dirOf(resolved.path).toLowerCase()) {
    pushHistory(true);
    for (const image of state.project.images) {
      if (isAbsolutePath(image.path || "")) continue;
      if (!image.resolved) continue;
      image.path = image.resolved;
      rebased += 1;
    }
    if (rebased) markDirty();
  }

  state.path = resolved.path;
  state.meta = { rev: -1, mtime_ns: "0", size: 0 };
  state.exists = false;
  await saveProject({ force: true, silent: false });

  if (!state.exists) {
    // 失败就把路径还原：否则界面显示已存到新文件，磁盘上其实还没有
    state.path = previousPath;
    state.meta = previousMeta;
    state.exists = previousExists;
    el.path.value = previousPath;
    return;
  }
  const note = rebased ? `（${rebased} 张图的相对路径已改成绝对路径）` : "";
  setStatus(`已另存为 ${resolved.path.split(/[\\/]/).pop()}${note}`, "ok");
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
    if (promptActive() && (state.promptPoints.length || state.promptBox || state.promptDrag)) {
      // 提示模式下 Esc 的第一优先级是"撤销我刚放的那些点"，
      // 而不是"切回选择工具"——用户按 Esc 的动机几乎都是点错了。
      clearPrompt();
      setStatus("已清空提示", "");
      return;
    }
    state.pending = [];
    state.selectedId = "";
    setTool("select");
    renderAll();
    renderCanvas();
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    if (isClassify()) assignImageClass(null);
    else deleteSelected();
    return;
  }
  if (isClassify() && /^[1-9]$/.test(event.key)) {
    const classes = state.project?.classes || [];
    const picked = classes[Number(event.key) - 1];
    if (picked) {
      event.preventDefault();
      assignImageClass(picked.id);
    }
    return;
  }
  if (event.key === "Enter") {
    // 提示模式下 Enter 的语义是"执行这次提示"，优先级高于"结束多边形"
    if (promptActive() && (state.promptPoints.length || state.promptBox)) {
      event.preventDefault();
      runPromptLabel();
      return;
    }
    // 多边形 / 关键点用 Enter 结束，比"精确点回起点"宽容
    if (state.tool === "polygon" && state.pending.length >= 3) finishShape(state.pending);
    else if (state.tool === "keypoints" && state.pending.length) finishShape(state.pending);
    return;
  }
  const lower = event.key.toLowerCase();
  if (lower === "k") {
    // K 标为已审核并跳下一张；Shift+K 取消。放在工具快捷键之前，
    // 因为这两个键每个工程都会反复按。
    event.preventDefault();
    if (event.shiftKey) unmarkReviewed();
    else markReviewed();
    return;
  }
  // 审核快捷键：A 接受 / D 丢弃 / 空格下一张。
  // 放在工具快捷键之前，且都 preventDefault——空格默认会滚页面，
  // 连按几下画布就"跑没了"，用户会以为图丢了。
  if (event.key === " " || event.code === "Space") {
    event.preventDefault();
    nextImage();
    return;
  }
  if (lower === "a") {
    event.preventDefault();
    acceptCurrentDraft();
    return;
  }
  if (lower === "d") {
    event.preventDefault();
    rejectCurrentDraft();
    return;
  }
  if (lower === "f") {
    fitView();
    renderCanvas();
    return;
  }
  if (isClassify()) return;
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
    // 五步引导条（v1.35.0）
    guide: document.getElementById("annoGuide"),
    steps: document.getElementById("annoSteps"),
    guideTip: document.getElementById("annoGuideTip"),
    guideAction: document.getElementById("annoGuideAction"),
    guideToggle: document.getElementById("annoGuideToggle"),
    // 进阶：预标注草稿与审核（折叠块）
    advDraft: document.getElementById("annoAdvDraft"),
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
    boxBody: document.getElementById("annoBoxBody"),
    boxCount: document.getElementById("annoBoxCount"),
    exportFormat: document.getElementById("annoExportFormat"),
    exportOut: document.getElementById("annoExportOut"),
    autoLabelBox: document.getElementById("annoAutoLabelBox"),
    alEngine: document.getElementById("annoAlEngine"),
    alEnginePick: document.getElementById("annoAlEnginePick"),
    alRefresh: document.getElementById("annoAlRefresh"),
    alLabels: document.getElementById("annoAlLabels"),
    alConf: document.getElementById("annoAlConf"),
    alNms: document.getElementById("annoAlNms"),
    alService: document.getElementById("annoAlService"),
    conflictModal: document.getElementById("annoConflictModal"),
    conflictText: document.getElementById("annoConflictText"),
    importSource: document.getElementById("annoImportSource"),
    importMode: document.getElementById("annoImportMode"),
    importDuplicate: document.getElementById("annoImportDuplicate"),
    importReport: document.getElementById("annoImportReport"),
    importApply: document.getElementById("annoImportApply"),
    importApplySave: document.getElementById("annoImportApplySave"),
    batchLabel: document.getElementById("annoBatchLabel"),
    batchStop: document.getElementById("annoBatchStop"),
    batchSkipDone: document.getElementById("annoBatchSkipDone"),
    batchProgress: document.getElementById("annoBatchProgress"),
    batchHint: document.getElementById("annoBatchHint"),
    acceptOne: document.getElementById("annoAcceptOne"),
    rejectOne: document.getElementById("annoRejectOne"),
    // 审核状态
    reviewCount: document.getElementById("annoReviewCount"),
    markReviewed: document.getElementById("annoMarkReviewed"),
    unmarkReviewed: document.getElementById("annoUnmarkReviewed"),
    // 预标注通道与提示参数
    alChannel: document.getElementById("annoAlChannel"),
    chEngine: document.getElementById("annoChEngine"),
    chPrompt: document.getElementById("annoChPrompt"),
    chText: document.getElementById("annoChText"),
    alPrompt: document.getElementById("annoAlPrompt"),
    alBoxThr: document.getElementById("annoAlBoxThr"),
    alTextThr: document.getElementById("annoAlTextThr"),
    alRefine: document.getElementById("annoAlRefine"),
    alEpsilon: document.getElementById("annoAlEpsilon"),
    promptClear: document.getElementById("annoPromptClear"),
    promptRun: document.getElementById("annoPromptRun"),
    // 模型仓库
    modelBox: document.getElementById("annoModelBox"),
    modelList: document.getElementById("annoModelList"),
    modelCount: document.getElementById("annoModelCount"),
    modelHint: document.getElementById("annoModelHint"),
  });

  document.getElementById("annoOpen")?.addEventListener("click", () => openProject((el.path.value || "").trim()));
  document.getElementById("annoCreate")?.addEventListener("click", createProject);
  // 引导条：主按钮按当前步骤分发；点步骤圆片走同一套动作
  el.guideAction?.addEventListener("click", () => {
    const step = currentStep(buildAnnoSteps(guideContext()));
    runGuideAction(step ? resolveGuideAction(step) : "");
  });
  el.guideToggle?.addEventListener("click", () => {
    const collapsed = el.guide?.classList.contains("is-collapsed");
    localStorage.setItem(GUIDE_STORAGE, collapsed ? "1" : "0");
    applyGuideVisibility();
    // 引导条高度变了，画布要重算一次尺寸，否则会被压扁或留白
    requestAnimationFrame(() => {
      resizeCanvas();
      renderCanvas();
    });
  });
  // 用户亲手折叠/展开进阶块后，就不再让"有草稿就自动展开"的规则改它。
  // 监听 <summary> 的 click 而不是 <details> 的 toggle：toggle 事件在**脚本**
  // 改 `open` 时也会触发，若挂在那里，自动展开一次就会被误判成"用户动过"，
  // 之后永远不会自动收起。
  el.advDraft?.querySelector("summary")?.addEventListener("click", () => {
    state.advDraftTouched = true;
  });
  document.getElementById("annoSaveNow")?.addEventListener("click", () => saveProject({ silent: false }));
  document.getElementById("annoSaveAs")?.addEventListener("click", saveAs);
  document.getElementById("annoAddImages")?.addEventListener("click", addImages);
  document.getElementById("annoAddFiles")?.addEventListener("click", addImagesByDialog);
  document.getElementById("annoImportPreview")?.addEventListener("click", previewImport);
  document.getElementById("annoImportApply")?.addEventListener("click", () => applyImport({ save: false }));
  document.getElementById("annoImportApplySave")?.addEventListener("click", () => applyImport({ save: true }));
  // 任何影响结果的参数一改，上一次预览就作废：不能拿旧报告去导新数据源
  el.importMode?.addEventListener("change", invalidateImport);
  el.importDuplicate?.addEventListener("change", invalidateImport);
  el.importSource?.addEventListener("input", invalidateImport);
  el.importSource?.addEventListener("change", invalidateImport);
  document.getElementById("annoImportSplits")?.addEventListener("change", invalidateImport);
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
  document.getElementById("annoAutoLabel")?.addEventListener("click", runCurrentChannel);
  document.getElementById("annoExport")?.addEventListener("click", runExport);
  document.getElementById("annoClassAdd")?.addEventListener("click", addClass);
  document.getElementById("annoAcceptAll")?.addEventListener("click", acceptAllDrafts);
  document.getElementById("annoRejectAll")?.addEventListener("click", rejectAllDrafts);
  el.acceptOne?.addEventListener("click", acceptCurrentDraft);
  el.rejectOne?.addEventListener("click", rejectCurrentDraft);
  el.markReviewed?.addEventListener("click", () => markReviewed());
  el.unmarkReviewed?.addEventListener("click", unmarkReviewed);
  el.alChannel?.addEventListener("change", () => setAlChannel(el.alChannel.value));
  el.promptClear?.addEventListener("click", clearPrompt);
  el.promptRun?.addEventListener("click", runPromptLabel);
  // 模型仓库是按需拉的：展开才去问后端，避免一次打开页面就多发一个请求。
  el.modelBox?.addEventListener("toggle", () => {
    if (el.modelBox.open && !state.modelzoo) refreshModelzoo();
  });
  el.batchLabel?.addEventListener("click", runBatchAutoLabel);
  el.batchStop?.addEventListener("click", stopBatchRun);
  el.alRefresh?.addEventListener("click", () => refreshEngineOptions({ manual: true }));
  el.alEnginePick?.addEventListener("change", applyEngineOption);
  // 训练产物扫描是懒加载：不展开预标注参数就不去扫盘（大工程的 runs/ 树很深）。
  el.autoLabelBox?.addEventListener("toggle", () => {
    if (el.autoLabelBox.open && !state.engineLoaded) refreshEngineOptions();
  });

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
    // 草稿有 800ms 防抖窗口，也算"未保存"：预标注跑了几百个框，
    // 因为刷新丢掉比丢一张手工框更让人崩溃。审核状态同理。
    if (state.dirty || state.draftDirty.size || reviewPending.size) {
      event.preventDefault();
      event.returnValue = "有未保存的标注或草稿，确定离开？";
    }
    flushDraftChanges({ silent: true });
    flushReviewChanges({ silent: true });
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
    // 预标注相关：草稿池、分档、批量进度。e2e 要断言"草稿确实落盘/确实分档"，
    // 否则只能靠截图看颜色，改个阈值就得重新校准像素。
    drafts: currentDrafts().map((draft) => ({
      id: draft.id,
      score: Number(draft.score),
      tier: draftTier(draft.score),
      label: draft.label || "",
    })),
    draftDirty: [...state.draftDirty.keys()],
    draftTotal: [...state.drafts.values()].reduce((sum, list) => sum + list.length, 0),
    // 提示模式与审核状态：e2e 要断言"点了几下""K 之后跳到哪张""刷新后还在"
    channel: state.alChannel,
    prompt: {
      points: state.promptPoints.length,
      negatives: state.promptLabels.filter((label) => Number(label) === 0).length,
      box: state.promptBox ? state.promptBox.map((p) => [Math.round(p[0]), Math.round(p[1])]) : null,
      busy: state.promptBusy,
    },
    review: [...state.review.keys()],
    reviewed: state.review.size,
    reviewPending: [...reviewPending.keys()],
    // 五步流程：e2e 断言"第几步该亮"不用去数 class
    guide: buildAnnoSteps(guideContext()).map((step) => ({ id: step.id, state: step.state })),
    advDraftOpen: Boolean(el.advDraft?.open),
    batch: state.batchJob
      ? {
          id: state.batchJob.id,
          status: state.batchJob.status,
          done: state.batchJob.done,
          total: state.batchJob.total,
          drafts: state.batchJob.drafts,
        }
      : null,
  });

  resizeCanvas();
  // 通道初始化：按默认值把三块参数区显示对
  setAlChannel(state.alChannel);
  // 引导条：读回上次的收起状态，再渲染第一帧
  applyGuideVisibility();
  renderAll();
  renderCanvas();
}

