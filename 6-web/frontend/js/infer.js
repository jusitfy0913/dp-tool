// 推理页：上传图片 → 调后端 → 展示结果。
//
// 数据流：
//   后端返回 base64 图（原图 / 检出图 / 热能图 / 轮廓图）+ 结构化 boxes。
//   图片直接挂到查看器；框由前端用 SVG 叠加层重绘，因此在任意倍率下都锐利。
//
// 关键设计：
//   · 叠加层与后端烧录的标注图共用同一份 `class_colors`，保证"屏幕上看到的颜色"
//     与"导出的标注图颜色"完全一致。
//   · 参数（conf/nms/双阈值）只改 UI 状态，点"开始推理"才提交。
//     实时推理会让拖动滑块变成卡顿，而且每次都要重建引擎句柄。
//   · 结果表、分布图、导出三者都从同一份 `lastResult` 生成，不会出现"导出的和看到的不一样"。

import { anomalyMetadataInfo, getJSON, postForm, postJSON } from "./api.js";
import {
  bindOcrPicks,
  buildOcrOverlay,
  highlightOcr,
  loadOcrModels,
  ocrCsv,
  ocrPlainText,
  renderOcrTable,
} from "./ocr.js";
import { applyOverlayScale, buildOverlay, setOverlayHighlight } from "./overlay.js";
import { createViewerStrip } from "./viewer.js";

let dom = {};
let viewers = null;
let sourceObjectUrl = "";

// 模型清单与上次参数，用于"命中缓存"提示与自动填充
let models = [];
let lastParamKey = "";
let lastResult = null;
let lastKind = "yolo";

// metadata.json 的推荐阈值回显。state: idle / loading / ok / error。
// 只用于界面展示——真正提交给后端的是"是否覆盖"这个语义，不是这里的数字。
let metaInfo = { state: "idle", path: "", image: null, pixel: null, model: "", size: null, accuracy: null, error: "" };
// 请求序号：路径连改几次时，只有最后一次的响应能落到界面上
let metaSeq = 0;

// 叠加层状态
let overlaySvg = null;
let overlayScale = 1;
let highlightIndex = -1;
let sortKey = "confidence";
let sortAsc = false;

export function initInfer() {
  dom = {
    form: document.getElementById("inferForm"),
    kind: document.getElementById("inferKind"),
    yoloGroup: document.getElementById("inferYoloGroup"),
    anomGroup: document.getElementById("inferAnomGroup"),
    ocrGroup: document.getElementById("inferOcrGroup"),
    artifactGroup: document.getElementById("inferArtifactGroup"),
    btnInfer: document.getElementById("btnInfer"),
    btnWorkerRestart: document.getElementById("btnWorkerRestart"),
    hint: document.getElementById("inferHint"),
    rebuildHint: document.getElementById("inferRebuildHint"),
    badge: document.getElementById("inferBadge"),
    metrics: document.getElementById("inferMetrics"),
    mount: document.getElementById("inferViewers"),
    file: document.getElementById("inferImage"),

    modelSelect: document.getElementById("inferModelSelect"),
    modelHint: document.getElementById("inferModelHint"),
    btnRefreshModels: document.getElementById("btnRefreshModels"),
    weight: document.getElementById("inferWeight"),
    weightHint: document.getElementById("inferWeightHint"),

    timing: document.getElementById("inferTiming"),
    timingBar: document.getElementById("timingBar"),
    timingLegend: document.getElementById("timingLegend"),

    toolbar: document.getElementById("inferToolbar"),
    detail: document.getElementById("inferDetail"),
    boxHead: document.getElementById("boxHead"),
    boxRows: document.getElementById("boxRows"),
    confHist: document.getElementById("confHist"),

    useMetaThresh: document.getElementById("anomUseMetaThresh"),
    anomThreshHint: document.getElementById("anomThreshHint"),
    metaReadout: document.getElementById("anomMetaReadout"),
    metaTag: document.getElementById("anomMetaTag"),
    metaImageThr: document.getElementById("anomMetaImageThr"),
    metaPixelThr: document.getElementById("anomMetaPixelThr"),
    metaModel: document.getElementById("anomMetaModel"),
    colormap: document.getElementById("anomColormap"),
    overlayOpacity: document.getElementById("anomOverlayOpacity"),
  };

  // 四个窗口：原图 / 检出图 / 热能图 / 异常轮廓。
  // 后两个默认关闭且标记为"暂无内容"——YOLO 模式下它们永远不会有图，
  // 一上来就占满一屏只会挤掉真正要看的东西。
  viewers = createViewerStrip({
    mount: dom.mount,
    id: "infer",
    windows: [
      { id: "source", title: "原图", enabled: true },
      { id: "result", title: "检出结果", enabled: true },
      { id: "heatmap", title: "热能图", enabled: false },
      { id: "mask", title: "异常轮廓", enabled: false },
    ],
    emptyText: "全部窗口已关闭，点击上方开关重新显示",
  });

  bindSliders();
  bindOcrPicks();
  dom.kind.addEventListener("change", syncKind);
  dom.form.addEventListener("submit", onSubmit);
  dom.btnWorkerRestart?.addEventListener("click", restartWorker);
  dom.file.addEventListener("change", onFileChange);
  dom.useMetaThresh.addEventListener("change", syncThresholdControls);
  dom.modelSelect.addEventListener("change", onModelChange);
  dom.btnRefreshModels.addEventListener("click", () => {
    loadModels(true);
    loadOcrModels();
  });
  // 切换权重格式只换路径，不重扫目录：引擎与 ONNX 都在同一条产物里
  dom.weight?.addEventListener("change", onWeightChange);
  // 手动改路径后让权重格式跟随后缀，避免界面显示和实际后端不一致
  for (const id of ["yoloEngine", "anomEngine"]) {
    document.getElementById(id)?.addEventListener("change", (event) => {
      const isOnnx = String(event.target.value || "").toLowerCase().endsWith(".onnx");
      if (dom.weight) dom.weight.value = isOnnx ? "onnx" : "engine";
      updateWeightHint();
    });
  }

  // metadata 路径一改就重读推荐值：手动输入走 change（失焦提交），
  // 点「浏览」选文件走 writeValue() 也会派发 change，两条路都能覆盖。
  document.getElementById("anomMetadata")?.addEventListener("change", () => refreshMetaThresholds());
  // 从别的页签切回来时再看一眼：可能在训练页/串联页里刚换了 metadata
  document.addEventListener("tab-change", (event) => {
    if (event.detail && event.detail.name === "infer") refreshMetaThresholds();
  });

  for (const id of ["optShowBoxes", "optShowLabels", "optFadeLow"]) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => rebuildOverlay());
  }
  dom.overlayOpacity.addEventListener("input", () => applyOverlayOpacity());

  document.getElementById("btnExportJson").addEventListener("click", exportJson);
  document.getElementById("btnExportCsv").addEventListener("click", exportCsv);
  document.getElementById("btnCopyOcr")?.addEventListener("click", copyOcrText);
  document.getElementById("btnExportTxt")?.addEventListener("click", exportTxt);

  // 标签尺寸随倍率反向缩放，保证屏幕字号恒定
  dom.mount.addEventListener("viewer-zoom", (event) => {
    overlayScale = event.detail.scale || 1;
    applyOverlayScale(overlaySvg, overlayScale);
  });

  // 参数一改就点亮"会重建引擎"提示，让用户知道慢在哪
  for (const id of ["inferConf", "inferNms", "anomImageThr", "anomPixelThr", "ocrDetThresh", "ocrBoxThresh", "ocrDet", "ocrRec", "ocrTextline", "ocrDevice"]) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", markParamsDirty);
  }

  syncKind();
  syncThresholdControls();
  refreshMetaThresholds();
  loadModels(false);
  loadOcrModels();
}

// ---------------------------------------------------------------------------
// 参数控件
// ---------------------------------------------------------------------------

/** 数字框 ↔ 滑块双向联动。 */
function bindSlider(numberId, rangeId) {
  const number = document.getElementById(numberId);
  const range = document.getElementById(rangeId);
  if (!number || !range) return;

  const clamp = (value) => {
    const low = Number(range.min);
    const high = Number(range.max);
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return Number(range.value);
    return Math.min(high, Math.max(low, parsed));
  };

  range.addEventListener("input", () => {
    number.value = range.value;
  });
  number.addEventListener("input", () => {
    // 输入框允许暂时为空（用户正在删除重输），此时不要强行写回
    if (number.value === "") return;
    range.value = String(clamp(number.value));
  });
  number.addEventListener("blur", () => {
    const value = clamp(number.value);
    number.value = String(value);
    range.value = String(value);
  });
}

function bindSliders() {
  bindSlider("inferConf", "inferConfRange");
  bindSlider("inferNms", "inferNmsRange");
  bindSlider("anomImageThr", "anomImageThrRange");
  bindSlider("anomPixelThr", "anomPixelThrRange");
  bindSlider("anomOverlayOpacity", "anomOverlayOpacityRange");
  bindSlider("ocrDetThresh", "ocrDetThreshRange");
  bindSlider("ocrBoxThresh", "ocrBoxThreshRange");
}

/** "沿用 metadata 默认值"开关：关掉后才允许编辑两个阈值。 */
function syncThresholdControls() {
  const useMeta = dom.useMetaThresh.checked;
  for (const key of ["imageThreshold", "pixelThreshold"]) {
    const block = document.querySelector(`.param[data-param="${key}"]`);
    if (!block) continue;
    block.classList.toggle("is-disabled", useMeta);
    for (const input of block.querySelectorAll("input")) {
      input.disabled = useMeta;
    }
  }
  // 勾选状态下把文件里的推荐值写进这两个只读框。
  // 之前这里只切 disabled，框里留着默认 0.5，用户完全看不出实际会用多少。
  if (useMeta) applyMetaThresholdsToInputs();
  dom.anomThreshHint.textContent = thresholdHint(useMeta);
}

// ---------------------------------------------------------------------------
// metadata 推荐值回显
// ---------------------------------------------------------------------------
/** 读一次 metadata.json 的推荐值并刷新界面。 */
async function refreshMetaThresholds() {
  const path = value("anomMetadata");
  const seq = ++metaSeq;
  if (!path) {
    metaInfo = { state: "idle", path: "", image: null, pixel: null, model: "", size: null, accuracy: null, error: "" };
    renderMetaReadout();
    syncThresholdControls();
    return;
  }
  metaInfo = { ...metaInfo, state: "loading", path };
  renderMetaReadout();
  try {
    const info = await anomalyMetadataInfo(path);
    if (seq !== metaSeq) return; // 路径又变了，这次响应已经过时
    metaInfo = {
      state: "ok",
      path,
      image: asNumber(info.image_threshold),
      pixel: asNumber(info.pixel_threshold),
      model: info.model_type || "",
      size: Array.isArray(info.input_size) ? info.input_size : null,
      accuracy: asNumber(info.sample_accuracy),
      error: "",
    };
  } catch (error) {
    if (seq !== metaSeq) return;
    metaInfo = { state: "error", path, image: null, pixel: null, model: "", size: null, accuracy: null,
      error: error && error.message ? error.message : String(error) };
  }
  renderMetaReadout();
  syncThresholdControls();
}

/** 把 metadata 的推荐阈值写进两个（此时只读的）输入框与滑块。 */
function applyMetaThresholdsToInputs() {
  const pairs = [
    ["anomImageThr", "anomImageThrRange", metaInfo.image],
    ["anomPixelThr", "anomPixelThrRange", metaInfo.pixel],
  ];
  for (const [numberId, rangeId, raw] of pairs) {
    const number = document.getElementById(numberId);
    const range = document.getElementById(rangeId);
    if (!Number.isFinite(raw)) {
      // 读不到就清空，别把上一个 metadata 的数字留在那里冒充本次的推荐值
      if (number) number.value = "";
      continue;
    }
    const text = String(round4(raw));
    if (number) number.value = text;
    if (range) range.value = text;
  }
}

/** 卡片：列出文件里记录的值；读不到时说明原因，而不是静默隐藏。 */
function renderMetaReadout() {
  if (!dom.metaReadout) return;
  const info = metaInfo;
  if (info.state === "idle") {
    dom.metaReadout.hidden = true;
    return;
  }
  dom.metaReadout.hidden = false;
  dom.metaImageThr.textContent = fmtThreshold(info.image);
  dom.metaPixelThr.textContent = fmtThreshold(info.pixel);
  dom.metaModel.textContent = [info.model || "未记录", info.size ? `${info.size[0]}×${info.size[1]}` : ""]
    .filter(Boolean)
    .join(" / ");
  for (const el of [dom.metaImageThr, dom.metaPixelThr]) {
    el.classList.toggle("is-missing", !Number.isFinite(
      el === dom.metaImageThr ? info.image : info.pixel,
    ));
  }
  if (info.state === "loading") {
    setMetaTag("读取中…", false);
  } else if (info.state === "error") {
    setMetaTag("读取失败", false);
  } else if (Number.isFinite(info.accuracy)) {
    // 有 calibration 区块说明阈值是按现场样本重新标定过的，可信度更高
    setMetaTag(`已校准 ${(info.accuracy * 100).toFixed(0)}%`, true);
  } else {
    setMetaTag("文件原值", false);
  }
  // 失败原因通常较长（后端原文），塞进卡片会撑爆两列网格，挂到 title 上按需查看
  if (dom.metaTag) dom.metaTag.title = info.state === "error" ? info.error : info.path || "";
}

function setMetaTag(text, calibrated) {
  if (!dom.metaTag) return;
  dom.metaTag.textContent = text;
  dom.metaTag.classList.toggle("is-calibrated", Boolean(calibrated));
}

/** 提示语：明确说清"现在用的是文件里的值"还是"你手填的值"，并把数字带上。 */
function thresholdHint(useMeta) {
  const hasValues = Number.isFinite(metaInfo.image) || Number.isFinite(metaInfo.pixel);
  const pairs = `图像 ${fmtThreshold(metaInfo.image)} ／ 像素 ${fmtThreshold(metaInfo.pixel)}`;
  if (useMeta) {
    if (metaInfo.state === "error") {
      return `沿用 metadata，但读取失败：${metaInfo.error}（推理时后端仍会读这个文件）`;
    }
    if (metaInfo.state === "loading") return "正在读取 metadata 里的推荐阈值…";
    if (hasValues) {
      return `沿用中：直接使用 metadata 记录的值（${pairs}）。取消勾选后可手动覆盖。`;
    }
    return "直接使用 metadata 里记录的阈值。取消勾选后可手动覆盖。";
  }
  return hasValues
    ? `手动覆盖中：当前用界面上的阈值，metadata 原值 ${pairs}。`
    : "手动覆盖 metadata 阈值。改阈值同样会重建引擎句柄。";
}

function asNumber(raw) {
  const parsed = Number(raw);
  return raw === null || raw === undefined || raw === "" || !Number.isFinite(parsed) ? null : parsed;
}

function round4(raw) {
  return Math.round(Number(raw) * 10000) / 10000;
}


/** 参数改动后提示"会重建引擎"，改回原值则提示"命中缓存"。 */
function markParamsDirty() {
  const key = currentParamKey();
  if (!lastParamKey) {
    dom.rebuildHint.classList.add("hidden");
    return;
  }
  const dirty = key !== lastParamKey;
  dom.rebuildHint.classList.remove("hidden");
  dom.rebuildHint.classList.toggle("hint-warn", dirty);
  dom.rebuildHint.classList.toggle("hint-ok", !dirty);
  dom.rebuildHint.classList.toggle("hint", true);
  dom.rebuildHint.textContent = dirty
    ? "参数已改动：下次推理会重建引擎句柄，首次约需数秒。再改回去可命中缓存、立即返回。"
    : "当前参数与上次推理一致，将直接命中已加载的引擎，无需重建。";
}

/** 当前参数组合的指纹，用于判断是否与上次相同（同参数会命中句柄缓存）。 */
function currentParamKey() {
  const parts = [dom.kind.value];
  if (dom.kind.value === "yolo") {
    parts.push(value("yoloEngine"), value("inferTask"), value("inferConf"), value("inferNms"));
  } else if (dom.kind.value === "ocr") {
    parts.push(value("ocrDet"), value("ocrRec"), value("ocrTextline"), value("ocrDevice"));
    parts.push(value("ocrThreads"), value("ocrBatch"), value("ocrDetThresh"), value("ocrBoxThresh"));
    parts.push(value("ocrUnclip"), value("ocrRecScore"), document.getElementById("ocrMkldnn")?.checked ? "1" : "0");
  } else {
    parts.push(value("anomEngine"), value("anomMetadata"), value("anomModelType"));
    parts.push(dom.useMetaThresh.checked ? "meta" : `${value("anomImageThr")}|${value("anomPixelThr")}`);
  }
  return parts.join("::");
}

// ---------------------------------------------------------------------------
// 模型下拉
// ---------------------------------------------------------------------------

async function loadModels(manual) {
  if (manual) dom.modelHint.textContent = "正在扫描训练产物…";
  try {
    const data = await getJSON("/api/models");
    models = data.entries || [];
    renderModelOptions(data);
  } catch (error) {
    dom.modelHint.textContent = `扫描失败：${error.message}`;
  }
}

function renderModelOptions(data) {
  const select = dom.modelSelect;
  const previous = select.value;
  select.replaceChildren();

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— 手工填写下面的路径 —";
  select.appendChild(placeholder);

  // 按类型分组，避免 YOLO 与 Anomalib 混在一个长列表里
  for (const kind of ["yolo", "anomalib"]) {
    const group = document.createElement("optgroup");
    group.label = kind === "yolo" ? "YOLO 目标检测" : "Anomalib 异常检测";
    const items = models.filter((item) => item.kind === kind);
    if (!items.length) {
      const none = document.createElement("option");
      none.disabled = true;
      none.textContent = "（该目录暂无引擎）";
      group.appendChild(none);
    }
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.path;
      const size = (item.size / (1024 * 1024)).toFixed(1);
      const warn = item.kind === "yolo" && !item.transposed ? " ⚠ 未转置" : "";
      const onnxTag = item.onnx ? " · ONNX" : "";
      option.textContent = `${item.label} · ${size}MB${onnxTag}${warn}`;
      group.appendChild(option);
    }
    select.appendChild(group);
  }

  if (previous && models.some((item) => item.path === previous)) {
    select.value = previous;
  }

  const broken = (data.roots || []).filter((root) => !root.ok);
  const parts = [`共 ${data.total ?? models.length} 个引擎`];
  if (data.truncated) parts.push(`（已截断到 ${models.length} 个）`);
  for (const root of broken) {
    parts.push(`· ${root.kind} 目录未扫描：${root.reason}`);
  }
  dom.modelHint.textContent = parts.join(" ");
}

/** 当前下拉选中的训练产物条目（手工输入时不指向任何条目）。 */
function currentModel() {
  return models.find((entry) => entry.path === dom.modelSelect.value) || null;
}

/** 该条目在当前权重格式下应该填哪个文件。 */
function modelPathFor(item) {
  if (!item) return "";
  if (dom.weight && dom.weight.value === "onnx" && item.onnx) return item.onnx;
  return item.path;
}

/** ONNX 选项只在条目真的有同名 .onnx 时可选，避免选了却填不出路径。 */
function syncWeightOptions(item) {
  if (!dom.weight) return;
  const option = dom.weight.querySelector('option[value="onnx"]');
  if (option) option.disabled = !item || !item.onnx;
  if (dom.weight.value === "onnx" && (!item || !item.onnx)) {
    dom.weight.value = "engine";
  }
  updateWeightHint();
}

function updateWeightHint() {
  if (!dom.weightHint) return;
  const item = currentModel();
  const onnx = dom.weight && dom.weight.value === "onnx";
  if (onnx) {
    dom.weightHint.textContent = item && item.onnx
      ? `将使用 ${fileName(item.onnx)}，后端为 onnx_infer.dll（不可用时退回 Python onnxruntime）。`
      : "ONNX 走 onnx_infer 动态库；当前产物没有同名 .onnx 时该选项不可选。";
  } else {
    dom.weightHint.textContent = "TensorRT 引擎走 yolo_infer / anomalib_infer 动态库。";
  }
}

function onWeightChange() {
  const item = currentModel();
  if (!item) {
    updateWeightHint();
    return;
  }
  applyModelToForm(item);
}

function onModelChange() {
  const item = currentModel();
  if (!item) {
    updateWeightHint();
    return;
  }

  dom.kind.value = item.kind === "anomalib" ? "anomalib" : "yolo";
  syncKind();
  syncWeightOptions(item);
  applyModelToForm(item);
}

/** 把条目内容填进表单：路径按当前权重格式取引擎或 ONNX，其余字段两份共用。 */
function applyModelToForm(item) {
  const chosen = modelPathFor(item);
  const isOnnx = chosen.toLowerCase().endsWith(".onnx");

  if (item.kind === "yolo") {
    setValue("yoloEngine", chosen);
    setValue("yoloLabels", item.labels || "");
    if (item.task && ["detect", "segment", "pose", "obb"].includes(item.task)) {
      setValue("inferTask", item.task);
    }
  } else {
    setValue("anomEngine", chosen);
    // 优先用校准过的 metadata：它是按现场数据重新标定过阈值的版本
    setValue("anomMetadata", item.metadata_calibrated || item.metadata || "");
    if (item.model_type) {
      const mapped = normalizeModelType(item.model_type);
      if (mapped) setValue("anomModelType", mapped);
    }
    // setValue 不派发 change，这里显式补一次，否则推荐阈值卡片还停在上一个模型
    refreshMetaThresholds();
  }

  // 未转置的引擎会让 DLL 解码越界，产出的框是垃圾值。必须提前拦。
  // 走 ONNX 时这条警告不适用：ONNX 侧按实际输出布局自适应解析。
  if (item.kind === "yolo" && !isOnnx && !item.transposed) {
    dom.modelHint.textContent =
      `⚠ ${item.name} 对应的 ONNX 是未转置的（${item.onnx ? fileName(item.onnx) : "无"}）。`
      + "用它编译的引擎解码会越界，结果不可信。请改用 *_fp16.engine 中对应 .transd.onnx 的那个。";
    dom.modelHint.classList.add("hint-warn");
  } else {
    dom.modelHint.classList.remove("hint-warn");
    dom.modelHint.textContent = isOnnx
      ? `已填入 ${fileName(chosen)}（ONNX，走 onnx_infer.dll）`
      : `已填入 ${item.name}${item.labels ? "（含标签文件）" : ""}`;
  }
  markParamsDirty();
}

/** `Patchcore` / `PATCHCORE` → 下拉里的 `PatchCore`。 */
function normalizeModelType(raw) {
  const lower = String(raw).toLowerCase().replace(/[\s_-]/g, "");
  if (lower.includes("patchcore")) return "PatchCore";
  if (lower.includes("padim")) return "PaDiM";
  if (lower.includes("efficientad")) return "EfficientAd";
  return "";
}

// ---------------------------------------------------------------------------
// 推理类型切换
// ---------------------------------------------------------------------------

function syncKind() {
  const kind = dom.kind.value;
  const isYolo = kind === "yolo";
  const isOcr = kind === "ocr";
  dom.yoloGroup.classList.toggle("hidden", !isYolo);
  dom.anomGroup.classList.toggle("hidden", kind !== "anomalib");
  dom.ocrGroup?.classList.toggle("hidden", !isOcr);
  dom.artifactGroup?.classList.toggle("hidden", isOcr);
  lastKind = kind;
  viewers.viewer("result").setTitle(isOcr ? "文字位置" : isYolo ? "检出结果" : "异常叠加");
  document.getElementById("btnCopyOcr")?.classList.toggle("hidden", !isOcr);
  document.getElementById("btnExportTxt")?.classList.toggle("hidden", !isOcr);
  if (isYolo || isOcr) {
    // YOLO 永远不产出热能图与轮廓图，直接标记为"无内容"并关掉，
    // 免得用户点开一个永远空着的窗口
    viewers.setAvailable("heatmap", false);
    viewers.setAvailable("mask", false);
    viewers.setEnabled("heatmap", false);
    viewers.setEnabled("mask", false);
  }
  // 切到 Anomalib 时不预先开放这两个窗口：等真有图了再开放，
  // 由 renderAnomalib 决定，避免"点开了却是空的"
  if (kind === "anomalib") refreshMetaThresholds();
  markParamsDirty();
}

function onFileChange() {
  const file = dom.file.files?.[0];
  if (!file) return;
  // 上一张图的 object URL 必须显式释放，否则大图会一直占着内存
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceObjectUrl = URL.createObjectURL(file);

  // 本地预览用 object URL，比等后端回传快得多；
  // 推理完成后如果后端回了 source_png，会换成它（两者像素一致）
  viewers.viewer("source").setImage(sourceObjectUrl);
  viewers.viewer("result").clear();
  viewers.viewer("heatmap").clear();
  viewers.viewer("mask").clear();
}

// ---------------------------------------------------------------------------
// 提交
// ---------------------------------------------------------------------------

async function onSubmit(event) {
  event.preventDefault();
  const kind = dom.kind.value;
  const file = dom.file.files?.[0];

  if (!file) {
    setHint("请先选择一张图片");
    return;
  }
  if (kind === "yolo" && !value("yoloEngine")) {
    setHint("请填写 YOLO 引擎路径");
    return;
  }
  if (kind === "anomalib" && (!value("anomEngine") || !value("anomMetadata"))) {
    setHint("请填写引擎与 metadata.json 路径");
    return;
  }
  if (kind === "ocr" && (!value("ocrDet") || !value("ocrRec"))) {
    setHint("请选择检测模型和识别模型");
    return;
  }

  const paramKey = currentParamKey();
  const cached = paramKey === lastParamKey;

  setHint("");
  dom.btnInfer.disabled = true;
  if (kind === "ocr") setBadge("OCR 推理中（CPU 单张约 10 秒以上）…", "run");
  else setBadge(cached ? "推理中（命中缓存）…" : "推理中（重建引擎）…", "run");
  dom.metrics.replaceChildren();

  const form = new FormData();
  form.append("image", file);

  let url;
  if (kind === "ocr") {
    url = "/api/infer/ocr";
    form.append("det_dir", value("ocrDet"));
    form.append("rec_dir", value("ocrRec"));
    const textline = value("ocrTextline");
    if (textline) form.append("textline_ori_dir", textline);
    form.append("device", value("ocrDevice") || "cpu");
    form.append("enable_mkldnn", document.getElementById("ocrMkldnn")?.checked ? "1" : "0");
    form.append("cpu_threads", value("ocrThreads") || "8");
    form.append("rec_batch_size", value("ocrBatch") || "6");
    form.append("det_thresh", value("ocrDetThresh") || "0.3");
    form.append("det_box_thresh", value("ocrBoxThresh") || "0.6");
    form.append("det_unclip_ratio", value("ocrUnclip") || "1.5");
    form.append("rec_score_thresh", value("ocrRecScore") || "0");
  } else if (kind === "yolo") {
    url = "/api/infer/yolo";
    form.append("engine", value("yoloEngine"));
    form.append("task", value("inferTask"));
    form.append("conf", value("inferConf"));
    form.append("nms", value("inferNms"));
    const labels = value("yoloLabels");
    if (labels) form.append("labels", labels);
  } else {
    url = "/api/infer/anomalib";
    form.append("engine", value("anomEngine"));
    form.append("metadata", value("anomMetadata"));
    form.append("model_type", value("anomModelType"));
    form.append("heatmap_colormap", value("anomColormap"));
    // 勾了"沿用 metadata"就不提交阈值字段，让后端用 -1 走默认分支
    if (!dom.useMetaThresh.checked) {
      form.append("image_threshold", value("anomImageThr"));
      form.append("pixel_threshold", value("anomPixelThr"));
    }
  }

  const started = performance.now();
  try {
    const result = await postForm(url, form);
    lastResult = result;
    lastKind = kind;
    lastParamKey = paramKey;
    // 完成后按"当前参数 vs 上次参数"重算提示，而不是简单地隐藏。
    // 不能只 hide：点按钮会让数字框失焦并触发一次 change，事件顺序不同就会把提示重新点亮，
    // 这里统一收敛到 markParamsDirty 里的同一套判断，结果与事件顺序无关。
    markParamsDirty();
    renderResult(result, kind, performance.now() - started);
    setBadge("完成", "done");
  } catch (error) {
    setHint(error.message);
    setBadge("失败", "fail");
    // 503/504 来自推理进程崩溃、超时或熔断：给一个手动重启入口
    if (error.status === 503 || error.status === 504) dom.btnWorkerRestart?.classList.remove("hidden");
  } finally {
    dom.btnInfer.disabled = false;
  }
}

async function restartWorker() {
  const button = dom.btnWorkerRestart;
  button.disabled = true;
  setHint("正在重启推理进程，首次加载引擎需要几秒…");
  try {
    const state = await postJSON("/api/infer/worker/restart", {});
    if (state.state === "running") {
      setHint("推理进程已重启，可以重新推理");
      button.classList.add("hidden");
    } else {
      setHint(`推理进程重启失败：${state.last_error || state.state}`);
    }
  } catch (error) {
    setHint(`推理进程重启失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 结果渲染
// ---------------------------------------------------------------------------

function renderResult(result, kind, roundTripMs) {
  if (result.warning) setMetric("降级提示", result.warning);

  if (kind === "yolo") renderYolo(result);
  else if (kind === "ocr") renderOcr(result);
  else renderAnomalib(result);

  renderTiming(result.timing, roundTripMs);
  dom.toolbar.classList.remove("hidden");

  if (kind === "yolo") {
    renderTable(result.boxes || []);
    renderHistogram(result.boxes || []);
    dom.detail.classList.remove("hidden");
  } else if (kind === "ocr") {
    renderOcrHead();
    renderOcrTable(dom.boxRows, result.lines || [], highlightIndex, pickOcrLine);
    renderHistogram((result.lines || []).map((line) => ({ confidence: line.score })));
    dom.detail.classList.remove("hidden");
  } else {
    // 异常检测没有"检出框"，明细表换成阈值与分数说明，避免留一张空表
    dom.detail.classList.add("hidden");
  }

  rebuildOverlay();
  applyOverlayOpacity();
}

function renderYolo(result) {
  const boxes = result.boxes || [];

  // 结果窗口显示"原图 + SVG 叠加层"，而不是后端烧录好框的 annotated_png。
  // 原因：两份画框会叠在一起（双重描边、标签互相盖住），而且烧录图放大后必然模糊。
  // annotated_png 仍然保留在 lastResult 里，只用于导出。
  if (sourceObjectUrl) {
    viewers.viewer("result").setImage(sourceObjectUrl);
  }
  viewers.setAvailable("result", true);
  viewers.setAvailable("heatmap", false);
  viewers.setAvailable("mask", false);
  document.getElementById("anomLegend")?.classList.add("hidden");

  setMetric("目标数", boxes.length);
  setMetric("后端", result.backend);
  if (result.timing) {
    setMetric("推理耗时", `${result.timing.inference_ms.toFixed(1)} ms`);
    setMetric("总耗时", `${result.timing.total_ms.toFixed(1)} ms`);
  }
  const labelInfo = result.labels || {};
  if (labelInfo.ok) {
    setMetric("标签文件", `${labelInfo.name || fileName(labelInfo.path)} · ${labelInfo.count} 类`);
  } else if (labelInfo.error) {
    setMetric("标签文件", labelInfo.error);
  }
  if (!boxes.length) setMetric("提示", "未检出目标，可尝试降低 conf");
}

function renderAnomalib(result) {
  if (result.source_png) {
    viewers.viewer("source").setImage(`data:image/png;base64,${result.source_png}`);
  }
  if (result.overlay_png) {
    viewers.viewer("result").setImage(`data:image/png;base64,${result.overlay_png}`);
    viewers.setAvailable("result", true);
  }
  if (result.heatmap_png) {
    viewers.viewer("heatmap").setImage(`data:image/png;base64,${result.heatmap_png}`);
    // 有内容再开放并默认打开：没有内容的窗口开着只会占地方
    viewers.setAvailable("heatmap", true);
    viewers.setEnabled("heatmap", true);
  } else {
    viewers.setAvailable("heatmap", false);
  }
  if (result.mask_png) {
    viewers.viewer("mask").setImage(`data:image/png;base64,${result.mask_png}`);
    viewers.setAvailable("mask", true);
    viewers.setEnabled("mask", true);
  } else {
    viewers.setAvailable("mask", false);
  }

  const verdict = result.is_anomaly ? "异常 NG" : "正常 OK";
  const badge = document.createElement("div");
  badge.className = `metric metric-verdict ${result.is_anomaly ? "is-ng" : "is-ok"}`;
  badge.innerHTML = `判定 <b>${escapeHtml(verdict)}</b>`;
  dom.metrics.appendChild(badge);

  setMetric("归一化分数", Number(result.normalized_score).toFixed(4));
  setMetric("原始分数", Number(result.raw_score).toFixed(4));

  // 生效阈值与 metadata 原值并列显示：覆盖之后如果只显示生效值，
  // 用户过几天回来会分不清"这个 0.6 是我设的还是文件里的"
  const thresholds = result.thresholds || {};
  const imageThr = thresholds.image || {};
  const pixelThr = thresholds.pixel || {};
  setMetric(
    "图像阈值",
    `${fmtThreshold(imageThr.effective)}${imageThr.overridden ? `（覆盖，原 ${fmtThreshold(imageThr.metadata)}）` : "（metadata）"}`,
  );
  setMetric(
    "像素阈值",
    `${fmtThreshold(pixelThr.effective)}${pixelThr.overridden ? `（覆盖，原 ${fmtThreshold(pixelThr.metadata)}）` : "（metadata）"}`,
  );
  if (result.heatmap_colormap) setMetric("色带", result.heatmap_colormap);
  renderPixelThresholdLegend(pixelThr.effective);
  if (result.timing) {
    setMetric("推理耗时", `${result.timing.inference_ms.toFixed(1)} ms`);
    setMetric("总耗时", `${result.timing.total_ms.toFixed(1)} ms`);
  }
}

function fmtThreshold(value) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(4);
}

/** 把像素阈值画到 0~1 色阶刻度上。 */
function renderPixelThresholdLegend(threshold) {
  const legend = document.getElementById("anomLegend");
  const mark = document.getElementById("pixelThrMark");
  const note = document.getElementById("pixelThrNote");
  if (!legend || !mark || !note) return;

  if (threshold === null || threshold === undefined || !Number.isFinite(Number(threshold))) {
    legend.classList.add("hidden");
    return;
  }
  const ratio = Math.min(1, Math.max(0, Number(threshold)));
  mark.style.left = `${ratio * 100}%`;
  note.textContent = `像素阈值 ${ratio.toFixed(4)}`;
  legend.classList.remove("hidden");
}

/** 三级耗时分解条。 */
function renderTiming(timing, roundTripMs) {
  if (!timing) {
    dom.timing.classList.add("hidden");
    return;
  }
  const pre = Number(timing.preprocess_ms) || 0;
  const infer = Number(timing.inference_ms) || 0;
  const post = Number(timing.postprocess_ms) || 0;
  const total = pre + infer + post || 1;

  const segments = [
    { key: "pre", label: "预处理", ms: pre, cls: "seg-pre" },
    { key: "infer", label: "推理", ms: infer, cls: "seg-infer" },
    { key: "post", label: "后处理", ms: post, cls: "seg-post" },
  ];

  dom.timingBar.replaceChildren();
  dom.timingLegend.replaceChildren();
  for (const seg of segments) {
    const bar = document.createElement("div");
    bar.className = `timing-seg ${seg.cls}`;
    bar.style.width = `${(seg.ms / total) * 100}%`;
    bar.title = `${seg.label} ${seg.ms.toFixed(2)} ms`;
    dom.timingBar.appendChild(bar);

    const item = document.createElement("div");
    item.className = "timing-item";
    item.innerHTML =
      `<span class="timing-dot ${seg.cls}"></span>`
      + `${escapeHtml(seg.label)} <b>${seg.ms.toFixed(2)} ms</b>`;
    dom.timingLegend.appendChild(item);
  }

  const totalItem = document.createElement("div");
  totalItem.className = "timing-item timing-total";
  totalItem.innerHTML = `合计 <b>${total.toFixed(2)} ms</b>`;
  dom.timingLegend.appendChild(totalItem);

  if (Number.isFinite(roundTripMs)) {
    const net = document.createElement("div");
    net.className = "timing-item timing-net";
    // 往返耗时与后端计时之差 = 上传 + 编码 + 传输，用来解释"为什么感觉比 40ms 慢"
    net.innerHTML = `含往返 <b>${roundTripMs.toFixed(0)} ms</b>`;
    dom.timingLegend.appendChild(net);
  }

  dom.timing.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// 明细表
// ---------------------------------------------------------------------------

const COLUMNS = [
  { key: "index", label: "#", align: "num", width: 4 },
  { key: "label_name", label: "类别", align: "" },
  { key: "confidence", label: "置信度", align: "num" },
  { key: "left", label: "left", align: "num" },
  { key: "top", label: "top", align: "num" },
  { key: "width", label: "宽", align: "num" },
  { key: "height", label: "高", align: "num" },
  { key: "area", label: "面积", align: "num" },
];

function renderTable(boxes) {
  dom.boxHead.replaceChildren();
  dom.boxRows.replaceChildren();

  const row = document.createElement("tr");
  for (const col of COLUMNS) {
    const th = document.createElement("th");
    th.textContent = col.label + (sortKey === col.key ? (sortAsc ? " ▲" : " ▼") : "");
    th.className = col.align;
    th.addEventListener("click", () => {
      if (sortKey === col.key) sortAsc = !sortAsc;
      else {
        sortKey = col.key;
        sortAsc = col.key === "index" || col.key === "label_name";
      }
      renderTable(lastResult?.boxes || []);
    });
    row.appendChild(th);
  }
  dom.boxHead.appendChild(row);

  if (!boxes.length) {
    const empty = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = COLUMNS.length;
    cell.className = "num";
    cell.textContent = "没有检出目标";
    empty.appendChild(cell);
    dom.boxRows.appendChild(empty);
    return;
  }

  const rows = boxes.map((box, index) => ({
    ...box,
    index,
    width: box.right - box.left,
    height: box.bottom - box.top,
    area: Math.max(0, (box.right - box.left) * (box.bottom - box.top)),
  }));

  const sortable = sortKey === "label_name"
    ? (a, b) => String(a[sortKey]).localeCompare(String(b[sortKey]), "zh")
    : (a, b) => Number(a[sortKey]) - Number(b[sortKey]);
  rows.sort((a, b) => (sortAsc ? sortable(a, b) : -sortable(a, b)));

  const median = medianConfidence(boxes);
  for (const item of rows) {
    const tr = document.createElement("tr");
    if (item.index === highlightIndex) tr.classList.add("is-active");
    if (boxes.length > 2 && item.confidence < median) tr.classList.add("is-low");

    for (const col of COLUMNS) {
      const td = document.createElement("td");
      td.className = col.align;
      if (col.key === "index") td.textContent = String(item.index + 1);
      else if (col.key === "label_name") {
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colorCss(colorOf(lastResult?.class_colors, item.class_label));
        td.appendChild(swatch);
        td.appendChild(document.createTextNode(item.label_name || `cls${item.class_label}`));
      } else if (col.key === "confidence") td.textContent = item.confidence.toFixed(4);
      else if (col.key === "area") td.textContent = Math.round(item.area).toLocaleString();
      else td.textContent = Number(item[col.key]).toFixed(1);
      tr.appendChild(td);
    }

    // 行与框联动：点行高亮对应框，再点一次取消
    tr.addEventListener("click", () => {
      highlightIndex = highlightIndex === item.index ? -1 : item.index;
      for (const other of dom.boxRows.children) other.classList.remove("is-active");
      if (highlightIndex >= 0) tr.classList.add("is-active");
      highlightBoxes(highlightIndex);
    });

    dom.boxRows.appendChild(tr);
  }
}

function medianConfidence(boxes) {
  if (!boxes.length) return 0;
  const sorted = [...boxes].map((b) => b.confidence).sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2;
}

function highlightBoxes(index) {
  setOverlayHighlight(overlaySvg, index);
}

// ---------------------------------------------------------------------------
// 置信度分布
// ---------------------------------------------------------------------------

function renderHistogram(boxes) {
  dom.confHist.replaceChildren();
  if (!boxes.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "没有检出目标";
    dom.confHist.appendChild(empty);
    return;
  }

  // 10 档、每档 0.1。用固定边界而不是自适应：不同次推理之间形状可比。
  const bins = new Array(10).fill(0);
  for (const box of boxes) {
    const idx = Math.min(9, Math.max(0, Math.floor(box.confidence * 10)));
    bins[idx] += 1;
  }
  const peak = Math.max(...bins, 1);

  bins.forEach((count, index) => {
    const row = document.createElement("div");
    row.className = "hist-row";

    const label = document.createElement("span");
    label.className = "hist-label";
    label.textContent = (index / 10).toFixed(1);

    const track = document.createElement("div");
    track.className = "hist-track";
    const bar = document.createElement("div");
    bar.className = "hist-bar";
    bar.style.width = `${(count / peak) * 100}%`;
    // 低分段用警示色，一眼看出"是不是靠低分框凑数"
    if (index < 3) bar.classList.add("is-low");
    track.appendChild(bar);

    const value = document.createElement("span");
    value.className = "hist-value";
    value.textContent = String(count);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    dom.confHist.appendChild(row);
  });

  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = `共 ${boxes.length} 个框，中位置信度 ${medianConfidence(boxes).toFixed(3)}`;
  dom.confHist.appendChild(note);
}

// ---------------------------------------------------------------------------
// 叠加层
// ---------------------------------------------------------------------------

function renderOcr(result) {
  if (sourceObjectUrl) viewers.viewer("result").setImage(sourceObjectUrl);
  viewers.setAvailable("result", true);
  viewers.setAvailable("heatmap", false);
  viewers.setAvailable("mask", false);
  setMetric("后端", result.backend || "ocr_infer.dll");
  setMetric("设备", `${result.device || ""} / ${result.run_mode || ""}`);
  setMetric("行数", result.count ?? (result.lines || []).length);
  setMetric("总耗时", `${Number(result.timing?.total_ms || 0).toFixed(0)} ms`);
  highlightIndex = -1;
  rebuildOverlay();
}

function renderOcrHead() {
  dom.boxHead.replaceChildren();
  const row = document.createElement("tr");
  for (const label of ["#", "文字", "置信度", "角度"]) {
    const th = document.createElement("th");
    th.textContent = label;
    row.appendChild(th);
  }
  dom.boxHead.appendChild(row);
}

function pickOcrLine(index) {
  highlightIndex = highlightIndex === index ? -1 : index;
  highlightOcr(overlaySvg, highlightIndex);
  renderOcrTable(dom.boxRows, lastResult?.lines || [], highlightIndex, pickOcrLine);
}

function rebuildOverlay() {
  if (lastKind === "ocr" && lastResult) {
    const viewer = viewers.viewer("result");
    if (!viewer) return;
    overlaySvg = buildOcrOverlay({
      lines: lastResult.lines || [],
      width: lastResult.src_w || lastResult.image_width || 1,
      height: lastResult.src_h || lastResult.image_height || 1,
      showBoxes: document.getElementById("optShowBoxes")?.checked !== false,
      showLabels: document.getElementById("optShowLabels")?.checked !== false,
      highlight: highlightIndex,
      onPick: pickOcrLine,
    });
    applyOverlayScale(overlaySvg, overlayScale);
    viewer.setOverlay(overlaySvg);
    return;
  }
  if (lastKind !== "yolo" || !lastResult) {
    const viewer = viewers.viewer("result");
    if (viewer) viewer.setOverlay(null);
    overlaySvg = null;
    return;
  }
  const boxes = lastResult.boxes || [];
  const viewer = viewers.viewer("result");
  if (!viewer) return;

  overlaySvg = buildOverlay({
    boxes,
    width: lastResult.image_width,
    height: lastResult.image_height,
    classColors: lastResult.class_colors,
    showBoxes: document.getElementById("optShowBoxes").checked,
    showLabels: document.getElementById("optShowLabels").checked,
    fadeLowConf: document.getElementById("optFadeLow").checked,
    highlight: highlightIndex,
  });
  applyOverlayScale(overlaySvg, overlayScale);
  viewer.setOverlay(overlaySvg);
}

function applyOverlayOpacity() {
  const percent = Number(dom.overlayOpacity.value);
  if (lastKind !== "anomalib") {
    // YOLO 的"检出结果"是后端烧录的标注图，压淡它只会让框变糊；
    // 这个控件只服务于异常叠加图
    viewers.viewer("result").setImageOpacity?.(1);
    return;
  }
  const viewer = viewers.viewer("result");
  if (!viewer || !viewer.setImageOpacity) return;
  viewer.setImageOpacity(Number.isFinite(percent) ? percent / 100 : 1);
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

function exportJson() {
  if (!lastResult) {
    setHint("还没有结果可导出");
    return;
  }
  // 导出与屏幕一致：图内联为 base64，明细直接取 boxes，不做任何裁剪
  const payload = {
    exported_at: new Date().toISOString(),
    kind: lastKind,
    backend: lastResult.backend,
    image: {
      width: lastResult.image_width,
      height: lastResult.image_height,
    },
    timing: lastResult.timing,
    count: lastResult.count ?? (lastResult.boxes || lastResult.lines || []).length,
    lines: lastResult.lines || [],
    class_names: lastResult.class_names || [],
    boxes: lastResult.boxes || [],
    thresholds: lastResult.thresholds || null,
    annotated_png: lastResult.annotated_png || null,
    heatmap_png: lastResult.heatmap_png || null,
  };
  download(`infer_${stamp()}.json`, JSON.stringify(payload, null, 2), "application/json");
}

function copyOcrText() {
  const text = ocrPlainText(lastResult?.lines || []);
  if (!text) {
    setHint("没有可复制的文字");
    return;
  }
  navigator.clipboard?.writeText(text).then(
    () => setHint("已复制全部文字"),
    () => setHint("复制失败，请改用导出 TXT"),
  );
}

function exportTxt() {
  if (lastKind !== "ocr" || !lastResult) {
    setHint("TXT 导出仅适用于 OCR 结果");
    return;
  }
  download(`ocr_${stamp()}.txt`, ocrPlainText(lastResult.lines || []), "text/plain");
}

function exportCsv() {
  if (lastKind === "ocr" && lastResult) {
    download(`ocr_${stamp()}.csv`, ocrCsv(lastResult.lines || []), "text/csv");
    return;
  }
  if (!lastResult || lastKind !== "yolo") {
    setHint("CSV 导出仅适用于检出结果");
    return;
  }
  const boxes = lastResult.boxes || [];
  const header = ["index", "class_label", "label_name", "confidence", "left", "top", "right", "bottom", "width", "height", "area"];
  const lines = [header.join(",")];
  boxes.forEach((box, index) => {
    const width = box.right - box.left;
    const height = box.bottom - box.top;
    const cells = [
      index + 1,
      box.class_label,
      csvCell(box.label_name || `cls${box.class_label}`),
      box.confidence.toFixed(6),
      box.left.toFixed(2),
      box.top.toFixed(2),
      box.right.toFixed(2),
      box.bottom.toFixed(2),
      width.toFixed(2),
      height.toFixed(2),
      Math.round(Math.max(0, width * height)),
    ];
    lines.push(cells.join(","));
  });
  // 前缀 BOM：Excel 打开中文 CSV 不加 BOM 会乱码
  download(`infer_${stamp()}.csv`, `\ufeff${lines.join("\r\n")}`, "text/csv");
}

function csvCell(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function download(filename, content, mime) {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // 立刻释放会让部分浏览器来不及下载，延后一拍即可
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function stamp() {
  const now = new Date();
  const pad = (v) => String(v).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

function value(id) {
  const el = document.getElementById(id);
  return el ? String(el.value).trim() : "";
}

function setValue(id, next) {
  const el = document.getElementById(id);
  if (el) el.value = next;
}

function fileName(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function colorOf(classColors, classLabel) {
  if (!classColors) return null;
  return classColors[String(classLabel)] || classColors[classLabel] || null;
}

function colorCss(rgb) {
  return rgb ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` : "var(--text-dim)";
}

function setMetric(name, value) {
  const chip = document.createElement("div");
  chip.className = "metric";
  chip.innerHTML = `${escapeHtml(name)} <b>${escapeHtml(String(value))}</b>`;
  dom.metrics.appendChild(chip);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function setBadge(text, cls = "") {
  dom.badge.textContent = text;
  dom.badge.className = `badge ${cls}`.trim();
}

function setHint(text) {
  dom.hint.textContent = text;
}
