// 全流程串联面板：训练 → 导出 → 校验 → 编译 → 补标签 → 引擎精度验证。
//
// 设计取舍：
// - 参数直接复用「训练参数」表单（YOLO），只额外暴露串联独有的几项（精度、跳过训练、是否验证）。
//   否则用户要在两块表单里各填一遍 dataset / epochs，改了一处忘了另一处。
// - 异常检测的目录结构与 YOLO 完全不同（Folder 数据集、metadata.json 阈值），
//   所以它有自己的一块参数区；但阶段列表、日志、产物表、历史、停止逻辑两边共用。
// - 轮询而不是 WebSocket：串联本身不产生日志，真正的日志在每一段的 run 里，
//   状态是低频变化的（一段几十分钟），1 秒一次的 HTTP 比多开一条 WS 简单得多。
// - 刷新页面后仍能看到上一次跑到哪：启动时拉一次 /api/pipeline，有进行中的就接着盯。

import { getJSON, postJSON } from "./api.js";
import { focusRun } from "./train.js";

/// 阶段状态 → 展示文案。与后端 pipeline.py 的常量保持一致。
const STAGE_TEXT = {
  pending: "等待",
  running: "进行中",
  success: "完成",
  failed: "失败",
  cancelled: "已停止",
  skipped: "跳过",
};

/// 整条链状态 → 徽标文案与配色。
const CHAIN_BADGE = {
  pending: ["待启动", ""],
  running: ["进行中", "run"],
  success: ["已完成", "done"],
  failed: ["失败", "fail"],
  cancelled: ["已停止", "warn"],
};

/// 产物键 → 展示名。顺序即表格顺序：从训练产物一路排到引擎。
/// 两种任务共用一个列表：某一行在本次产物里不存在就不显示，不会出现空行。
const ARTIFACT_LABELS = [
  ["run_dir", "训练目录"],
  ["pt", "最佳权重 best.pt"],
  ["onnx", "ONNX（Ultralytics 用）"],
  ["transd_onnx", "转置 ONNX（TensorRT 用）"],
  ["engine", "TensorRT 引擎"],
  ["metadata", "metadata.json（阈值）"],
  ["ckpt", "检查点 model.ckpt"],
  ["labels", "标签文件 labels_*.json"],
  ["val", "验证报告"],
];

const state = {
  id: null,
  record: null,
  timer: null,
};

let dom = {};

export function initPipeline() {
  dom = {
    badge: document.getElementById("pipeBadge"),
    kind: document.getElementById("pipeKind"),
    yoloOptions: document.getElementById("pipeYoloOptions"),
    anomalyOptions: document.getElementById("pipeAnomalyOptions"),
    precision: document.getElementById("pipePrecision"),
    split: document.getElementById("pipeSplit"),
    runVal: document.getElementById("pipeRunVal"),
    skipTrain: document.getElementById("pipeSkipTrain"),
    weightsBox: document.getElementById("pipeWeightsBox"),
    weights: document.getElementById("pipeWeights"),
    btnPickBest: document.getElementById("btnPipePickBest"),
    anomalyRunVal: document.getElementById("pipeAnomalyRunVal"),
    anomalySkipTrain: document.getElementById("pipeAnomalySkipTrain"),
    anomalySeedBox: document.getElementById("pipeAnomalySeedBox"),
    anomalyArtifacts: document.getElementById("pipeAnomalyArtifacts"),
    anomalyDirsBox: document.getElementById("pipeAnomalyDirsBox"),
    btnStart: document.getElementById("btnPipeStart"),
    btnStop: document.getElementById("btnPipeStop"),
    hint: document.getElementById("pipeHint"),
    stages: document.getElementById("pipeStages"),
    artifactsBox: document.getElementById("pipeArtifactsBox"),
    artifactRows: document.getElementById("pipeArtifactRows"),
    metrics: document.getElementById("pipeMetrics"),
    valDetail: document.getElementById("pipeValDetail"),
    historyRows: document.getElementById("pipeHistoryRows"),
    log: document.getElementById("pipeLog"),
    logTitle: document.querySelector("#pipeLog .pipe-log-head span"),
    logPre: document.getElementById("pipeLogPre"),
  };
  if (!dom.btnStart) return;

  dom.btnStart.addEventListener("click", onStart);
  dom.btnStop.addEventListener("click", onStop);
  dom.kind.addEventListener("change", syncKind);
  document.getElementById("yoloTask")?.addEventListener("change", syncClassifyNote);
  syncClassifyNote();
  dom.precision.addEventListener("change", syncCalib);
  dom.skipTrain.addEventListener("change", syncSkipTrain);
  dom.anomalySkipTrain.addEventListener("change", syncSkipTrain);
  dom.btnPickBest.addEventListener("click", pickBestWeights);
  document
    .getElementById("btnPipeLogClose")
    ?.addEventListener("click", () => {
      dom.log.hidden = true;
    });

  syncKind();
  loadHistory({ attachRunning: true });
}

// ---------------------------------------------------------------------------
// 表单 → 请求
// ---------------------------------------------------------------------------
function value(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

/// 任务类型切换：只换参数区和按钮文案，阶段列表/产物/历史沿用同一套渲染。
function syncClassifyNote() {
  const note = document.getElementById("pipeClassifyNote");
  if (!note) return;
  const classify = (document.getElementById("yoloTask")?.value || "") === "classify";
  const yolo = (document.getElementById("pipeKind")?.value || "yolo") === "yolo";
  note.hidden = !(classify && yolo);
}

function syncKind() {
  const kind = dom.kind.value;
  dom.yoloOptions.hidden = kind !== "yolo";
  dom.anomalyOptions.hidden = kind !== "anomalib";
  dom.btnStart.textContent = kind === "anomalib" ? "开始全流程（异常检测）" : "开始全流程";
  setHint("");
  syncSkipTrain();
  syncCalib();
}

function syncCalib() {
  const box = document.getElementById("pipeCalib");
  if (!box) return;
  box.classList.toggle("hidden", dom.kind.value !== "yolo" || dom.precision.value !== "int8");
}

function syncSkipTrain() {
  if (dom.kind.value === "anomalib") {
    const skip = dom.anomalySkipTrain.checked;
    dom.anomalySeedBox.hidden = !skip;
    return;
  }
  const skip = dom.skipTrain.checked;
  dom.weightsBox.hidden = !skip;
  if (skip && !dom.weights.value) pickBestWeights();
}

/// 用训练表单里已经算出来的实验目录补权重路径：project/name/weights/best.pt
function pickBestWeights() {
  const project = value("project") || "runs/trtforge";
  const name = value("runName");
  if (!name) {
    setHint("先填实验名称 name，才能推出 best.pt 路径", "warn");
    return;
  }
  // 后端把相对 project 解析到 launcher 目录，这里只拼相对片段并提示用户用「浏览」确认
  const sep = project.includes("\\") ? "\\" : "/";
  dom.weights.value = [project.replace(/[\\/]+$/, ""), name, "weights", "best.pt"].join(sep);
  setHint("已填入相对路径；若 launcher 不在当前目录下，请用「浏览」选绝对路径", "warn");
}

function buildPayload() {
  return {
    data: value("yoloData"),
    model: value("yoloModel") || "yolov8n.pt",
    task: value("yoloTask") || "detect",
    epochs: Number(value("yoloEpochs")) || 100,
    imgsz: Number(value("yoloImgsz")) || 640,
    batch: value("yoloBatch") || "16",
    device: value("yoloDevice") || "0",
    project: value("project") || "runs/trtforge",
    name: value("runName"),
    python: value("pythonPath") || null,
    precision: dom.precision.value,
    calib_dir: value("pipeCalibDir"),
    calib_count: Number(value("pipeCalibCount")) || 100,
    skip_train: dom.skipTrain.checked,
    weights: dom.weights.value.trim(),
    run_val: dom.runVal.checked,
    val_split: dom.split.value,
  };
}

function buildAnomalyPayload() {
  return {
    model: value("pipeAnomalyModel") || "patchcore",
    root: value("pipeAnomalyRoot"),
    normal_dir: value("pipeAnomalyNormal") || "train/good",
    abnormal_dir: value("pipeAnomalyAbnormal") || "test/abnormal",
    normal_test_dir: value("pipeAnomalyNormalTest") || "test/good",
    input_size: Number(value("pipeAnomalySize")) || 256,
    max_epochs: value("pipeAnomalyEpochs") || "auto",
    project: value("project") || "results",
    python: value("pythonPath") || null,
    precision: dom.precision.value,
    skip_train: dom.anomalySkipTrain.checked,
    artifacts_dir: dom.anomalyArtifacts.value.trim(),
    run_val: dom.anomalyRunVal.checked,
  };
}

async function onStart() {
  const kind = dom.kind.value;
  const anomaly = kind === "anomalib";
  if (anomaly && dom.precision.value === "int8") {
    setHint("异常检测还不能用 INT8，请改回 fp16 或 fp32", "warn");
    return;
  }
  const payload = anomaly ? buildAnomalyPayload() : buildPayload();

  if (anomaly) {
    if (payload.skip_train && !payload.artifacts_dir) {
      setHint("勾了「跳过训练」就必须给出已有产物目录（含 metadata.json 与 .onnx）", "bad");
      return;
    }
    if (!payload.skip_train && !payload.root) {
      setHint("请填写异常检测的数据集根目录（Folder 结构，含 train/good 与 test/abnormal）", "bad");
      return;
    }
  } else {
    if (!payload.data) {
      setHint("请先在上面的「训练参数」里填 dataset.yaml", "bad");
      return;
    }
    if (payload.skip_train && !payload.weights) {
      setHint("勾了「跳过训练」就必须给出已有 .pt 权重", "bad");
      return;
    }
  }

  dom.btnStart.disabled = true;
  setHint("正在提交…");
  try {
    const url = anomaly ? "/api/pipeline/anomalib" : "/api/pipeline/yolo";
    const record = await postJSON(url, payload);
    state.id = record.id;
    state.record = record;
    render(record);
    startPolling();
  } catch (error) {
    setHint(error.message, "bad");
    dom.btnStart.disabled = false;
  }
}

async function onStop() {
  if (!state.id) return;
  dom.btnStop.disabled = true;
  setHint("正在停止当前阶段…", "warn");
  try {
    const data = await postJSON(`/api/pipeline/${state.id}/stop`, {});
    if (data.pipeline) render(data.pipeline);
  } catch (error) {
    setHint(error.message, "bad");
    dom.btnStop.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 轮询
// ---------------------------------------------------------------------------
function startPolling() {
  stopPolling();
  state.timer = setInterval(poll, 1200);
}

function stopPolling() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function isTerminal(record) {
  if (!record) return true;
  if (["success", "failed", "cancelled"].includes(record.status)) return true;
  // live=false 且状态仍是 running：说明服务重启过，这条只是磁盘上的历史，
  // 继续轮询永远不会变，必须停。
  return record.live === false;
}

async function poll() {
  if (!state.id) return;
  try {
    const record = await getJSON(`/api/pipeline/${state.id}`);
    render(record);
    if (isTerminal(record)) {
      stopPolling();
      loadHistory();
    }
  } catch (error) {
    // 后端重启会让内存里的记录消失，这时停掉轮询并保留最后一次画面
    stopPolling();
    setHint(`状态读取失败：${error.message}`, "bad");
    dom.btnStart.disabled = false;
    dom.btnStop.disabled = true;
  }
}

// ---------------------------------------------------------------------------
// 渲染
// ---------------------------------------------------------------------------
function render(record) {
  if (!record) return;
  state.record = record;
  state.id = record.id;

  const [text, cls] = CHAIN_BADGE[record.status] || [record.status, ""];
  dom.badge.textContent = text;
  dom.badge.className = `badge ${cls}`.trim();

  const running = record.status === "running" || record.status === "pending";
  // live=false 的历史记录没有可停的东西：按钮给出去只会换来 409
  dom.btnStart.disabled = running && record.live !== false;
  dom.btnStop.disabled = !running || record.live === false;

  if (record.live === false && running) {
    setHint("这条记录来自上一次服务进程，任务已中断；产物文件仍在磁盘上", "warn");
  } else if (isTerminal(record)) {
    const seconds = record.elapsed ? `，用时 ${fmtDuration(record.elapsed)}` : "";
    if (record.status === "success") {
      setHint(`全流程完成${seconds}`, "ok");
    } else if (record.status === "cancelled") {
      setHint(`已停止${seconds}；已完成阶段的产物保留`, "warn");
    } else {
      setHint(`失败：${record.error || "未知原因"}${seconds}`, "bad");
    }
  } else if (record.stage_index >= 0 && record.stages[record.stage_index]) {
    setHint(`正在执行：${record.stages[record.stage_index].label}`, "");
  }

  renderStages(record);
  renderArtifacts(record);
  renderMetrics(record);
}

/// 看某一段的日志。
///
/// 两条路，按"这段还在不在跑"分流：
/// - 活着：切训练页的日志面板，能实时流（focusRun）；
/// - 已结束/服务重启过：run 已经不在内存里，实时流没得连，改读 `runtime/logs/{run_id}.log`。
///   排障要看的恰恰是这种已经结束的日志，所以不能只做前者。
async function openStageLog(stage, record) {
  const live = record && record.live !== false;
  if (live && focusRun(stage.run_id)) {
    document.getElementById("logBox")?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  showStageLog(`正在读取 ${stage.label || stage.name} 的日志…`, "");
  try {
    const text = await getJSON(
      `/api/pipeline/${record.id}/log?stage=${encodeURIComponent(stage.name)}&lines=400`,
    );
    showStageLog(text || "（日志是空的）", "", stage.label || stage.name);
  } catch (error) {
    showStageLog(`读取日志失败：${error.message}`, "bad", stage.label || stage.name);
  }
}

function showStageLog(text, cls, title = "阶段日志") {
  dom.log.hidden = false;
  if (dom.logTitle) dom.logTitle.textContent = title;
  dom.logPre.textContent = text;
  dom.logPre.className = `pipe-log-pre ${cls}`.trim();
  dom.log.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderStages(record) {  dom.stages.hidden = false;
  dom.stages.innerHTML = "";
  record.stages.forEach((stage, index) => {
    const li = document.createElement("li");
    li.className = `pipe-stage ${stage.status}`;

    const step = document.createElement("span");
    step.className = "pipe-step";
    step.textContent = String(index + 1).padStart(2, "0");

    const name = document.createElement("span");
    name.className = "pipe-name";
    name.textContent = stage.label || stage.name;

    const status = document.createElement("span");
    status.className = "pipe-status";
    status.textContent = STAGE_TEXT[stage.status] || stage.status;

    const time = document.createElement("span");
    time.className = "pipe-time";
    time.textContent = stage.elapsed != null ? fmtDuration(stage.elapsed) : "";

    // 每一段都有独立 run_id。活着的 run 直接切训练页日志面板（实时流）；
    // 历史记录没有 run 了（服务重启后内存里就没了），改读 runtime/logs 的日志文件。
    const logBtn = document.createElement("button");
    logBtn.type = "button";
    logBtn.className = "mini";
    logBtn.textContent = "日志";
    logBtn.hidden = !stage.run_id;
    logBtn.title = state.record?.live === false
      ? "查看已结束阶段的日志文件"
      : "在训练日志面板里实时查看";
    logBtn.addEventListener("click", () => openStageLog(stage, state.record));

    li.append(step, name, status, time, logBtn);
    if (stage.error) {
      const err = document.createElement("p");
      err.className = "pipe-error";
      err.textContent = stage.error;
      li.appendChild(err);
    }
    dom.stages.appendChild(li);
  });
}

function renderArtifacts(record) {
  const artifacts = record.artifacts || {};
  const rows = [];
  const seen = new Set();
  ARTIFACT_LABELS.forEach(([key, label]) => {
    if (artifacts[key]) {
      rows.push([key, label, artifacts[key]]);
      seen.add(key);
    }
  });
  Object.entries(artifacts).forEach(([key, path]) => {
    if (!seen.has(key) && typeof path === "string" && path) rows.push([key, key, path]);
  });

  dom.artifactRows.innerHTML = "";
  dom.artifactsBox.hidden = rows.length === 0;
  rows.forEach(([key, label, path]) => {
    const tr = document.createElement("tr");
    const tdLabel = document.createElement("td");
    tdLabel.textContent = label;
    const tdPath = document.createElement("td");
    tdPath.className = "mono";
    tdPath.textContent = path;

    const tdAct = document.createElement("td");
    if (key === "engine") {
      tdAct.appendChild(actionButton("去推理引擎", () => {
        // 异常检测的推理必须同时带 metadata.json（阈值在里面），
        // 只填引擎会让 DLL 在 ai_create 阶段直接失败。
        if (record.kind === "anomalib") {
          setValue("inferKind", "anomalib");
          setValue("anomEngine", path);
          const meta = (record.artifacts || {}).metadata;
          if (meta) setValue("anomMetadata", meta);
        } else {
          setValue("yoloEngine", path);
        }
        document.querySelector('.tab[data-tab="infer"]')?.click();
      }));
    } else if (key === "metadata" && record.kind === "anomalib") {
      tdAct.appendChild(actionButton("带到推理页", () => {
        setValue("inferKind", "anomalib");
        setValue("anomMetadata", path);
        const engine = (record.artifacts || {}).engine;
        if (engine) setValue("anomEngine", engine);
        document.querySelector('.tab[data-tab="infer"]')?.click();
      }));
    } else if (key === "labels") {
      tdAct.appendChild(actionButton("带到推理页", () => {
        setValue("yoloLabels", path);
        document.querySelector('.tab[data-tab="infer"]')?.click();
      }));
    }
    tdAct.appendChild(actionButton("复制路径", () => copyPath(path)));
    tr.append(tdLabel, tdPath, tdAct);
    dom.artifactRows.appendChild(tr);
  });
}

function renderMetrics(record) {
  const metrics = record.metrics;
  dom.metrics.innerHTML = "";
  dom.metrics.hidden = !metrics;
  if (dom.valDetail) {
    dom.valDetail.innerHTML = "";
    dom.valDetail.hidden = true;
  }
  if (!metrics) return;

  // 两条链的精度口径完全不同：检测是 mAP/P/R，异常检测是 AUROC/漏检/误报。
  // 用同一个渲染器 + 分支，而不是写两张卡片，是因为「有没有报告、报告在哪、
  // 怎么看历史」这些逻辑一模一样，复制一份只会让两边逐渐长得不一样。
  const items = metrics.kind === "anomaly"
    ? anomalyMetricItems(metrics).concat(pixelMetricItems(metrics))
    : yoloMetricItems(metrics);
  items.forEach(([label, text, flag]) => {
    const box = document.createElement("div");
    box.className = flag === "warn" ? "stat is-warn" : "stat";
    const span = document.createElement("span");
    span.className = "stat-label";
    span.textContent = label;
    const b = document.createElement("b");
    b.textContent = text;
    box.append(span, b);
    dom.metrics.appendChild(box);
  });

  // 类别太多时单独列出来会撑爆卡片，只在控制台留一份完整的
  if (Array.isArray(metrics.per_class) && metrics.per_class.length) {
    console.table(metrics.per_class);
  }
  renderValDetail(metrics);
  // 漏检-过杀曲线只有检测侧有：异常检测是单阈值判定，没有 conf 可扫
  const opSection = metrics.kind === "anomaly" ? null : operatingSection(metrics);
  if (opSection && dom.valDetail) {
    // 重复渲染同一份报告时不叠第二份曲线
    dom.valDetail.querySelector(".op-section")?.remove();
    dom.valDetail.appendChild(opSection);
    dom.valDetail.hidden = false;
  }
}

const KIND_TEXT = { miss: "漏检", extra: "过检", wrong: "判错" };

function classLabel(metrics, cls) {
  if (cls === null || cls === undefined || cls === "") return "—";
  const names = metrics.classes || [];
  const index = Number(cls);
  if (Number.isInteger(index) && names[index]) return names[index];
  return `类别 ${cls}`;
}

function renderValDetail(metrics) {
  const found = metrics.diagnosis;
  const host = dom.valDetail;
  if (!host || !found || !Array.isArray(found.matrix)) return;

  const names = (metrics.classes || []).slice();
  const background = found.background ?? names.length;
  while (names.length < background) names.push(`类别 ${names.length}`);
  const headers = names.concat(["没检出来"]);
  const rowNames = names.concat(["多检出来"]);

  const note = document.createElement("p");
  note.className = "pipe-val-note";
  note.textContent = "行是标注，列是预测。斜线是判对，最后一列是漏检，最后一行是过检，其他格子是判错。"
    + " 精确率 = 判对 / (判对 + 判错 + 过检)，召回率 = 判对 / (判对 + 判错 + 漏检)。";

  const table = document.createElement("table");
  table.className = "data-table conf-matrix";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["标注 \\ 预测"].concat(headers).forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  found.matrix.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = rowNames[rowIndex] || "";
    tr.appendChild(th);
    row.forEach((value, colIndex) => {
      const td = document.createElement("td");
      td.textContent = String(value);
      if (rowIndex === colIndex && rowIndex < background) td.className = "is-hit";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);

  const listTitle = document.createElement("p");
  listTitle.className = "pipe-val-note";
  const total = found.errors_total ?? (found.errors || []).length;
  const shown = (found.errors || []).slice(0, 100);
  listTitle.textContent = total
    ? `问题框 ${total} 条，下面显示前 ${shown.length} 条。`
    : "没有漏检、过检或判错。";
  if (found.images_csv) listTitle.textContent += ` 完整清单：${found.images_csv}`;

  host.append(note, table, listTitle);
  if (shown.length) {
    const list = document.createElement("table");
    list.className = "data-table";
    const listHead = document.createElement("tr");
    ["文件", "问题", "真实类别", "预测类别", "置信度"].forEach((text) => {
      const th = document.createElement("th");
      th.textContent = text;
      listHead.appendChild(th);
    });
    const listBody = document.createElement("tbody");
    shown.forEach((row) => {
      const tr = document.createElement("tr");
      [row.file || "—", KIND_TEXT[row.kind] || row.kind || "—",
        classLabel(metrics, row.gt_class), classLabel(metrics, row.pred_class),
        row.score == null ? "—" : String(row.score)].forEach((text) => {
        const td = document.createElement("td");
        td.textContent = text;
        tr.appendChild(td);
      });
      listBody.appendChild(tr);
    });
    list.append(listHead, listBody);
    host.appendChild(list);
  }
  host.hidden = false;
}

function yoloMetricItems(metrics) {
  const items = [
    ["mAP50", fmtNum(metrics.map50, 4)],
    ["mAP50-95", fmtNum(metrics.map5095, 4)],
    ["精确率 P", fmtNum(metrics.precision, 3)],
    ["召回率 R", fmtNum(metrics.recall, 3)],
    ["评估图片", metrics.num_images ?? "—"],
    ["GT 实例", metrics.num_gts ?? "—"],
    ["conf 阈值", fmtNum(metrics.conf, 2)],
  ];
  const found = metrics.diagnosis;
  if (found) {
    items.push(
      ["漏检", found.miss ?? 0],
      ["过检", found.extra ?? 0],
      ["判错", found.wrong ?? 0],
    );
  }
  // 工作点摘要：漏检率/过杀率才是产线口径，mAP 只是训练时的相对尺子
  const ops = metrics.operating_points;
  const recommended = metrics.recommended_operating_point || {};
  const point = recommended.point || (ops && ops.points ? ops.points[0] : null);
  if (point) {
    items.push(
      ["漏检率", fmtPct(point.miss_rate)],
      ["过杀", point.fp_per_image == null ? "—" : `${fmtNum(point.fp_per_image, 3)} 框/图`],
    );
    if (ops && ops.ok_images) {
      items.push([
        "良品误报",
        point.ok_fp_rate == null ? "—" : fmtPct(point.ok_fp_rate),
      ]);
      items.push(["良品图", ops.ok_images]);
    } else {
      items.push(["良品误报", "无良品图", "warn"]);
    }
  }
  return items;
}

// ---------------------------------------------------------------------------
// 漏检-过杀曲线（零依赖内联 SVG）
//
// x 轴是漏检率（0 → 1），y 轴是每图过杀框数。conf 从低到高走一遍时，
// 漏检单调上升、过杀单调下降——所以这条线本身就说明了「压漏检要多报多少」。
// 选中点之后可以写进引擎旁挂文件，推理页会默认用它。
// ---------------------------------------------------------------------------
const OP_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(OP_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function operatingChart(points) {
  const width = 420;
  const height = 180;
  const pad = 34;
  const usable = points.filter((p) => p.miss_rate != null && p.fp_per_image != null);
  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "op-chart",
    role: "img",
    "aria-label": "漏检-过杀曲线",
  });
  if (!usable.length) return svg;

  const maxFp = Math.max(0.01, ...usable.map((p) => p.fp_per_image));
  const x = (miss) => pad + miss * (width - pad - 10);
  const y = (fp) => height - pad - (fp / maxFp) * (height - pad - 12);

  // 坐标轴
  svg.appendChild(svgEl("line", { x1: pad, y1: height - pad, x2: width - 10, y2: height - pad, class: "op-axis" }));
  svg.appendChild(svgEl("line", { x1: pad, y1: 12, x2: pad, y2: height - pad, class: "op-axis" }));
  const xLabel = svgEl("text", { x: width - 10, y: height - pad + 16, class: "op-tick", "text-anchor": "end" });
  xLabel.textContent = "漏检率 →";
  svg.appendChild(xLabel);
  const yLabel = svgEl("text", { x: pad - 6, y: 16, class: "op-tick", "text-anchor": "end" });
  yLabel.textContent = "过杀/图";
  svg.appendChild(yLabel);

  // 曲线：按 conf 升序画，漏检从小到大
  const sorted = usable.slice().sort((a, b) => a.miss_rate - b.miss_rate);
  const path = sorted
    .map((p, index) => `${index === 0 ? "M" : "L"}${x(p.miss_rate).toFixed(1)},${y(p.fp_per_image).toFixed(1)}`)
    .join(" ");
  svg.appendChild(svgEl("path", { d: path, class: "op-line" }));
  sorted.forEach((p) => {
    const dot = svgEl("circle", { cx: x(p.miss_rate).toFixed(1), cy: y(p.fp_per_image).toFixed(1), r: 2.2, class: "op-dot" });
    const title = svgEl("title");
    title.textContent = `conf ${p.conf} · 漏检 ${fmtPct(p.miss_rate)} · 过杀 ${p.fp_per_image}/图`;
    dot.appendChild(title);
    svg.appendChild(dot);
  });
  return svg;
}

function operatingSection(metrics) {
  const ops = metrics.operating_points;
  if (!ops || !Array.isArray(ops.points) || !ops.points.length) return null;
  const host = document.createElement("div");
  host.className = "op-section";

  const title = document.createElement("p");
  title.className = "pipe-val-note";
  title.textContent =
    "漏检-过杀曲线：每个点是一个 conf 阈值。产线先定漏检上限，再取满足它的最高 conf（过杀最低）。";
  host.appendChild(title);

  const chartBox = document.createElement("div");
  chartBox.className = "op-chart-box";
  chartBox.appendChild(operatingChart(ops.points));
  host.appendChild(chartBox);

  // 关键工作点表：只列几个常用的漏检目标，避免 19 行数字糊在页面上
  const targets = [0.005, 0.01, 0.02, 0.05];
  const table = document.createElement("table");
  table.className = "data-table";
  const head = document.createElement("tr");
  ["目标漏检", "推荐 conf", "实际漏检", "过杀/图", "良品误报"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    head.appendChild(th);
  });
  table.appendChild(head);
  const bestByTarget = new Map();
  for (const target of targets) {
    const feasible = ops.points.filter((p) => p.miss_rate != null && p.miss_rate <= target + 1e-9);
    const pick = feasible.length ? feasible[feasible.length - 1] : null;
    bestByTarget.set(target, pick);
    const tr = document.createElement("tr");
    const cells = pick
      ? [fmtPct(target), String(pick.conf), fmtPct(pick.miss_rate), fmtNum(pick.fp_per_image, 3),
          pick.ok_fp_rate == null ? "无良品图" : fmtPct(pick.ok_fp_rate)]
      : [fmtPct(target), "无可行点", "—", "—", "—"];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  }
  host.appendChild(table);

  if (ops.missing_label_images) {
    const warn = document.createElement("p");
    warn.className = "pipe-val-note is-warn";
    warn.textContent =
      `${ops.missing_label_images} 张图没有标签文件，可能是漏标，未计入良品误报率。`;
    host.appendChild(warn);
  }

  const tools = document.createElement("div");
  tools.className = "row op-tools";
  const select = document.createElement("select");
  select.title = "选择目标漏检率";
  targets.forEach((target) => {
    const option = document.createElement("option");
    option.value = String(target);
    option.textContent = `目标漏检 ${fmtPct(target)}`;
    if (Math.abs(target - 0.01) < 1e-9) option.selected = true;
    select.appendChild(option);
  });
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "保存为引擎工作点";
  save.disabled = !metrics.engine;
  save.title = metrics.engine
    ? "写进 <engine>.meta.json，推理页选中该引擎时会预填这个 conf"
    : "报告里没有引擎路径，无法保存";
  const status = document.createElement("span");
  status.className = "hint";

  save.addEventListener("click", async () => {
    const target = Number(select.value);
    const point = bestByTarget.get(target);
    if (!point) {
      status.textContent = "该目标没有可行点，先放宽目标或补数据。";
      status.className = "hint bad";
      return;
    }
    save.disabled = true;
    try {
      await postJSON("/api/engine/operating-point", {
        engine: metrics.engine,
        conf: point.conf,
        iou: point.iou ?? ops.iou ?? 0.5,
        source: String(metrics.engine),
        target_miss_rate: target,
        metrics: {
          miss_rate: point.miss_rate,
          fp_per_image: point.fp_per_image,
          ok_fp_rate: point.ok_fp_rate,
          ok_images: ops.ok_images,
        },
      });
      status.textContent = `已保存：conf ${point.conf}（目标漏检 ${fmtPct(target)}）`;
      status.className = "hint";
    } catch (error) {
      status.textContent = `保存失败：${error.message}`;
      status.className = "hint bad";
    } finally {
      save.disabled = false;
    }
  });

  tools.append(select, save, status);
  host.appendChild(tools);
  return host;
}

function anomalyMetricItems(metrics) {
  const image = metrics.image || {};
  const at = image.at_metadata_threshold || {};
  const best = image.best_f1 || {};
  const counts = metrics.counts || {};
  const items = [
    ["图像 AUROC", image.auroc == null ? "无定义" : fmtNum(image.auroc, 4)],
    ["阈值下 F1", fmtNum(at.f1, 4)],
    ["漏检率 FNR", fmtPct(at.fnr)],
    ["误报率 FPR", fmtPct(at.fpr)],
    ["精确率 P", fmtNum(at.precision, 4)],
    ["召回率 R", fmtNum(at.recall, 4)],
    ["评估图片", metrics.num_images ?? "—"],
    ["正常 / 异常", `${counts.normal ?? "—"} / ${counts.abnormal ?? "—"}`],
    ["metadata 阈值", fmtNum(metrics.model?.image_threshold, 6)],
    ["最优阈值(F1)", fmtNum(best.threshold, 6)],
  ];
  // 判定口径不一致时，报告里的漏检率不可信；这不是"精度差"，是"数据或阈值被改过"
  if (image.decision_mismatch) {
    items.push(["⚠ 判定不一致", `${image.decision_mismatch} 张`, "warn"]);
  }
  return items;
}

/// 像素级诊断的 3 行。
/// 与图像级指标分开放：这三项回答的是「缺陷找没找准」，和「有没有判对」是两件事，
/// 混在一列里会让人把"定位糊了"误读成"模型不准"。
function pixelMetricItems(metrics) {
  const pixel = metrics.pixel || {};
  if (!pixel.available) {
    // 不给空数字：说明为什么没这个数，比摆三个 "—" 有用
    return [["像素级诊断", pixel.reason || "不可用（引擎未返回异常图）"]];
  }
  const diag = pixel.diagnostic || {};
  const items = [
    ["好图乱指像素", diag.normal_false_pixel_rate == null ? "n/a" : fmtPct(diag.normal_false_pixel_rate)],
    ["异常图定位面积", diag.abnormal_marked_rate == null ? "n/a" : fmtPct(diag.abnormal_marked_rate)],
  ];
  const undirected = Number(diag.abnormal_undirected) || 0;
  // 分数说坏、图上却一个像素都没超阈值：这是硬伤（分数与位置自相矛盾），
  // 比"定位面积偏大"严重，所以只在 > 0 时标红；0 是正常值，不给红色制造焦虑
  items.push(
    undirected > 0
      ? ["指不出位置", `${undirected} 张`, "warn"]
      : ["指不出位置", "0 张"]
  );
  return items;
}

// ---------------------------------------------------------------------------
// 历史记录
// ---------------------------------------------------------------------------
async function loadHistory({ attachRunning = false } = {}) {
  try {
    const data = await getJSON("/api/pipeline?limit=10");
    renderHistory(data.items || []);
    if (attachRunning) {
      const running = (data.items || []).find((item) => item.status === "running" || item.status === "pending");
      if (running) {
        render(running);
        startPolling();
      }
    }
  } catch (error) {
    dom.historyRows.innerHTML = "";
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = `读取历史失败：${error.message}`;
    tr.appendChild(td);
    dom.historyRows.appendChild(tr);
  }
}

function renderHistory(items) {
  dom.historyRows.innerHTML = "";
  if (!items.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "hint";
    td.textContent = "还没有串联记录";
    tr.appendChild(td);
    dom.historyRows.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    const tdTitle = document.createElement("td");
    tdTitle.textContent = item.title || item.id;

    const tdStatus = document.createElement("td");
    const [text, cls] = CHAIN_BADGE[item.status] || [item.status, ""];
    const badge = document.createElement("span");
    badge.className = `badge ${cls}`.trim();
    badge.textContent = text;
    tdStatus.appendChild(badge);

    const tdTime = document.createElement("td");
    tdTime.className = "num mono";
    tdTime.textContent = item.elapsed != null ? fmtDuration(item.elapsed) : "—";

    const tdStart = document.createElement("td");
    tdStart.className = "mono";
    tdStart.textContent = item.created_at ? fmtClock(item.created_at) : "—";

    const tdAct = document.createElement("td");
    tdAct.appendChild(actionButton("查看", async () => {
      try {
        render(await getJSON(`/api/pipeline/${item.id}`));
        if (item.status === "running" || item.status === "pending") startPolling();
      } catch (error) {
        setHint(error.message, "bad");
      }
    }));
    if (item.status === "running" || item.status === "pending") {
      tdAct.appendChild(actionButton("停止", async () => {
        try {
          const data = await postJSON(`/api/pipeline/${item.id}/stop`, {});
          if (data.pipeline) render(data.pipeline);
          stopPolling();
          loadHistory();
        } catch (error) {
          setHint(error.message, "bad");
        }
      }));
    }

    tr.append(tdTitle, tdStatus, tdTime, tdStart, tdAct);
    dom.historyRows.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
function actionButton(text, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "mini";
  btn.textContent = text;
  btn.addEventListener("click", onClick);
  return btn;
}

function setValue(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = text;
  // 推理页在 change 时会清掉上一次结果，这里补一次，避免「换了引擎还显示旧图」
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

/// 复制到剪贴板。局域网 http 页面里 navigator.clipboard 可能不可用（非安全上下文），
/// 退回到临时 textarea + execCommand，保证这个按钮永远有反应。
async function copyPath(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setHint("路径已复制", "ok");
  } catch (error) {
    setHint(`复制失败：${error.message}`, "bad");
  }
}

function setHint(text, cls = "") {
  dom.hint.textContent = text;
  dom.hint.className = `hint ${cls}`.trim();
}

function fmtNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

/// 0.125 → "12.5%"。漏检率/误报率写成小数没有尺度感，5% 的验收线看不出来。
function fmtPct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

/// 秒 → 人类可读。串联动辄一小时，只显示秒数没有尺度感。
function fmtDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  if (total < 60) return `${total} 秒`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} 分 ${total % 60} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes % 60} 分`;
}

/// epoch 秒 → 本地时间字符串
function fmtClock(seconds) {
  const date = new Date(Number(seconds) * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
