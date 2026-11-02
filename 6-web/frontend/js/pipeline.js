// 全流程串联面板：训练 → 导出 → 校验转置 → 编译 → 补标签 → 引擎精度验证。
//
// 设计取舍：
// - 参数直接复用「训练参数」表单，只额外暴露串联独有的三项（精度、跳过训练、是否验证）。
//   否则用户要在两块表单里各填一遍 dataset / epochs，改了一处忘了另一处。
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
const ARTIFACT_LABELS = [
  ["run_dir", "训练目录"],
  ["pt", "最佳权重 best.pt"],
  ["onnx", "ONNX（Ultralytics 用）"],
  ["transd_onnx", "转置 ONNX（TensorRT 用）"],
  ["engine", "TensorRT 引擎"],
  ["labels", "标签文件 labels_*.json"],
  ["val", "验证报告 val_report.json"],
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
    precision: document.getElementById("pipePrecision"),
    split: document.getElementById("pipeSplit"),
    runVal: document.getElementById("pipeRunVal"),
    skipTrain: document.getElementById("pipeSkipTrain"),
    weightsBox: document.getElementById("pipeWeightsBox"),
    weights: document.getElementById("pipeWeights"),
    btnPickBest: document.getElementById("btnPipePickBest"),
    btnStart: document.getElementById("btnPipeStart"),
    btnStop: document.getElementById("btnPipeStop"),
    hint: document.getElementById("pipeHint"),
    stages: document.getElementById("pipeStages"),
    artifactsBox: document.getElementById("pipeArtifactsBox"),
    artifactRows: document.getElementById("pipeArtifactRows"),
    metrics: document.getElementById("pipeMetrics"),
    historyRows: document.getElementById("pipeHistoryRows"),
    log: document.getElementById("pipeLog"),
    logTitle: document.querySelector("#pipeLog .pipe-log-head span"),
    logPre: document.getElementById("pipeLogPre"),
  };
  if (!dom.btnStart) return;

  dom.btnStart.addEventListener("click", onStart);
  dom.btnStop.addEventListener("click", onStop);
  dom.skipTrain.addEventListener("change", syncSkipTrain);
  dom.btnPickBest.addEventListener("click", pickBestWeights);
  document
    .getElementById("btnPipeLogClose")
    ?.addEventListener("click", () => {
      dom.log.hidden = true;
    });

  syncSkipTrain();
  loadHistory({ attachRunning: true });
}

// ---------------------------------------------------------------------------
// 表单 → 请求
// ---------------------------------------------------------------------------
function value(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function syncSkipTrain() {
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
    skip_train: dom.skipTrain.checked,
    weights: dom.weights.value.trim(),
    run_val: dom.runVal.checked,
    val_split: dom.split.value,
  };
}

async function onStart() {
  const kind = value("trainKind");
  if (kind !== "yolo") {
    setHint("全流程串联目前只支持 YOLO；Anomalib 请用上面的训练表单单独训练", "bad");
    return;
  }
  const payload = buildPayload();
  if (!payload.data) {
    setHint("请先在上面的「训练参数」里填 dataset.yaml", "bad");
    return;
  }
  if (payload.skip_train && !payload.weights) {
    setHint("勾了「跳过训练」就必须给出已有 .pt 权重", "bad");
    return;
  }

  dom.btnStart.disabled = true;
  setHint("正在提交…");
  try {
    const record = await postJSON("/api/pipeline/yolo", payload);
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
        setValue("yoloEngine", path);
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
  if (!metrics) return;

  const items = [
    ["mAP50", fmtNum(metrics.map50, 4)],
    ["mAP50-95", fmtNum(metrics.map5095, 4)],
    ["精确率 P", fmtNum(metrics.precision, 3)],
    ["召回率 R", fmtNum(metrics.recall, 3)],
    ["评估图片", metrics.num_images ?? "—"],
    ["GT 实例", metrics.num_gts ?? "—"],
    ["conf 阈值", fmtNum(metrics.conf, 2)],
  ];
  items.forEach(([label, text]) => {
    const box = document.createElement("div");
    box.className = "stat";
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
