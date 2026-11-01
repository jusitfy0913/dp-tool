// 训练页：表单 → 启动任务 → 事件流渲染。
// 只做展示与参数收集，参数语义完全沿用 launcher，不在前端做业务判断。

import { postJSON, stopRun } from "./api.js";
import { connectRunEvents } from "./ws.js";

const state = {
  runId: null,
  stream: null,
  artifacts: [],
};

/// 进度与曲线状态。
///
/// 为什么前端自己算 ETA：launcher 只在每轮结束发 `epoch_summary`，不给「还要多久」。
/// 首轮包含数据集扫描（缓存都要重建），拿平均值算必然偏高，所以取**最近 3 轮耗时的中位数**。
const progress = {
  total: 0,
  epoch: 0,
  pct: 0,
  startedAt: 0, // Date.now() 毫秒
  lastEpochAt: 0, // 上一轮结束的 Date.now() 毫秒
  epochDeltas: [], // 最近若干轮的单轮耗时（秒）
  timer: null,
  rows: [], // 每轮指标，供表格重绘
  columns: [], // 动态列，顺序即显示顺序
  losses: [], // 总 loss 序列，供曲线
  lossSource: "", // "训练" 或 "验证"，用于曲线标题
  metrics: {}, // {指标名: [值...]}，供曲线
};

//: 表格只留最近这么多轮，避免长时间训练把 DOM 撑大
const MAX_TABLE_ROWS = 20;
//: 曲线最多画这么多点（与表格解耦：曲线看趋势，多留一些更平滑）
const MAX_CHART_POINTS = 300;

let dom = {};

export function initTrain() {
  dom = {
    form: document.getElementById("trainForm"),
    kind: document.getElementById("trainKind"),
    pythonPath: document.getElementById("pythonPath"),
    yoloGroup: document.getElementById("yoloGroup"),
    anomalibGroup: document.getElementById("anomalibGroup"),
    project: document.getElementById("project"),
    runName: document.getElementById("runName"),
    btnTrain: document.getElementById("btnTrain"),
    btnCheckEnv: document.getElementById("btnCheckEnv"),
    btnStop: document.getElementById("btnStop"),
    hint: document.getElementById("trainHint"),
    logBox: document.getElementById("logBox"),
    progressBar: document.getElementById("progressBar"),
    metrics: document.getElementById("metrics"),
    badge: document.getElementById("runBadge"),
    // 进度面板
    stage: document.getElementById("trainStage"),
    epochText: document.getElementById("trainEpoch"),
    pctText: document.getElementById("trainPct"),
    elapsedText: document.getElementById("trainElapsed"),
    etaText: document.getElementById("trainEta"),
    speedText: document.getElementById("trainSpeed"),
    // 参数回显 / 表格 / 曲线
    commandBox: document.getElementById("trainCommandBox"),
    command: document.getElementById("trainCommand"),
    chartRow: document.getElementById("trainCharts"),
    lossChart: document.getElementById("lossChart"),
    lossCap: document.getElementById("lossChartCap"),
    metricChart: document.getElementById("metricChart"),
    metricCap: document.getElementById("metricChartCap"),
    tableBox: document.getElementById("epochTableBox"),
    tableHead: document.getElementById("epochHead"),
    tableRows: document.getElementById("epochRows"),
  };

  dom.kind.addEventListener("change", syncKind);
  dom.form.addEventListener("submit", onSubmit);
  dom.btnStop.addEventListener("click", onStop);
  dom.btnCheckEnv.addEventListener("click", onCheckEnv);
  // 切换任务类型时自动改写权重名，与 Qt 版一致：
  // detect -> yolov8n.pt / segment -> yolov8n-seg.pt / pose -> yolov8n-pose.pt / obb -> yolov8n-obb.pt
  document.getElementById("yoloTask").addEventListener("change", syncYoloModel);

  // 增强预设：只写「与默认不同」的项，其余留空，便于用户看出到底改了什么。
  // 小缺陷预设针对 PCB 这类 70px 缺陷：关掉 mosaic（拼图会让缺陷再缩小一半）、
  // 减弱几何抖动（避免把缺陷变形到认不出）、保留颜色抖动（应对打光差异）。
  document.getElementById("augSmallDefect").addEventListener("click", () => {
    const preset = {
      augMosaic: "0.1",
      augCloseMosaic: "20",
      augScale: "0.3",
      augTranslate: "0.05",
      augDegrees: "0",
      augMixup: "0",
    };
    for (const [id, value] of Object.entries(preset)) {
      const el = document.getElementById(id);
      if (el) el.value = value;
    }
    appendLog("已套用小缺陷预设：mosaic=0.1, close_mosaic=20, scale=0.3, translate=0.05", "ok");
  });

  document.getElementById("augReset").addEventListener("click", () => {
    document.querySelectorAll(".augbox input").forEach((el) => {
      el.value = "";
    });
    appendLog("已清空增强参数，将使用 Ultralytics 默认值", "ok");
  });

  syncKind();
  setDefaultRunName();
}

/** 生成带时间戳的默认实验名，避免多次训练覆盖同一目录。 */
function setDefaultRunName() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const stamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  dom.runName.value = `exp_${stamp}`;
}

/** 按 YOLO 任务类型改写默认权重名。只改默认别名，用户自定义路径不动。 */
function syncYoloModel() {
  const task = formValue("yoloTask");
  const suffix = { detect: "", segment: "-seg", pose: "-pose", obb: "-obb" }[task] ?? "";
  const current = (document.getElementById("yoloModel").value || "").trim();
  // 只有当前值是纯别名（yolov8n / yolov8n.pt 之类）时才改写，避免覆盖用户填的真实路径
  if (!current || /^yolo[\w]*n?(-seg|-pose|-obb)?\.pt$/i.test(current)) {
    document.getElementById("yoloModel").value = `yolov8n${suffix}.pt`;
  }
}

function syncKind() {
  const isYolo = dom.kind.value === "yolo";
  dom.yoloGroup.classList.toggle("hidden", !isYolo);
  dom.anomalibGroup.classList.toggle("hidden", isYolo);
  // 两种任务的默认输出目录不同：YOLO 落 runs，Anomalib 落 results
  dom.project.value = isYolo ? "runs/trtforge" : "results";
  // Anomalib 的实验名默认用 latest，与 launcher 默认值一致；
  // 切回 YOLO 时重新生成带时间戳的名字，避免沿用 "latest" 覆盖旧结果。
  if (!isYolo) {
    dom.runName.value = "latest";
  } else if (!dom.runName.value || dom.runName.value === "latest") {
    setDefaultRunName();
  }
}

function formValue(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function buildYoloPayload() {
  const payload = {
    task: formValue("yoloTask"),
    model: formValue("yoloModel"),
    data: formValue("yoloData"),
    epochs: Number(formValue("yoloEpochs")) || 100,
    imgsz: Number(formValue("yoloImgsz")) || 640,
    batch: formValue("yoloBatch") || "16",
    device: formValue("yoloDevice") || "0",
    project: formValue("project") || "runs/trtforge",
    name: formValue("runName") || "exp",
    python: formValue("pythonPath") || null,
  };

  // 增强参数：留空就不传，后端也不会拼进命令行，最终用 Ultralytics 默认值。
  // 这样「不填」与「升级前」行为完全一致，不会因为表单多了一堆字段而悄悄改变训练结果。
  const augmentFields = {
    hsv_h: "augHsvH",
    hsv_s: "augHsvS",
    hsv_v: "augHsvV",
    degrees: "augDegrees",
    translate: "augTranslate",
    scale: "augScale",
    shear: "augShear",
    perspective: "augPerspective",
    flipud: "augFlipud",
    fliplr: "augFliplr",
    bgr: "augBgr",
    mosaic: "augMosaic",
    mixup: "augMixup",
    copy_paste: "augCopyPaste",
    close_mosaic: "augCloseMosaic",
  };
  for (const [key, id] of Object.entries(augmentFields)) {
    const raw = formValue(id).trim();
    if (raw === "") continue;
    const value = Number(raw);
    if (Number.isNaN(value)) {
      throw new Error(`增强参数 ${key} 不是合法数字：${raw}`);
    }
    payload[key] = value;
  }
  return payload;
}

function buildAnomalibPayload() {
  return {
    model: formValue("anomModel"),
    root: formValue("anomRoot"),
    normal_dir: formValue("anomNormal") || "train/good",
    abnormal_dir: formValue("anomAbnormal") || "test/abnormal",
    normal_test_dir: formValue("anomNormalTest") || "test/good",
    project: formValue("project") || "results",
    name: formValue("runName") || "latest",
    input_size: Number(formValue("anomInputSize")) || 256,
    max_epochs: formValue("anomMaxEpochs") || "auto",
    python: formValue("pythonPath") || null,
  };
}

async function onSubmit(event) {
  event.preventDefault();
  const isYolo = dom.kind.value === "yolo";

  // 提交前只做「必填项是否存在」这种前端能判断的检查，
  // 路径是否真实存在交给后端校验，避免前后端两套规则。
  if (isYolo && !formValue("yoloData")) {
    setHint("请填写 dataset.yaml 路径");
    return;
  }
  if (!isYolo && !formValue("anomRoot")) {
    setHint("请填写 Anomalib 数据集 root");
    return;
  }

  setHint("");
  dom.btnTrain.disabled = true;
  resetRunView();
  appendLog("正在启动任务…", "ok");

  try {
    const payload = isYolo ? buildYoloPayload() : buildAnomalibPayload();
    const url = isYolo ? "/api/train/yolo" : "/api/train/anomalib";
    const run = await postJSON(url, payload);
    state.runId = run.run_id;
    setBadge("运行中", "run");
    dom.btnStop.disabled = false;
    appendLog(`任务已启动：${run.run_id}`, "ok");
    appendLog(`命令：${run.command.join(" ")}`);
    // 参数回显：折叠块里放完整命令行，用户能确认自己填的参数真的传进去了
    dom.commandBox.hidden = false;
    dom.command.textContent = run.command.join(" ");
    startProgressView();
    openStream(run.run_id);
  } catch (error) {
    setHint(error.message);
    appendLog(`启动失败：${error.message}`, "err");
    dom.btnTrain.disabled = false;
    setBadge("启动失败", "fail");
  }
}

function openStream(runId) {
  state.stream?.close();
  state.stream = connectRunEvents(runId, {
    onEvent: handleEvent,
    onClose: (payload) => {
      dom.btnTrain.disabled = false;
      dom.btnCheckEnv.disabled = false;
      dom.btnStop.disabled = true;
      stopProgressTimer();
      const ok = (payload.exit ?? 0) === 0;
      setBadge(ok ? "已完成" : "失败", ok ? "done" : "fail");
      setStage(ok ? "已完成" : "已结束", ok ? "done" : "fail");
      if (ok && progress.pct < 100) setProgressBar(100);
      appendLog(
        ok ? "任务完成。" : `任务结束，退出码 ${payload.exit}`,
        ok ? "ok" : "err",
      );
      // 结束那一刻把「已用」定格，避免定时器停掉后数字看起来停在中间
      renderElapsed(true);
      if (state.artifacts.length) {
        appendLog(`产物：\n${state.artifacts.join("\n")}`, "ok");
      }
    },
    onError: (message) => appendLog(message, "warn"),
  });
}

async function onStop() {
  if (!state.runId) return;
  dom.btnStop.disabled = true;
  appendLog("正在停止…", "warn");
  try {
    await stopRun(state.runId);
  } catch (error) {
    appendLog(`停止失败：${error.message}`, "err");
    dom.btnStop.disabled = false;
  }
}

/**
 * 环境自检：开训前先确认 Python / torch / CUDA 是否就绪。
 * 走与训练相同的事件通道，因此日志面板与进度条都能复用。
 */
async function onCheckEnv() {
  const kind = dom.kind.value;
  const python = formValue("pythonPath") || null;

  dom.btnCheckEnv.disabled = true;
  setHint("");
  resetRunView();
  setBadge("检测中…", "run");
  appendLog(`正在检测 ${kind} 训练环境…`, "ok");

  try {
    const run = await postJSON(`/api/train/${kind}/check-env?python=${encodeURIComponent(python || "")}`, {});
    state.runId = run.run_id;
    // 自检没有 epoch 概念，但「已用时间」仍然有意义（加载 torch/CUDA 可能要几十秒）
    startProgressView();
    setStage("环境检测", "run");
    setProgressBar(0);
    openStream(run.run_id);
  } catch (error) {
    setHint(error.message);
    appendLog(`检测失败：${error.message}`, "err");
    setBadge("检测失败", "fail");
    dom.btnCheckEnv.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 事件渲染
//
// 字段名必须与 launcher 实际发出的 JSON 一致，不能凭猜测。已核对的真实结构：
//   epoch_summary : { epoch, total, pct, losses:{k:v}, metrics:{k:v}, lr:{k:v}, gpu:{...} }
//   gpu_status    : { available, index, name, memory_total_mb, memory_used_mb,
//                     memory_free_mb, utilization_gpu_pct, error, nvidia_smi_error }
//   artifact      : { kind, name, path, message }
//   plot          : { name, path, mtime }
//   progress      : { epoch, total, pct, stage }
// ---------------------------------------------------------------------------
function handleEvent(event) {
  switch (event.type) {
    case "progress": {
      const pct = Number(event.pct ?? 0);
      setProgressBar(pct);
      const stage = event.stage ? `[${event.stage}] ` : "";
      const total = event.total ?? "-";
      if (event.total) progress.total = Number(event.total);
      if (event.epoch != null) progress.epoch = Number(event.epoch);
      renderEpochText();
      appendLog(`${stage}进度：epoch ${event.epoch ?? "-"}/${total}，${pct.toFixed(1)}%`);
      break;
    }
    case "metric": {
      const name = event.name || "metric";
      const value = typeof event.value === "number" ? event.value.toFixed(4) : event.value;
      setMetric(name, value);
      break;
    }
    case "epoch_summary": {
      // losses / metrics / lr 都是字典，不是标量
      applyMetricDict(event.losses, 6);
      applyMetricDict(event.metrics, 8);
      applyMetricDict(event.lr, 2);
      if (event.gpu) applyGpu(event.gpu);
      recordEpoch(event);
      if (event.total) {
        const pct = Number(event.pct ?? (event.epoch / event.total) * 100);
        setProgressBar(pct);
      }
      break;
    }
    case "gpu_status": {
      applyGpu(event);
      break;
    }
    case "env": {
      const parts = [];
      if (event.python) parts.push(`python=${event.python}`);
      if (event.env_name) parts.push(`env=${event.env_name}`);
      if (event.torch?.version) parts.push(`torch=${event.torch.version}`);
      if (event.cuda?.available) {
        const devices = event.cuda.devices || [];
        parts.push(`gpu=${devices.length ? devices.join(",") : "已启用"}`);
      } else {
        parts.push("gpu=不可用");
      }
      appendLog(`环境：${parts.join(" | ")}`, event.missing?.length ? "warn" : "ok");

      const missing = event.missing || [];
      if (missing.length) {
        for (const item of missing) {
          appendLog(`  缺少 ${item.package}：${item.error || ""}`, "warn");
          if (item.install) appendLog(`  修复命令：${item.install}`, "warn");
        }
      }
      break;
    }
    case "model": {
      // probe-model 事件，字段是 path / task / names / classes
      const names = event.names ? Object.keys(event.names).length : (event.classes ?? "?");
      appendLog(`模型：${event.path || event.name || ""}（task=${event.task || "-"}，类别数=${names}）`, "ok");
      break;
    }
    case "artifact": {
      const path = event.path || event.file || "";
      const label = event.kind || "产物";
      if (path) {
        state.artifacts.push(path);
        appendLog(`${label}：${path}`, "ok");
      } else if (event.message) {
        appendLog(`${label}：${event.message}`, "warn");
      }
      break;
    }
    case "plot": {
      // 训练曲线图，Qt 版缓存成缩略图；Web 版先只记录路径，不逐张渲染
      if (event.name) setMetric(`图 ${event.name}`, "已生成");
      break;
    }
    case "anomaly_model": {
      appendLog(
        `异常检测模型：${event.model || "-"}  input=${(event.input_size || []).join("x") || "-"}`,
        "ok",
      );
      if (event.model) setMetric("model", event.model);
      if (event.input_size) setMetric("input_size", (event.input_size || []).join("x"));
      break;
    }
    case "anomaly_export_summary": {
      appendLog(`导出完成：${event.model || ""}  contract=${event.output_contract || "-"}`, "ok");
      if (event.model_type) setMetric("model_type", event.model_type);
      if (event.image_threshold != null) {
        setMetric("image_threshold", Number(event.image_threshold).toFixed(4));
      }
      if (event.pixel_threshold != null) {
        setMetric("pixel_threshold", Number(event.pixel_threshold).toFixed(4));
      }
      for (const key of ["ckpt", "onnx", "metadata"]) {
        if (event[key]) {
          state.artifacts.push(event[key]);
          appendLog(`  ${key}：${event[key]}`, "ok");
        }
      }
      break;
    }
    case "event": {
      // Qt 版对这种"里程碑"事件没有分支，这里降级成一行日志，避免 JSON 刷屏
      const name = event.name || "事件";
      const extra = [];
      if (event.task) extra.push(`task=${event.task}`);
      if (event.model) extra.push(`model=${event.model}`);
      if (event.format) extra.push(`format=${event.format}`);
      if (event.max_epochs != null) extra.push(`max_epochs=${event.max_epochs}`);
      if (event.weights) extra.push(`weights=${event.weights}`);
      appendLog(`» ${name}${extra.length ? "  " + extra.join(" ") : ""}`);
      break;
    }
    case "error": {
      appendLog(`错误 ${event.code || ""}：${event.message || ""}`, "err");
      if (event.install) appendLog(`修复：${event.install}`, "warn");
      if (event.path) appendLog(`路径：${event.path}`, "err");
      if (event.supported) appendLog(`支持：${event.supported.join(", ")}`, "warn");
      if (event.missing?.length) {
        for (const item of event.missing) {
          appendLog(`  缺少 ${item.package}：${item.install || ""}`, "warn");
        }
      }
      if (event.traceback) appendLog(event.traceback, "err");
      break;
    }
    case "finished": {
      // 由 onClose 统一处理
      break;
    }
    case "_ws": {
      // 内部的连接状态信号（open 等），不是训练事件，不要打印到日志
      break;
    }
    case "log": {
      const line = event.line || "";
      const cls = /error|fail|traceback/i.test(line) ? "err" : "";
      appendLog(line, cls);
      break;
    }
    default: {
      appendLog(JSON.stringify(event));
    }
  }
}

/** 把 {指标名: 数值} 字典写进指标条，最多显示 limit 个，避免指标区被撑爆。 */
function applyMetricDict(dict, limit) {
  if (!dict || typeof dict !== "object") return;
  const entries = Object.entries(dict).filter(([, v]) => typeof v === "number");
  for (const [key, value] of entries.slice(0, limit)) {
    setMetric(key, Number(value).toFixed(4));
  }
}

/** GPU 状态渲染：字段名是 utilization_gpu_pct，不是 utilization。 */
function applyGpu(gpu) {
  if (!gpu || typeof gpu !== "object") return;
  if (gpu.available === false) {
    if (gpu.error) setMetric("GPU", "不可用");
    return;
  }
  if (gpu.memory_used_mb != null && gpu.memory_total_mb != null) {
    setMetric(
      "GPU 显存",
      `${Math.round(gpu.memory_used_mb)}/${Math.round(gpu.memory_total_mb)} MB`,
    );
  }
  if (gpu.utilization_gpu_pct != null) {
    setMetric("GPU 利用率", `${Math.round(gpu.utilization_gpu_pct)}%`);
  }
  if (gpu.name) setMetric("GPU 型号", gpu.name);
}

// ---------------------------------------------------------------------------
// 进度面板：epoch / 已用 / 每轮耗时 / 预计剩余
//
// 「预计剩余」是前端推算的，不是后端字段：
//   单轮耗时 = 最近 3 轮耗时的中位数（首轮含数据集扫描，用平均会被拉偏）
//   剩余     = 剩余轮数 × 单轮耗时 - 本轮已过去的秒数
// 定时器每秒刷新一次，所以即使一轮要跑几分钟，数字也在动。
// ---------------------------------------------------------------------------
function setProgressBar(pct) {
  const value = Math.min(100, Math.max(0, Number(pct) || 0));
  progress.pct = value;
  dom.progressBar.style.width = `${value.toFixed(1)}%`;
  dom.pctText.textContent = `${value.toFixed(1)}%`;
}

function setStage(text, cls = "") {
  dom.stage.textContent = text;
  dom.stage.className = cls ? `badge ${cls}` : "badge";
}

function renderEpochText() {
  const total = progress.total || "-";
  const epoch = progress.epoch || "-";
  dom.epochText.textContent = `epoch ${epoch}/${total}`;
}

function median(values) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
}

/** 单轮耗时的中位数（只取最近 3 轮），拿不到返回 0 */
function perEpochSeconds() {
  return median(progress.epochDeltas.slice(-3));
}

function fmtClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const ss = String(sec).padStart(2, "0");
  if (h) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
  return `${m}:${ss}`;
}

/** 刷新「已用 / 每轮 / 剩余」。freeze=true 时把已用定格在当前位置。 */
function renderElapsed(freeze = false) {
  if (!progress.startedAt) return;
  const elapsed = (Date.now() - progress.startedAt) / 1000;
  dom.elapsedText.textContent = `已用 ${fmtClock(elapsed)}`;

  const per = perEpochSeconds();
  dom.speedText.textContent = per ? `每轮 ≈ ${fmtClock(per)}` : "每轮 —";

  if (!per || !progress.total || !progress.epoch) {
    dom.etaText.textContent = "预计剩余 —";
    return;
  }
  const left = Math.max(0, progress.total - progress.epoch);
  // 本轮已经跑掉的时间要从剩余里扣掉，否则最后一轮的剩余时间会看着不动
  const inEpoch = progress.lastEpochAt ? (Date.now() - progress.lastEpochAt) / 1000 : 0;
  const remain = Math.max(0, left * per - (freeze ? 0 : inEpoch));
  dom.etaText.textContent = `预计剩余 ${fmtClock(remain)}`;
}

function startProgressView() {
  stopProgressTimer();
  progress.total = 0;
  progress.epoch = 0;
  progress.startedAt = Date.now();
  progress.lastEpochAt = 0;
  progress.epochDeltas = [];
  progress.pct = 0;
  progress.timer = setInterval(() => renderElapsed(false), 1000);
  setStage("运行中", "run");
  renderEpochText();
  renderElapsed();
}

function stopProgressTimer() {
  if (progress.timer) {
    clearInterval(progress.timer);
    progress.timer = null;
  }
}

/** 收到一轮的汇总：更新 ETA 基准、表格、曲线。 */
function recordEpoch(event) {
  const now = Date.now();
  const rawEpoch = Number(event.epoch) || progress.epoch + 1;
  if (event.total) progress.total = Number(event.total);
  // launcher 在训练结束后还会补一次「最终验证」的 epoch_summary，其 epoch = total + 1。
  // 直接照搬会显示成「epoch 4/3」，所以夹到 total，并在表格里标成 final。
  const isFinal = progress.total > 0 && rawEpoch > progress.total;
  progress.epoch = isFinal ? progress.total : rawEpoch;

  if (progress.lastEpochAt && !isFinal) {
    const delta = (now - progress.lastEpochAt) / 1000;
    if (delta > 0) progress.epochDeltas.push(delta);
  }
  progress.lastEpochAt = now;
  renderEpochText();
  renderElapsed();

  // 表格行：把 losses / metrics / lr 展平成 `group:name` 的列
  const values = {};
  collectNumbers(event.losses, "loss", values);
  collectNumbers(event.metrics, "metric", values);
  collectNumbers(event.lr, "lr", values);
  progress.rows.push({ epoch: rawEpoch, values, isFinal });
  if (progress.rows.length > MAX_TABLE_ROWS) {
    progress.rows = progress.rows.slice(-MAX_TABLE_ROWS);
  }
  // 新列出现时插到对应分组末尾，已有列顺序不变
  for (const key of Object.keys(values)) {
    if (!progress.columns.includes(key)) progress.columns.push(key);
  }
  progress.columns.sort(compareColumn);
  renderEpochTable();

  // 曲线数据：loss 取总和（各损失项相加，趋势与总损失一致）。
  // 训练损失优先；launcher 在 on_fit_epoch_end 里 `trainer.tloss` 常常是 None
  // （实测 8.x 会返回空 losses），这时回退到验证损失之和，否则曲线永远是空的。
  const trainLossValues = Object.entries(event.losses || {})
    .filter(([, v]) => typeof v === "number")
    .map(([, v]) => v);
  let lossSum = null;
  if (trainLossValues.length) {
    lossSum = trainLossValues.reduce((a, b) => a + b, 0);
    progress.lossSource = "训练";
  } else {
    const valLossValues = Object.entries(event.metrics || {})
      .filter(([key, v]) => /loss$/i.test(key) && typeof v === "number")
      .map(([, v]) => v);
    if (valLossValues.length) {
      lossSum = valLossValues.reduce((a, b) => a + b, 0);
      progress.lossSource = "验证";
    }
  }
  if (lossSum != null) {
    progress.losses.push(lossSum);
    if (progress.losses.length > MAX_CHART_POINTS) progress.losses.shift();
  }
  for (const key of preferredMetricKeys(event.metrics)) {
    const series = progress.metrics[key] || (progress.metrics[key] = []);
    series.push(Number(event.metrics[key]));
    if (series.length > MAX_CHART_POINTS) series.shift();
  }
  dom.chartRow.hidden = progress.losses.length < 2 && Object.keys(progress.metrics).length === 0;
  drawCharts();
}

function collectNumbers(dict, group, out) {
  if (!dict || typeof dict !== "object") return;
  for (const [name, value] of Object.entries(dict)) {
    if (typeof value === "number" && Number.isFinite(value)) out[`${group}:${name}`] = value;
  }
}

/** 列排序：loss 在前、metric 居中、lr 在后；同组内保持出现顺序（sort 稳定） */
function compareColumn(a, b) {
  const rank = (key) => (key.startsWith("loss:") ? 0 : key.startsWith("metric:") ? 1 : 2);
  return rank(a) - rank(b);
}

/** 优先画 mAP 这类用户最关心的指标，最多两条，避免曲线糊成一团 */
function preferredMetricKeys(metrics) {
  if (!metrics || typeof metrics !== "object") return [];
  const numeric = Object.keys(metrics).filter((key) => typeof metrics[key] === "number");
  const preferred = ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95", "metrics/mAP50"];
  const picked = preferred.filter((key) => numeric.includes(key));
  if (picked.length) return picked.slice(0, 2);
  return numeric.slice(0, 2);
}

function fmtNum(value) {
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function renderEpochTable() {
  if (!progress.rows.length) {
    dom.tableBox.hidden = true;
    return;
  }
  dom.tableBox.hidden = false;
  // 列太多会把表格撑爆，只留前 8 列（loss 优先，因为排序把 loss 放前面）
  const columns = progress.columns.slice(0, 8);

  const headRow = document.createElement("tr");
  headRow.appendChild(cellNode("th", "epoch"));
  for (const key of columns) {
    headRow.appendChild(cellNode("th", key, "num"));
  }
  dom.tableHead.innerHTML = "";
  dom.tableHead.appendChild(headRow);

  dom.tableRows.innerHTML = "";
  for (const row of progress.rows) {
    const tr = document.createElement("tr");
    tr.appendChild(cellNode("td", row.isFinal ? "final" : String(row.epoch), "num"));
    for (const key of columns) {
      const value = row.values[key];
      tr.appendChild(cellNode("td", typeof value === "number" ? fmtNum(value) : "—", "num"));
    }
    dom.tableRows.appendChild(tr);
  }
}

function cellNode(tag, text, cls = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

// ---------------------------------------------------------------------------
// 曲线：零依赖内联 SVG（前端保持零构建，不引第三方图表库）
// ---------------------------------------------------------------------------
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, String(value));
  }
  return node;
}

/**
 * 把若干条序列画进一个 svg。
 * `series` 形如 [{ name, values }]，自动根据全部数据定 Y 轴范围。
 * 只画折线 + 3 条横向参考线 + 最大/最小值文本，够看趋势就行。
 */
function drawChart(svg, series) {
  const W = 320;
  const H = 110;
  const PAD = 12;
  svg.innerHTML = "";

  const usable = series.filter((item) => item.values.length >= 1);
  if (!usable.length) {
    const empty = svgEl("text", { x: W / 2, y: H / 2, class: "chart-axis", "text-anchor": "middle" });
    empty.textContent = "暂无数据";
    svg.appendChild(empty);
    return;
  }

  const all = usable.flatMap((item) => item.values);
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (min === max) {
    // 一条水平线也要能看见：人为撑开一点范围
    min -= 0.5;
    max += 0.5;
  }
  const span = max - min;

  for (let i = 0; i <= 2; i += 1) {
    const y = PAD + ((H - 2 * PAD) / 2) * i;
    svg.appendChild(
      svgEl("line", { x1: PAD, y1: y, x2: W - PAD, y2: y, class: "chart-grid" }),
    );
  }

  usable.forEach((item, index) => {
    const count = item.values.length;
    const points = item.values
      .map((value, i) => {
        const x = count === 1 ? W / 2 : PAD + (i / (count - 1)) * (W - 2 * PAD);
        const y = H - PAD - ((value - min) / span) * (H - 2 * PAD);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    const line = svgEl("polyline", { points, class: `chart-line chart-s${index % 3}` });
    // 点少的时候补圆点，否则 1~2 个点时折线几乎看不见
    if (count <= 2) {
      for (const pair of points.split(" ")) {
        const [cx, cy] = pair.split(",");
        svg.appendChild(svgEl("circle", { cx, cy, r: 2, class: `chart-dot chart-f${index % 3}` }));
      }
    }
    svg.appendChild(line);
  });

  const maxText = svgEl("text", { x: PAD, y: PAD - 2, class: "chart-axis" });
  maxText.textContent = fmtNum(max);
  svg.appendChild(maxText);
  const minText = svgEl("text", { x: PAD, y: H - 2, class: "chart-axis" });
  minText.textContent = fmtNum(min);
  svg.appendChild(minText);
}

function drawCharts() {
  const lossSeries = progress.losses.length
    ? [{ name: "total loss", values: progress.losses }]
    : [];
  drawChart(dom.lossChart, lossSeries);
  dom.lossCap.textContent = lossSeries.length
    ? `loss（总，${progress.lossSource || "训练"}）`
    : "loss（暂无数据）";

  const metricSeries = Object.entries(progress.metrics)
    .filter(([, values]) => values.length)
    .map(([name, values]) => ({ name, values }));
  drawChart(dom.metricChart, metricSeries);
  dom.metricCap.textContent = metricSeries.length
    ? `指标：${metricSeries.map((item) => item.name).join(" / ")}`
    : "指标（暂无数据）";
}

// ---------------------------------------------------------------------------
// DOM 小工具
// ---------------------------------------------------------------------------
function appendLog(text, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  dom.logBox.appendChild(line);
  // 只保留最近 3000 行，避免长时间训练把浏览器拖慢
  while (dom.logBox.childElementCount > 3000) {
    dom.logBox.removeChild(dom.logBox.firstElementChild);
  }
  dom.logBox.scrollTop = dom.logBox.scrollHeight;
}

function setMetric(name, value) {
  const id = `metric-${name}`;
  let chip = document.getElementById(id);
  if (!chip) {
    chip = document.createElement("div");
    chip.className = "metric";
    chip.id = id;
    chip.dataset.name = name;
    dom.metrics.appendChild(chip);
  }
  chip.innerHTML = `${escapeHtml(name)} <b>${escapeHtml(String(value))}</b>`;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ({
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

function resetRunView() {
  state.artifacts = [];
  dom.logBox.innerHTML = "";
  dom.metrics.innerHTML = "";
  dom.progressBar.style.width = "0%";

  // 进度面板 / 表格 / 曲线全部清空，否则上一个任务的曲线会留在新任务下面
  stopProgressTimer();
  progress.total = 0;
  progress.epoch = 0;
  progress.startedAt = 0;
  progress.lastEpochAt = 0;
  progress.epochDeltas = [];
  progress.pct = 0;
  progress.rows = [];
  progress.columns = [];
  progress.losses = [];
  progress.metrics = {};
  progress.lossSource = "";
  setStage("待启动");
  dom.epochText.textContent = "epoch -/-";
  dom.pctText.textContent = "0.0%";
  dom.elapsedText.textContent = "已用 0:00";
  dom.etaText.textContent = "预计剩余 —";
  dom.speedText.textContent = "每轮 —";
  dom.commandBox.hidden = true;
  dom.command.textContent = "";
  dom.chartRow.hidden = true;
  dom.tableBox.hidden = true;
  dom.tableHead.innerHTML = "";
  dom.tableRows.innerHTML = "";
  dom.lossChart.innerHTML = "";
  dom.metricChart.innerHTML = "";
  dom.lossCap.textContent = "loss（暂无数据）";
  dom.metricCap.textContent = "指标（暂无数据）";
}
