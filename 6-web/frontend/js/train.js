// 训练页：表单 → 启动任务 → 事件流渲染。
// 只做展示与参数收集，参数语义完全沿用 launcher，不在前端做业务判断。

import { postJSON, stopRun } from "./api.js";
import { connectRunEvents } from "./ws.js";

const state = {
  runId: null,
  stream: null,
  artifacts: [],
};

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
      const ok = (payload.exit ?? 0) === 0;
      setBadge(ok ? "已完成" : "失败", ok ? "done" : "fail");
      appendLog(
        ok ? "任务完成。" : `任务结束，退出码 ${payload.exit}`,
        ok ? "ok" : "err",
      );
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
      dom.progressBar.style.width = `${Math.min(100, Math.max(0, pct)).toFixed(1)}%`;
      const stage = event.stage ? `[${event.stage}] ` : "";
      const total = event.total ?? "-";
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
      if (event.total) {
        const pct = Number(event.pct ?? (event.epoch / event.total) * 100);
        dom.progressBar.style.width = `${Math.min(100, Math.max(0, pct)).toFixed(1)}%`;
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
}
