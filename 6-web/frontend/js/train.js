// 训练页：表单 → 启动任务 → 事件流渲染。
// 只做展示与参数收集，参数语义完全沿用 launcher，不在前端做业务判断。

import { getJSON, listRuns, postJSON, stopRun } from "./api.js";
import { createLogActions } from "./logs.js";
import { connectRunEvents } from "./ws.js";

//: 训练页能接管的任务类型。
//: 自检（check-env）也算：它虽然短，但会把控制台变成只读，
//: 刷新后不接管的话按钮状态会自相矛盾（能点「检测环境」却不能点「停止」）。
const TRAIN_KINDS = new Set(["yolo-train", "anomalib-train", "yolo-check-env", "anomalib-check-env"]);


const state = {
  runId: null,
  stream: null,
  //: 当前日志面板跟踪的 run。串联的每段都能点「日志」，靠它避免把正在跟踪的
  //: 训练任务顶掉（顶掉后停止按钮会指向另一个 run，很难察觉）。
  streamRunId: null,
  artifacts: [],
};

/// 开训前体检 / 增强预览 / 断点续训的状态。
/// 全部只活在页面里：刷新后重新扫描即可，不落 localStorage——
/// 这些信息都依赖磁盘上的真实文件，缓存下来反而会误导。
const precheck = {
  overview: null,
  previewIndex: 0,
  previewSeed: null,
  resumeFrom: "",
  resumeLabel: "",
  // 微调起点（自己的 best.pt）。与 resumeFrom 互斥：
  //   resumeFrom = last.pt，恢复优化器与轮次，参数以检查点为准；
  //   finetuneFrom = best.pt，只载入权重，参数按表单，写回原实验目录。
  finetuneFrom: "",
  finetuneLabel: "",
  finetuneSource: null,
  // 算力体检结果（GET /api/train/preflight）与「显存不够但用户坚持要跑」的标记。
  // 后者只生效一次：下次点「开始训练」会重新走一遍预检。
  hardware: null,
  forceLowVram: false,
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
  // 启动阶段：spawned / boot / deps / model / data / train_start
  // 每一项 {phase, label, elapsed_ms, step_ms, warm}
  phases: [],
  warmHit: false,
};

//: 表格只留最近这么多轮，避免长时间训练把 DOM 撑大
const MAX_TABLE_ROWS = 20;
//: 曲线最多画这么多点（与表格解耦：曲线看趋势，多留一些更平滑）
const MAX_CHART_POINTS = 300;

let dom = {};
//: 日志出口（复制 / 打开日志文件）。在 initTrain 里创建，其它函数直接用。
let logActions = null;
//: 最近一次已结束的训练 run。刷新页面后控制台是空的，但磁盘上那份日志还在，
//: 「打开日志文件」应该指向它，而不是一直灰着。
let lastFinishedRunId = null;

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
    // 开训前体检
    btnScanData: document.getElementById("btnScanData"),
    overviewHint: document.getElementById("dsOverviewHint"),
    overviewBody: document.getElementById("dsOverviewBody"),
    overviewCounts: document.getElementById("dsCounts"),
    overviewBars: document.getElementById("dsClassBars"),
    overviewResolution: document.getElementById("dsResolution"),
    overviewWarnings: document.getElementById("dsWarnings"),
    // 增强预览
    btnPreview: document.getElementById("btnPreview"),
    btnPreviewSeed: document.getElementById("btnPreviewSeed"),
    btnPreviewNext: document.getElementById("btnPreviewNext"),
    previewHint: document.getElementById("previewHint"),
    previewBody: document.getElementById("previewBody"),
    previewOrigin: document.getElementById("previewOrigin"),
    previewAug: document.getElementById("previewAug"),
    previewAugCap: document.getElementById("previewAugCap"),
    previewApplied: document.getElementById("previewApplied"),
    // 断点续训 / 微调
    btnScanRuns: document.getElementById("btnScanRuns"),
    btnClearResume: document.getElementById("btnClearResume"),
    resumeHint: document.getElementById("resumeHint"),
    resumeActive: document.getElementById("resumeActive"),
    resumeTableBox: document.getElementById("resumeTableBox"),
    resumeRows: document.getElementById("resumeRows"),
    // 启动清单 + 预热状态
    startupSteps: document.getElementById("startupSteps"),
    startupSummary: document.getElementById("startupSummary"),
    warmBadge: document.getElementById("warmBadge"),
    // 日志出口
    btnCopyLog: document.getElementById("btnCopyLog"),
    btnOpenLog: document.getElementById("btnOpenLog"),
    // 算力体检（显存 + 混合精度）
    btnCheckHw: document.getElementById("btnCheckHw"),
    btnForceVram: document.getElementById("btnForceVram"),
    ampMode: document.getElementById("ampMode"),
    hwHint: document.getElementById("hwHint"),
    hwBody: document.getElementById("hwBody"),
    hwGpu: document.getElementById("hwGpu"),
    hwVram: document.getElementById("hwVram"),
    hwAmp: document.getElementById("hwAmp"),
    hwData: document.getElementById("hwData"),
  };

  logActions = createLogActions({
    copyBtn: dom.btnCopyLog,
    openBtn: dom.btnOpenLog,
    // innerText 而不是 textContent：面板里每行是一个 <div>，textContent 会把它们连成一行
    getText: () => dom.logBox.innerText || "",
    getRunId: () => state.runId || lastFinishedRunId,
  });

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

  // 开训前体检：三个面板都是「看一眼就能省一次训练」的信息，全部按需触发
  dom.btnScanData.addEventListener("click", () => scanDataset());
  dom.btnPreview.addEventListener("click", () => renderPreview({ reseed: true }));
  dom.btnPreviewSeed.addEventListener("click", () => renderPreview({ reseed: true }));
  dom.btnPreviewNext.addEventListener("click", () => {
    precheck.previewIndex += 1;
    renderPreview({ reseed: true });
  });
  dom.btnScanRuns.addEventListener("click", () => scanRuns());
  dom.btnClearResume.addEventListener("click", () => clearResume());

  // 算力体检：按钮手动触发；改 amp 档位时若已经展开过，顺手重算一次
  // （显存需求会随「预测精度」变，用户改了档位却看到旧结论才是真误导）。
  dom.btnCheckHw.addEventListener("click", () => scanHardware());
  dom.ampMode.addEventListener("change", () => {
    if (precheck.hardware) scanHardware();
  });
  dom.btnForceVram.addEventListener("click", () => {
    // 只越权这一次：标记用完即清，避免之后每次启动都绕过预检。
    precheck.forceLowVram = true;
    dom.btnForceVram.hidden = true;
    onSubmit(new Event("submit", { cancelable: true }));
  });

  syncKind();
  setDefaultRunName();
  refreshWarmStatus(true);

  // 刷新后自动接续：任务活在后端，页面只是窗口。不接的话用户会看到
  // 「未开始」+ 可点的「开始训练」，点下去就是第二个训练进程（抢同一块显卡）。
  document.addEventListener("trtf:attach-run", (event) => {
    const run = event.detail?.run;
    if (run) attachRunningRun(run);
  });
  attachIfRunning();
}

/**
 * 拉一次任务列表，有训练类任务在跑就接管。
 *
 * 失败静默：拿不到列表只说明后端暂时不可达，顶栏的「运行中」条会另外报出来，
 * 这里再弹一次提示只会重复打扰。
 */
async function attachIfRunning() {
  try {
    const data = await listRuns();
    const runs = data?.runs || [];
    const run = runs.find((item) => item.status === "running" && TRAIN_KINDS.has(item.kind));
    if (run) {
      attachRunningRun(run);
      return;
    }
    // 没有在跑的任务时，把「最近一次训练」的日志留着可查：
    // 刷新页面想复盘的，往往就是刚刚失败的那一次，而内存里的 run 已经结束了。
    const last = runs.find((item) => TRAIN_KINDS.has(item.kind));
    if (last) {
      lastFinishedRunId = last.run_id;
      logActions?.sync();
      dom.btnOpenLog.title = `最近一次任务 ${last.run_id} 的完整日志（${last.status}）`;
    }
  } catch {
    /* 由顶栏条负责报「后端未连接」 */
  }
}

/**
 * 接管一个正在跑的任务：恢复按钮、进度时钟，并从后端补发全部日志。
 *
 * 已结束的任务不接管：那时的事件流已经关闭，接上去也只会停在最后一帧，
 * 还不如让用户从「历史训练」里重新拿结果。
 */
function attachRunningRun(run) {
  if (!run || !TRAIN_KINDS.has(run.kind)) return false;
  // 已经在接一个任务了：不要用顶栏点来的另一个任务把控制台顶掉，
  // 否则「停止」按钮会指向用户没在看的那个 run。
  if (state.streamRunId) return false;

  state.runId = run.run_id;
  state.artifacts = [];
  resetRunView();
  logActions?.sync();
  setBadge("运行中", "run");
  dom.btnTrain.disabled = true;
  dom.btnCheckEnv.disabled = true;
  dom.btnStop.disabled = false;
  if (Array.isArray(run.command) && run.command.length) {
    dom.commandBox.hidden = false;
    dom.command.textContent = run.command.join(" ");
  }
  appendLog(
    `已接续正在运行的任务 ${run.run_id}（${run.kind}，已运行 ${Math.round(Number(run.elapsed) || 0)} 秒）`,
    "ok",
  );
  appendLog("正在从后端补发日志；进度会接着往下走，不用重新点开始。");
  // 时钟从任务真实的开始时间接着数，而不是从 0 重新开始
  startProgressView(run.elapsed);
  openStream(run.run_id);
  return true;
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
  const suffix = { detect: "", segment: "-seg", pose: "-pose", obb: "-obb", classify: "-cls" }[task] ?? "";
  const current = (document.getElementById("yoloModel").value || "").trim();
  // 只有当前值是纯别名（yolov8n / yolov8n.pt 之类）时才改写，避免覆盖用户填的真实路径
  if (!current || /^yolo[\w]*n?(-seg|-pose|-obb|-cls)?\.pt$/i.test(current)) {
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
  // 预热状态是分任务类型的：切到 Anomalib 时 yolo 的「预热就绪」不适用
  refreshWarmStatus(false);
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
    // 起点语义：微调 = 只载入 best.pt 的权重，优化器与轮次重置（参数按表单）。
    // 与断点续训互斥，续训另有 resume_from 字段。
    init_mode: precheck.finetuneFrom ? "finetune" : "preset",
  };

  // 学习率：留空表示「不填」，由后端决定（微调兜底 0.001，新训用 Ultralytics 默认 0.01）。
  const lr0 = formValue("yoloLr0").trim();
  if (lr0 !== "") {
    const value = Number(lr0);
    if (Number.isNaN(value) || value < 0) {
      throw new Error(`初始学习率 lr0 不是合法数字：${lr0}`);
    }
    payload.lr0 = value;
  }

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

  // 混合精度三态：auto 不传字段——后端会拼出与升级前完全一致的命令行，
  // 由 Ultralytics 自己在训练开始时判定（黑名单显卡上一定会回退 fp32）。
  const ampMode = formValue("ampMode");
  if (ampMode === "off") payload.amp = false;
  else if (ampMode === "on") payload.amp = true;

  // 显存预检没通过时，用户点了「仍然开始」才带这个标记，且只带这一次。
  if (precheck.forceLowVram) {
    payload.allow_low_vram = true;
    precheck.forceLowVram = false;
    appendLog("显存预检未通过，按用户要求强行开始（若 OOM，batch 可能被框架自动减半）", "warn");
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
  dom.btnTrain.textContent = "正在启动…";
  appendLog("正在启动任务…", "ok");

  try {
    const payload = isYolo ? buildYoloPayload() : buildAnomalibPayload();
    if (isYolo && precheck.resumeFrom) {
      // 续训走独立字段：后端只拼 --resume-from <绝对路径>，
      // 不再依赖 Ultralytics 的 get_latest_run() 去猜「最近一次 run」。
      payload.resume_from = precheck.resumeFrom;
      appendLog(`续训模式：从 ${precheck.resumeFrom} 继续，epochs 等参数以检查点记录为准`, "warn");
    }
    if (isYolo && precheck.finetuneFrom) {
      // 微调前先验一遍：任务头不一致要拦住，类别/尺寸差异要提前说，
      // 否则等训练跑完才发现一半权重没生效就太晚了。
      const check = await postJSON("/api/train/yolo/finetune-check", {
        weights: precheck.finetuneFrom,
        data: payload.data,
        task: payload.task,
        imgsz: payload.imgsz,
      });
      if (!check.ok) {
        throw new Error((check.blockers || []).join("；") || "微调检查未通过");
      }
      for (const warning of check.warnings || []) {
        appendLog(`微调提醒：${warning}`, "warn");
      }
      const source = check.source || {};
      appendLog(
        `微调模式：从 ${source.name || payload.model} 的 best.pt 继续` +
          `（已完成 ${source.epochs_done ?? "?"} 轮，最佳 mAP50-95 ${source.best_map ?? "—"}）`,
        "ok",
      );
    }
    const url = isYolo ? "/api/train/yolo" : "/api/train/anomalib";
    const run = await postJSON(url, payload);
    state.runId = run.run_id;
    logActions?.sync();
    setBadge("运行中", "run");
    dom.btnStop.disabled = false;
    appendLog(`任务已启动：${run.run_id}${run.warm ? "（热启动）" : ""}`, "ok");
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
    refreshTrainButton();
    setBadge("启动失败", "fail");
    // 409 有两种含义，必须分开处理：
    //   1. 已经有任务在跑 → 最有用的动作是直接接管它，而不是让用户自己去找停止按钮；
    //   2. 显存预检不通过 → 接管一个不存在的任务毫无意义，应该给「仍然开始」。
    if (error.status === 409 && !String(error.message).startsWith("VRAM_PREFLIGHT")) {
      attachIfRunning();
    } else if (error.status === 409) {
      // 后缀式提示：把前缀去掉再展示，前缀只是前后端约定的识别标记
      setHint(String(error.message).replace(/^VRAM_PREFLIGHT:\s*/, ""));
      dom.btnForceVram.hidden = false;
    }
  }
}

function openStream(runId) {
  state.stream?.close();
  state.streamRunId = runId;
  state.stream = connectRunEvents(runId, {
    onEvent: handleEvent,
    onClose: (payload) => {
      state.stream = null;
      state.streamRunId = null;
      dom.btnTrain.disabled = false;
      dom.btnCheckEnv.disabled = false;
      dom.btnStop.disabled = true;
      stopProgressTimer();
      refreshTrainButton();
      // 任务结束意味着预热池已经补了一个新进程，把状态刷新一下
      refreshWarmStatus(false);
      const ok = (payload.exit ?? 0) === 0;
      // 主动停止不是失败：Windows 上 terminate() 的退出码是 1，
      // 只能靠后端标好的 status 判断。以前这里只看退出码，
      // 于是「自己点的停止」被显示成「失败」，误导用户以为训练崩了。
      const cancelled = payload.status === "cancelled";
      setBadge(cancelled ? "已停止" : ok ? "已完成" : "失败", cancelled ? "warn" : ok ? "done" : "fail");
      setStage(cancelled ? "已停止" : ok ? "已完成" : "已结束", cancelled ? "warn" : ok ? "done" : "fail");
      if (ok && progress.pct < 100) setProgressBar(100);
      appendLog(
        cancelled
          ? "任务已停止（进度与产物保留，可从 last.pt 续训）"
          : ok
            ? "任务完成。"
            : `任务结束，退出码 ${payload.exit}`,
        cancelled ? "warn" : ok ? "ok" : "err",
      );
      // 结束那一刻把「已用」定格，避免定时器停掉后数字看起来停在中间
      renderElapsed(true);
      if (state.artifacts.length) {
        // 同一个产物会被后端发多次（训练中出图、校验后再出图、收尾再列一遍），
        // 直接 join 会打印出一堆重复路径，这里按首次出现顺序去重。
        const unique = [...new Set(state.artifacts)];
        appendLog(`产物：\n${unique.join("\n")}`, "ok");
      }
    },
    onError: (message) => appendLog(message, "warn"),
  });
}

/**
 * 在训练页的日志面板里查看任意一个 run（串联的每一段都有自己的 run_id）。
 * 只复用现成的日志/进度渲染。若面板正在跟踪另一个任务，返回 false 让调用方提示，
 * 而不是悄悄把它顶掉：顶掉后「停止」按钮会指向另一个 run，很难察觉。
 */
export function focusRun(runId) {
  if (!runId) return false;
  if (state.streamRunId && state.streamRunId !== runId) return false;
  resetRunView();
  state.runId = runId;
  logActions?.sync();
  setBadge("查看日志", "run");
  openStream(runId);
  return true;
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
    logActions?.sync();
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
    case "phase": {
      // 启动阶段：前端把每一项累加成清单，train_start 时折算总耗时。
      // 不写进日志（日志要留给训练本身），只在进度面板里展示。
      progress.phases.push({
        phase: event.phase || "unknown",
        label: event.label || event.phase || "未知阶段",
        elapsed_ms: Number(event.elapsed_ms ?? 0),
        step_ms: Number(event.step_ms ?? 0),
        warm: Boolean(event.warm),
      });
      if (event.warm) progress.warmHit = true;
      renderStartupSteps();
      renderStartupSummary();
      if (event.phase && event.phase !== "spawned") {
        setStage(`启动中：${event.label || event.phase}`, "run");
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
      if (name === "train_start") {
        // 启动阶段到此结束，按钮文案恢复成「当前模式」对应的样子
        setStage("训练中", "run");
        refreshTrainButton();
        renderStartupSummary();
      }
      if (name === "warm_ready") {
        // 预热进程就绪只对「下一次点击」有意义，不改变当前任务状态
        renderWarmStatus({ enabled: true, kinds: [{ kind: dom.kind.value, state: "ready" }] });
      }
      break;
    }
    case "amp_state": {
      // 实际生效的精度。必须显示「实际」而不是「请求」：黑名单显卡上请求 amp=True
      // 也会被框架关掉，用户看到的慢一倍只有这一个解释来源。
      const actual = event.precision === "fp16"
        ? "fp16（混合精度已启用）"
        : event.precision === "fp32"
          ? "fp32（未启用混合精度）"
          : "未知";
      const requested = event.amp_requested === undefined || event.amp_requested === null
        ? "未指定"
        : String(event.amp_requested);
      appendLog(
        `混合精度：实际 ${actual}，请求 amp=${requested}，GPU=${event.gpu || "-"}`,
        event.precision === "fp32" ? "warn" : "ok",
      );
      setMetric("精度", event.precision || "unknown");
      break;
    }
    case "vram_peak": {
      if (event.peak_mb) {
        appendLog(
          `峰值显存：${event.peak_mb} MB（已写入实测峰值表，下次预检用实测值判断）`,
          "ok",
        );
        setMetric("峰值显存 MB", String(event.peak_mb));
      }
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

// ---------------------------------------------------------------------------
// 启动清单与预热状态
// ---------------------------------------------------------------------------
// 点下「开始训练」到真正开训之间有几段固定开销（起进程 / 导入依赖 / 载权重 /
// 扫数据集）。以前这段只显示「运行中」，用户无法判断是在正常加载还是卡死了。
// 现在每完成一段就追加一行，并在 train_start 时折叠成一句总耗时。

const PHASE_ORDER = ["spawned", "boot", "deps", "model", "data", "train_start"];

function renderStartupSteps() {
  if (!dom.startupSteps) return;
  if (!progress.phases.length) {
    dom.startupSteps.hidden = true;
    return;
  }
  dom.startupSteps.hidden = false;
  dom.startupSteps.innerHTML = "";
  const seen = new Map(progress.phases.map((item) => [item.phase, item]));
  for (const phase of PHASE_ORDER) {
    const item = seen.get(phase);
    // 只画已经发生的阶段：没到的阶段画成灰色会让「进度」看着像假的
    if (!item) continue;
    const li = document.createElement("li");
    if (item.warm) li.classList.add("is-warm");
    const name = document.createElement("span");
    name.className = "step-name";
    name.textContent = item.label || phase;
    const cost = document.createElement("span");
    cost.className = "step-cost";
    // spawned 是「进程刚起来」的标记，没有可测量的耗时
    cost.textContent = item.phase === "spawned" || !item.step_ms
      ? ""
      : `${(item.step_ms / 1000).toFixed(2)}s`;
    li.append(name, cost);
    dom.startupSteps.appendChild(li);
  }
}

function renderStartupSummary() {
  if (!dom.startupSummary) return;
  const start = progress.phases.find((item) => item.phase === "train_start");
  if (!start) {
    dom.startupSummary.hidden = true;
    return;
  }
  const total = (start.elapsed_ms / 1000).toFixed(2);
  const parts = progress.phases
    .filter((item) => item.phase !== "spawned" && item.phase !== "train_start" && item.step_ms)
    .map((item) => `${item.label} ${(item.step_ms / 1000).toFixed(2)}s`);
  dom.startupSummary.hidden = false;
  dom.startupSummary.textContent =
    `启动用时 ${total} 秒` +
    (progress.warmHit ? "（命中热启动，已跳过依赖导入）" : "") +
    (parts.length ? `：${parts.join(" / ")}` : "");
}

async function refreshWarmStatus(prewarm = false) {
  try {
    if (prewarm) await postJSON("/api/warm/prewarm", {});
    const status = await getJSON("/api/warm/status");
    renderWarmStatus(status);
  } catch {
    if (dom.warmBadge) dom.warmBadge.hidden = true;
  }
}

function renderWarmStatus(status) {
  const badge = dom.warmBadge;
  if (!badge) return;
  if (!status || !status.enabled) {
    badge.hidden = true;
    return;
  }
  const kind = dom.kind.value === "yolo" ? "yolo" : "anomalib";
  const item = (status.kinds || []).find((entry) => entry.kind === kind);
  badge.hidden = false;
  if (!item || item.state === "idle") {
    badge.textContent = "预热未运行";
    badge.className = "badge warn";
  } else if (item.state === "warming") {
    badge.textContent = "预热中…";
    badge.className = "badge run";
  } else {
    badge.textContent = "预热就绪";
    badge.className = "badge done";
  }
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

function startProgressView(elapsed = 0) {
  stopProgressTimer();
  progress.total = 0;
  progress.epoch = 0;
  // 接续已跑了一段时间的任务时，把起点往前挪：否则「已用」会从 0 重新数，
  // 和顶栏条上的时长对不上，用户会以为接错了任务。
  progress.startedAt = Date.now() - Math.max(0, Number(elapsed) || 0) * 1000;
  progress.lastEpochAt = 0;
  progress.epochDeltas = [];
  progress.pct = 0;
  progress.timer = setInterval(() => renderElapsed(false), 1000);
  // 「正在启动」而不是「运行中」：此时训练循环还没开始，
  // 显示「运行中」会让人以为已经在训练了，其实在加载依赖。
  setStage("正在启动", "run");
  renderEpochText();
  renderElapsed();
  renderStartupSteps();
  renderStartupSummary();
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
  // 只要有一行内容，「复制」就该可点。这里不重复读 innerText（很贵）
  logActions?.markText();
}

/** 指标名的中文显示：分类只有 top1，别直接怼一串英文给用户 */
const METRIC_LABELS = {
  accuracy_top1: "top1 准确率",
  "metrics/accuracy_top1": "top1 准确率",
};

function setMetric(name, value) {
  const label = METRIC_LABELS[name] || name;
  const id = `metric-${name}`;
  let chip = document.getElementById(id);
  if (!chip) {
    chip = document.createElement("div");
    chip.className = "metric";
    chip.id = id;
    chip.dataset.name = name;
    dom.metrics.appendChild(chip);
  }
  chip.innerHTML = `${escapeHtml(label)} <b>${escapeHtml(String(value))}</b>`;
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
  progress.phases = [];
  progress.warmHit = false;
  setStage("待启动");
  if (dom.startupSteps) {
    dom.startupSteps.hidden = true;
    dom.startupSteps.innerHTML = "";
  }
  if (dom.startupSummary) {
    dom.startupSummary.hidden = true;
    dom.startupSummary.textContent = "";
  }
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

// ---------------------------------------------------------------------------
// 开训前体检（v1.14）：数据集概览 / 增强预览 / 断点续训
// ---------------------------------------------------------------------------
// 三块都是「只读看一眼」的功能：不写文件、不改训练参数，
// 唯一的副作用是扫描时多读几次磁盘，因此全部按需触发，不做自动轮询。

/** 表单里的 15 项增强参数，空串表示「用 launcher 默认值」。 */
function augmentFormValues() {
  const map = {
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
  };
  const params = {};
  for (const [key, id] of Object.entries(map)) {
    // 空串也照传：后端把空串理解为「默认值」，并在响应里回显真正用了多少
    params[key] = formValue(id);
  }
  return params;
}

// ---- 数据集概览 -----------------------------------------------------------
async function scanDataset() {
  const source = formValue("yoloData");
  if (!source) {
    setOverviewHint("先填 dataset.yaml 路径", "bad");
    return;
  }
  dom.btnScanData.disabled = true;
  setOverviewHint("扫描中…（几万张的目录会慢一点）");
  try {
    // refresh 传 false：后端按「yaml 路径 + mtime + size」缓存 60 秒。
    // 重新导出数据集会重写 dataset.yaml，缓存自然失效；连点两次不会重扫几万张。
    const data = await postJSON("/api/train/dataset-overview", { source, refresh: false });
    precheck.overview = data;
    renderOverview(data);
    setOverviewHint(`扫描完成，用时 ${data.elapsed_ms} ms${data.cached ? "（数据集未变，命中缓存）" : ""}`, "ok");
  } catch (error) {
    setOverviewHint(error.message, "bad");
  } finally {
    dom.btnScanData.disabled = false;
  }
}

function setOverviewHint(text, cls = "") {
  dom.overviewHint.textContent = text;
  dom.overviewHint.className = `hint ${cls}`.trim();
}

/** 渲染概览：计数格 + 每类实例条形 + 分辨率 + 警告清单。 */
function renderOverview(data) {
  dom.overviewBody.hidden = false;

  const splits = data.splits || {};
  const classify = data.task === "classify";
  const cells = [
    ["图片总数", data.counts.total],
    [classify ? "已分类图片" : "标注框总数", data.counts.instances],
    ["类别数", (data.classes || []).length],
  ];
  for (const name of ["train", "val", "test"]) {
    const stat = splits[name];
    if (!stat) continue;
    cells.push([
      name,
      stat.configured
        ? (classify ? `${stat.images} 张` : `${stat.images} 图 / ${stat.instances} 框`)
        : "未配置",
    ]);
  }

  dom.overviewCounts.innerHTML = "";
  for (const [label, value] of cells) {
    const box = document.createElement("div");
    box.className = "stat";
    const name = document.createElement("span");
    name.className = "stat-label";
    name.textContent = label;
    const number = document.createElement("b");
    number.textContent = String(value);
    box.append(name, number);
    dom.overviewCounts.appendChild(box);
  }

  // 每类实例数：条形长度相对最多的类，0 用红色标出来
  const classes = data.classes || [];
  const counts = data.class_instances || {};
  dom.overviewBars.innerHTML = "";
  const maxCount = Math.max(1, ...classes.map((item) => counts[String(item.id)] || 0));
  for (const item of classes) {
    const value = counts[String(item.id)] || 0;
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.name;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = value === 0 ? "bar-fill is-zero" : "bar-fill";
    fill.style.width = `${(value / maxCount) * 100}%`;
    track.appendChild(fill);
    const number = document.createElement("span");
    number.className = "bar-value";
    number.textContent = String(value);
    row.append(label, track, number);
    dom.overviewBars.appendChild(row);
  }

  const res = data.resolutions || {};
  if (res.count) {
    const top = (res.top || [])[0];
    dom.overviewResolution.textContent =
      `train 分辨率：宽 ${res.width[0]}~${res.width[2]}（中位 ${res.width[1]}），` +
      `高 ${res.height[0]}~${res.height[2]}（中位 ${res.height[1]}）` +
      (top ? `；最常见 ${top.width}×${top.height}（${top.count} 张）` : "");
  } else {
    dom.overviewResolution.textContent = "分辨率：没读到可用图片尺寸";
  }

  dom.overviewWarnings.innerHTML = "";
  const warnings = data.warnings || [];
  dom.overviewWarnings.hidden = warnings.length === 0;
  for (const item of warnings) {
    const li = document.createElement("li");
    li.className = `warn-${item.level}`;
    li.textContent = item.text;
    dom.overviewWarnings.appendChild(li);
  }
}

// ---- 增强预览 -------------------------------------------------------------
async function renderPreview({ reseed = false } = {}) {
  const data = formValue("yoloData");
  if (!data) {
    setPreviewHint("先填 dataset.yaml 路径", "bad");
    return;
  }
  if (reseed || precheck.previewSeed === null) {
    // seed 由前端产生并回显：用户想复现某次预览时可以把数字记下来
    precheck.previewSeed = Math.floor(Math.random() * 1e9);
  }
  dom.btnPreview.disabled = true;
  dom.btnPreviewSeed.disabled = true;
  dom.btnPreviewNext.disabled = true;
  setPreviewHint("生成中…");
  try {
    const result = await postJSON("/api/train/augment-preview", {
      data,
      params: augmentFormValues(),
      index: precheck.previewIndex,
      seed: precheck.previewSeed,
      imgsz: Number(formValue("yoloImgsz")) || 640,
      split: "train",
    });
    dom.previewBody.hidden = false;
    dom.previewOrigin.src = result.origin;
    dom.previewAug.src = result.augmented;
    dom.previewAugCap.textContent =
      `增强后（第 ${result.index + 1}/${result.total} 张，seed ${result.seed}）`;
    renderPreviewApplied(result);
    setPreviewHint(`已生成，输入尺寸 ${result.imgsz}`, "ok");
  } catch (error) {
    setPreviewHint(error.message, "bad");
  } finally {
    dom.btnPreview.disabled = false;
    dom.btnPreviewSeed.disabled = false;
    dom.btnPreviewNext.disabled = false;
  }
}

function setPreviewHint(text, cls = "") {
  dom.previewHint.textContent = text;
  dom.previewHint.className = `hint ${cls}`.trim();
}

/** 本次到底触发了什么、什么没生效、哪些参数压根不参与——全部照实列出来。 */
function renderPreviewApplied(result) {
  dom.previewApplied.innerHTML = "";
  const lines = [];
  for (const item of result.applied || []) {
    lines.push(["warn-info", `生效：${item.name} —— ${item.detail}`]);
  }
  for (const item of result.skipped || []) {
    lines.push(["warn-info", `未生效：${item.name} —— ${item.detail}`]);
  }
  for (const item of result.ignored || []) {
    lines.push(["warn-warn", `不参与预览：${item.name} —— ${item.detail}`]);
  }
  if (result.fell_back?.length) {
    lines.push(["warn-warn", `这些值不是数字，已按默认值渲染：${result.fell_back.join(", ")}`]);
  }
  dom.previewApplied.hidden = lines.length === 0;
  for (const [cls, text] of lines) {
    const li = document.createElement("li");
    li.className = cls;
    li.textContent = text;
    dom.previewApplied.appendChild(li);
  }
}

// ---- 算力体检（显存 + 混合精度）--------------------------------------------
// 只读探测：走 GET /api/train/preflight，不启动训练、不占显存。
// 不自动轮询：可用显存随时在变，频繁探测没有意义；用户点按钮或改档位时算一次就够。

async function scanHardware() {
  const imgsz = formValue("yoloImgsz") || "640";
  const batch = formValue("yoloBatch") || "16";
  const amp = formValue("ampMode") || "auto";
  const dataPath = formValue("yoloData");
  dom.btnCheckHw.disabled = true;
  setHwHint("探测中…");
  try {
    const query = new URLSearchParams({ imgsz, batch, amp }).toString();
    // 数据集指纹与显存体检一起取：用户点「检查算力」时最关心的就是
    // 「这台机器跑不跑得动」和「跑的是哪批数据」，两个问题一起回答。
    const [data, fp] = await Promise.all([
      getJSON(`/api/train/preflight?${query}`),
      dataPath
        ? getJSON(`/api/dataset/fingerprint?${new URLSearchParams({ source: dataPath })}`).catch(
            (error) => ({ error: error.message }),
          )
        : Promise.resolve(null),
    ]);
    precheck.hardware = data;
    renderHardware(data, fp);
  } catch (error) {
    setHwHint(error.message, "bad");
  } finally {
    dom.btnCheckHw.disabled = false;
  }
}

function setHwHint(text, cls = "") {
  dom.hwHint.textContent = text;
  dom.hwHint.className = `hint ${cls}`.trim();
}

/** 渲染体检结论：GPU / 显存需求与依据 / 混合精度 / 数据集指纹。 */
function renderHardware(data, fp = null) {
  dom.hwBody.hidden = false;
  const vram = data.vram || {};
  const amp = data.amp || {};
  const gpu = vram.gpu || amp.gpu;

  // 第一行：显卡与「表里有多少样本」。样本数就是这套估算的可信度来源，
  // 用户看到「0 条」就知道这次是按公式估的，可以不信那么多。
  const total = vram.total_mb ? `，总 ${fmtMB(vram.total_mb)} MB` : "";
  dom.hwGpu.textContent = gpu
    ? `GPU：${gpu}${total}｜实测峰值表：${data.sample_count ?? 0} 条组合`
    : `GPU：未检测到（没有 CUDA 或 nvidia-smi）`;

  // 第二行：显存结论。必须写清「依据是实测还是公式」，否则用户没法判断该不该信。
  const free = vram.free_mb == null ? null : fmtMB(vram.free_mb);
  const need = vram.needed_mb == null ? null : fmtMB(vram.needed_mb);
  const source = vram.source === "measured"
    ? `按实测峰值 ${fmtMB(vram.measured_mb)} MB × 1.15 + 400 MB`
    : vram.source === "scaled"
      ? `按 batch=${vram.sample_batch} 的实测峰值 ${fmtMB(vram.measured_mb)} MB 线性外推 × 1.15 + 400 MB`
      : vram.source === "capped"
        ? `取更大 batch 的实测峰值 ${fmtMB(vram.measured_mb)} MB 作上界 × 1.15 + 400 MB`
        : "按公式 2600 + 300 × batch × (imgsz/640)²（无实测样本）";
  if (vram.skipped) {
    dom.hwVram.textContent = `显存：${vram.skipped}`;
  } else if (free === null || need === null) {
    dom.hwVram.textContent = "显存：无法读取（没有 CUDA 或 nvidia-smi），启动时不做预检";
  } else {
    const verdict = vram.ok ? "够用" : "不够用";
    const hint = vram.hint ? `｜建议：${vram.hint}` : "";
    dom.hwVram.textContent = `显存：可用 ${free} MB / 需求 ${need} MB → ${verdict}（${source}${hint}）`;
  }

  // 第三行：混合精度。黑名单卡上「请求 fp16 也一定会 fp32」，必须在开训前就说明白，
  // 否则用户只会觉得「同一组参数怎么慢了一倍」。
  const parts = [`混合精度：${data.amp_text || "未知"}`];
  if (amp.available === false) {
    parts.push(amp.reason || "");
    parts.push("即使选「显式尝试」也无法开启，关闭发生在框架内部");
  } else if (amp.available === true) {
    parts.push("训练开始时框架还会做一次前向对比，极小概率仍回退 fp32");
  }
  dom.hwAmp.textContent = parts.filter(Boolean).join("｜");

  // 第四行：本次将用哪批数据。写清算法覆盖范围，避免被当成「图片内容校验」。
  // 指纹不同就意味着这次训练的结果和旧台账不能直接横比。
  if (fp && fp.dataset_fingerprint) {
    const counts = fp.counts || {};
    dom.hwData.textContent =
      `数据集：${fp.short}（${counts.train ?? 0} train / ${counts.val ?? 0} val）` +
      `｜算法 ${fp.algo}：路径+大小+标注，不含图片内容`;
    dom.hwData.title = `完整指纹 ${fp.dataset_fingerprint}｜标注指纹 ${fp.label_fingerprint || "—"}`;
  } else if (fp && fp.error) {
    dom.hwData.textContent = `数据集：指纹计算失败（${fp.error}）`;
    dom.hwData.title = "";
  } else {
    dom.hwData.textContent = "数据集：未选择 dataset.yaml，无法记录指纹";
    dom.hwData.title = "";
  }

  if (vram.ok) {
    setHwHint("体检通过", "ok");
    dom.btnForceVram.hidden = true;
  } else {
    setHwHint("显存不足：可以调小 batch/imgsz，或停掉常驻推理进程", "bad");
  }
}

/** MB 数字的显示格式：小于 10000 保留一位小数，避免「5460.4 MB」这种噪声。 */
function fmtMB(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num >= 1000 ? String(Math.round(num)) : num.toFixed(1);
}

// ---- 断点续训 -------------------------------------------------------------
async function scanRuns() {
  dom.btnScanRuns.disabled = true;
  setResumeHint("扫描中…");
  try {
    const project = formValue("project");
    const query = project ? `?project=${encodeURIComponent(project)}` : "";
    const data = await getJSON(`/api/runs/checkpoints${query}`);
    renderRuns(data);
    setResumeHint(`共 ${data.total} 次训练，列出 ${data.shown} 条`, "ok");
  } catch (error) {
    setResumeHint(error.message, "bad");
  } finally {
    dom.btnScanRuns.disabled = false;
  }
}

function setResumeHint(text, cls = "") {
  dom.resumeHint.textContent = text;
  dom.resumeHint.className = `hint ${cls}`.trim();
}

function renderRuns(data) {
  dom.resumeRows.innerHTML = "";
  dom.resumeTableBox.hidden = (data.items || []).length === 0;
  for (const item of data.items || []) {
    const tr = document.createElement("tr");

    const nameCell = document.createElement("td");
    const name = document.createElement("div");
    name.textContent = item.name;
    const meta = document.createElement("div");
    meta.className = "row-sub";
    meta.textContent = [item.task, item.imgsz ? `${item.imgsz}px` : "", item.optimizer]
      .filter(Boolean)
      .join(" · ");
    nameCell.append(name, meta);
    nameCell.title = item.run_dir;
    tr.appendChild(nameCell);

    const doneCell = document.createElement("td");
    doneCell.className = "num";
    doneCell.textContent = `${item.epochs_done}/${item.epochs_planned ?? "?"}`;
    tr.appendChild(doneCell);

    const mapCell = document.createElement("td");
    mapCell.className = "num";
    mapCell.textContent = item.best_map === null ? "—" : fmtNum(item.best_map);
    tr.appendChild(mapCell);

    const weightCell = document.createElement("td");
    weightCell.className = "num";
    weightCell.textContent = item.last_pt ? `${item.last_pt_mb} MB` : "缺失";
    tr.appendChild(weightCell);

    const bestCell = document.createElement("td");
    bestCell.className = "num";
    bestCell.textContent = item.best_pt ? `${item.best_pt_mb} MB` : "缺失";
    tr.appendChild(bestCell);

    const actionCell = document.createElement("td");
    // 两个入口语义不同，按钮必须分开：
    //   继续训练（last.pt）= 恢复优化器与轮次，参数以检查点为准；
    //   用它微调（best.pt）= 只载入权重，优化器与轮次重置，参数按表单。
    const resumeButton = document.createElement("button");
    resumeButton.type = "button";
    resumeButton.className = "mini";
    resumeButton.textContent = "继续训练";
    resumeButton.disabled = !item.resumable;
    if (item.blocked_reason) resumeButton.title = item.blocked_reason;
    resumeButton.addEventListener("click", () => applyResume(item));

    const finetuneButton = document.createElement("button");
    finetuneButton.type = "button";
    finetuneButton.className = "mini";
    finetuneButton.style.marginLeft = "4px";
    finetuneButton.textContent = "用它微调";
    finetuneButton.disabled = !item.finetunable;
    if (item.finetune_blocked_reason) finetuneButton.title = item.finetune_blocked_reason;
    finetuneButton.addEventListener("click", () => applyFinetune(item));

    actionCell.append(resumeButton, finetuneButton);
    if (item.blocked_reason && !item.resumable) {
      const why = document.createElement("div");
      why.className = "row-sub";
      why.textContent = item.blocked_reason;
      actionCell.appendChild(why);
    }
    tr.appendChild(actionCell);

    dom.resumeRows.appendChild(tr);
  }
}

/** 主按钮文案由当前模式决定，避免任务结束后被写死成「开始训练」。 */
function refreshTrainButton() {
  if (precheck.finetuneFrom) {
    dom.btnTrain.textContent = "微调训练（从 best.pt）";
  } else if (precheck.resumeFrom) {
    const epochs = precheck.resumeEpochs;
    dom.btnTrain.textContent = epochs
      ? `继续训练（从第 ${epochs} 轮）`
      : "继续训练";
  } else {
    dom.btnTrain.textContent = "开始训练";
  }
}

/** 选中一条历史：回填表单并把主按钮改成「继续训练」。 */
function applyResume(item) {
  precheck.resumeFrom = item.last_pt;
  precheck.resumeLabel = `${item.name}（第 ${item.epochs_done} 轮后）`;
  precheck.resumeEpochs = item.epochs_done;
  // 续训与微调互斥：否则两个起点会同时拼进命令行
  precheck.finetuneFrom = "";
  precheck.finetuneLabel = "";
  precheck.finetuneSource = null;
  precheck.previewIndex = 0;

  document.getElementById("yoloData").value = item.data;
  document.getElementById("yoloModel").value = item.last_pt;
  if (item.task) document.getElementById("yoloTask").value = item.task;
  if (item.imgsz) document.getElementById("yoloImgsz").value = String(item.imgsz);
  // project 取 run_dir 的父目录，name 取原实验名：这样续训写回同一个目录
  const runDir = item.run_dir.replace(/[\\/]+$/, "");
  const cut = Math.max(runDir.lastIndexOf("\\"), runDir.lastIndexOf("/"));
  if (cut > 0) {
    dom.project.value = runDir.slice(0, cut);
    dom.runName.value = runDir.slice(cut + 1);
  }

  dom.resumeActive.hidden = false;
  dom.resumeActive.textContent =
    `已选择续训点：${precheck.resumeLabel} → ${item.last_pt}。` +
    "epochs / 优化器等参数以该检查点内记录的为准，表单上改这些不会生效。";
  dom.btnClearResume.hidden = false;
  refreshTrainButton();
  setHint("已进入续训模式；只想改参数重新练的话，点「取消续训/微调」");
  appendLog(`已选择续训点：${item.last_pt}`, "ok");
}

/** 选中一条历史：用它训练好的 best.pt 当起点做微调（参数按表单生效）。 */
function applyFinetune(item) {
  if (!item.best_pt) return;
  precheck.finetuneFrom = item.best_pt;
  precheck.finetuneLabel = item.name;
  precheck.finetuneSource = item;
  precheck.resumeFrom = "";
  precheck.resumeLabel = "";
  precheck.resumeEpochs = undefined;
  precheck.previewIndex = 0;

  if (item.data) document.getElementById("yoloData").value = item.data;
  document.getElementById("yoloModel").value = item.best_pt;
  if (item.task) document.getElementById("yoloTask").value = item.task;
  if (item.imgsz) document.getElementById("yoloImgsz").value = String(item.imgsz);

  // 写回同一实验目录（后端也会按 weights 路径再推导一次，避免前端状态过期）
  const runDir = item.run_dir.replace(/[\\/]+$/, "");
  const cut = Math.max(runDir.lastIndexOf("\\"), runDir.lastIndexOf("/"));
  if (cut > 0) {
    dom.project.value = runDir.slice(0, cut);
    dom.runName.value = runDir.slice(cut + 1);
  }

  // 保守超参：微调不是重新训练。学习率与轮次按「小步修正」给，
  // mosaic 降下来避免小缺陷被拼图抹掉。每一项都写进日志，用户能看到改了什么。
  const preset = {};
  if (item.epochs_done && item.epochs_done > 0) preset.yoloEpochs = "30";
  preset.yoloLr0 = "0.001";
  preset.augMosaic = "0.3";
  preset.augCloseMosaic = "15";
  const applied = [];
  for (const [id, value] of Object.entries(preset)) {
    const el = document.getElementById(id);
    if (el && el.value !== value) {
      el.value = value;
      applied.push(`${id}=${value}`);
    }
  }

  dom.resumeActive.hidden = false;
  dom.resumeActive.textContent =
    `已选择微调起点：${item.name} 的 best.pt → ${item.best_pt}。` +
    "优化器与轮次重置，表单参数生效；结果写回原实验目录，旧结果会先备份到 _prev/。";
  dom.btnClearResume.hidden = false;
  refreshTrainButton();
  setHint("已进入微调模式；提交前会先检查类别与任务头是否兼容");
  appendLog(`已选择微调起点：${item.best_pt}`, "ok");
  if (applied.length) appendLog(`已套用小步微调参数：${applied.join(", ")}`, "ok");
}

function clearResume() {
  precheck.resumeFrom = "";
  precheck.resumeLabel = "";
  precheck.resumeEpochs = undefined;
  precheck.finetuneFrom = "";
  precheck.finetuneLabel = "";
  precheck.finetuneSource = null;
  dom.resumeActive.hidden = true;
  dom.resumeActive.textContent = "";
  dom.btnClearResume.hidden = true;
  refreshTrainButton();
  setHint("");
  appendLog("已取消续训/微调，下次提交按普通训练处理", "warn");
}
