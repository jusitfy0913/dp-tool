// 引擎构建页：ONNX -> TensorRT 引擎。
//
// 与训练页的区别：
// - trtexec 只输出普通文本日志，没有 TRTFORGE_JSON 结构事件，指标层为空；
// - **trtexec 不提供进度**。整个 tactic 选择阶段一行日志都不打（实测 PCB 960×960
//   从 `Local timing cache in use` 到 `Engine built in` 空白 473 秒），所以
//   界面上的百分比是按「历史耗时 + 阶段边界」估算的，不是真实进度。
//   估算逻辑与基线在 backend/progress.py，改口径只改那边，前端只负责呈现；
// - 为了让它看起来「活着」，已用时间与预计剩余由本地 1s 定时器刷新，
//   不依赖后端事件间隔（后端心跳是 2s 一条，且只推整数百分比变化）；
// - 日志量可达上千行，用 textContent 整体替换比逐行 append 更快。
//
// 事件流本身复用 /ws/runs/{id}，和训练页是同一条通道。

import { getJSON, listRuns, postJSON, stopRun } from "./api.js";
import { createLogActions } from "./logs.js";
import { connectRunEvents } from "./ws.js";

// 只保留最近若干行，防止长时间构建把内存和 DOM 拖慢
const MAX_LINES = 2000;

// 阶段清单必须与 backend/progress.py 的 STAGE_ORDER 一一对应（顺序也要一致）。
// 对不上时以事件里的 stage_label 为准，前端不会因此报错。
const STAGES = [
  { id: "start", label: "启动" },
  { id: "parse", label: "解析 ONNX" },
  { id: "parsed", label: "解析完成" },
  { id: "build", label: "构建 / tactic 搜索" },
  { id: "built", label: "序列化引擎" },
  { id: "deserialize", label: "校验引擎" },
  { id: "benchmark", label: "推理基准" },
  { id: "done", label: "完成" },
];

// ---------------------------------------------------------------------------
// 纯函数：输出引擎文件名推导（导出给 tests/test_engine_out.mjs，不依赖 DOM）
// ---------------------------------------------------------------------------
const ENGINE_SUFFIXES = /\.(engine|trt|plan)$/i;

/**
 * 由 ONNX 路径推导默认引擎文件名 `<onnx名>_<精度>.engine`。
 *
 * 必须与后端 build_engine 的默认推导一致，否则「留空自动推导」和
 * 「浏览目录后自动补名」会给出两个不同的文件名，用户会以为选错了。
 */
export function engineOutName(onnxPath, tag) {
  if (!onnxPath) return "";
  const base = onnxPath.replace(/^.*[\\/]/, "").replace(/\.onnx$/i, "");
  if (!base) return "";
  return `${base}_${tag}.engine`;
}

/**
 * 把「输出引擎」输入框的值规范成完整文件路径。
 *
 * 为什么需要它：浏览弹窗受浏览器限制只能选目录，选中后字段里只有一个目录
 * （例如 `D:\test\deeplearning\dp-tool`），后端会以
 * 「输出文件名需以 .engine / .trt / .plan 结尾」拒绝，用户卡在这里。
 * 这里把目录补成 `目录\<onnx名>_<精度>.engine`，让「选目录」也能直接构建。
 *
 * 规则：
 * - 空值：返回空字符串，保持「留空自动推导」的语义；
 * - 已带 .engine / .trt / .plan：原样返回（用户手工改名时不干扰）；
 * - .onnx 结尾：换后缀（贴错成 onnx 路径也能救回来）；
 * - 其它（视为目录）：补上推导出的文件名；没有路径成分时不猜，交后端报错。
 */
export function normalizeEngineOut(value, onnxPath, tag) {
  const raw = (value || "").trim();
  if (!raw) return "";
  if (ENGINE_SUFFIXES.test(raw)) return raw;
  if (/\.onnx$/i.test(raw)) return raw.replace(/\.onnx$/i, `_${tag}.engine`);
  const dir = raw.replace(/[\\/]+$/, "");
  if (!/[\\/]/.test(dir)) return raw;
  const sep = dir.includes("\\") ? "\\" : "/";
  return `${dir}${sep}${engineOutName(onnxPath, tag) || `model_${tag}.engine`}`;
}

export function initEngine() {
  const form = document.getElementById("engineForm");
  const onnxInput = document.getElementById("engOnnx");
  const outInput = document.getElementById("engOut");
  const precision = document.getElementById("engPrecision");
  const workspace = document.getElementById("engWorkspace");
  const dynamic = document.getElementById("engDynamic");
  const shapeGroup = document.getElementById("engShapeGroup");
  const shapeInputs = {
    min: document.getElementById("engMin"),
    opt: document.getElementById("engOpt"),
    max: document.getElementById("engMax"),
  };
  const extra = document.getElementById("engExtra");
  const probeHint = document.getElementById("engProbe");
  const hint = document.getElementById("engineHint");
  const logBox = document.getElementById("engineLog");
  const copyLogBtn = document.getElementById("btnCopyBuildLog");
  const openLogBtn = document.getElementById("btnOpenBuildLog");
  const badge = document.getElementById("engineBadge");
  const btnBuild = document.getElementById("btnBuild");
  const btnStop = document.getElementById("btnBuildStop");
  const stepsBox = document.getElementById("engSteps");
  const bar = document.getElementById("engBar");
  const pctText = document.getElementById("engPct");
  const stageText = document.getElementById("engStage");
  const elapsedText = document.getElementById("engElapsed");
  const etaText = document.getElementById("engEta");
  const estimateNote = document.getElementById("engEstimateNote");

  let active = null; // { runId, stream }
  let lines = [];
  //: 最近一次构建的 run。构建结束后 active 清空，但磁盘上的日志还在，
  //: 「日志文件」按钮应该继续指向它。
  let lastBuildRunId = null;

  const logActions = createLogActions({
    copyBtn: copyLogBtn,
    openBtn: openLogBtn,
    // 这里日志是用 textContent 整段重写的，取 textContent 就够了
    getText: () => logBox.textContent || "",
    getRunId: () => active?.runId || lastBuildRunId,
  });

  // 进度视图状态：时间基准放在本地，刷新/事件延迟都不会让数字停住
  const view = {
    estimate: null, // 估算总秒数；null 表示后端没给
    startedAt: 0, // Date.now() 毫秒
    pct: 0,
    stageIndex: -1,
    timer: null,
  };

  // -------------------------------------------------------------------------
  // 辅助
  // -------------------------------------------------------------------------
  function setBadge(text, cls) {
    badge.textContent = text;
    badge.className = cls ? `badge ${cls}` : "badge";
  }

  function setHint(text, bad = false) {
    hint.textContent = text || "";
    hint.className = bad ? "hint bad" : "hint";
  }

  function autoName(path) {
    if (!path) return "留空自动推导为 <onnx名>_fp16.engine";
    const tag = precision.value === "fp16" ? "fp16" : "fp32";
    return engineOutName(path, tag) || "留空自动推导为 <onnx名>_fp16.engine";
  }

  function fmtShape(shape) {
    return Array.isArray(shape) ? `[${shape.join(", ")}]` : "?";
  }

  /** 秒 -> `m:ss`（超过一小时才带小时），用于「已用 / 剩余」 */
  function fmtClock(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    const ss = String(sec).padStart(2, "0");
    if (h) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
    return `${m}:${ss}`;
  }

  /** 秒 -> 中文口语时长，用于「预计约 8 分 7 秒」这类说明文字 */
  function fmtSpoken(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const m = Math.floor(total / 60);
    if (m >= 1) return `${m} 分 ${total % 60} 秒`;
    return `${total} 秒`;
  }

  function render() {
    if (lines.length > MAX_LINES) lines = lines.slice(-MAX_LINES);
    logBox.textContent = lines.join("\n");
    logBox.scrollTop = logBox.scrollHeight;
    // 有内容就能复制。markText 只在「还没启用」时写一次 DOM，不会每行都动
    logActions.markText();
  }

  // -------------------------------------------------------------------------
  // 进度面板
  // -------------------------------------------------------------------------
  function renderSteps(activeIndex) {
    stepsBox.innerHTML = "";
    STAGES.forEach((stage, index) => {
      const chip = document.createElement("span");
      let cls = "build-step";
      if (index < activeIndex) cls += " done";
      else if (index === activeIndex) cls += " current";
      chip.className = cls;
      chip.textContent = stage.label;
      stepsBox.appendChild(chip);
    });
  }

  /** 每秒刷新一次时间：即使后端 2 秒才推一条事件，数字也在动 */
  function tickClock() {
    if (!view.startedAt) return;
    const elapsed = (Date.now() - view.startedAt) / 1000;
    elapsedText.textContent = `已用 ${fmtClock(elapsed)}`;
    if (view.estimate) {
      const left = Math.max(0, view.estimate - elapsed);
      etaText.textContent = `预计剩余 ${fmtClock(left)}`;
    }
  }

  function startTicker() {
    stopTicker();
    view.timer = setInterval(tickClock, 1000);
    tickClock();
  }

  function stopTicker() {
    if (view.timer) {
      clearInterval(view.timer);
      view.timer = null;
    }
  }

  function resetProgress(estimate) {
    view.estimate = Number.isFinite(estimate) && estimate > 0 ? estimate : null;
    view.startedAt = Date.now();
    view.pct = 0;
    view.stageIndex = -1;
    bar.style.width = "0%";
    pctText.textContent = "0%";
    stageText.textContent = "启动中";
    elapsedText.textContent = "已用 0:00";
    etaText.textContent = "预计剩余 —";
    renderSteps(-1);
    // 口径必须写在界面上：否则用户会以为这是 trtexec 报的真实进度
    estimateNote.textContent = view.estimate
      ? `预计约 ${fmtSpoken(view.estimate)}。trtexec 在 tactic 搜索阶段不输出任何日志，` +
        `这里的百分比是按历史构建耗时推算的估算值，不是真实进度；已用时间每秒在走，说明没卡死。`
      : "trtexec 不提供进度，也没有历史耗时可用，只能显示阶段与已用时间。";
    estimateNote.className = "hint";
  }

  function applyProgress(payload) {
    const pct = Number(payload.pct);
    if (Number.isFinite(pct)) {
      view.pct = Math.min(100, Math.max(0, pct));
      bar.style.width = `${view.pct.toFixed(1)}%`;
      pctText.textContent = `${view.pct.toFixed(0)}%`;
    }
    const index = STAGES.findIndex((stage) => stage.id === payload.stage);
    if (index >= 0) {
      view.stageIndex = index;
      // 结束阶段把「完成」这一格也打勾，否则收尾时会有一格停在 current
      renderSteps(payload.stage === "done" ? STAGES.length : index);
    }
    if (payload.stage_label || payload.stage) {
      stageText.textContent = payload.stage_label || payload.stage;
    }
    // 后端给的 elapsed/eta 更可靠（它掌握进程真实起点），优先采用
    if (typeof payload.elapsed === "number") {
      elapsedText.textContent = `已用 ${fmtClock(payload.elapsed)}`;
    }
    if (typeof payload.eta === "number") {
      etaText.textContent = `预计剩余 ${fmtClock(payload.eta)}`;
    } else if (payload.estimate === null) {
      etaText.textContent = "预计剩余 —";
    }
  }

  // -------------------------------------------------------------------------
  // 选完 ONNX 自动探测布局，把「未转置」这类问题提前暴露
  // -------------------------------------------------------------------------
  async function probe() {
    const path = onnxInput.value.trim();
    if (!path) {
      probeHint.textContent = "选好文件会自动探测输入/输出形状。";
      probeHint.className = "hint";
      return;
    }
    probeHint.textContent = "探测中…";
    probeHint.className = "hint";
    try {
      const data = await getJSON(`/api/engine/probe?onnx=${encodeURIComponent(path)}`);
      const bad = Boolean(data.untransposed_yolo);
      probeHint.textContent =
        `${data.size_mb} MB ｜ 输入 ${fmtShape(data.probe.input_shape)} ｜ ` +
        `输出 ${fmtShape(data.probe.output_shape)} ｜ ${data.verdict}`;
      probeHint.className = bad ? "hint bad" : "hint";
      // 输出路径留空时，实时展示会落到哪个文件名
      if (!outInput.value.trim()) outInput.placeholder = autoName(path);
    } catch (error) {
      probeHint.textContent = `探测失败：${error.message}`;
      probeHint.className = "hint bad";
    }
  }

  onnxInput.addEventListener("change", probe);
  onnxInput.addEventListener("blur", () => {
    if (onnxInput.value.trim()) probe();
  });
  precision.addEventListener("change", () => {
    if (!outInput.value.trim()) outInput.placeholder = autoName(onnxInput.value.trim());
  });

  // 输出引擎可能是「浏览」塞进来的纯目录，提交前必须补成文件名，
  // 否则后端只会回一句「输出文件名需以 .engine / .trt / .plan 结尾」。
  // 浏览弹窗选中目录时走 change（main.js 里补发），手工输入后离开走 blur。
  function normalizeOut(announce) {
    const before = outInput.value.trim();
    if (!before) return;
    const tag = precision.value === "fp16" ? "fp16" : "fp32";
    const after = normalizeEngineOut(before, onnxInput.value.trim(), tag);
    if (!after || after === before) return;
    outInput.value = after;
    if (announce) {
      setHint(`输出引擎已补全为 ${after.split(/[\\/]/).pop()}，文件名可手工改；留空则自动推导`);
    }
  }

  outInput.addEventListener("change", () => normalizeOut(true));
  outInput.addEventListener("blur", () => normalizeOut(false));

  dynamic.addEventListener("change", () => {
    shapeGroup.classList.toggle("hidden", !dynamic.checked);
  });

  // -------------------------------------------------------------------------
  // 提交构建
  // -------------------------------------------------------------------------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (active) return;

    const body = {
      onnx: onnxInput.value.trim(),
      engine: outInput.value.trim(),
      precision: precision.value,
      dynamic: dynamic.checked,
      min_shape: shapeInputs.min.value.trim(),
      opt_shape: shapeInputs.opt.value.trim(),
      max_shape: shapeInputs.max.value.trim(),
      workspace_mb: Number(workspace.value) || 4096,
      extra_args: extra.value.trim(),
    };

    if (!body.onnx) {
      setHint("请先选择 ONNX 文件", true);
      return;
    }
    if (body.dynamic && !(body.min_shape && body.opt_shape && body.max_shape)) {
      setHint("启用动态 shape 时必须同时填 min / opt / max", true);
      return;
    }

    // 再兜一次规范化：用户可能改完没离开输入框就直接点「开始构建」，
    // 那种情况下 blur/change 还没跑，目录会原样送到后端被拒。
    if (body.engine) {
      body.engine = normalizeEngineOut(body.engine, body.onnx, body.precision === "fp16" ? "fp16" : "fp32");
      outInput.value = body.engine;
    }

    setHint("");
    lines = [];
    render();
    logActions.reset();
    btnBuild.disabled = true;
    setBadge("启动中", "run");

    try {
      const run = await postJSON("/api/engine/build", body);
      active = { runId: run.run_id, stream: null };
      logActions.sync();
      setBadge("构建中", "run");
      btnStop.disabled = false;
      setHint(`输出目标：${run.engine}`);

      resetProgress(run.estimate_seconds);
      startTicker();

      lines.push(`$ ${run.command.join(" ")}`);
      lines.push("");
      render();

      bindStream(run.run_id, { engine: body.engine, onnx: body.onnx });
    } catch (error) {
      stopTicker();
      setHint(error.message, true);
      setBadge("失败", "fail");
      lines.push(`[错误] ${error.message}`);
      render();
      btnBuild.disabled = false;
      btnStop.disabled = true;
      // 409 = 机器上已经有任务在跑（见后端 _reject_if_busy）。
      // 比起让用户自己去找停止按钮，直接接管那个任务更省事。
      if (error.status === 409) attachIfRunning();
    }
  });

  /**
   * 把一个 run 的事件流接到界面上。
   *
   * 抽成函数是因为「刚点开始」和「刷新后接管」走的是同一条通道：
   * 两处各抄一份 onClose 里那套收尾文案，早晚会改漏其中一处。
   */
  function bindStream(runId, { engine, onnx }) {
    btnBuild.disabled = true;
    btnStop.disabled = false;
    active.stream = connectRunEvents(runId, {
      onEvent: (payload) => {
        if (payload.type === "log") {
          lines.push(payload.line);
        } else if (payload.type === "progress") {
          applyProgress(payload);
        } else if (payload.type === "error") {
          lines.push(`[错误] ${payload.message}`);
        }
        render();
      },
      onClose: (payload) => {
        stopTicker();
        const ok = payload.status === "finished";
        // 主动停止不是「失败」：进度停在中断处，用户在意的正是「停在哪、花了多久」
        const cancelled = payload.status === "cancelled";
        setBadge(cancelled ? "已停止" : ok ? "已完成" : "失败", cancelled ? "warn" : ok ? "done" : "fail");
        lines.push("");
        lines.push(`[结束] 退出码=${payload.exit} 状态=${payload.status}`);
        render();
        btnBuild.disabled = false;
        btnStop.disabled = true;
        // 结束后依然能打开完整日志：面板只留最近 MAX_LINES 行
        lastBuildRunId = active?.runId || lastBuildRunId;
        active = null;
        logActions.sync();

        const spent = view.startedAt ? (Date.now() - view.startedAt) / 1000 : null;
        if (ok) {
          if (view.pct < 100) applyProgress({ pct: 100, stage: "done", stage_label: "完成" });
          // 把「实际 vs 估算」摆出来，用户才知道下次该不该信这个数字
          const compare = spent != null && view.estimate
            ? `｜实际 ${fmtSpoken(spent)}，估算 ${fmtSpoken(view.estimate)}`
            : spent != null
              ? `｜实际 ${fmtSpoken(spent)}`
              : "";
          setHint(`构建完成：${engine || autoName(onnx)}${compare}`);
          estimateNote.textContent =
            `构建耗时已记入 runtime/engine_stats.json，下次同类构建的估算会更准。`;
          estimateNote.className = "hint ok";
        } else if (cancelled) {
          setHint(`已停止${spent != null ? `，已用 ${fmtSpoken(spent)}` : ""}（中断的构建不产出引擎）`, true);
          estimateNote.textContent =
            "进度条停在中断位置。若中断前已打出 `Engine built in`，那段真实耗时仍会计入统计。";
          estimateNote.className = "hint";
        } else {
          setHint("构建失败，看日志尾部定位原因", true);
          estimateNote.textContent =
            "进度条停在失败发生的位置。若失败前已打出 `Engine built in`，那段真实耗时仍会计入统计。";
          estimateNote.className = "hint bad";
        }
      },
      onError: (message) => {
        lines.push(`[事件流] ${message}`);
        render();
      },
    });
  }

  /**
   * 拉一次任务列表，有构建任务在跑就接管。
   *
   * 引擎构建动辄十几分钟，页面刷新后不接管的话，用户看到的是一块空进度条
   * 和可点的「开始构建」——再点一次就是第二个 trtexec 写同一个 .engine。
   */
  async function attachIfRunning() {
    try {
      const data = await listRuns();
      const runs = data?.runs || [];
      const run = runs.find((item) => item.status === "running" && item.kind === "engine-build");
      if (run) {
        attachRunningRun(run);
        return;
      }
      // 没有在跑：把最近一次构建的日志留着可查（失败的那次最需要）
      const last = runs.find((item) => item.kind === "engine-build");
      if (last) {
        lastBuildRunId = last.run_id;
        logActions.sync();
        openLogBtn.title = `最近一次构建 ${last.run_id} 的完整日志（${last.status}）`;
      }
    } catch {
      /* 由顶栏条负责报「后端未连接」 */
    }
  }

  function attachRunningRun(run) {
    if (!run || run.kind !== "engine-build" || active) return false;
    active = { runId: run.run_id, stream: null };
    logActions.reset();
    logActions.sync();
    setBadge("构建中", "run");
    setHint(`已接续正在运行的构建（run ${run.run_id}）`);

    resetProgress(run.estimate_seconds ?? run.meta?.estimate_seconds);
    // 时钟接着真实起点走：否则「已用」会从 0 重数，看起来像刚启动
    view.startedAt = Date.now() - Math.max(0, Number(run.elapsed) || 0) * 1000;
    startTicker();

    lines = [];
    if (Array.isArray(run.command) && run.command.length) {
      lines.push(`$ ${run.command.join(" ")}`);
      lines.push("");
    }
    lines.push(`[接续] 正在从后端补发日志…`);
    render();

    bindStream(run.run_id, { engine: run.meta?.engine || "", onnx: run.meta?.onnx || onnxInput.value.trim() });
    return true;
  }

  document.addEventListener("trtf:attach-run", (event) => {
    const run = event.detail?.run;
    if (run && run.kind === "engine-build") attachRunningRun(run);
  });

  attachIfRunning();

  btnStop.addEventListener("click", async () => {
    if (!active) return;
    // 先摆出「正在停止」，最终文案由 onClose 统一写（那里才知道是 cancelled 还是 failed）
    setHint("正在停止…");
    btnStop.disabled = true;
    try {
      await stopRun(active.runId);
    } catch (error) {
      setHint(error.message, true);
      btnStop.disabled = false;
    }
  });

  // 首屏就把阶段清单画出来，用户一眼知道要经过哪几步
  renderSteps(-1);
}
