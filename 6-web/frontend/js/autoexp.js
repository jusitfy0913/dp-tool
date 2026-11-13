// 自动实验（第 4 阶段：LLM 决策层）页面。
//
// 设计前提（改这个文件前先读一遍）：
//   1) 这是**外置模块**。它不碰 /api/train/yolo，也不写训练页的任何状态。
//      两者唯一的交集是同一套 GPU 互斥锁——所以不会同时抢显卡。
//   2) 模型只提建议；KEEP / DISCARD 由后端按噪声阈值算死。
//      因此界面上「模型提议」和「判定」永远是分开的两块，不能被合并成一句话，
//      否则用户会误以为「模型说好就是好」。
//   3) 日志里的决策块是核心交付物。用户要能一眼看出：这一轮模型想改什么、
//      为什么、结果算不算数。所以事件不是拼成一行 JSON，而是分行渲染。

import { getJSON, postJSON } from "./api.js";
import { connectRunEvents } from "./ws.js";

const API = "/api/autoexp";

const dom = {};
// 当前接管的 run_id 与事件流句柄
let currentRunId = "";
let watcher = null;
// 本轮统计（从事件流里累计，后端不复述）
let stats = { round: 0, rounds: 0, kept: 0, startedAt: 0 };

const MAX_LINES = 3000;

function q(id) {
  return document.getElementById(id);
}

function appendLine(text, cls = "") {
  if (!dom.log) return;
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  dom.log.appendChild(line);
  while (dom.log.childElementCount > MAX_LINES) {
    dom.log.removeChild(dom.log.firstElementChild);
  }
  dom.log.scrollTop = dom.log.scrollHeight;
  if (dom.copyLog) dom.copyLog.disabled = false;
}

/** 决策块：一条粗分隔线 + 标题，后面的行属于同一轮。 */
function beginBlock(title) {
  appendLine("");
  appendLine(`──────── ${title} ────────`, "ae-block");
}

function fmtClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

// ---------------------------------------------------------------------------
// 纯函数（导出给 tests/test_autoexp_fmt.mjs）
// ---------------------------------------------------------------------------
// 抽出来的原因：决策块的文字是这一版的核心交付物，用户全靠它判断
// 「模型想改什么、这一轮算不算数」。这类格式化逻辑不该埋在事件回调里
// 只能靠手点页面验证。

/** 差值显示：正数带 +，保留 6 位（与台账、审计 JSON 的精度一致）。 */
export function formatDelta(value) {
  if (value == null || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${number >= 0 ? "+" : ""}${number.toFixed(6)}`;
}

/**
 * 判定行的文字与样式类。
 *
 * 注意 DISCARD 用次要色而不是红色：它不是「错误」，只是「没超过噪声」。
 * 一整晚几十轮里大部分都会是 DISCARD，全标红会让真正的错误被淹没。
 */
export function verdictLine(event) {
  const verdict = String(event?.verdict || "");
  const expId = event?.exp_id || "—";
  const delta = formatDelta(event?.delta);
  if (verdict === "KEEP") {
    return { text: `[判定] KEEP  ${expId}  ${delta}`, cls: "ae-keep", tone: "ok" };
  }
  return { text: `[判定] DISCARD  ${expId}  ${delta}`, cls: "ae-discard", tone: "" };
}

/** 轮数估算文字。一轮 = 600 秒训练 + 约 60~90 秒启动与验证开销。 */
export function estimateBudget(rounds, hours, perRoundMinutes = 11) {
  const count = Number(rounds) || 0;
  const cap = Number(hours) || 0;
  const minutes = (count * perRoundMinutes);
  const estimate = minutes >= 60
    ? `${(minutes / 60).toFixed(1)} 小时`
    : `${minutes.toFixed(0)} 分钟`;
  const capText = cap > 0 ? `，但到 ${cap} 小时会强制停` : "，不限时间";
  return `一轮约 ${perRoundMinutes} 分钟（600 秒训练 + 启动与验证开销）。`
    + `${count} 轮 ≈ ${estimate}${capText}。`;
}

function setStage(text, tone = "") {
  if (!dom.stage) return;
  dom.stage.textContent = text;
  dom.stage.className = `badge${tone ? ` ${tone}` : ""}`;
}

function setRunBadge(text, tone = "") {
  if (!dom.runBadge) return;
  dom.runBadge.textContent = text;
  dom.runBadge.className = `badge${tone ? ` ${tone}` : ""}`;
}

function setRunning(running) {
  if (dom.start) dom.start.disabled = running;
  if (dom.stop) dom.stop.disabled = !running;
}

function renderProgress() {
  if (dom.round) {
    dom.round.textContent = stats.rounds
      ? `轮次 ${stats.round}/${stats.rounds}`
      : "轮次 -/-";
  }
  if (dom.kept) dom.kept.textContent = `KEEP ${stats.kept}`;
  if (dom.elapsed && stats.startedAt) {
    dom.elapsed.textContent = `已用 ${fmtClock((Date.now() - stats.startedAt) / 1000)}`;
  }
  if (dom.progressBar) {
    const pct = stats.rounds ? Math.min(100, (stats.round / stats.rounds) * 100) : 0;
    dom.progressBar.style.width = `${pct.toFixed(1)}%`;
  }
}

// ---------------------------------------------------------------------------
// 日志渲染：把结构化事件翻成人能读的决策块
// ---------------------------------------------------------------------------

function handleAgentEvent(event) {
  switch (event.name) {
    case "agent_start": {
      stats.rounds = Number(event.rounds) || 0;
      stats.round = 0;
      stats.kept = 0;
      stats.startedAt = Date.now();
      beginBlock("自动实验开始");
      appendLine(`模型：${event.model || "—"}`);
      if (event.max_tokens) {
        appendLine(
          `输出预算 max_tokens=${event.max_tokens}`
          + (Number(event.max_tokens) < 1500 ? "（偏小：推理型模型容易被截断）" : ""),
          Number(event.max_tokens) < 1500 ? "warn" : "hint"
        );
      }
      appendLine(
        `keep 阈值：${Number(event.keep_threshold).toFixed(6)}`
        + `（sigma=${event.sigma == null ? "未测" : Number(event.sigma).toFixed(6)}）`,
        "hint"
      );
      appendLine(
        `计划：最多 ${stats.rounds} 轮，时间上限 ${event.hours || "不限"} 小时。`
        + " 中途要用网页训练可先点停止。",
        "hint"
      );
      setStage("准备中", "run");
      renderProgress();
      break;
    }
    case "round_start": {
      stats.round = Number(event.round) || 0;
      stats.rounds = Number(event.rounds) || stats.rounds;
      beginBlock(`第 ${stats.round}/${stats.rounds} 轮`);
      setStage("等模型提议", "run");
      renderProgress();
      break;
    }
    case "agent_proposal": {
      // 这是本功能的核心可见性要求：模型想改什么、为什么，必须原样显示出来。
      appendLine("[模型提议]", "ae-propose");
      appendLine(`  假设：${event.hypothesis || "（未给出）"}`);
      if (event.changed) appendLine(`  改动：${event.changed}`);
      if (event.params && event.params !== "{}") appendLine(`  参数：${event.params}`);
      if (event.reasoning) appendLine(`  理由：${event.reasoning}`, "hint");
      setStage("训练中", "run");
      break;
    }
    case "agent_rejected": {
      appendLine(`[提议被拒] 第 ${event.attempt} 次，原因：`, "warn");
      for (const reason of event.reasons || []) appendLine(`  - ${reason}`, "warn");
      break;
    }
    case "agent_error": {
      if (event.code === "OUTPUT_TRUNCATED") {
        // 这类错误必须给出「改哪个数字」，否则用户只会去怀疑模型或提示词。
        appendLine("[输出被截断] 模型还没写完正文就用完了输出预算", "err");
        appendLine(`  ${event.message || ""}`, "err");
        appendLine("  处理：到「模型接口」把 max_tokens 调到 4096 以上，再重新开始。", "warn");
      } else {
        appendLine(`[调用失败] ${event.code || ""} ${event.message || ""}`, "err");
      }
      setStage("接口出错", "err");
      break;
    }
    case "agent_wait_gpu": {
      appendLine(`[等待显卡] 第 ${event.waits} 次等待，60 秒后重试`, "warn");
      setStage("等待显卡", "warn");
      break;
    }
    case "agent_stop": {
      // 三种停止原因含义完全不同，文案不能共用：
      // stale=搜索到瓶颈；errors=模型侧故障（以前这两者被报成同一句话）
      const tones = { stale: "warn", errors: "err", deadline: "hint" };
      appendLine(`[停止] ${event.message || event.code || ""}`, tones[event.code] || "warn");
      break;
    }
    case "agent_verdict": {
      const line = verdictLine(event);
      if (String(event.verdict || "") === "KEEP") stats.kept += 1;
      appendLine(line.text, line.cls);
      appendLine(`      ${event.reason || ""}`, "hint");
      appendLine(
        `      阈值 ${Number(event.threshold).toFixed(6)}；连续无进展 ${event.stale ?? 0} 轮`,
        "hint"
      );
      setStage(line.tone === "ok" ? "有新最佳" : "继续", line.tone);
      renderProgress();
      // 每轮结束刷一次台账，用户不用手动刷新页面
      refreshHistory();
      break;
    }
    case "agent_summary": {
      beginBlock("自动实验结束");
      appendLine(
        `完成 ${event.completed} 轮，KEEP ${event.kept} 次`
        + (event.errors ? `，失败 ${event.errors} 轮` : "")
        + `，用时 ${fmtClock(event.elapsed_seconds || 0)}`
      );
      setStage("已结束", "ok");
      break;
    }
    default:
      break;
  }
}

function handleEvent(payload) {
  if (payload.type === "agent") {
    handleAgentEvent(payload);
    return;
  }
  if (payload.type === "finished") {
    setRunning(false);
    setRunBadge(payload.status === "done" ? "已完成" : "已结束",
      payload.status === "done" ? "ok" : "");
    setStage("已停止", "");
    currentRunId = "";
    watcher?.close();
    watcher = null;
    refreshStatus();
    refreshHistory();
    return;
  }
  if (payload.type === "log" && payload.line) {
    // autoexp 自己的 [autoexp] 行与 launcher 的原始输出都从这里过。
    // 决策块已经单独渲染过，这里只补充原始上下文，不做二次加工。
    appendLine(payload.line.replace(/\s+$/, ""));
    return;
  }
  if (payload.type === "progress") {
    // launcher 在 autoexp 里也会发这条（见 cmd_exec 里注册的回调）
    const epoch = payload.epoch ?? payload.total;
    if (payload.epoch != null && payload.total) {
      setStage(`训练中 ${payload.epoch}/${payload.total} 轮`, "run");
    } else if (epoch != null) {
      setStage("训练中", "run");
    }
    return;
  }
  if (payload.type === "epoch_summary") {
    // 只留一行最关键的：mAP50-95。完整指标在台账里，日志不必全抄一遍。
    const metrics = payload.metrics || {};
    const key = Object.keys(metrics).find((name) => name.startsWith("metrics/mAP50-95"));
    const value = key == null ? null : Number(metrics[key]);
    if (value != null && Number.isFinite(value)) {
      appendLine(
        `  epoch ${payload.epoch}/${payload.total}  mAP50-95 ${value.toFixed(5)}`,
        "hint"
      );
    }
    return;
  }
  if (payload.type === "error") {
    appendLine(`错误 ${payload.code || ""}：${payload.message || ""}`, "err");
  }
}

/** 接管一个 run 的事件流。页面刷新后也能靠 status.active_run 接回来。 */
function attachRun(run) {
  if (!run || !run.run_id) return;
  if (currentRunId === run.run_id) return;
  watcher?.close();
  currentRunId = run.run_id;
  setRunning(true);
  setRunBadge("运行中", "run");
  stats.startedAt = stats.startedAt || Date.now();
  watcher = connectRunEvents(run.run_id, {
    onEvent: handleEvent,
    onError: (text) => appendLine(text, "warn"),
    onClose: () => setRunning(false),
  });
}

// ---------------------------------------------------------------------------
// 台账
// ---------------------------------------------------------------------------

//: 指纹配色板。只用于「区分组」，不承载好坏语义——所以不用红/绿。
const FP_COLORS = ["#4f8cff", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6", "#ef4444"];

/** 由指纹字符串稳定地映射到一个颜色：同一批数据永远同色。 */
function fpColor(fp) {
  let hash = 0;
  for (const ch of String(fp || "")) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return FP_COLORS[hash % FP_COLORS.length];
}

function renderRows(rows) {
  if (!dom.rows) return;
  dom.rows.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "hint";
    td.textContent = "还没有实验记录。点「开始自动实验」后，这里会逐轮出现。";
    tr.appendChild(td);
    dom.rows.appendChild(tr);
    return;
  }
  // 数据指纹分组：同一列里不同指纹的行，参数可能完全相同但数据不同。
  // 在这种行之间比 mAP 等于把「换数据」误读成「改参数有效」，所以必须显式区分。
  const fpSet = new Set(rows.map((row) => row.data_fp).filter(Boolean));
  if (dom.fpHint) {
    if (fpSet.size > 1) {
      dom.fpHint.hidden = false;
      dom.fpHint.className = "hint bad";
      dom.fpHint.textContent =
        `本台账出现 ${fpSet.size} 组不同的数据集指纹（${[...fpSet].join(" / ")}）。` +
        "跨组的行不可直接横向比较：数据变了，指标差异不一定是参数带来的。";
    } else {
      dom.fpHint.hidden = true;
    }
  }
  // 最新的在最上面：跑一整晚之后没人愿意滚到底部看最新一轮
  for (const row of [...rows].reverse()) {
    const tr = document.createElement("tr");

    const cellId = document.createElement("td");
    cellId.textContent = row.exp_id || "—";
    if (row.data_fp) {
      // 左边框上色：一眼看出哪些行属于同一批数据
      tr.style.borderLeft = `3px solid ${fpColor(row.data_fp)}`;
      cellId.title = `数据集指纹 ${row.data_fp}（数据不同的实验不可直接横比）`;
      const tag = document.createElement("span");
      tag.className = "fp-tag";
      tag.textContent = row.data_fp;
      cellId.appendChild(document.createElement("br"));
      cellId.appendChild(tag);
    } else {
      cellId.title = "未记录数据集指纹（升级前的实验，无法核对数据是否相同）";
    }

    const cellChange = document.createElement("td");
    cellChange.className = "ae-cell-change";
    const hypothesis = row.changed || row.hypothesis || "";
    cellChange.textContent = hypothesis || "（基线 / 未记录）";
    if (row.reason) cellChange.title = row.reason;

    const cellFitness = document.createElement("td");
    cellFitness.textContent = row.fitness_map50_95 === "" ? "—"
      : Number(row.fitness_map50_95).toFixed(5);

    const cellDelta = document.createElement("td");
    if (row.delta == null || row.delta === "") {
      cellDelta.textContent = "—";
    } else {
      const value = Number(row.delta);
      cellDelta.textContent = `${value >= 0 ? "+" : ""}${value.toFixed(5)}`;
      cellDelta.className = value > 0 ? "ae-keep" : "ae-discard";
    }

    const cellVerdict = document.createElement("td");
    const verdict = row.verdict || (row.status === "crash" ? "CRASH" : "—");
    cellVerdict.textContent = verdict;
    if (verdict === "KEEP") cellVerdict.className = "ae-keep";
    else if (verdict === "DISCARD") cellVerdict.className = "ae-discard";
    else if (verdict === "CRASH") cellVerdict.className = "err";

    tr.append(cellId, cellChange, cellFitness, cellDelta, cellVerdict);
    dom.rows.appendChild(tr);
  }
}

async function refreshHistory() {
  try {
    const data = await getJSON(`${API}/history?limit=80`);
    renderRows(data.rows || []);
    if (dom.ledgerCount) dom.ledgerCount.textContent = `${data.total || 0} 条`;
  } catch (error) {
    // 历史读不到不影响正在跑的实验，只是表格空着
    if (dom.ledgerCount) dom.ledgerCount.textContent = "读取失败";
  }
}

// ---------------------------------------------------------------------------
// 状态与配置
// ---------------------------------------------------------------------------

async function refreshStatus() {
  let status;
  try {
    status = await getJSON(`${API}/status`);
  } catch (error) {
    if (dom.startHint) dom.startHint.textContent = `后端未响应：${error.message}`;
    if (dom.start) dom.start.disabled = true;
    return;
  }

  if (dom.threshold) {
    const noise = status.noise || {};
    dom.threshold.textContent = noise.keep_threshold == null
      ? "未测（先跑 noise 子命令）"
      : `${Number(noise.keep_threshold).toFixed(6)}（σ=${Number(noise.sigma).toFixed(6)}）`;
  }

  // 让用户知道「不满足前提」时到底缺什么，而不是按钮灰着不给理由
  const problems = [];
  if (!status.available) problems.push("找不到 7-autoresearch-yolo/autoexp.py");
  if (!status.splits_ready) problems.push("还没有切分数据（先跑 autoexp.py split）");
  if (status.llm && status.llm.key_locked) {
    problems.push("API Key 需要在这台电脑上重新输入（换过电脑或 Windows 账户）");
  } else if (status.llm && !status.llm.has_key) {
    problems.push("还没填 API Key");
  }
  if (problems.length) {
    if (dom.startHint) dom.startHint.textContent = `暂时不能开始：${problems.join("；")}`;
  } else if (dom.startHint) {
    dom.startHint.textContent = `台账 ${status.ledger_rows} 条；当前最佳 ${status.ledger_best?.exp_id || "无"}`;
  }

  if (dom.keyBadge) {
    const locked = Boolean(status.llm?.key_locked);
    const hasKey = Boolean(status.llm?.has_key);
    dom.keyBadge.textContent = locked
      ? "需重输"
      : (hasKey ? status.llm.key_masked : "未配置");
    dom.keyBadge.className = `badge${hasKey ? " ok" : " warn"}`;
  }

  // 页面刷新后把正在跑的任务接回来（这是「关掉标签页不丢进度」的关键）
  if (status.active_run) {
    attachRun(status.active_run);
  } else if (!currentRunId) {
    setRunning(false);
    if (dom.runBadge?.textContent === "未开始") setStage("待启动");
  }
  return status;
}

async function refreshLlmConfig() {
  try {
    const cfg = await getJSON(`${API}/llm-config`);
    if (dom.baseUrl) dom.baseUrl.value = cfg.base_url || "";
    if (dom.model) dom.model.value = cfg.model || "";
    if (dom.temperature) dom.temperature.value = cfg.temperature ?? 0.7;
    if (dom.maxTokens) dom.maxTokens.value = cfg.max_tokens ?? 4096;
    if (dom.keyBadge) {
      // 「解不开」必须和「没配」分开显示：前者数据还在，只是换机/换账户，
      // 用户该做的是重输一次；后者是从没配过。两者提示完全不同。
      const locked = Boolean(cfg.key_locked);
      dom.keyBadge.textContent = locked
        ? "需重输"
        : (cfg.has_key ? cfg.key_masked : "未配置");
      dom.keyBadge.className = `badge${cfg.has_key ? " ok" : " warn"}`;
    }
    if (dom.llmHint) {
      if (cfg.key_locked) {
        dom.llmHint.textContent =
          "已保存的 Key 在这台电脑上解不开（换过电脑或 Windows 账户）。" +
          "请重新输入一次，覆盖后即可正常使用。";
      } else if (cfg.has_key) {
        dom.llmHint.textContent =
          `已保存：${cfg.key_masked}（本机 DPAPI 加密，不会回填明文，改 Key 直接覆盖）`;
      } else {
        dom.llmHint.textContent = "尚未保存 Key。也可以不用页面，改成设置环境变量后重启服务。";
      }
    }
  } catch (error) {
    if (dom.llmHint) dom.llmHint.textContent = `读取配置失败：${error.message}`;
  }
}

async function saveLlm() {
  if (dom.llmHint) dom.llmHint.textContent = "保存中…";
  try {
    const payload = {
      base_url: dom.baseUrl?.value.trim() || "",
      model: dom.model?.value.trim() || "",
      temperature: dom.temperature ? Number(dom.temperature.value) : null,
      // 显式传数字：null 表示「不改」，与 temperature 同一套语义
      max_tokens: dom.maxTokens?.value ? Number(dom.maxTokens.value) : null,
    };
    const key = dom.apiKey?.value.trim() || "";
    // 空串表示「不改动已保存的 Key」——后端按这个语义实现，别在这里传 null
    if (key) payload.api_key = key;
    await postJSON(`${API}/llm-config`, payload);
    if (dom.apiKey) dom.apiKey.value = "";
    if (dom.llmHint) dom.llmHint.textContent = "已保存。建议点一下「测试连通性」再开跑。";
    await refreshLlmConfig();
    await refreshStatus();
  } catch (error) {
    if (dom.llmHint) dom.llmHint.textContent = `保存失败：${error.message}`;
  }
}

async function testLlm() {
  if (dom.llmHint) dom.llmHint.textContent = "正在请求模型（最多约 2 分钟）…";
  try {
    const result = await postJSON(`${API}/llm-test`, {});
    if (result.ok) {
      if (dom.llmHint) {
        dom.llmHint.textContent = `连通正常：${result.model || ""} 回复「${result.reply || ""}」，${result.latency_ms} ms`;
      }
    } else if (dom.llmHint) {
      dom.llmHint.textContent = `连接失败：${result.error}`;
    }
  } catch (error) {
    if (dom.llmHint) dom.llmHint.textContent = `测试失败：${error.message}`;
  }
}

async function previewPrompt() {
  if (dom.previewCard) dom.previewCard.hidden = false;
  if (dom.previewText) dom.previewText.textContent = "正在生成（不调用 API）…";
  try {
    const result = await getJSON(`${API}/prompt-preview`);
    if (dom.previewText) dom.previewText.textContent = result.text || "（无输出）";
  } catch (error) {
    if (dom.previewText) dom.previewText.textContent = `生成失败：${error.message}`;
  }
}

async function startAgent() {
  if (dom.startHint) dom.startHint.textContent = "正在启动…";
  if (dom.start) dom.start.disabled = true;
  try {
    const run = await postJSON(`${API}/start`, {
      rounds: Number(dom.rounds?.value || 30),
      hours: Number(dom.hours?.value || 8),
      max_stale: Number(dom.maxStale?.value || 5),
      max_errors: Number(dom.maxErrors?.value || 3),
      lock_wait: Number(dom.lockWait?.value || 3600),
    });
    stats = { round: 0, rounds: 0, kept: 0, startedAt: Date.now() };
    if (dom.log) dom.log.textContent = "";
    beginBlock("已提交");
    appendLine("进程已启动，等待第一轮事件…");
    attachRun(run);
    if (dom.startHint) dom.startHint.textContent = `已启动（run ${run.run_id}）`;
  } catch (error) {
    // 409 是「已有任务在跑」，这条信息要原样给出来：用户正需要知道是谁占着
    if (dom.startHint) dom.startHint.textContent = `启动失败：${error.message}`;
    setRunning(false);
  }
}

async function stopAgent() {
  if (dom.stop) dom.stop.disabled = true;
  if (dom.startHint) dom.startHint.textContent = "正在停止（会杀掉整棵训练进程树）…";
  try {
    await postJSON(`${API}/stop`, {});
    if (dom.startHint) dom.startHint.textContent = "已停止";
  } catch (error) {
    if (dom.startHint) dom.startHint.textContent = `停止失败：${error.message}`;
  }
  setRunning(false);
  setTimeout(() => {
    refreshStatus();
    refreshHistory();
  }, 800);
}

/** 轮数/小时变化时给一句直观的时长估算，省得用户自己算。 */
function updateBudgetHint() {
  if (!dom.budgetHint) return;
  dom.budgetHint.textContent = estimateBudget(
    dom.rounds?.value || 0,
    dom.hours?.value || 0
  );
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------

export function initAutoexp() {
  dom.log = q("aeLog");
  dom.stage = q("aeStage");
  dom.runBadge = q("aeRunBadge");
  dom.keyBadge = q("aeKeyBadge");
  dom.round = q("aeRound");
  dom.kept = q("aeKept");
  dom.elapsed = q("aeElapsed");
  dom.progressBar = q("aeProgressBar");
  dom.threshold = q("aeThreshold");
  dom.start = q("aeStart");
  dom.stop = q("aeStop");
  dom.startHint = q("aeStartHint");
  dom.llmHint = q("aeLlmHint");
  dom.baseUrl = q("aeBaseUrl");
  dom.model = q("aeModel");
  dom.temperature = q("aeTemperature");
  dom.maxTokens = q("aeMaxTokens");
  dom.apiKey = q("aeApiKey");
  dom.rounds = q("aeRounds");
  dom.hours = q("aeHours");
  dom.maxStale = q("aeMaxStale");
  dom.maxErrors = q("aeMaxErrors");
  dom.lockWait = q("aeLockWait");
  dom.budgetHint = q("aeBudgetHint");
  dom.copyLog = q("aeCopyLog");
  dom.rows = q("aeRows");
  dom.fpHint = q("aeFpHint");
  dom.ledgerCount = q("aeLedgerCount");
  dom.previewCard = q("aePreviewCard");
  dom.previewText = q("aePreviewText");

  q("aeSaveLlm")?.addEventListener("click", saveLlm);
  q("aeTestLlm")?.addEventListener("click", testLlm);
  q("aePreviewPrompt")?.addEventListener("click", previewPrompt);
  q("aeClosePreview")?.addEventListener("click", () => {
    if (dom.previewCard) dom.previewCard.hidden = true;
  });
  dom.start?.addEventListener("click", startAgent);
  dom.stop?.addEventListener("click", stopAgent);
  dom.rounds?.addEventListener("input", updateBudgetHint);
  dom.hours?.addEventListener("input", updateBudgetHint);

  dom.copyLog?.addEventListener("click", async () => {
    if (!dom.log) return;
    try {
      await navigator.clipboard.writeText(dom.log.innerText);
      if (dom.startHint) dom.startHint.textContent = "日志已复制到剪贴板";
    } catch {
      if (dom.startHint) dom.startHint.textContent = "复制失败，请手动选中日志文本";
    }
  });

  setRunning(false);
  updateBudgetHint();
  refreshLlmConfig();
  refreshStatus().then(refreshHistory);

  // 每秒刷一次「已用」，只在有运行中任务时才写 DOM，避免空转
  setInterval(() => {
    if (currentRunId) renderProgress();
  }, 1000);

  // 切到这个页签时重新拉一次状态：用户可能刚在别处停掉了任务
  document.addEventListener("tab-change", (event) => {
    if (event.detail?.name !== "autoexp") return;
    refreshStatus().then(refreshHistory);
  });
}
