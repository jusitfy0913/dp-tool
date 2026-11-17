// 界面小工具：转义、剪贴板、时间格式化、状态徽标。
//
// ## 为什么要有这个文件
//
// 这些函数原本在 8 个模块里各抄一份，抄出来的版本已经开始分叉：
//
// - `escapeHtml` 有 2 份，其中一份的 `text` 参数没做 `String()`，传数字直接崩；
// - 剪贴板有 3 份，`infer.js` 那份**没有 execCommand 兜底**——局域网
//   `http://<同事机器>:8060` 不是安全上下文，`navigator.clipboard` 是 undefined，
//   用户点「复制」没有任何反应也不报错，这类问题只能靠人反复点才发现；
// - `fmtClock` 同名两种语义：`train.js`/`engine.js` 是「时长 m:ss」，
//   `runsbar.js`/`pipeline.js` 是「时间戳 11-11 14:12」。同名不同义是最费解的
//   一类 bug 来源（改一处，另一处静默变了），所以这里按语义拆成两个名字。
//
// ## 约定
//
// - 本文件**只依赖 DOM**，不 import 任何其它前端模块，避免环依赖；
// - `setHint` / `setBadge` 收元素而不是 id，调用方自己决定类名后缀，
//   因为各面板的 CSS 类约定不同（`hint bad` / `hint-warn` / `hidden`），
//   强行统一会改变界面外观，超出「不改行为」的范围。

/** HTML 转义。所有 `innerHTML` 拼接都必须先过这里（值里可能有 `<`）。 */
export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

/**
 * 复制文本到剪贴板。
 *
 * 两级兜底：`navigator.clipboard` 只在安全上下文可用（`http://局域网IP` 没有），
 * 所以必须有 textarea + execCommand 兜底。
 * @returns {Promise<boolean>} 是否真的复制成功（失败要如实告诉用户）
 */
export async function copyText(text) {
  if (!text) return false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // 落到下面的 execCommand 兜底
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  } finally {
    area.remove();
  }
  return ok;
}

/**
 * 时长（秒）→ `m:ss`，超过一小时才带小时。
 * 用于「已用 / 每轮 / 剩余」，**不是**时间戳格式化。
 */
export function fmtDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const ss = String(total % 60).padStart(2, "0");
  if (h) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
  return `${m}:${ss}`;
}

/**
 * 时间戳（epoch 秒）→ `MM-DD HH:mm`（`withYear` 时带年份）。
 * 历史列表用短格式省宽度；串流阶段用长格式，跨天/跨年要能分清。
 */
export function fmtStamp(seconds, { withYear = false } = {}) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) return "";
  const date = new Date(value * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  const clock = `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  return withYear ? `${date.getFullYear()}-${clock}` : clock;
}

/** 设置状态徽标。`cls` 为空时只留 `badge`。 */
export function setBadge(node, text, cls = "") {
  if (!node) return;
  node.textContent = text;
  node.className = `badge ${cls}`.trim();
}

/**
 * 黄色说明（p.hint.tip）平时收起，鼠标放在前一个词条上才展开。
 * 词条是紧挨着的标题、折叠名、输入项或一组输入框。
 */
export function bindHintTips(root = document) {
  root.querySelectorAll("p.hint.tip").forEach((tip) => {
    if (tip.dataset.tipBound === "1") return;
    const anchor = findHintAnchor(tip);
    if (!anchor) return;
    tip.dataset.tipBound = "1";
    anchor.classList.add("has-tip");
    if (anchor.matches("h2, h3") && !anchor.querySelector(".tip-mark")) {
      const mark = document.createElement("span");
      mark.className = "tip-mark";
      mark.textContent = "?";
      anchor.appendChild(mark);
    }
    if (anchor.matches(".grid2, .row")) {
      anchor.querySelectorAll(":scope > label").forEach((label) => label.classList.add("has-tip"));
    }
    const show = () => tip.classList.add("is-open");
    const hideIfLeft = (event) => {
      const next = event.relatedTarget;
      if (next && (anchor.contains(next) || tip.contains(next))) return;
      tip.classList.remove("is-open");
    };
    anchor.addEventListener("mouseenter", show);
    anchor.addEventListener("mouseleave", hideIfLeft);
    tip.addEventListener("mouseleave", hideIfLeft);
    anchor.addEventListener("focusin", show);
    anchor.addEventListener("focusout", hideIfLeft);
  });
}

/** 说明挂在前一个词条上。中间夹着的状态行要跳过。 */
function findHintAnchor(tip) {
  let node = tip.previousElementSibling;
  while (node && node.matches(".hint, .anno-hint")) node = node.previousElementSibling;
  if (node && node.matches("label, summary, h2, h3, .grid2, .row")) return node;
  return tip.parentElement?.querySelector(":scope > h2, :scope > h3, :scope > summary") || node;
}

/**
 * 设置提示行。`cls` 透传给类名（`bad` / `ok` / 空），**不做语义映射**：
 * 各面板的样式名不统一，映射会让「改这里、坏那里」。
 */
export function setHint(node, text, cls = "") {
  if (!node) return;
  node.textContent = text || "";
  node.className = `hint ${cls}`.trim();
}

/**
 * 指标小卡片。
 * @param {boolean} update 为真时按 `metric-<name>` 就地更新，否则总是新增。
 *   进度面板要就地更新（否则每轮多一张卡片），推理结果面板要新增。
 */
export function setMetric(container, name, value, { label = "", update = false } = {}) {
  if (!container) return;
  const text = String(value ?? "");
  if (!update) {
    const chip = document.createElement("div");
    chip.className = "metric";
    chip.innerHTML = `${escapeHtml(name)} <b>${escapeHtml(text)}</b>`;
    container.appendChild(chip);
    return;
  }
  const id = `metric-${name}`;
  let chip = container.querySelector(`#${CSS.escape(id)}`);
  if (!chip) {
    chip = document.createElement("div");
    chip.className = "metric";
    chip.id = id;
    chip.dataset.name = name;
    container.appendChild(chip);
  }
  chip.innerHTML = `${escapeHtml(label || name)} <b>${escapeHtml(text)}</b>`;
}
