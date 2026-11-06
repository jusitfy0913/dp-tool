// 各页左右并排的小窗口：拖中间的竖条按比例拉长或缩短。
//
// 宽度存成容器宽度的百分比，而不是像素。窗口变大变小时两边仍保持同一次拖动
// 定下的比例。双击竖条去掉记录，回到样式表里的默认宽度。
//
// 窄屏样式会把网格收成一列并藏起竖条。这里不在脚本里改 grid-template，
// 只改 CSS 变量，这样媒体查询仍能把整行覆盖掉。

const STORAGE = "trtforge.split.";

// 下面两个是纯函数，单测在 tests/test_splitpanes.mjs 里直接跑。
// 抽出来是因为「拖到底、换屏幕、改 localStorage」这几种情况在浏览器里手测
// 很容易漏，而漏掉的后果是表单被永久压成一条缝，用户只能靠双击猜回来。

// 把一次拖动的结果夹到 [minPx 对应的比例, maxRatio] 之间。
export function clampRatio(ratio, { width, minPx, maxRatio }) {
  if (!(width > 0)) return ratio;
  const minRatio = Math.min(0.8, minPx / width);
  return Math.min(maxRatio, Math.max(minRatio, ratio));
}

// 读回存下来的百分比。返回 null 表示这条记录不能用，应该删掉回默认宽度。
export function readStoredRatio(saved, { width, minPx, maxRatio }) {
  if (!saved) return null;
  const ratio = parseFloat(saved);
  // 8%~78% 之外的一律不信：那是手改过的、或者是几何坏掉的遗留值。
  if (!Number.isFinite(ratio) || ratio < 8 || ratio > 78) return null;
  // 隐藏的页签量不到宽度（0），这时不判像素下限，等可见时 CSS 的 minmax 兜底。
  if (width > 0 && (ratio / 100) * width < minPx - 1) return null;
  // 手改出来的超大值也要收回到拖动上限内，免得右边栏被挤没。
  const capped = typeof maxRatio === "number" ? Math.min(ratio, maxRatio * 100) : ratio;
  return capped;
}

function makeHandle() {
  const handle = document.createElement("div");
  handle.className = "split-handle";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "vertical");
  handle.title = "左右拖动拉长或缩短，双击恢复默认";
  return handle;
}

function bindHandle(handle, { container, variable, edge, minPx, maxRatio, storageKey }) {
  // 存的是百分比。读回来要按「当前」容器宽度折成像素再判断：换过窗口宽度、
  // 或以前拖到过最小宽度（15.61% 这种）时，同一个百分比会差出几百像素。
  // 折出来不够 minPx 就丢掉，回默认宽度——否则一次拖到底会把表单永久压成
  // 240px，输入框全都在横向滚动。
  const width = container.getBoundingClientRect().width;
  const savedRatio = readStoredRatio(localStorage.getItem(STORAGE + storageKey), {
    width,
    minPx,
    maxRatio,
  });
  if (savedRatio !== null) {
    container.style.setProperty(variable, `${savedRatio}%`);
  } else if (localStorage.getItem(STORAGE + storageKey)) {
    localStorage.removeItem(STORAGE + storageKey);
  }

  handle.addEventListener("dblclick", (event) => {
    event.preventDefault();
    container.style.removeProperty(variable);
    localStorage.removeItem(STORAGE + storageKey);
  });

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    // 窄屏下竖条是 display:none，offsetParent 为空，这时不该开始拖。
    if (handle.offsetParent === null && getComputedStyle(handle).display === "none") return;
    event.preventDefault();
    handle.classList.add("is-dragging");
    try {
      handle.setPointerCapture(event.pointerId);
    } catch {
      /* 捕获失败时仍可在竖条内拖，只是移出后会断 */
    }

    const move = (ev) => {
      const rect = container.getBoundingClientRect();
      if (rect.width < 80) return;
      const fromEdge = edge === "right" ? rect.right - ev.clientX : ev.clientX - rect.left;
      const ratio = clampRatio(fromEdge / rect.width, { width: rect.width, minPx, maxRatio });
      container.style.setProperty(variable, `${(ratio * 100).toFixed(2)}%`);
    };
    const finish = () => {
      handle.classList.remove("is-dragging");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", finish);
      handle.removeEventListener("pointercancel", finish);
      const current = container.style.getPropertyValue(variable).trim();
      if (current) localStorage.setItem(STORAGE + storageKey, current);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
    move(event);
  });
}

function insertOnce(container, before) {
  const existing = before.previousElementSibling;
  if (existing && existing.classList.contains("split-handle")) return existing;
  const handle = makeHandle();
  container.insertBefore(handle, before);
  return handle;
}

function panesOf(container) {
  return [...container.children].filter((el) => !el.classList.contains("split-handle"));
}

export function initSplitPanes() {
  document.querySelectorAll(".layout").forEach((container) => {
    const panes = panesOf(container);
    if (panes.length < 2) return;
    // 通栏卡片（.card.wide）本来就要另起一行，前面插个竖条没有意义。
    if (panes[1].classList.contains("wide")) return;
    const key = container.closest(".panel")?.id || "layout";
    const handle = insertOnce(container, panes[1]);
    bindHandle(handle, {
      container,
      variable: "--split-left",
      edge: "left",
      // 340 而不是 240：表单里是 label + input + 浏览按钮，240 时输入框会被
      // 压出横向滚动条。给「能正常填」的下限，而不是「还能看见」的下限。
      minPx: 340,
      maxRatio: 0.72,
      storageKey: `${key}-left`,
    });
  });

  document.querySelectorAll(".anno-layout").forEach((container) => {
    const panes = panesOf(container);
    if (panes.length < 3) return;
    const leftHandle = insertOnce(container, panes[1]);
    const rightHandle = insertOnce(container, panes[2]);
    bindHandle(leftHandle, {
      container,
      variable: "--anno-left",
      edge: "left",
      minPx: 220,
      maxRatio: 0.46,
      storageKey: "annotate-left",
    });
    bindHandle(rightHandle, {
      container,
      variable: "--anno-right",
      edge: "right",
      minPx: 260,
      maxRatio: 0.5,
      storageKey: "annotate-right",
    });
  });

  document.querySelectorAll(".detail-grid").forEach((container) => {
    const panes = panesOf(container);
    if (panes.length < 2) return;
    const handle = insertOnce(container, panes[1]);
    bindHandle(handle, {
      container,
      variable: "--detail-side",
      edge: "right",
      minPx: 180,
      maxRatio: 0.55,
      storageKey: "infer-detail",
    });
  });
}
