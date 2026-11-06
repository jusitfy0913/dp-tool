// 入口：页签切换、环境探测、目录浏览弹窗。

import { browse, cancelNativeDialog, getCapabilities, getHealth, nativeDialog } from "./api.js";
import { initAnnotate } from "./annotate.js";
import { initDataset } from "./dataset.js";
import { initEngine } from "./engine.js";
import { initExport } from "./export.js";
import { initInfer } from "./infer.js";
import { initInferBatch } from "./inferbatch.js";
import { initPipeline } from "./pipeline.js";
import { initRunsBar } from "./runsbar.js";
import { initSplitPanes } from "./splitpanes.js";
import { initTrain } from "./train.js";

// ---------------------------------------------------------------------------
// 页签
// ---------------------------------------------------------------------------
function activateTab(name) {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = Array.from(document.querySelectorAll(".panel"));
  const target = tabs.find((tab) => tab.dataset.tab === name);
  if (!target) return false;
  tabs.forEach((t) => t.classList.toggle("active", t === target));
  panels.forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  // 通知各模块"我这一页现在可见了"。
  // 标注画布必须先可见才能量到真实尺寸，否则会按 0 宽高初始化成 1×1。
  document.dispatchEvent(new CustomEvent("tab-change", { detail: { name } }));
  return true;
}

function initTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activateTab(tab.dataset.tab);
    });
  });
}

// ---------------------------------------------------------------------------
// 环境状态
// ---------------------------------------------------------------------------
// 能力表缓存：原生对话框是否可用，决定「浏览」按钮点下去走系统框还是页面内浏览。
// 缓存一份避免每次点按钮都发一次请求；刷新环境时一并更新。
let capsCache = null;

async function capabilitiesOnce() {
  if (capsCache) return capsCache;
  try {
    capsCache = await getCapabilities();
  } catch {
    // 拿不到就按「不支持原生框」处理：宁可退到页面内浏览，也不要弹一个失败框
    capsCache = {};
  }
  return capsCache;
}

async function refreshVersion() {
  const badge = document.getElementById("appVersion");
  if (!badge) return;
  try {
    const health = await getHealth();
    if (health.version) {
      badge.textContent = `v${health.version}`;
      badge.title = `后端版本 v${health.version}（来自 /api/health）`;
    }
  } catch {
    // 版本号拿不到不影响使用，顶栏状态灯会另外报「后端未连接」。
    badge.title = "未能获取后端版本";
  }
}

async function refreshEnvironment() {
  const dot = document.getElementById("envDot");
  const text = document.getElementById("envText");
  const capBox = document.getElementById("capBox");

  try {
    const caps = await getCapabilities();
    capsCache = caps;
    const problems = [];
    if (!caps.python) problems.push("缺少训练解释器");
    if (!caps.yolo_launcher) problems.push("缺少 YOLO launcher");
    if (!caps.anomalib_launcher) problems.push("缺少 Anomalib launcher");
    if (!caps.infer_yolo) problems.push("yolo_infer.dll 不可用");
    if (!caps.infer_anomalib) problems.push("anomalib_infer.dll 不可用");
    if (caps.engine_build === false) problems.push("缺少 trtexec（引擎构建不可用）");
    if (caps.infer_worker?.state === "tripped") problems.push("推理进程反复崩溃，已暂停");

    if (problems.length === 0) {
      dot.className = "dot ok";
      text.textContent = "环境就绪";
    } else {
      dot.className = "dot bad";
      text.textContent = `${problems.length} 项待处理`;
    }

    capBox.textContent = JSON.stringify(caps, null, 2);
  } catch (error) {
    dot.className = "dot bad";
    text.textContent = "后端未连接";
    capBox.textContent = `无法获取环境信息：${error.message}`;
  }
}

// ---------------------------------------------------------------------------
// 路径选择
//
// 两条路：
//   1. 原生对话框（首选）——由后端在服务端桌面弹 Windows 系统框，能选任意盘符、
//      带该字段专属的扩展名过滤。浏览器自己做不到这件事（安全沙箱），必须绕后端。
//   2. 页面内目录浏览（兜底/Shift）——远程访问、服务跑在无桌面会话、
//      或用户想直接切盘符时用。能力探测失败时自动走这条。
// ---------------------------------------------------------------------------
const browseState = { targetId: null, mode: "dir", current: "", patterns: [] };

/** 解析 data-patterns=".pt,.pth" → [".pt", ".pth"]；接受 pt / *.pt 写法。 */
function splitPatterns(text) {
  return String(text || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      if (item.startsWith("*.")) return item.slice(1);
      if (item.startsWith(".")) return item;
      return `.${item}`;
    });
}

/**
 * 顶部提示条。
 *
 * 为什么不用 alert：选路径时用户正在看输入框，alert 会把焦点抢走，
 * 点掉之后还得重新找位置；提示条不打断操作。
 */
function toast(message, kind = "warn") {
  let box = document.getElementById("pathToast");
  if (!box) {
    box = document.createElement("div");
    box.id = "pathToast";
    box.className = "path-toast";
    document.body.appendChild(box);
  }
  box.textContent = message;
  box.dataset.kind = kind;
  box.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => box.classList.remove("show"), 4500);
}

// 正在等系统选择窗的那一个按钮。再点它就是取消，不能再开第二个框。
let browseWaiting = null;

/**
 * 弹原生对话框，选中就写回字段。
 *
 * 按钮在等待时改成「取消」并且保持可点。以前会改成「…」再禁用，
 * 系统窗口要是没弹出来，请求五分钟不返回，按钮就一直灰着，只能刷新页面。
 *
 * @returns {Promise<boolean>} true=已处理（含用户取消）；false=应回退到页面内浏览
 */
async function pickNative(button) {
  if (browseWaiting?.button === button) {
    try {
      await cancelNativeDialog();
    } catch {
      /* 服务没接上时，等原来的请求自己结束即可 */
    }
    return true;
  }

  const kind = button.dataset.dialog;
  if (!kind) return false;

  const caps = await capabilitiesOnce();
  if (!caps.native_dialog) return false;

  const targetId = button.dataset.browse;
  const target = targetId ? document.getElementById(targetId) : null;
  const payload = {
    kind,
    field: targetId || "",
    title: button.dataset.title || "",
    // 字段里通常已经是上次选的文件路径，后端会取父目录作为起始位置
    initial: (target?.value || "").trim(),
    patterns: splitPatterns(button.dataset.patterns),
    label: button.dataset.label || "",
    defaultextension: button.dataset.defaultextension || "",
  };

  const label = button.textContent;
  const previousTitle = button.title;
  button.textContent = "取消";
  button.classList.add("is-waiting");
  button.title = "桌面没有弹出窗口时，再点一次就取消，并改用页面里的选择";
  browseWaiting = { button };
  const hint = setTimeout(() => toast("正在打开系统选择窗口。没有出现就再点一次「取消」", "info"), 1200);
  try {
    const result = await nativeDialog(payload);
    const reason = String(result.reason || "");
    // 窗口没出来、子进程出错：交给页面内浏览，不要假装用户取消了。
    if (reason === "dialog-not-shown" || reason === "dialog-error" || reason === "no-result") {
      toast("系统选择窗口没有出现，已改用页面内浏览", "warn");
      return false;
    }
    if (reason === "timeout") {
      toast("选择超时，已取消", "warn");
      return true;
    }
    // 网页上点了「取消」：系统框收掉，接着打开页面内的选择，避免人只能刷新。
    if (reason === "cancelled") {
      toast("已改用页面内浏览", "info");
      return false;
    }
    if (result.paths?.length) {
      writeValue(targetId, result.paths[0]);
      toast(`已选择：${result.paths[0]}`, "info");
    }
    // 取消（系统框里取消，或网页上点了「取消」）是有意行为：不动字段、不报错
    return true;
  } catch (error) {
    if (error.status === 409) {
      toast(error.message, "warn");
      return true;
    }
    const detail = error.payload?.detail || error.message;
    toast(`系统对话框打不开（${detail}），已改用页面内浏览`, "warn");
    return false;
  } finally {
    clearTimeout(hint);
    if (browseWaiting?.button === button) browseWaiting = null;
    button.classList.remove("is-waiting");
    button.disabled = false;
    button.textContent = label;
    button.title = previousTitle;
  }
}

function initBrowse() {
  const modal = document.getElementById("browseModal");
  const list = document.getElementById("browseList");
  const current = document.getElementById("browseCurrent");
  const btnUp = document.getElementById("browseUp");
  const btnPick = document.getElementById("browsePick");
  const btnClose = document.getElementById("browseClose");

  document.querySelectorAll("[data-browse]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      // Shift：强制页面内浏览。远程访问、无桌面会话、或想手动切盘符时用得上。
      if (!event.shiftKey) {
        const handled = await pickNative(button);
        if (handled) return;
      }
      openModal(button);
    });
  });

  function openModal(button) {
    browseState.targetId = button.dataset.browse;
    browseState.mode = button.dataset.dialog === "dir" || button.dataset.mode === "dir" ? "dir" : "file";
    browseState.patterns = splitPatterns(button.dataset.patterns);
    const start = document.getElementById(browseState.targetId)?.value || "";
    modal.classList.remove("hidden");
    btnPick.textContent = browseState.mode === "file" ? "选择文件/当前目录" : "选择当前目录";
    openAt(start);
  }

  /**
   * 从字段当前值开始浏览。
   *
   * 字段里可能已经是文件路径（选过一次之后再点「浏览」很常见），
   * 直接把它交给 /api/browse 会 404，弹窗里只剩一句「无法读取」，用户以为坏了。
   * 先按目录试一次，失败就退到父目录再试。
   */
  async function openAt(start) {
    const path = (start || "").trim();
    if (!path) {
      await loadDir("");
      return;
    }
    if (await loadDir(path)) return;
    const parent = path.replace(/[\\/][^\\/]+[\\/]?$/, "");
    if (parent && parent !== path) await loadDir(parent);
  }

  /** 没有 data-patterns 时的兜底白名单，保持与旧版本一致的行为。 */
  const LEGACY_FILE_RE = /\.(trt|engine|onnx|json|yaml|yml|tfanno|pt|pth|txt|exe|mp4|avi|mov|mkv|m4v|flv|ts|wmv)$/i;

  function matchFile(name) {
    if (browseState.patterns.length) {
      const lower = name.toLowerCase();
      return browseState.patterns.some((pattern) => lower.endsWith(pattern.toLowerCase()));
    }
    return LEGACY_FILE_RE.test(name);
  }

  /** 列目录。返回是否成功，调用方据此决定要不要退到父目录。 */
  async function loadDir(path) {
    try {
      const data = await browse(path);
      browseState.current = data.path;
      current.textContent = data.path;
      list.innerHTML = "";

      if (browseState.mode === "file") {
        data.entries
          .filter((e) => !e.is_dir && matchFile(e.name))
          .forEach((entry) => list.appendChild(makeItem(entry, btnPick)));
      }

      data.entries
        .filter((e) => e.is_dir)
        .forEach((entry) => list.appendChild(makeItem(entry, btnPick)));
      return true;
    } catch (error) {
      current.textContent = `无法读取：${error.message}`;
      list.innerHTML = "";
      return false;
    }
  }

  function makeItem(entry, pickButton) {
    const li = document.createElement("li");
    li.className = entry.is_dir ? "dir" : "file";
    li.textContent = entry.name;
    li.addEventListener("click", async () => {
      if (entry.is_dir) {
        await loadDir(entry.path);
      } else if (browseState.mode === "file") {
        writeValue(browseState.targetId, entry.path);
        modal.classList.add("hidden");
      }
    });
    // 双击文件直接选中
    li.addEventListener("dblclick", () => {
      if (!entry.is_dir) {
        writeValue(browseState.targetId, entry.path);
        modal.classList.add("hidden");
      }
    });
    return li;
  }

  btnUp.addEventListener("click", () => {
    const parent = browseState.current.replace(/[\\/][^\\/]+$/, "");
    if (parent && parent !== browseState.current) loadDir(parent);
  });

  btnPick.addEventListener("click", () => {
    writeValue(browseState.targetId, browseState.current);
    modal.classList.add("hidden");
  });

  btnClose.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (event) => {
    if (event.target === modal) modal.classList.add("hidden");
  });
}

/**
 * 写值到目标输入框并通知一次 change。
 *
 * 程序化赋值不会触发 change 事件。目标字段往往要据此做后续动作：
 * 「输出引擎」要把选中的目录补成完整文件名，ONNX 字段要重新探测形状。
 * 所以这里显式派发一次，让字段自己决定怎么处理。
 */
export function writeValue(targetId, value) {
  const input = targetId ? document.getElementById(targetId) : null;
  if (!input) return;
  input.value = value;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}


// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------

// ?reduced-motion=1 手动关闭发光、扫描线与过渡效果。
// 用途：验收可读性护栏，以及在核显机器上排除合成层开销。与系统偏好
// prefers-reduced-motion 走同一套令牌归零规则（见 style.css 第 1.9 节）。
if (new URLSearchParams(location.search).has("reduced-motion")) {
  document.documentElement.classList.add("reduced-motion");
}

initTabs();
initTrain();
initPipeline();
initExport();
initEngine();
initDataset();
initAnnotate();
initInfer();
initInferBatch();
initBrowse();
// 顶栏的「运行中任务」条要在各页初始化之后挂：它只是读 /api/runs，
// 不依赖页面状态，但放在最后能保证各页的接管监听已经注册好。
initRunsBar();

// 顶栏条里点某个任务 → 先把页签切到它所属的页面，再把事件交给页面自己接管
// （训练页恢复日志与按钮、构建页恢复进度）。页面没接管也不影响：条上仍能停。
document.addEventListener("trtf:attach-run", (event) => {
  const kind = String(event.detail?.run?.kind || "");
  if (kind === "engine-build") activateTab("engine");
  else if (kind === "infer-batch" || kind === "infer-video") activateTab("infer");
  else activateTab("train");
});
// 分栏拖动条必须在各页初始化之后插入：这些 init 会在卡片里生成表单与列表，
// 晚一步插入才不会打乱它们按 DOM 顺序取元素。漏掉这一句的后果不是「不能拖」，
// 而是第二块内容掉进 10px 宽的拖动条列里，被挤成一条竖线（v1.25.0 修）。
initSplitPanes();
refreshVersion();
refreshEnvironment();
