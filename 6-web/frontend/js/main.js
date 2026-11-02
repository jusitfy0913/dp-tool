// 入口：页签切换、环境探测、目录浏览弹窗。

import { browse, getCapabilities, getHealth } from "./api.js";
import { initAnnotate } from "./annotate.js";
import { initDataset } from "./dataset.js";
import { initEngine } from "./engine.js";
import { initExport } from "./export.js";
import { initInfer } from "./infer.js";
import { initInferBatch } from "./inferbatch.js";
import { initPipeline } from "./pipeline.js";
import { initTrain } from "./train.js";

// ---------------------------------------------------------------------------
// 页签
// ---------------------------------------------------------------------------
function initTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = Array.from(document.querySelectorAll(".panel"));

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      const name = tab.dataset.tab;
      panels.forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
      // 通知各模块"我这一页现在可见了"。
      // 标注画布必须先可见才能量到真实尺寸，否则会按 0 宽高初始化成 1×1。
      document.dispatchEvent(new CustomEvent("tab-change", { detail: { name } }));
    });
  });
}

// ---------------------------------------------------------------------------
// 环境状态
// ---------------------------------------------------------------------------
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
    const problems = [];
    if (!caps.python) problems.push("缺少训练解释器");
    if (!caps.yolo_launcher) problems.push("缺少 YOLO launcher");
    if (!caps.anomalib_launcher) problems.push("缺少 Anomalib launcher");
    if (!caps.infer_yolo) problems.push("yolo_infer.dll 不可用");
    if (!caps.infer_anomalib) problems.push("anomalib_infer.dll 不可用");
    if (caps.engine_build === false) problems.push("缺少 trtexec（引擎构建不可用）");

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
// 目录浏览弹窗
// 浏览器无法弹系统文件框，这里用受限的目录浏览补上「选路径」能力。
// ---------------------------------------------------------------------------
const browseState = { targetId: null, mode: "dir", current: "" };

function initBrowse() {
  const modal = document.getElementById("browseModal");
  const list = document.getElementById("browseList");
  const current = document.getElementById("browseCurrent");
  const btnUp = document.getElementById("browseUp");
  const btnPick = document.getElementById("browsePick");
  const btnClose = document.getElementById("browseClose");

  document.querySelectorAll("[data-browse]").forEach((button) => {
    button.addEventListener("click", async () => {
      browseState.targetId = button.dataset.browse;
      browseState.mode = button.dataset.mode || "dir";
      const start = document.getElementById(browseState.targetId)?.value || "";
      modal.classList.remove("hidden");
      btnPick.textContent = browseState.mode === "file" ? "选择文件/当前目录" : "选择当前目录";
      await loadDir(start);
    });
  });

  async function loadDir(path) {
    try {
      const data = await browse(path);
      browseState.current = data.path;
      current.textContent = data.path;
      list.innerHTML = "";

      if (browseState.mode === "file") {
        data.entries
          .filter((e) => !e.is_dir && /\.(trt|engine|onnx|json|yaml|yml|tfanno|pt|mp4|avi|mov|mkv|m4v|flv|ts|wmv)$/i.test(e.name))
          .forEach((entry) => list.appendChild(makeItem(entry, btnPick)));
      }

      data.entries
        .filter((e) => e.is_dir)
        .forEach((entry) => list.appendChild(makeItem(entry, btnPick)));
    } catch (error) {
      current.textContent = `无法读取：${error.message}`;
      list.innerHTML = "";
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
        applyValue(entry.path);
        modal.classList.add("hidden");
      }
    });
    // 双击文件直接选中
    li.addEventListener("dblclick", () => {
      if (!entry.is_dir) {
        applyValue(entry.path);
        modal.classList.add("hidden");
      }
    });
    return li;
  }

  function applyValue(value) {
    const input = document.getElementById(browseState.targetId);
    if (input) input.value = value;
  }

  btnUp.addEventListener("click", () => {
    const parent = browseState.current.replace(/[\\/][^\\/]+$/, "");
    if (parent && parent !== browseState.current) loadDir(parent);
  });

  btnPick.addEventListener("click", () => {
    applyValue(browseState.current);
    modal.classList.add("hidden");
  });

  btnClose.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (event) => {
    if (event.target === modal) modal.classList.add("hidden");
  });
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
refreshVersion();
refreshEnvironment();
