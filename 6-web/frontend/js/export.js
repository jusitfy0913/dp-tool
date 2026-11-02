// 导出 ONNX 面板：训练 -> 导出 -> 编译 的中间一环。
//
// 为什么单独一个模块：导出是独立任务（有自己的 run_id、自己的事件流），
// 但它和训练共用「子进程 + WebSocket」那套机制，不需要写第二遍。
//
// 关键约束（都来自后端与 launcher，不在这里二次判断）：
// - 产物与权重同目录同名：best.pt -> best.onnx / best.transd.onnx
// - 任务类型以权重自带为准；用户填错时后端会发 task_overridden 事件
// - .transd.onnx 的段序（proto 在前）由 launcher 负责，前端只负责展示

import { postJSON, stopRun } from "./api.js";
import { connectRunEvents } from "./ws.js";

const state = {
  runId: null,
  stream: null,
  artifacts: [],
};

let dom = {};

export function initExport() {
  dom = {
    weights: document.getElementById("expWeights"),
    imgsz: document.getElementById("expImgsz"),
    opset: document.getElementById("expOpset"),
    transd: document.getElementById("expTransd"),
    simplify: document.getElementById("expSimplify"),
    dynamic: document.getElementById("expDynamic"),
    half: document.getElementById("expHalf"),
    btnExport: document.getElementById("btnExport"),
    btnStop: document.getElementById("btnStopExport"),
    btnPickBest: document.getElementById("btnPickBest"),
    hint: document.getElementById("exportHint"),
    tableBox: document.getElementById("exportTableBox"),
    rows: document.getElementById("exportRows"),
    log: document.getElementById("exportLog"),
    project: document.getElementById("project"),
    runName: document.getElementById("runName"),
  };
  if (!dom.btnExport) return;

  dom.btnExport.addEventListener("click", onExport);
  dom.btnStop.addEventListener("click", onStop);
  dom.btnPickBest.addEventListener("click", pickCurrentBest);
}

/// 「用当前实验 best.pt」：按 project/name 推出权重路径。
/// 不在这里判断文件是否存在——浏览器拿不到文件系统，交给后端报错更准确。
function pickCurrentBest() {
  const project = (dom.project?.value || "").trim();
  const name = (dom.runName?.value || "").trim();
  if (!project || !name) {
    setHint("请先填输出目录与实验名称，或直接点「浏览」选权重", "warn");
    return;
  }
  // 统一成反斜杠：后端与 Windows 路径都按这个解析，正斜杠混用会让人看不出填错
  const dir = `${project.replace(/[\\/]+$/, "")}\\${name}`;
  dom.weights.value = `${dir}\\weights\\best.pt`;
  setHint("已填入当前实验的 best.pt；若还没训练过，请先训练或手动选择其他权重", "");
}

function setHint(text, tone = "") {
  if (!dom.hint) return;
  dom.hint.textContent = text;
  dom.hint.className = tone ? `hint ${tone}` : "hint";
}

function appendLog(line, tone = "") {
  if (!dom.log) return;
  dom.log.hidden = false;
  const div = document.createElement("div");
  div.textContent = line;
  if (tone) div.className = `log-${tone}`;
  dom.log.appendChild(div);
  dom.log.scrollTop = dom.log.scrollHeight;
}

function resetRows() {
  state.artifacts = [];
  if (dom.rows) dom.rows.innerHTML = "";
  if (dom.tableBox) dom.tableBox.hidden = true;
}

/// 产物行。transd 那一行多一个「去编译引擎」按钮：它才是 TensorRT 真正要的输入，
/// 把用户直接送到引擎页能省掉「我要填哪个文件」这个最常见的问题。
function addArtifactRow(kind, path) {
  if (!path) return;
  state.artifacts.push({ kind, path });
  if (!dom.rows) return;
  dom.tableBox.hidden = false;

  const tr = document.createElement("tr");
  const tdKind = document.createElement("td");
  tdKind.textContent = kind === "transd_onnx" ? "转置 ONNX（给 TensorRT 用）" : "ONNX（Ultralytics 用）";
  const tdPath = document.createElement("td");
  tdPath.className = "mono";
  tdPath.textContent = path;
  const tdAct = document.createElement("td");

  if (kind === "transd_onnx" || /\.transd\.onnx$/i.test(path)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mini primary";
    btn.textContent = "去编译引擎";
    btn.addEventListener("click", () => {
      const engOnnx = document.getElementById("engOnnx");
      if (engOnnx) {
        engOnnx.value = path;
        // 引擎页监听的是 change（用来触发形状探测），只发 input 不会自动探测
        engOnnx.dispatchEvent(new Event("input", { bubbles: true }));
        engOnnx.dispatchEvent(new Event("change", { bubbles: true }));
      }
      // 引擎页会自动探测形状，切过去就能直接点构建
      document.querySelector('.tab[data-tab="engine"]')?.click();
      document.getElementById("engOnnx")?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    tdAct.appendChild(btn);
  }

  tr.append(tdKind, tdPath, tdAct);
  dom.rows.appendChild(tr);
}

async function onExport() {
  const weights = (dom.weights?.value || "").trim();
  if (!weights) {
    setHint("请先选择 .pt 权重", "warn");
    return;
  }

  resetRows();
  dom.log.hidden = true;
  dom.log.textContent = "";
  dom.btnExport.disabled = true;
  dom.btnStop.disabled = false;
  setHint("导出中…", "");

  const body = {
    weights,
    imgsz: Number(dom.imgsz?.value || 640),
    make_transd: Boolean(dom.transd?.checked),
    simplify: Boolean(dom.simplify?.checked),
    dynamic: Boolean(dom.dynamic?.checked),
    half: Boolean(dom.half?.checked),
  };
  const opset = (dom.opset?.value || "").trim();
  if (opset) body.opset = Number(opset);

  try {
    const run = await postJSON("/api/export/yolo", body);
    state.runId = run.run_id;
    appendLog(`已启动导出：${run.run_id}`);
    openStream(run.run_id);
  } catch (error) {
    finish("failed");
    setHint(`启动失败：${error.message}`, "bad");
  }
}

function openStream(runId) {
  state.stream = connectRunEvents(runId, {
    onEvent: (event) => {
      switch (event.type) {
        case "log":
          appendLog(event.line || "");
          break;
        case "event":
          if (event.name === "export_start") {
            appendLog(`开始导出（task=${event.task || "-"}，权重=${event.weights || "-"}）`, "ok");
          } else if (event.name === "task_overridden") {
            appendLog(`注意：${event.message}`, "warn");
          } else if (event.name === "transd_written") {
            const shapes = (event.outputs || [])
              .map((o) => `${o.name}${JSON.stringify(o.dims)}`)
              .join("，");
            appendLog(`已生成 .transd.onnx（${shapes}）`, "ok");
          }
          break;
        case "artifact":
          if (event.kind === "onnx" || event.kind === "transd_onnx") {
            appendLog(`${event.kind}：${event.path}`, "ok");
            addArtifactRow(event.kind, event.path);
          }
          break;
        case "error":
          appendLog(`${event.code || "错误"}：${event.message || ""}`, "err");
          break;
        default:
          break;
      }
    },
    onClose: (payload) => {
      // 主动停止不等于失败：Windows 上 terminate 的退出码也是 1，
      // 只能看后端标好的 status（与训练页同一套判断）。
      if (payload.status === "cancelled") {
        finish("cancelled");
        setHint("导出已停止（已生成的产物保留）", "warn");
      } else if (payload.status === "finished") {
        finish("done");
        const hasTransd = state.artifacts.some((a) => a.kind === "transd_onnx");
        setHint(
          hasTransd ? "导出完成，可点「去编译引擎」继续" : "导出完成（未生成 .transd.onnx）",
          "ok",
        );
      } else if (payload.status === "disconnected") {
        finish("failed");
        setHint("事件流断开，请刷新页面查看结果", "warn");
      } else {
        finish("failed");
        setHint(`导出失败（退出码 ${payload.exit}）`, "bad");
      }
    },
    onError: (message) => appendLog(message, "warn"),
  });
}

function finish(status) {
  dom.btnExport.disabled = false;
  dom.btnStop.disabled = true;
  state.runId = null;
  state.stream = null;
  if (status === "done") appendLog("导出结束。", "ok");
}

async function onStop() {
  if (!state.runId) return;
  dom.btnStop.disabled = true;
  try {
    await stopRun(state.runId);
    appendLog("已请求停止…");
  } catch (error) {
    dom.btnStop.disabled = false;
    setHint(`停止失败：${error.message}`, "bad");
  }
}
