// 引擎构建页：ONNX -> TensorRT 引擎。
//
// 与训练页的区别：
// - trtexec 只输出普通文本日志，没有 TRTFORGE_JSON 结构事件，所以不做指标解析；
// - 进度无法量化（trtexec 不报百分比），用状态徽标表达 运行中/完成/失败 即可；
// - 日志量可达上千行，用 textContent 整体替换比逐行 append 更快。
//
// 事件流本身复用 /ws/runs/{id}，和训练页是同一条通道。

import { getJSON, postJSON, stopRun } from "./api.js";
import { connectRunEvents } from "./ws.js";

// 只保留最近若干行，防止长时间构建把内存和 DOM 拖慢
const MAX_LINES = 2000;

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
  const badge = document.getElementById("engineBadge");
  const btnBuild = document.getElementById("btnBuild");
  const btnStop = document.getElementById("btnBuildStop");

  let active = null; // { runId, stream }
  let lines = [];

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
    return `${path.replace(/\.onnx$/i, "")}_${tag}.engine`;
  }

  function fmtShape(shape) {
    return Array.isArray(shape) ? `[${shape.join(", ")}]` : "?";
  }

  function render() {
    if (lines.length > MAX_LINES) lines = lines.slice(-MAX_LINES);
    logBox.textContent = lines.join("\n");
    logBox.scrollTop = logBox.scrollHeight;
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

    setHint("");
    lines = [];
    render();
    btnBuild.disabled = true;
    setBadge("启动中", "run");

    try {
      const run = await postJSON("/api/engine/build", body);
      active = { runId: run.run_id, stream: null };
      setBadge("构建中", "run");
      btnStop.disabled = false;
      setHint(`输出目标：${run.engine}`);

      lines.push(`$ ${run.command.join(" ")}`);
      lines.push("");
      render();

      active.stream = connectRunEvents(run.run_id, {
        onEvent: (payload) => {
          if (payload.type === "log") {
            lines.push(payload.line);
          } else if (payload.type === "error") {
            lines.push(`[错误] ${payload.message}`);
          }
          render();
        },
        onClose: (payload) => {
          const ok = payload.status === "finished";
          setBadge(ok ? "已完成" : "失败", ok ? "done" : "fail");
          lines.push("");
          lines.push(`[结束] 退出码=${payload.exit} 状态=${payload.status}`);
          render();
          btnBuild.disabled = false;
          btnStop.disabled = true;
          active = null;
          if (ok) {
            setHint(`构建完成：${body.engine || autoName(body.onnx)}`);
          } else {
            setHint("构建失败，看日志尾部定位原因", true);
          }
        },
        onError: (message) => {
          lines.push(`[事件流] ${message}`);
          render();
        },
      });
    } catch (error) {
      setHint(error.message, true);
      setBadge("失败", "fail");
      lines.push(`[错误] ${error.message}`);
      render();
      btnBuild.disabled = false;
      btnStop.disabled = true;
    }
  });

  btnStop.addEventListener("click", async () => {
    if (!active) return;
    try {
      await stopRun(active.runId);
      setHint("已请求停止");
    } catch (error) {
      setHint(error.message, true);
    }
  });
}
