// 推理页：上传图片 → 调后端 → 展示结果图。
// 图片以 base64 返回，前端只负责挂到 <img>，不做像素处理。

import { postForm } from "./api.js";

let dom = {};

export function initInfer() {
  dom = {
    form: document.getElementById("inferForm"),
    kind: document.getElementById("inferKind"),
    yoloGroup: document.getElementById("inferYoloGroup"),
    anomGroup: document.getElementById("inferAnomGroup"),
    btnInfer: document.getElementById("btnInfer"),
    hint: document.getElementById("inferHint"),
    badge: document.getElementById("inferBadge"),
    metrics: document.getElementById("inferMetrics"),
    imgSource: document.getElementById("imgSource"),
    imgResult: document.getElementById("imgResult"),
    imgMask: document.getElementById("imgMask"),
    maskRow: document.getElementById("maskRow"),
    capResult: document.getElementById("capResult"),
    file: document.getElementById("inferImage"),
  };

  dom.kind.addEventListener("change", syncKind);
  dom.form.addEventListener("submit", onSubmit);
  dom.file.addEventListener("change", onFileChange);
  syncKind();
}

function syncKind() {
  const isYolo = dom.kind.value === "yolo";
  dom.yoloGroup.classList.toggle("hidden", !isYolo);
  dom.anomGroup.classList.toggle("hidden", isYolo);
  dom.capResult.textContent = isYolo ? "检测结果" : "异常叠加图";
}

function formValue(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function onFileChange() {
  const file = dom.file.files?.[0];
  if (!file) return;
  dom.imgSource.src = URL.createObjectURL(file);
  dom.imgResult.removeAttribute("src");
  dom.imgMask.removeAttribute("src");
  dom.maskRow.style.display = "none";
}

async function onSubmit(event) {
  event.preventDefault();
  const isYolo = dom.kind.value === "yolo";
  const file = dom.file.files?.[0];

  if (!file) {
    setHint("请先选择一张图片");
    return;
  }
  if (isYolo && !formValue("yoloEngine")) {
    setHint("请填写 YOLO 引擎路径");
    return;
  }
  if (!isYolo && (!formValue("anomEngine") || !formValue("anomMetadata"))) {
    setHint("请填写引擎与 metadata.json 路径");
    return;
  }

  setHint("");
  dom.btnInfer.disabled = true;
  setBadge("推理中…", "run");
  dom.metrics.innerHTML = "";

  const form = new FormData();
  form.append("image", file);

  let url;
  if (isYolo) {
    url = "/api/infer/yolo";
    form.append("engine", formValue("yoloEngine"));
    form.append("task", formValue("inferTask"));
    form.append("conf", formValue("inferConf"));
    form.append("nms", formValue("inferNms"));
  } else {
    url = "/api/infer/anomalib";
    form.append("engine", formValue("anomEngine"));
    form.append("metadata", formValue("anomMetadata"));
    form.append("model_type", formValue("anomModelType"));
  }

  try {
    const result = await postForm(url, form);
    renderResult(result, isYolo);
    setBadge("完成", "done");
  } catch (error) {
    setHint(error.message);
    setBadge("失败", "fail");
  } finally {
    dom.btnInfer.disabled = false;
  }
}

function renderResult(result, isYolo) {
  if (isYolo) {
    dom.imgResult.src = `data:image/png;base64,${result.annotated_png}`;
    setMetric("目标数", result.count);
    setMetric("后端", result.backend);
    if (result.timing) {
      setMetric("推理耗时", `${result.timing.inference_ms.toFixed(1)} ms`);
      setMetric("总耗时", `${result.timing.total_ms.toFixed(1)} ms`);
    }
    const summary = (result.boxes || [])
      .slice(0, 10)
      .map((b) => `cls${b.class_label} ${b.confidence.toFixed(3)}`)
      .join("  ");
    if (summary) setMetric("检出", summary);
  } else {
    if (result.overlay_png) {
      dom.imgResult.src = `data:image/png;base64,${result.overlay_png}`;
    }
    if (result.mask_png) {
      dom.imgMask.src = `data:image/png;base64,${result.mask_png}`;
      dom.maskRow.style.display = "grid";
    }
    setMetric("判定", result.is_anomaly ? "异常 NG" : "正常 OK");
    setMetric("原始分数", Number(result.raw_score).toFixed(4));
    setMetric("归一化分数", Number(result.normalized_score).toFixed(4));
    if (result.metadata?.image_threshold != null) {
      setMetric("图像阈值", Number(result.metadata.image_threshold).toFixed(4));
    }
    if (result.timing) {
      setMetric("推理耗时", `${result.timing.inference_ms.toFixed(1)} ms`);
    }
  }
}

function setMetric(name, value) {
  const chip = document.createElement("div");
  chip.className = "metric";
  chip.innerHTML = `${escapeHtml(name)} <b>${escapeHtml(String(value))}</b>`;
  dom.metrics.appendChild(chip);
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
