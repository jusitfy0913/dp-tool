// OCR 结果：多边形叠加、文本列表、复制与导出。
// 推理本身在 ocr_infer.dll 里，这里只负责把返回的行画出来。

import { getJSON, postJSON } from "./api.js";
import { setOverlayHighlight } from "./overlay.js";

const SVG_NS = "http://www.w3.org/2000/svg";

const PAIRS = [
  ["ocrDetPick", "ocrDet"],
  ["ocrRecPick", "ocrRec"],
  ["ocrTextlinePick", "ocrTextline"],
];

/** 把 /api/ocr/models 填进下拉。不兼容的项置灰，不能当默认值。 */
export async function loadOcrModels() {
  const hint = document.getElementById("ocrModelHint");
  let data;
  try {
    data = await getJSON("/api/ocr/models");
  } catch (error) {
    if (hint) hint.textContent = `模型列表读取失败：${error.message}`;
    return;
  }
  const entries = data.entries || [];
  fillPick("ocrDetPick", "ocrDet", entries.filter((item) => item.kind === "det"));
  fillPick("ocrRecPick", "ocrRec", entries.filter((item) => item.kind === "rec"));
  fillPick("ocrTextlinePick", "ocrTextline", entries.filter((item) => item.kind === "textline_ori"), true);
  const version = data.paddle_version || "未知";
  const bad = entries.filter((item) => !item.compatible).length;
  if (hint) {
    hint.textContent = `飞桨 ${version}。共 ${entries.length} 个模型，${bad} 个当前不可用。`;
  }
  // 模型列表出来后顺手预热一次：oi_create 首次要 1.3 s（加载 1.7 GB 的
  // paddle_inference.dll + 解析模型），这段时间不计入 DLL 的 timing，
  // 不预热的话用户第一次点推理会莫名卡住。
  // 失败不提示：预热只是优化，真正推理时该报的错还是会报。
  postJSON("/api/ocr/warm", {}).catch(() => {});
}

function fillPick(selectId, inputId, entries, optional) {
  const select = document.getElementById(selectId);
  const input = document.getElementById(inputId);
  if (!select || !input) return;
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = optional ? "不启用" : "— 请选择 —";
  select.appendChild(blank);
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.path;
    option.disabled = !entry.compatible;
    option.textContent = entry.compatible
      ? `${entry.name}（${entry.format}）`
      : `${entry.name}（${entry.reason || "不可用"}）`;
    select.appendChild(option);
  }
  const current = input.value;
  const match = [...select.options].find((option) => option.value === current && !option.disabled);
  select.value = match ? current : "";
  if (!optional && !input.value) {
    const first = [...select.options].find((option) => option.value && !option.disabled);
    if (first) {
      select.value = first.value;
      input.value = first.value;
    }
  }
}

export function bindOcrPicks() {
  for (const [selectId, inputId] of PAIRS) {
    document.getElementById(selectId)?.addEventListener("change", (event) => {
      const input = document.getElementById(inputId);
      if (input) input.value = event.target.value;
      input?.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
}

/** 原图上的文字多边形。点框会回调 onPick(index)。 */
export function buildOcrOverlay({ lines, width, height, showBoxes, showLabels, highlight, onPick }) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "detect-overlay");
  svg.setAttribute("viewBox", `0 0 ${Math.max(1, width)} ${Math.max(1, height)}`);
  svg.setAttribute("preserveAspectRatio", "none");
  (lines || []).forEach((line, index) => {
    const poly = line.poly || [];
    if (poly.length < 2) return;
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", "detect-box");
    group.dataset.index = String(index);
    if (index === highlight) group.classList.add("is-highlight");
    const shape = document.createElementNS(SVG_NS, "polygon");
    shape.setAttribute("points", poly.map(([x, y]) => `${x},${y}`).join(" "));
    if (showBoxes) {
      shape.setAttribute("fill", "none");
      shape.setAttribute("stroke", "#3dd6c6");
      shape.setAttribute("stroke-width", "2");
      shape.setAttribute("vector-effect", "non-scaling-stroke");
      group.appendChild(shape);
    }
    if (showLabels) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(poly[0][0]));
      label.setAttribute("y", String(Math.max(12, poly[0][1] - 4)));
      // 不要白字：OCR 处理的图多是白底/浅底（文档、标签、铭牌），白字会直接糊成一片。
      // 用琥珀色 + 深色描边（paint-order 把描边压在文字下面当"光晕"），任何底色都能读。
      label.setAttribute("fill", "#ffd54a");
      label.setAttribute("stroke", "#0b1016");
      label.setAttribute("stroke-width", "3");
      label.setAttribute("stroke-linejoin", "round");
      label.setAttribute("paint-order", "stroke");
      label.setAttribute("font-size", "14");
      label.setAttribute("font-weight", "600");
      label.textContent = `${index + 1} ${line.text || ""}`;
      group.appendChild(label);
    }
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      onPick?.(index);
    });
    svg.appendChild(group);
  });
  return svg;
}

export function highlightOcr(svg, index) {
  setOverlayHighlight(svg, index);
}

export function renderOcrTable(body, lines, highlight, onPick) {
  body.replaceChildren();
  if (!lines.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "没有识别到文字";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  lines.forEach((line, index) => {
    const row = document.createElement("tr");
    if (index === highlight) row.classList.add("is-active");
    row.dataset.index = String(index);
    const cells = [
      String(index + 1),
      line.text || "",
      Number(line.score || 0).toFixed(4),
      line.angle === undefined || line.angle < 0 ? "—" : `${line.angle}`,
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      row.appendChild(td);
    }
    row.addEventListener("click", () => onPick?.(index));
    body.appendChild(row);
  });
}

export function ocrPlainText(lines) {
  return (lines || []).map((line) => line.text || "").join("\n");
}

export function ocrCsv(lines) {
  const rows = ["index,text,score,angle,poly"];
  (lines || []).forEach((line, index) => {
    const text = csvCell(line.text || "");
    const poly = csvCell(JSON.stringify(line.poly || []));
    rows.push([index + 1, text, Number(line.score || 0).toFixed(6), line.angle ?? "", poly].join(","));
  });
  return `\ufeff${rows.join("\r\n")}`;
}

function csvCell(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
