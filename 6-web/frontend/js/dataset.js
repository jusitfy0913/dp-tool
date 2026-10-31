// 数据集页：扫描 YOLO / .tfanno → 按固定哈希预览划分 → 导出 YOLO / COCO / VOC。
//
// 划分状态不放在前端也不放在后端：比例确定后，同一份数据的结果必然一致，
// 所以预览与导出各算一次，不需要维护会话，刷新页面也不会丢东西。

import { postJSON } from "./api.js";

const MAX_ROWS = 200;

export function initDataset() {
  const form = document.getElementById("datasetForm");
  const source = document.getElementById("dsSource");
  const task = document.getElementById("dsTask");
  const format = document.getElementById("dsFormat");
  const train = document.getElementById("dsTrain");
  const val = document.getElementById("dsVal");
  const output = document.getElementById("dsOutput");
  const overwrite = document.getElementById("dsOverwrite");
  const ratioHint = document.getElementById("dsRatioHint");
  const hint = document.getElementById("dsHint");
  const badge = document.getElementById("dsBadge");
  const metrics = document.getElementById("dsMetrics");
  const warnings = document.getElementById("dsWarnings");
  const rows = document.getElementById("dsRows");
  const result = document.getElementById("dsResult");
  const btnScan = document.getElementById("btnScan");
  const btnExport = document.getElementById("btnExport");

  let lastScan = null; // 最近一次扫描结果，导出成功后用于对比

  function setHint(text, bad = false) {
    hint.textContent = text || "";
    hint.className = bad ? "hint bad" : "hint";
  }

  function setBadge(text, cls) {
    badge.textContent = text;
    badge.className = cls ? `badge ${cls}` : "badge";
  }

  function collect() {
    return {
      source: source.value.trim(),
      task: task.value,
      train: Number(train.value) || 0,
      val: Number(val.value) || 0,
      format: format.value,
      output: output.value.trim(),
      overwrite: overwrite.checked,
    };
  }

  // test 比例是余数，实时算给用户看，避免误以为填错了
  function refreshRatioHint() {
    const t = Number(train.value) || 0;
    const v = Number(val.value) || 0;
    if (t + v > 100) {
      ratioHint.textContent = `train ${t}% + val ${v}% 已超过 100%，导出会被拒绝`;
      ratioHint.className = "hint bad";
      return;
    }
    ratioHint.textContent = `test = ${100 - t - v}%（其余全部归 test，与 Qt 版一致）`;
    ratioHint.className = "hint";
  }

  train.addEventListener("input", refreshRatioHint);
  val.addEventListener("input", refreshRatioHint);

  function tag(split) {
    const span = document.createElement("span");
    span.className = `tag ${split}`;
    span.textContent = split;
    return span;
  }

  function renderMetrics(data) {
    metrics.innerHTML = "";
    const items = [
      ["总图片", data.counts.total],
      ["train", data.counts.train],
      ["val", data.counts.val],
      ["test", data.counts.test],
      ["空标注", data.counts.empty],
      ["类别", data.classes.length],
      ["任务", data.task],
    ];
    for (const [label, value] of items) {
      const box = document.createElement("div");
      box.className = "metric";
      box.innerHTML = `${label} <b>${value}</b>`;
      metrics.appendChild(box);
    }
  }

  function renderRows(data) {
    rows.innerHTML = "";
    const head = document.createElement("tr");
    for (const title of ["图片", "现有划分", "新划分", "标注数", "尺寸"]) {
      const td = document.createElement("td");
      td.textContent = title;
      head.appendChild(td);
    }
    rows.appendChild(head);

    for (const image of data.images) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = image.rel;
      name.title = image.rel;
      tr.appendChild(name);

      const current = document.createElement("td");
      current.appendChild(tag(image.current_split));
      tr.appendChild(current);

      const assigned = document.createElement("td");
      assigned.appendChild(tag(image.split));
      tr.appendChild(assigned);

      const objects = document.createElement("td");
      objects.textContent = image.objects;
      tr.appendChild(objects);

      const size = document.createElement("td");
      size.textContent = image.width && image.height ? `${image.width}×${image.height}` : "未知";
      tr.appendChild(size);

      rows.appendChild(tr);
    }
  }

  function renderWarnings(list) {
    warnings.textContent = (list || []).length ? `注意：${list.join("；")}` : "";
    warnings.className = (list || []).length ? "hint bad" : "hint";
  }

  async function scan() {
    const body = collect();
    if (!body.source) {
      setHint("请先选择数据源", true);
      return;
    }
    setHint("扫描中…");
    btnScan.disabled = true;
    try {
      const data = await postJSON("/api/dataset/scan", {
        source: body.source,
        task: body.task,
        train: body.train,
        val: body.val,
        limit: MAX_ROWS,
      });
      lastScan = data;
      renderMetrics(data);
      renderRows(data);
      renderWarnings(data.warnings);
      setBadge(data.truncated ? `预览前 ${MAX_ROWS} 张` : "已扫描", "run");
      setHint(
        data.truncated
          ? `共 ${data.counts.total} 张，列表只显示前 ${MAX_ROWS} 张；导出会处理全部图片。`
          : `共 ${data.counts.total} 张，可调整比例后重新扫描。`,
      );
      result.classList.add("hidden");
    } catch (error) {
      setBadge("扫描失败", "fail");
      setHint(error.message, true);
    } finally {
      btnScan.disabled = false;
    }
  }

  btnScan.addEventListener("click", scan);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = collect();
    if (!body.source) {
      setHint("请先选择数据源", true);
      return;
    }
    if (!body.output) {
      setHint("请填写输出目录", true);
      return;
    }

    if (body.train + body.val > 100) {
      setHint("train 与 val 比例之和不能超过 100%", true);
      return;
    }

    setHint("导出中…");
    btnExport.disabled = true;
    setBadge("导出中", "run");
    try {
      const data = await postJSON("/api/dataset/export", body);
      setBadge("导出完成", "done");
      const counts = Object.entries(data.counts || {})
        .map(([key, value]) => `${key}=${value}`)
        .join("  ");
      setHint(`已写入 ${data.output}（${counts}）`);
      renderWarnings(data.warnings);

      const lines = [
        `格式        : ${data.format}`,
        `任务        : ${data.task}`,
        `输出        : ${data.output}`,
        `划分        : ${JSON.stringify(data.split_counts)}`,
        `统计        : ${counts}`,
        `附加文件    : ${(data.extra_files || []).join(", ") || "无"}`,
      ];
      if (data.note) lines.push(`说明        : ${data.note}`);
      if ((data.skipped || []).length) {
        lines.push(`跳过 ${data.skipped.length} 项：`);
        for (const item of data.skipped.slice(0, 20)) lines.push(`  - ${item}`);
      }
      if ((data.warnings || []).length) {
        lines.push("警告：");
        for (const item of data.warnings) lines.push(`  - ${item}`);
      }
      result.textContent = lines.join("\n");
      result.classList.remove("hidden");
    } catch (error) {
      setBadge("导出失败", "fail");
      setHint(error.message, true);
      result.textContent = `[错误] ${error.message}`;
      result.classList.remove("hidden");
    } finally {
      btnExport.disabled = false;
    }
  });

  refreshRatioHint();
}
