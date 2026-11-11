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
  const groupBy = document.getElementById("dsGroupBy");
  const okDirs = document.getElementById("dsOkDirs");
  const okSplit = document.getElementById("dsOkSplit");
  const output = document.getElementById("dsOutput");
  const overwrite = document.getElementById("dsOverwrite");
  const ratioHint = document.getElementById("dsRatioHint");
  const hint = document.getElementById("dsHint");
  const badge = document.getElementById("dsBadge");
  const metrics = document.getElementById("dsMetrics");
  const warnings = document.getElementById("dsWarnings");
  const rows = document.getElementById("dsRows");
  const head = document.getElementById("dsHead");
  const result = document.getElementById("dsResult");
  const btnScan = document.getElementById("btnScan");
  const btnExport = document.getElementById("btnExport");
  const btnQc = document.getElementById("btnQc");
  const qcBadge = document.getElementById("dsQcBadge");
  const qcMetrics = document.getElementById("dsQcMetrics");
  const qcLeak = document.getElementById("dsQcLeak");
  const qcGroups = document.getElementById("dsQcGroups");
  const qcIssues = document.getElementById("dsQcIssues");

  let lastScan = null; // 最近一次扫描结果，导出成功后用于对比
  let lastQc = null; // 最近一次体检结果；导出时据此决定是否把结论写进 manifest

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
      group_by: groupBy.value,
      ok_dirs: okDirs.value
        .split(";")
        .map((item) => item.trim())
        .filter(Boolean),
      ok_split: okSplit.value,
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
    const ok = data.ok || {};
    const items = [
      ["总图片", data.counts.total],
      ["train", data.counts.train],
      ["val", data.counts.val],
      ["test", data.counts.test],
      ["空标注", data.counts.empty],
      ["良品", ok.images ?? 0],
      ["类别", data.classes.length],
      ["任务", data.task],
    ];
    for (const [label, value] of items) {
      const box = document.createElement("div");
      box.className = "metric";
      box.innerHTML = `${label} <b>${value}</b>`;
      metrics.appendChild(box);
    }
    // 良品与漏标必须分开说：空标注里既有良品也有漏标，混在一起会让过杀率失真
    if (ok.images) {
      const note = document.createElement("p");
      note.className = "hint";
      note.textContent =
        `${ok.images} 张良品图已归入 ${ok.split} 划分（导出写空标签、不参与训练）。` +
        "过杀率按「标签文件存在且为空」统计。";
      metrics.appendChild(note);
    }
  }

  function renderRows(data) {
    rows.innerHTML = "";
    head.innerHTML = "";

    // 表头用 th 而不是 td：语义正确，读屏会把列名关联到单元格。
    // 数值列加 num 类右对齐，等宽字体让位数相同的数字纵向对齐。
    const classify = data.task === "classify";
    const headRow = document.createElement("tr");
    for (const [title, numeric] of [["图片", false], ["现有划分", false], ["新划分", false], [classify ? "类别" : "标注数", !classify], ["尺寸", true]]) {
      const th = document.createElement("th");
      th.textContent = title;
      if (numeric) th.style.textAlign = "right";
      headRow.appendChild(th);
    }
    head.appendChild(headRow);

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
      objects.className = classify ? "" : "num";
      objects.textContent = classify ? (image.label || "未标注") : image.objects;
      tr.appendChild(objects);

      const size = document.createElement("td");
      size.className = "num";
      size.textContent = image.width && image.height ? `${image.width}×${image.height}` : "未知";
      tr.appendChild(size);

      rows.appendChild(tr);
    }
  }

  function renderWarnings(list) {
    warnings.textContent = (list || []).length ? `注意：${list.join("；")}` : "";
    warnings.className = (list || []).length ? "hint bad" : "hint";
  }

  // ------------------------------------------------------------------
  // 体检：分组 + 跨划分泄漏 + 标注问题
  // ------------------------------------------------------------------
  let rowLimit = 60; // 分组与问题各显示多少条，其余折叠

  function resetQc(reason) {
    lastQc = null;
    qcBadge.textContent = reason || "未体检";
    qcBadge.className = "badge";
    qcMetrics.innerHTML = "";
    qcLeak.classList.add("hidden");
    qcGroups.innerHTML = "";
    qcIssues.innerHTML = "";
  }

  function metric(label, value, cls) {
    const box = document.createElement("div");
    box.className = cls ? `metric ${cls}` : "metric";
    box.textContent = `${label} `;
    const strong = document.createElement("b");
    strong.textContent = value;
    box.appendChild(strong);
    return box;
  }

  // 分组/问题的条数可能上千，逐条建 DOM 会卡住页面；这里只渲染前 rowLimit 条。
  function renderQc(data) {
    lastQc = data;
    const groups = data.groups || {};
    const summary = data.summary || {};
    const leak = Number(groups.cross_split) || 0;
    qcBadge.textContent = `${groups.total || 0} 组${leak ? ` · ${leak} 跨划分` : ""}`;
    qcBadge.className = leak ? "badge fail" : "badge done";

    qcMetrics.innerHTML = "";
    const items = [
      ["图片", data.counts ? data.counts.images : 0, ""],
      ["底板组", groups.total || 0, ""],
      ["跨划分组", leak, leak ? "bad" : ""],
      ["精确重复组", data.exact_group_total || 0, ""],
      ["错误", summary.error || 0, summary.error ? "bad" : ""],
      ["警告", summary.warn || 0, summary.warn ? "warn" : ""],
      ["提示", summary.info || 0, ""],
      ["耗时", `${data.elapsed_ms || 0} ms`, ""],
    ];
    for (const [label, value, cls] of items) qcMetrics.appendChild(metric(label, value, cls));

    if (leak) {
      qcLeak.textContent =
        `${leak} 个组同时出现在多个划分里（涉及 ${groups.leak_images || 0} 张图）：` +
        "同一块板既在训练集也在验证集，指标不能当作泛化能力。建议把「划分口径」改成" +
        "按文件名前缀或按目录后重新扫描、导出。体检不会自动改你的数据。";
      qcLeak.classList.remove("hidden");
    } else {
      qcLeak.classList.add("hidden");
    }

    qcGroups.innerHTML = "";
    const groupItems = groups.items || [];
    if (!groupItems.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "没有发现同底板的多张图（每张都自成一组）。";
      qcGroups.appendChild(empty);
    }
    for (const [index, group] of groupItems.slice(0, rowLimit).entries()) {
      const line = document.createElement("div");
      line.className = group.cross_split ? "qc-group bad" : "qc-group";
      const splits = Object.entries(group.splits || {})
        .map(([name, count]) => `${name} ${count}`)
        .join(" / ");
      const title = document.createElement("span");
      title.textContent = `组 ${index + 1} · ${group.size} 张 · ${splits}`;
      line.appendChild(title);
      if (group.cross_split) {
        const mark = document.createElement("span");
        mark.className = "qc-tag";
        mark.textContent = "跨划分";
        line.appendChild(mark);
      }
      if (group.same_label_pairs) {
        const mark = document.createElement("span");
        mark.className = "qc-tag";
        mark.textContent = `${group.same_label_pairs} 对同标注（人工确认）`;
        line.appendChild(mark);
      }
      // 成员列表按需展开：650 张时全部展开会让页面很长
      const detail = document.createElement("details");
      const summaryLine = document.createElement("summary");
      summaryLine.textContent = "成员";
      detail.appendChild(summaryLine);
      const list = document.createElement("div");
      list.className = "qc-members";
      for (const member of group.members || []) {
        const item = document.createElement("div");
        item.textContent = `${member.split} · ${member.rel}`;
        list.appendChild(item);
      }
      if (group.members_truncated) {
        const more = document.createElement("div");
        more.className = "hint";
        more.textContent = `只列出前 ${(group.members || []).length} 张`;
        list.appendChild(more);
      }
      detail.appendChild(list);
      line.appendChild(detail);
      qcGroups.appendChild(line);
    }
    if (groupItems.length > rowLimit) {
      const more = document.createElement("p");
      more.className = "hint";
      more.textContent = `共 ${groupItems.length} 组，只显示前 ${rowLimit} 组。`;
      qcGroups.appendChild(more);
    }

    qcIssues.innerHTML = "";
    const issues = data.issues || [];
    if (!issues.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "标注未发现问题。";
      qcIssues.appendChild(empty);
      return;
    }
    // 按规则码聚合：用户关心的是「哪条规则出问题、有多少」，不是几千条平铺
    const buckets = new Map();
    for (const issue of issues) {
      if (!buckets.has(issue.code)) {
        buckets.set(issue.code, { title: issue.title, severity: issue.severity, items: [] });
      }
      buckets.get(issue.code).items.push(issue);
    }
    for (const [code, bucket] of buckets) {
      const total = (data.by_rule || {})[code];
      const count = total ? total.count : bucket.items.length;
      const box = document.createElement("details");
      box.className = `qc-issue ${bucket.severity}`;
      const summaryLine = document.createElement("summary");
      summaryLine.textContent = `${code} · ${bucket.title} · ${count} 条`;
      box.appendChild(summaryLine);
      const list = document.createElement("div");
      list.className = "qc-members";
      for (const issue of bucket.items.slice(0, rowLimit)) {
        const item = document.createElement("div");
        const where = issue.rel ? `${issue.rel}${issue.line ? `:${issue.line}` : ""}` : issue.split || "全局";
        item.textContent = `${where}　${issue.raw ? `「${issue.raw}」　` : ""}${issue.detail || ""}`;
        list.appendChild(item);
      }
      if (count > Math.min(rowLimit, bucket.items.length)) {
        const more = document.createElement("div");
        more.className = "hint";
        more.textContent = `共 ${count} 条，只显示前 ${Math.min(rowLimit, bucket.items.length)} 条。`;
        list.appendChild(more);
      }
      box.appendChild(list);
      qcIssues.appendChild(box);
    }
    if (data.issues_truncated) {
      const more = document.createElement("p");
      more.className = "hint";
      more.textContent = "问题列表已截断；上面的计数是完整的。";
      qcIssues.appendChild(more);
    }
  }

  async function runQc() {
    const body = collect();
    if (!body.source) {
      setHint("请先选择数据源", true);
      return;
    }
    qcBadge.textContent = "体检中…";
    qcBadge.className = "badge run";
    btnQc.disabled = true;
    try {
      const data = await postJSON("/api/dataset/qc", {
        source: body.source,
        task: body.task,
        threshold: 8,
        limit: 500,
        ok_dirs: body.ok_dirs,
      });
      renderQc(data);
      setHint(
        data.groups && data.groups.cross_split
          ? `体检完成：${data.groups.cross_split} 个组跨划分，指标可能虚高。`
          : `体检完成：${data.groups ? data.groups.total : 0} 组，${data.summary.error} 个错误。`,
      );
    } catch (error) {
      resetQc("体检失败");
      qcBadge.className = "badge fail";
      setHint(error.message, true);
    } finally {
      btnQc.disabled = false;
    }
  }

  btnQc.addEventListener("click", runQc);
  // 数据源/任务/口径一变，旧体检结论就不再对应当前数据，直接清掉而不是留个过期的红字
  for (const element of [source, task, groupBy, okDirs, okSplit]) {
    element.addEventListener("change", () => resetQc());
  }
  source.addEventListener("input", () => resetQc());

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
        group_by: body.group_by,
        ok_dirs: body.ok_dirs,
        ok_split: body.ok_split,
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
      const data = await postJSON("/api/dataset/export", {
        ...body,
        with_qc: Boolean(lastQc),
      });
      setBadge("导出完成", "done");
      const counts = Object.entries(data.counts || {})
        .map(([key, value]) => `${key}=${value}`)
        .join("  ");
      const fpText = data.data_fp
        ? ` · 数据指纹 ${data.data_fp}${data.unchanged ? "（内容未变）" : ""}`
        : "";
      setHint(`已写入 ${data.output}（${counts}）${fpText}`);
      renderWarnings(data.warnings);

      const lines = [
        `格式        : ${data.format}`,
        `任务        : ${data.task}`,
        `输出        : ${data.output}`,
        `划分        : ${JSON.stringify(data.split_counts)}`,
        `统计        : ${counts}`,
        `附加文件    : ${(data.extra_files || []).join(", ") || "无"}`,
      ];
      if (data.split_mode) {
        const mode = data.split_mode.mode || "ratio";
        const groups = data.split_mode.groups ? `，${data.split_mode.groups} 组` : "";
        lines.push(`划分口径    : ${mode}${groups}`);
      }
      if (data.qc) {
        const summary = data.qc.summary || {};
        const qcInfo = data.qc.groups || {};
        lines.push(
          `体检        : 错误 ${summary.error || 0} / 警告 ${summary.warn || 0} / ` +
            `提示 ${summary.info || 0}，${qcInfo.total || 0} 组，跨划分 ${qcInfo.cross_split || 0} 组（已写入 manifest）`,
        );
      }
      // 数据身份写在最显眼的位置：这份 manifest 存在的意义就是让别人事后能核对
      if (data.dataset_fingerprint) {
        lines.push(`数据集指纹  : ${data.data_fp}（完整 ${data.dataset_fingerprint}）`);
        lines.push(`标注指纹    : ${data.label_fingerprint || "—"}`);
        lines.push(
          `指纹算法    : ${data.fingerprint_algo}（路径+大小+标注，不含图片内容；改名也会变）`,
        );
        if (data.unchanged) {
          lines.push("内容比对    : 与上次导出完全一致（manifest 已刷新时间）");
        }
        lines.push(`清单文件    : ${data.manifest}`);
      } else {
        lines.push(
          `数据集指纹  : 未记录${data.fingerprint_error ? `（${data.fingerprint_error}）` : ""}`,
        );
      }
      if (data.manifest_error) lines.push(`清单写入失败: ${data.manifest_error}`);
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

  function syncTaskFormat() {
    const classify = task.value === "classify";
    for (const option of format.options) {
      option.disabled = classify && option.value !== "yolo";
    }
    if (classify && format.value !== "yolo") format.value = "yolo";
    const yolo = format.querySelector('option[value="yolo"]');
    if (yolo) yolo.textContent = classify
      ? "分类文件夹（train/类别名/图片）"
      : "YOLO（images/labels + dataset.yaml）";
  }
  task.addEventListener("change", syncTaskFormat);
  syncTaskFormat();

  refreshRatioHint();
}
