// 导入已有 YOLO 标签的"计划"与汇总文案：纯函数，不碰 DOM、不发请求。
//
// 现象：导入会一次性改掉几百张图的标注，点错一次就得靠撤销回退。
// 所以流程固定为「先预览 → 用户核对 → 再导入」，本模块负责回答三个问题：
//   1. 当前设置**能不能**导入（缺路径、模式缺工程、还没预览）→ `blocking`
//   2. 后端 dry_run 报告说的是什么（几张图、几个框、哪些被跳过）→ `summary`
//   3. 给用户看的那段文字 → `text`
//
// 注意：真正的扫描与合并由后端 `annotation.py` 的 dry_run 完成，
// 前端不做第二套解析，避免两边对"孤儿标签"这类边界的判定不一致。
//
// 单测：node 6-web/frontend/tests/test_importplan.mjs

/**
 * 把后端报告整理成固定字段的对象。
 *
 * 后端字段缺失时一律按 0 / 空数组补齐，文案就不会出现 `undefined 图`。
 */
export function summarizeReport(report) {
  const raw = report || {};
  const task = raw.task || "detect";
  return {
    task,
    classes: Array.isArray(raw.classes) ? raw.classes : [],
    images: Number(raw.images) || 0,
    objects: Number(raw.objects) || 0,
    addedImages: Number(raw.added_images) || 0,
    addedObjects: Number(raw.added_objects) || 0,
    skippedExisting: Number(raw.skipped_existing) || 0,
    replacedImages: Number(raw.replaced_images) || 0,
    emptyImages: Number(raw.empty_images) || 0,
    unknownClassObjects: Number(raw.unknown_class_objects) || 0,
    orphanLabels: Number(raw.orphan_labels) || 0,
    malformedLines: Number(raw.malformed_lines) || 0,
    truncated: Boolean(raw.truncated),
    pathMode: raw.path_mode === "relative" ? "relative" : "absolute",
    warnings: Array.isArray(raw.warnings) ? raw.warnings : [],
    perSplit: raw.per_split && typeof raw.per_split === "object" ? raw.per_split : {},
  };
}

/** 分类任务用"张"，其余用"框"。 */
export function countUnit(task) {
  return task === "classify" ? "张" : "框";
}

/**
 * 当前设置下无法导入的原因（按界面出现顺序）。
 *
 * 空数组 = 可以导入。文案直接给用户看，所以写"要做什么"而不是"哪里错了"。
 */
export function blockingReasons({
  mode = "new",
  source = "",
  projectPath = "",
  hasProject = false,
  previewReady = false,
  report = null,
} = {}) {
  const reasons = [];
  if (!String(source || "").trim()) reasons.push("请先填 data.yaml 或数据集根目录");
  if (mode === "merge" && !hasProject) reasons.push("先打开一个工程，或把导入方式改成「新建工程」");
  if (mode === "new" && !String(projectPath || "").trim()) reasons.push("新建工程要先填上面的 .tfanno 路径");
  if (!previewReady) {
    reasons.push("请先「预览」，确认图数与框数没问题再导入");
  } else if (report && (Number(report.images) || 0) <= 0) {
    // 0 图还继续点导入，新建模式下会把现有工程覆盖成空的
    reasons.push("这次预览里没有可导入的图片，检查数据集目录与 split 勾选");
  }
  return reasons;
}

/** 报告 → 多行文字（预览区与导入完成后的提示共用）。 */
export function importSummaryText(report, issues = []) {
  const data = summarizeReport(report);
  const lines = [];
  const splitText = Object.entries(data.perSplit)
    .map(([name, item]) => `${name} ${Number(item?.images) || 0} 图 / ${Number(item?.objects) || 0} 框`)
    .join("　");
  lines.push(
    `任务 ${data.task} ｜ ${data.classes.length} 类 ｜ ` +
      `${data.images} 图 / ${data.objects} ${data.task === "classify" ? "张" : "框"}`,
  );
  if (splitText) lines.push(splitText);

  const bits = [];
  if (data.addedImages) bits.push(`新增 ${data.addedImages} 图 / ${data.addedObjects} 框`);
  if (data.skippedExisting) bits.push(`跳过已存在 ${data.skippedExisting} 图`);
  if (data.replacedImages) bits.push(`覆盖 ${data.replacedImages} 图`);
  if (data.emptyImages) bits.push(`空标注 ${data.emptyImages} 图`);
  if (data.unknownClassObjects) bits.push(`未知类别 ${data.unknownClassObjects} 框`);
  if (data.orphanLabels) bits.push(`孤儿标签 ${data.orphanLabels}`);
  if (data.malformedLines) bits.push(`畸形行 ${data.malformedLines}`);
  if (data.truncated) bits.push("已按上限截断");
  if (bits.length) lines.push(bits.join(" ｜ "));

  lines.push(`图片路径：${data.pathMode === "relative" ? "相对工程目录（可搬移）" : "绝对路径"}`);
  data.warnings.forEach((text) => lines.push(`⚠ ${text}`));
  (issues || []).forEach((text) => lines.push(`✗ ${text}`));
  return lines.join("\n");
}

/**
 * 综合计划：能否导入 + 汇总文案。
 *
 * `ok=false` 时调用方把「预览 / 导入」按钮置灰并显示 `blocking[0]`；
 * `issues` 非空不算阻塞——它们是"需要人看一眼"的提示，后端已经合并好了。
 */
export function planImport({
  mode = "new",
  source = "",
  projectPath = "",
  hasProject = false,
  previewReady = false,
  report = null,
  issues = [],
} = {}) {
  const blocking = blockingReasons({ mode, source, projectPath, hasProject, previewReady, report });
  const summary = summarizeReport(report);
  return {
    ok: blocking.length === 0,
    blocking,
    needsReview: (issues || []).length > 0 || summary.warnings.length > 0,
    summary,
    text: importSummaryText(report, issues),
  };
}
