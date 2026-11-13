// 导入计划与汇总文案的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_importplan.mjs
//
// 为什么值得单测：
// - 阻塞原因判错，用户会看到"按钮灰着但没人告诉他为什么"；
// - 0 图的预览如果允许导入，新建模式会把现有工程覆盖成空的；
// - 汇总文案里的单位（框 / 张）与跳过原因必须来自后端报告，不能前端另算一套，
//   否则两边对"孤儿标签"的判定会不一致。

import {
  blockingReasons,
  countUnit,
  importSummaryText,
  planImport,
  summarizeReport,
} from "../js/annotate/importplan.js";

// ---------------------------------------------------------------------------
// 极简断言框架（与 test_annotate_flow.mjs 保持一致）
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function check(name, actual, expected) {
  const ok = Object.is(actual, expected);
  if (ok) {
    passed += 1;
  } else {
    failures.push(`${name}\n    期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
  }
  console.log(`${ok ? "✓" : "✗"} ${name}`);
}

function checkDeep(name, actual, expected) {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a === b) {
    passed += 1;
    console.log(`✓ ${name}`);
  } else {
    failures.push(`${name}\n    期望 ${b}，实际 ${a}`);
    console.log(`✗ ${name}`);
  }
}

function contains(name, text, needle) {
  const ok = String(text).includes(needle);
  if (ok) {
    passed += 1;
    console.log(`✓ ${name}`);
  } else {
    failures.push(`${name}\n    文案里找不到 ${JSON.stringify(needle)}：\n${text}`);
    console.log(`✗ ${name}`);
  }
}

const detectReport = {
  task: "detect",
  classes: ["a", "b"],
  images: 12,
  objects: 34,
  added_images: 10,
  added_objects: 30,
  skipped_existing: 2,
  replaced_images: 0,
  empty_images: 1,
  unknown_class_objects: 5,
  orphan_labels: 3,
  malformed_lines: 1,
  truncated: true,
  path_mode: "relative",
  warnings: ["还有 1 张图没有标签"],
  per_split: { train: { images: 10, objects: 28 }, val: { images: 2, objects: 6 } },
};

const classifyReport = {
  task: "classify",
  classes: ["合格", "不良"],
  images: 8,
  objects: 8,
  added_images: 8,
  added_objects: 8,
};

// ---------------------------------------------------------------------------
// 1. 报告归一化
// ---------------------------------------------------------------------------
const summary = summarizeReport(detectReport);
checkDeep("字段一一映射", [summary.images, summary.objects, summary.addedImages], [12, 34, 10]);
checkDeep("跳过/未知/孤儿都读出来", [summary.skippedExisting, summary.unknownClassObjects, summary.orphanLabels], [2, 5, 3]);
check("截断标记保留", summary.truncated, true);
check("相对路径标记保留", summary.pathMode, "relative");
check("警告条数", summary.warnings.length, 1);
check("split 明细保留", Object.keys(summary.perSplit).length, 2);

const empty = summarizeReport(null);
checkDeep("空报告全零", [empty.images, empty.objects, empty.addedImages], [0, 0, 0]);
checkDeep("空报告类别为空数组", empty.classes, []);
checkDeep("空报告警告为空数组", empty.warnings, []);
check("空报告默认检测任务", empty.task, "detect");
check("空报告默认绝对路径", empty.pathMode, "absolute");
const odd = summarizeReport({ task: "detect", images: "12", objects: null, classes: "a,b", path_mode: "weird" });
checkDeep("字符串数字被转成数值", [odd.images, odd.objects], [12, 0]);
checkDeep("非数组类别按空处理", odd.classes, []);
check("未知 path_mode 按绝对路径", odd.pathMode, "absolute");
check("分类任务用「张」", countUnit("classify"), "张");
check("检测任务用「框」", countUnit("detect"), "框");

// ---------------------------------------------------------------------------
// 2. 阻塞原因
// ---------------------------------------------------------------------------
checkDeep(
  "万事俱备没有阻塞",
  blockingReasons({
    mode: "new",
    source: "D:/data/data.yaml",
    projectPath: "D:/proj/demo.tfanno",
    previewReady: true,
    report: detectReport,
  }),
  [],
);
checkDeep("没填数据源就阻塞", blockingReasons({ mode: "new", projectPath: "p", previewReady: true, report: detectReport }), [
  "请先填 data.yaml 或数据集根目录",
]);
checkDeep(
  "合并模式没有工程就阻塞",
  blockingReasons({
    mode: "merge",
    source: "D:/data",
    projectPath: "p",
    hasProject: false,
    previewReady: true,
    report: detectReport,
  }),
  ["先打开一个工程，或把导入方式改成「新建工程」"],
);
checkDeep(
  "新建模式没填 .tfanno 路径就阻塞",
  blockingReasons({ mode: "new", source: "D:/data", previewReady: true, report: detectReport }),
  ["新建工程要先填上面的 .tfanno 路径"],
);
checkDeep(
  "没预览先要预览",
  blockingReasons({ mode: "new", source: "D:/data", projectPath: "p" }),
  ["请先「预览」，确认图数与框数没问题再导入"],
);
checkDeep(
  "预览没给出任何图也要拦下来",
  blockingReasons({ mode: "new", source: "D:/data", projectPath: "p", previewReady: true, report: { images: 0 } }),
  ["这次预览里没有可导入的图片，检查数据集目录与 split 勾选"],
);
checkDeep(
  "多个问题按顺序全部列出",
  blockingReasons({ mode: "merge", source: "", hasProject: false }),
  ["请先填 data.yaml 或数据集根目录", "先打开一个工程，或把导入方式改成「新建工程」", "请先「预览」，确认图数与框数没问题再导入"],
);

// ---------------------------------------------------------------------------
// 3. 汇总文案
// ---------------------------------------------------------------------------
const text = importSummaryText(detectReport, ["第 3 行类别越界"]);
contains("首行有任务/类别数/图数/框数", text, "任务 detect ｜ 2 类 ｜ 12 图 / 34 框");
contains("split 明细在一行里", text, "train 10 图 / 28 框");
contains("新增数量", text, "新增 10 图 / 30 框");
contains("跳过已存在", text, "跳过已存在 2 图");
contains("空标注数量", text, "空标注 1 图");
contains("类别对不上的框计入未知类别", text, "未知类别 5 框");
contains("孤儿标签", text, "孤儿标签 3");
contains("畸形行", text, "畸形行 1");
contains("截断提示", text, "已按上限截断");
contains("相对路径会说明可搬移", text, "相对工程目录（可搬移）");
contains("后端警告带 ⚠", text, "⚠ 还有 1 张图没有标签");
contains("issues 带 ✗", text, "✗ 第 3 行类别越界");

const classifyText = importSummaryText(classifyReport, []);
contains("分类任务的单位是张", classifyText, "8 图 / 8 张");
contains("分类任务不出现框数单位", classifyText, "任务 classify");
contains("空报告也能给出首行", importSummaryText(null, []), "任务 detect ｜ 0 类 ｜ 0 图 / 0 框");
contains("绝对路径会提示", importSummaryText({ task: "detect", images: 1, path_mode: "absolute" }, []), "图片路径：绝对路径");
check("0 计数不产生「新增 0」这类噪音", importSummaryText({ task: "detect", images: 3, objects: 3 }, []).includes("新增"), false);

// ---------------------------------------------------------------------------
// 4. 综合计划
// ---------------------------------------------------------------------------
const okPlan = planImport({
  mode: "new",
  source: "D:/data",
  projectPath: "D:/proj/a.tfanno",
  previewReady: true,
  report: detectReport,
});
check("设置齐全时可以导入", okPlan.ok, true);
checkDeep("可以导入时没有阻塞项", okPlan.blocking, []);
check("汇总字段跟着报告走", okPlan.summary.images, 12);
contains("计划自带文案", okPlan.text, "新增 10 图 / 30 框");
check("没有 issues 与警告以外的疑点", okPlan.needsReview, true);

const blocked = planImport({ mode: "new", source: "", previewReady: false });
check("缺设置时不能导入", blocked.ok, false);
check("阻塞项三缺三（数据源、路径、未预览）", blocked.blocking.length, 3);
check("缺设置时也能给出文案（全 0）", blocked.text.includes("0 图"), true);

const emptyPlan = planImport({
  mode: "new",
  source: "D:/data",
  projectPath: "D:/proj/a.tfanno",
  previewReady: true,
  report: { task: "detect", images: 0, objects: 0 },
});
check("空计划不能导入", emptyPlan.ok, false);
check("空计划只有一条阻塞", emptyPlan.blocking.length, 1);

const withIssues = planImport({
  mode: "merge",
  source: "D:/data",
  hasProject: true,
  previewReady: true,
  report: { task: "detect", images: 2, objects: 2 },
  issues: ["第 7 行少一个字段"],
});
check("issues 不阻塞（后端已经合并好）", withIssues.ok, true);
check("issues 会提示需要人工核对", withIssues.needsReview, true);
contains("issues 进文案", withIssues.text, "第 7 行少一个字段");

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log("");
if (failures.length) {
  console.error(`✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  failures.forEach((text) => console.error(`\n${text}`));
  process.exit(1);
}
console.log(`✓ 全部通过（${passed} 项）`);
