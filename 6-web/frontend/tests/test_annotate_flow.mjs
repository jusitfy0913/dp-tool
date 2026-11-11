// 标注页五步引导与图片移除的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_annotate_flow.mjs
//
// 只测 annotateflow.js 里的纯函数。为什么值得单测：
// - 步骤状态决定"用户以为自己做完了没有"，判错会让人卡在第 3 步却找不到原因；
// - removeImageAt 的下标夹取写错，会留下"列表还在、画布空白"的越界状态；
// - shouldExpandDrafts 的两个规则（自动展开 / 用户手动折叠）必须分清，
//   混在一起就会出现"刚收起又被弹开"的经典烦扰。

import {
  DEFAULT_CLASS_NAMES,
  buildAnnoSteps,
  countAllDrafts,
  currentStep,
  hasCustomClass,
  removeImageAt,
  resolveGuideAction,
  shouldExpandDrafts,
} from "../js/annotateflow.js";

// ---------------------------------------------------------------------------
// 极简断言框架
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
  const ok = a === b;
  if (ok) {
    passed += 1;
  } else {
    failures.push(`${name}\n    期望 ${b}\n    实际 ${a}`);
  }
  console.log(`${ok ? "✓" : "✗"} ${name}`);
}

function group(title) {
  console.log(`\n── ${title}`);
}

const states = (steps) => steps.map((step) => step.state);
const currentId = (steps) => currentStep(steps)?.id;

// ---------------------------------------------------------------------------
// hasCustomClass
// ---------------------------------------------------------------------------
group("类别是否已经设置过");
check("空数组不算", hasCustomClass([], "detect"), false);
check("只有占位 object 不算", hasCustomClass([{ name: "object" }], "detect"), false);
check("改了名才算", hasCustomClass([{ name: "object" }, { name: "划痕" }], "detect"), true);
check("空格名字不算", hasCustomClass([{ name: "   " }], "detect"), false);
check("分类任务有默认良品即算", hasCustomClass([{ name: "良品" }], "classify"), true);
check("分类任务空类别仍不算", hasCustomClass([], "classify"), false);
check("非数组输入不抛异常", hasCustomClass(null, "detect"), false);
check("占位名清单与新建工程一致", DEFAULT_CLASS_NAMES.join(","), "object,良品,不良");

// ---------------------------------------------------------------------------
// buildAnnoSteps：五种阶段
// ---------------------------------------------------------------------------
group("五步状态：从空白工程一路走完");

// 1) 什么都没做：第 1 步 current，其余 todo
{
  const steps = buildAnnoSteps({});
  check("步骤数为 5", steps.length, 5);
  checkDeep("顺序固定", steps.map((step) => step.id), ["project", "images", "classes", "draw", "save"]);
  checkDeep("只有第 1 步 current", states(steps), ["current", "todo", "todo", "todo", "todo"]);
  check("当前步骤是工程", currentId(steps), "project");
  check("编号从 1 开始", steps[0].no, 1);
}

// 2) 新建了工程、还没加图：第 2 步 current
{
  const steps = buildAnnoSteps({ hasProject: true });
  checkDeep("工程 done 后停在加图", states(steps), ["done", "current", "todo", "todo", "todo"]);
}

// 3) 加了图、类别还是占位 object：第 3 步 current（这是最容易判错的一档）
{
  const steps = buildAnnoSteps({
    hasProject: true,
    imageCount: 12,
    classCount: 1,
    customClass: false,
  });
  checkDeep("占位类别不算完成", states(steps), ["done", "done", "current", "todo", "todo"]);
  check("提示点出 object 是占位名", steps[2].hint.includes("object"), true);
}

// 4) 类别改过名、还没画框：第 4 步 current
{
  const steps = buildAnnoSteps({
    hasProject: true,
    imageCount: 12,
    classCount: 1,
    customClass: true,
    annotationCount: 0,
  });
  checkDeep("停在画框", states(steps), ["done", "done", "done", "current", "todo"]);
}

// 5) 画了框但还没保存：第 5 步 current
{
  const steps = buildAnnoSteps({
    hasProject: true,
    imageCount: 12,
    classCount: 1,
    customClass: true,
    annotationCount: 3,
    dirty: true,
  });
  checkDeep("停在保存", states(steps), ["done", "done", "done", "done", "current"]);
  check("提示给出 Ctrl+S", steps[4].hint.includes("Ctrl+S"), true);
}

// 6) 全部完成：没有 current，不会被误判成"还差一步"
{
  const steps = buildAnnoSteps({
    hasProject: true,
    imageCount: 12,
    classCount: 1,
    customClass: true,
    annotationCount: 3,
    dirty: false,
  });
  checkDeep("五步全绿", states(steps), ["done", "done", "done", "done", "done"]);
  check("currentStep 兜底为最后一步", currentId(steps), "save");
}

group("五步状态：分类任务的差异");
{
  // 分类任务用默认「良品 / 不良」就算类别 OK，不需要改名
  const steps = buildAnnoSteps({
    task: "classify",
    hasProject: true,
    imageCount: 5,
    classCount: 2,
    customClass: false,
    classifiedCount: 0,
  });
  checkDeep("分类任务类别默认即算完成", states(steps), ["done", "done", "done", "current", "todo"]);
  check("分类任务提示讲整张图打标签", steps[3].hint.includes("整张图"), true);
}
{
  const steps = buildAnnoSteps({
    task: "classify",
    hasProject: true,
    imageCount: 5,
    classCount: 2,
    classifiedCount: 5,
    dirty: false,
  });
  checkDeep("分类任务标完即完成", states(steps), ["done", "done", "done", "done", "done"]);
}
{
  // 分类任务下 annotationCount 不应被当成进度（它恒为 0）
  const steps = buildAnnoSteps({
    task: "classify",
    hasProject: true,
    imageCount: 5,
    classCount: 2,
    annotationCount: 99,
    classifiedCount: 0,
  });
  check("分类任务不拿框数当进度", steps[3].state, "current");
}

group("五步状态：脏标记与保存步");
{
  // 前四步没走完时，"没改动"也不能算第 5 步完成
  const steps = buildAnnoSteps({ hasProject: true, imageCount: 3, classCount: 0, dirty: false });
  check("前置未完成时保存步不算完成", steps[4].state, "todo");
  check("保存步提示不会被提前点亮", steps[4].hint.includes("Ctrl+S"), true);
}

// ---------------------------------------------------------------------------
// resolveGuideAction：每个动作都可枚举
// ---------------------------------------------------------------------------
group("引导按钮动作");
{
  const steps = buildAnnoSteps({});
  checkDeep(
    "五个动作与步骤一一对应",
    steps.map((step) => resolveGuideAction(step)),
    ["create", "addFiles", "focusClass", "toolRect", "save"]
  );
  checkDeep(
    "主按钮文案",
    steps.map((step) => step.actionLabel),
    ["新建工程", "选图片…", "填写类别", "选矩形工具", "保存"]
  );
}
check("未知动作返回空串", resolveGuideAction({ action: "explode" }), "");
check("空步骤返回空串", resolveGuideAction(null), "");
check("原型链上的键不算动作", resolveGuideAction({ action: "toString" }), "");

// ---------------------------------------------------------------------------
// removeImageAt：下标夹取与不改入参
// ---------------------------------------------------------------------------
group("移除图片");
const sample = () => [
  { id: "a" },
  { id: "b" },
  { id: "c" },
  { id: "d" },
];
{
  const list = sample();
  const result = removeImageAt(list, 1);
  checkDeep("删中间：剩下 a/c/d", result.images.map((item) => item.id), ["a", "c", "d"]);
  check("删中间：被删的是 b", result.removed.id, "b");
  check("删中间：停在原下标（c 顶上来）", result.focusIndex, 1);
  check("不改入参", list.length, 4);
}
{
  const list = sample();
  const result = removeImageAt(list, 3);
  check("删最后一张：退到前一张", result.focusIndex, 2);
  checkDeep("删最后一张：剩 a/b/c", result.images.map((item) => item.id), ["a", "b", "c"]);
}
{
  const result = removeImageAt(sample(), 0);
  check("删第一张：停在 0", result.focusIndex, 0);
}
{
  const result = removeImageAt([{ id: "only" }], 0);
  check("删唯一一张：空了", result.images.length, 0);
  check("删唯一一张：focusIndex 为 -1", result.focusIndex, -1);
}
{
  const result = removeImageAt(sample(), 99);
  check("越界下标不删任何东西", result.images.length, 4);
  check("越界下标没有 removed", result.removed, null);
}
{
  const result = removeImageAt(null, 0);
  check("空输入不抛异常", result.images.length, 0);
  check("空输入 focusIndex 为 -1", result.focusIndex, -1);
}
{
  // 负下标 / 小数都不能被当成合法下标（slice 会把 -1 解释成"最后一个"）
  const result = removeImageAt(sample(), -1);
  check("负下标不生效", result.images.length, 4);
  const decimal = removeImageAt(sample(), 1.5);
  check("小数下标不生效", decimal.images.length, 4);
}

// ---------------------------------------------------------------------------
// shouldExpandDrafts / countAllDrafts
// ---------------------------------------------------------------------------
group("草稿面板展开策略");
check("没草稿不展开", shouldExpandDrafts({ draftCount: 0, reviewCount: 0 }), false);
check("有草稿自动展开", shouldExpandDrafts({ draftCount: 3, reviewCount: 0 }), true);
check("只有审核记录也展开", shouldExpandDrafts({ draftCount: 0, reviewCount: 2 }), true);
check("用户收起后不再自动弹开", shouldExpandDrafts({ draftCount: 99, reviewCount: 9, touched: true, open: false }), false);
check("用户展开后保持展开", shouldExpandDrafts({ draftCount: 0, reviewCount: 0, touched: true, open: true }), true);
check("缺省参数不抛异常", shouldExpandDrafts(), false);
{
  const map = new Map([
    ["img1", [{ id: "d1" }, { id: "d2" }]],
    ["img2", [{ id: "d3" }]],
    ["img3", []],
  ]);
  check("Map 草稿计数", countAllDrafts(map), 3);
  check("普通对象草稿计数", countAllDrafts({ a: [1, 2], b: [] }), 2);
  check("null 草稿计数为 0", countAllDrafts(null), 0);
  // 坏数据不能把整页统计带崩
  check("非数组值被忽略", countAllDrafts({ a: "oops", b: [1] }), 1);
}

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log("");
if (failures.length) {
  console.log(`✗ 失败 ${failures.length} 项：`);
  for (const item of failures) console.log(`  - ${item}`);
  console.log(`\n通过 ${passed} 项，失败 ${failures.length} 项`);
  process.exit(1);
}
console.log(`✓ 全部通过：${passed} 项`);
