// 标注页「五步流程」的纯逻辑：步骤状态、引导按钮、图片移除。
//
// 为什么单独一个文件而不是塞进 annotate.js：
// 1) annotate.js 已经 4400 多行，任何改动都要重新读懂一大片才能确认没碰坏画布；
// 2) 这里的判定（第几步该高亮、移除后当前图跳到哪、草稿面板该不该自动展开）全是
//    「输入 → 输出」的纯函数，可以用 node 直接跑单测，不必起浏览器；
// 3) 不 import 任何 DOM / 浏览器 API，测试里 import 这个文件不会因为缺少
//    document 而炸掉。annotate.js 负责把结果画到界面上。

/** 五步的顺序。改成别的顺序只需要动这一个数组。 */
export const ANNO_STEP_ORDER = ["project", "images", "classes", "draw", "save"];

/**
 * 新建工程时自带的占位类别名。
 *
 * 检测任务新建后只有一个 `object`，它是「还没想好叫什么」的占位，不能算
 * 「第 3 步已完成」，否则引导条会在用户还没想清楚缺陷种类时就跳到第 4 步。
 */
export const DEFAULT_CLASS_NAMES = ["object", "良品", "不良"];

/**
 * 类别是否已经「真的设置过」。
 *
 * 分类任务（整张图打标签）只要存在类别就算完成：它的默认类别是「良品 / 不良」，
 * 本身就是可用的业务含义。检测类任务则要求至少有一个改名后的类别。
 */
export function hasCustomClass(classes, task = "detect") {
  const list = Array.isArray(classes) ? classes : [];
  if (!list.length) return false;
  if (task === "classify") return true;
  return list.some((item) => {
    const name = String(item?.name ?? "").trim();
    return Boolean(name) && !DEFAULT_CLASS_NAMES.includes(name);
  });
}

/** 每一步在界面上的固定部分：编号、标题、点击/主按钮要触发的动作。 */
const STEP_META = {
  project: { no: 1, title: "工程", action: "create" },
  images: { no: 2, title: "加入图片", action: "addFiles" },
  classes: { no: 3, title: "设置类别", action: "focusClass" },
  draw: { no: 4, title: "画框标注", action: "toolRect" },
  save: { no: 5, title: "保存 / 导出", action: "save" },
};

/** 主按钮文案。动作名到按钮文案是一对一，所以放在同一张表里。 */
const ACTION_LABEL = {
  create: "新建工程",
  addFiles: "选图片…",
  focusClass: "填写类别",
  toolRect: "选矩形工具",
  save: "保存",
};

/** 每一步「现在该做什么」的说明。读的是调用方传进来的现场数据。 */
function stepHint(id, context) {
  const { task, hasProject, imageCount, classCount, customClass, annotationCount, classifiedCount, dirty } = context;
  const classify = task === "classify";
  switch (id) {
    case "project":
      return hasProject
        ? "工程已就绪；改动会在停手 3 秒后自动保存"
        : "填 .tfanno 路径后点「新建工程」（新工程）或「打开」（已有工程）";
    case "images":
      return imageCount > 0
        ? `已加入 ${imageCount} 张图片；点列表可切换，点行尾「×」可移出`
        : "填图片目录点「批量加入」，或点「选图片…」逐张挑";
    case "classes":
      if (classCount > 0 && (classify || customClass)) return `共 ${classCount} 个类别`;
      if (classCount > 0) return "类别名还是占位的 object，改成真实的缺陷名";
      return "在右栏输入缺陷名（如「划痕」）后点「加」";
    case "draw":
      if (classify) {
        return classifiedCount > 0
          ? `已标 ${classifiedCount}/${imageCount} 张；点右栏类别给整张图打标签`
          : "点右栏类别给整张图打标签，标完自动跳到下一张未标图";
      }
      return annotationCount > 0
        ? `当前图片有 ${annotationCount} 个框`
        : "选「矩形」工具，在图上按住拖出一个框";
    case "save":
      return dirty ? "有未保存改动：按 Ctrl+S 或点「保存」" : "按 Ctrl+S 保存，再到右栏「导出」生成训练目录";
    default:
      return "";
  }
}

/**
 * 生成五步状态。
 *
 * 返回的每项都是 `{ id, no, title, hint, action, actionLabel, state }`，
 * `state` 只有 `done / current / todo` 三种，界面只按这三个值上色。
 * 判断规则刻意保守：宁可多提醒一步，也不要让用户以为自己已经做完了。
 */
export function buildAnnoSteps(rawContext = {}) {
  const context = {
    task: "detect",
    hasProject: false,
    imageCount: 0,
    classCount: 0,
    customClass: false,
    annotationCount: 0,
    classifiedCount: 0,
    dirty: false,
    ...rawContext,
  };
  const classify = context.task === "classify";
  const done = {
    project: Boolean(context.hasProject),
    images: Number(context.imageCount) > 0,
    // 分类任务只判「有没有类别」，检测类任务要判「类别名是否已经改过」
    classes: classify
      ? Number(context.classCount) > 0
      : Number(context.classCount) > 0 && Boolean(context.customClass),
    draw: classify ? Number(context.classifiedCount) > 0 : Number(context.annotationCount) > 0,
  };
  // 保存这一步有前置条件：前面四步没走完谈不上"已保存"
  done.save = done.project && done.images && done.classes && done.draw && !context.dirty;

  let currentAssigned = false;
  return ANNO_STEP_ORDER.map((id) => {
    let state = done[id] ? "done" : "todo";
    if (state !== "done" && !currentAssigned) {
      state = "current";
      currentAssigned = true;
    }
    const meta = STEP_META[id];
    return {
      id,
      no: meta.no,
      title: meta.title,
      action: meta.action,
      actionLabel: ACTION_LABEL[meta.action] || "",
      hint: stepHint(id, context),
      state,
    };
  });
}

/** 当前应该高亮的那一步；全部完成时返回最后一步（保存），不会返回 null。 */
export function currentStep(steps) {
  const list = Array.isArray(steps) ? steps : [];
  return list.find((step) => step.state === "current") || list[list.length - 1] || null;
}

/**
 * 步骤 → 动作名。界面按这个名字分发，不做任何业务判断。
 *
 * 单独一个函数是为了让「点了哪个按钮会做什么」可枚举：测试里把所有步骤过一遍，
 * 就能保证以后加步骤时不会出现点了没反应的死按钮。
 */
export function resolveGuideAction(step) {
  if (!step || typeof step !== "object") return "";
  const action = String(step.action || "");
  return Object.prototype.hasOwnProperty.call(ACTION_LABEL, action) ? action : "";
}

/**
 * 从图片数组里移除一张，返回新数组（不改入参）。
 *
 * `focusIndex` 是"删完之后应该切到哪张"：删中间那张就停在原位置（下一张顶上来），
 * 删最后一张就退回前一张，删到空则 `-1`（画布回到空状态）。
 * 前端拿它修正 `state.currentIndex`，避免出现下标越界后画布空白但列表还在的怪状态。
 */
export function removeImageAt(images, index) {
  const list = Array.isArray(images) ? images : [];
  if (!Number.isInteger(index) || index < 0 || index >= list.length) {
    return { images: list.slice(), removed: null, focusIndex: list.length ? 0 : -1 };
  }
  const removed = list[index];
  const next = list.slice(0, index).concat(list.slice(index + 1));
  return { images: next, removed, focusIndex: next.length === 0 ? -1 : Math.min(index, next.length - 1) };
}

/**
 * 「进阶：预标注草稿与审核」该不该默认展开。
 *
 * 两条规则会互相打架，所以必须显式区分：
 * - 自动规则：工程里有草稿或有审核记录就展开——跑完批量预标注后结果要立刻可见；
 * - 用户意图：只要用户亲手点过折叠条（`touched`），自动规则永久让位。
 *
 * 不然会出现最烦人的情况：用户刚把那块收起来，下一次 renderAll 又给他弹开。
 * `open` 传当前折叠状态，`touched` 时原样返回它。
 */
export function shouldExpandDrafts({ draftCount = 0, reviewCount = 0, touched = false, open = false } = {}) {
  if (touched) return Boolean(open);
  return Number(draftCount) > 0 || Number(reviewCount) > 0;
}

/**
 * 画布 / 列表上显示的名字：`label` 优先，空则回退类别名。
 *
 * 抽出来是为了让"显示名"只有一处定义。导入的工程每个框都带 `label`，
 * 如果显示规则散落在画布、表格、状态栏各处，改一处就会漏一处。
 */
export function displayName(label, className) {
  const text = String(label ?? "").trim();
  return text || String(className ?? "").trim();
}

/**
 * 改了类别之后，这个框的 `label` 应该变成什么。
 *
 * 规则只有一条：**label 是不是用户自己起的名字**。
 * - 空 label：表示"跟随类别名"，改完仍然是空（显示层回退到新类别名）；
 * - label 与旧类别名相同：说明它是导入时自动写进去的副本（`annotationimport`
 *   会把类别名塞进 label），必须同步成新名字，否则界面上会一直显示旧类别；
 * - 其它情况：用户手工写的别名，原样保留。
 *
 * 这条规则同时解决了两个"看起来改不动"的场景：改框的类别、改名后旧框没跟着变。
 */
export function labelAfterClassChange(label, fromName, toName) {
  const current = String(label ?? "").trim();
  if (!current) return "";
  if (current === String(fromName ?? "").trim()) return String(toName ?? "").trim();
  return String(label ?? "");
}

/**
 * 批量改「集合」前的计划：只挑出确实需要改的图片，不改入参。
 *
 * 返回 `targets` 是为了让调用方统一"先判断有没有变化、再进撤销栈"——
 * 选中的图本来就都在目标集合里时不该产生一条空的撤销记录。
 */
export function planSplitChange(images, ids, split) {
  const wanted = new Set(Array.isArray(ids) ? ids : []);
  const target = String(split ?? "");
  const list = Array.isArray(images) ? images : [];
  const targets = target ? list.filter((image) => wanted.has(image.id) && image.split !== target) : [];
  return { targets, changed: targets.length };
}

/** 全部图片的草稿总数，供 `shouldExpandDrafts` 使用。 */
export function countAllDrafts(drafts) {
  if (!drafts) return 0;
  let total = 0;
  // 兼容 Map 与普通对象两种来源：前者是前端状态，后者是接口返回
  const values = typeof drafts.values === "function" ? drafts.values() : Object.values(drafts);
  for (const list of values) {
    if (Array.isArray(list)) total += list.length;
  }
  return total;
}
