// 训练页与全流程页共用的「任务类型 → 默认输出目录」规则。
//
// 为什么单独一个模块：两个页面共用一组 project / name 输入框，默认值必须一致；
// 而 pipeline.js 已经 import 了 train.js（focusRun），train.js 再反向 import
// pipeline.js 就成了循环依赖。默认值抄两份迟早漂移——v1.42.0 之前训练页就是
// 无条件覆盖，用户刚用「浏览」选好的输出目录，切一下任务类型就没了。

/** 两种任务的默认写盘位置：YOLO 落 runs，异常检测落 results。 */
export const KIND_OUTPUT_DEFAULTS = {
  yolo: { project: "runs/trtforge", name: "exp" },
  anomalib: { project: "results", name: "latest" },
};

/**
 * 切换任务类型时同步 project / name 的默认值。
 *
 * 只替换「仍等于上一个任务默认值」或空着的字段：用户手填（或刚用浏览按钮选好）
 * 的目录不能被切一下任务就冲掉，那比默认值不对更让人火大。
 *
 * @param {string} kind 目标任务类型（yolo / anomalib）
 * @param {{value: string}|null} projectEl 输出目录输入框
 * @param {{value: string}|null} nameEl 实验名输入框
 * @param {string} fromKind 切换前的任务类型，用来判断字段里是不是「旧默认值」
 */
export function syncOutputDefaults(kind, projectEl, nameEl, fromKind) {
  const target = KIND_OUTPUT_DEFAULTS[kind];
  const previous = KIND_OUTPUT_DEFAULTS[fromKind] || target;
  if (!target) return;
  if (projectEl && (!projectEl.value.trim() || projectEl.value.trim() === previous.project)) {
    projectEl.value = target.project;
  }
  if (nameEl && (!nameEl.value.trim() || nameEl.value.trim() === previous.name)) {
    nameEl.value = target.name;
  }
}
