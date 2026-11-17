// 全流程页「任务类型切换 → 输出目录默认值」的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_pipeline_defaults.mjs
//
// 为什么值得测：
// 训练页和全流程页共用一组 project / name 输入框，两种任务的默认写盘位置又不同
// （YOLO：runs/trtforge + exp；异常检测：results + latest）。切换任务时如果不换默认值，
// 异常检测的产物会写进 YOLO 的实验目录，和历史训练结果混在一起，而且界面上看不出来。
//
// 反过来，切换时如果无条件覆盖，就会把用户刚手填的目录冲掉——这比默认值不对更让人火大。
// 所以这里同时守住两个方向。

import { syncOutputDefaults } from "../js/pipeline.js";

let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) passed += 1;
  else failures.push(message);
}

function equal(actual, expected, message) {
  assert(
    actual === expected,
    `${message}（实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}）`,
  );
}

function group(name) {
  console.log(`\n── ${name}`);
}

/** 造一个够用的 input 桩：只需要 value 与 trim()。 */
function field(value) {
  return { value };
}

// ---------------------------------------------------------------------------
group("切到异常检测：仍是 YOLO 默认值的字段跟着换");
// ---------------------------------------------------------------------------
{
  const project = field("runs/trtforge");
  const name = field("exp");
  syncOutputDefaults("anomalib", project, name, "yolo");
  equal(project.value, "results", "project 换到 results");
  equal(name.value, "latest", "name 换到 latest");
}

// ---------------------------------------------------------------------------
group("切回 YOLO：还原训练页默认值");
// ---------------------------------------------------------------------------
{
  const project = field("results");
  const name = field("latest");
  syncOutputDefaults("yolo", project, name, "anomalib");
  equal(project.value, "runs/trtforge", "project 还原 runs/trtforge");
  equal(name.value, "exp", "name 还原 exp");
}

// ---------------------------------------------------------------------------
group("用户手填的值不能被切任务冲掉");
// ---------------------------------------------------------------------------
{
  const project = field("D:/exp/pcb-run");
  const name = field("my-exp-001");
  syncOutputDefaults("anomalib", project, name, "yolo");
  equal(project.value, "D:/exp/pcb-run", "自定义 project 保留");
  equal(name.value, "my-exp-001", "自定义 name 保留");
}

// ---------------------------------------------------------------------------
group("空输入框按目标任务的默认值补齐");
// ---------------------------------------------------------------------------
{
  const project = field("");
  const name = field("   ");
  syncOutputDefaults("anomalib", project, name, "yolo");
  equal(project.value, "results", "空 project 补 results");
  equal(name.value, "latest", "空白 name 补 latest");
}

// ---------------------------------------------------------------------------
group("边界：缺元素 / 未知任务类型不抛异常");
// ---------------------------------------------------------------------------
{
  syncOutputDefaults("anomalib", null, null, "yolo");
  passed += 1; // 只要没抛异常就算过
}
{
  const project = field("runs/trtforge");
  const name = field("exp");
  syncOutputDefaults("unknown-kind", project, name, "yolo");
  equal(project.value, "runs/trtforge", "未知任务类型不动字段");
  equal(name.value, "exp", "未知任务类型不动字段");
}

// ---------------------------------------------------------------------------
if (failures.length) {
  console.log(`\n✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  for (const message of failures) console.log(`  - ${message}`);
  process.exit(1);
}
console.log(`\n✓ 全部通过：${passed} 项`);
