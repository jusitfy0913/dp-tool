// 顶栏「运行中任务」条的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_runsbar.mjs
//
// 为什么这层值得测：
// 刷新页面后「停止」按钮能不能找到任务，全靠这条子里的判断。
// 一旦过滤条件写错（比如把「已结束」也算进来、或把真在跑的训练过滤掉），
// 用户看到的就是一条假的任务条，或者一条都没有——两种情况都比报错更难排查。

import { fmtElapsed, isNoteworthyRun, kindLabel } from "../js/runsbar.js";

// ---------------------------------------------------------------------------
// 极简断言框架（与 test_engine_out.mjs 保持一致）
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) passed += 1;
  else failures.push(message);
}

function equal(actual, expected, message) {
  assert(actual === expected, `${message}（实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}）`);
}

function group(name) {
  console.log(`\n── ${name}`);
}

function run(kind, status = "running", extra = {}) {
  return { run_id: "r1", kind, status, elapsed: 10, ...extra };
}

// ---------------------------------------------------------------------------
group("时长格式");
// ---------------------------------------------------------------------------
equal(fmtElapsed(0), "0:00", "0 秒");
equal(fmtElapsed(9), "0:09", "秒数补零");
equal(fmtElapsed(59), "0:59", "59 秒不进位");
equal(fmtElapsed(60), "1:00", "整分钟");
equal(fmtElapsed(3725), "1:02:05", "超过 1 小时要带小时位");
equal(fmtElapsed(-5), "0:00", "负数按 0 处理（时钟回拨也不能显示负时长）");
equal(fmtElapsed("88"), "1:28", "字符串输入按数字解析");
equal(fmtElapsed(undefined), "0:00", "缺字段不抛异常");

// ---------------------------------------------------------------------------
group("任务中文名");
// ---------------------------------------------------------------------------
equal(kindLabel("yolo-train"), "YOLO 训练", "训练页的 YOLO");
equal(kindLabel("anomalib-train"), "Anomalib 训练", "训练页的 Anomalib");
equal(kindLabel("engine-build"), "引擎构建", "构建页");
equal(kindLabel("infer-batch"), "目录批量推理", "批量推理");
equal(kindLabel("infer-video"), "视频推理", "视频推理");
equal(kindLabel("pipeline-build"), "全流程串联", "串联阶段统一叫「全流程串联」，不暴露内部 kind");
equal(kindLabel("yolo-check-env"), "环境自检", "自检");
equal(kindLabel("some-future-kind"), "some-future-kind", "未知 kind 原样显示，不能变成 undefined");

// ---------------------------------------------------------------------------
group("哪些任务该出现在顶栏");
// ---------------------------------------------------------------------------
assert(isNoteworthyRun(run("yolo-train")), "正在跑的训练必须出现（否则用户没有停止入口）");
assert(isNoteworthyRun(run("engine-build")), "正在跑的构建必须出现");
assert(isNoteworthyRun(run("infer-video")), "正在跑的视频推理必须出现");
assert(isNoteworthyRun(run("pipeline-train")), "串联阶段必须出现");

assert(!isNoteworthyRun(run("yolo-train", "finished")), "已结束的任务不该占着条");
assert(!isNoteworthyRun(run("yolo-train", "failed")), "失败的任务不该占着条");
assert(!isNoteworthyRun(run("yolo-train", "cancelled")), "已停止的任务不该占着条");
assert(!isNoteworthyRun(null), "空值不抛异常");
assert(!isNoteworthyRun({}), "缺 status 不抛异常");

// 自检 2 秒就结束，出现只会让条子闪一下，像故障
assert(!isNoteworthyRun(run("yolo-check-env")), "自检任务不该闪现");
assert(!isNoteworthyRun(run("anomalib-check-env")), "Anomalib 自检同样不该闪现");
// 但不能因为「叫 check-env」就把真的训练过滤掉
assert(isNoteworthyRun(run("yolo-train")), "过滤自检不能误伤训练");

// ---------------------------------------------------------------------------
if (failures.length) {
  console.log(`\n✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  for (const message of failures) console.log(`  - ${message}`);
  process.exit(1);
}
console.log(`\n✓ 全部通过：${passed} 项`);
