// 「输出引擎」文件名规范化的单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_engine_out.mjs
//
// 为什么需要这层：浏览弹窗受浏览器限制只能选目录，选中后字段里是个纯目录，
// 后端会以「输出文件名需以 .engine / .trt / .plan 结尾」拒绝。
// 规范化必须在前端把目录补成完整文件名，且不能覆盖用户手工改的名字。
// 这类规则一旦写错，用户看到的是「选完还是报错」，只能靠人肉复现，所以抽成纯函数单测。

import { engineOutName, normalizeEngineOut } from "../js/engine.js";

// ---------------------------------------------------------------------------
// 极简断言框架（与 test_viewer_math.mjs 保持一致，避免多一套工具）
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function assert(condition, message) {
  if (condition) {
    passed += 1;
  } else {
    failures.push(message);
  }
}

function equal(actual, expected, message) {
  assert(actual === expected, `${message}（实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}）`);
}

function group(name) {
  console.log(`\n── ${name}`);
}

// ---------------------------------------------------------------------------
group("默认文件名推导");
// ---------------------------------------------------------------------------
equal(
  engineOutName("D:\\test\\dp-tool\\weights\\best.onnx", "fp16"),
  "best_fp16.engine",
  "应取 onnx 文件名 + 精度",
);
equal(
  engineOutName("/home/u/run/model.transd.onnx", "fp32"),
  "model.transd_fp32.engine",
  "只去掉最后一个 .onnx，中间的点要保留",
);
equal(engineOutName("", "fp16"), "", "没有 onnx 时不应该编造名字");
equal(
  engineOutName("D:\\weights\\best.ONNX", "fp16"),
  "best_fp16.engine",
  "后缀大小写不应影响推导",
);

// ---------------------------------------------------------------------------
group("目录补全成完整文件名");
// ---------------------------------------------------------------------------
equal(
  normalizeEngineOut("D:\\test\\deeplearning\\dp-tool", "D:\\test\\dp-tool\\weights\\best.onnx", "fp16"),
  "D:\\test\\deeplearning\\dp-tool\\best_fp16.engine",
  "纯目录应补上 onnx 推导出的文件名",
);
equal(
  normalizeEngineOut("D:\\out\\", "D:\\w\\best.onnx", "fp16"),
  "D:\\out\\best_fp16.engine",
  "结尾分隔符不应产生 \\\\ 双反斜杠",
);
equal(
  normalizeEngineOut("/home/u/out", "/home/u/w/best.onnx", "fp32"),
  "/home/u/out/best_fp32.engine",
  "POSIX 路径应使用 / 拼接",
);
equal(
  normalizeEngineOut("D:\\out", "", "fp16"),
  "D:\\out\\model_fp16.engine",
  "还没选 onnx 时也要给出可用路径，不能把目录原样送后端",
);

// ---------------------------------------------------------------------------
group("不该动的情况");
// ---------------------------------------------------------------------------
equal(
  normalizeEngineOut("D:\\out\\my.engine", "D:\\w\\best.onnx", "fp16"),
  "D:\\out\\my.engine",
  "已是 .engine 时必须原样保留",
);
equal(
  normalizeEngineOut("D:\\out\\my.trt", "", "fp16"),
  "D:\\out\\my.trt",
  ".trt 也要认",
);
equal(
  normalizeEngineOut("D:\\out\\my.plan", "", "fp32"),
  "D:\\out\\my.plan",
  ".plan 也要认",
);
equal(
  normalizeEngineOut("", "D:\\w\\best.onnx", "fp16"),
  "",
  "空值保持「留空自动推导」语义，不能替用户决定",
);
equal(
  normalizeEngineOut("   ", "D:\\w\\best.onnx", "fp16"),
  "   ".trim(),
  "纯空白等同空值",
);
equal(
  normalizeEngineOut("best.engine", "", "fp16"),
  "best.engine",
  "相对文件名不擅自加目录",
);

// ---------------------------------------------------------------------------
group("贴错成 onnx 路径的补救");
// ---------------------------------------------------------------------------
equal(
  normalizeEngineOut("D:\\out\\best.onnx", "D:\\w\\best.onnx", "fp16"),
  "D:\\out\\best_fp16.engine",
  "把 onnx 路径填进输出框时应换后缀，而不是拼成 best.onnx\\xxx.engine",
);
equal(
  normalizeEngineOut("best.ONNX", "", "fp16"),
  "best_fp16.engine",
  "后缀大小写也要能救",
);

// ---------------------------------------------------------------------------
group("没有路径成分时不猜");
// ---------------------------------------------------------------------------
equal(
  normalizeEngineOut("myengine", "D:\\w\\best.onnx", "fp16"),
  "myengine",
  "只给了个名字，不擅自加目录（否则会落到服务端 cwd，路径不可预期）",
);

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
console.log(`\n${failures.length ? "失败" : "通过"}：${passed} 项断言`);
if (failures.length) {
  failures.forEach((message) => console.error(`  ✗ ${message}`));
  process.exit(1);
}
console.log("全部通过");
