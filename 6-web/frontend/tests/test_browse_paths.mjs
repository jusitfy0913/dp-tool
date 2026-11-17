// 「浏览」按钮选完路径后的改写规则单测 —— 零依赖，直接用 node 跑：
//   node 6-web/frontend/tests/test_browse_paths.mjs
//
// 为什么值得测：
// 异常检测的 normal_dir / abnormal_dir / normal_test_dir 是「相对数据集 root」的子目录，
// 启动器按 root / value 解析。用户用系统选择框挑了一个绝对目录后，如果改写错了，
// 轻则字段变得又长又看不懂，重则路径指向别处、训练读到空目录。
//
// 这里守住四条：root 内 → 相对；root 外 → 原样；root 为空/相对 → 原样；
// 大小写不同的盘符仍算同一个 root。

import { isAbsolutePath, relativizePath } from "../js/paths.js";

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

// ---------------------------------------------------------------------------
group("绝对路径判断");
// ---------------------------------------------------------------------------
equal(isAbsolutePath("D:\\data\\bottle"), true, "盘符 + 反斜杠");
equal(isAbsolutePath("d:/data/bottle"), true, "盘符 + 正斜杠");
equal(isAbsolutePath("\\\\server\\share\\data"), true, "UNC 路径");
equal(isAbsolutePath("/mnt/data"), true, "POSIX 路径");
equal(isAbsolutePath("train/good"), false, "相对子目录");
equal(isAbsolutePath("results"), false, "相对输出目录");
equal(isAbsolutePath(""), false, "空串");
equal(isAbsolutePath("   "), false, "纯空白");

// ---------------------------------------------------------------------------
group("root 内部：改写成相对路径，分隔符统一成 /");
// ---------------------------------------------------------------------------
equal(
  relativizePath("D:\\datasets\\bottle", "D:\\datasets\\bottle\\train\\good"),
  "train/good",
  "正常训练图",
);
equal(
  relativizePath("D:\\datasets\\bottle", "D:/datasets/bottle/test/abnormal"),
  "test/abnormal",
  "选择框返回正斜杠也能比较",
);
equal(
  relativizePath("D:\\datasets\\bottle\\", "D:\\datasets\\bottle\\test\\good"),
  "test/good",
  "root 末尾多余的斜杠不影响",
);
equal(
  relativizePath("d:\\datasets\\bottle", "D:\\datasets\\bottle\\train\\good"),
  "train/good",
  "盘符大小写不同仍算同一个 root，相对部分保留原始大小写",
);
equal(
  relativizePath("D:\\datasets\\bottle", "D:\\datasets\\bottle\\a\\b\\c"),
  "a/b/c",
  "多层子目录一并保留",
);

// ---------------------------------------------------------------------------
group("不改写的情况：原样返回，绝不猜");
// ---------------------------------------------------------------------------
equal(
  relativizePath("D:\\datasets\\bottle", "D:\\other\\train\\good"),
  "D:\\other\\train\\good",
  "root 外面的目录（比如另一块盘上的正常图库）保持绝对路径",
);
equal(
  relativizePath("D:\\datasets\\bottle", "D:\\datasets\\bottle2\\train"),
  "D:\\datasets\\bottle2\\train",
  "前缀相同但不是同一个目录（bottle vs bottle2）不能算子目录",
);
equal(
  relativizePath("", "D:\\datasets\\bottle\\train\\good"),
  "D:\\datasets\\bottle\\train\\good",
  "root 为空时无从比较，保持绝对路径",
);
equal(
  relativizePath("datasets/bottle", "D:\\datasets\\bottle\\train\\good"),
  "D:\\datasets\\bottle\\train\\good",
  "root 本身是相对路径（仓库内写法），不去猜它对应哪个盘",
);
equal(
  relativizePath("D:\\datasets\\bottle", "train/good"),
  "train/good",
  "字段里本来就是相对值，原样保留",
);
equal(
  relativizePath("D:\\datasets\\bottle", "D:\\datasets\\bottle"),
  "D:\\datasets\\bottle",
  "选中的就是 root 本身：多半是选错层级，保持原样让人自己发现",
);

// ---------------------------------------------------------------------------
group("边界：空值与纯空白不抛异常");
// ---------------------------------------------------------------------------
equal(relativizePath("", ""), "", "两边都为空");
equal(relativizePath("   ", "   "), "", "纯空白被 trim 掉");
equal(relativizePath(null, "D:\\a\\b"), "D:\\a\\b", "null root");
equal(relativizePath("D:\\a", null), "", "null path");

// ---------------------------------------------------------------------------
if (failures.length) {
  console.log(`\n✗ ${failures.length} 项失败 / 共 ${passed + failures.length} 项`);
  for (const message of failures) console.log(`  - ${message}`);
  process.exit(1);
}
console.log(`\n✓ 全部通过：${passed} 项`);
