// 路径判断与「相对 root」改写 —— 各页「浏览」按钮共用。
//
// 为什么单独一个模块：main.js 在 import 时就会绑定页面事件，node 里无法直接
// 单测；把纯函数抽出来后 test_browse_paths.mjs 才能零依赖地守住这些边界
// （盘符大小写、root 为空、选中的目录在 root 外面……）。

/**
 * 是不是绝对路径。
 *
 * 覆盖三种写法：盘符（`D:\data`、`D:/data`）、UNC（`\\server\share`）、
 * POSIX（`/home/data`）。其余一律按「相对路径」处理——训练页的
 * `train/good`、全流程页的输出目录 `results` 都属于这一类。
 */
export function isAbsolutePath(value) {
  const text = String(value || "").trim();
  if (!text) return false;
  return /^[a-zA-Z]:[\\/]/.test(text) || text.startsWith("\\\\") || text.startsWith("/");
}

/**
 * 把 `path` 改写成相对 `rootText` 的子路径（分隔符统一成 `/`）。
 *
 * 用在异常检测那三个子目录字段（normal_dir / abnormal_dir / normal_test_dir）上：
 * 启动器按 `root / value` 解析它们，所以 `train/good` 才是正确写法。用户用系统
 * 选择框挑了一个绝对路径后，如果能判断出它在数据集 root 里面，就还原成相对写法，
 * 让人一眼看出这三个字段和 root 是绑在一起的。
 *
 * 不动的情况（原样返回）：
 * - root 为空、或两边有一边不是绝对路径（比如 root 相对仓库根，无从比较）；
 * - 选中的目录不在 root 里面（可能是另一块盘上的正常图库）；
 * - 选中的就是 root 本身（多半是选错了层级，保持原样让人自己发现）。
 *
 * Windows 路径大小写不敏感，比较时统一转小写；返回的相对部分保留原始大小写。
 */
export function relativizePath(rootText, path) {
  const root = String(rootText || "").trim();
  const target = String(path || "").trim();
  if (!root || !target) return target;
  if (!isAbsolutePath(root) || !isAbsolutePath(target)) return target;

  const strip = (value) => value.replace(/\//g, "\\").replace(/\\+$/, "");
  const rootClean = strip(root);
  const targetClean = strip(target);
  if (!rootClean) return target;

  const lowerRoot = rootClean.toLowerCase();
  const lowerTarget = targetClean.toLowerCase();
  if (lowerTarget === lowerRoot) return target;
  if (!lowerTarget.startsWith(`${lowerRoot}\\`)) return target;
  return targetClean.slice(rootClean.length + 1).replace(/\\/g, "/");
}
