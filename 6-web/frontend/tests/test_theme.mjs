// 主题切换的零依赖单测 —— 直接用 node 跑：
//   node 6-web/frontend/tests/test_theme.mjs
//
// 被测对象是 js/theme.js：一个 classic script（不是 module），运行在浏览器里。
// 为了不引入 jsdom（本项目零构建、零依赖），这里用 node:vm 造一个最小环境：
// 只实现 theme.js 真正用到的几个成员（location / localStorage / documentElement /
// getElementById / addEventListener）。
//
// 覆盖的是"优先级"和"持久化"这两件最容易写错的事：
//   1. 默认深色（不能跟随系统，否则老用户第一次打开会以为界面坏了）
//   2. 本地存储优先于默认
//   3. 网址参数 ?theme= 优先于本地存储，并且会写回存储
//   4. 非法值（拼错、"auto"、旧版本残留）一律回落到深色
//   5. toggle() 深浅互切并持久化
//   6. 多标签页 storage 事件同步
//   7. index.html / login.html 在 <head> 里同步引入它（否则会白闪一帧）

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(resolve(here, "../js/theme.js"), "utf8");

let passed = 0;
const failures = [];

function check(label, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) passed += 1;
  else failures.push(`${label}：期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
  console.log(`${ok ? "✓" : "✗"} ${label}${ok ? "" : `  ← 实际 ${JSON.stringify(actual)}`}`);
}

/**
 * 在一个隔离的 vm 上下文里跑一遍 theme.js。
 * @param {object} options
 * @param {string} [options.search]  网址参数（含 ?）
 * @param {string|null} [options.stored] 本地存储里已有的值
 * @returns {{ theme:string, saved:string[], root:object, win:object }}
 */
function boot({ search = "", stored = null } = {}) {
  const saved = [];
  const store = new Map();
  if (stored !== null) store.set("trtforge.theme", stored);

  const windowHandlers = new Map();
  const win = {
    location: { search },
    localStorage: {
      getItem(key) {
        return store.has(key) ? store.get(key) : null;
      },
      setItem(key, value) {
        store.set(key, value);
        saved.push(value);
      },
    },
    addEventListener(type, handler) {
      windowHandlers.set(type, handler);
    },
  };

  const rootHandlers = new Map();
  const root = {
    dataset: {},
    style: {},
    setAttribute(name, value) {
      root.attrs = { ...(root.attrs || {}), [name]: value };
    },
  };
  const doc = {
    readyState: "complete",
    documentElement: root,
    getElementById() {
      return null; // 没有按钮：theme.js 必须自己容错
    },
    querySelectorAll() {
      return [];
    },
    addEventListener(type, handler) {
      rootHandlers.set(type, handler);
    },
  };

  const context = vm.createContext({ window: win, document: doc, URLSearchParams, console });
  vm.runInContext(source, context, { filename: "theme.js" });

  return { theme: root.dataset.theme, saved, root, win, windowHandlers, store };
}

console.log("── 默认值与优先级");
check("无存储、无参数 → 深色", boot().theme, "dark");
check("存储 light → 浅色", boot({ stored: "light" }).theme, "light");
check("存储 dark → 深色", boot({ stored: "dark" }).theme, "dark");
check("参数 ?theme=light → 浅色", boot({ search: "?theme=light" }).theme, "light");
check(
  "参数优先于存储（存储 dark + 参数 light）",
  boot({ search: "?theme=light", stored: "dark" }).theme,
  "light",
);

console.log("\n── 非法输入一律回落");
check("存储里是拼错的值", boot({ stored: "lite" }).theme, "dark");
check("存储里是 auto（旧版本可能写过）", boot({ stored: "auto" }).theme, "dark");
check("参数是非法值", boot({ search: "?theme=blue", stored: "light" }).theme, "light");

console.log("\n── 持久化");
const fromQuery = boot({ search: "?theme=light" });
check("网址参数会写回存储", fromQuery.saved, ["light"]);
check("只是读取存储时不重复写入", boot({ stored: "light" }).saved, []);

console.log("\n── 运行时接口");
const runtime = boot();
check("暴露存储键", runtime.win.TRTTheme.key, "trtforge.theme");
check("get() 返回当前主题", runtime.win.TRTTheme.get(), "dark");
check("toggle() 切到浅色", runtime.win.TRTTheme.toggle(), "light");
check("toggle() 写回存储", runtime.saved, ["light"]);
check("toggle() 之后 get() 同步", runtime.win.TRTTheme.get(), "light");
check("toggle() 再切回深色", runtime.win.TRTTheme.toggle(), "dark");
check("set() 接受合法值", runtime.win.TRTTheme.set("light"), "light");
check("set() 拒绝非法值并回落深色", runtime.win.TRTTheme.set("neon"), "dark");

console.log("\n── 多标签页同步");
const synced = boot({ stored: "dark" });
const storageHandler = synced.windowHandlers.get("storage");
check("注册了 storage 监听", typeof storageHandler, "function");
if (typeof storageHandler === "function") {
  storageHandler({ key: "trtforge.theme", newValue: "light" });
  check("其它标签页切浅色 → 本页跟随", synced.root.dataset.theme, "light");
  storageHandler({ key: "别的键", newValue: "light" });
  check("无关的存储键 → 不响应", synced.root.dataset.theme, "light");
}

console.log("\n── 引入位置（防白闪）");
for (const page of ["index.html", "login.html"]) {
  const html = readFileSync(resolve(here, `../${page}`), "utf8");
  const scriptAt = html.indexOf("/static/js/theme.js");
  const headEnd = html.indexOf("</head>");
  const bodyAt = html.indexOf("<body");
  const ok = scriptAt > 0 && scriptAt < headEnd && scriptAt < bodyAt;
  if (ok) passed += 1;
  else failures.push(`${page} 里的 theme.js 不在 <head> 内或晚于 <body>`);
  console.log(`${ok ? "✓" : "✗"} ${page}：theme.js 在 <head> 内同步加载`);
}

console.log("");
if (failures.length === 0) {
  console.log(`✓ 全部通过：${passed} 项`);
  process.exit(0);
} else {
  console.error(`✗ 失败 ${failures.length} 项（通过 ${passed} 项）：`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
