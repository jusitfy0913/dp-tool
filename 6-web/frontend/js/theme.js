/* ============================================================================
 * 主题切换 —— 深色 / 浅色
 *
 * 为什么是一个同步的 classic script（而不是 module）：
 *   module 天然是 defer 语义，要等 DOM 解析完才执行，那时首帧已经画过了。
 *   深色主题的用户会先看到一帧浅色底再"闪"成深色（FOUC 的孪生问题）。
 *   这个文件必须在 <head> 里、样式表之后同步执行，赶在首帧之前把
 *   html[data-theme] 写上，样式表里那套令牌才会从一开始就生效。
 *
 * 优先级（高 → 低）：
 *   1. 网址参数 ?theme=light|dark   —— 出文档截图、验收时强制指定，并记进本地
 *   2. localStorage["trtforge.theme"]
 *   3. 默认深色
 *
 * 为什么不跟随系统 prefers-color-scheme 作为默认值：
 *   这是给产线用户看检测结果的工业工具，浅色/深色是"判读条件"的一部分。
 *   第一次打开就按系统偏好变成白底，会让老用户以为界面坏了。
 *   想跟随系统的用户点一次切换，选择就会被记住。
 * ========================================================================== */
(function () {
  "use strict";

  var KEY = "trtforge.theme";
  var DARK = "dark";
  var LIGHT = "light";

  /** 只接受两个合法值，其余（含 null）返回 null。 */
  function normalize(value) {
    if (value === LIGHT) return LIGHT;
    if (value === DARK) return DARK;
    return null;
  }

  /** 读网址参数。用 try/catch 兜住 file:// 或异常环境下 URL 解析失败。 */
  function fromQuery() {
    try {
      return normalize(new URLSearchParams(window.location.search).get("theme"));
    } catch (error) {
      return null;
    }
  }

  /** 读本地存储。隐私模式 / 禁用了存储时 localStorage 会抛异常，不能让整页挂掉。 */
  function fromStorage() {
    try {
      return normalize(window.localStorage.getItem(KEY));
    } catch (error) {
      return null;
    }
  }

  function save(value) {
    try {
      window.localStorage.setItem(KEY, value);
    } catch (error) {
      // 存不下就算了：本次会话内主题仍然正确，只是下次打开回到默认。
    }
  }

  /**
   * 用户手动切换后，把网址里的 ?theme= 摘掉。
   *
   * 不摘的话会出现一个反直觉的现象：用 ?theme=light 打开（出截图常用），
   * 手动切成深色，刷新一下又变回浅色 —— 因为网址参数的优先级高于存储。
   * 摘掉之后，"用户最后的选择"才真正是唯一权威。
   */
  function dropQueryHint() {
    try {
      if (!window.location.href) return; // 测试桩里可能没有完整的 location
      const url = new URL(window.location.href);
      if (!url.searchParams.has("theme")) return;
      url.searchParams.delete("theme");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    } catch (error) {
      // 拿不到 history（老浏览器 / 沙箱）时保持原样，不影响主题本身。
    }
  }

  var initial = fromQuery() || fromStorage() || DARK;

  /** 写入 DOM。color-scheme 一并设置，让原生控件（下拉、滚动条、日期选择器）
   *  跟着换色，否则浅色主题里会出现黑色的系统下拉箭头。 */
  function apply(value) {
    var theme = normalize(value) || DARK;
    var root = document.documentElement;
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    return theme;
  }

  var current = apply(initial);
  // 网址参数是"显式指定"，顺带记住：否则刷新一下又回到旧主题，截图对不上。
  if (fromQuery()) save(current);

  /** 顶栏和登录遮罩上的按钮文案随主题变化。
   *  现取、不缓存：按钮在 DOMContentLoaded 之后才存在，登录遮罩上还有一颗。 */
  function themeButtons() {
    if (typeof document.querySelectorAll === "function") {
      return document.querySelectorAll(".theme-toggle");
    }
    var one = document.getElementById("themeToggle");
    return one ? [one] : [];
  }

  function syncToggleLabel() {
    var text = "切换到" + (current === LIGHT ? "深色" : "浅色") + "主题";
    var buttons = themeButtons();
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-label", text);
      buttons[i].title = text;
    }
  }

  /** 供测试与其它模块使用的最小接口。 */
  window.TRTTheme = {
    key: KEY,
    get: function () {
      return current;
    },
    set: function (value) {
      current = apply(value);
      save(current);
      dropQueryHint();
      syncToggleLabel();
      return current;
    },
    toggle: function () {
      return window.TRTTheme.set(current === LIGHT ? DARK : LIGHT);
    },
  };

  /** 绑定顶栏按钮。
   *  脚本在 <head> 里执行，此时按钮还不存在，所以挂到 DOMContentLoaded。
   *  用 addEventListener 而不是内联 onclick，保持零构建与 CSP 友好。 */
  function bindToggle() {
    var buttons = themeButtons();
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function () {
        window.TRTTheme.toggle();
      });
    }
    syncToggleLabel();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindToggle, { once: true });
  } else {
    bindToggle();
  }

  /** 多标签页同步：在 A 标签切了主题，B 标签不该停在上一个配色上。 */
  window.addEventListener("storage", function (event) {
    if (event.key !== KEY) return;
    current = apply(normalize(event.newValue) || DARK);
    syncToggleLabel();
  });
})();
