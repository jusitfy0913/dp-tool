// 日志出口：把面板里的内容复制走，或打开磁盘上那份完整日志。
//
// 为什么需要这一小块：
// 1. 面板只留最近若干行（训练 3000 行），真正的错误往往在更早的地方，
//    用户要贴给别人看得能拿到全文；
// 2. 任务结束后内存里的 run 就没了，但 `runtime/logs/{run_id}.log` 一直在，
//    后端 `/api/runs/{id}/log` 直接吐文件，比让用户自己去翻目录靠谱。

import { copyText } from "./ui.js";

/** 任务日志的下载地址（后端返回 text/plain，浏览器里可以直接全选复制）。 */
export function runLogUrl(runId) {
  return `/api/runs/${encodeURIComponent(runId)}/log`;
}

// 复制能力统一在 ui.copyText 实现：这里保留同名再导出，
// 是为了不改动已经 import 本模块的调用方（logs.js 属于最早的使用者）。
export { copyText };

/**
 * 把「复制 / 日志文件」两个按钮接到一块日志面板上。
 *
 * @param {object} opts
 * @param {HTMLButtonElement} opts.copyBtn 复制按钮
 * @param {HTMLButtonElement} opts.openBtn 打开日志文件按钮
 * @param {() => string} opts.getText 取当前面板文本
 * @param {() => string|null} opts.getRunId 取当前 run_id
 */
export function createLogActions({ copyBtn, openBtn, getText, getRunId }) {
  let flashTimer = 0;
  const originalLabel = copyBtn.textContent;

  async function onCopy() {
    const text = getText();
    if (!text) return;
    const ok = await copyText(text);
    // 就地反馈：按钮自己变绿 1.2 秒，不弹 toast 打断盯训练
    clearTimeout(flashTimer);
    copyBtn.classList.toggle("copied", ok);
    copyBtn.textContent = ok ? "已复制" : "复制失败";
    flashTimer = setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyBtn.textContent = originalLabel;
    }, 1200);
  }

  function onOpen() {
    const runId = getRunId();
    if (!runId) return;
    // noopener：日志页是同源的纯文本，没必要把 window.opener 交出去
    window.open(runLogUrl(runId), "_blank", "noopener");
  }

  copyBtn.addEventListener("click", onCopy);
  openBtn.addEventListener("click", onOpen);

  return {
    /** 任务开始 / 接管 / 有历史 run 时调用：有 run 才能打开日志文件。 */
    sync() {
      openBtn.disabled = !getRunId();
    },
    /** 出现第一行日志时调用（不必每行都写 DOM）。 */
    markText() {
      if (copyBtn.disabled) copyBtn.disabled = false;
    },
    /** 切到新任务前清空：上一个任务的日志不能拿来复制。 */
    reset() {
      copyBtn.disabled = true;
      openBtn.disabled = true;
    },
  };
}
