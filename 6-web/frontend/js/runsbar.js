// 顶栏「运行中任务」条：全局的任务可见性与止损入口。
//
// 为什么需要它：
// 任务活在**后端进程**里，页面只是它的一个「窗口」。窗口一刷新，任务并没有停，
// 但页面退回「未开始」、停止按钮变灰——用户既看不到进度，也点不到停止，
// 却又可以直接再点一次「开始训练」，于是两个训练抢同一块显卡（v1.25.2 前的实测行为）。
//
// 这一条不依赖任何页面自己的状态：刷新、切页、甚至在标注页，都能看见谁在跑、
// 跑了多久，并且随时能停掉它。同时它还兼任「后端心跳」——拉不到
// /api/runs 就说明服务没了，立刻在顶栏写出来，而不是等用户点按钮收一句
// 英文的 "Failed to fetch"。

import { listRuns, stopRun } from "./api.js";

//: 轮询间隔。3 秒足够让「开训 / 结束」在一眼之内出现在顶栏，
//: 又不会让 127.0.0.1 上的服务多出可观察的负担。
const POLL_MS = 3000;

/** kind → 中文名与它所属的页签。写全是为了让用户不用猜 "infer-batch" 是什么。 */
const KIND_META = {
  "yolo-train": { label: "YOLO 训练", tab: "train" },
  "anomalib-train": { label: "Anomalib 训练", tab: "train" },
  "engine-build": { label: "引擎构建", tab: "engine" },
  "infer-batch": { label: "目录批量推理", tab: "infer" },
  "infer-video": { label: "视频推理", tab: "infer" },
};

function metaOf(kind) {
  if (KIND_META[kind]) return KIND_META[kind];
  if (kind.startsWith("pipeline-")) return { label: "全流程串联", tab: "train" };
  if (kind.endsWith("-check-env")) return { label: "环境自检", tab: "train", brief: true };
  return { label: kind, tab: "train" };
}

/** 该 kind 的中文名。导出给 test_runsbar.mjs 单测，别让内部 kind 泄漏到界面上。 */
export function kindLabel(kind) {
  return metaOf(String(kind || "")).label;
}

/**
 * 这条任务该不该出现在顶栏。
 *
 * 自检任务（`*-check-env`）通常 2 秒就结束：出现在条里只会闪一下，
 * 看起来像界面在抖，反而让人以为出了故障。所以按「短暂任务」跳过。
 */
export function isNoteworthyRun(run) {
  if (!run || run.status !== "running") return false;
  return !metaOf(String(run.kind || "")).brief;
}

/** 秒数 → 「12:03」/「1:02:03」。导出给单测：时长格式一旦错位，用户会读成别的意思。 */
export function fmtElapsed(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function initRunsBar() {
  const bar = document.getElementById("runsBar");
  if (!bar) return;

  let timer = 0;
  //: 上一次渲染的「签名」。内容没变就不重画，避免 3 秒一次的无谓 DOM 抖动
  //: （更重要的是别把用户正在点的「停止」按钮换成一个新节点）。
  let lastSignature = "";
  let stopping = new Set();
  let offline = false;

  function render(runs) {
    const signature = runs.map((run) => `${run.run_id}:${run.status}:${stopping.has(run.run_id)}`).join("|");
    if (signature === lastSignature && !offline) return;
    lastSignature = signature;

    bar.innerHTML = "";
    if (!runs.length) {
      bar.hidden = true;
      bar.className = "runs";
      return;
    }
    bar.hidden = false;
    bar.className = "runs";

    const head = document.createElement("span");
    head.className = "runs-title";
    head.textContent = `运行中 ${runs.length}`;
    bar.appendChild(head);

    for (const run of runs) {
      const meta = metaOf(String(run.kind || ""));
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "run-chip";
      chip.title = `${run.kind} · run ${run.run_id}\n点击接续这个任务的页面`;
      chip.addEventListener("click", () => {
        // 页面自己决定怎么接管（训练页恢复日志与按钮、构建页恢复进度）。
        // 冒泡到 document 由 main.js 负责把页签切过去。
        document.dispatchEvent(new CustomEvent("trtf:attach-run", { detail: { run } }));
      });

      const dot = document.createElement("span");
      dot.className = "run-dot";
      const text = document.createElement("span");
      text.className = "run-label";
      text.textContent = `${meta.label} ${fmtElapsed(run.elapsed)}`;
      chip.append(dot, text);

      const stop = document.createElement("span");
      stop.className = "run-stop";
      stop.textContent = stopping.has(run.run_id) ? "停止中…" : "停止";
      stop.addEventListener("click", async (event) => {
        // 别让点击冒泡成「接续」：想停的时候最不希望页面跳走
        event.stopPropagation();
        if (stopping.has(run.run_id)) return;
        stopping.add(run.run_id);
        render(runs);
        try {
          await stopRun(run.run_id);
        } catch (error) {
          // 任务可能刚好自己结束了（404）：不用打扰用户，下一轮轮询自然消失
          if (error?.status !== 404) {
            text.textContent = `${meta.label} 停止失败`;
            text.title = String(error?.message || error);
          }
        } finally {
          stopping.delete(run.run_id);
          poll();
        }
      });
      chip.appendChild(stop);
      bar.appendChild(chip);
    }
  }

  function renderOffline() {
    bar.hidden = false;
    bar.className = "runs offline";
    if (offline) return;
    offline = true;
    lastSignature = "";
    bar.innerHTML = "";
    const chip = document.createElement("span");
    chip.className = "run-chip";
    const dot = document.createElement("span");
    dot.className = "run-dot";
    const text = document.createElement("span");
    text.className = "run-label";
    text.textContent = "后端未连接 · 正在重试（服务可能已退出）";
    chip.append(dot, text);
    bar.appendChild(chip);
  }

  async function poll() {
    // 后台标签页不轮询：没人看的时候每 3 秒发一次请求纯属浪费
    if (document.hidden) return;
    try {
      const data = await listRuns();
      offline = false;
      // 自检任务通常 2 秒就结束，出现在条里只会闪一下，反而像故障
      const runs = (data?.runs || []).filter(isNoteworthyRun);
      render(runs);
    } catch {
      renderOffline();
    }
  }

  function loop() {
    window.clearInterval(timer);
    timer = window.setInterval(poll, POLL_MS);
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) poll();
  });

  poll();
  loop();
}
