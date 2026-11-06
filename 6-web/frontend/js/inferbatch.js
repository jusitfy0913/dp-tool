// 批量 / 视频 / 摄像头推理的前端控制。
//
// 三件事放在一个模块里，是因为它们共用同一套"引擎 + 标签 + conf/nms"设置，
// 也共用同一条 WebSocket 事件通道与进度区。拆三个文件只会让设置读取逻辑
// 复制三遍，然后慢慢漂移。
//
// 三种模式的执行方式完全不同，这是有意的：
//   目录批量 / 视频 → 子进程任务，进度走 /ws/runs/{id}，可停止、可断线重连
//   摄像头          → 同步单帧接口，一次一个请求，前端自己画框（低延迟）

import { postForm, postJSON, stopRun } from "./api.js";
import { connectRunEvents } from "./ws.js";

const state = {
  mode: "dir",
  run: null,          // { runId, stream, kind, artifact_url }
  seen: { detections: 0 },
  camera: null,       // { stream, timer, ... }
};

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const value = (id) => ($(id)?.value || "").trim();
const num = (id, fallback) => {
  const parsed = Number(value(id));
  return Number.isFinite(parsed) ? parsed : fallback;
};

function fmtDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function fmtMs(ms) {
  const value = Number(ms) || 0;
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function shortName(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function badge(text, cls = "") {
  const node = $("batchBadge");
  if (!node) return;
  node.textContent = text;
  node.className = `badge ${cls}`.trim();
}

function setHint(text, isError = false) {
  const node = $("batchHint");
  if (!node) return;
  node.textContent = text || "";
  node.className = `hint${text ? "" : " hidden"}${isError ? " hint-warn" : ""}`;
}

// 引擎内部日志（[yolo.cu:913]: ... 这类）每帧都会打印一次 num_image，
// 一次批量几百张就会把真正有用的信息挤出屏幕，这里做最小化过滤：
//   1) num_image 行直接丢弃（零信息量）；
//   2) 其余引擎行只在内容变化时保留一条（输入/输出形状这类只关心首次）。
let lastEngineLine = "";

function appendLog(line) {
  const box = $("batchLog");
  if (!box) return;
  const text = String(line ?? "");
  if (/^\[[a-z_]+\.cu:\d+\]:\s*num_image\b/i.test(text)) return;
  if (/^\[[a-z_]+\.cu:\d+\]:/i.test(text)) {
    if (text === lastEngineLine) return;
    lastEngineLine = text;
  }
  box.classList.remove("hidden");
  box.textContent += `${text}\n`;
  box.scrollTop = box.scrollHeight;
}

function resetLog() {
  const box = $("batchLog");
  if (box) {
    box.textContent = "";
    box.classList.add("hidden");
  }
  lastEngineLine = "";
}

function setProgress(percent) {
  $("batchProgressWrap")?.classList.remove("hidden");
  $("batchLine")?.classList.remove("hidden");
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));
  const bar = $("batchBar");
  if (bar) bar.style.width = `${pct}%`;
  const label = $("batchPct");
  if (label) label.textContent = `${pct.toFixed(pct >= 99.95 ? 0 : 1)}%`;
}

function setStage(text) {
  const node = $("batchStage");
  if (node) node.textContent = text;
}

function setElapsed(seconds) {
  const node = $("batchElapsed");
  if (node) node.textContent = `已用 ${fmtDuration(seconds)}`;
}

function setEta(seconds) {
  const node = $("batchEta");
  if (node) node.textContent = seconds == null ? "预计剩余 —" : `预计剩余 ${fmtDuration(seconds)}`;
}

// ---------------------------------------------------------------------------
// 共用：读取上方"推理设置"
// ---------------------------------------------------------------------------
function readSettings() {
  const kind = value("inferKind") || "yolo";
  const anomalib = kind === "anomalib";
  const useMeta = $("anomUseMetaThresh")?.checked ?? true;

  return {
    kind,
    engine: kind === "ocr" ? "" : (anomalib ? value("anomEngine") : value("yoloEngine")),
    det_dir: kind === "ocr" ? value("ocrDet") : "",
    rec_dir: kind === "ocr" ? value("ocrRec") : "",
    textline_ori_dir: kind === "ocr" ? value("ocrTextline") : "",
    device: value("ocrDevice") || "cpu",
    enable_mkldnn: $("ocrMkldnn")?.checked ? 1 : 0,
    cpu_threads: num("ocrThreads", 8),
    rec_batch_size: num("ocrBatch", 6),
    det_thresh: num("ocrDetThresh", 0.3),
    det_box_thresh: num("ocrBoxThresh", 0.6),
    det_unclip_ratio: num("ocrUnclip", 1.5),
    rec_score_thresh: num("ocrRecScore", 0),
    labels: anomalib ? "" : value("yoloLabels"),
    task: value("inferTask") || "detect",
    conf: num("inferConf", 0.25),
    nms: num("inferNms", 0.45),
    metadata: anomalib ? value("anomMetadata") : "",
    model_type: anomalib ? value("anomModelType") || "Auto" : "Auto",
    image_threshold: anomalib && !useMeta ? num("anomImageThr", 0.5) : -1,
    pixel_threshold: anomalib && !useMeta ? num("anomPixelThr", 0.5) : -1,
    colormap: anomalib ? value("anomColormap") || "TURBO" : "TURBO",
  };
}

function requireEngine(settings) {
  if (settings.kind === "ocr") {
    if (!settings.det_dir || !settings.rec_dir) {
      setHint("请先在上方选择检测模型和识别模型", true);
      return false;
    }
    return true;
  }
  if (!settings.engine) {
    setHint("请先在上方选择引擎文件（.engine / .trt / .onnx）", true);
    return false;
  }
  if (settings.kind === "anomalib" && !settings.metadata) {
    setHint("Anomalib 还需要 metadata.json 路径", true);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// 共用：表格
// ---------------------------------------------------------------------------
function clearTable() {
  const body = $("batchRows");
  if (body) body.innerHTML = "";
  $("batchTableWrap")?.classList.remove("hidden");
}

function addRow(index, file, count, ok = true, note = "") {
  const body = $("batchRows");
  if (!body) return;
  const row = document.createElement("tr");
  const cells = [
    String(index),
    shortName(file),
    count == null ? "—" : String(count),
    ok ? "OK" : note || "失败",
  ];
  cells.forEach((text) => {
    const td = document.createElement("td");
    td.textContent = text;
    if (!ok) td.className = "bad";
    row.appendChild(td);
  });
  row.title = file;
  // 最新的放最上面：批量跑几千张时，用户只关心"刚刚处理的是哪张"
  body.insertBefore(row, body.firstChild);
  while (body.children.length > 200) body.removeChild(body.lastChild);
}

// ---------------------------------------------------------------------------
// 共用：任务事件流
// ---------------------------------------------------------------------------
function attachRun(run, kind) {
  state.run = { runId: run.run_id, kind, stream: null, artifact_url: run.artifact_url || "" };
  state.seen = { detections: 0 };
  clearTable();
  resetLog();
  setProgress(0);
  setStage("已启动，等待第一批结果…");
  setEta(null);
  setElapsed(0);
  badge("运行中", "run");

  const startedAt = Date.now();
  state.run.stream = connectRunEvents(run.run_id, {
    onEvent: (payload) => {
      if (payload.type === "log") {
        appendLog(payload.line);
      } else if (payload.type === "started") {
        const total = payload.total ?? payload.total_frames;
        setStage(
          `${payload.backend || ""}${payload.codec ? ` · ${payload.codec}` : ""}` +
          `${total ? ` · 共 ${total} ${kind === "video" ? "帧" : "张"}` : ""}`,
        );
        if (payload.output) appendLog(`产物目录：${payload.output}`);
        if (payload.codec_note) appendLog(payload.codec_note);
      } else if (payload.type === "progress") {
        const done = payload.done ?? payload.processed ?? 0;
        const total = payload.total ?? payload.total_frames ?? 0;
        const percent = payload.percent ?? (total ? (done * 100) / total : 0);
        setProgress(percent);
        setElapsed(payload.elapsed ?? (Date.now() - startedAt) / 1000);
        setEta(payload.eta);

        const detections = Number(payload.detections) || 0;
        const delta = Math.max(0, detections - state.seen.detections);
        state.seen.detections = detections;

        if (payload.file) addRow(done, payload.file, delta, true);
        setStage(
          kind === "video"
            ? `已处理 ${payload.processed ?? done} 帧 · ${payload.fps ?? "—"} fps · 累计检出 ${detections}`
            : `已完成 ${done} / ${total || "?"} 张 · 累计检出 ${detections}`,
        );
      } else if (payload.type === "warning") {
        appendLog(`[警告] ${payload.message}`);
        if (payload.file) addRow(doneOrDash(payload), payload.file, null, false, "失败");
      } else if (payload.type === "error") {
        appendLog(`[错误] ${payload.message}`);
        setHint(payload.message, true);
        badge("失败", "fail");
      } else if (payload.type === "summary") {
        renderSummary(kind, payload, state.run);
      }
    },
    onClose: (payload) => {
      const ok = payload.status === "finished";
      const cancelled = payload.status === "cancelled";
      appendLog(`[结束] 退出码=${payload.exit ?? "?"} 状态=${payload.status ?? "?"}`);
      badge(cancelled ? "已停止" : ok ? "已完成" : "失败", cancelled ? "warn" : ok ? "done" : "fail");
      if (ok) setProgress(100);
      setStage(cancelled ? "已停止" : ok ? "完成" : "失败，看日志定位原因");
      setEta(null);
      enableButtons(kind, true);
      if (ok && kind === "dir") setHint("批量推理完成，可下载标注包或查看 results.csv");
      state.run = null;
    },
    onError: (message) => {
      appendLog(`[事件流] ${message}`);
      setHint(message, true);
    },
  });
}

function doneOrDash(payload) {
  return payload.done ?? payload.processed ?? "—";
}

function enableButtons(kind, enabled) {
  if (kind === "dir") {
    $("btnBatchStart").disabled = !enabled;
    $("btnBatchStop").disabled = enabled;
  } else if (kind === "video") {
    $("btnVideoStart").disabled = !enabled;
    $("btnVideoStop").disabled = enabled;
  }
}

function renderSummary(kind, payload, run) {
  if (kind === "dir") {
    const link = $("batchDownload");
    if (payload.zip) {
      link.href = `/api/infer/artifact?path=${encodeURIComponent(payload.zip)}&download=1`;
      link.textContent = `下载标注包（${payload.annotated} 张 + CSV）`;
      link.classList.remove("hidden");
    } else {
      link.classList.add("hidden");
    }
    const classes = Object.entries(payload.class_histogram || {})
      .map(([name, count]) => `${name}×${count}`)
      .join("，");
    setHint(
      `完成 ${payload.processed}/${payload.total} 张，检出 ${payload.detections} 个` +
      `${payload.failed ? `，失败 ${payload.failed} 张` : ""}` +
      `｜平均 ${fmtMs(payload.avg_ms)}/张｜总耗时 ${payload.elapsed_seconds}s` +
      `${classes ? `｜分布：${classes}` : ""}`,
    );
    appendLog(`results.json：${payload.results_json || "—"}`);
    appendLog(`results.csv：${payload.results_csv || "—"}`);
    if (payload.zip) appendLog(`annotated.zip：${payload.zip}`);
  } else {
    const player = $("videoResult");
    if (run?.artifact_url) {
      player.src = run.artifact_url;
      player.classList.remove("hidden");
    }
    const link = $("videoDownload");
    if (payload.output) {
      link.href = `/api/infer/artifact?path=${encodeURIComponent(payload.output)}&download=1`;
      link.textContent = `下载 mp4（${payload.size_mb} MB）`;
      link.classList.remove("hidden");
    }
    const classes = Object.entries(payload.class_histogram || {})
      .map(([name, count]) => `${name}×${count}`)
      .join("，");
    let note =
      `编码器 ${payload.codec}｜${payload.frames_written} 帧写入｜检出 ${payload.detections} 个` +
      `｜处理速度 ${payload.avg_fps ?? "—"} fps｜总耗时 ${payload.elapsed_seconds}s` +
      `${classes ? `｜分布：${classes}` : ""}`;
    if (payload.codec !== "avc1" && payload.codec !== "H264") {
      note += "。注意：该编码器部分浏览器无法直接播放，若视频黑屏请用下载按钮交给系统播放器。";
    }
    $("videoNote").textContent = note;
    appendLog(`输出视频：${payload.output}`);
  }
}

// ---------------------------------------------------------------------------
// 目录批量
// ---------------------------------------------------------------------------
function bindDir() {
  $("btnBatchStart")?.addEventListener("click", async () => {
    const settings = readSettings();
    if (!requireEngine(settings)) return;
    if (settings.kind === "anomalib") {
      appendLog("Anomalib 批量：按 metadata（或上方阈值）逐张判定 OK / NG");
    }
    const source = value("batchSource");
    if (!source) {
      setHint("请填写图片目录路径（可点浏览选择）", true);
      return;
    }

    $("batchDownload")?.classList.add("hidden");
    enableButtons("dir", false);
    setHint("正在启动批量任务…");

    try {
      const run = await postJSON("/api/infer/batch", {
        kind: settings.kind,
        engine: settings.engine,
        det_dir: settings.det_dir,
        rec_dir: settings.rec_dir,
        textline_ori_dir: settings.textline_ori_dir,
        device: settings.device,
        enable_mkldnn: settings.enable_mkldnn,
        cpu_threads: settings.cpu_threads,
        rec_batch_size: settings.rec_batch_size,
        det_thresh: settings.det_thresh,
        det_box_thresh: settings.det_box_thresh,
        det_unclip_ratio: settings.det_unclip_ratio,
        rec_score_thresh: settings.rec_score_thresh,
        task: settings.task,
        labels: settings.labels,
        conf: settings.conf,
        nms: settings.nms,
        source,
        limit: num("batchLimit", 0),
        recursive: $("batchRecursive")?.checked ?? true,
        jpeg_quality: num("batchQuality", 92),
        make_zip: $("batchZip")?.checked ?? true,
        full_json: $("batchFullJson")?.checked ?? true,
        metadata: settings.metadata,
        model_type: settings.model_type,
        image_threshold: settings.image_threshold,
        pixel_threshold: settings.pixel_threshold,
        colormap: settings.colormap,
        heatmap: false,
      });
      attachRun(run, "dir");
      setHint("");
    } catch (error) {
      setHint(error.message, true);
      badge("失败", "fail");
      enableButtons("dir", true);
    }
  });

  $("btnBatchStop")?.addEventListener("click", async () => {
    if (!state.run || state.run.kind !== "dir") return;
    try {
      await stopRun(state.run.runId);
      appendLog("[操作] 已请求停止");
    } catch (error) {
      setHint(error.message, true);
    }
  });
}

// ---------------------------------------------------------------------------
// 视频
// ---------------------------------------------------------------------------
function bindVideo() {
  $("btnVideoStart")?.addEventListener("click", async () => {
    const settings = readSettings();
    if (!requireEngine(settings)) return;
    if (settings.kind === "ocr") {
      setHint("OCR 不支持视频推理：单张约 10 秒以上，请改用目录批量", true);
      return;
    }
    if (settings.kind !== "yolo") {
      setHint("视频推理当前只支持 YOLO；Anomalib 请用目录批量", true);
      return;
    }
    const source = value("videoSource");
    if (!source) {
      setHint("请填写视频文件路径（可点浏览选择）", true);
      return;
    }

    $("videoDownload")?.classList.add("hidden");
    $("videoNote").textContent = "";
    const player = $("videoResult");
    if (player) {
      player.classList.add("hidden");
      player.removeAttribute("src");
      player.load();
    }

    enableButtons("video", false);
    setHint("正在启动视频任务（首帧要加载引擎，约数秒）…");

    try {
      const run = await postJSON("/api/infer/video", {
        engine: settings.engine,
        task: settings.task,
        labels: settings.labels,
        conf: settings.conf,
        nms: settings.nms,
        source,
        stride: num("videoStride", 1),
        max_frames: num("videoMaxFrames", 0),
        max_side: num("videoMaxSide", 0),
        codec: value("videoCodec") || "auto",
        save_detections: $("videoSaveDetections")?.checked ?? false,
      });
      attachRun(run, "video");
      setHint("");
    } catch (error) {
      setHint(error.message, true);
      badge("失败", "fail");
      enableButtons("video", true);
    }
  });

  $("btnVideoStop")?.addEventListener("click", async () => {
    if (!state.run || state.run.kind !== "video") return;
    try {
      await stopRun(state.run.runId);
      appendLog("[操作] 已请求停止（已写入的帧会保留）");
    } catch (error) {
      setHint(error.message, true);
    }
  });
}

// ---------------------------------------------------------------------------
// 摄像头
// ---------------------------------------------------------------------------
function bindCamera() {
  $("btnCameraStart")?.addEventListener("click", startCamera);
  $("btnCameraStop")?.addEventListener("click", () => stopCamera("已关闭摄像头"));
}

async function startCamera() {
  const settings = readSettings();
  if (!requireEngine(settings)) return;
  if (settings.kind === "ocr") {
    setHint("OCR 不支持摄像头：单张约 10 秒以上，请改用目录批量", true);
    return;
  }
  if (settings.kind !== "yolo") {
    setHint("摄像头只支持 YOLO；Anomalib 请用目录批量", true);
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setHint("当前浏览器不支持摄像头采集（需要 https 或 localhost 打开页面）", true);
    return;
  }
  if (state.camera) stopCamera("");

  const [width, height] = (value("cameraResolution") || "1280x720").split("x").map(Number);
  const video = $("cameraVideo");
  const canvas = $("cameraCanvas");
  if (!video || !canvas) return;

  $("batchProgressWrap")?.classList.add("hidden");
  $("batchLine")?.classList.add("hidden");
  $("batchTableWrap")?.classList.add("hidden");
  resetLog();
  badge("摄像头启动中", "run");
  setHint("正在请求摄像头权限…");

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: width }, height: { ideal: height } },
      audio: false,
    });
  } catch (error) {
    badge("未启动", "");
    setHint(`无法打开摄像头：${error.message}`, true);
    return;
  }

  video.srcObject = stream;
  try {
    await video.play();
  } catch {
    // 少数浏览器要求用户手势后才真正播放；muted + autoplay 通常能过
  }

  // 采集用离屏 canvas（原始分辨率），叠加层 canvas 与视频同一坐标
  const capture = document.createElement("canvas");
  const captureCtx = capture.getContext("2d");
  const overlayCtx = canvas.getContext("2d");

  state.camera = {
    stream,
    capture,
    captureCtx,
    overlayCtx,
    canvas,
    video,
    inflight: false,
    errors: 0,
    frames: 0,
    detections: 0,
    lastTick: 0,
    fpsSamples: [],
    fpsCap: Math.max(1, Math.min(30, num("cameraFps", 8))),
    settings,
  };

  $("btnCameraStart").disabled = true;
  $("btnCameraStop").disabled = false;
  badge("摄像头运行中", "run");
  setHint("");

  // 用 setInterval 而不是 requestAnimationFrame：rAF 在后台标签页会停，
  // 而这里本来就该按固定帧率跑，定时器更容易控制节奏。
  state.camera.timer = setInterval(cameraTick, Math.max(30, 1000 / state.camera.fpsCap));
}

function cameraTick() {
  const cam = state.camera;
  if (!cam) return;
  // 上一帧还没回来就跳过：引擎慢时靠丢帧降延迟，而不是排队积压
  if (cam.inflight || document.hidden) return;

  const { video, capture, captureCtx } = cam;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  if (capture.width !== width || capture.height !== height) {
    capture.width = width;
    capture.height = height;
  }
  captureCtx.drawImage(video, 0, 0, width, height);
  cam.inflight = true;
  const started = performance.now();

  capture.toBlob(async (blob) => {
    if (!blob || !state.camera) {
      if (state.camera) state.camera.inflight = false;
      return;
    }
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    form.append("engine", cam.settings.engine);
    form.append("task", cam.settings.task);
    form.append("conf", String(cam.settings.conf));
    form.append("nms", String(cam.settings.nms));
    form.append("labels", cam.settings.labels);

    try {
      const result = await postForm("/api/infer/frame", form);
      if (!state.camera) return;
      const roundTrip = performance.now() - started;
      cam.errors = 0;
      cam.frames += 1;
      cam.detections += Number(result.count) || 0;
      drawOverlay(cam, result, width, height);
      updateCameraStat(cam, result, roundTrip);
    } catch (error) {
      if (!state.camera) return;
      cam.errors += 1;
      setHint(`单帧推理失败：${error.message}`, true);
      // 连续失败 8 次才停：偶发一两帧超时不该直接把摄像头关掉
      if (cam.errors >= 8) stopCamera("连续多帧失败，已停止摄像头");
    } finally {
      if (state.camera) state.camera.inflight = false;
    }
  }, "image/jpeg", 0.7);
}

function drawOverlay(cam, result, width, height) {
  const { canvas, overlayCtx } = cam;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  overlayCtx.clearRect(0, 0, width, height);

  const colors = result.class_colors || {};
  const thickness = Math.max(2, Math.round(Math.min(width, height) / 320));
  const showLabels = $("cameraDrawLabels")?.checked ?? true;

  for (const box of result.boxes || []) {
    const rgb = colors[String(box.class_label)] || [34, 211, 238];
    const color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = thickness;

    const obb = box.is_obb ? box.obb : null;
    if (obb) {
      drawObb(overlayCtx, obb, width, height);
    } else {
      overlayCtx.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    }

    for (const kpt of box.kpts || []) {
      overlayCtx.beginPath();
      overlayCtx.arc(kpt[0], kpt[1], thickness + 1, 0, Math.PI * 2);
      overlayCtx.fillStyle = "#facc15";
      overlayCtx.fill();
    }

    if (showLabels) {
      const label = `${box.label_name || `cls${box.class_label}`} ${Number(box.confidence).toFixed(2)}`;
      const fontSize = Math.max(12, Math.round(Math.min(width, height) / 40));
      const anchorX = obb ? Math.max(0, obb.cx - obb.w / 2) : box.left;
      const anchorY = obb
        ? Math.max(0, obb.cy - obb.h / 2 - fontSize - 4)
        : Math.max(0, box.top - fontSize - 4);
      overlayCtx.font = `600 ${fontSize}px "Segoe UI", sans-serif`;
      const textWidth = overlayCtx.measureText(label).width;
      overlayCtx.fillStyle = "rgba(12, 14, 18, 0.72)";
      overlayCtx.fillRect(anchorX, anchorY, textWidth + 10, fontSize + 6);
      overlayCtx.fillStyle = "#ffffff";
      overlayCtx.fillText(label, anchorX + 5, anchorY + fontSize);
    }
  }
}

function drawObb(ctx, obb, maxWidth, maxHeight) {
  const rad = Number(obb.angle) || 0;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const hw = obb.w / 2;
  const hh = obb.h / 2;
  const corners = [
    [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
  ].map(([dx, dy]) => [
    Math.max(0, Math.min(maxWidth, obb.cx + dx * cos - dy * sin)),
    Math.max(0, Math.min(maxHeight, obb.cy + dx * sin + dy * cos)),
  ]);

  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  for (let i = 1; i < corners.length; i += 1) ctx.lineTo(corners[i][0], corners[i][1]);
  ctx.closePath();
  ctx.stroke();
}

function updateCameraStat(cam, result, roundTripMs) {
  const now = performance.now();
  if (cam.lastTick) {
    const delta = now - cam.lastTick;
    if (delta > 0) {
      cam.fpsSamples.push(1000 / delta);
      if (cam.fpsSamples.length > 20) cam.fpsSamples.shift();
    }
  }
  cam.lastTick = now;
  const fps = cam.fpsSamples.length
    ? cam.fpsSamples.reduce((a, b) => a + b, 0) / cam.fpsSamples.length
    : 0;

  const timing = result.timing || {};
  const node = $("cameraStat");
  if (node) {
    node.textContent =
      `${fps.toFixed(1)} fps｜检出 ${result.count}｜推理 ${fmtMs(timing.inference_ms)}` +
      `｜往返 ${fmtMs(roundTripMs)}｜累计 ${cam.frames} 帧`;
  }
  badge(`摄像头运行中 · ${result.count} 个目标`, "run");
}

function stopCamera(message) {
  const cam = state.camera;
  if (!cam) return;
  state.camera = null;
  clearInterval(cam.timer);
  try {
    cam.stream.getTracks().forEach((track) => track.stop());
  } catch {
    // 有些浏览器在流已关闭时抛错，忽略即可
  }
  const video = $("cameraVideo");
  if (video) video.srcObject = null;
  const canvas = $("cameraCanvas");
  const ctx = canvas?.getContext("2d");
  if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  const stat = $("cameraStat");
  if (stat) stat.textContent = "未启动";
  $("btnCameraStart").disabled = false;
  $("btnCameraStop").disabled = true;
  badge("已停止", "warn");
  if (message) setHint(message);
}

// ---------------------------------------------------------------------------
// 模式切换
// ---------------------------------------------------------------------------
function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#batchModes .mode-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  document.querySelectorAll("#batchCard .batch-pane").forEach((pane) => {
    pane.classList.toggle("hidden", pane.dataset.pane !== mode);
  });
  setHint("");
  // 切走时关掉摄像头：用户明确切到别的模式，让摄像头继续占着设备不合理
  if (mode !== "camera") stopCamera("");
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
export function initInferBatch() {
  const card = $("batchCard");
  if (!card) return;

  // 共用提示位：三个模式如果各写一条 hint，切换时会互相覆盖
  const hint = document.createElement("p");
  hint.className = "hint hidden";
  hint.id = "batchHint";
  $("batchLine")?.insertAdjacentElement("afterend", hint);

  document.querySelectorAll("#batchModes .mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchMode(tab.dataset.mode));
  });

  bindDir();
  bindVideo();
  bindCamera();
  document.getElementById("inferKind")?.addEventListener("change", syncRealtimeForOcr);
  syncRealtimeForOcr();
  switchMode("dir");
}

function syncRealtimeForOcr() {
  const ocr = value("inferKind") === "ocr";
  document.querySelectorAll("#batchModes .mode-tab").forEach((tab) => {
    const blocked = ocr && tab.dataset.mode !== "dir";
    tab.disabled = blocked;
    tab.title = blocked ? "OCR 单张约 10 秒以上，不支持实时推理" : "";
  });
  if (ocr && state.mode !== "dir") {
    switchMode("dir");
    setHint("OCR 不支持视频和摄像头，已切回目录批量", true);
  }
}
