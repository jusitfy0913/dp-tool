// 统一的 HTTP 封装：把后端返回的错误转成可读文案，避免到处写 try/catch。

/**
 * 带状态码与原始响应的错误。
 *
 * 只有 message 的话，调用方无法区分「409 修订号冲突」和「403 路径越界」——
 * 前者要弹冲突对话框，后者只要一行提示。标注页的乐观锁依赖这个区分。
 */
export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function readPayload(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json().catch(() => null);
  }
  return response.text().catch(() => null);
}

function throwIfFailed(response, payload) {
  if (response.ok) return;
  const detail = (payload && (payload.detail || payload.message)) || payload || response.statusText;
  throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), response.status, payload);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await readPayload(response);
  throwIfFailed(response, payload);
  return payload;
}

export function getJSON(url) {
  return request(url);
}

export function postJSON(url, body) {
  return request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postForm(url, formData) {
  // 不设置 Content-Type，交给浏览器自动带 multipart boundary
  return request(url, { method: "POST", body: formData });
}

/**
 * 上传表单并返回分段耗时，供推理页把「含往返」拆成三段。
 *
 * 为什么要单独一个函数：`fetch` 的 `await` 在「响应头到达」时就返回了，
 * 后面的 `json()` 才是真正的下载 + 解析。两段耗时恰好对应「上传+排队+服务端」
 * 和「下载+解析」，配合响应里的 `timing.server_ms` 就能算出网络那一段。
 *
 * 返回 `{ payload, waitMs, parseMs }`：
 * - waitMs：发出请求到收到响应头（上传 + 排队 + 服务端计算 + 首字节）
 * - parseMs：收到响应头到 JSON 解析完（下载 + 解析，体积大时非常明显）
 */
export async function postFormTimed(url, formData) {
  const started = performance.now();
  const response = await fetch(url, { method: "POST", body: formData });
  const headersAt = performance.now();
  const payload = await readPayload(response);
  const parsedAt = performance.now();
  throwIfFailed(response, payload);
  return { payload, waitMs: headersAt - started, parseMs: parsedAt - headersAt };
}

export function getCapabilities() {
  return getJSON("/api/capabilities");
}

/** 只读 anomalib 的 metadata.json：拿推荐阈值/模型类型/输入尺寸用于界面回显。 */
export function anomalyMetadataInfo(path) {
  return getJSON(`/api/infer/anomalib/metadata?${new URLSearchParams({ path: String(path || "") })}`);
}

/** 存活探针，同时带回后端版本号（用来识别「连到了旧进程」）。 */
export function getHealth() {
  return getJSON("/api/health");
}

export function listRuns() {
  // 只要最近 20 条：顶栏每 3 秒拉一次，不截断的话服务跑久了会白白搬一堆旧 run
  // （每条还带完整命令行）。正在跑的任务一定是最新的那几条。
  return getJSON("/api/runs?limit=20");
}

export function stopRun(runId) {
  return postJSON(`/api/runs/${runId}/stop`, {});
}

export function browse(path) {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return getJSON(`/api/browse${query}`);
}

export function nativeDialog(payload) {
  // 服务端在本机桌面弹系统文件框。**同步返回**：请求会一直挂到用户点确定或取消，
  // 调用方不能当成普通快速请求——要提示「请到桌面完成选择」。
  // 窗口没出现时，服务端会在十几秒内带回 reason=dialog-not-shown，不要干等五分钟。
  return postJSON("/api/browse/dialog", payload);
}

/** 关掉当前正在等的系统选择窗，让按钮从「取消」恢复成「浏览」。 */
export function cancelNativeDialog() {
  return postJSON("/api/browse/cancel", {});
}

// ---------------------------------------------------------------------------
// 标注工程
// ---------------------------------------------------------------------------
function query(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  return search.toString();
}

export function loadAnnotationProject(path) {
  return getJSON(`/api/annotation/project?${query({ path })}`);
}

export function saveAnnotationProject(payload) {
  return postJSON("/api/annotation/project", payload);
}

export function annotationImageUrl(path) {
  return `/api/annotation/image?${query({ path })}`;
}

export function annotationImageMeta(path) {
  return getJSON(`/api/annotation/image-meta?${query({ path })}`);
}

export function scanAnnotationImages(root, recursive = true) {
  // 后端是 @app.post（参数走查询串）：用 GET 会拿到 405，"批量加入图片"整条流程都失效。
  return postJSON(`/api/annotation/scan-images?${query({ root, recursive })}`, {});
}

export function exportAnnotationProject(payload) {
  return postJSON("/api/annotation/export", payload);
}

export function importAnnotationProject(payload) {
  // 导入是「读数据源 + 生成工程」，不改磁盘；落盘仍走 saveAnnotationProject，
  // 这样导入结果能参与撤销、冲突检测和自动保存。
  return postJSON("/api/annotation/import", payload);
}

export function autoLabel(payload) {
  return postJSON("/api/annotation/autolabel", payload);
}

// ---------------------------------------------------------------------------
// 预标注草稿（旁挂 `<工程名>.drafts.json`）与批量预标注
// ---------------------------------------------------------------------------
export function loadAnnotationDrafts(path) {
  return getJSON(`/api/annotation/drafts?${query({ path })}`);
}

export function saveAnnotationDrafts(payload) {
  // 默认增量：payload.images 只带被改过的图片，其余图片的草稿由后端保留。
  return postJSON("/api/annotation/drafts", payload);
}

/**
 * 只写审核状态（`K` / `Shift+K`）。
 *
 * 单独一个入口而不是复用 saveAnnotationDrafts：审核是连按的高频操作，
 * 带上草稿会与防抖自动保存互相覆盖（两个请求都在写同一份文件）。
 */
export function saveAnnotationReview(payload) {
  return postJSON("/api/annotation/drafts/review", payload);
}

/** 点 / 框提示式预标注（SAM）。 */
export function promptLabel(payload) {
  return postJSON("/api/annotation/prompt-label", payload);
}

/** 文本提示预标注（GroundingDINO，可选 SAM 细化）。 */
export function textLabel(payload) {
  return postJSON("/api/annotation/text-label", payload);
}

/** 内置模型仓库：清单 + 本地状态 + 进行中的下载任务。 */
export function listModelzoo() {
  return getJSON("/api/modelzoo");
}

export function downloadModel(id) {
  return postJSON("/api/modelzoo/download", { id });
}

export function getModelDownload(jobId) {
  return getJSON(`/api/modelzoo/download/${encodeURIComponent(jobId)}`);
}

export function cancelModelDownload(jobId) {
  return postJSON(`/api/modelzoo/download/${encodeURIComponent(jobId)}/cancel`, {});
}

export function startBatchAutoLabel(payload) {
  return postJSON("/api/annotation/autolabel/batch", payload);
}

export function getBatchAutoLabel(jobId) {
  return getJSON(`/api/annotation/autolabel/batch/${encodeURIComponent(jobId)}`);
}

export function listBatchAutoLabel() {
  return getJSON("/api/annotation/autolabel/batch");
}

export function stopBatchAutoLabel(jobId) {
  return postJSON(`/api/annotation/autolabel/batch/${encodeURIComponent(jobId)}/stop`, {});
}

/** 训练产物扫描结果（推理页与标注页共用）。 */
export function listModels() {
  return getJSON("/api/models");
}
