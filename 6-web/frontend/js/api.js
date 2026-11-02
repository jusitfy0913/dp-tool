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

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";

  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json().catch(() => null);
  } else {
    payload = await response.text().catch(() => null);
  }

  if (!response.ok) {
    const detail = (payload && (payload.detail || payload.message)) || payload || response.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), response.status, payload);
  }
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
  return getJSON("/api/runs");
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
  return postJSON("/api/browse/dialog", payload);
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
