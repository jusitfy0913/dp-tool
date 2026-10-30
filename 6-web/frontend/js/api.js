// 统一的 HTTP 封装：把后端返回的错误转成可读文案，避免到处写 try/catch。

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
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
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
