// WebSocket 事件流：训练/转换任务的实时日志通道。
// 断线时自动重连，后端会补发最近事件，刷新页面不丢日志。

/**
 * 判断一条事件是不是「还没处理过」的。
 *
 * 后端每次发事件都会打一个从 1 开始自增的 `seq`。重连时后端会把最近
 * EVENT_BUFFER_SIZE 条历史整段补发，前端只要认 `seq > lastSeq` 就不会重复。
 * 没有 `seq` 的（老版本后端、或者 onmessage 里拼出来的兜底日志）一律放行——
 * 去重不能变成丢日志。
 */
export function isFreshEvent(payload, lastSeq) {
  if (typeof payload?.seq !== "number") return true;
  return payload.seq > lastSeq;
}

export function connectRunEvents(runId, handlers = {}) {
  const { onEvent, onClose, onError } = handlers;
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/runs/${runId}`;

  let socket = null;
  let closedByUser = false;
  let retries = 0;
  // 已处理到的事件序号。断线重连时后端会补发最近 EVENT_BUFFER_SIZE 条历史，
  // 没有这个游标就只能整段再追加一遍——实测网络闪断一次，日志区里每个文件事件
  // 都出现两遍，指标表也会重复记同一轮。放在 open() 外面，跨重连保持。
  let lastSeq = 0;

  function open() {
    socket = new WebSocket(url);

    socket.onopen = () => {
      retries = 0;
      onEvent?.({ type: "_ws", name: "open" });
    };

    socket.onmessage = (message) => {
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch {
        payload = { type: "log", line: message.data };
      }
      // 去重：seq 由后端单调递增分配，只认比游标大的。
      // 老版本后端不带 seq（undefined），此时照旧全部放行，不做破坏性假设。
      if (!isFreshEvent(payload, lastSeq)) return;
      if (typeof payload.seq === "number") lastSeq = payload.seq;
      onEvent?.(payload);
      if (payload.type === "finished") {
        closedByUser = true;
        socket.close();
        onClose?.(payload);
      }
    };

    socket.onerror = () => {
      onError?.("事件流连接出错");
    };

    socket.onclose = () => {
      if (closedByUser) return;
      // 任务没结束时短暂断线，退避重连
      retries += 1;
      if (retries > 5) {
        onError?.("事件流已断开，请刷新页面查看结果");
        onClose?.({ status: "disconnected" });
        return;
      }
      setTimeout(open, 400 * retries);
    };
  }

  open();

  return {
    close() {
      closedByUser = true;
      socket?.close();
    },
  };
}
