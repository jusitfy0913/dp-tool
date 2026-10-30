// WebSocket 事件流：训练/转换任务的实时日志通道。
// 断线时自动重连，后端会补发最近事件，刷新页面不丢日志。

export function connectRunEvents(runId, handlers = {}) {
  const { onEvent, onClose, onError } = handlers;
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/runs/${runId}`;

  let socket = null;
  let closedByUser = false;
  let retries = 0;

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
