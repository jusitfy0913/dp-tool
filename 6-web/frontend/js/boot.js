// 进入软件前的两步：密码，然后等训练环境预热完。
//
// 不走 api.js 的 postJSON：那个封装把 401 一律送到令牌登录页。
// 这里的 401 是「进入密码不对」，人还在这张卡片上，不该被带走。

/** 和 index.html <head> 里那段防闪脚本用的是同一个键。 */
const SESSION_KEY = "trtforge.operator.ok";

/** 从进入等待起最多干等多久。到点给「仍然进入」，不把人关在画面里。 */
const WARM_TIMEOUT_MS = 120_000;
/** 某一类一直是 idle（进程没起来）就停止干等。 */
const IDLE_GIVE_UP_MS = 15_000;
const POLL_MS = 500;

const KIND_LABEL = {
  yolo: "YOLO",
  anomalib: "异常检测",
  anomalydino: "AnomalyDINO",
};

function kindLabel(kind) {
  return KIND_LABEL[kind] || kind;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readSession() {
  try {
    return sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

function writeSession() {
  try {
    sessionStorage.setItem(SESSION_KEY, "1");
  } catch {
    // 存不下也让这次进入。关掉窗口后下次还会再问。
  }
}

/**
 * 自己读 JSON。401 / 429 留给调用方决定怎么提示，不跳转。
 * @returns {Promise<{ok:boolean, status:number, payload:any}>}
 */
async function readJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => null);
  return { ok: response.ok, status: response.status, payload };
}

async function postJson(url, body) {
  return readJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function detailOf(payload, fallback) {
  const detail = payload && (payload.detail || payload.message);
  return typeof detail === "string" && detail ? detail : fallback;
}

/**
 * @param {() => void} startApp 密码通过后、等待画面还盖着的时候，把各页初始化跑起来
 */
export function startBoot(startApp) {
  const gate = document.getElementById("bootGate");
  const loginForm = document.getElementById("bootLogin");
  const password = document.getElementById("bootPassword");
  const submit = document.getElementById("bootSubmit");
  const loginMsg = document.getElementById("bootMsg");
  const hint = document.getElementById("bootHint");
  const step = document.getElementById("bootStep");
  const waitMsg = document.getElementById("bootWaitMsg");
  const enterBtn = document.getElementById("bootEnter");
  const bar = document.getElementById("bootBar");
  if (!gate || !loginForm) {
    document.body.classList.add("app-ready");
    startApp();
    return;
  }

  let configured = false;
  let entered = false;
  let bootStarted = false;
  let loginAttempted = false;

  function showStep(text) {
    if (step) step.textContent = text;
  }

  function showLogin(message) {
    gate.classList.add("is-login");
    gate.classList.remove("is-wait");
    document.documentElement.classList.remove("boot-resume");
    if (loginMsg) {
      loginMsg.textContent = message || "";
      loginMsg.classList.remove("is-ok");
    }
  }

  function showWait() {
    gate.classList.add("is-wait");
    gate.classList.remove("is-login");
    document.documentElement.classList.remove("boot-resume");
    if (bar) bar.classList.add("is-live");
    if (enterBtn) enterBtn.hidden = true;
    if (waitMsg) waitMsg.textContent = "";
  }

  function showEscape(message) {
    if (bar) bar.classList.remove("is-live");
    if (waitMsg) waitMsg.textContent = message;
    if (enterBtn) enterBtn.hidden = false;
  }

  function enterApp() {
    if (entered) return;
    entered = true;
    document.body.classList.add("app-ready");
    document.documentElement.classList.remove("boot-resume");
  }

  function applyHints() {
    if (hint) {
      hint.textContent = configured
        ? "请输入进入密码。"
        : "还没设置密码，留空或随便填都能进入。设好之后在「环境说明」里修改。";
    }
    const cardHint = document.getElementById("operatorHint");
    const oldLabel = document.getElementById("operatorOldLabel");
    const clearHint = document.getElementById("operatorClearHint");
    if (cardHint) {
      cardHint.textContent = configured
        ? "已经设置过密码。修改时要填当前密码；两个新密码都留空就是清除。"
        : "还没设置密码。留空或随便填都能进入。在这里设好之后，下次打开必须输对。";
    }
    if (oldLabel) oldLabel.hidden = !configured;
    if (clearHint) clearHint.hidden = !configured;
  }

  async function loadStatus() {
    try {
      const result = await readJson("/api/operator/status");
      configured = !!(result.ok && result.payload && result.payload.configured);
    } catch {
      configured = false;
    }
    applyHints();
  }

  function bindOperatorCard() {
    const save = document.getElementById("operatorSave");
    const msg = document.getElementById("operatorMsg");
    const oldInput = document.getElementById("operatorOld");
    const newer = document.getElementById("operatorNew");
    const again = document.getElementById("operatorNew2");
    if (!save || !newer || !again) return;
    save.addEventListener("click", async () => {
      if (msg) {
        msg.textContent = "";
        msg.classList.remove("is-ok");
      }
      if (newer.value !== again.value) {
        if (msg) msg.textContent = "两次新密码不一致";
        return;
      }
      if (!configured && newer.value === "") {
        if (msg) msg.textContent = "请填写新密码";
        return;
      }
      save.disabled = true;
      try {
        const result = await postJson("/api/operator/password", {
          old_password: oldInput ? oldInput.value : "",
          new_password: newer.value,
        });
        if (!result.ok) {
          if (msg) msg.textContent = detailOf(result.payload, `保存失败（HTTP ${result.status}）`);
          return;
        }
        configured = !!(result.payload && result.payload.configured);
        if (oldInput) oldInput.value = "";
        newer.value = "";
        again.value = "";
        applyHints();
        if (msg) {
          msg.textContent = configured ? "密码已保存。下次打开软件要输这一次的密码。" : "密码已清除。留空或随便填又能进入。";
          msg.classList.add("is-ok");
        }
      } catch (error) {
        if (msg) msg.textContent = "无法连接服务：" + error.message;
      } finally {
        save.disabled = false;
      }
    });
  }

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    loginAttempted = true;
    if (loginMsg) loginMsg.textContent = "";
    if (submit) submit.disabled = true;
    try {
      const result = await postJson("/api/operator/login", {
        password: password ? password.value : "",
      });
      if (!result.ok) {
        showLogin(detailOf(result.payload, `登录失败（HTTP ${result.status}）`));
        if (password) password.select();
        return;
      }
      writeSession();
      await runWait();
    } catch (error) {
      showLogin("无法连接服务：" + error.message);
    } finally {
      if (submit) submit.disabled = false;
    }
  });

  if (enterBtn) enterBtn.addEventListener("click", enterApp);

  async function runWait() {
    if (bootStarted) return;
    bootStarted = true;
    showWait();
    try {
      startApp();
    } catch (error) {
      console.error("[boot] 页面初始化失败", error);
    }
    const outcome = await waitWarm();
    if (outcome === "enter") enterApp();
  }

  async function waitWarm() {
    showStep("正在连接服务…");
    try {
      await readJson("/api/health");
    } catch {
      showStep("服务还没连上，仍会继续检查。");
    }
    showStep("正在检查训练与推理环境…");
    try {
      await readJson("/api/capabilities");
    } catch {
      // 环境探测失败不拦进入，主界面顶栏会显示「后端未连接」。
    }
    showStep("正在预热训练环境…");
    try {
      await postJson("/api/warm/prewarm", {});
    } catch {
      // 预热接口失败时下面的状态查询会说明要不要等。
    }

    const started = Date.now();
    /** @type {Map<string, number>} */
    const idleSince = new Map();
    while (!entered) {
      const now = Date.now();
      if (now - started >= WARM_TIMEOUT_MS) {
        showEscape("预热超过 2 分钟。可以先进入，训练时会再等一次。");
        return "wait";
      }
      let status;
      try {
        status = await readJson("/api/warm/status");
      } catch {
        showStep("还在等预热状态…");
        await sleep(POLL_MS);
        continue;
      }
      const data = status.payload || {};
      if (!status.ok) {
        showStep("还在等预热状态…");
        await sleep(POLL_MS);
        continue;
      }
      if (!data.enabled) {
        showStep("这台机器没有开启训练预热，直接进入。");
        await sleep(800);
        return "enter";
      }
      const kinds = Array.isArray(data.kinds) ? data.kinds : [];
      if (kinds.length === 0) {
        showStep("没有需要预热的训练环境，直接进入。");
        await sleep(800);
        return "enter";
      }
      const pending = kinds.filter((item) => item.state !== "ready");
      if (pending.length === 0) {
        showStep("训练环境已就绪");
        return "enter";
      }
      showStep(`正在预热 ${pending.map((item) => kindLabel(item.kind)).join("、")}…`);
      const stuck = [];
      for (const item of kinds) {
        if (item.state === "idle") {
          if (!idleSince.has(item.kind)) idleSince.set(item.kind, now);
          if (now - idleSince.get(item.kind) >= IDLE_GIVE_UP_MS) stuck.push(item);
        } else {
          idleSince.delete(item.kind);
        }
      }
      if (stuck.length) {
        const names = stuck.map((item) => kindLabel(item.kind)).join("、");
        showEscape(`${names} 没有启动起来。可以先进入，训练时会改走冷启动。`);
        return "wait";
      }
      await sleep(POLL_MS);
    }
    return "enter";
  }

  bindOperatorCard();
  loadStatus().finally(() => {
    if (bootStarted || loginAttempted) return;
    if (readSession()) {
      runWait();
      return;
    }
    showLogin("");
    if (password) password.focus();
  });
}
