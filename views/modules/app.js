const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
const activePhases = new Set(["queued", "installing", "downloading", "validating"]);
const failurePhases = new Set(["error", "interrupted", "restart_required"]);
const phaseNames = {
  not_installed: "未安装", installed: "已安装", enabled: "已启用", queued: "排队中",
  installing: "安装依赖", downloading: "下载模型", validating: "加载验证",
  error: "安装失败", interrupted: "安装中断", restart_required: "需要重启",
};
const cards = new Map();
let busy = false;
let stopped = false;
let networkError = false;
const size = (bytes) => bytes < 1024 ? `${bytes} B`
  : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KiB`
  : `${(bytes / 1048576).toFixed(1)} MiB`;

function notice(message, error = false) {
  const el = document.getElementById("notice");
  el.textContent = message;
  el.hidden = !message;
  el.classList.toggle("error", error);
}

async function perform(id, action, button) {
  button.disabled = true;
  try {
    render(await bridge.apiPost("modules/action", { module: id, action }));
    notice(action === "disable" ? "模块已关闭，资源已保留。" : "任务已提交，可以在此查看进度。关闭页面后安装会继续。");
  } catch (error) {
    notice(error.message, true);
    button.disabled = false;
  }
}

async function loadSettings(details, id) {
  const form = details.querySelector("form");
  if (!details.open || form.dataset.loaded) return;
  form.dataset.loaded = "loading";
  try {
    const fields = await bridge.apiPost("modules/settings", { module: id });
    form.replaceChildren();
    for (const [key, field] of Object.entries(fields)) {
      const label = document.createElement("label");
      label.className = "field";
      const title = document.createElement("span");
      title.className = "field-title";
      title.textContent = field.description;
      const input = document.createElement(key === "fallback_policy" ? "select" : "input");
      input.name = key;
      input.dataset.kind = field.type;
      if (key === "fallback_policy") {
        for (const [value, name] of [["adaptive", "按需补强"], ["all", "运行全部已启用模型"]]) {
          input.add(new Option(name, value));
        }
        input.value = field.value;
      } else if (field.type === "bool") {
        input.type = "checkbox";
        input.checked = field.value;
      } else {
        input.type = ["int", "float"].includes(field.type) ? "number" : "text";
        input.step = field.type === "float" ? "any" : "1";
        input.value = field.value;
        input.required = input.type === "number";
      }
      const hint = document.createElement("small");
      hint.textContent = field.hint || "";
      label.append(title, input, hint);
      form.append(label);
    }
    const save = document.createElement("button");
    save.type = "submit";
    save.textContent = "保存设置";
    save.className = "primary";
    save.disabled = busy;
    form.append(save);
    form.onsubmit = async (event) => {
      event.preventDefault();
      const values = {};
      for (const input of form.querySelectorAll("input, select")) {
        values[input.name] = input.dataset.kind === "bool" ? input.checked
          : ["int", "float"].includes(input.dataset.kind) ? Number(input.value) : input.value;
      }
      save.disabled = true;
      try {
        await bridge.apiPost("modules/settings", { module: id, values });
        notice("设置已保存并生效。");
        render(await bridge.apiGet("modules"));
      } catch (error) {
        notice(error.message, true);
      } finally {
        save.disabled = busy;
      }
    };
    form.dataset.loaded = "true";
  } catch (error) {
    delete form.dataset.loaded;
    notice(error.message, true);
  }
}

function render(state) {
  busy = state.busy;
  const container = document.getElementById("modules");
  state.modules.forEach((item, index) => {
    let card = cards.get(item.id);
    if (!card) {
      card = document.getElementById("module-card").content.firstElementChild.cloneNode(true);
      card.dataset.module = item.id;
      card.querySelector(".module-number").textContent = `0${index + 1}`;
      card.querySelector("h2").textContent = item.name;
      card.querySelector(".description").textContent = item.description;
      card.querySelector(".size").textContent = item.model_bytes
        ? `模型约 ${size(item.model_bytes)} · 共享运行库按需安装`
        : "需要 OpenCV · 安装体积随系统变化";
      const settings = card.querySelector(".module-settings");
      settings.addEventListener("toggle", () => loadSettings(settings, item.id));
      container.append(card);
      cards.set(item.id, card);
    }
    const active = activePhases.has(item.phase);
    const failed = failurePhases.has(item.phase);
    const badge = card.querySelector(".badge");
    badge.textContent = phaseNames[item.phase] || item.phase;
    badge.className = `badge${item.enabled ? " enabled" : ""}${failed ? " failed" : ""}`;
    card.querySelector(".progress-area").hidden = !active;
    card.querySelectorAll(".steps span").forEach((el, i) => {
      el.classList.toggle("current", ["installing", "downloading", "validating"][i] === item.phase);
    });
    const progress = card.querySelector("progress");
    let label = phaseNames[item.phase];
    if (item.phase === "downloading") {
      if (item.total_bytes) {
        progress.value = Math.min(100, item.downloaded_bytes / item.total_bytes * 100);
        label = `${progress.value.toFixed(0)}% · ${size(item.downloaded_bytes)} / ${size(item.total_bytes)}`;
      } else {
        progress.removeAttribute("value");
        label = `已下载 ${size(item.downloaded_bytes)}`;
      }
      if (item.bytes_per_second) label += ` · ${size(item.bytes_per_second)}/s`;
    } else {
      progress.removeAttribute("value");
    }
    card.querySelector(".progress-label").textContent = label;
    const message = card.querySelector(".message");
    message.textContent = item.message;
    message.classList.toggle("error", failed);
    const button = card.querySelector(".actions button");
    button.disabled = active;
    button.textContent = active ? "安装进行中…" : item.enabled ? "关闭模块"
      : failed ? "重试安装并启用" : item.installed ? "启用模块" : "安装并启用";
    button.className = item.enabled ? "secondary" : "primary";
    button.onclick = () => perform(item.id, item.enabled ? "disable" : item.installed && !failed ? "enable" : "install", button);
    card.querySelector(".module-settings").hidden = !item.installed;
    const log = card.querySelector(".logs");
    log.hidden = !item.logs.length;
    const pre = log.querySelector("pre");
    const nearEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 30;
    pre.textContent = item.logs.join("\n");
    if (nearEnd) pre.scrollTop = pre.scrollHeight;
  });
  document.querySelectorAll('.settings button[type="submit"]').forEach((el) => { el.disabled = busy; });
  container.setAttribute("aria-busy", "false");
}

async function poll() {
  if (stopped) return;
  try {
    render(await bridge.apiGet("modules"));
    if (networkError) notice("");
    networkError = false;
  } catch (error) {
    networkError = true;
    notice(`暂时无法刷新状态：${error.message}。正在重试，后台安装不会因此停止。`, true);
  }
  if (!stopped) window.setTimeout(poll, busy ? 1000 : 3000);
}

if (!bridge) {
  notice("请从 AstrBot WebUI 的插件视图打开模块管理。旧版可使用管理员指令“框模块 状态”和“框模块 安装 前景”。", true);
} else {
  try {
    const context = await bridge.ready();
    const theme = (context) => { document.documentElement.dataset.dark = String(Boolean(context.isDark)); };
    theme(context);
    const off = bridge.onContext(theme);
    window.addEventListener("beforeunload", () => { stopped = true; off(); });
    const common = document.getElementById("common");
    common.addEventListener("toggle", () => loadSettings(common, "common"));
    await poll();
  } catch (error) {
    notice(`模块管理连接失败：${error.message}`, true);
  }
}
