/* hipfire embedded chat UI — vanilla JS over the OpenAI gateway.
 *
 * Wire contract (same as every other client of this gateway):
 *   POST /v1/chat/completions, {"stream": true, "stream_options":
 *   {"include_usage": true}} → SSE `data:` frames whose delta carries
 *   `content` (visible tokens) and `reasoning_content` (thinking). Tool
 *   calls are released in terminal chunks as `delta.tool_calls` with
 *   finish_reason "tool_calls"; the final choice chunk carries `timings`
 *   and a `hipfire` extension with the *resolved* reasoning mode/effort.
 *   Client abort → server-side cancellation is wired engine-side, so
 *   AbortController.abort() really does stop GPU work.
 *
 * Rendering rule: model output only ever lands in textContent — never
 * innerHTML — so no sanitisation dependency is needed (and the CSP
 * served with this page forbids inline script anyway).
 */
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  model: $("model"), badges: $("model-badges"), status: $("status"),
  settings: $("settings"), log: $("log"), input: $("input"),
  send: $("send"), stop: $("stop"), attach: $("attach"), file: $("file"),
  attachments: $("attachments"), system: $("system"),
  temperature: $("temperature"), top_p: $("top_p"), max_tokens: $("max_tokens"),
  thinking: $("thinking"), thinkingWrap: $("thinking-wrap"),
  effort: $("effort"), effortWrap: $("effort-wrap"),
  tools: $("tools"), toolsWrap: $("tools-wrap"),
  toolChoiceRow: $("tool-choice-row"), toolChoice: $("tool_choice"),
  hint: $("settings-hint"), newChat: $("new-chat"),
  toggleSettings: $("toggle-settings"),
};

const state = {
  messages: [],          // OpenAI message array sent verbatim
  models: [],            // /v1/models entries {id, capabilities}
  health: null,          // last /health payload
  sending: false,
  abort: null,
  attachment: null,      // {dataUrl} — at most one image per request
};

/* ---------------- boot ---------------- */

async function boot() {
  $("toggle-settings").addEventListener("click", () => {
    els.settings.hidden = !els.settings.hidden;
  });
  els.newChat.addEventListener("click", newChat);
  els.send.addEventListener("click", send);
  els.stop.addEventListener("click", () => state.abort?.abort());
  els.attach.addEventListener("click", () => els.file.click());
  els.file.addEventListener("change", onFile);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  els.input.addEventListener("input", autosize);
  els.tools.addEventListener("input", validateTools);
  els.model.addEventListener("change", renderThinkingControls);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollStatus();
  });

  await Promise.all([loadModels(), pollStatus()]);
  setInterval(() => { if (!document.hidden) pollStatus(); }, 2000);
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, window.innerHeight * 0.3) + "px";
}

/* ---------------- discovery ---------------- */

async function loadModels() {
  try {
    const r = await fetch("/v1/models");
    const j = await r.json();
    state.models = (j.data || []);
    els.model.replaceChildren(...state.models.map((m) => {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.id;
      return o;
    }));
    renderModelBadges();
    renderThinkingControls();
  } catch (e) {
    setStatus("model list failed: " + e.message);
  }
}
function renderModelBadges() {
  // v0.4.0 wire: per-model facts live on the resident entry's
  // architecture.input_modalities; route facts on /health.capabilities.
  const resident = state.models.find((m) => m.id === (state.health?.model || ""));
  const canSee = !!resident?.architecture?.input_modalities?.includes("image");
  const route = state.health?.capabilities || {};
  const badges = [
    ["multi-slot", route.multi_slot],
    ["image input", canSee],
  ];
  els.badges.replaceChildren(...badges.map(([label, on]) => {
    const s = document.createElement("span");
    s.className = "badge" + (on ? " on" : "");
    s.textContent = label;
    return s;
  }));
  els.attach.hidden = !canSee;
  if (!canSee) clearAttachment();
  const refused = route.refused_request_fields || [];
  els.toolsWrap.hidden = refused.includes("tools");
  // The multi-slot route refuses tools COMBINED with an image — warn rather
  // than silently drop either.
  const toolsSet = els.tools.value.trim().length > 0;
  if (refused.includes("tools+image") && toolsSet && state.attachment) {
    els.hint.textContent = "this serve route refuses tools combined with an image — pick one";
  }
}


/* Thinking controls are driven by what the *loaded* model advertises on
 * /health (reasoning_contract + reasoning_efforts, probed from the chat
 * template at load). Contracts:
 *   unsupported          → hide everything
 *   muse_glimmer         → always-on reasoning; hide toggle AND effort
 *   gemma_boolean        → on/off select only
 *   qwen_jinja/deepseek4 → on/off select + effort dropdown (advertised rungs
 *                          plus "auto" and "off")
 */
function renderThinkingControls() {
  const h = state.health || {};
  const contract = h.reasoning_contract || "unsupported";
  const efforts = h.reasoning_efforts || [];

  if (contract === "unsupported" || contract === "muse_glimmer") {
    els.thinkingWrap.hidden = true;
    els.effortWrap.hidden = true;
    els.hint.textContent = contract === "muse_glimmer"
      ? "This model always reasons; thinking cannot be disabled."
      : "";
    return;
  }

  els.thinkingWrap.hidden = false;
  if (!els.thinking.options.length) {
    els.thinking.replaceChildren(
      opt("auto", "auto (model default)"), opt("on", "on"), opt("off", "off"));
  }

  // The multi-slot route refuses the effort field outright regardless of
  // what the model could do — hide the picker so nothing is sent.
  const refused = state.health?.capabilities?.refused_request_fields || [];
  const effortRefused = refused.includes("reasoning_effort");
  if (efforts.length && !effortRefused) {
    els.effortWrap.hidden = false;
    els.effort.replaceChildren(
      opt("", "auto"),
      ...efforts.map((e) => opt(e, e)),
      opt("off", "off"));
  } else {
    els.effortWrap.hidden = true;
  }
  els.hint.textContent =
    `reasoning contract: ${contract}` +
    (efforts.length ? ` — efforts ${efforts.join(", ")}` : "");
}

function opt(value, label) {
  const o = document.createElement("option");
  o.value = value; o.textContent = label;
  return o;
}

async function pollStatus() {
  try {
    const [health, stats] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/stats").then((r) => r.json()),
    ]);
    state.health = health;
    const parts = [stats.model || "(no model)"];
    if (stats.queue_depth) parts.push(`queue ${stats.queue_depth}`);
    if (stats.recent_tok_s) parts.push(`${stats.recent_tok_s.toFixed(1)} tok/s`);
    if (health.loading_model) parts.push(`loading ${health.loading_model}…`);
    setStatus(parts.join(" · "));
    renderModelBadges();
    renderThinkingControls();
    // Follow the served model when the user hasn't picked one. The loaded
    // model may be a path (not in /v1/models) — inject it as an option so
    // `model` is always sent on the wire (the gateway 400s without it).
    if (!els.model.value && health.model) {
      if (![...els.model.options].some((o) => o.value === health.model)) {
        els.model.append(opt(health.model, health.model));
      }
      els.model.value = health.model;
    }
  } catch {
    setStatus("serve unreachable");
  }
}

function setStatus(t) { els.status.textContent = t; }

/* ---------------- attachments (vision) ---------------- */

function onFile() {
  const f = els.file.files[0];
  els.file.value = "";
  if (!f) return;
  if (f.type !== "image/png" && f.type !== "image/jpeg") {
    pushError("only PNG and JPEG are accepted (gateway rejects other data URIs)");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.attachment = { dataUrl: reader.result };
    renderAttachment();
  };
  reader.readAsDataURL(f);
}

function renderAttachment() {
  els.attachments.replaceChildren();
  if (!state.attachment) return;
  const wrap = document.createElement("span");
  wrap.className = "thumb";
  const img = document.createElement("img");
  img.src = state.attachment.dataUrl;
  const x = document.createElement("button");
  x.textContent = "×";
  x.addEventListener("click", clearAttachment);
  wrap.append(img, x);
  els.attachments.append(wrap);
}

function clearAttachment() {
  state.attachment = null;
  renderAttachment();
}

/* ---------------- request shaping ---------------- */

function validateTools() {
  const raw = els.tools.value.trim();
  if (!raw) { els.toolChoiceRow.hidden = true; return true; }
  try {
    const t = JSON.parse(raw);
    if (!Array.isArray(t)) throw new Error("expected a JSON array");
    els.tools.style.borderColor = "";
    els.toolChoiceRow.hidden = false;
    return true;
  } catch (e) {
    els.tools.style.borderColor = "var(--error)";
    els.toolChoiceRow.hidden = true;
    return false;
  }
}

/* Translate the thinking controls into wire fields. Off wins over effort:
 * "reasoning_effort": "off" is the effort-native spelling of disable and
 * the resolver normalises it through disabled-wins. */
function applyThinking(body) {
  if (els.thinkingWrap.hidden) return;
  const t = els.thinking.value;
  if (t === "on") body.enable_thinking = true;
  else if (t === "off") body.enable_thinking = false;
  // effort only applies while thinking is on (or auto→on).
  if (!els.effortWrap.hidden && els.effort.value && t !== "off") {
    body.reasoning_effort = els.effort.value; // rung or "off" → resolver-normalised
  }
}

function buildBody() {
  const body = {
    model: els.model.value || undefined,
    stream: true,
    stream_options: { include_usage: true },
    messages: state.messages.slice(),
  };
  const sys = els.system.value.trim();
  if (sys) body.messages.unshift({ role: "system", content: sys });
  const t = parseFloat(els.temperature.value);
  if (Number.isFinite(t)) body.temperature = t;
  const p = parseFloat(els.top_p.value);
  if (Number.isFinite(p)) body.top_p = p;
  const n = parseInt(els.max_tokens.value, 10);
  if (Number.isFinite(n)) body.max_tokens = n;
  applyThinking(body);
  const raw = els.tools.value.trim();
  if (raw && !els.toolsWrap.hidden) {
    body.tools = JSON.parse(raw);
    const tc = els.toolChoice.value;
    if (tc !== "auto") body.tool_choice = tc;
  }
  return body;
}

/* ---------------- chat flow ---------------- */

function newChat() {
  state.messages = [];
  els.log.replaceChildren();
  clearAttachment();
}

function pushMsg(role, text, imageDataUrl) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  const r = document.createElement("div");
  r.className = "role"; r.textContent = role;
  const b = document.createElement("div");
  b.className = "bubble"; b.textContent = text;
  div.append(r, b);
  if (imageDataUrl) {
    const img = document.createElement("img");
    img.src = imageDataUrl;
    b.append(img);
  }
  els.log.append(div);
  div.scrollIntoView({ block: "end" });
  return { div, bubble: b };
}

function pushError(text) {
  pushMsg("error", text);
}

async function send() {
  const text = els.input.value.trim();
  if (!text || state.sending) return;
  if (els.tools.value.trim() && !els.toolsWrap.hidden && !validateTools()) {
    pushError("tools field is not a valid JSON array");
    return;
  }

  // Content shape: plain string, or the OpenAI multipart content array
  // when an image is attached (single image — gateway cap).
  const content = state.attachment
    ? [{ type: "text", text },
       { type: "image_url", image_url: { url: state.attachment.dataUrl } }]
    : text;
  state.messages.push({ role: "user", content });
  pushMsg("user", text, state.attachment?.dataUrl);
  els.input.value = ""; autosize();
  clearAttachment();
  await runTurn();
}

/* One assistant turn over whatever tail state.messages holds — used by
 * send() for user turns and by "reply as tool" for tool-role turns
 * (which have no user text and must not be gated on it). */
async function runTurn() {
  if (state.sending) return;
  const view = pushMsg("assistant", "");
  view.bubble.textContent = "";
  const thinkDetails = document.createElement("details");
  thinkDetails.className = "think";
  const thinkSummary = document.createElement("summary");
  thinkSummary.textContent = "thinking…";
  const thinkBody = document.createElement("div");
  thinkBody.className = "think-body";
  thinkDetails.append(thinkSummary, thinkBody);
  const meta = document.createElement("div");
  meta.className = "meta-line";

  setSending(true);
  state.abort = new AbortController();

  let sawThinking = false;
  const toolCalls = new Map(); // index → {index, id, name, arguments}
  let finishReason = null;

  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildBody()),
      signal: state.abort.signal,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      throw new Error(err?.error?.message || `HTTP ${resp.status}`);
    }
    await readSSE(resp.body, (chunk) => {
      const choice = chunk.choices?.[0];
      const delta = choice?.delta;
      if (delta?.content) {
        view.bubble.textContent += delta.content;
        view.div.scrollIntoView({ block: "end" });
      }
      if (delta?.reasoning_content) {
        if (!sawThinking) {
          sawThinking = true;
          view.div.insertBefore(thinkDetails, view.bubble);
        }
        thinkBody.textContent += delta.reasoning_content;
      }
      for (const tc of delta?.tool_calls || []) {
        const prev = toolCalls.get(tc.index)
          || { index: tc.index, id: "", name: "", arguments: "" };
        if (tc.id) prev.id = tc.id;
        if (tc.function?.name) prev.name += tc.function.name;
        if (tc.function?.arguments) prev.arguments += tc.function.arguments;
        toolCalls.set(tc.index, prev);
      }
      if (choice?.finish_reason) {
        finishReason = choice.finish_reason;
        // Resolved reasoning (hipfire extension) + timings on the terminal
        // chunk — show what actually ran, not what was requested.
        const hip = chunk.hipfire;
        const t = chunk.timings;
        const parts = [];
        if (t?.ttft_ms != null) parts.push(`ttft ${Math.round(t.ttft_ms)}ms`);
        if (t?.decode_tok_s != null) parts.push(`${Number(t.decode_tok_s).toFixed(1)} tok/s`);
        if (t?.tau != null) parts.push(`τ ${Number(t.tau).toFixed(2)}`);
        const r = hip?.reasoning;
        if (r && r.mode === "enabled") {
          parts.push(`thinking ${r.effort || "on"}`);
        }
        for (const w of hip?.config_warnings || []) parts.push(`⚠ ${w}`);
        meta.textContent = parts.join(" · ");
        if (meta.textContent) view.div.append(meta);
        thinkSummary.textContent = "thinking";
      }
      if (chunk.usage) {
        const u = chunk.usage;
        meta.textContent +=
          (meta.textContent ? " · " : "") +
          `${u.prompt_tokens}+${u.completion_tokens} tok`;
      }
    });
    const assistantMsg = { role: "assistant", content: view.bubble.textContent };
    if (finishReason === "tool_calls" && toolCalls.size) {
      // Preserve the wire shape: a following tool-role turn is only
      // well-formed against an assistant message carrying tool_calls.
      assistantMsg.tool_calls = [...toolCalls.values()]
        .sort((a, b) => a.index - b.index)
        .map((tc) => ({
          id: tc.id, type: "function",
          function: { name: tc.name, arguments: tc.arguments },
        }));
    }
    state.messages.push(assistantMsg);
  } catch (e) {
    if (e.name === "AbortError") {
      view.bubble.textContent += "\n[stopped]";
      state.messages.push({ role: "assistant", content: view.bubble.textContent });
    } else {
      pushError(e.message);
      // Do not push a failed assistant turn into history.
    }
  } finally {
    for (const tc of [...toolCalls.values()].sort((a, b) => a.index - b.index)) {
      renderToolCall(view.div, tc);
    }
    setSending(false);
  }
}

function renderToolCall(parent, tc) {
  const div = document.createElement("div");
  div.className = "toolcall";
  const fn = document.createElement("span");
  fn.className = "fn";
  fn.textContent = "tool_call: " + (tc.name || "?");
  const pre = document.createElement("pre");
  try { pre.textContent = JSON.stringify(JSON.parse(tc.arguments), null, 2); }
  catch { pre.textContent = tc.arguments; }
  div.append(fn, pre);
  // Manual round-trip: paste the tool result back and run the next turn.
  const btn = document.createElement("button");
  btn.textContent = "reply as tool";
  btn.addEventListener("click", () => {
    const result = prompt(`Result for ${tc.name}:`);
    if (result == null) return;
    state.messages.push({
      role: "tool", tool_call_id: tc.id || undefined,
      name: tc.name || undefined, content: result,
    });
    pushMsg("tool", result);
    runTurn();
  });
  div.append(btn);
  parent.append(div);
}

function setSending(on) {
  state.sending = on;
  els.send.disabled = on;
  els.stop.hidden = !on;
}

/* Minimal SSE reader (EventSource cannot POST). Splits on the event
 * boundary and parses each `data:` payload; [DONE] terminates. */
async function readSSE(body, onChunk) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        try { onChunk(JSON.parse(payload)); } catch { /* keepalive / junk */ }
      }
    }
  }
}

boot();
