/**
 * app.js — Voice AI · Demo Mode
 *
 * Pipeline (no PyTorch / Whisper / Piper needed):
 *   🎤  Browser SpeechRecognition  (live interim + final STT)
 *   →   WebSocket /ws/{id}         (transcript_direct message)
 *   →   FastAPI + OpenAI           (streaming LLM)
 *   ←   response_token / llm_response  (streamed text)
 *   🔊  Browser speechSynthesis    (TTS)
 *
 * Features:
 *   • Live interim transcript in the strip while you speak
 *   • Token-by-token AI response with blinking cursor
 *   • Thinking animation while LLM processes
 *   • Copy + Replay buttons on every AI bubble
 *   • Session timer, message & word counters
 *   • Download transcript as .txt
 *   • Clear chat
 *   • Keyboard shortcut: Space = toggle mic
 *   • Waveform animation during listening
 *   • Animated debug console (toggle 🐛)
 *   • Toast notifications
 */

// ── Config ────────────────────────────────────────────────────────────
const WS_HOST = window.location.hostname || "localhost";
const WS_PORT = window.location.port    || "8000";
const WS_URL  = `ws://${WS_HOST}:${WS_PORT}/ws/`;

// ── State ─────────────────────────────────────────────────────────────
let ws           = null;
let recognition  = null;
let sessionId    = null;
let appState     = "disconnected";
let currentUtter = null;
let aiTurnEl     = null;   // current .bubble element being streamed into
let aiTurnText   = "";
let aiMsgEl      = null;   // full .msg element (for adding action buttons)
let thinkingEl   = null;   // "..." thinking bubble element

// Stats
let sessionStart  = null;
let timerInterval = null;
let msgCount      = 0;
let wordCount     = 0;

// ── DOM shorthand ─────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const dot        = $("status-dot");
const statusText = $("status-text");
const btnStart   = $("btn-start");
const btnStop    = $("btn-stop");
const btnMic     = $("btn-mic");
const chatEl     = $("chat");
const liveStrip  = $("live-strip");
const liveText   = $("live-text");
const debugLog   = $("debug-log");
const toastEl    = $("toast");
const waveformEl = $("waveform");
const waveBars   = Array.from(waveformEl.children);

// ── State machine ─────────────────────────────────────────────────────
function setState(s) {
  appState = s;
  const C = {
    disconnected: { txt: "Disconnected",  cls: "",           mic: false },
    connecting:   { txt: "Connecting…",   cls: "connecting", mic: false },
    idle:         { txt: "Ready",         cls: "idle",       mic: true  },
    listening:    { txt: "Listening…",    cls: "listening",  mic: true  },
    processing:   { txt: "Thinking…",     cls: "processing", mic: false },
    speaking:     { txt: "AI Speaking",   cls: "speaking",   mic: true  },
    error:        { txt: "Error",         cls: "error",      mic: false },
  };
  const cfg = C[s] ?? C.disconnected;
  dot.className         = cfg.cls;
  statusText.textContent = cfg.txt;

  btnMic.disabled = !cfg.mic || !ws;
  btnMic.className = "btn" + (s === "listening" ? " listening" : s === "processing" ? " processing" : "");
  btnMic.textContent = s === "listening" ? "⏹" : "🎤";
  btnMic.title = s === "listening" ? "Stop (Space)"
               : s === "speaking"  ? "Interrupt AI (Space)"
               : "Speak (Space)";

  btnStart.disabled = (ws !== null);
  btnStop.disabled  = (ws === null);

  // Live strip
  if (s === "listening") {
    liveStrip.classList.add("active");
    liveText.textContent = "Listening…";
    animateWave(true);
  } else if (s === "processing") {
    liveStrip.classList.remove("active");
    liveText.textContent = "Processing your message…";
    animateWave(false);
  } else if (s === "speaking") {
    liveStrip.classList.remove("active");
    liveText.textContent = "AI is speaking…";
    animateWave(false);
  } else {
    liveStrip.classList.remove("active");
    liveText.textContent = ws ? "Press the mic button or Space to speak…"
                              : "Connect to get started…";
    animateWave(false);
  }
}

// ── Log ───────────────────────────────────────────────────────────────
function log(msg) {
  const el = document.createElement("div");
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  debugLog.appendChild(el);
  debugLog.scrollTop = debugLog.scrollHeight;
}

// ── Toast ─────────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg, dur = 2600) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toastEl.classList.remove("show"), dur);
}

// ── Stats ─────────────────────────────────────────────────────────────
function startTimer() {
  sessionStart = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const s   = Math.floor((Date.now() - sessionStart) / 1000);
    const mm  = String(Math.floor(s / 60)).padStart(2, "0");
    const ss  = String(s % 60).padStart(2, "0");
    $("stat-time").textContent = `${mm}:${ss}`;
  }, 1000);
}

function stopTimer() { clearInterval(timerInterval); timerInterval = null; }

function incMsgs()      { msgCount++;  $("stat-msgs").textContent = msgCount; }
function addWords(text) {
  wordCount += text.trim().split(/\s+/).filter(Boolean).length;
  $("stat-words").textContent = wordCount;
}

// ── Escape HTML ───────────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Timestamp helper ──────────────────────────────────────────────────
function ts() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Hide empty state ──────────────────────────────────────────────────
function hideEmpty() {
  const el = document.getElementById("empty-state");
  if (el) el.remove();
}

// ── Chat bubble factories ─────────────────────────────────────────────
function addUserBubble(text) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `
    <div class="msg-meta">
      <span class="msg-role">You</span>
      <span class="msg-ts">${ts()}</span>
    </div>
    <div class="bubble">${esc(text)}</div>`;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  incMsgs(); addWords(text);
}

function addThinking() {
  hideEmpty();
  removeThinking();
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `
    <div class="msg-meta"><span class="msg-role">AI</span></div>
    <div class="thinking-dots"><span></span><span></span><span></span></div>`;
  chatEl.appendChild(el);
  thinkingEl = el;
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeThinking() {
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
}

function startAIBubble() {
  hideEmpty();
  removeThinking();
  const el = document.createElement("div");
  el.className = "msg ai";
  el.innerHTML = `
    <div class="msg-meta">
      <span class="msg-role">AI</span>
      <span class="msg-ts">${ts()}</span>
    </div>
    <div class="bubble streaming"></div>
    <div class="bubble-actions"></div>`;
  chatEl.appendChild(el);
  aiTurnEl   = el.querySelector(".bubble");
  aiMsgEl    = el;
  aiTurnText = "";
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function appendToken(token) {
  if (!aiTurnEl) startAIBubble();
  aiTurnText += token;
  aiTurnEl.textContent = aiTurnText;
  chatEl.scrollTop = chatEl.scrollHeight;
}

function finaliseAI(fullText) {
  // Finish or create the bubble
  if (aiTurnEl) {
    aiTurnEl.classList.remove("streaming");
    aiTurnEl.textContent = fullText;
  } else if (fullText) {
    aiMsgEl = startAIBubble();
    aiTurnEl = aiMsgEl.querySelector(".bubble");
    aiTurnEl.classList.remove("streaming");
    aiTurnEl.textContent = fullText;
  }

  // Add Copy + Replay action buttons
  const actions = aiMsgEl?.querySelector(".bubble-actions");
  if (actions && fullText) {
    const cp = document.createElement("button");
    cp.className = "bubble-btn"; cp.textContent = "📋 Copy";
    cp.addEventListener("click", () =>
      navigator.clipboard.writeText(fullText).then(() => toast("Copied to clipboard")));

    const rp = document.createElement("button");
    rp.className = "bubble-btn"; rp.textContent = "🔊 Replay";
    rp.addEventListener("click", () => speakText(fullText));

    actions.appendChild(cp);
    actions.appendChild(rp);
  }

  aiTurnEl = null; aiMsgEl = null; aiTurnText = "";
  incMsgs(); addWords(fullText);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addSystemMsg(text, isError = false) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg system" + (isError ? " is-error" : "");
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ── WebSocket ─────────────────────────────────────────────────────────
function connectWS(sid) {
  log(`Connecting → ${WS_URL}${sid}`);
  ws = new WebSocket(WS_URL + sid);

  ws.onopen  = () => log("WebSocket open");
  ws.onerror = ()  => log("WebSocket error — check console");

  ws.onclose = () => {
    log("WebSocket closed");
    ws = null;
    stopTimer();
    removeThinking();
    if (aiTurnEl) { aiTurnEl.classList.remove("streaming"); aiTurnEl = null; }
    setState("disconnected");
    addSystemMsg("Session ended");
  };

  ws.onmessage = (evt) => {
    if (typeof evt.data !== "string") return;
    handleMsg(JSON.parse(evt.data));
  };
}

function handleMsg(msg) {
  log(`← ${msg.type}${msg.text ? ` "${msg.text.slice(0, 55)}"` : ""}`);

  switch (msg.type) {

    case "ready":
      setState("idle");
      startTimer();
      addSystemMsg(`Connected · session ${msg.session_id}`);
      break;

    case "transcript":
      // Backend echoed the recognised text
      liveText.textContent = msg.text;
      addUserBubble(msg.text);
      addThinking();
      setState("processing");
      break;

    case "response_token":
      if (appState !== "speaking") {
        startAIBubble();
        setState("speaking");
      }
      appendToken(msg.token);
      break;

    case "llm_response":
      finaliseAI(msg.text);
      speakText(msg.text);
      break;

    case "response_complete":
      // TTS onend handles transition back to idle
      break;

    case "interrupted":
      stopSpeaking();
      setState("idle");
      break;

    case "error":
      removeThinking();
      if (aiTurnEl) { aiTurnEl.classList.remove("streaming"); aiTurnEl = null; }
      addSystemMsg(msg.message, true);
      setState("idle");
      log(`SERVER ERROR: ${msg.message}`);
      break;
  }
}

// ── Speech Recognition (STT) ──────────────────────────────────────────
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

function startListening() {
  if (!SR) {
    toast("SpeechRecognition not supported — use Chrome or Edge", 5000);
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    toast("Not connected — click Connect first");
    return;
  }

  if (recognition) { recognition.abort(); recognition = null; }

  recognition = new SR();
  recognition.lang           = "en-US";
  recognition.interimResults = true;
  recognition.continuous     = false;

  let finalSent = false;

  recognition.onstart = () => {
    setState("listening");
    log("Mic open");
  };

  recognition.onresult = (evt) => {
    let interim = "", final_ = "";
    for (let i = evt.resultIndex; i < evt.results.length; i++) {
      if (evt.results[i].isFinal) final_ += evt.results[i][0].transcript;
      else                         interim += evt.results[i][0].transcript;
    }
    // Show interim words live in the strip
    if (interim) liveText.textContent = interim + "…";
    if (final_ && !finalSent) {
      finalSent = true;
      liveText.textContent = final_.trim();
      sendTranscript(final_.trim());
    }
  };

  recognition.onerror = (e) => {
    log(`Mic error: ${e.error}`);
    if (e.error === "not-allowed")
      toast("Microphone access denied — allow in browser settings", 5000);
    if (e.error !== "aborted") setState("idle");
    animateWave(false);
  };

  recognition.onend = () => {
    recognition = null;
    animateWave(false);
    if (appState === "listening") setState("processing");
  };

  recognition.start();
}

function stopListening() {
  if (recognition) { recognition.stop(); recognition = null; }
  animateWave(false);
}

// ── Send transcript to backend ────────────────────────────────────────
function sendTranscript(text) {
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  log(`→ transcript_direct: "${text}"`);
  ws.send(JSON.stringify({ type: "transcript_direct", text }));
}

// ── TTS (speechSynthesis) ─────────────────────────────────────────────
// Load voices lazily — Chrome populates them asynchronously
let _voices = [];
window.speechSynthesis.onvoiceschanged = () => {
  _voices = window.speechSynthesis.getVoices();
};

function speakText(text) {
  if (!text) return;
  window.speechSynthesis.cancel();

  currentUtter       = new SpeechSynthesisUtterance(text);
  currentUtter.lang  = "en-US";
  currentUtter.rate  = 1.0;
  currentUtter.pitch = 1.0;

  const voices = _voices.length ? _voices : window.speechSynthesis.getVoices();
  const voice  = voices.find(v => v.lang === "en-US" && /natural|neural/i.test(v.name))
              || voices.find(v => v.lang === "en-US")
              || voices[0];
  if (voice) currentUtter.voice = voice;

  currentUtter.onstart = () => { setState("speaking"); log("TTS started"); };
  currentUtter.onend   = () => { setState("idle"); log("TTS done"); currentUtter = null; };
  currentUtter.onerror = (e) => { log(`TTS error: ${e.error}`); setState("idle"); };

  // Small delay ensures voices are populated in Chrome
  setTimeout(() => window.speechSynthesis.speak(currentUtter), 80);
}

function stopSpeaking() {
  window.speechSynthesis.cancel();
  currentUtter = null;
}

// ── Waveform animation ────────────────────────────────────────────────
let _waveFrame = null;
function animateWave(on) {
  cancelAnimationFrame(_waveFrame); _waveFrame = null;
  if (!on) { waveBars.forEach(b => { b.style.height = "3px"; }); return; }
  const tick = () => {
    waveBars.forEach(b => { b.style.height = (3 + Math.random() * 15) + "px"; });
    _waveFrame = requestAnimationFrame(tick);
  };
  tick();
}

// ── Button: Connect ───────────────────────────────────────────────────
btnStart.addEventListener("click", () => {
  sessionId = "sess-" + Math.random().toString(36).slice(2, 10);
  setState("connecting");
  connectWS(sessionId);
});

// ── Button: End session ───────────────────────────────────────────────
btnStop.addEventListener("click", () => {
  stopListening(); stopSpeaking();
  if (ws) { ws.close(); ws = null; }
  stopTimer(); setState("disconnected");
  log("Disconnected by user");
});

// ── Button: Mic ───────────────────────────────────────────────────────
btnMic.addEventListener("click", () => {
  if (appState === "listening") {
    stopListening(); setState("idle");
  } else {
    if (appState === "speaking") {
      stopSpeaking();
      if (ws?.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "interrupt" }));
    }
    startListening();
  }
});

// ── Button: Clear chat ────────────────────────────────────────────────
$("btn-clear").addEventListener("click", () => {
  chatEl.innerHTML = `
    <div id="empty-state">
      <div class="empty-icon">🎙</div>
      <div class="empty-h">Chat cleared</div>
      <div class="empty-s">Press <kbd>Space</kbd> or the mic button to start speaking.</div>
    </div>`;
  msgCount = 0; wordCount = 0; aiTurnEl = null; thinkingEl = null; aiMsgEl = null;
  $("stat-msgs").textContent  = "0";
  $("stat-words").textContent = "0";
  toast("Chat cleared");
});

// ── Button: Download transcript ───────────────────────────────────────
$("btn-download").addEventListener("click", () => {
  const lines = [];
  chatEl.querySelectorAll(".msg").forEach(msg => {
    const role = msg.querySelector(".msg-role")?.textContent ?? "System";
    const time = msg.querySelector(".msg-ts")?.textContent  ?? "";
    const text = msg.querySelector(".bubble")?.textContent  ?? "";
    if (text) lines.push(`[${time}] ${role.padEnd(6)}: ${text}`);
  });
  if (!lines.length) { toast("Nothing to download yet"); return; }
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `voice-ai-${new Date().toISOString().slice(0, 10)}.txt`;
  a.click();
  toast("Transcript downloaded ✓");
});

// ── Button: Debug console ─────────────────────────────────────────────
let debugOpen = false;
$("btn-debug").addEventListener("click", () => {
  debugOpen = !debugOpen;
  $("debug-wrap").classList.toggle("open", debugOpen);
});
$("debug-header").addEventListener("click", () => {
  debugOpen = false; $("debug-wrap").classList.remove("open");
});

// ── Keyboard shortcut: Space = toggle mic ─────────────────────────────
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    btnMic.click();
  }
});
