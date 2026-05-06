const fileSelect = document.getElementById("fileSelect");
const audio = document.getElementById("audio");
const waveformEl = document.getElementById("waveform");
const clipRange = document.getElementById("clipRange");
const analyzeClip = document.getElementById("analyzeClip");
const playClipButton = document.getElementById("playClip");
const refreshFiles = document.getElementById("refreshFiles");
const instrument = document.getElementById("instrument");
const targetPart = document.getElementById("targetPart");
const promptBox = document.getElementById("prompt");
const axesEl = document.getElementById("axes");
const startRun = document.getElementById("startRun");
const logEl = document.getElementById("log");
const artifactsEl = document.getElementById("artifacts");
const statusEl = document.getElementById("serviceStatus");
const keyboardPanel = document.getElementById("keyboardPanel");
const keyboardEl = document.getElementById("keyboard");
const refreshRuns = document.getElementById("refreshRuns");
const runHistory = document.getElementById("runHistory");

let currentFile = null;
let currentClip = null;
let currentInstrument = null;
let wavesurfer = null;
let regionsPlugin = null;
let activeRegion = null;
let activePatch = null;
let audioContext = null;
const activeVoices = new Map();
const CLIP_SECONDS = 5;
const KEYBOARD = [
  ["KeyA", "A", 60, "C", false],
  ["KeyW", "W", 61, "C#", true],
  ["KeyS", "S", 62, "D", false],
  ["KeyE", "E", 63, "D#", true],
  ["KeyD", "D", 64, "E", false],
  ["KeyF", "F", 65, "F", false],
  ["KeyT", "T", 66, "F#", true],
  ["KeyG", "G", 67, "G", false],
  ["KeyY", "Y", 68, "G#", true],
  ["KeyH", "H", 69, "A", false],
  ["KeyU", "U", 70, "A#", true],
  ["KeyJ", "J", 71, "B", false],
  ["KeyK", "K", 72, "C", false],
];

function setStatus(text) {
  statusEl.textContent = text;
}

function appendLog(line) {
  logEl.textContent += `${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

async function loadFiles() {
  const data = await api("/api/references");
  fileSelect.innerHTML = "";
  data.files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = file.name;
    fileSelect.appendChild(option);
  });
  if (data.files.length) {
    await selectFile(data.files[0].name);
  }
}

async function loadRuns() {
  const data = await api("/api/runs");
  renderRunHistory(data.runs);
}

async function selectFile(name) {
  currentFile = name;
  currentClip = null;
  currentInstrument = null;
  instrument.value = "";
  keyboardPanel.hidden = true;
  activePatch = null;
  startRun.disabled = true;
  axesEl.innerHTML = "";
  promptBox.value = "";
  targetPart.value = "";
  const url = `/media/references/${encodeURIComponent(name)}`;
  audio.src = url;
  setStatus("Loading");

  if (wavesurfer) {
    wavesurfer.destroy();
  }
  regionsPlugin = WaveSurfer.Regions.create();
  wavesurfer = WaveSurfer.create({
    container: waveformEl,
    url,
    height: 220,
    waveColor: "#cfdaf2",
    progressColor: "#323a85",
    cursorColor: "#262626",
    cursorWidth: 1,
    barWidth: 2,
    barGap: 1,
    barRadius: 1,
    normalize: true,
    plugins: [regionsPlugin],
  });

  wavesurfer.on("ready", () => {
    setStatus("Ready");
    createOrResetRegion(0);
  });

  wavesurfer.on("interaction", () => {
    const time = wavesurfer.getCurrentTime();
    createOrResetRegion(Math.max(0, Math.min(wavesurfer.getDuration() - CLIP_SECONDS, time)));
  });

  regionsPlugin.on("region-updated", (region) => {
    activeRegion = region;
    enforceFiveSeconds(region);
    updateClipReadout();
  });

  regionsPlugin.on("region-clicked", (region, event) => {
    event.stopPropagation();
    activeRegion = region;
    playRegion();
  });
}

function createOrResetRegion(start) {
  const duration = wavesurfer.getDuration();
  const safeStart = Math.max(0, Math.min(Math.max(0, duration - CLIP_SECONDS), start));
  regionsPlugin.clearRegions();
  activeRegion = regionsPlugin.addRegion({
    start: safeStart,
    end: safeStart + CLIP_SECONDS,
    color: "rgba(50, 58, 133, 0.18)",
    drag: true,
    resize: false,
  });
  wavesurfer.setTime(safeStart);
  updateClipReadout();
}

function enforceFiveSeconds(region) {
  const duration = wavesurfer.getDuration();
  const start = Math.max(0, Math.min(Math.max(0, duration - CLIP_SECONDS), region.start));
  if (Math.abs(region.start - start) > 0.001 || Math.abs(region.end - (start + CLIP_SECONDS)) > 0.001) {
    region.setOptions({ start, end: start + CLIP_SECONDS });
  }
  wavesurfer.setTime(start);
}

function selectedRange() {
  if (!activeRegion) return { start: 0, end: CLIP_SECONDS };
  return { start: activeRegion.start, end: activeRegion.start + CLIP_SECONDS };
}

function updateClipReadout() {
  const { start, end } = selectedRange();
  clipRange.textContent = `${start.toFixed(2)}s - ${end.toFixed(2)}s`;
}

function playRegion() {
  if (!activeRegion) return;
  if (wavesurfer.isPlaying()) {
    wavesurfer.pause();
    playClipButton.textContent = "Play Selected 5s";
    return;
  }
  playClipButton.textContent = "Pause";
  wavesurfer.play(activeRegion.start, activeRegion.end);
}

function stopRegionPlaybackLabel() {
  playClipButton.textContent = "Play Selected 5s";
}

function renderAxes(axes) {
  axesEl.innerHTML = "";
  Object.entries(axes).forEach(([name, values]) => {
    const div = document.createElement("div");
    div.className = "axis";
    div.innerHTML = `<strong>${name}</strong><span>${values.join(", ")}</span>`;
    axesEl.appendChild(div);
  });
}

function formatRunDate(id) {
  const match = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
  if (!match) return id;
  const [, year, month, day, hour, minute, second] = match;
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function renderRunHistory(runs) {
  runHistory.innerHTML = "";
  if (!runs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = "No completed UI runs found yet.";
    runHistory.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = "run-card";
    button.type = "button";
    const score = Number.isFinite(run.best_score) ? run.best_score.toFixed(3) : "n/a";
    button.innerHTML = `
      <span class="run-time">${formatRunDate(run.id)}</span>
      <strong>${run.instrument_type || "unknown instrument"}</strong>
      <span>${run.prompt || "No prompt recorded"}</span>
      <em>${run.status} · ${run.best_axis || "no axis"} · ${score}</em>
    `;
    button.addEventListener("click", () => loadPastRun(run));
    runHistory.appendChild(button);
  });
}

async function loadPastRun(run) {
  logEl.textContent = "";
  artifactsEl.innerHTML = "";
  keyboardPanel.hidden = true;
  activePatch = null;
  setStatus("Loaded run");
  appendLog(`loaded past run: ${run.id}`);
  appendLog(`status: ${run.status}`);
  const reportArtifact = run.artifacts.find((artifact) => artifact.name === "match_report.json");
  if (!reportArtifact) {
    appendLog("match_report.json is missing for this run");
    renderArtifacts(run.artifacts);
    return;
  }
  const report = await fetch(reportArtifact.url).then((res) => {
    if (!res.ok) throw new Error(`Could not load ${reportArtifact.url}`);
    return res.json();
  });
  const analysis = report.analysis || {};
  const finalRecipe = report.final_recipe || {};
  const instrumentType = finalRecipe.instrument_type || analysis.instrument_type || run.instrument_type || "";
  currentInstrument = instrumentType;
  instrument.value = "";
  promptBox.value = report.prompt || run.prompt || "";
  if (analysis.axes) renderAxes(analysis.axes);
  appendLog(`instrument: ${instrumentType || "not recorded"}`);
  appendLog(`prompt: ${promptBox.value || "not recorded"}`);
  if (Array.isArray(report.best_candidates) && report.best_candidates.length) {
    const best = report.best_candidates[0];
    appendLog(`best candidate: ${best.name || "candidate"} / ${best.axis || "axis"} / ${JSON.stringify(best.scores || {})}`);
  }
  if (report.codex_synthesis?.answer_path) {
    appendLog("codex synthesis answer is available in the artifacts below");
  }
  await renderArtifacts(run.artifacts);
}

analyzeClip.addEventListener("click", async () => {
  if (!currentFile || !activeRegion) return;
  setStatus("Extracting");
  startRun.disabled = true;
  logEl.textContent = "";
  const { start } = selectedRange();
  appendLog(`extracting exact clip: ${currentFile} @ ${start.toFixed(2)}s-${(start + CLIP_SECONDS).toFixed(2)}s`);
  const focus = targetPart.value.trim();
  appendLog(`target part: ${focus || "model should infer the main synth part"}`);
  appendLog("sending extracted WAV to Gemini for focused per-axis audio analysis...");
  try {
    const data = await api("/api/clip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference: currentFile, start, duration: CLIP_SECONDS, target_part: focus }),
    });
    currentClip = data.clip;
    currentInstrument = data.analysis.instrument_type;
    instrument.value = "";
    renderAxes(data.analysis.axes);
    promptBox.value = data.analysis.prompt;
    appendLog(`clip extracted: ${data.clip}`);
    appendLog(`analysis source: ${data.analysis.analysis_source}`);
    appendLog(`detected instrument: ${currentInstrument}`);
    appendLog(`target prompt: ${data.analysis.prompt}`);
    startRun.disabled = false;
    setStatus(`Clip ${start.toFixed(2)}s`);
  } catch (error) {
    setStatus("Analyze failed");
    appendLog(error.stack || String(error));
  }
});

playClipButton.addEventListener("click", playRegion);

startRun.addEventListener("click", async () => {
  if (!currentClip) return;
  logEl.textContent = "";
  artifactsEl.innerHTML = "";
  setStatus("Running");
  startRun.disabled = true;
  const selectedInstrument = instrument.value || currentInstrument;
  appendLog(`starting strict run for ${currentClip}`);
  appendLog(`instrument: ${selectedInstrument}`);
  appendLog("waiting for subprocess log stream...");
  try {
    const data = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip: currentClip,
        prompt: promptBox.value,
        instrument_type: selectedInstrument,
        candidates: 2,
        axis_trials: 1,
      }),
    });
    const runId = data.run_id;
    appendLog(`run id: ${runId}`);
    const events = new EventSource(`/api/jobs/${runId}/events`);
    events.onmessage = async (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") {
        appendLog(payload.line);
      }
      if (payload.type === "heartbeat") {
        appendLog("heartbeat: process still running");
      }
      if (payload.type === "done") {
        events.close();
        setStatus(payload.status);
        startRun.disabled = false;
        appendLog(`run ${payload.status} with return code ${payload.returncode}`);
        const job = await api(`/api/jobs/${runId}`);
        renderArtifacts(job.artifacts);
        loadRuns().catch((error) => appendLog(error.stack || String(error)));
      }
    };
    events.onerror = () => {
      appendLog("event stream error; check server process");
    };
  } catch (error) {
    setStatus("Run failed");
    startRun.disabled = false;
    appendLog(error.stack || String(error));
  }
});

async function renderArtifacts(artifacts) {
  artifactsEl.innerHTML = "";
  for (const artifact of artifacts) {
    const item = document.createElement("div");
    item.className = "artifact";
    const link = document.createElement("a");
    link.href = artifact.url;
    link.target = "_blank";
    link.textContent = artifact.name;
    item.appendChild(link);
    if (artifact.name.endsWith(".wav")) {
      const player = document.createElement("audio");
      player.controls = true;
      player.src = artifact.url;
      item.appendChild(player);
    }
    artifactsEl.appendChild(item);
    if (artifact.name === "playable_patch.json") {
      const patch = await fetch(artifact.url).then((res) => res.json());
      setupKeyboardPatch(patch);
    }
  }
}

function setupKeyboardPatch(patch) {
  activePatch = patch;
  keyboardPanel.hidden = false;
  keyboardEl.innerHTML = "";
  KEYBOARD.forEach(([code, label, midi, note, black]) => {
    const el = document.createElement("div");
    el.className = `key ${black ? "black" : ""}`;
    el.dataset.code = code;
    el.innerHTML = `<strong>${label}</strong><span>${note}</span>`;
    keyboardEl.appendChild(el);
  });
}

function midiToHz(note) {
  return 440 * Math.pow(2, (note - 69) / 12);
}

function ensureAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
  return audioContext;
}

function startVoice(code, midi) {
  if (!activePatch || activeVoices.has(code)) return;
  const ctx = ensureAudioContext();
  const synth = activePatch.synth;
  const effects = activePatch.effects;
  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const square = ctx.createOscillator();
  const sineGain = ctx.createGain();
  const squareGain = ctx.createGain();
  const voiceGain = ctx.createGain();
  const filter = ctx.createBiquadFilter();
  const out = ctx.createGain();

  osc.type = "sine";
  square.type = "square";
  osc.frequency.value = midiToHz(midi);
  square.frequency.value = midiToHz(midi);
  sineGain.gain.value = 1 - synth.sine_square;
  squareGain.gain.value = synth.sine_square;
  filter.type = "lowpass";
  filter.frequency.value = Math.max(600, 4500 + effects.high_gain_db * 350);
  filter.Q.value = 0.7;
  out.gain.value = 0.18;
  voiceGain.gain.setValueAtTime(0.0001, now);
  voiceGain.gain.exponentialRampToValueAtTime(1, now + Math.max(0.005, synth.attack));
  voiceGain.gain.setTargetAtTime(Math.max(0.001, synth.sustain), now + synth.attack, Math.max(0.01, synth.decay));

  osc.connect(sineGain).connect(voiceGain);
  square.connect(squareGain).connect(voiceGain);
  voiceGain.connect(filter).connect(out).connect(ctx.destination);
  osc.start(now);
  square.start(now);
  activeVoices.set(code, { osc, square, voiceGain, out });
  document.querySelector(`.key[data-code="${code}"]`)?.classList.add("active");
}

function stopVoice(code) {
  const voice = activeVoices.get(code);
  if (!voice || !activePatch || !audioContext) return;
  const now = audioContext.currentTime;
  const release = Math.max(0.03, activePatch.synth.release);
  voice.voiceGain.gain.cancelScheduledValues(now);
  voice.voiceGain.gain.setTargetAtTime(0.0001, now, release / 4);
  voice.osc.stop(now + release);
  voice.square.stop(now + release);
  activeVoices.delete(code);
  document.querySelector(`.key[data-code="${code}"]`)?.classList.remove("active");
}

function isTextEditingTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

window.addEventListener("keydown", (event) => {
  if (isTextEditingTarget(event.target)) return;
  if (event.repeat) return;
  const found = KEYBOARD.find(([code]) => code === event.code);
  if (!found) return;
  event.preventDefault();
  startVoice(found[0], found[2]);
});

window.addEventListener("keyup", (event) => {
  if (isTextEditingTarget(event.target)) return;
  const found = KEYBOARD.find(([code]) => code === event.code);
  if (!found) return;
  event.preventDefault();
  stopVoice(found[0]);
});

document.addEventListener("click", (event) => {
  const keyEl = event.target.closest?.(".key");
  if (!keyEl) return;
  const found = KEYBOARD.find(([code]) => code === keyEl.dataset.code);
  if (!found) return;
  startVoice(found[0], found[2]);
  setTimeout(() => stopVoice(found[0]), 350);
});

fileSelect.addEventListener("change", () => selectFile(fileSelect.value));
refreshFiles.addEventListener("click", loadFiles);
refreshRuns.addEventListener("click", () => {
  loadRuns().catch((error) => appendLog(error.stack || String(error)));
});

Promise.all([loadFiles(), loadRuns()]).catch((error) => {
  setStatus("Error");
  logEl.textContent = error.stack || String(error);
});
