const fileSelect = document.getElementById("fileSelect");
const audio = document.getElementById("audio");
const waveformEl = document.getElementById("waveform");
const clipRange = document.getElementById("clipRange");
const playClipButton = document.getElementById("playClip");
const refreshFiles = document.getElementById("refreshFiles");
const extractClip = document.getElementById("extractClip");
const startReconstruction = document.getElementById("startReconstruction");
const targetPart = document.getElementById("targetPart");
const clipLogEl = document.getElementById("clipLog");
const runLogEl = document.getElementById("runLog");
const stepLogsEl = document.getElementById("stepLogs");
const artifactsEl = document.getElementById("artifacts");
const statusEl = document.getElementById("serviceStatus");
const scoreboardEl = document.getElementById("scoreboard");
const stepsEl = document.getElementById("steps");
const localTrialsEl = document.getElementById("localTrials");
const maxLayersEl = document.getElementById("maxLayers");

let currentFile = null;
let currentClip = null;
let wavesurfer = null;
let regionsPlugin = null;
let activeRegion = null;
const CLIP_SECONDS = 5;

function setStatus(text) {
  statusEl.textContent = text;
}

function appendToLog(element, line) {
  element.textContent += `${line}\n`;
  element.scrollTop = element.scrollHeight;
}

function appendClipLog(line) {
  appendToLog(clipLogEl, line);
}

function appendRunLog(line) {
  appendToLog(runLogEl, line);
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

async function selectFile(name) {
  currentFile = name;
  currentClip = null;
  startReconstruction.disabled = true;
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  stepLogsEl.innerHTML = "";
  clipLogEl.textContent = "";
  runLogEl.textContent = "";
  const url = `/media/references/${encodeURIComponent(name)}`;
  audio.src = url;
  setStatus("Loading");

  if (wavesurfer) wavesurfer.destroy();
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

function parseStepIndex(line) {
  const direct = line.match(/\bstep=(\d+)/);
  if (direct) return Number(direct[1]);
  const done = line.match(/\bstep_complete index=(\d+)/);
  return done ? Number(done[1]) : null;
}

function stepPanel(index) {
  let panel = stepLogsEl.querySelector(`[data-step="${index}"]`);
  if (panel) return panel;
  panel = document.createElement("section");
  panel.className = "candidate-panel running";
  panel.dataset.step = String(index);
  panel.innerHTML = `
    <div class="candidate-head">
      <div>
        <span>Step ${index}</span>
        <strong class="candidate-axis">scoring</strong>
      </div>
      <div class="spinner" aria-label="running"></div>
    </div>
    <pre class="candidate-log"></pre>
  `;
  stepLogsEl.appendChild(panel);
  return panel;
}

function routeRunLog(line) {
  const step = parseStepIndex(line);
  if (step === null) {
    appendRunLog(line);
    return;
  }
  const panel = stepPanel(step);
  appendToLog(panel.querySelector(".candidate-log"), line);
  const scoreMatch = line.match(/\bscore=([0-9.]+)/);
  if (scoreMatch) {
    panel.querySelector(".candidate-axis").textContent = `score ${Number(scoreMatch[1]).toFixed(3)}`;
  }
  if (line.startsWith("step_complete")) {
    panel.classList.remove("running");
    panel.classList.add("completed");
  }
}

extractClip.addEventListener("click", async () => {
  if (!currentFile || !activeRegion) return;
  setStatus("Extracting");
  startReconstruction.disabled = true;
  clipLogEl.textContent = "";
  runLogEl.textContent = "";
  stepLogsEl.innerHTML = "";
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  const { start } = selectedRange();
  const focus = targetPart.value.trim();
  appendClipLog(`extracting exact clip: ${currentFile} @ ${start.toFixed(2)}s-${(start + CLIP_SECONDS).toFixed(2)}s`);
  appendClipLog(`focus: ${focus || "full prominent synth construction"}`);
  try {
    const data = await api("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference: currentFile, start, duration: CLIP_SECONDS }),
    });
    appendClipLog(`clip job id: ${data.clip_id}`);
    const events = new EventSource(`/api/clips/${data.clip_id}/events`);
    events.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") appendClipLog(payload.line);
      if (payload.type === "heartbeat") appendClipLog("heartbeat: extraction still running");
      if (payload.type === "done") {
        events.close();
        if (payload.status !== "completed") {
          setStatus("Extract failed");
          appendClipLog(payload.error || "extract failed");
          return;
        }
        currentClip = payload.result.clip;
        appendClipLog(`source clip ready: ${currentClip}`);
        startReconstruction.disabled = false;
        setStatus("Clip ready");
      }
    };
    events.onerror = () => appendClipLog("event stream error; check server process");
  } catch (error) {
    setStatus("Extract failed");
    appendClipLog(error.stack || String(error));
  }
});

startReconstruction.addEventListener("click", async () => {
  if (!currentClip) return;
  runLogEl.textContent = "";
  stepLogsEl.innerHTML = "";
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  setStatus("Running");
  startReconstruction.disabled = true;
  appendRunLog(`starting v1 reconstruction for ${currentClip}`);
  appendRunLog(`steps=${stepsEl.value} local_trials=${localTrialsEl.value} max_layers=${maxLayersEl.value}`);
  try {
    const data = await api("/api/reconstruct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip: currentClip,
        target_part: targetPart.value.trim(),
        steps: Number(stepsEl.value),
        local_trials: Number(localTrialsEl.value),
        max_layers: Number(maxLayersEl.value),
      }),
    });
    appendRunLog(`run id: ${data.run_id}`);
    const events = new EventSource(`/api/reconstructions/${data.run_id}/events`);
    events.onmessage = async (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") routeRunLog(payload.line);
      if (payload.type === "heartbeat") appendRunLog("heartbeat: reconstruction still running");
      if (payload.type === "done") {
        events.close();
        setStatus(payload.status);
        startReconstruction.disabled = false;
        appendRunLog(`run ${payload.status} with return code ${payload.returncode}`);
        if (payload.status !== "completed") {
          stepLogsEl.querySelectorAll(".candidate-panel.running").forEach((panel) => {
            panel.classList.remove("running");
            panel.classList.add("failed");
          });
        }
        const job = await api(`/api/reconstructions/${data.run_id}`);
        await renderArtifacts(job.artifacts);
      }
    };
    events.onerror = () => appendRunLog("event stream error; check server process");
  } catch (error) {
    setStatus("Run failed");
    startReconstruction.disabled = false;
    appendRunLog(error.stack || String(error));
  }
});

function renderScoreboard(report) {
  const scores = report.best_scores || {};
  scoreboardEl.innerHTML = "";
  ["final", "spectral", "envelope", "chroma", "trajectory", "stereo"].forEach((name) => {
    const value = Number(scores[name] || 0);
    const item = document.createElement("div");
    item.className = "score-card";
    item.innerHTML = `
      <span>${name}</span>
      <strong>${value.toFixed(3)}</strong>
      <div class="score-bar"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div>
    `;
    scoreboardEl.appendChild(item);
  });
}

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
    if (artifact.name === "reconstruction_report.json") {
      const report = await fetch(artifact.url).then((res) => res.json());
      renderScoreboard(report);
    }
  }
}

playClipButton.addEventListener("click", playRegion);
fileSelect.addEventListener("change", () => selectFile(fileSelect.value));
refreshFiles.addEventListener("click", loadFiles);

loadFiles().catch((error) => {
  setStatus("Error");
  runLogEl.textContent = error.stack || String(error);
});
