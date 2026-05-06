const fileSelect = document.getElementById("fileSelect");
const audio = document.getElementById("audio");
const waveformEl = document.getElementById("waveform");
const clipRange = document.getElementById("clipRange");
const playClipButton = document.getElementById("playClip");
const refreshFiles = document.getElementById("refreshFiles");
const startReconstruction = document.getElementById("startReconstruction");
const clipLogEl = document.getElementById("clipLog");
const runLogEl = document.getElementById("runLog");
const stepLogsEl = document.getElementById("stepLogs");
const artifactsEl = document.getElementById("artifacts");
const statusEl = document.getElementById("serviceStatus");
const scoreboardEl = document.getElementById("scoreboard");
const refreshRuns = document.getElementById("refreshRuns");
const runHistory = document.getElementById("runHistory");
const comparisonPanel = document.getElementById("comparisonPanel");
const sourceCompareWaveform = document.getElementById("sourceCompareWaveform");
const finalCompareWaveform = document.getElementById("finalCompareWaveform");
const sourceCompareAudio = document.getElementById("sourceCompareAudio");
const finalCompareAudio = document.getElementById("finalCompareAudio");

let currentFile = null;
let currentClip = null;
let wavesurfer = null;
let regionsPlugin = null;
let activeRegion = null;
let sourceCompareWave = null;
let finalCompareWave = null;
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

function clearComparison() {
  if (sourceCompareWave) {
    sourceCompareWave.destroy();
    sourceCompareWave = null;
  }
  if (finalCompareWave) {
    finalCompareWave.destroy();
    finalCompareWave = null;
  }
  sourceCompareWaveform.innerHTML = "";
  finalCompareWaveform.innerHTML = "";
  sourceCompareAudio.removeAttribute("src");
  finalCompareAudio.removeAttribute("src");
  comparisonPanel.hidden = true;
}

function createComparisonWave(container, url, progressColor) {
  return WaveSurfer.create({
    container,
    url,
    height: 108,
    waveColor: "#cfdaf2",
    progressColor,
    cursorColor: "#262626",
    cursorWidth: 1,
    barWidth: 2,
    barGap: 1,
    barRadius: 1,
    normalize: true,
  });
}

function renderComparison(sourceUrl, finalUrl) {
  if (!sourceUrl || !finalUrl) {
    clearComparison();
    return;
  }
  clearComparison();
  comparisonPanel.hidden = false;
  sourceCompareAudio.src = sourceUrl;
  finalCompareAudio.src = finalUrl;
  sourceCompareWave = createComparisonWave(sourceCompareWaveform, sourceUrl, "#323a85");
  finalCompareWave = createComparisonWave(finalCompareWaveform, finalUrl, "#657cc2");
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
  const data = await api("/api/reconstruction-runs");
  renderRunHistory(data.runs);
}

async function selectFile(name) {
  currentFile = name;
  currentClip = null;
  startReconstruction.disabled = false;
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  stepLogsEl.innerHTML = "";
  clipLogEl.textContent = "";
  runLogEl.textContent = "";
  clearComparison();
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

function agentPanel(id, label) {
  let panel = stepLogsEl.querySelector(`[data-step="${id}"]`);
  if (panel) return panel;
  panel = document.createElement("section");
  panel.className = "candidate-panel running";
  panel.dataset.step = String(id);
  panel.innerHTML = `
    <div class="candidate-head">
      <div>
        <span>${label}</span>
        <strong class="candidate-axis">scoring</strong>
      </div>
      <div class="spinner" aria-label="running"></div>
    </div>
    <pre class="candidate-log"></pre>
  `;
  stepLogsEl.appendChild(panel);
  return panel;
}

function fileUrlFromTracePath(path) {
  const match = path.match(/\/ui_runs\/([^/]+)\/(.+)$/);
  if (!match) return null;
  return `/media/runs/${encodeURIComponent(match[1])}/${match[2].split("/").map(encodeURIComponent).join("/")}`;
}

function tracePanelId(agent, step) {
  if (step !== null && step !== undefined) return `${agent}_${step}`;
  const match = agent.match(/(.+)_step_(\d+)/);
  return match ? `${match[1]}_${Number(match[2])}` : agent;
}

async function addTraceFile(line) {
  const agent = line.match(/\bagent=([^ ]+)/)?.[1] || "trace";
  const role = line.match(/\brole=([^ ]+)/)?.[1] || "file";
  const stepMatch = line.match(/\bstep=(\d+)/);
  const step = stepMatch ? Number(stepMatch[1]) : null;
  const pathMatch = line.match(/\bpath=(.+)$/);
  if (!pathMatch) {
    appendRunLog(line);
    return;
  }
  const path = pathMatch[1];
  const url = fileUrlFromTracePath(path);
  const fileName = path.split("/").pop();
  const panel = agentPanel(tracePanelId(agent, step), agent.replaceAll("_", " "));
  panel.querySelector(".candidate-axis").textContent = role.replaceAll("_", " ");
  const log = panel.querySelector(".candidate-log");
  appendToLog(log, line);

  const trace = document.createElement("details");
  trace.className = "trace-file";
  trace.open = role.includes("recommendation") || role.includes("audio_diff") || role.includes("answer");
  trace.innerHTML = `<summary><span>${role.replaceAll("_", " ")}</span><a href="${url || "#"}" target="_blank">${fileName}</a></summary>`;
  const body = document.createElement("pre");
  body.textContent = "loading...";
  trace.appendChild(body);
  panel.appendChild(trace);

  if (!url) {
    body.textContent = path;
    return;
  }
  if (fileName.endsWith(".wav")) {
    body.remove();
    const player = document.createElement("audio");
    player.controls = true;
    player.src = url;
    trace.appendChild(player);
    return;
  }
  try {
    const text = await fetch(url).then((res) => {
      if (!res.ok) throw new Error(`Could not load ${url}`);
      return res.text();
    });
    body.textContent = text.length > 24000 ? `${text.slice(0, 24000)}\n... [truncated in UI; open artifact for full file]` : text;
  } catch (error) {
    body.textContent = error.stack || String(error);
  }
}

async function routeRunLog(line) {
  if (line.startsWith("trace_file")) {
    await addTraceFile(line);
    return;
  }
  const step = parseStepIndex(line);
  const agentMatch = line.match(/\bagent_stage ([a-z_]+)(?: step=(\d+))?/);
  if (agentMatch) {
    const id = agentMatch[2] ? `${agentMatch[1]}_${agentMatch[2]}` : agentMatch[1];
    const panel = agentPanel(id, agentMatch[1].replaceAll("_", " "));
    appendToLog(panel.querySelector(".candidate-log"), line);
    panel.querySelector(".candidate-axis").textContent = "running";
    return;
  }
  if (step === null) {
    appendRunLog(line);
    return;
  }
  const panel = agentPanel(`builder_${step}`, `builder ${step + 1}`);
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

async function renderTraceArtifacts(artifacts) {
  const traceArtifacts = artifacts.filter((artifact) => {
    const name = artifact.name;
    return (
      name.startsWith("codex_") ||
      name.startsWith("audio_diff_") ||
      name.match(/^reconstruction_step_\d+_.+\.wav$/) ||
      name === "mixer_reconstruction.wav" ||
      name === "simplifier_reconstruction.wav" ||
      name.startsWith("recommendation_step_") ||
      name === "recommendation_initial.json" ||
      name === "source_profile.json" ||
      name === "layer_analysis.json" ||
      name.match(/^session_step_\d+_(codex_proposal|accepted)\.json$/)
    );
  });
  for (const artifact of traceArtifacts) {
    const pseudoPath = `/ui_runs/${artifact.url.split("/")[3]}/${artifact.name}`;
    const role = artifact.name.startsWith("codex_")
      ? artifact.name.includes("_prompt") ? "prompt" : "answer"
      : artifact.name.startsWith("audio_diff") ? "audio_diff"
        : artifact.name.endsWith(".wav") ? "render"
        : artifact.name.startsWith("session") ? "session"
          : artifact.name.startsWith("recommendation") ? "recommendation"
            : "file";
    const agent = artifact.name
      .replace(/^codex_/, "")
      .replace(/_(prompt|answer)\.txt$/, "")
      .replace(/\.json$/, "");
    await addTraceFile(`trace_file agent=${agent} role=${role} path=${pseudoPath}`);
  }
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
    empty.textContent = "No completed V1 reconstruction runs found yet.";
    runHistory.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = "run-card";
    button.type = "button";
    const finalScore = Number.isFinite(run.final_score) ? run.final_score.toFixed(3) : "n/a";
    const melScore = Number.isFinite(run.mel_score) ? run.mel_score.toFixed(3) : "n/a";
    button.innerHTML = `
      <span class="run-time">${formatRunDate(run.id)}</span>
      <strong>${finalScore}</strong>
      <span>${run.overall_mix || "Reconstruction run"}</span>
      <em>${run.stage_count || 0} stages · mel ${melScore}</em>
    `;
    button.addEventListener("click", () => loadPastRun(run));
    runHistory.appendChild(button);
  });
}

async function loadPastRun(run) {
  clipLogEl.textContent = "";
  runLogEl.textContent = "";
  stepLogsEl.innerHTML = "";
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  clearComparison();
  setStatus("Loaded run");
  appendRunLog(`loaded v1 run: ${run.id}`);
  appendRunLog(`status: ${run.status}`);
  const reportArtifact = run.artifacts.find((artifact) => artifact.name === "reconstruction_report.json");
  if (!reportArtifact) {
    appendRunLog("reconstruction_report.json is missing for this run");
    await renderArtifacts(run.artifacts);
    return;
  }
  const report = await fetch(reportArtifact.url).then((res) => {
    if (!res.ok) throw new Error(`Could not load ${reportArtifact.url}`);
    return res.json();
  });
  renderScoreboard(report);
  renderHistoryTimeline(report.history || []);
  await renderTraceArtifacts(run.artifacts);
  const scores = report.best_scores || {};
  appendRunLog(`final score: ${Number(scores.final || 0).toFixed(3)}`);
  appendRunLog(`final audio: ${report.final_path || "not recorded"}`);
  await renderArtifacts(run.artifacts);
}

function renderHistoryTimeline(history) {
  stepLogsEl.innerHTML = "";
  history.forEach((item, index) => {
    const id = item.step === undefined || item.step === null ? `${item.stage}_${index}` : `${item.stage}_${item.step}`;
    const panel = agentPanel(id, item.stage || `stage ${index + 1}`);
    panel.classList.remove("running");
    panel.classList.add(item.accepted === false ? "failed" : "completed");
    const log = panel.querySelector(".candidate-log");
    const score = item.scores?.final;
    panel.querySelector(".candidate-axis").textContent = Number.isFinite(score) ? `score ${score.toFixed(3)}` : "loaded";
    appendToLog(log, `stage: ${item.stage || "unknown"}`);
    if (item.step !== undefined && item.step !== null) appendToLog(log, `step: ${item.step}`);
    if (item.winner) appendToLog(log, `winner: ${item.winner}`);
    if (item.accepted !== undefined) appendToLog(log, `accepted: ${item.accepted}`);
    if (item.scores) appendToLog(log, `scores: ${JSON.stringify(item.scores, null, 2)}`);
    if (item.recommendation_path) appendToLog(log, `recommendation: ${item.recommendation_path}`);
  });
}

async function extractSelectedClip() {
  if (!currentFile || !activeRegion) return;
  setStatus("Extracting");
  startReconstruction.disabled = true;
  clipLogEl.textContent = "";
  runLogEl.textContent = "";
  stepLogsEl.innerHTML = "";
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  clearComparison();
  const { start } = selectedRange();
  appendClipLog(`extracting exact clip: ${currentFile} @ ${start.toFixed(2)}s-${(start + CLIP_SECONDS).toFixed(2)}s`);
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
        setStatus("Clip ready");
        startAutonomousRun();
      }
    };
    events.onerror = () => appendClipLog("event stream error; check server process");
  } catch (error) {
    setStatus("Extract failed");
    startReconstruction.disabled = false;
    appendClipLog(error.stack || String(error));
  }
}

async function startAutonomousRun() {
  if (!currentClip) return;
  runLogEl.textContent = "";
  stepLogsEl.innerHTML = "";
  artifactsEl.innerHTML = "";
  scoreboardEl.innerHTML = "";
  clearComparison();
  setStatus("Running");
  startReconstruction.disabled = true;
  appendRunLog(`starting v1 reconstruction for ${currentClip}`);
  appendRunLog("defaults: 5 builder passes, 4 local trials per pass, max 5 layers, mixer pass, simplifier pass");
  try {
    const data = await api("/api/reconstruct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip: currentClip,
        steps: 5,
        local_trials: 4,
        max_layers: 5,
      }),
    });
    appendRunLog(`run id: ${data.run_id}`);
    const events = new EventSource(`/api/reconstructions/${data.run_id}/events`);
    events.onmessage = async (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") await routeRunLog(payload.line);
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
        loadRuns().catch((error) => appendRunLog(error.stack || String(error)));
      }
    };
    events.onerror = () => appendRunLog("event stream error; check server process");
  } catch (error) {
    setStatus("Run failed");
    startReconstruction.disabled = false;
    appendRunLog(error.stack || String(error));
  }
}

startReconstruction.addEventListener("click", extractSelectedClip);

function renderScoreboard(report) {
  const scores = report.best_scores || {};
  scoreboardEl.innerHTML = "";
  [
    "final",
    "multi_resolution_spectral",
    "mel_spectrogram",
    "a_weighted_spectral",
    "envelope",
    "segment_envelope",
    "late_energy_ratio",
    "sustain_coverage",
    "frontload_balance",
    "band_envelope_by_time",
    "pitch_chroma",
    "f0_contour",
    "spectral_motion",
    "centroid_trajectory",
    "spectral_features",
    "transient_onset",
    "onset_count",
    "onset_timing",
    "stereo_width",
    "modulation",
    "harmonic_noise",
    "cepstral",
    "embedding",
    "codec_latent",
  ].forEach((name) => {
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
  const sourceArtifact = artifacts.find((artifact) => artifact.name === "source_clip.wav");
  const finalArtifact = artifacts.find((artifact) => artifact.name === "final_reconstruction.wav");
  renderComparison(sourceArtifact?.url, finalArtifact?.url);
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
refreshRuns.addEventListener("click", () => {
  loadRuns().catch((error) => appendRunLog(error.stack || String(error)));
});

Promise.all([loadFiles(), loadRuns()]).catch((error) => {
  setStatus("Error");
  runLogEl.textContent = error.stack || String(error);
});
