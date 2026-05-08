import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Activity, ArrowLeft, MessageSquareText, RefreshCw, SlidersHorizontal } from 'lucide-react';
import { Terminal } from './components/ui/terminal.jsx';
import { Button } from './components/ui/button.jsx';
import { Badge } from './components/ui/badge.jsx';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card.jsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select.jsx';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './components/ui/accordion.jsx';

const h = React.createElement;

const CLIP_SECONDS = 5;
const SCORE_FIELDS = [
  "final",
  "exact_envelope_50ms",
  "exact_band_50ms",
  "beat_grid_mel",
  "beat_grid_band",
  "beat_grid_envelope",
  "modulation_periodicity",
  "modulation_rate",
  "modulation_depth",
  "directional_delta",
  "transient_classification",
  "mel_spectrogram",
  "multi_resolution_spectral",
  "spectral_motion",
  "pitch_chroma",
  "stereo_width",
  "embedding",
];

const ROLE_LABELS = {
  source_profile: "Source measurements",
  prompt: "Instructions",
  answer: "Output",
  parsed_answer: "Parsed plan",
  layer_analysis: "Layer plan",
  recommendation_initial: "Starting brief",
  recommendation: "Next Producer brief",
  session_proposal: "Proposed session",
  accepted_session: "Accepted session",
  candidate_session: "Candidate session",
  audio_diff: "Accuracy report",
  source_clip: "Target audio",
  winner_audio_diff: "Winning accuracy report",
  winner_render: "Winning audio",
  render: "Audio",
};

const AGENT_LABELS = {
  producer: "Producer",
  residual_critic: "Critic",
  harness_improver: "Harness Improver",
  loss: "Calculate Accuracy",
};

const RUN_ID_PATTERN = /^\d{8}_\d{6}_v1_[A-Za-z0-9]+$/;

function api(path, options = {}) {
  return fetch(path, options).then(async (res) => {
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.statusText);
    return data;
  });
}

function fileUrlFromTracePath(path) {
  const match = path.match(/\/ui_runs\/([^/]+)\/(.+)$/);
  if (!match) return null;
  return `/media/runs/${encodeURIComponent(match[1])}/${match[2].split("/").map(encodeURIComponent).join("/")}`;
}

function formatRunDate(id) {
  const match = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
  if (!match) return id;
  const [, year, month, day, hour, minute, second] = match;
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function normalizeAgent(agent) {
  if (!agent) return "producer";
  if (agent.startsWith("layer_builder")) return agent.replace("layer_builder", "producer");
  if (agent.startsWith("producer")) return agent;
  if (agent.startsWith("residual_critic")) return agent;
  if (agent.startsWith("harness_improver")) return agent;
  if (agent.startsWith("loss")) return agent;
  return agent;
}

function agentBase(agent) {
  return normalizeAgent(agent).replace(/_step_\d+$/, "");
}

function agentStep(agent, fallback = null) {
  const match = normalizeAgent(agent).match(/_step_(\d+)/);
  if (match) return Number(match[1]);
  return fallback;
}

function agentName(agent) {
  const base = agentBase(agent);
  if (base === "producer") return "Producer";
  if (base === "residual_critic") return "Critic";
  if (base === "harness_improver") return "Harness Improver";
  if (base === "loss") return "Calculate Accuracy";
  return AGENT_LABELS[base] || base.replaceAll("_", " ");
}

function roleName(role, agent) {
  if (role?.startsWith("producer_audio_diff")) return "Producer trial accuracy";
  if (role?.startsWith("producer_render")) return "Producer trial audio";
  if (role?.startsWith("audio_diff")) return "Accuracy report";
  if (role?.startsWith("render")) return "Audio";
  if (role === "prompt") {
    if (agentBase(agent) === "producer") return "Producer instructions";
    if (agentBase(agent) === "residual_critic") return "Critic instructions";
    if (agentBase(agent) === "harness_improver") return "Harness instructions";
  }
  return ROLE_LABELS[role] || String(role || "File").replaceAll("_", " ");
}

function friendlyWinner(name) {
  if (!name) return "unknown";
  const inner = String(name).match(/^codex_inner_(\d+)/);
  if (inner) return `Producer trial ${Number(inner[1]) + 1}`;
  if (name === "codex") return "Producer proposal";
  return String(name).replaceAll("_", " ");
}

function runStatusTitle(run) {
  if (run.status === "running") return "Running";
  if (run.status === "failed") return "Failed";
  if (run.status === "cancelled") return "Cancelled";
  if (run.status === "interrupted") return "Interrupted";
  if (Number.isFinite(run.final_score)) return run.final_score.toFixed(3);
  return "n/a";
}

function viewStatusForRun(statusValue) {
  if (statusValue === "running") return "Running";
  if (statusValue === "failed") return "Failed";
  if (statusValue === "cancelled") return "Cancelled";
  if (statusValue === "interrupted") return "Interrupted";
  return "Viewing past run";
}

function traceKey(trace) {
  return `${trace.agent}:${trace.step ?? "root"}:${trace.role}:${trace.url || trace.path}`;
}

function codexEventKey(event, index = 0) {
  return `${event.type}:${event.agent}:${event.step ?? "root"}:${event.url || event.path || event.line || index}`;
}

function parseStep(line) {
  const direct = line.match(/\bstep=(\d+)/);
  if (direct) return Number(direct[1]);
  const done = line.match(/\bstep_complete index=(\d+)/);
  return done ? Number(done[1]) : null;
}

function cleanLogLine(line) {
  if (!line) return null;
  if (line.includes("/site-packages/") || line.includes("FutureWarning")) return null;
  if (line.startsWith("codex_prompt_path") || line.startsWith("codex_prompt_hidden")) return null;
  if (line.startsWith("trace_file") || line.startsWith("analysis_start")) return null;
  if (line.includes("/Users/ryanprendergast/")) return null;
  return line
    .replaceAll("layer_builder", "producer")
    .replaceAll("residual_critic", "critic")
    .replaceAll("Loop ", "Iteration ");
}

function cleanCodexLine(line) {
  const text = String(line || "").trim();
  if (!text) return null;
  if (text.includes("/site-packages/") || text.includes("FutureWarning")) return null;
  if (text.startsWith("codex_") || text.startsWith("trace_file")) return null;
  return text
    .replaceAll("•", "-")
    .replace(/\s+/g, " ")
    .replace(/^codex\s*/i, "")
    .trim();
}

function WaveformPlayer({ url, label, compact = false, color = "#323a85", onReady }) {
  const waveRef = useRef(null);
  const playerRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!url || !waveRef.current || !window.WaveSurfer) return undefined;
    const wave = WaveSurfer.create({
      container: waveRef.current,
      url,
      height: compact ? 72 : 104,
      waveColor: "#d8dfed",
      progressColor: color,
      cursorColor: "#2d2d2d",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: true,
    });
    playerRef.current = wave;
    wave.on("finish", () => setPlaying(false));
    wave.on("play", () => setPlaying(true));
    wave.on("pause", () => setPlaying(false));
    onReady?.(wave);
    return () => {
      onReady?.(null);
      wave.destroy();
      playerRef.current = null;
    };
  }, [url, compact, color, onReady]);

  if (!url) return null;
  return h("div", { className: compact ? "audio-player compact" : "audio-player" },
    h("div", { className: "audio-player-head" },
      h("button", {
        type: "button",
        className: "icon-button",
        onClick: () => playerRef.current?.playPause(),
        "aria-label": playing ? "Pause" : "Play",
      }, playing ? "Pause" : "Play"),
      h("strong", null, label || "Audio"),
      h("a", { href: url, target: "_blank" }, "Open")
    ),
    h("div", { className: "audio-wave", ref: waveRef })
  );
}

function AudioComparePlayers({ sourceUrl, renderUrl, renderLabel = "Rendered audio" }) {
  const sourceRef = useRef(null);
  const renderRef = useRef(null);
  const [playingBoth, setPlayingBoth] = useState(false);
  const setSourceReady = useCallback((wave) => { sourceRef.current = wave; }, []);
  const setRenderReady = useCallback((wave) => { renderRef.current = wave; }, []);
  const toggleBoth = useCallback(() => {
    const source = sourceRef.current;
    const render = renderRef.current;
    if (!source || !render) return;
    if (playingBoth) {
      source.pause();
      render.pause();
      setPlayingBoth(false);
      return;
    }
    source.setTime(0);
    render.setTime(0);
    source.play();
    render.play();
    setPlayingBoth(true);
  }, [playingBoth]);

  return h("div", { className: "audio-compare-stack" },
    sourceUrl && renderUrl ? h("button", {
      type: "button",
      className: "play-both-button",
      onClick: toggleBoth,
    }, playingBoth ? "Pause both" : "Play both") : null,
    sourceUrl ? h(WaveformPlayer, { url: sourceUrl, label: "Target source", compact: true, onReady: setSourceReady }) : null,
    renderUrl ? h(WaveformPlayer, { url: renderUrl, label: renderLabel, compact: true, color: "#657cc2", onReady: setRenderReady }) : null
  );
}

function SourceSelector({ songs, currentSongId, onSelect, onWaveReady, audioUrl, onPlayClip, playLabel, onRefresh, onStart, disabled, running }) {
  const waveRef = useRef(null);
  const wave = useRef(null);
  const regions = useRef(null);

  useEffect(() => {
    if (!audioUrl || !waveRef.current || !window.WaveSurfer) return undefined;
    regions.current = WaveSurfer.Regions.create();
    wave.current = WaveSurfer.create({
      container: waveRef.current,
      url: audioUrl,
      height: 172,
      waveColor: "#d8dfed",
      progressColor: "#323a85",
      cursorColor: "#262626",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: true,
      plugins: [regions.current],
    });
    onWaveReady(wave.current, regions.current);
    wave.current.on("ready", () => onSelect(currentSongId, { ready: true }));
    wave.current.on("interaction", () => onSelect(currentSongId, { seek: wave.current.getCurrentTime() }));
    regions.current.on("region-updated", (region) => onSelect(currentSongId, { region }));
    regions.current.on("region-clicked", (region, event) => {
      event.stopPropagation();
      onSelect(currentSongId, { playRegion: region });
    });
    return () => {
      wave.current?.destroy();
      wave.current = null;
      regions.current = null;
    };
  }, [audioUrl]);

  return h("section", { className: "hero-panel" },
    h("div", { className: "hero-main" },
      h("div", { className: "top-line" },
        h("h1", null, "Patchex")
      ),
      h("div", { className: "source-toolbar" },
        h("div", { className: "select-field" },
          h("span", null, "Song"),
          h(Select, { value: currentSongId || "", onValueChange: (value) => onSelect(value, { newFile: true }) },
            h(SelectTrigger, { className: "source-select-trigger", "aria-label": "Song" },
              h(SelectValue, { placeholder: "Choose song" })
            ),
            h(SelectContent, { className: "source-select-content" },
              songs.map((song) => h(SelectItem, { key: song.id, value: song.id }, song.label || song.title || song.id))
            )
          )
        ),
        h(Button, { type: "button", variant: "outline", onClick: onRefresh }, "Refresh"),
        h("a", { href: "/", className: "quiet-link" }, "V0")
      ),
      h("div", { className: "source-wave-frame" },
        h(Button, { type: "button", variant: "outline", className: "wave-play", onClick: onPlayClip }, playLabel === "Pause" ? "Pause" : "Play"),
        h("div", { className: "source-wave", ref: waveRef })
      ),
      h("div", { className: "clip-controls" },
        h(Button, { type: "button", className: running ? "primary inline-primary start-inline is-running" : "primary inline-primary start-inline", onClick: onStart, disabled },
          running ? h("span", { className: "button-spinner", "aria-hidden": "true" }) : null,
          h("span", null, "Start reconstruction")
        )
      )
    )
  );
}

function Scoreboard({ report }) {
  const scores = report?.best_scores || {};
  if (!report) return null;
  return h("section", { className: "section-block" },
    h("div", { className: "section-title" },
      h("h2", null, "Accuracy"),
      h("p", null, "Time-based loss components make obvious mismatches visible.")
    ),
    h("div", { className: "scoreboard react-scoreboard" },
      SCORE_FIELDS.map((name) => {
        const value = Number(scores[name] || 0);
        return h(Card, { className: "score-card", key: name },
          h(CardHeader, { className: "score-card-header" },
            h(CardDescription, null, name.replaceAll("_", " ")),
            h(CardTitle, null, value.toFixed(3))
          ),
          h(CardContent, null,
            h("div", { className: "score-bar" }, h("i", { style: { width: `${Math.max(0, Math.min(100, value * 100))}%` } }))
          )
        );
      })
    )
  );
}

function traceOrder(trace) {
  const role = trace.role || "";
  if (role === "prompt") return 0;
  if (role === "answer") return 1;
  if (role === "parsed_answer" || role === "layer_analysis") return 2;
  if (role.includes("recommendation")) return 3;
  if (role.includes("audio_diff") || role.includes("accuracy")) return 4;
  if (role.includes("render") || role.endsWith("audio")) return 5;
  return 6;
}

function transcriptLabel(trace) {
  const name = trace.name || trace.path?.split("/").pop() || "artifact";
  return `${roleName(trace.role, trace.agent)} (${name})`;
}

function currentRouteRunId() {
  const path = window.location.pathname.replace(/^\/+/, "");
  if (!path || path === "v1" || path === "v1.html") return null;
  if (path.startsWith("v1/")) {
    const id = path.split("/")[1];
    return RUN_ID_PATTERN.test(id) ? id : null;
  }
  return RUN_ID_PATTERN.test(path) ? path : null;
}

function pushRunRoute(runId) {
  if (!runId) return;
  const target = `/${runId}`;
  if (window.location.pathname !== target) window.history.pushState({}, "", target);
}

function AgentTranscript({ traces }) {
  const [items, setItems] = useState([]);
  const tracesKey = traces.map(traceKey).join("|");

  useEffect(() => {
    let cancelled = false;
    const sorted = [...traces].sort((a, b) => traceOrder(a) - traceOrder(b) || transcriptLabel(a).localeCompare(transcriptLabel(b)));
    Promise.all(sorted.map(async (trace) => {
      const name = trace.name || trace.path?.split("/").pop() || "artifact";
      if (name.endsWith(".wav")) {
        return { trace, text: `[audio] ${name}\n${trace.url || trace.path || ""}` };
      }
      if (!trace.url) return { trace, text: trace.path || "" };
      try {
        const res = await fetch(trace.url);
        if (!res.ok) throw new Error(`Could not load ${trace.url}`);
        const text = await res.text();
        return { trace, text: text.length > 24000 ? `${text.slice(0, 24000)}\n... [truncated; open artifact for full file]` : text };
      } catch (error) {
        return { trace, text: error.stack || String(error) };
      }
    })).then((loaded) => {
      if (!cancelled) setItems(loaded);
    });
    return () => { cancelled = true; };
  }, [tracesKey]);

  const transcript = !traces.length
    ? "waiting for files..."
    : !items.length
      ? "loading..."
      : items.map(({ text }) => text || "").join("\n");

  return h(Terminal, { className: "agent-transcript agent-terminal", sequence: false, startOnView: false },
    h("span", { className: "agent-terminal-text" }, transcript)
  );
}

function FlowAgentNode({ data, selected }) {
  const classes = ["workflow-node", `status-${data.status || "waiting"}`];
  if (selected) classes.push("selected");
  const openRow = (row, event) => {
    event.stopPropagation();
    data.onOpen?.(row.trace, data);
  };
  return h("div", { className: classes.join(" ") },
    h(Handle, { className: "workflow-handle workflow-handle-left", type: "target", position: Position.Left }),
    h("span", { className: "workflow-node-head" },
      h("span", { className: "workflow-node-icon" }, data.icon ? h(data.icon, { size: 18, strokeWidth: 2.2 }) : data.label.slice(0, 1)),
      h("span", { className: "workflow-node-title" }, data.label),
      data.status === "running" ? h("span", { className: "node-spinner", "aria-label": "In progress" }) : null
    ),
    (data.rows || []).map((row) => h(row.trace ? "button" : "span", {
      type: row.trace ? "button" : undefined,
      className: row.trace ? "workflow-node-row clickable nodrag nopan" : "workflow-node-row",
      key: `${row.label}-${row.value}`,
      onClick: row.trace ? (event) => openRow(row, event) : undefined,
      onPointerDown: row.trace ? (event) => event.stopPropagation() : undefined,
      onMouseDown: row.trace ? (event) => event.stopPropagation() : undefined,
      onKeyDown: row.trace ? (event) => {
        if (event.key === "Enter" || event.key === " ") openRow(row, event);
      } : undefined,
    },
      h("span", null, row.label),
      h("strong", null, row.value)
    )),
    h(Handle, { className: "workflow-handle workflow-handle-right", type: "source", position: Position.Right })
  );
}

function IterationFrameNode({ data }) {
  return h("div", { className: "iteration-frame-node" });
}

function AccuracyNode({ data, selected }) {
  const classes = ["accuracy-node", `status-${data.status || "waiting"}`];
  if (selected) classes.push("selected");
  return h("button", {
    type: "button",
    className: classes.join(" "),
    title: "Calculate accuracy",
    onClick: (event) => {
      event.stopPropagation();
      data.onOpen?.();
    },
  },
    h(Handle, { className: "workflow-handle workflow-handle-left", type: "target", position: Position.Left }),
    h(Activity, { size: 14, strokeWidth: 2.4 }),
    h("span", null, data.score || "..."),
    h(Handle, { className: "workflow-handle workflow-handle-right", type: "source", position: Position.Right })
  );
}

const nodeTypes = { agent: FlowAgentNode, iterationFrame: IterationFrameNode, accuracy: AccuracyNode };

function traceSet(traces, base, step = null) {
  return traces.filter((trace) => {
    const sameStep = step === null ? trace.step === null || trace.step === undefined : trace.step === step;
    if (!sameStep) return false;
    if (base === "loss") return agentBase(trace.agent) === "loss" || trace.role?.includes("audio_diff") || trace.role?.includes("winner_render");
    return agentBase(trace.agent) === base;
  });
}

function outputRowsForAgent(base, step, nodeTraces, winners, notes, statusKey, allTraces = []) {
  const byRole = (...patterns) => nodeTraces.find((trace) => patterns.some((pattern) => trace.role === pattern || trace.role?.includes(pattern)));
  const sameStepTraces = allTraces.filter((trace) => {
    const traceStep = trace.step ?? agentStep(trace.agent, null);
    return step === null ? traceStep === null || traceStep === undefined : traceStep === step;
  });
  const sameStepByRole = (...patterns) => sameStepTraces.find((trace) => patterns.some((pattern) => trace.role === pattern || trace.role?.includes(pattern)));
  const filename = (trace) => trace?.name || trace?.path?.split("/").pop() || "Open";
  if (base === "producer") {
    const audioTrace = byRole("producer_render", "render") || sameStepByRole("winner_render", "render");
    const sessionTrace = byRole("accepted_session", "session_proposal", "candidate_session", "session") || sameStepByRole("accepted_session", "session_proposal", "candidate_session", "session");
    return [
      { label: "Audio", value: filename(audioTrace), trace: audioTrace },
      { label: "Session", value: filename(sessionTrace), trace: sessionTrace },
    ];
  }
  if (base === "loss") {
    const winner = step !== null && step !== undefined ? winners[step] : null;
    return [
      { label: "Score", value: winner?.score || "pending", trace: byRole("winner_audio_diff", "audio_diff") },
      { label: "Audio", value: filename(byRole("winner_render", "render")), trace: byRole("winner_render", "render") },
    ];
  }
  if (base === "baseline") {
    const sourceTrace = byRole("source_clip") || allTraces.find((trace) => trace.name === "source_clip.wav");
    const renderTrace = byRole("winner_render") || allTraces.find((trace) => trace.name === "current_render_step_initial.wav");
    const lossTrace = byRole("audio_diff") || allTraces.find((trace) => trace.name === "loss_report_step_initial.json");
    return [
      { label: "Target", value: filename(sourceTrace), trace: sourceTrace },
      { label: "First render", value: filename(renderTrace), trace: renderTrace },
      { label: "Initial loss", value: filename(lossTrace), trace: lossTrace },
    ];
  }
  const latestNote = (notes[statusKey] || []).at(-1);
  if (base === "harness_improver") {
    return [
      { label: "Improvement plan", value: filename(byRole("harness_improvement", "file")), trace: byRole("harness_improvement", "file") },
      { label: "Focus", value: latestNote || "loss, graph, prompt advice" },
    ];
  }
  return [
    { label: "Brief", value: filename(byRole("recommendation")), trace: byRole("recommendation") },
    { label: "Prompt", value: filename(byRole("prompt")) || latestNote || "Open", trace: byRole("prompt") },
  ];
}

function scoreMeaning(name) {
  if (name.includes("envelope")) return "Timing and loudness shape over the clip.";
  if (name.includes("beat_grid")) return "Beat-aligned similarity, so wrong moments are penalized.";
  if (name.includes("band")) return "Energy match across frequency bands over time.";
  if (name.includes("modulation")) return "LFO-like motion rate and depth.";
  if (name.includes("spectral") || name.includes("mel")) return "Overall tone and frequency shape.";
  if (name.includes("chroma") || name.includes("pitch")) return "Pitch and harmonic content.";
  if (name.includes("stereo")) return "Width and mid-side similarity.";
  if (name.includes("transient")) return "Attack and onset behavior.";
  return "Higher is closer to the target.";
}

const SCORE_GROUPS = [
  { title: "Timing / Automation", keys: ["exact_envelope_50ms", "exact_band_50ms", "beat_grid_envelope", "beat_grid_band", "directional_delta", "band_envelope_by_time"] },
  { title: "Timbre", keys: ["mel_spectrogram", "multi_resolution_spectral", "a_weighted_spectral", "spectral_features", "harmonic_noise", "codec_latent"] },
  { title: "Motion", keys: ["spectral_motion", "centroid_trajectory", "modulation", "modulation_periodicity", "modulation_rate", "modulation_depth", "transient_classification"] },
  { title: "Pitch", keys: ["pitch_chroma", "f0_contour"] },
  { title: "Stereo", keys: ["stereo_width", "beat_grid_mid_side"] },
];

const SCORE_LABELS = {
  time_series_core: "Time series",
  timbre_core: "Timbre",
  motion_core: "Motion",
  pitch_core: "Pitch",
  stereo_core: "Stereo",
  structural_penalty: "Penalty",
};

const BAND_NAMES = ["sub", "bass", "low_mid", "mid", "presence", "air"];
const HIDDEN_SCORE_KEYS = new Set(["arrangement_preservation"]);

function accuracyPayload(data) {
  if (data?.global_mix_diff?.scores) {
    return {
      scores: data.global_mix_diff.scores || {},
      diagnostics: data.global_mix_diff.diagnostics || {},
      source: "global_mix_diff",
    };
  }
  return {
    scores: data?.scores || data?.best_scores || {},
    diagnostics: data?.diagnostics || {},
    source: "report",
  };
}

function formatScoreKey(key) {
  return String(key || "").replaceAll("_", " ");
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function scoreTone(value) {
  if (value >= 0.85) return "good";
  if (value >= 0.7) return "ok";
  if (value >= 0.55) return "warn";
  return "bad";
}

function scoreAverage(scores, keys) {
  const values = keys.map((key) => Number(scores[key])).filter(Number.isFinite);
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function windowSeverity(row) {
  const deltaRms = Math.abs(Number(row?.delta_rms) || 0);
  const bands = Object.values(row?.delta_bands || {}).map((value) => Math.abs(Number(value) || 0));
  return deltaRms + (bands.length ? Math.max(...bands) : 0);
}

function windowLocalScore(row, maxSeverity) {
  if (!maxSeverity) return 1;
  return clamp01(1 - windowSeverity(row) / maxSeverity);
}

function deltaColor(value, maxAbs) {
  const delta = Number(value) || 0;
  const strength = maxAbs ? Math.min(1, Math.abs(delta) / maxAbs) : 0;
  if (strength < 0.08) return "rgba(77, 130, 92, .22)";
  const alpha = 0.18 + strength * 0.66;
  return delta > 0 ? `rgba(183, 75, 49, ${alpha})` : `rgba(43, 99, 157, ${alpha})`;
}

function ScoreBar({ value, label }) {
  const numeric = clamp01(value);
  return h("div", { className: `score-bar score-${scoreTone(numeric)}`, "aria-label": label },
    h("i", { style: { width: `${numeric * 100}%` } })
  );
}

function ScoreGroup({ title, scores, keys }) {
  const available = keys.filter((key) => Number.isFinite(Number(scores[key])));
  if (!available.length) return null;
  const average = scoreAverage(scores, available);
  return h("section", { className: "accuracy-group" },
    h("div", { className: "accuracy-group-head" },
      h("span", null, title),
      h("strong", { className: `score-text-${scoreTone(average)}` }, average.toFixed(3))
    ),
    h(ScoreBar, { value: average, label: `${title} score` }),
    h("div", { className: "accuracy-group-metrics" },
      available.map((key) => {
        const value = Number(scores[key]);
        return h("div", { className: "accuracy-metric", key },
          h("div", { className: "accuracy-row-head" },
            h("span", null, formatScoreKey(key)),
            h("strong", { className: `score-text-${scoreTone(value)}` }, value.toFixed(3))
          ),
          h(ScoreBar, { value, label: formatScoreKey(key) }),
          h("p", null, scoreMeaning(key))
        );
      })
    )
  );
}

function ScoreGroupTiles({ groups }) {
  const keys = ["time_series_core", "timbre_core", "motion_core", "pitch_core", "stereo_core", "structural_penalty"];
  const available = keys.filter((key) => Number.isFinite(Number(groups[key])));
  if (!available.length) return null;
  return h("div", { className: "accuracy-core-grid" },
    available.map((key) => {
      const value = Number(groups[key]);
      return h("div", { className: `accuracy-core-tile score-tile-${scoreTone(value)}`, key },
        h("span", null, SCORE_LABELS[key] || formatScoreKey(key)),
        h("strong", null, value.toFixed(3))
      );
    })
  );
}

function AccuracySparkline({ windows, maxSeverity }) {
  if (!windows.length) return null;
  const width = 240;
  const height = 52;
  const points = windows.map((row, index) => {
    const x = windows.length === 1 ? 0 : (index / (windows.length - 1)) * width;
    const y = height - windowLocalScore(row, maxSeverity) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return h("svg", { className: "accuracy-sparkline", viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none", role: "img", "aria-label": "50 millisecond accuracy over time" },
    h("line", { x1: 0, x2: width, y1: height * .3, y2: height * .3 }),
    h("line", { x1: 0, x2: width, y1: height * .7, y2: height * .7 }),
    h("polyline", { points })
  );
}

function FixedWindowTimeline({ diagnostics }) {
  const fixed = diagnostics?.fixed_50ms || {};
  const windows = Array.isArray(fixed.windows) ? fixed.windows : [];
  if (!windows.length) return h("div", { className: "accuracy-empty" }, "No 50ms window diagnostics were recorded for this report.");
  const maxSeverity = Math.max(...windows.map(windowSeverity), 1e-9);
  const maxRms = Math.max(...windows.map((row) => Math.abs(Number(row.delta_rms) || 0)), 1e-9);
  return h("section", { className: "accuracy-timeline-panel" },
    h("div", { className: "accuracy-section-head" },
      h("span", null, "50ms Timeline"),
      h("strong", null, `${windows.length} chunks`)
    ),
    h(AccuracySparkline, { windows, maxSeverity }),
    h("div", { className: "accuracy-delta-strip", role: "img", "aria-label": "RMS error by 50 millisecond chunk" },
      windows.map((row) => {
        const delta = Number(row.delta_rms) || 0;
        return h("i", {
          key: row.index,
          style: { background: deltaColor(delta, maxRms) },
          title: `${Number(row.start || 0).toFixed(2)}-${Number(row.end || 0).toFixed(2)}s, ${delta >= 0 ? "+" : ""}${delta.toFixed(5)} RMS, worst ${row.largest_band_error || "band"}`,
        });
      })
    ),
    h("div", { className: "accuracy-legend" },
      h("span", null, h("i", { className: "under" }), "Candidate quiet"),
      h("span", null, h("i", { className: "match" }), "Close"),
      h("span", null, h("i", { className: "over" }), "Candidate loud")
    )
  );
}

function BandHeatmap({ diagnostics }) {
  const fixed = diagnostics?.fixed_50ms || {};
  const windows = Array.isArray(fixed.windows) ? fixed.windows : [];
  if (!windows.length) return null;
  const maxBand = Math.max(...windows.flatMap((row) => BAND_NAMES.map((name) => Math.abs(Number(row?.delta_bands?.[name]) || 0))), 1e-9);
  return h("section", { className: "accuracy-heatmap-panel" },
    h("div", { className: "accuracy-section-head" },
      h("span", null, "Band Error"),
      h("strong", null, "50ms cells")
    ),
    h("div", { className: "accuracy-heatmap" },
      BAND_NAMES.map((band) => h(React.Fragment, { key: band },
        h("span", { className: "accuracy-heatmap-label" }, band.replace("_", " ")),
        h("div", { className: "accuracy-heatmap-row" },
          windows.map((row) => {
            const delta = Number(row?.delta_bands?.[band]) || 0;
            return h("i", {
              key: `${band}-${row.index}`,
              style: { background: deltaColor(delta, maxBand) },
              title: `${band.replace("_", " ")} ${Number(row.start || 0).toFixed(2)}-${Number(row.end || 0).toFixed(2)}s: ${delta >= 0 ? "+" : ""}${delta.toFixed(5)}`,
            });
          })
        )
      ))
    )
  );
}

function WeakWindows({ diagnostics }) {
  const fixed = diagnostics?.fixed_50ms || {};
  const weak = Array.isArray(fixed.weak_windows) ? fixed.weak_windows.slice(0, 8) : [];
  if (!weak.length) return null;
  const maxSeverity = Math.max(...weak.map(windowSeverity), 1e-9);
  return h("section", { className: "accuracy-weak-panel" },
    h("div", { className: "accuracy-section-head" },
      h("span", null, "Worst Windows"),
      h("strong", null, "largest local errors")
    ),
    weak.map((row) => {
      const delta = Number(row.delta_rms) || 0;
      const severity = windowSeverity(row) / maxSeverity;
      return h("div", { className: "accuracy-weak-row", key: row.index },
        h("div", null,
          h("strong", null, `${Number(row.start || 0).toFixed(2)}-${Number(row.end || 0).toFixed(2)}s`),
          h("span", null, `${delta >= 0 ? "too loud" : "too quiet"} · ${row.largest_band_error || "band"}`)
        ),
        h("em", { style: { width: `${Math.max(7, severity * 100)}%` } })
      );
    })
  );
}

function AccuracyViewer({ trace }) {
  const [data, setData] = useState(null);
  const [tab, setTab] = useState("overview");
  useEffect(() => {
    if (!trace?.url) return undefined;
    let cancelled = false;
    fetch(trace.url)
      .then((res) => res.json())
      .then((json) => { if (!cancelled) setData(json); })
      .catch((error) => { if (!cancelled) setData({ error: error.stack || String(error) }); });
    return () => { cancelled = true; };
  }, [trace?.url]);
  const payload = accuracyPayload(data);
  const scores = payload.scores;
  const diagnostics = payload.diagnostics;
  const groups = diagnostics.score_groups || {};
  const keys = Object.keys(scores).filter((key) => Number.isFinite(Number(scores[key])) && key !== "final" && !HIDDEN_SCORE_KEYS.has(key)).slice(0, 24);
  const finalScore = Number(scores.final || 0);
  if (!trace) return null;
  if (!data) return h("div", { className: "sidebar-loading" }, "Loading accuracy report...");
  if (data.error) return h("pre", { className: "sidebar-pre" }, data.error);
  return h("div", { className: "accuracy-viewer" },
    h("div", { className: "accuracy-summary" },
      h("div", null,
        h("strong", { className: `score-text-${scoreTone(finalScore)}` }, finalScore.toFixed(3)),
        h("span", null, payload.source === "global_mix_diff" ? "Global mix similarity" : "Final similarity")
      ),
      h("em", { className: `accuracy-verdict score-tile-${scoreTone(finalScore)}` },
        finalScore >= 0.85 ? "Strong match" : finalScore >= 0.7 ? "Close, inspect weak areas" : finalScore >= 0.55 ? "Needs targeted repair" : "Major mismatch"
      )
    ),
    h(ScoreGroupTiles, { groups }),
    h("div", { className: "accuracy-tabs" },
      ["overview", "timeline", "bands", "raw"].map((name) => h("button", {
        key: name,
        type: "button",
        className: tab === name ? "active" : "",
        onClick: () => setTab(name),
      }, name))
    ),
    tab === "overview" ? h("div", { className: "accuracy-bars" },
      SCORE_GROUPS.map((group) => h(ScoreGroup, { key: group.title, title: group.title, scores, keys: group.keys })),
      h(WeakWindows, { diagnostics })
    ) : null,
    tab === "timeline" ? h("div", { className: "accuracy-bars" },
      h(FixedWindowTimeline, { diagnostics }),
      h(WeakWindows, { diagnostics })
    ) : null,
    tab === "bands" ? h("div", { className: "accuracy-bars" },
      h(BandHeatmap, { diagnostics }),
      h(WeakWindows, { diagnostics })
    ) : null,
    tab === "raw" ? h("div", { className: "accuracy-bars" },
      keys.map((key) => {
        const value = Number(scores[key] || 0);
        return h("div", { className: "accuracy-row", key },
          h("div", { className: "accuracy-row-head" },
            h("span", null, formatScoreKey(key)),
            h("strong", { className: `score-text-${scoreTone(value)}` }, value.toFixed(3))
          ),
          h(ScoreBar, { value, label: formatScoreKey(key) }),
          h("p", null, scoreMeaning(key))
        );
      })
    ) : null
  );
}

function TextArtifactViewer({ trace }) {
  return h(AgentTranscript, { traces: trace ? [trace] : [] });
}

function noteNumber(note) {
  if (typeof note === "number") return note;
  const text = String(note || "").trim().toUpperCase();
  const match = text.match(/^([A-G])([#B]?)(-?\d+)$/);
  if (!match) return 60;
  const offsets = { C: 0, "C#": 1, DB: 1, D: 2, "D#": 3, EB: 3, E: 4, F: 5, "F#": 6, GB: 6, G: 7, "G#": 8, AB: 8, A: 9, "A#": 10, BB: 10, B: 11 };
  return (Number(match[3]) + 1) * 12 + (offsets[`${match[1]}${match[2]}`] ?? offsets[match[1]] ?? 0);
}

function noteLabel(note) {
  const midi = noteNumber(note);
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

function MidiTimeline({ session }) {
  const layers = Array.isArray(session?.layers) ? session.layers : [];
  const notes = layers.flatMap((layer, layerIndex) =>
    (Array.isArray(layer.notes) ? layer.notes : []).map((note) => ({
      ...note,
      layerId: layer.id || `layer_${layerIndex + 1}`,
      layerRole: layer.role || "Layer",
      midi: noteNumber(note.note),
      label: noteLabel(note.note),
      start: Number(note.start || 0),
      duration: Number(note.duration || 0.01),
      velocity: Number(note.velocity || 0.7),
    }))
  );
  const duration = Math.max(Number(session?.duration || 5), ...notes.map((note) => note.start + note.duration), 1);
  const minMidi = notes.length ? Math.min(...notes.map((note) => note.midi)) : 48;
  const maxMidi = notes.length ? Math.max(...notes.map((note) => note.midi)) : 72;
  const pitchSpan = Math.max(1, maxMidi - minMidi);
  const laneColors = ["#323a85", "#4f8a6b", "#a45d3f", "#6f5aa8", "#9a7b24", "#2d7f8f"];

  return h("section", { className: "midi-viewer" },
    h("div", { className: "midi-viewer-head" },
      h("strong", null, "MIDI"),
      h("span", null, `${notes.length} notes`)
    ),
    notes.length ? h("div", { className: "midi-roll", style: { "--midi-rows": String(Math.min(18, pitchSpan + 1)) } },
      h("div", { className: "midi-grid", "aria-hidden": "true" },
        [0, 1, 2, 3, 4, 5].map((tick) => h("i", { key: tick, style: { left: `${(tick / 5) * 100}%` } }))
      ),
      notes.map((note, index) => h("div", {
        className: "midi-note",
        key: `${note.layerId}-${index}-${note.start}`,
        title: `${note.layerId} ${note.label} ${note.start.toFixed(2)}s`,
        style: {
          left: `${Math.max(0, Math.min(100, (note.start / duration) * 100))}%`,
          width: `${Math.max(1.2, Math.min(100, (note.duration / duration) * 100))}%`,
          top: `${Math.max(0, Math.min(92, ((maxMidi - note.midi) / pitchSpan) * 88 + 4))}%`,
          background: laneColors[index % laneColors.length],
          opacity: Math.max(0.45, Math.min(1, note.velocity)),
        },
      }, h("span", null, note.label)))
    ) : h("div", { className: "midi-empty" }, "No note events in this session."),
    layers.length ? h("div", { className: "midi-layers" },
      layers.map((layer) => h("span", { key: layer.id || layer.role }, layer.id || layer.role || "layer"))
    ) : null
  );
}

function SessionArtifactViewer({ trace, selected, sourceArtifact, allTraces = [] }) {
  const [session, setSession] = useState(null);
  useEffect(() => {
    if (!trace?.url) return undefined;
    let cancelled = false;
    fetch(trace.url)
      .then((res) => res.json())
      .then((json) => { if (!cancelled) setSession(json); })
      .catch((error) => { if (!cancelled) setSession({ error: error.stack || String(error) }); });
    return () => { cancelled = true; };
  }, [trace?.url]);

  const isRenderAudio = (item) => {
    const name = item?.name || item?.path?.split("/").pop() || "";
    return name !== "source_clip.wav" && (
      item?.role?.includes("render") ||
      name === "current_render_step_initial.wav" ||
      name.match(/^producer_reconstruction_step_.*\.wav$/) ||
      name.match(/^patch_render_step_.*\.wav$/) ||
      name === "final_reconstruction.wav"
    );
  };
  const outputAudio = (selected.traces || []).find(isRenderAudio) ||
    allTraces.find((item) => item.step === selected.step && (
      isRenderAudio(item)
    ));
  return h("div", { className: "session-artifact-viewer" },
    h("div", { className: "sidebar-audio-compare" },
      h(AudioComparePlayers, {
        sourceUrl: sourceArtifact?.url,
        renderUrl: outputAudio?.url,
        renderLabel: "Producer audio",
      })
    ),
    session?.error ? h("pre", { className: "sidebar-pre" }, session.error) : session ? h(MidiTimeline, { session }) : h("div", { className: "sidebar-loading" }, "Loading session...")
  );
}

function SidebarDetail({ selected, detail, sourceArtifact, allTraces = [] }) {
  const trace = detail?.trace;
  const name = trace?.name || trace?.path?.split("/").pop() || "";
  const isAudio = name.endsWith(".wav");
  const isSourceAudio = name === "source_clip.wav" || trace?.role === "source_clip";
  const isAccuracy = trace?.role?.includes("audio_diff") || name.includes("audio_diff") || name.includes("accuracy");
  const isSession = trace?.role?.includes("session") || name.includes("session_step_") || name === "session.json";
  return h("aside", { className: "workflow-sidebar" },
    h("div", { className: "sidebar-head" },
      h("div", null,
        h("span", null, selected.step === null || selected.step === undefined ? "Agent" : `Iteration ${selected.step + 1}`),
        h("h3", null, detail?.label || selected.label)
      ),
      h(Badge, { variant: "outline" }, selected.status === "completed" ? "Done" : selected.status === "running" ? "Working" : "Waiting")
    ),
    selected.notes?.length && !detail ? h("p", { className: "sidebar-note" }, selected.notes[selected.notes.length - 1]) : null,
    isAudio ? h("div", { className: "sidebar-audio-compare" },
      isSourceAudio
        ? h(WaveformPlayer, { url: trace.url, label: "Target source", compact: true })
        : h(AudioComparePlayers, {
            sourceUrl: sourceArtifact?.url,
            renderUrl: trace.url,
            renderLabel: detail?.label || "Rendered audio",
          })
    ) : isAccuracy ? h(AccuracyViewer, { trace }) : isSession ? h(SessionArtifactViewer, { trace, selected, sourceArtifact, allTraces }) : detail ? h(TextArtifactViewer, { trace }) : h(AgentActivity, { selected })
  );
}

function activityKind(trace) {
  if (trace.role === "prompt") return "Prompt";
  if (trace.role === "answer") return "Answer";
  if (trace.role?.includes("render") || trace.name?.endsWith(".wav")) return "Audio render";
  if (trace.role?.includes("audio_diff")) return "Accuracy report";
  if (trace.role?.includes("recommendation")) return "Critic brief";
  if (trace.role?.includes("session")) return "Session file";
  if (trace.role?.includes("layer_analysis")) return "Layer plan";
  return roleName(trace.role, trace.agent);
}

function AgentActivity({ selected }) {
  const traces = [...(selected.traces || [])].sort((a, b) => traceOrder(a) - traceOrder(b));
  const active = selected.status === "running";
  const [elapsed, setElapsed] = useState(0);
  const [loadedText, setLoadedText] = useState({});
  const codexEvents = selected.codexEvents || [];
  const requestEvent = codexEvents.find((event) => event.type === "codex_request") || traces.find((trace) => trace.role === "prompt");
  const responseEvent = [...codexEvents].reverse().find((event) => event.type === "codex_response") || traces.find((trace) => trace.role === "answer");
  const progressLines = codexEvents
    .filter((event) => event.type === "codex_log")
    .map((event) => cleanCodexLine(event.line))
    .filter(Boolean)
    .slice(-8);
  const textKey = [requestEvent?.url || requestEvent?.path || "", responseEvent?.url || responseEvent?.path || ""].join("|");
  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return undefined;
    }
    setElapsed(0);
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.max(1, Math.floor((Date.now() - started) / 1000))), 1000);
    return () => clearInterval(timer);
  }, [active, selected.id]);
  useEffect(() => {
    let cancelled = false;
    const events = [requestEvent, responseEvent].filter((event) => event?.url);
    if (!events.length) return undefined;
    Promise.all(events.map(async (event) => {
      try {
        const res = await fetch(event.url);
        if (!res.ok) throw new Error(`Could not load ${event.url}`);
        const text = await res.text();
        return [event.url, text.length > 18000 ? `${text.slice(0, 18000)}\n\n[Open the file for the full text.]` : text];
      } catch (error) {
        return [event.url, error.stack || String(error)];
      }
    })).then((entries) => {
      if (!cancelled) setLoadedText((current) => ({ ...current, ...Object.fromEntries(entries) }));
    });
    return () => { cancelled = true; };
  }, [textKey]);
  const latestNote = selected.notes?.at(-1);
  const requestText = requestEvent?.url ? loadedText[requestEvent.url] : "";
  const responseText = responseEvent?.url ? loadedText[responseEvent.url] : "";
  const artifacts = traces.filter((trace) => !["prompt", "answer"].includes(trace.role));
  return h("div", { className: "agent-activity" },
    h("div", { className: "codex-status-line" }, active ? `Working for ${elapsed || 1}s` : selected.status === "completed" ? "Finished" : "Waiting"),
    requestEvent ? h("p", { className: "producer-plain-line" }, requestText || "Loading request...") : null,
    progressLines.length ? progressLines.map((line, index) => h("p", { className: "producer-plain-line", key: `${line}-${index}` }, line)) : latestNote ? h("p", { className: "producer-plain-line" }, latestNote) : null,
    responseEvent ? h("p", { className: "producer-plain-line" }, responseText || (active ? "Waiting for the Producer response..." : "Response file is not available yet.")) : null,
    artifacts.length ? h("p", { className: "producer-plain-line" }, artifacts.slice(0, 8).map((trace) => roleName(trace.role, trace.agent)).join(", ")) : null,
    active ? h("div", { className: "thinking-shimmer" }, "Thinking") : null
  );
}

function WorkflowCanvas({ traces, statuses, notes, winners, artifacts, codexEvents }) {
  const traceSteps = traces.map((trace) => trace.step).filter((step) => step !== null && step !== undefined);
  const statusSteps = Object.keys(statuses)
    .map((key) => key.match(/_(\d+)$/)?.[1])
    .filter((step) => step !== undefined)
    .map((step) => Number(step));
  const steps = Array.from(new Set([...traceSteps, ...statusSteps])).sort((a, b) => a - b);
  const [selectedId, setSelectedId] = useState("critic-0");
  const [selectedDetail, setSelectedDetail] = useState(null);

  useEffect(() => {
    if (steps.some((step) => selectedId.endsWith(`-${step}`))) return;
    setSelectedId(steps.length ? `critic-${steps[0]}` : "critic-0");
  }, [steps.join(","), selectedId]);

  const makeAgentNode = (id, label, base, step, x, y, parentId = undefined, detail = "") => {
    const statusKey = step === null ? base : `${base}_${step}`;
    const icon = base === "producer" ? SlidersHorizontal : base === "loss" ? Activity : base === "harness_improver" ? RefreshCw : MessageSquareText;
    const nodeTraces = traceSet(traces, base, step);
    const nodeCodexEvents = (codexEvents || []).filter((event) => {
      const eventStep = event.step ?? agentStep(event.agent, null);
      const sameStep = step === null ? eventStep === null || eventStep === undefined : eventStep === step;
      return sameStep && agentBase(event.agent) === base;
    });
    const rows = outputRowsForAgent(base, step, nodeTraces, winners, notes, statusKey, traces);
    return {
      id,
      type: "agent",
      position: { x, y },
      parentId,
      extent: parentId ? "parent" : undefined,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      draggable: false,
      data: {
        id,
        label,
        detail,
        icon,
        rows: rows.map((row) => ({ ...row, value: row.value || "pending" })),
        onOpen: (trace, nodeData) => {
          setSelectedId(nodeData.id);
          setSelectedDetail({ trace, label: `${nodeData.label}: ${roleName(trace.role, trace.agent)}` });
        },
        base,
        step,
        status: statuses[statusKey] || (traceSet(traces, base, step).length ? "running" : "waiting"),
        traces: nodeTraces,
        codexEvents: nodeCodexEvents,
        notes: notes[statusKey] || [],
      },
    };
  };

  const arrowMarker = { type: MarkerType.ArrowClosed, width: 24, height: 24, color: "#323a85" };
  const makeEdge = (id, source, target, animated = false) => ({
    id,
    source,
    target,
    type: "smoothstep",
    animated,
    markerEnd: arrowMarker,
    style: { stroke: "#323a85", strokeWidth: 2 },
  });

  const nodes = [];
  const edges = [];
  const baselineTraces = traces.filter((trace) => ["source_clip.wav", "current_render_step_initial.wav", "loss_report_step_initial.json"].includes(trace.name));
  if (baselineTraces.length) {
    const rows = outputRowsForAgent("baseline", null, baselineTraces, winners, notes, "baseline", traces);
    nodes.push({
      id: "baseline",
      type: "agent",
      position: { x: 300, y: 28 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      draggable: false,
      data: {
        id: "baseline",
        label: "Initial Baseline",
        detail: "before first Critic brief",
        icon: Activity,
        rows: rows.map((row) => ({ ...row, value: row.value || "pending" })),
        onOpen: (trace, nodeData) => {
          setSelectedId(nodeData.id);
          setSelectedDetail({ trace, label: `${nodeData.label}: ${roleName(trace.role, trace.agent)}` });
        },
        base: "baseline",
        step: null,
        status: baselineTraces.some((trace) => trace.name === "loss_report_step_initial.json") ? "completed" : "running",
        traces: baselineTraces,
        codexEvents: [],
        notes: [],
      },
    });
  }
  const makeAccuracyNode = (step, x, y, parentId) => {
    const winner = winners[step];
    const accuracyTrace = traceSet(traces, "loss", step).find((trace) => trace.role?.includes("winner_audio_diff") || trace.role?.includes("audio_diff"));
    return {
      id: `accuracy-${step}`,
      type: "accuracy",
      position: { x, y },
      parentId,
      extent: parentId ? "parent" : undefined,
      draggable: false,
      data: {
        score: winner?.score,
        status: statuses[`loss_${step}`] || (accuracyTrace ? "completed" : "waiting"),
        onOpen: accuracyTrace ? () => {
          setSelectedId(`producer-${step}`);
          setSelectedDetail({ trace: accuracyTrace, label: `Accuracy: ${roleName(accuracyTrace.role, accuracyTrace.agent)}` });
        } : undefined,
      },
    };
  };
  steps.forEach((step, index) => {
    const frameId = `iteration-${step}`;
    const y = 250 + index * 290;
    nodes.push({
      id: frameId,
      type: "iterationFrame",
      position: { x: 8, y: y + 22 },
      draggable: false,
      selectable: false,
      style: { width: 880, height: 196 },
      data: {},
    });
    nodes.push(makeAgentNode(`critic-${step}`, "Critic", "residual_critic", step, 36, 34, frameId, "writes Producer brief"));
    nodes.push(makeAgentNode(`producer-${step}`, "Producer", "producer", step, 336, 34, frameId, "writes session files"));
    nodes.push(makeAccuracyNode(step, 636, 78, frameId));
    if (index === 0 && baselineTraces.length) edges.push(makeEdge("baseline-critic-0", "baseline", `critic-${step}`, statuses[`residual_critic_${step}`] === "running"));
    edges.push(makeEdge(`c-p-${step}`, `critic-${step}`, `producer-${step}`, statuses[`producer_${step}`] === "running"));
    edges.push(makeEdge(`p-a-${step}`, `producer-${step}`, `accuracy-${step}`, statuses[`loss_${step}`] === "running"));
    const nextStep = steps[index + 1];
    if (nextStep !== undefined) edges.push(makeEdge(`a-c-${step}-${nextStep}`, `accuracy-${step}`, `critic-${nextStep}`, statuses[`residual_critic_${nextStep}`] === "running"));
  });

  const selectedNode = nodes.find((node) => node.id === selectedId && node.type === "agent") || nodes.find((node) => node.type === "agent");
  const selected = selectedNode?.data || { label: "Select a block", traces: [] };
  const liveSourceArtifact = artifacts?.find((artifact) => artifact.name === "source_clip.wav") ||
    traces.find((trace) => trace.name === "source_clip.wav" || trace.role === "source_clip");

  return h("section", { className: "workflow-section" },
    h("div", { className: "workflow-layout" },
      h("div", { className: "workflow-canvas" },
        h(ReactFlow, {
          key: `flow-${steps.join("-") || "empty"}`,
          nodes,
          edges,
          nodeTypes,
          fitView: true,
          fitViewOptions: { padding: 0.18, maxZoom: 0.98, includeHiddenNodes: false },
          minZoom: 0.18,
          maxZoom: 1.25,
          defaultEdgeOptions: { markerEnd: arrowMarker, style: { stroke: "#323a85", strokeWidth: 2 } },
          nodesDraggable: false,
          nodesConnectable: false,
          elementsSelectable: true,
          panOnScroll: true,
          onNodeClick: (_, node) => {
            if (node.type === "agent") {
              setSelectedId(node.id);
              setSelectedDetail(null);
            }
          },
        },
          h(Background, { gap: 18, color: "#ececec" }),
          h(Controls, { showInteractive: false })
        )
      ),
      h(SidebarDetail, { selected, detail: selectedDetail, sourceArtifact: liveSourceArtifact, allTraces: traces })
    )
  );
}

function Artifacts({ artifacts, onReport }) {
  useEffect(() => {
    const reportArtifact = artifacts.find((artifact) => artifact.name === "reconstruction_report.json");
    if (!reportArtifact) return undefined;
    let cancelled = false;
    fetch(reportArtifact.url)
      .then((res) => res.json())
      .then((report) => { if (!cancelled) onReport(report); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [artifacts]);

  if (!artifacts.length) return null;
  const sortedArtifacts = [...artifacts].sort((a, b) => a.name.localeCompare(b.name));
  return h("section", { className: "section-block" },
    h("div", { className: "section-title" },
      h("h2", null, "Files"),
      h("p", null, "Every output from the run.")
    ),
    h(Accordion, { className: "artifact-groups", type: "single", collapsible: true, defaultValue: "files" },
      h(AccordionItem, { className: "artifact-group", value: "files" },
        h(AccordionTrigger, null,
          h("span", null, "Files"),
          h(Badge, { variant: "outline" }, String(sortedArtifacts.length))
        ),
        h(AccordionContent, null,
          h("div", { className: "artifact-list" },
            sortedArtifacts.map((artifact) => h("a", { className: artifact.name.endsWith(".wav") ? "audio-artifact-link" : "", href: artifact.url, target: "_blank", key: artifact.url },
              h("span", null, artifact.name),
              artifact.name.endsWith(".wav") ? h(Badge, { variant: "secondary" }, "audio") : null
            ))
          )
        )
      )
    )
  );
}

function Comparison({ artifacts }) {
  const source = artifacts.find((artifact) => artifact.name === "source_clip.wav");
  const final = artifacts.find((artifact) => artifact.name === "final_reconstruction.wav");
  if (!source || !final) return null;
  return h("section", { className: "section-block" },
    h("div", { className: "section-title" },
      h("h2", null, "Source vs Output"),
      h("p", null, "Scrub either waveform and compare the same five seconds directly.")
    ),
    h("div", { className: "comparison-grid react-comparison" },
      h("section", { className: "comparison-item" },
        h("div", { className: "comparison-head" }, h("span", null, "Source"), h("strong", null, "Selected clip")),
        h(WaveformPlayer, { url: source.url, label: "Source" })
      ),
      h("section", { className: "comparison-item" },
        h("div", { className: "comparison-head" }, h("span", null, "Output"), h("strong", null, "Final reconstruction")),
        h(WaveformPlayer, { url: final.url, label: "Output", color: "#657cc2" })
      )
    )
  );
}

function RunHistory({ runs, onLoad, onRefresh }) {
  return h("section", { className: "section-block" },
    h("div", { className: "section-title with-action" },
      h("div", null, h("h2", null, "Past Runs"), h("p", null, "Open a previous reconstruction and inspect its trace.")),
      h(Button, { type: "button", variant: "outline", onClick: onRefresh }, "Refresh")
    ),
    h("div", { className: "run-history react-run-history" },
      runs.length
        ? runs.map((run) => h(Card, { key: run.id, className: run.status === "running" ? "run-card active-run-card" : `run-card status-${run.status || "unknown"}`, role: "button", tabIndex: 0, onClick: () => onLoad(run), onKeyDown: (event) => {
          if (event.key === "Enter" || event.key === " ") onLoad(run);
        } },
          h(CardHeader, null,
            h(CardDescription, { className: "run-time" }, formatRunDate(run.id)),
            h(CardTitle, null, runStatusTitle(run))
          ),
          h(CardContent, null,
            h("span", null, run.overall_mix || "Reconstruction run"),
            h("em", null, `${run.status || "unknown"} · ${run.stage_count || 0} stages`)
          )
        ))
        : h("div", { className: "empty-history" }, "No V1 runs yet.")
    )
  );
}

function App() {
  const [songs, setSongs] = useState([]);
  const [runs, setRuns] = useState([]);
  const [currentSongId, setCurrentSongId] = useState("");
  const [clipStart, setClipStart] = useState(0);
  const [currentClip, setCurrentClip] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [starting, setStarting] = useState(false);
  const [playLabel, setPlayLabel] = useState("Play selection");
  const [runNotes, setRunNotes] = useState([]);
  const [traces, setTraces] = useState([]);
  const [codexEvents, setCodexEvents] = useState([]);
  const [statuses, setStatuses] = useState({});
  const [notes, setNotes] = useState({});
  const [winners, setWinners] = useState({});
  const [artifacts, setArtifacts] = useState([]);
  const [report, setReport] = useState(null);
  const activeRun = useRef(localStorage.getItem("v1ActiveRunId"));
  const sourceTools = useRef({ wave: null, regions: null, activeRegion: null });

  const addRunNote = useCallback((line) => {
    const clean = cleanLogLine(line);
    if (!clean) return;
    setRunNotes((items) => [...items.slice(-40), clean]);
  }, []);

  const addAgentNote = useCallback((id, note) => {
    if (!note) return;
    setNotes((current) => ({ ...current, [id]: [...(current[id] || []).slice(-200), note] }));
  }, []);

  const addTrace = useCallback((payload) => {
    let agent = normalizeAgent(payload.agent || "producer");
    const role = payload.role || "file";
    if (agent === "session" && role === "current") return;
    let step = payload.step ?? agentStep(agent, null);
    if (step !== null && role.startsWith("producer_render")) agent = `producer_step_${String(step).padStart(2, "0")}`;
    if (step !== null && role.startsWith("audio_diff")) agent = `loss_step_${String(step).padStart(2, "0")}`;
    const path = payload.path || "";
    const trace = {
      agent,
      step,
      role,
      path,
      url: payload.url || fileUrlFromTracePath(path),
      name: payload.name || path.split("/").pop(),
    };
    setTraces((current) => current.some((item) => traceKey(item) === traceKey(trace)) ? current : [...current, trace]);
    const base = agentBase(agent);
    const statusId = step === null || step === undefined ? base : `${base}_${step}`;
    setStatuses((current) => ({ ...current, [statusId]: current[statusId] === "completed" ? "completed" : "running" }));
    if (role === "layer_analysis") addAgentNote(statusId, "Layer plan ready.");
    if (role === "session_proposal") addAgentNote(statusId, "Producer wrote a session proposal.");
    if (role === "accepted_session") addAgentNote(statusId, "Accepted as current session.");
    if (role === "recommendation") addAgentNote(statusId, "Critic wrote the next Producer brief.");
    if (["answer", "layer_analysis", "recommendation", "recommendation_initial", "accepted_session", "candidate_session"].includes(role)) {
      setStatuses((current) => ({ ...current, [statusId]: "completed" }));
    }
  }, [addAgentNote]);

  const addCodexEvent = useCallback((payload) => {
    const agent = normalizeAgent(payload.agent || "producer");
    const step = payload.step ?? agentStep(agent, null);
    const item = { ...payload, agent, step };
    setCodexEvents((current) => current.some((event, index) => codexEventKey(event, index) === codexEventKey(item, index)) ? current : [...current, item]);
    const statusId = step === null || step === undefined ? agentBase(agent) : `${agentBase(agent)}_${step}`;
    if (payload.type === "codex_request" || payload.type === "codex_log") {
      setStatuses((current) => ({ ...current, [statusId]: current[statusId] === "completed" ? "completed" : "running" }));
    }
  }, []);

  const loadSongs = useCallback(async () => {
    const data = await api("/api/songs");
    setSongs(data.songs || []);
    if ((data.songs || []).length && !currentSongId) setCurrentSongId(data.songs[0].id);
  }, [currentSongId]);

  const loadRuns = useCallback(async () => {
    const data = await api("/api/reconstruction-runs");
    const serverRuns = data.runs || [];
    const activeId = localStorage.getItem("v1ActiveRunId");
    const merged = activeId && !serverRuns.some((run) => run.id === activeId)
      ? [{ id: activeId, status: "running", overall_mix: "Reconnect to running reconstruction", artifacts: [], stage_count: 0 }, ...serverRuns]
      : serverRuns;
    setRuns(merged);
    return merged;
  }, []);

  const resetRunView = useCallback(() => {
    setRunNotes([]);
    setTraces([]);
    setCodexEvents([]);
    setStatuses({});
    setNotes({});
    setWinners({});
    setArtifacts([]);
    setReport(null);
  }, []);

  const ensureFiveSecondRegion = useCallback((start = 0) => {
    const wave = sourceTools.current.wave;
    const regions = sourceTools.current.regions;
    if (!wave || !regions) return;
    const duration = wave.getDuration() || CLIP_SECONDS;
    const safeStart = Math.max(0, Math.min(Math.max(0, duration - CLIP_SECONDS), start));
    regions.clearRegions();
    sourceTools.current.activeRegion = regions.addRegion({
      start: safeStart,
      end: safeStart + CLIP_SECONDS,
      color: "rgba(50, 58, 133, 0.16)",
      drag: true,
      resize: false,
    });
    wave.setTime(safeStart);
    setClipStart(safeStart);
  }, []);

  const handleSourceEvent = useCallback((songId, action = {}) => {
    if (action.newFile) {
      setCurrentSongId(songId);
      resetRunView();
      return;
    }
    if (action.ready) {
      ensureFiveSecondRegion(0);
      return;
    }
    if (action.seek !== undefined) {
      ensureFiveSecondRegion(action.seek);
      return;
    }
    if (action.region) {
      const wave = sourceTools.current.wave;
      const duration = wave?.getDuration() || CLIP_SECONDS;
      const safeStart = Math.max(0, Math.min(Math.max(0, duration - CLIP_SECONDS), action.region.start));
      if (Math.abs(action.region.start - safeStart) > 0.001 || Math.abs(action.region.end - (safeStart + CLIP_SECONDS)) > 0.001) {
        action.region.setOptions({ start: safeStart, end: safeStart + CLIP_SECONDS });
      }
      wave?.setTime(safeStart);
      sourceTools.current.activeRegion = action.region;
      setClipStart(safeStart);
    }
    if (action.playRegion) playSelectedClip();
  }, [ensureFiveSecondRegion, resetRunView]);

  const playSelectedClip = useCallback(() => {
    const wave = sourceTools.current.wave;
    const region = sourceTools.current.activeRegion;
    if (!wave || !region) return;
    if (wave.isPlaying()) {
      wave.pause();
      setPlayLabel("Play selection");
      return;
    }
    setPlayLabel("Pause");
    wave.play(region.start, region.end);
    setTimeout(() => setPlayLabel("Play selection"), CLIP_SECONDS * 1000 + 250);
  }, []);

  const routeRunLog = useCallback((line) => {
    const step = parseStep(line);
    if (line.startsWith("codex_log ")) {
      const agent = normalizeAgent(line.match(/\bagent=([^ ]+)/)?.[1] || "");
      const stepId = agentStep(agent, null);
      const id = stepId === null ? agentBase(agent) : `${agentBase(agent)}_${stepId}`;
      addAgentNote(id, line);
      return;
    }
    if (line.startsWith("codex_start")) {
      const agent = normalizeAgent(line.match(/\bagent=([^ ]+)/)?.[1] || "");
      const stepId = agentStep(agent, null);
      const id = stepId === null ? agentBase(agent) : `${agentBase(agent)}_${stepId}`;
      setStatuses((current) => ({ ...current, [id]: "running" }));
      addAgentNote(id, line);
      return;
    }
    if (line.startsWith("winner_summary") || line.startsWith("producer_winner")) {
      const winner = line.match(/\bwinner=([^ ]+)/)?.[1] || "unknown";
      const score = line.match(/\bscore=([0-9.]+)/)?.[1] || "n/a";
      const idx = Number(line.match(/\bstep=(\d+)/)?.[1] || 0);
      setWinners((current) => ({ ...current, [idx]: { winner, score } }));
      addAgentNote(`loss_${idx}`, `Winner: ${friendlyWinner(winner)} (${score}).`);
      setStatuses((current) => ({ ...current, [`loss_${idx}`]: "completed" }));
      return;
    }
    if (line.startsWith("codex_done")) {
      const agent = normalizeAgent(line.match(/\bagent=([^ ]+)/)?.[1] || "");
      const stepId = agentStep(agent, null);
      const id = stepId === null ? agentBase(agent) : `${agentBase(agent)}_${stepId}`;
      setStatuses((current) => ({ ...current, [id]: "completed" }));
      addAgentNote(id, line);
      return;
    }
    const agentMatch = line.match(/\bagent_stage ([a-z_]+)(?: step=(\d+))?/);
    if (agentMatch) {
      const base = normalizeAgent(agentMatch[1]);
      const idx = agentMatch[2] !== undefined ? Number(agentMatch[2]) : null;
      const id = idx === null ? agentBase(base) : `${agentBase(base)}_${idx}`;
      setStatuses((current) => ({ ...current, [id]: "running" }));
      addAgentNote(id, "Started.");
      return;
    }
    if (line.startsWith("step_complete")) {
      const idx = parseStep(line);
      if (idx !== null) {
        setStatuses((current) => ({
          ...current,
          [`producer_${idx}`]: "completed",
          [`loss_${idx}`]: "completed",
          [`residual_critic_${idx}`]: "completed",
        }));
      }
      return;
    }
    addRunNote(line);
  }, [addAgentNote, addRunNote]);

  const renderTraceArtifacts = useCallback(async (artifactList, loadedReport = null) => {
    const traceArtifacts = artifactList.filter((artifact) => {
      const name = artifact.name;
      return (
        name.startsWith("codex_") ||
        name.startsWith("audio_diff_") ||
        name.startsWith("producer_audio_diff_") ||
        name === "arrangement.json" ||
        name === "full_arrangement.json" ||
        name === "patch_session_current.json" ||
        name.startsWith("patch_session_step_") ||
        name.startsWith("patch_report_step_") ||
        name.startsWith("patch_render_step_") ||
        name.startsWith("patch_ops_step_") ||
        name.startsWith("patch_ops_applied_step_") ||
        name === "source_clip.wav" ||
        name === "current_render_step_initial.wav" ||
        name === "loss_report_step_initial.json" ||
        name.startsWith("critic_brief_step_") ||
        name.startsWith("harness_improver_step_") ||
        name.match(/^producer_reconstruction_step_\d+_.+\.wav$/) ||
        name === "final_reconstruction.wav" ||
        name.startsWith("recommendation_step_") ||
        name === "recommendation_initial.json" ||
        name === "source_profile.json" ||
        name === "pattern_constraints.json" ||
        name === "beat_grid.json" ||
        name === "layer_analysis.json" ||
        name.match(/^session_step_\d+_(codex_proposal|producer_winner|accepted)\.json$/)
      );
    });
    traceArtifacts.forEach((artifact) => {
      const pseudoPath = `/ui_runs/${artifact.url.split("/")[3]}/${artifact.name}`;
      const role = artifact.name.startsWith("codex_")
        ? artifact.name.includes("_prompt") ? "prompt" : "answer"
        : artifact.name.startsWith("producer_audio_diff") ? "producer_audio_diff"
        : artifact.name.startsWith("patch_session_step_") ? "accepted_session"
        : artifact.name.startsWith("patch_report_step_") ? "audio_diff"
        : artifact.name.startsWith("patch_render_step_") ? "winner_render"
        : artifact.name.startsWith("patch_ops_step_") ? "session"
        : artifact.name.startsWith("patch_ops_applied_step_") ? "session"
        : artifact.name === "source_clip.wav" ? "source_clip"
        : artifact.name === "current_render_step_initial.wav" ? "winner_render"
        : artifact.name === "loss_report_step_initial.json" ? "audio_diff"
        : artifact.name.startsWith("critic_brief_step_") ? "recommendation"
        : artifact.name === "arrangement.json" || artifact.name === "full_arrangement.json" ? "layer_analysis"
        : artifact.name.startsWith("harness_improver_step_") ? "harness_improvement"
        : artifact.name.startsWith("audio_diff") ? "audio_diff"
        : artifact.name.match(/^producer_reconstruction_step_/) ? "producer_render"
        : artifact.name === "final_reconstruction.wav" ? "render"
        : artifact.name.endsWith(".wav") ? "render"
        : artifact.name.match(/^session_step_\d+_accepted\.json$/) ? "accepted_session"
        : artifact.name.match(/^session_step_\d+_producer_winner\.json$/) ? "candidate_session"
        : artifact.name.match(/^session_step_\d+_codex_proposal\.json$/) ? "session_proposal"
        : artifact.name.startsWith("session") ? "session"
        : artifact.name.startsWith("recommendation") ? "recommendation"
        : "file";
      let agent = artifact.name
        .replace(/^codex_/, "")
        .replace(/_(prompt|answer)\.txt$/, "")
        .replace(/\.json$/, "");
      const stepMatch = artifact.name.match(/step_(\d+)/);
      const step = stepMatch ? Number(stepMatch[1]) : null;
      if (role === "producer_render" && step !== null) agent = `producer_step_${String(step).padStart(2, "0")}`;
      if (["accepted_session", "candidate_session", "session_proposal"].includes(role) && step !== null) agent = `producer_step_${String(step).padStart(2, "0")}`;
      if (role === "audio_diff" && step !== null) agent = `loss_step_${String(step).padStart(2, "0")}`;
      if (["source_clip.wav", "loss_report_step_initial.json", "current_render_step_initial.wav"].includes(artifact.name)) agent = "baseline";
      if (role === "harness_improvement" && step !== null) agent = `harness_improver_step_${String(step).padStart(2, "0")}`;
      if (role === "recommendation" && step !== null) agent = `residual_critic_step_${String(step).padStart(2, "0")}`;
      addTrace({ agent, role, path: pseudoPath, url: artifact.url, name: artifact.name, step });
    });
    (loadedReport?.history || []).forEach((item) => {
      if (item.step === undefined || item.step === null || !item.winner) return;
      const step = Number(item.step);
      const score = Number(item.scores?.final || item.scores || 0);
      setWinners((current) => ({ ...current, [step]: { winner: item.winner, score: score.toFixed(3) } }));
      setStatuses((current) => ({ ...current, [`producer_${step}`]: "completed", [`loss_${step}`]: "completed", [`residual_critic_${step}`]: "completed" }));
      const audioName = item.audio_path?.split("/").pop();
      const diffName = item.audio_diff_path?.split("/").pop();
      const audioArtifact = artifactList.find((artifact) => artifact.name === audioName);
      const diffArtifact = artifactList.find((artifact) => artifact.name === diffName);
      if (audioArtifact) addTrace({ agent: `loss_step_${String(step).padStart(2, "0")}`, role: "winner_render", step, path: `/ui_runs/${audioArtifact.url.split("/")[3]}/${audioArtifact.name}`, url: audioArtifact.url, name: audioArtifact.name });
      if (diffArtifact) addTrace({ agent: `loss_step_${String(step).padStart(2, "0")}`, role: "winner_audio_diff", step, path: `/ui_runs/${diffArtifact.url.split("/")[3]}/${diffArtifact.name}`, url: diffArtifact.url, name: diffArtifact.name });
    });
  }, [addTrace]);

  const attachEvents = useCallback((runId) => {
    const events = new EventSource(`/api/reconstructions/${runId}/events`);
    events.onmessage = async (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "trace_file") addTrace(payload);
      if (["codex_request", "codex_response", "codex_log"].includes(payload.type)) addCodexEvent(payload);
      if (payload.type === "log") routeRunLog(payload.line);
      if (payload.type === "heartbeat") addRunNote("Still running.");
      if (payload.type === "done") {
        events.close();
        setStatus(payload.status === "completed" ? "Complete" : payload.status === "cancelled" ? "Cancelled" : "Failed");
        localStorage.removeItem("v1ActiveRunId");
        activeRun.current = null;
        addRunNote(`Run ${payload.status}.`);
        const job = await api(`/api/reconstructions/${runId}`);
        setArtifacts(job.artifacts || []);
        await renderTraceArtifacts(job.artifacts || []);
        loadRuns().catch((error) => addRunNote(error.stack || String(error)));
      }
    };
    events.onerror = () => addRunNote("Event stream disconnected; refresh to reconnect.");
    return events;
  }, [addCodexEvent, addRunNote, addTrace, loadRuns, renderTraceArtifacts, routeRunLog]);

  const startRun = useCallback(async () => {
    if (!currentSongId) return;
    const song = songs.find((item) => item.id === currentSongId);
    try {
      resetRunView();
      setStarting(true);
      setStatus("Extracting");
      const start = clipStart;
      addRunNote(`Extracting ${song?.label || currentSongId} from ${start.toFixed(2)}s to ${(start + CLIP_SECONDS).toFixed(2)}s.`);
      const extracted = await api("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ song_id: currentSongId, start, duration: CLIP_SECONDS }),
      });
      const clipEvents = new EventSource(`/api/clips/${extracted.clip_id}/events`);
      clipEvents.onmessage = async (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "log") addRunNote(payload.line);
        if (payload.type === "done") {
          clipEvents.close();
          if (payload.status !== "completed") {
            setStatus("Failed");
            addRunNote(payload.error || "Clip extraction failed.");
            return;
          }
          setCurrentClip(payload.result.clip);
          setStatus("Running");
          addRunNote("Clip ready. Starting reconstruction.");
          const data = await api("/api/reconstruct", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clip: payload.result.clip, song_id: currentSongId, clip_start: start, steps: 5, local_trials: 0, max_layers: 5 }),
          });
          activeRun.current = data.run_id;
          localStorage.setItem("v1ActiveRunId", data.run_id);
          setStatuses({ residual_critic_0: "running" });
          addAgentNote("residual_critic_0", "Started.");
          pushRunRoute(data.run_id);
          attachEvents(data.run_id);
          loadRuns().catch((error) => addRunNote(error.stack || String(error)));
          setStarting(false);
        }
      };
      clipEvents.onerror = () => addRunNote("Clip event stream disconnected.");
    } catch (error) {
      setStatus("Failed");
      addRunNote(error.stack || String(error));
      setStarting(false);
    }
  }, [addAgentNote, addRunNote, attachEvents, clipStart, currentSongId, loadRuns, resetRunView, songs]);

  const loadPastRun = useCallback(async (run, options = {}) => {
    resetRunView();
    if (options.push !== false) pushRunRoute(run.id);
    if (run.status === "running") {
      setStatus("Running");
      activeRun.current = run.id;
      localStorage.setItem("v1ActiveRunId", run.id);
      try {
        const job = await api(`/api/reconstructions/${run.id}`);
        setArtifacts(job.artifacts || []);
        await renderTraceArtifacts(job.artifacts || []);
      } catch (error) {
        addRunNote(error.stack || String(error));
      }
      attachEvents(run.id);
      addRunNote(`Reconnected to ${run.id}.`);
      return;
    }
    setStatus(viewStatusForRun(run.status));
    setArtifacts(run.artifacts || []);
    const reportArtifact = run.artifacts.find((artifact) => artifact.name === "reconstruction_report.json");
    let loadedReport = null;
    if (reportArtifact) {
      loadedReport = await fetch(reportArtifact.url).then((res) => res.json());
      setReport(loadedReport);
    }
    await renderTraceArtifacts(run.artifacts || [], loadedReport);
    addRunNote(`Opened ${run.id}.`);
  }, [addRunNote, renderTraceArtifacts, resetRunView]);

  const killRun = useCallback(async () => {
    const runId = currentRouteRunId();
    if (!runId) return;
    try {
      await api(`/api/reconstructions/${runId}/kill`, { method: "POST" });
      setStatus("Cancelled");
      localStorage.removeItem("v1ActiveRunId");
      if (activeRun.current === runId) activeRun.current = null;
      addRunNote("run killed by user");
      loadRuns().catch((error) => addRunNote(error.stack || String(error)));
    } catch (error) {
      addRunNote(error.stack || String(error));
    }
  }, [addRunNote, loadRuns]);

  useEffect(() => {
    loadSongs().catch((error) => addRunNote(error.stack || String(error)));
    loadRuns()
      .then((loadedRuns) => {
        const routeId = currentRouteRunId();
        if (!routeId) return;
        const run = loadedRuns.find((item) => item.id === routeId);
        if (run) return loadPastRun(run, { push: false });
        return api(`/api/reconstructions/${routeId}`)
          .then(async (job) => {
            setStatus(viewStatusForRun(job.status));
            setArtifacts(job.artifacts || []);
            await renderTraceArtifacts(job.artifacts || []);
            if (job.status === "running") {
              activeRun.current = routeId;
              localStorage.setItem("v1ActiveRunId", routeId);
              attachEvents(routeId);
            }
          })
          .catch(() => addRunNote(`Run ${routeId} is not available.`));
      })
      .catch((error) => addRunNote(error.stack || String(error)));
  }, []);

  useEffect(() => {
    if (currentRouteRunId()) return undefined;
    if (!activeRun.current) return undefined;
    let events = null;
    api(`/api/reconstructions/${activeRun.current}`)
      .then(async (job) => {
        if (job.status !== "running") {
          localStorage.removeItem("v1ActiveRunId");
          activeRun.current = null;
          return;
        }
        setStatus("Running");
        setArtifacts(job.artifacts || []);
        await renderTraceArtifacts(job.artifacts || []);
        events = attachEvents(activeRun.current);
        addRunNote("Reconnected to active run.");
      })
      .catch(() => {
        localStorage.removeItem("v1ActiveRunId");
        activeRun.current = null;
      });
    return () => events?.close();
  }, []);

  const audioUrl = currentSongId ? `/media/songs/${encodeURIComponent(currentSongId)}/audio` : "";
  const runActive = status === "Running" || status === "Extracting";
  const routeRunId = currentRouteRunId();
  const hasLoadedRun = runActive || traces.length > 0 || artifacts.length > 0 || report;
  const showRunDetail = Boolean(routeRunId);
  const canKillRun = showRunDetail && runActive && activeRun.current === routeRunId;
  return h("main", { className: showRunDetail ? "react-shell run-detail-shell" : "react-shell" },
    showRunDetail ? h("div", { className: "run-topbar" },
      h("a", { className: "back-link", href: "/v1" },
        h(ArrowLeft, { size: 16, strokeWidth: 2 }),
        h("span", null, "Back")
      ),
      canKillRun ? h(Button, { type: "button", variant: "outline", className: "kill-run-button", onClick: killRun }, "Kill Run") : null
    ) : null,
    showRunDetail ? null : h(SourceSelector, {
      songs,
      currentSongId,
      onSelect: handleSourceEvent,
      onWaveReady: (wave, regions) => {
        sourceTools.current.wave = wave;
        sourceTools.current.regions = regions;
      },
      audioUrl,
      onPlayClip: playSelectedClip,
      playLabel,
      onRefresh: loadSongs,
      onStart: startRun,
      disabled: starting || !currentSongId,
      running: starting,
    }),
    showRunDetail && hasLoadedRun ? h(WorkflowCanvas, { traces, statuses, notes, winners, artifacts, codexEvents }) : null,
    showRunDetail && hasLoadedRun ? h(Comparison, { artifacts }) : null,
    showRunDetail && hasLoadedRun ? h(Scoreboard, { report }) : null,
    showRunDetail && hasLoadedRun ? h(Artifacts, { artifacts, onReport: setReport }) : null,
    showRunDetail ? null : h(RunHistory, { runs, onLoad: loadPastRun, onRefresh: loadRuns })
  );
}

createRoot(document.getElementById("v1-root")).render(h(App));
