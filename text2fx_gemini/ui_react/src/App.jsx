import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Activity, ArrowLeft, Brain, MessageSquareText, SlidersHorizontal } from 'lucide-react';
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
  winner_audio_diff: "Winning accuracy report",
  winner_render: "Winning audio",
  render: "Audio",
};

const AGENT_LABELS = {
  analyzer: "Analyzer",
  producer: "Producer",
  residual_critic: "Critic",
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
    if (agentBase(agent) === "analyzer") return "Analyzer instructions";
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

function traceKey(trace) {
  return `${trace.agent}:${trace.step ?? "root"}:${trace.role}:${trace.url || trace.path}`;
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

function WaveformPlayer({ url, label, compact = false, color = "#323a85" }) {
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
    return () => {
      wave.destroy();
      playerRef.current = null;
    };
  }, [url, compact, color]);

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

function SourceSelector({ files, currentFile, onSelect, onWaveReady, audioUrl, onPlayClip, playLabel, onRefresh, onStart, disabled, running }) {
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
    wave.current.on("ready", () => onSelect(currentFile, { ready: true }));
    wave.current.on("interaction", () => onSelect(currentFile, { seek: wave.current.getCurrentTime() }));
    regions.current.on("region-updated", (region) => onSelect(currentFile, { region }));
    regions.current.on("region-clicked", (region, event) => {
      event.stopPropagation();
      onSelect(currentFile, { playRegion: region });
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
          h("span", null, "Source"),
          h(Select, { value: currentFile || "", onValueChange: (value) => onSelect(value, { newFile: true }) },
            h(SelectTrigger, { className: "source-select-trigger", "aria-label": "Source audio file" },
              h(SelectValue, { placeholder: "Choose source" })
            ),
            h(SelectContent, { className: "source-select-content" },
              files.map((file) => h(SelectItem, { key: file.name, value: file.name }, file.name))
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
  return h("div", { className: classes.join(" ") },
    h(Handle, { className: "workflow-handle workflow-handle-left", type: "target", position: Position.Left }),
    h("span", { className: "workflow-node-head" },
      h("span", { className: "workflow-node-icon" }, data.icon ? h(data.icon, { size: 18, strokeWidth: 2.2 }) : data.label.slice(0, 1)),
      h("span", { className: "workflow-node-title" }, data.label),
      data.status === "running" ? h("span", { className: "node-spinner", "aria-label": "In progress" }) : null
    ),
    (data.rows || []).map((row) => h(row.trace ? "button" : "span", {
      type: row.trace ? "button" : undefined,
      className: row.trace ? "workflow-node-row clickable" : "workflow-node-row",
      key: `${row.label}-${row.value}`,
      onClick: row.trace ? (event) => {
        event.stopPropagation();
        data.onOpen?.(row.trace, data);
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

const nodeTypes = { agent: FlowAgentNode, iterationFrame: IterationFrameNode };

function traceSet(traces, base, step = null) {
  return traces.filter((trace) => {
    const sameStep = step === null ? trace.step === null || trace.step === undefined : trace.step === step;
    if (!sameStep) return false;
    if (base === "loss") return agentBase(trace.agent) === "loss" || trace.role?.includes("audio_diff") || trace.role?.includes("winner_render");
    return agentBase(trace.agent) === base;
  });
}

function outputRowsForAgent(base, step, nodeTraces, winners, notes, statusKey) {
  const byRole = (...patterns) => nodeTraces.find((trace) => patterns.some((pattern) => trace.role === pattern || trace.role?.includes(pattern)));
  const filename = (trace) => trace?.name || trace?.path?.split("/").pop() || "Open";
  if (base === "analyzer") {
    return [
      { label: "Plan", value: filename(byRole("layer_analysis", "parsed_answer")), trace: byRole("layer_analysis", "parsed_answer") },
      { label: "Prompt", value: filename(byRole("prompt")), trace: byRole("prompt") },
    ];
  }
  if (base === "producer") {
    return [
      { label: "Audio", value: filename(byRole("producer_render", "render")), trace: byRole("producer_render", "render") },
      { label: "Session", value: filename(byRole("accepted_session", "session_proposal")), trace: byRole("accepted_session", "session_proposal") },
    ];
  }
  if (base === "loss") {
    const winner = step !== null && step !== undefined ? winners[step] : null;
    return [
      { label: "Score", value: winner?.score || "pending", trace: byRole("winner_audio_diff", "audio_diff") },
      { label: "Audio", value: filename(byRole("winner_render", "render")), trace: byRole("winner_render", "render") },
    ];
  }
  const latestNote = (notes[statusKey] || []).at(-1);
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

function AccuracyViewer({ trace }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!trace?.url) return undefined;
    let cancelled = false;
    fetch(trace.url)
      .then((res) => res.json())
      .then((json) => { if (!cancelled) setData(json); })
      .catch((error) => { if (!cancelled) setData({ error: error.stack || String(error) }); });
    return () => { cancelled = true; };
  }, [trace?.url]);
  const scores = data?.scores || data?.best_scores || {};
  const keys = Object.keys(scores).filter((key) => Number.isFinite(Number(scores[key]))).slice(0, 18);
  if (!trace) return null;
  if (!data) return h("div", { className: "sidebar-loading" }, "Loading accuracy report...");
  if (data.error) return h("pre", { className: "sidebar-pre" }, data.error);
  return h("div", { className: "accuracy-viewer" },
    h("div", { className: "accuracy-summary" },
      h("strong", null, Number(scores.final || 0).toFixed(3)),
      h("span", null, "Final similarity")
    ),
    h("div", { className: "accuracy-bars" },
      keys.map((key) => {
        const value = Number(scores[key] || 0);
        return h("div", { className: "accuracy-row", key },
          h("div", { className: "accuracy-row-head" },
            h("span", null, key.replaceAll("_", " ")),
            h("strong", null, value.toFixed(3))
          ),
          h("div", { className: "score-bar" }, h("i", { style: { width: `${Math.max(0, Math.min(100, value * 100))}%` } })),
          h("p", null, scoreMeaning(key))
        );
      })
    )
  );
}

function TextArtifactViewer({ trace }) {
  return h(AgentTranscript, { traces: trace ? [trace] : [] });
}

function SidebarDetail({ selected, detail, sourceArtifact }) {
  const trace = detail?.trace;
  const name = trace?.name || trace?.path?.split("/").pop() || "";
  const isAudio = name.endsWith(".wav");
  const isAccuracy = trace?.role?.includes("audio_diff") || name.includes("audio_diff") || name.includes("accuracy");
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
      sourceArtifact ? h(WaveformPlayer, { url: sourceArtifact.url, label: "Target source", compact: true }) : null,
      h(WaveformPlayer, { url: trace.url, label: detail?.label || "Output audio", compact: true, color: "#657cc2" })
    ) : isAccuracy ? h(AccuracyViewer, { trace }) : detail ? h(TextArtifactViewer, { trace }) : h(AgentActivity, { selected })
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
  const latestNote = selected.notes?.at(-1);
  const rawLines = [
    ...(selected.notes || []).map((note) => note),
    ...traces.map((trace) => `trace_file agent=${trace.agent} role=${trace.role} path=${trace.path || trace.name || ""}`),
  ];
  return h("div", { className: "agent-activity" },
    h("div", { className: "codex-status-line" }, active ? `Working for ${elapsed || 1}s` : selected.status === "completed" ? "Finished" : "Waiting"),
    h("div", { className: "codex-message" },
      h("p", null, latestNote || (active ? `${selected.label} is running. Waiting for new log output.` : `${selected.label} run log.`))
    ),
    h("pre", { className: "raw-agent-log" }, rawLines.length ? rawLines.join("\n") : active ? "waiting for agent logs..." : "no logs for this agent"),
    active ? h("div", { className: "thinking-shimmer" }, "Thinking") : null
  );
}

function WorkflowCanvas({ traces, statuses, notes, winners, artifacts }) {
  const steps = Array.from(new Set(traces.map((trace) => trace.step).filter((step) => step !== null && step !== undefined))).sort((a, b) => a - b);
  const [selectedId, setSelectedId] = useState("analyzer");
  const [selectedDetail, setSelectedDetail] = useState(null);

  useEffect(() => {
    if (selectedId === "analyzer" || steps.some((step) => selectedId.endsWith(`-${step}`))) return;
    setSelectedId("analyzer");
  }, [steps.join(","), selectedId]);

  const makeAgentNode = (id, label, base, step, x, y, parentId = undefined, detail = "") => {
    const statusKey = step === null ? base : `${base}_${step}`;
    const icon = base === "analyzer" ? Brain : base === "producer" ? SlidersHorizontal : base === "loss" ? Activity : MessageSquareText;
    const nodeTraces = traceSet(traces, base, step);
    const rows = outputRowsForAgent(base, step, nodeTraces, winners, notes, statusKey);
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

  const nodes = [
    makeAgentNode("analyzer", "Analyzer", "analyzer", null, 24, 26, undefined, "one-time source analysis"),
  ];
  const edges = [];
  steps.forEach((step, index) => {
    const frameId = `iteration-${step}`;
    const y = 180 + index * 290;
    const winner = winners[step];
    nodes.push({
      id: frameId,
      type: "iterationFrame",
      position: { x: 8, y: y + 22 },
      draggable: false,
      selectable: false,
      style: { width: 920, height: 196 },
      data: {},
    });
    nodes.push(makeAgentNode(`producer-${step}`, "Producer", "producer", step, 36, 34, frameId, "writes session files"));
    nodes.push(makeAgentNode(`loss-${step}`, "Calculate Accuracy", "loss", step, 336, 34, frameId, "renders and scores"));
    nodes.push(makeAgentNode(`critic-${step}`, "Critic", "residual_critic", step, 636, 34, frameId, "writes next brief"));
    edges.push(makeEdge(`p-l-${step}`, `producer-${step}`, `loss-${step}`, statuses[`loss_${step}`] === "running"));
    edges.push(makeEdge(`l-c-${step}`, `loss-${step}`, `critic-${step}`, statuses[`residual_critic_${step}`] === "running"));
    if (index === 0) edges.push(makeEdge("a-p-0", "analyzer", `producer-${step}`));
    if (index > 0) edges.push(makeEdge(`c-p-${step}`, `critic-${steps[index - 1]}`, `producer-${step}`));
  });

  const selectedNode = nodes.find((node) => node.id === selectedId && node.type === "agent") || nodes.find((node) => node.id === "analyzer");
  const selected = selectedNode?.data || { label: "Select a block", traces: [] };

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
      h(SidebarDetail, { selected, detail: selectedDetail, sourceArtifact: artifacts?.find((artifact) => artifact.name === "source_clip.wav") })
    )
  );
}

function artifactGroup(name) {
  if (name === "final_reconstruction.wav" || name === "source_clip.wav" || name === "distilled_playable_patch.json") return "Essentials";
  if (name === "reconstruction_report.json" || name.startsWith("audio_diff_") || name.startsWith("producer_audio_diff_") || name.includes("accuracy")) return "Accuracy reports";
  if (name.startsWith("session") || name === "manifest.json" || name === "master.json" || name === "routing.json" || name === "timeline.json") return "Session state";
  if (name.startsWith("codex_") || name.startsWith("recommendation") || name === "layer_analysis.json" || name === "source_profile.json" || name === "beat_grid.json" || name === "pattern_constraints.json") return "Agent files";
  if (name.endsWith(".wav")) return "Audio renders";
  return "Other";
}

function groupArtifacts(artifacts) {
  const order = ["Essentials", "Accuracy reports", "Session state", "Agent files", "Audio renders", "Other"];
  const groups = new Map(order.map((name) => [name, []]));
  artifacts.forEach((artifact) => {
    const group = artifactGroup(artifact.name);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(artifact);
  });
  return order.map((name) => [name, groups.get(name) || []]).filter(([, items]) => items.length);
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
  return h("section", { className: "section-block" },
    h("div", { className: "section-title" },
      h("h2", null, "Files"),
      h("p", null, "Grouped outputs from the run.")
    ),
    h(Accordion, { className: "artifact-groups", type: "multiple", defaultValue: ["Essentials"] },
      groupArtifacts(artifacts).map(([group, items]) =>
        h(AccordionItem, { className: "artifact-group", key: group, value: group },
          h(AccordionTrigger, null,
            h("span", null, group),
            h(Badge, { variant: "outline" }, String(items.length))
          ),
          h(AccordionContent, null,
            h("div", { className: "artifact-list" },
              items.map((artifact) => h("a", { className: artifact.name.endsWith(".wav") ? "audio-artifact-link" : "", href: artifact.url, target: "_blank", key: artifact.url },
                h("span", null, artifact.name),
                artifact.name.endsWith(".wav") ? h(Badge, { variant: "secondary" }, "audio") : null
              ))
            )
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
        ? runs.map((run) => h(Card, { key: run.id, className: run.status === "running" ? "run-card active-run-card" : "run-card", role: "button", tabIndex: 0, onClick: () => onLoad(run), onKeyDown: (event) => {
          if (event.key === "Enter" || event.key === " ") onLoad(run);
        } },
          h(CardHeader, null,
            h(CardDescription, { className: "run-time" }, formatRunDate(run.id)),
            h(CardTitle, null, run.status === "running" ? "Running" : Number.isFinite(run.final_score) ? run.final_score.toFixed(3) : "n/a")
          ),
          h(CardContent, null,
            h("span", null, run.overall_mix || "Reconstruction run"),
            h("em", null, `${run.stage_count || 0} stages`)
          )
        ))
        : h("div", { className: "empty-history" }, "No V1 runs yet.")
    )
  );
}

function App() {
  const [files, setFiles] = useState([]);
  const [runs, setRuns] = useState([]);
  const [currentFile, setCurrentFile] = useState("");
  const [clipStart, setClipStart] = useState(0);
  const [currentClip, setCurrentClip] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [playLabel, setPlayLabel] = useState("Play selection");
  const [runNotes, setRunNotes] = useState([]);
  const [traces, setTraces] = useState([]);
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
    setNotes((current) => ({ ...current, [id]: [...(current[id] || []).slice(-4), note] }));
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

  const loadFiles = useCallback(async () => {
    const data = await api("/api/references");
    setFiles(data.files || []);
    if ((data.files || []).length && !currentFile) setCurrentFile(data.files[0].name);
  }, [currentFile]);

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

  const handleSourceEvent = useCallback((file, action = {}) => {
    if (action.newFile) {
      setCurrentFile(file);
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
    if (step !== null && line.includes("score=")) {
      const score = line.match(/\bscore=([0-9.]+)/)?.[1];
      if (score) addAgentNote(`loss_${step}`, `Score ${Number(score).toFixed(3)}.`);
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
        : artifact.name.startsWith("audio_diff") ? "audio_diff"
        : artifact.name.match(/^producer_reconstruction_step_/) ? "producer_render"
        : artifact.name === "final_reconstruction.wav" ? "render"
        : artifact.name.endsWith(".wav") ? "render"
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
      if (role === "audio_diff" && step !== null) agent = `loss_step_${String(step).padStart(2, "0")}`;
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
      if (payload.type === "log") routeRunLog(payload.line);
      if (payload.type === "heartbeat") addRunNote("Still running.");
      if (payload.type === "done") {
        events.close();
        setStatus(payload.status === "completed" ? "Complete" : "Failed");
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
  }, [addRunNote, addTrace, loadRuns, renderTraceArtifacts, routeRunLog]);

  const startRun = useCallback(async () => {
    if (!currentFile) return;
    try {
      resetRunView();
      setStatus("Extracting");
      const start = clipStart;
      addRunNote(`Extracting ${currentFile} from ${start.toFixed(2)}s to ${(start + CLIP_SECONDS).toFixed(2)}s.`);
      const extracted = await api("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reference: currentFile, start, duration: CLIP_SECONDS }),
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
            body: JSON.stringify({ clip: payload.result.clip, steps: 5, local_trials: 0, max_layers: 5 }),
          });
          activeRun.current = data.run_id;
          localStorage.setItem("v1ActiveRunId", data.run_id);
          pushRunRoute(data.run_id);
          attachEvents(data.run_id);
        }
      };
      clipEvents.onerror = () => addRunNote("Clip event stream disconnected.");
    } catch (error) {
      setStatus("Failed");
      addRunNote(error.stack || String(error));
    }
  }, [addRunNote, attachEvents, clipStart, currentFile, resetRunView]);

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
    setStatus("Viewing past run");
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

  useEffect(() => {
    loadFiles().catch((error) => addRunNote(error.stack || String(error)));
    loadRuns()
      .then((loadedRuns) => {
        const routeId = currentRouteRunId();
        if (!routeId) return;
        const run = loadedRuns.find((item) => item.id === routeId);
        if (run) return loadPastRun(run, { push: false });
        return api(`/api/reconstructions/${routeId}`)
          .then(async (job) => {
            setStatus(job.status === "running" ? "Running" : "Viewing past run");
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

  const audioUrl = currentFile ? `/media/references/${encodeURIComponent(currentFile)}` : "";
  const runActive = status === "Running" || status === "Extracting";
  const hasRunView = runActive || traces.length > 0 || artifacts.length > 0 || report;
  const routeRunId = currentRouteRunId();
  return h("main", { className: routeRunId || hasRunView ? "react-shell run-detail-shell" : "react-shell" },
    routeRunId || hasRunView ? h("div", { className: "run-topbar" },
      h("a", { className: "back-link", href: "/v1" },
        h(ArrowLeft, { size: 16, strokeWidth: 2 }),
        h("span", null, "Back")
      )
    ) : null,
    routeRunId || hasRunView ? null : h(SourceSelector, {
      files,
      currentFile,
      onSelect: handleSourceEvent,
      onWaveReady: (wave, regions) => {
        sourceTools.current.wave = wave;
        sourceTools.current.regions = regions;
      },
      audioUrl,
      onPlayClip: playSelectedClip,
      playLabel,
      onRefresh: loadFiles,
      onStart: startRun,
      disabled: runActive || !currentFile,
      running: runActive,
    }),
    hasRunView ? h(WorkflowCanvas, { traces, statuses, notes, winners, artifacts }) : null,
    hasRunView ? h(Comparison, { artifacts }) : null,
    hasRunView ? h(Scoreboard, { report }) : null,
    hasRunView ? h(Artifacts, { artifacts, onReport: setReport }) : null,
    routeRunId || hasRunView ? null : h(RunHistory, { runs, onLoad: loadPastRun, onRefresh: loadRuns })
  );
}

createRoot(document.getElementById("v1-root")).render(h(App));
