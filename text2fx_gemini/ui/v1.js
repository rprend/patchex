const { useCallback, useEffect, useRef, useState } = React;
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

function SourceSelector({ files, currentFile, onSelect, onWaveReady, audioUrl, clipRange, onPlayClip, playLabel, onRefresh, onStart, disabled }) {
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
        h("label", { className: "select-field" },
          h("span", null, "Source"),
          h("select", { value: currentFile || "", onChange: (event) => onSelect(event.target.value, { newFile: true }) },
            files.map((file) => h("option", { key: file.name, value: file.name }, file.name))
          )
        ),
        h("button", { type: "button", onClick: onRefresh }, "Refresh"),
        h("a", { href: "/", className: "quiet-link" }, "V0")
      ),
      h("div", { className: "source-wave-frame" },
        h("button", { type: "button", className: "wave-play", onClick: onPlayClip }, playLabel === "Pause" ? "Pause" : "Play"),
        h("div", { className: "source-wave", ref: waveRef })
      ),
      h("div", { className: "clip-controls" },
        h("strong", null, clipRange),
        h("button", { type: "button", className: "primary inline-primary start-inline", onClick: onStart, disabled }, disabled ? "Running" : "Start reconstruction")
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
        return h("div", { className: "score-card", key: name },
          h("span", null, name.replaceAll("_", " ")),
          h("strong", null, value.toFixed(3)),
          h("div", { className: "score-bar" }, h("i", { style: { width: `${Math.max(0, Math.min(100, value * 100))}%` } }))
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

  if (!traces.length) return h("pre", { className: "agent-transcript" }, "waiting for files...");
  if (!items.length) return h("pre", { className: "agent-transcript" }, "loading...");

  return h("pre", { className: "agent-transcript" },
    items.map(({ text }) => text || "").join("\n")
  );
}

function TraceFile({ trace }) {
  const [content, setContent] = useState(trace.text || "");
  const [loading, setLoading] = useState(false);
  const name = trace.name || trace.path?.split("/").pop() || "artifact";
  const isAudio = name.endsWith(".wav");
  const startsOpen = isAudio || trace.role?.includes("recommendation") || trace.role?.includes("winner_audio_diff");

  useEffect(() => {
    if (!trace.url || isAudio || content) return undefined;
    let cancelled = false;
    setLoading(true);
    fetch(trace.url)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load ${trace.url}`);
        return res.text();
      })
      .then((text) => {
        if (!cancelled) setContent(text.length > 24000 ? `${text.slice(0, 24000)}\n... [truncated; open artifact for full file]` : text);
      })
      .catch((error) => {
        if (!cancelled) setContent(error.stack || String(error));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [trace.url, isAudio]);

  return h("details", { className: "trace-file react-trace-file", open: startsOpen },
    h("summary", null,
      h("span", null, roleName(trace.role, trace.agent)),
      trace.url ? h("a", { href: trace.url, target: "_blank" }, "Open") : null
    ),
    isAudio
      ? h(WaveformPlayer, { url: trace.url, label: roleName(trace.role, trace.agent), compact: true })
      : h("pre", null, loading ? "loading..." : content || trace.path)
  );
}

function AgentCard({ agent, traces, status, notes }) {
  const completed = status === "completed";
  const title = agentName(agent);
  return h("details", { className: `agent-card ${completed ? "completed" : "running"}`, open: false },
    h("summary", { className: "agent-card-head" },
      h("div", null,
        h("strong", null, title),
        notes?.length ? h("p", null, notes[notes.length - 1]) : null
      ),
      h("span", { className: "agent-state" }, completed ? "Done" : "Working")
    ),
    h(AgentTranscript, { traces })
  );
}

function IterationGroup({ step, traces, statuses, notes, winners }) {
  const producer = traces.filter((trace) => agentBase(trace.agent) === "producer");
  const loss = traces.filter((trace) => agentBase(trace.agent) === "loss" || trace.role?.includes("audio_diff") || trace.role?.includes("winner_render"));
  const critic = traces.filter((trace) => agentBase(trace.agent) === "residual_critic");
  const winner = winners[step];
  return h("details", { className: "iteration-group", open: true },
    h("summary", null,
      h("span", null, `Iteration ${step + 1}`),
      h("strong", null, winner ? `Winner: ${friendlyWinner(winner.winner)} · ${winner.score}` : "Produce, calculate accuracy, critique")
    ),
    h("div", { className: "iteration-body" },
      h(AgentCard, { agent: "producer", traces: producer, status: statuses[`producer_${step}`], notes: notes[`producer_${step}`] || [] }),
      h(AgentCard, { agent: "loss", traces: loss, status: statuses[`loss_${step}`], notes: notes[`loss_${step}`] || [] }),
      h(AgentCard, { agent: "residual_critic", traces: critic, status: statuses[`residual_critic_${step}`], notes: notes[`residual_critic_${step}`] || [] })
    )
  );
}

function Timeline({ traces, statuses, notes, winners, runNotes }) {
  const analyzerTraces = traces.filter((trace) => agentBase(trace.agent) === "analyzer");
  const steps = Array.from(new Set(traces.map((trace) => trace.step).filter((step) => step !== null && step !== undefined))).sort((a, b) => a - b);
  return h("section", { className: "section-block timeline-section" },
    h("div", { className: "section-title" },
      h("h2", null, "Trace"),
      h("p", null, "Analyzer plans the target once. Each iteration then shows the Producer change, the scored audio, and the Critic brief that drives the next pass.")
    ),
    runNotes.length ? h("div", { className: "activity-strip" }, runNotes.slice(-5).map((note, index) => h("span", { key: `${note}-${index}` }, note))) : null,
    analyzerTraces.length ? h(AgentCard, { agent: "analyzer", traces: analyzerTraces, status: statuses.analyzer, notes: notes.analyzer || [] }) : null,
    h("div", { className: "iterations" },
      steps.map((step) => h(IterationGroup, {
        key: step,
        step,
        traces: traces.filter((trace) => trace.step === step),
        statuses,
        notes,
        winners,
      }))
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
    h("div", { className: "artifact-groups" },
      groupArtifacts(artifacts).map(([group, items]) => h("details", { className: "artifact-group", key: group, open: group === "Essentials" },
        h("summary", null, h("span", null, group), h("strong", null, String(items.length))),
        h("div", { className: "artifact-list" },
          items.map((artifact) => h("a", { className: artifact.name.endsWith(".wav") ? "audio-artifact-link" : "", href: artifact.url, target: "_blank", key: artifact.url },
            h("span", null, artifact.name),
            artifact.name.endsWith(".wav") ? h("em", null, "audio") : null
          ))
        )
      ))
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
      h("button", { type: "button", onClick: onRefresh }, "Refresh")
    ),
    h("div", { className: "run-history react-run-history" },
      runs.length
        ? runs.map((run) => h("button", { key: run.id, className: "run-card", type: "button", onClick: () => onLoad(run) },
          h("span", { className: "run-time" }, formatRunDate(run.id)),
          h("strong", null, Number.isFinite(run.final_score) ? run.final_score.toFixed(3) : "n/a"),
          h("span", null, run.overall_mix || "Reconstruction run"),
          h("em", null, `${run.stage_count || 0} stages`)
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
    setRuns(data.runs || []);
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
          attachEvents(data.run_id);
        }
      };
      clipEvents.onerror = () => addRunNote("Clip event stream disconnected.");
    } catch (error) {
      setStatus("Failed");
      addRunNote(error.stack || String(error));
    }
  }, [addRunNote, attachEvents, clipStart, currentFile, resetRunView]);

  const loadPastRun = useCallback(async (run) => {
    resetRunView();
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
    loadRuns().catch((error) => addRunNote(error.stack || String(error)));
  }, []);

  useEffect(() => {
    if (!currentFile) return;
    resetRunView();
  }, [currentFile]);

  useEffect(() => {
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
  const clipRange = `${clipStart.toFixed(2)}s - ${(clipStart + CLIP_SECONDS).toFixed(2)}s`;
  return h("main", { className: "react-shell" },
    h(SourceSelector, {
      files,
      currentFile,
      onSelect: handleSourceEvent,
      onWaveReady: (wave, regions) => {
        sourceTools.current.wave = wave;
        sourceTools.current.regions = regions;
      },
      audioUrl,
      clipRange,
      onPlayClip: playSelectedClip,
      playLabel,
      onRefresh: loadFiles,
      onStart: startRun,
      disabled: status === "Running" || status === "Extracting" || !currentFile,
    }),
    h(Timeline, { traces, statuses, notes, winners, runNotes }),
    h(Comparison, { artifacts }),
    h(Scoreboard, { report }),
    h(Artifacts, { artifacts, onReport: setReport }),
    h(RunHistory, { runs, onLoad: loadPastRun, onRefresh: loadRuns })
  );
}

ReactDOM.createRoot(document.getElementById("v1-root")).render(h(App));
