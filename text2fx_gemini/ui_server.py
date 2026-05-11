#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from reference_match import analyze_reference, load_audio, runtime
from text2fx import LLM_MODEL, extract_json_object


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
REFERENCES = ROOT / "references"
SONGS = ROOT / "songs"
UI = ROOT / "ui"
RUNS = ROOT / "ui_runs"
FFMPEG = shutil.which("ffmpeg")

jobs: dict[str, dict] = {}
analysis_jobs: dict[str, dict] = {}
clip_jobs: dict[str, dict] = {}
reconstruction_jobs: dict[str, dict] = {}
EXTERNAL_RUN_ACTIVE_WINDOW_SECONDS = 2 * 60 * 60

NOISY_LOG_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwarn(ing)?\b",
        r"futurewarning",
        r"deprecationwarning",
        r"^\s*\[?debug\]?",
        r"^\s*\[?trace\]?",
        r"token",
        r"rate limit header",
        r"rust_log",
        r"telemetry",
    ]
]

IMPORTANT_LOG_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^codex_start",
        r"^codex_done",
        r"^codex_request",
        r"^codex_response",
        r"^codex_prompt_path",
        r"^candidate=",
        r"^step=",
        r"^step_complete",
        r"^analysis_start",
        r"^analysis_done",
        r"^agent_stage",
        r"^trace_file",
        r"^winner_summary",
        r"^producer_winner",
        r"^layer_building_stopped",
        r"^wrote ",
        r"^analysis ",
        r"^Traceback",
        r"^RuntimeError",
        r"^ValueError",
        r"^FileNotFoundError",
    ]
]


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length))


def safe_reference_path(name: str) -> Path:
    candidate = (REFERENCES / name).resolve()
    if REFERENCES.resolve() not in candidate.parents and candidate != REFERENCES.resolve():
        raise ValueError("Reference path escapes references directory.")
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def safe_song_dir(song_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_-]+$", song_id):
        raise ValueError("Invalid song id.")
    song_dir = (SONGS / song_id).resolve()
    if SONGS.resolve() not in song_dir.parents and song_dir != SONGS.resolve():
        raise ValueError("Song path escapes songs directory.")
    if not song_dir.is_dir():
        raise FileNotFoundError(song_dir)
    return song_dir


def load_song(song_id: str) -> dict:
    song_dir = safe_song_dir(song_id)
    manifest_path = song_dir / "song.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["id"] = str(manifest.get("id") or song_id)
    manifest["_dir"] = str(song_dir)
    manifest["_manifest"] = str(manifest_path)
    return manifest


def song_asset_path(song: dict, key: str) -> Path:
    song_dir = Path(song["_dir"]).resolve()
    rel = song.get(key)
    if not rel:
        raise ValueError(f"Song {song.get('id')} is missing {key}.")
    path = (song_dir / str(rel)).resolve()
    if song_dir not in path.parents and path != song_dir:
        raise ValueError(f"Song asset {key} escapes song directory.")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html"
    if suffix == ".css":
        return "text/css"
    if suffix == ".js":
        return "application/javascript"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def make_clip(source_name: str, start: float, duration: float) -> Path:
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is required for clip extraction.")
    if duration <= 0:
        raise ValueError("Clip duration must be positive.")
    if duration > 12:
        raise ValueError("Clip duration must be 12 seconds or less.")
    source = safe_reference_path(source_name)
    out = REFERENCES / f"{source.stem}_clip_{start:.2f}_{duration:.2f}.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-ac",
            "2",
            "-ar",
            "44100",
            str(out),
        ],
        check=True,
    )
    return out


def make_song_clip(song_id: str, start: float, duration: float) -> Path:
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is required for clip extraction.")
    if duration <= 0:
        raise ValueError("Clip duration must be positive.")
    if duration > 12:
        raise ValueError("Clip duration must be 12 seconds or less.")
    song = load_song(song_id)
    source = song_asset_path(song, "audio")
    out = REFERENCES / f"{song_id}_clip_{start:.2f}_{duration:.2f}.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-ac",
            "2",
            "-ar",
            "44100",
            str(out),
        ],
        check=True,
    )
    return out


def analyze_clip(path: Path, target_part: str = "") -> dict:
    runtime()
    audio, sr = load_audio(path)
    local = analyze_reference(audio, sr)
    return describe_clip_with_gemini(path, local, target_part)


def start_clip_analysis(reference: str, start: float, duration: float, target_part: str) -> str:
    analysis_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    event_queue: queue.Queue[dict] = queue.Queue()
    analysis_jobs[analysis_id] = {
        "id": analysis_id,
        "status": "running",
        "queue": event_queue,
        "result": None,
        "error": None,
    }

    def worker() -> None:
        try:
            event_queue.put({"type": "log", "line": f"extracting exact clip: {reference} @ {start:.2f}s-{(start + duration):.2f}s"})
            clip = make_clip(reference, start, duration)
            event_queue.put({"type": "log", "line": f"clip written: {clip.name}"})
            event_queue.put({"type": "log", "line": "loading extracted WAV and computing local DSP features"})
            runtime()
            audio, sr = load_audio(clip)
            local = analyze_reference(audio, sr)
            event_queue.put({"type": "log", "line": f"local preliminary instrument: {local['instrument_type']}"})
            event_queue.put({"type": "log", "line": "starting Gemini audio analysis"})
            analysis = describe_clip_with_gemini(clip, local, target_part)
            event_queue.put({"type": "log", "line": "Gemini response parsed and validated"})
            event_queue.put({"type": "log", "line": f"detected instrument: {analysis['instrument_type']}"})
            event_queue.put({"type": "log", "line": f"target prompt: {analysis['prompt']}"})
            result = {"clip": clip.name, "url": f"/media/references/{clip.name}", "analysis": analysis}
            analysis_jobs[analysis_id]["result"] = result
            analysis_jobs[analysis_id]["status"] = "completed"
            event_queue.put({"type": "done", "status": "completed", "result": result})
        except Exception as exc:
            analysis_jobs[analysis_id]["status"] = "failed"
            analysis_jobs[analysis_id]["error"] = str(exc)
            event_queue.put({"type": "log", "line": f"analysis failed: {exc}"})
            event_queue.put({"type": "done", "status": "failed", "error": str(exc)})

    threading.Thread(target=worker, daemon=True).start()
    return analysis_id


def start_clip_extract(reference: str, start: float, duration: float, song_id: str = "") -> str:
    clip_id = time.strftime("%Y%m%d_%H%M%S_clip_") + uuid.uuid4().hex[:8]
    event_queue: queue.Queue[dict] = queue.Queue()
    clip_jobs[clip_id] = {
        "id": clip_id,
        "status": "running",
        "queue": event_queue,
        "result": None,
        "error": None,
    }

    def worker() -> None:
        try:
            label = song_id or reference
            event_queue.put({"type": "log", "line": f"extracting exact clip: {label} @ {start:.2f}s-{(start + duration):.2f}s"})
            clip = make_song_clip(song_id, start, duration) if song_id else make_clip(reference, start, duration)
            event_queue.put({"type": "log", "line": f"clip written: {clip.name}"})
            result = {"clip": clip.name, "url": f"/media/references/{clip.name}"}
            clip_jobs[clip_id]["result"] = result
            clip_jobs[clip_id]["status"] = "completed"
            event_queue.put({"type": "done", "status": "completed", "result": result})
        except Exception as exc:
            clip_jobs[clip_id]["status"] = "failed"
            clip_jobs[clip_id]["error"] = str(exc)
            event_queue.put({"type": "log", "line": f"clip extraction failed: {exc}"})
            event_queue.put({"type": "done", "status": "failed", "error": str(exc)})

    threading.Thread(target=worker, daemon=True).start()
    return clip_id


def load_secrets_into_env(env: dict[str, str]) -> None:
    secrets = Path.home() / ".codex/secrets.env"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                if stripped.startswith("export "):
                    stripped = stripped.removeprefix("export ").strip()
                key, value = stripped.split("=", 1)
                env.setdefault(key, value.strip().strip("'\""))


def gemini_client():
    env = os.environ.copy()
    load_secrets_into_env(env)
    key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before analyzing clips.")
    from google import genai

    return genai.Client(api_key=key)


def describe_clip_with_gemini(path: Path, local: dict, target_part: str = "") -> dict:
    from google.genai import types

    focus = target_part.strip() or "Infer the main synth-like part in the clip. Ignore drums, vocals, and full-mix traits unless they are central to the synth patch."
    instruction = f"""
You are analyzing exactly one extracted five-second audio clip for synth patch matching.

Target part to recreate:
{focus}

Describe the target part, not the whole mix. If multiple instruments are present, choose the instrument family and axes for the target part above.

Return only JSON with this exact shape:
{{
  "instrument_type": "bass_synth|lead_synth|pad_synth|pluck_synth|arp_synth|texture_synth",
  "axes": {{
    "spectral_tone": ["2-4 short production phrases"],
    "harmonic_texture": ["2-4 short production phrases"],
    "envelope": ["2-4 short production phrases"],
    "space": ["2-4 short production phrases"],
    "motion": ["2-4 short production phrases"],
    "rhythm": ["2-4 short production phrases"],
    "mix_role": ["2-4 short production phrases"]
  }},
  "fixed_pattern": {{
    "tempo": 60-180,
    "grid": "16th",
    "steps": [16 semitone offsets from root, integers -24..24],
    "velocity": [16 values from 0..1 matching steps]
  }},
  "prompt": "one compact target prompt summarizing the axes for recipe generation"
}}

Use the local DSP features only as supporting evidence; the audio clip is authoritative.
Choose fixed_pattern from the audio and target part. For pads/background swells, prefer held or slow-pulse notes. For arps/plucks, use active 16th-note motion. For bass, use sparse root/octave/fifth pulses. Do not use a deterministic template.
Local features:
{json.dumps(local["features"], indent=2)}

Local preliminary guess:
{json.dumps({"instrument_type": local["instrument_type"], "axes": local["axes"]}, indent=2)}
"""
    client = gemini_client()
    wav_bytes = path.read_bytes()
    response = generate_clip_description_with_retry(
        client,
        [
            types.Content(
                parts=[
                    types.Part(text=instruction),
                    types.Part(inline_data=types.Blob(mime_type="audio/wav", data=wav_bytes)),
                ]
            )
        ],
    )
    payload = extract_json_object(response.text or "")
    required_axes = {"spectral_tone", "harmonic_texture", "envelope", "space", "motion", "rhythm", "mix_role"}
    if set(payload.get("axes", {})) != required_axes:
        raise ValueError(f"Gemini axis analysis returned wrong axes: {payload}")
    if payload.get("instrument_type") not in {"bass_synth", "lead_synth", "pad_synth", "pluck_synth", "arp_synth", "texture_synth"}:
        raise ValueError(f"Gemini axis analysis returned invalid instrument_type: {payload}")
    fixed_pattern = sanitize_fixed_pattern(payload.get("fixed_pattern"))
    return {"features": local["features"], "axes": payload["axes"], "instrument_type": payload["instrument_type"], "fixed_pattern": fixed_pattern, "prompt": payload["prompt"], "target_part": target_part, "analysis_source": "gemini_audio"}


def sanitize_fixed_pattern(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini analysis missing fixed_pattern: {payload}")
    required = {"tempo", "grid", "steps", "velocity"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Gemini fixed_pattern missing keys: {missing}")
    steps = payload["steps"]
    velocity = payload["velocity"]
    if not isinstance(steps, list) or len(steps) != 16:
        raise ValueError("Gemini fixed_pattern.steps must be a 16-item list.")
    if not isinstance(velocity, list) or len(velocity) != 16:
        raise ValueError("Gemini fixed_pattern.velocity must be a 16-item list.")
    return {
        "tempo": float(max(60, min(180, float(payload["tempo"])))),
        "grid": "16th",
        "steps": [int(max(-24, min(24, float(step)))) for step in steps],
        "velocity": [float(max(0, min(1, float(v)))) for v in velocity],
    }


def generate_clip_description_with_retry(client, contents, attempts: int = 4):
    last_error = None
    for attempt in range(attempts):
        try:
            return client.models.generate_content(model=LLM_MODEL, contents=contents)
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini clip analysis failed after retries.") from last_error


def compact_log_line(line: str, state: dict[str, int]) -> str | None:
    stripped = line.rstrip()
    if not stripped:
        return None
    if stripped == "codex_prompt_begin":
        state["in_prompt"] = 1
        state["prompt_lines"] = 0
        return "codex_prompt_begin (full prompt saved to prompt_path artifact)"
    if stripped == "codex_prompt_end":
        skipped = state.get("prompt_lines", 0)
        state["in_prompt"] = 0
        state["prompt_lines"] = 0
        return f"codex_prompt_end ({skipped} prompt lines hidden from UI)"
    if state.get("in_prompt"):
        state["prompt_lines"] = state.get("prompt_lines", 0) + 1
        return None
    if stripped.startswith("codex_log "):
        return stripped[:1600] + (" ... [truncated]" if len(stripped) > 1600 else "")
    if "FutureWarning:" in stripped or stripped.startswith("warnings.warn(") or "/site-packages/" in stripped:
        return None
    if stripped.startswith("codex_prompt_hidden") or stripped.startswith("codex_prompt_path"):
        return None
    if any(pattern.search(stripped) for pattern in IMPORTANT_LOG_PATTERNS):
        return stripped[:1200] + (" ... [truncated]" if len(stripped) > 1200 else "")
    if len(stripped) > 600:
        return stripped[:600] + " ... [truncated]"
    return stripped


def run_media_url(path: Path) -> str | None:
    try:
        resolved = path.resolve()
        runs = RUNS.resolve()
    except FileNotFoundError:
        return None
    if runs not in resolved.parents:
        return None
    run_dir = resolved.parent
    return f"/media/runs/{run_dir.name}/{resolved.name}"


def parse_trace_event(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith("trace_file "):
        return None
    payload: dict[str, str | int | None] = {"type": "trace_file", "line": stripped}
    for key, value in re.findall(r"\b(agent|role|step)=([^ ]+)", stripped):
        payload[key] = int(value) if key == "step" else value
    if " path=" in stripped:
        payload["path"] = stripped.split(" path=", 1)[1]
    path = payload.get("path")
    if isinstance(path, str):
        payload["url"] = run_media_url(Path(path))
        payload["name"] = Path(path).name
    return payload


def parse_codex_file_event(line: str) -> dict | None:
    stripped = line.strip()
    if not (stripped.startswith("codex_request ") or stripped.startswith("codex_response ")):
        return None
    event_type = "codex_request" if stripped.startswith("codex_request ") else "codex_response"
    payload: dict[str, str | int | None] = {"type": event_type, "line": stripped}
    for key, value in re.findall(r"\b(agent|step)=([^ ]+)", stripped):
        payload[key] = int(value) if key == "step" else value
    if " path=" in stripped:
        payload["path"] = stripped.split(" path=", 1)[1]
    path = payload.get("path")
    if isinstance(path, str):
        payload["url"] = run_media_url(Path(path))
        payload["name"] = Path(path).name
    return payload


def parse_codex_log_event(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith("codex_log "):
        return None
    agent, step = agent_from_log_line(stripped)
    text = re.sub(r"^codex_log agent=[^ ]+\s*", "", stripped)
    return {"type": "codex_log", "agent": agent, "step": step, "line": text}


def agent_from_log_line(line: str) -> tuple[str | None, int | None]:
    agent = None
    step = None
    agent_match = re.search(r"\bagent=([^ ]+)", line)
    if agent_match:
        agent = agent_match.group(1)
    stage_match = re.search(r"\bagent_stage ([a-z_]+)(?: step=(\d+))?", line)
    if stage_match:
        agent = stage_match.group(1)
        if stage_match.group(2) is not None:
            step = int(stage_match.group(2))
    step_match = re.search(r"\bstep=(\d+)", line)
    if step_match:
        step = int(step_match.group(1))
    if agent == "producer" and step is not None:
        agent = f"producer_step_{step:02d}"
    if agent == "residual_critic" and step is not None:
        agent = f"residual_critic_step_{step:02d}"
    if agent == "loss" and step is not None:
        agent = f"loss_step_{step:02d}"
    return agent, step


def append_run_event(out_dir: Path, event: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with (out_dir / "events.jsonl").open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def append_agent_log(out_dir: Path, agent: str | None, line: str) -> None:
    if not agent:
        return
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent)
    with (logs_dir / f"{safe}.log").open("a") as fh:
        fh.write(line.rstrip() + "\n")


def write_run_state(out_dir: Path, status: str, returncode: int | None = None, pid: int | None = None) -> None:
    payload = {
        "run_id": out_dir.name,
        "status": status,
        "returncode": returncode,
        "pid": pid,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_state.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        manifest["status"] = status
        manifest["returncode"] = returncode
        manifest["updated_at"] = payload["updated_at"]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def enqueue_compacted(event_queue: queue.Queue[dict], compacted: str, out_dir: Path | None = None) -> None:
    trace_event = parse_trace_event(compacted)
    if trace_event is not None:
        if out_dir is not None:
            append_run_event(out_dir, {k: v for k, v in trace_event.items() if k != "type"} | {"type": "file_written"})
            append_agent_log(out_dir, str(trace_event.get("agent") or ""), compacted)
        event_queue.put(trace_event)
    elif (codex_file_event := parse_codex_file_event(compacted)) is not None:
        if out_dir is not None:
            append_run_event(out_dir, {k: v for k, v in codex_file_event.items() if k != "type"} | {"type": codex_file_event["type"]})
            append_agent_log(out_dir, str(codex_file_event.get("agent") or ""), compacted)
        event_queue.put(codex_file_event)
    elif (codex_log_event := parse_codex_log_event(compacted)) is not None:
        if out_dir is not None:
            append_run_event(out_dir, codex_log_event)
            append_agent_log(out_dir, str(codex_log_event.get("agent") or ""), codex_log_event["line"])
        event_queue.put(codex_log_event)
    elif out_dir is not None:
        agent, step = agent_from_log_line(compacted)
        append_run_event(out_dir, {"type": "log", "agent": agent, "step": step, "line": compacted})
        append_agent_log(out_dir, agent, compacted)
        event_queue.put({"type": "log", "line": compacted})


def start_run(reference: str, prompt: str, instrument_type: str, candidates: int, axis_trials: int, target_part: str = "", analysis: dict | None = None) -> str:
    reference_path = safe_reference_path(reference)
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    event_queue: queue.Queue[dict] = queue.Queue()
    analysis_path = out_dir / "analysis_input.json"
    if analysis is None:
        raise ValueError("Build requires AI clip analysis, including fixed_pattern. Run Analyze Clip first.")
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    cmd = [
        str(WORKSPACE / ".venv/bin/python"),
        str(ROOT / "reference_match.py"),
        "--reference",
        str(reference_path),
        "--prompt",
        prompt,
        "--output-dir",
        str(out_dir),
        "--candidates",
        str(candidates),
        "--axis-trials",
        str(axis_trials),
        "--instrument-type",
        instrument_type,
        "--analysis-json",
        str(analysis_path),
    ]
    if target_part:
        cmd.extend(["--target-part", target_part])
    jobs[run_id] = {
        "id": run_id,
        "status": "running",
        "queue": event_queue,
        "output_dir": str(out_dir),
        "cmd": cmd,
        "returncode": None,
    }

    def worker() -> None:
        env = os.environ.copy()
        load_secrets_into_env(env)
        raw_log_path = out_dir / "raw_subprocess.log"
        filter_state = {"in_prompt": 0, "prompt_lines": 0, "suppressed_codex": 0}
        recent_raw_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        jobs[run_id]["pid"] = process.pid
        assert process.stdout is not None
        with raw_log_path.open("w") as raw_log:
            for line in process.stdout:
                raw_log.write(line)
                raw_log.flush()
                recent_raw_lines.append(line.rstrip())
                recent_raw_lines = recent_raw_lines[-40:]
                compacted = compact_log_line(line, filter_state)
                if compacted is not None:
                    enqueue_compacted(event_queue, compacted, out_dir)
        returncode = process.wait()
        if filter_state.get("suppressed_codex"):
            pass
        jobs[run_id]["returncode"] = returncode
        jobs[run_id]["status"] = "completed" if returncode == 0 else "failed"
        if returncode != 0:
            event_queue.put({"type": "log", "line": "last raw subprocess lines before failure:"})
            for raw_line in recent_raw_lines:
                if raw_line:
                    event_queue.put({"type": "log", "line": raw_line[:1200] + (" ... [truncated]" if len(raw_line) > 1200 else "")})
        event_queue.put({"type": "done", "status": jobs[run_id]["status"], "returncode": returncode})

    threading.Thread(target=worker, daemon=True).start()
    return run_id


def start_reconstruction(reference: str, steps: int = 5, local_trials: int = 0, max_layers: int = 5, clip_start: float | None = None, song_id: str = "") -> str:
    reference_path = safe_reference_path(reference)
    run_id = time.strftime("%Y%m%d_%H%M%S_v1_") + uuid.uuid4().hex[:8]
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    event_queue: queue.Queue[dict] = queue.Queue()
    song = load_song(song_id) if song_id else None
    midi_path = song_asset_path(song, "midi") if song else None
    role_map_path = Path(song["_manifest"]) if song else None
    if midi_path is not None:
        cmd = [
            str(WORKSPACE / ".venv/bin/python"),
            str(ROOT / "midi_locked_patch.py"),
            "run",
            "--midi",
            str(midi_path),
            "--role-map",
            str(role_map_path),
            "--reference",
            str(reference_path),
            "--output-dir",
            str(out_dir),
            "--steps",
            str(max(1, min(10, steps))),
            "--seconds",
            "5",
        ]
        if clip_start is not None:
            cmd.extend(["--clip-start", f"{clip_start:.6f}"])
    else:
        cmd = [
            str(WORKSPACE / ".venv/bin/python"),
            str(ROOT / "reconstruct_match.py"),
            "--reference",
            str(reference_path),
            "--output-dir",
            str(out_dir),
            "--steps",
            str(max(1, min(10, steps))),
            "--local-trials",
            str(max(0, min(24, local_trials))),
            "--max-layers",
            str(max(1, min(6, max_layers))),
        ]
    reconstruction_jobs[run_id] = {
        "id": run_id,
        "status": "running",
        "queue": event_queue,
        "output_dir": str(out_dir),
        "cmd": cmd,
        "returncode": None,
    }
    (out_dir / "logs").mkdir(exist_ok=True)
    append_run_event(
        out_dir,
        {
            "type": "run_created",
            "run_id": run_id,
            "status": "running",
            "reference": str(reference_path),
            "workflow": "midi_locked_patch" if midi_path is not None else "freeform_reconstruction",
            "song_id": song_id,
            "midi": str(midi_path) if midi_path is not None else None,
            "clip_start": clip_start,
            "cmd": cmd,
        },
    )
    write_run_state(out_dir, "running")

    def worker() -> None:
        env = os.environ.copy()
        load_secrets_into_env(env)
        raw_log_path = out_dir / "raw_subprocess.log"
        filter_state = {"in_prompt": 0, "prompt_lines": 0, "suppressed_codex": 0}
        recent_raw_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        reconstruction_jobs[run_id]["pid"] = process.pid
        write_run_state(out_dir, "running", pid=process.pid)
        append_run_event(out_dir, {"type": "process_start", "run_id": run_id, "pid": process.pid})
        assert process.stdout is not None
        with raw_log_path.open("w") as raw_log:
            for line in process.stdout:
                raw_log.write(line)
                raw_log.flush()
                recent_raw_lines.append(line.rstrip())
                recent_raw_lines = recent_raw_lines[-40:]
                compacted = compact_log_line(line, filter_state)
                if compacted is not None:
                    enqueue_compacted(event_queue, compacted, out_dir)
        returncode = process.wait()
        if filter_state.get("suppressed_codex"):
            pass
        was_cancelled = reconstruction_jobs[run_id].get("status") == "cancelled"
        status = "cancelled" if was_cancelled else ("completed" if returncode == 0 else "failed")
        reconstruction_jobs[run_id]["returncode"] = returncode
        reconstruction_jobs[run_id]["status"] = status
        write_run_state(out_dir, status, returncode=returncode, pid=process.pid)
        append_run_event(out_dir, {"type": "process_done", "run_id": run_id, "status": status, "returncode": returncode})
        if returncode != 0 and not was_cancelled:
            event_queue.put({"type": "log", "line": "last raw subprocess lines before failure:"})
            for raw_line in recent_raw_lines:
                if raw_line:
                    event_queue.put({"type": "log", "line": raw_line[:1200] + (" ... [truncated]" if len(raw_line) > 1200 else "")})
        if not was_cancelled:
            event_queue.put({"type": "done", "status": status, "returncode": returncode})

    threading.Thread(target=worker, daemon=True).start()
    return run_id


def kill_reconstruction(run_id: str) -> dict:
    job = reconstruction_jobs.get(run_id)
    if not job:
        raise KeyError(run_id)
    out_dir = Path(job["output_dir"])
    if job.get("status") != "running":
        return {"id": run_id, "status": job.get("status"), "killed": False}
    pid = job.get("pid")
    if pid:
        try:
            os.kill(int(pid), 15)
        except ProcessLookupError:
            pass
    job["status"] = "cancelled"
    job["returncode"] = -15
    write_run_state(out_dir, "cancelled", returncode=-15, pid=pid)
    append_run_event(out_dir, {"type": "process_kill", "run_id": run_id, "status": "cancelled", "pid": pid})
    try:
        job["queue"].put({"type": "log", "line": "run killed by user"})
        job["queue"].put({"type": "done", "status": "cancelled", "returncode": -15})
    except Exception:
        pass
    return {"id": run_id, "status": "cancelled", "killed": True}


def disk_reconstruction_status(run_dir: Path, manifest: dict | None = None) -> str:
    if (run_dir / "reconstruction_report.json").exists():
        return "completed"
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        try:
            status = json.loads(state_path.read_text()).get("status")
            if status:
                return "interrupted" if status == "running" else status
        except (json.JSONDecodeError, OSError):
            pass
    if manifest is None:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                manifest = None
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        try:
            for line in reversed(events_path.read_text().splitlines()):
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") == "process_kill":
                    return "cancelled"
                if event.get("type") == "process_done":
                    return event.get("status") or "partial"
        except (json.JSONDecodeError, OSError):
            pass
    status = (manifest or {}).get("status")
    if status:
        return "interrupted" if status == "running" else status
    latest_mtime = latest_artifact_mtime(run_dir)
    if latest_mtime and time.time() - latest_mtime <= EXTERNAL_RUN_ACTIVE_WINDOW_SECONDS:
        return "running"
    return "partial"


def latest_artifact_mtime(run_dir: Path) -> float | None:
    try:
        mtimes = [path.stat().st_mtime for path in run_dir.rglob("*") if path.is_file()]
    except OSError:
        return None
    return max(mtimes) if mtimes else None


def read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def score_from_report(report: dict | None) -> float | None:
    if not report:
        return None
    candidates = [
        ((report.get("best_scores") or {}).get("final") if isinstance(report.get("best_scores"), dict) else None),
        ((report.get("scores") or {}).get("final") if isinstance(report.get("scores"), dict) else None),
        (((report.get("global_mix_diff") or {}).get("scores") or {}).get("final") if isinstance(report.get("global_mix_diff"), dict) else None),
    ]
    for value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            return numeric
    return None


def score_summary_from_artifacts(run_dir: Path, report: dict | None) -> dict:
    scored_steps = []
    final_score = score_from_report(report)
    if final_score is not None:
        scored_steps.append({"step": "final", "score": final_score, "name": "reconstruction_report.json"})
    for path in sorted(run_dir.glob("patch_report_step_*.json")):
        step_match = re.search(r"step_(\d+)", path.name)
        score = score_from_report(read_json_file(path))
        if score is not None:
            scored_steps.append({"step": int(step_match.group(1)) if step_match else None, "score": score, "name": path.name})
    initial_score = score_from_report(read_json_file(run_dir / "loss_report_step_initial.json"))
    if initial_score is not None:
        scored_steps.append({"step": "initial", "score": initial_score, "name": "loss_report_step_initial.json"})
    if not scored_steps:
        return {"final_score": None, "best_step": None, "best_name": None}
    best = max(scored_steps, key=lambda item: item["score"])
    return {"final_score": best["score"], "best_step": best["step"], "best_name": best["name"]}


def reconstruction_run_summary(run_dir: Path, report: dict | None, manifest: dict | None = None) -> dict:
    history = (report or {}).get("history") or []
    score_summary = score_summary_from_artifacts(run_dir, report)
    patch_reports = sorted(run_dir.glob("patch_report_step_*.json"))
    patch_ops = sorted(run_dir.glob("patch_ops_step_*.json"))
    accepted_sessions = sorted(run_dir.glob("patch_session_step_*.json"))
    stage_count = len(history) or max(len(patch_reports), len(patch_ops))
    final_score = score_summary["final_score"]
    if report:
        overall_mix = (((report.get("analysis") or {}).get("global") or {}).get("overall_mix") or "").strip()
    else:
        overall_mix = ""
    if not overall_mix:
        if final_score is not None:
            best_step = score_summary["best_step"]
            if best_step == "initial":
                overall_mix = f"Initial score {final_score:.3f}; waiting for patch iterations"
            elif best_step == "final":
                overall_mix = f"Final score {final_score:.3f}"
            else:
                overall_mix = f"Best patch score {final_score:.3f} at step {best_step}"
        elif patch_ops or (run_dir / "patch_session_current.json").exists():
            overall_mix = "Patch goal run in progress; waiting for first score"
        else:
            overall_mix = (manifest or {}).get("reference", "") or "Partial reconstruction run"
    return {
        "overall_mix": overall_mix,
        "final_score": final_score,
        "mel_score": ((report or {}).get("best_scores") or {}).get("mel_spectrogram"),
        "envelope_score": ((report or {}).get("best_scores") or {}).get("envelope"),
        "stage_count": stage_count,
        "accepted_layers": len(((report or {}).get("analysis") or {}).get("layers") or []) or len(accepted_sessions),
    }


def read_text_limited(path: Path, max_chars: int = 200_000) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def artifact_record(run_dir: Path, path: Path, inline_max_chars: int = 200_000) -> dict:
    rel = str(path.relative_to(run_dir))
    record = {
        "name": rel,
        "size": path.stat().st_size,
        "url": f"/media/runs/{run_dir.name}/{'/'.join(quote(part) for part in path.relative_to(run_dir).parts)}",
    }
    if inline_max_chars > 0 and path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".log", ".sh", ".csv", ".tsv"}:
        record["content"] = read_text_limited(path, inline_max_chars)
        record["truncated"] = path.stat().st_size > inline_max_chars
    return record


def run_log_bundle(run_id: str, inline_artifacts: bool = True) -> dict:
    run_dir = (RUNS / run_id).resolve()
    if not run_dir.is_dir():
        raise KeyError(run_id)
    if RUNS.resolve() not in run_dir.parents and run_dir != RUNS.resolve():
        raise ValueError("Run path escapes runs directory.")
    manifest = None
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = None
    events = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "raw", "line": line})
    agent_logs = []
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.log")):
            agent_logs.append({"name": path.name, "content": read_text_limited(path), "size": path.stat().st_size})
    artifacts = [artifact_record(run_dir, path, inline_max_chars=80_000 if inline_artifacts else 0) for path in sorted(run_dir.rglob("*")) if path.is_file()]
    return {
        "run_id": run_id,
        "run_path": str(run_dir),
        "status": disk_reconstruction_status(run_dir, manifest),
        "manifest": manifest,
        "events": events,
        "raw_subprocess_log": read_text_limited(run_dir / "raw_subprocess.log"),
        "agent_logs": agent_logs,
        "artifacts": artifacts,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/references":
                files = [
                    {"name": file.name, "size": file.stat().st_size, "url": f"/media/references/{file.name}"}
                    for file in sorted(REFERENCES.glob("*"))
                    if file.is_file() and file.suffix.lower() in {".wav", ".mp3", ".aiff", ".aif", ".m4a"}
                ]
                json_response(self, {"files": files})
                return
            if path == "/api/songs":
                songs = []
                errors = []
                for song_dir in sorted(SONGS.glob("*")):
                    if not song_dir.is_dir() or not (song_dir / "song.json").exists():
                        continue
                    try:
                        song = load_song(song_dir.name)
                        audio = song_asset_path(song, "audio")
                        midi = song_asset_path(song, "midi")
                        arrangement = song_asset_path(song, "arrangement") if song.get("arrangement") else None
                    except Exception as exc:
                        errors.append({"id": song_dir.name, "path": str(song_dir), "error": str(exc)})
                        continue
                    songs.append(
                        {
                            "id": song["id"],
                            "title": song.get("title", song["id"]),
                            "artist": song.get("artist", ""),
                            "label": f"{song.get('artist', '').strip()} - {song.get('title', song['id']).strip()}".strip(" -"),
                            "audio": song.get("audio"),
                            "midi": song.get("midi"),
                            "arrangement": song.get("arrangement"),
                            "audio_size": audio.stat().st_size,
                            "midi_size": midi.stat().st_size,
                            "arrangement_size": arrangement.stat().st_size if arrangement is not None else None,
                            "url": f"/media/songs/{quote(song['id'])}/audio",
                        }
                    )
                json_response(self, {"songs": songs, "errors": errors})
                return
            if path == "/api/runs":
                json_response(self, {"runs": list_runs()})
                return
            if path == "/api/reconstruction-runs":
                json_response(self, {"runs": list_reconstruction_runs()})
                return
            if path.startswith("/api/analysis/") and path.endswith("/events"):
                analysis_id = path.split("/")[-2]
                self.stream_analysis_events(analysis_id)
                return
            if path.startswith("/api/clips/") and path.endswith("/events"):
                clip_id = path.split("/")[-2]
                self.stream_clip_events(clip_id)
                return
            if path.startswith("/api/analysis/"):
                analysis_id = path.split("/")[-1]
                job = analysis_jobs[analysis_id]
                json_response(
                    self,
                    {
                        "id": analysis_id,
                        "status": job["status"],
                        "result": job["result"],
                        "error": job["error"],
                    },
                )
                return
            if path.startswith("/api/jobs/") and path.endswith("/events"):
                run_id = path.split("/")[-2]
                self.stream_events(run_id)
                return
            if path.startswith("/api/jobs/"):
                run_id = path.split("/")[-1]
                job = jobs[run_id]
                json_response(
                    self,
                    {
                        "id": run_id,
                        "status": job["status"],
                        "returncode": job["returncode"],
                        "output_dir": job["output_dir"],
                        "artifacts": list_artifacts(Path(job["output_dir"])),
                    },
                )
                return
            if path.startswith("/api/reconstructions/") and path.endswith("/events"):
                run_id = path.split("/")[-2]
                self.stream_reconstruction_events(run_id)
                return
            log_match = re.match(r"^/api/reconstructions/([^/]+)/(?:logs|harness)$", path)
            if log_match:
                json_response(self, run_log_bundle(log_match.group(1)))
                return
            if path.startswith("/api/reconstructions/"):
                run_id = path.split("/")[-1]
                job = reconstruction_jobs.get(run_id)
                if not job:
                    run_dir = (RUNS / run_id).resolve()
                    if not run_dir.is_dir():
                        raise KeyError(run_id)
                    json_response(
                        self,
                        {
                            "id": run_id,
                            "status": disk_reconstruction_status(run_dir),
                            "returncode": None,
                            "output_dir": str(run_dir),
                            "artifacts": list_artifacts(run_dir),
                        },
                    )
                    return
                json_response(
                    self,
                    {
                        "id": run_id,
                        "status": job["status"],
                        "returncode": job["returncode"],
                        "output_dir": job["output_dir"],
                        "artifacts": list_artifacts(Path(job["output_dir"])),
                    },
                )
                return
            if path.startswith("/media/references/"):
                self.serve_file(safe_reference_path(unquote(path.removeprefix("/media/references/"))))
                return
            song_media_match = re.match(r"^/media/songs/([^/]+)/(audio|midi|arrangement)$", path)
            if song_media_match:
                song = load_song(unquote(song_media_match.group(1)))
                key = song_media_match.group(2)
                self.serve_file(song_asset_path(song, key))
                return
            if path.startswith("/media/runs/"):
                parts = path.removeprefix("/media/runs/").split("/", 1)
                if len(parts) != 2:
                    raise FileNotFoundError(path)
                run_id, rel = parts
                run_dir = (RUNS / run_id).resolve()
                target = (run_dir / unquote(rel)).resolve()
                if run_dir not in target.parents and target != run_dir:
                    raise ValueError("Run media path escapes run directory.")
                self.serve_file(target)
                return
            route_run_id = path.strip("/").split("/", 1)[0]
            is_run_route = (RUNS / route_run_id).is_dir() if route_run_id else False
            if path in {"/v1", "/v1.html", "/v1_reconstruct", "/v1_reconstruct.html"} or path.startswith("/v1/") or is_run_route:
                target = UI / "v1.html"
            else:
                target = UI / ("index.html" if path == "/" else path.lstrip("/"))
            self.serve_file(target)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/clip":
                payload = read_json(self)
                analysis_id = start_clip_analysis(
                    reference=payload["reference"],
                    start=float(payload["start"]),
                    duration=float(payload.get("duration", 5)),
                    target_part=str(payload.get("target_part", "")),
                )
                json_response(self, {"analysis_id": analysis_id})
                return
            if parsed.path == "/api/extract":
                payload = read_json(self)
                clip_id = start_clip_extract(
                    reference=str(payload.get("reference", "")),
                    start=float(payload["start"]),
                    duration=float(payload.get("duration", 5)),
                    song_id=str(payload.get("song_id", "")),
                )
                json_response(self, {"clip_id": clip_id})
                return
            if parsed.path == "/api/run":
                payload = read_json(self)
                run_id = start_run(
                    reference=payload["clip"],
                    prompt=payload["prompt"],
                    instrument_type=payload["instrument_type"],
                    candidates=int(payload.get("candidates", 4)),
                    axis_trials=int(payload.get("axis_trials", 1)),
                    target_part=str(payload.get("target_part", "")),
                    analysis=payload.get("analysis"),
                )
                json_response(self, {"run_id": run_id})
                return
            if parsed.path == "/api/reconstruct":
                payload = read_json(self)
                run_id = start_reconstruction(
                    reference=payload["clip"],
                    steps=int(payload.get("steps", 5)),
                    local_trials=int(payload.get("local_trials", 0)),
                    max_layers=int(payload.get("max_layers", 5)),
                    clip_start=float(payload["clip_start"]) if payload.get("clip_start") is not None else None,
                    song_id=str(payload.get("song_id", "")),
                )
                json_response(self, {"run_id": run_id})
                return
            kill_match = re.match(r"^/api/reconstructions/([^/]+)/kill$", parsed.path)
            if kill_match:
                json_response(self, kill_reconstruction(kill_match.group(1)))
                return
            json_response(self, {"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type(path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream_events(self, run_id: str) -> None:
        job = jobs[run_id]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue[dict] = job["queue"]
        while True:
            try:
                event = q.get(timeout=15)
            except queue.Empty:
                event = {"type": "heartbeat"}
            data = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()
            if event.get("type") == "done":
                break

    def stream_analysis_events(self, analysis_id: str) -> None:
        job = analysis_jobs[analysis_id]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue[dict] = job["queue"]
        while True:
            try:
                event = q.get(timeout=5)
            except queue.Empty:
                event = {"type": "heartbeat"}
            data = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()
            if event.get("type") == "done":
                break

    def stream_clip_events(self, clip_id: str) -> None:
        job = clip_jobs[clip_id]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue[dict] = job["queue"]
        while True:
            try:
                event = q.get(timeout=5)
            except queue.Empty:
                event = {"type": "heartbeat"}
            data = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()
            if event.get("type") == "done":
                break

    def stream_reconstruction_events(self, run_id: str) -> None:
        job = reconstruction_jobs.get(run_id)
        if not job:
            self.stream_external_reconstruction_events(run_id)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue[dict] = job["queue"]
        while True:
            try:
                event = q.get(timeout=15)
            except queue.Empty:
                event = {"type": "heartbeat"}
            data = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()
            if event.get("type") == "done":
                break

    def stream_external_reconstruction_events(self, run_id: str) -> None:
        run_dir = (RUNS / run_id).resolve()
        if not run_dir.is_dir():
            raise KeyError(run_id)
        if RUNS.resolve() not in run_dir.parents and run_dir != RUNS.resolve():
            raise ValueError("Run path escapes runs directory.")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        seen: dict[str, tuple[int, float]] = {}
        self.write_sse({"type": "log", "line": "Watching external Codex goal run artifacts on disk."})
        while True:
            try:
                files = [path for path in sorted(run_dir.rglob("*")) if path.is_file()]
                changed = []
                for path in files:
                    stat = path.stat()
                    rel = str(path.relative_to(run_dir))
                    stamp = (stat.st_size, stat.st_mtime)
                    if seen.get(rel) != stamp:
                        seen[rel] = stamp
                        changed.append(path)
                for path in changed:
                    event = trace_event_for_artifact(run_dir, path)
                    if event:
                        self.write_sse(event)
                status = disk_reconstruction_status(run_dir)
                if status == "completed":
                    self.write_sse({"type": "done", "status": "completed", "returncode": 0})
                    break
                self.write_sse({"type": "heartbeat"})
                time.sleep(2)
            except (BrokenPipeError, ConnectionResetError):
                break

    def write_sse(self, event: dict) -> None:
        data = f"data: {json.dumps(event)}\n\n".encode()
        self.wfile.write(data)
        self.wfile.flush()


def list_artifacts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        {
            "name": str(file.relative_to(path)),
            "size": file.stat().st_size,
            "url": f"/media/runs/{path.name}/{'/'.join(quote(part) for part in file.relative_to(path).parts)}",
        }
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]


def trace_event_for_artifact(run_dir: Path, path: Path) -> dict | None:
    name = str(path.relative_to(run_dir))
    if not path.is_file():
        return None
    base = path.name
    role = "file"
    agent = "producer"
    step = None
    step_match = re.search(r"step_(\d+)", base)
    if step_match:
        step = int(step_match.group(1))
    if base.startswith("codex_"):
        role = "prompt" if "_prompt" in base else "answer"
        agent = re.sub(r"^codex_|_(prompt|answer)\.txt$", "", base)
    elif base.startswith("critic_brief_step_"):
        role = "recommendation"
        agent = f"residual_critic_step_{step:02d}" if step is not None else "residual_critic"
    elif base.startswith("patch_report_step_"):
        role = "audio_diff"
        agent = f"loss_step_{step:02d}" if step is not None else "loss"
    elif base.startswith("patch_render_step_"):
        role = "winner_render"
        agent = f"loss_step_{step:02d}" if step is not None else "loss"
    elif base.startswith("patch_session_step_"):
        role = "accepted_session"
        agent = f"producer_step_{step:02d}" if step is not None else "producer"
    elif base.startswith("patch_ops_step_") or base.startswith("patch_ops_applied_step_"):
        role = "session"
        agent = f"producer_step_{step:02d}" if step is not None else "producer"
    elif base in {"patch_session_current.json", "arrangement.json", "full_arrangement.json"}:
        role = "layer_analysis"
        agent = "baseline"
    elif base == "source_clip.wav":
        role = "source_clip"
        agent = "baseline"
    elif base == "current_render_step_initial.wav":
        role = "winner_render"
        agent = "baseline"
    elif base == "loss_report_step_initial.json":
        role = "audio_diff"
        agent = "baseline"
    elif base == "final_reconstruction.wav":
        role = "render"
        agent = "loss"
    elif base.endswith(".wav"):
        role = "render"
    elif base == "reconstruction_report.json":
        role = "audio_diff"
        agent = "loss"
    return {
        "type": "trace_file",
        "agent": agent,
        "role": role,
        "step": step,
        "name": name,
        "path": f"/ui_runs/{run_dir.name}/{name}",
        "url": f"/media/runs/{run_dir.name}/{'/'.join(quote(part) for part in path.relative_to(run_dir).parts)}",
    }


def list_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(RUNS.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        report_path = run_dir / "match_report.json"
        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                report = None
        runs.append(
            {
                "id": run_dir.name,
                "status": "completed" if report_path.exists() else "unknown",
                "prompt": (report or {}).get("prompt", ""),
                "instrument_type": ((report or {}).get("final_recipe") or {}).get("instrument_type", ""),
                "best_axis": (((report or {}).get("best_candidates") or [{}])[0]).get("axis", ""),
                "best_score": ((((report or {}).get("best_candidates") or [{}])[0]).get("scores") or {}).get("final"),
                "artifacts": list_artifacts(run_dir),
            }
        )
    return runs


def list_reconstruction_runs() -> list[dict]:
    runs = []
    seen = set()
    for run_id, job in sorted(reconstruction_jobs.items(), reverse=True):
        if job.get("status") != "running":
            continue
        output_dir = Path(job["output_dir"])
        artifacts = list_artifacts(output_dir) if output_dir.exists() else []
        summary = reconstruction_run_summary(output_dir, None) if output_dir.exists() else {}
        seen.add(run_id)
        runs.append(
            {
                "id": run_id,
                "status": "running",
                "overall_mix": summary.get("overall_mix") or "Reconstruction currently running",
                "final_score": summary.get("final_score"),
                "mel_score": summary.get("mel_score"),
                "envelope_score": summary.get("envelope_score"),
                "stage_count": summary.get("stage_count") or 0,
                "accepted_layers": summary.get("accepted_layers") or 0,
                "artifacts": artifacts,
            }
        )
    for run_dir in sorted(RUNS.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        if run_dir.name in seen:
            continue
        report_path = run_dir / "reconstruction_report.json"
        running_job = reconstruction_jobs.get(run_dir.name)
        if running_job and running_job.get("status") == "running":
            continue
        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                report = None
        manifest = None
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                manifest = None
        artifacts = list_artifacts(run_dir)
        status = "running" if running_job and running_job.get("status") == "running" else disk_reconstruction_status(run_dir, manifest)
        summary = reconstruction_run_summary(run_dir, report, manifest)
        runs.append(
            {
                "id": run_dir.name,
                "status": status,
                "overall_mix": summary["overall_mix"],
                "final_score": summary["final_score"],
                "mel_score": summary["mel_score"],
                "envelope_score": summary["envelope_score"],
                "stage_count": summary["stage_count"],
                "accepted_layers": summary["accepted_layers"],
                "artifacts": artifacts,
            }
        )
    return runs


def main() -> None:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    SONGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Text2FX UI running at http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
