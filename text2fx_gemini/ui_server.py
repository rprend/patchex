#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from reference_match import analyze_reference, load_audio, runtime
from text2fx import LLM_MODEL, extract_json_object


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
REFERENCES = ROOT / "references"
UI = ROOT / "ui"
RUNS = ROOT / "ui_runs"
FFMPEG = shutil.which("ffmpeg")

jobs: dict[str, dict] = {}
analysis_jobs: dict[str, dict] = {}


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
            "1",
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
  "prompt": "one compact target prompt summarizing the axes for recipe generation"
}}

Use the local DSP features only as supporting evidence; the audio clip is authoritative.
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
    return {"features": local["features"], "axes": payload["axes"], "instrument_type": payload["instrument_type"], "prompt": payload["prompt"], "target_part": target_part, "analysis_source": "gemini_audio"}


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


def start_run(reference: str, prompt: str, instrument_type: str, candidates: int, axis_trials: int) -> str:
    reference_path = safe_reference_path(reference)
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    event_queue: queue.Queue[dict] = queue.Queue()
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
    ]
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
        for line in process.stdout:
            event_queue.put({"type": "log", "line": line.rstrip()})
        returncode = process.wait()
        jobs[run_id]["returncode"] = returncode
        jobs[run_id]["status"] = "completed" if returncode == 0 else "failed"
        event_queue.put({"type": "done", "status": jobs[run_id]["status"], "returncode": returncode})

    threading.Thread(target=worker, daemon=True).start()
    return run_id


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
            if path == "/api/runs":
                json_response(self, {"runs": list_runs()})
                return
            if path.startswith("/api/analysis/") and path.endswith("/events"):
                analysis_id = path.split("/")[-2]
                self.stream_analysis_events(analysis_id)
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
            if path.startswith("/media/references/"):
                self.serve_file(safe_reference_path(unquote(path.removeprefix("/media/references/"))))
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
            if parsed.path == "/api/run":
                payload = read_json(self)
                run_id = start_run(
                    reference=payload["clip"],
                    prompt=payload["prompt"],
                    instrument_type=payload["instrument_type"],
                    candidates=int(payload.get("candidates", 4)),
                    axis_trials=int(payload.get("axis_trials", 1)),
                )
                json_response(self, {"run_id": run_id})
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


def list_artifacts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        {"name": file.name, "size": file.stat().st_size, "url": f"/media/runs/{path.name}/{file.name}"}
        for file in sorted(path.glob("*"))
        if file.is_file()
    ]


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


def main() -> None:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Text2FX UI running at http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
