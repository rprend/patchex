#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log", ".sh", ".csv", ".tsv"}


def resolve_run(run_id: str) -> Path:
    path = Path(run_id).expanduser()
    if path.exists():
        return path.resolve()
    run = RUNS / run_id
    if run.exists():
        return run.resolve()
    raise FileNotFoundError(f"run not found: {run_id}")


def read_text_limited(path: Path, max_chars: int) -> str:
    text = path.read_text(errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... truncated {len(text) - max_chars} chars from {path.name} ...\n\n" + text[-half:]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def infer_status(run: Path, manifest: dict | None) -> str:
    if (run / "reconstruction_report.json").exists():
        return "completed"
    state = read_json(run / "run_state.json")
    if state and state.get("status"):
        return "interrupted" if state["status"] == "running" else str(state["status"])
    events_path = run / "events.jsonl"
    if events_path.exists():
        for line in reversed(events_path.read_text(errors="replace").splitlines()):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "process_kill":
                return "cancelled"
            if event.get("type") == "process_done":
                return str(event.get("status") or "partial")
    status = (manifest or {}).get("status")
    return "interrupted" if status == "running" else str(status or "partial")


def collect_events(run: Path) -> list[dict]:
    events_path = run / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"type": "raw", "line": line})
    return events


def collect_artifacts(run: Path, max_chars: int) -> list[dict]:
    artifacts = []
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run))
        item = {
            "name": rel,
            "size": path.stat().st_size,
            "url": f"/media/runs/{run.name}/{'/'.join(quote(part) for part in path.relative_to(run).parts)}",
        }
        if path.suffix.lower() in TEXT_SUFFIXES:
            item["content"] = read_text_limited(path, max_chars)
            item["truncated"] = path.stat().st_size > max_chars
        artifacts.append(item)
    return artifacts


def build_bundle(run: Path, max_chars: int) -> dict:
    manifest = read_json(run / "run_manifest.json")
    logs = []
    logs_dir = run / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.log")):
            logs.append({"name": path.name, "size": path.stat().st_size, "content": read_text_limited(path, max_chars)})
    raw_path = run / "raw_subprocess.log"
    return {
        "run_id": run.name,
        "run_path": str(run),
        "status": infer_status(run, manifest),
        "manifest": manifest,
        "events": collect_events(run),
        "raw_subprocess_log": read_text_limited(raw_path, max_chars) if raw_path.exists() else "",
        "agent_logs": logs,
        "artifacts": collect_artifacts(run, max_chars),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a complete Patchex run harness bundle by run id.")
    parser.add_argument("run_id", help="Run id under text2fx_gemini/ui_runs or a direct run directory path.")
    parser.add_argument("--max-chars", type=int, default=200_000, help="Max chars to inline per text artifact.")
    args = parser.parse_args()
    try:
        bundle = build_bundle(resolve_run(args.run_id), args.max_chars)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
