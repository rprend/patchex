#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"
CODEX_PATH = "/Applications/Codex.app/Contents/Resources/codex"
DEFAULT_MAX_CHARS = 120_000

REPORT_NAMES = (
    "reconstruction_report.json",
    "match_report.json",
    "history.json",
    "run_summary.md",
    "run_manifest.json",
)

TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log", ".sh"}
IMPORTANT_PREFIXES = (
    "audio_diff",
    "producer_audio_diff",
    "harness_improver",
    "recommendation",
    "history_item",
    "source_profile",
    "beat_grid",
    "layer_analysis",
    "pattern_constraints",
    "codex_producer",
    "codex_residual_critic",
    "codex_synthesis",
)


def resolve_run(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    run = RUNS / value
    if run.exists():
        return run.resolve()
    raise FileNotFoundError(f"Run not found as path or ui_runs id: {value}")


def read_text_limited(path: Path, max_chars: int) -> str:
    text = path.read_text(errors="replace")
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n\n... truncated {len(text) - max_chars} chars from {path.name} ...\n\n" + text[-tail:]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        return {"_error": f"Could not parse JSON: {exc}", "_path": str(path)}


def summarize_scores(report: dict[str, Any]) -> dict[str, Any]:
    if "history" in report:
        history = []
        for item in report.get("history") or []:
            scores = item.get("scores") or item.get("best_scores") or {}
            history.append(
                {
                    "step": item.get("step"),
                    "winner": item.get("winner"),
                    "accepted": item.get("accepted"),
                    "final": scores.get("final"),
                    "weakest": (item.get("residual") or {}).get("missing", [])[:4],
                }
            )
        return {"kind": "reconstruction_report", "best_scores": report.get("best_scores"), "history": history}
    if "best_candidates" in report:
        return {
            "kind": "match_report",
            "best_candidates": [
                {
                    "index": item.get("index"),
                    "axis": item.get("axis"),
                    "loss": item.get("loss"),
                    "final": (item.get("scores") or {}).get("final"),
                    "weakest": (item.get("residual") or {}).get("missing", [])[:4],
                    "harness": item.get("harness_improvement", {}),
                }
                for item in (report.get("best_candidates") or [])[:8]
            ],
        }
    return {"kind": "unknown_report", "keys": sorted(report.keys())[:30]}


def collect_context(run: Path, max_chars: int) -> dict[str, Any]:
    reports = {}
    for name in REPORT_NAMES:
        path = run / name
        if path.exists():
            if path.suffix == ".json":
                data = load_json(path)
                reports[name] = {"summary": summarize_scores(data) if isinstance(data, dict) else None, "data": data}
            else:
                reports[name] = {"text": read_text_limited(path, max_chars // 8)}

    artifacts = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in REPORT_NAMES:
            continue
        if not (path.name.startswith(IMPORTANT_PREFIXES) or path.name == "raw_subprocess.log" or path.name == "events.jsonl"):
            continue
        artifacts.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "content": read_text_limited(path, max(2_000, max_chars // 20)),
            }
        )

    logs = []
    logs_dir = run / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.log")):
            logs.append({"name": f"logs/{path.name}", "size": path.stat().st_size, "content": read_text_limited(path, max(2_000, max_chars // 20))})

    return {
        "run_id": run.name,
        "run_path": str(run),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
        "artifacts": artifacts,
        "logs": logs,
    }


def build_prompt(context: dict[str, Any]) -> str:
    return (
        "You are the Harness Improver for an audio-agent architecture.\n"
        "Inspect this completed run's outputs, loss reports, artifacts, and logs. Recommend concrete improvements to the harness so future agent loops get closer to the desired result.\n\n"
        "Scope of allowed recommendations:\n"
        "1. Loss function changes: metrics, weights, gates, diagnostics, or residual messages.\n"
        "2. Graph changes: add, delete, split, merge, or reorder nodes in the agent graph.\n"
        "3. Prompt changes: update producer, critic, analyzer, synthesis, or harness prompts.\n\n"
        "Return markdown with these sections:\n"
        "- Run Diagnosis\n"
        "- Loss Function Improvements\n"
        "- Graph/Node Improvements\n"
        "- Prompt Improvements\n"
        "- Priority Patch Plan\n"
        "- Evidence\n\n"
        "Be specific. Cite artifact names and score/log details from the context. Prefer changes that are implementable in this repository.\n\n"
        f"RUN CONTEXT JSON:\n{json.dumps(context, indent=2)}\n"
    )


def run_codex(prompt: str, answer_path: Path, timeout: int) -> None:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    process = subprocess.run(
        [
            CODEX_PATH,
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(answer_path),
            "-C",
            str(ROOT),
            "-",
        ],
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Codex failed with return code {process.returncode}:\n{process.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Codex harness-improver review against a completed UI run.")
    parser.add_argument("run", help="Run id under text2fx_gemini/ui_runs or a direct run directory path.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Approximate character budget for collected text context.")
    parser.add_argument("--timeout", type=int, default=180, help="Codex subprocess timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Write the prompt/context but do not call Codex.")
    args = parser.parse_args()

    run = resolve_run(args.run)
    if not run.is_dir():
        print(f"Run is not a directory: {run}", file=sys.stderr)
        return 1

    context = collect_context(run, args.max_chars)
    prompt = build_prompt(context)
    prompt_path = run / "harness_improver_prompt.md"
    context_path = run / "harness_improver_context.json"
    answer_path = run / "harness_improver_recommendations.md"
    context_path.write_text(json.dumps(context, indent=2) + "\n")
    prompt_path.write_text(prompt)

    if args.dry_run:
        print(f"wrote {context_path}")
        print(f"wrote {prompt_path}")
        print("dry run: skipped Codex")
        return 0

    run_codex(prompt, answer_path, args.timeout)
    print(f"wrote {context_path}")
    print(f"wrote {prompt_path}")
    print(f"wrote {answer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
