#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scripts/show_run.py <run_id>", file=sys.stderr)
        return 2
    run = RUNS / sys.argv[1]
    manifest = run / "run_manifest.json"
    summary = run / "run_summary.md"
    if not run.exists():
        print(f"run not found: {run}", file=sys.stderr)
        return 1
    if manifest.exists():
        data = json.loads(manifest.read_text())
        print(json.dumps(data, indent=2))
    if summary.exists():
        print("\n" + summary.read_text())
    print("\nArtifacts:")
    for path in sorted(run.iterdir()):
        if path.is_file():
            print(f"- {path.name}")
    logs = run / "logs"
    if logs.exists():
        print("\nAgent logs:")
        for path in sorted(logs.glob("*.log")):
            print(f"- {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
