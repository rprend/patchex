#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scripts/summarize_run.py <run_id>", file=sys.stderr)
        return 2
    run = RUNS / sys.argv[1]
    report_path = run / "reconstruction_report.json"
    if not report_path.exists():
        summary = run / "run_summary.md"
        print(summary.read_text() if summary.exists() else f"no summary/report for {run.name}")
        return 0
    report = json.loads(report_path.read_text())
    print(f"run: {run.name}")
    print(f"final: {float((report.get('best_scores') or {}).get('final') or 0):.4f}")
    print("iterations:")
    for item in report.get("history") or []:
        score = float((item.get("scores") or {}).get("final") or 0)
        best = float((item.get("best_scores") or {}).get("final") or 0)
        print(f"- step {item.get('step')}: winner={item.get('winner')} accepted={item.get('accepted')} score={score:.4f} best={best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
