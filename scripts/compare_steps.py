#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scripts/compare_steps.py <run_id>", file=sys.stderr)
        return 2
    report_path = RUNS / sys.argv[1] / "reconstruction_report.json"
    if not report_path.exists():
        print(f"report not found: {report_path}", file=sys.stderr)
        return 1
    history = json.loads(report_path.read_text()).get("history") or []
    previous = None
    for item in history:
        score = float((item.get("scores") or {}).get("final") or 0)
        delta = 0.0 if previous is None else score - previous
        previous = score
        print(f"step={item.get('step')} score={score:.4f} delta={delta:+.4f} winner={item.get('winner')} accepted={item.get('accepted')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
