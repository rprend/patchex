#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "text2fx_gemini" / "ui_runs"


def follow(path: Path) -> None:
    with path.open() as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--agent")
    parser.add_argument("--no-follow", action="store_true")
    args = parser.parse_args()
    run = RUNS / args.run_id
    path = run / "events.jsonl"
    if args.agent:
        path = run / "logs" / f"{args.agent}.log"
    if not path.exists():
        print(f"log not found: {path}")
        return 1
    if args.no_follow:
        print(path.read_text(), end="")
    else:
        follow(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
