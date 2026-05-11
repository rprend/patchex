#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOAL_FILE="$ROOT/prompts/patchex_first5_score_over_08.goal.md"
CODEX_BIN="${CODEX_BIN:-codex}"

if [[ ! -f "$GOAL_FILE" ]]; then
  echo "Goal file not found: $GOAL_FILE" >&2
  exit 1
fi

echo
echo "Goal prompt:"
echo "------------"
cat "$GOAL_FILE"
echo "------------"
echo
echo "Starting Codex CLI in:"
echo "  $ROOT"
echo
echo "Paste the /goal prompt above into the Codex prompt if it is not already submitted."
echo

exec "$CODEX_BIN" -C "$ROOT"
