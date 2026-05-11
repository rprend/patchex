/goal Update the Patchex synth patches, effects, modulation, mix, and scoring harness usage until the first five seconds of the canonical song reconstruction reaches best_scores.final >= 0.8.

Repository and target:
- Work in /Users/ryanprendergast/Documents/Zenobia Pay/ableton-tracks.
- Follow AGENTS.md exactly, including secret handling.
- Use the canonical song workspace text2fx_gemini/songs/between_the_buttons/.
- Optimize the first five seconds only: clip_start 0.0, seconds 5.
- Use the MIDI-locked workflow so the musical arrangement stays fixed and only synth patches, effects, modulation, automation, and mix settings change.
- The success score is the loss-function accuracy stored as best_scores.final in reconstruction_report.json. Stop only after best_scores.final >= 0.8.

Run/eval loop:
1. Start by reading AGENTS.md, text2fx_gemini/README.md, scripts/summarize_run.py, scripts/inspect_run.py, and the run loop in text2fx_gemini/midi_locked_patch.py.
2. Run a baseline reconstruction/eval for the first five seconds if no current suitable run exists.
3. After every run or iteration, inspect the full run harness bundle with scripts/inspect_run.py <run_id> or the run directory, not just the UI shell.
4. Read reconstruction_report.json, patch_report_step_*.json, history_item_step_*.json, critic_brief_step_*.md, patch_ops_step_*.json, and the rendered WAV artifacts that explain the current score.
5. Identify the largest bottleneck from the score components, weakest tracks, sustain/envelope diagnostics, global mix diagnostics, and artifacts.
6. Make one focused change per iteration. Prefer changes that the existing patch operation system can apply and score cleanly: synth source, envelope, filter movement, gain automation, modulation, effects, stereo, saturation, EQ, returns, or master settings.
7. Re-run the reconstruction/scoring loop for the first five seconds.
8. Append or update a running log in the active run directory recording:
   - current best score
   - latest candidate score
   - accepted/rejected result
   - what changed
   - what improved or regressed
   - next planned bottleneck
9. Continue until best_scores.final >= 0.8.

Commands and constraints:
- Use ~/.codex/scripts/ask-secret.sh for any missing secret; never ask the user to paste secrets.
- If GEMINI_API_KEY is needed, load it from ~/.codex/secrets.env using the existing project convention.
- Do not change MIDI notes, note timing, velocities, track ordering, or composition structure.
- Preserve the fixed first-five-second arrangement.
- Do not stop just because a fixed --steps run completes below 0.8. Start another focused iteration or modify the local harness to support a target-score loop if that is the most reliable path.
- If you modify repository files, run relevant tests/checks, then commit and push.

Suggested starting command if you need a fresh run:

set -a; source ~/.codex/secrets.env; set +a
python text2fx_gemini/midi_locked_patch.py run \
  --midi text2fx_gemini/songs/between_the_buttons/source.mid \
  --role-map text2fx_gemini/songs/between_the_buttons/song.json \
  --reference text2fx_gemini/songs/between_the_buttons/source.mp3 \
  --output-dir text2fx_gemini/ui_runs/goal_first5_$(date +%Y%m%d_%H%M%S) \
  --clip-start 0 \
  --seconds 5 \
  --steps 5

Use scripts/summarize_run.py and scripts/inspect_run.py on the resulting run directory before deciding the next change.
