# MIDI-Locked Loss v1

This documents the MIDI-locked patch loss before the strict loudness and modulation update.

## Scope

The v1 loss is implemented in `text2fx_gemini/midi_locked_patch.py` by `score_midi_locked`.
It renders the patch session against a MIDI-locked arrangement, compares the full mix to the
reference audio, then combines full-mix, active-window, patch-control, sustain-shape, and
arrangement-preservation terms.

## Final Score

The v1 loss is:

```text
loss =
  0.45 * (1 - global_mix)
+ 0.30 * (1 - track_active_window)
+ 0.15 * (1 - patch_control)
+ 0.05 * (1 - sustain_shape)
+ 0.05 * arrangement_preservation_penalty

final = max(0, 1 - loss)
```

## Components

- `global_mix`: `audio_diff.compare_audio()` final score for the whole candidate render against the whole reference clip.
- `track_active_window`: mean `compare_audio()` final score over each MIDI track's active windows.
- `track_isolation_proxy`: mean score of each solo-rendered layer against the same active-window reference. This is reported but not directly weighted into the v1 final score.
- `patch_control`: mean of patch-control diagnostics across tracks:
  - ADSR/envelope fit
  - filter brightness fit
  - modulation fit
  - space fit
  - saturation/noise fit
- `sustain_shape`: diagnostic based on 50ms RMS windows from the full-mix diff.
- `arrangement_preservation`: score derived from whether locked MIDI notes, timings, velocities, and layer ids were preserved.

## Acceptance Gate

A candidate is accepted only if:

- final score is at least the previous best, and
- `sustain_shape` does not regress by more than `0.02`, and
- largest 50ms RMS error does not grow by more than `20%`, and
- these components do not regress past tolerance:
  - `track_isolation_proxy`: `0.04`
  - `patch_control`: `0.03`
  - `global_mix`: `0.03`

## Known Weaknesses

The v1 loss can over-reward broad spectral or embedding similarity while under-penalizing audible production failures:

- Sustained sections can become too quiet because `sustain_shape` is only 5% of the final loss.
- Local 50ms or beat-grid dropouts are diagnostics, not hard failures.
- Modulation is summarized too broadly; the loss does not strongly distinguish slow drift, cyclic filter LFO, tremolo, pitch vibrato, chorus drift, or one-off automation.
- Beat-synchronous envelope and mid-side movement are buried inside `global_mix` instead of being decisive.
- A candidate can pass with the wrong musical movement type if it matches pitch/chroma and broad spectrum well enough.
