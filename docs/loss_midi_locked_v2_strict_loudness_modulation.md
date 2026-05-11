# MIDI-Locked Loss v2: Strict Loudness And Modulation

This documents the updated MIDI-locked patch loss after adding stricter loudness, groove-envelope, and modulation-identity scoring.

## Scope

The v2 loss is implemented in `text2fx_gemini/midi_locked_patch.py` by `score_midi_locked`.
It still uses `audio_diff.compare_audio()` for detailed audio comparison, but the aggregate MIDI-locked score now gives explicit weight to production defects that were previously too easy to hide:

- active-window quietness and dropouts
- beat-synchronous envelope mismatch
- wrong modulation rate/depth/type
- stereo/mid-side motion mismatch

The report includes `loss_version: midi_locked_patch.v2_strict_loudness_modulation`.

## Final Score

The v2 weighted score before gates is:

```text
weighted_final =
  0.25 * spectral_timbre
+ 0.20 * loudness_envelope
+ 0.20 * groove_envelope
+ 0.15 * modulation_identity
+ 0.10 * stereo_motion
+ 0.05 * pitch_chroma
+ 0.05 * patch_control
```

Then hard gates and dropout penalties are applied:

```text
hard_gate_min = min(
  loudness_floor,
  groove_envelope,
  modulation_identity,
  arrangement_preservation
)

gate_multiplier = min(1.0, 0.58 + 0.42 * hard_gate_min)
dropout_penalty = exp(-0.08 * dropout_window_count)

final = clamp01(weighted_final * gate_multiplier * dropout_penalty - 0.05 * arrangement_preservation_penalty)
loss = 1 - final
```

In code this is represented as `final_loss = 1 - gated_score + arrangement_penalty`.

## Components

### `spectral_timbre`

Mean of broad timbre and spectrum scores:

- `multi_resolution_spectral`
- `mel_spectrogram`
- `a_weighted_spectral`
- `spectral_features`
- `harmonic_noise`
- `codec_latent`

### `loudness_envelope`

Mean of envelope and loudness-floor terms:

- `envelope`
- `segment_envelope`
- `exact_envelope_50ms`
- `loudness_floor`
- `sustain_shape`

### `loudness_floor`

New strict diagnostic from fixed 50ms windows. It measures whether the candidate stays loud enough wherever the reference is active.

Reported fields:

- `median_rms_ratio`
- `p10_rms_ratio`
- `max_under_db`
- `dropout_window_count`
- `severe_under_window_count`
- `weak_windows`

Scoring intent:

- reward median RMS ratio near the reference
- reward the 10th-percentile RMS ratio so brief quiet gaps matter
- penalize windows below `0.25x` reference RMS as dropouts
- penalize windows below `0.50x` reference RMS as severe under-target windows

### `groove_envelope`

New beat/time movement diagnostic. It makes the loss care about local envelope and band movement, not just whole-clip averages.

Mean of:

- `beat_grid_envelope`
- `beat_grid_band`
- `beat_grid_mid_side`
- `directional_delta`
- `exact_envelope_50ms`

### `modulation_identity`

New movement-type diagnostic. It compares the reference and candidate modulation signatures instead of treating all motion as equivalent.

Mean of:

- `modulation_rate`
- `modulation_depth`
- `modulation_periodicity`
- `modulation`
- `spectral_motion`

It also reports:

- reference rate/depth/periodicity
- candidate rate/depth/periodicity
- whether the reference appears `cyclic_lfo_like`
- whether the candidate is too slow
- whether the candidate is too shallow

### `stereo_motion`

Mean of:

- `stereo_width`
- `beat_grid_mid_side`

### `pitch_chroma`

Mean of:

- `pitch_chroma`
- `f0_contour`

### `patch_control`

Same as v1: mean patch-control diagnostics across active tracks.

## Acceptance Gate

The v2 acceptance gate keeps the v1 checks and adds strict production-failure checks:

- reject if `p10_rms_ratio < 0.45`
- reject if any dropout window exists below `0.25x` reference RMS
- reject if reference cyclic modulation exists and candidate rate is less than `60%` of reference rate
- reject if reference cyclic modulation exists and candidate depth is less than `60%` of reference depth
- reject if `groove_envelope < 0.55`
- reject regressions beyond tolerance for:
  - `track_isolation_proxy`
  - `patch_control`
  - `global_mix`
  - `loudness_floor`
  - `groove_envelope`
  - `modulation_identity`

## Agent Feedback

The new diagnostics are designed to tell the Critic and Producer more concrete production facts:

- whether the render is globally close but locally too quiet
- where the exact weak 50ms windows are
- whether movement should be cyclic LFO-like or broad automation
- whether the candidate LFO is too slow or too shallow
- whether beat-grid envelope and mid-side movement are failing

This should steer the agent toward the right class of patch changes: routed LFOs, beat-aligned gain/filter movement, sustain-floor fixes, and local automation rather than generic master gain or broad timbre changes.
