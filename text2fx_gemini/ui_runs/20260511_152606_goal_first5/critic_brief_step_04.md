# Critic Brief Step 04

## Primary Objective

Make the next pass darker, bass-weighted, wider, and steadier. The MIDI arrangement is already preserved perfectly, and the full-clip RMS is close, so do not change notes, timings, tempo, or track activity. Focus on timbre, envelope/gain automation, stereo width, and modulation identity.

## Evidence

- Final score is still low at `0.294`; global mix and active-window score are both `0.150`.
- Arrangement preservation is `1.0`, so the problem is patch/mix, not performance.
- Pitch chroma is strong globally (`0.959`), but f0 contour is weak (`0.216`): treat this as source/octave/support uncertainty, not a reason to edit notes.
- The target is much darker and more bass-dominant. Target centroid falls from about `368 Hz` to `244 Hz`; candidate rises from about `718 Hz` to `1005 Hz`.
- Band balance is the biggest blocker: target bass is `0.811`, candidate bass is only `0.463`; candidate has too much mid/presence/air.
- Candidate stereo is too collapsed: target width `0.399` and correlation `0.739`; candidate width `0.069` and correlation `0.991`.
- Modulation identity is off: target reads as cyclic motion around `3.81 Hz` at shallow depth `0.18`; candidate reads as slower `0.80 Hz` with excessive depth `0.50`.
- Sustain shape still has 14 pumping events and 11 directional reversals. Largest 50 ms errors are around `2.60-2.80s`, where the candidate jumps far above target, and `4.90-5.00s`, where it drops far below target.

## What I Hear / Musical Read

The target feels like a low, dark, sustained organ/string pad with subtle ensemble shimmer and steady tremolo-like movement. The current render has the right chord skeleton, but it is too synthy and bright after the attack, too narrow in the center, and its gain automation makes it breathe in obvious waves instead of sitting as a continuous pad.

## Next Production Moves

1. `electric_piano_1` - make this the low fundamental body, not a bright organ layer.
   - Lower the filter substantially: set cutoff around `450-650 Hz` at the start and let it drift down toward `350-500 Hz` by the end. Keep resonance low.
   - Reduce or remove the current upper-harmonic square/saw character. Move the source toward a smoother organ/triangle-sine blend with only a small square component.
   - Keep the sub/octave support, but rebalance it as musical bass body rather than sub boom: raise the fundamental/bass portion and reduce mids/presence. If available, use a gentle low shelf/body boost around `100-250 Hz` and cut `700 Hz-4 kHz`.
   - Shorten the attack slightly only if needed for the first transient, but avoid a sharp pluck. The target onset is softer than the candidate's first 50 ms overshoot.
   - Smooth the gain automation. Remove the big late lift and any valley that causes under-windows at `0.75-1.15s`, `2.45-2.60s`, and `4.85-5.00s`. Aim for a slow rise across the clip, not pulsing.

2. `strings` - use this for width and shallow ensemble motion, but darken it hard.
   - Lower the filter even more than the current `1400 -> 1220 Hz`; try `650 -> 450 Hz` or similar. The current string layer is likely driving the excessive presence/air and wrong rising centroid.
   - Reduce bright saw-stack edge. Use fewer high harmonics or a warmer string/organ pad source. If using saws, reduce blend/brightness and detune enough for width without making the sound fizzy.
   - Increase stereo width from the layer itself, not from long reverb. Keep `width` high, increase unison/stereo spread if needed, and consider a subtle chorus/ensemble, but filter the chorus dark.
   - Replace the current slow/large gain motion with a shallow synced/free LFO near `3.8 Hz`, depth around `0.12-0.20`. Put it mostly on amplitude or filter cutoff, not both at high depth.
   - Avoid the exaggerated swell at the chord change. The candidate is far too loud from `2.60-2.80s`; reduce gain automation there by several dB or flatten the chord-change bump.

3. Global balance - trade highs for bass while keeping total RMS similar.
   - Do not raise master gain; RMS is already close (`0.0654` candidate vs `0.0647` target).
   - Apply a broad master or per-track high cut / high shelf reduction above roughly `1 kHz`, especially presence and air.
   - Shift energy into the bass band, not the sub band. The target is bass-heavy but not bright; the candidate already has excess sub proportion in some windows.
   - Keep reverb short. Current late-energy ratio matches well, so do not add a longer wash. If adding space, use dark, wide early reflections with low wet level.

4. End sustain - fix the last half second.
   - The candidate collapses at `4.90-5.00s` while the target stays strong. Extend releases or flatten end gain so both layers continue through the clip boundary.
   - Avoid a last-second brightness tail. The end should remain dark and bass-led.

## Avoid This Pass

- Do not edit note pitches, durations, velocities, tempo, or active ranges.
- Do not add drums or new rhythmic note events to chase the onset-count diagnostic; the detected target onsets are mostly modulation/texture in a sustained source.
- Do not add bright saturation, bitcrush, phaser, or long delay.
- Do not solve width with a large wet reverb tail; the target is wider than the render but still controlled and sustained.

## Priority Order

1. Darken both tracks and restore bass-band dominance.
2. Flatten the bad gain pumping, especially `2.60-2.80s` and `4.90-5.00s`.
3. Add shallow `~3.8 Hz` cyclic motion.
4. Widen the string/ensemble layer while keeping lows mostly centered.
