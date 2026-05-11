# Critic Brief Step 00

## Primary objective

Make the two sustained chord layers behave like a steady, wide, bass-heavy organ/string pad instead of the current narrow triangle patch that decays away. Keep the MIDI arrangement locked: chroma is very high and arrangement preservation is perfect, so this pass should fix envelope, gain automation, width, low/low-mid body, and modulation identity.

Avoid in this pass: changing notes, start times, durations, tempo, or adding rhythmic MIDI events. Also avoid chasing bright/presence detail until the sustain level and stereo body are corrected.

## Evidence

- Global score is very low: final 0.029, global mix 0.224.
- Target RMS is 0.0647; render RMS is 0.0340. The render is roughly half the target level overall.
- The render starts too loud then drops badly: target 50 ms RMS stays around 0.04-0.087, while render falls from about 0.078 at the start to 0.005 at the end.
- Loudness floor is the worst gate: median RMS ratio 0.34, p10 ratio 0.19, 32 dropout windows and 65 severe-under windows.
- Sustain shape says candidate is mostly under target, with worst misses at 4.90-5.00 s, 3.85-3.90 s, 4.10-4.20 s, and 2.50-2.60 s.
- Stereo is far too narrow: target width 0.399 / correlation 0.739; render width 0.056 / correlation 0.994.
- Spectral balance is wrong in the weak windows: target is dominated by bass/low-mid body, while render has too much mid/presence proportion when it drops.
- Modulation identity is wrong: target rate about 3.81 Hz at shallow depth 0.18; render shows slow 0.40 Hz pumping at excessive depth 0.65.
- Both tracks score similarly because they play the same sustained chord ranges for the full 5 seconds; isolation does not cleanly identify one culprit. Treat this as a layer design and mix-envelope problem, not a note problem.

## Recommended changes

1. `electric_piano_1`: replace the decaying triangle tone with a steadier warm electric-organ/electric-piano body.
   - Raise `gain_db` by about +4 to +6 dB, but remove the front-loaded feel by increasing sustain to 0.95-1.0 and release to about 0.45-0.7 s.
   - Reduce decay impact: set decay longer or nearly neutral, around 0.8-1.2 s, so the first chord does not collapse after the attack.
   - Change source from pure triangle toward organ/electric-piano: add a low saw or square/pulse component under the triangle, or use a warmer organ-like blend. Keep it low-harmonic, not bright.
   - Add low body: add a small sub or octave support on this layer, around `sub_level` 0.15-0.25 if available. This addresses the weak f0 contour without changing the notes.
   - Darken the mids: lower the filter range from 2600-3600 Hz to roughly 900-1800 Hz, with little resonance. The target centroid falls from about 368 Hz to 244 Hz; the render instead rises slightly.

2. `strings`: make this the wide sustained body layer rather than a duplicate narrow triangle.
   - Raise `gain_db` by about +3 to +5 dB and set sustain to 1.0 with release around 0.8-1.2 s.
   - Use a string pad / ensemble / soft saw source instead of triangle. Use 4-6 voices, detune around 8-14 cents, and more stereo spread.
   - Widen this track aggressively: set track `width` near 1.2-1.5, synth `stereo_spread` near 1.0, and increase chorus mix to roughly 0.18-0.28. The target has real side energy across bass through air; the current render is almost mono.
   - Keep the filter dark and warm: cutoff around 1000-1800 Hz, low resonance, no extra presence boost. The current presence/mid proportion is too high in the quiet sections.

3. Fix the full-clip gain shape before fine timbre tweaks.
   - Add gain automation to both tracks, or at least the dominant body layer, that rises after the first half instead of fading away.
   - Suggested shape: start around -2 to -3 dB relative automation, reach 0 dB around 1.0 s, +2 dB around 2.6 s, and +4 to +5 dB by 4.8-5.0 s.
   - Pay special attention to 2.50-2.60 s, 3.60-4.20 s, and 4.90-5.00 s. These are not master-gain-only errors; they need sustained layer body and low/low-mid energy in those windows.

4. Replace slow pumping with shallow fast motion.
   - Remove any broad 0.4 Hz gain/filter movement implied by the current decay/automation shape.
   - Add a shallow LFO around 3.8 Hz on amplitude or filter cutoff, depth about 0.12-0.20. Keep it subtle and continuous.
   - If only one track gets modulation, put it on `strings` width/filter or level so the pad has movement without making the core chord wobble too much.

5. Add width/space without washing out the envelope.
   - Raise return/track reverb only moderately: current reverb score is not the biggest issue. Use a slightly wider short room/plate, decay around 0.5-0.8 s, wet/send enough for width but not enough to blur the 50 ms envelope.
   - Increase side energy mainly through chorus/unison/stereo spread on `strings`, not a long reverb tail.

6. Add a little harmonic density/noise only after the level shape is fixed.
   - Saturation/noise fit is weak, but this is secondary. Add light tape/soft saturation on both tracks, around 0.05-0.12, to thicken bass/low-mid harmonics.
   - Do not add bright hiss or air yet; target air is low and the current render is already proportionally too present when quiet.

## Expected next-pass result

The next render should no longer fade into dropout at the end. It should hold a wide, warm, low-centered sustained chord with shallow fast motion, stronger late energy, and much less mono correlation. Once that is stable, the following pass can refine the exact organ-vs-string source blend and the small onset/texture details.
