# Critic Brief Step 02

## Primary objective

Make the patch sound like a dark, bass-dominant sustained organ/string chord bed, not a bright synth pad. The notes and timing are already preserved perfectly, and full-clip RMS is close, so the next pass should focus on spectral balance, local envelope stability, and stereo width.

Avoid changing MIDI notes, chord timing, tempo, meter, or adding new rhythmic parts. Also avoid broad master gain moves unless a local fix changes RMS substantially.

## Evidence summary

- Final score is still low at `0.294`; global/active-window score is only `0.150`.
- Arrangement preservation is `1.0`, and pitch chroma is high (`0.959`), so do not treat the poor `f0_contour` as a note problem. This is a tone/source and octave-support problem on dense sustained chords.
- Target energy is overwhelmingly bass-band. The report shows reference bass `0.811` vs candidate `0.463`, while candidate has too much low-mid/mid/presence/air. My FFT check also reads target as almost entirely 60-250 Hz energy, with the render carrying extra low-mid and mid content.
- The render is far too narrow: reference side/mid is about `0.399`, candidate about `0.069`; report width is reference `0.399` vs candidate `0.069`, correlation `0.739` vs `0.991`.
- Sustain shape is not fixed: largest 50 ms errors happen around the chord change (`2.60-2.80s`) where candidate overshoots, and near the end (`4.90-5.00s`) where candidate drops out. There are `14` pumping events and `11` reversals.
- Modulation identity is wrong: target is shallow cyclic motion near `3.81 Hz` / depth `0.18`; candidate is detected around `0.80 Hz` / depth `0.50`, so the audible motion is too slow and too lumpy.

## Recommended changes

1. `electric_piano_1`: make it the dark fundamental anchor.
   - Change the source away from the current bright `organ-like` square/saw blend. Use a smoother organ/triangle-sine blend or very filtered organ source with much less saw/square edge.
   - Reduce upper harmonic content: lower `wavetable_position`/blend toward the smoother side, reduce unison brightness, and keep detune subtle.
   - Lower the low-pass cutoff substantially: try `cutoff_start_hz` around `450-650 Hz`, `cutoff_end_hz` around `350-500 Hz`, with low resonance. The target centroid falls from about `368 Hz` to `244 Hz`; the candidate rises from about `718 Hz` to `1005 Hz`, which is the opposite motion.
   - Increase true bass/fundamental support without adding fizz: raise `sub_level` moderately from `0.14` to about `0.22-0.30`, or use a quiet sine/triangle octave support layer if available. Keep this centered/mono.
   - Reduce or remove saturation if it is adding upper harmonics. Current mid/presence excess matters more than harmonic density.

2. `strings`: stop it from being the bright layer; make it a wide, filtered ensemble.
   - Keep the string role, but darken it heavily. Lower `cutoff_start_hz` from `1400` to roughly `550-750 Hz`, and `cutoff_end_hz` to roughly `400-600 Hz`.
   - Reduce saw-stack brightness: use fewer bright partials, less blend toward saw, and softer filtering. The isolation proxy for `strings` is the weakest (`0.128`) with terrible spectral features/centroid scores, so this is the main timbre offender.
   - Keep ensemble width, but do not use chorus as a brightener. If chorus stays at `0.24`, filter it darker or reduce it to `0.12-0.18` and increase stereo spread/width instead.
   - Add shallow 3.8 Hz amplitude or filter shimmer only on this layer: rate `3.7-3.9 Hz`, depth around `0.10-0.18`. Remove any slow gain movement that produces the detected `0.8 Hz` pumping.

3. Both tracks: repair local envelope shape instead of changing overall loudness.
   - The first 250 ms is too loud, while `0.75-1.15s`, `2.45-2.60s`, and `4.85-5.00s` are under or unstable. Reduce the initial transient/front gain by about `2-3 dB`.
   - Smooth the gain automation. Current gain points create lumpy rises and a severe end falloff. Use a more gradual ramp: lower at `0.0s`, near-neutral by `1.0s`, slight lift after `2.6s`, and maintain level through `5.0s` instead of letting releases vanish.
   - Increase release or add a final hold so the last `4.90-5.00s` window does not collapse. Target remains strong to the end; candidate RMS falls to near dropout there.
   - Around the chord change (`2.58s`), avoid stacking a gain boost exactly on the new note attack. The report shows candidate overshoots by `+0.07` to `+0.088 RMS` from `2.60-2.80s`. Ease the lift after the change, or add a short dip/softer attack on the second chord.

4. Stereo/space: widen the sustained body, not the reverb tail.
   - Increase `strings.width`/stereo spread enough to bring side energy up, but keep `electric_piano_1` bass support centered.
   - The return reverb is not the main issue because late-energy ratio is already excellent (`0.997`). Do not lengthen reverb much. If using the `space` return, keep decay short/damped and use it only for side width.
   - Consider widening only above the bass fundamentals with the strings/chorus layer, while keeping sub and main low bass mono.

5. Mix balance after timbre changes.
   - After darkening, rebalance so `electric_piano_1` supplies most of the 60-250 Hz body and `strings` supplies width/soft ensemble color.
   - If the mix becomes too quiet after filtering, raise track gains slightly rather than reopening the filters. The target is not bright; it is a sustained low, warm chord bed.

## Conservative target for next render

Aim for: darker centroid trajectory, bass-band dominance, side/mid width closer to the reference, fewer 50 ms envelope reversals, and no end dropout. A successful pass may not improve every spectral metric immediately, but it should reduce the glaring mid/presence excess and the narrow, pumping character.
