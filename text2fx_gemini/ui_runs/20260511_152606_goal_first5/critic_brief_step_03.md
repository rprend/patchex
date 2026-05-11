# Critic Brief Step 03

## Primary objective

Make the render a steadier, darker, wider bass-weighted sustained chord. The current pass has the right locked notes and nearly the same whole-clip RMS, so do not change arrangement, timing, notes, tempo, or active ranges. Fix the patch/mix shape: reduce the exaggerated gain pumping, remove excess mid/presence/air, add warm bass body, and restore moderate stereo side energy.

## Evidence

- Global mix score is very low at 0.150; spectral timbre is 0.264 and A-weighted spectral is 0.048, so timbre/EQ is the main blocker.
- Arrangement preservation is perfect at 1.0. Pitch chroma is high at 0.959 while f0 contour is weak at 0.216; treat this as source/octave/timbre confusion on a dense sustained pad, not a note problem.
- Target energy is bass-dominant: bass 0.811, mid 0.009, presence 0.002, air 0.005. Render is much brighter/thinner: bass 0.463, mid 0.144, presence 0.078, air 0.064.
- Target centroid falls from about 368 Hz to 244 Hz. Render rises from about 718 Hz to 1005 Hz, opposite of the target.
- Stereo is too collapsed: target overall width 0.399/correlation 0.739; render width 0.069/correlation 0.991.
- Sustain shape score is 0.668 with 14 pumping events and 11 directional reversals. Largest 50 ms errors are around the chord change, especially 2.60-2.80s where the render jumps too loud and too bright, and at 4.90-5.00s where it drops out too hard.
- Modulation identity is poor at 0.436. The report sees target motion around 3.81 Hz/depth 0.18, while the render behaves like slow 0.80 Hz/depth 0.50 pumping. Listening-wise this reads as broad level automation, not the subtle tremolo/ensemble movement of the source.

## Changes to make next

1. `electric_piano_1`: replace the current bright organ-like square/saw emphasis with a warmer, rounder electric-piano/organ body.
   - Move waveform/source toward triangle/sine-organ or mellow electric piano, not saw/square buzz.
   - Reduce `blend` and upper harmonic content; keep only a small buzz layer if needed.
   - Raise low fundamental support slightly, but avoid excessive sub. Try `sub_level` around 0.18-0.22 only if the bass band still lacks weight.
   - Lower filter cutoff substantially: start around 650-750 Hz and end around 500-600 Hz. Add a gentle downward cutoff slope over the clip to match the target centroid falling, not rising.
   - Reduce or remove saturation on this track; saturation/noise fit is very weak and the render has too much presence/air.

2. `strings`: make this a dark, supportive pad, not a bright saw stack.
   - Reduce saw-stack brightness: fewer voices or a darker wavetable position, less detune if the upper harmonics smear.
   - Lower filter cutoff more aggressively than the current 1400 -> 1220 Hz. Try about 700 -> 520 Hz with low resonance.
   - Add a high-shelf cut or equivalent `high_gain_db` reduction if available, roughly -6 to -10 dB, and a smaller mid cut around -3 to -6 dB.
   - Keep it behind the electric piano; if the render remains too bright, lower `strings` gain 1-2 dB before raising master.

3. Both tracks: flatten the broad gain automation that is causing pumping.
   - Current gain points ramp from roughly -2 dB to +2.5/+2.8 dB and create large local overshoots/dropouts. Replace with much flatter gain points.
   - Target envelope slowly grows from about 0.050 to 0.072 across the 10 segments. Use a gentle rise of only about +1.0 to +1.5 dB across 5 seconds, not +4.5 to +5 dB.
   - Specifically reduce the spike after the chord change: at 2.60-2.80s the render is up to 0.141 RMS against target 0.049-0.071. Do not boost there; keep the level continuous through the change.
   - Prevent the final dropout: 4.90-5.00s target remains strong but render falls to 0.018 then 0.007. Extend release/hold or remove end-of-clip gain falloff so the sustain stays present through 5.0s.

4. Both tracks: use subtle 3.8 Hz motion, but not deep pumping.
   - Add a small synced/free LFO near 3.8 Hz to amplitude or filter on one or both layers, depth about 0.08-0.18.
   - Remove any slow 0.8 Hz-feeling gain movement. The report says candidate depth is already too large at 0.50; do not increase tremolo depth until the broad automation is flattened.
   - Prefer very shallow ensemble/tremolo motion on `strings` and a steadier `electric_piano_1`.

5. Stereo/space: widen the actual sustained layers rather than adding wet blur.
   - Increase side energy while keeping lows controlled. Try `electric_piano_1` width around 0.9-1.0 and `strings` width around 1.5-1.7.
   - Use chorus/ensemble width lightly on `strings`; the target is wider than the render, but not washed out.
   - Do not raise reverb much. Space fit is already decent and late-energy ratio is nearly perfect; more reverb will not fix the main mismatch.

6. Mix balance: rebalance toward bass/low-mid body.
   - After darkening filters, raise track gains only if whole-clip RMS drops. The current RMS average is already close, so avoid a master-gain fix first.
   - If low end remains weak after filtering, add body to `electric_piano_1` rather than boosting `strings` highs.
   - Keep presence and air very low; the target has almost no sustained energy above the bass/low-mid bands.

## Avoid this pass

- Do not edit MIDI notes, chord timing, tempo, meter, or active ranges.
- Do not add new percussion or transient layers to chase the onset count; the source onsets appear to come from modulation/noise within a sustained sound, not a new arrangement part.
- Do not add more reverb, delay, distortion, or bright chorus as the primary fix.
- Do not use a global master gain change to hide local 50 ms errors; the biggest failures are time-local pumping and wrong band balance.
