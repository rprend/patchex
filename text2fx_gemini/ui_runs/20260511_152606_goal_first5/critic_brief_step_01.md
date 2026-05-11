# Critic Brief Step 01

## Primary objective

Keep the successful full-clip loudness recovery, but make the patch darker, bass-centered, wider, and locally smoother. The target is a sustained low organ/string chord bed with shallow fast motion; the current render is close in total RMS but has too much mid/presence/air, not enough bass dominance, too little side energy, and obvious 50 ms pumping around the chord change and ending.

Avoid in this pass: changing MIDI notes, timings, durations, tempo, or adding rhythmic events. Also avoid further broad gain boosts, brighter filters, extra air/hiss, long reverb tails, or stronger slow pumping automation.

## Evidence

- Overall score is still low: final `0.294`, global mix `0.150`, track active-window `0.150`, isolation proxy `0.156`; arrangement preservation is perfect at `1.0`.
- Full-clip RMS is now close: target about `0.0647`, render about `0.0654`, so the next improvement is not master gain.
- Chroma is high in the active window (`0.959`), but `f0_contour` is low (`0.216`). Treat this as tone/source/octave-support uncertainty, not a note problem.
- Spectral/tone fit is the main blocker: `spectral_motion 0.0086`, `centroid_trajectory 0.0149`, `spectral_features 0.0098`, `a_weighted_spectral 0.048`.
- Target centroid falls from about `368 Hz` to `244 Hz`; render centroid rises from about `718 Hz` to `1005 Hz`. The patch is moving brighter while the reference gets darker.
- Global band balance is wrong: target is about `81% bass`, `15% low-mid`, with almost no mid/presence/air. Render is only `46% bass` and has excess `mid 14%`, `presence 7.8%`, `air 6.4%`.
- Sustain shape is still unstable: score `0.668`, `14` pumping events, `11` directional reversals. Worst 50 ms windows are around `2.60-2.80 s`, `4.90-5.00 s`, and the opening.
- At `2.65-2.75 s`, render RMS is more than twice the target and has excess mids/presence/air while still under-representing bass proportion. At `4.90-5.00 s`, render drops too low but remains too bright proportionally.
- Stereo is improved but still too mono: target width `0.399` / correlation `0.739`; render width `0.069` / correlation `0.991`. Side energy is especially missing in the bass/low-mid region.
- Modulation identity is closer conceptually but still off: reference is about `3.81 Hz`, shallow depth `0.18`; candidate reads around `0.80 Hz`, depth `0.50`, so the perceived motion is too slow and too deep.

## Recommended changes

1. `electric_piano_1`: make this the centered dark fundamental/body layer.
   - Reduce brightness hard: lower cutoff from the current `1150->1000 Hz` shape to roughly `650 Hz` at start, `580 Hz` by `2.6 s`, and `500-540 Hz` at end. Keep resonance very low, around `0.05-0.10`.
   - Increase bass/fundamental support without raising total RMS: raise `sub_level` from `0.14` to about `0.22-0.30`, but lower track `gain_db` by about `1-2 dB` if the full RMS rises.
   - Make the source less buzzy in the upper partials: move the organ blend toward sine/triangle plus low square, not saw-heavy. If `blend` controls saw/square brightness, reduce it from `0.55` toward `0.35-0.45`.
   - Narrow this layer slightly and keep it central: set `width` around `0.45-0.6`, `stereo_spread` around `0.35-0.5`. Let `strings` provide width.
   - Smooth the local gain automation: reduce the `2.6 s` lift by about `1.5-2 dB` because the render overshoots badly at `2.60-2.80 s`; keep some late lift near `4.8-5.0 s` but make it gradual.

2. `strings`: make this a dark wide ensemble, not a bright saw pad.
   - Lower filter cutoff from `1400->1220 Hz` to roughly `850 Hz` at start, `720 Hz` at `2.6 s`, and `620-680 Hz` at end. This should pull the centroid downward with the reference.
   - Reduce upper-harmonic density: if using `saw_stack`, lower `wavetable_position`/bright blend from `0.45/0.55` toward `0.25-0.35`, or switch toward soft string/organ ensemble with fewer sharp saw partials.
   - Keep width high but create width from chorus/unison, not reverb: `width 1.35-1.5`, `stereo_spread 1.0`, chorus around `0.18-0.24`. Do not increase reverb decay.
   - Add shallow fast motion only: keep/add an LFO around `3.8 Hz`, but reduce depth to about `0.05-0.09` on gain or filter. Current motion reads too deep and slow; avoid broad tremolo.
   - Consider a small low-mid EQ boost or body emphasis around `180-300 Hz` if available, paired with a high/presence cut. The target is bass/low-mid weighted, not airy.

3. Full mix envelope: fix pumping before further timbre detail.
   - Do not raise master or both tracks globally; full RMS is already close.
   - Tame the chord-change hump: reduce both tracks' gain automation around `2.6-2.8 s`, especially `electric_piano_1`, until the 50 ms RMS no longer spikes above the target.
   - Protect the final sustain without adding brightness: keep a smooth late gain ramp into `4.8-5.0 s`, but darken it and rely on the body layer/sub support rather than presence-band energy.
   - If compression is available, use only very gentle leveling on the summed layers: low ratio around `1.3-1.6:1`, slow-ish attack, medium release. Avoid audible pumping.

4. Space/stereo:
   - Keep reverb short: return decay around `0.5-0.7 s`, low wet/send. The target has width, but the envelope is still measured in tight 50 ms windows.
   - Increase side energy through `strings` width/chorus and possibly a subtle stereo spread on low-mid content. Do not widen the sub/fundamental enough to destabilize the center.

5. Saturation/noise:
   - Reduce or keep saturation modest (`0.03-0.07`) if it is adding presence/air. The saturation/noise fit is weak, but adding grit is less important than removing the excess upper bands.
   - Do not add hiss, air lift, bitcrush, or bright tape noise in this pass.

## Expected next-pass result

The next render should keep the current near-correct average loudness but shift energy downward into bass/low-mid, lower the centroid over time, reduce the `2.6-2.8 s` RMS hump, preserve the final sustain without bright fizz, and widen mainly through the string ensemble. If this improves the spectral and stereo scores, the following pass can refine the exact organ/string source blend and the shallow 3.8 Hz motion.
