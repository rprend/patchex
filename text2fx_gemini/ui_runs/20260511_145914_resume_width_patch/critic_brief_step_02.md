# Critic Brief Step 02

## Primary objective

Make the two locked chord layers behave like one steady, dark, bass-dominant sustained organ/pad source. The arrangement is correct; the next pass should fix the envelope pumping/dropouts and the bright synthetic upper harmonics before making any fine timbre tweaks.

Avoid in this pass: MIDI/note edits, new parts, delay, wetter or longer reverb, brighter filter ramps, extra sub spikes, or deep LFO/gain modulation on both tracks.

## Evidence

- Overall match is still poor: final 0.333, global mix 0.200, active-window 0.200, isolation proxy 0.238. Arrangement preservation is perfect at 1.000, so keep the performance fixed.
- The render is close in whole-clip RMS, but locally wrong: source RMS 0.0647 vs candidate 0.0685, while exact 50 ms envelope is only 0.414. Sustain shape has 12 pumping events, 9 reversals, and largest 50 ms RMS error 0.080.
- Biggest envelope failures are musical, not subtle: too loud at 0.10-0.25 s and 2.70-2.80 s, nearly absent at 2.50-2.60 s and 4.90-5.00 s. The target gradually rises across the clip; the render jumps and collapses.
- Tonal balance is too bright and not bass-weighted enough. Target bass band is 0.811 vs candidate 0.626; candidate has excess low-mid, mid, presence, and air. Candidate centroid ends much brighter than target, 558 Hz vs 244 Hz.
- `strings` remains the weak track: isolation final 0.142, spectral features 0.035, spectral motion 0.037, centroid trajectory 0.052, filter brightness fit 0.474, saturation/noise fit 0.203. `electric_piano_1` is closer, especially filter brightness fit 0.984, so treat it as the chord identity and make `strings` the dark body/width layer.
- Modulation identity is wrong: reference reads about 3.81 Hz at shallow depth 0.18, while candidate reads 0.40 Hz at deep depth 0.51. This supports what the envelope report shows: the current render is dominated by slow pumping rather than shallow texture.

## Recommended changes

1. `strings` - remove the bright saw-stack fingerprint.
   - Change the source toward a darker organ/string pad: reduce saw dominance, reduce unison buzz, and blend more triangle/sine/organ body if the patch format allows it.
   - Keep the low-pass dark and nearly flat, around 600-700 Hz. Do not open the filter late.
   - Cut high EQ harder on this track: reduce presence/air another 3-6 dB if available. The target has very little energy above the bass/low-mid body.
   - Reduce `saturation` on `strings` or make it warmer/low-order. The current string layer is adding synthetic fizz, not useful texture.

2. `strings` - flatten the amplitude shape before raising level.
   - Remove the gain dip at the chord change. The 2.50-2.60 s windows are severe under-target/dropout windows, so the string bed must stay present through the change.
   - Remove the rebound boost after the change. The 2.70-2.80 s windows overshoot badly; do not compensate the dip with a spike.
   - Use a smooth, mostly level sustain curve with only a slight late lift into 4.85-5.00 s. The target tail remains strong; the render collapses there.
   - Keep sustain high and lengthen release to at least 0.8 s so the final window does not fall away before the clip ends.

3. `electric_piano_1` - keep the identity, but tame the front edge.
   - The first 250 ms is too loud. Lower early gain automation by roughly 3-5 dB or increase attack slightly, but keep the note onset smooth rather than plucky.
   - Do not brighten this layer. Its filter fit is already good; preserve the dark triangle/electric-piano body and keep high EQ cut.
   - Shorten or control release only enough to avoid smearing the 2.58 s chord change; do not create a tail dropout.

4. Both tracks - fix the 50 ms envelope as the first-order target.
   - Replace the current sharp gain points with a continuous sustain shape: no deep point just before 2.58 s, no positive bump just after 2.65 s, and no falloff at the very end.
   - If the overall render becomes quieter after removing spikes, recover with small static gain changes after the local envelope is stable. Do not solve this with more automation.
   - Watch these pass/fail windows: 0.10-0.25 s should come down, 2.50-2.60 s should come up, 2.70-2.80 s should come down, and 4.90-5.00 s should come up.

5. Mix/EQ - make the body bass-forward without adding sub.
   - The missing weight is mainly bass-band dominance, not lack of sub. Sub is already slightly high in several weak windows, so avoid adding more sub oscillator.
   - Shift apparent weight by cutting mids/presence/air and strengthening stable fundamental/body. `strings` should supply muted low-mid width; `electric_piano_1` should remain the more defined chord layer.
   - Keep master gain conservative; the whole-clip RMS is already close, and the problem is local pumping.

6. Stereo/space - widen direct harmonics, not reverb.
   - Increase `strings` direct width/chorus/spread modestly after darkening it. The target has more side energy, but the late-energy score is already good enough that extra reverb will blur the envelope.
   - Keep low fundamentals centered. Widen only the muted pad harmonics.
   - Do not increase return send or reverb decay in this pass.

7. Modulation - remove slow pumping, then add only shallow fast motion if needed.
   - Any current slow gain/filter movement around 0.4 Hz should be removed or greatly reduced.
   - If the next render becomes too static, add a very shallow 3.5-4 Hz tremolo/filter/chorus motion on `strings` only. Keep depth low enough that 50 ms RMS stays steady.
   - Do not apply deep synchronized modulation to both tracks.

## Pass criteria

The next render should sound like a continuous dark sustained chord, not a bright saw pad with automation jumps. It should have no hole at 2.50-2.60 s, no rebound spike at 2.70-2.80 s, no tail collapse at 4.90-5.00 s, less mid/presence/air fizz, stronger bass-band dominance, and wider muted pad harmonics without added reverb wash.
