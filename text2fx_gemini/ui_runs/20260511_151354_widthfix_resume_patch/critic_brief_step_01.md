# Critic Brief Step 01

## Primary Objective

Make the render a stable, dark, bass-forward sustained chord bed. The arrangement is correct: keep the two locked chord tracks, notes, timing, tempo, and meter unchanged. The next Producer pass should prioritize local envelope correction over broad timbre experiments, because whole-clip RMS is already close while the 50 ms windows are badly wrong.

Avoid in this pass: note edits, chord/octave changes, new rhythmic gating, new delay patterns, big reverb changes, or additional sharp gain-point swings. Do not chase the low `f0_contour` by changing pitches; pitch chroma is strong enough that this is mainly source/tone/octave-support ambiguity in a dense sustained pad.

## Evidence

- Global score is low (`0.341`), with global mix and active-window scores both at `0.203`; arrangement preservation is perfect (`1.0`).
- Target RMS is `0.0647`; render RMS is close at `0.0686`, but the render peak is much higher (`0.330` vs `0.255`) and the local envelope is unstable.
- Target 50 ms RMS stays sustained (`0.0399-0.0870`, median `0.0631`). Render ranges from near-dropout to over-bloom (`0.0078-0.1331`, median `0.0542`).
- Worst over-target windows: `0.10-0.25s` and `2.70-2.80s`. Worst under-target/dropout windows: `2.55-2.60s` and `4.90-5.00s`.
- Sustain diagnostics: score `0.660`, `12` pumping events, `9` directional reversals, `3` large level jumps. Dominant error says under-target overall, but the largest individual errors alternate between loud blooms and deep dropouts.
- Spectral/timbre is still a blocker: spectral timbre `0.389`; global spectral features `0.158`; candidate centroid rises from `376 Hz` to `555 Hz` while target falls from `368 Hz` to `244 Hz`.
- `strings` is the weakest isolated track (`0.144` final) with very poor spectral features (`0.035`), centroid trajectory (`0.052`), filter brightness fit (`0.474`), and saturation/noise fit (`0.204`).
- Stereo/space is not the main issue: stereo motion `0.748`, width is reasonably close, and late-energy ratio is high. The render is slightly too narrow by stats, but more reverb will not fix the main mismatch.

## Recommended Changes

1. `strings` - remove the gain automation that causes the chord-change pump.
   - Problem: the current curve has a boost at `2.52s`, a hard dip at `2.63s`, then recovery; this matches the measured collapse at `2.55-2.60s` and bloom at `2.70-2.80s`.
   - Move: replace `strings.gain_points` with a smoother, nearly continuous body. Start about `-4.5 dB` to `-5 dB` relative automation at `0.0s`, rise slowly to about `-3 dB` by `2.3s`, keep `2.50-2.85s` flat within `1 dB`, then rise only gently to the end. Do not use a local boost/dip pair around the chord change.

2. `electric_piano_1` - reduce the opening bloom and keep the transition filled.
   - Problem: the render over-shoots the target from `0.10-0.25s`, but then under-shoots at `2.55s` and the final tail. The electric piano is less wrong than strings, but its current gain points still reinforce the pump.
   - Move: lower the first `250 ms` by about `3 dB` compared with the current patch, remove the local `2.52s` boost / `2.64s` dip shape, and keep a small sustained lift only after `4.5s` so the final `4.90-5.00s` window does not collapse.

3. `strings` - darken and simplify the source.
   - Problem: the saw stack is too bright and synthetic for the target; the target is bass/low-mid dominant with very little mid, presence, or air.
   - Move: lower the string filter from the current `700-760 Hz` range to roughly `420-540 Hz`, keep resonance very low, reduce `voices` or `detune_cents` if the layer still beats/fizzes, and consider blending the source away from full saw toward a warmer organ/triangle-like pad. Keep `high_gain_db` at `-8 dB` or darker and reduce saturation further.

4. `electric_piano_1` - keep it warm and supportive, not percussive.
   - Problem: filter brightness fit is good on this track, but saturation/noise and modulation are weak, and the audible role should be a soft chord body rather than a tine attack.
   - Move: keep the triangle/electric-piano source, attack around `0.10-0.14s`, high sustain, and low resonance. Cut upper mids/presence more if needed (`mid_gain_db` near `-5 dB`, `high_gain_db` `-6 dB` or lower). Do not add transient punch.

5. Both tracks - rebalance energy downward without adding sub spikes.
   - Problem: weak windows consistently show the target dominated by bass around `0.80-0.85` of band energy, while the render leaks too much mid/presence/air. Some render windows also have excess sub during blooms.
   - Move: use low-mid/bass EQ support conservatively, but mainly cut mids and highs. Keep `strings.sub_level` very quiet or reduce it if the `2.7s` bloom persists. The goal is warmer and darker, not louder or boomier.

6. Modulation - replace automation pumping with subtle shimmer only after the envelope is stable.
   - Problem: modulation identity is poor (`0.487`); the render reads as slow level pumping while the target has gentle, faster motion.
   - Move: after flattening gain points, add only a shallow `3.5-4.0 Hz` tremolo or filter LFO to `strings` at low depth (`0.05-0.10`). Avoid deep amplitude LFO and avoid applying it to both tracks.

7. Stereo/space - leave reverb mostly alone.
   - Problem: space score is relatively good, and the target mismatch is direct tone/envelope rather than ambience.
   - Move: keep the shared reverb decay and send close to current values. If the darker patch becomes too centered, widen via `strings.width` or gentle chorus/source spread, not by increasing reverb wet level. Keep the low-frequency body effectively centered.

## Priority Order

First flatten `strings` gain points and remove the chord-change pump. Second smooth `electric_piano_1` level so the opening no longer blooms and the final tail does not vanish. Third darken the string source and cut mid/presence on both tracks. Only then add subtle faster shimmer if the render feels too static.
