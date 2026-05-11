# Critic Brief Step 00

## Primary objective

Rebuild the patch as a steady, low-register sustained chord bed with stronger body, wider stereo image, and gentle fast tremolo/ensemble motion. Do not change the MIDI notes, timing, tempo, or track activity: arrangement preservation is perfect and pitch chroma is already high enough that the weak f0 score is more likely a tone/source/octave-support issue than a note issue.

Avoid in this pass: adding new rhythmic note events, changing chord voicings, making bright plucks, adding long delay, or chasing tiny EQ tweaks before fixing sustain level and stereo width.

## Evidence summary

- Global score is very low: final `0.029`, global mix `0.224`.
- Target RMS is about `0.0647`; current render is `0.0340`, and ffmpeg level checks show the current file is roughly 24 dB lower overall than the target.
- The target envelope gradually rises across the 5 seconds (`0.050 -> 0.072` by report segment RMS). The render starts close but collapses after the initial onset (`0.058, 0.019, 0.014, 0.021...`).
- Sustain shape is mostly under target: largest 50 ms RMS errors are about `-0.074` near `4.90-5.00s`; there are 4 pumping events and 3 directional reversals.
- Stereo is far too narrow: target width `0.399`, candidate width `0.056`, candidate correlation `0.994`.
- Spectrum is too exposed in mid/presence and short on low-mid/sub support: candidate mid `+0.0466`, presence `+0.0095`, low-mid `-0.0227`, sub `-0.0129`.
- Modulation identity misses the target: reference motion is about `3.81 Hz` with moderate depth `0.18`; candidate is detected around `0.40 Hz` with too-large slow level movement.
- Both tracks have the same sustained chord blocks, so isolation is weak. Still, `strings` is the weaker proxy (`0.218`) and should become the main body/wide pad, while `electric_piano_1` should supply controlled attack and harmonic definition.

## Recommended changes

### `strings` - make this the main sustained body

Problem: current `strings` is a triangle duplicate of the electric piano, too quiet, too narrow, too clean/thin, and not carrying the long low-mid sustain. The target sounds like a sustained ensemble/organ-like low chord bed, not a short-decay triangle patch.

Production moves:

- Raise `strings.gain_db` substantially, about `+10 to +14 dB` from the current `-15.5 dB`. This track should carry most of the perceived level after the first 300 ms.
- Change source from pure triangle toward a warm string/organ pad: use saw/triangle blend or string-pad wavetable if available. Keep it mellow, not buzzy; target has dominant bass/low-mid and very little presence.
- Add low support: enable a sub or octave-support layer lightly (`sub_level` around `0.15-0.25`) or blend in a sine/triangle fundamental under the pad. This should help the weak f0 contour without changing notes.
- Fix the envelope for continuous sustain: attack `0.08-0.16s`, decay `0.4-0.8s`, sustain `0.9-1.0`, release `0.45-0.8s`. The current sustain `0.65` and release `0.25` are letting the body fall away.
- Add gentle amplitude or filter tremolo near the measured target rate: LFO around `3.7-3.9 Hz`, shallow depth only (`0.08-0.15`). Avoid slow 0.4 Hz gain sweeps or deep pumping.
- Widen it aggressively: set width/stereo spread near `1.0-1.4`, increase chorus/ensemble mix to about `0.18-0.28`, and keep lows mostly centered if the engine supports band-limited widening.
- Darken the top: lower filter cutoff from the current `2600-3600 Hz` toward `900-1600 Hz`, with low resonance. Add a small low-mid boost or high/presence cut if available.
- Add mild saturation/noise texture, not distortion: saturation around `0.05-0.12` to thicken the sustain and improve harmonic/noise fit.

### `electric_piano_1` - reduce duplicate pad role, keep soft attack/detail

Problem: it currently duplicates the strings patch, adds too much mid/presence relative to target, and the initial onset is too high while later windows are too low. It should be a quieter attack/color layer rather than half of the whole pad.

Production moves:

- Keep `electric_piano_1` quieter than `strings`; raise less than the strings track, roughly `+4 to +7 dB` from current if master gain is not changed. If the producer raises master gain, leave this closer to current level.
- Use a warmer electric-piano/organ triangle tone with less unison brightness: reduce detune/voices if it creates beating, or use 2 voices instead of 3.
- Soften the transient: attack `0.04-0.08s`; decay `0.5-0.9s`; sustain `0.75-0.9`; release `0.35-0.6s`.
- Darken it more than the current patch: cutoff about `1200-2200 Hz`, reduce high/presence EQ by `2-4 dB`, and avoid extra resonance.
- Narrow this track relative to `strings` (width around `0.5-0.8`) so the wide image comes from the pad/ensemble, not from two identical wide layers.
- Add only shallow 3.8 Hz tremolo/filter motion if needed; do not let it create obvious pumping.

### Mix/master and space

Problem: the render is globally under-level, nearly mono, and the target has more sustained low-mid/side energy without sounding washed in reverb.

Production moves:

- After raising track sustain, set master gain so full-mix RMS lands close to target. A rough starting move is master `+6 to +9 dB`, but avoid clipping; prefer track gain first so the sustain shape improves before final level.
- Keep reverb short and supportive. Current return decay `0.35s` is acceptable; raise send/wet only slightly if widening still falls short. Do not add a long tail, because late-energy coverage is already good.
- Add ensemble/chorus width before adding reverb. The report’s stereo failure is width/correlation, not missing ambience.
- Correct spectral balance globally after level fixes: reduce broad mids/presence and restore low-mid/sub. A starting EQ target is `-3 dB` around upper mids/presence, `+1 to +2 dB` low-mid/body if needed, and no bright air boost.

## Priority order for the next Producer pass

1. Make `strings` the louder, wider, sustained low-mid pad with high sustain and gentle 3.8 Hz motion.
2. Recast `electric_piano_1` as a softer, darker support/attack layer rather than a duplicate equal pad.
3. Bring overall RMS up after the sustain no longer collapses.
4. Re-check stereo width and local 50 ms RMS windows, especially `3.6-4.2s` and `4.9-5.0s`, where the current render is far below the target.
