# Patchex Plugin Guide

This note captures the paid plugin direction for later and the free path to use now when improving Patchex MIDI-locked reconstructions. The goal is not to buy plugins first. The goal is to make the harness prefer the right production moves: subtractive synth tone, tempo-synced delay, chorus width, sidechain-style movement, rhythmic filter/gate motion, saturation, compression, EQ, and reverb.

## Current Patchex Baseline

- Instrument: Vital is installed locally and is the preferred synth engine for new MIDI-locked sessions when the Audio Unit path is available.
- Current effects: Patchex renders most effects internally in Python DSP, not through external VST/AU plugins. The implemented primitives include tempo delay, sidechain pump, step gate, filter sequencer, Juno-style chorus, reverb/returns, saturation, compression, EQ, phaser/flanger-style master movement, and stereo motion.
- Best next architecture: keep using internal DSP for scoring speed and determinism, but map each primitive to a known plugin vocabulary so prompts and patch operations resemble real production choices.

## Paid Later

These are strong commercial targets for a later real-plugin rendering pass.

| Role | Paid target | Why |
| --- | --- | --- |
| Synth | Vital Pro or keep Vital Free | Vital already works here and is more than good enough for saw/pulse, Juno-ish, Prophet-ish, and MicroKORG-ish patches. |
| Tempo delay | Soundtoys EchoBoy or FabFilter Timeless | Dotted/eighth filtered delay, ping-pong, saturation, and rhythmic repeats. |
| Sidechain/pump | Cableguys ShaperBox or Kickstart 2 | Fast, predictable French-electro volume shaping. |
| Step gate | Xfer LFOTool or Cableguys ShaperBox | Drawn rhythmic gates without adding MIDI notes. |
| Filter sequencer | Cableguys ShaperBox | Patterned cutoff movement and rhythmic filter accents. |
| Juno chorus | Arturia Chorus JUN-6 or TAL-Chorus-LX | Classic wide Juno ensemble behavior. |
| Reverb | Valhalla VintageVerb | Big plate/hall send that can stay musical at low wet levels. |
| Saturation | Soundtoys Decapitator | Tone, thickness, and controllable dirt. |
| Compression | FabFilter Pro-C 2 | Clean master/bus compression and sidechain-aware control. |
| EQ | FabFilter Pro-Q | Surgical and broad shaping with fast workflow. |
| Phaser/flanger master motion | Soundtoys PhaseMistress or Eventide Instant Flanger | Subtle global movement for the "effects on the whole title" idea. |

## Free Now

Use this stack before spending money. Prefer AU on macOS when integrating with Patchex, because the Vital Audio Unit path is already proven locally.

| Role | Free target | Notes |
| --- | --- | --- |
| Synth | Vital Free | The practical default. Use saw/pulse, unison lightly, low-pass filter, filter envelope, LFO cutoff movement, and onboard chorus/delay only if the harness can expose those controls cleanly. |
| Alternate subtractive synth | Surge XT or u-he Tyrell N6 | Useful for Juno/Prophet-style patches if Vital is unavailable or too wavetable-clean. |
| Tempo delay | Voxengo Tempo Delay, Tritik Tymee, or Valhalla Supermassive | Voxengo is the most literal tempo-delay choice. Tymee adds filter/downsample texture. Supermassive is excellent for delay-wash and huge space, but can be too smeary if overused. |
| Sidechain/pump | ZEEK STFU or DuckTool | Free volume shaping for kick-style pumping. In Patchex, this maps to `sidechain_pump`, not arbitrary dense `gain_points`. |
| Step gate | ZEEK STFU, DuckTool, or Ableton Auto Pan/Utility automation | Use tempo-locked shape/gate behavior. Avoid random volume automation points. |
| Filter sequencer | Audiomodern Filterstep, Ableton Auto Filter with synced LFO, or Surge XT/Vital modulation | Best mapped to `filter_sequencer`: divisions, pattern, depth, smoothing, resonance. |
| Juno chorus | OSL Chorus or TAL-Chorus-LX | OSL is explicitly Juno-60 modeled. TAL-Chorus-LX is a common free Juno chorus option. |
| Stereo panning motion | Cableguys PanCake | Free drawn LFO panner. Use sparingly for width/motion, not as a substitute for tone. |
| Reverb | Valhalla Supermassive or TAL-Reverb-4 | Supermassive is the best free "big space" option. TAL-Reverb-4 is better when the target needs vintage plate-like 80s space without endless wash. |
| Saturation/clip | Venomode Mesa Lite, Kilohearts Essentials, or Ableton Saturator | Mesa Lite is a free clipper/saturation option. Ableton Saturator is enough for most harness-matching work. |
| Compression | Klanghelm DC1A or TDR Kotelnikov | DC1A is fast for pump/glue; Kotelnikov is cleaner for bus/master dynamics. |
| EQ/dynamic EQ | TDR Nova | Free dynamic EQ with standard EQ duties covered. |
| Phaser/flanger | Blue Cat Phaser and Blue Cat Flanger | Free modulation effects for subtle master or bus movement. |

## Patchex Primitive Mapping

These names should stay stable in prompts, patch operations, scoring diagnostics, and future plugin rendering code.

| Patchex primitive | Production meaning | Free implementation target |
| --- | --- | --- |
| `synth.engine = "vital"` | Real synth tone, not samples | Vital AU |
| `effects.tempo_delay` | Sparkly synced repeats between notes | Voxengo Tempo Delay, Tymee, Valhalla Supermassive, or internal DSP |
| `effects.sidechain_pump` | Kick-driven breathing | STFU, DuckTool, Ableton Compressor/Utility, or internal DSP |
| `effects.step_gate` | Rhythmic chops without changing MIDI | STFU, DuckTool, Auto Pan, or internal DSP |
| `effects.filter_sequencer` | Patterned cutoff motion | Filterstep, Auto Filter LFO, Vital modulation, or internal DSP |
| `effects.juno_chorus` | Juno-style width/ensemble | OSL Chorus, TAL-Chorus-LX, or internal DSP |
| `returns.reverb` | Big shared plate/hall space | Valhalla Supermassive, TAL-Reverb-4, or internal DSP |
| `master.saturation` | Mild glue and harmonic density | Mesa Lite, Ableton Saturator, or internal DSP |
| `master.compression_*` | Bus compression/glue/pump | DC1A, Kotelnikov, Ableton Compressor, or internal DSP |
| `master.movement` | Subtle phaser/flanger/title-wide motion | Blue Cat Phaser/Flanger or internal DSP |

## Prompting Rules

- If MIDI is sparse but the reference has lots of onsets, treat it as a texture/effects problem. Use delay, sidechain, step gate, filter sequencing, and LFOs. Do not invent notes.
- Prefer named primitives over raw automation. Use `gain_points` only for simple fades, one duck, or a phrase-level correction.
- The first question should be "what production device explains this sound?" not "what curve can fit the RMS envelope?"
- For Between the Buttons-style first-five-second work, start from Vital saw/pulse into low-pass, modest resonance, per-note filter envelope, Juno chorus, dotted/eighth filtered delay, restrained hall/plate send, sidechain pump, and mild saturation/compression.

## Source Links

- Vital: https://vital.audio/
- Voxengo Tempo Delay: https://www.voxengo.com/product/tempodelay/
- Tritik Tymee: https://www.tritik.com/product/tymee/
- Valhalla Supermassive: https://valhalladsp.com/shop/reverb/valhalla-supermassive/
- ZEEK STFU: https://zeeks.app/
- DuckTool: https://dsgdnb.com/plugins/ducktool
- OSL Chorus: https://oblivionsoundlab.com/product/osl-chorus/
- Cableguys PanCake: https://www.cableguys.com/pancake.html
- Venomode Mesa Lite: https://venomode.com/mesa-lite
- Klanghelm DC1A: https://klanghelm.com/contents/products/DC1A.html
- TDR Nova: https://www.tokyodawn.net/tdr-nova/
- Blue Cat Flanger: https://www.bluecataudio.com/Products/Product_Flanger/
- Blue Cat Phaser: https://www.bluecataudio.com/Products/Product_Phaser/
