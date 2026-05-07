# Text2FX Gemini Prototype

This is a small hybrid Text2FX-style pipeline:

1. Gemini LLM proposes initial EQ/reverb/compression/saturation settings.
2. Local audio effects render the candidate audio.
3. Gemini Embedding 2 embeds both the text prompt and rendered audio.
4. Cosine similarity scores prompt/audio alignment.
5. Local refinement perturbs the initial preset and keeps the best-scoring render.

## Core Idea

Text2FX needs a model that places text prompts and audio clips in the same
embedding space. Gemini Embedding 2 is suitable for that because Google describes
it as a multimodal embedding model for text, images, video, audio, and documents.

Unlike the CLAP paper implementation, this prototype uses gradient-free
refinement:

1. Ask the LLM for a conservative initial preset.
2. Render that preset.
3. Render nearby randomized variations.
4. Embed and score each render.
5. Save the highest-scoring WAV and JSON sidecar.

This works with black-box effects and remote embedding APIs. It is slower than
differentiable DSP, but simpler and closer to how a real plugin/render pipeline
will behave.

## Recommended Minimal Effects

Start with this effect set:

- `eq`: 3-band tone shaping, implemented with low-shelf, peak, and high-shelf EQ.
- `reverb`: space/depth.
- `saturation`: warmth, grit, density.
- `delay`: rhythmic echo and width.
- `compressor`: punch, sustain, dynamic control.
- `limiter`: safety output stage.

The current script includes all six stages in the render chain. The LLM proposes
all exposed parameters, and refinement searches around that proposal.

## Recommended Synth Setup

Keep synthesis separate from Text2FX at first.

Smallest useful synth palette:

- Wavetable/subtractive synth for basses, leads, pads, plucks.
- Sampler for drums, vocals, acoustic instruments, one-shots, and reference audio.
- Simple drum synth or drum sampler for kick/snare/hat.

The Text2FX loop should operate on rendered audio stems, not MIDI directly. That
keeps the optimizer focused on sound transformation instead of composition.

This folder includes a tiny source synth for testing:

```bash
python text2fx_gemini/synth.py \
  --output text2fx_gemini/synth_input.wav \
  --note 45 \
  --sine-square 0.35 \
  --attack 0.02 \
  --decay 0.25 \
  --sustain 0.65 \
  --release 0.75
```

Synth parameters:

- `--sine-square`: oscillator morph, where `0` is sine and `1` is square.
- `--attack`: seconds to rise from silence to full level.
- `--decay`: seconds to fall from full level to sustain level.
- `--sustain`: held level from `0` to `1`.
- `--release`: seconds to fade after note-off.
- `--gate`: note length before release starts.

This is not a full wavetable synth yet. It is a minimal oscillator/envelope test
source that makes the Text2FX pipeline easier to evaluate.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r text2fx_gemini/requirements.txt
```

You need a Gemini API key in the environment. Use the secret helper rather than
pasting secrets into chat:

```bash
~/.codex/scripts/ask-secret.sh GEMINI_API_KEY ~/.codex/secrets.env "Enter Gemini API key"
set -a; source ~/.codex/secrets.env; set +a
```

Do not commit `.env` or secret files.

## Usage

```bash
python text2fx_gemini/text2fx.py \
  --input input.wav \
  --prompt "make it warmer and more spacious" \
  --output output.wav \
  --refine-trials 24 \
  --random-trials 8
```

The script writes the best rendered WAV and a JSON sidecar containing:

- the LLM proposal
- the best score
- the best effect parameters
- the full candidate history

To run only the LLM proposal and score it without local refinement:

```bash
python text2fx_gemini/text2fx.py \
  --input input.wav \
  --prompt "make it warmer and more spacious" \
  --output output.wav \
  --no-refine
```

## Model Environment Variables

Defaults:

```bash
GEMINI_LLM_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
```

Override these if your account exposes different model IDs.

## Reference Matching Pipeline

The full reference-matching loop is implemented in:

```bash
text2fx_gemini/reference_match.py
```

It does:

1. Analyze a reference WAV into description axes and DSP features.
2. Choose an instrument type such as `bass_synth`, `lead_synth`,
   `pad_synth`, `pluck_synth`, or `arp_synth`.
3. Ask the LLM for an initial synth/pattern/effects recipe.
4. Run candidate families for each description axis.
5. Render the recipe's step pattern through the synth/effect chain.
6. Score candidates against the reference using audio-audio embedding,
   audio-text embedding, and DSP feature distance.
7. Synthesize the top candidates into a final playable patch and render.

Run with Gemini:

```bash
. .venv/bin/activate
set -a; source ~/.codex/secrets.env; set +a

python text2fx_gemini/reference_match.py \
  --reference reference_5s.wav \
  --prompt "warm rounded pulsing foreground synth hook" \
  --output-dir text2fx_gemini/reference_run \
  --candidates 8 \
  --axis-trials 2
```

The reference matcher intentionally fails loudly. It requires Gemini proposal,
Gemini embedding scoring, and Codex CLI synthesis. If any required service fails,
the run exits instead of producing a degraded patch.

Outputs:

- `candidate_*.wav` - rendered candidate patches.
- `final_synthesized.wav` - merged recipe from the top candidates.
- `playable_patch.json` - keyboard-playable patch recipe.
- `match_report.json` - axes, features, scores, recipes, and final recipe.

Render a playable patch at a different note:

```bash
python text2fx_gemini/play_patch.py \
  --patch text2fx_gemini/reference_run/playable_patch.json \
  --note 64 \
  --velocity 0.9 \
  --output text2fx_gemini/reference_run/note_64.wav
```

Render with macro overrides:

```bash
python text2fx_gemini/play_patch.py \
  --patch text2fx_gemini/reference_run/playable_patch.json \
  --note 52 \
  --brightness 0.35 \
  --warmth 0.8 \
  --crunch 0.4 \
  --space 0.25 \
  --output text2fx_gemini/reference_run/note_52_warm.wav
```

## MIDI-Locked Patch Finding

Use this path when the arrangement is known from MIDI and the agent should only
search synth, effects, modulation, and mix settings.

```bash
python text2fx_gemini/midi_locked_patch.py import-midi \
  --midi ~/Downloads/french-79-between-the-buttons-20240211083935-nonstop2k.com.mid \
  --output text2fx_gemini/midi_locked_run/arrangement.json

python text2fx_gemini/midi_locked_patch.py neutral-session \
  --arrangement text2fx_gemini/midi_locked_run/arrangement.json \
  --output text2fx_gemini/midi_locked_run/patch_session.json \
  --seconds 5

python text2fx_gemini/midi_locked_patch.py score-session \
  --arrangement text2fx_gemini/midi_locked_run/arrangement.json \
  --session text2fx_gemini/midi_locked_run/patch_session.json \
  --reference reference_5s.wav \
  --output text2fx_gemini/midi_locked_run/patch_report.json \
  --render-output text2fx_gemini/midi_locked_run/patch_render.wav \
  --seconds 5
```

The report includes global full-mix similarity, per-track active-window scores,
solo-track isolation proxy scores, patch-control diagnostics, and an arrangement
preservation penalty. Any note, timing, or velocity change makes the preservation
penalty nonzero.

The first canonical song workspace is:

```bash
text2fx_gemini/songs/between_the_buttons/
```

It contains `source.mp3`, `source.mid`, and full-song `arrangement.json`.
When the V1 UI reconstructs a five-second clip from "French 79 Between the
Buttons" and that song MIDI exists, the backend automatically uses the
MIDI-locked patch workflow and slices the MIDI to the selected five-second
audio region before the Producer starts.
