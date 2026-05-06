#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference_match import Recipe, SynthParams, PatternParams, runtime
from text2fx import FxParams, render_audio
from synth import render_synth


def load_patch(path: Path) -> Recipe:
    payload = json.loads(path.read_text())
    return Recipe(
        instrument_type=payload.get("instrument_type", "lead_synth"),
        synth=SynthParams(**payload["synth"]),
        pattern=PatternParams(**payload["pattern"]),
        effects=FxParams(**payload["effects"]),
        macros=payload.get("macros", {}),
        keyboard_mapping=payload.get("keyboard_mapping", {}),
    )


def apply_macros(recipe: Recipe, brightness: float | None, warmth: float | None, crunch: float | None, space: float | None) -> Recipe:
    synth = SynthParams(**recipe.synth.__dict__)
    effects = FxParams(**recipe.effects.__dict__)
    if brightness is not None:
        synth.sine_square = float(np.clip(0.15 + 0.85 * brightness, 0, 1))
        effects.high_gain_db = float(np.interp(brightness, [0, 1], [-6, 6]))
    if warmth is not None:
        effects.low_gain_db = float(np.interp(warmth, [0, 1], [-4, 6]))
    if crunch is not None:
        effects.saturation_drive_db = float(np.interp(crunch, [0, 1], [0, 18]))
    if space is not None:
        effects.reverb_room_size = float(np.interp(space, [0, 1], [0.05, 0.85]))
        effects.reverb_wet_level = float(np.interp(space, [0, 1], [0.0, 0.28]))
    return Recipe(
        instrument_type=recipe.instrument_type,
        synth=synth,
        pattern=recipe.pattern,
        effects=effects,
        macros=recipe.macros,
        keyboard_mapping=recipe.keyboard_mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--velocity", type=float, default=0.85)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--brightness", type=float)
    parser.add_argument("--warmth", type=float)
    parser.add_argument("--crunch", type=float)
    parser.add_argument("--space", type=float)
    args = parser.parse_args()

    runtime()
    import soundfile as sf

    recipe = load_patch(args.patch)
    recipe = apply_macros(recipe, args.brightness, args.warmth, args.crunch, args.space)
    audio = render_synth(
        note=args.note,
        seconds=args.seconds,
        sample_rate=args.sample_rate,
        morph=recipe.synth.sine_square,
        attack=recipe.synth.attack,
        decay=recipe.synth.decay,
        sustain=recipe.synth.sustain,
        release=recipe.synth.release,
        gate_seconds=min(recipe.synth.gate, args.seconds * 0.8),
    ) * float(np.clip(args.velocity, 0, 1))
    rendered = render_audio(audio[None, :], args.sample_rate, recipe.effects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, rendered.T, args.sample_rate)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
