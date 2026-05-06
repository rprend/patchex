#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def adsr_envelope(
    total_samples: int,
    sample_rate: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    gate_seconds: float,
) -> np.ndarray:
    gate_samples = min(total_samples, int(gate_seconds * sample_rate))
    release_samples = min(total_samples - gate_samples, int(release * sample_rate))
    held_samples = total_samples - release_samples

    attack_samples = min(held_samples, int(attack * sample_rate))
    decay_samples = min(max(0, held_samples - attack_samples), int(decay * sample_rate))
    sustain_samples = max(0, held_samples - attack_samples - decay_samples)

    parts: list[np.ndarray] = []
    if attack_samples:
        parts.append(np.linspace(0.0, 1.0, attack_samples, endpoint=False))
    if decay_samples:
        parts.append(np.linspace(1.0, sustain, decay_samples, endpoint=False))
    if sustain_samples:
        parts.append(np.full(sustain_samples, sustain))

    held = np.concatenate(parts) if parts else np.zeros(0)
    release_start = held[-1] if held.size else 0.0
    if release_samples:
        release_part = np.linspace(release_start, 0.0, release_samples, endpoint=True)
        env = np.concatenate([held, release_part])
    else:
        env = held

    if env.size < total_samples:
        env = np.pad(env, (0, total_samples - env.size))
    return env[:total_samples]


def sine_square_morph(phase: np.ndarray, morph: float) -> np.ndarray:
    sine = np.sin(phase)
    square = np.sign(sine)
    square[square == 0] = 1.0
    signal = (1.0 - morph) * sine + morph * square
    peak = np.max(np.abs(signal))
    return signal / peak if peak else signal


def render_synth(
    note: int,
    seconds: float,
    sample_rate: int,
    morph: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    gate_seconds: float,
) -> np.ndarray:
    samples = int(seconds * sample_rate)
    if samples <= 0:
        raise ValueError(f"Cannot render synth with non-positive duration: seconds={seconds}, sample_rate={sample_rate}.")
    t = np.arange(samples) / sample_rate
    frequency = midi_to_hz(note)
    phase = 2.0 * np.pi * frequency * t

    oscillator = sine_square_morph(phase, np.clip(morph, 0.0, 1.0))
    sub = 0.25 * np.sin(phase * 0.5)
    envelope = adsr_envelope(samples, sample_rate, attack, decay, sustain, release, gate_seconds)
    audio = (oscillator + sub) * envelope * 0.28
    return audio.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--note", type=int, default=45, help="MIDI note number")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--sine-square", type=float, default=0.35, help="0=sine, 1=square")
    parser.add_argument("--attack", type=float, default=0.02)
    parser.add_argument("--decay", type=float, default=0.25)
    parser.add_argument("--sustain", type=float, default=0.65)
    parser.add_argument("--release", type=float, default=0.75)
    parser.add_argument("--gate", type=float, default=2.8)
    args = parser.parse_args()

    audio = render_synth(
        note=args.note,
        seconds=args.seconds,
        sample_rate=args.sample_rate,
        morph=args.sine_square,
        attack=args.attack,
        decay=args.decay,
        sustain=args.sustain,
        release=args.release,
        gate_seconds=args.gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, args.sample_rate)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
