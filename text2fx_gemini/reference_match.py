#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=r"google(\.|$)")

from synth import render_synth
from text2fx import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    FxParams,
    cosine,
    embed_text,
    embed_wav,
    extract_json_object,
    load_runtime_dependencies,
    render_audio,
    sanitize_params,
)

sf: Any = None
genai: Any = None


@dataclass
class SynthParams:
    note: int = 45
    sine_square: float = 0.35
    attack: float = 0.02
    decay: float = 0.25
    sustain: float = 0.65
    release: float = 0.45
    gate: float = 0.45


@dataclass
class PatternParams:
    tempo: float = 118.0
    grid: str = "16th"
    steps: list[int] | None = None
    velocity: list[float] | None = None


@dataclass
class Recipe:
    instrument_type: str
    synth: SynthParams
    pattern: PatternParams
    effects: FxParams
    macros: dict[str, float] | None = None
    keyboard_mapping: dict[str, Any] | None = None


BAND_NAMES = ("low", "mid", "high")


def runtime() -> None:
    global sf, genai
    load_runtime_dependencies()
    import soundfile as soundfile_module
    from google import genai as genai_module

    sf = soundfile_module
    genai = genai_module


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.T.astype(np.float32), sample_rate


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def spectral_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    y = mono(audio)
    if y.size == 0:
        raise ValueError("Audio is empty.")
    window = np.hanning(y.size)
    spectrum = np.abs(np.fft.rfft(y * window)) + 1e-12
    freqs = np.fft.rfftfreq(y.size, 1.0 / sample_rate)
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total)
    cumulative = np.cumsum(spectrum)
    rolloff = float(freqs[np.searchsorted(cumulative, cumulative[-1] * 0.85)])
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spectrum[mask]) / total)

    return {
        "rms": rms(y),
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(y))))),
        "low_energy": band(20, 250),
        "mid_energy": band(250, 2500),
        "high_energy": band(2500, sample_rate / 2),
    }


def envelope_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    y = np.abs(mono(audio))
    frame = max(128, int(sample_rate * 0.01))
    usable = y[: (y.size // frame) * frame]
    if usable.size == 0:
        return {"transient_strength": 0.0, "onset_density": 0.0}
    env = usable.reshape(-1, frame).mean(axis=1)
    diff = np.maximum(0, np.diff(env))
    threshold = float(np.mean(diff) + np.std(diff))
    onsets = int(np.sum(diff > threshold))
    return {
        "transient_strength": float(np.max(diff) / (np.mean(env) + 1e-9)),
        "onset_density": float(onsets / max(1.0, y.size / sample_rate)),
    }


def analyze_reference(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    features = spectral_features(audio, sample_rate)
    features.update(envelope_features(audio, sample_rate))
    axes = {
        "spectral_tone": [],
        "harmonic_texture": [],
        "envelope": [],
        "space": [],
        "motion": [],
        "rhythm": [],
        "mix_role": [],
    }

    if features["spectral_centroid"] < 900:
        axes["spectral_tone"].extend(["dark", "warm"])
    elif features["spectral_centroid"] > 2200:
        axes["spectral_tone"].extend(["bright", "airy"])
    else:
        axes["spectral_tone"].append("balanced")

    if features["low_energy"] > 0.35:
        axes["spectral_tone"].append("full-bodied")
    if features["high_energy"] < 0.12:
        axes["spectral_tone"].append("soft highs")
    if features["spectral_flatness"] > 0.18:
        axes["harmonic_texture"].extend(["noisy", "gritty"])
    elif features["zero_crossing_rate"] > 0.1:
        axes["harmonic_texture"].append("crunchy")
    else:
        axes["harmonic_texture"].extend(["smooth", "rounded"])

    if features["transient_strength"] > 4.0:
        axes["envelope"].extend(["sharp attack", "percussive"])
    else:
        axes["envelope"].extend(["soft attack", "sustained"])

    if features["onset_density"] > 4.0:
        axes["rhythm"].extend(["dense", "driving"])
    elif features["onset_density"] > 1.0:
        axes["rhythm"].extend(["repetitive", "pulsing"])
    else:
        axes["rhythm"].append("sparse")

    axes["space"].append("unknown from mono features")
    axes["motion"].append("estimate from rendered candidates")
    axes["mix_role"].append("foreground synth candidate")
    instrument_type = choose_instrument_type(features, axes)
    return {"features": features, "axes": axes, "instrument_type": instrument_type}


def choose_instrument_type(features: dict[str, float], axes: dict[str, list[str]]) -> str:
    if features["low_energy"] > 0.55 and features["spectral_centroid"] < 900:
        return "bass_synth"
    if features["onset_density"] > 8.0:
        return "arp_synth"
    if "soft attack" in axes["envelope"] and features["onset_density"] < 2.0:
        return "pad_synth"
    if "sharp attack" in axes["envelope"]:
        return "pluck_synth"
    return "lead_synth"


def phrase_summary(axes: dict[str, list[str]]) -> str:
    return "; ".join(f"{name}: {', '.join(values)}" for name, values in axes.items())


def sanitize_recipe_payload(payload: dict[str, Any]) -> Recipe:
    required_top = {"instrument_type", "synth", "pattern", "effects", "macros", "keyboard_mapping"}
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        raise ValueError(f"Recipe missing required top-level keys: {missing_top}")
    instrument_type = str(payload["instrument_type"])
    if instrument_type not in {"bass_synth", "lead_synth", "pad_synth", "pluck_synth", "arp_synth", "texture_synth"}:
        raise ValueError(f"Invalid instrument_type: {instrument_type}")
    synth_payload = payload["synth"]
    pattern_payload = payload["pattern"]
    effects_payload = payload["effects"]
    required_synth = {"note", "sine_square", "attack", "decay", "sustain", "release", "gate"}
    required_pattern = {"tempo", "grid", "steps", "velocity"}
    required_effects = {
        "low_gain_db",
        "mid_gain_db",
        "high_gain_db",
        "reverb_room_size",
        "reverb_wet_level",
        "saturation_drive_db",
        "delay_mix",
        "compressor_threshold_db",
    }
    for name, required, section in [
        ("synth", required_synth, synth_payload),
        ("pattern", required_pattern, pattern_payload),
        ("effects", required_effects, effects_payload),
    ]:
        missing = sorted(required - set(section))
        if missing:
            raise ValueError(f"Recipe {name} missing required keys: {missing}")
    synth = SynthParams(
        note=int(np.clip(float(synth_payload["note"]), 24, 84)),
        sine_square=float(np.clip(float(synth_payload["sine_square"]), 0, 1)),
        attack=float(np.clip(float(synth_payload["attack"]), 0.001, 2.0)),
        decay=float(np.clip(float(synth_payload["decay"]), 0.001, 2.0)),
        sustain=float(np.clip(float(synth_payload["sustain"]), 0, 1)),
        release=float(np.clip(float(synth_payload["release"]), 0.001, 3.0)),
        gate=float(np.clip(float(synth_payload["gate"]), 0.05, 2.0)),
    )
    steps = pattern_payload["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Recipe pattern.steps must be a non-empty list.")
    steps = [int(np.clip(float(step), -24, 24)) for step in steps[:32]]
    velocity = pattern_payload["velocity"]
    if not isinstance(velocity, list) or len(velocity) != len(steps):
        raise ValueError("Recipe pattern.velocity must be a list with the same length as steps.")
    velocity = [float(np.clip(float(v), 0, 1)) for v in velocity[: len(steps)]]
    pattern = PatternParams(
        tempo=float(np.clip(float(pattern_payload["tempo"]), 60, 180)),
        grid=str(pattern_payload["grid"]),
        steps=steps,
        velocity=velocity,
    )
    effects = sanitize_params(
        FxParams(
            low_gain_db=float(effects_payload["low_gain_db"]),
            mid_gain_db=float(effects_payload["mid_gain_db"]),
            high_gain_db=float(effects_payload["high_gain_db"]),
            reverb_room_size=float(effects_payload["reverb_room_size"]),
            reverb_wet_level=float(effects_payload["reverb_wet_level"]),
            saturation_drive_db=float(effects_payload["saturation_drive_db"]),
            delay_mix=float(effects_payload["delay_mix"]),
            compressor_threshold_db=float(effects_payload["compressor_threshold_db"]),
        )
    )
    macros = payload["macros"]
    keyboard_mapping = payload["keyboard_mapping"]
    if not isinstance(macros, dict):
        raise ValueError("Recipe macros must be an object.")
    if not isinstance(keyboard_mapping, dict):
        raise ValueError("Recipe keyboard_mapping must be an object.")
    return Recipe(instrument_type=instrument_type, synth=synth, pattern=pattern, effects=effects, macros=macros, keyboard_mapping=keyboard_mapping)


def with_fixed_pattern(recipe: Recipe, pattern: PatternParams) -> Recipe:
    return Recipe(
        instrument_type=recipe.instrument_type,
        synth=recipe.synth,
        pattern=PatternParams(**asdict(pattern)),
        effects=recipe.effects,
        macros=recipe.macros or default_macros(recipe.synth, recipe.effects),
        keyboard_mapping=recipe.keyboard_mapping or default_keyboard_mapping(),
    )


def fixed_pattern_from_analysis(analysis: dict[str, Any]) -> PatternParams:
    payload = analysis.get("fixed_pattern")
    if not isinstance(payload, dict):
        raise ValueError("Analysis is missing AI-generated fixed_pattern. Analyze the clip again before building.")
    steps = payload.get("steps")
    velocity = payload.get("velocity")
    if not isinstance(steps, list) or len(steps) != 16:
        raise ValueError("Analysis fixed_pattern.steps must contain exactly 16 values.")
    if not isinstance(velocity, list) or len(velocity) != 16:
        raise ValueError("Analysis fixed_pattern.velocity must contain exactly 16 values.")
    return PatternParams(
        tempo=float(np.clip(float(payload["tempo"]), 60, 180)),
        grid="16th",
        steps=[int(np.clip(float(step), -24, 24)) for step in steps],
        velocity=[float(np.clip(float(v), 0, 1)) for v in velocity],
    )


def default_macros(synth: SynthParams, effects: FxParams) -> dict[str, float]:
    return {
        "brightness": float(np.clip(0.5 * synth.sine_square + (effects.high_gain_db + 9) / 36, 0, 1)),
        "warmth": float(np.clip((effects.low_gain_db + 9) / 18, 0, 1)),
        "crunch": float(np.clip(effects.saturation_drive_db / 18, 0, 1)),
        "space": float(np.clip(0.5 * effects.reverb_room_size + effects.reverb_wet_level / 0.7, 0, 1)),
        "pluck": float(np.clip(1.0 - synth.attack * 5.0, 0, 1)),
    }


def default_keyboard_mapping() -> dict[str, Any]:
    return {
        "root_midi_note": 60,
        "pitch_tracking": True,
        "velocity_to_amp": True,
        "mod_wheel": "brightness",
        "aftertouch": "space",
    }


def propose_recipe(client: Any, axes: dict[str, list[str]], features: dict[str, float], instrument_type: str) -> Recipe:
    instruction = f"""
You are designing a synth patch, note pattern, and effects chain to match a five-second reference.

Description axes:
{json.dumps(axes, indent=2)}

Chosen instrument type: {instrument_type}

Measured features:
{json.dumps(features, indent=2)}

Return only JSON with this shape:
{{
  "instrument_type": "{instrument_type}",
  "synth": {{
    "note": 24-84 MIDI integer,
    "sine_square": 0-1,
    "attack": seconds,
    "decay": seconds,
    "sustain": 0-1,
    "release": seconds,
    "gate": seconds
  }},
  "pattern": {{
    "tempo": 60-180,
    "grid": "16th",
    "steps": list of semitone offsets from root,
    "velocity": list of 0-1 values same length as steps
  }},
  "effects": {{
    "low_gain_db": -9..9,
    "mid_gain_db": -9..9,
    "high_gain_db": -9..9,
    "reverb_room_size": 0..1,
    "reverb_wet_level": 0..0.35,
    "saturation_drive_db": 0..18,
    "delay_mix": 0..0.25,
    "compressor_threshold_db": -28..-8
  }},
  "macros": {{
    "brightness": 0-1,
    "warmth": 0-1,
    "crunch": 0-1,
    "space": 0-1,
    "pluck": 0-1
  }},
  "keyboard_mapping": {{
    "root_midi_note": 60,
    "pitch_tracking": true,
    "velocity_to_amp": true,
    "mod_wheel": "brightness",
    "aftertouch": "space"
  }}
}}
"""
    response = generate_content_with_retry(client, instruction)
    return sanitize_recipe_payload(extract_json_object(response.text or ""))


def generate_content_with_retry(client: Any, instruction: str, attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.models.generate_content(model=LLM_MODEL, contents=instruction)
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            sleep_seconds = 2 ** attempt
            print(f"Gemini generate_content failed; retrying in {sleep_seconds}s: {exc}", flush=True)
            time.sleep(sleep_seconds)
    raise RuntimeError("Gemini generate_content failed after retries.") from last_error


def mutate_recipe(recipe: Recipe, rng: np.random.Generator, amount: float) -> Recipe:
    synth = SynthParams(
        note=int(np.clip(recipe.synth.note + rng.choice([-12, -7, 0, 7, 12]), 24, 84)),
        sine_square=float(np.clip(recipe.synth.sine_square + rng.normal(0, amount), 0, 1)),
        attack=float(np.clip(recipe.synth.attack * np.exp(rng.normal(0, amount)), 0.001, 2.0)),
        decay=float(np.clip(recipe.synth.decay * np.exp(rng.normal(0, amount)), 0.001, 2.0)),
        sustain=float(np.clip(recipe.synth.sustain + rng.normal(0, amount), 0, 1)),
        release=float(np.clip(recipe.synth.release * np.exp(rng.normal(0, amount)), 0.001, 3.0)),
        gate=float(np.clip(recipe.synth.gate * np.exp(rng.normal(0, amount)), 0.05, 2.0)),
    )
    fx = recipe.effects
    effects = sanitize_params(
        FxParams(
            low_gain_db=fx.low_gain_db + rng.normal(0, 3 * amount),
            mid_gain_db=fx.mid_gain_db + rng.normal(0, 3 * amount),
            high_gain_db=fx.high_gain_db + rng.normal(0, 3 * amount),
            reverb_room_size=fx.reverb_room_size + rng.normal(0, amount),
            reverb_wet_level=fx.reverb_wet_level + rng.normal(0, 0.08 * amount),
            saturation_drive_db=fx.saturation_drive_db + rng.normal(0, 5 * amount),
            delay_mix=fx.delay_mix + rng.normal(0, 0.08 * amount),
            compressor_threshold_db=fx.compressor_threshold_db + rng.normal(0, 5 * amount),
        )
    )
    return Recipe(
        instrument_type=recipe.instrument_type,
        synth=synth,
        pattern=recipe.pattern,
        effects=effects,
        macros=default_macros(synth, effects),
        keyboard_mapping=recipe.keyboard_mapping or default_keyboard_mapping(),
    )


def clone_recipe(recipe: Recipe) -> Recipe:
    return Recipe(
        instrument_type=recipe.instrument_type,
        synth=SynthParams(**asdict(recipe.synth)),
        pattern=PatternParams(**asdict(recipe.pattern)),
        effects=FxParams(**asdict(recipe.effects)),
        macros=dict(recipe.macros or default_macros(recipe.synth, recipe.effects)),
        keyboard_mapping=dict(recipe.keyboard_mapping or default_keyboard_mapping()),
    )


def render_recipe(recipe: Recipe, seconds: float, sample_rate: int) -> np.ndarray:
    dry = render_pattern(recipe, seconds, sample_rate)
    return render_audio(dry[None, :], sample_rate, recipe.effects)


def render_pattern(recipe: Recipe, seconds: float, sample_rate: int, transpose: int = 0) -> np.ndarray:
    pattern = recipe.pattern
    if not pattern.steps:
        raise ValueError("Cannot render pattern without steps.")
    if not pattern.velocity or len(pattern.velocity) != len(pattern.steps):
        raise ValueError("Cannot render pattern without velocity values matching steps.")
    steps = pattern.steps
    velocity = pattern.velocity
    step_duration = step_duration_seconds(pattern.tempo, pattern.grid)
    total_samples = int(seconds * sample_rate)
    out = np.zeros(total_samples, dtype=np.float32)
    step_index = 0
    start_time = 0.0
    while start_time < seconds:
        semitone = steps[step_index % len(steps)]
        vel = velocity[step_index % len(velocity)]
        note = recipe.synth.note + semitone + transpose
        gate = min(recipe.synth.gate * step_duration, step_duration * 0.95)
        note_seconds = min(step_duration, seconds - start_time)
        if note_seconds * sample_rate < 1:
            break
        rendered = render_synth(
            note=note,
            seconds=note_seconds,
            sample_rate=sample_rate,
            morph=recipe.synth.sine_square,
            attack=recipe.synth.attack,
            decay=recipe.synth.decay,
            sustain=recipe.synth.sustain,
            release=min(recipe.synth.release, max(0.001, note_seconds - gate)),
            gate_seconds=gate,
        ) * float(vel)
        if rendered.size == 0:
            raise ValueError(f"Synth rendered zero samples for note_seconds={note_seconds}, tempo={pattern.tempo}, grid={pattern.grid}.")
        start = int(start_time * sample_rate)
        end = min(total_samples, start + rendered.size)
        out[start:end] += rendered[: end - start]
        start_time += step_duration
        step_index += 1
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out * (0.8 / peak) if peak > 0.8 else out


def step_duration_seconds(tempo: float, grid: str) -> float:
    quarter = 60.0 / max(1.0, tempo)
    normalized = grid.lower()
    if normalized == "32nd":
        return quarter / 8.0
    if normalized == "8th":
        return quarter / 2.0
    return quarter / 4.0


def feature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = ["spectral_centroid", "spectral_rolloff", "spectral_flatness", "low_energy", "mid_energy", "high_energy", "rms"]
    total = 0.0
    for key in keys:
        scale = max(abs(a[key]), abs(b[key]), 1e-6)
        total += abs(a[key] - b[key]) / scale
    return total / len(keys)


def score_candidate(
    client: Any,
    reference_embedding: np.ndarray,
    text_embedding: np.ndarray,
    reference_features: dict[str, float],
    candidate_path: Path,
) -> dict[str, float]:
    candidate_audio, sr = load_audio(candidate_path)
    candidate_features = spectral_features(candidate_audio, sr)
    candidate_embedding = embed_wav(client, candidate_path)
    audio_audio = cosine(reference_embedding, candidate_embedding)
    audio_text = cosine(text_embedding, candidate_embedding)
    feature_score = math.exp(-feature_distance(reference_features, candidate_features))
    final = 0.45 * audio_audio + 0.25 * audio_text + 0.30 * feature_score
    return {
        "final": float(final),
        "audio_audio": float(audio_audio),
        "audio_text": float(audio_text),
        "feature": float(feature_score),
    }


def benchmark_loss(scores: dict[str, float]) -> float:
    return float(1.0 - scores["final"])


def run_iterative_codex_candidate(
    output_dir: Path,
    client: Any,
    analysis: dict[str, Any],
    target_prompt: str,
    instrument_type: str,
    fixed_pattern: PatternParams,
    candidate_index: int,
    axis: str,
    objective: str,
    iterations: int,
    seconds: float,
    sample_rate: int,
    reference_embedding: np.ndarray,
    text_embedding: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if iterations < 1:
        raise ValueError("Candidate iterations must be at least 1.")
    attempts: list[dict[str, Any]] = []
    codex_runs: list[dict[str, Any]] = []
    prior_recipe: Recipe | None = None
    benchmark_history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        recipe, codex_run = codex_generate_candidate_recipe(
            output_dir=output_dir,
            analysis=analysis,
            target_prompt=target_prompt,
            instrument_type=instrument_type,
            fixed_pattern=fixed_pattern,
            candidate_index=candidate_index,
            axis=axis,
            objective=objective,
            prior_recipe=prior_recipe,
            iteration=iteration,
            benchmark_history=benchmark_history,
        )
        attempt_path = output_dir / f"candidate_{candidate_index:03d}_iter_{iteration:02d}.wav"
        rendered = render_recipe(recipe, seconds, sample_rate)
        sf.write(attempt_path, rendered.T, sample_rate)
        scores = score_candidate(client, reference_embedding, text_embedding, analysis["features"], attempt_path)
        loss = benchmark_loss(scores)
        attempt = {
            "candidate_index": candidate_index,
            "iteration": iteration,
            "axis": axis,
            "path": str(attempt_path),
            "scores": scores,
            "loss": loss,
            "recipe": recipe,
        }
        attempts.append(attempt)
        codex_runs.append(codex_run)
        benchmark_history.append(
            {
                "iteration": iteration,
                "loss": loss,
                "scores": scores,
                "recipe": recipe_to_json(recipe),
                "audio_path": str(attempt_path),
            }
        )
        prior_recipe = recipe
        print(f"candidate={candidate_index} axis={axis} iteration={iteration} loss={loss:.4f} score={scores['final']:.4f}", flush=True)
    best = min(attempts, key=lambda item: item["loss"])
    final_path = output_dir / f"candidate_{candidate_index:03d}.wav"
    best_audio, best_sr = load_audio(Path(best["path"]))
    sf.write(final_path, best_audio.T, best_sr)
    return {
        "index": candidate_index,
        "axis": axis,
        "path": str(final_path),
        "scores": best["scores"],
        "loss": best["loss"],
        "recipe": best["recipe"],
        "iterations": [
            {
                "iteration": item["iteration"],
                "path": item["path"],
                "scores": item["scores"],
                "loss": item["loss"],
                "recipe": recipe_to_json(item["recipe"]),
            }
            for item in attempts
        ],
    }, codex_runs


def synthesize_recipe(best: list[dict[str, Any]]) -> Recipe:
    top = best[: max(1, min(3, len(best)))]
    synth = SynthParams(
        note=int(round(np.median([item["recipe"].synth.note for item in top]))),
        sine_square=float(np.mean([item["recipe"].synth.sine_square for item in top])),
        attack=float(np.mean([item["recipe"].synth.attack for item in top])),
        decay=float(np.mean([item["recipe"].synth.decay for item in top])),
        sustain=float(np.mean([item["recipe"].synth.sustain for item in top])),
        release=float(np.mean([item["recipe"].synth.release for item in top])),
        gate=float(np.mean([item["recipe"].synth.gate for item in top])),
    )
    effects = sanitize_params(
        FxParams(
            low_gain_db=float(np.mean([item["recipe"].effects.low_gain_db for item in top])),
            mid_gain_db=float(np.mean([item["recipe"].effects.mid_gain_db for item in top])),
            high_gain_db=float(np.mean([item["recipe"].effects.high_gain_db for item in top])),
            reverb_room_size=float(np.mean([item["recipe"].effects.reverb_room_size for item in top])),
            reverb_wet_level=float(np.mean([item["recipe"].effects.reverb_wet_level for item in top])),
            saturation_drive_db=float(np.mean([item["recipe"].effects.saturation_drive_db for item in top])),
            delay_mix=float(np.mean([item["recipe"].effects.delay_mix for item in top])),
            compressor_threshold_db=float(np.mean([item["recipe"].effects.compressor_threshold_db for item in top])),
        )
    )
    return Recipe(
        instrument_type=top[0]["recipe"].instrument_type,
        synth=synth,
        pattern=top[0]["recipe"].pattern,
        effects=effects,
        macros=default_macros(synth, effects),
        keyboard_mapping=top[0]["recipe"].keyboard_mapping or default_keyboard_mapping(),
    )


def recipe_to_json(recipe: Recipe) -> dict[str, Any]:
    return {
        "instrument_type": recipe.instrument_type,
        "synth": asdict(recipe.synth),
        "pattern": asdict(recipe.pattern),
        "effects": asdict(recipe.effects),
        "macros": recipe.macros or default_macros(recipe.synth, recipe.effects),
        "keyboard_mapping": recipe.keyboard_mapping or default_keyboard_mapping(),
    }


AXIS_MUTATION_GROUPS = {
    "spectral_tone": ["sine_square", "low_gain_db", "mid_gain_db", "high_gain_db"],
    "harmonic_texture": ["sine_square", "saturation_drive_db"],
    "envelope": ["attack", "decay", "sustain", "release", "gate", "compressor_threshold_db"],
    "space": ["reverb_room_size", "reverb_wet_level", "delay_mix"],
    "motion": ["sine_square", "delay_mix", "reverb_wet_level"],
    "rhythm": ["tempo", "gate"],
    "mix_role": ["low_gain_db", "mid_gain_db", "high_gain_db", "compressor_threshold_db", "reverb_wet_level"],
}


CODEX_PATH = "/Applications/Codex.app/Contents/Resources/codex"


def require_codex_path() -> str:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    return CODEX_PATH


def codex_recipe_prompt(
    analysis: dict[str, Any],
    target_prompt: str,
    instrument_type: str,
    fixed_pattern: PatternParams,
    candidate_index: int,
    axis: str,
    objective: str,
    prior_recipe: Recipe | None,
    benchmark_history: list[dict[str, Any]] | None = None,
) -> str:
    prior = "None. Create this candidate from the analysis and target prompt."
    if prior_recipe is not None:
        prior = json.dumps(recipe_to_json(prior_recipe), indent=2)
    axis_phrases = analysis["axes"].get(axis, []) if axis in analysis.get("axes", {}) else []
    history = benchmark_history or []
    return (
        "You are producing one candidate synth patch for an audio reference matching system.\n"
        "The downstream renderer is fixed: a sine-to-square morph synth, ADSR envelope, a fixed note pattern, EQ, reverb, saturation, delay, and compression.\n"
        "Your job is to choose playable synth and effect parameters that satisfy this candidate objective.\n"
        "Do not design the musical pattern. The pattern is fixed across every candidate and every iteration; copy it exactly into the JSON.\n\n"
        "Quantitative benchmark used after each answer:\n"
        "- Render your JSON to audio with the fixed renderer.\n"
        "- Embed the rendered WAV and reference WAV with the configured audio embedding model.\n"
        "- Embed the target prompt with the configured text embedding model.\n"
        "- Compute final_score = 0.45 * audio_audio_similarity + 0.25 * audio_text_similarity + 0.30 * feature_similarity.\n"
        "- Compute loss = 1.0 - final_score. Lower loss is better.\n"
        "Use benchmark history below to make a concrete improvement, not a generic rewrite.\n\n"
        f"Candidate index: {candidate_index}\n"
        f"Candidate axis: {axis}\n"
        f"Candidate objective: {objective}\n"
        f"Target prompt: {target_prompt}\n"
        f"Chosen instrument type: {instrument_type}\n"
        f"Axis phrases for this candidate: {json.dumps(axis_phrases)}\n\n"
        f"Fixed pattern to copy exactly:\n{json.dumps(asdict(fixed_pattern), indent=2)}\n\n"
        f"Full clip analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"Prior recipe to improve:\n{prior}\n\n"
        f"Benchmark history:\n{json.dumps(history, indent=2)}\n\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        f'  "instrument_type": "{instrument_type}",\n'
        '  "synth": {\n'
        '    "note": 24-84 MIDI integer,\n'
        '    "sine_square": 0-1,\n'
        '    "attack": seconds,\n'
        '    "decay": seconds,\n'
        '    "sustain": 0-1,\n'
        '    "release": seconds,\n'
        '    "gate": seconds\n'
        "  },\n"
        '  "pattern": {\n'
        f'    "tempo": {fixed_pattern.tempo},\n'
        '    "grid": "16th",\n'
        f'    "steps": {json.dumps(fixed_pattern.steps)},\n'
        f'    "velocity": {json.dumps(fixed_pattern.velocity)}\n'
        "  },\n"
        '  "effects": {\n'
        '    "low_gain_db": -9..9,\n'
        '    "mid_gain_db": -9..9,\n'
        '    "high_gain_db": -9..9,\n'
        '    "reverb_room_size": 0..1,\n'
        '    "reverb_wet_level": 0..0.35,\n'
        '    "saturation_drive_db": 0..18,\n'
        '    "delay_mix": 0..0.25,\n'
        '    "compressor_threshold_db": -28..-8\n'
        "  },\n"
        '  "macros": {\n'
        '    "brightness": 0-1,\n'
        '    "warmth": 0-1,\n'
        '    "crunch": 0-1,\n'
        '    "space": 0-1,\n'
        '    "pluck": 0-1\n'
        "  },\n"
        '  "keyboard_mapping": {\n'
        '    "root_midi_note": 60,\n'
        '    "pitch_tracking": true,\n'
        '    "velocity_to_amp": true,\n'
        '    "mod_wheel": "brightness",\n'
        '    "aftertouch": "space"\n'
        "  }\n"
        "}\n"
    )


def codex_generate_candidate_recipe(
    output_dir: Path,
    analysis: dict[str, Any],
    target_prompt: str,
    instrument_type: str,
    fixed_pattern: PatternParams,
    candidate_index: int,
    axis: str,
    objective: str,
    prior_recipe: Recipe | None,
    iteration: int,
    benchmark_history: list[dict[str, Any]],
) -> tuple[Recipe, dict[str, Any]]:
    codex_path = require_codex_path()
    prompt_path = output_dir / f"codex_candidate_{candidate_index:03d}_{axis}_iter_{iteration:02d}_prompt.txt"
    answer_path = output_dir / f"codex_candidate_{candidate_index:03d}_{axis}_iter_{iteration:02d}_answer.txt"
    prompt = codex_recipe_prompt(analysis, target_prompt, instrument_type, fixed_pattern, candidate_index, axis, objective, prior_recipe, benchmark_history)
    prompt_path.write_text(prompt)
    print(f"codex_start candidate={candidate_index} axis={axis} iteration={iteration}", flush=True)
    print(f"codex_prompt_path {prompt_path}", flush=True)
    if os.environ.get("PATCHEX_STREAM_FULL_PROMPTS") == "1":
        print("codex_prompt_begin", flush=True)
        print(prompt, flush=True)
        print("codex_prompt_end", flush=True)
    else:
        print(f"codex_prompt_hidden candidate={candidate_index} axis={axis} iteration={iteration} bytes={len(prompt.encode())}", flush=True)
    process = subprocess.Popen(
        [
            codex_path,
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(answer_path),
            "-C",
            str(Path.cwd()),
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(prompt)
    process.stdin.close()
    for line in process.stdout:
        print(f"codex_log candidate={candidate_index} axis={axis} iteration={iteration} {line.rstrip()}", flush=True)
    try:
        returncode = process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"Codex candidate {candidate_index} iteration {iteration} timed out.")
    if returncode != 0:
        raise RuntimeError(f"Codex candidate {candidate_index} iteration {iteration} failed with return code {returncode}.")
    print(f"codex_done candidate={candidate_index} axis={axis} iteration={iteration} answer_path={answer_path}", flush=True)
    answer = answer_path.read_text()
    if not answer.strip():
        raise RuntimeError(f"Codex candidate {candidate_index} iteration {iteration} returned an empty answer.")
    return with_fixed_pattern(sanitize_recipe_payload(extract_json_object(answer)), fixed_pattern), {
        "used": True,
        "candidate_index": candidate_index,
        "axis": axis,
        "iteration": iteration,
        "answer_path": str(answer_path),
        "prompt_path": str(prompt_path),
    }


def build_candidate_specs(candidate_count: int, axis_trials: int) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [("seed", "Create the strongest overall first-pass patch for the target prompt.")]
    for index in range(max(0, candidate_count - 1)):
        specs.append(("combined", f"Create an alternate full-patch candidate #{index + 1}; explore a materially different but plausible synth/effects/pattern balance."))
    for axis in AXIS_MUTATION_GROUPS:
        for trial in range(axis_trials):
            specs.append((axis, f"Optimize this candidate primarily for the {axis} axis. Keep other axes plausible, but make the {axis} decision intentionally strong. Trial {trial + 1}."))
    return specs


def mutate_recipe_for_axis(recipe: Recipe, axis: str, rng: np.random.Generator, amount: float) -> Recipe:
    mutated = clone_recipe(recipe)
    fields = AXIS_MUTATION_GROUPS[axis]
    synth = mutated.synth
    effects = mutated.effects
    pattern = mutated.pattern
    if "sine_square" in fields:
        synth.sine_square = float(np.clip(synth.sine_square + rng.normal(0, amount), 0, 1))
    if "attack" in fields:
        synth.attack = float(np.clip(synth.attack * np.exp(rng.normal(0, amount)), 0.001, 2.0))
    if "decay" in fields:
        synth.decay = float(np.clip(synth.decay * np.exp(rng.normal(0, amount)), 0.001, 2.0))
    if "sustain" in fields:
        synth.sustain = float(np.clip(synth.sustain + rng.normal(0, amount), 0, 1))
    if "release" in fields:
        synth.release = float(np.clip(synth.release * np.exp(rng.normal(0, amount)), 0.001, 3.0))
    if "gate" in fields:
        synth.gate = float(np.clip(synth.gate * np.exp(rng.normal(0, amount)), 0.05, 2.0))
    if "tempo" in fields:
        pattern.tempo = float(np.clip(pattern.tempo + rng.normal(0, 8), 60, 180))
    fx_values = asdict(effects)
    for field in fields:
        if field in fx_values:
            scale = 3.0 if field.endswith("_db") else 0.08
            if field == "saturation_drive_db":
                scale = 4.0
            if field == "compressor_threshold_db":
                scale = 4.0
            fx_values[field] = fx_values[field] + rng.normal(0, scale * amount)
    effects = sanitize_params(FxParams(**fx_values))
    return Recipe(
        instrument_type=mutated.instrument_type,
        synth=synth,
        pattern=pattern,
        effects=effects,
        macros=default_macros(synth, effects),
        keyboard_mapping=mutated.keyboard_mapping or default_keyboard_mapping(),
    )


def codex_synthesize_patch(output_dir: Path, analysis: dict[str, Any], results: list[dict[str, Any]], merged_recipe: Recipe) -> tuple[Recipe, dict[str, Any]]:
    codex_path = require_codex_path()
    prompt_path = output_dir / "codex_synthesis_prompt.txt"
    answer_path = output_dir / "codex_synthesis_answer.txt"
    top_payload = [
        {"axis": item.get("axis"), "scores": item["scores"], "recipe": recipe_to_json(item["recipe"])}
        for item in results[:8]
    ]
    prompt = (
        "You are synthesizing a playable synth patch from axis-specific candidate results.\n"
        "Return ONLY JSON matching the recipe shape. Preserve playable keyboard mapping.\n\n"
        f"Analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"Top candidates:\n{json.dumps(top_payload, indent=2)}\n\n"
        f"Merged candidate recipe to improve:\n{json.dumps(recipe_to_json(merged_recipe), indent=2)}\n",
    )
    prompt = "".join(prompt)
    prompt_path.write_text(prompt)
    print("codex_start synthesis", flush=True)
    print(f"codex_prompt_path {prompt_path}", flush=True)
    if os.environ.get("PATCHEX_STREAM_FULL_PROMPTS") == "1":
        print("codex_prompt_begin", flush=True)
        print(prompt, flush=True)
        print("codex_prompt_end", flush=True)
    else:
        print(f"codex_prompt_hidden synthesis bytes={len(prompt.encode())}", flush=True)
    process = subprocess.Popen(
        [
            codex_path,
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(answer_path),
            "-C",
            str(Path.cwd()),
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(prompt)
    process.stdin.close()
    for line in process.stdout:
        print(f"codex_log synthesis {line.rstrip()}", flush=True)
    try:
        returncode = process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("Codex synthesis timed out.")
    if returncode != 0:
        raise RuntimeError(f"Codex synthesis failed with return code {returncode}.")
    print(f"codex_done synthesis answer_path={answer_path}", flush=True)
    answer = answer_path.read_text()
    if not answer.strip():
        raise RuntimeError("Codex synthesis returned an empty answer.")
    return sanitize_recipe_payload(extract_json_object(answer)), {"used": True, "answer_path": str(answer_path), "prompt_path": str(prompt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--axis-trials", type=int, default=2)
    parser.add_argument("--candidate-iterations", type=int, default=3)
    parser.add_argument("--instrument-type", choices=["bass_synth", "lead_synth", "pad_synth", "pluck_synth", "arp_synth", "texture_synth"])
    parser.add_argument("--target-part", default="")
    parser.add_argument("--analysis-json", type=Path)
    args = parser.parse_args()

    runtime()
    reference_audio, reference_sr = load_audio(args.reference)
    if args.analysis_json:
        analysis = json.loads(args.analysis_json.read_text())
    else:
        analysis = analyze_reference(reference_audio, reference_sr)
    if args.instrument_type:
        analysis["instrument_type"] = args.instrument_type
    if args.target_part:
        analysis["target_part"] = args.target_part
    target_prompt = args.prompt or phrase_summary(analysis["axes"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    proposal_status: dict[str, Any] = {"source": "codex_cli", "codex_path": CODEX_PATH}
    codex_candidates: list[dict[str, Any]] = []
    fixed_pattern = fixed_pattern_from_analysis(analysis)
    print(f"fixed_pattern {json.dumps(asdict(fixed_pattern))}", flush=True)
    seed_recipe: Recipe | None = None

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reference_clip = tmp_path / "reference.wav"
        sf.write(reference_clip, reference_audio.T, reference_sr)
        reference_embedding = embed_wav(client, reference_clip)
        text_embedding = embed_text(client, target_prompt)
        specs = list(enumerate(build_candidate_specs(args.candidates, args.axis_trials)))
        max_workers = max(1, min(4, len(specs)))
        print(f"parallel_candidates count={len(specs)} max_workers={max_workers}", flush=True)

        def run_candidate(spec: tuple[int, tuple[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            candidate_index, (axis, objective) = spec
            return run_iterative_codex_candidate(
                output_dir=args.output_dir,
                client=client,
                analysis=analysis,
                target_prompt=target_prompt,
                instrument_type=analysis["instrument_type"],
                fixed_pattern=fixed_pattern,
                candidate_index=candidate_index,
                axis=axis,
                objective=objective,
                iterations=args.candidate_iterations,
                seconds=args.seconds,
                sample_rate=args.sample_rate,
                reference_embedding=reference_embedding,
                text_embedding=text_embedding,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(run_candidate, spec): spec[0] for spec in specs}
            for future in as_completed(future_to_index):
                candidate_index = future_to_index[future]
                result, codex_runs = future.result()
                results.append(result)
                codex_candidates.extend(codex_runs)
                print(f"candidate_complete index={candidate_index} axis={result['axis']} best_loss={result['loss']:.4f} best_score={result['scores']['final']:.4f}", flush=True)
    results.sort(key=lambda item: item["index"])
    seed_recipe = results[0]["recipe"] if results else None
    if seed_recipe is None:
        raise RuntimeError("No candidate recipes were generated.")

    results.sort(key=lambda item: item["scores"]["final"], reverse=True)
    final_recipe = synthesize_recipe(results)
    final_recipe = with_fixed_pattern(final_recipe, fixed_pattern)
    final_recipe, codex_synthesis = codex_synthesize_patch(args.output_dir, analysis, results, final_recipe)
    final_recipe = with_fixed_pattern(final_recipe, fixed_pattern)
    final_audio = render_recipe(final_recipe, args.seconds, args.sample_rate)
    final_path = args.output_dir / "final_synthesized.wav"
    sf.write(final_path, final_audio.T, args.sample_rate)
    patch_path = args.output_dir / "playable_patch.json"
    patch_path.write_text(json.dumps(recipe_to_json(final_recipe), indent=2) + "\n")

    metadata = {
        "reference": str(args.reference),
        "prompt": target_prompt,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "analysis": analysis,
        "proposal_status": proposal_status,
        "fixed_pattern": asdict(fixed_pattern),
        "seed_recipe": recipe_to_json(seed_recipe),
        "codex_candidates": codex_candidates,
        "best_candidates": [
            {
                "index": item["index"],
                "axis": item["axis"],
                "path": item["path"],
                "scores": item["scores"],
                "loss": item["loss"],
                "recipe": recipe_to_json(item["recipe"]),
                "iterations": item["iterations"],
            }
            for item in results[:5]
        ],
        "final_recipe": recipe_to_json(final_recipe),
        "codex_synthesis": codex_synthesis,
        "final_path": str(final_path),
        "patch_path": str(patch_path),
    }
    (args.output_dir / "match_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {final_path}")
    print(f"wrote {patch_path}")
    print(f"wrote {args.output_dir / 'match_report.json'}")


if __name__ == "__main__":
    main()
