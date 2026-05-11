#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import subprocess
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from audio_diff import AudioScore, compare_audio, estimate_beat_grid, score_to_json
    from text2fx import extract_json_object, load_runtime_dependencies
except ModuleNotFoundError:
    from .audio_diff import AudioScore, compare_audio, estimate_beat_grid, score_to_json
    from .text2fx import extract_json_object, load_runtime_dependencies

sf: Any = None


DEFAULT_SESSION = {
    "version": 1,
    "sample_rate": 44100,
    "duration": 5.0,
    "layers": [],
    "returns": [{"id": "space", "type": "reverb", "gain_db": -12.0, "decay": 0.35, "width": 1.0}],
    "master": {"gain_db": -1.0, "width": 1.0},
}

NOTE_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
CODEX_PATH = "/Applications/Codex.app/Contents/Resources/codex"
VITAL_AU_RENDER_SOURCE = Path(__file__).resolve().parent / "vital_au_render.swift"
VITAL_AU_RENDER_BIN = Path(tempfile.gettempdir()) / "text2fx_vital_au_render"
MAX_NOTES_PER_LAYER = 128
MAX_AUTOMATION_POINTS = 64
VITAL_PLUGIN_CANDIDATES = [
    Path("/Library/Audio/Plug-Ins/VST3/Vital.vst3"),
    Path("/Library/Audio/Plug-Ins/Components/Vital.component"),
    Path("/Library/Audio/Plug-Ins/VST/Vital.vst"),
]


def setup() -> None:
    global sf
    load_runtime_dependencies()
    import soundfile as soundfile_module

    sf = soundfile_module


def vital_status() -> dict[str, Any]:
    installed = [str(path) for path in VITAL_PLUGIN_CANDIDATES if path.exists()]
    loadable = False
    error = ""
    if installed:
        try:
            from pedalboard import load_plugin

            load_plugin(installed[0])
            loadable = True
        except Exception as exc:
            error = str(exc).splitlines()[0]
    auval_loadable = False
    auval_error = ""
    try:
        result = subprocess.run(["auval", "-v", "aumu", "Vita", "Tyte"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        auval_loadable = result.returncode == 0 and "AU VALIDATION SUCCEEDED" in result.stdout
        if not auval_loadable:
            auval_error = "\n".join(result.stdout.splitlines()[-5:])
    except Exception as exc:
        auval_error = str(exc)
    return {"installed_paths": installed, "pedalboard_loadable": loadable, "pedalboard_error": error, "audio_unit_loadable": auval_loadable, "audio_unit_error": auval_error}


def ensure_vital_au_renderer() -> Path:
    if not VITAL_AU_RENDER_BIN.exists() or VITAL_AU_RENDER_BIN.stat().st_mtime < VITAL_AU_RENDER_SOURCE.stat().st_mtime:
        subprocess.run(["swiftc", str(VITAL_AU_RENDER_SOURCE), "-o", str(VITAL_AU_RENDER_BIN)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return VITAL_AU_RENDER_BIN


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.T.astype(np.float32), int(sample_rate)


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def note_to_midi(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(np.clip(value, 24, 96))
    text = str(value).strip().upper().replace("♯", "#").replace("♭", "B")
    if len(text) < 2:
        return 60
    name = text[:-1]
    try:
        octave = int(text[-1])
    except ValueError:
        return 60
    return int(np.clip((octave + 1) * 12 + NOTE_NAMES.get(name, 0), 24, 96))


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def sanitize_points(points: Any, duration: float, value_key: str, default: list[dict[str, float]], lo: float, hi: float) -> list[dict[str, float]]:
    if not isinstance(points, list) or not points:
        points = default
    out = []
    for point in points[:MAX_AUTOMATION_POINTS]:
        if not isinstance(point, dict):
            continue
        time_value = float(np.clip(float(point.get("time", 0.0)), 0.0, duration))
        value = float(np.clip(float(point.get(value_key, point.get("value", 0.0))), lo, hi))
        out.append({"time": time_value, value_key: value})
    out.sort(key=lambda item: item["time"])
    return out or default


def automation_curve(points: list[dict[str, float]], value_key: str, samples: int, duration: float) -> np.ndarray:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    times = np.asarray([float(point["time"]) for point in points], dtype=np.float32)
    values = np.asarray([float(point[value_key]) for point in points], dtype=np.float32)
    if times.size == 1:
        return np.full(samples, values[0], dtype=np.float32)
    t = np.linspace(0.0, duration, samples, endpoint=False, dtype=np.float32)
    return np.interp(t, times, values).astype(np.float32)


def lfo_shape(shape: str, rate_hz: float, phase: float, samples: int, sr: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / sr
    cycle = (rate_hz * t + phase) % 1.0
    if shape == "triangle":
        return (4.0 * np.abs(cycle - 0.5) - 1.0).astype(np.float32)
    if shape in {"square", "pulse"}:
        return np.where(cycle < 0.5, 1.0, -1.0).astype(np.float32)
    if shape in {"saw", "ramp"}:
        return (2.0 * cycle - 1.0).astype(np.float32)
    return np.sin(2.0 * np.pi * cycle).astype(np.float32)


def modulation_curves(modulation: dict[str, Any], samples: int, sr: int, duration: float) -> dict[str, np.ndarray]:
    curves = {
        "gain_db": np.zeros(samples, dtype=np.float32),
        "filter_hz": np.zeros(samples, dtype=np.float32),
        "pan": np.zeros(samples, dtype=np.float32),
        "width": np.zeros(samples, dtype=np.float32),
    }
    for lfo in modulation.get("lfos", []) if isinstance(modulation.get("lfos", []), list) else []:
        target = str(lfo.get("target", "")).lower()
        amount = float(lfo.get("amount", 0.0)) * float(lfo.get("depth", 1.0))
        signal = lfo_shape(str(lfo.get("shape", "sine")), float(lfo.get("rate_hz", 1.0)), float(lfo.get("phase", 0.0)), samples, sr)
        if "gain" in target or "amp" in target or "volume" in target:
            curves["gain_db"] += signal * (amount if amount else 2.0 * float(lfo.get("depth", 1.0)))
        elif "pan" in target:
            curves["pan"] += signal * (amount if amount else 0.3 * float(lfo.get("depth", 1.0)))
        elif "width" in target or "stereo" in target:
            curves["width"] += signal * (amount if amount else 0.2 * float(lfo.get("depth", 1.0)))
        elif "filter" in target or "cutoff" in target:
            curves["filter_hz"] += signal * (amount if amount else 800.0 * float(lfo.get("depth", 1.0)))
    return curves


def sanitize_session(payload: dict[str, Any], duration: float, sample_rate: int) -> dict[str, Any]:
    session = dict(DEFAULT_SESSION)
    session["duration"] = float(duration)
    session["sample_rate"] = int(sample_rate)
    session["layers"] = []
    returns = payload.get("returns", DEFAULT_SESSION["returns"])
    session["returns"] = []
    if isinstance(returns, list):
        for index, ret in enumerate(returns[:3]):
            if not isinstance(ret, dict):
                continue
            session["returns"].append(
                {
                    "id": str(ret.get("id") or f"return_{index + 1}"),
                    "type": str(ret.get("type", "reverb")).lower(),
                    "gain_db": float(np.clip(float(ret.get("gain_db", -12.0)), -48.0, 6.0)),
                    "decay": float(np.clip(float(ret.get("decay", 0.35)), 0.05, 1.0)),
                    "width": float(np.clip(float(ret.get("width", 1.0)), 0.0, 1.5)),
                }
            )
    for index, layer in enumerate(payload.get("layers", [])[:6]):
        synth = layer.get("synth", {})
        amp = layer.get("amp_envelope", {})
        filt = layer.get("filter", {})
        effects = layer.get("effects", {})
        if not isinstance(effects, dict):
            effects = {}
        compression_payload = effects.get("compression", effects.get("compressor", {}))
        compression = compression_payload if isinstance(compression_payload, dict) else {}
        compression_mix = compression.get("mix", compression.get("amount", effects.get("compression_mix", effects.get("compressor_mix", 0.0))))
        compression_threshold_db = compression.get("threshold_db", effects.get("compression_threshold_db", effects.get("threshold_db", -18.0)))
        compression_ratio = compression.get("ratio", effects.get("compression_ratio", effects.get("ratio", 2.0)))
        compression_attack = compression.get("attack", effects.get("compression_attack", effects.get("attack", 0.01)))
        compression_release = compression.get("release", effects.get("compression_release", effects.get("release", 0.16)))
        notes = layer.get("notes", [])
        if not isinstance(notes, list) or not notes:
            notes = [{"note": synth.get("note", 48), "start": 0.0, "duration": duration, "velocity": 0.7}]
        modulation_payload = layer.get("modulation", {})
        raw_lfos = modulation_payload.get("lfos", [])
        sanitized_lfos = []
        if isinstance(raw_lfos, list):
            for lfo in raw_lfos[:8]:
                if not isinstance(lfo, dict):
                    continue
                sanitized_lfos.append(
                    {
                        "id": str(lfo.get("id") or f"lfo_{len(sanitized_lfos) + 1}"),
                        "target": str(lfo.get("target", "filter.cutoff_hz")),
                        "shape": str(lfo.get("shape", "sine")).lower(),
                        "rate_hz": float(np.clip(float(lfo.get("rate_hz", lfo.get("rate", 1.0))), 0.01, 20.0)),
                        "depth": float(np.clip(float(lfo.get("depth", 1.0)), 0.0, 1.0)),
                        "phase": float(np.clip(float(lfo.get("phase", 0.0)), 0.0, 1.0)),
                        "amount": float(np.clip(float(lfo.get("amount", lfo.get("amount_hz", lfo.get("amount_db", 0.0)))), -12000.0, 12000.0)),
                        "center": float(np.clip(float(lfo.get("center", 0.0)), -12000.0, 20000.0)),
                    }
                )
        waveform = str(synth.get("waveform", "saw")).lower()
        wavetable_name = str(synth.get("wavetable", synth.get("waveform", "saw_stack"))).lower()
        default_wavetable_position = {
            "sine": 0.0,
            "triangle": 0.22,
            "tri": 0.22,
            "digital": 0.38,
            "formant": 0.45,
            "saw": 0.55,
            "saw_stack": 0.62,
            "square_saw": 0.74,
            "square": 0.82,
            "noise": 0.95,
            "air": 0.95,
        }.get(wavetable_name, 0.5)
        requested_engine = str(synth.get("engine", "")).lower()
        engine = requested_engine or "vital"
        sanitized_notes = []
        for note in notes[:MAX_NOTES_PER_LAYER]:
            start = float(np.clip(float(note.get("start", 0.0)), 0.0, duration))
            note_duration = float(np.clip(float(note.get("duration", duration - start)), 0.01, duration - start if duration > start else 0.01))
            sanitized_notes.append(
                {
                    "note": note_to_midi(note.get("note", synth.get("note", 48))),
                    "start": start,
                    "duration": note_duration,
                    "velocity": float(np.clip(float(note.get("velocity", 0.7)), 0.0, 1.0)),
                }
            )
        session["layers"].append(
            {
                "id": str(layer.get("id") or f"layer_{index + 1}"),
                "role": str(layer.get("role") or "synth layer"),
                "gain_db": float(np.clip(float(layer.get("gain_db", -8.0)), -48.0, 6.0)),
                "pan": float(np.clip(float(layer.get("pan", 0.0)), -1.0, 1.0)),
                "width": float(np.clip(float(layer.get("width", 0.6)), 0.0, 1.0)),
                "notes": sanitized_notes,
                "synth": {
                    "waveform": waveform,
                    "engine": engine,
                    "wavetable": wavetable_name,
                    "wavetable_position": float(np.clip(float(synth.get("wavetable_position", default_wavetable_position)), 0.0, 1.0)),
                    "warp": float(np.clip(float(synth.get("warp", 0.0)), 0.0, 1.0)),
                    "fm_amount": float(np.clip(float(synth.get("fm_amount", 0.0)), 0.0, 1.0)),
                    "fm_ratio": float(np.clip(float(synth.get("fm_ratio", 2.0)), 0.25, 12.0)),
                    "blend": float(np.clip(float(synth.get("blend", 0.35)), 0.0, 1.0)),
                    "voices": int(np.clip(int(synth.get("voices", 3)), 1, 16)),
                    "detune_cents": float(np.clip(float(synth.get("detune_cents", 8.0)), 0.0, 70.0)),
                    "stereo_spread": float(np.clip(float(synth.get("stereo_spread", 0.55)), 0.0, 1.0)),
                    "sub_level": float(np.clip(float(synth.get("sub_level", 0.15)), 0.0, 1.0)),
                    "vital_parameters": synth.get("vital_parameters", synth.get("parameters", {})) if isinstance(synth.get("vital_parameters", synth.get("parameters", {})), dict) else {},
                },
                "amp_envelope": {
                    "attack": float(np.clip(float(amp.get("attack", 1.2)), 0.001, 4.0)),
                    "decay": float(np.clip(float(amp.get("decay", 0.6)), 0.001, 4.0)),
                    "sustain": float(np.clip(float(amp.get("sustain", 0.85)), 0.0, 1.0)),
                    "release": float(np.clip(float(amp.get("release", 1.0)), 0.001, 4.0)),
                },
                "filter": {
                    "cutoff_start_hz": float(np.clip(float(filt.get("cutoff_start_hz", 550.0)), 80.0, 12000.0)),
                    "cutoff_end_hz": float(np.clip(float(filt.get("cutoff_end_hz", 1800.0)), 80.0, 14000.0)),
                    "resonance": float(np.clip(float(filt.get("resonance", 0.25)), 0.0, 1.0)),
                    "drive": float(np.clip(float(filt.get("drive", 0.0)), 0.0, 1.0)),
                    "cutoff_points": sanitize_points(
                        filt.get("cutoff_points"),
                        duration,
                        "hz",
                        [
                            {"time": 0.0, "hz": float(np.clip(float(filt.get("cutoff_start_hz", 550.0)), 80.0, 12000.0))},
                            {"time": duration, "hz": float(np.clip(float(filt.get("cutoff_end_hz", 1800.0)), 80.0, 14000.0))},
                        ],
                        80.0,
                        14000.0,
                    ),
                },
                "modulation": {
                    "lfo_rate_hz": float(np.clip(float(modulation_payload.get("lfo_rate_hz", 0.15)), 0.0, 8.0)),
                    "lfo_depth": float(np.clip(float(modulation_payload.get("lfo_depth", 0.05)), 0.0, 1.0)),
                    "gate_points": sanitize_points(modulation_payload.get("gate_points"), duration, "level", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}], 0.0, 1.0),
                    "lfos": sanitized_lfos,
                },
                "gain_points": sanitize_points(layer.get("gain_points"), duration, "db", [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}], -36.0, 12.0),
                "effects": {
                    "chorus_mix": float(np.clip(float(effects.get("chorus_mix", 0.18)), 0.0, 1.0)),
                    "reverb_mix": float(np.clip(float(effects.get("reverb_mix", 0.18)), 0.0, 0.7)),
                    "delay_mix": float(np.clip(float(effects.get("delay_mix", 0.0)), 0.0, 0.6)),
                    "delay_time": float(np.clip(float(effects.get("delay_time", 0.18)), 0.03, 0.75)),
                    "phaser_mix": float(np.clip(float(effects.get("phaser_mix", 0.0)), 0.0, 1.0)),
                    "saturation": float(np.clip(float(effects.get("saturation", 0.0)), 0.0, 1.0)),
                    "low_gain_db": float(np.clip(float(effects.get("low_gain_db", 0.0)), -18.0, 18.0)),
                    "mid_gain_db": float(np.clip(float(effects.get("mid_gain_db", 0.0)), -18.0, 18.0)),
                    "high_gain_db": float(np.clip(float(effects.get("high_gain_db", 0.0)), -18.0, 18.0)),
                    "compression_mix": float(np.clip(float(compression_mix), 0.0, 1.0)),
                    "compression_threshold_db": float(np.clip(float(compression_threshold_db), -60.0, 0.0)),
                    "compression_ratio": float(np.clip(float(compression_ratio), 1.0, 20.0)),
                    "compression_attack": float(np.clip(float(compression_attack), 0.001, 0.5)),
                    "compression_release": float(np.clip(float(compression_release), 0.01, 2.0)),
                    "return_send": sanitize_return_send(effects.get("return_send", 0.0)),
                },
            }
        )
    master = payload.get("master", {})
    session["master"] = {
        "gain_db": float(np.clip(float(master.get("gain_db", -1.0)), -24.0, 6.0)),
        "width": float(np.clip(float(master.get("width", 1.0)), 0.0, 1.5)),
    }
    return session


def sanitize_return_send(value: Any) -> float:
    if isinstance(value, dict):
        values = []
        for amount in value.values():
            try:
                values.append(float(amount))
            except (TypeError, ValueError):
                continue
        value = max(values) if values else 0.0
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def oscillator(phase: np.ndarray, waveform: str, blend: float) -> np.ndarray:
    sine = np.sin(phase)
    if waveform in {"noise", "air", "transient"}:
        rng = np.random.default_rng(int(phase.size + blend * 1000))
        return rng.normal(0.0, 0.5 + 0.5 * blend, phase.size)
    saw = 2.0 * ((phase / (2.0 * np.pi)) % 1.0) - 1.0
    square = np.sign(sine)
    tri = 2.0 * np.abs(saw) - 1.0
    if waveform == "square":
        base = square
    elif waveform == "triangle":
        base = tri
    elif waveform == "sine":
        base = sine
    else:
        base = saw
    return (1.0 - blend) * sine + blend * base


def wavetable_oscillator(phase: np.ndarray, synth: dict[str, Any], freq: float) -> np.ndarray:
    wavetable = str(synth.get("wavetable", synth.get("waveform", "saw_stack"))).lower()
    position = float(synth.get("wavetable_position", 0.5))
    blend = float(synth.get("blend", 0.35))
    warp = float(synth.get("warp", 0.0))
    fm_amount = float(synth.get("fm_amount", 0.0))
    fm_ratio = float(synth.get("fm_ratio", 2.0))
    if fm_amount:
        phase = phase + fm_amount * 2.0 * np.sin(phase * fm_ratio)
    if warp:
        phase = phase + warp * np.sin(2.0 * phase)
    sine = np.sin(phase)
    saw = 2.0 * ((phase / (2.0 * np.pi)) % 1.0) - 1.0
    square = np.tanh(5.0 * np.sin(phase))
    tri = 2.0 * np.abs(saw) - 1.0
    formant = np.sin(phase) + 0.35 * np.sin(phase * 2.01) + 0.18 * np.sin(phase * 3.02)
    formant = np.tanh(formant * (1.0 + 2.0 * position))
    glass = np.sin(phase) + 0.25 * np.sin(phase * (2.0 + 3.0 * position)) + 0.12 * np.sin(phase * 7.0)
    supersaw = saw + 0.55 * np.sin(phase * 0.997) + 0.35 * np.sin(phase * 1.003)
    if wavetable in {"saw_stack", "supersaw", "vital_supersaw"}:
        raw = (1.0 - position) * saw + position * supersaw / 1.9
    elif wavetable in {"square_saw", "pulse"}:
        raw = (1.0 - position) * square + position * saw
    elif wavetable in {"formant", "vocal"}:
        raw = (1.0 - position) * saw + position * formant
    elif wavetable in {"digital", "glass", "metal"}:
        raw = (1.0 - position) * tri + position * glass
    elif wavetable in {"triangle", "tri"}:
        raw = tri
    elif wavetable == "sine":
        raw = sine
    else:
        raw = oscillator(phase, str(synth.get("waveform", "saw")), blend)
    raw = (1.0 - blend) * sine + blend * raw
    return raw / (np.max(np.abs(raw)) + 1e-8)


def envelope(total: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    a = min(total, int(attack * sr))
    d = min(max(0, total - a), int(decay * sr))
    r = min(max(0, total - a - d), int(release * sr))
    hold = max(0, total - a - d - r)
    parts = []
    if a:
        parts.append(np.linspace(0.0, 1.0, a, endpoint=False))
    if d:
        parts.append(np.linspace(1.0, sustain, d, endpoint=False))
    if hold:
        parts.append(np.full(hold, sustain))
    start = parts[-1][-1] if parts else sustain
    if r:
        parts.append(np.linspace(start, 0.0, r, endpoint=True))
    env = np.concatenate(parts) if parts else np.zeros(total)
    return np.pad(env, (0, max(0, total - env.size)))[:total]


def moving_lowpass(audio: np.ndarray, sr: int, start_hz: float, end_hz: float, resonance: float, cutoff_curve: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros_like(audio)
    y = 0.0
    for index, sample in enumerate(audio):
        if cutoff_curve is not None and cutoff_curve.size:
            cutoff = float(cutoff_curve[min(index, cutoff_curve.size - 1)])
        else:
            frac = index / max(1, audio.size - 1)
            cutoff = start_hz + (end_hz - start_hz) * frac
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sr)
        y += alpha * (sample - y)
        out[index] = y
    return out * (1.0 + 0.2 * resonance)


def one_pole_highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    low = moving_lowpass(audio, sr, cutoff_hz, cutoff_hz, 0.0)
    return audio - low


def layer_eq(audio: np.ndarray, sr: int, effects: dict[str, Any]) -> np.ndarray:
    low = moving_lowpass(audio, sr, 220.0, 220.0, 0.0)
    high = one_pole_highpass(audio, sr, 2400.0)
    mid = audio - low - high
    return (
        low * db_to_amp(float(effects.get("low_gain_db", 0.0)))
        + mid * db_to_amp(float(effects.get("mid_gain_db", 0.0)))
        + high * db_to_amp(float(effects.get("high_gain_db", 0.0)))
    ).astype(np.float32)


def stereo_layer_eq(stereo: np.ndarray, sr: int, effects: dict[str, Any]) -> np.ndarray:
    return np.vstack([layer_eq(stereo[0], sr, effects), layer_eq(stereo[1], sr, effects)]).astype(np.float32)


def apply_saturation(stereo: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return stereo
    drive = 1.0 + 6.0 * float(amount)
    return (np.tanh(stereo * drive) / np.tanh(drive)).astype(np.float32)


def apply_compressor(stereo: np.ndarray, sr: int, effects: dict[str, Any]) -> np.ndarray:
    mix = float(effects.get("compression_mix", 0.0))
    if mix <= 0:
        return stereo
    threshold = db_to_amp(float(effects.get("compression_threshold_db", -18.0)))
    ratio = max(1.0, float(effects.get("compression_ratio", 2.0)))
    attack = max(0.001, float(effects.get("compression_attack", 0.01)))
    release = max(0.01, float(effects.get("compression_release", 0.16)))
    attack_coeff = math.exp(-1.0 / (attack * sr))
    release_coeff = math.exp(-1.0 / (release * sr))
    envelope_follow = np.zeros(stereo.shape[1], dtype=np.float32)
    level = 0.0
    mono_level = np.max(np.abs(stereo), axis=0)
    for index, sample_level in enumerate(mono_level):
        coeff = attack_coeff if sample_level > level else release_coeff
        level = coeff * level + (1.0 - coeff) * float(sample_level)
        envelope_follow[index] = level
    gain = np.ones_like(envelope_follow)
    over = envelope_follow > threshold
    compressed_level = threshold + (envelope_follow[over] - threshold) / ratio
    gain[over] = compressed_level / np.maximum(envelope_follow[over], 1e-8)
    compressed = stereo * gain[None, :]
    return (stereo * (1.0 - mix) + compressed * mix).astype(np.float32)


def apply_pan_width(stereo: np.ndarray, pan: float, width: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    width = float(np.clip(width, 0.0, 2.0))
    mid = 0.5 * (stereo[0] + stereo[1])
    side = 0.5 * (stereo[0] - stereo[1]) * width
    widened = np.vstack([mid + side, mid - side])
    angle = (pan + 1.0) * math.pi * 0.25
    left_gain = math.cos(angle) * math.sqrt(2.0)
    right_gain = math.sin(angle) * math.sqrt(2.0)
    widened[0] *= left_gain
    widened[1] *= right_gain
    return widened.astype(np.float32)


def apply_chorus(stereo: np.ndarray, sr: int, mix: float, width: float) -> np.ndarray:
    if mix <= 0:
        return stereo
    delay = max(1, int(0.012 * sr))
    if stereo.shape[1] <= delay:
        return stereo
    delayed = np.zeros_like(stereo)
    delayed[0, delay:] = stereo[1, :-delay]
    delayed[1, delay:] = -stereo[0, :-delay]
    return (stereo * (1.0 - mix) + delayed * mix * max(0.1, width)).astype(np.float32)


def apply_decorrelation_width(stereo: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.75))
    if amount <= 0.0:
        return stereo
    samples = stereo.shape[1]
    delay_a = max(1, int(0.007 * sr))
    delay_b = max(delay_a + 1, int(0.013 * sr))
    if samples <= delay_b:
        return stereo
    mid = 0.5 * (stereo[0] + stereo[1])
    delayed_a = np.zeros_like(mid)
    delayed_b = np.zeros_like(mid)
    delayed_a[delay_a:] = mid[:-delay_a]
    delayed_b[delay_b:] = mid[:-delay_b]
    side = (delayed_a - 0.65 * delayed_b) * amount
    widened = np.vstack([mid + side, mid - side])
    return widened.astype(np.float32)


def apply_stereo_reverb(stereo: np.ndarray, sr: int, mix: float) -> np.ndarray:
    if mix <= 0:
        return stereo
    tail = np.zeros_like(stereo)
    samples = stereo.shape[1]
    delays = [int(sr * 0.083), int(sr * 0.137), int(sr * 0.211)]
    for delay_samples in delays:
        if delay_samples < samples:
            tail[:, delay_samples:] += stereo[:, :-delay_samples] * (0.35 / len(delays))
    return (stereo * (1.0 - mix) + tail * mix).astype(np.float32)


def hz_to_vital_norm(hz: float) -> float:
    hz = float(np.clip(hz, 20.0, 20000.0))
    return float(np.clip((math.log(hz) - math.log(20.0)) / (math.log(20000.0) - math.log(20.0)), 0.0, 1.0))


def seconds_to_vital_norm(seconds: float) -> float:
    seconds = max(0.0, float(seconds))
    return float(np.clip(math.log1p(seconds * 20.0) / math.log1p(80.0), 0.0, 1.0))


def build_vital_parameters(layer: dict[str, Any]) -> dict[str, float]:
    synth = layer.get("synth", {})
    amp = layer.get("amp_envelope", {})
    filt = layer.get("filter", {})
    effects = layer.get("effects", {})
    waveform_positions = {
        "sine": 0.0,
        "triangle": 0.22,
        "tri": 0.22,
        "saw": 0.55,
        "saw_stack": 0.62,
        "square": 0.82,
        "square_saw": 0.74,
        "digital": 0.38,
        "formant": 0.45,
        "noise": 0.95,
        "air": 0.95,
    }
    wave_position = float(synth.get("wavetable_position", waveform_positions.get(str(synth.get("waveform", "saw")), 0.55)))
    stereo_spread = float(synth.get("stereo_spread", layer.get("width", 1.0)))
    warp = float(synth.get("warp", 0.0))
    fm_amount = float(synth.get("fm_amount", 0.0))
    params: dict[str, float] = {
        "Oscillator 1 Switch": 1.0,
        "Oscillator 1 Level": float(np.clip(synth.get("blend", 0.8), 0.0, 1.0)),
        "Oscillator 1 Blend": float(np.clip(wave_position, 0.0, 1.0)),
        "Oscillator 1 Wave Frame": float(np.clip(wave_position, 0.0, 1.0)),
        "Oscillator 1 Unison Voices": float(np.clip(int(synth.get("voices", 1)) - 1, 0, 15)),
        "Oscillator 1 Unison Detune": float(np.clip(float(synth.get("detune_cents", 0.0)) / 50.0, 0.0, 1.0)),
        "Oscillator 1 Stereo Spread": float(np.clip(stereo_spread, 0.0, 1.0)),
        "Oscillator 1 Distortion Amount": float(np.clip(max(warp, fm_amount), 0.0, 1.0)),
        "Oscillator 1 Frequency Morph Amount": float(np.clip(fm_amount, 0.0, 1.0)),
        "Oscillator 1 Frequency Morph Spread": float(np.clip(float(synth.get("fm_ratio", 2.0)) / 12.0, 0.0, 1.0)),
        "Oscillator 1 Pan": float(np.clip(0.5 + 0.5 * float(layer.get("pan", 0.0)), 0.0, 1.0)),
        "Oscillator 2 Switch": 1.0 if float(synth.get("sub_level", 0.0)) > 0.001 else 0.0,
        "Oscillator 2 Level": float(np.clip(synth.get("sub_level", 0.0), 0.0, 1.0)),
        "Oscillator 2 Transpose": 36.0,
        "Envelope 1 Attack": seconds_to_vital_norm(float(amp.get("attack", 0.01))),
        "Envelope 1 Decay": seconds_to_vital_norm(float(amp.get("decay", 0.2))),
        "Envelope 1 Sustain": float(np.clip(amp.get("sustain", 0.75), 0.0, 1.0)),
        "Envelope 1 Release": seconds_to_vital_norm(float(amp.get("release", 0.2))),
        "Filter 1 Switch": 1.0,
        "Filter 1 Cutoff": hz_to_vital_norm(float(filt.get("cutoff_end_hz", filt.get("cutoff_start_hz", 1200.0)))),
        "Filter 1 Resonance": float(np.clip(filt.get("resonance", 0.1), 0.0, 1.0)),
        "Filter 1 Drive": float(np.clip(filt.get("drive", 0.0), 0.0, 1.0)),
        "Filter 1 Mix": 1.0,
        "Chorus Switch": 1.0 if float(effects.get("chorus_mix", 0.0)) > 0.001 else 0.0,
        "Chorus Mix": float(np.clip(effects.get("chorus_mix", 0.0), 0.0, 1.0)),
        "Delay Switch": 1.0 if float(effects.get("delay_mix", 0.0)) > 0.001 else 0.0,
        "Delay Mix": float(np.clip(effects.get("delay_mix", 0.0), 0.0, 1.0)),
        "Distortion Switch": 1.0 if float(effects.get("saturation", 0.0)) > 0.001 else 0.0,
        "Distortion Drive": float(np.clip(effects.get("saturation", 0.0), 0.0, 1.0)),
        "Distortion Mix": 1.0,
        "Compressor Switch": 1.0 if float(effects.get("compression_mix", 0.0)) > 0.001 else 0.0,
        "Compressor Mix": float(np.clip(effects.get("compression_mix", 0.0), 0.0, 1.0)),
        "Compressor Attack": float(np.clip(effects.get("compression_attack", 0.01) / 0.5, 0.0, 1.0)),
        "Compressor Release": float(np.clip(effects.get("compression_release", 0.16) / 2.0, 0.0, 1.0)),
        "Reverb Switch": 1.0 if float(effects.get("reverb_mix", 0.0)) > 0.001 else 0.0,
        "Reverb Mix": float(np.clip(effects.get("reverb_mix", 0.0), 0.0, 1.0)),
        "Volume": 0.8,
    }
    explicit = synth.get("vital_parameters", {})
    if isinstance(explicit, dict):
        for key, value in explicit.items():
            try:
                params[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return params


def build_vital_parameter_automation(layer: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    filt = layer.get("filter", {})
    mod = layer.get("modulation", {})
    cutoff_points = filt.get("cutoff_points", [{"time": 0.0, "hz": filt.get("cutoff_start_hz", 1200.0)}, {"time": duration, "hz": filt.get("cutoff_end_hz", 1200.0)}])
    gain_points = layer.get("gain_points", [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}])
    gate_points = mod.get("gate_points", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}])
    cutoff_automation = [
        {"time": float(point.get("time", 0.0)), "value": hz_to_vital_norm(float(point.get("hz", filt.get("cutoff_end_hz", 1200.0))))}
        for point in cutoff_points
    ]
    volume_automation = []
    for point in gain_points:
        time = float(point.get("time", 0.0))
        gate_level = float(np.interp(time, [float(p["time"]) for p in gate_points], [float(p["level"]) for p in gate_points])) if gate_points else 1.0
        db = float(point.get("db", 0.0))
        volume_automation.append({"time": time, "value": float(np.clip(0.8 * db_to_amp(db) * gate_level, 0.0, 1.0))})
    return [
        {"parameter": "Filter 1 Cutoff", "points": cutoff_automation},
        {"parameter": "Volume", "points": volume_automation},
    ]


def apply_delay(stereo: np.ndarray, sr: int, delay_time: float, mix: float) -> np.ndarray:
    if mix <= 0:
        return stereo
    delay_samples = max(1, int(delay_time * sr))
    if delay_samples >= stereo.shape[1]:
        return stereo
    delayed = np.zeros_like(stereo)
    delayed[0, delay_samples:] = stereo[1, :-delay_samples] * 0.55
    delayed[1, delay_samples:] = stereo[0, :-delay_samples] * 0.55
    return stereo * (1.0 - mix) + delayed * mix


def apply_phaser(stereo: np.ndarray, sr: int, mix: float, rate_hz: float = 0.23) -> np.ndarray:
    if mix <= 0:
        return stereo
    samples = stereo.shape[1]
    lfo = (0.5 + 0.5 * np.sin(2.0 * np.pi * rate_hz * np.arange(samples) / sr)).astype(np.float32)
    out = np.copy(stereo)
    for channel in range(2):
        shifted = np.pad(stereo[channel, :-3], (3, 0)) if samples > 3 else stereo[channel]
        out[channel] = stereo[channel] * (1.0 - 0.35 * lfo) + shifted * (0.35 * lfo)
    return stereo * (1.0 - mix) + out * mix


def render_vital_layer(layer: dict[str, Any], duration: float, sr: int) -> np.ndarray:
    renderer = ensure_vital_au_renderer()
    with tempfile.TemporaryDirectory(prefix="text2fx_vital_") as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / "request.json"
        output_path = tmp_path / "vital.wav"
        applied_path = tmp_path / "applied_parameters.json"
        notes = [
            {
                "note": int(note.get("note", 60)),
                "start": float(note.get("start", 0.0)),
                "duration": float(note.get("duration", 0.1)),
                "velocity": int(np.clip(round(float(note.get("velocity", 0.7)) * 127.0), 1, 127)),
            }
            for note in layer.get("notes", [])
        ]
        write_json(
            request_path,
            {
                "sample_rate": sr,
                "duration": duration,
                "output_path": str(output_path),
                "notes": notes,
                "parameters": build_vital_parameters(layer),
                "parameter_automation": build_vital_parameter_automation(layer, duration),
                "dump_parameters_path": str(tmp_path / "vital_parameters.json"),
                "applied_parameters_path": str(applied_path),
            },
        )
        subprocess.run([str(renderer), str(request_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if applied_path.exists():
            report = json.loads(applied_path.read_text())
            ignored = report.get("ignored", [])
            explicit = set((layer.get("synth", {}).get("vital_parameters") or {}).keys())
            ignored_explicit = [name for name in ignored if name in explicit]
            if ignored_explicit:
                raise RuntimeError(f"Vital ignored unknown synth.vital_parameters keys: {ignored_explicit}")
        audio, rendered_sr = sf.read(output_path, always_2d=True)
        if int(rendered_sr) != int(sr):
            raise RuntimeError(f"Vital AU rendered at {rendered_sr} Hz, expected {sr} Hz")
        rendered = audio.T.astype(np.float32)
        samples = int(duration * sr)
        if rendered.shape[1] < samples:
            rendered = np.pad(rendered, ((0, 0), (0, samples - rendered.shape[1])))
        return rendered[:, :samples]


def render_layer(layer: dict[str, Any], duration: float, sr: int) -> np.ndarray:
    samples = int(duration * sr)
    synth = layer["synth"]
    if synth.get("engine") == "vital":
        stereo = render_vital_layer(layer, duration, sr)
        mod = layer["modulation"]
        filt = layer["filter"]
        gain_curve = automation_curve(layer.get("gain_points", [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}]), "db", samples, duration)
        gate_curve = automation_curve(mod.get("gate_points", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}]), "level", samples, duration)
        cutoff_curve = automation_curve(filt.get("cutoff_points", [{"time": 0.0, "hz": filt["cutoff_start_hz"]}, {"time": duration, "hz": filt["cutoff_end_hz"]}]), "hz", samples, duration)
        lfo_curves = modulation_curves(mod, samples, sr, duration)
        cutoff_curve = np.clip(cutoff_curve + lfo_curves["filter_hz"], 40.0, sr / 2.0)
        for channel in range(2):
            stereo[channel] = moving_lowpass(stereo[channel].astype(np.float32), sr, filt["cutoff_start_hz"], filt["cutoff_end_hz"], filt["resonance"], cutoff_curve)
        if float(mod.get("lfo_depth", 0.0)) > 0 and float(mod.get("lfo_rate_hz", 0.0)) > 0:
            t = np.arange(samples) / sr
            tremolo = 1.0 + float(mod["lfo_depth"]) * 0.15 * np.sin(2.0 * np.pi * float(mod["lfo_rate_hz"]) * t)
            stereo *= tremolo[None, :].astype(np.float32)
        stereo *= gate_curve[None, :]
        stereo *= np.power(10.0, (gain_curve + lfo_curves["gain_db"]) / 20.0)[None, :]
        stereo = stereo_layer_eq(stereo, sr, layer["effects"])
        stereo = apply_saturation(stereo, float(layer["effects"].get("saturation", 0.0)))
        stereo = apply_compressor(stereo, sr, layer["effects"])
        stereo = apply_chorus(stereo, sr, float(layer["effects"].get("chorus_mix", 0.0)), float(layer.get("width", 1.0)))
        stereo = apply_phaser(stereo, sr, float(layer["effects"].get("phaser_mix", 0.0)), float(mod.get("lfo_rate_hz", 0.23)) or 0.23)
        stereo = apply_delay(stereo, sr, float(layer["effects"].get("delay_time", 0.18)), float(layer["effects"].get("delay_mix", 0.0)))
        dynamic_pan = float(layer.get("pan", 0.0)) + float(np.mean(lfo_curves["pan"])) if lfo_curves["pan"].size else float(layer.get("pan", 0.0))
        dynamic_width = float(layer.get("width", 1.0)) + float(np.mean(lfo_curves["width"])) if lfo_curves["width"].size else float(layer.get("width", 1.0))
        stereo = apply_pan_width(stereo, dynamic_pan, dynamic_width)
        stereo *= db_to_amp(layer["gain_db"])
        stereo = apply_stereo_reverb(stereo, sr, float(layer["effects"].get("reverb_mix", 0.0)))
        return stereo.astype(np.float32)
    mono = np.zeros(samples, dtype=np.float32)
    amp = layer["amp_envelope"]
    filt = layer["filter"]
    mod = layer["modulation"]
    gain_curve = automation_curve(layer.get("gain_points", [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}]), "db", samples, duration)
    gate_curve = automation_curve(mod.get("gate_points", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}]), "level", samples, duration)
    cutoff_curve = automation_curve(filt.get("cutoff_points", [{"time": 0.0, "hz": filt["cutoff_start_hz"]}, {"time": duration, "hz": filt["cutoff_end_hz"]}]), "hz", samples, duration)
    lfo_curves = modulation_curves(mod, samples, sr, duration)
    cutoff_curve = np.clip(cutoff_curve + lfo_curves["filter_hz"], 40.0, sr / 2.0)
    for note in layer["notes"]:
        start = int(note["start"] * sr)
        count = max(1, int(note["duration"] * sr))
        end = min(samples, start + count)
        if end <= start:
            continue
        t = np.arange(end - start) / sr
        note_audio = np.zeros(end - start)
        voices = max(1, synth["voices"])
        for voice in range(voices):
            spread = 0.0 if voices == 1 else (voice / (voices - 1) - 0.5) * 2.0
            cents = spread * synth["detune_cents"]
            freq = midi_to_hz(note["note"] + cents / 100.0)
            phase = 2.0 * np.pi * freq * t
            note_audio += wavetable_oscillator(phase, synth, freq) / voices
        if synth["sub_level"] > 0:
            note_audio += synth["sub_level"] * np.sin(2.0 * np.pi * midi_to_hz(note["note"] - 12) * t)
        if mod["lfo_depth"] > 0 and mod["lfo_rate_hz"] > 0:
            note_audio *= 1.0 + mod["lfo_depth"] * 0.15 * np.sin(2.0 * np.pi * mod["lfo_rate_hz"] * t)
        note_audio *= envelope(end - start, sr, amp["attack"], amp["decay"], amp["sustain"], amp["release"])
        note_audio *= note["velocity"]
        drive = float(filt.get("drive", 0.0))
        if drive:
            note_audio = np.tanh(note_audio * (1.0 + 4.0 * drive)) / np.tanh(1.0 + 4.0 * drive)
        note_audio = moving_lowpass(note_audio.astype(np.float32), sr, filt["cutoff_start_hz"], filt["cutoff_end_hz"], filt["resonance"], cutoff_curve[start:end])
        mono[start:end] += note_audio.astype(np.float32)
    mono *= gate_curve
    mono *= np.power(10.0, (gain_curve + lfo_curves["gain_db"]) / 20.0)
    mono = layer_eq(mono, sr, layer["effects"])
    saturation = float(layer["effects"].get("saturation", 0.0))
    if saturation:
        mono = np.tanh(mono * (1.0 + 6.0 * saturation)) / np.tanh(1.0 + 6.0 * saturation)
    mono_stereo = np.vstack([mono, mono])
    mono = apply_compressor(mono_stereo, sr, layer["effects"]).mean(axis=0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono *= 1.0 / peak
    chorus_mix = layer["effects"]["chorus_mix"]
    delay = max(1, int(0.012 * sr))
    delayed = np.pad(mono[:-delay], (delay, 0)) if mono.size > delay else mono
    left = mono * (1.0 - 0.35 * layer["pan"]) + delayed * chorus_mix * layer["width"]
    right = mono * (1.0 + 0.35 * layer["pan"]) - delayed * chorus_mix * layer["width"]
    stereo = np.vstack([left, right])
    width_amount = max(
        0.0,
        (float(layer.get("width", 1.0)) - 1.0) * 0.35
        + (float(synth.get("stereo_spread", layer.get("width", 1.0))) - 1.0) * 0.25,
    )
    stereo = apply_decorrelation_width(stereo, sr, width_amount)
    stereo = apply_pan_width(stereo, float(layer.get("pan", 0.0)), 1.0)
    stereo *= db_to_amp(layer["gain_db"])
    stereo = apply_phaser(stereo, sr, float(layer["effects"].get("phaser_mix", 0.0)), float(mod.get("lfo_rate_hz", 0.23)) or 0.23)
    stereo = apply_delay(stereo, sr, float(layer["effects"].get("delay_time", 0.18)), float(layer["effects"].get("delay_mix", 0.0)))
    reverb_mix = layer["effects"]["reverb_mix"]
    if reverb_mix > 0:
        tail = np.zeros_like(stereo)
        delays = [int(sr * 0.083), int(sr * 0.137), int(sr * 0.211)]
        for delay_samples in delays:
            if delay_samples < samples:
                tail[:, delay_samples:] += stereo[:, :-delay_samples] * (0.35 / len(delays))
        stereo = stereo * (1.0 - reverb_mix) + tail * reverb_mix
    return stereo.astype(np.float32)


def apply_return_effect(audio: np.ndarray, ret: dict[str, Any], sr: int) -> np.ndarray:
    if ret.get("type", "reverb") != "reverb":
        return audio * db_to_amp(ret.get("gain_db", -12.0))
    samples = audio.shape[1]
    tail = np.zeros_like(audio)
    decay = float(ret.get("decay", 0.35))
    delays = [int(sr * 0.061), int(sr * 0.127), int(sr * 0.193), int(sr * 0.311)]
    for index, delay_samples in enumerate(delays):
        if delay_samples < samples:
            tail[:, delay_samples:] += audio[:, :-delay_samples] * (decay ** (index + 1)) / len(delays)
    width = float(ret.get("width", 1.0))
    mid = tail.mean(axis=0)
    side = (tail[0] - tail[1]) * 0.5 * width
    widened = np.vstack([mid + side, mid - side])
    return widened * db_to_amp(ret.get("gain_db", -12.0))


def render_session(session: dict[str, Any]) -> np.ndarray:
    sr = int(session["sample_rate"])
    duration = float(session["duration"])
    samples = int(duration * sr)
    mix = np.zeros((2, samples), dtype=np.float32)
    return_input = np.zeros((2, samples), dtype=np.float32)
    for layer in session.get("layers", []):
        rendered = render_layer(layer, duration, sr)
        mix += rendered
        return_input += rendered * float(layer.get("effects", {}).get("return_send", 0.0))
    for ret in session.get("returns", []):
        mix += apply_return_effect(return_input, ret, sr)
    width = session.get("master", {}).get("width", 1.0)
    mid = mix.mean(axis=0)
    side = (mix[0] - mix[1]) * 0.5 * width
    mix = np.vstack([mid + side, mid - side])
    mix *= db_to_amp(session.get("master", {}).get("gain_db", -1.0))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    return mix.astype(np.float32)


def session_shape() -> str:
    return json.dumps(DEFAULT_SESSION, indent=2)


def codex_producer_prompt(
    source_profile_path: Path,
    current_session_path: Path,
    recommendation_path: Path,
    step: int,
    max_layers: int,
) -> str:
    return f"""
You are the Producer agent in an autonomous audio reconstruction pipeline.

Read these files before answering:
- Source audio profile JSON: {source_profile_path}
- Current best reconstruction session JSON: {current_session_path}
- Critic recommendation JSON: {recommendation_path}
- Target source audio: {source_profile_path.parent / "source_clip.wav"}
- Source pattern constraints JSON: {source_profile_path.parent / "pattern_constraints.json"}
- Deterministic beat grid JSON: {source_profile_path.parent / "beat_grid.json"}

Task: Write the next complete reconstruction session JSON. Use the current session as the starting point and make the smallest useful DAW change that follows the Critic recommendation.

Renderer capabilities:
- Up to {max_layers} synth layers.
- Each layer has note events, gain/pan/width, wavetable synth params, ADSR amp envelope, automated lowpass cutoff_start_hz -> cutoff_end_hz, optional filter.cutoff_points, optional layer gain_points, optional modulation.gate_points, LFO tremolo, chorus_mix, phaser_mix, delay_mix, saturation, EQ, compression, and reverb_mix.
- Synth engine defaults to engine=vital for every layer.
- Vital layers render through the native Vital AudioUnit path. Normalized JSON controls are active: MIDI notes/velocity, oscillator blend/voices/detune/sub/wavetable position, ADSR envelope, filter cutoff/resonance/drive plus cutoff_points automation, gain_points, gate_points, LFO tremolo, pan/width, chorus/phaser/delay/reverb, saturation, compression, and EQ. Use synth.vital_parameters only for explicit raw Vital AU overrides; keys may be Vital parameter display names, identifiers, or numeric AudioUnit parameter addresses.
- Wavetable fields are renderer controls: wavetable/waveform choose a default Vital wave-frame region, and wavetable_position, warp, fm_amount, fm_ratio, blend, voices, detune_cents, stereo_spread, and sub_level are mapped into Vital oscillator parameters where available.
- Each layer can send to session returns through effects.return_send. Returns currently support reverb with id/type/gain_db/decay/width.
- waveform may be sine, triangle, saw, square, noise, air, or transient for fallback/noise layers.
- Session has returns and master fields. Preserve them even if empty.
- Dense MIDI/pattern layers may use up to {MAX_NOTES_PER_LAYER} note events. If pattern_constraints.json shows source onsets, write note events near those onset times instead of summarizing them as a held layer.
- Timing must be aligned to beat_grid.json. Use its tempo, meter, downbeat_time_seconds, and 1/16 grid edges when placing arpeggio/pulse notes or automation changes.

Rules:
- Write a complete session JSON file, not prose and not a patch fragment.
- Preserve useful existing layers and stable layer ids.
- Add/remove tracks, set Vital synth parameters, add LFO/modulation, add sidechain-like gate/gain movement, add compression/EQ/reverb/chorus/phaser/delay, or change MIDI/patterns as needed.
- Use synth/noise-like approximation only; do not reference external samples.
- The Critic's producer_prompt and success_metrics are binding. Follow the intent, but choose the actual session edits yourself.
- If the Critic or pattern constraints say MIDI/pattern/onsets are weak, update the notes list concretely. Do not describe an arpeggio without writing the note events.
- If the source motion is cyclic/modulated, use continuous modulation/LFO-style settings or dense smooth automation. Do not fake an LFO with one isolated chop or note hit.

The session JSON must use this shape:
{session_shape()}
"""


def codex_residual_critic_prompt(source_profile_path: Path, session_path: Path, audio_diff_path: Path | None) -> str:
    diff_line = f"- Latest audio similarity/diff JSON: {audio_diff_path}" if audio_diff_path is not None else "- No reconstruction has been scored yet."
    diff_guidance = (
        "Use the diff JSON to identify the biggest measurable mismatch: exact time ranges, band/envelope errors, modulation rate/depth errors, onset errors, pitch, stereo, or timbre."
        if audio_diff_path is not None
        else "Use the source measurements to recommend the first reconstruction move."
    )
    return f"""
You are the Critic agent in an audio reconstruction pipeline.

Read these files before answering:
- Source audio profile JSON: {source_profile_path}
- Current best reconstruction session JSON: {session_path}
- Target source audio: {source_profile_path.parent / "source_clip.wav"}
{diff_line}
- Source pattern constraints JSON: {source_profile_path.parent / "pattern_constraints.json"}
- Deterministic beat grid JSON: {source_profile_path.parent / "beat_grid.json"}

Task: Write the next Producer brief. {diff_guidance}

The recommendation JSON must use this shape:
{{
  "missing": ["specific measurable mismatches"],
  "producer_prompt": "plain-language prompt for the next Producer. It should reason like a DAW producer comparing target audio and current reconstruction, not prescribe a rigid JSON action list.",
  "success_metrics": {{
    "primary": ["loss components that must improve next"],
    "targets": ["concrete numeric or directional targets"],
    "failure_modes": ["what would prove the Producer made the wrong kind of change"]
  }},
  "recommendations": ["optional high-level DAW moves to consider, not mandatory actions"],
  "must_fix": ["highest priority perceptual/metric fixes to include in producer_prompt"],
  "do_not": ["specific mistakes the next Producer should avoid"],
  "success_criteria": ["numbers the next score should move toward"],
  "priority": "layer|automation|pitch|envelope|stereo|mix",
  "target_files": ["tracks/track_id.json or effects/effect_id.json that should change"],
  "target_slices": [{{"index": 0, "start": 0.0, "end": 0.1, "problem": "what is wrong here", "action": "specific fix at this time"}}],
  "stop_layer_building": true_or_false
}}
"""


def run_codex_json(
    output_dir: Path,
    agent: str,
    prompt: str,
    duration: float | None = None,
    sample_rate: int | None = None,
    json_output_path: Path | None = None,
) -> dict[str, Any]:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    answer_path = output_dir / f"codex_{agent}_answer.txt"
    if json_output_path is not None:
        prompt = (
            f"{prompt.rstrip()}\n\n"
            "File-driven orchestration requirement:\n"
            f"- Working timeline folder: {output_dir}\n"
            f"- Write or update the JSON artifact at: {json_output_path}\n"
            "- Base your edit on the files listed above. The orchestrator will read that JSON file after this run.\n"
            "- Your final chat message can be brief; the file is the source of truth.\n"
        )
    prompt_path.write_text(prompt)
    print(f"codex_request agent={agent} path={prompt_path}", flush=True)
    print(f"codex_start agent={agent}", flush=True)
    print(f"codex_prompt_path {prompt_path}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    print(f"codex_prompt_hidden agent={agent} bytes={len(prompt.encode())}", flush=True)
    process = subprocess.Popen(
        [
            CODEX_PATH,
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
        print(f"codex_log agent={agent} {line.rstrip()}", flush=True)
    try:
        returncode = process.wait(timeout=150)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"Codex agent {agent} timed out.")
    if returncode != 0:
        raise RuntimeError(f"Codex agent {agent} failed with return code {returncode}.")
    print(f"codex_done agent={agent} answer_path={answer_path}", flush=True)
    print(f"codex_response agent={agent} path={answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
    payload_text = answer_path.read_text()
    if json_output_path is not None and json_output_path.exists() and json_output_path.read_text().strip():
        payload_text = json_output_path.read_text()
    payload = extract_json_object(payload_text)
    if duration is not None and sample_rate is not None:
        return sanitize_session(payload, duration, sample_rate)
    return payload


def local_mutation(session: dict[str, Any], rng: np.random.Generator, amount: float, duration: float, sample_rate: int) -> dict[str, Any]:
    mutated = json.loads(json.dumps(session))
    if not mutated.get("layers"):
        return mutated
    layer = mutated["layers"][int(rng.integers(0, len(mutated["layers"])))]
    choice = rng.choice(["gain", "filter", "width", "detune", "envelope", "blend", "pattern", "wavetable", "effects"])
    if choice == "gain":
        layer["gain_db"] += float(rng.normal(0, 3.0 * amount))
    elif choice == "filter":
        layer["filter"]["cutoff_start_hz"] *= float(np.exp(rng.normal(0, amount)))
        layer["filter"]["cutoff_end_hz"] *= float(np.exp(rng.normal(0, amount)))
        layer["filter"]["drive"] = float(layer["filter"].get("drive", 0.0) + rng.normal(0, 0.25 * amount))
    elif choice == "width":
        layer["width"] += float(rng.normal(0, amount))
        layer["effects"]["chorus_mix"] += float(rng.normal(0, 0.2 * amount))
    elif choice == "detune":
        layer["synth"]["detune_cents"] += float(rng.normal(0, 10.0 * amount))
        layer["synth"]["voices"] = int(np.clip(layer["synth"]["voices"] + rng.choice([-1, 0, 1]), 1, 16))
    elif choice == "envelope":
        layer["amp_envelope"]["attack"] *= float(np.exp(rng.normal(0, amount)))
        layer["amp_envelope"]["release"] *= float(np.exp(rng.normal(0, amount)))
    elif choice == "blend":
        layer["synth"]["blend"] += float(rng.normal(0, amount))
        layer["synth"]["wavetable_position"] = float(layer["synth"].get("wavetable_position", 0.5) + rng.normal(0, amount))
    elif choice == "wavetable":
        layer["synth"]["wavetable"] = str(rng.choice(["saw_stack", "square_saw", "formant", "digital", "triangle"]))
        layer["synth"]["fm_amount"] = float(layer["synth"].get("fm_amount", 0.0) + rng.normal(0, 0.2 * amount))
        layer["synth"]["warp"] = float(layer["synth"].get("warp", 0.0) + rng.normal(0, 0.2 * amount))
    elif choice == "effects":
        layer["effects"]["phaser_mix"] = float(layer["effects"].get("phaser_mix", 0.0) + rng.normal(0, 0.3 * amount))
        layer["effects"]["saturation"] = float(layer["effects"].get("saturation", 0.0) + rng.normal(0, 0.4 * amount))
        layer["effects"]["mid_gain_db"] = float(layer["effects"].get("mid_gain_db", 0.0) + rng.normal(0, 4.0 * amount))
        layer["effects"]["high_gain_db"] = float(layer["effects"].get("high_gain_db", 0.0) + rng.normal(0, 4.0 * amount))
    elif layer.get("notes"):
        for note in layer["notes"]:
            if rng.random() < 0.25:
                note["start"] = float(np.clip(note.get("start", 0.0) + rng.normal(0, 0.018), 0.0, max(0.0, duration - 0.01)))
            if rng.random() < 0.2:
                note["duration"] = float(np.clip(note.get("duration", 0.08) * np.exp(rng.normal(0, 0.35 * amount)), 0.015, duration))
            if rng.random() < 0.16:
                note["note"] = int(np.clip(note.get("note", 52) + int(rng.choice([-12, -7, -5, 0, 5, 7, 12])), 24, 96))
    return sanitize_session(mutated, duration, sample_rate)


def local_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def trace_file(agent: str, role: str, path: Path) -> Path:
    print(f"trace_file agent={agent} role={role} path={path}", flush=True)
    return path


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_replay_command(output_dir: Path, argv: list[str]) -> Path:
    path = output_dir / "replay_command.sh"
    command = " ".join(shell_quote(str(part)) for part in argv)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    path.chmod(0o755)
    return path


def write_run_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    return write_json(output_dir / "run_manifest.json", payload)


def write_run_summary(output_dir: Path, status: str, history: list[dict[str, Any]] | None = None, best_scores: dict[str, Any] | None = None, note: str = "") -> Path:
    history = history or []
    lines = [
        "# Reconstruction Run",
        "",
        f"- Status: {status}",
        f"- Updated: {datetime.now(timezone.utc).isoformat()}",
    ]
    if best_scores:
        lines.append(f"- Best final score: {float(best_scores.get('final', 0.0)):.4f}")
    if note:
        lines.extend(["", note])
    if history:
        lines.extend(["", "## Iterations"])
        for item in history:
            score = float((item.get("scores") or {}).get("final") or 0.0)
            best = float((item.get("best_scores") or {}).get("final") or 0.0)
            lines.append(f"- Step {item.get('step')}: winner={item.get('winner')} accepted={item.get('accepted')} score={score:.4f} best={best:.4f}")
    lines.extend(["", "## Debug Files", "- events.jsonl", "- raw_subprocess.log", "- logs/*.log", "- run_manifest.json", "- replay_command.sh"])
    path = output_dir / "run_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_chunked_session(root: Path, session: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tracks_dir = root / "tracks"
    instruments_dir = root / "instruments"
    modulators_dir = root / "modulators"
    effects_dir = root / "effects"
    tracks_dir.mkdir(exist_ok=True)
    instruments_dir.mkdir(exist_ok=True)
    modulators_dir.mkdir(exist_ok=True)
    effects_dir.mkdir(exist_ok=True)
    manifest = {
        "version": session.get("version", 1),
        "sample_rate": session.get("sample_rate", 44100),
        "duration": session.get("duration", 5.0),
        "tracks": [],
        "instruments": [],
        "modulators": [],
        "effects": [],
        "returns": [],
        "master": "master.json",
        "routing": "routing.json",
        "timeline": "timeline.json",
    }
    routing = {"track_order": [], "sends": []}
    timeline = {"duration": session.get("duration", 5.0), "notes": [], "automation": []}
    for index, layer in enumerate(session.get("layers", [])):
        track_id = str(layer.get("id") or f"track_{index + 1}")
        track_path = tracks_dir / f"{track_id}.json"
        track_payload = {key: value for key, value in layer.items() if key not in {"synth", "modulation", "effects"}}
        track_payload["instrument"] = f"../instruments/{track_id}.json"
        track_payload["modulators"] = []
        track_payload["effects"] = f"../effects/{track_id}.json"
        write_json(track_path, track_payload)
        write_json(instruments_dir / f"{track_id}.json", {"track_id": track_id, **layer.get("synth", {})})
        write_json(effects_dir / f"{track_id}.json", {"track_id": track_id, **layer.get("effects", {})})
        manifest["tracks"].append(str(track_path.relative_to(root)))
        manifest["instruments"].append(str((instruments_dir / f"{track_id}.json").relative_to(root)))
        manifest["effects"].append(str((effects_dir / f"{track_id}.json").relative_to(root)))
        routing["track_order"].append(track_id)
        routing["sends"].append({"track_id": track_id, "return_id": "space", "amount": float(layer.get("effects", {}).get("return_send", 0.0))})
        timeline["notes"].append({"track_id": track_id, "notes": layer.get("notes", [])})
        timeline["automation"].append({"track_id": track_id, "gain_points": layer.get("gain_points", []), "filter": layer.get("filter", {})})
        for lfo_index, lfo in enumerate(layer.get("modulation", {}).get("lfos", [])):
            mod_path = modulators_dir / f"{track_id}_lfo_{lfo_index + 1}.json"
            write_json(mod_path, {"track_id": track_id, **lfo})
            track_payload["modulators"].append(f"../modulators/{mod_path.name}")
            manifest["modulators"].append(str(mod_path.relative_to(root)))
        write_json(track_path, track_payload)
    for index, ret in enumerate(session.get("returns", [])):
        return_id = str(ret.get("id") or f"return_{index + 1}")
        return_path = effects_dir / f"{return_id}.json"
        write_json(return_path, ret)
        manifest["returns"].append(str(return_path.relative_to(root)))
    write_json(root / "master.json", session.get("master", {}))
    write_json(root / "routing.json", routing)
    write_json(root / "timeline.json", timeline)
    return write_json(root / "manifest.json", manifest)


def read_chunked_session(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    session = {
        "version": manifest.get("version", 1),
        "sample_rate": manifest.get("sample_rate", 44100),
        "duration": manifest.get("duration", 5.0),
        "layers": [],
        "returns": [],
        "master": json.loads((root / manifest.get("master", "master.json")).read_text()),
    }
    for track in manifest.get("tracks", []):
        track_payload = json.loads((root / track).read_text())
        track_id = str(track_payload.get("id") or Path(track).stem)
        instrument_path = track_payload.get("instrument", f"../instruments/{track_id}.json")
        effects_path = track_payload.get("effects", f"../effects/{track_id}.json")
        instrument = json.loads((root / Path(track).parent / instrument_path).resolve().read_text()) if (root / Path(track).parent / instrument_path).resolve().exists() else {}
        effects = json.loads((root / Path(track).parent / effects_path).resolve().read_text()) if (root / Path(track).parent / effects_path).resolve().exists() else {}
        lfos = []
        for mod_path in track_payload.get("modulators", []):
            full = (root / Path(track).parent / mod_path).resolve()
            if full.exists():
                mod = json.loads(full.read_text())
                mod.pop("track_id", None)
                lfos.append(mod)
        layer = {key: value for key, value in track_payload.items() if key not in {"instrument", "effects", "modulators"}}
        instrument.pop("track_id", None)
        effects.pop("track_id", None)
        layer["synth"] = instrument
        layer["effects"] = effects
        modulation = layer.get("modulation", {})
        if not isinstance(modulation, dict):
            modulation = {}
        modulation["lfos"] = lfos
        layer["modulation"] = modulation
        session["layers"].append(layer)
    for ret in manifest.get("returns", []):
        session["returns"].append(json.loads((root / ret).read_text()))
    return sanitize_session(session, float(session["duration"]), int(session["sample_rate"]))


def onset_frames_to_notes(diagnostics: dict[str, Any], sample_rate: int, midi_note: int = 52) -> list[dict[str, Any]]:
    positions = diagnostics.get("reference_onset_positions", [])
    notes = []
    for index, frame in enumerate(positions[:MAX_NOTES_PER_LAYER]):
        start = max(0.0, float(frame) * 512.0 / sample_rate)
        note = midi_note + [0, 7, 12, 7, 3, 10][index % 6]
        notes.append({"note": int(np.clip(note, 24, 96)), "start": round(start, 4), "duration": 0.07, "velocity": 0.45 + 0.25 * ((index % 4) / 3.0)})
    return notes


def pattern_constraints(profile: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    diagnostics = profile.get("diagnostics", {})
    beat_grid = profile.get("beat_grid", {})
    onset_count = int(profile.get("onset_count", diagnostics.get("reference_onset_count", 0)) or 0)
    return {
        "requires_pattern": onset_count >= 12,
        "target_onset_count": onset_count,
        "tempo": beat_grid.get("tempo"),
        "meter": beat_grid.get("meter", "4/4"),
        "subdivision": beat_grid.get("subdivision", "1/16"),
        "step_seconds": beat_grid.get("step_seconds"),
        "downbeat_time_seconds": beat_grid.get("downbeat_time_seconds", beat_grid.get("phase_seconds", 0.0)),
        "grid_edges_seconds": beat_grid.get("edges_seconds", []),
        "reference_onset_positions": diagnostics.get("reference_onset_positions", []),
        "reference_onset_times_seconds": [round(float(frame) * 512.0 / sample_rate, 4) for frame in diagnostics.get("reference_onset_positions", [])],
        "starter_notes": onset_frames_to_notes(diagnostics, sample_rate),
    }


def add_or_replace_pattern_layer(session: dict[str, Any], constraints: dict[str, Any], duration: float, sample_rate: int) -> dict[str, Any]:
    if not constraints.get("requires_pattern"):
        return sanitize_session(session, duration, sample_rate)
    mutated = json.loads(json.dumps(session))
    notes = constraints.get("starter_notes", [])
    if not notes:
        return sanitize_session(mutated, duration, sample_rate)
    layer = {
        "id": "arp_pattern",
        "role": "detected arpeggiated or pulsed MIDI pattern reconstructed from source onset times",
        "gain_db": -8.0,
        "pan": 0.0,
        "width": 0.72,
        "notes": notes,
        "synth": {
            "engine": "vital",
            "waveform": "saw",
            "wavetable": "saw_stack",
            "wavetable_position": 0.62,
            "warp": 0.1,
            "fm_amount": 0.04,
            "fm_ratio": 2.0,
            "blend": 0.5,
            "voices": 5,
            "detune_cents": 12.0,
            "stereo_spread": 0.75,
            "sub_level": 0.0,
        },
        "amp_envelope": {"attack": 0.004, "decay": 0.08, "sustain": 0.28, "release": 0.08},
        "filter": {"cutoff_start_hz": 650.0, "cutoff_end_hz": 900.0, "resonance": 0.18, "drive": 0.18},
        "modulation": {"lfo_rate_hz": 0.3, "lfo_depth": 0.05},
        "effects": {"chorus_mix": 0.18, "phaser_mix": 0.08, "saturation": 0.12, "reverb_mix": 0.02, "return_send": 0.02},
    }
    layers = [existing for existing in mutated.get("layers", []) if existing.get("id") not in {"arp_pattern", "pulse_onsets"}]
    layers.append(layer)
    mutated["layers"] = layers
    return sanitize_session(mutated, duration, sample_rate)


def temporal_scores_ok(score: AudioScore) -> bool:
    thresholds = {
        "segment_envelope": 0.72,
        "late_energy_ratio": 0.72,
        "sustain_coverage": 0.72,
        "frontload_balance": 0.72,
        "band_envelope_by_time": 0.68,
        "exact_envelope_50ms": 0.72,
        "exact_band_50ms": 0.70,
        "directional_delta": 0.70,
        "modulation_periodicity": 0.62,
        "onset_count": 0.72,
        "onset_timing": 0.72,
        "centroid_trajectory": 0.68,
    }
    return all(float(getattr(score, name)) >= threshold for name, threshold in thresholds.items())


def structural_scores_ok(score: AudioScore, diagnostics: dict[str, Any]) -> bool:
    ref_count = int(diagnostics.get("reference_onset_count", 0) or 0)
    cand_count = int(diagnostics.get("candidate_onset_count", 0) or 0)
    if ref_count >= 16 and cand_count < max(10, int(ref_count * 0.65)):
        return False
    if ref_count >= 16 and (score.onset_count < 0.62 or score.onset_timing < 0.42):
        return False
    if score.f0_contour < 0.24:
        return False
    if score.spectral_features < 0.38 or score.harmonic_noise < 0.38:
        return False
    if score.beat_grid_mel < 0.58 or score.beat_grid_band < 0.58 or score.exact_band_50ms < 0.58:
        return False
    return True


def score_from_json(payload: dict[str, float]) -> AudioScore:
    return AudioScore(**{field: float(payload.get(field, 0.0)) for field in AudioScore.__dataclass_fields__})


def write_audio_diff(reference_audio: np.ndarray, session: dict[str, Any], sample_rate: int, audio_path: Path, diff_path: Path, beat_grid: dict[str, Any] | None = None) -> tuple[AudioScore, dict[str, Any], dict[str, Any]]:
    rendered = render_session(session)
    sf.write(audio_path, rendered.T, sample_rate)
    diff = compare_audio(reference_audio, rendered, sample_rate, beat_grid=beat_grid)
    write_json(diff_path, diff)
    return score_from_json(diff["scores"]), diff["diagnostics"], diff["residual"]


def score_candidate_with_inner_trials(
    reference_audio: np.ndarray,
    base_session: dict[str, Any],
    output_dir: Path,
    step: int,
    label_prefix: str,
    rng: np.random.Generator,
    duration: float,
    sample_rate: int,
    trials: int,
    pattern_info: dict[str, Any],
    beat_grid: dict[str, Any],
) -> list[tuple[float, str, dict[str, Any], AudioScore, dict[str, Any], dict[str, Any], Path, Path]]:
    candidates = [(label_prefix, sanitize_session(base_session, duration, sample_rate))]
    if pattern_info.get("requires_pattern"):
        candidates.append((f"{label_prefix}_pattern_seed", add_or_replace_pattern_layer(base_session, pattern_info, duration, sample_rate)))
    for trial in range(trials):
        source = candidates[min(len(candidates) - 1, 1)][1] if len(candidates) > 1 and trial % 2 else base_session
        candidates.append((f"{label_prefix}_inner_{trial}", local_mutation(source, rng, 0.18 + 0.04 * trial, duration, sample_rate)))
    results = []
    for label, candidate_session in candidates:
        out_path = output_dir / f"producer_reconstruction_step_{step:02d}_{label}.wav"
        diff_path = output_dir / f"producer_audio_diff_step_{step:02d}_{label}.json"
        score, diagnostics, residual = write_audio_diff(reference_audio, candidate_session, sample_rate, out_path, diff_path, beat_grid=beat_grid)
        write_json(output_dir / f"session_step_{step:02d}_{label}.json", candidate_session)
        print(f"trace_file agent=loss step={step} role=producer_audio_diff_{label} path={diff_path}", flush=True)
        print(f"trace_file agent=loss step={step} role=producer_render_{label} path={out_path}", flush=True)
        gate = structural_scores_ok(score, diagnostics)
        gated_final = score.final if gate else score.final * 0.82
        results.append((score.final, label, candidate_session, score, diagnostics, residual, out_path, diff_path))
        print(
            f"step={step} candidate={label} phase=producer_trial score={score.final:.4f} gated={gated_final:.4f} structural_gate={str(gate).lower()} exact_env_50ms={score.exact_envelope_50ms:.4f} exact_band_50ms={score.exact_band_50ms:.4f} mod_period={score.modulation_periodicity:.4f} mod_rate={score.modulation_rate:.4f} mod_depth={score.modulation_depth:.4f} direction={score.directional_delta:.4f} transient_class={score.transient_classification:.4f} mel={score.mel_spectrogram:.4f} beat_mel={score.beat_grid_mel:.4f} beat_band={score.beat_grid_band:.4f} beat_env={score.beat_grid_envelope:.4f} envelope={score.envelope:.4f} segment_envelope={score.segment_envelope:.4f} late={score.late_energy_ratio:.4f} sustain={score.sustain_coverage:.4f} frontload={score.frontload_balance:.4f} band_time={score.band_envelope_by_time:.4f} chroma={score.pitch_chroma:.4f} f0={score.f0_contour:.4f} motion={score.spectral_motion:.4f} timbre={score.spectral_features:.4f} onset_count={score.onset_count:.4f} onset_timing={score.onset_timing:.4f} stereo={score.stereo_width:.4f}",
            flush=True,
        )
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def source_profile(reference_audio: np.ndarray, sample_rate: int, duration: float, beat_grid: dict[str, Any]) -> dict[str, Any]:
    silence = np.zeros_like(reference_audio)
    profile_diff = compare_audio(reference_audio, silence, sample_rate, beat_grid=beat_grid)
    diagnostics = profile_diff["diagnostics"]
    return {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "beat_grid": beat_grid,
        "tempo": beat_grid.get("tempo"),
        "meter": beat_grid.get("meter", "4/4"),
        "subdivision": beat_grid.get("subdivision", "1/16"),
        "downbeat_time_seconds": beat_grid.get("downbeat_time_seconds", beat_grid.get("phase_seconds", 0.0)),
        "downbeat_frame": beat_grid.get("downbeat_frame", 0),
        "rms": float(np.sqrt(np.mean(local_mono(reference_audio) ** 2) + 1e-12)),
        "diagnostics": diagnostics,
        "band_energy": diagnostics.get("band_energy", {}),
        "centroid": {
            "start_hz": diagnostics.get("reference_centroid_start_hz", 0.0),
            "end_hz": diagnostics.get("reference_centroid_end_hz", 0.0),
        },
        "onset_count": diagnostics.get("reference_onset_count", 0),
        "stereo": diagnostics.get("reference_stereo", {}),
    }


def distilled_playable_patch(session: dict[str, Any]) -> dict[str, Any]:
    layers = session.get("layers") or []
    primary = layers[0] if layers else sanitize_session({"layers": [{}]}, session.get("duration", 5.0), session.get("sample_rate", 44100))["layers"][0]
    notes = primary.get("notes") or [{"note": 60}]
    effects = primary.get("effects", {})
    filt = primary.get("filter", {})
    synth = primary.get("synth", {})
    amp = primary.get("amp_envelope", {})
    return {
        "instrument_type": "reconstructed_session_patch",
        "source": "v1_reconstruction_session",
        "root_midi_note": int(notes[0].get("note", 60)),
        "layers": [
            {
                "id": layer.get("id", f"layer_{index + 1}"),
                "role": layer.get("role", "synth layer"),
                "relative_gain_db": layer.get("gain_db", -8.0),
                "pan": layer.get("pan", 0.0),
                "width": layer.get("width", 0.6),
                "synth": layer.get("synth", {}),
                "amp_envelope": layer.get("amp_envelope", {}),
                "filter": layer.get("filter", {}),
                "effects": layer.get("effects", {}),
            }
            for index, layer in enumerate(layers[:4])
        ],
        "macros": {
            "brightness": float(np.clip((filt.get("cutoff_end_hz", 1800.0) - 200.0) / 6000.0, 0.0, 1.0)),
            "movement": float(np.clip(primary.get("modulation", {}).get("lfo_depth", 0.0), 0.0, 1.0)),
            "space": float(np.clip(effects.get("reverb_mix", 0.0) / 0.7, 0.0, 1.0)),
            "width": float(np.clip(primary.get("width", 0.6), 0.0, 1.0)),
            "attack": float(np.clip(amp.get("attack", 0.05) / 4.0, 0.0, 1.0)),
        },
        "keyboard_mapping": {
            "pitch_tracking": True,
            "velocity_to_amp": True,
            "mod_wheel": "brightness",
            "aftertouch": "space",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--render-session-dir", type=Path, help="Render a chunked session directory containing manifest.json and exit.")
    parser.add_argument("--render-output", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--local-trials", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    setup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": str(args.reference) if args.reference else None,
        "output_dir": str(args.output_dir),
        "argv": sys.argv,
        "steps": args.steps,
        "local_trials": args.local_trials,
        "max_layers": args.max_layers,
        "sample_rate": args.sample_rate,
        "seconds": args.seconds,
        "artifacts": {
            "events": "events.jsonl",
            "raw_subprocess_log": "raw_subprocess.log",
            "agent_logs": "logs/*.log",
        },
    }
    write_run_manifest(args.output_dir, manifest)
    write_replay_command(args.output_dir, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    write_run_summary(args.output_dir, "running", note="Run initialized.")
    if args.render_session_dir:
        session = read_chunked_session(args.render_session_dir)
        audio = render_session(session)
        render_output = args.render_output or args.output_dir / "chunked_render.wav"
        sf.write(render_output, audio.T, int(session["sample_rate"]))
        print(f"wrote {render_output}", flush=True)
        return
    if args.reference is None:
        raise SystemExit("--reference is required unless --render-session-dir is provided")
    reference_audio, reference_sr = load_audio(args.reference)
    if reference_sr != args.sample_rate:
        raise RuntimeError(f"Expected {args.sample_rate} Hz reference, got {reference_sr}. Extract the clip through the UI first.")
    reference_audio = reference_audio[:, : int(args.seconds * args.sample_rate)]
    reference_clip = args.output_dir / "source_clip.wav"
    sf.write(reference_clip, reference_audio.T, args.sample_rate)
    manifest["source_clip"] = str(reference_clip)
    write_run_manifest(args.output_dir, manifest)

    beat_grid = estimate_beat_grid(reference_audio, args.sample_rate, subdivision=4)
    trace_file("residual_critic", "beat_grid", write_json(args.output_dir / "beat_grid.json", beat_grid))
    profile = source_profile(reference_audio, args.sample_rate, args.seconds, beat_grid)
    pattern_info = pattern_constraints(profile, args.sample_rate)
    profile["pattern_constraints"] = pattern_info
    profile["synth_engine"] = {"active": "vital_audio_unit", "fallback": "internal_wavetable", "vital": vital_status()}
    source_profile_path = trace_file("residual_critic", "source_profile", write_json(args.output_dir / "source_profile.json", profile))
    trace_file("residual_critic", "pattern_constraints", write_json(args.output_dir / "pattern_constraints.json", pattern_info))

    session = sanitize_session(DEFAULT_SESSION, args.seconds, args.sample_rate)
    current_session_path = trace_file("session", "current", write_json(args.output_dir / "session_current.json", session))
    chunk_manifest_path = trace_file("session", "chunked_manifest", write_chunked_session(args.output_dir / "session_chunks" / "current", session))
    history: list[dict[str, Any]] = []
    history_path = write_json(args.output_dir / "history.json", history)
    best_session = session
    best_score = AudioScore(**{field: 0.0 for field in AudioScore.__dataclass_fields__})
    previous_diff_path: Path | None = None
    rng = np.random.default_rng(20260506)
    min_builder_steps = min(args.steps, 5)

    for step in range(args.steps):
        print(f"agent_stage residual_critic step={step}", flush=True)
        critic = run_codex_json(
            args.output_dir,
            f"residual_critic_step_{step:02d}",
            codex_residual_critic_prompt(source_profile_path, current_session_path, previous_diff_path),
            json_output_path=args.output_dir / f"recommendation_step_{step:02d}.json",
        )
        recommendation = {
            "missing": critic.get("missing", ["empty session"] if previous_diff_path is None else []),
            "producer_prompt": critic.get("producer_prompt", "Make the smallest DAW-style session change that should improve the current reconstruction."),
            "success_metrics": critic.get("success_metrics", {"primary": [], "targets": [], "failure_modes": []}),
            "recommendations": critic.get("recommendations", []),
            "must_fix": critic.get("must_fix", critic.get("missing", [])[:4]),
            "do_not": critic.get("do_not", []),
            "success_criteria": critic.get("success_criteria", []),
            "priority": critic.get("priority", "layer"),
            "target_files": critic.get("target_files", []),
            "target_slices": critic.get("target_slices", []),
            "stop_layer_building": bool(critic.get("stop_layer_building", False)),
        }
        recommendation_path = trace_file(f"residual_critic_step_{step:02d}", "recommendation", write_json(args.output_dir / f"recommendation_step_{step:02d}.json", recommendation))

        print(f"agent_stage producer step={step}", flush=True)
        proposed = run_codex_json(
            args.output_dir,
            f"producer_step_{step:02d}",
            codex_producer_prompt(source_profile_path, current_session_path, recommendation_path, step, args.max_layers),
            args.seconds,
            args.sample_rate,
            json_output_path=args.output_dir / f"session_step_{step:02d}_codex_proposal.json",
        )
        proposed_session_path = trace_file(f"producer_step_{step:02d}", "session_proposal", write_json(args.output_dir / f"session_step_{step:02d}_codex_proposal.json", proposed))
        step_results = score_candidate_with_inner_trials(
            reference_audio,
            proposed,
            args.output_dir,
            step,
            "codex",
            rng,
            args.seconds,
            args.sample_rate,
            args.local_trials,
            pattern_info,
            beat_grid,
        )
        score_value, label, session, score, diagnostics, candidate_residual, out_path, diff_path = step_results[0]
        trace_file(f"producer_step_{step:02d}", "candidate_session", write_json(args.output_dir / f"session_step_{step:02d}_producer_winner.json", session))
        print(f"producer_winner step={step} winner={label} score={score_value:.4f}", flush=True)
        print(f"trace_file agent=loss step={step} role=audio_diff_winner path={diff_path}", flush=True)
        print(f"trace_file agent=loss step={step} role=render_winner path={out_path}", flush=True)
        print(
            f"step={step} candidate={label} phase=producer score={score.final:.4f} exact_env_50ms={score.exact_envelope_50ms:.4f} exact_band_50ms={score.exact_band_50ms:.4f} mod_period={score.modulation_periodicity:.4f} mod_rate={score.modulation_rate:.4f} mod_depth={score.modulation_depth:.4f} direction={score.directional_delta:.4f} transient_class={score.transient_classification:.4f} mel={score.mel_spectrogram:.4f} beat_mel={score.beat_grid_mel:.4f} beat_band={score.beat_grid_band:.4f} beat_env={score.beat_grid_envelope:.4f} envelope={score.envelope:.4f} segment_envelope={score.segment_envelope:.4f} late={score.late_energy_ratio:.4f} sustain={score.sustain_coverage:.4f} frontload={score.frontload_balance:.4f} band_time={score.band_envelope_by_time:.4f} chroma={score.pitch_chroma:.4f} motion={score.spectral_motion:.4f} onset_count={score.onset_count:.4f} onset_timing={score.onset_timing:.4f} stereo={score.stereo_width:.4f}",
            flush=True,
        )
        print(f"winner_summary step={step} codex_proposed=true winner={label} codex_won={str(label == 'codex').lower()} score={score.final:.4f}", flush=True)
        trace_file(f"loss_step_{step:02d}", "winner_render", out_path)
        trace_file(f"loss_step_{step:02d}", "winner_audio_diff", diff_path)
        structural_ok = structural_scores_ok(score, diagnostics)
        accepted = score.final >= best_score.final
        if accepted:
            best_session = session
            best_score = score
            current_session_path = trace_file("session", "current", write_json(args.output_dir / "session_current.json", best_session))
            chunk_manifest_path = trace_file("session", "chunked_manifest", write_chunked_session(args.output_dir / "session_chunks" / "current", best_session))
        deterministic_residual = dict(candidate_residual)
        if len(best_session.get("layers", [])) < 2 and best_score.final < 0.72:
            deterministic_residual.setdefault("recommendations", []).append("prefer adding one concrete layer over overfitting the first layer")
        history_item = {
            "stage": "producer",
            "step": step,
            "accepted": accepted,
            "winner": label,
            "audio_path": str(out_path),
            "audio_diff_path": str(diff_path),
            "proposal_session_path": str(proposed_session_path),
            "chunk_manifest_path": str(chunk_manifest_path),
            "scores": score_to_json(score),
            "best_scores": score_to_json(best_score),
            "structural_gate": structural_ok,
            "layers": [{"id": layer["id"], "role": layer["role"]} for layer in best_session.get("layers", [])],
            "deterministic_residual": deterministic_residual,
            "recommendation_path": str(recommendation_path),
            "residual_critic": recommendation,
        }
        history_item_path = trace_file(f"loss_step_{step:02d}", "history_item", write_json(args.output_dir / f"history_item_step_{step:02d}.json", history_item))
        temporal_ok = temporal_scores_ok(best_score)
        structural_ok = structural_scores_ok(best_score, diagnostics)
        forced_continue = step + 1 < min_builder_steps or best_score.final < 0.78 or not temporal_ok or not structural_ok
        if forced_continue and recommendation["stop_layer_building"]:
            recommendation["stop_layer_building"] = False
            recommendation_path = trace_file(f"residual_critic_step_{step:02d}", "recommendation", write_json(args.output_dir / f"recommendation_step_{step:02d}.json", recommendation))
            history_item["recommendation_path"] = str(recommendation_path)
            history_item["residual_critic"] = recommendation
            history_item_path = trace_file(f"loss_step_{step:02d}", "history_item", write_json(args.output_dir / f"history_item_step_{step:02d}.json", history_item))
        history.append(history_item)
        history_path = write_json(args.output_dir / "history.json", history)
        write_run_summary(args.output_dir, "running", history, score_to_json(best_score))
        trace_file(f"producer_step_{step:02d}", "accepted_session", write_json(args.output_dir / f"session_step_{step:02d}_accepted.json", best_session))
        print(f"step_complete index={step} winner={label} accepted={str(accepted).lower()} best_score={best_score.final:.4f}", flush=True)
        previous_diff_path = Path(diff_path)
        if recommendation.get("stop_layer_building") and len(best_session.get("layers", [])) >= 1 and step + 1 >= min_builder_steps and best_score.final >= 0.78 and temporal_scores_ok(best_score) and structural_scores_ok(best_score, diagnostics):
            print(f"layer_building_stopped step={step} reason=residual_critic", flush=True)
            break

    final_audio = render_session(best_session)
    final_path = args.output_dir / "final_reconstruction.wav"
    session_path = args.output_dir / "reconstruction_session.json"
    session_alias_path = args.output_dir / "session.json"
    playable_path = args.output_dir / "distilled_playable_patch.json"
    report_path = args.output_dir / "reconstruction_report.json"
    sf.write(final_path, final_audio.T, args.sample_rate)
    session_path.write_text(json.dumps(best_session, indent=2) + "\n")
    session_alias_path.write_text(json.dumps(best_session, indent=2) + "\n")
    playable_path.write_text(json.dumps(distilled_playable_patch(best_session), indent=2) + "\n")
    report = {
        "reference": str(args.reference),
        "source_clip": str(reference_clip),
        "source_profile": profile,
        "empty_session_schema": sanitize_session(DEFAULT_SESSION, args.seconds, args.sample_rate),
        "history": history,
        "best_scores": score_to_json(best_score),
        "final_path": str(final_path),
        "session_path": str(session_path),
        "session_alias_path": str(session_alias_path),
        "distilled_playable_patch_path": str(playable_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_path"] = str(final_path)
    manifest["report_path"] = str(report_path)
    manifest["best_scores"] = score_to_json(best_score)
    write_run_manifest(args.output_dir, manifest)
    write_run_summary(args.output_dir, "completed", history, score_to_json(best_score))
    print(f"wrote {final_path}", flush=True)
    print(f"wrote {session_path}", flush=True)
    print(f"wrote {session_alias_path}", flush=True)
    print(f"wrote {playable_path}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
