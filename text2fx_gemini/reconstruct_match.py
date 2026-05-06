#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
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
    return {"installed_paths": installed, "pedalboard_loadable": loadable, "error": error}


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
        notes = layer.get("notes", [])
        if not isinstance(notes, list) or not notes:
            notes = [{"note": synth.get("note", 48), "start": 0.0, "duration": duration, "velocity": 0.7}]
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
                    "waveform": str(synth.get("waveform", "saw")).lower(),
                    "engine": str(synth.get("engine", "wavetable")).lower(),
                    "wavetable": str(synth.get("wavetable", synth.get("waveform", "saw_stack"))).lower(),
                    "wavetable_position": float(np.clip(float(synth.get("wavetable_position", 0.5)), 0.0, 1.0)),
                    "warp": float(np.clip(float(synth.get("warp", 0.0)), 0.0, 1.0)),
                    "fm_amount": float(np.clip(float(synth.get("fm_amount", 0.0)), 0.0, 1.0)),
                    "fm_ratio": float(np.clip(float(synth.get("fm_ratio", 2.0)), 0.25, 12.0)),
                    "blend": float(np.clip(float(synth.get("blend", 0.35)), 0.0, 1.0)),
                    "voices": int(np.clip(int(synth.get("voices", 3)), 1, 16)),
                    "detune_cents": float(np.clip(float(synth.get("detune_cents", 8.0)), 0.0, 70.0)),
                    "stereo_spread": float(np.clip(float(synth.get("stereo_spread", 0.55)), 0.0, 1.0)),
                    "sub_level": float(np.clip(float(synth.get("sub_level", 0.15)), 0.0, 1.0)),
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
                    "lfo_rate_hz": float(np.clip(float(layer.get("modulation", {}).get("lfo_rate_hz", 0.15)), 0.0, 8.0)),
                    "lfo_depth": float(np.clip(float(layer.get("modulation", {}).get("lfo_depth", 0.05)), 0.0, 1.0)),
                    "gate_points": sanitize_points(layer.get("modulation", {}).get("gate_points"), duration, "level", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}], 0.0, 1.0),
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
                    "return_send": float(np.clip(float(effects.get("return_send", 0.0)), 0.0, 1.0)),
                },
            }
        )
    master = payload.get("master", {})
    session["master"] = {
        "gain_db": float(np.clip(float(master.get("gain_db", -1.0)), -24.0, 6.0)),
        "width": float(np.clip(float(master.get("width", 1.0)), 0.0, 1.5)),
    }
    return session


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


def render_layer(layer: dict[str, Any], duration: float, sr: int) -> np.ndarray:
    samples = int(duration * sr)
    mono = np.zeros(samples, dtype=np.float32)
    synth = layer["synth"]
    amp = layer["amp_envelope"]
    filt = layer["filter"]
    mod = layer["modulation"]
    gain_curve = automation_curve(layer.get("gain_points", [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}]), "db", samples, duration)
    gate_curve = automation_curve(mod.get("gate_points", [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}]), "level", samples, duration)
    cutoff_curve = automation_curve(filt.get("cutoff_points", [{"time": 0.0, "hz": filt["cutoff_start_hz"]}, {"time": duration, "hz": filt["cutoff_end_hz"]}]), "hz", samples, duration)
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
    mono *= np.power(10.0, gain_curve / 20.0)
    mono = layer_eq(mono, sr, layer["effects"])
    saturation = float(layer["effects"].get("saturation", 0.0))
    if saturation:
        mono = np.tanh(mono * (1.0 + 6.0 * saturation)) / np.tanh(1.0 + 6.0 * saturation)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono *= 1.0 / peak
    chorus_mix = layer["effects"]["chorus_mix"]
    delay = max(1, int(0.012 * sr))
    delayed = np.pad(mono[:-delay], (delay, 0)) if mono.size > delay else mono
    left = mono * (1.0 - 0.35 * layer["pan"]) + delayed * chorus_mix * layer["width"]
    right = mono * (1.0 + 0.35 * layer["pan"]) - delayed * chorus_mix * layer["width"]
    stereo = np.vstack([left, right]) * db_to_amp(layer["gain_db"])
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


def codex_analyzer_prompt(source_clip_path: Path, source_profile_path: Path, duration: float) -> str:
    return f"""
You are the Analyzer agent in a file-driven audio reconstruction pipeline.

Read these files before answering:
- Source clip WAV: {source_clip_path}
- Source audio profile JSON: {source_profile_path}

Task: Produce a structured layer plan and reconstruction constraints. This is not vibe analysis. Infer likely layer architecture from the source profile: band energy, centroid motion, onset count, stereo stats, and duration.

Update the JSON artifact requested by the orchestrator with this structure:
{{
  "global": {{
    "tempo": 60-180 or null,
    "key": "estimated key or unknown",
    "meter": "4/4 or unknown",
    "duration_seconds": {duration},
    "overall_mix": "compact reconstruction summary"
  }},
  "layers": [
    {{
      "id": "short_layer_id",
      "role": "what this layer contributes",
      "instrument": "specific synth/instrument family",
      "confidence": 0-1,
      "pitch_content": ["note/chord estimates"],
      "timing": "held, pulsed, arpeggiated, transient, or ambience",
      "synth_hypothesis": {{
        "oscillators": "concrete oscillator/unison idea",
        "filter": "concrete filter idea",
        "amp_envelope": "concrete envelope idea",
        "modulation": "concrete modulation idea"
      }},
      "effects": ["specific effects"]
    }}
  ],
  "constraints": ["specific measurable constraints for reconstruction"],
  "strategy": ["ordered reconstruction steps, each adding or modifying a measurable layer"]
}}

Prefer 2-5 layers for dense clips and 1-3 layers for sparse clips. Treat this as a hypothesis that later audio-diff files can correct.
"""


def codex_layer_builder_prompt(
    analyzer_path: Path,
    current_session_path: Path,
    recommendation_path: Path,
    history_path: Path,
    step: int,
    max_layers: int,
) -> str:
    phase = [
        "create the dominant audible layer",
        "inspect residual and add or modify the second most important layer",
        "add movement/filter automation or correct pitch/envelope mismatch",
        "add stereo width, reverb/space, chorus/phaser-like motion, or ambience layer",
        "make a conservative reconstruction pass before mixing",
    ][min(step, 4)]
    return f"""
You are the Producer agent in an autonomous audio reconstruction pipeline.

Read these files before answering:
- Analyzer layer plan: {analyzer_path}
- Current best reconstruction session JSON: {current_session_path}
- Previous Critic recommendation JSON: {recommendation_path}
- Score/history JSON: {history_path}
- Source pattern constraints JSON: {analyzer_path.parent / "pattern_constraints.json"}
- Deterministic beat grid JSON: {analyzer_path.parent / "beat_grid.json"}
- Current chunked session manifest if present: {analyzer_path.parent / "session_chunks" / "current" / "manifest.json"}

Task: Given those files, update the requested session artifact with the smallest concrete full-session JSON change that should reduce reconstruction error.

Current Producer run: {step + 1}
Allowed action for this run: {phase}

Renderer capabilities:
- Up to {max_layers} synth layers.
- Each layer has note events, gain/pan/width, wavetable synth params, ADSR amp envelope, automated lowpass cutoff_start_hz -> cutoff_end_hz, optional filter.cutoff_points, optional layer gain_points, optional modulation.gate_points, LFO tremolo, chorus_mix, phaser_mix, delay_mix, saturation, EQ, reverb_mix.
- Wavetable synth fields: engine=wavetable, wavetable=saw_stack|square_saw|formant|digital|triangle|sine, wavetable_position, warp, fm_amount, fm_ratio, blend, voices, detune_cents, stereo_spread, sub_level.
- Each layer can send to session returns through effects.return_send. Returns currently support reverb with id/type/gain_db/decay/width.
- waveform may be sine, triangle, saw, square, noise, air, or transient for fallback/noise layers.
- Session has returns and master fields. Preserve them even if empty.
- Dense MIDI/pattern layers may use up to {MAX_NOTES_PER_LAYER} note events. If pattern_constraints.json shows source onsets, write note events near those onset times instead of summarizing them as a held layer.
- Timing must be aligned to beat_grid.json. Use its tempo, meter, downbeat_time_seconds, and 1/16 grid edges when placing arpeggio/pulse notes or automation changes.

Rules:
- Write a complete session JSON file, not prose and not a patch fragment.
- Preserve useful existing layers and stable layer ids.
- Add at most one new layer unless the residual clearly demands a paired support/noise layer.
- Use synth/noise-like approximation only; do not reference external samples.
- The previous Critic recommendation is binding: address its highest-priority missing items first and avoid its do_not guidance if present.
- If the Critic or pattern constraints say MIDI/pattern/onsets are weak, update the notes list concretely. Do not describe an arpeggio without writing the note events.
- Optimize actual reconstruction metrics: multi-resolution spectral, mel spectrogram, envelope, segment_envelope, late_energy_ratio, sustain_coverage, frontload_balance, band_envelope_by_time, pitch chroma, spectral motion, transient/onset, stereo width, embedding.

The session JSON must use this shape:
{session_shape()}
"""


def codex_residual_critic_prompt(analyzer_path: Path, session_path: Path, history_item_path: Path, audio_diff_path: Path) -> str:
    return f"""
You are the Critic agent in an audio reconstruction pipeline.

Read these files before answering:
- Analyzer layer plan: {analyzer_path}
- Current best reconstruction session JSON: {session_path}
- Latest builder history item JSON: {history_item_path}
- Latest audio similarity/diff JSON: {audio_diff_path}
- Source pattern constraints JSON: {analyzer_path.parent / "pattern_constraints.json"}
- Deterministic beat grid JSON: {analyzer_path.parent / "beat_grid.json"}

Task: Update the requested recommendation artifact for the next Producer run. Include perceptual language plus concrete file-level edits. Tie every critique to reconstruction metrics, source analysis, or session structure.

The recommendation JSON must use this shape:
{{
  "missing": ["specific measurable mismatches"],
  "recommendations": ["concrete next actions for the Producer or Mixer"],
  "must_fix": ["highest priority concrete measurable fixes for the next Producer prompt"],
  "do_not": ["specific changes the next Producer should avoid"],
  "success_criteria": ["numbers the next score should move toward"],
  "priority": "layer|automation|pitch|envelope|stereo|mix",
  "target_files": ["tracks/track_id.json or effects/effect_id.json that should change"],
  "producer_work_order": "one compact paragraph telling Producer exactly which track/effect file to update and what note/synth/effect changes to test",
  "stop_layer_building": true_or_false
}}
"""


def codex_mixer_prompt(analyzer_path: Path, session_path: Path, recommendation_path: Path, history_path: Path) -> str:
    return f"""
You are the Mixer agent in an audio reconstruction pipeline.

Read these files before answering:
- Analyzer layer plan: {analyzer_path}
- Current best reconstruction session JSON: {session_path}
- Latest Critic recommendation JSON: {recommendation_path}
- Score/history JSON: {history_path}

Task: Update the requested session artifact by balancing the accepted layers, stereo width, layer gains, pan, reverb/chorus mix, and master gain/width. Do not add new musical content unless the session is empty.

The session JSON must use this shape:
{session_shape()}
"""


def codex_step_mixer_prompt(analyzer_path: Path, candidate_session_path: Path, recommendation_path: Path, history_path: Path, step: int) -> str:
    return f"""
You are the Mixer agent inside the reconstruction loop.

Read these files before answering:
- Analyzer layer plan: {analyzer_path}
- Current Producer candidate session JSON: {candidate_session_path}
- Latest Critic recommendation JSON: {recommendation_path}
- Score/history JSON: {history_path}

Task: Update the requested session artifact by mixing this step's Producer candidate before simplification and scoring. Balance layer gains, pan, width, chorus/reverb, return_send, returns, and master gain/width so the reconstruction loss has a fair mixed candidate to evaluate. Do not remove musical layers and do not add new notes unless required to prevent an empty session.

Current loop step: {step + 1}

The session JSON must use this shape:
{session_shape()}
"""


def codex_simplifier_prompt(analyzer_path: Path, session_path: Path, history_path: Path) -> str:
    return f"""
You are the Simplifier agent in an audio reconstruction pipeline.

Read these files before answering:
- Analyzer layer plan: {analyzer_path}
- Mixed reconstruction session JSON: {session_path}
- Score/history JSON: {history_path}

Task: Update the requested session artifact by removing redundant layers, collapsing overlapping roles, keeping the reconstruction close, and making the session understandable. Also preserve enough structure to distill a playable patch.

Rules:
- Do not make the session empty.
- Prefer 1-4 meaningful layers.
- Preserve the most important layer ids when possible.

The session JSON must use this shape:
{session_shape()}
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


def write_chunked_session(root: Path, session: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tracks_dir = root / "tracks"
    effects_dir = root / "effects"
    tracks_dir.mkdir(exist_ok=True)
    effects_dir.mkdir(exist_ok=True)
    manifest = {
        "version": session.get("version", 1),
        "sample_rate": session.get("sample_rate", 44100),
        "duration": session.get("duration", 5.0),
        "tracks": [],
        "returns": [],
        "master": "master.json",
    }
    for index, layer in enumerate(session.get("layers", [])):
        track_id = str(layer.get("id") or f"track_{index + 1}")
        track_path = tracks_dir / f"{track_id}.json"
        write_json(track_path, layer)
        manifest["tracks"].append(str(track_path.relative_to(root)))
    for index, ret in enumerate(session.get("returns", [])):
        return_id = str(ret.get("id") or f"return_{index + 1}")
        return_path = effects_dir / f"{return_id}.json"
        write_json(return_path, ret)
        manifest["returns"].append(str(return_path.relative_to(root)))
    write_json(root / "master.json", session.get("master", {}))
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
        session["layers"].append(json.loads((root / track).read_text()))
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
            "engine": "wavetable",
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
    if score.beat_grid_mel < 0.58 or score.beat_grid_band < 0.58:
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
        out_path = output_dir / f"premix_reconstruction_step_{step:02d}_{label}.wav"
        diff_path = output_dir / f"premix_audio_diff_step_{step:02d}_{label}.json"
        score, diagnostics, residual = write_audio_diff(reference_audio, candidate_session, sample_rate, out_path, diff_path, beat_grid=beat_grid)
        write_json(output_dir / f"session_step_{step:02d}_{label}.json", candidate_session)
        print(f"trace_file agent=loss step={step} role=premix_audio_diff_{label} path={diff_path}", flush=True)
        print(f"trace_file agent=loss step={step} role=premix_render_{label} path={out_path}", flush=True)
        gate = structural_scores_ok(score, diagnostics)
        gated_final = score.final if gate else score.final * 0.82
        results.append((gated_final, label, candidate_session, score, diagnostics, residual, out_path, diff_path))
        print(
            f"step={step} candidate={label} phase=premix score={score.final:.4f} gated={gated_final:.4f} structural_gate={str(gate).lower()} mel={score.mel_spectrogram:.4f} beat_mel={score.beat_grid_mel:.4f} beat_band={score.beat_grid_band:.4f} beat_env={score.beat_grid_envelope:.4f} envelope={score.envelope:.4f} segment_envelope={score.segment_envelope:.4f} late={score.late_energy_ratio:.4f} sustain={score.sustain_coverage:.4f} frontload={score.frontload_balance:.4f} band_time={score.band_envelope_by_time:.4f} chroma={score.pitch_chroma:.4f} f0={score.f0_contour:.4f} motion={score.spectral_motion:.4f} timbre={score.spectral_features:.4f} onset_count={score.onset_count:.4f} onset_timing={score.onset_timing:.4f} stereo={score.stereo_width:.4f}",
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
        "source": "v1_simplifier",
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
    parser.add_argument("--local-trials", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    setup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    beat_grid = estimate_beat_grid(reference_audio, args.sample_rate, subdivision=4)
    trace_file("analyzer", "beat_grid", write_json(args.output_dir / "beat_grid.json", beat_grid))
    profile = source_profile(reference_audio, args.sample_rate, args.seconds, beat_grid)
    pattern_info = pattern_constraints(profile, args.sample_rate)
    profile["pattern_constraints"] = pattern_info
    profile["synth_engine"] = {"active": "internal_wavetable", "vital": vital_status()}
    source_profile_path = trace_file("analyzer", "source_profile", write_json(args.output_dir / "source_profile.json", profile))
    trace_file("analyzer", "pattern_constraints", write_json(args.output_dir / "pattern_constraints.json", pattern_info))
    print("analysis_start layer_plan", flush=True)
    analysis = run_codex_json(
        args.output_dir,
        "analyzer",
        codex_analyzer_prompt(reference_clip, source_profile_path, args.seconds),
        json_output_path=args.output_dir / "analyzer_answer.json",
    )
    global_payload = analysis.setdefault("global", {})
    global_payload.setdefault("meter", "unknown")
    global_payload.setdefault("duration_seconds", args.seconds)
    global_payload.setdefault("overall_mix", "unknown overall mix")
    if not isinstance(analysis.get("layers"), list):
        raise ValueError(f"Codex Analyzer did not return layers: {analysis}")
    analyzer_path = trace_file("analyzer", "parsed_answer", write_json(args.output_dir / "analyzer_answer.json", analysis))
    trace_file("analyzer", "layer_analysis", write_json(args.output_dir / "layer_analysis.json", analysis))
    print(f"analysis_done layers={len(analysis.get('layers', []))}", flush=True)

    session = sanitize_session(DEFAULT_SESSION, args.seconds, args.sample_rate)
    current_session_path = trace_file("session", "current", write_json(args.output_dir / "session_current.json", session))
    chunk_manifest_path = trace_file("session", "chunked_manifest", write_chunked_session(args.output_dir / "session_chunks" / "current", session))
    history: list[dict[str, Any]] = []
    history_path = write_json(args.output_dir / "history.json", history)
    best_session = session
    best_score = AudioScore(**{field: 0.0 for field in AudioScore.__dataclass_fields__})
    recommendation = {
        "missing": ["empty session"],
        "recommendations": ["add the dominant audible layer first"],
        "priority": "layer",
        "stop_layer_building": False,
    }
    recommendation_path = trace_file("residual_critic", "recommendation_initial", write_json(args.output_dir / "recommendation_initial.json", recommendation))
    rng = np.random.default_rng(20260506)
    min_builder_steps = min(args.steps, 5)

    for step in range(args.steps):
        print(f"agent_stage layer_builder step={step}", flush=True)
        proposed = run_codex_json(
            args.output_dir,
            f"layer_builder_step_{step:02d}",
            codex_layer_builder_prompt(analyzer_path, current_session_path, recommendation_path, history_path, step, args.max_layers),
            args.seconds,
            args.sample_rate,
            json_output_path=args.output_dir / f"session_step_{step:02d}_codex_proposal.json",
        )
        proposed_session_path = trace_file(f"layer_builder_step_{step:02d}", "session_proposal", write_json(args.output_dir / f"session_step_{step:02d}_codex_proposal.json", proposed))
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
        premix_score_value, premix_label, premix_session, premix_score, premix_diagnostics, premix_residual, premix_out_path, premix_diff_path = step_results[0]
        print(f"premix_winner step={step} winner={premix_label} score={premix_score_value:.4f}", flush=True)
        print(f"agent_stage mixer step={step}", flush=True)
        premix_session_path = trace_file(f"mixer_step_{step:02d}", "premix_session", write_json(args.output_dir / f"session_step_{step:02d}_premix_winner.json", premix_session))
        mixed_candidate = run_codex_json(
            args.output_dir,
            f"mixer_step_{step:02d}",
            codex_step_mixer_prompt(analyzer_path, premix_session_path, recommendation_path, history_path, step),
            args.seconds,
            args.sample_rate,
            json_output_path=args.output_dir / f"session_step_{step:02d}_mixed.json",
        )
        mixed_session_path = trace_file(f"mixer_step_{step:02d}", "mixed_session", write_json(args.output_dir / f"session_step_{step:02d}_mixed.json", mixed_candidate))
        print(f"agent_stage simplifier step={step}", flush=True)
        simplified_candidate = run_codex_json(
            args.output_dir,
            f"simplifier_step_{step:02d}",
            codex_simplifier_prompt(analyzer_path, mixed_session_path, history_path),
            args.seconds,
            args.sample_rate,
            json_output_path=args.output_dir / f"session_step_{step:02d}_simplified.json",
        )
        simplified_session_path = trace_file(f"simplifier_step_{step:02d}", "simplified_session", write_json(args.output_dir / f"session_step_{step:02d}_simplified.json", simplified_candidate))
        simplified_out_path = args.output_dir / f"reconstruction_step_{step:02d}_simplified.wav"
        simplified_diff_path = args.output_dir / f"audio_diff_step_{step:02d}_simplified.json"
        score, diagnostics, candidate_residual = write_audio_diff(reference_audio, simplified_candidate, args.sample_rate, simplified_out_path, simplified_diff_path, beat_grid=beat_grid)
        label = f"{premix_label}+mixer+simplifier"
        session = simplified_candidate
        out_path = simplified_out_path
        diff_path = simplified_diff_path
        print(f"trace_file agent=loss step={step} role=audio_diff_simplified path={simplified_diff_path}", flush=True)
        print(f"trace_file agent=loss step={step} role=render_simplified path={simplified_out_path}", flush=True)
        print(
            f"step={step} candidate=simplified_from_{premix_label} phase=simplified score={score.final:.4f} mel={score.mel_spectrogram:.4f} beat_mel={score.beat_grid_mel:.4f} beat_band={score.beat_grid_band:.4f} beat_env={score.beat_grid_envelope:.4f} envelope={score.envelope:.4f} segment_envelope={score.segment_envelope:.4f} late={score.late_energy_ratio:.4f} sustain={score.sustain_coverage:.4f} frontload={score.frontload_balance:.4f} band_time={score.band_envelope_by_time:.4f} chroma={score.pitch_chroma:.4f} motion={score.spectral_motion:.4f} onset_count={score.onset_count:.4f} onset_timing={score.onset_timing:.4f} stereo={score.stereo_width:.4f}",
            flush=True,
        )
        print(f"winner_summary step={step} codex_proposed=true winner={label} codex_won={str(premix_label == 'codex').lower()} score={score.final:.4f}", flush=True)
        trace_file(f"loss_step_{step:02d}", "winner_render", out_path)
        trace_file(f"loss_step_{step:02d}", "winner_audio_diff", diff_path)
        accepted = score.final >= best_score.final and structural_scores_ok(score, diagnostics)
        if accepted:
            best_session = session
            best_score = score
            current_session_path = trace_file("session", "current", write_json(args.output_dir / "session_current.json", best_session))
            chunk_manifest_path = trace_file("session", "chunked_manifest", write_chunked_session(args.output_dir / "session_chunks" / "current", best_session))
        deterministic_residual = dict(candidate_residual)
        if len(best_session.get("layers", [])) < 2 and best_score.final < 0.72:
            deterministic_residual.setdefault("recommendations", []).append("prefer adding one concrete layer over overfitting the first layer")
        history_item = {
            "stage": "layer_builder",
            "step": step,
            "accepted": accepted,
            "winner": label,
            "audio_path": str(out_path),
            "audio_diff_path": str(diff_path),
            "proposal_session_path": str(proposed_session_path),
            "premix_winner": premix_label,
            "premix_audio_path": str(premix_out_path),
            "premix_audio_diff_path": str(premix_diff_path),
            "mixer_session_path": str(mixed_session_path),
            "simplifier_session_path": str(simplified_session_path),
            "chunk_manifest_path": str(chunk_manifest_path),
            "scores": score_to_json(score),
            "best_scores": score_to_json(best_score),
            "layers": [{"id": layer["id"], "role": layer["role"]} for layer in best_session.get("layers", [])],
            "deterministic_residual": deterministic_residual,
        }
        history_item_path = trace_file(f"loss_step_{step:02d}", "history_item", write_json(args.output_dir / f"history_item_step_{step:02d}.json", history_item))
        print(f"agent_stage residual_critic step={step}", flush=True)
        critic = run_codex_json(
            args.output_dir,
            f"residual_critic_step_{step:02d}",
            codex_residual_critic_prompt(analyzer_path, current_session_path, history_item_path, diff_path),
            json_output_path=args.output_dir / f"recommendation_step_{step:02d}.json",
        )
        recommendation = {
            "missing": critic.get("missing", deterministic_residual["missing"]),
            "recommendations": critic.get("recommendations", deterministic_residual["recommendations"]),
            "must_fix": critic.get("must_fix", deterministic_residual["missing"][:4]),
            "do_not": critic.get("do_not", []),
            "success_criteria": critic.get("success_criteria", []),
            "priority": critic.get("priority", "layer"),
            "target_files": critic.get("target_files", []),
            "producer_work_order": critic.get("producer_work_order", ""),
            "stop_layer_building": bool(critic.get("stop_layer_building", False)),
            "diagnostics": deterministic_residual.get("diagnostics", {}),
        }
        temporal_ok = temporal_scores_ok(best_score)
        structural_ok = structural_scores_ok(best_score, diagnostics)
        forced_continue = step + 1 < min_builder_steps or best_score.final < 0.78 or not temporal_ok or not structural_ok
        if forced_continue and recommendation["stop_layer_building"]:
            recommendation["stop_layer_building"] = False
            recommendation.setdefault("do_not", []).append("do not stop yet; the orchestrator requires more builder passes, a higher final score, and passing temporal scores before mixer")
            recommendation.setdefault("recommendations", []).append("continue the Builder/Critic loop and address the weakest temporal and spectral metrics")
            recommendation.setdefault("success_criteria", []).append("before stopping, final >= 0.78, temporal scores pass, and structural gate passes: enough onsets, onset timing, f0 contour, timbre features")
        recommendation_path = trace_file(f"residual_critic_step_{step:02d}", "recommendation", write_json(args.output_dir / f"recommendation_step_{step:02d}.json", recommendation))
        history_item["recommendation_path"] = str(recommendation_path)
        history_item["residual_critic"] = recommendation
        history.append(history_item)
        history_path = write_json(args.output_dir / "history.json", history)
        trace_file(f"layer_builder_step_{step:02d}", "accepted_session", write_json(args.output_dir / f"session_step_{step:02d}_accepted.json", best_session))
        print(f"step_complete index={step} winner={label} accepted={str(accepted).lower()} best_score={best_score.final:.4f}", flush=True)
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
        "analysis": analysis,
        "empty_session_schema": sanitize_session(DEFAULT_SESSION, args.seconds, args.sample_rate),
        "history": history,
        "best_scores": score_to_json(best_score),
        "final_path": str(final_path),
        "session_path": str(session_path),
        "session_alias_path": str(session_alias_path),
        "distilled_playable_patch_path": str(playable_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {final_path}", flush=True)
    print(f"wrote {session_path}", flush=True)
    print(f"wrote {session_alias_path}", flush=True)
    print(f"wrote {playable_path}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
