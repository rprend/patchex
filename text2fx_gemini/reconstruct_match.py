#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from text2fx import LLM_MODEL, extract_json_object, load_runtime_dependencies
except ModuleNotFoundError:
    from .text2fx import LLM_MODEL, extract_json_object, load_runtime_dependencies

sf: Any = None
genai: Any = None


@dataclass
class Score:
    final: float
    multi_resolution_spectral: float
    mel_spectrogram: float
    envelope: float
    pitch_chroma: float
    spectral_motion: float
    transient_onset: float
    stereo_width: float
    embedding: float


DEFAULT_SESSION = {
    "version": 1,
    "sample_rate": 44100,
    "duration": 5.0,
    "layers": [],
    "returns": [],
    "master": {"gain_db": -1.0, "width": 1.0},
}

NOTE_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
CODEX_PATH = "/Applications/Codex.app/Contents/Resources/codex"


def setup() -> None:
    global sf, genai
    load_runtime_dependencies()
    import soundfile as soundfile_module
    from google import genai as genai_module

    sf = soundfile_module
    genai = genai_module


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


def sanitize_session(payload: dict[str, Any], duration: float, sample_rate: int) -> dict[str, Any]:
    session = dict(DEFAULT_SESSION)
    session["duration"] = float(duration)
    session["sample_rate"] = int(sample_rate)
    session["layers"] = []
    returns = payload.get("returns", [])
    session["returns"] = returns if isinstance(returns, list) else []
    for index, layer in enumerate(payload.get("layers", [])[:6]):
        synth = layer.get("synth", {})
        amp = layer.get("amp_envelope", {})
        filt = layer.get("filter", {})
        effects = layer.get("effects", {})
        notes = layer.get("notes", [])
        if not isinstance(notes, list) or not notes:
            notes = [{"note": synth.get("note", 48), "start": 0.0, "duration": duration, "velocity": 0.7}]
        sanitized_notes = []
        for note in notes[:16]:
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
                    "blend": float(np.clip(float(synth.get("blend", 0.35)), 0.0, 1.0)),
                    "voices": int(np.clip(int(synth.get("voices", 3)), 1, 8)),
                    "detune_cents": float(np.clip(float(synth.get("detune_cents", 8.0)), 0.0, 35.0)),
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
                },
                "modulation": {
                    "lfo_rate_hz": float(np.clip(float(layer.get("modulation", {}).get("lfo_rate_hz", 0.15)), 0.0, 8.0)),
                    "lfo_depth": float(np.clip(float(layer.get("modulation", {}).get("lfo_depth", 0.05)), 0.0, 1.0)),
                },
                "effects": {
                    "chorus_mix": float(np.clip(float(effects.get("chorus_mix", 0.18)), 0.0, 1.0)),
                    "reverb_mix": float(np.clip(float(effects.get("reverb_mix", 0.18)), 0.0, 0.7)),
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


def moving_lowpass(audio: np.ndarray, sr: int, start_hz: float, end_hz: float, resonance: float) -> np.ndarray:
    out = np.zeros_like(audio)
    y = 0.0
    for index, sample in enumerate(audio):
        frac = index / max(1, audio.size - 1)
        cutoff = start_hz + (end_hz - start_hz) * frac
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sr)
        y += alpha * (sample - y)
        out[index] = y
    return out * (1.0 + 0.2 * resonance)


def render_layer(layer: dict[str, Any], duration: float, sr: int) -> np.ndarray:
    samples = int(duration * sr)
    mono = np.zeros(samples, dtype=np.float32)
    synth = layer["synth"]
    amp = layer["amp_envelope"]
    filt = layer["filter"]
    mod = layer["modulation"]
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
            note_audio += oscillator(phase, synth["waveform"], synth["blend"]) / voices
        if synth["sub_level"] > 0:
            note_audio += synth["sub_level"] * np.sin(2.0 * np.pi * midi_to_hz(note["note"] - 12) * t)
        if mod["lfo_depth"] > 0 and mod["lfo_rate_hz"] > 0:
            note_audio *= 1.0 + mod["lfo_depth"] * 0.15 * np.sin(2.0 * np.pi * mod["lfo_rate_hz"] * t)
        note_audio *= envelope(end - start, sr, amp["attack"], amp["decay"], amp["sustain"], amp["release"])
        note_audio *= note["velocity"]
        note_audio = moving_lowpass(note_audio.astype(np.float32), sr, filt["cutoff_start_hz"], filt["cutoff_end_hz"], filt["resonance"])
        mono[start:end] += note_audio.astype(np.float32)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono *= 1.0 / peak
    chorus_mix = layer["effects"]["chorus_mix"]
    delay = max(1, int(0.012 * sr))
    delayed = np.pad(mono[:-delay], (delay, 0)) if mono.size > delay else mono
    left = mono * (1.0 - 0.35 * layer["pan"]) + delayed * chorus_mix * layer["width"]
    right = mono * (1.0 + 0.35 * layer["pan"]) - delayed * chorus_mix * layer["width"]
    stereo = np.vstack([left, right]) * db_to_amp(layer["gain_db"])
    reverb_mix = layer["effects"]["reverb_mix"]
    if reverb_mix > 0:
        tail = np.zeros_like(stereo)
        delays = [int(sr * 0.083), int(sr * 0.137), int(sr * 0.211)]
        for delay_samples in delays:
            if delay_samples < samples:
                tail[:, delay_samples:] += stereo[:, :-delay_samples] * (0.35 / len(delays))
        stereo = stereo * (1.0 - reverb_mix) + tail * reverb_mix
    return stereo.astype(np.float32)


def render_session(session: dict[str, Any]) -> np.ndarray:
    sr = int(session["sample_rate"])
    duration = float(session["duration"])
    samples = int(duration * sr)
    mix = np.zeros((2, samples), dtype=np.float32)
    for layer in session.get("layers", []):
        mix += render_layer(layer, duration, sr)
    width = session.get("master", {}).get("width", 1.0)
    mid = mix.mean(axis=0)
    side = (mix[0] - mix[1]) * 0.5 * width
    mix = np.vstack([mid + side, mid - side])
    mix *= db_to_amp(session.get("master", {}).get("gain_db", -1.0))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    return mix.astype(np.float32)


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def stft_mag(y: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    frames = []
    window = np.hanning(frame)
    for start in range(0, y.size - frame + 1, hop):
        frames.append(np.abs(np.fft.rfft(y[start : start + frame] * window)) + 1e-8)
    return np.asarray(frames)


def frame_rms(y: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    values = []
    for start in range(0, y.size - frame + 1, hop):
        chunk = y[start : start + frame]
        values.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))
    return np.asarray(values)


def mel_like_bands(mag: np.ndarray, bands: int = 32) -> np.ndarray:
    if mag.size == 0:
        return mag
    edges = np.geomspace(1, mag.shape[1], bands + 1).astype(int) - 1
    edges[0] = 0
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        end = max(start + 1, end)
        out.append(np.mean(mag[:, start:end], axis=1))
    return np.asarray(out).T


def onset_curve(env: np.ndarray) -> np.ndarray:
    if env.size < 2:
        return np.zeros_like(env)
    diff = np.maximum(0, np.diff(env, prepend=env[0]))
    peak = float(np.max(diff))
    return diff / peak if peak > 0 else diff


def band_chroma(mag: np.ndarray, sr: int) -> np.ndarray:
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    chroma = np.zeros(12)
    for idx, freq in enumerate(freqs):
        if 40 <= freq <= 5000:
            midi = int(round(69 + 12 * math.log2(freq / 440.0)))
            chroma[midi % 12] += float(np.mean(mag[:, idx]))
    norm = np.linalg.norm(chroma)
    return chroma / norm if norm else chroma


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def score_audio(reference: np.ndarray, candidate: np.ndarray, sr: int) -> tuple[Score, dict[str, Any]]:
    ref = mono(reference)
    cand = mono(candidate)
    size = min(ref.size, cand.size)
    ref = ref[:size]
    cand = cand[:size]
    ref_mag = stft_mag(ref)
    cand_mag = stft_mag(cand)
    frames = min(ref_mag.shape[0], cand_mag.shape[0])
    bins = min(ref_mag.shape[1], cand_mag.shape[1])
    ref_mag = ref_mag[:frames, :bins]
    cand_mag = cand_mag[:frames, :bins]
    spectral_distances = []
    for frame, hop in [(1024, 256), (2048, 512), (4096, 1024)]:
        r = stft_mag(ref, frame, hop)
        c = stft_mag(cand, frame, hop)
        f = min(r.shape[0], c.shape[0])
        b = min(r.shape[1], c.shape[1])
        r = r[:f, :b]
        c = c[:f, :b]
        spectral_distances.append(float(np.mean(np.abs(np.log1p(r) - np.log1p(c))) / (np.mean(np.log1p(r)) + 1e-8)))
    multi_resolution_spectral = math.exp(-float(np.mean(spectral_distances)))
    ref_mel = mel_like_bands(ref_mag)
    cand_mel = mel_like_bands(cand_mag)
    mel_spectrogram = math.exp(-float(np.mean(np.abs(np.log1p(ref_mel) - np.log1p(cand_mel))) / (np.mean(np.log1p(ref_mel)) + 1e-8)))
    ref_env = frame_rms(ref)
    cand_env = frame_rms(cand)
    env_frames = min(ref_env.size, cand_env.size)
    ref_env = ref_env[:env_frames]
    cand_env = cand_env[:env_frames]
    envelope_score = math.exp(-float(np.mean(np.abs(ref_env - cand_env)) / (np.mean(ref_env) + 1e-8)))
    pitch_chroma = max(0.0, cosine(band_chroma(ref_mag, sr), band_chroma(cand_mag, sr)))
    freqs = np.fft.rfftfreq((bins - 1) * 2, 1.0 / sr)[:bins]
    ref_centroid = np.sum(ref_mag * freqs[None, :], axis=1) / (np.sum(ref_mag, axis=1) + 1e-8)
    cand_centroid = np.sum(cand_mag * freqs[None, :], axis=1) / (np.sum(cand_mag, axis=1) + 1e-8)
    spectral_motion = math.exp(-float(np.mean(np.abs(ref_centroid - cand_centroid)) / (np.mean(ref_centroid) + 1e-8)))
    transient_onset = math.exp(-float(np.mean(np.abs(onset_curve(ref_env) - onset_curve(cand_env)))))
    if reference.ndim == 2 and candidate.ndim == 2:
        ref_side = float(np.std(reference[0, :size] - reference[1, :size]) / (np.std(reference[:, :size]) + 1e-8))
        cand_side = float(np.std(candidate[0, :size] - candidate[1, :size]) / (np.std(candidate[:, :size]) + 1e-8))
        stereo_width = math.exp(-abs(ref_side - cand_side) / max(ref_side, cand_side, 1e-6))
    else:
        stereo_width = 1.0
    embedding = float(np.clip(0.50 * multi_resolution_spectral + 0.25 * pitch_chroma + 0.25 * envelope_score, 0.0, 1.0))
    final = (
        0.24 * multi_resolution_spectral
        + 0.20 * mel_spectrogram
        + 0.16 * envelope_score
        + 0.13 * pitch_chroma
        + 0.12 * spectral_motion
        + 0.07 * transient_onset
        + 0.06 * stereo_width
        + 0.02 * embedding
    )
    diagnostics = {
        "reference_centroid_start": float(ref_centroid[0]) if ref_centroid.size else 0.0,
        "reference_centroid_end": float(ref_centroid[-1]) if ref_centroid.size else 0.0,
        "candidate_centroid_start": float(cand_centroid[0]) if cand_centroid.size else 0.0,
        "candidate_centroid_end": float(cand_centroid[-1]) if cand_centroid.size else 0.0,
        "reference_rms": float(np.sqrt(np.mean(ref**2) + 1e-12)),
        "candidate_rms": float(np.sqrt(np.mean(cand**2) + 1e-12)),
    }
    return Score(final, multi_resolution_spectral, mel_spectrogram, envelope_score, pitch_chroma, spectral_motion, transient_onset, stereo_width, embedding), diagnostics


def score_to_json(score: Score) -> dict[str, float]:
    return {
        "final": float(score.final),
        "multi_resolution_spectral": float(score.multi_resolution_spectral),
        "mel_spectrogram": float(score.mel_spectrogram),
        "envelope": float(score.envelope),
        "pitch_chroma": float(score.pitch_chroma),
        "spectral_motion": float(score.spectral_motion),
        "transient_onset": float(score.transient_onset),
        "stereo_width": float(score.stereo_width),
        "embedding": float(score.embedding),
    }


def residual_report(score: Score, diagnostics: dict[str, Any], current_layers: int) -> dict[str, Any]:
    missing = []
    recs = []
    if score.multi_resolution_spectral < 0.62 or score.mel_spectrogram < 0.62:
        missing.append("spectral balance still differs from the source")
        recs.append("adjust oscillator blend, filter cutoff range, or add a support layer")
    if score.envelope < 0.65:
        missing.append("amplitude contour does not follow the source")
        recs.append("change note durations, gain, attack, release, or layer timing")
    if score.spectral_motion < 0.65:
        missing.append("brightness motion does not track the source")
        recs.append("add or adjust lowpass cutoff automation")
    if score.pitch_chroma < 0.55:
        missing.append("pitch/chord content appears mismatched")
        recs.append("change layer notes or add a lower harmonic layer")
    if score.transient_onset < 0.65:
        missing.append("transient/onset contour does not match the source")
        recs.append("reduce unwanted retriggers or adjust note starts")
    if score.stereo_width < 0.6:
        missing.append("stereo width differs from the source")
        recs.append("adjust width, pan, chorus, or reverb mix")
    if current_layers < 2 and score.final < 0.72:
        recs.append("prefer adding one concrete layer over overfitting the first layer")
    return {"missing": missing or ["largest remaining error is subtle parameter mismatch"], "recommendations": recs or ["make a conservative local adjustment"], "diagnostics": diagnostics}


def gemini_client() -> Any:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running reconstruction.")
    return genai.Client(api_key=key)


def analyze_layers(client: Any, reference_path: Path, local_features: dict[str, Any], duration: float) -> dict[str, Any]:
    from google.genai import types

    prompt = f"""
Analyze this exact five-second audio clip as a reconstruction target, not as a vibe prompt.

Reconstruct the whole selected clip autonomously. If drums, vocals, or non-synth material are present, represent what can be approximated with synth/noise/space layers and mark confidence honestly.

Return ONLY JSON:
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

Prefer 2-5 layers for dense clips and 1-3 layers for sparse clips. Use the source audio as authority. Local DSP features:
{json.dumps(local_features, indent=2)}
"""
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="audio/wav", data=reference_path.read_bytes())),
                ]
            )
        ],
    )
    payload = extract_json_object(response.text or "")
    if not isinstance(payload.get("layers"), list):
        raise ValueError(f"Layer analysis did not return layers: {payload}")
    global_payload = payload.setdefault("global", {})
    global_payload.setdefault("meter", "unknown")
    global_payload.setdefault("duration_seconds", duration)
    if "summary" in global_payload and "overall_mix" not in global_payload:
        global_payload["overall_mix"] = global_payload.pop("summary")
    global_payload.setdefault("overall_mix", "unknown overall mix")
    return payload


def session_shape() -> str:
    return json.dumps(DEFAULT_SESSION, indent=2)


def codex_layer_builder_prompt(analysis: dict[str, Any], session: dict[str, Any], history: list[dict[str, Any]], residual: dict[str, Any], step: int, max_layers: int) -> str:
    phase = [
        "create the dominant audible layer",
        "inspect residual and add or modify the second most important layer",
        "add movement/filter automation or correct pitch/envelope mismatch",
        "add stereo width, reverb/space, chorus/phaser-like motion, or ambience layer",
        "make a conservative reconstruction pass before mixing",
    ][min(step, 4)]
    return f"""
You are the Layer Builder agent in an autonomous audio reconstruction pipeline.

Task: Given the source layer plan, current session, score history, and residual report, make the smallest concrete full-session JSON change that should reduce reconstruction error.

Current builder run: {step + 1}
Allowed action for this run: {phase}

Renderer capabilities:
- Up to {max_layers} synth layers.
- Each layer has note events, gain/pan/width, oscillator waveform/blend/voices/detune/sub, ADSR amp envelope, automated lowpass cutoff_start_hz -> cutoff_end_hz, LFO tremolo, chorus_mix, reverb_mix.
- Session has returns and master fields. Preserve them even if empty.

Rules:
- Return ONLY full JSON session, not prose and not a patch fragment.
- Preserve useful existing layers and stable layer ids.
- Add at most one new layer unless the residual clearly demands a paired support/noise layer.
- Use synth/noise-like approximation only; do not reference external samples.
- Optimize actual reconstruction metrics: multi-resolution spectral, mel spectrogram, envelope, pitch chroma, spectral motion, transient/onset, stereo width, embedding.

Source analysis:
{json.dumps(analysis, indent=2)}

Current session:
{json.dumps(session, indent=2)}

Score history:
{json.dumps(history, indent=2)}

Residual report:
{json.dumps(residual, indent=2)}

Return full JSON session with this shape:
{session_shape()}
"""


def codex_residual_critic_prompt(analysis: dict[str, Any], session: dict[str, Any], history_item: dict[str, Any], deterministic_residual: dict[str, Any]) -> str:
    return f"""
You are the Residual Critic agent in an audio reconstruction pipeline.

Read the latest accepted/rejected render scores and produce measurable residual guidance for the next builder. Do not suggest vibes. Tie every critique to reconstruction metrics, source analysis, or session structure.

Return ONLY JSON:
{{
  "missing": ["specific measurable mismatches"],
  "recommendations": ["concrete next actions for the Layer Builder or Mixer"],
  "priority": "layer|automation|pitch|envelope|stereo|mix",
  "stop_layer_building": true_or_false
}}

Source analysis:
{json.dumps(analysis, indent=2)}

Current session:
{json.dumps(session, indent=2)}

Latest history item:
{json.dumps(history_item, indent=2)}

Deterministic residual:
{json.dumps(deterministic_residual, indent=2)}
"""


def codex_mixer_prompt(analysis: dict[str, Any], session: dict[str, Any], history: list[dict[str, Any]], residual: dict[str, Any]) -> str:
    return f"""
You are the Mixer agent in an audio reconstruction pipeline.

Task: Balance the accepted layers, stereo width, layer gains, pan, reverb/chorus mix, and master gain/width. Do not add new musical content unless the session is empty. Return a full JSON session.

Source analysis:
{json.dumps(analysis, indent=2)}

Current session:
{json.dumps(session, indent=2)}

Score history:
{json.dumps(history, indent=2)}

Residual report:
{json.dumps(residual, indent=2)}

Return ONLY full JSON session with this shape:
{session_shape()}
"""


def codex_simplifier_prompt(analysis: dict[str, Any], session: dict[str, Any], history: list[dict[str, Any]]) -> str:
    return f"""
You are the Simplifier agent in an audio reconstruction pipeline.

Task: Remove redundant layers, collapse overlapping roles, keep the reconstruction close, and make the final session understandable. Return full JSON session. Also preserve enough structure to distill a playable patch.

Rules:
- Do not make the session empty.
- Prefer 1-4 meaningful layers.
- Preserve the most important layer ids when possible.
- Return ONLY full JSON session.

Source analysis:
{json.dumps(analysis, indent=2)}

Current mixed session:
{json.dumps(session, indent=2)}

History:
{json.dumps(history, indent=2)}

Return full JSON session with this shape:
{session_shape()}
"""


def run_codex_json(output_dir: Path, agent: str, prompt: str, duration: float | None = None, sample_rate: int | None = None) -> dict[str, Any]:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    answer_path = output_dir / f"codex_{agent}_answer.txt"
    prompt_path.write_text(prompt)
    print(f"codex_start agent={agent}", flush=True)
    print(f"codex_prompt_path {prompt_path}", flush=True)
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
    payload = extract_json_object(answer_path.read_text())
    if duration is not None and sample_rate is not None:
        return sanitize_session(payload, duration, sample_rate)
    return payload


def local_mutation(session: dict[str, Any], rng: np.random.Generator, amount: float, duration: float, sample_rate: int) -> dict[str, Any]:
    mutated = json.loads(json.dumps(session))
    if not mutated.get("layers"):
        return mutated
    layer = mutated["layers"][int(rng.integers(0, len(mutated["layers"])))]
    choice = rng.choice(["gain", "filter", "width", "detune", "envelope", "blend"])
    if choice == "gain":
        layer["gain_db"] += float(rng.normal(0, 3.0 * amount))
    elif choice == "filter":
        layer["filter"]["cutoff_start_hz"] *= float(np.exp(rng.normal(0, amount)))
        layer["filter"]["cutoff_end_hz"] *= float(np.exp(rng.normal(0, amount)))
    elif choice == "width":
        layer["width"] += float(rng.normal(0, amount))
        layer["effects"]["chorus_mix"] += float(rng.normal(0, 0.2 * amount))
    elif choice == "detune":
        layer["synth"]["detune_cents"] += float(rng.normal(0, 10.0 * amount))
        layer["synth"]["voices"] = int(np.clip(layer["synth"]["voices"] + rng.choice([-1, 0, 1]), 1, 8))
    elif choice == "envelope":
        layer["amp_envelope"]["attack"] *= float(np.exp(rng.normal(0, amount)))
        layer["amp_envelope"]["release"] *= float(np.exp(rng.normal(0, amount)))
    else:
        layer["synth"]["blend"] += float(rng.normal(0, amount))
    return sanitize_session(mutated, duration, sample_rate)


def score_session(reference_audio: np.ndarray, session: dict[str, Any], sample_rate: int, output_path: Path) -> tuple[Score, dict[str, Any]]:
    rendered = render_session(session)
    sf.write(output_path, rendered.T, sample_rate)
    return score_audio(reference_audio, rendered, sample_rate)


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
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--local-trials", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    setup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_audio, reference_sr = load_audio(args.reference)
    if reference_sr != args.sample_rate:
        raise RuntimeError(f"Expected {args.sample_rate} Hz reference, got {reference_sr}. Extract the clip through the UI first.")
    reference_audio = reference_audio[:, : int(args.seconds * args.sample_rate)]
    reference_clip = args.output_dir / "source_clip.wav"
    sf.write(reference_clip, reference_audio.T, args.sample_rate)
    client = gemini_client()

    local_features = {
        "rms": float(np.sqrt(np.mean(mono(reference_audio) ** 2) + 1e-12)),
        "duration": args.seconds,
    }
    print("analysis_start layer_plan", flush=True)
    analysis = analyze_layers(client, reference_clip, local_features, args.seconds)
    (args.output_dir / "layer_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (args.output_dir / "analyzer_answer.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"analysis_done layers={len(analysis.get('layers', []))}", flush=True)

    session = sanitize_session(DEFAULT_SESSION, args.seconds, args.sample_rate)
    history: list[dict[str, Any]] = []
    best_session = session
    best_score = Score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    residual = {"missing": ["empty session"], "recommendations": ["add the dominant audible layer first"], "diagnostics": local_features}
    rng = np.random.default_rng(20260506)

    for step in range(args.steps):
        print(f"agent_stage layer_builder step={step}", flush=True)
        proposed = run_codex_json(
            args.output_dir,
            f"layer_builder_step_{step:02d}",
            codex_layer_builder_prompt(analysis, best_session, history, residual, step, args.max_layers),
            args.seconds,
            args.sample_rate,
        )
        candidates = [("codex", proposed)]
        for trial in range(args.local_trials):
            candidates.append((f"local_{trial}", local_mutation(proposed, rng, 0.22, args.seconds, args.sample_rate)))
        step_results = []
        for label, candidate_session in candidates:
            out_path = args.output_dir / f"reconstruction_step_{step:02d}_{label}.wav"
            score, diagnostics = score_session(reference_audio, candidate_session, args.sample_rate, out_path)
            step_results.append((score.final, label, candidate_session, score, diagnostics, out_path))
            print(
                f"step={step} candidate={label} score={score.final:.4f} mel={score.mel_spectrogram:.4f} envelope={score.envelope:.4f} chroma={score.pitch_chroma:.4f} motion={score.spectral_motion:.4f} onset={score.transient_onset:.4f} stereo={score.stereo_width:.4f}",
                flush=True,
            )
        step_results.sort(key=lambda item: item[0], reverse=True)
        _, label, session, score, diagnostics, out_path = step_results[0]
        accepted = score.final >= best_score.final
        if accepted:
            best_session = session
            best_score = score
        deterministic_residual = residual_report(best_score, diagnostics, len(best_session.get("layers", [])))
        history_item = {
            "stage": "layer_builder",
            "step": step,
            "accepted": accepted,
            "winner": label,
            "audio_path": str(out_path),
            "scores": score_to_json(score),
            "best_scores": score_to_json(best_score),
            "layers": [{"id": layer["id"], "role": layer["role"]} for layer in best_session.get("layers", [])],
            "deterministic_residual": deterministic_residual,
        }
        print(f"agent_stage residual_critic step={step}", flush=True)
        critic = run_codex_json(
            args.output_dir,
            f"residual_critic_step_{step:02d}",
            codex_residual_critic_prompt(analysis, best_session, history_item, deterministic_residual),
        )
        residual = {
            "missing": critic.get("missing", deterministic_residual["missing"]),
            "recommendations": critic.get("recommendations", deterministic_residual["recommendations"]),
            "priority": critic.get("priority", "layer"),
            "stop_layer_building": bool(critic.get("stop_layer_building", False)),
            "diagnostics": deterministic_residual.get("diagnostics", {}),
        }
        history_item["residual_critic"] = residual
        history.append(history_item)
        (args.output_dir / f"session_step_{step:02d}.json").write_text(json.dumps(best_session, indent=2) + "\n")
        print(f"step_complete index={step} winner={label} accepted={str(accepted).lower()} best_score={best_score.final:.4f}", flush=True)
        if residual.get("stop_layer_building") and len(best_session.get("layers", [])) >= 1:
            print(f"layer_building_stopped step={step} reason=residual_critic", flush=True)
            break

    print("agent_stage mixer", flush=True)
    mixed_session = run_codex_json(
        args.output_dir,
        "mixer",
        codex_mixer_prompt(analysis, best_session, history, residual),
        args.seconds,
        args.sample_rate,
    )
    mixed_score, mixed_diagnostics = score_session(reference_audio, mixed_session, args.sample_rate, args.output_dir / "mixer_reconstruction.wav")
    mixed_accepted = mixed_score.final >= best_score.final
    if mixed_accepted:
        best_session = mixed_session
        best_score = mixed_score
    history.append({"stage": "mixer", "scores": score_to_json(mixed_score), "accepted": mixed_accepted, "diagnostics": mixed_diagnostics})

    print("agent_stage simplifier", flush=True)
    simplified_session = run_codex_json(
        args.output_dir,
        "simplifier",
        codex_simplifier_prompt(analysis, best_session, history),
        args.seconds,
        args.sample_rate,
    )
    simplified_score, simplified_diagnostics = score_session(reference_audio, simplified_session, args.sample_rate, args.output_dir / "simplifier_reconstruction.wav")
    simplified_accepted = simplified_score.final >= best_score.final * 0.97
    if simplified_accepted:
        best_session = simplified_session
        best_score = simplified_score
    history.append({"stage": "simplifier", "scores": score_to_json(simplified_score), "accepted": simplified_accepted, "diagnostics": simplified_diagnostics})

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
