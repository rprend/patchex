#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from audio_diff import compare_audio, estimate_beat_grid, frame_rms, mono, stft_mag, stereo_stats
    from reconstruct_match import CODEX_PATH, DEFAULT_SESSION, render_layer, render_session
    from text2fx import extract_json_object
except ModuleNotFoundError:
    from .audio_diff import compare_audio, estimate_beat_grid, frame_rms, mono, stft_mag, stereo_stats
    from .reconstruct_match import CODEX_PATH, DEFAULT_SESSION, render_layer, render_session
    from .text2fx import extract_json_object


sf: Any = None

GM_PROGRAMS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavi", "Celesta", "Glockenspiel",
    "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion",
    "Harmonica", "Tango Accordion", "Acoustic Guitar nylon", "Acoustic Guitar steel",
    "Electric Guitar jazz", "Electric Guitar clean", "Electric Guitar muted", "Overdriven Guitar",
    "Distortion Guitar", "Guitar harmonics", "Acoustic Bass", "Electric Bass finger",
    "Electric Bass pick", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1",
    "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2",
    "SynthStrings 1", "SynthStrings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
    "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn",
    "Brass Section", "SynthBrass 1", "SynthBrass 2", "Soprano Sax", "Alto Sax", "Tenor Sax",
    "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute",
    "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 square", "Lead 2 sawtooth", "Lead 3 calliope", "Lead 4 chiff", "Lead 5 charang",
    "Lead 6 voice", "Lead 7 fifths", "Lead 8 bass+lead", "Pad 1 new age", "Pad 2 warm",
    "Pad 3 polysynth", "Pad 4 choir", "Pad 5 bowed", "Pad 6 metallic", "Pad 7 halo",
    "Pad 8 sweep", "FX 1 rain", "FX 2 soundtrack", "FX 3 crystal", "FX 4 atmosphere",
    "FX 5 brightness", "FX 6 goblins", "FX 7 echoes", "FX 8 sci-fi",
]

ROLE_BY_PROGRAM = {
    33: "fingered_bass",
    38: "synth_bass_1",
    4: "electric_piano_1",
    80: "square_wave",
    81: "saw_wave",
    86: "fifth_saw_wave",
    49: "strings",
}

ROLE_BY_NAME = {
    "fingered": "fingered_bass",
    "synth bass": "synth_bass_1",
    "e.piano": "electric_piano_1",
    "square": "square_wave",
    "saw wave": "saw_wave",
    "5th": "fifth_saw_wave",
    "track 7": "strings",
    "standard": "drums",
}


def load_role_map(path: Path | None) -> dict[str, dict[Any, str]]:
    if path is None:
        return {"programs": ROLE_BY_PROGRAM, "track_names": ROLE_BY_NAME}
    payload = json.loads(path.read_text())
    role_map = payload.get("role_map", payload)
    programs = {int(key): str(value) for key, value in role_map.get("programs", {}).items()}
    names = {str(key).lower(): str(value) for key, value in role_map.get("track_names", {}).items()}
    return {"programs": programs or ROLE_BY_PROGRAM, "track_names": names or ROLE_BY_NAME}


@dataclass
class MidiNote:
    channel: int
    note: int
    start_tick: int
    duration_ticks: int
    velocity: int


def runtime() -> None:
    global sf
    if sf is None:
        import soundfile as soundfile_module

        sf = soundfile_module


def read_vlq(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def ticks_to_seconds(tick: int, tempo_map: list[tuple[int, int]], ticks_per_beat: int) -> float:
    tempo_map = sorted(tempo_map) or [(0, 500000)]
    seconds = 0.0
    previous_tick = tempo_map[0][0]
    current_mpq = tempo_map[0][1]
    if tick < previous_tick:
        return tick * current_mpq / ticks_per_beat / 1_000_000.0
    for next_tick, next_mpq in tempo_map[1:]:
        if tick <= next_tick:
            break
        seconds += (next_tick - previous_tick) * current_mpq / ticks_per_beat / 1_000_000.0
        previous_tick = next_tick
        current_mpq = next_mpq
    seconds += max(0, tick - previous_tick) * current_mpq / ticks_per_beat / 1_000_000.0
    return float(seconds)


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "track"


def role_for_track(name: str, program: int | None, channel: int | None, role_map: dict[str, dict[Any, str]] | None = None) -> str:
    role_map = role_map or {"programs": ROLE_BY_PROGRAM, "track_names": ROLE_BY_NAME}
    if channel == 9:
        return "drums"
    program_roles = role_map.get("programs", {})
    if program in program_roles:
        return program_roles[int(program)]
    lowered = name.lower()
    for token, role in role_map.get("track_names", {}).items():
        if token in lowered:
            return role
    return slug(name) if name else f"channel_{channel if channel is not None else 'unknown'}"


def parse_midi(path: Path, role_map_path: Path | None = None) -> dict[str, Any]:
    role_map = load_role_map(role_map_path)
    data = path.read_bytes()
    index = 0
    if data[index : index + 4] != b"MThd":
        raise ValueError(f"Not a MIDI file: {path}")
    index += 4
    header_len = struct.unpack(">I", data[index : index + 4])[0]
    index += 4
    midi_format, track_count, ticks_per_beat = struct.unpack(">HHH", data[index : index + 6])
    index += header_len
    tempo_map: list[tuple[int, int]] = [(0, 500000)]
    time_signatures: list[dict[str, Any]] = []
    raw_tracks: list[dict[str, Any]] = []

    for track_index in range(track_count):
        if data[index : index + 4] != b"MTrk":
            raise ValueError(f"Invalid MIDI track header at track {track_index}")
        index += 4
        length = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        track_data = data[index : index + length]
        index += length
        pointer = 0
        tick = 0
        running_status: int | None = None
        name = ""
        programs: list[dict[str, Any]] = []
        channels: set[int] = set()
        notes: list[MidiNote] = []
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}

        while pointer < len(track_data):
            delta, pointer = read_vlq(track_data, pointer)
            tick += delta
            status = track_data[pointer]
            if status < 0x80:
                if running_status is None:
                    raise ValueError(f"Running status without prior status in track {track_index}")
                status = running_status
            else:
                pointer += 1
                running_status = status

            if status == 0xFF:
                meta_type = track_data[pointer]
                pointer += 1
                size, pointer = read_vlq(track_data, pointer)
                payload = track_data[pointer : pointer + size]
                pointer += size
                if meta_type == 0x03:
                    name = payload.decode("latin1", "replace")
                elif meta_type == 0x51 and size == 3:
                    tempo_map.append((tick, int.from_bytes(payload, "big")))
                elif meta_type == 0x58 and size >= 2:
                    time_signatures.append({"tick": tick, "numerator": payload[0], "denominator": 2 ** payload[1]})
                continue
            if status in (0xF0, 0xF7):
                size, pointer = read_vlq(track_data, pointer)
                pointer += size
                continue

            command = status & 0xF0
            channel = status & 0x0F
            channels.add(channel)
            size = 1 if command in (0xC0, 0xD0) else 2
            values = track_data[pointer : pointer + size]
            pointer += size
            if command == 0xC0:
                program = int(values[0])
                programs.append({"channel": channel, "program": program, "name": GM_PROGRAMS[program] if program < len(GM_PROGRAMS) else f"Program {program}"})
            elif command == 0x90:
                note, velocity = int(values[0]), int(values[1])
                if velocity > 0:
                    active.setdefault((channel, note), []).append((tick, velocity))
                else:
                    starts = active.get((channel, note), [])
                    if starts:
                        start_tick, start_velocity = starts.pop(0)
                        notes.append(MidiNote(channel, note, start_tick, max(0, tick - start_tick), start_velocity))
            elif command == 0x80:
                note = int(values[0])
                starts = active.get((channel, note), [])
                if starts:
                    start_tick, start_velocity = starts.pop(0)
                    notes.append(MidiNote(channel, note, start_tick, max(0, tick - start_tick), start_velocity))
        raw_tracks.append({"index": track_index, "name": name, "programs": programs, "channels": sorted(channels), "notes": notes})

    tempo_map = sorted(set(tempo_map))
    tracks = []
    used_ids: set[str] = set()
    for track in raw_tracks:
        notes = track["notes"]
        if not notes:
            continue
        primary_program = track["programs"][0]["program"] if track["programs"] else None
        primary_channel = notes[0].channel if notes else (track["channels"][0] if track["channels"] else None)
        role = role_for_track(track["name"], primary_program, primary_channel, role_map)
        track_id = role
        suffix = 2
        while track_id in used_ids:
            track_id = f"{role}_{suffix}"
            suffix += 1
        used_ids.add(track_id)
        note_events = []
        for note in sorted(notes, key=lambda item: (item.start_tick, item.note, item.channel)):
            start = ticks_to_seconds(note.start_tick, tempo_map, ticks_per_beat)
            end = ticks_to_seconds(note.start_tick + note.duration_ticks, tempo_map, ticks_per_beat)
            note_events.append(
                {
                    "note": note.note,
                    "start": round(start, 6),
                    "duration": round(max(0.001, end - start), 6),
                    "velocity": round(note.velocity / 127.0, 6),
                    "start_tick": note.start_tick,
                    "duration_ticks": note.duration_ticks,
                    "channel": note.channel,
                }
            )
        first = min(item["start"] for item in note_events)
        last = max(item["start"] + item["duration"] for item in note_events)
        tracks.append(
            {
                "id": track_id,
                "source_track_index": track["index"],
                "source_name": track["name"],
                "role": role,
                "channel": primary_channel,
                "gm_program": primary_program,
                "gm_program_name": GM_PROGRAMS[primary_program] if primary_program is not None and primary_program < len(GM_PROGRAMS) else None,
                "note_count": len(note_events),
                "pitch_range": [min(item["note"] for item in note_events), max(item["note"] for item in note_events)],
                "time_range": [round(first, 6), round(last, 6)],
                "notes": note_events,
            }
        )

    first_tempo = tempo_map[0][1]
    return {
        "version": 1,
        "schema": "patchex.arrangement.v1",
        "source_midi": str(path),
        "midi_format": midi_format,
        "source_track_count": track_count,
        "ticks_per_beat": ticks_per_beat,
        "tempo_map": [{"tick": tick, "bpm": round(60_000_000 / mpq, 6), "microseconds_per_quarter": mpq} for tick, mpq in tempo_map],
        "tempo": round(60_000_000 / first_tempo, 6),
        "meter": f"{time_signatures[0]['numerator']}/{time_signatures[0]['denominator']}" if time_signatures else "4/4",
        "time_signatures": time_signatures,
        "duration": round(max((track["time_range"][1] for track in tracks), default=0.0), 6),
        "tracks": tracks,
        "ownership": {
            "arrangement_workflow": ["tracks[].notes", "tracks[].role", "tracks[].time_range", "tempo", "meter"],
            "patch_workflow": ["tracks[].instrument", "tracks[].effects", "tracks[].modulation", "returns", "master"],
            "patch_workflow_may_change_notes": False,
        },
    }


def default_layer_for_track(track: dict[str, Any], sample_rate: int, duration: float) -> dict[str, Any]:
    role = str(track.get("role", "track"))
    pitch_min, pitch_max = track.get("pitch_range", [48, 72])
    is_bass = "bass" in role or pitch_max < 48
    is_drums = role == "drums"
    wavetable = "square_saw" if "square" in role else "saw_stack"
    if "piano" in role:
        wavetable = "digital"
    if "strings" in role:
        wavetable = "triangle"
    if is_drums:
        wavetable = "noise"
    return {
        "id": track["id"],
        "role": role,
        "gain_db": -14.0 if is_drums else (-10.0 if is_bass else -16.0),
        "pan": 0.0,
        "width": 0.25 if is_bass else 0.7,
        "notes": [{"note": n["note"], "start": n["start"], "duration": n["duration"], "velocity": n["velocity"]} for n in track.get("notes", [])],
        "synth": {
            "waveform": "noise" if is_drums else "saw",
            "engine": "internal",
            "wavetable": wavetable,
            "wavetable_position": 0.95 if is_drums else (0.62 if "saw" in wavetable else 0.45),
            "warp": 0.0,
            "fm_amount": 0.0,
            "fm_ratio": 2.0,
            "blend": 0.9 if is_drums else 0.55,
            "voices": 1 if is_bass or is_drums else 4,
            "detune_cents": 0.0 if is_bass or is_drums else 8.0,
            "stereo_spread": 0.2 if is_bass else 0.65,
            "sub_level": 0.35 if is_bass else 0.0,
            "vital_parameters": {},
        },
        "amp_envelope": {
            "attack": 0.001 if is_drums else (0.01 if is_bass else 0.02),
            "decay": 0.08 if is_drums else 0.25,
            "sustain": 0.0 if is_drums else 0.65,
            "release": 0.04 if is_drums else 0.25,
        },
        "filter": {
            "cutoff_start_hz": 900.0 if is_bass else 2600.0,
            "cutoff_end_hz": 1400.0 if is_bass else 4200.0,
            "resonance": 0.12,
            "drive": 0.08 if is_bass else 0.0,
            "cutoff_points": [{"time": 0.0, "hz": 900.0 if is_bass else 2600.0}, {"time": duration, "hz": 1400.0 if is_bass else 4200.0}],
        },
        "modulation": {"lfo_rate_hz": 0.0, "lfo_depth": 0.0, "gate_points": [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}], "lfos": []},
        "gain_points": [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}],
        "effects": {
            "chorus_mix": 0.0 if is_bass or is_drums else 0.12,
            "reverb_mix": 0.04 if is_bass else 0.12,
            "delay_mix": 0.0,
            "delay_time": 0.18,
            "phaser_mix": 0.0,
            "saturation": 0.08 if is_bass else 0.0,
            "low_gain_db": 0.0,
            "mid_gain_db": 0.0,
            "high_gain_db": -4.0 if is_bass else 0.0,
            "compression_mix": 0.0,
            "compression_threshold_db": -18.0,
            "compression_ratio": 2.0,
            "compression_attack": 0.01,
            "compression_release": 0.16,
            "return_send": 0.0 if is_bass or is_drums else 0.12,
        },
        "arrangement_locked": True,
        "source_midi_track_index": track.get("source_track_index"),
        "gm_program": track.get("gm_program"),
    }


def neutral_session(arrangement: dict[str, Any], sample_rate: int = 44100, seconds: float | None = None) -> dict[str, Any]:
    duration = float(seconds if seconds is not None else arrangement.get("duration", 0.0))
    session = deepcopy(DEFAULT_SESSION)
    session["schema"] = "patchex.patch_session.v1"
    session["arrangement_schema"] = arrangement.get("schema", "patchex.arrangement.v1")
    session["sample_rate"] = sample_rate
    session["duration"] = duration
    session["layers"] = [default_layer_for_track(track, sample_rate, duration) for track in arrangement.get("tracks", [])]
    session["master"] = {"gain_db": -3.0, "width": 1.0}
    return session


def slice_arrangement(arrangement: dict[str, Any], start: float, seconds: float) -> dict[str, Any]:
    """Return a clip-local arrangement with overlapping notes shifted to 0s."""
    end = start + seconds
    sliced = deepcopy(arrangement)
    sliced["source_arrangement_duration"] = arrangement.get("duration")
    sliced["clip_start"] = float(start)
    sliced["duration"] = float(seconds)
    tracks = []
    for track in arrangement.get("tracks", []):
        notes = []
        for note in track.get("notes", []):
            note_start = float(note["start"])
            note_end = note_start + float(note["duration"])
            if note_end <= start or note_start >= end:
                continue
            clipped_start = max(note_start, start)
            clipped_end = min(note_end, end)
            payload = deepcopy(note)
            payload["source_start"] = note_start
            payload["source_duration"] = float(note["duration"])
            payload["start"] = round(clipped_start - start, 6)
            payload["duration"] = round(max(0.001, clipped_end - clipped_start), 6)
            notes.append(payload)
        if not notes:
            continue
        clipped = deepcopy(track)
        clipped["notes"] = sorted(notes, key=lambda item: (item["start"], item["note"], item.get("channel", 0)))
        clipped["note_count"] = len(notes)
        clipped["pitch_range"] = [min(item["note"] for item in notes), max(item["note"] for item in notes)]
        clipped["time_range"] = [round(min(item["start"] for item in notes), 6), round(max(item["start"] + item["duration"] for item in notes), 6)]
        tracks.append(clipped)
    sliced["tracks"] = tracks
    return sliced


def canonical_notes(notes: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    out = []
    for note in notes:
        out.append(
            (
                int(note.get("note", 0)),
                int(round(float(note.get("start", 0.0)) * 1_000_000)),
                int(round(float(note.get("duration", 0.0)) * 1_000_000)),
                int(round(float(note.get("velocity", 0.0)) * 1_000_000)),
            )
        )
    return sorted(out)


def arrangement_preservation(arrangement: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    layers = {str(layer.get("id")): layer for layer in session.get("layers", [])}
    changed = []
    missing = []
    for track in arrangement.get("tracks", []):
        layer = layers.get(str(track["id"]))
        if layer is None:
            missing.append(track["id"])
            continue
        expected = canonical_notes(track.get("notes", []))
        actual = canonical_notes(layer.get("notes", []))
        if expected != actual:
            changed.append({"track_id": track["id"], "expected_notes": len(expected), "actual_notes": len(actual)})
    extra = sorted(set(layers) - {str(track["id"]) for track in arrangement.get("tracks", [])})
    total = max(1, len(arrangement.get("tracks", [])))
    penalty = min(1.0, (len(changed) + len(missing) + 0.5 * len(extra)) / total)
    return {"penalty": float(penalty), "changed_tracks": changed, "missing_tracks": missing, "extra_tracks": extra}


def enforce_arrangement_lock(arrangement: dict[str, Any], session: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    locked = deepcopy(session)
    source_tracks = {str(track["id"]): track for track in arrangement.get("tracks", [])}
    layers = []
    repaired: list[str] = []
    for track_id, track in source_tracks.items():
        existing = next((layer for layer in locked.get("layers", []) if str(layer.get("id")) == track_id), None)
        if existing is None:
            existing = default_layer_for_track(track, int(locked.get("sample_rate", 44100)), float(locked.get("duration", arrangement.get("duration", 5.0))))
            repaired.append(track_id)
        expected_notes = [{"note": n["note"], "start": n["start"], "duration": n["duration"], "velocity": n["velocity"]} for n in track.get("notes", [])]
        if canonical_notes(existing.get("notes", [])) != canonical_notes(expected_notes):
            repaired.append(track_id)
        existing["id"] = track_id
        existing["role"] = track.get("role", existing.get("role", track_id))
        existing["notes"] = expected_notes
        existing["arrangement_locked"] = True
        layers.append(existing)
    dropped = [str(layer.get("id")) for layer in locked.get("layers", []) if str(layer.get("id")) not in source_tracks]
    locked["layers"] = layers
    before = arrangement_preservation(arrangement, session)
    after = arrangement_preservation(arrangement, locked)
    return locked, {"before": before, "after": after, "repaired_tracks": sorted(set(repaired)), "dropped_extra_tracks": dropped}


def active_windows(track: dict[str, Any], duration: float, pad: float = 0.04) -> list[tuple[float, float]]:
    windows = []
    for note in track.get("notes", []):
        start = max(0.0, float(note["start"]) - pad)
        end = min(duration, float(note["start"]) + float(note["duration"]) + pad)
        if end > start:
            windows.append((start, end))
    if not windows:
        return []
    windows.sort()
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_windows(audio: np.ndarray, windows: list[tuple[float, float]], sr: int, max_seconds: float = 8.0) -> np.ndarray:
    chunks = []
    total = 0
    max_samples = int(max_seconds * sr)
    for start, end in windows:
        start_i = max(0, int(round(start * sr)))
        end_i = min(audio.shape[-1], int(round(end * sr)))
        if end_i <= start_i:
            continue
        chunk = audio[..., start_i:end_i]
        if total + chunk.shape[-1] > max_samples:
            chunk = chunk[..., : max(0, max_samples - total)]
        if chunk.shape[-1]:
            chunks.append(chunk)
            total += chunk.shape[-1]
        if total >= max_samples:
            break
    if not chunks:
        return np.zeros((2, max(1, int(0.05 * sr))), dtype=np.float32)
    return np.concatenate(chunks, axis=-1).astype(np.float32)


def rms_shape_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_env = frame_rms(mono(reference))
    cand_env = frame_rms(mono(candidate))
    rows = min(ref_env.size, cand_env.size)
    if rows == 0:
        return 1.0
    ref_env = ref_env[:rows]
    cand_env = cand_env[:rows]
    denom = np.mean(np.abs(ref_env)) + 1e-8
    return float(math.exp(-np.mean(np.abs(ref_env - cand_env)) / denom))


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    mag = stft_mag(mono(audio))
    if mag.size == 0:
        return 0.0
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    energy = np.sum(mag, axis=1) + 1e-8
    return float(np.mean(np.sum(mag * freqs[None, :], axis=1) / energy))


def patch_control_diagnostics(reference: np.ndarray, candidate: np.ndarray, sr: int) -> dict[str, Any]:
    ref_centroid = spectral_centroid(reference, sr)
    cand_centroid = spectral_centroid(candidate, sr)
    brightness = math.exp(-abs(ref_centroid - cand_centroid) / max(ref_centroid, cand_centroid, 1.0))
    envelope = rms_shape_score(reference, candidate)
    diff = compare_audio(reference, candidate, sr)
    scores = diff["scores"]
    space = float(np.mean([scores.get("late_energy_ratio", 1.0), scores.get("stereo_width", 1.0)]))
    modulation = float(np.mean([scores.get("modulation_periodicity", 1.0), scores.get("modulation_rate", 1.0), scores.get("modulation_depth", 1.0)]))
    saturation_noise = float(np.mean([scores.get("harmonic_noise", 1.0), scores.get("spectral_features", 1.0)]))
    component_scores = {
        "adsr_fit": envelope,
        "filter_brightness_fit": float(brightness),
        "modulation_fit": modulation,
        "space_fit": space,
        "saturation_noise_fit": saturation_noise,
    }
    return {"score": float(np.mean(list(component_scores.values()))), "components": component_scores}


def score_midi_locked(arrangement: dict[str, Any], session: dict[str, Any], reference_audio: np.ndarray, sr: int) -> dict[str, Any]:
    duration = float(session["duration"])
    candidate_audio = render_session(session)
    beat_grid = estimate_beat_grid(reference_audio, sr, subdivision=4)
    global_diff = compare_audio(reference_audio, candidate_audio, sr, beat_grid)
    preservation = arrangement_preservation(arrangement, session)
    layers = {str(layer.get("id")): layer for layer in session.get("layers", [])}
    track_scores = []
    active_values = []
    proxy_values = []
    control_values = []
    for track in arrangement.get("tracks", []):
        layer = layers.get(str(track["id"]))
        if layer is None:
            continue
        windows = active_windows(track, duration)
        if not windows:
            continue
        ref_slice = extract_windows(reference_audio, windows, sr)
        cand_slice = extract_windows(candidate_audio, windows, sr)
        active_diff = compare_audio(ref_slice, cand_slice, sr)
        solo = render_layer(layer, duration, sr)
        solo_slice = extract_windows(solo, windows, sr)
        proxy_diff = compare_audio(ref_slice, solo_slice, sr)
        controls = patch_control_diagnostics(ref_slice, solo_slice, sr)
        active_values.append(float(active_diff["scores"]["final"]))
        proxy_values.append(float(proxy_diff["scores"]["final"]))
        control_values.append(float(controls["score"]))
        track_scores.append(
            {
                "track_id": track["id"],
                "role": track["role"],
                "active_window_score": active_diff["scores"],
                "isolation_proxy_score": proxy_diff["scores"],
                "patch_control_diagnostics": controls,
                "window_count": len(windows),
                "active_seconds": round(sum(end - start for start, end in windows), 6),
                "weakest_active_components": active_diff["diagnostics"].get("weakest_components", []),
            }
        )
    track_active_score = float(np.mean(active_values)) if active_values else 0.0
    isolation_proxy_score = float(np.mean(proxy_values)) if proxy_values else 0.0
    patch_control_score = float(np.mean(control_values)) if control_values else 0.0
    final_loss = (
        0.50 * (1.0 - float(global_diff["scores"]["final"]))
        + 0.30 * (1.0 - track_active_score)
        + 0.15 * (1.0 - patch_control_score)
        + 0.05 * float(preservation["penalty"])
    )
    return {
        "scores": {
            "final": float(max(0.0, 1.0 - final_loss)),
            "loss": float(final_loss),
            "global_mix": float(global_diff["scores"]["final"]),
            "track_active_window": track_active_score,
            "track_isolation_proxy": isolation_proxy_score,
            "patch_control": patch_control_score,
            "arrangement_preservation": float(1.0 - preservation["penalty"]),
        },
        "global_mix_diff": global_diff,
        "track_scores": track_scores,
        "arrangement_preservation": preservation,
        "weakest_tracks": sorted(track_scores, key=lambda item: item["active_window_score"].get("final", 0.0))[:5],
    }


def codex_patch_prompt(arrangement_path: Path, current_session_path: Path, recommendation_path: Path, previous_report_path: Path | None, output_path: Path) -> str:
    previous_line = f"- Previous MIDI-locked patch report: {previous_report_path}" if previous_report_path else "- No previous patch report exists."
    return f"""
You are the Patch Producer in a MIDI-locked patch finding workflow.

Read these files:
- Authoritative arrangement JSON: {arrangement_path}
- Current patch session JSON: {current_session_path}
- Critic recommendation JSON: {recommendation_path}
{previous_line}

Task: Write a complete patch session JSON to {output_path}.

Hard boundary:
- The Arrangement/MIDI workflow owns track ids, roles, and every note event.
- You may change synth, amp_envelope, filter, modulation, gain_points, effects, returns, master, pan, width, and gain_db.
- You must not add, delete, reorder, retime, transpose, or change velocity of notes.
- Preserve every layer id from arrangement.tracks[].id.

Optimization target:
- Improve global full-mix similarity.
- Improve per-track active-window scores.
- Use track patch diagnostics to choose oscillator, ADSR, filter, modulation, space, saturation/noise, and mix changes.
- Drums may be left approximate unless their active-window score is the main weakness.

Return only JSON matching the current session shape. Do not include prose.
"""


def codex_critic_prompt(arrangement_path: Path, current_session_path: Path, previous_report_path: Path | None, output_path: Path) -> str:
    previous_line = f"- Latest MIDI-locked patch report JSON: {previous_report_path}" if previous_report_path else "- No patch has been scored yet."
    return f"""
You are the Critic agent in a MIDI-locked patch finding workflow.

Read these files:
- Authoritative arrangement JSON: {arrangement_path}
- Current patch session JSON: {current_session_path}
{previous_line}

Task: Write the next Producer brief to {output_path}.

Important:
- MIDI lock is the default workflow contract, not a degraded mode.
- Do not recommend changing notes, timing, velocities, layer ids, or roles.
- Use the latest report's global_mix, track_active_window, patch_control, weakest_tracks, per-track active-window scores, isolation proxy scores, and patch-control diagnostics.
- Name the specific track ids and parameter families that should change next.

Return only JSON with this shape:
{{
  "workflow": "midi_locked_patch",
  "missing": ["specific measurable mismatches"],
  "producer_prompt": "specific brief for the Producer focused on patch, effects, modulation, and mix changes",
  "success_metrics": {{
    "primary": ["metric names to improve"],
    "targets": ["concrete directional targets"],
    "failure_modes": ["what would prove the Producer made the wrong type of patch change"]
  }},
  "recommendations": ["specific patch/mix moves"],
  "must_fix": ["track_id: parameter family and why"],
  "do_not": ["specific mistakes to avoid"],
  "priority": "oscillator|envelope|filter|modulation|effects|mix|stereo",
  "target_tracks": ["track ids that should change"]
}}
"""


def run_codex_patch(agent: str, prompt: str, output_dir: Path, answer_path: Path, timeout: int = 180) -> None:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    prompt_path.write_text(prompt)
    print(f"codex_start agent={agent}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    process = subprocess.run(
        [CODEX_PATH, "exec", "--skip-git-repo-check", "--output-last-message", str(output_dir / f"codex_{agent}_answer.txt"), "-C", str(Path.cwd()), "-"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Codex patch producer failed with return code {process.returncode}:\n{process.stdout}")
    codex_answer_path = output_dir / f"codex_{agent}_answer.txt"
    print(f"codex_done agent={agent} answer_path={codex_answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={codex_answer_path}", flush=True)
    if not answer_path.exists():
        payload = extract_json_object(codex_answer_path.read_text())
        answer_path.write_text(json.dumps(payload, indent=2) + "\n")


def run_codex_json_agent(agent: str, prompt: str, output_dir: Path, json_output_path: Path, timeout: int = 180) -> dict[str, Any]:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    answer_path = output_dir / f"codex_{agent}_answer.txt"
    prompt = (
        f"{prompt.rstrip()}\n\n"
        "File-driven orchestration requirement:\n"
        f"- Write the JSON artifact at: {json_output_path}\n"
        "- The orchestrator will read that JSON file after this run.\n"
    )
    prompt_path.write_text(prompt)
    print(f"codex_start agent={agent}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    process = subprocess.run(
        [CODEX_PATH, "exec", "--skip-git-repo-check", "--output-last-message", str(answer_path), "-C", str(Path.cwd()), "-"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Codex agent {agent} failed with return code {process.returncode}:\n{process.stdout}")
    print(f"codex_done agent={agent} answer_path={answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
    payload_text = json_output_path.read_text() if json_output_path.exists() and json_output_path.read_text().strip() else answer_path.read_text()
    payload = extract_json_object(payload_text)
    json_output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_audio(path: Path, seconds: float | None = None) -> tuple[np.ndarray, int]:
    runtime()
    audio, sr = sf.read(path, always_2d=True)
    data = audio.T.astype(np.float32)
    if seconds is not None:
        data = data[:, : int(seconds * sr)]
    return data, int(sr)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def command_import(args: argparse.Namespace) -> int:
    arrangement = parse_midi(args.midi, args.role_map)
    write_json(args.output, arrangement)
    print(f"wrote {args.output}")
    return 0


def command_neutral(args: argparse.Namespace) -> int:
    arrangement = json.loads(args.arrangement.read_text())
    if args.clip_start is not None:
        arrangement = slice_arrangement(arrangement, args.clip_start, args.seconds or 5.0)
    session = neutral_session(arrangement, args.sample_rate, args.seconds)
    write_json(args.output, session)
    print(f"wrote {args.output}")
    return 0


def command_score(args: argparse.Namespace) -> int:
    runtime()
    arrangement = json.loads(args.arrangement.read_text())
    session = json.loads(args.session.read_text())
    reference_audio, sr = load_audio(args.reference, args.seconds or float(session.get("duration", arrangement.get("duration", 0.0))))
    if sr != int(session["sample_rate"]):
        raise RuntimeError(f"Reference sample rate {sr} does not match session sample rate {session['sample_rate']}")
    report = score_midi_locked(arrangement, session, reference_audio, sr)
    write_json(args.output, report)
    if args.render_output:
        sf.write(args.render_output, render_session(session).T, sr)
        print(f"wrote {args.render_output}")
    print(f"wrote {args.output}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrangement_path = args.output_dir / "arrangement.json"
    current_session_path = args.output_dir / "patch_session_current.json"
    full_arrangement = parse_midi(args.midi, args.role_map)
    full_arrangement_path = write_json(args.output_dir / "full_arrangement.json", full_arrangement)
    print(f"trace_file agent=analyzer role=full_arrangement path={full_arrangement_path}", flush=True)
    arrangement = slice_arrangement(full_arrangement, args.clip_start, args.seconds) if args.clip_start is not None else full_arrangement
    write_json(arrangement_path, arrangement)
    print(f"trace_file agent=analyzer role=arrangement path={arrangement_path}", flush=True)
    session = neutral_session(arrangement, args.sample_rate, args.seconds)
    write_json(current_session_path, session)
    print(f"trace_file agent=producer_step_00 role=session_proposal path={current_session_path}", flush=True)
    reference_audio, sr = load_audio(args.reference, args.seconds)
    source_clip_path = args.output_dir / "source_clip.wav"
    sf.write(source_clip_path, reference_audio.T, sr)
    print(f"trace_file agent=analyzer role=source_clip path={source_clip_path}", flush=True)
    previous_report_path: Path | None = None
    history: list[dict[str, Any]] = []
    best_report: dict[str, Any] | None = None
    best_session = session
    best_render_path: Path | None = None
    for step in range(args.steps):
        lock_report: dict[str, Any] = {}
        print(f"agent_stage residual_critic step={step}", flush=True)
        recommendation_path = args.output_dir / f"recommendation_step_{step:02d}.json"
        if args.neutral_only:
            recommendation = {
                "workflow": "midi_locked_patch",
                "missing": ["neutral-session smoke run"],
                "producer_prompt": "Render the neutral MIDI-locked session without Codex patch changes.",
                "success_metrics": {"primary": ["arrangement_preservation"], "targets": ["arrangement_preservation stays 1.0"], "failure_modes": ["notes changed"]},
                "recommendations": [],
                "must_fix": [],
                "do_not": ["Do not alter MIDI notes."],
                "priority": "mix",
                "target_tracks": [],
            }
            write_json(recommendation_path, recommendation)
        else:
            recommendation = run_codex_json_agent(
                f"residual_critic_step_{step:02d}",
                codex_critic_prompt(arrangement_path, current_session_path, previous_report_path, recommendation_path),
                args.output_dir,
                recommendation_path,
                args.timeout,
            )
        print(f"trace_file agent=residual_critic_step_{step:02d} role=recommendation path={recommendation_path}", flush=True)
        print(f"agent_stage producer step={step}", flush=True)
        proposal_path = args.output_dir / f"patch_session_step_{step:02d}.json"
        if step == 0 and args.neutral_only:
            proposal = session
            proposal, lock_report = enforce_arrangement_lock(arrangement, proposal)
            write_json(proposal_path, proposal)
        else:
            prompt = codex_patch_prompt(arrangement_path, current_session_path, recommendation_path, previous_report_path, proposal_path)
            run_codex_patch(f"producer_step_{step:02d}", prompt, args.output_dir, proposal_path, args.timeout)
            raw_proposal = json.loads(proposal_path.read_text())
            proposal, lock_report = enforce_arrangement_lock(arrangement, raw_proposal)
            write_json(proposal_path, proposal)
            write_json(args.output_dir / f"arrangement_lock_step_{step:02d}.json", lock_report)
        report = score_midi_locked(arrangement, proposal, reference_audio, sr)
        report_path = write_json(args.output_dir / f"patch_report_step_{step:02d}.json", report)
        print(f"trace_file agent=loss step={step} role=audio_diff path={report_path}", flush=True)
        render_path = args.output_dir / f"patch_render_step_{step:02d}.wav"
        sf.write(render_path, render_session(proposal).T, sr)
        print(f"trace_file agent=loss step={step} role=winner_render path={render_path}", flush=True)
        accepted = best_report is None or report["scores"]["final"] >= best_report["scores"]["final"]
        if accepted:
            best_report = report
            best_session = proposal
            best_render_path = render_path
        session = proposal
        current_session_path = write_json(args.output_dir / "patch_session_current.json", session)
        previous_report_path = report_path
        history_item = {
            "stage": "midi_locked_patch",
            "step": step,
            "accepted": accepted,
            "winner": "midi_locked_patch",
            "audio_path": str(render_path),
            "audio_diff_path": str(report_path),
            "proposal_session_path": str(proposal_path),
            "scores": report["scores"],
            "best_scores": (best_report or report)["scores"],
            "layers": [{"id": layer.get("id"), "role": layer.get("role")} for layer in proposal.get("layers", [])],
            "arrangement_preservation": report["arrangement_preservation"],
            "arrangement_lock_report": lock_report or {"before": report["arrangement_preservation"], "after": report["arrangement_preservation"]},
            "residual_critic": recommendation,
        }
        history.append(history_item)
        write_json(args.output_dir / f"history_item_step_{step:02d}.json", history_item)
        print(f"winner_summary step={step} winner=midi_locked_patch score={report['scores']['final']:.4f}", flush=True)
        print(f"step_complete index={step} winner=midi_locked_patch accepted={str(accepted).lower()} best_score={(best_report or report)['scores']['final']:.4f}", flush=True)
    final_render = args.output_dir / "final_reconstruction.wav"
    if best_render_path is not None:
        final_render.write_bytes(best_render_path.read_bytes())
    session_path = write_json(args.output_dir / "reconstruction_session.json", best_session)
    report_payload = {
        "workflow": "midi_locked_patch",
        "reference": str(args.reference),
        "source_clip": str(args.reference),
        "midi": str(args.midi),
        "clip_start": args.clip_start,
        "arrangement_path": str(arrangement_path),
        "full_arrangement_path": str(full_arrangement_path),
        "history": history,
        "best_scores": (best_report or {"scores": {}})["scores"],
        "final_path": str(final_render),
        "session_path": str(session_path),
        "midi_locked_patch_report": best_report,
    }
    write_json(args.output_dir / "reconstruction_report.json", report_payload)
    write_json(args.output_dir / "midi_locked_patch_report.json", {"arrangement_path": str(arrangement_path), "current_session_path": str(current_session_path), "last_report_path": str(previous_report_path), "best_scores": report_payload["best_scores"]})
    print(f"wrote {final_render}", flush=True)
    print(f"wrote {args.output_dir / 'reconstruction_report.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIDI-locked arrangement import and patch scoring workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_import = sub.add_parser("import-midi")
    p_import.add_argument("--midi", required=True, type=Path)
    p_import.add_argument("--output", required=True, type=Path)
    p_import.add_argument("--role-map", type=Path)
    p_import.set_defaults(func=command_import)

    p_neutral = sub.add_parser("neutral-session")
    p_neutral.add_argument("--arrangement", required=True, type=Path)
    p_neutral.add_argument("--output", required=True, type=Path)
    p_neutral.add_argument("--sample-rate", type=int, default=44100)
    p_neutral.add_argument("--seconds", type=float)
    p_neutral.add_argument("--clip-start", type=float)
    p_neutral.set_defaults(func=command_neutral)

    p_score = sub.add_parser("score-session")
    p_score.add_argument("--arrangement", required=True, type=Path)
    p_score.add_argument("--session", required=True, type=Path)
    p_score.add_argument("--reference", required=True, type=Path)
    p_score.add_argument("--output", required=True, type=Path)
    p_score.add_argument("--render-output", type=Path)
    p_score.add_argument("--seconds", type=float)
    p_score.set_defaults(func=command_score)

    p_run = sub.add_parser("run")
    p_run.add_argument("--midi", required=True, type=Path)
    p_run.add_argument("--role-map", type=Path)
    p_run.add_argument("--reference", required=True, type=Path)
    p_run.add_argument("--output-dir", required=True, type=Path)
    p_run.add_argument("--sample-rate", type=int, default=44100)
    p_run.add_argument("--seconds", type=float, default=5.0)
    p_run.add_argument("--clip-start", type=float)
    p_run.add_argument("--steps", type=int, default=1)
    p_run.add_argument("--timeout", type=int, default=180)
    p_run.add_argument("--neutral-only", action="store_true")
    p_run.set_defaults(func=command_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
