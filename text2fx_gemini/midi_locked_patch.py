#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import time
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


def normalize_tempo_map(tempo_map: list[tuple[int, int]]) -> list[tuple[int, int]]:
    by_tick: dict[int, int] = {}
    for tick, mpq in tempo_map:
        by_tick[int(tick)] = int(mpq)
    return sorted(by_tick.items()) or [(0, 500000)]


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

    tempo_map = normalize_tempo_map(tempo_map)
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
    is_lead = "saw" in role or "square" in role or "fifth" in role
    wavetable = "square" if "square" in role else "saw_stack"
    waveform = "saw"
    gain_db = -14.0 if is_drums else (-9.0 if is_bass else (-12.5 if is_lead else -15.5))
    width = 0.15 if is_bass or is_drums else (0.45 if is_lead else 0.7)
    voices = 1 if is_bass or is_drums or is_lead else 3
    detune = 0.0 if is_bass or is_drums or is_lead else 4.0
    cutoff_start = 850.0 if is_bass else (7200.0 if is_drums else (5200.0 if is_lead else 2600.0))
    cutoff_end = 1500.0 if is_bass else (9200.0 if is_drums else (6800.0 if is_lead else 3600.0))
    attack = 0.001 if is_drums else (0.008 if is_bass or is_lead else 0.025)
    decay = 0.05 if is_drums else (0.12 if is_lead else 0.25)
    sustain = 0.0 if is_drums else (0.35 if is_lead else 0.65)
    release = 0.035 if is_drums else (0.08 if is_lead else 0.25)
    if "piano" in role:
        wavetable = "digital"
        waveform = "triangle"
    if "strings" in role:
        wavetable = "triangle"
        waveform = "triangle"
    if is_drums:
        wavetable = "noise"
        waveform = "noise"
    return {
        "id": track["id"],
        "role": role,
        "gain_db": gain_db,
        "pan": 0.0,
        "width": width,
        "notes": [{"note": n["note"], "start": n["start"], "duration": n["duration"], "velocity": n["velocity"]} for n in track.get("notes", [])],
        "synth": {
            "waveform": waveform,
            "engine": "internal",
            "wavetable": wavetable,
            "wavetable_position": 0.95 if is_drums else (0.78 if is_lead else (0.62 if "saw" in wavetable else 0.45)),
            "warp": 0.0,
            "fm_amount": 0.0,
            "fm_ratio": 2.0,
            "blend": 0.9 if is_drums else (0.72 if is_lead else 0.55),
            "voices": voices,
            "detune_cents": detune,
            "stereo_spread": width,
            "sub_level": 0.35 if is_bass else 0.0,
            "vital_parameters": {},
        },
        "amp_envelope": {
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
        },
        "filter": {
            "cutoff_start_hz": cutoff_start,
            "cutoff_end_hz": cutoff_end,
            "resonance": 0.12,
            "drive": 0.08 if is_bass else 0.0,
            "cutoff_points": [{"time": 0.0, "hz": cutoff_start}, {"time": duration, "hz": cutoff_end}],
        },
        "modulation": {"lfo_rate_hz": 0.0, "lfo_depth": 0.0, "gate_points": [{"time": 0.0, "level": 1.0}, {"time": duration, "level": 1.0}], "lfos": []},
        "gain_points": [{"time": 0.0, "db": 0.0}, {"time": duration, "db": 0.0}],
        "effects": {
            "chorus_mix": 0.0 if is_bass or is_drums or is_lead else 0.08,
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


def slice_patch_session(session: dict[str, Any], start: float, seconds: float) -> dict[str, Any]:
    end = start + seconds
    sliced = deepcopy(session)
    sliced["source_session_duration"] = session.get("duration")
    sliced["window_start"] = float(start)
    sliced["duration"] = float(seconds)
    layers = []
    for layer in session.get("layers", []):
        clipped = deepcopy(layer)
        notes = []
        for note in layer.get("notes", []):
            note_start = float(note.get("start", 0.0))
            note_end = note_start + float(note.get("duration", 0.0))
            if note_end <= start or note_start >= end:
                continue
            clipped_start = max(note_start, start)
            clipped_end = min(note_end, end)
            payload = deepcopy(note)
            payload["source_start"] = note_start
            payload["source_duration"] = float(note.get("duration", 0.0))
            payload["start"] = round(clipped_start - start, 6)
            payload["duration"] = round(max(0.0, clipped_end - clipped_start), 6)
            notes.append(payload)
        clipped["notes"] = sorted(notes, key=lambda item: (float(item.get("start", 0.0)), int(item.get("note", 0))))
        layers.append(clipped)
    sliced["layers"] = layers
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


BLOCKED_PATCH_PATH_PARTS = {"notes", "id", "role", "arrangement_locked", "source_midi_track_index", "gm_program"}
ALLOWED_PATCH_ROOTS = {"layers", "returns", "master", "production_notes"}


def load_patch_ops_payload(path: Path) -> dict[str, Any]:
    if path.exists() and path.read_text().strip():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Patch operation file must contain a JSON object.")
        payload.setdefault("schema", "patchex.patch_ops.v1")
        payload.setdefault("operations", [])
        payload.setdefault("loss_trials", [])
        return payload
    return {"schema": "patchex.patch_ops.v1", "hypothesis": "", "critic_brief_used": "", "operations": [], "loss_trials": []}


def write_patch_ops_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["schema"] = "patchex.patch_ops.v1"
    payload.setdefault("operations", [])
    payload.setdefault("loss_trials", [])
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def set_production_hypothesis(operations_path: Path | str, hypothesis: str, critic_brief_used: Path | str = "") -> Path:
    path = Path(operations_path)
    payload = load_patch_ops_payload(path)
    payload["hypothesis"] = str(hypothesis)
    if critic_brief_used:
        payload["critic_brief_used"] = str(critic_brief_used)
    return write_patch_ops_payload(path, payload)


def save_patch_change(
    operations_path: Path | str,
    *,
    path: str,
    value: Any,
    track_id: str = "",
    change: str,
    reason: str,
    op: str = "set",
) -> Path:
    parse_patch_path(path)
    payload = load_patch_ops_payload(Path(operations_path))
    payload["operations"].append(
        {
            "op": str(op),
            "path": str(path),
            "value": deepcopy(value),
            "track_id": str(track_id),
            "change": str(change),
            "reason": str(reason),
        }
    )
    return write_patch_ops_payload(Path(operations_path), payload)


def save_loss_trial(
    operations_path: Path | str,
    *,
    command: str,
    score: float | None = None,
    loss: float | None = None,
    window_start: float | None = None,
    window_duration: float | None = None,
    notes: str = "",
) -> Path:
    payload = load_patch_ops_payload(Path(operations_path))
    trial: dict[str, Any] = {"command": str(command), "notes": str(notes)}
    if score is not None:
        trial["score"] = float(score)
    if loss is not None:
        trial["loss"] = float(loss)
    if window_start is not None:
        trial["window_start"] = float(window_start)
    if window_duration is not None:
        trial["window_duration"] = float(window_duration)
    payload["loss_trials"].append(trial)
    return write_patch_ops_payload(Path(operations_path), payload)


def parse_patch_path(path: str) -> list[str]:
    parts = [part for part in str(path).strip().split(".") if part]
    if not parts:
        raise ValueError("Patch operation path is empty.")
    if parts[0] not in ALLOWED_PATCH_ROOTS:
        raise ValueError(f"Patch operation root is not editable: {parts[0]}")
    if any(part in BLOCKED_PATCH_PATH_PARTS for part in parts):
        raise ValueError(f"Patch operation may not edit arrangement-owned path: {path}")
    return parts


def resolve_patch_parent(session: dict[str, Any], parts: list[str]) -> tuple[Any, str]:
    current: Any = session
    index = 0
    while index < len(parts) - 1:
        part = parts[index]
        if part == "layers":
            if index + 1 >= len(parts) - 1:
                raise ValueError("Layer patch path must include a field after the layer id.")
            layer_id = parts[index + 1]
            if not isinstance(current.get("layers"), list):
                raise ValueError("Session layers must be a list.")
            current = next((layer for layer in current["layers"] if str(layer.get("id")) == layer_id), None)
            if current is None:
                raise ValueError(f"Unknown layer id in patch operation path: {layer_id}")
            index += 2
            continue
        if part == "returns":
            if index + 1 >= len(parts) - 1:
                raise ValueError("Return patch path must include a field after the return id.")
            return_id = parts[index + 1]
            if not isinstance(current.get("returns"), list):
                raise ValueError("Session returns must be a list.")
            current = next((ret for ret in current["returns"] if str(ret.get("id")) == return_id), None)
            if current is None:
                raise ValueError(f"Unknown return id in patch operation path: {return_id}")
            index += 2
            continue
        if isinstance(current, dict):
            next_part = parts[index + 1]
            if part not in current or current[part] is None:
                current[part] = [] if next_part.isdigit() else {}
            current = current[part]
            index += 1
            continue
        if isinstance(current, list) and part.isdigit():
            current = current[int(part)]
            index += 1
            continue
        raise ValueError(f"Cannot resolve patch operation path: {'.'.join(parts)}")
    return current, parts[-1]


def apply_patch_operations(session: dict[str, Any], operations_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(session)
    operations = operations_payload.get("operations", operations_payload.get("changes", []))
    if not isinstance(operations, list) or not operations:
        raise ValueError("Producer must return a non-empty operations list.")
    notes = updated.setdefault("production_notes", {})
    if operations_payload.get("hypothesis"):
        notes["hypothesis"] = str(operations_payload["hypothesis"])
    if operations_payload.get("critic_brief_used"):
        notes["critic_brief_used"] = str(operations_payload["critic_brief_used"])
    notes.setdefault("change_log", [])
    notes.setdefault("loss_trials", [])
    if isinstance(operations_payload.get("loss_trials"), list):
        notes["loss_trials"].extend(operations_payload["loss_trials"])
    applied = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each patch operation must be an object.")
        op = str(operation.get("op", "set")).lower()
        path = str(operation.get("path", ""))
        parts = parse_patch_path(path)
        parent, key = resolve_patch_parent(updated, parts)
        value = deepcopy(operation.get("value"))
        if op == "set":
            if isinstance(parent, dict):
                parent[key] = value
            elif isinstance(parent, list) and key.isdigit():
                parent[int(key)] = value
            else:
                raise ValueError(f"Cannot set patch operation path: {path}")
        elif op == "append":
            target = parent.get(key) if isinstance(parent, dict) else parent[int(key)]
            if not isinstance(target, list):
                raise ValueError(f"Append target is not a list: {path}")
            target.append(value)
        elif op == "extend":
            target = parent.get(key) if isinstance(parent, dict) else parent[int(key)]
            if not isinstance(target, list) or not isinstance(value, list):
                raise ValueError(f"Extend target/value must be lists: {path}")
            target.extend(value)
        else:
            raise ValueError(f"Unsupported patch operation: {op}")
        entry = {
            "track_id": str(operation.get("track_id") or (parts[1] if parts[0] == "layers" and len(parts) > 1 else parts[0])),
            "parameter_path": path,
            "change": str(operation.get("change") or f"{op} {path}"),
            "reason": str(operation.get("reason") or operations_payload.get("hypothesis") or ""),
        }
        notes["change_log"].append(entry)
        applied.append(entry)
    return updated, {"applied": applied, "operation_count": len(applied)}


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


def sustain_shape_diagnostics(diff: dict[str, Any]) -> dict[str, Any]:
    fixed = diff.get("diagnostics", {}).get("fixed_50ms", {})
    windows = fixed.get("windows") or []
    if len(windows) < 3:
        return {
            "score": 1.0,
            "largest_abs_rms_error": 0.0,
            "pumping_events": 0,
            "weak_windows": [],
            "summary": "Not enough 50ms windows to judge sustained-envelope shape.",
        }
    ref = np.asarray([float(item.get("source_rms", 0.0)) for item in windows], dtype=np.float32)
    cand = np.asarray([float(item.get("candidate_rms", 0.0)) for item in windows], dtype=np.float32)
    delta = cand - ref
    ref_scale = float(np.mean(np.abs(ref)) + 1e-8)
    abs_error = np.abs(delta)
    derivative_error = np.abs(np.diff(cand) - np.diff(ref))
    sign_changes = int(np.sum(np.diff(np.signbit(delta)) != 0)) if delta.size > 1 else 0
    large_reversals = int(np.sum(np.abs(np.diff(cand)) > max(ref_scale * 0.35, 1e-4)))
    error_score = math.exp(-float(np.mean(abs_error)) / ref_scale)
    derivative_score = math.exp(-float(np.mean(derivative_error)) / ref_scale) if derivative_error.size else 1.0
    pumping_penalty = math.exp(-0.08 * float(sign_changes + large_reversals))
    score = float(max(0.0, min(1.0, 0.55 * error_score + 0.30 * derivative_score + 0.15 * pumping_penalty)))
    weak_windows = sorted(
        windows,
        key=lambda item: abs(float(item.get("candidate_rms", 0.0)) - float(item.get("source_rms", 0.0))),
        reverse=True,
    )[:8]
    direction = "under" if float(np.mean(delta)) < 0 else "over"
    return {
        "score": score,
        "largest_abs_rms_error": float(np.max(abs_error)),
        "mean_abs_rms_error": float(np.mean(abs_error)),
        "directional_reversal_count": sign_changes,
        "large_candidate_level_jumps": large_reversals,
        "pumping_events": sign_changes + large_reversals,
        "dominant_error": direction,
        "weak_windows": weak_windows,
        "summary": (
            f"50ms sustain shape score {score:.3f}; candidate is mostly {direction} target with "
            f"{sign_changes + large_reversals} pumping/reversal events."
        ),
    }


def loudness_floor_diagnostics(diff: dict[str, Any]) -> dict[str, Any]:
    fixed = diff.get("diagnostics", {}).get("fixed_50ms", {})
    windows = fixed.get("windows") or []
    if not windows:
        return {
            "score": 1.0,
            "median_rms_ratio": 1.0,
            "p10_rms_ratio": 1.0,
            "max_under_db": 0.0,
            "dropout_window_count": 0,
            "severe_under_window_count": 0,
            "weak_windows": [],
            "summary": "No fixed-window loudness diagnostics available.",
        }
    ratios = []
    under_windows = []
    for item in windows:
        source = float(item.get("source_rms", 0.0))
        candidate = float(item.get("candidate_rms", 0.0))
        if source < 1e-5:
            continue
        ratio = candidate / max(source, 1e-8)
        ratios.append(ratio)
        under_db = 20.0 * math.log10(max(ratio, 1e-8))
        if under_db < -6.0:
            under_windows.append({**item, "rms_ratio": ratio, "under_db": under_db})
    if not ratios:
        return {
            "score": 1.0,
            "median_rms_ratio": 1.0,
            "p10_rms_ratio": 1.0,
            "max_under_db": 0.0,
            "dropout_window_count": 0,
            "severe_under_window_count": 0,
            "weak_windows": [],
            "summary": "Reference is silent enough that no loudness floor was required.",
        }
    ratios_arr = np.asarray(ratios, dtype=np.float32)
    median_ratio = float(np.median(ratios_arr))
    p10_ratio = float(np.percentile(ratios_arr, 10))
    min_ratio = float(np.min(ratios_arr))
    max_under_db = float(20.0 * math.log10(max(min_ratio, 1e-8)))
    dropout_count = int(np.sum(ratios_arr < 0.25))
    severe_under_count = int(np.sum(ratios_arr < 0.50))
    median_score = min(1.0, median_ratio / 0.85)
    p10_score = min(1.0, p10_ratio / 0.55)
    dropout_score = math.exp(-0.18 * dropout_count - 0.06 * severe_under_count)
    score = float(max(0.0, min(1.0, 0.42 * median_score + 0.38 * p10_score + 0.20 * dropout_score)))
    weak_windows = sorted(under_windows, key=lambda item: item["rms_ratio"])[:10]
    return {
        "score": score,
        "median_rms_ratio": median_ratio,
        "p10_rms_ratio": p10_ratio,
        "max_under_db": max_under_db,
        "dropout_window_count": dropout_count,
        "severe_under_window_count": severe_under_count,
        "weak_windows": weak_windows,
        "summary": (
            f"Loudness floor score {score:.3f}; median RMS ratio {median_ratio:.2f}, "
            f"p10 ratio {p10_ratio:.2f}, {dropout_count} dropout windows below -12 dB."
        ),
    }


def groove_envelope_diagnostics(diff: dict[str, Any]) -> dict[str, Any]:
    scores = diff.get("scores", {})
    beat_grid = diff.get("diagnostics", {}).get("beat_grid_scores", {})
    fixed = diff.get("diagnostics", {}).get("fixed_50ms", {})
    component_scores = {
        "beat_grid_envelope": float(scores.get("beat_grid_envelope", beat_grid.get("envelope", 1.0))),
        "beat_grid_band": float(scores.get("beat_grid_band", beat_grid.get("band", 1.0))),
        "beat_grid_mid_side": float(scores.get("beat_grid_mid_side", beat_grid.get("mid_side", 1.0))),
        "directional_delta": float(scores.get("directional_delta", 1.0)),
        "exact_envelope_50ms": float(scores.get("exact_envelope_50ms", fixed.get("envelope", 1.0))),
    }
    score = float(np.mean(list(component_scores.values())))
    weak_slices = beat_grid.get("weak_slices", [])[:10]
    return {
        "score": score,
        "components": component_scores,
        "weak_slices": weak_slices,
        "summary": f"Groove/envelope score {score:.3f}; weakest component is {min(component_scores.items(), key=lambda item: item[1])[0]}.",
    }


def modulation_identity_diagnostics(diff: dict[str, Any]) -> dict[str, Any]:
    scores = diff.get("scores", {})
    analysis = diff.get("diagnostics", {}).get("modulation_analysis", {})
    reference = analysis.get("reference", {})
    candidate = analysis.get("candidate", {})
    reference_rate = float(reference.get("rate_hz", 0.0))
    candidate_rate = float(candidate.get("rate_hz", 0.0))
    reference_depth = float(reference.get("depth", 0.0))
    candidate_depth = float(candidate.get("depth", 0.0))
    reference_periodicity = float(reference.get("periodicity", 0.0))
    candidate_periodicity = float(candidate.get("periodicity", 0.0))
    component_scores = {
        "rate": float(scores.get("modulation_rate", 1.0)),
        "depth": float(scores.get("modulation_depth", 1.0)),
        "periodicity": float(scores.get("modulation_periodicity", 1.0)),
        "band_motion": float(scores.get("modulation", 1.0)),
        "spectral_motion": float(scores.get("spectral_motion", 1.0)),
    }
    score = float(np.mean(list(component_scores.values())))
    reference_has_cyclic_motion = reference_periodicity >= 0.10 and reference_depth >= 0.04 and reference_rate >= 0.20
    candidate_too_slow = reference_has_cyclic_motion and candidate_rate < reference_rate * 0.60
    candidate_too_shallow = reference_has_cyclic_motion and candidate_depth < reference_depth * 0.60
    movement_type = "cyclic_lfo_like" if reference_has_cyclic_motion else "mostly_automation_or_drift"
    return {
        "score": score,
        "components": component_scores,
        "reference": reference,
        "candidate": candidate,
        "reference_movement_type": movement_type,
        "candidate_too_slow": bool(candidate_too_slow),
        "candidate_too_shallow": bool(candidate_too_shallow),
        "summary": (
            f"Modulation identity score {score:.3f}; reference {reference_rate:.2f}Hz/depth {reference_depth:.2f}, "
            f"candidate {candidate_rate:.2f}Hz/depth {candidate_depth:.2f}."
        ),
    }


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
    sustain_shape = sustain_shape_diagnostics(global_diff)
    loudness_floor = loudness_floor_diagnostics(global_diff)
    groove_envelope = groove_envelope_diagnostics(global_diff)
    modulation_identity = modulation_identity_diagnostics(global_diff)
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
    global_scores = global_diff["scores"]
    spectral_timbre = float(
        np.mean(
            [
                global_scores.get("multi_resolution_spectral", 0.0),
                global_scores.get("mel_spectrogram", 0.0),
                global_scores.get("a_weighted_spectral", 0.0),
                global_scores.get("spectral_features", 0.0),
                global_scores.get("harmonic_noise", 0.0),
                global_scores.get("codec_latent", 0.0),
            ]
        )
    )
    loudness_envelope = float(
        np.mean(
            [
                global_scores.get("envelope", 0.0),
                global_scores.get("segment_envelope", 0.0),
                global_scores.get("exact_envelope_50ms", 0.0),
                loudness_floor["score"],
                sustain_shape["score"],
            ]
        )
    )
    stereo_motion = float(np.mean([global_scores.get("stereo_width", 0.0), global_scores.get("beat_grid_mid_side", 0.0)]))
    pitch_chroma = float(np.mean([global_scores.get("pitch_chroma", 0.0), global_scores.get("f0_contour", 0.0)]))
    weighted_final = (
        0.25 * spectral_timbre
        + 0.20 * loudness_envelope
        + 0.20 * float(groove_envelope["score"])
        + 0.15 * float(modulation_identity["score"])
        + 0.10 * stereo_motion
        + 0.05 * pitch_chroma
        + 0.05 * patch_control_score
    )
    hard_gates = {
        "loudness_floor": float(loudness_floor["score"]),
        "groove_envelope": float(groove_envelope["score"]),
        "modulation_identity": float(modulation_identity["score"]),
        "arrangement_preservation": float(1.0 - preservation["penalty"]),
    }
    gate_multiplier = float(min(1.0, 0.58 + 0.42 * min(hard_gates.values())))
    dropout_penalty = math.exp(-0.08 * float(loudness_floor["dropout_window_count"]))
    final_loss = (
        1.0
        - max(0.0, min(1.0, weighted_final * gate_multiplier * dropout_penalty))
        + 0.05 * float(preservation["penalty"])
    )
    final_score = float(max(0.0, 1.0 - final_loss))
    return {
        "scores": {
            "final": final_score,
            "loss": float(final_loss),
            "global_mix": float(global_diff["scores"]["final"]),
            "track_active_window": track_active_score,
            "track_isolation_proxy": isolation_proxy_score,
            "patch_control": patch_control_score,
            "sustain_shape": float(sustain_shape["score"]),
            "loudness_envelope": loudness_envelope,
            "loudness_floor": float(loudness_floor["score"]),
            "groove_envelope": float(groove_envelope["score"]),
            "modulation_identity": float(modulation_identity["score"]),
            "spectral_timbre": spectral_timbre,
            "stereo_motion": stereo_motion,
            "pitch_chroma": pitch_chroma,
            "arrangement_preservation": float(1.0 - preservation["penalty"]),
        },
        "loss_version": "midi_locked_patch.v2_strict_loudness_modulation",
        "loss_components": {
            "weighted_final_before_gates": weighted_final,
            "gate_multiplier": gate_multiplier,
            "dropout_penalty": dropout_penalty,
            "hard_gates": hard_gates,
            "weights": {
                "spectral_timbre": 0.25,
                "loudness_envelope": 0.20,
                "groove_envelope": 0.20,
                "modulation_identity": 0.15,
                "stereo_motion": 0.10,
                "pitch_chroma": 0.05,
                "patch_control": 0.05,
            },
        },
        "global_mix_diff": global_diff,
        "sustain_shape_diagnostics": sustain_shape,
        "loudness_floor_diagnostics": loudness_floor,
        "groove_envelope_diagnostics": groove_envelope,
        "modulation_identity_diagnostics": modulation_identity,
        "track_scores": track_scores,
        "arrangement_preservation": preservation,
        "weakest_tracks": sorted(track_scores, key=lambda item: item["active_window_score"].get("final", 0.0))[:5],
    }


def acceptance_gate(previous: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {"accepted": True, "reasons": ["no previous best report"], "regressions": []}
    prev_scores = previous.get("scores", {})
    cand_scores = candidate.get("scores", {})
    prev_final = float(prev_scores.get("final", 0.0))
    cand_final = float(cand_scores.get("final", 0.0))
    reasons = [f"final score {cand_final:.4f} vs previous {prev_final:.4f}"]
    regressions = []
    strong_final_gain = cand_final >= prev_final + 0.05

    prev_sustain = previous.get("sustain_shape_diagnostics", {})
    cand_sustain = candidate.get("sustain_shape_diagnostics", {})
    prev_sustain_score = float(prev_scores.get("sustain_shape", prev_sustain.get("score", 0.0)))
    cand_sustain_score = float(cand_scores.get("sustain_shape", cand_sustain.get("score", 0.0)))
    if cand_sustain_score + 0.02 < prev_sustain_score:
        regressions.append(
            f"sustain_shape regressed from {prev_sustain_score:.4f} to {cand_sustain_score:.4f}"
        )

    prev_largest = float(prev_sustain.get("largest_abs_rms_error", 0.0))
    cand_largest = float(cand_sustain.get("largest_abs_rms_error", 0.0))
    if prev_largest > 0.0 and cand_largest > prev_largest * 1.20:
        regressions.append(
            f"largest 50ms RMS error grew from {prev_largest:.5f} to {cand_largest:.5f}"
        )

    if not strong_final_gain:
        cand_loudness = candidate.get("loudness_floor_diagnostics", {})
        cand_modulation = candidate.get("modulation_identity_diagnostics", {})
        cand_groove = candidate.get("groove_envelope_diagnostics", {})
        if float(cand_loudness.get("p10_rms_ratio", 1.0)) < 0.45:
            regressions.append(
                f"active-window loudness floor failed: p10 RMS ratio {float(cand_loudness.get('p10_rms_ratio', 0.0)):.2f}"
            )
        if int(cand_loudness.get("dropout_window_count", 0)) > 0:
            regressions.append(f"candidate has {int(cand_loudness.get('dropout_window_count', 0))} dropout windows below -12 dB")
        if bool(cand_modulation.get("candidate_too_slow", False)):
            ref = cand_modulation.get("reference", {})
            cand = cand_modulation.get("candidate", {})
            regressions.append(
                f"modulation rate is too slow: source {float(ref.get('rate_hz', 0.0)):.2f}Hz vs candidate {float(cand.get('rate_hz', 0.0)):.2f}Hz"
            )
        if bool(cand_modulation.get("candidate_too_shallow", False)):
            ref = cand_modulation.get("reference", {})
            cand = cand_modulation.get("candidate", {})
            regressions.append(
                f"modulation depth is too shallow: source {float(ref.get('depth', 0.0)):.2f} vs candidate {float(cand.get('depth', 0.0)):.2f}"
            )
        if float(cand_groove.get("score", 1.0)) < 0.55:
            regressions.append(f"beat/groove envelope gate failed: {float(cand_groove.get('score', 0.0)):.4f}")
    else:
        reasons.append("large final-score gain; absolute diagnostic gates remain reflected in the final score")

    secondary_regressions = []
    for key, tolerance in (
        ("track_isolation_proxy", 0.04),
        ("patch_control", 0.03),
        ("global_mix", 0.03),
        ("loudness_floor", 0.04),
        ("groove_envelope", 0.04),
        ("modulation_identity", 0.04),
    ):
        previous_value = float(prev_scores.get(key, 0.0))
        candidate_value = float(cand_scores.get(key, 0.0))
        if candidate_value + tolerance < previous_value:
            secondary_regressions.append(f"{key} regressed from {previous_value:.4f} to {candidate_value:.4f}")

    if strong_final_gain:
        reasons.extend(f"warning: {item}" for item in regressions + secondary_regressions)
        return {"accepted": True, "reasons": reasons, "regressions": []}

    regressions.extend(secondary_regressions)
    accepted = cand_final >= prev_final and not regressions
    if regressions:
        reasons.extend(regressions)
    return {"accepted": accepted, "reasons": reasons, "regressions": regressions}


def codex_patch_prompt(arrangement_path: Path, current_session_path: Path, critic_brief_path: Path, loss_report_path: Path, operations_output_path: Path, session_output_path: Path) -> str:
    return f"""
You are the Producer agent for an audio reconstruction loop.

Your goal is to produce the fixed composition so the rendered audio sounds exactly like the target source. The Critic has listened to the target and current render, studied the loss report, and written a production briefing. Start from that Critic briefing, form a production hypothesis, make concrete patch/mix changes through the patch operation function interface, run loss checks, and save patch operations for the harness to apply.

Read these files:
- Composition JSON: {arrangement_path}
- Current patch session JSON: {current_session_path}
- Critic briefing markdown: {critic_brief_path}
- Current loss report JSON: {loss_report_path}

Task: Call the patch operation helper functions to save changes to {operations_output_path}. The harness will apply those operations to {current_session_path} and write the resulting patch session to {session_output_path}.

Definitions:
- Composition is the fixed musical performance: track ids, roles, note events, velocities, timing, active ranges, tempo, and meter.
- Patch session is the sound design, effects, and mix state: synth settings, envelopes, filters, modulation, effects, sends, gain, pan, width, returns, master, and production_notes.
- Critic briefing is the production advice you should follow first.
- Loss report is the measured difference between the target audio and current rendered audio.

Workflow:
1. Read the Critic briefing first. Treat it as the starting point for your pass.
2. Read the current patch session and identify the exact fields you will edit.
3. Read the composition only to understand track roles and active timing. Do not solve composition in this step.
4. Write a short production hypothesis through `set_production_hypothesis(...)`. Phrase it as production decisions, for example: "The bass is too static and dry, so add subtle filter LFO and saturation while lowering the keys that mask it" or "The pad needs a wider saw-stack source with slower attack and longer reverb tail."
5. Choose one primary objective for this pass: envelope/automation, timbre/source, or space/modulation. If `exact_envelope_50ms`, `directional_delta`, `modulation`, or `sustain_shape` is among the weakest areas, make an envelope/automation-focused pass unless the Critic explicitly says a different blocker is more important.
6. Make the smallest set of high-impact patch/mix changes that follows that one objective and the Critic briefing. Avoid broad multi-domain rewrites that make it impossible to tell which edit helped.
7. Save each intended patch/mix edit by calling `save_patch_change(...)`. This is how changelog entries are created.
8. Apply your candidate operations to create a candidate session, then run loss checks on the full clip and on the weakest 50ms/beat windows called out by the loss report.
9. Record every loss command you ran and the score/loss result by calling `save_loss_trial(...)`.
10. Leave {operations_output_path} as the final patch operation file. Do not write the patch session directly.

Patch operation function interface:
- Do not hand-write {operations_output_path} as raw JSON.
- Do not edit {current_session_path} or {session_output_path} directly.
- Create/update {operations_output_path} only by calling these functions from Python:

```python
from pathlib import Path
from text2fx_gemini.midi_locked_patch import set_production_hypothesis, save_patch_change, save_loss_trial

ops = Path("{operations_output_path}")

set_production_hypothesis(
    ops,
    "The keys are too bright and static, so darken the filter and add slow filter motion while pushing the bass slightly forward.",
    critic_brief_used=Path("{critic_brief_path}"),
)

save_patch_change(
    ops,
    path="layers.<track_id>.filter.cutoff_start_hz",
    value=1800.0,
    track_id="<track_id>",
    change="lower starting filter cutoff",
    reason="Critic says this track is brighter than the target during its active windows.",
)

save_loss_trial(
    ops,
    command="python3 text2fx_gemini/midi_locked_patch.py score-session ...",
    score=0.42,
    loss=0.58,
    window_start=1.0,
    window_duration=0.75,
    notes="Targeted window improved after the filter move.",
)
```

Scoring commands you can run while iterating:
- Apply a candidate operation file to create a candidate session:
  `python3 text2fx_gemini/midi_locked_patch.py apply-patch-ops --session {current_session_path} --operations <candidate_ops.json> --output <candidate_session.json>`
- Full clip:
  `python3 text2fx_gemini/midi_locked_patch.py score-session --arrangement {arrangement_path} --session <candidate_session.json> --reference {loss_report_path.parent / "source_clip.wav"} --output <candidate_loss.json> --render-output <candidate_render.wav> --seconds 5`
- Specific time window inside the selected 5-second clip:
  `python3 text2fx_gemini/midi_locked_patch.py score-session --arrangement {arrangement_path} --session <candidate_session.json> --reference {loss_report_path.parent / "source_clip.wav"} --output <candidate_loss_window.json> --render-output <candidate_render_window.wav> --seconds 5 --window-start <seconds_from_clip_start> --window-duration <seconds>`

Use targeted windows for exact beats/times called out by the Critic, weak active windows in the loss report, or dense moments where one track dominates. Times are local to the 5-second clip, so `--window-start 1.0 --window-duration 0.75` scores 1.0s-1.75s of the selected clip.

Output contract:
- Your final answer can be brief, but the required artifact is {operations_output_path}.
- The required artifact must be created through the helper functions above.
- Do not return a full patch session.
- The operation payload created by the helper functions has schema `patchex.patch_ops.v1`.
- Supported operation types: `set`, `append`, and `extend`.
- Operation paths use layer ids, not numeric layer indexes: `layers.<track_id>.synth.waveform`, `layers.<track_id>.amp_envelope.attack`, `layers.<track_id>.filter.cutoff_points`, `layers.<track_id>.effects.reverb_mix`, `returns.space.decay`, `master.gain_db`, or `production_notes.hypothesis`.
- Each operation should include `track_id`, `path`, `value`, `change`, and `reason`.
- `loss_trials` entries should include the command, score/loss numbers if available, the time window if applicable, and a short note.

Hard boundary:
- Do not add, delete, reorder, retime, transpose, or change velocity of notes.
- Preserve every layer id from composition tracks.
- Preserve every layer's `notes` array exactly from the current patch session. The harness will repair note changes, but you should not rely on repair.
- Do not add/delete composition tracks. If a track should be silent or tucked back, change gain/mix, not notes.

Allowed patch and production changes:

Oscillator/source:
- Change `synth.waveform` toward sine, triangle, square, pulse, saw, supersaw-style saw, fifth-saw, organ-like, electric-piano-like, string-pad-like, bass-like, noise, air, transient, or drum-like source.
- Change `synth.wavetable` among renderer-friendly names such as sine, triangle, digital, formant, saw, saw_stack, square_saw, square, noise, or air.
- Adjust `synth.wavetable_position` to move between purer, brighter, buzzier, hollower, noisier, or more digital tones.
- Adjust `synth.blend`, `synth.warp`, `synth.fm_amount`, and `synth.fm_ratio`.
- Add or reduce `synth.sub_level`.
- Change `synth.voices`, `synth.detune_cents`, and `synth.stereo_spread`.
- Use `synth.vital_parameters` for named synth parameters when the Critic asks for a more specific patch character.

Pitch and tuning behavior:
- Add or reduce pitch-like character using FM, detune, unison, wavetable position, or subtle LFOs.
- Add or reduce vibrato with `modulation.lfos` targeting pitch-like synth parameters when supported by the patch representation.
- Tighten pitch stability by reducing detune/FM/LFO depth.
- Make a sound thicker with octave/sub support rather than changing MIDI notes.

Envelope and articulation:
- Adjust `amp_envelope.attack`, `decay`, `sustain`, and `release`.
- Make notes pluckier, sharper, softer, more legato, more gated, more sustained, punchier, smoother, shorter, or longer.
- Shorten releases that smear rhythm.
- Lengthen releases when the target has more sustain or ambience.
- Use `gain_points` for clip-level fades, swells, ducking, pulsing, or phrase emphasis.

Filter and brightness:
- Adjust `filter.cutoff_start_hz`, `filter.cutoff_end_hz`, `filter.cutoff_points`, `filter.resonance`, and `filter.drive`.
- Make a track brighter, darker, warmer, thinner, more muffled, more open, more nasal, more resonant, or more driven.
- Use moving `filter.cutoff_points` for sweeps or evolving brightness.
- Use `modulation.lfos` targeting `filter.cutoff_hz` for periodic brightness motion.

Amplitude, dynamics, and sidechain-like motion:
- Adjust layer `gain_db`.
- Adjust `gain_points` for automation, rhythmic pumping, phrase-level fades, or sidechain-like ducking.
- Use `modulation.gate_points` for amplitude gating.
- Use `modulation.lfos` targeting gain/amp/volume for tremolo or pumping.
- Adjust `effects.compression_mix`, `compression_threshold_db`, `compression_ratio`, `compression_attack`, and `compression_release`.
- Make tracks more forward, tucked back, even, dynamic, aggressive, soft, punchy, or controlled.
- To approximate sidechain X to Y, duck the masked track with gain automation or gain LFO during the dominant track's active windows.

Modulation and motion:
- Add, remove, or change `modulation.lfos`.
- Adjust LFO `target`, `shape`, `rate_hz`, `depth`, `phase`, `amount`, and `center`.
- Add or reduce tremolo, vibrato-like motion, filter LFO, amplitude LFO, pan LFO, width LFO, or slow pad evolution.
- Sync motion by choosing rates that match the clip feel, or make it freer if the target is less grid-locked.
- Remove motion when the target is steadier.

Effects and space:
- Adjust `effects.reverb_mix`, `return_send`, and session `returns`.
- Adjust return `gain_db`, `decay`, and `width`.
- Adjust `effects.delay_mix`, `delay_time`, `phaser_mix`, and `chorus_mix`.
- Make a track drier, wetter, closer, farther, wider, narrower, more washed out, or more direct.
- Shorten reverb tails if the mix is blurry.
- Increase space if the target has more late energy or stereo spread.

Saturation, distortion, EQ, and noise:
- Adjust `effects.saturation`.
- Adjust `filter.drive`.
- Adjust `effects.low_gain_db`, `mid_gain_db`, and `high_gain_db`.
- Make a sound cleaner, dirtier, warmer, harsher, softer, brighter, duller, boomier, thinner, less muddy, or more harmonically dense.
- Add noise-like character with waveform/wavetable choices instead of changing notes.

Stereo and placement:
- Adjust layer `pan` from -1.0 left to 1.0 right.
- Adjust layer `width`.
- Adjust `synth.stereo_spread`.
- Adjust master `width`.
- Keep bass and low-frequency fundamentals narrower when needed.
- Widen pads, keys, strings, effects, or hooks when the target image is broader.
- Narrow tracks that sound too diffuse.

Mix and master:
- Adjust layer gains to rebalance bass, keys, lead, strings, drums, and effects.
- Push the main hook forward or tuck supporting material behind it.
- Reduce masking between bass and keys.
- Adjust `master.gain_db` if the render is globally too loud or quiet.
- Adjust `master.width` if the full image is too narrow or too wide.
- Prefer track-level changes before broad master changes unless the Critic identifies a global issue.

Drums and percussion, if present:
- Adjust drum layer gain, envelope, noise/source character, saturation, EQ, reverb, compression, and width.
- Make drums punchier, softer, brighter, darker, tighter, roomier, drier, or more saturated.
- Add or reduce transient emphasis with envelope attack/decay and compression.

Decision rules:
- Follow the Critic briefing unless the loss report clearly contradicts it.
- Prefer high-impact production moves over tiny random parameter changes.
- Make changes that are coherent as a production hypothesis, not isolated parameter noise.
- For sustained locked arrangements, do not create `gain_points` that alternate high/low across 0.5s segments unless the reference segment envelope also alternates. Prefer smooth holds, ramps, or one intentional duck/recovery.
- Reject your own candidate if it improves full score but worsens the weakest local envelope windows, `sustain_shape`, `exact_envelope_50ms`, or `directional_delta` versus the current report.
- Treat low `f0_contour` cautiously when `pitch_chroma` is high and arrangement preservation is perfect; do not infer note or chord edits from that metric.
- In your recorded loss trial notes, include the before/after effect on weakest components such as `harmonic_noise`, `modulation`, `exact_envelope_50ms`, `directional_delta`, `band_envelope_by_time`, `stereo_width`, and `sustain_shape` when available.
- If you try multiple candidates, write the operations for the best one to {operations_output_path} and record the rejected trials in `loss_trials`.
- If a targeted window improves but full-clip loss worsens, choose the version that best supports the stated production goal and explain that tradeoff in `production_notes`.
"""


def codex_critic_prompt(
    composition_path: Path,
    patch_session_path: Path,
    target_audio_path: Path,
    current_render_path: Path,
    loss_report_path: Path,
    critic_brief_path: Path,
) -> str:
    return f"""
You are the Critic agent for an audio reconstruction loop.

Your goal is to help produce this composition so it sounds exactly like the target source. The music Producer has made a pass at it, and you are an expert musician, sound designer, and mix engineer giving advice on what production steps to take next. You will write this guidance to a file, and we will pass that file to the Producer.

You can listen to the audio files themselves. We have also done an in-depth audio analysis of the differences between the rendered audio and the target audio. Reason over both the audio and the analysis.

Read and use these inputs:
- Composition: {composition_path}
- Current patch session: {patch_session_path}
- Target audio: {target_audio_path}
- Current rendered audio: {current_render_path}
- Loss report: {loss_report_path}

Definitions:
- Composition is the fixed musical performance: tracks, roles, note events, velocities, timing, active ranges, tempo, and meter.
- Patch session is the current sound design, effects, and mix attempt: synth settings, envelopes, filters, modulation, effects, sends, gain, pan, width, returns, and master settings.
- Target audio is the reference clip we are trying to reconstruct.
- Current rendered audio is what the current patch session produces from the fixed composition.
- Loss report contains measured inaccuracies between the target audio and current rendered audio.

Task:
Write a concise markdown briefing to:
{critic_brief_path}

Your briefing should tell the Producer what to change next in the patch session so the rendered audio gets closer to the target source.

Do this:
1. Listen to the target audio.
2. Listen to the current rendered audio.
3. Compare what you hear against the loss report.
4. Read the composition so you understand which tracks are active and what role each track plays.
5. Read the current patch session so you know what synth, effects, and mix choices are already being used.
6. Identify the most important inaccuracies blocking a better reconstruction.
7. Recommend concrete production changes for the next Producer pass.

Use all available evidence:
- global mix score
- per-track active-window scores
- isolation proxy scores
- weakest tracks
- envelope/ADSR diagnostics
- brightness/filter diagnostics
- modulation diagnostics
- stereo/space diagnostics
- saturation/noise diagnostics
- arrangement preservation diagnostics
- sustain_shape diagnostics, especially largest 50ms RMS errors and pumping/reversal counts
- audible differences between target audio and current rendered audio

Be specific. For each recommendation, name the track id, the problem, and the exact production move.

Possible production changes include:

Oscillator/source:
- Change a track synth toward sine, triangle, square, pulse, saw, supersaw, fifth-saw, organ, electric piano, string pad, bass, noise, or drum-like source.
- Add, remove, or rebalance oscillator layers.
- Detune oscillators more or less.
- Add octave doubling, sub oscillator, fifth, or unison.
- Change pulse width or waveform blend.
- Make the source more pure, buzzy, hollow, nasal, glassy, warm, thin, thick, metallic, or noisy.

Pitch and tuning behavior:
- Add or reduce pitch drift.
- Add or reduce pitch envelope.
- Add or reduce portamento/glide.
- Change vibrato rate or depth.
- Tighten or loosen pitch stability.

Envelope and articulation:
- Increase or decrease attack.
- Increase or decrease decay.
- Increase or decrease sustain.
- Increase or decrease release.
- Make notes more plucky, more legato, more gated, more sustained, softer, sharper, punchier, or smoother.
- Shorten long tails that smear the rhythm.
- Lengthen releases if the target has more sustain or ambience.

Filter and brightness:
- Raise or lower filter cutoff.
- Increase or decrease resonance.
- Use low-pass, high-pass, band-pass, notch, or shelving behavior.
- Add or reduce filter envelope amount.
- Make the sound brighter, darker, thinner, warmer, more muffled, more open, more nasal, or more resonant.
- Change filter attack, decay, sustain, or release if the brightness evolves incorrectly.

Amplitude and dynamics:
- Raise or lower track gain.
- Add or reduce velocity sensitivity.
- Add or reduce compression.
- Add or reduce transient punch.
- Make a track more forward, more tucked back, more even, more dynamic, more aggressive, or softer.
- Change gain automation if a part should grow, fade, pulse, or duck.

Modulation and motion:
- Add or reduce tremolo.
- Add or reduce vibrato.
- Add or reduce filter LFO.
- Add or reduce amplitude LFO.
- Change modulation rate.
- Change modulation depth.
- Sync modulation to tempo or make it freer.
- Add slow evolving motion for pads.
- Remove motion if the target is steadier.

Effects and space:
- Add or reduce reverb.
- Change reverb size, decay, pre-delay, damping, or wet level.
- Add or reduce delay.
- Change delay time, feedback, filtering, stereo spread, or wet level.
- Add or reduce chorus, ensemble, phaser, flanger, or widening.
- Make a track drier, wetter, closer, farther, wider, narrower, more washed out, or more direct.
- Shorten reverb tails if the mix is blurry.
- Increase space if the target has more late energy or width.

Saturation, distortion, and noise:
- Add or reduce saturation.
- Add or reduce drive, clipping, bitcrush, tape color, harmonic density, or grit.
- Add or reduce noise floor, breath, hiss, or texture.
- Make a sound cleaner, dirtier, warmer, harsher, softer, or more compressed.

Stereo and placement:
- Move a track left, right, or center.
- Increase or decrease stereo width.
- Collapse low-frequency tracks toward mono.
- Widen pads, keys, or effects if the target is broader.
- Narrow tracks that are too diffuse.
- Adjust pan or width if the stereo image does not match.

Mix balance:
- Raise or lower specific tracks.
- Rebalance bass, chords, lead, strings, drums, and effects.
- Push the main hook forward.
- Tuck supporting parts behind the lead.
- Reduce masking between bass and keys.
- Adjust master gain if the render is globally too loud or quiet.
- Adjust EQ balance if the whole render is too bright, dull, boomy, thin, harsh, or muddy.

Drums and percussion, if present:
- Change drum level.
- Change kick/snare/hat balance.
- Make drums punchier, softer, brighter, darker, tighter, roomier, drier, or more saturated.
- Add or reduce transient emphasis.
- Add or reduce drum room/reverb.
- Adjust cymbal/hat brightness and decay.

Prioritization:
- Start with the changes most likely to improve the full rendered audio.
- Prefer high-impact track and mix changes over tiny parameter tweaks.
- If arrangement preservation is perfect and the target is sustained, rank envelope automation before timbre when whole-clip RMS is close but 50ms/local windows are extreme.
- When `pitch_chroma` is high but `f0_contour` is low on a dense sustained pad, describe it as a tone/source or octave-support uncertainty, not as evidence to change notes.
- Provide one primary objective for the next Producer pass and list which edit families should be avoided in that pass.
- If the loss report and your listening disagree, explain the disagreement and choose the more musically plausible action.
- If the full mix does not provide enough evidence to isolate a track, say so and recommend a conservative change.

Write markdown only. Keep it practical and direct. The Producer should be able to act from the briefing without guessing.
"""


def load_codex_json_artifact(json_output_path: Path, answer_path: Path) -> dict[str, Any]:
    sources: list[tuple[str, Path]] = []
    if json_output_path.exists() and json_output_path.read_text().strip():
        sources.append(("artifact", json_output_path))
    if answer_path.exists() and answer_path.read_text().strip():
        sources.append(("answer", answer_path))
    errors: list[str] = []
    for label, path in sources:
        try:
            payload = extract_json_object(path.read_text())
        except Exception as exc:
            errors.append(f"{label} {path}: {exc}")
            continue
        json_output_path.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    detail = "; ".join(errors) if errors else "no non-empty JSON output was written"
    raise ValueError(f"Codex did not produce valid JSON for {json_output_path}: {detail}")


def stream_codex_exec(agent: str, prompt: str, answer_path: Path, timeout: int) -> int:
    process = subprocess.Popen(
        [CODEX_PATH, "exec", "--skip-git-repo-check", "--output-last-message", str(answer_path), "-C", str(Path.cwd()), "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(prompt)
    process.stdin.close()
    started = time.monotonic()
    for line in process.stdout:
        print(f"codex_log agent={agent} {line.rstrip()}", flush=True)
        if time.monotonic() - started > timeout:
            process.kill()
            print(f"codex_timeout agent={agent} seconds={timeout}", flush=True)
            return 124
    try:
        return process.wait(timeout=max(1, int(timeout - (time.monotonic() - started))))
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"codex_timeout agent={agent} seconds={timeout}", flush=True)
        return 124


def recover_codex_json(agent: str, json_output_path: Path, answer_path: Path, returncode: int) -> dict[str, Any]:
    try:
        payload = load_codex_json_artifact(json_output_path, answer_path)
    except Exception as exc:
        raise RuntimeError(f"Codex agent {agent} failed with return code {returncode} and no valid JSON artifact could be recovered: {exc}") from exc
    print(f"codex_recovered agent={agent} artifact={json_output_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
    return payload


def run_codex_patch(agent: str, prompt: str, output_dir: Path, answer_path: Path, timeout: int = 600) -> dict[str, Any]:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    prompt_path.write_text(prompt)
    print(f"codex_request agent={agent} path={prompt_path}", flush=True)
    print(f"codex_start agent={agent}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    codex_answer_path = output_dir / f"codex_{agent}_answer.txt"
    returncode = stream_codex_exec(
        agent,
        prompt,
        codex_answer_path,
        timeout,
    )
    if returncode != 0:
        return recover_codex_json(agent, answer_path, codex_answer_path, returncode)
    print(f"codex_done agent={agent} answer_path={codex_answer_path}", flush=True)
    print(f"codex_response agent={agent} path={codex_answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={codex_answer_path}", flush=True)
    return load_codex_json_artifact(answer_path, codex_answer_path)


def run_codex_json_agent(agent: str, prompt: str, output_dir: Path, json_output_path: Path, timeout: int = 600) -> dict[str, Any]:
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
    print(f"codex_request agent={agent} path={prompt_path}", flush=True)
    print(f"codex_start agent={agent}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    returncode = stream_codex_exec(
        agent,
        prompt,
        answer_path,
        timeout,
    )
    if returncode != 0:
        return recover_codex_json(agent, json_output_path, answer_path, returncode)
    print(f"codex_done agent={agent} answer_path={answer_path}", flush=True)
    print(f"codex_response agent={agent} path={answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
    return load_codex_json_artifact(json_output_path, answer_path)


def run_codex_markdown_agent(agent: str, prompt: str, output_dir: Path, markdown_output_path: Path, timeout: int = 600) -> str:
    if not Path(CODEX_PATH).exists():
        raise FileNotFoundError(f"Codex command not found: {CODEX_PATH}")
    prompt_path = output_dir / f"codex_{agent}_prompt.txt"
    answer_path = output_dir / f"codex_{agent}_answer.txt"
    prompt = (
        f"{prompt.rstrip()}\n\n"
        "File-driven orchestration requirement:\n"
        f"- Write the markdown briefing at: {markdown_output_path}\n"
        "- The orchestrator will pass that markdown file to the Producer after this run.\n"
    )
    prompt_path.write_text(prompt)
    print(f"codex_request agent={agent} path={prompt_path}", flush=True)
    print(f"codex_start agent={agent}", flush=True)
    print(f"trace_file agent={agent} role=prompt path={prompt_path}", flush=True)
    returncode = stream_codex_exec(
        agent,
        prompt,
        answer_path,
        timeout,
    )
    if returncode != 0:
        if markdown_output_path.exists() and markdown_output_path.read_text().strip():
            briefing = markdown_output_path.read_text()
            print(f"codex_recovered agent={agent} artifact={markdown_output_path}", flush=True)
            print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
            return briefing
        if answer_path.exists() and answer_path.read_text().strip():
            briefing = answer_path.read_text()
            markdown_output_path.write_text(briefing.rstrip() + "\n")
            print(f"codex_recovered agent={agent} artifact={markdown_output_path}", flush=True)
            print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
            return briefing
        raise RuntimeError(f"Codex agent {agent} failed with return code {returncode} and no markdown artifact could be recovered.")
    print(f"codex_done agent={agent} answer_path={answer_path}", flush=True)
    print(f"codex_response agent={agent} path={answer_path}", flush=True)
    print(f"trace_file agent={agent} role=answer path={answer_path}", flush=True)
    if markdown_output_path.exists() and markdown_output_path.read_text().strip():
        briefing = markdown_output_path.read_text()
    else:
        briefing = answer_path.read_text()
        markdown_output_path.write_text(briefing.rstrip() + "\n")
    return briefing


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
    if args.window_start is not None or args.window_duration is not None:
        window_start = float(args.window_start or 0.0)
        max_duration = max(0.0, reference_audio.shape[-1] / sr - window_start)
        window_duration = float(args.window_duration if args.window_duration is not None else max_duration)
        window_duration = max(0.01, min(window_duration, max_duration if max_duration > 0 else window_duration))
        start_i = max(0, int(round(window_start * sr)))
        end_i = min(reference_audio.shape[-1], start_i + int(round(window_duration * sr)))
        reference_audio = reference_audio[:, start_i:end_i]
        arrangement = slice_arrangement(arrangement, window_start, window_duration)
        session = slice_patch_session(session, window_start, window_duration)
    report = score_midi_locked(arrangement, session, reference_audio, sr)
    write_json(args.output, report)
    if args.render_output:
        sf.write(args.render_output, render_session(session).T, sr)
        print(f"wrote {args.render_output}")
    print(f"wrote {args.output}")
    return 0


def command_apply_patch_ops(args: argparse.Namespace) -> int:
    session = json.loads(args.session.read_text())
    operations = json.loads(args.operations.read_text())
    updated, report = apply_patch_operations(session, operations)
    write_json(args.output, updated)
    if args.report:
        write_json(args.report, report)
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
    reference_audio, sr = load_audio(args.reference, args.seconds)
    arrangement = slice_arrangement(full_arrangement, args.clip_start, args.seconds) if args.clip_start is not None else full_arrangement
    write_json(arrangement_path, arrangement)
    print(f"trace_file agent=analyzer role=arrangement path={arrangement_path}", flush=True)
    session_sample_rate = sr if int(args.sample_rate) == 44100 else int(args.sample_rate)
    if args.initial_session:
        session = json.loads(args.initial_session.read_text())
        session["sample_rate"] = session_sample_rate
        session["duration"] = float(args.seconds)
        session, initial_lock_report = enforce_arrangement_lock(arrangement, session)
        write_json(args.output_dir / "initial_session_lock.json", initial_lock_report)
    else:
        session = neutral_session(arrangement, session_sample_rate, args.seconds)
    write_json(current_session_path, session)
    print(f"trace_file agent=session role=current path={current_session_path}", flush=True)
    source_clip_path = args.output_dir / "source_clip.wav"
    sf.write(source_clip_path, reference_audio.T, sr)
    print(f"trace_file agent=analyzer role=source_clip path={source_clip_path}", flush=True)
    current_render_path = args.output_dir / "current_render_step_initial.wav"
    sf.write(current_render_path, render_session(session).T, sr)
    print(f"trace_file agent=loss role=winner_render path={current_render_path}", flush=True)
    current_loss_report = score_midi_locked(arrangement, session, reference_audio, sr)
    current_loss_report_path = write_json(args.output_dir / "loss_report_step_initial.json", current_loss_report)
    print(f"trace_file agent=loss role=audio_diff path={current_loss_report_path}", flush=True)
    history: list[dict[str, Any]] = []
    best_report: dict[str, Any] | None = current_loss_report
    best_session = session
    best_render_path: Path | None = current_render_path
    for step in range(args.steps):
        lock_report: dict[str, Any] = {}
        print(f"agent_stage residual_critic step={step}", flush=True)
        critic_brief_path = args.output_dir / f"critic_brief_step_{step:02d}.md"
        if args.neutral_only:
            critic_brief = "# Critic Brief\n\nNeutral-session smoke run. Render the current session without Codex patch changes.\n"
            critic_brief_path.write_text(critic_brief)
        else:
            critic_brief = run_codex_markdown_agent(
                f"residual_critic_step_{step:02d}",
                codex_critic_prompt(
                    arrangement_path,
                    current_session_path,
                    source_clip_path,
                    current_render_path,
                    current_loss_report_path,
                    critic_brief_path,
                ),
                args.output_dir,
                critic_brief_path,
                args.timeout,
            )
        print(f"trace_file agent=residual_critic_step_{step:02d} role=recommendation path={critic_brief_path}", flush=True)
        print(f"agent_stage producer step={step}", flush=True)
        proposal_path = args.output_dir / f"patch_session_step_{step:02d}.json"
        operations_path = args.output_dir / f"patch_ops_step_{step:02d}.json"
        if step == 0 and args.neutral_only:
            proposal = session
            proposal, lock_report = enforce_arrangement_lock(arrangement, proposal)
            write_json(proposal_path, proposal)
            write_json(operations_path, {"schema": "patchex.patch_ops.v1", "hypothesis": "neutral-session smoke run", "operations": [], "loss_trials": []})
        else:
            prompt = codex_patch_prompt(arrangement_path, current_session_path, critic_brief_path, current_loss_report_path, operations_path, proposal_path)
            raw_operations = run_codex_patch(f"producer_step_{step:02d}", prompt, args.output_dir, operations_path, args.timeout)
            raw_proposal, operations_report = apply_patch_operations(session, raw_operations)
            write_json(args.output_dir / f"patch_ops_applied_step_{step:02d}.json", operations_report)
            proposal, lock_report = enforce_arrangement_lock(arrangement, raw_proposal)
            write_json(proposal_path, proposal)
            write_json(args.output_dir / f"arrangement_lock_step_{step:02d}.json", lock_report)
        report = score_midi_locked(arrangement, proposal, reference_audio, sr)
        report_path = write_json(args.output_dir / f"patch_report_step_{step:02d}.json", report)
        print(f"trace_file agent=loss step={step} role=audio_diff path={report_path}", flush=True)
        render_path = args.output_dir / f"patch_render_step_{step:02d}.wav"
        sf.write(render_path, render_session(proposal).T, sr)
        print(f"trace_file agent=loss step={step} role=winner_render path={render_path}", flush=True)
        gate_report = acceptance_gate(best_report, report)
        accepted = bool(gate_report["accepted"])
        if accepted:
            best_report = report
            best_session = proposal
            best_render_path = render_path
        session = best_session if best_session is not None else proposal
        current_session_path = write_json(args.output_dir / "patch_session_current.json", session)
        current_loss_report_path = report_path if accepted else current_loss_report_path
        current_render_path = render_path if accepted else current_render_path
        history_item = {
            "stage": "midi_locked_patch",
            "step": step,
            "accepted": accepted,
            "winner": "midi_locked_patch",
            "audio_path": str(render_path),
            "audio_diff_path": str(report_path),
            "proposal_session_path": str(proposal_path),
            "patch_ops_path": str(operations_path),
            "scores": report["scores"],
            "best_scores": (best_report or report)["scores"],
            "layers": [{"id": layer.get("id"), "role": layer.get("role")} for layer in proposal.get("layers", [])],
            "arrangement_preservation": report["arrangement_preservation"],
            "arrangement_lock_report": lock_report or {"before": report["arrangement_preservation"], "after": report["arrangement_preservation"]},
            "acceptance_gate": gate_report,
            "critic_brief_path": str(critic_brief_path),
            "critic_brief": critic_brief,
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
    write_json(
        args.output_dir / "midi_locked_patch_report.json",
        {
            "arrangement_path": str(arrangement_path),
            "current_session_path": str(current_session_path),
            "current_loss_report_path": str(current_loss_report_path),
            "current_render_path": str(current_render_path),
            "best_scores": report_payload["best_scores"],
        },
    )
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
    p_score.add_argument("--window-start", type=float)
    p_score.add_argument("--window-duration", type=float)
    p_score.set_defaults(func=command_score)

    p_apply_ops = sub.add_parser("apply-patch-ops")
    p_apply_ops.add_argument("--session", required=True, type=Path)
    p_apply_ops.add_argument("--operations", required=True, type=Path)
    p_apply_ops.add_argument("--output", required=True, type=Path)
    p_apply_ops.add_argument("--report", type=Path)
    p_apply_ops.set_defaults(func=command_apply_patch_ops)

    p_run = sub.add_parser("run")
    p_run.add_argument("--midi", required=True, type=Path)
    p_run.add_argument("--role-map", type=Path)
    p_run.add_argument("--reference", required=True, type=Path)
    p_run.add_argument("--output-dir", required=True, type=Path)
    p_run.add_argument("--sample-rate", type=int, default=44100)
    p_run.add_argument("--seconds", type=float, default=5.0)
    p_run.add_argument("--clip-start", type=float)
    p_run.add_argument("--steps", type=int, default=1)
    p_run.add_argument("--timeout", type=int, default=600)
    p_run.add_argument("--neutral-only", action="store_true")
    p_run.add_argument("--initial-session", type=Path)
    p_run.set_defaults(func=command_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
