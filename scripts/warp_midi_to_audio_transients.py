#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text2fx_gemini.midi_locked_patch import parse_midi  # noqa: E402


def read_vlq(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def write_vlq(value: int) -> bytes:
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def seconds_to_ticks(seconds: float, ticks_per_beat: int, mpq: int) -> int:
    return max(0, int(round(seconds * 1_000_000.0 * ticks_per_beat / mpq)))


def ticks_to_seconds(ticks: int, ticks_per_beat: int, mpq: int) -> float:
    return ticks * mpq / ticks_per_beat / 1_000_000.0


def audio_transients(audio_path: Path, threshold: float = 0.10) -> list[dict]:
    audio, sr = sf.read(audio_path, always_2d=True)
    samples = audio.mean(axis=1).astype(np.float64)
    frame = 1024
    hop = 128
    window = np.hanning(frame)
    values = []
    times = []
    for start in range(0, samples.size - frame + 1, hop):
        segment = samples[start:start + frame] * window
        values.append(math.log(float(np.sqrt(np.mean(segment * segment))) + 1e-12))
        times.append((start + frame / 2) / sr)
    values = np.asarray(values)
    flux = np.maximum(0.0, np.diff(values, prepend=values[0]))
    if flux.size >= 5:
        flux = np.convolve(flux, np.ones(3) / 3.0, mode="same")
    norm = (flux - flux.min()) / (flux.max() - flux.min() + 1e-12)
    candidates = []
    for i in range(1, norm.size - 1):
        if norm[i] > norm[i - 1] and norm[i] >= norm[i + 1] and norm[i] >= threshold:
            candidates.append((float(norm[i]), float(times[i])))
    selected = []
    for strength, time in sorted(candidates, reverse=True):
        if all(abs(time - item["time"]) >= 0.12 for item in selected):
            selected.append({"time": time, "strength": strength})
    return sorted(selected, key=lambda item: item["time"])


def midi_clusters(arrangement: dict, min_count: int = 3) -> list[dict]:
    starts = []
    for track in arrangement.get("tracks", []):
        for note in track.get("notes", []):
            starts.append((float(note["start"]), float(note.get("velocity", 0.7)), str(track["id"])))
    clusters = []
    for time, velocity, track_id in sorted(starts):
        if not clusters or time - clusters[-1]["last"] > 0.025:
            clusters.append({"times": [time], "velocities": [velocity], "tracks": {track_id}, "last": time})
        else:
            clusters[-1]["times"].append(time)
            clusters[-1]["velocities"].append(velocity)
            clusters[-1]["tracks"].add(track_id)
            clusters[-1]["last"] = time
    output = []
    for cluster in clusters:
        if len(cluster["times"]) < min_count:
            continue
        output.append(
            {
                "time": float(np.mean(cluster["times"])),
                "count": len(cluster["times"]),
                "strength": float(sum(cluster["velocities"])),
                "tracks": sorted(cluster["tracks"]),
            }
        )
    return output


def match_transients(arrangement: dict, audio_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    audio = audio_transients(audio_path)
    midi = midi_clusters(arrangement)
    audio_times = np.asarray([item["time"] for item in audio])
    matches = []
    for cluster in midi:
        if audio_times.size == 0:
            break
        index = int(np.argmin(np.abs(audio_times - cluster["time"])))
        offset = float(audio_times[index] - cluster["time"])
        if abs(offset) <= 0.80:
            item = audio[index]
            matches.append(
                {
                    "midi_time": cluster["time"],
                    "audio_time": item["time"],
                    "offset_seconds": offset,
                    "midi_count": cluster["count"],
                    "midi_strength": cluster["strength"],
                    "audio_strength": item["strength"],
                    "tracks": cluster["tracks"],
                }
            )
    return audio, midi, matches


def section_anchors(matches: list[dict], duration: float, window: float = 10.0) -> list[tuple[float, float]]:
    strong = [m for m in matches if m["audio_strength"] >= 0.12 and abs(m["offset_seconds"]) <= 0.65]
    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    for start in np.arange(0.0, duration, window):
        local = [m["offset_seconds"] for m in strong if start <= m["midi_time"] < start + window]
        if len(local) < 2:
            continue
        center = min(duration, start + window / 2.0)
        median = float(np.median(local))
        anchors.append((center, median))
    anchors.append((duration, anchors[-1][1] if anchors else 0.0))
    deduped = []
    for time, offset in sorted(anchors):
        if deduped and abs(time - deduped[-1][0]) < 1e-6:
            deduped[-1] = (time, offset)
        else:
            deduped.append((time, offset))
    return deduped


def interpolate_offset(seconds: float, anchors: list[tuple[float, float]]) -> float:
    if seconds <= anchors[0][0]:
        return anchors[0][1]
    for (left_t, left_o), (right_t, right_o) in zip(anchors, anchors[1:]):
        if seconds <= right_t:
            span = max(1e-9, right_t - left_t)
            amount = (seconds - left_t) / span
            return left_o + (right_o - left_o) * amount
    return anchors[-1][1]


def warp_midi_file(input_path: Path, output_path: Path, anchors: list[tuple[float, float]]) -> None:
    data = input_path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{input_path} is not a MIDI file")
    header_len = struct.unpack(">I", data[4:8])[0]
    header = data[: 8 + header_len]
    _, _, ticks_per_beat = struct.unpack(">HHH", data[8 : 8 + 6])
    mpq = 500000
    pos = 8 + header_len
    tracks = []
    while pos < len(data):
        if data[pos : pos + 4] != b"MTrk":
            raise ValueError(f"Expected MTrk at byte {pos}")
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        track_data = data[pos + 8 : pos + 8 + length]
        pos += 8 + length
        events = []
        index = 0
        running_status = None
        absolute = 0
        while index < len(track_data):
            delta, event_start = read_vlq(track_data, index)
            absolute += delta
            status_or_data = track_data[event_start]
            if status_or_data == 0xFF:
                meta_type = track_data[event_start + 1]
                meta_len, payload_start = read_vlq(track_data, event_start + 2)
                event_bytes = track_data[event_start : payload_start + meta_len]
                index = payload_start + meta_len
                running_status = None
                if meta_type == 0x51 and meta_len == 3 and absolute == 0:
                    mpq = int.from_bytes(track_data[payload_start : payload_start + 3], "big")
                new_absolute = absolute
            elif status_or_data in (0xF0, 0xF7):
                syx_len, payload_start = read_vlq(track_data, event_start + 1)
                event_bytes = track_data[event_start : payload_start + syx_len]
                index = payload_start + syx_len
                running_status = None
                new_absolute = absolute
            else:
                if status_or_data & 0x80:
                    status = status_or_data
                    running_status = status
                    payload_start = event_start + 1
                    event_len = 1 if 0xC0 <= status <= 0xDF else 2
                    event_bytes = track_data[event_start : payload_start + event_len]
                    index = payload_start + event_len
                else:
                    if running_status is None:
                        raise ValueError("Running status without previous status")
                    status = running_status
                    event_len = 1 if 0xC0 <= status <= 0xDF else 2
                    event_bytes = track_data[event_start : event_start + event_len]
                    index = event_start + event_len
                old_seconds = ticks_to_seconds(absolute, ticks_per_beat, mpq)
                new_seconds = max(0.0, old_seconds + interpolate_offset(old_seconds, anchors))
                new_absolute = seconds_to_ticks(new_seconds, ticks_per_beat, mpq)
            events.append((new_absolute, event_bytes))
        events.sort(key=lambda item: item[0])
        rebuilt = bytearray()
        previous = 0
        for new_absolute, event_bytes in events:
            new_absolute = max(previous, int(new_absolute))
            rebuilt += write_vlq(new_absolute - previous)
            rebuilt += event_bytes
            previous = new_absolute
        tracks.append(b"MTrk" + struct.pack(">I", len(rebuilt)) + bytes(rebuilt))
    output_path.write_bytes(header + b"".join(tracks))


def write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Warp a song MIDI file to whole-song audio transient anchors.")
    parser.add_argument("--song-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    song_dir = args.song_dir
    manifest = json.loads((song_dir / "song.json").read_text())
    midi_path = song_dir / manifest["midi"]
    audio_path = song_dir / manifest["audio"]
    arrangement_path = song_dir / manifest.get("arrangement", "arrangement.json")
    arrangement = parse_midi(midi_path, song_dir / "song.json")
    audio, midi, matches = match_transients(arrangement, audio_path)
    anchors = section_anchors(matches, float(arrangement["duration"]))
    report = {
        "song": manifest.get("id", song_dir.name),
        "method": "section-median transient warp",
        "audio_transient_count": len(audio),
        "midi_transient_cluster_count": len(midi),
        "matched_count": len(matches),
        "anchors": [{"time": time, "offset_seconds": offset} for time, offset in anchors],
        "matches": matches,
    }
    write_report(song_dir / "warp_map.json", report)
    if args.apply:
        backup = midi_path.with_suffix(".prewarp.mid")
        if not backup.exists():
            backup.write_bytes(midi_path.read_bytes())
        tmp = midi_path.with_suffix(".warped.mid")
        warp_midi_file(midi_path, tmp, anchors)
        tmp.replace(midi_path)
        warped_arrangement = parse_midi(midi_path, song_dir / "song.json")
        write_report(arrangement_path, warped_arrangement)
    print(json.dumps({"anchors": len(anchors), "matches": len(matches), "applied": args.apply}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
