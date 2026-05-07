from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class AudioScore:
    final: float
    multi_resolution_spectral: float
    mel_spectrogram: float
    a_weighted_spectral: float
    envelope: float
    segment_envelope: float
    late_energy_ratio: float
    sustain_coverage: float
    frontload_balance: float
    band_envelope_by_time: float
    beat_grid_mel: float
    beat_grid_band: float
    beat_grid_envelope: float
    beat_grid_mid_side: float
    pitch_chroma: float
    f0_contour: float
    spectral_motion: float
    centroid_trajectory: float
    spectral_features: float
    transient_onset: float
    onset_count: float
    onset_timing: float
    stereo_width: float
    modulation: float
    harmonic_noise: float
    cepstral: float
    embedding: float
    codec_latent: float
    exact_envelope_50ms: float
    exact_band_50ms: float
    modulation_periodicity: float
    modulation_rate: float
    modulation_depth: float
    directional_delta: float
    transient_classification: float


def score_to_json(score: AudioScore) -> dict[str, float]:
    return asdict(score)


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = min(reference.shape[-1], candidate.shape[-1])
    return reference[..., :size], candidate[..., :size]


def safe_exp_distance(distance: float) -> float:
    return float(math.exp(-max(0.0, distance)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def stft_mag(y: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    frames = []
    window = np.hanning(frame)
    for start in range(0, y.size - frame + 1, hop):
        frames.append(np.abs(np.fft.rfft(y[start : start + frame] * window)) + 1e-8)
    return np.asarray(frames) if frames else np.zeros((0, frame // 2 + 1))


def frame_rms(y: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    values = []
    for start in range(0, y.size - frame + 1, hop):
        chunk = y[start : start + frame]
        values.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))
    return np.asarray(values)


def band_matrix(mag: np.ndarray, bands: int = 40) -> np.ndarray:
    if mag.size == 0:
        return mag
    edges = np.geomspace(1, mag.shape[1], bands + 1).astype(int) - 1
    edges[0] = 0
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        end = max(start + 1, end)
        out.append(np.mean(mag[:, start:end], axis=1))
    return np.asarray(out).T


def a_weighting(freqs: np.ndarray) -> np.ndarray:
    f2 = np.square(np.maximum(freqs, 1.0))
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    db = 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.0
    weights = 10.0 ** (db / 20.0)
    return weights / (np.max(weights) + 1e-12)


def onset_curve(env: np.ndarray) -> np.ndarray:
    if env.size < 2:
        return np.zeros_like(env)
    diff = np.maximum(0, np.diff(env, prepend=env[0]))
    peak = float(np.max(diff))
    return diff / peak if peak > 0 else diff


def onset_positions(env: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    curve = onset_curve(env)
    peaks = []
    for index in range(1, max(1, curve.size - 1)):
        if curve[index] >= threshold and curve[index] >= curve[index - 1] and curve[index] >= curve[index + 1]:
            peaks.append(index)
    if curve.size and curve[0] >= threshold:
        peaks.insert(0, 0)
    return np.asarray(peaks, dtype=np.float32)


def estimate_beat_grid(reference: np.ndarray, sr: int, subdivision: int = 4) -> dict[str, Any]:
    y = mono(reference)
    duration = y.size / sr
    env = frame_rms(y, frame=1024, hop=512)
    positions = onset_positions(env, threshold=0.32)
    onset_times = positions * 512.0 / sr
    if env.size < 4:
        tempo = 120.0
    else:
        curve = onset_curve(env)
        tempos = np.arange(60.0, 181.0, 1.0)
        scores = []
        for bpm in tempos:
            beat_frames = max(1.0, (60.0 / bpm) * sr / 512.0)
            lags = [beat_frames * mul for mul in (1.0, 2.0, 0.5)]
            score = 0.0
            for lag in lags:
                lag_i = int(round(lag))
                if 1 <= lag_i < curve.size:
                    score += float(np.dot(curve[:-lag_i], curve[lag_i:]) / max(1, curve.size - lag_i))
            scores.append(score)
        tempo = float(tempos[int(np.argmax(scores))])
    step_seconds = 60.0 / tempo / subdivision
    if onset_times.size:
        phases = np.linspace(0.0, step_seconds, 32, endpoint=False)
        phase_scores = []
        for phase in phases:
            distances = np.abs(((onset_times - phase + step_seconds / 2.0) % step_seconds) - step_seconds / 2.0)
            phase_scores.append(float(np.mean(distances)))
        phase = float(phases[int(np.argmin(phase_scores))])
    else:
        phase = 0.0
    edges = np.arange(phase, duration + step_seconds, step_seconds)
    if edges.size < 2 or edges[0] > 0.0:
        edges = np.insert(edges, 0, 0.0)
    edges = np.unique(np.clip(edges, 0.0, duration))
    if edges[-1] < duration:
        edges = np.append(edges, duration)
    downbeat_time = float(phase)
    return {
        "tempo": tempo,
        "meter": "4/4",
        "subdivision": f"1/{4 * subdivision}",
        "subdivision_per_beat": subdivision,
        "step_seconds": float(step_seconds),
        "phase_seconds": phase,
        "downbeat_time_seconds": downbeat_time,
        "downbeat_frame": int(round(downbeat_time * sr)),
        "edges_seconds": [float(x) for x in edges.tolist()],
        "onset_times_seconds": [float(x) for x in onset_times.tolist()],
    }


def chroma(mag: np.ndarray, sr: int) -> np.ndarray:
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    out = np.zeros(12)
    for idx, freq in enumerate(freqs):
        if 40 <= freq <= 5000:
            midi = int(round(69 + 12 * math.log2(freq / 440.0)))
            out[midi % 12] += float(np.mean(mag[:, idx]))
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def centroid_rolloff_flatness(mag: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    energy = np.sum(mag, axis=1) + 1e-8
    centroid = np.sum(mag * freqs[None, :], axis=1) / energy
    cumulative = np.cumsum(mag, axis=1)
    rolloff_idx = np.argmax(cumulative >= cumulative[:, -1:] * 0.85, axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, freqs.size - 1)]
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-8)
    bandwidth = np.sqrt(np.sum(mag * np.square(freqs[None, :] - centroid[:, None]), axis=1) / energy)
    return {"centroid": centroid, "rolloff": rolloff, "flatness": flatness, "bandwidth": bandwidth}


def band_energies(mag: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    total = np.sum(mag, axis=1) + 1e-8
    bands = {
        "sub": (20, 80),
        "bass": (80, 250),
        "low_mid": (250, 700),
        "mid": (700, 2000),
        "presence": (2000, 5000),
        "air": (5000, sr / 2),
    }
    return {name: np.sum(mag[:, (freqs >= lo) & (freqs < hi)], axis=1) / total for name, (lo, hi) in bands.items()}


def estimate_f0_track(y: np.ndarray, sr: int, frame: int = 2048, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    lo = max(1, int(sr / 1000))
    hi = min(frame - 1, int(sr / 45))
    values = []
    for start in range(0, y.size - frame + 1, hop):
        chunk = y[start : start + frame]
        chunk = chunk - np.mean(chunk)
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        if rms < 1e-4:
            values.append(0.0)
            continue
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1 :]
        segment = corr[lo:hi]
        if segment.size == 0:
            values.append(0.0)
            continue
        lag = lo + int(np.argmax(segment))
        confidence = float(segment[lag - lo] / (corr[0] + 1e-8))
        values.append(sr / lag if confidence > 0.2 else 0.0)
    return np.asarray(values)


def harmonic_noise_ratio(mag: np.ndarray) -> np.ndarray:
    if mag.shape[1] < 5:
        return np.zeros(mag.shape[0])
    sorted_bins = np.sort(mag, axis=1)
    tonal = np.mean(sorted_bins[:, -max(1, mag.shape[1] // 20) :], axis=1)
    noisy = np.median(mag, axis=1) + 1e-8
    return tonal / noisy


def cepstral_vector(mag: np.ndarray, count: int = 20) -> np.ndarray:
    log_mag = np.log(np.maximum(mag, 1e-8))
    cep = np.fft.irfft(log_mag, axis=1)
    return np.mean(cep[:, 1 : count + 1], axis=0)


def latent_vector(mag: np.ndarray, env: np.ndarray, sr: int) -> np.ndarray:
    feats = centroid_rolloff_flatness(mag, sr)
    bands = band_energies(mag, sr)
    values: list[float] = []
    for arr in [feats["centroid"], feats["rolloff"], feats["flatness"], feats["bandwidth"], env]:
        values.extend([float(np.mean(arr)), float(np.std(arr)), float(np.percentile(arr, 90) - np.percentile(arr, 10))])
    for arr in bands.values():
        values.extend([float(np.mean(arr)), float(np.std(arr))])
    values.extend(cepstral_vector(mag, 12).tolist())
    return np.asarray(values, dtype=np.float32)


def stereo_stats(audio: np.ndarray, sr: int) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[0] < 2:
        return {"overall_width": 0.0, "correlation": 1.0, "band_width": {}}
    left, right = audio[0], audio[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    overall_width = float(np.std(side) / (np.std(mid) + 1e-8))
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-8 and np.std(right) > 1e-8 else 1.0
    mid_mag = stft_mag(mid)
    side_mag = stft_mag(side)
    mid_bands = band_energies(mid_mag, sr)
    side_bands = band_energies(side_mag, sr)
    band_width = {name: float(np.mean(side_bands[name]) / (np.mean(mid_bands[name]) + 1e-8)) for name in mid_bands}
    return {"overall_width": overall_width, "correlation": correlation, "band_width": band_width}


def relative_score(ref: np.ndarray, cand: np.ndarray, scale_floor: float = 1e-8) -> float:
    frames = min(ref.size, cand.size)
    ref = ref[:frames]
    cand = cand[:frames]
    distance = float(np.mean(np.abs(ref - cand)) / (np.mean(np.abs(ref)) + scale_floor))
    return safe_exp_distance(distance)


def ratio_score(reference: float, candidate: float, floor: float = 1e-8) -> float:
    distance = abs(reference - candidate) / max(abs(reference), abs(candidate), floor)
    return safe_exp_distance(float(distance))


def segment_values(values: np.ndarray, segments: int = 10) -> np.ndarray:
    if values.size == 0:
        return np.zeros(segments)
    edges = np.linspace(0, values.size, segments + 1).astype(int)
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        end = max(start + 1, end)
        out.append(float(np.mean(values[start:end])))
    return np.asarray(out)


def segmented_relative_score(ref: np.ndarray, cand: np.ndarray, segments: int = 10) -> float:
    return relative_score(segment_values(ref, segments), segment_values(cand, segments))


def late_energy_score(ref_env: np.ndarray, cand_env: np.ndarray) -> float:
    if ref_env.size == 0 or cand_env.size == 0:
        return 1.0
    split_ref = max(1, int(ref_env.size * 0.5))
    split_cand = max(1, int(cand_env.size * 0.5))
    ref_late = float(np.sum(ref_env[split_ref:]) / (np.sum(ref_env) + 1e-8))
    cand_late = float(np.sum(cand_env[split_cand:]) / (np.sum(cand_env) + 1e-8))
    return ratio_score(ref_late, cand_late)


def sustain_score(ref_env: np.ndarray, cand_env: np.ndarray) -> float:
    if ref_env.size == 0 or cand_env.size == 0:
        return 1.0
    ref_threshold = max(float(np.percentile(ref_env, 70)) * 0.35, 1e-6)
    cand_threshold = max(float(np.percentile(cand_env, 70)) * 0.35, 1e-6)
    ref_coverage = float(np.mean(ref_env > ref_threshold))
    cand_coverage = float(np.mean(cand_env > cand_threshold))
    return ratio_score(ref_coverage, cand_coverage)


def frontload_score(ref_env: np.ndarray, cand_env: np.ndarray) -> float:
    if ref_env.size == 0 or cand_env.size == 0:
        return 1.0
    ref_end = max(1, int(ref_env.size * 0.3))
    cand_end = max(1, int(cand_env.size * 0.3))
    ref_front = float(np.sum(ref_env[:ref_end]) / (np.sum(ref_env) + 1e-8))
    cand_front = float(np.sum(cand_env[:cand_end]) / (np.sum(cand_env) + 1e-8))
    return ratio_score(ref_front, cand_front)


def fixed_window_metrics(reference: np.ndarray, candidate: np.ndarray, sr: int, window_seconds: float = 0.05) -> dict[str, Any]:
    samples = min(reference.shape[-1], candidate.shape[-1])
    step = max(1, int(window_seconds * sr))
    ref_mono = mono(reference)[..., :samples]
    cand_mono = mono(candidate)[..., :samples]
    band_names = ["sub", "bass", "low_mid", "mid", "presence", "air"]
    rows = []
    ref_env = []
    cand_env = []
    ref_bands = []
    cand_bands = []
    for index, start in enumerate(range(0, samples, step)):
        end = min(samples, start + step)
        if end <= start:
            continue
        r = ref_mono[start:end]
        c = cand_mono[start:end]
        rb = slice_band_vector(r, sr)
        cb = slice_band_vector(c, sr)
        rr = float(np.sqrt(np.mean(r**2) + 1e-12))
        cr = float(np.sqrt(np.mean(c**2) + 1e-12))
        ref_env.append(rr)
        cand_env.append(cr)
        ref_bands.append(rb)
        cand_bands.append(cb)
        delta_bands = cb - rb
        worst = int(np.argmax(np.abs(delta_bands))) if delta_bands.size else 0
        rows.append(
            {
                "index": index,
                "start": float(start / sr),
                "end": float(end / sr),
                "source_rms": rr,
                "candidate_rms": cr,
                "delta_rms": float(cr - rr),
                "source_bands": {name: float(rb[pos]) for pos, name in enumerate(band_names[: rb.size])},
                "candidate_bands": {name: float(cb[pos]) for pos, name in enumerate(band_names[: cb.size])},
                "delta_bands": {name: float(delta_bands[pos]) for pos, name in enumerate(band_names[: delta_bands.size])},
                "largest_band_error": band_names[worst] if worst < len(band_names) else "unknown",
            }
        )
    ref_env_arr = np.asarray(ref_env, dtype=np.float32)
    cand_env_arr = np.asarray(cand_env, dtype=np.float32)
    ref_band_arr = np.asarray(ref_bands, dtype=np.float32) if ref_bands else np.zeros((0, 6), dtype=np.float32)
    cand_band_arr = np.asarray(cand_bands, dtype=np.float32) if cand_bands else np.zeros((0, 6), dtype=np.float32)
    envelope_distance = float(np.mean(np.abs(ref_env_arr - cand_env_arr)) / (np.mean(np.abs(ref_env_arr)) + 1e-8)) if ref_env_arr.size else 0.0
    envelope_score = safe_exp_distance(2.0 * envelope_distance)
    band_score = matrix_score(ref_band_arr, cand_band_arr) if ref_band_arr.size and cand_band_arr.size else 1.0
    weak = sorted(rows, key=lambda row: abs(row["delta_rms"]) + max(abs(v) for v in row["delta_bands"].values()))[:]
    weak = list(reversed(weak[-16:]))
    return {"envelope": envelope_score, "band": band_score, "windows": rows, "weak_windows": weak}


def dominant_modulation(values: np.ndarray, sr: int, hop: int = 512) -> dict[str, float]:
    if values.size < 8:
        return {"rate_hz": 0.0, "depth": 0.0, "periodicity": 0.0}
    centered = values.astype(np.float32) - float(np.mean(values))
    depth = float(np.std(centered) / (float(np.mean(np.abs(values))) + 1e-8))
    if float(np.std(centered)) < 1e-8:
        return {"rate_hz": 0.0, "depth": depth, "periodicity": 0.0}
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    freqs = np.fft.rfftfreq(centered.size, hop / sr)
    mask = (freqs >= 0.2) & (freqs <= 12.0)
    if not np.any(mask):
        return {"rate_hz": 0.0, "depth": depth, "periodicity": 0.0}
    masked = spectrum[mask]
    masked_freqs = freqs[mask]
    peak_index = int(np.argmax(masked))
    periodicity = float(masked[peak_index] / (np.sum(masked) + 1e-8))
    return {"rate_hz": float(masked_freqs[peak_index]), "depth": depth, "periodicity": periodicity}


def modulation_scores(ref_env: np.ndarray, cand_env: np.ndarray, ref_centroid: np.ndarray, cand_centroid: np.ndarray, sr: int) -> dict[str, Any]:
    ref_sig = dominant_modulation(ref_env, sr)
    cand_sig = dominant_modulation(cand_env, sr)
    ref_cent = dominant_modulation(ref_centroid, sr)
    cand_cent = dominant_modulation(cand_centroid, sr)
    ref_rate = max(ref_sig["rate_hz"], ref_cent["rate_hz"])
    cand_rate = max(cand_sig["rate_hz"], cand_cent["rate_hz"])
    ref_depth = max(ref_sig["depth"], ref_cent["depth"])
    cand_depth = max(cand_sig["depth"], cand_cent["depth"])
    ref_periodicity = max(ref_sig["periodicity"], ref_cent["periodicity"])
    cand_periodicity = max(cand_sig["periodicity"], cand_cent["periodicity"])
    rate_score = ratio_score(ref_rate, cand_rate, 0.25) if max(ref_rate, cand_rate) > 0 else 1.0
    depth_score = ratio_score(ref_depth, cand_depth, 0.05)
    periodicity_score = ratio_score(ref_periodicity, cand_periodicity, 0.05)
    return {
        "rate": rate_score,
        "depth": depth_score,
        "periodicity": periodicity_score,
        "score": float(np.mean([rate_score, depth_score, periodicity_score])),
        "reference": {"rate_hz": ref_rate, "depth": ref_depth, "periodicity": ref_periodicity},
        "candidate": {"rate_hz": cand_rate, "depth": cand_depth, "periodicity": cand_periodicity},
    }


def transient_classification_score(ref_env: np.ndarray, cand_env: np.ndarray) -> float:
    ref_curve = onset_curve(ref_env)
    cand_curve = onset_curve(cand_env)
    ref_diffuse = float(np.mean(ref_curve > 0.08))
    cand_diffuse = float(np.mean(cand_curve > 0.08))
    ref_peak = float(np.max(ref_curve)) if ref_curve.size else 0.0
    cand_peak = float(np.max(cand_curve)) if cand_curve.size else 0.0
    diffuse_score = ratio_score(ref_diffuse, cand_diffuse, 0.02)
    peak_score = ratio_score(ref_peak, cand_peak, 0.05)
    return float(np.mean([diffuse_score, peak_score]))


def band_envelope_score(ref_mag: np.ndarray, cand_mag: np.ndarray, sr: int) -> float:
    ref_bands = band_energies(ref_mag, sr)
    cand_bands = band_energies(cand_mag, sr)
    scores = [segmented_relative_score(ref_bands[name], cand_bands[name], 8) for name in ref_bands]
    return float(np.mean(scores)) if scores else 1.0


def slice_edges_to_samples(edges_seconds: list[float], samples: int, sr: int) -> list[tuple[int, int]]:
    edges = [int(np.clip(round(edge * sr), 0, samples)) for edge in edges_seconds]
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end > start:
            out.append((start, end))
    return out


def slice_band_vector(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size < 64:
        y = np.pad(y, (0, 64 - y.size))
    mag = stft_mag(y, frame=min(1024, max(128, int(2 ** math.ceil(math.log2(max(64, y.size))))), 1024), hop=max(64, min(512, y.size // 2)))
    bands = band_energies(mag, sr)
    return np.asarray([float(np.mean(values)) for values in bands.values()], dtype=np.float32)


def beat_grid_scores(reference: np.ndarray, candidate: np.ndarray, sr: int, grid: dict[str, Any]) -> dict[str, Any]:
    samples = min(reference.shape[-1], candidate.shape[-1])
    slices = slice_edges_to_samples(grid.get("edges_seconds", []), samples, sr)
    if not slices:
        return {"mel": 1.0, "band": 1.0, "envelope": 1.0, "mid_side": 1.0, "slice_count": 0, "weak_slices": [], "slices": []}
    ref_mono = mono(reference)[..., :samples]
    cand_mono = mono(candidate)[..., :samples]
    mel_ref = []
    mel_cand = []
    band_ref = []
    band_cand = []
    env_ref = []
    env_cand = []
    ms_ref = []
    ms_cand = []
    slice_reports = []
    band_names = ["sub", "bass", "low_mid", "mid", "presence", "air"]
    for index, (start, end) in enumerate(slices):
        r = ref_mono[start:end]
        c = cand_mono[start:end]
        rb = slice_band_vector(r, sr)
        cb = slice_band_vector(c, sr)
        band_ref.append(rb)
        band_cand.append(cb)
        mel_ref.append(rb)
        mel_cand.append(cb)
        env_ref.append(float(np.sqrt(np.mean(r**2) + 1e-12)))
        env_cand.append(float(np.sqrt(np.mean(c**2) + 1e-12)))
        ref_ms = [0.0, 0.0]
        cand_ms = [0.0, 0.0]
        mid_side_score = 1.0
        if reference.ndim == 2 and candidate.ndim == 2 and reference.shape[0] >= 2 and candidate.shape[0] >= 2:
            ref_mid = 0.5 * (reference[0, start:end] + reference[1, start:end])
            ref_side = 0.5 * (reference[0, start:end] - reference[1, start:end])
            cand_mid = 0.5 * (candidate[0, start:end] + candidate[1, start:end])
            cand_side = 0.5 * (candidate[0, start:end] - candidate[1, start:end])
            ref_ms = [float(np.sqrt(np.mean(ref_mid**2) + 1e-12)), float(np.sqrt(np.mean(ref_side**2) + 1e-12))]
            cand_ms = [float(np.sqrt(np.mean(cand_mid**2) + 1e-12)), float(np.sqrt(np.mean(cand_side**2) + 1e-12))]
            ms_ref.append(ref_ms)
            ms_cand.append(cand_ms)
            mid_side_score = relative_score(np.asarray(ref_ms), np.asarray(cand_ms))
        band_score = relative_score(rb, cb)
        envelope_score = ratio_score(env_ref[-1], env_cand[-1])
        mel_score = band_score
        deltas = cb - rb
        worst_band_index = int(np.argmax(np.abs(deltas))) if deltas.size else 0
        slice_reports.append(
            {
                "index": index,
                "start": float(start / sr),
                "end": float(end / sr),
                "scores": {
                    "mel": float(mel_score),
                    "band": float(band_score),
                    "envelope": float(envelope_score),
                    "mid_side": float(mid_side_score),
                },
                "source": {
                    "rms": float(env_ref[-1]),
                    "bands": {name: float(rb[pos]) for pos, name in enumerate(band_names[: rb.size])},
                    "mid": ref_ms[0],
                    "side": ref_ms[1],
                },
                "candidate": {
                    "rms": float(env_cand[-1]),
                    "bands": {name: float(cb[pos]) for pos, name in enumerate(band_names[: cb.size])},
                    "mid": cand_ms[0],
                    "side": cand_ms[1],
                },
                "delta": {
                    "rms": float(env_cand[-1] - env_ref[-1]),
                    "bands": {name: float(deltas[pos]) for pos, name in enumerate(band_names[: deltas.size])},
                    "largest_band": band_names[worst_band_index] if worst_band_index < len(band_names) else "unknown",
                },
            }
        )
    mel = matrix_score(np.asarray(mel_ref), np.asarray(mel_cand))
    band = matrix_score(np.asarray(band_ref), np.asarray(band_cand))
    envelope = relative_score(np.asarray(env_ref), np.asarray(env_cand))
    mid_side = matrix_score(np.asarray(ms_ref), np.asarray(ms_cand)) if ms_ref else 1.0
    weak = sorted(
        slice_reports,
        key=lambda item: min(item["scores"]["mel"], item["scores"]["band"], item["scores"]["envelope"], item["scores"]["mid_side"]),
    )[:12]
    return {"mel": mel, "band": band, "envelope": envelope, "mid_side": mid_side, "slice_count": len(slices), "weak_slices": weak, "slices": slice_reports}


def trajectory_score(ref: np.ndarray, cand: np.ndarray, segments: int = 10) -> float:
    ref_segments = segment_values(ref, segments)
    cand_segments = segment_values(cand, segments)
    return relative_score(ref_segments, cand_segments)


def onset_count_score(ref_positions: np.ndarray, cand_positions: np.ndarray) -> float:
    ref_count = float(ref_positions.size)
    cand_count = float(cand_positions.size)
    return ratio_score(ref_count, cand_count, 1.0)


def onset_timing_score(ref_positions: np.ndarray, cand_positions: np.ndarray, length: int) -> float:
    if ref_positions.size == 0 and cand_positions.size == 0:
        return 1.0
    if ref_positions.size == 0 or cand_positions.size == 0:
        return 0.0
    ref_norm = ref_positions / max(1, length - 1)
    cand_norm = cand_positions / max(1, length - 1)
    distances = [float(np.min(np.abs(cand_norm - pos))) for pos in ref_norm]
    extra_penalty = abs(ref_positions.size - cand_positions.size) / max(ref_positions.size, cand_positions.size, 1)
    return safe_exp_distance(float(np.mean(distances) * 8.0 + extra_penalty))


def structural_gate_penalty(
    ref_positions: np.ndarray,
    cand_positions: np.ndarray,
    f0_contour: float,
    spectral_features: float,
    harmonic_noise: float,
    beat_grid_mel: float = 1.0,
    beat_grid_band: float = 1.0,
) -> float:
    """Harder musical-identity penalty for patterns that average metrics can hide."""
    penalty = 1.0
    ref_count = int(ref_positions.size)
    cand_count = int(cand_positions.size)
    if ref_count >= 16:
        count_ratio = cand_count / max(1, ref_count)
        if count_ratio < 0.55:
            penalty *= 0.72
        elif count_ratio < 0.8:
            penalty *= 0.88
    if f0_contour < 0.35:
        penalty *= 0.82
    if spectral_features < 0.5:
        penalty *= 0.9
    if harmonic_noise < 0.5:
        penalty *= 0.94
    if beat_grid_mel < 0.58 or beat_grid_band < 0.58:
        penalty *= 0.82
    return float(np.clip(penalty, 0.45, 1.0))


def matrix_score(ref: np.ndarray, cand: np.ndarray) -> float:
    rows = min(ref.shape[0], cand.shape[0])
    cols = min(ref.shape[1], cand.shape[1])
    ref = ref[:rows, :cols]
    cand = cand[:rows, :cols]
    distance = float(np.mean(np.abs(np.log1p(ref) - np.log1p(cand))) / (np.mean(np.log1p(ref)) + 1e-8))
    return safe_exp_distance(distance)


def compare_audio(reference: np.ndarray, candidate: np.ndarray, sr: int, beat_grid: dict[str, Any] | None = None) -> dict[str, Any]:
    reference, candidate = align(reference, candidate)
    ref = mono(reference)
    cand = mono(candidate)
    ref_mag = stft_mag(ref)
    cand_mag = stft_mag(cand)
    beat_grid = beat_grid or estimate_beat_grid(reference, sr)
    grid_scores = beat_grid_scores(reference, candidate, sr, beat_grid)

    spectral_scores = []
    for frame, hop in [(512, 128), (1024, 256), (2048, 512), (4096, 1024)]:
        spectral_scores.append(matrix_score(stft_mag(ref, frame, hop), stft_mag(cand, frame, hop)))
    multi_resolution_spectral = float(np.mean(spectral_scores))
    mel_spectrogram = matrix_score(band_matrix(ref_mag, 48), band_matrix(cand_mag, 48))

    freqs = np.fft.rfftfreq((ref_mag.shape[1] - 1) * 2, 1.0 / sr)
    weights = a_weighting(freqs)
    a_weighted_spectral = matrix_score(ref_mag * weights[None, :], cand_mag * weights[None, :])

    ref_env = frame_rms(ref)
    cand_env = frame_rms(cand)
    fixed_50ms = fixed_window_metrics(reference, candidate, sr, 0.05)
    envelope = relative_score(ref_env, cand_env)
    segment_envelope = segmented_relative_score(ref_env, cand_env, 10)
    late_energy_ratio = late_energy_score(ref_env, cand_env)
    sustain_coverage = sustain_score(ref_env, cand_env)
    frontload_balance = frontload_score(ref_env, cand_env)
    band_envelope_by_time = band_envelope_score(ref_mag, cand_mag, sr)
    ref_onsets = onset_positions(ref_env)
    cand_onsets = onset_positions(cand_env)
    transient_onset = relative_score(onset_curve(ref_env), onset_curve(cand_env), 1.0)
    onset_count = onset_count_score(ref_onsets, cand_onsets)
    onset_timing = onset_timing_score(ref_onsets, cand_onsets, min(ref_env.size, cand_env.size))
    pitch_chroma = max(0.0, cosine(chroma(ref_mag, sr), chroma(cand_mag, sr)))

    ref_f0 = estimate_f0_track(ref, sr)
    cand_f0 = estimate_f0_track(cand, sr)
    voiced = (ref_f0 > 0) | (cand_f0 > 0)
    if np.any(voiced):
        f0_contour = safe_exp_distance(float(np.mean(np.abs(np.log2((ref_f0[voiced] + 1e-6) / (cand_f0[voiced] + 1e-6))))))
    else:
        f0_contour = 1.0

    ref_features = centroid_rolloff_flatness(ref_mag, sr)
    cand_features = centroid_rolloff_flatness(cand_mag, sr)
    mod_detail = modulation_scores(ref_env, cand_env, ref_features["centroid"], cand_features["centroid"], sr)
    spectral_motion_parts = [
        relative_score(ref_features["centroid"], cand_features["centroid"]),
        relative_score(ref_features["rolloff"], cand_features["rolloff"]),
    ]
    spectral_motion = float(np.mean(spectral_motion_parts))
    centroid_trajectory = trajectory_score(ref_features["centroid"], cand_features["centroid"], 10)
    spectral_features = float(
        np.mean(
            [
                relative_score(ref_features["centroid"], cand_features["centroid"]),
                relative_score(ref_features["rolloff"], cand_features["rolloff"]),
                relative_score(ref_features["flatness"], cand_features["flatness"]),
                relative_score(ref_features["bandwidth"], cand_features["bandwidth"]),
            ]
        )
    )

    ref_band_env = band_matrix(ref_mag, 16)
    cand_band_env = band_matrix(cand_mag, 16)
    modulation = matrix_score(np.abs(np.diff(ref_band_env, axis=0)), np.abs(np.diff(cand_band_env, axis=0)))
    transient_classification = transient_classification_score(ref_env, cand_env)
    directional_delta = float(
        np.mean(
            [
                relative_score(np.diff(ref_env), np.diff(cand_env), 1.0),
                relative_score(np.diff(ref_features["centroid"]), np.diff(cand_features["centroid"]), 1.0),
            ]
        )
    )

    harmonic_noise = relative_score(harmonic_noise_ratio(ref_mag), harmonic_noise_ratio(cand_mag))
    cepstral = max(0.0, cosine(cepstral_vector(ref_mag), cepstral_vector(cand_mag)))
    embedding = max(0.0, cosine(latent_vector(ref_mag, ref_env, sr), latent_vector(cand_mag, cand_env, sr)))
    codec_latent = matrix_score(band_matrix(ref_mag, 12), band_matrix(cand_mag, 12))

    ref_stereo = stereo_stats(reference, sr)
    cand_stereo = stereo_stats(candidate, sr)
    band_width_scores = []
    for name, value in ref_stereo["band_width"].items():
        band_width_scores.append(safe_exp_distance(abs(value - cand_stereo["band_width"].get(name, 0.0)) / max(abs(value), abs(cand_stereo["band_width"].get(name, 0.0)), 1e-6)))
    stereo_width = float(
        np.mean(
            [
                safe_exp_distance(abs(ref_stereo["overall_width"] - cand_stereo["overall_width"]) / max(abs(ref_stereo["overall_width"]), abs(cand_stereo["overall_width"]), 1e-6)),
                safe_exp_distance(abs(ref_stereo["correlation"] - cand_stereo["correlation"])),
                float(np.mean(band_width_scores)) if band_width_scores else 1.0,
            ]
        )
    )

    time_series_core = float(
        np.mean(
            [
                fixed_50ms["envelope"],
                fixed_50ms["band"],
                grid_scores["envelope"],
                grid_scores["band"],
                band_envelope_by_time,
                directional_delta,
            ]
        )
    )
    timbre_core = float(np.mean([multi_resolution_spectral, mel_spectrogram, a_weighted_spectral, spectral_features, harmonic_noise, codec_latent]))
    motion_core = float(np.mean([spectral_motion, centroid_trajectory, modulation, mod_detail["score"], transient_classification]))
    pitch_core = float(np.mean([pitch_chroma, f0_contour]))
    stereo_core = float(np.mean([stereo_width, grid_scores["mid_side"]]))
    weighted_final = 0.38 * time_series_core + 0.24 * timbre_core + 0.16 * motion_core + 0.12 * pitch_core + 0.07 * stereo_core + 0.03 * cepstral
    gate = min(time_series_core, max(0.52, motion_core), max(0.50, timbre_core))
    penalty = structural_gate_penalty(ref_onsets, cand_onsets, f0_contour, spectral_features, harmonic_noise, grid_scores["mel"], grid_scores["band"])

    score = AudioScore(
        final=float(weighted_final * penalty * (0.55 + 0.45 * gate)),
        multi_resolution_spectral=multi_resolution_spectral,
        mel_spectrogram=mel_spectrogram,
        a_weighted_spectral=a_weighted_spectral,
        envelope=envelope,
        segment_envelope=segment_envelope,
        late_energy_ratio=late_energy_ratio,
        sustain_coverage=sustain_coverage,
        frontload_balance=frontload_balance,
        band_envelope_by_time=band_envelope_by_time,
        beat_grid_mel=grid_scores["mel"],
        beat_grid_band=grid_scores["band"],
        beat_grid_envelope=grid_scores["envelope"],
        beat_grid_mid_side=grid_scores["mid_side"],
        pitch_chroma=pitch_chroma,
        f0_contour=f0_contour,
        spectral_motion=spectral_motion,
        centroid_trajectory=centroid_trajectory,
        spectral_features=spectral_features,
        transient_onset=transient_onset,
        onset_count=onset_count,
        onset_timing=onset_timing,
        stereo_width=stereo_width,
        modulation=modulation,
        harmonic_noise=harmonic_noise,
        cepstral=cepstral,
        embedding=embedding,
        codec_latent=codec_latent,
        exact_envelope_50ms=float(fixed_50ms["envelope"]),
        exact_band_50ms=float(fixed_50ms["band"]),
        modulation_periodicity=float(mod_detail["periodicity"]),
        modulation_rate=float(mod_detail["rate"]),
        modulation_depth=float(mod_detail["depth"]),
        directional_delta=directional_delta,
        transient_classification=transient_classification,
    )
    diagnostics = build_diagnostics(reference, candidate, sr, ref_features, cand_features, ref_env, cand_env, ref_stereo, cand_stereo, score, beat_grid, grid_scores)
    diagnostics["fixed_50ms"] = fixed_50ms
    diagnostics["modulation_analysis"] = mod_detail
    diagnostics["score_groups"] = {
        "time_series_core": time_series_core,
        "timbre_core": timbre_core,
        "motion_core": motion_core,
        "pitch_core": pitch_core,
        "stereo_core": stereo_core,
        "gate": gate,
        "structural_penalty": penalty,
    }
    return {"scores": score_to_json(score), "diagnostics": diagnostics, "residual": residual_from_diff(score, diagnostics)}


def build_diagnostics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    ref_features: dict[str, np.ndarray],
    cand_features: dict[str, np.ndarray],
    ref_env: np.ndarray,
    cand_env: np.ndarray,
    ref_stereo: dict[str, Any],
    cand_stereo: dict[str, Any],
    score: AudioScore,
    beat_grid: dict[str, Any],
    grid_scores: dict[str, Any],
) -> dict[str, Any]:
    ref_mag = stft_mag(mono(reference))
    cand_mag = stft_mag(mono(candidate))
    ref_bands = band_energies(ref_mag, sr)
    cand_bands = band_energies(cand_mag, sr)
    band_deltas = {
        name: {
            "reference": float(np.mean(ref_bands[name])),
            "candidate": float(np.mean(cand_bands[name])),
            "delta": float(np.mean(cand_bands[name]) - np.mean(ref_bands[name])),
        }
        for name in ref_bands
    }
    ref_onsets = onset_positions(ref_env)
    cand_onsets = onset_positions(cand_env)
    ref_segment_env = segment_values(ref_env, 10)
    cand_segment_env = segment_values(cand_env, 10)
    ref_late = float(np.sum(ref_env[max(1, int(ref_env.size * 0.5)) :]) / (np.sum(ref_env) + 1e-8)) if ref_env.size else 0.0
    cand_late = float(np.sum(cand_env[max(1, int(cand_env.size * 0.5)) :]) / (np.sum(cand_env) + 1e-8)) if cand_env.size else 0.0
    ref_front = float(np.sum(ref_env[: max(1, int(ref_env.size * 0.3))]) / (np.sum(ref_env) + 1e-8)) if ref_env.size else 0.0
    cand_front = float(np.sum(cand_env[: max(1, int(cand_env.size * 0.3))]) / (np.sum(cand_env) + 1e-8)) if cand_env.size else 0.0
    return {
        "reference_centroid_start_hz": float(ref_features["centroid"][0]) if ref_features["centroid"].size else 0.0,
        "reference_centroid_end_hz": float(ref_features["centroid"][-1]) if ref_features["centroid"].size else 0.0,
        "candidate_centroid_start_hz": float(cand_features["centroid"][0]) if cand_features["centroid"].size else 0.0,
        "candidate_centroid_end_hz": float(cand_features["centroid"][-1]) if cand_features["centroid"].size else 0.0,
        "reference_rms": float(np.sqrt(np.mean(np.square(mono(reference))) + 1e-12)),
        "candidate_rms": float(np.sqrt(np.mean(np.square(mono(candidate))) + 1e-12)),
        "reference_onset_count": int(ref_onsets.size),
        "candidate_onset_count": int(cand_onsets.size),
        "reference_onset_positions": ref_onsets.astype(int).tolist(),
        "candidate_onset_positions": cand_onsets.astype(int).tolist(),
        "reference_late_energy_ratio": ref_late,
        "candidate_late_energy_ratio": cand_late,
        "reference_front_energy_ratio": ref_front,
        "candidate_front_energy_ratio": cand_front,
        "reference_segment_envelope": ref_segment_env.tolist(),
        "candidate_segment_envelope": cand_segment_env.tolist(),
        "band_energy": band_deltas,
        "reference_stereo": ref_stereo,
        "candidate_stereo": cand_stereo,
        "beat_grid": beat_grid,
        "beat_grid_scores": {
            "mel": grid_scores.get("mel", 1.0),
            "band": grid_scores.get("band", 1.0),
            "envelope": grid_scores.get("envelope", 1.0),
            "mid_side": grid_scores.get("mid_side", 1.0),
            "slice_count": grid_scores.get("slice_count", 0),
            "weak_slices": grid_scores.get("weak_slices", []),
            "slices": grid_scores.get("slices", []),
        },
        "weakest_components": sorted(score_to_json(score).items(), key=lambda item: item[1])[:5],
    }


def residual_from_diff(score: AudioScore, diagnostics: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    recommendations: list[str] = []
    if score.mel_spectrogram < 0.68 or score.multi_resolution_spectral < 0.68:
        missing.append("candidate broad timbre/spectrum differs from the source")
        recommendations.append("adjust oscillator blend, filter range, layer gain, or add a missing support layer")
    if score.a_weighted_spectral < 0.68:
        missing.append("perceptually weighted brightness/presence balance is off")
        recommendations.append("adjust cutoff_end_hz, high layer level, or presence/air layer")
    if score.envelope < 0.7:
        missing.append("RMS envelope does not follow the source")
        recommendations.append("adjust note durations, attack/release, layer timing, or master gain automation")
    if score.segment_envelope < 0.72:
        missing.append("time-segmented envelope differs across the 5-second clip")
        recommendations.append("add or adjust gain automation, note lengths, and sustained layer body across the full clip")
    if score.late_energy_ratio < 0.78 or score.sustain_coverage < 0.78:
        ref_late = diagnostics.get("reference_late_energy_ratio", 0.0)
        cand_late = diagnostics.get("candidate_late_energy_ratio", 0.0)
        missing.append(f"candidate late/sustained energy differs: source late energy {ref_late:.2f}, candidate {cand_late:.2f}")
        recommendations.append("increase sustained body after the first half or reduce front-only decay")
    if score.frontload_balance < 0.78:
        ref_front = diagnostics.get("reference_front_energy_ratio", 0.0)
        cand_front = diagnostics.get("candidate_front_energy_ratio", 0.0)
        missing.append(f"candidate front/back energy balance differs: source front energy {ref_front:.2f}, candidate {cand_front:.2f}")
        recommendations.append("reduce front-loaded attack energy or add later sustained layer energy")
    if score.band_envelope_by_time < 0.7:
        missing.append("band energy changes over time do not match the source")
        recommendations.append("use filter/gain automation or layer timing to match low/mid/high energy across the whole clip")
    if score.exact_envelope_50ms < 0.72 or score.exact_band_50ms < 0.72:
        weak_windows = diagnostics.get("fixed_50ms", {}).get("weak_windows", [])[:5]
        ranges = ", ".join(f"{item.get('start', 0):.2f}-{item.get('end', 0):.2f}s {item.get('largest_band_error', 'band')}" for item in weak_windows)
        missing.append(f"50ms time-series match is poor; worst windows: {ranges}")
        recommendations.append("fix the exact time-local envelope/band errors before optimizing whole-clip averages")
    if score.beat_grid_mel < 0.68 or score.beat_grid_band < 0.68 or score.beat_grid_envelope < 0.68:
        grid = diagnostics.get("beat_grid", {})
        grid_scores = diagnostics.get("beat_grid_scores", {})
        missing.append(f"beat-grid reconstruction differs across {grid_scores.get('slice_count', 0)} {grid.get('subdivision', 'grid')} slices")
        recommendations.append("move note events, automation, and layer energy onto the detected beat grid instead of matching only whole-clip averages")
    if score.transient_onset < 0.7 or score.onset_count < 0.78 or score.onset_timing < 0.78:
        ref_count = diagnostics.get("reference_onset_count", 0)
        cand_count = diagnostics.get("candidate_onset_count", 0)
        missing.append(f"onset contour differs: source has {ref_count} prominent onset frames, candidate has {cand_count}")
        recommendations.append("reduce unwanted retriggers or add missing attacks")
    if score.pitch_chroma < 0.65 or score.f0_contour < 0.65:
        missing.append("pitch/chord contour appears mismatched")
        recommendations.append("change note choices, octave support, or add missing chord tones")
    if score.spectral_motion < 0.7 or score.centroid_trajectory < 0.72 or score.modulation < 0.7:
        ref_start = diagnostics.get("reference_centroid_start_hz", 0.0)
        ref_end = diagnostics.get("reference_centroid_end_hz", 0.0)
        cand_start = diagnostics.get("candidate_centroid_start_hz", 0.0)
        cand_end = diagnostics.get("candidate_centroid_end_hz", 0.0)
        missing.append(f"spectral motion differs: source centroid {ref_start:.0f}->{ref_end:.0f}Hz, candidate {cand_start:.0f}->{cand_end:.0f}Hz")
        recommendations.append("adjust filter cutoff automation, LFO depth/rate, or add movement layer")
    if score.modulation_periodicity < 0.72 or score.modulation_rate < 0.72 or score.modulation_depth < 0.72:
        mod = diagnostics.get("modulation_analysis", {})
        ref_mod = mod.get("reference", {})
        cand_mod = mod.get("candidate", {})
        missing.append(
            f"cyclic modulation differs: source rate {ref_mod.get('rate_hz', 0.0):.2f}Hz/depth {ref_mod.get('depth', 0.0):.2f}, candidate rate {cand_mod.get('rate_hz', 0.0):.2f}Hz/depth {cand_mod.get('depth', 0.0):.2f}"
        )
        recommendations.append("use a routed LFO or smooth repeated automation for filter/gain/pan; do not replace cyclic motion with a single chop")
    if score.transient_classification < 0.72:
        missing.append("transient shape classification differs; source modulation/attacks are being confused with candidate chops or missing pulses")
        recommendations.append("decide whether the target is cyclic LFO motion or discrete MIDI attacks, then remove the wrong type of event")
    if score.directional_delta < 0.72:
        missing.append("directional movement is wrong: candidate rises/falls at different times than the source")
        recommendations.append("align gain/filter movement direction per 50ms window instead of matching only average level")
    if score.stereo_width < 0.72:
        missing.append("stereo image differs from the source")
        recommendations.append("adjust width, pan, chorus_mix, reverb_mix, or high-band side energy")
    if score.harmonic_noise < 0.68:
        missing.append("harmonic-to-noise character differs")
        recommendations.append("adjust detune/voices/saturation or add/remove noise-like air layer")
    if score.cepstral < 0.68 or score.codec_latent < 0.68:
        missing.append("compact perceptual timbre features remain mismatched")
        recommendations.append("make a conservative timbre pass before changing arrangement")
    return {
        "missing": missing or ["largest remaining error is subtle parameter mismatch"],
        "recommendations": recommendations or ["make a conservative local adjustment to the weakest score components"],
        "diagnostics": diagnostics,
    }
