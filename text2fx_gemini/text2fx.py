#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sf: Any = None
genai: Any = None
types: Any = None
Compressor: Any = None
Delay: Any = None
Gain: Any = None
HighShelfFilter: Any = None
Limiter: Any = None
LowShelfFilter: Any = None
Pedalboard: Any = None
PeakFilter: Any = None
Reverb: Any = None
Distortion: Any = None


EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")


@dataclass
class FxParams:
    low_gain_db: float
    mid_gain_db: float
    high_gain_db: float
    reverb_room_size: float
    reverb_wet_level: float
    saturation_drive_db: float
    delay_mix: float
    compressor_threshold_db: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def vector_to_params(x: np.ndarray) -> FxParams:
    return FxParams(
        low_gain_db=float(np.interp(x[0], [0, 1], [-9, 9])),
        mid_gain_db=float(np.interp(x[1], [0, 1], [-9, 9])),
        high_gain_db=float(np.interp(x[2], [0, 1], [-9, 9])),
        reverb_room_size=clamp01(float(x[3])),
        reverb_wet_level=float(np.interp(x[4], [0, 1], [0.0, 0.35])),
        saturation_drive_db=float(np.interp(x[5], [0, 1], [0, 18])),
        delay_mix=float(np.interp(x[6], [0, 1], [0.0, 0.25])),
        compressor_threshold_db=float(np.interp(x[7], [0, 1], [-28, -8])),
    )


def params_to_vector(params: FxParams) -> np.ndarray:
    return np.asarray(
        [
            np.interp(params.low_gain_db, [-9, 9], [0, 1]),
            np.interp(params.mid_gain_db, [-9, 9], [0, 1]),
            np.interp(params.high_gain_db, [-9, 9], [0, 1]),
            params.reverb_room_size,
            np.interp(params.reverb_wet_level, [0.0, 0.35], [0, 1]),
            np.interp(params.saturation_drive_db, [0, 18], [0, 1]),
            np.interp(params.delay_mix, [0.0, 0.25], [0, 1]),
            np.interp(params.compressor_threshold_db, [-28, -8], [0, 1]),
        ],
        dtype=np.float32,
    )


def sanitize_params(params: FxParams) -> FxParams:
    return FxParams(
        low_gain_db=float(np.clip(params.low_gain_db, -9, 9)),
        mid_gain_db=float(np.clip(params.mid_gain_db, -9, 9)),
        high_gain_db=float(np.clip(params.high_gain_db, -9, 9)),
        reverb_room_size=float(np.clip(params.reverb_room_size, 0, 1)),
        reverb_wet_level=float(np.clip(params.reverb_wet_level, 0, 0.35)),
        saturation_drive_db=float(np.clip(params.saturation_drive_db, 0, 18)),
        delay_mix=float(np.clip(params.delay_mix, 0, 0.25)),
        compressor_threshold_db=float(np.clip(params.compressor_threshold_db, -28, -8)),
    )


def load_runtime_dependencies() -> None:
    global sf, genai, types
    global Compressor, Delay, Gain, HighShelfFilter, Limiter, LowShelfFilter, Pedalboard, PeakFilter, Reverb, Distortion

    try:
        import soundfile as soundfile_module
        from google import genai as genai_module
        from google.genai import types as genai_types
        from pedalboard import (
            Compressor as PedalboardCompressor,
            Delay as PedalboardDelay,
            Distortion as PedalboardDistortion,
            Gain as PedalboardGain,
            HighShelfFilter as PedalboardHighShelfFilter,
            Limiter as PedalboardLimiter,
            LowShelfFilter as PedalboardLowShelfFilter,
            PeakFilter as PedalboardPeakFilter,
            Pedalboard as PedalboardClass,
            Reverb as PedalboardReverb,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency. Run: pip install -r text2fx_gemini/requirements.txt"
        ) from exc

    sf = soundfile_module
    genai = genai_module
    types = genai_types
    Compressor = PedalboardCompressor
    Delay = PedalboardDelay
    Distortion = PedalboardDistortion
    Gain = PedalboardGain
    HighShelfFilter = PedalboardHighShelfFilter
    Limiter = PedalboardLimiter
    LowShelfFilter = PedalboardLowShelfFilter
    PeakFilter = PedalboardPeakFilter
    Pedalboard = PedalboardClass
    Reverb = PedalboardReverb


def build_board(params: FxParams) -> Any:
    return Pedalboard(
        [
            LowShelfFilter(cutoff_frequency_hz=180, gain_db=params.low_gain_db, q=0.707),
            PeakFilter(cutoff_frequency_hz=1200, gain_db=params.mid_gain_db, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=4500, gain_db=params.high_gain_db, q=0.707),
            Distortion(drive_db=params.saturation_drive_db),
            Compressor(threshold_db=params.compressor_threshold_db, ratio=2.0, attack_ms=10, release_ms=120),
            Delay(delay_seconds=0.25, feedback=0.22, mix=params.delay_mix),
            Reverb(room_size=params.reverb_room_size, damping=0.45, wet_level=params.reverb_wet_level, dry_level=1.0),
            Gain(gain_db=-1.0),
            Limiter(threshold_db=-1.0, release_ms=80),
        ]
    )


def render_audio(audio: np.ndarray, sample_rate: int, params: FxParams) -> np.ndarray:
    board = build_board(params)
    rendered = board(audio, sample_rate)
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    if peak > 0.99:
        rendered = rendered * (0.99 / peak)
    return rendered


def embed_text(client: Any, text: str) -> np.ndarray:
    response = embed_content_with_retry(
        client,
        contents=[types.Content(parts=[types.Part(text=text)])],
    )
    return np.asarray(response.embeddings[0].values, dtype=np.float32)


def embed_wav(client: Any, wav_path: Path) -> np.ndarray:
    data = wav_path.read_bytes()
    response = embed_content_with_retry(
        client,
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="audio/wav",
                            data=data,
                        )
                    )
                ]
            )
        ],
    )
    return np.asarray(response.embeddings[0].values, dtype=np.float32)


def embed_content_with_retry(client: Any, contents: list[Any], attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            sleep_seconds = 2 ** attempt
            print(f"Gemini embed_content failed; retrying in {sleep_seconds}s: {exc}", flush=True)
            time.sleep(sleep_seconds)
    raise RuntimeError("Gemini embed_content failed after retries.") from last_error


def score_params(
    client: Any,
    text_embedding: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    params: FxParams,
    temp_dir: Path,
    index: int,
) -> float:
    rendered = render_audio(audio, sample_rate, params)
    wav_path = temp_dir / f"candidate_{index:04d}.wav"
    sf.write(wav_path, rendered.T if rendered.ndim == 2 else rendered, sample_rate)
    audio_embedding = embed_wav(client, wav_path)
    return cosine(audio_embedding, text_embedding)


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM response did not contain JSON: {text}")
    return json.loads(match.group(0))


def propose_params_with_llm(client: Any, prompt: str) -> FxParams:
    schema = {
        "low_gain_db": "float -9..9, low shelf at 180 Hz",
        "mid_gain_db": "float -9..9, peak EQ at 1200 Hz",
        "high_gain_db": "float -9..9, high shelf at 4500 Hz",
        "reverb_room_size": "float 0..1",
        "reverb_wet_level": "float 0..0.35",
        "saturation_drive_db": "float 0..18",
        "delay_mix": "float 0..0.25",
        "compressor_threshold_db": "float -28..-8",
    }
    instruction = f"""
You are an expert audio engineer. Convert the requested sound change into one conservative audio effects preset.

Effect chain, in order:
1. low shelf EQ at 180 Hz
2. mid peak EQ at 1200 Hz
3. high shelf EQ at 4500 Hz
4. saturation/distortion
5. compressor, ratio fixed at 2:1
6. delay, fixed quarter-second delay
7. reverb
8. limiter

Return only a JSON object with these exact keys and numeric ranges:
{json.dumps(schema, indent=2)}

Use subtle-to-moderate values. Avoid extreme settings unless the prompt asks for an obvious special effect.

Prompt: {prompt}
"""
    response = client.models.generate_content(model=LLM_MODEL, contents=instruction)
    payload = extract_json_object(response.text or "")
    return sanitize_params(FxParams(**{field: float(payload[field]) for field in FxParams.__dataclass_fields__}))


def candidate_vectors(seed: np.ndarray, trials: int, random_trials: int, spread: float) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    vectors = [np.clip(seed, 0, 1)]
    for _ in range(max(0, trials - 1)):
        perturbation = rng.normal(0, spread, size=seed.shape)
        vectors.append(np.clip(seed + perturbation, 0, 1))
    for _ in range(random_trials):
        vectors.append(rng.random(seed.shape[0]))
    return vectors


def hybrid_search(
    client: Any,
    audio: np.ndarray,
    sample_rate: int,
    prompt: str,
    initial_params: FxParams,
    refine_trials: int,
    random_trials: int,
    spread: float,
) -> tuple[FxParams, float, list[dict]]:
    text_embedding = embed_text(client, f"This sound is {prompt}")
    best_params: FxParams | None = None
    best_score = -math.inf
    history: list[dict] = []
    seed = params_to_vector(initial_params)
    vectors = candidate_vectors(seed, refine_trials, random_trials, spread)

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        for i, vector in enumerate(vectors):
            params = vector_to_params(vector)
            score = score_params(client, text_embedding, audio, sample_rate, params, temp_dir, i)
            result = {"trial": i, "score": score, "params": asdict(params)}
            history.append(result)
            if score > best_score:
                best_score = score
                best_params = params
                print(f"trial={i + 1}/{len(vectors)} score={score:.4f} params={json.dumps(asdict(params))}", flush=True)

    if best_params is None:
        raise RuntimeError("No candidates were scored.")
    return best_params, best_score, history


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.T.astype(np.float32), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refine-trials", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--spread", type=float, default=0.16)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")

    load_runtime_dependencies()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    audio, sample_rate = load_audio(args.input)
    proposed_params = propose_params_with_llm(client, args.prompt)
    print(f"llm_proposal={json.dumps(asdict(proposed_params))}", flush=True)

    if args.no_refine:
        with tempfile.TemporaryDirectory() as tmp:
            text_embedding = embed_text(client, f"This sound is {args.prompt}")
            score = score_params(client, text_embedding, audio, sample_rate, proposed_params, Path(tmp), 0)
        params = proposed_params
        history = [{"trial": 0, "score": score, "params": asdict(params)}]
    else:
        params, score, history = hybrid_search(
            client,
            audio,
            sample_rate,
            args.prompt,
            proposed_params,
            args.refine_trials,
            args.random_trials,
            args.spread,
        )
    rendered = render_audio(audio, sample_rate, params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, rendered.T, sample_rate)
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": LLM_MODEL,
                "llm_proposal": asdict(proposed_params),
                "best_score": score,
                "best_params": asdict(params),
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
