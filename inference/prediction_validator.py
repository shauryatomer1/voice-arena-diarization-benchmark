"""Validate normalised predictions (schema, timestamps, ids) and exported RTTM files."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from preprocessing.rttm import RTTMError, read_rttm

REQUIRED = ("recording_id", "model_name", "model_version", "segments")


def validate_prediction(pred: dict[str, Any], recording_id: str, audio_duration_s: float | None,
                        tol_s: float = 0.05) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors make the prediction invalid; an empty prediction is a warning."""
    errors: list[str] = []
    warnings: list[str] = []
    for k in REQUIRED:
        if k not in pred:
            errors.append(f"missing key {k!r}")
    if errors:
        return errors, warnings
    if pred["recording_id"] != recording_id:
        errors.append(f"recording_id {pred['recording_id']!r} != expected {recording_id!r}")
    if not pred["model_version"]:
        errors.append("model_version is empty")
    segs = pred["segments"]
    if not isinstance(segs, list):
        return errors + ["segments is not a list"], warnings
    if not segs:
        warnings.append("empty prediction (no speech detected)")
    for i, s in enumerate(segs):
        try:
            a, b, spk = s["start"], s["end"], s["speaker"]
        except (KeyError, TypeError):
            errors.append(f"segment {i}: needs start/end/speaker")
            continue
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)):
            errors.append(f"segment {i}: non-finite or non-numeric time")
            continue
        if a < 0:
            errors.append(f"segment {i}: negative start {a}")
        if b <= a:
            errors.append(f"segment {i}: end {b} <= start {a}")
        if audio_duration_s is not None and b > audio_duration_s + tol_s:
            errors.append(f"segment {i}: end {b:.3f} beyond audio duration {audio_duration_s:.3f}")
        if not isinstance(spk, str) or not spk:
            errors.append(f"segment {i}: invalid speaker {spk!r}")
    if len(errors) > 20:
        errors = errors[:20] + [f"... {len(errors) - 20} more errors"]
    return errors, warnings


def validate_rttm_against_prediction(rttm_path: Path, pred: dict[str, Any], tol_s: float = 0.0015) -> list[str]:
    """RTTM must contain exactly the prediction's segments (to ms rounding) with the same recording ID."""
    errors: list[str] = []
    try:
        df = read_rttm(rttm_path)
    except (RTTMError, OSError) as exc:
        return [f"RTTM unreadable: {exc}"]
    if len(df) != len(pred["segments"]):
        errors.append(f"RTTM has {len(df)} segments, prediction has {len(pred['segments'])}")
        return errors
    if len(df) and set(df["file_id"]) != {pred["recording_id"]}:
        errors.append(f"RTTM file ids {sorted(set(df['file_id']))} != {pred['recording_id']!r}")
    exp = sorted((s["start"], s["end"], s["speaker"]) for s in pred["segments"])
    got = sorted(zip(df["start"], df["end"], df["speaker"]))
    for (a, b, sp), (c, d, sq) in zip(exp, got):
        if abs(a - c) > tol_s or abs(b - d) > 2 * tol_s or sp != sq:
            errors.append(f"RTTM mismatch: expected {(a, b, sp)} got {(c, d, sq)}")
            break
    return errors
