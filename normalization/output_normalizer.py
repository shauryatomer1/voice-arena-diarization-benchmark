"""Convert a model's raw output to the common prediction schema and to RTTM.

Normalisation rules (deliberately minimal):
  * segments are sorted by (start, end); NOTHING is merged, split, clipped or dropped, so overlaps survive
  * timestamps are kept as full-precision floats in JSON (no rounding)
  * speaker labels are renamed speaker_0, speaker_1, ... by first appearance (labels are arbitrary
    per recording; we never try to match them to reference labels). The original->normalised map is stored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from models.base_model import RawSegment
from preprocessing.rttm import write_rttm

SCHEMA_VERSION = "1.0"


def normalize_output(recording_id: str, model_name: str, model_version: str,
                     segments: list[RawSegment], experiment_id: str | None = None) -> dict[str, Any]:
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
    label_map: dict[str, str] = {}
    out_segments = []
    for s in ordered:
        if s.speaker not in label_map:
            label_map[s.speaker] = f"speaker_{len(label_map)}"
        out_segments.append({"start": float(s.start), "end": float(s.end), "speaker": label_map[s.speaker]})
    return {
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording_id,
        "model_name": model_name,
        "model_version": model_version,
        "experiment_id": experiment_id,
        "num_speakers": len(label_map),
        "speaker_label_map": label_map,
        "segments": out_segments,
    }


def prediction_to_dataframe(pred: dict[str, Any]) -> pd.DataFrame:
    rows = [{"file_id": pred["recording_id"], "channel": "1", "start": s["start"],
             "duration": s["end"] - s["start"], "speaker": s["speaker"]} for s in pred["segments"]]
    return pd.DataFrame(rows, columns=["file_id", "channel", "start", "duration", "speaker"])


def prediction_to_rttm(pred: dict[str, Any], path: Path, decimals: int = 3) -> None:
    """Write RTTM (millisecond resolution, the standard). An empty prediction yields an empty file."""
    path = Path(path)
    if not pred["segments"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    write_rttm(prediction_to_dataframe(pred), path, decimals)
