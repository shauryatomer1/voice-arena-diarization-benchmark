"""RTTM reading, writing and statistics (original files are never modified)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
COLUMNS = ["type", "file_id", "channel", "start", "duration", "ortho",
           "stype", "speaker", "conf", "slat"]


class RTTMError(ValueError):
    """Raised when an RTTM file is malformed."""


def read_rttm(path: Path) -> pd.DataFrame:
    """Parse an RTTM file into a DataFrame with an extra `end` column.

    Blank lines and ';;' comments are ignored. Any other malformed line raises
    RTTMError, so nothing is skipped silently.
    """
    path = Path(path)
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 9:
            raise RTTMError(f"{path.name}:{n}: expected >=9 fields, got {len(parts)}")
        if parts[0] != "SPEAKER":
            raise RTTMError(f"{path.name}:{n}: unsupported record type {parts[0]!r}")
        try:
            start, dur = float(parts[3]), float(parts[4])
        except ValueError as exc:
            raise RTTMError(f"{path.name}:{n}: non-numeric time ({exc})") from exc
        rows.append([parts[0], parts[1], parts[2], start, dur, parts[5], parts[6],
                     parts[7], parts[8], parts[9] if len(parts) > 9 else "<NA>"])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["end"] = df["start"] + df["duration"]
    return df


def normalize_rttm(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Return a cleaned copy: channel 1, placeholder fields, rounded times.

    Segments are neither merged nor dropped, so overlapping speech is preserved.
    """
    out = df.copy()
    out["channel"] = "1"
    for col in ("ortho", "stype", "conf", "slat"):
        out[col] = "<NA>"
    out["start"] = out["start"].round(decimals)
    out["duration"] = out["duration"].round(decimals)
    out["end"] = out["start"] + out["duration"]
    return out.reset_index(drop=True)


def write_rttm(df: pd.DataFrame, path: Path, decimals: int = 3) -> None:
    """Write segments as RTTM, sorted by start time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"SPEAKER {r.file_id} {r.channel} {r.start:.{decimals}f} "
             f"{r.duration:.{decimals}f} <NA> <NA> {r.speaker} <NA> <NA>"
             for r in df.sort_values(["start", "speaker"]).itertuples()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class RTTMStats:
    n_segments: int
    n_speakers: int
    speech_s: float        # union of all speech
    overlap_s: float       # time with >= 2 simultaneous speakers
    overlap_ratio: float   # overlap_s / speech_s


def compute_stats(df: pd.DataFrame) -> RTTMStats:
    """Speaker count, speech time and overlap time via a boundary sweep.

    Overlap is counted by segments, so two overlapping segments of the same speaker
    would also count (validation flags those separately as duplicates).
    """
    if df.empty:
        return RTTMStats(0, 0, 0.0, 0.0, 0.0)
    events = sorted([(s, 1) for s in df["start"]] + [(e, -1) for e in df["end"]])
    active, last, speech, overlap = 0, events[0][0], 0.0, 0.0
    for t, d in events:
        if active >= 1:
            speech += t - last
        if active >= 2:
            overlap += t - last
        active += d
        last = t
    return RTTMStats(len(df), df["speaker"].nunique(), speech, overlap,
                     overlap / speech if speech else 0.0)
