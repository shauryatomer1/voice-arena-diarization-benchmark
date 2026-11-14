"""Automated data validation for audio + reference annotations.

Each check yields a `CheckResult`; failures are always reported, never skipped.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from preprocessing.audio_processor import AudioError, get_audio_metadata, validate_audio
from preprocessing.rttm import RTTMError, read_rttm

LOGGER = logging.getLogger(__name__)
SPEAKER_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


@dataclass
class CheckResult:
    recording_id: str
    check: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _r(rid: str, check: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(rid, check, bool(ok), detail if not ok else (detail or "ok"))


def check_supported_extension(rid: str, path: Path, allowed: list[str]) -> CheckResult:
    return _r(rid, "audio_extension_supported", path.suffix.lower() in allowed,
              f"{path.suffix!r} not in {allowed}")


def check_audio(rid: str, path: Path, cfg_audio: dict) -> list[CheckResult]:
    """Existence, readability, format, sample rate, channels, emptiness, corruption."""
    if not path.is_file():
        return [_r(rid, "audio_exists", False, f"missing: {path}")]
    problems = validate_audio(path, cfg_audio["target_sample_rate"], cfg_audio["target_channels"],
                              cfg_audio["target_subtype"], cfg_audio["min_duration_s"])
    out = [_r(rid, "audio_exists", True)]
    out.append(_r(rid, "audio_valid_format_16k_mono_pcm16_nonempty_uncorrupted", not problems, "; ".join(problems)))
    return out


def check_reference(rid: str, path: Path, duration_s: float | None, tol_s: float,
                    label: str = "reference") -> list[CheckResult]:
    """Structural checks on one RTTM file."""
    if not path.is_file():
        return [_r(rid, f"{label}_exists", False, f"missing: {path}")]
    out = [_r(rid, f"{label}_exists", True)]
    try:
        df = read_rttm(path)
    except RTTMError as exc:
        return out + [_r(rid, f"{label}_parses", False, str(exc))]
    out.append(_r(rid, f"{label}_parses", not df.empty, "no segments"))
    if df.empty:
        return out
    bad_ids = sorted(set(df["file_id"]) - {rid})
    out.append(_r(rid, f"{label}_recording_id_consistent", not bad_ids, f"unexpected ids: {bad_ids}"))
    out.append(_r(rid, f"{label}_timestamps_valid",
                  bool(((df["start"] >= 0) & df["start"].notna() & df["duration"].notna()).all()),
                  "negative or missing start/duration"))
    n_nonpos = int((df["duration"] <= 0).sum())
    out.append(_r(rid, f"{label}_durations_positive", n_nonpos == 0, f"{n_nonpos} non-positive durations"))
    bad_spk = sorted(s for s in df["speaker"].unique() if not SPEAKER_RE.match(s) or s == "<NA>")
    out.append(_r(rid, f"{label}_speaker_ids_valid", not bad_spk, f"invalid ids: {bad_spk}"))
    dups = int(df.duplicated(subset=["speaker", "start", "duration"]).sum())
    out.append(_r(rid, f"{label}_no_duplicate_segments", dups == 0, f"{dups} duplicates"))
    if duration_s is not None:
        over = df[df["end"] > duration_s + tol_s]
        out.append(_r(rid, f"{label}_within_audio_duration", over.empty,
                      f"{len(over)} segments end after {duration_s:.2f}s (max end {df['end'].max():.2f}s)"))
    return out


def check_overlap_preserved(rid: str, original: Path, normalized: Path, decimals: int) -> CheckResult:
    """Normalised RTTM must contain exactly the same segments (after rounding) as the original."""
    if not (original.is_file() and normalized.is_file()):
        return _r(rid, "overlap_preserved_in_normalized", False, "original or normalized RTTM missing")
    o, n = read_rttm(original), read_rttm(normalized)
    key = lambda d: sorted(zip(d["speaker"], d["start"].round(decimals), d["duration"].round(decimals)))
    same = len(o) == len(n) and key(o) == key(n)
    return _r(rid, "overlap_preserved_in_normalized", same, f"segments original={len(o)} normalized={len(n)}")


def check_manifest_ids(manifest: pd.DataFrame, audio_dir: Path, ref_dir: Path) -> list[CheckResult]:
    """Every manifest ID has audio+reference, and no stray files exist without a manifest entry."""
    ids = set(manifest["recording_id"])
    audio_ids = {p.stem for p in audio_dir.glob("*.wav")}
    ref_ids = {p.stem for p in ref_dir.glob("*.rttm")}
    return [
        _r("ALL", "manifest_ids_match_audio", ids == audio_ids, f"manifest-only={ids - audio_ids} audio-only={audio_ids - ids}"),
        _r("ALL", "manifest_ids_match_references", ids == ref_ids, f"manifest-only={ids - ref_ids} ref-only={ref_ids - ids}"),
    ]


def validate_all(manifest: pd.DataFrame, root: Path, cfg: dict) -> list[CheckResult]:
    """Run every check for every recording in the manifest."""
    results: list[CheckResult] = []
    tol, dec = cfg["rttm"]["duration_tolerance_s"], cfg["rttm"]["normalized_decimals"]
    for row in manifest.itertuples():
        rid = row.recording_id
        audio = root / row.audio_filename
        results.append(check_supported_extension(rid, audio, cfg["audio"]["allowed_extensions"]))
        results += check_audio(rid, audio, cfg["audio"])
        duration = None
        if audio.is_file():
            try:
                duration = get_audio_metadata(audio).duration_s
            except AudioError:
                pass
        results += check_reference(rid, root / row.annotation_filename, duration, tol, "normalized_ref")
        results += check_reference(rid, root / row.original_annotation_filename, duration, tol, "original_ref")
        results.append(check_overlap_preserved(rid, root / row.original_annotation_filename,
                                               root / row.annotation_filename, dec))
    results += check_manifest_ids(manifest, root / cfg["paths"]["processed_audio"],
                                  root / cfg["paths"]["ref_normalized"])
    return results
