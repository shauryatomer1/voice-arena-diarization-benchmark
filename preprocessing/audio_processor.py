"""Audio inspection and standardisation to WAV / PCM-16 / 16 kHz / mono.

SoundFile does the I/O (no FFmpeg needed). Raw audio is never modified: every
function writes to a new path, and `convert_to_wav` refuses to overwrite its source.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Raised when audio cannot be read, is empty, or is invalid."""


@dataclass
class AudioMetadata:
    path: str
    format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    file_size_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_audio(path: Path) -> sf.Info:
    """Return the SoundFile header info; raise AudioError if missing/unreadable/unsupported."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Audio file not found: {path}")
    try:
        return sf.info(str(path))
    except Exception as exc:
        raise AudioError(f"Cannot read {path.name}: {exc}") from exc


def get_audio_metadata(path: Path) -> AudioMetadata:
    """Collect metadata from the file header (no full decode)."""
    info = inspect_audio(path)
    return AudioMetadata(str(path), info.format, info.subtype, info.samplerate,
                         info.channels, info.frames, info.frames / info.samplerate,
                         Path(path).stat().st_size)


def convert_to_mono(data: np.ndarray) -> np.ndarray:
    """Average channels to mono; 1-D input is returned unchanged.

    This is a down-mix. It is NOT equivalent to evaluating a channel-aware system.
    """
    return data if data.ndim == 1 else data.mean(axis=1)


def resample_audio(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample 1-D audio with a polyphase anti-aliasing filter."""
    if sr_in == sr_out:
        return data
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(data, sr_out // g, sr_in // g).astype(np.float32)


def convert_to_wav(src: Path, dst: Path, sample_rate: int = 16000, channels: int = 1,
                   subtype: str = "PCM_16", overwrite: bool = False) -> AudioMetadata:
    """Write `src` as WAV PCM-16 at `sample_rate` (mono when channels == 1).

    Refuses to write onto the source or an existing file unless overwrite=True.
    Every operation applied (downmix / resample) is logged.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise AudioError("Refusing to overwrite the source audio file")
    if dst.exists() and not overwrite:
        raise AudioError(f"Destination exists: {dst}")
    info = inspect_audio(src)
    if info.frames == 0:
        raise AudioError(f"Empty audio: {src.name}")
    data, sr = sf.read(str(src), dtype="float32")
    ops = []
    if info.channels > 1 and channels == 1:
        data = convert_to_mono(data)
        ops.append(f"downmix {info.channels}ch->mono (mean)")
    if sr != sample_rate:
        data = resample_audio(data, sr, sample_rate)
        ops.append(f"resample {sr}->{sample_rate} Hz (polyphase)")
    data = np.clip(data, -1.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sample_rate, subtype=subtype, format="WAV")
    LOGGER.info("%s -> %s | ops: %s", src.name, dst.name, "; ".join(ops) or "none (re-encode to PCM_16 only)")
    return get_audio_metadata(dst)


def _wav_truncated(path: Path) -> str | None:
    """Detect a RIFF/WAVE file whose `data` chunk declares more bytes than the file holds.

    libsndfile silently clamps such files, so a truncated download would otherwise pass.
    """
    import struct
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return None
        pos = 12
        while pos + 8 <= size:
            fh.seek(pos)
            cid, csz = struct.unpack("<4sI", fh.read(8))
            if cid == b"data":
                if csz != 0xFFFFFFFF and pos + 8 + csz > size:
                    return f"data chunk declares {csz} bytes but only {size - pos - 8} present (truncated)"
                return None
            pos += 8 + csz + (csz & 1)
    return None


def validate_audio(path: Path, sample_rate: int = 16000, channels: int = 1,
                   subtype: str = "PCM_16", min_duration_s: float = 1.0) -> list[str]:
    """Return a list of problems (empty == valid). Fully decodes to detect corruption."""
    try:
        info = inspect_audio(path)
    except AudioError as exc:
        return [str(exc)]
    problems: list[str] = []
    truncated = _wav_truncated(Path(path))
    if truncated:
        problems.append(truncated)
    if info.format != "WAV":
        problems.append(f"format is {info.format}, expected WAV")
    if info.subtype != subtype:
        problems.append(f"subtype is {info.subtype}, expected {subtype}")
    if info.samplerate != sample_rate:
        problems.append(f"sample rate is {info.samplerate}, expected {sample_rate}")
    if info.channels != channels:
        problems.append(f"channels is {info.channels}, expected {channels}")
    if info.frames == 0:
        problems.append("audio is empty")
        return problems
    if info.frames / info.samplerate < min_duration_s:
        problems.append(f"duration {info.frames / info.samplerate:.2f}s below minimum {min_duration_s}s")
    try:
        n, peak = 0, 0.0
        for block in sf.blocks(str(path), blocksize=1 << 20, dtype="float32"):
            n += len(block)
            if not np.isfinite(block).all():
                problems.append("non-finite samples found")
                break
            peak = max(peak, float(np.abs(block).max()))
        if n != info.frames:
            problems.append(f"decoded {n} frames but header says {info.frames}")
        if peak == 0.0:
            problems.append("audio is entirely silent")
    except Exception as exc:
        problems.append(f"decode failed: {exc}")
    return problems
