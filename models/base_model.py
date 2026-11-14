"""Common interface for diarization systems.

A model receives ONLY an audio path (plus its own configuration). It must never receive reference
annotations, transcripts, or the reference speaker count in the primary track.
"""
from __future__ import annotations

import abc
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawSegment:
    start: float
    end: float
    speaker: str


@dataclass
class RawOutput:
    """What a model produced, before normalisation."""
    segments: list[RawSegment]
    native: dict[str, Any] = field(default_factory=dict)   # model-specific raw artefacts (JSON-serialisable)
    inference_time_s: float | None = None                  # pure model compute, excluding model load
    notes: list[str] = field(default_factory=list)


class BaseDiarizationModel(abc.ABC):
    """Interface every diarization system implements."""

    name: str = "base"

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.loaded = False
        self.load_time_s: float | None = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load weights / pipeline. Must set self.loaded = True."""

    @abc.abstractmethod
    def diarize(self, audio_path: Path) -> RawOutput:
        """Run diarization on one audio file and return RawOutput."""

    @abc.abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        """Exact model identity/versions/settings actually used (never guessed)."""

    def unload(self) -> None:  # optional hook
        self.loaded = False

    # ---- shared helpers ----
    def environment_metadata(self) -> dict[str, Any]:
        """Python / torch / CUDA / device facts for the experiment manifest."""
        meta: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
        try:
            import torch
            meta.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                        cuda_available=torch.cuda.is_available(),
                        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        except Exception as exc:  # torch missing -> recorded, not hidden
            meta["torch_error"] = repr(exc)
        return meta

    def resolve_device(self) -> str:
        want = self.config.get("device", "auto")
        if want != "auto":
            return want
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
