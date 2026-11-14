"""pyannote.audio integration (Community-1 by default; id/revision come from model_config.yaml)."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from .base_model import BaseDiarizationModel, RawOutput, RawSegment

LOGGER = logging.getLogger(__name__)


class PyannoteModel(BaseDiarizationModel):
    """Wraps `pyannote.audio.Pipeline`. Audio is decoded with SoundFile and passed as a waveform
    tensor, so no FFmpeg/torchcodec is needed and every model decodes the same samples."""

    def load_model(self) -> None:
        import huggingface_hub
        import pyannote.audio
        import torch
        from pyannote.audio import Pipeline

        cfg = self.config
        env_var = cfg["auth"]["env_var"]
        token = os.environ.get(env_var)
        if not token:
            raise RuntimeError(
                f"{env_var} is not set. {cfg['model_id']} is gated: accept its conditions on Hugging Face "
                f"and put a read token in .env as {env_var}=...")
        model_id = cfg["model_id"]
        revision = cfg.get("revision")
        if not revision:  # pin the exact commit actually used
            revision = huggingface_hub.model_info(model_id, token=token).sha
        self.resolved_revision = revision
        t0 = time.perf_counter()
        pipeline = Pipeline.from_pretrained(model_id, revision=revision, token=token)
        if pipeline is None:
            raise RuntimeError(f"Pipeline.from_pretrained returned None for {model_id} (access not granted?)")
        self.device = self.resolve_device()
        pipeline.to(torch.device(self.device))
        self.pipeline = pipeline
        self.framework_version = pyannote.audio.__version__
        self.load_time_s = time.perf_counter() - t0
        self.loaded = True
        LOGGER.info("Loaded %s@%s on %s in %.1fs", model_id, revision[:8], self.device, self.load_time_s)

    def diarize(self, audio_path: Path) -> RawOutput:
        import torch
        if not self.loaded:
            raise RuntimeError("load_model() has not been called")
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)   # (time, channel)
        wav = torch.from_numpy(data.T.copy())                                   # (channel, time)
        p = self.config.get("params", {})
        kwargs = {k: p[k] for k in ("num_speakers", "min_speakers", "max_speakers") if p.get(k) is not None}
        if self.config.get("track", "primary") == "primary" and "num_speakers" in kwargs:
            raise RuntimeError("num_speakers must be null in the primary track (no reference leakage)")
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.pipeline({"waveform": wav, "sample_rate": sr, "uri": audio_path.stem}, **kwargs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ann = getattr(out, self.config["output_field"], out)   # older pipelines return a bare Annotation
        segs = [RawSegment(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]
        excl = getattr(out, "exclusive_speaker_diarization", None)
        native = {"output_field": self.config["output_field"],
                  "exclusive_speaker_diarization": [
                      {"start": float(t.start), "end": float(t.end), "speaker": str(s)}
                      for t, _, s in excl.itertracks(yield_label=True)] if excl is not None else None}
        return RawOutput(segs, native, elapsed)

    def get_model_metadata(self) -> dict[str, Any]:
        c = self.config
        return {
            "provider": c["provider"], "model_name": self.name, "model_id": c["model_id"],
            "revision": getattr(self, "resolved_revision", c.get("revision")),
            "framework": "pyannote.audio", "framework_version": getattr(self, "framework_version", None),
            "licence": c["auth"].get("licence"), "gated": c["auth"].get("gated"),
            "input_sample_rate": c["input"]["sample_rate"], "input_channels": c["input"]["channels"],
            "inference_params": c.get("params", {}), "output_field": c["output_field"],
            "vad_settings": "internal to pipeline (pyannote segmentation-3.0 powerset; defaults kept)",
            "clustering_settings": "internal to pipeline (VBx-style clustering; defaults kept)",
            "device": getattr(self, "device", None), "load_time_s": self.load_time_s,
            **self.environment_metadata(),
        }
