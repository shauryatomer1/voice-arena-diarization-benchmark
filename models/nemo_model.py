"""NVIDIA NeMo cascaded diarizer: MarbleNet VAD -> TitaNet multiscale embeddings -> NME-SC clustering -> MSDD.

The framework YAML (config/nemo/diar_infer_meeting.yaml, from NeMo tag in config/nemo/SOURCE_TAG.txt) is NeMo's
own recommended inference config for 3-5 speaker meetings and is kept unchanged except for the checkpoint names,
paths, and device. No reference RTTM / speaker count is written into the input manifest.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from preprocessing.common import ROOT

from .base_model import BaseDiarizationModel, RawOutput, RawSegment

LOGGER = logging.getLogger(__name__)


class NemoMSDDModel(BaseDiarizationModel):

    def load_model(self) -> None:
        import nemo
        from omegaconf import OmegaConf

        t0 = time.perf_counter()
        self.device = self.resolve_device()
        cfg = OmegaConf.load(ROOT / self.config["framework_config"])
        pre, p = self.config["pretrained"], self.config["params"]
        cfg.device = self.device
        cfg.diarizer.vad.model_path = pre["vad_model"]
        cfg.diarizer.speaker_embeddings.model_path = pre["speaker_embedding_model"]
        cfg.diarizer.msdd_model.model_path = pre["msdd_model"]
        cfg.diarizer.clustering.parameters.oracle_num_speakers = bool(p["oracle_num_speakers"])
        if p["oracle_num_speakers"] and self.config.get("track") == "primary":
            raise RuntimeError("oracle_num_speakers must be false in the primary track (no reference leakage)")
        cfg.diarizer.clustering.parameters.max_num_speakers = int(p["max_num_speakers"])
        cfg.diarizer.oracle_vad = False
        cfg.diarizer.asr.model_path = None           # ASR not used: pure diarization
        cfg.verbose = False
        self.cfg = cfg
        self.framework_version = nemo.__version__
        self.load_time_s = time.perf_counter() - t0
        self.loaded = True

    def diarize(self, audio_path: Path) -> RawOutput:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        from omegaconf import OmegaConf
        import torch

        if not self.loaded:
            raise RuntimeError("load_model() has not been called")
        work = Path(tempfile.mkdtemp(prefix="nemo_diar_"))
        try:
            manifest = work / "input_manifest.json"
            # Input contract: path only. No RTTM, UEM, transcript or speaker count.
            manifest.write_text(json.dumps({"audio_filepath": str(audio_path.resolve()), "offset": 0,
                                            "duration": None, "label": "infer", "text": "-",
                                            "num_speakers": None, "rttm_filepath": None,
                                            "uem_filepath": None}) + "\n", encoding="utf-8")
            cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
            cfg.diarizer.manifest_filepath = str(manifest)
            cfg.diarizer.out_dir = str(work / "out")
            t0_total = time.perf_counter()
            diarizer = NeuralDiarizer(cfg=cfg)
            setup_s = time.perf_counter() - t0_total      # builds VAD/TitaNet/MSDD (+ first-time checkpoint load)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            diarizer.diarize()
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rttm = work / "out" / "pred_rttms" / f"{audio_path.stem}.rttm"
            segs = _read_native_rttm(rttm)
            native = {"native_rttm": rttm.read_text(encoding="utf-8"),
                      "per_recording_model_setup_s": round(setup_s, 3)}
            notes = [f"inference_time_s excludes per-recording NeuralDiarizer construction ({setup_s:.1f}s) "
                     "which re-instantiates the three networks"]
            return RawOutput(segs, native, elapsed, notes)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def get_model_metadata(self) -> dict[str, Any]:
        c = self.config
        cfg = getattr(self, "cfg", None)
        from_cfg = (lambda path: None) if cfg is None else (lambda path: _get(cfg, path))
        return {
            "provider": c["provider"], "model_name": self.name, "framework": "NVIDIA NeMo",
            "framework_version": getattr(self, "framework_version", None),
            "revision": f"nemo_toolkit=={getattr(self, 'framework_version', '?')}; "
                        f"msdd={c['pretrained']['msdd_model']}; embedding={c['pretrained']['speaker_embedding_model']}; "
                        f"vad={c['pretrained']['vad_model']}",
            "pretrained_checkpoints": c["pretrained"],
            "framework_config": c["framework_config"],
            "framework_config_source_tag": (ROOT / "config/nemo/SOURCE_TAG.txt").read_text().strip(),
            "input_sample_rate": c["input"]["sample_rate"], "input_channels": c["input"]["channels"],
            "inference_params": c["params"],
            "vad_settings": from_cfg("diarizer.vad.parameters"),
            "clustering_settings": from_cfg("diarizer.clustering.parameters"),
            "embedding_settings": from_cfg("diarizer.speaker_embeddings.parameters"),
            "msdd_settings": from_cfg("diarizer.msdd_model.parameters"),
            "device": getattr(self, "device", None), "load_time_s": self.load_time_s,
            **self.environment_metadata(),
        }


def _get(cfg, dotted: str):
    from omegaconf import OmegaConf
    node = cfg
    for k in dotted.split("."):
        node = node[k]
    return OmegaConf.to_container(node, resolve=True)


def _read_native_rttm(path: Path) -> list[RawSegment]:
    """Read NeMo's own RTTM output (SPEAKER <id> 1 start dur <NA> <NA> speaker_N <NA> <NA>)."""
    if not path.exists():
        raise FileNotFoundError(f"NeMo produced no RTTM at {path}")
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            segs.append(RawSegment(start, start + dur, parts[7]))
    return segs
