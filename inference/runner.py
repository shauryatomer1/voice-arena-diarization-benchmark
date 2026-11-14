"""Single-model inference runner (runs inside the model's own virtualenv).

Pipeline per recording: read Phase-1 audio -> diarize (audio path only) -> save raw output ->
normalise -> validate -> export RTTM -> record performance. A failure is logged and recorded and
the run continues with the next recording; nothing is silently skipped.
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from inference.experiment import (PerfMonitor, hardware_info, load_dotenv, new_experiment_id, sha256_file,
                                  software_versions)
from inference.prediction_validator import validate_prediction, validate_rttm_against_prediction
from models.base_model import BaseDiarizationModel
from models.model_registry import create_model, load_model_config
from normalization.output_normalizer import normalize_output, prediction_to_rttm
from preprocessing.audio_processor import get_audio_metadata
from preprocessing.common import ROOT, load_yaml

LOGGER = logging.getLogger(__name__)


def save_prediction(pred: dict[str, Any], path: Path) -> str:
    """Write JSON (never overwrites an existing file) and return its SHA-256."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pred, indent=2), encoding="utf-8")
    return sha256_file(path)


def load_prediction(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_prediction_file(path: Path, recording_id: str, duration_s: float | None) -> tuple[list[str], list[str]]:
    return validate_prediction(load_prediction(path), recording_id, duration_s)


def _paths(cfg: dict, mcfg: dict, exp_id: str, model_name: str, rid: str) -> dict[str, Path]:
    root = ROOT / cfg["output"]["root"]
    return {"raw": root / mcfg["raw_dir"] / exp_id / f"{rid}.raw.json",
            "normalized": root / "normalized" / exp_id / model_name / f"{rid}.json",
            "rttm": root / "rttm" / exp_id / model_name / f"{rid}.rttm"}


def run_on_recording(model: BaseDiarizationModel, recording_id: str, audio_path: Path, exp_id: str,
                     cfg: dict | None = None) -> dict[str, Any]:
    """Run one recording. Always returns a result dict (status success|success_empty|failed)."""
    cfg = cfg or load_model_config()
    paths = _paths(cfg, model.config, exp_id, model.name, recording_id)
    res: dict[str, Any] = {"recording_id": recording_id, "model_name": model.name, "status": "failed",
                           "errors": [], "warnings": [], "audio_sha256": None}
    try:
        meta = get_audio_metadata(audio_path)          # used only for duration bookkeeping, not given to the model
        res.update(audio_duration_s=meta.duration_s, audio_sample_rate=meta.sample_rate,
                   audio_channels=meta.channels, audio_sha256=sha256_file(audio_path))
        if (meta.sample_rate, meta.channels) != (model.config["input"]["sample_rate"], model.config["input"]["channels"]):
            raise ValueError(f"audio {meta.sample_rate} Hz/{meta.channels} ch does not match model input contract")
        with PerfMonitor() as perf:
            raw = model.diarize(audio_path)            # ONLY the audio path crosses the model boundary
        inf = raw.inference_time_s if raw.inference_time_s is not None else perf.wall_s
        version = model.get_model_metadata().get("revision") or model.get_model_metadata().get("model_id")
        raw_doc = {"recording_id": recording_id, "model_name": model.name, "model_version": version,
                   "experiment_id": exp_id, "native": raw.native, "notes": raw.notes,
                   "segments": [asdict(s) for s in raw.segments]}
        res["raw_sha256"] = save_prediction(raw_doc, paths["raw"])
        pred = normalize_output(recording_id, model.name, str(version), raw.segments, exp_id)
        res["normalized_sha256"] = save_prediction(pred, paths["normalized"])
        errs, warns = validate_prediction(pred, recording_id, meta.duration_s)
        prediction_to_rttm(pred, paths["rttm"])
        errs += validate_rttm_against_prediction(paths["rttm"], pred)
        res["rttm_sha256"] = sha256_file(paths["rttm"])
        res.update(
            inference_time_s=round(inf, 3), total_wall_time_s=round(perf.wall_s, 3),
            rtf=round(inf / meta.duration_s, 4),   # RTF = processing time / audio duration
            processing_speed_x_realtime=round(meta.duration_s / inf, 2) if inf > 0 else None,
            num_predicted_speakers=pred["num_speakers"], num_predicted_segments=len(pred["segments"]),
            **perf.summary(), errors=errs, warnings=warns + raw.notes)
        res["status"] = "failed" if errs else ("success_empty" if not pred["segments"] else "success")
    except Exception as exc:
        res["errors"].append(f"{type(exc).__name__}: {exc}")
        res["traceback"] = traceback.format_exc()
        LOGGER.error("FAILED %s / %s: %s", model.name, recording_id, exc)
    if res["status"] == "failed":
        LOGGER.error("%s / %s FAILED: %s", model.name, recording_id, res["errors"])
    else:
        LOGGER.info("%s / %s %s: %.1fs inference, RTF %.3f, %d speakers, %d segments", model.name, recording_id,
                    res["status"], res["inference_time_s"], res["rtf"], res["num_predicted_speakers"],
                    res["num_predicted_segments"])
    res["paths"] = {k: _rel(v) for k, v in paths.items() if v.exists()}
    return res


def _rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:  # outside the project (e.g. tests using a temp dir)
        return str(p)


def load_recordings(cfg: dict, only: list[str] | None = None) -> list[tuple[str, Path]]:
    """Recording IDs + processed audio paths from the Phase-1 manifest. Annotations are NOT read."""
    manifest = pd.read_csv(ROOT / cfg["input"]["manifest"])
    recs = [(r.recording_id, ROOT / r.audio_filename) for r in manifest.itertuples()]
    if only:
        missing = set(only) - {r for r, _ in recs}
        if missing:
            raise KeyError(f"Recording ids not in manifest: {sorted(missing)}")
        recs = [r for r in recs if r[0] in only]
    return recs


def run_on_dataset(model: BaseDiarizationModel, recordings: list[tuple[str, Path]], exp_id: str,
                   cfg: dict | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_model_config()
    return [run_on_recording(model, rid, p, exp_id, cfg) for rid, p in recordings]


def run_single_model(name: str, exp_id: str | None = None, recording_ids: list[str] | None = None) -> dict[str, Any]:
    """Load one model, run it on the dataset, write experiments/<exp>/parts/<model>.json and return it."""
    load_dotenv(ROOT / ".env")
    cfg = load_model_config()
    exp_id = exp_id or new_experiment_id(cfg["models"][name])
    recordings = load_recordings(cfg, recording_ids)
    model = create_model(name, cfg)
    started = datetime.now(timezone.utc).isoformat()
    load_error = None
    try:
        model.load_model()
    except Exception as exc:  # recorded for every recording, never swallowed
        load_error = f"{type(exc).__name__}: {exc}"
        LOGGER.error("MODEL LOAD FAILED (%s): %s", name, load_error)
    if load_error:
        results = [{"recording_id": rid, "model_name": name, "status": "failed",
                    "errors": [f"model load failed: {load_error}"], "warnings": []} for rid, _ in recordings]
    else:
        results = run_on_dataset(model, recordings, exp_id, cfg)
    part = {
        "experiment_id": exp_id, "model_name": name, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "track": cfg.get("track"),
        "config": {k: v for k, v in cfg["models"][name].items()},
        "model_metadata": model.get_model_metadata() if not load_error else {"load_error": load_error},
        "hardware": hardware_info(), "software": software_versions(), "load_error": load_error,
        "results": results,
    }
    out = ROOT / cfg["output"]["experiments"] / exp_id / "parts" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(part, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %s", out.relative_to(ROOT))
    return part
