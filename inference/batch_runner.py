"""Orchestrator: run every configured model in its OWN virtualenv, then merge into one experiment manifest.

Each model is a subprocess of `scripts/run_inference.py --model X --experiment-id E`, using the model's
`env_python`. This avoids pyannote.audio / NeMo dependency conflicts in a single environment.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from inference.experiment import hardware_info, new_experiment_id, sha256_file
from models.model_registry import load_model_config
from preprocessing.common import ROOT

LOGGER = logging.getLogger(__name__)


def _python_for(mcfg: dict) -> Path:
    base = ROOT / mcfg["env_python"]
    candidates = [base, base.with_name(base.name + ".exe"),            # Windows: Scripts/python.exe
                  ROOT / mcfg["env_python"].replace("Scripts/python", "bin/python")]  # POSIX
    return next((c for c in candidates if c.exists()), base)


def run_all_models(models: list[str] | None = None, recording_ids: list[str] | None = None,
                   exp_id: str | None = None) -> dict[str, Any]:
    cfg = load_model_config()
    models = models or list(cfg["models"])
    exp_id = exp_id or new_experiment_id({m: cfg["models"][m] for m in models})
    exp_dir = ROOT / cfg["output"]["experiments"] / exp_id
    if exp_dir.exists():
        raise FileExistsError(f"Experiment already exists (never overwritten): {exp_dir}")
    exp_dir.mkdir(parents=True)
    for m in models:
        py = _python_for(cfg["models"][m])
        if not py.exists():
            LOGGER.error("%s: interpreter %s not found - model will be recorded as failed", m, py)
            continue
        cmd = [str(py), str(ROOT / "scripts" / "run_inference.py"), "--model", m, "--experiment-id", exp_id]
        if recording_ids:
            cmd += ["--recordings", *recording_ids]
        LOGGER.info("Running %s: %s", m, " ".join(cmd))
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            LOGGER.error("%s subprocess exited with code %d", m, rc)
    return merge_experiment(exp_id, models)


def merge_experiment(exp_id: str, models: list[str]) -> dict[str, Any]:
    """Merge per-model parts into manifest.json + CSV reports. Missing parts are recorded as failures."""
    cfg = load_model_config()
    exp_dir = ROOT / cfg["output"]["experiments"] / exp_id
    parts, missing = {}, []
    for m in models:
        f = exp_dir / "parts" / f"{m}.json"
        if f.exists():
            parts[m] = json.loads(f.read_text(encoding="utf-8"))
        else:
            missing.append(m)
            LOGGER.error("No result part for model %s (its process failed before writing results)", m)
    rows, errors = [], []
    for m, p in parts.items():
        for r in p["results"]:
            rows.append({k: r.get(k) for k in (
                "model_name", "recording_id", "status", "audio_duration_s", "inference_time_s", "total_wall_time_s",
                "rtf", "processing_speed_x_realtime", "peak_rss_mb", "gpu_util_mean_pct", "gpu_util_max_pct",
                "gpu_mem_used_max_mb", "torch_peak_gpu_alloc_mb", "num_predicted_speakers",
                "num_predicted_segments")} | {"hardware": _hw_label(p["hardware"], p["model_metadata"]),
                                              "errors": "; ".join(r.get("errors", [])),
                                              "warnings": "; ".join(r.get("warnings", []))})
            errors += [{"model": m, "recording_id": r["recording_id"], "error": e} for e in r.get("errors", [])]
    perf = pd.DataFrame(rows)
    perf.to_csv(exp_dir / "performance_report.csv", index=False)
    model_meta = {m: p["model_metadata"] for m, p in parts.items()}
    (exp_dir / "model_metadata_report.json").write_text(json.dumps(model_meta, indent=2, default=str), encoding="utf-8")
    manifest = {
        "experiment_id": exp_id, "track": cfg.get("track"), "models_requested": models,
        "models_without_results": missing, "hardware": hardware_info(),
        "software_per_model": {m: p["software"] for m, p in parts.items()},
        "model_configs": {m: p["config"] for m, p in parts.items()},
        "model_metadata": model_meta,
        "input_hashes": {r["recording_id"]: r.get("audio_sha256") for p in parts.values() for r in p["results"]},
        "prediction_hashes": {m: {r["recording_id"]: {k: r.get(k) for k in ("raw_sha256", "normalized_sha256", "rttm_sha256")}
                                  for r in p["results"]} for m, p in parts.items()},
        "started_utc": min((p["started_utc"] for p in parts.values()), default=None),
        "finished_utc": max((p["finished_utc"] for p in parts.values()), default=None),
        "errors": errors, "results": {m: p["results"] for m, p in parts.items()},
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Experiment manifest: %s", (exp_dir / "manifest.json").relative_to(ROOT))
    return manifest


def _hw_label(hw: dict, meta: dict) -> str:
    dev = meta.get("device") or meta.get("device_name") or "?"
    gpu = hw.get("gpu", {})
    return f"{dev}: {gpu.get('name', hw.get('cpu'))}" if dev == "cuda" else f"cpu: {hw.get('cpu')}"
