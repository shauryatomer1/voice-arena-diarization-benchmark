"""CLI for Phase 2 inference.

Orchestrate all models (each in its own venv):
    python scripts/run_inference.py --all [--recordings ES2004a IS1009a]
Run ONE model inside the current environment:
    python scripts/run_inference.py --model pyannote_community_1 [--experiment-id EXP] [--recordings ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.common import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="run all models via their own virtualenvs")
    g.add_argument("--model", help="run a single model in THIS environment")
    ap.add_argument("--models", nargs="*", help="with --all: subset of models")
    ap.add_argument("--experiment-id")
    ap.add_argument("--recordings", nargs="*")
    a = ap.parse_args()
    setup_logging("inference")
    if a.all:
        from inference.batch_runner import run_all_models
        m = run_all_models(a.models, a.recordings, a.experiment_id)
        return 1 if (m["errors"] or m["models_without_results"]) else 0
    from inference.runner import run_single_model
    part = run_single_model(a.model, a.experiment_id, a.recordings)
    return 1 if any(r["status"] == "failed" for r in part["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
