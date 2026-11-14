"""Run all validation checks and write CSV + JSON reports.

Run: python scripts/run_validation.py        (exit code 1 if any check fails)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.common import ROOT, load_yaml, project_path, setup_logging  # noqa: E402
from validation.validators import validate_all  # noqa: E402


def main() -> int:
    cfg = load_yaml("config.yaml")
    log = setup_logging("validation", cfg)
    meta = project_path(cfg, "metadata")
    manifest_path = meta / "dataset_manifest.csv"
    if not manifest_path.is_file():
        log.error("Manifest missing: run scripts/prepare_dataset.py first")
        return 1
    results = [r.to_dict() for r in validate_all(pd.read_csv(manifest_path), ROOT, cfg)]
    df = pd.DataFrame(results)
    df.to_csv(meta / "validation_report.csv", index=False)
    failed = df[~df["passed"]]
    summary = {"total_checks": len(df), "passed": int(df["passed"].sum()), "failed": len(failed),
               "all_passed": failed.empty, "results": results}
    (meta / "validation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in failed.itertuples():
        log.error("FAIL %s %s: %s", r.recording_id, r.check, r.detail)
    log.info("Validation: %d/%d checks passed", summary["passed"], summary["total_checks"])
    return 0 if failed.empty else 1


if __name__ == "__main__":
    sys.exit(main())
