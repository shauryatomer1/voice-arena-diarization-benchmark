"""Score all RTTMs under predictions/rttm/<experiment>/<model>/ against data/references/normalized.
Run inside .venv-pyannote:  python scripts/run_scoring.py [--experiment EXP]
Writes results/scores_<exp>.csv (per file x setting) and results/leaderboard_<exp>.csv (mean + micro DER).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.common import ROOT
from scoring.scorer import load_annotation, score_one

SETTINGS = [(0.25, False), (0.0, False), (0.25, True)]   # primary, strict, no-overlap-region


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment"); a = ap.parse_args()
    man = pd.read_csv(ROOT / "data/metadata/dataset_manifest.csv").set_index("recording_id")
    base = ROOT / "predictions/rttm"
    exps = [a.experiment] if a.experiment else sorted(p.name for p in base.iterdir() if p.is_dir())
    rows = []
    for exp in exps:
        for mdir in sorted((base / exp).iterdir()):
            for rid in man.index:
                f = mdir / f"{rid}.rttm"
                if not f.exists():
                    rows.append({"experiment": exp, "model": mdir.name, "recording": rid, "status": "missing"}); continue
                ref = load_annotation(ROOT / man.loc[rid, "annotation_filename"], rid)
                hyp = load_annotation(f, rid)
                for collar, skip in SETTINGS:
                    r = score_one(ref, hyp, float(man.loc[rid, "duration_s"]), collar, skip)
                    rows.append({"experiment": exp, "model": mdir.name, "recording": rid, "status": "ok", **r})
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); tag = a.experiment or "all"
    df.to_csv(out / f"scores_{tag}.csv", index=False)
    ok = df[df.status == "ok"].copy()
    # A model may have several experiment folders (reruns): score only the LATEST per model/recording
    # so reruns never double-count in the aggregates. (Experiment ids sort chronologically.)
    ok = ok.sort_values("experiment").drop_duplicates(["model", "recording", "collar_s", "skip_overlap"], keep="last")
    ok["w"] = ok.ref_speech_s
    g = ok.groupby(["model", "collar_s", "skip_overlap"])
    lb = g.apply(lambda d: pd.Series({
        "DER_macro": d.DER.mean(), "DER_micro": (d.DER * d.w).sum() / d.w.sum(), "JER_macro": d.JER.mean(),
        "miss": (d.miss * d.w).sum() / d.w.sum(), "fa": (d.false_alarm * d.w).sum() / d.w.sum(),
        "confusion": (d.confusion * d.w).sum() / d.w.sum(), "n_files": len(d)}), include_groups=False).reset_index()
    lb.to_csv(out / f"leaderboard_{tag}.csv", index=False)
    pd.set_option("display.width", 200); print(lb.round(3).to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
