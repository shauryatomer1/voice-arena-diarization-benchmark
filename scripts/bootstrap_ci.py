"""Block-bootstrap confidence intervals for DER and paired model differences.

Why blocks: with only a handful of recordings, resampling whole recordings is degenerate. We cut every
recording into fixed-length blocks (default 60 s), score each block, then resample blocks (pooled over
recordings, with replacement) 2000x. DER = sum(errors)/sum(reference speech) over the resampled blocks.
Differences between two models use the SAME resampled blocks (paired), so shared difficulty cancels.

Speaker mapping: the optimal hyp->ref mapping is computed ONCE per recording on the full file, applied to the
hypothesis, and then FROZEN inside every block (pyannote.metrics would otherwise re-optimise it per block,
which is more lenient and under-reports DER). The script checks that pooled blocks reproduce the full-file DER.
Caveat: blocks within a meeting are not independent (same speakers, same topic), so CIs are somewhat
optimistic; with 3 recordings, read them as a lower bound on uncertainty, not a verdict.

Run inside .venv-pyannote:  python scripts/bootstrap_ci.py [--block 60] [--collar 0.25] [--skip-overlap]
Writes results/bootstrap_ci.csv (per model) and results/bootstrap_pairs.csv (paired differences).
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate
from preprocessing.common import ROOT
from scoring.scorer import load_annotation

KEYS = ("missed detection", "false alarm", "confusion", "total")


def latest_rttm_dirs() -> dict[str, Path]:
    """model name -> directory of its latest experiment."""
    base, found = ROOT / "predictions/rttm", {}
    for exp in sorted(p for p in base.iterdir() if p.is_dir()):
        for m in sorted(p for p in exp.iterdir() if p.is_dir()):
            if any(m.glob("*.rttm")):
                found[m.name] = m          # later experiments overwrite earlier ones
    return found


def block_components(ref, hyp, dur, block, collar, skip):
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip)
    full = Timeline([Segment(0, dur)])
    mapping = metric.optimal_mapping(ref, hyp, uem=full)          # hyp label -> ref label
    hyp = hyp.rename_labels(mapping={h: mapping.get(h, f"_extra_{h}") for h in hyp.labels()})
    out = []
    for t in np.arange(0, dur, block):
        uem = Timeline([Segment(t, min(t + block, dur))])
        # IdentificationErrorRate = DER's base class: compares labels as given (no re-mapping).
        c = IdentificationErrorRate(collar=collar, skip_overlap=skip)(ref, hyp, uem=uem, detailed=True)
        out.append([c[k] for k in KEYS])
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=float, default=60.0); ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--skip-overlap", action="store_true"); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    man = pd.read_csv(ROOT / "data/metadata/dataset_manifest.csv").set_index("recording_id")
    dirs = latest_rttm_dirs()
    comps = {}
    for model, d in dirs.items():
        parts = []
        for rid in man.index:
            f = d / f"{rid}.rttm"
            if not f.exists(): continue
            ref = load_annotation(ROOT / man.loc[rid, "annotation_filename"], rid)
            parts.append(block_components(ref, load_annotation(f, rid), float(man.loc[rid, "duration_s"]),
                                          a.block, a.collar, a.skip_overlap))
        if len(parts) == len(man):
            comps[model] = np.vstack(parts)
            full = [DiarizationErrorRate(collar=a.collar, skip_overlap=a.skip_overlap)(
                        load_annotation(ROOT / man.loc[r, "annotation_filename"], r), load_annotation(d / f"{r}.rttm", r),
                        uem=Timeline([Segment(0, float(man.loc[r, "duration_s"]))]), detailed=True) for r in man.index]
            full_der = sum(c["missed detection"] + c["false alarm"] + c["confusion"] for c in full) / sum(c["total"] for c in full)
            blk = comps[model]; blk_der = blk[:, :3].sum() / blk[:, 3].sum()
            flag = "OK" if abs(blk_der - full_der) < 0.005 else "MISMATCH - investigate"
            print(f"check {model}: pooled-block DER {blk_der:.4f} vs full-file micro DER {full_der:.4f}  [{flag}]")
        else: print(f"skip {model}: only {len(parts)}/{len(man)} recordings")
    n = len(next(iter(comps.values())))
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n, size=(a.n_boot, n))                  # SAME resamples for every model (paired)
    def der(c, ix): s = c[ix].sum(axis=1); return (s[:, 0] + s[:, 1] + s[:, 2]) / s[:, 3]
    boots = {m: der(c, idx) for m, c in comps.items()}
    lo, hi = 2.5, 97.5
    rows = [{"model": m, "n_blocks": n, "block_s": a.block, "collar_s": a.collar, "skip_overlap": a.skip_overlap,
             "DER": (c[:, :3].sum() / c[:, 3].sum()), "CI95_low": np.percentile(boots[m], lo),
             "CI95_high": np.percentile(boots[m], hi)} for m, c in comps.items()]
    pairs = []
    for x, y in itertools.combinations(comps, 2):
        d = boots[x] - boots[y]
        pairs.append({"model_a": x, "model_b": y, "DER_a_minus_b": d.mean(), "CI95_low": np.percentile(d, lo),
                      "CI95_high": np.percentile(d, hi), "P(a better than b)": float((d < 0).mean()),
                      "verdict": "a < b (a better)" if np.percentile(d, hi) < 0 else
                                 "b < a (b better)" if np.percentile(d, lo) > 0 else "not distinguishable"})
    out = ROOT / "results"
    pd.DataFrame(rows).round(4).to_csv(out / "bootstrap_ci.csv", index=False)
    pd.DataFrame(pairs).round(4).to_csv(out / "bootstrap_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(); print(pd.DataFrame(pairs).round(3).to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
