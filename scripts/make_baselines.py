"""Write trivial no-information baselines as RTTMs (the floor every real system must beat).
  baseline_single_speaker : whole file = one speaker.
  baseline_oracle_vad_1spk: reference speech regions merged, labelled one speaker (isolates speaker-confusion/overlap cost).
  sanity_reference        : reference copied as prediction -> must score DER 0 (checks the scorer).
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.common import ROOT
from scoring.scorer import load_annotation

exp = sys.argv[1] if len(sys.argv) > 1 else "baselines"
man = pd.read_csv(ROOT / "data/metadata/dataset_manifest.csv")
for r in man.itertuples():
    ref = load_annotation(ROOT / r.annotation_filename, r.recording_id)
    outs = {"baseline_single_speaker": [(0.0, r.duration_s)],
            "baseline_oracle_vad_1spk": [(s.start, s.end) for s in ref.get_timeline().support()],
            "sanity_reference": None}
    for name, segs in outs.items():
        d = ROOT / "predictions/rttm" / exp / name; d.mkdir(parents=True, exist_ok=True)
        if segs is None:
            lines = [f"SPEAKER {r.recording_id} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> {l} <NA> <NA>"
                     for s, _, l in ref.itertracks(yield_label=True)]
        else:
            lines = [f"SPEAKER {r.recording_id} 1 {a:.3f} {b-a:.3f} <NA> <NA> spk0 <NA> <NA>" for a, b in segs]
        (d / f"{r.recording_id}.rttm").write_text("\n".join(lines) + "\n")
print("baselines written to predictions/rttm/" + exp)
