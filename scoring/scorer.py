"""Phase 3 scoring with pyannote.metrics (run inside .venv-pyannote).

Settings are explicit and recorded with every result:
  collar_s       forgiveness around reference boundaries (0.0 strict, 0.25 = common AMI/NIST convention)
  skip_overlap   False = overlapped reference speech IS scored (our primary setting)
  UEM            full file [0, duration] (no recording part is excluded)
Speaker labels are mapped to the reference with the Hungarian algorithm inside pyannote.metrics.
"""
from __future__ import annotations

from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def load_annotation(rttm: Path, rid: str) -> Annotation:
    d = load_rttm(str(rttm))
    return d.get(rid, Annotation(uri=rid))


def score_one(ref: Annotation, hyp: Annotation, duration: float, collar: float, skip_overlap: bool) -> dict:
    uem = Timeline([Segment(0.0, duration)])
    ref.uri = hyp.uri = ref.uri or "x"
    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    comp = der(ref, hyp, uem=uem, detailed=True)
    total = comp["total"] or 1.0
    # JER is defined per reference speaker with no collar; reported as a supporting metric.
    jer = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)(ref, hyp, uem=uem)
    return {"collar_s": collar, "skip_overlap": skip_overlap,
            "DER": comp["diarization error rate"], "JER": jer,
            "miss": comp["missed detection"] / total, "false_alarm": comp["false alarm"] / total,
            "confusion": comp["confusion"] / total, "ref_speech_s": comp["total"],
            "n_ref_speakers": len(ref.labels()), "n_hyp_speakers": len(hyp.labels())}
