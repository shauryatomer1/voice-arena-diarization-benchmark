"""Phase 2 framework tests using a fake model (no weights/tokens needed).

Run: python -m unittest inference.test_phase2 -v
Real-model tests (both models load / process the same recording) are the actual inference runs
documented in docs/phase2_technical.md.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from inference.prediction_validator import validate_prediction, validate_rttm_against_prediction
from inference.runner import load_prediction, run_on_dataset, save_prediction
from models.base_model import BaseDiarizationModel, RawOutput, RawSegment
from normalization.output_normalizer import normalize_output, prediction_to_rttm
from preprocessing.rttm import read_rttm


class FakeModel(BaseDiarizationModel):
    """Behaviour chosen by file stem: ovl -> overlapping speakers, empty -> nothing, boom -> raises."""

    def load_model(self):
        self.loaded = True

    def diarize(self, audio_path):
        if audio_path.stem == "boom":
            raise RuntimeError("synthetic failure")
        if audio_path.stem == "empty":
            return RawOutput([], {}, 0.01)
        if audio_path.stem == "bad":
            return RawOutput([RawSegment(1.0, 0.5, "x")], {}, 0.01)
        return RawOutput([RawSegment(0.5, 2.8, "B"), RawSegment(2.4, 4.1, "A"), RawSegment(2.6, 3.0, "C")],
                         {"k": 1}, 0.02)

    def get_model_metadata(self):
        return {"revision": "fake-1"}


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = {"output": {"root": str(self.tmp / "pred")}}
        self.model = FakeModel("fake", {"raw_dir": "fake", "input": {"sample_rate": 16000, "channels": 1}})
        self.model.load_model()

    def _wav(self, stem, secs=5.0):
        p = self.tmp / f"{stem}.wav"
        sf.write(str(p), 0.1 * np.random.randn(int(16000 * secs)), 16000, subtype="PCM_16")
        return p

    def test_overlap_preserved_json_and_rttm(self):
        recs = [("ovl", self._wav("ovl"))]
        (r,) = run_on_dataset(self.model, recs, "exp_t", self.cfg)
        self.assertEqual(r["status"], "success", r)
        pred = load_prediction(Path(r["paths"]["normalized"]))
        self.assertEqual(len(pred["segments"]), 3)
        self.assertEqual(pred["num_speakers"], 3)
        self.assertEqual(pred["segments"][0]["speaker"], "speaker_0")      # relabelled by first appearance
        self.assertEqual(pred["segments"][0]["end"], 2.8)                  # unrounded, preserved
        df = read_rttm(Path(r["paths"]["rttm"]))
        self.assertEqual(df.file_id.unique().tolist(), ["ovl"])
        self.assertEqual(validate_rttm_against_prediction(Path(r["paths"]["rttm"]), pred), [])
        self.assertAlmostEqual(r["rtf"], 0.02 / 5.0, places=3)             # RTF = processing / duration
        json.dumps(pred)                                                    # valid JSON

    def test_empty_failed_and_continue(self):
        recs = [(s, self._wav(s)) for s in ("boom", "empty", "bad", "ovl")]
        res = {r["recording_id"]: r for r in run_on_dataset(self.model, recs, "exp_t2", self.cfg)}
        self.assertEqual(res["boom"]["status"], "failed")
        self.assertIn("synthetic failure", res["boom"]["errors"][0])
        self.assertEqual(res["empty"]["status"], "success_empty")
        self.assertTrue(res["empty"]["warnings"])
        self.assertEqual(res["bad"]["status"], "failed")                    # end <= start flagged
        self.assertEqual(res["ovl"]["status"], "success")                   # run continued after failures

    def test_no_overwrite(self):
        p = self.tmp / "x.json"
        save_prediction({"a": 1}, p)
        with self.assertRaises(FileExistsError):
            save_prediction({"a": 2}, p)

    def test_validator_rules(self):
        base = normalize_output("r", "m", "v", [RawSegment(0, 1, "a")])
        self.assertEqual(validate_prediction(base, "r", 10.0), ([], []))
        self.assertTrue(validate_prediction(base, "other", 10.0)[0])        # id mismatch
        self.assertTrue(validate_prediction(base, "r", 0.5)[0])             # beyond duration
        bad = dict(base, segments=[{"start": float("nan"), "end": 1.0, "speaker": "a"}])
        self.assertTrue(validate_prediction(bad, "r", 10.0)[0])
        self.assertTrue(validate_prediction({"recording_id": "r"}, "r", 1.0)[0])

    def test_rttm_conversion_correct(self):
        pred = normalize_output("rec", "m", "v", [RawSegment(1.2345, 2.5, "q"), RawSegment(2.0, 3.0, "z")])
        p = self.tmp / "o.rttm"
        prediction_to_rttm(pred, p)
        self.assertTrue(p.read_text().startswith("SPEAKER rec 1 "))
        df = read_rttm(p)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.start.iloc[0], 1.2345, places=3)

    def test_no_reference_leakage_guard(self):
        from models.pyannote_model import PyannoteModel
        m = PyannoteModel("p", {"track": "primary", "params": {"num_speakers": 4}, "output_field": "x",
                                "input": {}, "provider": "", "model_id": "", "auth": {"env_var": "X"}})
        m.loaded, m.device = True, "cpu"
        m.pipeline = lambda *a, **k: None
        p = self.tmp / "g.wav"
        sf.write(str(p), np.zeros(1600), 16000)
        with self.assertRaises(RuntimeError):
            m.diarize(p)


if __name__ == "__main__":
    unittest.main()
