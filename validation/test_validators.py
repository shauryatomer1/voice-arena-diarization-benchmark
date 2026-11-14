"""Negative tests proving the validators catch bad data. Run: python -m unittest validation.test_validators -v"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from preprocessing.audio_processor import AudioError, convert_to_wav, validate_audio
from preprocessing.rttm import compute_stats, read_rttm, RTTMError
from validation.validators import check_reference

LINE = "SPEAKER {rid} 1 {s} {d} <NA> <NA> {spk} <NA> <NA>\n"


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _wav(self, name, sr=16000, ch=1, secs=2.0, subtype="PCM_16", silent=False):
        n = int(sr * secs)
        x = np.zeros((n, ch)) if silent else 0.1 * np.random.randn(n, ch)
        p = self.tmp / name
        sf.write(str(p), x if ch > 1 else x[:, 0], sr, subtype=subtype)
        return p

    def test_good(self):
        self.assertEqual(validate_audio(self._wav("a.wav")), [])

    def test_wrong_rate_and_channels(self):
        probs = validate_audio(self._wav("b.wav", sr=44100, ch=2))
        self.assertTrue(any("sample rate" in p for p in probs) and any("channels" in p for p in probs))

    def test_empty_missing_corrupt_silent(self):
        e = self.tmp / "e.wav"
        sf.write(str(e), np.zeros(0), 16000, subtype="PCM_16")
        self.assertIn("audio is empty", validate_audio(e))
        self.assertTrue(validate_audio(self.tmp / "nope.wav"))
        c = self.tmp / "c.wav"
        c.write_bytes(b"not a wav file")
        self.assertTrue(validate_audio(c))
        self.assertTrue(any("silent" in p for p in validate_audio(self._wav("s.wav", silent=True))))

    def test_truncated(self):
        p = self._wav("t.wav", secs=3)
        p.write_bytes(p.read_bytes()[:-20001])
        self.assertTrue(validate_audio(p))

    def test_convert_stereo_44k(self):
        src = self._wav("st.wav", sr=44100, ch=2, secs=1.5)
        meta = convert_to_wav(src, self.tmp / "out.wav")
        self.assertEqual((meta.sample_rate, meta.channels, meta.subtype), (16000, 1, "PCM_16"))
        self.assertAlmostEqual(meta.duration_s, 1.5, places=2)
        self.assertEqual(validate_audio(self.tmp / "out.wav"), [])

    def test_never_overwrite_source(self):
        src = self._wav("o.wav")
        with self.assertRaises(AudioError):
            convert_to_wav(src, src, overwrite=True)


class RTTMTests(unittest.TestCase):
    def _rttm(self, lines):
        p = Path(tempfile.mkdtemp()) / "r.rttm"
        p.write_text("".join(lines))
        return p

    def results(self, lines, dur=100.0):
        return {r.check: r.passed for r in check_reference("r", self._rttm(lines), dur, 0.05)}

    def test_good_with_overlap_preserved(self):
        p = self._rttm([LINE.format(rid="r", s=0, d=5, spk="A"), LINE.format(rid="r", s=3, d=4, spk="B")])
        st = compute_stats(read_rttm(p))
        self.assertAlmostEqual(st.overlap_s, 2.0)
        self.assertEqual(st.n_speakers, 2)

    def test_bad_cases(self):
        r = self.results([LINE.format(rid="x", s=0, d=1, spk="A")])
        self.assertFalse(r["reference_recording_id_consistent"])
        r = self.results([LINE.format(rid="r", s=0, d=0, spk="A")])
        self.assertFalse(r["reference_durations_positive"])
        r = self.results([LINE.format(rid="r", s=99, d=5, spk="A")])
        self.assertFalse(r["reference_within_audio_duration"])
        r = self.results([LINE.format(rid="r", s=1, d=1, spk="A")] * 2)
        self.assertFalse(r["reference_no_duplicate_segments"])
        r = self.results([LINE.format(rid="r", s=1, d=1, spk="bad$id")])
        self.assertFalse(r["reference_speaker_ids_valid"])

    def test_malformed_raises(self):
        with self.assertRaises(RTTMError):
            read_rttm(self._rttm(["SPEAKER r 1 abc 2 <NA> <NA> A <NA> <NA>\n"]))


if __name__ == "__main__":
    unittest.main()
