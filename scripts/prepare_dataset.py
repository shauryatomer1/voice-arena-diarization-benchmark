"""Standardise audio, normalise RTTMs, and build the manifest + audio metadata report.

Run: python scripts/prepare_dataset.py [--overwrite]
Raw audio and original RTTMs are only read, never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.audio_processor import convert_to_wav, get_audio_metadata  # noqa: E402
from preprocessing.common import ROOT, load_yaml, project_path, setup_logging  # noqa: E402
from preprocessing.rttm import compute_stats, normalize_rttm, read_rttm, write_rttm  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="re-create processed files")
    args = ap.parse_args()
    cfg, sel = load_yaml("config.yaml"), load_yaml("selection.yaml")
    log = setup_logging("prepare", cfg)
    a, ds = cfg["audio"], sel["dataset"]
    raw, proc = project_path(cfg, "raw_audio"), project_path(cfg, "processed_audio")
    ref_o, ref_n = project_path(cfg, "ref_original"), project_path(cfg, "ref_normalized")
    meta_dir = project_path(cfg, "metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest, audio_report, failures = [], [], []
    for rec in sel["recordings"]:
        rid = rec["id"]
        try:
            src = raw / f"{rid}.Mix-Headset.wav"
            dst = proc / f"{rid}.wav"
            raw_meta = get_audio_metadata(src)
            proc_meta = convert_to_wav(src, dst, a["target_sample_rate"], a["target_channels"],
                                       a["target_subtype"], overwrite=args.overwrite) \
                if (args.overwrite or not dst.exists()) else get_audio_metadata(dst)

            df = normalize_rttm(read_rttm(ref_o / f"{rid}.rttm"), cfg["rttm"]["normalized_decimals"])
            write_rttm(df, ref_n / f"{rid}.rttm", cfg["rttm"]["normalized_decimals"])
            st = compute_stats(df)

            for stage, m in (("raw", raw_meta), ("processed", proc_meta)):
                audio_report.append({"recording_id": rid, "stage": stage, **m.to_dict(),
                                     "path": rel(Path(m.path))})
            manifest.append({
                "recording_id": rid, "dataset": ds["name"], "original_split": rec["split"],
                "source_url_audio": ds["audio_url_template"].format(id=rid),
                "source_url_annotation": ds["rttm_url_template"].format(id=rid, split=rec["split"]),
                "audio_filename": rel(dst), "raw_audio_filename": rel(src),
                "annotation_filename": rel(ref_n / f"{rid}.rttm"),
                "original_annotation_filename": rel(ref_o / f"{rid}.rttm"),
                "duration_s": round(proc_meta.duration_s, 3), "num_speakers": st.n_speakers,
                "sample_rate": proc_meta.sample_rate, "channels": proc_meta.channels,
                "original_sample_rate": raw_meta.sample_rate, "original_channels": raw_meta.channels,
                "language": ds["language"], "meeting_type": rec["meeting_type"],
                "has_overlap": st.overlap_s > 0, "overlap_s": round(st.overlap_s, 2),
                "overlap_ratio_of_speech": round(st.overlap_ratio, 4), "num_segments": st.n_segments,
                "raw_audio_sha256": sha256(src),
                "licence_access_notes": f"Audio: {ds['licence']}. RTTM: {ds['rttm_licence']}. "
                                        "No registration needed.",
            })
            log.info("%s ok: %.0fs, %d speakers, overlap %.1f%% of speech", rid,
                     proc_meta.duration_s, st.n_speakers, 100 * st.overlap_ratio)
        except Exception as exc:  # report, never silently skip
            log.exception("FAILED %s: %s", rid, exc)
            failures.append(rid)

    pd.DataFrame(manifest).to_csv(meta_dir / "dataset_manifest.csv", index=False)
    pd.DataFrame(audio_report).to_csv(meta_dir / "audio_metadata_report.csv", index=False)
    (meta_dir / "audio_metadata_report.json").write_text(json.dumps(audio_report, indent=2), encoding="utf-8")
    if failures:
        log.error("Failed: %s", failures)
        return 1
    log.info("Wrote manifest and audio metadata report to %s", rel(meta_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
