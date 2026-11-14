"""Generate notebooks/01_audio_and_annotation_explorer.ipynb. Run: python scripts/build_notebook.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_audio_and_annotation_explorer.ipynb"

CELLS = [
("md", """# Phase 1 - Audio & reference-annotation explorer
Inspect each prepared recording: metadata, waveform, per-speaker timeline lanes, overlap regions, speaker count, and audio playback.
Run from the `notebooks/` folder (or anywhere - paths are resolved from the project root)."""),
("code", """import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config" / "config.yaml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from preprocessing.rttm import read_rttm, compute_stats
from preprocessing.audio_processor import get_audio_metadata

manifest = pd.read_csv(ROOT / "data/metadata/dataset_manifest.csv")
pd.set_option("display.width", 200, "display.max_columns", 30)
manifest[["recording_id", "dataset", "original_split", "duration_s", "num_speakers", "sample_rate",
          "channels", "language", "has_overlap", "overlap_ratio_of_speech", "num_segments"]]"""),
("md", "## Helpers"),
("code", """PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]

def load(rid):
    row = manifest.set_index("recording_id").loc[rid]
    data, sr = sf.read(ROOT / row.audio_filename, dtype="float32")
    return data, sr, read_rttm(ROOT / row.annotation_filename)

def overlap_regions(df):
    \"\"\"Intervals where >= 2 segments are active.\"\"\"
    ev = sorted([(s, 1) for s in df.start] + [(e, -1) for e in df.end])
    out, active, t0 = [], 0, None
    for t, d in ev:
        was = active; active += d
        if was < 2 <= active: t0 = t
        elif was >= 2 > active: out.append((t0, t))
    return out

def plot_recording(rid, t_start=0, t_end=None):
    data, sr, df = load(rid)
    t_end = t_end or len(data) / sr
    spk = sorted(df.speaker.unique())
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(spk)}
    fig, (aw, at) = plt.subplots(2, 1, figsize=(14, 2 + 0.6 * len(spk) + 2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, max(1, len(spk) * 0.6)]})
    i0, i1 = int(t_start * sr), int(t_end * sr)
    seg = data[i0:i1]; step = max(1, len(seg) // 6000)           # min/max envelope for speed
    n = len(seg) // step
    blocks = seg[: n * step].reshape(n, step)
    tt = t_start + np.arange(n) * step / sr
    aw.fill_between(tt, blocks.min(1), blocks.max(1), color="#555", lw=0)
    aw.set_ylabel("amplitude"); aw.set_title(f"{rid} - waveform ({(t_end - t_start):.0f}s shown)")
    for k, s in enumerate(spk):
        for r in df[df.speaker == s].itertuples():
            if r.end < t_start or r.start > t_end: continue
            at.barh(k, r.duration, left=r.start, height=0.7, color=colors[s], edgecolor="none")
    for a, b in overlap_regions(df):
        if b < t_start or a > t_end: continue
        at.axvspan(a, b, color="red", alpha=0.25, lw=0)
        aw.axvspan(a, b, color="red", alpha=0.2, lw=0)
    at.set_yticks(range(len(spk))); at.set_yticklabels(spk); at.invert_yaxis()
    at.set_xlim(t_start, t_end); at.set_xlabel("time (s)  -  red shading = overlapping speech")
    plt.tight_layout(); plt.show()

def summary(rid):
    row = manifest.set_index("recording_id").loc[rid]
    _, _, df = load(rid)
    st = compute_stats(df)
    m = get_audio_metadata(ROOT / row.audio_filename)
    per = df.groupby("speaker").duration.sum().sort_values(ascending=False).round(1)
    print(f"{rid}: {m.duration_s:.1f}s | {m.sample_rate} Hz | {m.channels} ch | {m.subtype} | speakers={st.n_speakers} | "
          f"segments={st.n_segments} | speech={st.speech_s:.0f}s | overlap={st.overlap_s:.0f}s ({st.overlap_ratio:.1%} of speech)")
    display(per.rename("speaking time (s)").to_frame().T)

def play(rid, start, end):
    data, sr, _ = load(rid)
    display(Audio(data[int(start * sr): int(end * sr)], rate=sr))"""),
("md", "## Per-recording summary, full-length waveform and speaker timeline"),
("code", """for rid in manifest.recording_id:
    summary(rid)
    plot_recording(rid)"""),
("md", "## Zoom on an overlap region (first overlap longer than 1 s) with playback"),
("code", """for rid in manifest.recording_id:
    _, _, df = load(rid)
    ov = [r for r in overlap_regions(df) if r[1] - r[0] > 1.0]
    a, b = ov[0]
    print(f"{rid}: {len(ov)} overlap regions > 1 s; zooming on {a:.1f}-{b:.1f}s")
    plot_recording(rid, max(0, a - 10), b + 10)
    play(rid, max(0, a - 3), b + 3)"""),
("md", "## Speaker count and overlap overview"),
("code", """overview = manifest.set_index("recording_id")[["num_speakers", "overlap_ratio_of_speech"]]
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
overview.num_speakers.plot.bar(ax=ax[0], color="#4C78A8", title="Speakers per recording")
(overview.overlap_ratio_of_speech * 100).plot.bar(ax=ax[1], color="#E45756", title="Overlap (% of speech time)")
for a in ax: a.set_xlabel("")
plt.tight_layout(); plt.show()"""),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
