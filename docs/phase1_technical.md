# Phase 1 Technical Documentation

## 1. Diarization concepts
**Input:** a single audio recording (here WAV, PCM-16, 16 kHz, mono). The system is not told who the speakers are or how many.

**Output:** a list of segments `(start, end, speaker label)`, e.g. `0.37 s -> 1.76 s : SPK_A`. Labels are arbitrary and
only meaningful within one recording (`SPK_A` in file 1 is unrelated to `SPK_A` in file 2), so scoring must find the best
mapping between system and reference labels. Segments of different speakers may overlap in time.

**Reference annotation:** the "correct answer", built by humans (AMI: word-level manual transcription timings, converted to speaker
segments). It is the ground truth every system is compared with. It is imperfect (boundary and short-word decisions), which matters
for collar settings in Phase 2.

**RTTM (Rich Transcription Time Marked):** plain text, one line per segment:
```
SPEAKER ES2004a 1 0.370 1.390 <NA> <NA> MEO015 <NA> <NA>
type    file_id chan start dur  ortho stype speaker conf slat
```
Fields used: `file_id` (must equal the recording ID), `start`, `duration` (end = start + duration), `speaker`. Overlap is
represented simply by two lines whose intervals intersect.

## 2. Why standardise audio
Systems differ in the audio they accept. Scoring only compares times, so every system must hear the *same* signal, and segment times
must refer to the same timeline. We use WAV / PCM-16 / 16 kHz / mono because it is the lowest common denominator of open-source and API
systems and avoids codec/resampler differences leaking into results. Operations are logged per file (`logs/prepare.log`):
the AMI headset-mix files were already 16 kHz mono, so conversion was a re-encode to PCM-16 with **no resampling and no channel mixing**.
If a stereo/multichannel source is ever used, the mean down-mix is a convenience for the mono track only; it is *not* a fair stand-in for
channel-aware systems, and originals are kept for a separate multichannel track.

## 3. Why overlapping speech is preserved
Overlap is a main failure mode of diarization (the assessment names it explicitly) and 5-16% of speech time here. Merging or truncating
overlapping reference segments would silently remove the hardest cases, make scores look better, and make the later
"overlap-specific performance" analysis impossible. The normaliser only rounds times to ms and unifies placeholder fields; validation
(`overlap_preserved_in_normalized`) proves segment-for-segment equality with the original.

## 4. Datasets
**Selected: AMI Meeting Corpus (test split)** - ES2004a (17.5 min), IS1009a (14 min), TS3003a (25 min). All 4 speakers, English, 4.6-15.8% overlap.
URLs verified live (HTTP 200); membership in the official test list verified by `download_data.py`.
- Audio: `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<ID>/audio/<ID>.Mix-Headset.wav` - AMI "Headset mix", CC BY 4.0.
- References: `pyannote/AMI-diarization-setup` `only_words/rttms/test/<ID>.rttm` (Apache-2.0), the standard diarization conversion of the official AMI manual annotations.
  Chosen over parsing the raw NXT XML (`ami_public_manual_1.6.2.zip`) because it is the community-standard, reproducible conversion and the one pyannote reports against.

**Why not VoxConverse (the preferred option):** audio zips at `robots.ox.ac.uk/~vgg/data/voxconverse/...` return **HTTP 401** - they need the
VoxCeleb username/password (request via the Google form linked on the VoxConverse page). RTTMs on GitHub are open, but without audio they are unusable,
and nothing was substituted without your agreement. **To add VoxConverse later:** fill the form, put `VOXCELEB_USER`/`VOXCELEB_PASSWORD` in `.env`,
download selected files from the dev/test zips, and add entries to `config/selection.yaml`. Licence: CC BY 4.0 (research use; copyright with video owners).

**Why AMI is a reasonable start, and why it is not enough (limitations):**
- Only the headset *mix* is used. AMI also has far-field array audio (`Array1-01.wav`, verified to exist) - not downloaded, so far-field conditions are not yet tested.
- All three meetings are **scenario design meetings** (role-played, 4 speakers, same task), English only - no phone audio, noise, accent variety, or varying speaker counts.
- Test split only, 3 recordings, ~57 minutes: enough to exercise the pipeline, far too small for statistically meaningful system comparison.
- References come from word timings: pauses inside a turn and very short backchannels are handled by the conversion rules, not by dedicated diarization annotators.
- Headset mix contains close-talk cross-talk; it is cleaner than real far-field meeting audio and will flatter systems.
- `ES2004a`-style meetings overlap far more than `TS3003a`, so per-file scores will vary a lot; do not average blindly.

## 5. Code map and decisions
| File | Purpose |
|---|---|
| `preprocessing/audio_processor.py` | `inspect_audio`, `get_audio_metadata`, `convert_to_mono`, `resample_audio`, `convert_to_wav`, `validate_audio` |
| `preprocessing/rttm.py` | strict RTTM parser, normaliser, writer, speaker/overlap statistics |
| `validation/validators.py` | all data checks; reports never skip a failure |
| `scripts/*.py` | thin command-line entry points; config in `config/*.yaml`; paths via `pathlib`, relative to project root |

Decisions: SoundFile instead of FFmpeg (no external binary; FFmpeg not installed on this machine; still reported by diagnostics).
`validate_audio` additionally detects truncated WAVs by comparing the `data` chunk size to the file size, because libsndfile silently clamps
truncated files (found via a failing negative test). The parser raises on any malformed line instead of skipping it.

## 6. Validation (data/metadata/validation_report.csv/.json)
Per recording: supported extension, audio exists, valid WAV/PCM-16/16 kHz/mono/non-empty/non-silent/decodable/untruncated; normalised and original RTTM exist
and parse, ID consistency, valid timestamps, positive durations, valid speaker IDs, no duplicate segments, segments inside audio duration (50 ms tolerance),
overlap preserved; globally: manifest IDs == audio IDs == reference IDs. Result: **62/62 checks passed**. `validation/test_validators.py` (9 tests) feeds
corrupt, truncated, empty, silent, stereo/44.1 kHz, bad-ID, zero-duration, out-of-range and duplicate inputs to confirm checks can fail.

## 7. Results summary
| Recording | Duration | Speakers | Segments | Overlap (s / % of speech) |
|---|---|---|---|---|
| ES2004a | 1049.4 s | 4 | 260 | 124 s / 15.8% |
| IS1009a | 838.8 s | 4 | 195 | 82 s / 13.6% |
| TS3003a | 1505.6 s | 4 | 242 | 45 s / 4.6% |

(Overlap here is segment-level overlap in the reference; Phase 2 will define the scored overlap region.)
