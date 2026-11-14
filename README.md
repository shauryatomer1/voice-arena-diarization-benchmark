# Voice Arena - Speaker Diarization Evaluation Benchmark (Phase 1)

Phase 1 = environment, dataset and audio preparation. **No model inference or scoring yet** (Phase 2+).
Full write-up: [docs/phase1_technical.md](docs/phase1_technical.md).

## Reproduce (Windows PowerShell)

```powershell
cd voice-arena
python -m venv .venv                      # Python 3.10-3.12 (built on 3.11.9)
.\.venv\Scripts\Activate.ps1              # deactivate with: deactivate
pip install -r requirements.txt
python scripts/check_env.py               # environment diagnostic
python scripts/download_data.py           # downloads ONLY the 3 selected AMI meetings + RTTMs (~109 MB)
python scripts/prepare_dataset.py         # 16 kHz mono PCM-16 WAVs, normalised RTTMs, manifest, metadata
python scripts/run_validation.py          # writes validation_report.csv/.json (exit 1 on any failure)
python -m unittest validation.test_validators -v   # negative tests of the validators
python scripts/build_notebook.py          # (re)generate the notebook
jupyter notebook notebooks/01_audio_and_annotation_explorer.ipynb
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`, or call `.\.venv\Scripts\python.exe` directly.

## Reproduce (Linux / macOS)

```bash
cd voice-arena
python3 -m venv .venv && source .venv/bin/activate     # deactivate with: deactivate
pip install -r requirements.txt
python scripts/check_env.py && python scripts/download_data.py && python scripts/prepare_dataset.py \
  && python scripts/run_validation.py
jupyter notebook notebooks/01_audio_and_annotation_explorer.ipynb
```

## Notes
- **FFmpeg is optional.** Conversion uses SoundFile (+ SciPy polyphase resampling). `check_env.py` reports whether FFmpeg exists
  (Windows install: `winget install Gyan.FFmpeg`). No CUDA/GPU is needed in Phase 1.
- pandas is pinned `<2.3`: on the development machine a Windows Application Control policy blocked pandas 3.x binaries.
- Raw audio and original RTTMs are never modified; transformed copies live in `data/processed` and `data/references/normalized`.
- Data and credentials are git-ignored. `.env.example` lists variables only needed for VoxConverse audio / later phases.

## Layout
```
config/         config.yaml (paths, audio target, tolerances), selection.yaml (chosen recordings + verified URLs)
data/raw        original audio (untouched)        data/processed   standardised WAVs
data/references original/ (untouched RTTM) and normalized/ (transformed copies)
data/metadata   dataset_manifest.csv, audio_metadata_report.csv/json, validation_report.csv/json
preprocessing/  audio_processor.py, rttm.py, common.py
validation/     validators.py, test_validators.py
scripts/        check_env, download_data, prepare_dataset, run_validation, build_notebook
notebooks/      01_audio_and_annotation_explorer.ipynb
docs/           phase1_technical.md           logs/            run logs
```

## Phase 2/3 additions
- `requirements-pyannote.txt`: exact pins for `.venv-pyannote` (inference + scoring).
- `python scripts/run_inference.py --all --models pyannote_community_1 pyannote_3_1` (needs HF_TOKEN + accepted model terms).
- `python scripts/run_scoring.py` then `python scripts/bootstrap_ci.py` (block-bootstrap CIs + paired differences, self-checked against full-file DER).
- NeMo MSDD is registered but could not be run on the dev machine (compiled dependencies blocked by Windows Application Control).
