"""Environment diagnostic. Run: python scripts/check_env.py"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

PACKAGES = ["numpy", "pandas", "soundfile", "yaml", "scipy", "matplotlib", "requests", "IPython", "nbformat"]


def main() -> int:
    ok = True
    print(f"OS            : {platform.platform()}")
    print(f"Python        : {sys.version.split()[0]} ({sys.executable})")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        print("  WARNING: Python 3.10-3.12 recommended")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtualenv    : {'yes' if in_venv else 'NO (activate .venv)'}")
    for pkg in PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            print(f"  [ok] {pkg:<11} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # also catches DLL / policy errors
            ok = False
            print(f"  [FAIL] {pkg}: {exc}")
    try:
        import soundfile as sf
        print(f"libsndfile    : {sf.__libsndfile_version__}")
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ver = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        print(f"FFmpeg        : {ver}")
    else:
        print("FFmpeg        : not found (optional; pipeline uses SoundFile. Install: winget install Gyan.FFmpeg)")
    smi = shutil.which("nvidia-smi")
    print(f"NVIDIA GPU    : {'nvidia-smi found (not needed in Phase 1)' if smi else 'none detected (not needed)'}")
    print("RESULT        :", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
