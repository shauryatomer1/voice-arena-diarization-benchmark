"""Experiment bookkeeping: IDs, hashes, hardware/software snapshots, GPU/memory monitoring, .env loading."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

KEY_PACKAGES = ["torch", "torchaudio", "pyannote.audio", "nemo_toolkit", "numpy", "pandas", "soundfile",
                "scipy", "huggingface_hub", "omegaconf", "lightning", "pytorch-lightning", "PyYAML"]


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE); existing environment variables win. Values are never logged."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def new_experiment_id(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return f"exp_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{hashlib.sha256(blob).hexdigest()[:8]}"


def software_versions() -> dict[str, str]:
    out = {}
    for pkg in KEY_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
                            "logical_cpus": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            name, mem, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            info["gpu"] = {"name": name, "memory_total": mem, "driver": drv}
        except Exception as exc:
            info["gpu_error"] = repr(exc)
    return info


class PerfMonitor:
    """Context manager measuring wall time, peak process RSS, and sampled GPU utilisation / memory."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.gpu_util: list[float] = []
        self.gpu_mem_mb: list[float] = []
        self.peak_rss_mb: float | None = None
        self.wall_s: float = 0.0

    def _sample(self) -> None:
        smi = shutil.which("nvidia-smi")
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            proc = None
        while not self._stop.is_set():
            if proc is not None:
                rss = proc.memory_info().rss / 1e6
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss)
            if smi:
                try:
                    r = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    u, m = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
                    self.gpu_util.append(u)
                    self.gpu_mem_mb.append(m)
                except Exception:
                    pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PerfMonitor":
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._sample, daemon=True)
        self._t.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_s = time.perf_counter() - self._t0
        self._stop.set()
        self._t.join(timeout=5)
        try:  # Windows exposes the true peak working set
            import psutil
            peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
            if peak:
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, peak / 1e6)
        except ImportError:
            pass

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {"peak_rss_mb": round(self.peak_rss_mb, 1) if self.peak_rss_mb else None}
        if self.gpu_util:
            d.update(gpu_util_mean_pct=round(sum(self.gpu_util) / len(self.gpu_util), 1),
                     gpu_util_max_pct=max(self.gpu_util), gpu_mem_used_max_mb=max(self.gpu_mem_mb),
                     gpu_samples=len(self.gpu_util))
        try:
            import torch
            if torch.cuda.is_available():
                d["torch_peak_gpu_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except ImportError:
            pass
        return d
