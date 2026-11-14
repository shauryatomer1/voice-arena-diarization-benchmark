"""Download the selected AMI recordings and their reference RTTMs.

Run: python scripts/download_data.py
Only the files listed in config/selection.yaml are fetched (never the full corpus).
Existing files are kept (skipped with a log line), never overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.common import load_yaml, project_path, setup_logging  # noqa: E402


def fetch(url: str, dst: Path, timeout: int, chunk: int, log) -> None:
    """Stream `url` to `dst` via a .part file; raise on any HTTP error."""
    if dst.exists():
        log.info("exists, skipping: %s", dst.name)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log.info("GET %s", url)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        # Content-Length is the compressed size when Content-Encoding is set, so only compare otherwise.
        expected = 0 if r.headers.get("Content-Encoding") else int(r.headers.get("Content-Length", 0))
        done = 0
        with tmp.open("wb") as fh:
            for part in r.iter_content(chunk):
                fh.write(part)
                done += len(part)
    if expected and done != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Incomplete download for {url}: {done}/{expected} bytes")
    tmp.rename(dst)
    log.info("saved %s (%.1f MB)", dst.name, done / 1e6)


def main() -> int:
    cfg, sel = load_yaml("config.yaml"), load_yaml("selection.yaml")
    log = setup_logging("download", cfg)
    ds = sel["dataset"]
    raw, ref = project_path(cfg, "raw_audio"), project_path(cfg, "ref_original")
    t, c = cfg["download"]["timeout_s"], cfg["download"]["chunk_bytes"]
    failures = []
    for rec in sel["recordings"]:
        rid, split = rec["id"], rec["split"]
        try:  # confirm the recording really belongs to the stated split
            lst = requests.get(ds["split_list_url"].format(split=split), timeout=t)
            lst.raise_for_status()
            if rid not in lst.text.split():
                raise ValueError(f"{rid} not in official '{split}' list")
            fetch(ds["rttm_url_template"].format(id=rid, split=split), ref / f"{rid}.rttm", t, c, log)
            fetch(ds["audio_url_template"].format(id=rid), raw / f"{rid}.Mix-Headset.wav", t, c, log)
        except Exception as exc:
            log.error("FAILED %s: %s", rid, exc)
            failures.append(rid)
    if failures:
        log.error("Failed recordings: %s", failures)
        return 1
    log.info("All downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
