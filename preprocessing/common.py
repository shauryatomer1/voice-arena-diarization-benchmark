"""Shared helpers: project root, config loading, logging."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    """Load a YAML file from the config directory."""
    with (ROOT / "config" / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(cfg: dict, key: str) -> Path:
    """Resolve a `paths:` entry of config.yaml relative to the project root."""
    return ROOT / cfg["paths"][key]


def setup_logging(name: str, cfg: dict | None = None) -> logging.Logger:
    """Log to console and logs/<name>.log."""
    cfg = cfg or load_yaml("config.yaml")
    log_dir = project_path(cfg, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")])
    return logging.getLogger(name)
