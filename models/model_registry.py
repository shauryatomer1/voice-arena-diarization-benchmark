"""Registry: model name -> class, driven by config/model_config.yaml. Add a model = add a YAML block + a class."""
from __future__ import annotations

import importlib
from typing import Any

from preprocessing.common import load_yaml

from .base_model import BaseDiarizationModel


def load_model_config() -> dict[str, Any]:
    return load_yaml("model_config.yaml")


def list_models() -> list[str]:
    return list(load_model_config()["models"])


def create_model(name: str, cfg: dict[str, Any] | None = None) -> BaseDiarizationModel:
    """Instantiate (not load) a model. Imports only the requested model's module, so a venv that lacks
    the other framework still works."""
    cfg = cfg or load_model_config()
    if name not in cfg["models"]:
        raise KeyError(f"Unknown model {name!r}; available: {list(cfg['models'])}")
    mcfg = dict(cfg["models"][name])
    mcfg["track"] = cfg.get("track", "primary")
    module_path, cls_name = mcfg["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(name, mcfg)
