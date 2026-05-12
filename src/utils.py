"""Shared utilities for reproducibility and path handling."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device() -> torch.device:
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_output_dir(config: dict[str, Any], stage_key: str) -> Path:
    """Create and return the output directory for a training stage."""
    output_dir = Path(config[stage_key]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
