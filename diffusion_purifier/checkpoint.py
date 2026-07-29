"""Checkpoint helpers for diffusion purifier models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, model: torch.nn.Module, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
    """Save a diffusion purifier checkpoint."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": dict(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a diffusion purifier checkpoint and normalize old formats."""
    if path in [None, "None", "none", ""]:
        raise ValueError("A diffusion checkpoint path must be provided when diffusion purifier is enabled.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return {"model": checkpoint["state_dict"], "config": checkpoint.get("config", {})}
    if isinstance(checkpoint, dict):
        return {"model": checkpoint, "config": {}}
    raise TypeError(f"Unsupported diffusion checkpoint format: {type(checkpoint)}")
