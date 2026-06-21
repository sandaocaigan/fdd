"""Probe image construction utilities."""

from __future__ import annotations

from typing import Iterable, List

import torch


def parse_probe_scales(probe_scales) -> List[float]:
    """Parse a comma-separated scale string or iterable into a float list."""
    if probe_scales in [None, "None", "none", ""]:
        return [1.0]
    if isinstance(probe_scales, str):
        return [float(x.strip()) for x in probe_scales.split(",") if x.strip()]
    if isinstance(probe_scales, Iterable):
        return [float(x) for x in probe_scales]
    return [float(probe_scales)]


def build_scaled_probe(image: torch.Tensor, scale: float, clip_min=None, clip_max=None) -> torch.Tensor:
    """Build a SCALE-UP style intensity-scaled probe from an image tensor."""
    probe = image.detach().clone() * float(scale)
    if clip_min not in [None, "None", "none"] or clip_max not in [None, "None", "none"]:
        min_val = None if clip_min in [None, "None", "none"] else float(clip_min)
        max_val = None if clip_max in [None, "None", "none"] else float(clip_max)
        probe = torch.clamp(probe, min=min_val, max=max_val)
    return probe


def build_probe_variant(image: torch.Tensor, probe_type: str, scale: float, clip_min=None, clip_max=None) -> torch.Tensor:
    """Dispatch probe construction by type; new probe types can be added here."""
    probe_type = (probe_type or "scale").lower()
    if probe_type == "scale":
        return build_scaled_probe(image=image, scale=scale, clip_min=clip_min, clip_max=clip_max)
    raise NotImplementedError(f"Probe type {probe_type} is not implemented yet.")
