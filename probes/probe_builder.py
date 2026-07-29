"""Probe image construction utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional

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


def build_blend_probe(
    image: torch.Tensor,
    partner_image: torch.Tensor,
    alpha: float = 0.5,
    clip_min=None,
    clip_max=None,
) -> torch.Tensor:
    """Build an ambiguous probe by linearly blending two public images."""
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)
    probe = alpha * image.detach().clone() + (1.0 - alpha) * partner_image.detach().clone()
    if clip_min not in [None, "None", "none"] or clip_max not in [None, "None", "none"]:
        min_val = None if clip_min in [None, "None", "none"] else float(clip_min)
        max_val = None if clip_max in [None, "None", "none"] else float(clip_max)
        probe = torch.clamp(probe, min=min_val, max=max_val)
    return probe


def build_noise_probe(
    image: torch.Tensor,
    scale: float,
    noise_scale: float = 0.15,
    clip_min=None,
    clip_max=None,
) -> torch.Tensor:
    """Build a deterministic noise-perturbed probe from an image tensor."""
    probe = image.detach().clone()
    noise = torch.sin(torch.arange(probe.numel(), device=probe.device, dtype=probe.dtype)).reshape_as(probe)
    probe = probe + float(scale) * float(noise_scale) * noise
    if clip_min not in [None, "None", "none"] or clip_max not in [None, "None", "none"]:
        min_val = None if clip_min in [None, "None", "none"] else float(clip_min)
        max_val = None if clip_max in [None, "None", "none"] else float(clip_max)
        probe = torch.clamp(probe, min=min_val, max=max_val)
    return probe


def build_probe_variant(
    image: torch.Tensor,
    probe_type: str,
    scale: float,
    partner_image: Optional[torch.Tensor] = None,
    blend_alpha: float = 0.5,
    clip_min=None,
    clip_max=None,
) -> torch.Tensor:
    """Dispatch probe construction by type; new probe types can be added here."""
    probe_type = (probe_type or "scale").lower()
    if probe_type in ["clean", "anchor", "clean_anchor", "accuracy", "accuracy_anchor"]:
        return image.detach().clone()
    if probe_type in ["noise", "noise_anchor", "accuracy_noise"]:
        return build_noise_probe(image=image, scale=scale, clip_min=clip_min, clip_max=clip_max)
    if probe_type == "scale":
        return build_scaled_probe(image=image, scale=scale, clip_min=clip_min, clip_max=clip_max)
    if probe_type in ["blend", "scale_blend", "blend_scale"]:
        if partner_image is None:
            partner_image = image
        blended = build_blend_probe(
            image=image,
            partner_image=partner_image,
            alpha=blend_alpha,
            clip_min=None,
            clip_max=None,
        )
        return build_scaled_probe(image=blended, scale=scale, clip_min=clip_min, clip_max=clip_max)
    raise NotImplementedError(f"Probe type {probe_type} is not implemented yet.")
