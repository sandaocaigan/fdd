"""Network builders for 1D diffusion purification."""

from __future__ import annotations

from typing import Iterable, Tuple


def parse_dim_mults(dim_mults) -> Tuple[int, ...]:
    """Parse dim_mults from a tuple/list/string such as '1,2,4'."""
    if dim_mults is None:
        return (1, 2, 4)
    if isinstance(dim_mults, str):
        values = [x.strip() for x in dim_mults.split(",") if x.strip()]
        return tuple(int(x) for x in values) or (1,)
    if isinstance(dim_mults, Iterable):
        return tuple(int(x) for x in dim_mults)
    return (int(dim_mults),)


def build_unet1d(dim: int = 32, dim_mults=(1, 2, 4), channels: int = 1):
    """Build lucidrains' Unet1D with a thin local wrapper."""
    try:
        from denoising_diffusion_pytorch import Unet1D
    except Exception as exc:
        raise ImportError(
            "denoising-diffusion-pytorch is required for DiffusionLogitPurifier. "
            "Install it with: python -m pip install denoising-diffusion-pytorch"
        ) from exc

    return Unet1D(
        dim=int(dim),
        dim_mults=parse_dim_mults(dim_mults),
        channels=int(channels),
    )
