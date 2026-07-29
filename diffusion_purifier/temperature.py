"""Optional temperature calibration helpers for future purifier versions."""

from __future__ import annotations

import torch


def entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits / float(temperature), dim=-1)
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def match_entropy_temperature(
    logits: torch.Tensor,
    target_entropy: torch.Tensor | float,
    low: float = 1e-2,
    high: float = 100.0,
    n_iter: int = 30,
) -> float:
    """Find a scalar temperature whose mean entropy matches target_entropy."""
    if isinstance(target_entropy, torch.Tensor):
        target = float(target_entropy.detach().mean().cpu())
    else:
        target = float(target_entropy)

    left, right = float(low), float(high)
    for _ in range(int(n_iter)):
        mid = 0.5 * (left + right)
        entropy = float(entropy_from_logits(logits, mid).mean().detach().cpu())
        if entropy < target:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)
