"""Preprocessing helpers for logit diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to value."""
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def pad_logits(logits: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad [B, C] logits to [B, target_length] on the class dimension."""
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
    target_length = int(target_length)
    if logits.shape[1] > target_length:
        raise ValueError(f"Cannot pad class dimension {logits.shape[1]} to smaller length {target_length}.")
    if logits.shape[1] == target_length:
        return logits
    return F.pad(logits, (0, target_length - logits.shape[1]))


def unpad_logits(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Remove right-side padding and return [B, n_classes]."""
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
    return logits[:, :int(n_classes)]


def to_unet_sequence(logits: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Convert [B, C] logits to [B, 1, seq_length] for Unet1D."""
    padded = pad_logits(logits.float(), seq_length)
    return padded.unsqueeze(1)


def from_unet_sequence(sequence: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert [B, 1, L] Unet1D output back to [B, n_classes]."""
    if sequence.dim() != 3 or sequence.shape[1] != 1:
        raise ValueError(f"Expected sequence with shape [B, 1, L], got {tuple(sequence.shape)}")
    return unpad_logits(sequence.squeeze(1), n_classes)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities."""
    return torch.softmax(logits, dim=-1)


def probabilities_to_logits(probabilities: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert probabilities to log-probability logits for fallback use."""
    return torch.log(probabilities.clamp_min(eps))
