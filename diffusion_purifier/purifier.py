"""Runtime interface for diffusion-based logit purification."""

from __future__ import annotations

from typing import Optional

import torch

from diffusion_purifier.checkpoint import load_checkpoint
from diffusion_purifier.models import build_unet1d, parse_dim_mults
from diffusion_purifier.preprocess import (
    from_unet_sequence,
    logits_to_probabilities,
    next_power_of_two,
    to_unet_sequence,
)
from diffusion_purifier.sampler import LogitGaussianDiffusion


class DiffusionLogitPurifier:
    """Load a pretrained 1D diffusion model and purify client logits."""

    def __init__(
        self,
        model: torch.nn.Module,
        n_classes: int,
        seq_length: int,
        timesteps: int,
        device,
        batch_size: int = 512,
        diffusion_steps: int = 10,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.n_classes = int(n_classes)
        self.seq_length = int(seq_length)
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion = LogitGaussianDiffusion(
            model=self.model,
            seq_length=self.seq_length,
            timesteps=self.timesteps,
        ).to(self.device)
        self.diffusion.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device,
        expected_dataset: Optional[str] = None,
        expected_n_classes: Optional[int] = None,
        batch_size: int = 512,
        diffusion_steps: int = 10,
    ):
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        config = dict(checkpoint.get("config", {}))
        n_classes = int(config.get("n_classes", expected_n_classes))
        if expected_n_classes is not None and n_classes != int(expected_n_classes):
            raise ValueError(
                f"Diffusion checkpoint n_classes={n_classes}, expected {expected_n_classes}."
            )
        dataset = config.get("dataset")
        if expected_dataset not in [None, "None", "none"] and dataset not in [None, expected_dataset]:
            raise ValueError(f"Diffusion checkpoint dataset={dataset}, expected {expected_dataset}.")

        seq_length = int(config.get("seq_length", next_power_of_two(n_classes)))
        model_dim = int(config.get("model_dim", 32))
        dim_mults = parse_dim_mults(config.get("dim_mults", "1,2,4"))
        timesteps = int(config.get("timesteps", 100))

        model = build_unet1d(dim=model_dim, dim_mults=dim_mults, channels=1)
        model.load_state_dict(checkpoint["model"], strict=True)
        return cls(
            model=model,
            n_classes=n_classes,
            seq_length=seq_length,
            timesteps=timesteps,
            device=device,
            batch_size=batch_size,
            diffusion_steps=diffusion_steps,
        )

    @torch.no_grad()
    def purify(self, logits: torch.Tensor, return_probs: bool = True):
        """Purify [N, C] logits and optionally return softmax probabilities."""
        if logits.dim() != 2:
            raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
        purified_chunks = []
        for start in range(0, logits.shape[0], self.batch_size):
            batch = logits[start:start + self.batch_size].to(self.device).float()
            sequence = to_unet_sequence(batch, self.seq_length)
            purified_sequence = self.diffusion.denoise_from_input(
                x_start=sequence,
                sampling_steps=self.diffusion_steps,
            )
            purified_logits = from_unet_sequence(purified_sequence, self.n_classes)
            purified_chunks.append(purified_logits)

        purified_logits = torch.cat(purified_chunks, dim=0)
        if not return_probs:
            return purified_logits
        return purified_logits, logits_to_probabilities(purified_logits)
