"""Training utilities for V1 Unet1D logit diffusion purifier."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from diffusion_purifier.checkpoint import save_checkpoint
from diffusion_purifier.models import build_unet1d, parse_dim_mults
from diffusion_purifier.preprocess import next_power_of_two, to_unet_sequence
from diffusion_purifier.sampler import LogitGaussianDiffusion


def load_logits_tensor(path: str) -> torch.Tensor:
    """Load logits from a tensor or a dict containing a 'logits' tensor."""
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        logits = payload
    elif isinstance(payload, dict) and "logits" in payload:
        logits = payload["logits"]
    else:
        raise ValueError(f"Expected a tensor or a dict with key 'logits' in {path}.")
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
    return logits.float()


def train_diffusion_from_logits(
    logits: torch.Tensor,
    output_path: str,
    dataset: str,
    arch: str,
    n_classes: int,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-4,
    timesteps: int = 100,
    model_dim: int = 32,
    dim_mults="1,2,4",
    device: str = "cuda:0",
) -> Dict:
    """Train a V1 diffusion purifier on clean logits and save a checkpoint."""
    device = torch.device(device if torch.cuda.is_available() or "cuda" not in str(device) else "cpu")
    n_classes = int(n_classes)
    seq_length = next_power_of_two(n_classes)
    sequence = to_unet_sequence(logits[:, :n_classes], seq_length)
    loader = DataLoader(TensorDataset(sequence), batch_size=int(batch_size), shuffle=True, drop_last=False)

    model = build_unet1d(dim=int(model_dim), dim_mults=parse_dim_mults(dim_mults), channels=1).to(device)
    diffusion = LogitGaussianDiffusion(model=model, seq_length=seq_length, timesteps=int(timesteps)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    model.train()
    for epoch in range(1, int(epochs) + 1):
        losses = []
        with tqdm(loader, desc=f"diffusion epoch {epoch}/{epochs}", leave=True) as pbar:
            for (batch,) in pbar:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = diffusion.training_loss(batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=f"{losses[-1]:.6f}")

    config = {
        "dataset": dataset,
        "arch": arch,
        "n_classes": n_classes,
        "seq_length": seq_length,
        "timesteps": int(timesteps),
        "model_dim": int(model_dim),
        "dim_mults": ",".join(str(x) for x in parse_dim_mults(dim_mults)),
        "version": "v1_raw_unet1d",
    }
    save_checkpoint(output_path, model=model, config=config)
    return config
