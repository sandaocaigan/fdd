"""A small DDPM-style sampler for logit purification.

The purifier uses lucidrains' Unet1D as the noise predictor, but keeps the
diffusion math local so inference can denoise a given logit vector instead of
only sampling from pure noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape):
    """Extract per-sample coefficients and reshape them for broadcasting."""
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class LogitGaussianDiffusion(nn.Module):
    """Minimal Gaussian diffusion wrapper for [B, 1, L] logit sequences."""

    def __init__(
        self,
        model: nn.Module,
        seq_length: int,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.model = model
        self.seq_length = int(seq_length)
        self.timesteps = int(timesteps)

        betas = torch.linspace(float(beta_start), float(beta_end), self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None):
        """Diffuse clean samples x_start to timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def training_loss(self, x_start: torch.Tensor):
        """Standard noise-prediction DDPM objective."""
        if x_start.dim() != 3:
            raise ValueError(f"Expected [B, 1, L] input, got {tuple(x_start.shape)}")
        batch_size = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.model(noisy, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, timesteps: torch.Tensor, step_index: int):
        """One reverse DDPM step."""
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, timesteps, x.shape)

        predicted_noise = self.model(x, timesteps)
        model_mean = sqrt_recip_alpha * (x - betas_t * predicted_noise / sqrt_one_minus)

        if step_index == 0:
            return model_mean
        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

    @torch.no_grad()
    def denoise_from_input(self, x_start: torch.Tensor, sampling_steps: int = 10):
        """Add moderate noise to x_start and run the reverse process back to t=0."""
        if sampling_steps <= 0:
            return x_start
        start_step = min(int(sampling_steps), self.timesteps) - 1
        batch_size = x_start.shape[0]
        timesteps = torch.full((batch_size,), start_step, device=x_start.device, dtype=torch.long)
        x = self.q_sample(x_start, timesteps)

        for step in range(start_step, -1, -1):
            t = torch.full((batch_size,), step, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, step)
        return x

    @torch.no_grad()
    def sample(self, batch_size: int, device=None):
        """Generate samples from pure noise. Used only for diagnostics."""
        device = device or next(self.model.parameters()).device
        x = torch.randn(int(batch_size), 1, self.seq_length, device=device)
        for step in range(self.timesteps - 1, -1, -1):
            t = torch.full((int(batch_size),), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, step)
        return x
