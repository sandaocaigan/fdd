"""Smoke test a diffusion purifier checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from diffusion_purifier import DiffusionLogitPurifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu"
    purifier = DiffusionLogitPurifier.from_checkpoint(
        checkpoint_path=args.ckpt,
        device=device,
        expected_dataset=args.dataset,
        expected_n_classes=args.n_classes,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
    )
    logits = torch.randn(args.batch_size, args.n_classes, device=device)
    purified_logits, purified_probs = purifier.purify(logits)
    print("input logits:", tuple(logits.shape))
    print("purified logits:", tuple(purified_logits.shape))
    print("purified probs:", tuple(purified_probs.shape))
    print("has nan:", bool(torch.isnan(purified_probs).any().item()))
    print("prob row sum min/max:", float(purified_probs.sum(dim=1).min()), float(purified_probs.sum(dim=1).max()))


if __name__ == "__main__":
    main()
