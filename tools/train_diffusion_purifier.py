"""Train a V1 Unet1D diffusion purifier from exported clean public logits."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from config import n_classesDict
from diffusion_purifier.trainer import load_logits_tensor, train_diffusion_from_logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="ResNet18")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--dim_mults", default="1,2,4")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    n_classes = args.n_classes or n_classesDict[args.dataset]
    logits = load_logits_tensor(args.logits_path)
    if logits.shape[1] < n_classes:
        raise ValueError(f"Logits have {logits.shape[1]} classes, expected at least {n_classes}.")

    config = train_diffusion_from_logits(
        logits=logits,
        output_path=args.output,
        dataset=args.dataset,
        arch=args.arch,
        n_classes=n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        model_dim=args.model_dim,
        dim_mults=args.dim_mults,
        device=args.device,
    )
    print("Saved diffusion purifier:", args.output)
    print("Config:", config)


if __name__ == "__main__":
    main()
