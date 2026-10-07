"""Create the fixed class-balanced CIFAR-10 public/validation split manifest."""

import argparse
from pathlib import Path
import sys

from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_splits import build_fixed_stratified_manifest, save_split_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets_pytorch/cifar10")
    parser.add_argument(
        "--output",
        default="splits/cifar10_public_500x10_seed20260805.json",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--public-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=250)
    args = parser.parse_args()

    dataset = CIFAR10(root=args.dataset_root, train=True, download=True)
    manifest = build_fixed_stratified_manifest(
        targets=dataset.targets,
        public_samples_per_class=args.public_per_class,
        validation_samples_per_class=args.validation_per_class,
        seed=args.seed,
    )
    digest = save_split_manifest(manifest, args.output)

    print(f"Saved split manifest: {args.output}")
    print(f"Public samples: {len(manifest['public_indices'])}")
    print(f"Validation samples: {len(manifest['validation_indices'])}")
    print(f"Private samples: {len(dataset) - len(manifest['public_indices']) - len(manifest['validation_indices'])}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
