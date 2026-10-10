"""Train a supervised server initializer on the fixed labeled public split only."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import datasetDict, n_classesDict, num_workersDict, testTransformDict, trainTransformDict
from data_splits import ReindexedSubset, load_and_validate_fixed_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--arch", default="ResNet18")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument(
        "--public_split_manifest",
        default="splits/cifar10_public_500x10_seed20260805.json",
    )
    parser.add_argument("--public_samples_per_class", type=int, default=500)
    parser.add_argument("--public_val_samples_per_class", type=int, default=250)
    parser.add_argument("--public_split_seed", type=int, default=20260805)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def build_dataset(args, train: bool, transform):
    from utilities import Utilities as Utils

    root = args.data_root or os.path.join("./datasets_pytorch", args.dataset)
    dataset_cls = Utils.get_index_dataset(datasetDict[args.dataset])
    return dataset_cls(root=root, train=train, download=True, transform=transform)


def build_fixed_public_subsets(
    train_base,
    validation_base,
    manifest_path,
    public_samples_per_class: int,
    validation_samples_per_class: int,
    split_seed: int,
):
    """Select public training and validation samples from one fixed manifest."""

    if len(train_base) != len(validation_base):
        raise ValueError("Training and validation dataset views must have equal length.")
    if list(train_base.targets) != list(validation_base.targets):
        raise ValueError("Training and validation dataset views must have identical labels.")

    public_indices, validation_indices, _, digest, public_counts, validation_counts = (
        load_and_validate_fixed_split(
            path=manifest_path,
            targets=train_base.targets,
            expected_public_per_class=public_samples_per_class,
            expected_validation_per_class=validation_samples_per_class,
            expected_seed=split_seed,
        )
    )
    return (
        ReindexedSubset(train_base, public_indices),
        ReindexedSubset(validation_base, validation_indices),
        {
            "manifest_digest": digest,
            "public_samples": len(public_indices),
            "validation_samples": len(validation_indices),
            "public_class_counts": public_counts,
            "validation_class_counts": validation_counts,
        },
    )


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x_input, y_target, _ in loader:
        x_input = x_input.to(device, non_blocking=True)
        y_target = y_target.to(device, non_blocking=True)
        pred = model(x_input).argmax(dim=1)
        correct += int((pred == y_target).sum().item())
        total += int(y_target.numel())
    return correct / max(total, 1)


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.public_val_samples_per_class <= 0:
        raise ValueError("public_val_samples_per_class must be positive for model selection.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    train_base = build_dataset(args, train=True, transform=trainTransformDict[args.dataset])
    validation_base = build_dataset(args, train=True, transform=testTransformDict[args.dataset])
    test_data = build_dataset(args, train=False, transform=testTransformDict[args.dataset])
    train_data, validation_data, split_metadata = build_fixed_public_subsets(
        train_base=train_base,
        validation_base=validation_base,
        manifest_path=args.public_split_manifest,
        public_samples_per_class=args.public_samples_per_class,
        validation_samples_per_class=args.public_val_samples_per_class,
        split_seed=args.public_split_seed,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": num_workersDict.get(args.dataset, 0),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    print(
        f"Public-only pretraining: train={len(train_data)}, "
        f"validation={len(validation_data)}, test={len(test_data)}"
    )
    print(f"Split manifest SHA256: {split_metadata['manifest_digest']}")

    model = getattr(importlib.import_module("models." + args.dataset), args.arch)().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.weight_decay > 0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(train_loader, desc=f"teacher epoch {epoch}/{args.epochs}", leave=True) as pbar:
            for x_input, y_target, _ in pbar:
                x_input = x_input.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(x_input)
                    loss = criterion(logits, y_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        scheduler.step()

        val_acc = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} val_acc={val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 1,
            "model": best_state,
            "dataset": args.dataset,
            "arch": args.arch,
            "n_classes": n_classesDict[args.dataset],
            "manifest_digest": split_metadata["manifest_digest"],
            "manifest_path": os.path.normpath(args.public_split_manifest),
            "public_samples": split_metadata["public_samples"],
            "validation_samples": split_metadata["validation_samples"],
            "public_samples_per_class": args.public_samples_per_class,
            "validation_samples_per_class": args.public_val_samples_per_class,
            "split_seed": args.public_split_seed,
            "training_seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_test_acc": test_acc,
        },
        args.output,
    )
    print("Saved public-only server initializer:", args.output)
    print(f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} final_test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
