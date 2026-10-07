"""Train a clean supervised teacher checkpoint on the selected labeled dataset.

This is a utility script. It is intentionally outside the federated training
loop and does not touch client-private splits.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import datasetDict, n_classesDict, num_workersDict, testTransformDict, trainTransformDict
from utilities import Utilities as Utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="ResNet18")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def build_dataset(args, train: bool):
    root = args.data_root or os.path.join("./datasets_pytorch", args.dataset)
    dataset_cls = Utils.get_index_dataset(datasetDict[args.dataset])
    if args.dataset in ["cinic10", "imagenet100", "clothing1m"]:
        split = "train" if train else "test"
        return dataset_cls(
            root=os.path.join(root, split),
            transform=trainTransformDict[args.dataset] if train else testTransformDict[args.dataset],
        )
    return dataset_cls(
        root=root,
        train=train,
        download=True,
        transform=trainTransformDict[args.dataset] if train else testTransformDict[args.dataset],
    )


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    train_data = build_dataset(args, train=True)
    test_data = build_dataset(args, train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workersDict.get(args.dataset, 0),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workersDict.get(args.dataset, 0),
    )

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

    best_acc = 0.0
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

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_input, y_target, _ in test_loader:
                x_input = x_input.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                pred = model(x_input).argmax(dim=1)
                correct += int((pred == y_target).sum().item())
                total += int(y_target.numel())
        acc = correct / max(total, 1)
        print(f"epoch={epoch} test_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model": best_state or model.state_dict(),
            "dataset": args.dataset,
            "arch": args.arch,
            "n_classes": n_classesDict[args.dataset],
            "best_acc": best_acc,
        },
        args.output,
    )
    print("Saved clean teacher:", args.output)
    print("best_acc:", best_acc)


if __name__ == "__main__":
    main()
