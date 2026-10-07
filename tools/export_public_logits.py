"""Export clean teacher logits on the server-side public dataset."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import n_classesDict, num_workersDict
from public_config import public_datasetAssignmentDict, public_testDataset_dict
from utilities import Utilities as Utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="ResNet18")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public_ds", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def load_state_dict(path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported teacher checkpoint format: {type(payload)}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    n_classes = n_classesDict[args.dataset]
    public_ds_name = args.public_ds or public_datasetAssignmentDict[args.dataset]
    public_dataset_cls = public_testDataset_dict[public_ds_name]
    public_root = os.path.join("./datasets_pytorch", public_ds_name)
    if public_ds_name == "cinic10":
        public_root = os.path.join(public_root, "valid")

    dataset = Utils.get_index_dataset(public_dataset_cls)(root=public_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workersDict.get(args.dataset, 0),
    )

    model = getattr(importlib.import_module("models." + args.dataset), args.arch)().to(device)
    model.load_state_dict(load_state_dict(args.teacher_ckpt, device), strict=True)
    model.eval()

    logits_store = torch.zeros(len(dataset), n_classes)
    with torch.no_grad():
        with tqdm(loader, desc="export public logits", leave=True) as pbar:
            for x_input, _, indices in pbar:
                x_input = x_input.to(device, non_blocking=True)
                with autocast(enabled=args.use_amp):
                    logits = model(x_input)
                logits_store[indices] = logits.detach().cpu()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "logits": logits_store,
            "dataset": args.dataset,
            "public_ds": public_ds_name,
            "arch": args.arch,
            "teacher_ckpt": args.teacher_ckpt,
        },
        args.output,
    )
    print("Saved public logits:", args.output)
    print("shape:", tuple(logits_store.shape))


if __name__ == "__main__":
    main()
