"""Validation and loading for optional server initialization checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch


def validate_server_initialization_config(
    checkpoint_path,
    dataset: str,
    public_split_mode: str,
    reinit_server: bool,
) -> None:
    """Reject configurations that cannot preserve checkpoint provenance or weights."""

    if checkpoint_path in [None, "none", "None"]:
        return
    if not (dataset == "cifar10" and str(public_split_mode).lower() == "fixed_stratified"):
        raise ValueError(
            "server_init_ckpt requires the fixed_stratified CIFAR-10 split so "
            "the checkpoint's public-data manifest can be verified."
        )
    if reinit_server:
        raise ValueError(
            "server_init_ckpt cannot be combined with reinit_server=true because "
            "server reinitialization would discard the pretrained weights."
        )


def _validate_metadata(payload: Dict[str, Any], expected: Dict[str, Any]) -> None:
    missing = [key for key in expected if key not in payload]
    if missing:
        raise ValueError(
            "Server checkpoint is missing required metadata: " + ", ".join(missing) + "."
        )

    for key, expected_value in expected.items():
        actual_value = payload[key]
        if key in {"n_classes", "public_samples"}:
            actual_value = int(actual_value)
        if actual_value != expected_value:
            raise ValueError(
                f"Server checkpoint metadata mismatch for {key}: "
                f"found {actual_value!r}, expected {expected_value!r}."
            )


def _state_dict_for_model(state_dict: Dict[str, torch.Tensor], model) -> OrderedDict:
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Server checkpoint model state must be a non-empty dictionary.")

    model_uses_module_prefix = all(key.startswith("module.") for key in model.state_dict())
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("Server checkpoint model state contains a non-string key.")
        clean_key = key[7:] if key.startswith("module.") else key
        target_key = f"module.{clean_key}" if model_uses_module_prefix else clean_key
        normalized[target_key] = value
    return normalized


def load_server_initial_checkpoint(
    model,
    checkpoint_path,
    dataset: str,
    arch: str,
    n_classes: int,
    manifest_digest: str,
    public_samples: int,
    map_location="cpu",
) -> Dict[str, Any]:
    """Validate checkpoint provenance and strictly load its model weights."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Server initialization checkpoint not found: {path}.")
    if not manifest_digest:
        raise ValueError("The fixed public split manifest digest is unavailable.")

    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Server checkpoint must be a metadata dictionary containing 'model'.")

    state_dict = payload.get("model", payload.get("state_dict"))
    if state_dict is None:
        raise ValueError(
            "Server checkpoint must contain a 'model' or 'state_dict' entry plus metadata."
        )

    expected = {
        "dataset": str(dataset),
        "arch": str(arch),
        "n_classes": int(n_classes),
        "manifest_digest": str(manifest_digest),
        "public_samples": int(public_samples),
    }
    _validate_metadata(payload, expected)
    model.load_state_dict(_state_dict_for_model(state_dict, model), strict=True)

    return {
        **{key: payload[key] for key in expected},
        "best_val_acc": payload.get("best_val_acc"),
        "checkpoint_path": str(path.resolve()),
    }
