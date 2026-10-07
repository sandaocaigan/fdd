"""Reproducible, disjoint dataset splits used by the experiment runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class ReindexedSubset(Dataset):
    """Expose selected source samples with contiguous local indices."""

    def __init__(self, dataset: Dataset, source_indices: Sequence[int]):
        self.dataset = dataset
        self.source_indices = tuple(int(index) for index in source_indices)

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, local_index: int):
        source_index = self.source_indices[local_index]
        item = self.dataset[source_index]
        if not isinstance(item, (tuple, list)) or len(item) not in (2, 3):
            raise ValueError("Indexed datasets must return (image, label) or (image, label, index).")
        image, label = item[0], item[1]
        return image, label, local_index


def build_fixed_stratified_manifest(
    targets: Iterable[int],
    public_samples_per_class: int,
    validation_samples_per_class: int,
    seed: int,
) -> Dict:
    """Select fixed public and validation indices independently in every class."""

    labels = torch.as_tensor(list(targets), dtype=torch.long)
    classes = sorted(int(value) for value in labels.unique().tolist())
    generator = torch.Generator().manual_seed(int(seed))
    public_indices: List[int] = []
    validation_indices: List[int] = []

    for class_id in classes:
        class_indices = torch.where(labels == class_id)[0]
        required = int(public_samples_per_class) + int(validation_samples_per_class)
        if required > len(class_indices):
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} samples, but {required} are required."
            )
        permutation = torch.randperm(len(class_indices), generator=generator)
        shuffled = class_indices[permutation]
        public_indices.extend(shuffled[:public_samples_per_class].tolist())
        validation_indices.extend(
            shuffled[
                public_samples_per_class:
                public_samples_per_class + validation_samples_per_class
            ].tolist()
        )

    return {
        "version": 1,
        "dataset": "cifar10",
        "dataset_size": len(labels),
        "split_seed": int(seed),
        "public_samples_per_class": int(public_samples_per_class),
        "validation_samples_per_class": int(validation_samples_per_class),
        "classes": classes,
        "public_indices": sorted(int(index) for index in public_indices),
        "validation_indices": sorted(int(index) for index in validation_indices),
    }


def save_split_manifest(manifest: Dict, path) -> str:
    """Write a split manifest and return its canonical content digest."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    serialized_bytes = serialized.encode("utf-8")
    manifest_path.write_bytes(serialized_bytes)
    return _manifest_digest(manifest)


def load_and_validate_fixed_split(
    path,
    targets: Iterable[int],
    expected_public_per_class: int,
    expected_validation_per_class: int,
    expected_seed: int,
) -> Tuple[List[int], List[int], List[int], str, List[int], List[int]]:
    """Load a manifest and return validated public, validation and private indices."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fixed public split manifest not found: {manifest_path}. "
            "Run python tools/create_cifar10_fixed_split.py first."
        )

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    labels = [int(value) for value in targets]
    dataset_size = len(labels)

    expected_fields = {
        "dataset": "cifar10",
        "dataset_size": dataset_size,
        "split_seed": int(expected_seed),
        "public_samples_per_class": int(expected_public_per_class),
        "validation_samples_per_class": int(expected_validation_per_class),
    }
    for field, expected in expected_fields.items():
        actual = manifest.get(field)
        if actual != expected:
            raise ValueError(
                f"Split manifest field '{field}' is {actual!r}, expected {expected!r}."
            )

    public_indices = [int(index) for index in manifest.get("public_indices", [])]
    validation_indices = [int(index) for index in manifest.get("validation_indices", [])]
    _validate_indices(public_indices, dataset_size, "public")
    _validate_indices(validation_indices, dataset_size, "validation")

    public_set = set(public_indices)
    validation_set = set(validation_indices)
    overlap = public_set & validation_set
    if overlap:
        raise ValueError(f"Public and validation splits overlap at {len(overlap)} indices.")

    classes = sorted(set(labels))
    public_counts = _class_counts(public_indices, labels, classes)
    validation_counts = _class_counts(validation_indices, labels, classes)
    if any(count != expected_public_per_class for count in public_counts):
        raise ValueError(f"Invalid public class counts: {public_counts}.")
    if any(count != expected_validation_per_class for count in validation_counts):
        raise ValueError(f"Invalid validation class counts: {validation_counts}.")

    excluded = public_set | validation_set
    private_indices = [index for index in range(dataset_size) if index not in excluded]
    digest = _manifest_digest(manifest)
    return (
        public_indices,
        validation_indices,
        private_indices,
        digest,
        public_counts,
        validation_counts,
    )


def _validate_indices(indices: Sequence[int], dataset_size: int, split_name: str) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError(f"The {split_name} split contains duplicate indices.")
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise ValueError(f"The {split_name} split contains out-of-range indices: {invalid[:5]}.")


def _class_counts(indices: Sequence[int], labels: Sequence[int], classes: Sequence[int]) -> List[int]:
    counts = {class_id: 0 for class_id in classes}
    for index in indices:
        counts[labels[index]] += 1
    return [counts[class_id] for class_id in classes]


def _manifest_digest(manifest: Dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
