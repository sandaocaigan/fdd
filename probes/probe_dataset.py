"""Dataset wrapper that replaces a small subset of public samples with probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from probes.probe_builder import build_probe_variant, parse_probe_scales


@dataclass(frozen=True)
class ProbeEntry:
    replacement_index: int
    base_index: int
    group_id: int
    scale: float


class ProbeDatasetWrapper(Dataset):
    """Wrap a public dataset and replace selected indices with generated probe samples."""

    def __init__(
        self,
        base_dataset: Dataset,
        probe_base_count: int,
        probe_scales,
        probe_type: str = "scale",
        probe_seed: int = 0,
        mode: str = "replace",
        metadata: Optional[Dict] = None,
        clip_min=None,
        clip_max=None,
    ):
        if mode != "replace":
            raise NotImplementedError("Only replace mode is implemented for the first probe version.")
        self.base_dataset = base_dataset
        self.probe_type = probe_type
        self.probe_scales = parse_probe_scales(probe_scales)
        self.probe_base_count = int(probe_base_count)
        self.probe_seed = int(probe_seed)
        self.mode = mode
        self.clip_min = clip_min
        self.clip_max = clip_max

        if metadata is None:
            metadata = self._build_metadata()
        self.metadata = metadata
        self.entries_by_index = {
            int(entry["replacement_index"]): ProbeEntry(
                replacement_index=int(entry["replacement_index"]),
                base_index=int(entry["base_index"]),
                group_id=int(entry["group_id"]),
                scale=float(entry["scale"]),
            )
            for entry in self.metadata["entries"]
        }

    def _build_metadata(self) -> Dict:
        dataset_len = len(self.base_dataset)
        n_probe_samples = self.probe_base_count * len(self.probe_scales)
        if n_probe_samples <= 0:
            return {"probe_indices": [], "probe_groups": [], "entries": [], "probe_scales": self.probe_scales}
        if n_probe_samples + self.probe_base_count > dataset_len:
            raise ValueError(
                f"Probe setup needs {n_probe_samples + self.probe_base_count} public samples, "
                f"but dataset only has {dataset_len}."
            )

        generator = torch.Generator().manual_seed(self.probe_seed)
        perm = torch.randperm(dataset_len, generator=generator).tolist()
        replacement_indices = perm[:n_probe_samples]
        base_indices = perm[n_probe_samples:n_probe_samples + self.probe_base_count]

        entries: List[Dict] = []
        probe_groups: List[List[int]] = []
        cursor = 0
        for group_id, base_index in enumerate(base_indices):
            group_indices = []
            for scale in self.probe_scales:
                replacement_index = replacement_indices[cursor]
                cursor += 1
                entries.append({
                    "replacement_index": int(replacement_index),
                    "base_index": int(base_index),
                    "group_id": int(group_id),
                    "scale": float(scale),
                })
                group_indices.append(int(replacement_index))
            probe_groups.append(group_indices)

        return {
            "probe_indices": [int(x) for x in replacement_indices],
            "probe_groups": probe_groups,
            "entries": entries,
            "probe_scales": [float(x) for x in self.probe_scales],
            "probe_type": self.probe_type,
            "mode": self.mode,
            "probe_seed": self.probe_seed,
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if idx in self.entries_by_index:
            entry = self.entries_by_index[idx]
            image, label, _ = self.base_dataset[entry.base_index]
            image = build_probe_variant(
                image=image,
                probe_type=self.probe_type,
                scale=entry.scale,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
            return image, label, idx
        image, label, _ = self.base_dataset[idx]
        return image, label, idx
