"""Dataset wrapper for active probes and untouched-public-data audit controls."""

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
    partner_index: Optional[int]
    group_id: int
    scale: float
    label: Optional[int]


class ProbeDatasetWrapper(Dataset):
    """Wrap a public dataset with generated probes or an audit-only index selection."""

    def __init__(
        self,
        base_dataset: Dataset,
        probe_base_count: int,
        probe_scales,
        probe_type: str = "scale",
        probe_seed: int = 0,
        mode: str = "replace",
        metadata: Optional[Dict] = None,
        blend_alpha: float = 0.5,
        blend_mode: str = "random",
        clip_min=None,
        clip_max=None,
    ):
        if mode not in ["replace", "audit_only"]:
            raise ValueError("probe_mode must be 'replace' or 'audit_only'.")
        self.base_dataset = base_dataset
        self.probe_type = probe_type
        self.probe_scales = parse_probe_scales(probe_scales)
        self.probe_base_count = int(probe_base_count)
        self.probe_seed = int(probe_seed)
        self.mode = mode
        self.blend_alpha = float(blend_alpha)
        self.blend_mode = str(blend_mode or "random").lower()
        self.clip_min = clip_min
        self.clip_max = clip_max

        if metadata is None:
            metadata = self._build_metadata()
        self.metadata = metadata
        self.entries_by_index = {
            int(entry["replacement_index"]): ProbeEntry(
                replacement_index=int(entry["replacement_index"]),
                base_index=int(entry["base_index"]),
                partner_index=None if entry.get("partner_index") is None else int(entry["partner_index"]),
                group_id=int(entry["group_id"]),
                scale=float(entry["scale"]),
                label=None if entry.get("label") is None else int(entry["label"]),
            )
            for entry in self.metadata["entries"]
        }

    def _uses_partner_image(self) -> bool:
        return str(self.probe_type or "scale").lower() in ["blend", "scale_blend", "blend_scale"]

    def _build_metadata(self) -> Dict:
        dataset_len = len(self.base_dataset)
        n_probe_samples = self.probe_base_count * len(self.probe_scales)
        if n_probe_samples <= 0:
            return {"probe_indices": [], "probe_groups": [], "entries": [], "probe_scales": self.probe_scales}

        generator = torch.Generator().manual_seed(self.probe_seed)
        perm = torch.randperm(dataset_len, generator=generator).tolist()
        if self.mode == "audit_only":
            if n_probe_samples > dataset_len:
                raise ValueError(
                    f"Audit setup needs {n_probe_samples} public samples, "
                    f"but dataset only has {dataset_len}."
                )
            audit_indices = perm[:n_probe_samples]
            entries = []
            for group_id, audit_index in enumerate(audit_indices):
                _, label, _ = self.base_dataset[audit_index]
                entries.append({
                    "replacement_index": int(audit_index),
                    "base_index": int(audit_index),
                    "partner_index": None,
                    "group_id": int(group_id),
                    "scale": 1.0,
                    "label": self._label_to_int(label),
                })
            return {
                "probe_indices": [int(x) for x in audit_indices],
                # Singleton groups make SPC=1, so hybrid scoring reduces to consensus.
                "probe_groups": [[int(x)] for x in audit_indices],
                "entries": entries,
                "probe_scales": [1.0],
                "probe_type": "audit_only",
                "mode": self.mode,
                "probe_seed": self.probe_seed,
                "blend_alpha": self.blend_alpha,
                "blend_mode": self.blend_mode,
            }

        uses_partner = self._uses_partner_image()
        n_partner_samples = self.probe_base_count if uses_partner else 0
        n_required_samples = n_probe_samples + self.probe_base_count + n_partner_samples
        if n_required_samples > dataset_len:
            raise ValueError(
                f"Probe setup needs {n_required_samples} public samples, "
                f"but dataset only has {dataset_len}."
            )

        replacement_indices = perm[:n_probe_samples]
        base_indices = perm[n_probe_samples:n_probe_samples + self.probe_base_count]
        partner_start = n_probe_samples + self.probe_base_count
        partner_indices = (
            perm[partner_start:partner_start + n_partner_samples]
            if uses_partner
            else [None for _ in range(self.probe_base_count)]
        )

        entries: List[Dict] = []
        probe_groups: List[List[int]] = []
        cursor = 0
        for group_id, base_index in enumerate(base_indices):
            partner_index = partner_indices[group_id]
            _, base_label, _ = self.base_dataset[base_index]
            base_label = self._label_to_int(base_label)
            group_indices = []
            for scale in self.probe_scales:
                replacement_index = replacement_indices[cursor]
                cursor += 1
                entries.append({
                    "replacement_index": int(replacement_index),
                    "base_index": int(base_index),
                    "partner_index": None if partner_index is None else int(partner_index),
                    "group_id": int(group_id),
                    "scale": float(scale),
                    "label": base_label,
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
            "blend_alpha": self.blend_alpha,
            "blend_mode": self.blend_mode,
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.mode == "audit_only":
            return self.base_dataset[idx]
        if idx in self.entries_by_index:
            entry = self.entries_by_index[idx]
            image, label, _ = self.base_dataset[entry.base_index]
            partner_image = None
            if entry.partner_index is not None:
                partner_image, _, _ = self.base_dataset[entry.partner_index]
            image = build_probe_variant(
                image=image,
                probe_type=self.probe_type,
                scale=entry.scale,
                partner_image=partner_image,
                blend_alpha=self.blend_alpha,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
            return image, label, idx
        image, label, _ = self.base_dataset[idx]
        return image, label, idx

    @staticmethod
    def _label_to_int(label):
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                return None
            label = label.item()
        try:
            return int(label)
        except (TypeError, ValueError):
            return None
