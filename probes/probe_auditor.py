"""Probe auditing utilities for public prediction tensors."""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional

import torch
import wandb


class ProbeAuditor:
    """Compute entropy, confidence, SPC, and a combined risk score for each client."""

    def __init__(
        self,
        probe_metadata: Dict,
        n_classes: int,
        start_round: int = 1,
        w_entropy: float = 1.0,
        w_confidence: float = 1.0,
        w_spc: float = 1.0,
        log_each_round: bool = True,
    ):
        self.probe_metadata = probe_metadata or {}
        self.probe_indices = [int(x) for x in self.probe_metadata.get("probe_indices", [])]
        self.probe_groups = [[int(x) for x in group] for group in self.probe_metadata.get("probe_groups", [])]
        self.n_classes = int(n_classes)
        self.start_round = int(start_round or 1)
        self.w_entropy = float(w_entropy)
        self.w_confidence = float(w_confidence)
        self.w_spc = float(w_spc)
        self.log_each_round = bool(log_each_round)

    def should_audit(self, current_round: Optional[int]) -> bool:
        if not self.probe_indices:
            return False
        if current_round is None:
            return True
        return int(current_round) >= self.start_round

    def audit(self, client_prediction_list: List[torch.Tensor], clients=None, current_round: Optional[int] = None):
        if not self.should_audit(current_round):
            return []

        rows = []
        for client_idx, predictions in enumerate(client_prediction_list):
            probe_predictions = predictions[self.probe_indices]
            entropy = self._normalized_entropy(probe_predictions).mean()
            confidence = probe_predictions.max(dim=1).values.mean()
            spc = self._spc(predictions)

            entropy_risk = 1.0 - entropy
            weighted_sum = (
                self.w_entropy * entropy_risk
                + self.w_confidence * confidence
                + self.w_spc * spc
            )
            weight_total = self.w_entropy + self.w_confidence + self.w_spc
            risk = weighted_sum / max(weight_total, 1e-12)

            client = clients[client_idx] if clients is not None else None
            role = "malicious" if getattr(client, "is_byzantine", False) else "benign"
            client_id = getattr(client, "client_id", client_idx)
            rows.append({
                "client_id": int(client_id),
                "role": role,
                "entropy": float(entropy.detach().cpu()),
                "confidence": float(confidence.detach().cpu()),
                "spc": float(spc.detach().cpu()),
                "risk": float(risk.detach().cpu()),
            })

        if self.log_each_round:
            self.print_rows(rows=rows, current_round=current_round)
            self.log_to_wandb(rows=rows, current_round=current_round)
        return rows

    def _normalized_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        probabilities = probabilities.clamp_min(eps)
        entropy = -(probabilities * probabilities.log()).sum(dim=1)
        return entropy / math.log(self.n_classes)

    def _spc(self, predictions: torch.Tensor) -> torch.Tensor:
        if not self.probe_groups:
            return torch.tensor(0.0, device=predictions.device)
        group_scores = []
        for group in self.probe_groups:
            group_pred = predictions[group].argmax(dim=1)
            base_pred = group_pred[0]
            group_scores.append((group_pred == base_pred).float().mean())
        return torch.stack(group_scores).mean()

    def print_rows(self, rows: List[Dict], current_round: Optional[int]):
        sys.stdout.write(f"\n[Probe Audit] Round {current_round}\n")
        sys.stdout.write("client_id | role      | entropy | confidence | spc    | risk\n")
        for row in sorted(rows, key=lambda x: x["client_id"]):
            sys.stdout.write(
                f"{row['client_id']:<9} | {row['role']:<9} | "
                f"{row['entropy']:.4f}  | {row['confidence']:.4f}     | "
                f"{row['spc']:.4f} | {row['risk']:.4f}\n"
            )

    def log_to_wandb(self, rows: List[Dict], current_round: Optional[int]):
        if wandb.run is None:
            return
        log_dict = {}
        for row in rows:
            prefix = f"probe/client{row['client_id']}"
            log_dict[f"{prefix}/entropy"] = row["entropy"]
            log_dict[f"{prefix}/confidence"] = row["confidence"]
            log_dict[f"{prefix}/spc"] = row["spc"]
            log_dict[f"{prefix}/risk"] = row["risk"]
            log_dict[f"{prefix}/is_malicious"] = 1 if row["role"] == "malicious" else 0
        if current_round is not None:
            log_dict["probe/round"] = int(current_round)
        wandb.log(log_dict, commit=False)
