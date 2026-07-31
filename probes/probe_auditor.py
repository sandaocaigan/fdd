"""Probe auditing utilities for public prediction tensors."""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional

import torch
import wandb


class ProbeAuditor:
    """Compute prediction, hard-SPC, and soft multi-scale consistency audit metrics."""

    def __init__(
        self,
        probe_metadata: Dict,
        n_classes: int,
        start_round: int = 1,
        score_mode: str = "hybrid",
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        w_entropy: float = 1.0,
        w_confidence: float = 1.0,
        w_spc: float = 1.0,
        w_target_bias: float = 0.0,
        w_consensus: float = 1.0,
        w_label: float = 1.0,
        log_each_round: bool = True,
    ):
        self.probe_metadata = probe_metadata or {}
        self.probe_indices = [int(x) for x in self.probe_metadata.get("probe_indices", [])]
        self.probe_groups = [[int(x) for x in group] for group in self.probe_metadata.get("probe_groups", [])]
        self.probe_labels = self._build_probe_labels()
        self.n_classes = int(n_classes)
        self.start_round = int(start_round or 1)
        self.score_mode = str(score_mode or "hybrid").lower()
        self.target_class = None if target_class in [None, "None", "none"] else int(target_class)
        self.source_class = None if source_class in [None, "None", "none"] else int(source_class)
        self.w_entropy = float(w_entropy)
        self.w_confidence = float(w_confidence)
        self.w_spc = float(w_spc)
        self.w_target_bias = float(w_target_bias)
        self.w_consensus = float(w_consensus)
        self.w_label = float(w_label)
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

        stacked_predictions = torch.stack([pred[self.probe_indices] for pred in client_prediction_list], dim=0)
        consensus_predictions = torch.median(stacked_predictions, dim=0).values

        rows = []
        for client_idx, predictions in enumerate(client_prediction_list):
            probe_predictions = predictions[self.probe_indices]
            entropy = self._normalized_entropy(probe_predictions).mean()
            confidence = probe_predictions.max(dim=1).values.mean()
            spc = self._spc(predictions)
            soft_spc_stats = self._soft_spc(predictions)
            target_bias_stats = self._target_bias(probe_predictions)
            consensus_stats = self._consensus_deviation(probe_predictions, consensus_predictions)
            label_stats = self._label_quality(probe_predictions)

            entropy_risk = 1.0 - entropy
            uncertainty_risk = ((entropy + (1.0 - confidence)) / 2.0).clamp(0.0, 1.0)
            weighted_sum, weight_total = self._weighted_risk(
                entropy_risk=entropy_risk,
                uncertainty_risk=uncertainty_risk,
                confidence=confidence,
                spc=spc,
                soft_spc_tv_risk=soft_spc_stats["tv_risk"],
                soft_spc_js_risk=soft_spc_stats["js_risk"],
                target_bias_score=target_bias_stats["target_bias_score"],
                consensus_risk=consensus_stats["consensus_risk"],
                label_error=label_stats["label_error"],
                has_valid_labels=label_stats["has_valid_labels"],
            )
            risk = weighted_sum / weight_total.clamp_min(1e-12)

            client = clients[client_idx] if clients is not None else None
            role = "malicious" if getattr(client, "is_byzantine", False) else "benign"
            client_id = getattr(client, "client_id", client_idx)
            rows.append({
                "client_id": int(client_id),
                "role": role,
                "entropy": float(entropy.detach().cpu()),
                "confidence": float(confidence.detach().cpu()),
                "spc": float(spc.detach().cpu()),
                "soft_spc_tv": float(soft_spc_stats["tv_risk"].detach().cpu()),
                "soft_spc_js": float(soft_spc_stats["js_risk"].detach().cpu()),
                "target_prob": float(target_bias_stats["target_prob"].detach().cpu()),
                "source_prob": float(target_bias_stats["source_prob"].detach().cpu()),
                "target_bias": float(target_bias_stats["target_bias"].detach().cpu()),
                "consensus_risk": float(consensus_stats["consensus_risk"].detach().cpu()),
                "label_prob": float(label_stats["label_prob"].detach().cpu()),
                "label_error": float(label_stats["label_error"].detach().cpu()),
                "has_valid_labels": bool(label_stats["has_valid_labels"]),
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

    def _weighted_risk(
        self,
        entropy_risk,
        uncertainty_risk,
        confidence,
        spc,
        soft_spc_tv_risk,
        soft_spc_js_risk,
        target_bias_score,
        consensus_risk,
        label_error,
        has_valid_labels,
    ):
        if self.score_mode == "target_bias":
            return target_bias_score, torch.tensor(1.0, device=target_bias_score.device)
        if self.score_mode in ["consensus", "consensus_risk"]:
            return consensus_risk, torch.tensor(1.0, device=consensus_risk.device)
        spc_risk = (1.0 - spc).clamp(0.0, 1.0)
        if self.score_mode in ["spc", "spc_risk", "scale", "scale_risk", "consistency"]:
            return spc_risk, torch.tensor(1.0, device=spc_risk.device)
        if self.score_mode in ["soft_spc_tv", "soft_tv", "tv_spc", "tv_consistency"]:
            return soft_spc_tv_risk, torch.tensor(1.0, device=soft_spc_tv_risk.device)
        if self.score_mode in ["soft_spc_js", "soft_js", "js_spc", "js_consistency"]:
            return soft_spc_js_risk, torch.tensor(1.0, device=soft_spc_js_risk.device)
        if self.score_mode in ["hybrid", "consensus_spc", "probe_hybrid"]:
            consensus_weight = max(float(self.w_consensus), 0.0)
            spc_weight = max(float(self.w_spc), 0.0)
            weight_total = consensus_weight + spc_weight
            if weight_total <= 0:
                return consensus_risk, torch.tensor(1.0, device=consensus_risk.device)
            weighted_sum = consensus_weight * consensus_risk + spc_weight * spc_risk
            return weighted_sum, torch.tensor(float(weight_total), device=weighted_sum.device)
        if self.score_mode in ["uniform", "uniform_risk", "uncertainty", "flatness"]:
            return uncertainty_risk, torch.tensor(1.0, device=uncertainty_risk.device)
        if self.score_mode in ["degradation", "degradation_risk", "accuracy_degradation"]:
            degradation_risk = torch.maximum(consensus_risk, uncertainty_risk)
            return degradation_risk, torch.tensor(1.0, device=degradation_risk.device)
        if self.score_mode in ["label", "label_risk", "accuracy", "accuracy_risk"]:
            if has_valid_labels:
                return label_error, torch.tensor(1.0, device=label_error.device)
            return consensus_risk, torch.tensor(1.0, device=consensus_risk.device)

        weighted_sum = (
            self.w_entropy * entropy_risk
            + self.w_confidence * confidence
            + self.w_spc * spc_risk
        )
        weight_total = self.w_entropy + self.w_confidence + self.w_spc
        if self.score_mode in ["hybrid", "scale_target", "target_scale"] and self.w_target_bias > 0:
            weighted_sum = weighted_sum + self.w_target_bias * target_bias_score
            weight_total = weight_total + self.w_target_bias
        if self.score_mode in ["uniform_hybrid", "uncertainty_hybrid", "flatness_hybrid"]:
            weighted_sum = self.w_consensus * consensus_risk + self.w_entropy * uncertainty_risk
            weight_total = self.w_consensus + self.w_entropy
            if has_valid_labels and self.w_label > 0:
                weighted_sum = weighted_sum + self.w_label * label_error
                weight_total = weight_total + self.w_label
            return weighted_sum, torch.tensor(float(weight_total), device=weighted_sum.device)
        if self.score_mode in ["accuracy_hybrid", "quality", "quality_hybrid"]:
            weighted_sum = self.w_consensus * consensus_risk
            weight_total = self.w_consensus
            if has_valid_labels and self.w_label > 0:
                weighted_sum = weighted_sum + self.w_label * label_error
                weight_total = weight_total + self.w_label
            if self.w_spc > 0:
                weighted_sum = weighted_sum + self.w_spc * spc_risk
                weight_total = weight_total + self.w_spc
        return weighted_sum, torch.tensor(float(weight_total), device=weighted_sum.device)

    def _build_probe_labels(self) -> List[Optional[int]]:
        label_by_index = {}
        for entry in self.probe_metadata.get("entries", []):
            label = entry.get("label")
            if label is not None:
                try:
                    label = int(label)
                except (TypeError, ValueError):
                    label = None
            label_by_index[int(entry["replacement_index"])] = label
        return [label_by_index.get(int(index)) for index in self.probe_indices]

    def _target_bias(self, probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=probabilities.device)
        if self.target_class is None or self.target_class < 0 or self.target_class >= self.n_classes:
            return {
                "target_prob": zero,
                "source_prob": zero,
                "target_bias": zero,
                "target_bias_score": zero,
            }

        target_prob_per_sample = probabilities[:, self.target_class]
        target_prob = target_prob_per_sample.mean()
        if self.source_class is not None and 0 <= self.source_class < self.n_classes:
            source_prob_per_sample = probabilities[:, self.source_class]
        else:
            non_target_sum = probabilities.sum(dim=1) - target_prob_per_sample
            source_prob_per_sample = non_target_sum / max(self.n_classes - 1, 1)
        source_prob = source_prob_per_sample.mean()
        target_bias = target_prob - source_prob
        target_bias_score = ((target_bias + 1.0) / 2.0).clamp(0.0, 1.0)
        return {
            "target_prob": target_prob,
            "source_prob": source_prob,
            "target_bias": target_bias,
            "target_bias_score": target_bias_score,
        }

    def _consensus_deviation(self, probabilities: torch.Tensor, consensus_probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_variation = 0.5 * torch.abs(probabilities - consensus_probabilities).sum(dim=1)
        consensus_risk = total_variation.mean().clamp(0.0, 1.0)
        return {"consensus_risk": consensus_risk}

    def _label_quality(self, probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = probabilities.device
        valid = [
            (idx, int(label))
            for idx, label in enumerate(self.probe_labels)
            if label is not None and 0 <= int(label) < self.n_classes
        ]
        if not valid:
            zero = torch.tensor(0.0, device=device)
            return {
                "label_prob": zero,
                "label_error": zero,
                "has_valid_labels": False,
            }
        row_indices = torch.tensor([idx for idx, _ in valid], device=device, dtype=torch.long)
        labels = torch.tensor([label for _, label in valid], device=device, dtype=torch.long)
        true_probs = probabilities[row_indices, labels]
        label_prob = true_probs.mean()
        label_error = (1.0 - label_prob).clamp(0.0, 1.0)
        return {
            "label_prob": label_prob,
            "label_error": label_error,
            "has_valid_labels": True,
        }

    def _spc(self, predictions: torch.Tensor) -> torch.Tensor:
        if not self.probe_groups:
            return torch.tensor(0.0, device=predictions.device)
        group_scores = []
        for group in self.probe_groups:
            group_pred = predictions[group].argmax(dim=1)
            base_pred = group_pred[0]
            group_scores.append((group_pred == base_pred).float().mean())
        return torch.stack(group_scores).mean()

    def _soft_spc(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Measure pairwise multi-scale probability divergence within each probe group.

        TV and JS retain confidence changes that hard SPC discards.  Pairwise JS is
        normalized by log(2), its maximum for two probability distributions, so
        both returned values are risks in [0, 1].
        """
        zero = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        if not self.probe_groups:
            return {"tv_risk": zero, "js_risk": zero}

        tv_scores = []
        js_scores = []
        eps = torch.finfo(predictions.dtype).eps
        for group in self.probe_groups:
            if len(group) < 2:
                continue
            group_probabilities = predictions[group].clamp_min(eps)
            left, right = torch.triu_indices(
                len(group), len(group), offset=1, device=predictions.device
            )
            first = group_probabilities[left]
            second = group_probabilities[right]

            total_variation = 0.5 * torch.abs(first - second).sum(dim=1)
            midpoint = 0.5 * (first + second)
            js_divergence = 0.5 * (
                (first * (first.log() - midpoint.log())).sum(dim=1)
                + (second * (second.log() - midpoint.log())).sum(dim=1)
            )
            tv_scores.append(total_variation.mean())
            js_scores.append((js_divergence / math.log(2.0)).mean())

        if not tv_scores:
            return {"tv_risk": zero, "js_risk": zero}
        return {
            "tv_risk": torch.stack(tv_scores).mean().clamp(0.0, 1.0),
            "js_risk": torch.stack(js_scores).mean().clamp(0.0, 1.0),
        }

    def print_rows(self, rows: List[Dict], current_round: Optional[int]):
        sys.stdout.write(f"\n[Probe Audit] Round {current_round}\n")
        sys.stdout.write(
            "client_id | role      | entropy | confidence | spc    | soft_tv | soft_js | "
            "consensus | label_p | target_p | target_bias | risk\n"
        )
        for row in sorted(rows, key=lambda x: x["client_id"]):
            sys.stdout.write(
                f"{row['client_id']:<9} | {row['role']:<9} | "
                f"{row['entropy']:.4f}  | {row['confidence']:.4f}     | "
                f"{row['spc']:.4f} | {row['soft_spc_tv']:.4f}  | "
                f"{row['soft_spc_js']:.4f}  | {row['consensus_risk']:.4f}    | "
                f"{row['label_prob']:.4f}  | {row['target_prob']:.4f}   | "
                f"{row['target_bias']:.4f}      | {row['risk']:.4f}\n"
            )

    def log_to_wandb(self, rows: List[Dict], current_round: Optional[int]):
        if getattr(wandb, "run", None) is None:
            return
        log_dict = {}
        for row in rows:
            prefix = f"probe/client{row['client_id']}"
            log_dict[f"{prefix}/entropy"] = row["entropy"]
            log_dict[f"{prefix}/confidence"] = row["confidence"]
            log_dict[f"{prefix}/spc"] = row["spc"]
            log_dict[f"{prefix}/soft_spc_tv"] = row["soft_spc_tv"]
            log_dict[f"{prefix}/soft_spc_js"] = row["soft_spc_js"]
            log_dict[f"{prefix}/target_prob"] = row["target_prob"]
            log_dict[f"{prefix}/source_prob"] = row["source_prob"]
            log_dict[f"{prefix}/target_bias"] = row["target_bias"]
            log_dict[f"{prefix}/consensus_risk"] = row["consensus_risk"]
            log_dict[f"{prefix}/label_prob"] = row["label_prob"]
            log_dict[f"{prefix}/label_error"] = row["label_error"]
            log_dict[f"{prefix}/risk"] = row["risk"]
            log_dict[f"{prefix}/is_malicious"] = 1 if row["role"] == "malicious" else 0
        if current_round is not None:
            log_dict["probe/round"] = int(current_round)
        wandb.log(log_dict, commit=False)
