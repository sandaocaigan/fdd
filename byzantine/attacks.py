# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         byzantine/attacks.py
# Description:  Byzantine Attack classes
# ===========================================================================
from collections import OrderedDict
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from public_config import public_datasetAssignmentDict
from utilities import Utilities as Utils
import os

#### Attack Base Class
class NoAttack:
    """Attack base class"""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.clients = kwargs['clients']
        self.runner = kwargs['runner_instance']
        self._attack_generators = {}
        self._attack_seed = (int(self.runner.seed) + 104729) % (2 ** 63 - 1)

    def _attack_generator(self, device):
        """Return a device-local RNG that does not advance the training RNG."""
        device = torch.device(device)
        key = str(device)
        if key not in self._attack_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed((self._attack_seed + len(self._attack_generators)) % (2 ** 63 - 1))
            self._attack_generators[key] = generator
        return self._attack_generators[key]

    def _attack_randn_like(self, reference):
        return torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=self._attack_generator(reference.device),
        )

    def _attack_randperm(self, n, device=None):
        device = torch.device("cpu") if device is None else torch.device(device)
        return torch.randperm(
            n,
            device=device,
            generator=self._attack_generator(device),
        )

    def is_attack_active(self):
        """Return True after attack_start_round; used for delayed attacks."""
        start_round = getattr(self.config, 'attack_start_round', 1)
        if start_round in [None, 'None', 'none']:
            return True
        current_round = getattr(self.runner, 'current_round', None)
        if current_round is None:
            return True
        return int(current_round) >= int(start_round)

    def trains_byzantine_locally(self):
        """Upload-only attacks retain ordinary local training for Byzantine clients."""
        return True

    def train_byzantine_epoch(self, client, epoch=None, current_round=None):
        self.runner.train_epoch(actor=client, data="train", epoch=epoch)


    def get_perturbed_client_models(self, **kwargs):
        """Called before the model is communicated. Defaults to returning the individual client models (unchanged)."""
        return Utils.get_client_models(self.clients)

    @staticmethod
    def probabilities_to_logits(probabilities: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Convert probability vectors to equivalent logits for the logit-upload path."""
        return probabilities.clamp_min(eps).log()

    def get_perturbed_client_predictions(self, **kwargs):
        """Legacy probability-upload path."""
        return self.runner.get_client_predictions(mode='train')

    def get_perturbed_client_logits(self, **kwargs):
        """Default logit-upload path.

        Attacks that only implement the legacy probability interface are mapped to
        equivalent logits via log(probability), so existing attacks keep working
        while FedDistill communicates raw logits internally.
        """
        if self.__class__.get_perturbed_client_predictions is NoAttack.get_perturbed_client_predictions:
            client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode='train')
            return client_logits_list
        client_prediction_list = self.get_perturbed_client_predictions(**kwargs)
        return [self.probabilities_to_logits(predictions) for predictions in client_prediction_list]


class ParameterRandomVector(NoAttack):
    """Random vector attack"""

    @torch.no_grad()
    def perturb_byzantine_model(self, client_state_dict: OrderedDict) -> OrderedDict:
        """Takes the state_dict of a client and perturbs it."""
        client_state_dict = client_state_dict.copy()
        for key in client_state_dict:
            if not torch.is_floating_point(client_state_dict[key]):
                continue
            client_state_dict[key] = self._attack_randn_like(client_state_dict[key].float())
        return client_state_dict

    def get_perturbed_client_models(self, **kwargs):
        if not self.is_attack_active():
            return Utils.get_client_models(self.clients)
        client_model_list = []
        for client in self.clients:
            client_state_dict = client.model.state_dict()
            if client.is_byzantine:
                client_state_dict = self.perturb_byzantine_model(client_state_dict)
            client_model_list.append(client_state_dict)

        return client_model_list

class ParameterRandomVectorScaled(ParameterRandomVector):
    """Random vector attack but scale to have the same L2 norm as the original model."""

    @torch.no_grad()
    def perturb_byzantine_model(self, client_state_dict: OrderedDict) -> OrderedDict:
        """Takes the state_dict of a client and perturbs it."""
        client_state_dict = client_state_dict.copy()
        for key in client_state_dict:
            if not torch.is_floating_point(client_state_dict[key]):
                continue
            p_old = client_state_dict[key].float()
            old_norm = torch.norm(p_old)
            random_vector = self._attack_randn_like(p_old)
            new_norm = torch.norm(random_vector)
            assert new_norm > 0
            client_state_dict[key] = random_vector * (old_norm / new_norm)

        return client_state_dict


class PredictionNaiveSignFlip(NoAttack):
    """Naive prediction sign flip attack, just uses a random one-hot vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                random_logits = self._attack_randn_like(client_predictions)
                random_predictions = torch.argmax(random_logits, dim=1)
                client_prediction_list[client_idx] = torch.nn.functional.one_hot(random_predictions,
                                                                                 num_classes=client_predictions.shape[
                                                                                     1]).float()
        return client_prediction_list


class PredictionFixedSignFlip(NoAttack):
    """Fixed prediction sign flip attack, just uses a fixed one-hot vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        fixed_prediction = torch.zeros_like(client_prediction_list[0])
        byz_prediction  = fixed_prediction
        byz_prediction[:,0] = 1.
        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            byz_idx = self._attack_randperm(fixed_prediction.shape[0])[:int(fixed_prediction.shape[0] * self.config['sample_attack_frac'])]

        for client_idx in range(len(client_prediction_list)):
            if self.clients[client_idx].is_byzantine:
                if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
                    client_pred = client_prediction_list[client_idx]
                    client_pred[byz_idx,:] = byz_prediction[byz_idx,:]
                    client_prediction_list[client_idx] = client_pred
                else:
                    client_prediction_list[client_idx] = byz_prediction

        return client_prediction_list


class PredictionAdversarialSignFlip(PredictionNaiveSignFlip):
    """Byzantine clients put full emphasis (one hot) on second most likely class of benign clients."""

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        # Get the list of predictions, but only the benign ones
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        avg_honest_client_predictions = torch.mean(torch.stack(honest_client_predictions, dim=0), dim=0)
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                # Get the second most likely class and set full probability to that class
                # Index 1 corresponds to the second most likely class
                second_most_likely_class = torch.topk(avg_honest_client_predictions, k=2, dim=1).indices[:, 1]

                client_prediction_list[client_idx] = torch.nn.functional.one_hot(second_most_likely_class,
                                                                                 num_classes=client_predictions.shape[
                                                                                     1]).float()
        return client_prediction_list


class PredictionRandomLabel(PredictionNaiveSignFlip):
    """Accuracy attack: Byzantine clients upload random one-hot labels."""


class PredictionUniform(NoAttack):
    """Accuracy attack: Byzantine clients upload uninformative uniform probabilities."""

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                client_prediction_list[client_idx] = torch.full_like(
                    client_predictions,
                    fill_value=1.0 / float(client_predictions.shape[1]),
                )
        return client_prediction_list


class PredictionLeastLikely(PredictionAdversarialSignFlip):
    """Accuracy attack: Byzantine clients upload the least likely class under benign consensus."""

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        avg_honest_client_predictions = torch.mean(torch.stack(honest_client_predictions, dim=0), dim=0)
        least_likely_class = torch.argmin(avg_honest_client_predictions, dim=1)
        byz_predictions = torch.nn.functional.one_hot(
            least_likely_class,
            num_classes=client_prediction_list[0].shape[1],
        ).float()
        for client_idx in range(len(client_prediction_list)):
            if self.clients[client_idx].is_byzantine:
                client_prediction_list[client_idx] = byz_predictions
        return client_prediction_list
class Gaussian(NoAttack):
    """Paper baseline: add Gaussian logit noise fitted from honest uploads."""

    @torch.no_grad()
    def get_perturbed_client_logits(self, **kwargs):
        client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode="train")
        if not self.is_attack_active():
            return client_logits_list

        honest_logits = [
            logits for idx, logits in enumerate(client_logits_list)
            if not self.clients[idx].is_byzantine
        ]
        if not honest_logits:
            return client_logits_list

        honest_stack = torch.stack(honest_logits, dim=0)
        noise_mean = honest_stack.mean(dim=0)
        noise_std = honest_stack.std(dim=0, unbiased=False).clamp_min(1e-6)
        scale = float(getattr(self.config, "logit_gaussian_scale", 1.0) or 1.0)
        perturbed_logits = list(client_logits_list)
        for idx, client in enumerate(self.clients):
            if client.is_byzantine:
                noise = noise_mean + noise_std * self._attack_randn_like(noise_std)
                perturbed_logits[idx] = client_logits_list[idx] + scale * noise
        print(f"[Gaussian] Added honest-fitted Gaussian logit noise (scale={scale}).")
        return perturbed_logits


class Disruption(NoAttack):
    """Paper baseline: shuffle each class-logit dimension over public samples."""

    @torch.no_grad()
    def get_perturbed_client_logits(self, **kwargs):
        client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode="train")
        if not self.is_attack_active():
            return client_logits_list

        perturbed_logits = list(client_logits_list)
        for idx, client in enumerate(self.clients):
            if not client.is_byzantine:
                continue
            logits = client_logits_list[idx]
            disrupted = torch.empty_like(logits)
            for class_idx in range(logits.shape[1]):
                order = self._attack_randperm(logits.shape[0], device=logits.device)
                disrupted[:, class_idx] = logits[order, class_idx]
            perturbed_logits[idx] = disrupted
        print("[Disruption] Shuffled every class-logit dimension across public samples.")
        return perturbed_logits


class Shuffling(NoAttack):
    """Paper baseline: promote the runner-up class and suppress top-1 logits.

    The configured public dataset is unlabeled, so the client's current top-1
    class is used as a pseudo label and the runner-up as the plausible wrong
    class. This does not use inaccessible ground-truth public labels.
    """

    @torch.no_grad()
    def get_perturbed_client_logits(self, **kwargs):
        client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode="train")
        if not self.is_attack_active():
            return client_logits_list

        perturbed_logits = list(client_logits_list)
        for idx, client in enumerate(self.clients):
            if not client.is_byzantine:
                continue
            logits = client_logits_list[idx]
            ranking = logits.argsort(dim=1, descending=True)
            pseudo_label = ranking[:, 0:1]
            plausible_wrong = ranking[:, 1:2]
            lowest = logits.min(dim=1, keepdim=True).values
            highest = logits.max(dim=1, keepdim=True).values
            shuffled = logits.clone()
            shuffled.scatter_(1, pseudo_label, lowest)
            shuffled.scatter_(1, plausible_wrong, highest)
            perturbed_logits[idx] = shuffled
        print("[Shuffling] Used per-client top-1 pseudo labels on the unlabeled public dataset.")
        return perturbed_logits


class LabelFlip(NoAttack):
    """Paper baseline: train active Byzantine clients with l -> L - l - 1."""

    def trains_byzantine_locally(self):
        return True

    def train_byzantine_epoch(self, client, epoch=None, current_round=None):
        epoch_str = f"\nEpoch {epoch} - " if epoch is not None else ""
        sys.stdout.write(f"{epoch_str}Label-flip training {client.actor_name}.\n")
        with torch.set_grad_enabled(True):
            with tqdm(client.dataloader, leave=True) as pbar:
                for x_input, y_target, _ in pbar:
                    x_input = x_input.to(self.runner.device, non_blocking=True)
                    y_target = y_target.to(self.runner.device, non_blocking=True)
                    flipped_target = self.runner.n_classes - y_target - 1
                    client.optimizer.zero_grad()
                    with autocast(enabled=(self.runner.use_amp is True)):
                        output = client.model.train()(x_input)
                        loss = client.loss_criterion(output, flipped_target)
                    client.gradScaler.scale(loss).backward()
                    client.gradScaler.step(client.optimizer)
                    client.gradScaler.update()
                    client.scheduler.step()
                    client.update_batch_metrics(
                        mode="train", loss=loss, output=output, y_target=flipped_target
                    )



class ManipulatingKD(NoAttack):
    """ManipulatingKD-style optimized raw-logit poisoning with stealth constraints.

    This implementation directly optimizes the logits uploaded by Byzantine
    clients. It keeps the core attack independent from the server training loop:
    the real server still consumes the malicious logits and distills normally.
    Aggregation-specific surrogate hooks are exposed so stronger Median,
    TrimmedMean, Krum, or GeoMedian constraints can be added later without
    changing the FedDistill communication path.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paper_history = []

    def _cfg(self, name, default=None):
        try:
            value = getattr(self.config, name)
        except Exception:
            try:
                value = self.config[name]
            except Exception:
                value = default
        return default if value in [None, "None", "none"] else value

    def _constraint_enabled(self, name):
        constraints = str(self._cfg("manipkd_constraints", "l2,range,moments,entropy,outlier")).lower()
        if constraints in ["all", "*"]:
            return True
        enabled = [x.strip() for x in constraints.split(",") if x.strip()]
        return name in enabled

    def _surrogate_aggregate(self, honest_stack, malicious_logits):
        mode = str(self._cfg("manipkd_surrogate", "mean")).lower()
        stack = torch.cat([honest_stack, malicious_logits], dim=0)

        if mode in ["mean", "avg", "average", "nodefence", "none"]:
            return stack.mean(dim=0)

        if mode in ["median", "predictionmedian"]:
            return torch.quantile(stack, 0.5, dim=0, interpolation="linear")

        if mode in ["trimmed_mean", "trimmedmean", "filter", "predictionfilter"]:
            trim_count = self._cfg("manipkd_trim_count", None)
            if trim_count in [None, "None", "none"]:
                trim_count = int(self._cfg("n_byzantine_clients", 0) or 0)
            trim_count = int(trim_count)
            sorted_stack, _ = torch.sort(stack, dim=0)
            if trim_count > 0 and 2 * trim_count < sorted_stack.shape[0]:
                sorted_stack = sorted_stack[trim_count:-trim_count]
            return sorted_stack.mean(dim=0)

        if mode in ["geomedian", "geom_median", "predictiongeomedian"]:
            gm = stack.mean(dim=0)
            n_iters = int(self._cfg("manipkd_geomedian_iters", 5) or 5)
            for _ in range(max(n_iters, 1)):
                distances = torch.linalg.norm(stack - gm.unsqueeze(0), dim=2).clamp_min(1e-6)
                weights = 1.0 / distances
                gm = (weights.unsqueeze(2) * stack).sum(dim=0) / weights.sum(dim=0).unsqueeze(1).clamp_min(1e-6)
            return gm

        if mode in ["krum", "soft_krum", "softkrum"]:
            flat = stack.reshape(stack.shape[0], -1)
            pairwise = torch.cdist(flat, flat, p=2).pow(2)
            n_clients = stack.shape[0]
            n_byz = int(self._cfg("n_byzantine_clients", 0) or 0)
            k = max(n_clients - n_byz - 2, 1)
            values = torch.topk(pairwise, k=min(k + 1, n_clients), dim=1, largest=False).values
            scores = values[:, 1:].sum(dim=1) if values.shape[1] > 1 else values[:, 0]
            temperature = float(self._cfg("manipkd_krum_temperature", 1.0) or 1.0)
            weights = torch.softmax(-scores / max(temperature, 1e-6), dim=0)
            return (weights.view(-1, 1, 1) * stack).sum(dim=0)

        raise NotImplementedError(f"ManipulatingKD surrogate '{mode}' is not implemented.")

    def _target_loss(self, aggregate_logits, honest_mean_logits):
        target = str(self._cfg("manipkd_target", "least_likely")).lower()
        honest_probs = torch.softmax(honest_mean_logits, dim=1)

        if target in ["least_likely", "leastlikely", "argmin"]:
            target_labels = torch.argmin(honest_probs, dim=1)
            return F.cross_entropy(aggregate_logits, target_labels)

        if target in ["second_likely", "secondlikely", "second"]:
            target_labels = torch.topk(honest_probs, k=2, dim=1).indices[:, 1]
            return F.cross_entropy(aggregate_logits, target_labels)

        if target in ["maximize_kl", "max_kl", "kl"]:
            aggregate_log_probs = torch.log_softmax(aggregate_logits, dim=1)
            honest_log_probs = torch.log(honest_probs.clamp_min(1e-12))
            kl = torch.sum(honest_probs * (honest_log_probs - aggregate_log_probs), dim=1).mean()
            return -kl

        if target in ["uniform", "entropy"]:
            uniform = torch.full_like(honest_probs, 1.0 / float(honest_probs.shape[1]))
            aggregate_log_probs = torch.log_softmax(aggregate_logits, dim=1)
            return F.kl_div(aggregate_log_probs, uniform, reduction="batchmean")

        raise NotImplementedError(f"ManipulatingKD target '{target}' is not implemented.")

    def _stealth_penalty(self, malicious_logits, stats):
        penalty = malicious_logits.new_tensor(0.0)

        if self._constraint_enabled("l2"):
            distances = torch.linalg.norm(malicious_logits - stats["center"].unsqueeze(0), dim=2)
            l2_excess = torch.relu(distances - stats["sample_radius"].unsqueeze(0))
            penalty = penalty + float(self._cfg("manipkd_lambda_l2", 1.0)) * l2_excess.pow(2).mean()

        if self._constraint_enabled("range"):
            high_excess = torch.relu(malicious_logits - stats["high"].unsqueeze(0))
            low_excess = torch.relu(stats["low"].unsqueeze(0) - malicious_logits)
            penalty = penalty + float(self._cfg("manipkd_lambda_range", 1.0)) * (
                high_excess.pow(2).mean() + low_excess.pow(2).mean()
            )

        if self._constraint_enabled("moments"):
            mal_mean = malicious_logits.mean(dim=0)
            mean_penalty = F.mse_loss(mal_mean, stats["center"])
            if malicious_logits.shape[0] > 1:
                mal_std = malicious_logits.std(dim=0, unbiased=False)
                std_penalty = F.mse_loss(mal_std, stats["std"])
            else:
                std_penalty = malicious_logits.new_tensor(0.0)
            penalty = penalty + float(self._cfg("manipkd_lambda_moments", 0.25)) * (mean_penalty + std_penalty)

        if self._constraint_enabled("entropy"):
            mal_probs = torch.softmax(malicious_logits, dim=2).clamp_min(1e-12)
            mal_entropy = -(mal_probs * mal_probs.log()).sum(dim=2)
            mal_confidence = mal_probs.max(dim=2).values
            entropy_penalty = F.mse_loss(mal_entropy.mean(dim=0), stats["entropy"])
            confidence_penalty = F.mse_loss(mal_confidence.mean(dim=0), stats["confidence"])
            penalty = penalty + float(self._cfg("manipkd_lambda_entropy", 0.25)) * (
                entropy_penalty + confidence_penalty
            )

        if self._constraint_enabled("outlier"):
            mal_client_scores = torch.linalg.norm(
                malicious_logits - stats["center"].unsqueeze(0),
                dim=2,
            ).mean(dim=1)
            outlier_excess = torch.relu(mal_client_scores - stats["client_outlier_threshold"])
            penalty = penalty + float(self._cfg("manipkd_lambda_outlier", 1.0)) * outlier_excess.pow(2).mean()

        return penalty

    @torch.no_grad()
    def _project_to_stealth_set(self, malicious_logits, stats):
        if not bool(self._cfg("manipkd_project", True)):
            return
        if self._constraint_enabled("range"):
            malicious_logits.copy_(
                torch.max(
                    torch.min(malicious_logits, stats["high"].unsqueeze(0)),
                    stats["low"].unsqueeze(0),
                )
            )
        if self._constraint_enabled("l2"):
            delta = malicious_logits - stats["center"].unsqueeze(0)
            distances = torch.linalg.norm(delta, dim=2, keepdim=True).clamp_min(1e-6)
            radius = stats["sample_radius"].view(1, -1, 1)
            scale = torch.clamp(radius / distances, max=1.0)
            malicious_logits.copy_(stats["center"].unsqueeze(0) + delta * scale)

    def _repeat_stack_to_count(self, stack, count):
        if count <= 0:
            return stack[:0]
        if stack.shape[0] == count:
            return stack
        repeats = (count + stack.shape[0] - 1) // stack.shape[0]
        return stack.repeat((repeats, 1, 1))[:count]

    def _build_stats(self, source_stack):
        q_low = float(self._cfg("manipkd_q_low", 0.05))
        q_high = float(self._cfg("manipkd_q_high", 0.95))
        q_radius = float(self._cfg("manipkd_q_radius", 0.95))
        q_outlier = float(self._cfg("manipkd_q_outlier", 0.95))
        radius_multiplier = float(self._cfg("manipkd_radius_multiplier", 1.25))

        center = source_stack.mean(dim=0)

        if source_stack.shape[0] < 2:
            single_radius = float(self._cfg("manipkd_single_client_radius", 0.5) or 0.5)
            within_sample_std = center.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
            margin = single_radius * within_sample_std.expand_as(center)
            std = margin
            low = center - margin
            high = center + margin
            sample_radius = torch.linalg.norm(margin, dim=1).mul(radius_multiplier).clamp_min(1e-6)
            client_outlier_threshold = sample_radius.mean().clamp_min(1e-6)
        else:
            std = source_stack.std(dim=0, unbiased=False)
            low = torch.quantile(source_stack, q_low, dim=0, interpolation="linear")
            high = torch.quantile(source_stack, q_high, dim=0, interpolation="linear")

            sample_distances = torch.linalg.norm(source_stack - center.unsqueeze(0), dim=2)
            sample_radius = torch.quantile(sample_distances, q_radius, dim=0, interpolation="linear")
            sample_radius = (sample_radius * radius_multiplier).clamp_min(1e-6)

            source_client_scores = sample_distances.mean(dim=1)
            client_outlier_threshold = torch.quantile(source_client_scores, q_outlier, interpolation="linear")
            client_outlier_threshold = (client_outlier_threshold * radius_multiplier).clamp_min(1e-6)

        source_probs = torch.softmax(source_stack, dim=2).clamp_min(1e-12)
        entropy = -(source_probs * source_probs.log()).sum(dim=2).mean(dim=0)
        confidence = source_probs.max(dim=2).values.mean(dim=0)

        return {
            "center": center,
            "std": std,
            "low": low,
            "high": high,
            "sample_radius": sample_radius,
            "client_outlier_threshold": client_outlier_threshold,
            "entropy": entropy,
            "confidence": confidence,
        }

    def _get_legacy_perturbed_client_logits(self, **kwargs):
        client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode="train")
        if not self.is_attack_active():
            return client_logits_list

        honest_indices = [idx for idx, client in enumerate(self.clients) if not client.is_byzantine]
        byzantine_indices = [idx for idx, client in enumerate(self.clients) if client.is_byzantine]
        if not byzantine_indices or not honest_indices:
            return client_logits_list

        honest_stack = torch.stack([client_logits_list[idx].detach() for idx in honest_indices], dim=0)
        true_byzantine_stack = torch.stack([client_logits_list[idx].detach() for idx in byzantine_indices], dim=0)

        knowledge = str(self._cfg("manipkd_knowledge", "full_local")).lower().replace("-", "_")
        if knowledge in ["full", "full_local", "fl", "oracle"]:
            stats_source = honest_stack
            surrogate_honest_stack = honest_stack
            stats_source_name = "honest_clients"
        elif knowledge in ["partial", "partial_local", "pl", "colluding_only"]:
            stats_source = true_byzantine_stack
            surrogate_honest_stack = self._repeat_stack_to_count(true_byzantine_stack, len(honest_indices))
            stats_source_name = "byzantine_clean_logits"
        else:
            raise NotImplementedError(f"ManipulatingKD knowledge mode '{knowledge}' is not implemented.")

        stats = self._build_stats(stats_source)

        init_mode = str(self._cfg("manipkd_init", "honest_mean")).lower()
        if init_mode in ["client", "true_client", "local"]:
            malicious_logits = true_byzantine_stack.clone()
        else:
            malicious_logits = stats["center"].unsqueeze(0).repeat(len(byzantine_indices), 1, 1).clone()

        noise_std = float(self._cfg("manipkd_noise_std", 0.01) or 0.0)
        if noise_std > 0:
            malicious_logits = malicious_logits + noise_std * stats["std"].clamp_min(1e-3).unsqueeze(0) * self._attack_randn_like(malicious_logits)
        self._project_to_stealth_set(malicious_logits, stats)

        malicious_logits = malicious_logits.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([malicious_logits], lr=float(self._cfg("manipkd_lr", 0.05)))
        steps = int(self._cfg("manipkd_steps", 20) or 20)
        log_every = int(self._cfg("manipkd_log_every", 0) or 0)
        honest_mean_logits = stats["center"]

        final_attack_loss = 0.0
        final_stealth_penalty = 0.0
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            aggregate_logits = self._surrogate_aggregate(surrogate_honest_stack, malicious_logits)
            attack_loss = self._target_loss(aggregate_logits, honest_mean_logits)
            stealth_penalty = self._stealth_penalty(malicious_logits, stats)
            loss = attack_loss + stealth_penalty
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self._project_to_stealth_set(malicious_logits, stats)
            final_attack_loss = float(attack_loss.detach().cpu())
            final_stealth_penalty = float(stealth_penalty.detach().cpu())
            if log_every > 0 and (step % log_every == 0 or step == steps):
                print(
                    f"[ManipulatingKD] step={step}/{steps} "
                    f"attack_loss={final_attack_loss:.6f} stealth_penalty={final_stealth_penalty:.6f}"
                )

        perturbed_logits = list(client_logits_list)
        optimized_logits = malicious_logits.detach()
        for local_idx, client_idx in enumerate(byzantine_indices):
            perturbed_logits[client_idx] = optimized_logits[local_idx]

        print(
            f"[ManipulatingKD] optimized raw logits for {len(byzantine_indices)} Byzantine clients "
            f"with knowledge={knowledge}, stats_source={stats_source_name}, "
            f"surrogate={self._cfg('manipkd_surrogate', 'mean')}, "
            f"target={self._cfg('manipkd_target', 'least_likely')}, "
            f"attack_loss={final_attack_loss:.6f}, stealth_penalty={final_stealth_penalty:.6f}."
        )
        return perturbed_logits

    def trains_byzantine_locally(self):
        """Paper ManipulatingKD changes uploads, not private local training."""
        return True

    def train_byzantine_epoch(self, client, epoch, current_round):
        self.runner.train_epoch(actor=client, data="train", epoch=epoch)

    @staticmethod
    def _paper_mse_per_sample(left, right):
        return (left - right).pow(2).mean(dim=-1)

    def _paper_kl_per_sample(self, reference_logits, poisoned_logits):
        temperature = float(self._cfg("distill_temperature", 1.0) or 1.0)
        if temperature <= 0:
            raise ValueError("distill_temperature must be positive for ManipulatingKD.")
        reference_probs = torch.softmax(reference_logits / temperature, dim=1)
        poisoned_log_probs = torch.log_softmax(poisoned_logits / temperature, dim=1)
        return (
            reference_probs
            * (reference_probs.clamp_min(1e-12).log() - poisoned_log_probs)
        ).sum(dim=1)

    @torch.no_grad()
    def record_aggregated_logits(self, aggregated_logits, client_logits_list):
        """Store public aggregate broadcasts and malicious uploads for P-L."""
        byzantine_indices = [
            idx for idx, client in enumerate(self.clients) if client.is_byzantine
        ]
        if not byzantine_indices:
            return
        self._paper_history.append({
            "aggregate": aggregated_logits.detach().clone(),
            "malicious_uploads": torch.stack(
                [client_logits_list[idx].detach() for idx in byzantine_indices], dim=0
            ).clone(),
        })
        keep = max(int(self._cfg("manipkd_history_rounds", 5) or 5), 1)
        self._paper_history = self._paper_history[-keep:]

    def _get_paper_perturbed_client_logits(self):
        """Implement the paper's F-L/P-L ManipulatingKD objectives (Eq. 16-23)."""
        client_logits_list, _ = self.runner.get_client_logits_and_predictions(mode="train")
        if not self.is_attack_active():
            return client_logits_list

        byzantine_indices = [
            idx for idx, client in enumerate(self.clients) if client.is_byzantine
        ]
        honest_indices = [
            idx for idx, client in enumerate(self.clients) if not client.is_byzantine
        ]
        if not byzantine_indices or not honest_indices:
            return client_logits_list

        clean_stack = torch.stack([logits.detach() for logits in client_logits_list], dim=0)
        clean_malicious = clean_stack[byzantine_indices]
        n_clients = clean_stack.shape[0]
        n_malicious = len(byzantine_indices)
        knowledge = str(self._cfg("manipkd_knowledge", "full_local")).lower().replace("-", "_")

        if knowledge in ["full", "full_local", "fl", "oracle"]:
            model_name = "F-L"
            honest_sum = clean_stack[honest_indices].sum(dim=0)
            reference_aggregate = clean_stack.mean(dim=0)
            reference_center = clean_stack.mean(dim=0)
            max_reference_distance = self._paper_mse_per_sample(
                clean_stack, reference_center.unsqueeze(0)
            ).max(dim=0).values
        elif knowledge in ["partial", "partial_local", "pl", "colluding_only"]:
            model_name = "P-L"
            required_history = max(int(self._cfg("manipkd_history_rounds", 5) or 5), 1)
            if len(self._paper_history) < required_history:
                print(
                    f"[ManipulatingKD paper P-L] warmup: {len(self._paper_history)}/{required_history} "
                    "aggregate broadcasts available; uploading clean logits."
                )
                return client_logits_list
            history = self._paper_history[-required_history:]
            estimated_benign_sum = torch.stack([
                item["aggregate"] * n_clients - item["malicious_uploads"].sum(dim=0)
                for item in history
            ], dim=0).mean(dim=0)
            reference_aggregate = (
                estimated_benign_sum + clean_malicious.sum(dim=0)
            ) / float(n_clients)
            reference_center = clean_malicious.mean(dim=0)
            max_reference_distance = self._paper_mse_per_sample(
                clean_malicious, reference_center.unsqueeze(0)
            ).max(dim=0).values
            honest_sum = estimated_benign_sum
        elif knowledge in ["full_global", "fg", "partial_global", "pg"]:
            raise NotImplementedError(
                "Paper F-G/P-G require a shadow global model and are not implemented. "
                "Use manipkd_knowledge=full_local (F-L) or partial_local (P-L)."
            )
        else:
            raise NotImplementedError(f"Unknown paper ManipulatingKD knowledge mode '{knowledge}'.")

        candidate = clean_malicious.mean(dim=0).detach().clone().requires_grad_(True)
        # KL is stationary when the candidate exactly equals the clean aggregate.
        # Start inside Eq. (5)'s feasible ball so gradient ascent can leave that point.
        noise_scale = float(self._cfg("manipkd_noise_std", 0.01) or 0.0)
        if noise_scale > 0:
            with torch.no_grad():
                direction = self._attack_randn_like(candidate)
                direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
                feasible_radius = (n_malicious * max_reference_distance).clamp_min(0.0).sqrt()
                candidate.add_(noise_scale * feasible_radius.unsqueeze(1) * direction)
        dual = torch.zeros_like(max_reference_distance)
        steps = max(int(self._cfg("manipkd_steps", 30) or 30), 1)
        primal_lr = float(self._cfg("manipkd_lr", 0.05) or 0.05)
        dual_lr = float(self._cfg("manipkd_lagrange_lr", 1.0) or 1.0)

        for _ in range(steps):
            poisoned_aggregate = (honest_sum + n_malicious * candidate) / float(n_clients)
            divergence = self._paper_kl_per_sample(reference_aggregate, poisoned_aggregate)
            constraint = (
                self._paper_mse_per_sample(candidate, reference_center)
                - n_malicious * max_reference_distance
            )
            objective = divergence.mean() - (dual * constraint).mean()
            gradient, = torch.autograd.grad(objective, candidate)
            with torch.no_grad():
                candidate.add_(primal_lr * gradient)
                dual.copy_(torch.clamp(dual + dual_lr * constraint, min=0.0))

        with torch.no_grad():
            poisoned_aggregate = (honest_sum + n_malicious * candidate) / float(n_clients)
            divergence = self._paper_kl_per_sample(reference_aggregate, poisoned_aggregate)
            constraint = (
                self._paper_mse_per_sample(candidate, reference_center)
                - n_malicious * max_reference_distance
            )
            violation_rate = (constraint > 1e-6).float().mean()

        perturbed_logits = list(client_logits_list)
        optimized_upload = candidate.detach()
        for client_idx in byzantine_indices:
            perturbed_logits[client_idx] = optimized_upload.clone()
        print(
            f"[ManipulatingKD paper {model_name}] shared malicious upload optimized: "
            f"KL={float(divergence.mean()):.6f}, constraint_mean={float(constraint.mean()):.6f}, "
            f"violation_rate={float(violation_rate):.4f}, steps={steps}."
        )
        return perturbed_logits

    def get_perturbed_client_logits(self, **kwargs):
        formulation = str(self._cfg("manipkd_formulation", "paper")).lower()
        if formulation in ["paper", "original"]:
            return self._get_paper_perturbed_client_logits()
        if formulation in ["legacy", "style"]:
            return self._get_legacy_perturbed_client_logits(**kwargs)
        raise NotImplementedError(
            f"Unknown ManipulatingKD formulation '{formulation}'. Use paper or legacy."
        )
    """Byzantine clients put full emphasis (one hot) on least corrrelated class."""
class CPA(PredictionNaiveSignFlip):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.config['public_ds'] in [None, 'none', 'None']:
            public_ds = public_datasetAssignmentDict[self.config['dataset']]
        else:
            public_ds = self.config['dataset']
        cpa_tensor_path = os.path.join('byzantine', 'cpa_info', f"{public_ds}-cov.pt")
        self.cpa_tensor = torch.load(cpa_tensor_path, map_location=self.config.device)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        honest_client_predictions = torch.stack(honest_client_predictions, dim=0)
        mean_honest_predictions = torch.mean(honest_client_predictions, dim=0)
        honest_max_pred = torch.argmax(mean_honest_predictions, dim=1)

        if self.config['hips'] == True:
            # Get the covariance vector corresponding to the honest_max_pred
            cpa_tensor = self.cpa_tensor[honest_max_pred, :]

            selected_vertices = torch.mul(cpa_tensor, honest_client_predictions).sum(dim=2).argmin(dim=0)
            byz_predictions = honest_client_predictions[selected_vertices, torch.arange(selected_vertices.size()[0]), :]
        else:
            cpa_tensor = self.cpa_tensor.argmin(dim=0)
            byz_label = cpa_tensor[honest_max_pred]
            byz_predictions = torch.nn.functional.one_hot(byz_label, num_classes=client_prediction_list[0].shape[1]).float()

        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            assert not self.config['hips'], "Cannot sample attack fraction when hips is True."
            p_honest = 1. - self.config['sample_attack_frac']
            honest_idx = self._attack_randperm(byz_label.shape[0])[:int(byz_label.shape[0] * p_honest)]
            byz_predictions[honest_idx,:] = mean_honest_predictions[honest_idx,:]

        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                # Get the least likely class and set full probability to that class
                client_prediction_list[client_idx] = byz_predictions
        return client_prediction_list


class CELMAX(PredictionNaiveSignFlip):
    """Byzantine clients put full emphasis (one hot) on the class that is least likely when averaging all honest clients."""

    @torch.no_grad()
    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        if not self.is_attack_active():
            return client_prediction_list
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        honest_client_predictions = torch.stack(honest_client_predictions, dim=0)
        mean_honest_predictions = torch.mean(honest_client_predictions, dim=0)

        if self.config['hips'] == True:
            alpha = float(self.config['n_byzantine_clients']) / float(self.config['n_clients'])
            potential_predictions = alpha * mean_honest_predictions.unsqueeze(0) + (1. - alpha) * honest_client_predictions
            deviations = torch.sum(-1. * mean_honest_predictions.unsqueeze(0) * torch.log(potential_predictions),dim=2)
            argmax_deviations = torch.argmax(deviations, dim=0)
            byz_predictions = honest_client_predictions[argmax_deviations,torch.arange(argmax_deviations.size()[0]),:]
        else:
            del honest_client_predictions
            honest_client_least_likely_predictions = torch.argmin(mean_honest_predictions, dim=1)
            byz_predictions = torch.nn.functional.one_hot(honest_client_least_likely_predictions,
                                                                                num_classes=
                                                                                client_prediction_list[0].shape[
                                                                                    1]).float()
        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            p_honest = 1. - self.config['sample_attack_frac']
            honest_idx = self._attack_randperm(byz_predictions.size()[0])[:int(byz_predictions.size()[0] * p_honest)]
            byz_predictions[honest_idx,:] = mean_honest_predictions[honest_idx,:]
        del mean_honest_predictions
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                client_prediction_list[client_idx] = byz_predictions
        return client_prediction_list


AccuracyRandom = PredictionRandomLabel
AccuracyUniform = PredictionUniform
AccuracyLeastLikely = PredictionLeastLikely
AccuracySecondLikely = PredictionAdversarialSignFlip

from byzantine.backdoor_attacks import BadNets, WaNet, LabelConsistent


