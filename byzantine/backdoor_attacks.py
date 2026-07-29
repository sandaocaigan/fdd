# ===========================================================================
# Backdoor attacks for prediction/logit-based federated distillation.
# These attacks train Byzantine clients locally with input-level triggers and
# let them upload predictions generated from triggered public inputs.
# ===========================================================================
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from config import means as private_means
from config import stds as private_stds
from public_config import means as public_means
from public_config import public_datasetAssignmentDict
from public_config import stds as public_stds
from utilities import Utilities as Utils


class BackdoorAttackBase:
    """Base class for local-training backdoor attacks in FedDistill."""

    attack_name = "BackdoorAttackBase"

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.clients = kwargs["clients"]
        self.runner = kwargs["runner_instance"]
        self.device = self.runner.device
        self.n_classes = self.runner.n_classes

        self.target_class = int(getattr(self.config, "backdoor_target_class", 0))
        source = getattr(self.config, "backdoor_source_class", None)
        self.source_class = None if source in [None, "None", "none"] else int(source)
        self.poison_rate = float(getattr(self.config, "backdoor_poison_rate", 0.25))
        self.upload_rate = float(getattr(self.config, "backdoor_upload_rate", 1.0))
        self.trigger_size = int(getattr(self.config, "backdoor_trigger_size", 4))
        self.trigger_value = float(getattr(self.config, "backdoor_trigger_value", 1.0))
        self.trigger_position = str(getattr(self.config, "backdoor_trigger_position", "bottom_right"))
        self.wanet_strength = float(getattr(self.config, "wanet_grid_strength", 0.15))

        self.private_dataset_name = self.config.dataset
        self.public_dataset_name = self._resolve_public_dataset_name()
        self.private_mean, self.private_std = self._make_norm_tensors(
            dataset_name=self.private_dataset_name,
            means_dict=private_means,
            stds_dict=private_stds,
        )
        self.public_mean, self.public_std = self._make_norm_tensors(
            dataset_name=self.public_dataset_name,
            means_dict=public_means,
            stds_dict=public_stds,
        )
        if self.public_mean is None or self.public_std is None:
            self.public_mean, self.public_std = self.private_mean, self.private_std

    def _is_none_like(self, value):
        return value in [None, "None", "none"]

    def _resolve_public_dataset_name(self):
        if not self._is_none_like(getattr(self.config, "public_ds_fraction", None)):
            return self.config.dataset
        public_ds = getattr(self.config, "public_ds", None)
        if not self._is_none_like(public_ds):
            return public_ds
        return public_datasetAssignmentDict.get(self.config.dataset, self.config.dataset)

    def _make_norm_tensors(self, dataset_name, means_dict, stds_dict):
        mean = means_dict.get(dataset_name, None)
        std = stds_dict.get(dataset_name, None)
        mean_tensor = None if mean is None else torch.tensor(mean, device=self.device).view(1, -1, 1, 1)
        std_tensor = None if std is None else torch.tensor(std, device=self.device).view(1, -1, 1, 1)
        return mean_tensor, std_tensor

    def _norm_stats_for_domain(self, domain):
        if str(domain).lower() == "public":
            return self.public_mean, self.public_std
        return self.private_mean, self.private_std

    def is_attack_active(self):
        start_round = getattr(self.config, "attack_start_round", 1)
        if start_round in [None, "None", "none"]:
            return True
        current_round = getattr(self.runner, "current_round", None)
        if current_round is None:
            return True
        return int(current_round) >= int(start_round)

    def trains_byzantine_locally(self):
        return True

    def get_perturbed_client_models(self, **kwargs):
        return Utils.get_client_models(self.clients)

    def _target_value_like(self, x, domain="private"):
        mean, std = self._norm_stats_for_domain(domain)
        if mean is None or std is None:
            return torch.full((1, x.shape[1], 1, 1), self.trigger_value, device=x.device, dtype=x.dtype)
        value = torch.full((1, x.shape[1], 1, 1), self.trigger_value, device=x.device, dtype=x.dtype)
        return (value - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)

    def _select_by_label_or_rate(self, y, rate):
        if self.source_class is not None:
            return y == self.source_class
        if rate >= 1.0:
            return torch.ones_like(y, dtype=torch.bool)
        if rate <= 0.0:
            return torch.zeros_like(y, dtype=torch.bool)
        return torch.rand(y.shape, device=y.device) < rate

    def _select_target_class(self, y, rate=1.0):
        mask = y == self.target_class
        if rate < 1.0:
            mask = mask & (torch.rand(y.shape, device=y.device) < rate)
        return mask

    def apply_trigger(self, x, domain="private"):
        raise NotImplementedError

    def poison_train_batch(self, x, y):
        mask = self._select_by_label_or_rate(y, self.poison_rate)
        if not torch.any(mask):
            return x, y
        x_poison = x.clone()
        y_poison = y.clone()
        x_poison[mask] = self.apply_trigger(x_poison[mask], domain="private")
        y_poison[mask] = self.target_class
        return x_poison, y_poison

    def poison_upload_batch(self, x, y):
        mask = self._select_by_label_or_rate(y, self.upload_rate)
        if not torch.any(mask):
            return x
        x_poison = x.clone()
        x_poison[mask] = self.apply_trigger(x_poison[mask], domain="public")
        return x_poison

    def build_asr_batch(self, x, y):
        """Create triggered evaluation samples and target labels for ASR."""
        if self.source_class is not None:
            mask = y == self.source_class
        else:
            mask = y != self.target_class
        if not torch.any(mask):
            return None, None
        x_triggered = self.apply_trigger(x[mask].clone(), domain="private")
        y_target = torch.full(
            (x_triggered.shape[0],),
            self.target_class,
            device=y.device,
            dtype=y.dtype,
        )
        return x_triggered, y_target

    def train_byzantine_epoch(self, client, epoch=None, current_round=None):
        epoch_str = f"\nEpoch {epoch} - " if epoch is not None else ""
        sys.stdout.write(
            f"{epoch_str}Backdoor training {client.actor_name} with {self.attack_name}: "
            f"target={self.target_class}, source={self.source_class}, poison_rate={self.poison_rate}.\n"
        )
        loader = client.dataloader
        with torch.set_grad_enabled(True):
            with tqdm(loader, leave=True) as pbar:
                for x_input, y_target, _ in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    x_input, y_target = self.poison_train_batch(x_input, y_target)

                    client.optimizer.zero_grad()
                    with autocast(enabled=(self.runner.use_amp is True)):
                        output = client.model.train()(x_input)
                        loss = client.loss_criterion(output, y_target)

                    client.gradScaler.scale(loss).backward()
                    client.gradScaler.step(client.optimizer)
                    client.gradScaler.update()
                    client.scheduler.step()
                    client.update_batch_metrics(mode="train", loss=loss, output=output, y_target=y_target)

    @torch.no_grad()
    def get_perturbed_client_logits(self, **kwargs):
        loader = self.runner.dataloaders_public["train"]
        logits_store_tensors = [
            torch.zeros(len(loader.dataset), self.n_classes, device=self.device)
            for _ in range(len(self.clients))
        ]

        sys.stdout.write(
            f"\nCollecting client logits with {self.attack_name} upload-time trigger "
            f"for active Byzantine clients.\n"
        )
        with tqdm(loader, leave=True) as pbar:
            for x_input, y_target, indices in pbar:
                x_input = x_input.to(self.device, non_blocking=True)
                y_target = y_target.to(self.device, non_blocking=True)
                for client_idx, client in enumerate(self.clients):
                    x_eval = x_input
                    if client.is_byzantine and self.is_attack_active():
                        x_eval = self.poison_upload_batch(x_input, y_target)
                    with autocast(enabled=(self.runner.use_amp is True)):
                        output = client.model.eval()(x_eval).float()
                    logits_store_tensors[client_idx][indices] = output.detach()
        return logits_store_tensors

    @torch.no_grad()
    def get_perturbed_client_predictions(self, **kwargs):
        logits_store_tensors = self.get_perturbed_client_logits(**kwargs)
        return [torch.nn.functional.softmax(logits, dim=1) for logits in logits_store_tensors]

class BadNets(BackdoorAttackBase):
    """Classic square patch trigger with poisoned labels."""

    attack_name = "BadNets"

    def _patch_slice(self, h, w):
        s = min(self.trigger_size, h, w)
        pos = self.trigger_position.lower()
        if pos == "top_left":
            return slice(0, s), slice(0, s)
        if pos == "top_right":
            return slice(0, s), slice(w - s, w)
        if pos == "bottom_left":
            return slice(h - s, h), slice(0, s)
        return slice(h - s, h), slice(w - s, w)

    def apply_trigger(self, x, domain="private"):
        x_out = x.clone()
        _, _, h, w = x_out.shape
        hs, ws = self._patch_slice(h, w)
        x_out[:, :, hs, ws] = self._target_value_like(x_out, domain=domain)
        return x_out


class WaNet(BackdoorAttackBase):
    """Warping-based trigger using a deterministic smooth sinusoidal grid."""

    attack_name = "WaNet"

    def apply_trigger(self, x, domain="private"):
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        flow_x = self.wanet_strength * torch.sin(3.14159265 * grid_y) * torch.sin(2.0 * 3.14159265 * grid_x)
        flow_y = self.wanet_strength * torch.sin(3.14159265 * grid_x) * torch.sin(2.0 * 3.14159265 * grid_y)
        grid = torch.stack((grid_x + flow_x, grid_y + flow_y), dim=-1)
        grid = grid.clamp(-1, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


class LabelConsistent(BadNets):
    """Clean-label patch backdoor: poison target-class samples while keeping labels unchanged."""

    attack_name = "LabelConsistent"

    def poison_train_batch(self, x, y):
        mask = self._select_target_class(y, self.poison_rate)
        if not torch.any(mask):
            return x, y
        x_poison = x.clone()
        x_poison[mask] = self.apply_trigger(x_poison[mask], domain="private")
        return x_poison, y

