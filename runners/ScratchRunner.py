# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         runners/ScratchRunner.py
# Description:  Scratch Runner class, used for starting the run from scratch
# ===========================================================================
import sys
import time
from typing import Optional

import torch
import wandb
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from actors import Actor
from byzantine import attacks, defences
from runners.BaseRunner import BaseRunner
from utilities import Utilities as Utils


class ScratchRunner(BaseRunner):
    """Handles federated training from random initialization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_round = None
        self.artifact = None

        entity, project = wandb.run.entity, wandb.run.project
        self.initial_artifact_name = (
            f"seed_placeholder-{entity}-{project}-{self.config.arch}-"
            f"{self.config.dataset}-{self.config.run_id}"
        )

    def find_existing_seed(self):
        """Find an existing wandb artifact and pull the seed."""
        try:
            self.artifact = wandb.run.use_artifact(f"{self.initial_artifact_name}:latest")
            seed = self.artifact.metadata["seed"]
            self.seed = seed
        except Exception as e:
            print(e)

        output_str = (
            f"Found {self.initial_artifact_name} with seed {seed}"
            if self.artifact is not None
            else "Nothing found."
        )
        sys.stdout.write(f"Trying to find reference artifact in project: {output_str}\n")

    def save_artifact_seed(self):
        """Save artifact and seed before training so other runs can fetch it."""
        if self.artifact is None:
            self.artifact = wandb.Artifact(
                self.initial_artifact_name,
                type="seed_placeholder",
                metadata={"seed": self.seed},
            )
            sys.stdout.write(f"Creating {self.initial_artifact_name}.\n")
            wandb.run.use_artifact(self.artifact)

    @torch.no_grad()
    def broadcast_server_model_to_clients(self):
        """Broadcast server model params to all clients."""
        sys.stdout.write("Broadcasting server model to clients.\n")
        server_state_dict = self.server.model.state_dict()
        for client in self.clients:
            client.model.load_state_dict(server_state_dict)

        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    @torch.no_grad()
    def broadcast_agg_client_models_to_server(self):
        """FedAVG path: aggregate client models and update the server."""
        sys.stdout.write(f"{self.config.attack}: Broadcasting clients models and applying attack.\n")
        client_model_list = self.attack.get_perturbed_client_models()

        sys.stdout.write(f"{self.config.defence}: Averaging models with defence mechanism.\n")
        averaged_model = self.defence.get_aggregated_model(client_model_list)
        self.server.model.load_state_dict(averaged_model)

        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    def set_attack_defence(self):
        """Set attack and defence objects, and mark Byzantine clients."""
        if self.config.attack not in [None, "None", "none", "NoAttack"]:
            n_byzantine_clients = self.config.n_byzantine_clients or 0
            assert 0 <= n_byzantine_clients <= self.config.n_clients, (
                "Number of byzantine clients must be in [0, n_clients]."
            )
            sys.stdout.write(f"{n_byzantine_clients} byzantine clients with attack {self.config.attack}.\n")

            attack_selection_generator = torch.Generator(device="cpu")
            attack_selection_generator.manual_seed((int(self.seed) + 7919) % (2 ** 63 - 1))
            byzantine_client_indices = torch.randperm(
                len(self.clients),
                generator=attack_selection_generator,
            )[:n_byzantine_clients]
            byzantine_ids_str = ", ".join(
                str(self.clients[int(x)].client_id) for x in byzantine_client_indices.tolist()
            )
            sys.stdout.write(f"Client(s): {byzantine_ids_str} are byzantine.\n")
            for idx in byzantine_client_indices:
                self.clients[idx].is_byzantine = True

            try:
                self.attack = getattr(attacks, self.config.attack)(
                    clients=self.clients,
                    config=self.config,
                    runner_instance=self,
                )
            except AttributeError:
                raise AttributeError(f"Attack {self.config.attack} not found.")
        else:
            sys.stdout.write("No attack.\n")
            assert self.config.n_byzantine_clients in [0, None, "None"], (
                "If no attack is used, n_byzantine_clients must be 0."
            )
            self.attack = attacks.NoAttack(clients=self.clients, config=self.config, runner_instance=self)

        if self.config.defence not in [None, "None", "none"]:
            if self.config.memory_method is not None:
                robust_method = getattr(defences, self.config.defence)
                if self.config.memory_method == "expweights":
                    self.defence = defences.choose_aggregation_expweights(robust_method)(
                        clients=self.clients,
                        config=self.config,
                        runner_instance=self,
                    )
                else:
                    raise AttributeError(f"Memory method {self.config.memory_method} not found.")
            else:
                try:
                    self.defence = getattr(defences, self.config.defence)(
                        clients=self.clients,
                        config=self.config,
                        runner_instance=self,
                    )
                except AttributeError:
                    raise AttributeError(f"Defence {self.config.defence} not found.")

            sys.stdout.write(f"Using defence {self.config.defence}.\n")
        else:
            sys.stdout.write("No defence.\n")
            self.defence = defences.NoDefence(clients=self.clients, config=self.config, runner_instance=self)

    def set_client_models(self):
        """Initialize one model per client."""
        for client in self.clients:
            client.set_model(reinit=True, fileName=None)

    def set_client_optimizers(self, reinit_optimizer: bool = True, lr_duration: Optional[int] = None):
        """Set optimizers and schedulers for all clients."""
        clients_train_on_public = self.strategy.do_clients_train_on_public_data()
        add_public = 0 if not clients_train_on_public else len(self.dataloaders_public["train"])
        assert add_public == 0, "FED does not work with the current schedulers."
        for client in self.clients:
            n_batches_per_epoch = len(client.dataloader)
            n_epochs = lr_duration or self.config["n_total_local_epochs"]
            client.set_optimizer_and_scheduler(
                n_epochs=n_epochs,
                n_batches_per_epoch=n_batches_per_epoch,
                reinit_optimizer=reinit_optimizer,
            )

    def set_server_optimizer(self, reinit_server: bool, first_init: bool):
        """Set optimizer and scheduler for the server model."""
        n_base_epochs = self.config.n_server_epochs_per_round

        if reinit_server:
            sys.stdout.write("Reinitializing server optimizer and scheduler.\n")
            n_epochs = n_base_epochs
        else:
            n_epochs = self.config.n_communications * n_base_epochs
        n_batches_per_epoch = len(self.server.dataloader)
        self.server.set_optimizer_and_scheduler(
            n_epochs=n_epochs,
            n_batches_per_epoch=n_batches_per_epoch,
            reinit_optimizer=(reinit_server or first_init),
        )

    @torch.no_grad()
    def compute_accuracy(self, loader, prediction):
        """Compute accuracy for a tensor containing all predictions of an ensemble."""
        sys.stdout.write("Evaluating accuracy of ensemble.\n")
        accuracy_meter = Accuracy(num_classes=self.n_classes).to(device=self.device)
        with tqdm(loader, leave=True) as pbar:
            for _, y_target, indices in pbar:
                y_target = y_target.to(device=self.device)
                accuracy_meter(prediction[indices], y_target)
        return accuracy_meter.compute()

    @torch.no_grad()
    def get_client_logits_and_predictions(self, mode: str):
        """Collect client logits and softmax probabilities on the public train/test set."""
        assert mode in ["train", "test"]
        loader = self.dataloaders_public[mode]
        sys.stdout.write("\nCollecting logits and predictions of all clients.\n")

        logits_store_tensors = [
            torch.zeros(len(loader.dataset), self.n_classes, device=self.device)
            for _ in range(len(self.clients))
        ]
        prediction_store_tensors = [
            torch.zeros(len(loader.dataset), self.n_classes, device=self.device)
            for _ in range(len(self.clients))
        ]

        with tqdm(loader, leave=True) as pbar:
            for x_input, _, indices in pbar:
                x_input = x_input.to(self.device, non_blocking=True)
                with autocast(enabled=(self.use_amp is True)):
                    for client_idx, client in enumerate(self.clients):
                        output = client.model.eval()(x_input).float()
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        logits_store_tensors[client_idx][indices] = output.detach()
                        prediction_store_tensors[client_idx][indices] = probabilities.detach()

        return logits_store_tensors, prediction_store_tensors

    @torch.no_grad()
    def get_client_predictions(self, mode: str):
        """Collect only client softmax probabilities on public train/test data."""
        _, prediction_store_tensors = self.get_client_logits_and_predictions(mode=mode)
        return prediction_store_tensors

    def logits_to_probabilities(self, logits: torch.Tensor, temperature: Optional[float] = None):
        """Convert uploaded/aggregated logits to soft labels for auditing or distillation."""
        if temperature is None:
            temperature = getattr(self.config, "distill_temperature", 1.0)
        temperature = float(temperature or 1.0)
        if temperature <= 0:
            raise ValueError(f"distill_temperature must be positive, got {temperature}.")
        return torch.nn.functional.softmax(logits / temperature, dim=1)

    def logits_list_to_probabilities(self, client_logits_list, temperature: Optional[float] = None):
        return [self.logits_to_probabilities(logits, temperature=temperature) for logits in client_logits_list]

    def distill(self, actor: Actor, avg_output: torch.tensor, is_training: bool = True):
        """Train or evaluate an actor using averaged public soft labels."""
        if actor.actor_type == "server":
            loader = self.dataloaders_public["train_server"]
        else:
            loader = self.dataloaders_public["train"]
        sys.stdout.write(
            f"\n{'Training' if is_training else 'Evaluating'} {actor.actor_name} "
            f"on temperature-softmax labels from aggregated logits.\n"
        )
        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, _, indices in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)
                    target = avg_output[indices].to(self.device, non_blocking=True)
                    actor.optimizer.zero_grad()

                    with autocast(enabled=(self.use_amp is True)):
                        output = actor.model.train(mode=is_training)(x_input)
                        temperature = float(getattr(self.config, "distill_temperature", 1.0) or 1.0)
                        student_log_probs = torch.nn.functional.log_softmax(output / temperature, dim=1)
                        loss = torch.nn.functional.kl_div(
                            student_log_probs, target, reduction="batchmean"
                        ) * (temperature ** 2)
                    if is_training:
                        actor.gradScaler.scale(loss).backward()
                        actor.gradScaler.step(actor.optimizer)
                        actor.gradScaler.update()
                        actor.scheduler.step()

                    if actor.actor_type == "server":
                        actor.update_batch_metrics(mode="train", loss=loss, output=output, y_target=None)

    def train_client_local(self, n_epochs: int, current_round: int):
        """Train each non-active Byzantine client locally on its private dataset."""
        for epoch in range(1, n_epochs + 1, 1):
            for client in self.clients:
                client.reset_averaged_metrics()
                if client.is_byzantine and self.attack.is_attack_active():
                    trains_byzantine = getattr(self.attack, "trains_byzantine_locally", lambda: False)()
                    if not trains_byzantine:
                        sys.stdout.write(
                            f"\nRound {current_round}/{self.config.n_communications} - "
                            f"Local Epoch {epoch}/{n_epochs}: "
                            f"Skipping active byzantine client-{client.client_id}."
                        )
                        continue
                    sys.stdout.write(
                        f"\nRound {current_round}/{self.config.n_communications} - "
                        f"Local Epoch {epoch}/{n_epochs}: Backdoor training client-{client.client_id}."
                    )
                    self.attack.train_byzantine_epoch(client=client, epoch=epoch, current_round=current_round)
                else:
                    sys.stdout.write(
                        f"\nRound {current_round}/{self.config.n_communications} - "
                        f"Local Epoch {epoch}/{n_epochs}: Locally training client-{client.client_id}."
                    )
                    self.train_epoch(actor=client, data="train", epoch=epoch)
                if self.config.client_early_stopping:
                    self.evaluate_model(actor=client, data="val")
                if epoch == n_epochs:
                    self.evaluate_model(actor=client, data="test")

                if self.config.client_early_stopping:
                    client.update_checkpoint()

            self.log_clients_at_epoch_end(epoch=self.client_epochs_done + epoch, commit=True)
        self.client_epochs_done += n_epochs

    def maybe_audit_probe_predictions(self, client_prediction_list):
        """Run probe auditing on probability tensors after attack perturbation."""
        if self.probe_auditor is None:
            return []
        return self.probe_auditor.audit(
            client_prediction_list=client_prediction_list,
            clients=self.clients,
            current_round=self.current_round,
        )

    def maybe_audit_probe_logits(self, client_logits_list):
        """Audit uploaded logits by first converting them to probabilities."""
        client_prediction_list = self.logits_list_to_probabilities(client_logits_list, temperature=1.0)
        return self.maybe_audit_probe_predictions(client_prediction_list)

    @torch.no_grad()
    def maybe_evaluate_backdoor_asr(self):
        """Evaluate attack success rate for input-trigger backdoor attacks."""
        if not getattr(self.config, "evaluate_backdoor_asr", False):
            return []
        if self.current_round in [None, 0]:
            return []
        if self.attack is None or not hasattr(self.attack, "build_asr_batch"):
            return []
        if not self.attack.is_attack_active():
            return []

        frequency = int(getattr(self.config, "asr_eval_frequency", 1) or 1)
        if frequency <= 0:
            return []
        is_final_round = int(self.current_round) == int(self.config.n_communications)
        if int(self.current_round) % frequency != 0 and not is_final_round:
            return []

        data = str(getattr(self.config, "asr_eval_data", "test") or "test")
        if data not in ["val", "test"]:
            raise ValueError("Backdoor ASR needs labeled validation/test data; use asr_eval_data='val' or 'test'.")
        if data not in self.dataloaders_public:
            raise ValueError(f"ASR eval data '{data}' not found in public dataloaders.")
        batch_limit = getattr(self.config, "asr_batch_limit", None)
        batch_limit = None if batch_limit in [None, "None", "none"] else int(batch_limit)

        loader = self.dataloaders_public[data]
        successes = torch.zeros(len(self.clients), device=self.device)
        totals = torch.zeros(len(self.clients), device=self.device)

        sys.stdout.write(
            f"\n[Backdoor ASR] Round {self.current_round}: evaluating triggered {data} samples.\n"
        )
        with tqdm(loader, leave=True) as pbar:
            for batch_idx, (x_input, y_target, _) in enumerate(pbar):
                if batch_limit is not None and batch_idx >= batch_limit:
                    break
                x_input = x_input.to(self.device, non_blocking=True)
                y_target = y_target.to(self.device, non_blocking=True)
                x_triggered, y_asr_target = self.attack.build_asr_batch(x_input, y_target)
                if x_triggered is None:
                    continue

                for client_idx, client in enumerate(self.clients):
                    with autocast(enabled=(self.use_amp is True)):
                        output = client.model.eval()(x_triggered).float()
                    predicted = output.argmax(dim=1)
                    successes[client_idx] += (predicted == y_asr_target).sum()
                    totals[client_idx] += y_asr_target.numel()

        rows = []
        for client_idx, client in enumerate(self.clients):
            total = float(totals[client_idx].detach().cpu())
            asr = 0.0 if total == 0 else float((successes[client_idx] / totals[client_idx]).detach().cpu())
            rows.append({
                "client_id": int(client.client_id),
                "role": "malicious" if getattr(client, "is_byzantine", False) else "benign",
                "asr": asr,
                "n": int(total),
            })

        self.print_and_log_asr_rows(rows)
        return rows

    def print_and_log_asr_rows(self, rows):
        """Print ASR rows and log them to wandb without committing the round."""
        if not rows:
            return

        malicious = [row["asr"] for row in rows if row["role"] == "malicious"]
        benign = [row["asr"] for row in rows if row["role"] == "benign"]
        malicious_mean = sum(malicious) / len(malicious) if malicious else 0.0
        benign_mean = sum(benign) / len(benign) if benign else 0.0
        gap = malicious_mean - benign_mean

        sys.stdout.write("client_id | role      | asr    | n_triggered\n")
        for row in sorted(rows, key=lambda x: x["client_id"]):
            sys.stdout.write(
                f"{row['client_id']:<9} | {row['role']:<9} | "
                f"{row['asr']:.4f} | {row['n']}\n"
            )
        sys.stdout.write(
            f"ASR summary: malicious_mean={malicious_mean:.4f}, "
            f"benign_mean={benign_mean:.4f}, gap={gap:.4f}\n"
        )

        if getattr(wandb, "run", None) is None:
            return
        log_dict = {
            "backdoor/asr_mean_malicious": malicious_mean,
            "backdoor/asr_mean_benign": benign_mean,
            "backdoor/asr_gap": gap,
            "backdoor/round": int(self.current_round),
        }
        for row in rows:
            prefix = f"backdoor/client{row['client_id']}"
            log_dict[f"{prefix}/asr"] = row["asr"]
            log_dict[f"{prefix}/is_malicious"] = 1 if row["role"] == "malicious" else 0
            log_dict[f"{prefix}/n_triggered"] = row["n"]
        wandb.log(log_dict, commit=False)

    def maybe_purify_suspicious_logits(self, client_logits_list, audit_rows):
        """Purify selected clients' uploaded logits before defence aggregation."""
        if self.diffusion_purifier is None:
            return client_logits_list
        if not audit_rows:
            return client_logits_list

        target = str(getattr(self.config, "diffusion_target", "yellow") or "yellow").lower()
        yellow_low = float(getattr(self.config, "diffusion_yellow_low", 0.65))
        yellow_high = float(getattr(self.config, "diffusion_yellow_high", 0.88))
        red_threshold = float(getattr(self.config, "diffusion_red_threshold", yellow_high))

        client_id_to_idx = {client.client_id: idx for idx, client in enumerate(self.clients)}
        selected_indices = []
        for row in audit_rows:
            risk = float(row["risk"])
            client_idx = client_id_to_idx.get(int(row["client_id"]))
            if client_idx is None:
                continue
            if target == "yellow" and yellow_low <= risk < yellow_high:
                selected_indices.append(client_idx)
            elif target == "red" and risk >= red_threshold:
                selected_indices.append(client_idx)
            elif target in ["suspicious", "all_suspicious"] and risk >= yellow_low:
                selected_indices.append(client_idx)
            elif target == "all":
                selected_indices.append(client_idx)

        if not selected_indices:
            return client_logits_list

        selected_ids = [self.clients[idx].client_id for idx in selected_indices]
        sys.stdout.write(
            f"\nDiffusion purifier: purifying uploaded logits from client ids {selected_ids} "
            f"with target='{target}'.\n"
        )
        purified_logits_list = list(client_logits_list)
        for client_idx in selected_indices:
            purified_logits = self.diffusion_purifier.purify(
                client_logits_list[client_idx],
                return_probs=False,
            )
            purified_logits_list[client_idx] = purified_logits.to(self.device)
        return purified_logits_list

    def collect_avg_output_and_distill_to_server(self):
        """FedDistill core: collect logits, attack, probe, purify, aggregate, distill."""
        sys.stdout.write(f"{self.config.attack}: Broadcasting client logits and applying attack.\n")
        client_logits_list = self.attack.get_perturbed_client_logits()

        audit_rows = self.maybe_audit_probe_logits(client_logits_list)
        client_logits_list = self.maybe_purify_suspicious_logits(
            client_logits_list=client_logits_list,
            audit_rows=audit_rows,
        )

        sys.stdout.write(f"{self.config.defence}: Aggregating uploaded logits with defence mechanism.\n")
        defence_start = time.time()
        if not hasattr(self.defence, "get_aggregated_logits"):
            raise AttributeError(
                f"Defence {self.config.defence} must implement get_aggregated_logits for FedDistill logit upload."
            )
        averaged_logits, mean_outlier_scores = self.defence.get_aggregated_logits(client_logits_list)
        if hasattr(self.attack, "record_aggregated_logits"):
            self.attack.record_aggregated_logits(
                aggregated_logits=averaged_logits,
                client_logits_list=client_logits_list,
            )
        self.defence_time = time.time() - defence_start

        indices = [idx for idx in range(len(mean_outlier_scores))]
        scores = [float(mean_outlier_scores[idx]) for idx in range(len(mean_outlier_scores))]
        Utils.dump_bar_plot_to_wandb(
            x=indices,
            y=scores,
            xlabel="Client ID",
            ylabel="Outlier Score",
            title="Mean Logit Outlier Scores by Client Index",
            wandb_identifier="outlier_scores",
        )

        averaged_soft_labels = self.logits_to_probabilities(averaged_logits)
        sys.stdout.write(
            f"\nDistilling to server in round {self.current_round}/{self.config.n_communications} "
            f"with temperature={float(getattr(self.config, 'distill_temperature', 1.0) or 1.0)}."
        )
        length = self.config.n_server_epochs_per_round
        for epoch in range(1, length + 1, 1):
            self.server.reset_averaged_metrics()
            self.distill(actor=self.server, avg_output=averaged_soft_labels, is_training=True)

            self.evaluate_model(actor=self.server, data="val")
            self.evaluate_model(actor=self.server, data="test")

            if self.config.server_early_stopping:
                self.server.update_checkpoint()

            if epoch == length:
                self.server.reset_val_and_test_metrics()
            self.log_server(epoch=self.server_epochs_done + epoch, commit=(epoch < length))
        self.total_bytes_communicated += Utils.calculate_communication_cost(client_logits_list)
        self.server_epochs_done += length

        if self.config.server_early_stopping:
            self.server.load_checkpoint()

    def train_federated(self):
        """Train the server and clients in a federated way."""
        for current_round in range(0, self.config.n_communications + 1, 1):
            self.current_round = current_round
            is_training = current_round > 0
            sys.stdout.write(f"\nFL - Round {current_round}/{self.config.n_communications}\n") if is_training \
                else sys.stdout.write("\nFL - Evaluation round.\n")
            t_start = time.time()

            for client in self.clients:
                client.reset_averaged_metrics()
            self.server.reset_averaged_metrics()

            if is_training:
                self.strategy.before_local_training()
                round_n_epochs = self.strategy.get_phase_length(current_round=current_round)

                if self.config.restart_client_lr:
                    self.set_client_optimizers(reinit_optimizer=False, lr_duration=round_n_epochs)
                if self.config.reinit_server:
                    self.server.set_model(reinit=True)
                    self.set_server_optimizer(reinit_server=self.config.reinit_server, first_init=False)
                if self.config.warm_restarts:
                    warmup_steps_client = int(0.05 * round_n_epochs * len(self.clients[0].dataloader))
                    sys.stdout.write("Warming up momentum for 5% of the iterations.\n")
                    for client in self.clients:
                        client.warmup_scheduler(warmup_steps=warmup_steps_client)

                    server_train_length = self.config.n_server_epochs_per_round or 0
                    if server_train_length > 0:
                        warmup_steps_server = int(
                            0.05 * server_train_length * len(self.dataloaders_public["train_server"])
                        )
                        self.server.warmup_scheduler(warmup_steps=warmup_steps_server)

                self.train_client_local(n_epochs=round_n_epochs, current_round=current_round)
                if self.config.client_early_stopping:
                    for client in self.clients:
                        client.load_checkpoint()

                self.strategy.after_local_training()
            else:
                round_n_epochs = 0

            self.evaluate_model(actor=self.server, data="val")
            self.evaluate_model(actor=self.server, data="test")
            self.strategy.at_round_end()
            self.maybe_evaluate_backdoor_asr()
            self.total_epochs_completed += round_n_epochs
            self.log_at_round_end(
                round=current_round,
                round_n_epochs=round_n_epochs,
                round_runtime=time.time() - t_start,
            )

    def run(self):
        """Complete experiment entrypoint."""
        # self.find_existing_seed()
        self.set_seed()
        self.set_client_models()
        self.server.set_model(reinit=True)
        # self.save_artifact_seed()

        self.assign_dataloaders()
        self.maybe_init_diffusion_purifier()

        self.set_client_optimizers()
        self.set_server_optimizer(reinit_server=self.config.reinit_server, first_init=True)

        self.set_attack_defence()
        self.train_federated()
        self.final_log()




