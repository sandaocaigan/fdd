"""Short fixed-strength branches restored from the same clean FL state."""

import copy
import hashlib
import json
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from byzantine.defences import NoDefence
from runners.ScratchRunner import ScratchRunner


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def actor_state(actor):
    return {"model": cpu_copy(actor.model.state_dict()),
            "optimizer": cpu_copy(actor.optimizer.state_dict()),
            "scheduler": cpu_copy(actor.scheduler.state_dict()),
            "scaler": cpu_copy(actor.gradScaler.state_dict()),
            "modes": [m.training for m in actor.model.modules()]}


def restore_actor(actor, state):
    actor.model.load_state_dict(state["model"])
    actor.optimizer.load_state_dict(state["optimizer"])
    actor.scheduler.load_state_dict(state["scheduler"])
    actor.gradScaler.load_state_dict(state["scaler"])
    for module, training in zip(actor.model.modules(), state["modes"]):
        module.training = training
    actor.optimizer.zero_grad(set_to_none=True)
    actor.reset_averaged_metrics()


def dataset_description(dataset):
    result = {"type": type(dataset).__name__, "length": len(dataset)}
    if hasattr(dataset, "indices"):
        result["indices"] = [int(i) for i in dataset.indices]
    if hasattr(dataset, "dataset"):
        result["dataset"] = dataset_description(dataset.dataset)
    if hasattr(dataset, "base_dataset"):
        result["base"] = dataset_description(dataset.base_dataset)
    return result


class Tee:
    def __init__(self, parent, file):
        self.parent, self.file = parent, file

    def write(self, value):
        self.parent.write(value)
        return self.file.write(value)

    def flush(self):
        self.parent.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.parent, name)


@contextmanager
def phase_log(path):
    with path.open("x", encoding="utf-8") as stream:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, stream), Tee(err, stream)
        try:
            print("Start:", datetime.now().astimezone().isoformat())
            yield
        finally:
            print("End:", datetime.now().astimezone().isoformat())
            sys.stdout, sys.stderr = out, err


class ObservedMean(NoDefence):
    @torch.no_grad()
    def get_aggregated_logits(self, uploads):
        aggregate, scores = super().get_aggregated_logits(uploads)
        self.runner.observe_aggregate(aggregate)
        return aggregate, scores


class ObservedAggregation:
    """Observe the configured aggregator without replacing its rule or state."""

    def __init__(self, defence, runner):
        self.wrapped_defence = defence
        self.runner = runner

    def __getattr__(self, name):
        return getattr(self.wrapped_defence, name)

    @torch.no_grad()
    def get_aggregated_logits(self, uploads):
        aggregate, scores = self.wrapped_defence.get_aggregated_logits(uploads)
        self.runner.observe_aggregate(aggregate)
        return aggregate, scores


class PoisonedFLPairRunner(ScratchRunner):
    """Use the ordinary training methods, with optional output-only diagnostics."""

    counters = ("total_epochs_completed", "total_bytes_communicated",
                "client_epochs_done", "server_epochs_done")

    def _configure_branch(self, strength, spc=False, defence="NoDefence"):
        values = {"attack": "NoAttack" if strength == 0 else "PoisonedFL",
                  "n_byzantine_clients": 0 if strength == 0 else self.pair_byzantine,
                  "poisonedfl_strength": strength or 1.0,
                  "defence": defence}
        # Branch-only settings: keep the saved Mean warmup signature unchanged.
        if defence == "TriMean":
            values.update(trimean_trim_ratio=0.3)
        elif defence == "Krum":
            values.update(krum_f=self.pair_byzantine, krum_num_selected=3,
                          krum_fallback="error")
        elif defence == "AFA":
            values.update(afa_xi=2.0, afa_delta_xi=0.5, afa_prior_alpha=3.0,
                          afa_prior_beta=3.0, afa_block_threshold=0.95)
        elif defence != "NoDefence":
            raise ValueError(f"Unsupported paired aggregation: {defence}")
        if isinstance(self.config, dict):
            self.config.update(values)
        else:
            self.config.update(values, allow_val_change=True)
        # SPC is a branch condition; warmup always uses the ordinary path.
        spc_value = bool(spc)
        if isinstance(self.config, dict):
            self.config["probe_core_sigma_filter"] = spc_value
        else:
            self.config.update({"probe_core_sigma_filter": spc_value}, allow_val_change=True)
        for client in self.clients:
            client.is_byzantine = False
        self.set_attack_defence()
        if self.diagnosing:
            if defence == "NoDefence":
                self.defence = ObservedMean(config=self.config, clients=self.clients,
                                            runner_instance=self)
            else:
                self.defence = ObservedAggregation(self.defence, self)
        self._filter_diagnostics = {}

    def _signature(self):
        excluded = {"run_id", "computer", "device", "attack", "n_byzantine_clients",
                    "poisonedfl_strength", "poisonedfl_pair_rounds",
                    "poisonedfl_pair_samples", "poisonedfl_pair_checkpoint",
                    "poisonedfl_pair_spc", "poisonedfl_pair_spc_only",
                    "poisonedfl_pair_traditional"}
        settings = {k: v for k, v in dict(self.config).items() if k not in excluded}
        content = {"settings": settings, "torch": str(torch.__version__),
                   "private": [dataset_description(c.trainData) for c in self.clients],
                   "public": dataset_description(self.dataloaders_public["train"].dataset),
                   "probe_metadata": self.probe_metadata}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def _save_state(self, path, signature):
        state = {"format": 1, "signature": signature, "round": self.current_round,
                 "actors": [actor_state(a) for a in [self.server] + self.clients],
                 "rng": capture_rng(),
                 "counters": {k: getattr(self, k) for k in self.counters},
                 "round_metrics": copy.deepcopy(self.round_metrics),
                 "ensemble": cpu_copy(self.ensemble_test_acc)}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("xb") as stream:
            torch.save(state, stream)
        temporary.replace(path)
        print(f"[Paired checkpoint] saved={path}, round={self.current_round}, "
              f"signature={signature}, bytes={path.stat().st_size}")

    def _restore_state(self, path, signature):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state.get("format") != 1 or state.get("signature") != signature
                or state.get("round") != self.warmup_rounds
                or len(state["actors"]) != len(self.clients) + 1):
            raise ValueError("Paired checkpoint does not match this configuration/data split.")
        for actor, saved in zip([self.server] + self.clients, state["actors"]):
            restore_actor(actor, saved)
        for key, value in state["counters"].items():
            setattr(self, key, value)
        self.current_round = state["round"]
        self.ensemble_test_acc = state["ensemble"]
        self.round_metrics = state["round_metrics"]
        self.defence_time = None
        restore_rng(state["rng"])
        print(f"[Paired checkpoint] restored={path}, round={self.current_round}, "
              f"signature={signature}")

    def _run_round(self, number):
        self.current_round = number
        started = time.time()
        print(f"\nFL - Round {number}/{self.config.n_communications}")
        for actor in [self.server] + self.clients:
            actor.reset_averaged_metrics()
        length = 0
        if number:
            self.strategy.before_local_training()
            length = self.strategy.get_phase_length(current_round=number)
            self.train_client_local(n_epochs=length, current_round=number)
            self.strategy.after_local_training()
        self.evaluate_model(actor=self.server, data="val")
        self.evaluate_model(actor=self.server, data="test")
        self.strategy.at_round_end()
        self.total_epochs_completed += length
        self.log_at_round_end(round=number, round_n_epochs=length,
                              round_runtime=time.time() - started)
        metrics = self.server.get_metrics()
        self.round_metrics = {"round": number,
                              "server_test_acc": float(metrics["test"]["accuracy"]),
                              "server_test_loss": float(metrics["test"]["loss"]),
                              "ensemble_test_acc": float(self.ensemble_test_acc)}
        print("[Paired accuracy] " + json.dumps(self.round_metrics, allow_nan=False))

    @torch.no_grad()
    def get_client_logits_and_predictions(self, mode, include_server=False):
        result = super().get_client_logits_and_predictions(mode, include_server)
        if getattr(self, "diagnosing", False) and mode == "train":
            self.clean_aggregate = torch.stack(result[0]).mean(dim=0)
        return result

    @torch.no_grad()
    def _public_outputs(self):
        modes = [m.training for m in self.server.model.modules()]
        outputs = []
        try:
            with preserve_rng():
                self.server.model.eval()
                for images, _, _ in self.diagnostic_loader:
                    outputs.append(self.server.model(images.to(self.device)).float())
        finally:
            for module, training in zip(self.server.model.modules(), modes):
                module.training = training
        result = torch.cat(outputs)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite server diagnostic outputs.")
        return result

    def _kd(self, output, target):
        temperature = float(self.config.distill_temperature)
        return float(F.kl_div((output / temperature).log_softmax(1),
                              (target / temperature).softmax(1),
                              reduction="batchmean") * temperature ** 2)

    @torch.no_grad()
    def observe_aggregate(self, aggregate):
        ids = self.diagnostic_ids.to(aggregate.device)
        self.clean_sample = self.clean_aggregate[ids].clone()
        self.attack_sample = aggregate[ids].clone()
        self.before_outputs = self._public_outputs()
        self.before_parameters = [p.detach().cpu().clone() for p in self.server.model.parameters()]
        shift = aggregate - self.clean_aggregate
        self.diagnostics = {
            "clean_aggregate_rms": float(self.clean_aggregate.square().mean().sqrt()),
            "aggregate_shift_rms": float(shift.square().mean().sqrt()),
            "aggregate_top1_change": float((aggregate.argmax(1) !=
                                             self.clean_aggregate.argmax(1)).float().mean()),
            "kd_clean_before": self._kd(self.before_outputs, self.clean_sample),
            "kd_attack_before": self._kd(self.before_outputs, self.attack_sample),
        }
        for name, target in (("clean", self.clean_aggregate), ("attack", aggregate)):
            log_probs = (target / float(self.config.distill_temperature)).log_softmax(1)
            self.diagnostics[f"{name}_target_entropy"] = float(
                -(log_probs.exp() * log_probs).sum(1).mean())
        self.clean_aggregate = None

    def maybe_filter_core_sigma_logits(self, client_logits_list, audit_rows):
        """Record raw/effective upload membership around the normal SPC hook."""
        raw_mean = torch.stack(client_logits_list).mean(dim=0)
        clean = getattr(self, "clean_aggregate", None)
        raw_shift = None if clean is None else raw_mean - clean
        retained, retained_ids = super().maybe_filter_core_sigma_logits(
            client_logits_list, audit_rows)
        all_ids = [int(client.client_id) for client in self.clients]
        flagged_ids = sorted(int(row["client_id"]) for row in audit_rows
                             if bool(row.get("core_sigma_flagged", False)))
        malicious_ids = sorted(int(client.client_id) for client in self.clients
                               if getattr(client, "is_byzantine", False))
        excluded_ids = sorted(set(all_ids) - set(retained_ids))
        flagged, malicious = set(flagged_ids), set(malicious_ids)
        self._filter_diagnostics = {
            "spc_enabled": bool(getattr(self.config, "probe_core_sigma_filter", False)),
            "spc_audited_clients": len(audit_rows),
            "spc_flagged_ids": flagged_ids,
            "spc_retained_ids": [int(x) for x in retained_ids],
            "spc_excluded_ids": excluded_ids,
            "spc_malicious_ids": malicious_ids,
            "spc_tp": len(flagged & malicious),
            "spc_fp": len(flagged - malicious),
            "spc_fn": len(malicious - flagged),
            "spc_tn": len(set(all_ids) - flagged - malicious),
            "spc_excluded_malicious": len(set(excluded_ids) & malicious),
            "spc_excluded_benign": len(set(excluded_ids) - malicious),
        }
        if raw_shift is not None:
            self._filter_diagnostics["raw_pre_filter_aggregate_shift_rms"] = float(
                raw_shift.square().mean().sqrt())
        return retained, retained_ids

    def collect_avg_output_and_distill_to_server(self):
        super().collect_avg_output_and_distill_to_server()
        if not self.diagnosing:
            return
        self.diagnostics.update(getattr(self, "_filter_diagnostics", {}))
        after = self._public_outputs()
        delta = after - self.before_outputs
        shift = self.attack_sample - self.clean_sample
        delta = delta - delta.mean(1, keepdim=True)
        shift = shift - shift.mean(1, keepdim=True)
        shift_norm, delta_norm = shift.norm(), delta.norm()
        cosine = None if min(float(shift_norm), float(delta_norm)) < 1e-10 else float(
            (delta * shift).sum() / (shift_norm * delta_norm))
        update_squared = sum(float((p.detach().cpu().double() - previous.double()).square().sum())
                             for p, previous in zip(self.server.model.parameters(), self.before_parameters))
        self.diagnostics.update({
            "kd_clean_after": self._kd(after, self.clean_sample),
            "kd_attack_after": self._kd(after, self.attack_sample),
            "server_public_top1_change": float((after.argmax(1) !=
                                                  self.before_outputs.argmax(1)).float().mean()),
            "server_output_shift_cosine": cosine,
            "server_parameter_update_l2": update_squared ** 0.5,
            "diagnostic_samples": len(after),
        })
        self.before_parameters = None
        print("[Paired signal] " + json.dumps(self.diagnostics, allow_nan=False))

    def train_federated(self):
        self.warmup_rounds = int(self.config.attack_start_round) - 1
        branch_rounds = int(self.config.poisonedfl_pair_rounds)
        self.pair_byzantine = int(self.config.n_byzantine_clients)
        use_spc_matrix = bool(getattr(self.config, "poisonedfl_pair_spc", False))
        use_spc_only = bool(getattr(self.config, "poisonedfl_pair_spc_only", False))
        use_traditional = bool(getattr(self.config, "poisonedfl_pair_traditional", False))
        if sum((use_spc_matrix, use_spc_only, use_traditional)) > 1:
            raise ValueError("Choose either one SPC branch mode or the traditional-defence branches.")
        if (self.config.strategy != "FedDistill" or self.config.defence != "NoDefence"
                or self.config.memory_method is not None or self.config.poisonedfl_adaptive
                or self.config.poisonedfl_direction != "fixed"
                or (self.config.probe_core_sigma_filter and not (use_spc_matrix or use_spc_only))
                or self.config.use_diffusion_purifier
                or any(getattr(self.config, k) for k in ("restart_client_lr", "reinit_server",
                        "warm_restarts", "client_early_stopping", "server_early_stopping"))):
            raise ValueError("Paired runner requires fixed, nonadaptive Mean and no restart/early stopping.")
        if (self.warmup_rounds < 1 or branch_rounds < 1
                or self.warmup_rounds + branch_rounds > self.config.n_communications
                or not 0 < self.pair_byzantine < len(self.clients)
                or not self.config.poisonedfl_min_strength <= 1
                or not self.config.poisonedfl_max_strength >= 10
                or self.config.poisonedfl_pair_samples < 1):
            raise ValueError("Invalid paired rounds, malicious count, diagnostic sample count, or strength bounds.")
        if use_traditional and (len(self.clients) <= 2 * self.pair_byzantine + 2
                                or len(self.clients) - self.pair_byzantine - 2 < 3):
            raise ValueError("Traditional branches require feasible Multi-Krum f and three selections.")
        base = int(self.config.run_id)
        branch_count = 6 if use_traditional else (2 if use_spc_only else (8 if use_spc_matrix else 4))
        summary = Path(f"logs/{base}_{base+branch_count}_poisonedfl_paired.summary.txt")
        checkpoint = Path(self.config.poisonedfl_pair_checkpoint or
                          f"checkpoints/poisonedfl_pair_{base}_round{self.warmup_rounds}.pt")
        conditions = (("clear", 0, False, "NoDefence"),
                      ("strength1", 1, False, "NoDefence"),
                      ("strength5", 5, False, "NoDefence"),
                      ("strength10", 10, False, "NoDefence"))
        if use_spc_matrix:
            conditions = tuple((f"strength{strength}_{'spc' if spc else 'nospc'}",
                                strength, spc, "NoDefence")
                               for strength in (0, 1, 5, 10)
                               for spc in (False, True))
        elif use_spc_only:
            conditions = (("clear_spc", 0, True, "NoDefence"),
                          ("strength10_spc", 10, True, "NoDefence"))
        elif use_traditional:
            conditions = tuple((f"{'clear' if strength == 0 else 'strength10'}_{label}",
                                strength, False, defence)
                               for label, defence in (("trimean", "TriMean"),
                                                      ("multikrum", "Krum"), ("afa", "AFA"))
                               for strength in (0, 10))
        phases = [(base, "warmup")] + [(base+i+1, name) for i, (name, _, _, _) in enumerate(conditions)]
        paths = [Path(f"logs/{run}_poisonedfl_pair_{name}.log") for run, name in phases]
        metric_paths = [p.with_suffix(".metrics.jsonl") for p in paths[1:]]
        if any(p.exists() for p in [summary] + paths + metric_paths):
            raise FileExistsError("Paired output already exists; use fresh run IDs.")
        summary.parent.mkdir(parents=True, exist_ok=True)
        signature = self._signature()
        self.diagnosing = False
        self._configure_branch(0)
        with phase_log(paths[0]):
            if checkpoint.exists():
                self._restore_state(checkpoint, signature)
            else:
                for number in range(self.warmup_rounds + 1):
                    self._run_round(number)
                self._save_state(checkpoint, signature)
            print("Exit code: 0")
        with summary.open("x", encoding="utf-8") as stream:
            stream.write(f"Shared clean checkpoint: {checkpoint}\nSignature: {signature}\n"
                         f"Warmup accuracy: {json.dumps(self.round_metrics)}\n"
                         f"Warmup={self.warmup_rounds}; branch_rounds={branch_rounds}; "
                         f"LR horizon={self.config.n_communications}; fixed directions; warmup=Mean; "
                         f"SPC matrix={use_spc_matrix}; SPC only={use_spc_only}; "
                         f"traditional={use_traditional}\n")
        dataset = self.dataloaders_public["train"].dataset
        generator = torch.Generator().manual_seed(int(self.seed) + 17713)
        self.diagnostic_ids = torch.randperm(len(dataset), generator=generator)[
            :min(len(dataset), int(self.config.poisonedfl_pair_samples))].sort().values
        self.diagnostic_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, self.diagnostic_ids.tolist()),
            batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        baseline = {}
        defence_baselines = {}
        self.diagnosing = True
        for index, (name, strength, spc, defence) in enumerate(conditions):
            run_id, _ = phases[index + 1]
            with phase_log(paths[index + 1]), metric_paths[index].open("x", encoding="utf-8") as metrics:
                self._configure_branch(strength, spc=spc, defence=defence)
                self._restore_state(checkpoint, signature)
                print(f"[Paired branch] run_id={run_id}, condition={name}, strength={strength}")
                for number in range(self.warmup_rounds + 1, self.warmup_rounds + branch_rounds + 1):
                    self._run_round(number)
                    record = dict(self.round_metrics, **self.diagnostics, run_id=run_id,
                                  condition=name, strength=strength)
                    if use_traditional:
                        record["defence"] = defence
                        clean = defence_baselines.setdefault(defence, {})
                        if strength == 0:
                            clean[number] = record["server_test_acc"]
                        record["delta_vs_clean_defence_pp"] = 100 * (
                            record["server_test_acc"] - clean[number])
                        record["delta_vs_clear_pp"] = record["delta_vs_clean_defence_pp"]
                    elif use_spc_only:
                        if strength == 0:
                            baseline[number] = record["server_test_acc"]
                        record["delta_vs_clear_spc_pp"] = 100 * (
                            record["server_test_acc"] - baseline[number])
                        record["delta_vs_clear_pp"] = record["delta_vs_clear_spc_pp"]
                    else:
                        if strength == 0 and not spc:
                            baseline[number] = record["server_test_acc"]
                        record["delta_vs_clear_nospc_pp"] = 100 * (
                            record["server_test_acc"] - baseline.get(number, record["server_test_acc"]))
                        # Preserve the existing four/eight-branch metric convention.
                        record["delta_vs_clear_pp"] = record["delta_vs_clear_nospc_pp"]
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                    metrics.flush()
                result = (f"Run {run_id}: condition={name}, exit=0, "
                          f"server_test_acc={record['server_test_acc']:.6f}, "
                          f"ensemble_test_acc={record['ensemble_test_acc']:.6f}, "
                          f"delta_vs_clear_pp={record['delta_vs_clear_pp']:.4f}")
                print(result)
                print("Exit code: 0")
                with summary.open("a", encoding="utf-8") as stream:
                    stream.write(result + "\n")
        with summary.open("a", encoding="utf-8") as stream:
            stream.write("All PoisonedFL paired branches finished.\n")
        print("All PoisonedFL paired branches finished.")
