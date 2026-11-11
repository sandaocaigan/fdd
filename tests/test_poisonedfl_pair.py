import ast
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from actors import Client, Server
from byzantine.defences import AFA, Krum, TriMean
from runners.PoisonedFLPairRunner import (ObservedAggregation, PoisonedFLPairRunner, actor_state,
                                         capture_rng, preserve_rng, restore_actor,
                                         restore_rng)
from runners.ScratchRunner import ScratchRunner
from strategies import FedDistill


class Config(dict):
    __getattr__ = dict.__getitem__


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class PoisonedFLPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "defaults" for t in n.targets))
        cls.defaults = {k.arg: ast.literal_eval(k.value) for k in defaults.keywords}

    def make_runner(self, directory, run_id=90001, checkpoint=None, n_clients=4):
        config = Config(self.defaults)
        config.update(run_id=run_id, dataset="cifar10", arch="synthetic", batch_size=4,
                      strategy="FedDistill", attack="PoisonedFL", defence="NoDefence",
                      seed=20260805, n_clients=n_clients, n_byzantine_clients=1,
                      n_communications=70, n_total_local_epochs=70,
                      attack_start_round=2, probe_core_sigma_filter=False,
                      poisonedfl_pair_rounds=2, poisonedfl_pair_samples=8,
                      poisonedfl_pair_checkpoint=str(checkpoint or Path(directory)/"warmup.pt"),
                      use_amp=False, device="cpu", server_early_stopping=False)
        runner = object.__new__(PoisonedFLPairRunner)
        runner.config, runner.device = config, torch.device("cpu")
        runner.seed, runner.n_classes, runner.use_amp = 20260805, 3, False
        runner.tmp_dir, runner.current_round = directory, None
        runner.probe_auditor = runner.probe_metadata = runner.diffusion_purifier = None
        runner.defence_time = runner.ensemble_test_acc = None
        for name in runner.counters:
            setattr(runner, name, 0)
        generator = torch.Generator().manual_seed(107)
        x = torch.randn(8, 4, generator=generator)
        y = torch.arange(8) % 3
        data = torch.utils.data.TensorDataset(x, y, torch.arange(8))
        runner.dataloaders_public = {
            name: torch.utils.data.DataLoader(data, batch_size=4, shuffle="train" in name)
            for name in ("train", "train_server", "val", "test")}
        runner.clients = []
        for client_id in range(1, n_clients + 1):
            actor = Client(use_amp=False, client_id=client_id, n_classes=3, tmp_dir=directory,
                           num_workers=0, config=config, callbacks=None, device=runner.device)
            runner.clients.append(actor)
        runner.server = Server(use_amp=False, n_classes=3, tmp_dir=directory, num_workers=0,
                               config=config, callbacks=None, device=runner.device)
        for actor in [runner.server] + runner.clients:
            actor.model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 3))
            actor.assign_dataset(data)
            actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.05, momentum=0.9)
            actor.scheduler = torch.optim.lr_scheduler.LinearLR(
                actor.optimizer, start_factor=1, end_factor=0, total_iters=140)
        runner.strategy = FedDistill(config=config, runner_instance=runner)
        return runner

    def quiet(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(patch("wandb.log"))
        stack.enter_context(patch("runners.ScratchRunner.Utils.dump_bar_plot_to_wandb"))
        return stack

    def test_checkpoint_restores_optimizer_scheduler_modes_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory)
            actor = runner.server
            x = torch.ones(4, 4)
            actor.model.train()
            actor.model(x).square().mean().backward()
            actor.optimizer.step()
            actor.scheduler.step()
            state = actor_state(actor)
            self.assertTrue(state["optimizer"]["state"])
            def next_update():
                actor.optimizer.zero_grad()
                actor.model(x * 2).square().mean().backward()
                actor.optimizer.step()
                actor.scheduler.step()
                return actor_state(actor)
            expected = next_update()
            actor.model.eval()
            restore_actor(actor, state)
            actual = next_update()
            for key in expected["model"]:
                torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
            self.assertEqual(actual["scheduler"], expected["scheduler"])
            self.assertEqual(actual["optimizer"]["param_groups"], expected["optimizer"]["param_groups"])

    def test_preserve_rng_including_exception(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(5))
        restore_rng(state)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with preserve_rng():
                random.random()
                np.random.rand()
                torch.rand(5)
                raise RuntimeError("deliberate")
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)

    def test_new_matrix_switch_does_not_invalidate_legacy_checkpoint_signature(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory)
            with_key = runner._signature()
            runner.config.pop("poisonedfl_pair_spc", None)
            runner.config.pop("poisonedfl_pair_spc_only", None)
            runner.config.pop("poisonedfl_pair_traditional", None)
            without_key = runner._signature()
            self.assertEqual(with_key, without_key)

    def test_four_real_branches_and_checkpoint_reuse_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            checkpoint = Path(directory)/"warmup.pt"
            with working_directory(directory):
                for run_id in (90001, 90011):
                    runner = self.make_runner(directory, run_id, checkpoint)
                    runner.train_federated()
                    summary = Path(f"logs/{run_id}_{run_id+4}_poisonedfl_paired.summary.txt").read_text()
                    self.assertIn("All PoisonedFL paired branches finished.", summary)
                records = []
                for base in (90001, 90011):
                    branch_records = []
                    for i, condition in enumerate(("clear", "strength1", "strength5", "strength10")):
                        path = Path(f"logs/{base+i+1}_poisonedfl_pair_{condition}.metrics.jsonl")
                        rows = [json.loads(s) for s in path.read_text().splitlines()]
                        self.assertEqual([r["round"] for r in rows], [2, 3])
                        for row in rows:
                            self.assertAlmostEqual(row["aggregate_shift_rms"], row["strength"] / 4, places=5)
                            self.assertGreater(row["server_parameter_update_l2"], 0)
                            self.assertEqual(row["diagnostic_samples"], 8)
                            row.pop("run_id")
                        branch_records.append(rows)
                    first_clean = branch_records[0][0]["clean_aggregate_rms"]
                    self.assertTrue(all(rows[0]["clean_aggregate_rms"] == first_clean for rows in branch_records))
                    records.append(branch_records)
                self.assertEqual(records[0], records[1])

    def test_diagnostics_do_not_change_training_update_or_rng(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory)
            runner.pair_byzantine, runner.diagnosing = 1, False
            runner._configure_branch(0)
            runner.current_round = 2
            model_state = actor_state(runner.server)
            rng = capture_rng()
            ScratchRunner.collect_avg_output_and_distill_to_server(runner)
            expected = actor_state(runner.server)
            next_random = torch.rand(5)
            restore_actor(runner.server, model_state)
            runner.diagnosing = True
            runner._configure_branch(0)
            runner.diagnostic_ids = torch.arange(8)
            runner.diagnostic_loader = runner.dataloaders_public["val"]
            restore_rng(rng)
            runner.collect_avg_output_and_distill_to_server()
            actual = actor_state(runner.server)
            for key in expected["model"]:
                torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(torch.rand(5), next_random, rtol=0, atol=0)
            self.assertEqual(runner.diagnostics["aggregate_shift_rms"], 0)
            self.assertIsNone(runner.diagnostics["server_output_shift_cosine"])

    def test_incompatible_snapshot_and_output_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet(), working_directory(directory):
            runner = self.make_runner(directory)
            runner.train_federated()
            checkpoint = Path(directory)/"warmup.pt"
            with self.assertRaisesRegex(FileExistsError, "fresh run"):
                runner.train_federated()
            other = self.make_runner(directory, 90011, checkpoint)
            other.config["momentum"] = 0.8
            with self.assertRaisesRegex(ValueError, "does not match"):
                other.train_federated()

    def test_two_spc_only_branches_reuse_checkpoint_and_compare_clean_spc(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet(), working_directory(directory):
            legacy = self.make_runner(directory, 90201)
            legacy.train_federated()
            runner = self.make_runner(directory, 90301)
            runner.config["poisonedfl_pair_spc_only"] = True

            class DeterministicAudit:
                def audit(self, client_prediction_list, clients=None, current_round=None):
                    return [{"client_id": int(client.client_id),
                             "core_sigma_flagged": bool(getattr(client, "is_byzantine", False))}
                            for client in clients]

            runner.probe_auditor = DeterministicAudit()
            runner.train_federated()
            warmup = Path("logs/90301_poisonedfl_pair_warmup.log").read_text()
            self.assertIn("[Paired checkpoint] restored=", warmup)
            self.assertNotIn("FL - Round", warmup)
            self.assertEqual(len(list(Path("logs").glob("9030*_*.metrics.jsonl"))), 2)
            clean = [json.loads(s) for s in Path(
                "logs/90302_poisonedfl_pair_clear_spc.metrics.jsonl").read_text().splitlines()]
            attacked = [json.loads(s) for s in Path(
                "logs/90303_poisonedfl_pair_strength10_spc.metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["round"] for r in clean], [2, 3])
            self.assertEqual([r["round"] for r in attacked], [2, 3])
            for normal, poisoned in zip(clean, attacked):
                self.assertTrue(normal["spc_enabled"] and poisoned["spc_enabled"])
                self.assertEqual(normal["spc_tp"], 0)
                self.assertEqual(normal["delta_vs_clear_spc_pp"], 0)
                self.assertEqual(poisoned["spc_tp"], 1)
                self.assertEqual(poisoned["spc_excluded_malicious"], 1)
                self.assertEqual(poisoned["spc_excluded_benign"], 0)
                self.assertAlmostEqual(poisoned["raw_pre_filter_aggregate_shift_rms"], 2.5, places=5)
                self.assertAlmostEqual(poisoned["delta_vs_clear_spc_pp"],
                    100 * (poisoned["server_test_acc"] - normal["server_test_acc"]))
                self.assertEqual(poisoned["delta_vs_clear_pp"], poisoned["delta_vs_clear_spc_pp"])
                self.assertNotIn("delta_vs_clear_nospc_pp", poisoned)
            summary = Path("logs/90301_90303_poisonedfl_paired.summary.txt").read_text()
            self.assertIn("SPC only=True", summary)
            self.assertIn("All PoisonedFL paired branches finished.", summary)

    def test_conflicting_spc_branch_modes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory)
            runner.config.update(poisonedfl_pair_spc=True, poisonedfl_pair_spc_only=True)
            with self.assertRaisesRegex(ValueError, "Choose either"):
                runner.train_federated()

    def test_observer_preserves_each_robust_aggregator_and_afa_state(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory, n_clients=6)
            runner.pair_byzantine, runner.diagnosing = 1, True
            generator = torch.Generator().manual_seed(902)
            uploads = [torch.randn(8, 3, generator=generator) for _ in runner.clients]
            with patch.object(runner, "observe_aggregate") as observe:
                for name, cls in (("TriMean", TriMean), ("Krum", Krum), ("AFA", AFA)):
                    runner._configure_branch(10, defence=name)
                    wrapped = runner.defence
                    self.assertIsInstance(wrapped, ObservedAggregation)
                    self.assertIsInstance(wrapped.wrapped_defence, cls)
                    expected = cls(config=runner.config, clients=runner.clients, runner_instance=runner)
                    ids = [client.client_id for client in runner.clients]
                    wrapped.set_active_client_ids(ids)
                    expected.set_active_client_ids(ids)
                    for _ in range(2):
                        actual_aggregate, actual_scores = wrapped.get_aggregated_logits(uploads)
                        expected_aggregate, expected_scores = expected.get_aggregated_logits(uploads)
                        torch.testing.assert_close(actual_aggregate, expected_aggregate, rtol=0, atol=0)
                        torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
                    if name == "AFA":
                        self.assertEqual(wrapped.good_counts, expected.good_counts)
                        self.assertEqual(wrapped.bad_counts, expected.bad_counts)
                        self.assertTrue(any(wrapped.good_counts.values()) or any(wrapped.bad_counts.values()))
                        runner._configure_branch(0, defence="AFA")
                        self.assertIsNot(runner.defence.wrapped_defence, wrapped.wrapped_defence)
                        self.assertFalse(any(runner.defence.good_counts.values()))
                        self.assertFalse(any(runner.defence.bad_counts.values()))
                        self.assertFalse(runner.defence.blocked_client_ids)
                self.assertEqual(observe.call_count, 6)

    def test_six_traditional_branches_restore_same_checkpoint_and_use_own_clean_baselines(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet(), working_directory(directory):
            legacy = self.make_runner(directory, 90401, n_clients=6)
            legacy.train_federated()
            runner = self.make_runner(directory, 90501, n_clients=6)
            runner.config["poisonedfl_pair_traditional"] = True
            runner.train_federated()
            self.assertEqual(len(list(Path("logs").glob("9050*_*.metrics.jsonl"))), 6)
            first_clean_rms = []
            for index, (label, defence) in enumerate((("trimean", "TriMean"),
                                                      ("multikrum", "Krum"), ("afa", "AFA"))):
                groups = []
                for offset, (prefix, strength) in enumerate((("clear", 0), ("strength10", 10))):
                    run_id = 90502 + 2 * index + offset
                    phase = Path(f"logs/{run_id}_poisonedfl_pair_{prefix}_{label}")
                    records = [json.loads(line) for line in phase.with_suffix(".metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([r["round"] for r in records], [2, 3])
                    self.assertIn("[Paired checkpoint] restored=", phase.with_suffix(".log").read_text())
                    for record in records:
                        self.assertEqual(record["defence"], defence)
                        self.assertFalse(record["spc_enabled"])
                        self.assertEqual(len(record["spc_retained_ids"]), 6)
                        self.assertAlmostEqual(record["raw_pre_filter_aggregate_shift_rms"], strength / 6, places=5)
                        self.assertGreater(record["server_parameter_update_l2"], 0)
                        self.assertNotIn("delta_vs_clear_nospc_pp", record)
                    first_clean_rms.append(records[0]["clean_aggregate_rms"])
                    groups.append(records)
                for clean, attacked in zip(*groups):
                    self.assertEqual(clean["delta_vs_clean_defence_pp"], 0)
                    self.assertAlmostEqual(attacked["delta_vs_clean_defence_pp"],
                        100 * (attacked["server_test_acc"] - clean["server_test_acc"]))
                    self.assertEqual(attacked["delta_vs_clear_pp"], attacked["delta_vs_clean_defence_pp"])
            self.assertTrue(all(value == first_clean_rms[0] for value in first_clean_rms))
            summary = Path("logs/90501_90507_poisonedfl_paired.summary.txt").read_text()
            self.assertIn("traditional=True", summary)
            self.assertIn("All PoisonedFL paired branches finished.", summary)

    def test_traditional_mode_rejects_conflicting_modes_and_infeasible_multikrum(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            runner = self.make_runner(directory)
            runner.config.update(poisonedfl_pair_traditional=True, poisonedfl_pair_spc_only=True)
            with self.assertRaisesRegex(ValueError, "Choose either"):
                runner.train_federated()
            runner.config["poisonedfl_pair_spc_only"] = False
            with self.assertRaisesRegex(ValueError, "feasible Multi-Krum"):
                runner.train_federated()

    def test_eight_branch_spc_matrix_records_filter_confusion_and_shift(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet(), working_directory(directory):
            runner = self.make_runner(directory, 90101)
            runner.config["poisonedfl_pair_spc"] = True

            class DeterministicAudit:
                def audit(self, client_prediction_list, clients=None, current_round=None):
                    return [{"client_id": int(client.client_id),
                             "core_sigma_flagged": bool(getattr(client, "is_byzantine", False))}
                            for client in clients]

            runner.probe_auditor = DeterministicAudit()
            runner.train_federated()
            conditions = [(strength, suffix)
                          for strength in (0, 1, 5, 10)
                          for suffix in ("nospc", "spc")]
            for offset, (strength, suffix) in enumerate(conditions, 1):
                path = Path(f"logs/{90101 + offset}_poisonedfl_pair_strength{strength}_{suffix}.metrics.jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([row["round"] for row in rows], [2, 3])
                self.assertTrue(all("raw_pre_filter_aggregate_shift_rms" in row for row in rows))
                if suffix == "spc":
                    self.assertTrue(all(row["spc_enabled"] for row in rows))
                    expected_malicious = 0 if strength == 0 else 1
                    self.assertTrue(all(row["spc_tp"] == expected_malicious for row in rows))
                    self.assertTrue(all(row["spc_excluded_malicious"] == expected_malicious
                                        for row in rows))
                else:
                    self.assertTrue(all(not row["spc_enabled"] for row in rows))
            self.assertIn("SPC matrix=True", Path(
                "logs/90101_90109_poisonedfl_paired.summary.txt").read_text())


if __name__ == "__main__":
    unittest.main()
