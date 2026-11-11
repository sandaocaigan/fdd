import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from byzantine.attacks import PoisonedFL
from byzantine.defences import AFA, Krum, NoDefence, TriMean
from byzantine.poisonedfl import PoisonedFLLogitController
from probes.probe_auditor import ProbeAuditor
from runners.ScratchRunner import ScratchRunner


class PoisonedFLControllerTests(unittest.TestCase):
    def test_fixed_offsets_are_shared_and_persistent_without_changing_inputs(self):
        ctrl = PoisonedFLLogitController(7, strength=2)
        inputs = [torch.randn(30, 10), torch.randn(30, 10)]
        copies = [x.clone() for x in inputs]
        first, _ = ctrl.poison(inputs, 15)
        second, diag = ctrl.poison(inputs, 16)
        torch.testing.assert_close(first[0] - inputs[0], first[1] - inputs[1])
        torch.testing.assert_close(first[0], second[0])
        for a, b in zip(inputs, copies):
            torch.testing.assert_close(a, b)
        self.assertAlmostEqual(diag["direction_cosine"], 1)

    def test_centered_direction_has_unit_row_rms_and_changes_softmax(self):
        for classes in (2, 3, 10, 100):
            inputs = torch.zeros(100, classes, dtype=torch.float64)
            result, _ = PoisonedFLLogitController(9).poison([inputs], 1)
            torch.testing.assert_close(result[0].mean(1), torch.zeros(100, dtype=torch.float64))
            torch.testing.assert_close(result[0].square().mean(1), torch.ones(100, dtype=torch.float64))
            self.assertFalse(torch.equal(result[0].softmax(1), inputs.softmax(1)))

    def test_fresh_direction_is_reproducible_but_changes_between_rounds(self):
        inputs = [torch.zeros(1024, 10)]
        ctrl = PoisonedFLLogitController(13, direction="fresh")
        a, _ = ctrl.poison(inputs, 15)
        b, diag = ctrl.poison(inputs, 16)
        self.assertFalse(torch.equal(a[0], b[0]))
        self.assertLess(abs(diag["direction_cosine"]), 0.1)
        again, _ = PoisonedFLLogitController(13, direction="fresh").poison(inputs, 15)
        torch.testing.assert_close(a[0], again[0])

    def test_row_reordering_uses_sample_ids(self):
        x = torch.arange(60, dtype=torch.float64).reshape(20, 3)
        order = torch.arange(19, -1, -1)
        ctrl = PoisonedFLLogitController(42)
        first, _ = ctrl.poison([x], 15)
        permuted, _ = ctrl.poison([x[order]], 16, sample_ids=order)
        torch.testing.assert_close(permuted[0], first[0][order])

    def test_private_direction_rng_preserves_global_rng(self):
        state = torch.random.get_rng_state().clone()
        PoisonedFLLogitController(42).poison([torch.zeros(10, 3)], 1)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_seed_changes_direction(self):
        x = [torch.zeros(100, 10)]
        a, _ = PoisonedFLLogitController(1).poison(x, 1)
        b, _ = PoisonedFLLogitController(2).poison(x, 1)
        self.assertFalse(torch.equal(a[0], b[0]))

    def test_adaptive_growth_uses_previous_direction_and_respects_ceiling(self):
        ctrl = PoisonedFLLogitController(3, adaptive=True, feedback_interval=1,
                                         max_strength=1.25)
        x = torch.zeros(20, 3, dtype=torch.float64)
        first, _ = ctrl.poison([x], 1, reference_logits=x)
        _, diag = ctrl.poison([x], 2, reference_logits=first[0])
        self.assertEqual(diag["adjustment"], "grow")
        self.assertAlmostEqual(diag["feedback_alignment"], 1)
        self.assertEqual(diag["strength"], 1.25)
        _, diag = ctrl.poison([x], 3, reference_logits=first[0] * 2)
        self.assertEqual(diag["strength"], 1.25)

    def test_adaptive_shrink_respects_floor(self):
        ctrl = PoisonedFLLogitController(3, adaptive=True, feedback_interval=1,
                                         min_strength=0.8)
        x = torch.zeros(20, 3, dtype=torch.float64)
        first, _ = ctrl.poison([x], 1, reference_logits=x)
        _, diag = ctrl.poison([x], 2, reference_logits=-first[0])
        self.assertEqual(diag["adjustment"], "shrink")
        self.assertEqual(diag["strength"], 0.8)
        _, diag = ctrl.poison([x], 3, reference_logits=-first[0] * 2)
        self.assertEqual(diag["strength"], 0.8)

    def test_feedback_waits_for_interval_and_holds_on_zero_signal(self):
        ctrl = PoisonedFLLogitController(3, adaptive=True, feedback_interval=2)
        x = torch.zeros(20, 3)
        for r in (1, 2, 3):
            _, diag = ctrl.poison([x], r, reference_logits=x)
            self.assertEqual(diag["strength"], 1)
            self.assertEqual(diag["adjustment"], "hold")
            self.assertIsNone(diag["feedback_alignment"])

    def test_feedback_is_aligned_after_row_permutation_and_copies_reference(self):
        ctrl = PoisonedFLLogitController(3, adaptive=True, feedback_interval=1)
        x = torch.zeros(20, 3, dtype=torch.float64)
        first, _ = ctrl.poison([x], 1, reference_logits=x)
        x.fill_(99)
        order = torch.arange(19, -1, -1)
        _, diag = ctrl.poison([torch.zeros_like(x)], 2,
                              reference_logits=first[0][order], sample_ids=order)
        self.assertEqual(diag["adjustment"], "grow")

    def test_invalid_settings_are_rejected(self):
        cases = [dict(strength=0), dict(strength=-1), dict(strength=float("nan")),
                 dict(max_strength=float("inf")), dict(min_strength=2),
                 dict(direction="random"), dict(feedback_interval=0),
                 dict(feedback_target=0.5), dict(feedback_target=1),
                 dict(adjust_factor=1), dict(adaptive="false")]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PoisonedFLLogitController(3, **settings)

    def test_invalid_uploads_and_references_are_rejected(self):
        for x in (torch.ones(3), torch.ones(3, 1), torch.ones(3, 2, dtype=torch.int64),
                  torch.full((3, 2), float("inf")), torch.full((3, 2), float("nan"))):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                PoisonedFLLogitController(3).poison([x], 1)
        with self.assertRaisesRegex(ValueError, "aligned"):
            PoisonedFLLogitController(3).poison([torch.zeros(3, 2), torch.zeros(4, 2)], 1)
        with self.assertRaisesRegex(ValueError, "server"):
            PoisonedFLLogitController(3, adaptive=True).poison([torch.zeros(3, 2)], 1)
        with self.assertRaisesRegex(ValueError, "aligned"):
            PoisonedFLLogitController(3, adaptive=True).poison(
                [torch.zeros(3, 2)], 1, reference_logits=torch.zeros(4, 2))

    def test_invalid_ids_and_dataset_changes_are_rejected(self):
        for ids in ([0, 0, 2], [0, 1], [0, 1, 3], [-1, 1, 2],
                    [0., 1., 2.], [False, True, True]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "sample IDs"):
                PoisonedFLLogitController(3).poison([torch.zeros(3, 2)], 1, sample_ids=ids)
        ctrl = PoisonedFLLogitController(3)
        ctrl.poison([torch.zeros(3, 2)], 1)
        with self.assertRaisesRegex(ValueError, "shape changed"):
            ctrl.poison([torch.zeros(4, 2)], 2)

    def test_rounds_must_increase(self):
        ctrl = PoisonedFLLogitController(3)
        ctrl.poison([torch.zeros(3, 2)], 2)
        for r in (1, 2):
            with self.assertRaisesRegex(ValueError, "increasing"):
                ctrl.poison([torch.zeros(3, 2)], r)

    def test_dtype_empty_inputs_and_overflow(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.zeros(3, 2, dtype=dtype)
            result, _ = PoisonedFLLogitController(3).poison([x], 1)
            self.assertEqual(result[0].dtype, dtype)
            self.assertEqual(result[0].device, x.device)
            self.assertFalse(result[0].requires_grad)
        result, diag = PoisonedFLLogitController(3).poison([torch.zeros(0, 3)], 1)
        self.assertEqual(result[0].shape, (0, 3))
        self.assertEqual(diag["offset_rms"], 0)
        self.assertEqual(PoisonedFLLogitController(3).poison([], 1)[0], [])
        with self.assertRaisesRegex(ValueError, "overflows"):
            PoisonedFLLogitController(3, strength=1e5, max_strength=1e6).poison(
                [torch.zeros(3, 2, dtype=torch.float16)], 1)

    def test_rms_diagnostic_uses_actual_rounded_upload_offsets(self):
        x = torch.full((3, 2), 60000., dtype=torch.float16)
        result, diag = PoisonedFLLogitController(3).poison([x], 1)
        torch.testing.assert_close(result[0], x)
        self.assertEqual(diag["offset_rms"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_direction_matches_cpu(self):
        cpu, _ = PoisonedFLLogitController(3).poison([torch.zeros(30, 10)], 1)
        gpu, _ = PoisonedFLLogitController(3).poison([torch.zeros(30, 10, device="cuda")], 1)
        torch.testing.assert_close(cpu[0], gpu[0].cpu())


class PoisonedFLIntegrationTests(unittest.TestCase):
    def make_attack(self, adaptive=False, current_round=15):
        self.uploads = [torch.tensor([[2., 1., 0.], [0., 2., 1.]]) + i for i in range(3)]
        self.calls = []
        def collect(mode, include_server=False):
            self.calls.append((mode, include_server))
            data = (self.uploads, None)
            return data + (torch.zeros_like(self.uploads[0]),) if include_server else data
        runner = SimpleNamespace(seed=7, current_round=current_round,
                                 get_client_logits_and_predictions=collect)
        clients = [SimpleNamespace(client_id=i+1, is_byzantine=i > 0) for i in range(3)]
        return PoisonedFL(config=SimpleNamespace(strategy="FedDistill", attack_start_round=15,
                                                poisonedfl_adaptive=adaptive),
                          clients=clients, runner_instance=runner)

    def test_only_active_malicious_uploads_change(self):
        attack = self.make_attack(current_round=14)
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)
        self.assertIsNone(attack.controller._last_round)
        attack.runner.current_round = 15
        originals = [x.clone() for x in self.uploads]
        with redirect_stdout(io.StringIO()) as output:
            result = attack.get_perturbed_client_logits()
        self.assertIs(result[0], self.uploads[0])
        self.assertFalse(torch.equal(result[1], self.uploads[1]))
        torch.testing.assert_close(result[1] - self.uploads[1], result[2] - self.uploads[2])
        for a, b in zip(originals, self.uploads):
            torch.testing.assert_close(a, b)
        self.assertIn("PoisonedFL inspired", output.getvalue())

    def test_feedback_collection_is_opt_in_and_inactive_rounds_do_not_use_it(self):
        attack = self.make_attack(adaptive=True, current_round=14)
        attack.get_perturbed_client_logits()
        self.assertEqual(self.calls[-1], ("train", False))
        attack.runner.current_round = 15
        with redirect_stdout(io.StringIO()):
            attack.get_perturbed_client_logits()
        self.assertEqual(self.calls[-1], ("train", True))

    def test_zero_malicious_clients_are_an_exact_noop(self):
        attack = self.make_attack(adaptive=True)
        for c in attack.clients:
            c.is_byzantine = False
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)
        self.assertEqual(self.calls[-1], ("train", False))

    def test_invalid_strategy_and_client_alignment(self):
        attack = self.make_attack()
        self.uploads.pop()
        with self.assertRaisesRegex(ValueError, "aligned"):
            attack.get_perturbed_client_logits()
        attack.config.strategy = "FedAVG"
        with self.assertRaisesRegex(ValueError, "FedDistill"):
            PoisonedFL(config=attack.config, clients=attack.clients, runner_instance=attack.runner)

    def test_cli_types_are_coerced(self):
        # Load only pure parser helpers; importing main would launch a training run.
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ("_parse_bool", "_coerce_cli_value")]
        import argparse
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "main.py", "exec"), namespace)
        coerce = namespace["_coerce_cli_value"]
        self.assertIs(coerce("poisonedfl_adaptive", "false", False), False)
        self.assertIs(coerce("poisonedfl_adaptive", "true", False), True)
        self.assertEqual(coerce("poisonedfl_feedback_interval", "3", 5), 3)
        self.assertEqual(coerce("poisonedfl_strength", "2.5", 1), 2.5)

    def test_logits_remain_compatible_with_probes_and_all_aggregators(self):
        clients = [SimpleNamespace(client_id=i+1, is_byzantine=i == 5, trainData=range(5))
                   for i in range(6)]
        uploads = [torch.tensor([[2., 0., 1.], [1.9, 0., 1.], [0., 2., 1.], [0., 1.9, 1.]])
                   for _ in clients]
        uploads[-1] = PoisonedFLLogitController(7).poison([uploads[-1]], 15)[0][0]
        auditor = ProbeAuditor({"probe_indices": [0, 1, 2, 3],
                                "probe_groups": [[0, 1], [2, 3]], "entries": []},
                               n_classes=3, log_each_round=False)
        rows = auditor.audit([x.softmax(1) for x in uploads], clients=clients, current_round=15)
        auditor._apply_core_sigma(rows)
        self.assertEqual(len(rows), 6)
        with redirect_stdout(io.StringIO()):
            for cls in (NoDefence, TriMean, Krum, AFA):
                defence = cls(clients=clients, config=SimpleNamespace(krum_f=1), runner_instance=None)
                defence.set_active_client_ids(range(1, 7))
                aggregate, scores = defence.get_aggregated_logits(uploads)
                self.assertEqual(aggregate.shape, (4, 3))
                self.assertTrue(torch.isfinite(aggregate).all())
                self.assertEqual(scores.shape, (6,))


class ServerFeedbackCollectionTests(unittest.TestCase):
    def make_runner(self):
        runner = object.__new__(ScratchRunner)
        x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        ids = torch.arange(8)
        runner.device = torch.device("cpu")
        runner.n_classes = 3
        runner.use_amp = False
        generator = torch.Generator().manual_seed(81)
        runner.dataloaders_public = {"train": torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, torch.full((8,), -999), ids), batch_size=3,
            shuffle=True, generator=generator)}
        runner.clients = [SimpleNamespace(model=torch.nn.Linear(4, 3)) for _ in range(2)]
        runner.server = SimpleNamespace(model=torch.nn.Sequential(
            torch.nn.BatchNorm1d(4), torch.nn.Dropout(), torch.nn.Linear(4, 3)))
        return runner, x

    def test_server_feedback_uses_same_indexed_images_and_preserves_model_state(self):
        runner, x = self.make_runner()
        runner.server.model.train()
        runner.server.model[1].eval()
        modes = [m.training for m in runner.server.model.modules()]
        state = {k: v.clone() for k, v in runner.server.model.state_dict().items()}
        with redirect_stdout(io.StringIO()):
            uploads, probabilities, reference = runner.get_client_logits_and_predictions(
                "train", include_server=True)
        self.assertEqual([m.training for m in runner.server.model.modules()], modes)
        for k, v in state.items():
            torch.testing.assert_close(v, runner.server.model.state_dict()[k])
        runner.server.model.eval()
        torch.testing.assert_close(reference, runner.server.model(x))
        for i, client in enumerate(runner.clients):
            torch.testing.assert_close(uploads[i], client.model(x))
            torch.testing.assert_close(probabilities[i], uploads[i].softmax(1))

    def test_default_return_contract_stays_two_items(self):
        runner, _ = self.make_runner()
        with redirect_stdout(io.StringIO()):
            result = runner.get_client_logits_and_predictions("train")
        self.assertEqual(len(result), 2)

    def test_feedback_does_not_advance_additional_global_rng(self):
        runner, _ = self.make_runner()
        initial = torch.random.get_rng_state()
        with redirect_stdout(io.StringIO()):
            runner.get_client_logits_and_predictions("train")
        after_default = torch.random.get_rng_state()
        torch.random.set_rng_state(initial)
        with redirect_stdout(io.StringIO()):
            runner.get_client_logits_and_predictions("train", include_server=True)
        self.assertTrue(torch.equal(after_default, torch.random.get_rng_state()))

    def test_server_mode_is_restored_on_forward_failure(self):
        runner, _ = self.make_runner()
        runner.server.model.train()
        modes = [m.training for m in runner.server.model.modules()]
        def broken(*args):
            raise RuntimeError("synthetic forward failure")
        runner.clients[0].model.forward = broken
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic"):
            runner.get_client_logits_and_predictions("train", include_server=True)
        self.assertEqual([m.training for m in runner.server.model.modules()], modes)

    def test_two_round_attack_spc_aggregation_and_real_server_distillation(self):
        for adaptive in (False, True):
            for filtering in (False, True):
                with self.subTest(adaptive=adaptive, filtering=filtering):
                    runner, _ = self.make_runner()
                    runner.seed = 9
                    runner.current_round = 15
                    runner.clients = [SimpleNamespace(
                        model=torch.nn.Linear(4, 3), client_id=i+1, is_byzantine=i == 3,
                    ) for i in range(4)]
                    runner.config = SimpleNamespace(
                        strategy="FedDistill", attack="PoisonedFL", defence="NoDefence",
                        poisonedfl_adaptive=adaptive, poisonedfl_feedback_interval=1,
                        attack_start_round=15, probe_core_sigma_filter=filtering,
                        distill_temperature=1., n_server_epochs_per_round=1,
                        n_communications=70, server_early_stopping=False,
                    )
                    server = runner.server
                    server.model = torch.nn.Linear(4, 3)
                    server.actor_type = "server"
                    server.actor_name = "server"
                    server.optimizer = torch.optim.SGD(server.model.parameters(), lr=0.01)
                    server.scheduler = torch.optim.lr_scheduler.StepLR(server.optimizer, step_size=100)
                    server.gradScaler = torch.cuda.amp.GradScaler(enabled=False)
                    server.update_batch_metrics = lambda **kwargs: None
                    server.reset_averaged_metrics = lambda: None
                    server.reset_val_and_test_metrics = lambda: None
                    runner.dataloaders_public["train_server"] = runner.dataloaders_public["train"]
                    runner.server_epochs_done = 0
                    runner.total_bytes_communicated = 0
                    runner.diffusion_purifier = None
                    runner.probe_auditor = ProbeAuditor({
                        "probe_indices": [0, 1, 2, 3],
                        "probe_groups": [[0, 1], [2, 3]], "entries": [],
                    }, n_classes=3, log_each_round=False, core_sigma_enabled=filtering)
                    runner.evaluate_model = lambda **kwargs: None
                    runner.log_server = lambda **kwargs: None
                    runner.attack = PoisonedFL(config=runner.config, clients=runner.clients,
                                               runner_instance=runner)
                    runner.defence = NoDefence(config=runner.config, clients=runner.clients,
                                               runner_instance=runner)
                    initial = {k: v.clone() for k, v in server.model.state_dict().items()}
                    with patch("runners.ScratchRunner.Utils.dump_bar_plot_to_wandb"), \
                            redirect_stdout(io.StringIO()):
                        for current in (15, 16):
                            runner.current_round = current
                            runner.collect_avg_output_and_distill_to_server()
                    self.assertEqual(runner.server_epochs_done, 2)
                    self.assertEqual(runner.attack.controller._last_round, 16)
                    self.assertGreater(runner.total_bytes_communicated, 0)
                    self.assertTrue(all(torch.isfinite(v).all() for v in server.model.state_dict().values()))
                    self.assertTrue(any(not torch.equal(initial[k], v)
                                        for k, v in server.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
