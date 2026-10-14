import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from byzantine.attacks import ManipulatingKD
from byzantine.defences import AFA, Krum, TriMean


class DummyClient:
    def __init__(self, client_id, is_byzantine):
        self.client_id = client_id
        self.is_byzantine = is_byzantine
        self.trainData = range(5)


class ManipulatingKDHistoryTests(unittest.TestCase):
    def setUp(self):
        self.attack = object.__new__(ManipulatingKD)
        self.attack.clients = [
            DummyClient(10, False),
            DummyClient(20, False),
            DummyClient(30, True),
            DummyClient(40, True),
        ]
        self.attack.config = SimpleNamespace(manipkd_history_rounds=5)
        self.attack._paper_history = []

    @staticmethod
    def logits(value):
        return torch.full((2, 3), float(value))

    def record(self, client_ids, values):
        uploads = [self.logits(value) for value in values]
        aggregate = torch.stack(uploads).mean(dim=0)
        self.attack.record_aggregated_logits(aggregate, uploads, client_ids)

    def assert_estimated_honest_mean(self, expected):
        estimated_sum = self.attack._estimate_paper_benign_sum(current_honest_count=2)
        self.assertTrue(torch.allclose(estimated_sum / 2.0, self.logits(expected)))

    def test_no_filtering_matches_original_reconstruction(self):
        self.record([10, 20, 30, 40], [1, 3, 10, 20])
        self.assert_estimated_honest_mean(2)

    def test_removed_honest_client_does_not_shift_byzantine_indices(self):
        self.record([10, 30, 40], [1, 10, 20])
        self.assert_estimated_honest_mean(1)
        self.assertEqual(self.attack._paper_history[-1]["retained_malicious_count"], 2)

    def test_some_byzantine_clients_can_be_removed(self):
        self.record([10, 20, 40], [1, 3, 20])
        self.assert_estimated_honest_mean(2)
        self.assertEqual(self.attack._paper_history[-1]["retained_malicious_count"], 1)

    def test_all_byzantine_clients_can_be_removed(self):
        self.record([10, 20], [1, 3])
        self.assert_estimated_honest_mean(2)
        self.assertEqual(self.attack._paper_history[-1]["malicious_uploads"].shape[0], 0)

    def test_history_without_retained_honest_client_is_ignored(self):
        self.record([10, 20], [1, 3])
        self.record([30, 40], [10, 20])
        self.assertEqual(len(self.attack._paper_history), 1)
        self.assert_estimated_honest_mean(2)

    def test_upload_and_client_id_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "one stable client ID"):
            self.attack.record_aggregated_logits(
                self.logits(1), [self.logits(1)], [10, 20]
            )

    def test_mean_surrogate_averages_retained_honest_estimates(self):
        self.attack.config.manipkd_pl_estimator = "mean_surrogate"
        self.record([10, 20, 30, 40], [1, 3, 10, 20])
        self.record([10, 30, 40], [5, 10, 20])
        self.assert_estimated_honest_mean(3.5)

    def test_strict_mean_keeps_latest_round_with_multiple_records(self):
        self.record([10, 20, 30, 40], [1, 3, 10, 20])
        self.record([10, 30, 40], [5, 10, 20])
        self.assert_estimated_honest_mean(5)

    def test_nonlinear_defences_require_explicit_proxy(self):
        self.attack.config.manipkd_formulation = "paper"
        self.attack.config.manipkd_knowledge = "partial_local"
        for defence in ["Krum", "TriMean", "AFA"]:
            with self.subTest(defence=defence):
                self.attack.config.defence = defence
                with self.assertRaisesRegex(ValueError, "mean_surrogate"):
                    self.attack._validate_paper_pl_defence()
                self.attack.config.manipkd_pl_estimator = "mean_surrogate"
                self.attack._validate_paper_pl_defence()
                del self.attack.config.manipkd_pl_estimator

    def test_proxy_attack_optimizes_shared_upload_for_robust_defences(self):
        class DummyRunner:
            current_round = 15

            def __init__(self, uploads):
                self.uploads = uploads

            def get_client_logits_and_predictions(self, mode):
                return self.uploads, None

        self.attack.clients = [DummyClient(index, index >= 21) for index in range(30)]
        uploads = [
            torch.tensor([[0.03 * index, 1.0, -0.5], [0.0, 0.02 * index, 1.0]])
            for index in range(30)
        ]
        stack = torch.stack(uploads)
        self.attack.runner = DummyRunner(uploads)
        self.attack._attack_seed = 123
        self.attack._attack_generators = {}
        self.attack._paper_previous_delta = None
        self.attack.config = SimpleNamespace(
            manipkd_formulation="paper",
            manipkd_knowledge="partial_local",
            manipkd_pl_estimator="mean_surrogate",
            manipkd_history_rounds=5,
            attack_start_round=15,
            manipkd_steps=2,
            manipkd_lr=0.1,
            manipkd_noise_std=0.1,
        )

        aggregates = {
            "Krum": stack[21],
            "TriMean": stack.sort(dim=0).values[9:-9].mean(dim=0),
            "AFA": stack[:25].mean(dim=0),
        }
        for defence, aggregate in aggregates.items():
            with self.subTest(defence=defence):
                self.attack.config.defence = defence
                self.attack._paper_history = []
                self.attack._paper_previous_delta = None
                self.attack._validate_paper_pl_defence()
                for _ in range(5):
                    self.attack.record_aggregated_logits(
                        aggregate, uploads, list(range(30))
                    )
                attacked = self.attack.get_perturbed_client_logits()
                self.assertEqual(len(attacked), 30)
                self.assertTrue(torch.isfinite(attacked[21]).all())
                for index in range(21, 30):
                    self.assertTrue(torch.equal(attacked[index], attacked[21]))
                for index in range(21):
                    self.assertIs(attacked[index], uploads[index])

    def test_robust_defences_aggregate_and_report_malicious_uploads(self):
        clients = [DummyClient(index, index >= 21) for index in range(30)]
        uploads = [
            torch.tensor([[0.03 * index, 1.0, -0.5], [0.0, 0.02 * index, 1.0]])
            for index in range(30)
        ]
        settings = [
            (Krum, SimpleNamespace(krum_f=9, krum_num_selected=1), "[Krum selection audit]"),
            (TriMean, SimpleNamespace(trimean_trim_ratio=0.3), "[TriMean]"),
            (AFA, SimpleNamespace(), "[AFA retention audit]"),
        ]
        for defence_type, config, audit_marker in settings:
            with self.subTest(defence=defence_type.__name__):
                defence = defence_type(clients=clients, config=config, runner_instance=None)
                defence.set_active_client_ids(range(30))
                output = io.StringIO()
                with redirect_stdout(output):
                    aggregate, scores = defence.get_aggregated_logits(uploads)
                self.assertEqual(aggregate.shape, uploads[0].shape)
                self.assertTrue(torch.isfinite(aggregate).all())
                self.assertEqual(len(scores), len(clients))
                self.assertIn(audit_marker, output.getvalue())


if __name__ == "__main__":
    unittest.main()
