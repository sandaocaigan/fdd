import unittest
from types import SimpleNamespace

import torch

from byzantine.attacks import ManipulatingKD


class DummyClient:
    def __init__(self, client_id, is_byzantine):
        self.client_id = client_id
        self.is_byzantine = is_byzantine


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


if __name__ == "__main__":
    unittest.main()
