import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from byzantine.attacks import PCFDLA
from byzantine.defences import AFA, Krum, TriMean


class PCFDLADefenceTests(unittest.TestCase):
    def setUp(self):
        self.clients = [
            SimpleNamespace(client_id=i + 1, is_byzantine=i >= 21, trainData=range(5))
            for i in range(30)
        ]
        self.clean = [
            torch.tensor([[2.0 + i * 0.001, 0.2, -0.3], [-0.2, 1.8 + i * 0.001, 0.3]])
            for i in range(30)
        ]
        self.attacked = [
            PCFDLA.poison_logits(x, 10.) if client.is_byzantine else x
            for x, client in zip(self.clean, self.clients)
        ]

    def make_defence(self, cls, **settings):
        defence = cls(clients=self.clients, config=SimpleNamespace(**settings), runner_instance=None)
        defence.set_active_client_ids(range(1, 31))
        return defence

    def test_trimean_trims_nine_per_tail_for_clean_and_attacked_uploads(self):
        defence = self.make_defence(TriMean, trimean_trim_ratio=0.3)
        for uploads in (self.clean, self.attacked):
            with redirect_stdout(io.StringIO()) as output:
                aggregate, scores = defence.get_aggregated_logits(uploads)
            self.assertIn("Trimmed 9 lowest and 9 highest", output.getvalue())
            self.assertEqual(scores.shape, (30,))
            # Nine peak-controlled extremes should not survive this example.
            honest = torch.stack(self.clean[:21])
            self.assertTrue(torch.all(aggregate >= honest.min(dim=0).values - 1e-6))
            # Clean rows also include clients 22-30 with a slightly higher top logit.
            upper = torch.stack(uploads if uploads is self.clean else self.clean[:21]).max(dim=0).values
            self.assertTrue(torch.all(aggregate <= upper + 1e-6))

    def test_multikrum_uses_same_f_and_three_selected_for_both_groups(self):
        for uploads in (self.clean, self.attacked):
            defence = self.make_defence(Krum, krum_f=9, krum_num_selected=3, krum_fallback="error")
            with redirect_stdout(io.StringIO()) as output:
                aggregate, scores = defence.get_aggregated_logits(uploads)
            self.assertIn("[Multi-Krum] f=9", output.getvalue())
            self.assertIn("nearest=19", output.getvalue())
            self.assertIn("/3.", output.getvalue())
            if uploads is self.attacked:
                self.assertIn("malicious_selected=0/3", output.getvalue())
            self.assertEqual(aggregate.shape, (2, 3))
            self.assertTrue(torch.isfinite(aggregate).all())
            self.assertEqual(scores.shape, (30,))

    def test_afa_accepts_pcfdla_outputs_and_reports_retention(self):
        for uploads in (self.clean, self.attacked):
            defence = self.make_defence(
                AFA, afa_xi=2., afa_delta_xi=0.5,
                afa_prior_alpha=3., afa_prior_beta=3., afa_block_threshold=0.95,
            )
            with redirect_stdout(io.StringIO()) as output:
                for _ in range(3):
                    aggregate, scores = defence.get_aggregated_logits(uploads)
                    self.assertEqual(aggregate.shape, (2, 3))
                    self.assertTrue(torch.isfinite(aggregate).all())
                    self.assertEqual(scores.shape, (30,))
            self.assertIn("[AFA retention audit]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
