import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from byzantine.attacks import PCFDLA


class PCFDLATests(unittest.TestCase):
    def make_attack(self, peak=5.0, current_round=15):
        # Different runner-up classes verify each attacker uses its own outputs.
        self.uploads = [
            torch.tensor([[9., 1., 0.], [0., 2., 5.]]),
            torch.tensor([[7., 6., 0.], [2., 4., 0.]]),
            torch.tensor([[0., 8., 3.], [6., 0., 7.]]),
        ]

        def get_uploads(mode):
            self.assertEqual(mode, "train")
            return self.uploads, None

        self.runner = SimpleNamespace(
            seed=20260805, current_round=current_round,
            get_client_logits_and_predictions=get_uploads,
        )
        self.clients = [SimpleNamespace(is_byzantine=x) for x in (False, True, True)]
        return PCFDLA(
            config=SimpleNamespace(
                strategy="FedDistill", pcfdla_peak=peak, attack_start_round=15,
            ), clients=self.clients, runner_instance=self.runner,
        )

    def test_exact_peaks_follow_own_runner_up_and_preserve_inputs(self):
        attack = self.make_attack()
        originals = [x.clone() for x in self.uploads]
        with redirect_stdout(io.StringIO()):
            result = attack.get_perturbed_client_logits()
        self.assertIs(result[0], self.uploads[0])
        torch.testing.assert_close(result[1], torch.tensor([[-5., 5., -5.], [5., -5., -5.]]))
        torch.testing.assert_close(result[2], torch.tensor([[-5., -5., 5.], [5., -5., -5.]]))
        for before, after in zip(originals, self.uploads):
            torch.testing.assert_close(before, after)

    def test_attack_starts_at_configured_round(self):
        attack = self.make_attack(current_round=14)
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)
        self.runner.current_round = 15
        with redirect_stdout(io.StringIO()):
            result = attack.get_perturbed_client_logits()
        self.assertFalse(torch.equal(result[1], self.uploads[1]))

    def test_peak_grid_changes_amplitude_without_changing_targets(self):
        logits = torch.tensor([[9., 0., 3.], [-8., -3., -5.]])
        for peak in (1., 5., 10.):
            poisoned = PCFDLA.poison_logits(logits, peak)
            torch.testing.assert_close(poisoned.argmax(1), torch.tensor([2, 2]))
            self.assertTrue(torch.all(poisoned.abs() == peak))
            self.assertTrue(torch.all((poisoned > 0).sum(1) == 1))

    def test_ties_row_permutation_and_rng_independence(self):
        logits = torch.tensor([[3., 3., 3.], [0., 5., 5.], [8., 7., 0.]])
        rng = torch.random.get_rng_state().clone()
        poisoned = PCFDLA.poison_logits(logits, 5.)
        torch.testing.assert_close(poisoned.argmax(1), torch.tensor([1, 2, 1]))
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(PCFDLA.poison_logits(logits[order], 5.), poisoned[order])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_no_byzantine_clients_leave_all_uploads_unchanged(self):
        attack = self.make_attack()
        for client in self.clients:
            client.is_byzantine = False
        with redirect_stdout(io.StringIO()):
            result = attack.get_perturbed_client_logits()
        for a, b in zip(result, self.uploads):
            self.assertIs(a, b)

    def test_invalid_peaks_and_strategy_are_rejected(self):
        for peak in (0., -1., float("inf"), float("nan")):
            with self.subTest(peak=peak), self.assertRaisesRegex(ValueError, "pcfdla_peak"):
                self.make_attack(peak=peak)
        attack = self.make_attack()
        attack.config.strategy = "FedAVG"
        with self.assertRaisesRegex(ValueError, "FedDistill"):
            PCFDLA(config=attack.config, clients=self.clients, runner_instance=self.runner)

    def test_invalid_shape_dtype_and_alignment_are_rejected(self):
        for tensor in (torch.ones(3), torch.ones(3, 1), torch.ones(2, 3, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                PCFDLA.poison_logits(tensor, 5.)
        with self.assertRaises(ValueError):
            PCFDLA.poison_logits(torch.zeros(2, 3, dtype=torch.float16), 1e6)
        attack = self.make_attack()
        self.uploads.pop()
        with self.assertRaisesRegex(ValueError, "aligned"):
            attack.get_perturbed_client_logits()

    def test_device_dtype_and_empty_samples_are_preserved(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            logits = torch.tensor([[8., 2., 0.]], dtype=torch.float64, device=device)
            result = PCFDLA.poison_logits(logits, 2.)
            self.assertEqual(result.device, logits.device)
            self.assertEqual(result.dtype, logits.dtype)
        empty = PCFDLA.poison_logits(torch.zeros(0, 10), 5.)
        self.assertEqual(empty.shape, (0, 10))


if __name__ == "__main__":
    unittest.main()
