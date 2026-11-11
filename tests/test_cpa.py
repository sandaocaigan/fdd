"""Regression tests for CPA's data selection and real logit-upload path."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from byzantine.attacks import CPA


class CPATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.matrix_path = str(Path(self.temp.name) / "cov.pt")
        self.matrix = torch.tensor([[1., -.4, -.1], [-.4, 1., -.2], [-.1, -.2, 1.]])
        torch.save(self.matrix, self.matrix_path)
        self.uploads = [
            torch.tensor([[8., 2., 1.], [1., 7., 2.], [0., 1., 9.], [6., 1., 0.]]),
            torch.tensor([[5., 1., 0.], [2., 8., 0.], [0., 2., 7.], [7., 0., 1.]]),
            torch.tensor([[3., 2., 0.], [2., 4., 1.], [1., 2., 5.], [4., 1., 2.]]),
        ]
        self.clients = [SimpleNamespace(is_byzantine=value) for value in (False, False, True)]
        self.calls = 0

        def collect(mode):
            self.assertEqual(mode, "train")
            self.calls += 1
            return self.uploads, [value.softmax(1) for value in self.uploads]

        self.runner = SimpleNamespace(
            seed=20260805, n_classes=3, current_round=15,
            get_client_logits_and_predictions=collect,
            get_client_predictions=lambda mode: collect(mode)[1],
        )

    def make_attack(self, **overrides):
        values = dict(
            dataset="cifar10", public_ds="stl10", public_ds_fraction=None,
            public_split_mode="legacy", strategy="FedDistill", cpa_cov_path=self.matrix_path,
            cpa_logit_epsilon=1e-12, sample_attack_frac=None, hips=False, attack_start_round=15,
        )
        values.update(overrides)
        with redirect_stdout(io.StringIO()):
            return CPA(config=SimpleNamespace(**values), clients=self.clients, runner_instance=self.runner)

    def test_plain_targets_and_real_logit_path_preserve_honest_inputs(self):
        attack = self.make_attack()
        originals = [value.clone() for value in self.uploads]
        result = attack.get_perturbed_client_logits()
        self.assertEqual(self.calls, 1)
        self.assertIs(result[0], self.uploads[0])
        self.assertIs(result[1], self.uploads[1])
        torch.testing.assert_close(result[2].argmax(1), torch.tensor([1, 0, 1, 1]))
        self.assertTrue(torch.isfinite(result[2]).all())
        self.assertAlmostEqual(float(result[2].min()), -27.631021, places=4)
        for original, value in zip(originals, self.uploads):
            torch.testing.assert_close(original, value, rtol=0, atol=0)

    def test_warmup_and_zero_fraction_leave_exact_uploads_unchanged(self):
        attack = self.make_attack()
        self.runner.current_round = 14
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)
        self.runner.current_round = 15
        attack = self.make_attack(sample_attack_frac=0.)
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)

    def test_partial_attack_preserves_unselected_samples_and_training_rng(self):
        attack = self.make_attack(sample_attack_frac=.5)
        rng_before = torch.random.get_rng_state().clone()
        first = attack.get_perturbed_client_logits()
        changed = (first[2] != self.uploads[2]).any(1)
        self.assertEqual(int(changed.sum()), 2)
        torch.testing.assert_close(first[2][~changed], self.uploads[2][~changed], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        second = self.make_attack(sample_attack_frac=.5).get_perturbed_client_logits()
        torch.testing.assert_close(first[2], second[2])

    def test_hips_selects_an_honest_prediction_per_sample(self):
        attack = self.make_attack(hips=True)
        result = attack.get_perturbed_client_logits()
        honest = torch.stack([value.softmax(1) for value in self.uploads[:2]])
        actual = result[2].softmax(1)
        belongs_to_honest = (actual.unsqueeze(0) - honest).abs().amax(2).lt(1e-6).any(0)
        self.assertTrue(belongs_to_honest.all())
        honest_logits = torch.stack(self.uploads[:2])
        raw_vertex = (result[2].unsqueeze(0) == honest_logits).all(2).any(0)
        self.assertTrue(raw_vertex.all())
        self.assertIs(result[0], self.uploads[0])

    def test_legacy_probabilities_and_logits_agree_on_attack_predictions(self):
        attack = self.make_attack()
        probabilities = attack.get_perturbed_client_predictions()
        logits = attack.get_perturbed_client_logits()
        torch.testing.assert_close(logits[2].softmax(1), probabilities[2], atol=1e-10, rtol=0)

    def test_no_malicious_is_identity_and_all_malicious_is_rejected(self):
        attack = self.make_attack()
        for client in self.clients:
            client.is_byzantine = False
        self.assertIs(attack.get_perturbed_client_logits(), self.uploads)
        for client in self.clients:
            client.is_byzantine = True
        with self.assertRaisesRegex(ValueError, "honest client"):
            attack.get_perturbed_client_logits()

    def test_explicit_and_implicit_stl10_resolve_from_module_not_cwd(self):
        self.runner.n_classes = 10
        previous = os.getcwd()
        try:
            os.chdir(self.temp.name)
            for public_ds in ("stl10", None):
                attack = self.make_attack(cpa_cov_path=None, public_ds=public_ds)
                self.assertEqual(attack.cpa_tensor.shape, (10, 10))
        finally:
            os.chdir(previous)

    def test_in_domain_split_does_not_silently_use_stl10_matrix(self):
        for options in (dict(public_split_mode="fixed_stratified", public_ds=None),
                        dict(public_ds_fraction=.1)):
            with self.subTest(options=options), self.assertRaisesRegex(FileNotFoundError, "cifar10-cov.pt"):
                self.make_attack(cpa_cov_path=None, **options)

    def test_bad_matrices_and_missing_file_fail_early(self):
        for matrix in (torch.eye(2), torch.ones(3, 3, dtype=torch.int64),
                       torch.full((3, 3), float("nan")),
                       torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 0., 1.]])):
            torch.save(matrix, self.matrix_path)
            with self.subTest(matrix=matrix), self.assertRaisesRegex(ValueError, "covariance"):
                self.make_attack()
        with self.assertRaises(FileNotFoundError):
            self.make_attack(cpa_cov_path=str(Path(self.temp.name) / "missing.pt"))

    def test_invalid_configuration_and_alignment(self):
        for epsilon in (0, -1, 1, float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                self.make_attack(cpa_logit_epsilon=epsilon)
        for fraction in (-.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "sample_attack_frac"):
                self.make_attack(sample_attack_frac=fraction)
        with self.assertRaisesRegex(ValueError, "HIPS"):
            self.make_attack(hips=True, sample_attack_frac=.5)
        with self.assertRaisesRegex(ValueError, "FedDistill"):
            self.make_attack(strategy="FedAVG")
        attack = self.make_attack()
        self.uploads.pop()
        with self.assertRaisesRegex(ValueError, "aligned"):
            attack.get_perturbed_client_logits()


if __name__ == "__main__":
    unittest.main()
