"""Exercise the real CPA logit entry point with the comparison aggregators."""
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from byzantine.attacks import CPA
from byzantine.defences import AFA, Krum, TriMean


class CPADefenceIntegrationTests(unittest.TestCase):
    def check_profile(self, n_clients, f):
        generator = torch.Generator().manual_seed(20260805)
        uploads = [torch.randn(8, 10, generator=generator) + 2 for _ in range(n_clients)]
        probabilities = [value.softmax(1) for value in uploads]
        for attacked in (False, True):
            clients = [SimpleNamespace(client_id=i + 1, is_byzantine=attacked and i >= n_clients - f,
                                       trainData=range(20)) for i in range(n_clients)]
            config = SimpleNamespace(
                dataset="cifar10", public_ds="stl10", public_split_mode="legacy",
                strategy="FedDistill", hips=False, attack_start_round=15,
                trimean_trim_ratio=.3, krum_f=f, krum_num_selected=3, krum_fallback="error",
                probe_core_sigma_filter=False, n_byzantine_clients=f if attacked else 0,
                afa_xi=2., afa_delta_xi=.5, afa_prior_alpha=3., afa_prior_beta=3.,
                afa_block_threshold=.95,
            )
            runner = SimpleNamespace(seed=20260805, current_round=14, n_classes=10,
                                     get_client_logits_and_predictions=lambda mode: (uploads, probabilities))
            with redirect_stdout(io.StringIO()):
                attack = CPA(config=config, clients=clients, runner_instance=runner)
                warmup = attack.get_perturbed_client_logits()
                self.assertIs(warmup, uploads)
                runner.current_round = 15
                poisoned = attack.get_perturbed_client_logits()
                for i, client in enumerate(clients):
                    if not client.is_byzantine:
                        self.assertIs(poisoned[i], uploads[i])
                    else:
                        self.assertFalse(torch.equal(poisoned[i], uploads[i]))
                for cls in (TriMean, Krum, AFA):
                    with self.subTest(attacked=attacked, defense=cls.__name__, n=n_clients):
                        defense = cls(config=config, clients=clients, runner_instance=runner)
                        defense.set_active_client_ids(range(1, n_clients + 1))
                        for round_index in range(3):
                            aggregate, scores = defense.get_aggregated_logits(poisoned)
                            self.assertEqual(aggregate.shape, (8, 10))
                            self.assertEqual(scores.shape, (n_clients,))
                            self.assertTrue(torch.isfinite(aggregate).all())
                            target = aggregate.softmax(1)
                            self.assertTrue(torch.isfinite(target).all())
                            torch.testing.assert_close(target.sum(1), torch.ones(8))
                        if cls is AFA:
                            self.assertTrue(any(value > 0 for value in defense.good_counts.values()))

    def test_full_profile_same_security_parameters_for_clean_and_attack(self):
        self.check_profile(n_clients=30, f=9)

    def test_smoke_profile_satisfies_multikrum_requirements(self):
        self.check_profile(n_clients=6, f=1)


if __name__ == "__main__":
    unittest.main()
