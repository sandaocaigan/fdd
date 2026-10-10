import tempfile
import unittest
from pathlib import Path

import torch

from server_pretraining import (
    load_server_initial_checkpoint,
    validate_server_initialization_config,
)


class ServerPretrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 2)
        self.expected = {
            "dataset": "cifar10",
            "arch": "LinearTestModel",
            "n_classes": 10,
            "manifest_digest": "manifest-sha256",
            "public_samples": 5000,
        }

    def save_checkpoint(self, path, **changes):
        source = torch.nn.Linear(3, 2)
        with torch.no_grad():
            source.weight.fill_(1.25)
            source.bias.fill_(-0.5)
        payload = {"model": source.state_dict(), **self.expected, "best_val_acc": 0.75}
        payload.update(changes)
        torch.save(payload, path)
        return source

    def load(self, path):
        return load_server_initial_checkpoint(
            model=self.model,
            checkpoint_path=path,
            map_location="cpu",
            **self.expected,
        )

    def test_valid_checkpoint_loads_exact_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.pt"
            source = self.save_checkpoint(path)
            metadata = self.load(path)

        for loaded, expected in zip(self.model.parameters(), source.parameters()):
            self.assertTrue(torch.equal(loaded, expected))
        self.assertEqual(metadata["best_val_acc"], 0.75)

    def test_module_prefix_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.pt"
            source = self.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu")
            payload["model"] = {
                f"module.{key}": value for key, value in payload["model"].items()
            }
            torch.save(payload, path)
            self.load(path)

        for loaded, expected in zip(self.model.parameters(), source.parameters()):
            self.assertTrue(torch.equal(loaded, expected))

    def test_missing_file_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            self.load("does-not-exist.pt")

    def test_metadata_mismatches_are_rejected(self):
        cases = {
            "dataset": "cifar100",
            "arch": "OtherModel",
            "n_classes": 100,
            "manifest_digest": "other-manifest",
            "public_samples": 4999,
        }
        for field, value in cases.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "server.pt"
                self.save_checkpoint(path, **{field: value})
                with self.assertRaisesRegex(ValueError, field):
                    self.load(path)

    def test_raw_state_dict_without_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.pt"
            torch.save(self.model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "model.*metadata"):
                self.load(path)

    def test_no_checkpoint_preserves_legacy_configuration(self):
        validate_server_initialization_config(None, "cifar100", "legacy", True)
        validate_server_initialization_config("none", "cifar100", "legacy", True)

    def test_checkpoint_rejects_server_reinitialization(self):
        with self.assertRaisesRegex(ValueError, "reinit_server"):
            validate_server_initialization_config(
                "server.pt", "cifar10", "fixed_stratified", True
            )

    def test_checkpoint_requires_verifiable_fixed_split(self):
        with self.assertRaisesRegex(ValueError, "fixed_stratified"):
            validate_server_initialization_config(
                "server.pt", "cifar10", "legacy", False
            )


if __name__ == "__main__":
    unittest.main()
