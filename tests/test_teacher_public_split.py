import tempfile
import unittest
from pathlib import Path

import torch

from data_splits import build_fixed_stratified_manifest, save_split_manifest
from tools.train_clean_teacher import build_fixed_public_subsets


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor(index), self.targets[index], index


class TeacherPublicSplitTests(unittest.TestCase):
    def test_teacher_uses_only_public_and_validation_indices(self):
        targets = [class_id for class_id in range(3) for _ in range(10)]
        manifest = build_fixed_stratified_manifest(targets, 3, 2, seed=17)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            digest = save_split_manifest(manifest, path)
            train, validation, metadata = build_fixed_public_subsets(
                train_base=DummyDataset(targets),
                validation_base=DummyDataset(targets),
                manifest_path=path,
                public_samples_per_class=3,
                validation_samples_per_class=2,
                split_seed=17,
            )

        self.assertEqual(set(train.source_indices), set(manifest["public_indices"]))
        self.assertEqual(set(validation.source_indices), set(manifest["validation_indices"]))
        self.assertTrue(set(train.source_indices).isdisjoint(validation.source_indices))
        self.assertEqual(metadata["public_samples"], 9)
        self.assertEqual(metadata["validation_samples"], 6)
        self.assertEqual(metadata["manifest_digest"], digest)


if __name__ == "__main__":
    unittest.main()
