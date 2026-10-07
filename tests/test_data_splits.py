import tempfile
import unittest
from pathlib import Path

import torch

from data_splits import (
    ReindexedSubset,
    build_fixed_stratified_manifest,
    load_and_validate_fixed_split,
    save_split_manifest,
)


class DummyIndexedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.tensor(index), index % 2, index


class FixedDataSplitTests(unittest.TestCase):
    def test_manifest_is_balanced_disjoint_and_reproducible(self):
        targets = [class_id for class_id in range(3) for _ in range(10)]
        first = build_fixed_stratified_manifest(targets, 3, 2, seed=17)
        second = build_fixed_stratified_manifest(targets, 3, 2, seed=17)
        self.assertEqual(first, second)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            saved_digest = save_split_manifest(first, path)
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            public, validation, private, loaded_digest, public_counts, validation_counts = (
                load_and_validate_fixed_split(path, targets, 3, 2, 17)
            )

        self.assertEqual(saved_digest, loaded_digest)
        self.assertEqual(public_counts, [3, 3, 3])
        self.assertEqual(validation_counts, [2, 2, 2])
        self.assertEqual(len(public), 9)
        self.assertEqual(len(validation), 6)
        self.assertEqual(len(private), 15)
        self.assertTrue(set(public).isdisjoint(validation))
        self.assertTrue(set(public).isdisjoint(private))
        self.assertTrue(set(validation).isdisjoint(private))

    def test_reindexed_subset_preserves_source_selection(self):
        subset = ReindexedSubset(DummyIndexedDataset(), [5, 2, 4])
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0][0].item(), 5)
        self.assertEqual(subset[0][2], 0)
        self.assertEqual(subset[1][0].item(), 2)
        self.assertEqual(subset[1][2], 1)


if __name__ == "__main__":
    unittest.main()
