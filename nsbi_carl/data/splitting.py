"""Pipeline step: deterministic train/validation/test split.

The split is drawn *per input sample* with a generator seeded by
``(seed, crc32(sample_name))``. Consequences:

* For a fixed seed, the events pulled from a given reference sample are
  always the same — completely independent of which target samples (or
  which other reference samples) are in the run.
* The split is stratified by construction: every sample contributes the
  same fractions to train/val/test.
"""

from __future__ import annotations

import zlib

import numpy as np

from .dataset import NSBIDataset, SplitIndices


class SplitStep:
    def __init__(self, train_fraction: float = 0.8, val_fraction: float = 0.1, seed: int = 52):
        if not 0.0 < train_fraction < 1.0 or val_fraction < 0.0:
            raise ValueError("Invalid split fractions.")
        if train_fraction + val_fraction > 1.0:
            raise ValueError("train_fraction + val_fraction must be <= 1.")
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.seed = seed

    def split(self, dataset: NSBIDataset) -> SplitIndices:
        train_parts, val_parts, test_parts = [], [], []

        for sid, name in enumerate(dataset.sample_names):
            idx = np.flatnonzero(dataset.sample_id == sid)
            rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
            perm = idx[rng.permutation(len(idx))]

            n_train = int(self.train_fraction * len(idx))
            n_val = int(self.val_fraction * len(idx))
            train_parts.append(perm[:n_train])
            val_parts.append(perm[n_train : n_train + n_val])
            test_parts.append(perm[n_train + n_val :])

        # Shuffle the concatenated splits (order only, membership unchanged).
        rng = np.random.default_rng(self.seed)
        return SplitIndices(
            train=rng.permutation(np.concatenate(train_parts)),
            val=rng.permutation(np.concatenate(val_parts)),
            test=rng.permutation(np.concatenate(test_parts)),
        )
