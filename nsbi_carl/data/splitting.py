"""Pipeline step: deterministic train/validation/test split.

The split is drawn *per input sample* with a generator seeded by
``(seed, crc32(sample_name))``. Consequences:

* For a fixed seed, the events pulled from a given reference sample are
  always the same — completely independent of which target samples (or
  which other reference samples) are in the run. The draw depends only on
  the sample's own name, its own event count, and the seed; nothing about
  the rest of the run enters it.
* The split is stratified by construction: every sample contributes the
  same fractions to train/val/test.

Events that arrive with a ``split_label`` already set (>= 0) keep it and are
not redrawn. That is how a reference sample restored from a cache file keeps
the exact split it was originally constructed with, even if the seed or the
split fractions in the config have since changed.

``reference_fingerprint`` turns "the reference sample is identical" from an
assumption into something checkable: it hashes the actual reference events in
each split, and the pipeline writes it to the run record. Two runs that agree
on that hash used byte-identical reference training data.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np

from .dataset import NSBIDataset, SplitIndices

TRAIN, VAL, TEST = 0, 1, 2


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
        label = dataset.split_label

        for sid, name in enumerate(dataset.sample_names):
            idx = np.flatnonzero(dataset.sample_id == sid)
            if idx.size == 0:
                continue
            # Respect a pre-assigned split (restored reference cache).
            if np.all(label[idx] >= 0):
                continue
            if np.any(label[idx] >= 0):
                raise ValueError(
                    f"Sample '{name}' is only partially pre-split; a sample must be either "
                    "fully pre-assigned (from a reference cache) or fully unassigned."
                )

            rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
            local = rng.permutation(len(idx))

            n_train = int(self.train_fraction * len(idx))
            n_val = int(self.val_fraction * len(idx))
            label[idx[local[:n_train]]] = TRAIN
            label[idx[local[n_train : n_train + n_val]]] = VAL
            label[idx[local[n_train + n_val :]]] = TEST

        # Shuffle the concatenated splits (order only, membership unchanged).
        rng = np.random.default_rng(self.seed)
        return SplitIndices(
            train=rng.permutation(np.flatnonzero(label == TRAIN)),
            val=rng.permutation(np.flatnonzero(label == VAL)),
            test=rng.permutation(np.flatnonzero(label == TEST)),
        )


def reference_fingerprint(dataset: NSBIDataset, splits: SplitIndices) -> dict[str, str | int]:
    """Hash the reference events of each split, order-independently.

    The hash covers the reference features and the sample each event came
    from, sorted so that it does not depend on the shuffle order or on how
    many target samples were concatenated ahead of them. It deliberately does
    NOT cover the weights, which are rescaled later by ``ReweightStep``
    relative to the target yield.
    """
    y = dataset.y.numpy().reshape(-1)
    x = dataset.x.numpy()
    names = np.array(dataset.sample_names)

    out: dict[str, str | int] = {}
    for key, idx in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        ref = idx[y[idx] == 0.0]
        if ref.size == 0:
            out[f"{key}_sha1"] = ""
            out[f"{key}_n"] = 0
            continue
        tags = names[dataset.sample_id[ref]]
        order = np.lexsort((*x[ref].T[::-1], tags))
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(x[ref][order]).tobytes())
        h.update("\0".join(tags[order]).encode())
        out[f"{key}_sha1"] = h.hexdigest()
        out[f"{key}_n"] = int(ref.size)
    return out
