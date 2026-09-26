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


# ---------------------------------------------------------------------------
# k-fold cross-validation (INT note Sec. 2.7.2)
# ---------------------------------------------------------------------------
# Every event of every sample is put in exactly one of k folds. Fold f's
# ensemble trains on the other k-1 folds and is the ONLY ensemble ever used to
# score fold f's events, so no event is scored by a network that saw it.
#
# The fold of an event depends only on (fold_seed, sample file name, k, row
# index within that sample) -- not on the run seed, the other samples, or
# whether the sample is the target or part of the reference. Two consequences
# the likelihood relies on:
#
#  * a reference event sits in the same fold for EVERY template, so all the
#    ratios combined for it come from networks that never saw it;
#  * a sample that is both a target (S for the S network) and part of the
#    reference gets the identical assignment in both roles.
#
# `fold_seed` must therefore be the same for every template of an analysis;
# validate_templates() checks this.
#
# Before the folds are dealt, a `test_fraction` of every sample is set aside
# as fold TEST_FOLD (-1): a final test set that no member of any fold ever
# trains or validates on. Being keyed the same way, it is the same events for
# every template, so a reference test event is untouched by ALL networks.
DEFAULT_FOLD_SEED = 20240607
TEST_FOLD = -1


def fold_assignment(
    sample_name: str,
    n_events: int,
    k: int,
    fold_seed: int = DEFAULT_FOLD_SEED,
    test_fraction: float = 0.0,
) -> np.ndarray:
    """Fold of every row of one sample: ``TEST_FOLD`` (-1) for the untouched
    test set, else 0..k-1, balanced to +-1 event."""
    if k < 2:
        raise ValueError(f"k-fold needs k >= 2, got {k}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    rng = np.random.default_rng([int(fold_seed), zlib.crc32(sample_name.encode()), int(k)])
    perm = rng.permutation(n_events)
    n_test = int(round(test_fraction * n_events))
    folds = np.empty(n_events, dtype=np.int16)
    folds[perm[:n_test]] = TEST_FOLD
    folds[perm[n_test:]] = np.arange(n_events - n_test) % k
    return folds


class KFoldStep:
    """Assigns folds and builds the per-member index sets of a k-fold ensemble.

    A ``test_fraction`` of every sample is first set aside (fold -1) and
    never enters any fold's pool: it is the final test set of the ensemble.

    Inside the k-1 training folds, each member gets its own train/validation
    split, drawn WITHOUT replacement (``val_fraction`` goes to validation):

    * target events: redrawn for every member (seeded by the member seed), as
      in the note, where each member sees a different 80/20 split;
    * reference events: one split per fold, seeded by ``fold_seed`` only, so
      every member of every template trained on fold f shares byte-identical
      reference train and validation events -- the common-denominator
      guarantee the bootstrap path gives, now per fold.
    """

    def __init__(
        self,
        k: int = 10,
        fold_seed: int = DEFAULT_FOLD_SEED,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
    ):
        if k < 2:
            raise ValueError(f"split.k_folds must be >= 2, got {k}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"split.kfold_val_fraction must be in (0, 1), got {val_fraction}")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError(f"split.kfold_test_fraction must be in [0, 1), got {test_fraction}")
        self.k = int(k)
        self.fold_seed = int(fold_seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

    def assign(self, dataset: NSBIDataset) -> np.ndarray:
        """Per-event fold index for the whole dataset (rows in sample order)."""
        folds = np.empty(len(dataset.sample_id), dtype=np.int16)
        for sid, name in enumerate(dataset.sample_names):
            idx = np.flatnonzero(dataset.sample_id == sid)
            if idx.size:
                folds[idx] = fold_assignment(
                    name, idx.size, self.k, self.fold_seed, self.test_fraction
                )
        return folds

    def member_indices(
        self,
        dataset: NSBIDataset,
        folds: np.ndarray,
        fold: int,
        member_seed: int,
    ) -> SplitIndices:
        """``SplitIndices(train, val, test=holdout)`` for one member of ``fold``.

        The pool is the other k-1 folds; the untouched test set (fold -1) is
        in neither the pool nor the holdout.
        """
        if not 0 <= fold < self.k:
            raise ValueError(f"fold must be in 0..{self.k - 1}, got {fold}")
        y = dataset.y.numpy().reshape(-1)
        train_parts, val_parts = [], []
        for sid, name in enumerate(dataset.sample_names):
            pool = np.flatnonzero(
                (dataset.sample_id == sid) & (folds != fold) & (folds != TEST_FOLD)
            )
            if pool.size == 0:
                continue
            is_ref = bool(y[pool[0]] == 0.0)
            if is_ref:
                rng = np.random.default_rng([self.fold_seed, zlib.crc32(name.encode()), fold, 1])
            else:
                rng = np.random.default_rng([int(member_seed), zlib.crc32(name.encode()), fold, 2])
            perm = pool[rng.permutation(pool.size)]
            n_val = int(round(self.val_fraction * pool.size))
            val_parts.append(perm[:n_val])
            train_parts.append(perm[n_val:])

        # Target rows first, reference rows last and in a fixed order, so the
        # reference block is identical across the members of a fold.
        def _ordered(parts):
            idx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            t, r = idx[y[idx] == 1.0], np.sort(idx[y[idx] == 0.0])
            return np.concatenate([t, r]).astype(np.int64)

        return SplitIndices(
            train=_ordered(train_parts),
            val=_ordered(val_parts),
            test=np.flatnonzero(folds == fold).astype(np.int64),
        )
