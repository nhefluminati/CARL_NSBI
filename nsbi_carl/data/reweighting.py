"""Pipeline step: NSBI weight rescaling.

Three stages, executed in this order (the order matters):

1. **Reference equalization.** Every reference sample is rescaled so that its
   total event weight is identical (normalized to 1). All reference samples
   therefore contribute the same effective statistics to the training.

2. **Target/reference balancing.** The target weights are multiplied by
   ``w_reference_train.sum() / w_target_train.sum()`` where the sums run over
   the TRAIN split only. The resulting scalar is then applied to *all* target
   events — train, validation and test — so the validation/test sets use the
   numerically identical reweighting as the training set.

3. **Global normalization** (optional, ``normalize_weights``). One scalar
   applied to every event of both classes, chosen so the mean weight over the
   combined train split is exactly 1. Because it multiplies target and
   reference alike it cancels out of the likelihood ratio, so it cannot
   affect the NSBI result — see the note on ``normalize_weights`` below for
   what it does and does not change in practice.

This whole step runs *after* the reference cache is written, so none of these
scale factors are baked into the persisted reference sample: a cache built by
one run can be reloaded by another whose target — and therefore whose
balancing scale — is completely different.

All scale factors are returned so the pipeline can persist them in the run
record YAML.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import NSBIDataset, SplitIndices


class ReweightStep:
    def __init__(self, reference_unit_weights: bool = True, normalize_weights: bool = False):
        # When true, every reference weight is overwritten with 1 before the
        # samples are equalized — an easy way to build a synthetic reference
        # distribution from physics processes with the desired domain.
        #
        # NOTE: while this is on, `data.absolute_weights` has no observable
        # effect, because the reference weights are discarded here anyway.
        # Turn it off to train against the reference sample's own weights.
        self.reference_unit_weights = bool(reference_unit_weights)

        # Scale every weight of both classes by one scalar, so that the mean
        # weight over the combined train split is 1.
        #
        # This is safe for NSBI — a common factor cancels from the likelihood
        # ratio — but be aware that it is very nearly a no-op for the
        # optimisation too, because the weighted BCE this package trains with
        # is already scale invariant:
        #
        #     loss = (bce * w).sum() / w.sum()
        #
        # Multiplying every w by c multiplies numerator and denominator alike,
        # so it cancels analytically. It does NOT cancel exactly in finite
        # precision: those sums are accumulated in float32 in the training
        # loop, and measured end to end the loss and gradients move by a few
        # float32 ULP (~2e-7 relative — tests/weights_test.py pins this).
        #
        # So the gain is conditioning, not a different optimum. With the
        # yields this pipeline produces by default (the target balance carries
        # a 1e+6 factor) a large batch drives the running float32 sums to
        # ~1e5-1e6, where the spacing between representable values starts to
        # matter relative to an individual term; normalising keeps both sums
        # near the batch size. If you were hoping this would change how
        # training behaves, the batch size and learning rate are the knobs
        # that will — this one mostly buys numerical headroom and weights
        # that are readable in a log.
        self.normalize_weights = bool(normalize_weights)

    def apply(self, dataset: NSBIDataset, splits: SplitIndices) -> dict:
        w = dataset.w.numpy().reshape(-1).astype(np.float64).copy()
        y = dataset.y.numpy().reshape(-1)

        # -- 1) equalize reference samples (before target/ref balancing) ---
        reference_scales: dict[str, float] = {}
        for sid, name in enumerate(dataset.sample_names):
            mask = (dataset.sample_id == sid) & (y == 0.0)
            if not mask.any():
                continue  # target sample
            if self.reference_unit_weights:
                w[mask] = 1
            total = w[mask].sum()
            if total <= 0:
                raise ValueError(
                    f"Reference sample '{name}' has non-positive total weight "
                    f"({total:.6g}). If this sample has negative MC weights, set "
                    "data.absolute_weights: true."
                )
            scale = 1.0 / total
            w[mask] *= scale
            reference_scales[name] = float(scale)

        # -- 2) balance target vs reference on the TRAIN split -------------
        train = splits.train
        y_train = y[train]
        target_train_sum = w[train][y_train == 1.0].sum()
        reference_train_sum = w[train][y_train == 0.0].sum()
        if target_train_sum <= 0 or reference_train_sum <= 0:
            raise ValueError("Train split must contain positive target and reference yields.")

        target_scale = float(reference_train_sum * 1e+6 / target_train_sum)
        w[y == 1.0] *= target_scale  # identical value applied to train/val/test

        # -- 3) global normalization (both classes, one scalar) ------------
        # Fixed on the TRAIN split and applied to train/val/test alike, for
        # the same reason as stage 2: val/test must be weighted exactly as
        # train is, or their losses are not comparable.
        normalization_scale = 1.0
        if self.normalize_weights:
            combined_train_sum = w[train].sum()
            if combined_train_sum <= 0:
                raise ValueError(
                    f"Combined train weight sum is non-positive ({combined_train_sum:.6g}); "
                    "cannot normalize to a mean weight of 1."
                )
            normalization_scale = float(len(train) / combined_train_sum)
            w *= normalization_scale

        dataset.w = torch.as_tensor(w, dtype=torch.float64).reshape(-1, 1)
        return {
            "reference_sample_scales": reference_scales,
            "reference_unit_weights": self.reference_unit_weights,
            "target_balance_scale": target_scale,
            "normalize_weights": self.normalize_weights,
            "global_normalization_scale": normalization_scale,
            "train_mean_weight": float(w[train].mean()),
            "train_target_yield": float(target_train_sum * target_scale * normalization_scale),
            "train_reference_yield": float(reference_train_sum * normalization_scale),
        }
