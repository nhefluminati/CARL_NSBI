"""Pipeline step: NSBI weight rescaling.

Two stages, executed in this order (the order matters):

1. **Reference equalization.** Every reference sample is rescaled so that its
   total event weight is identical (normalized to 1). All reference samples
   therefore contribute the same effective statistics to the training.

2. **Target/reference balancing.** The target weights are multiplied by
   ``w_reference_train.sum() / w_target_train.sum()`` where the sums run over
   the TRAIN split only. The resulting scalar is then applied to *all* target
   events — train, validation and test — so the validation/test sets use the
   numerically identical reweighting as the training set.

All scale factors are returned so the pipeline can persist them in the run
record YAML.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import NSBIDataset, SplitIndices


class ReweightStep:
    def __init__(self, reference_unit_weights: bool = True):
        # When true, every reference weight is overwritten with 1 before the
        # samples are equalized — an easy way to build a synthetic reference
        # distribution from physics processes with the desired domain.
        #
        # NOTE: while this is on, `data.absolute_weights` has no observable
        # effect, because the reference weights are discarded here anyway.
        # Turn it off to train against the reference sample's own weights.
        self.reference_unit_weights = bool(reference_unit_weights)

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

        dataset.w = torch.as_tensor(w, dtype=torch.float64).reshape(-1, 1)
        return {
            "reference_sample_scales": reference_scales,
            "reference_unit_weights": self.reference_unit_weights,
            "target_balance_scale": target_scale,
            "train_target_yield": float(target_train_sum * target_scale),
            "train_reference_yield": float(reference_train_sum),
        }
