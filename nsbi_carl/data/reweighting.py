"""Pipeline step: NSBI weight rescaling.

Three stages, executed in this order (the order matters):

1. **Reference equalization.** Every reference sample is rescaled so that its
   total event weight is identical (normalized to 1). All reference samples
   therefore contribute the same effective statistics to the training.

2. **Target/reference balancing.** The target weights are multiplied by
   ``target_balance_factor * w_reference_train.sum() / w_target_train.sum()``
   where the sums run over the TRAIN split only. The resulting scalar is then
   applied to *all* target events — train, validation and test — so the
   validation/test sets use the numerically identical reweighting as the
   training set.

   ``target_balance_factor`` must be 1 for CARL to estimate the right thing:
   it is the ratio of the total target weight to the total reference weight
   after balancing, and the classifier's optimum is
   ``s = w_t p_t / (w_t p_t + w_r p_r)``. Only at 1 does
   ``r = s/(1-s)`` equal the density ratio ``p_t/p_r``. At any other value the
   network learns ``factor * p_t/p_r``, and a large factor also drives the
   loss to ~0 by saturating the output, because the reference then carries a
   negligible share of the loss. The step warns when the configured value
   would leave the classes more than 100x apart.

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

import warnings

import numpy as np
import torch

from .dataset import NSBIDataset, SplitIndices


def weight_summary(weights, labels) -> dict:
    """Per-class weight statistics for a set of events.

    Reported separately for target and reference because the two means are
    *not* the same number, and confusing them is the easy mistake here:
    ``normalize_weights`` fixes the mean over the COMBINED set to 1, while the
    per-class means come out as ``N_combined / (2 * N_class)`` once the classes
    carry equal total weight. With one target sample against three reference
    samples, for instance, the target mean is ~2 and the reference mean ~0.67,
    and both are correct.

    The number to check against 1 is ``combined_mean``. The number to check
    for CARL's validity is ``balance`` (total target weight over total
    reference weight), which must be 1.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    t, r = w[y == 1.0], w[y == 0.0]
    out = {
        "n": int(w.size),
        "combined_mean": float(w.mean()) if w.size else float("nan"),
        "n_target": int(t.size),
        "n_reference": int(r.size),
        "target_mean": float(t.mean()) if t.size else float("nan"),
        "reference_mean": float(r.mean()) if r.size else float("nan"),
        "target_sum": float(t.sum()),
        "reference_sum": float(r.sum()),
        "target_min": float(t.min()) if t.size else float("nan"),
        "target_max": float(t.max()) if t.size else float("nan"),
        "n_negative": int((w < 0).sum()),
    }
    out["balance"] = (
        out["target_sum"] / out["reference_sum"] if out["reference_sum"] else float("nan")
    )
    return out


def log_weight_summary(weights, labels, tag: str = "", split: str = "train",
                       strict_balance: bool = True) -> dict:
    """Print :func:`weight_summary` as a few aligned lines.

    ``strict_balance`` flags a class balance away from 1. Use it for the
    TRAIN split, where the balance is fixed by construction and any
    deviation is a bug. The val/test splits are drawn per sample, so their
    balance only holds to sampling accuracy and a small deviation there is
    expected -- they are used for model selection, not for the ratio.
    """
    s = weight_summary(weights, labels)
    p = f"[{tag}] " if tag else ""
    print(
        f"{p}{split} weights: combined mean = {s['combined_mean']:.6f}  "
        f"(this is what normalize_weights sets to 1)",
        flush=True,
    )
    print(
        f"{p}  target    n={s['n_target']:>9,d}  mean={s['target_mean']:.6g}  "
        f"sum={s['target_sum']:.6g}  range=[{s['target_min']:.3g}, {s['target_max']:.3g}]",
        flush=True,
    )
    print(
        f"{p}  reference n={s['n_reference']:>9,d}  mean={s['reference_mean']:.6g}  "
        f"sum={s['reference_sum']:.6g}",
        flush=True,
    )
    off = abs(s["balance"] - 1.0)
    if strict_balance and off > 1e-6:
        note = "   <-- MUST be 1 for r = s/(1-s)"
    elif not strict_balance and off > 0.05:
        note = "   (split fluctuation; large enough to be worth a look)"
    else:
        note = ""
    print(f"{p}  target/reference weight ratio = {s['balance']:.6g}{note}", flush=True)
    if s["n_negative"]:
        print(
            f"{p}  WARNING: {s['n_negative']:,} negative weights in this split; "
            "a weighted BCE is not bounded below with those.",
            flush=True,
        )
    return s


class ReweightStep:
    def __init__(
        self,
        reference_unit_weights: bool = True,
        normalize_weights: bool = False,
        target_balance_factor: float = 1.0,
    ):
        # Ratio of total target weight to total reference weight after
        # balancing. 1.0 is the only value for which r = s/(1-s) is the
        # density ratio; see the module docstring.
        if target_balance_factor <= 0:
            raise ValueError("target_balance_factor must be positive.")
        self.target_balance_factor = float(target_balance_factor)
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
        # So the gain is conditioning, not a different optimum: it keeps the
        # running float32 sums near the batch size instead of wherever the
        # sample yields happen to put them, and makes the weights readable in
        # a log. If you were hoping this would change how training behaves,
        # the batch size and learning rate are the knobs that will.
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

        target_scale = float(
            reference_train_sum * self.target_balance_factor / target_train_sum
        )
        w[y == 1.0] *= target_scale  # identical value applied to train/val/test

        if not 0.01 <= self.target_balance_factor <= 100.0:
            warnings.warn(
                f"target_balance_factor={self.target_balance_factor:g} leaves the target and "
                "reference classes far apart in total weight. The classifier optimum is "
                "s = w_t.p_t / (w_t.p_t + w_r.p_r), so r = s/(1-s) will estimate "
                f"{self.target_balance_factor:g} * p_t/p_r, not the density ratio, and the "
                "loss will fall towards 0 as the output saturates. Use 1.0 unless you are "
                "deliberately studying this.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            "target_balance_factor": self.target_balance_factor,
            "normalize_weights": self.normalize_weights,
            "global_normalization_scale": normalization_scale,
            "train_mean_weight": float(w[train].mean()),
            "train_target_yield": float(target_train_sum * target_scale * normalization_scale),
            "train_reference_yield": float(reference_train_sum * normalization_scale),
        }
