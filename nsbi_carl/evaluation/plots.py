"""Auxiliary plots. Each plot is its own class and only needs an
:class:`EvaluationContext`, so all of them can be invoked on-the-fly on any
dataset/prediction pair, during training, after training, and for ensembles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..registry import register_plot
from .base import EvaluationContext, Plot


@register_plot("loss_curves")
class LossCurvePlot(Plot):
    """Train/validation loss vs epoch. With several histories (ensemble) every
    member is drawn with reduced opacity."""

    def plot(self, ctx: EvaluationContext) -> Path:
        path = ctx.out_path("loss_curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        many = len(ctx.histories) > 1
        for hist in ctx.histories:
            label_suffix = f" m{hist.get('member', '')}" if many else ""
            ax.plot(hist.get("train_loss", []), alpha=0.5 if many else 1.0, label=f"train{label_suffix}")
            ax.plot(hist.get("val_loss", []), linestyle="--", alpha=0.5 if many else 1.0, label=f"val{label_suffix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted BCE loss")
        ax.set_title("Training / validation loss")
        ax.grid(alpha=0.3)
        if len(ctx.histories) <= 8:
            ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path


def effective_n(sum_w: np.ndarray, sum_w2: np.ndarray) -> np.ndarray:
    """Effective entry count of a weighted sample, ``(Σw)² / Σw²``.

    For unit weights this is just the number of entries; for spread-out
    weights it is smaller, which is the whole point — a bin holding one
    event of weight 100 carries the statistical power of one event, not 100.
    """
    out = np.zeros_like(sum_w, dtype=np.float64)
    good = sum_w2 > 0
    out[good] = sum_w[good] ** 2 / sum_w2[good]
    return out


def wilson_interval(p: np.ndarray, n_eff: np.ndarray, z: float = 1.0):
    """Wilson score interval for a (weighted) binomial fraction.

    Returned as ``(err_low, err_high)`` relative to ``p``, so it drops
    straight into ``ax.errorbar(yerr=...)``.

    The normal approximation ``sqrt(p(1-p)/n)`` is used almost everywhere for
    this, but it collapses to *zero width* at p = 0 and p = 1 — exactly the
    bins a calibration curve lives or dies on, since a well-trained
    classifier piles events up at both ends. Wilson stays finite there and is
    asymmetric, which is the honest shape for a fraction bounded in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    n = np.asarray(n_eff, dtype=np.float64)
    lo = np.full_like(p, np.nan)
    hi = np.full_like(p, np.nan)

    good = n > 0
    pn, nn = p[good], n[good]
    denom = 1.0 + z**2 / nn
    center = (pn + z**2 / (2.0 * nn)) / denom
    half = (z / denom) * np.sqrt(pn * (1.0 - pn) / nn + z**2 / (4.0 * nn**2))
    lo[good] = np.clip(center - half, 0.0, 1.0)
    hi[good] = np.clip(center + half, 0.0, 1.0)

    # The Wilson interval always contains the estimate, so these are >= 0 up
    # to rounding; clip the rounding away rather than hand matplotlib a
    # negative error bar.
    return np.clip(p - lo, 0.0, None), np.clip(hi - p, 0.0, None)


@register_plot("calibration_curve")
class CalibrationCurvePlot(Plot):
    """Weighted fraction of target events per score bin vs the score itself.
    A perfectly calibrated classifier lies on the diagonal.

    The y error bars are the statistical uncertainty on that fraction, from
    the effective number of entries in the bin. Without them the curve
    invites over-reading of the sparsely populated bins, which for a
    classifier are usually the interesting ones at either end.
    """

    def __init__(self, n_bins: int = 30, error_method: str = "wilson", z: float = 1.0):
        self.n_bins = n_bins
        self.error_method = str(error_method).lower()
        self.z = float(z)

    def plot(self, ctx: EvaluationContext) -> Path:
        path = ctx.out_path("calibration_curve")
        s = np.clip(ctx.scores, 0.0, 1.0)
        y, w = ctx.labels, ctx.weights.astype(np.float64)

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        idx = np.clip(np.digitize(s, bins) - 1, 0, self.n_bins - 1)

        num = np.zeros(self.n_bins)
        den = np.zeros(self.n_bins)
        den_w2 = np.zeros(self.n_bins)
        np.add.at(num, idx[y == 1.0], w[y == 1.0])
        np.add.at(den, idx, w)
        np.add.at(den_w2, idx, w**2)

        valid = den > 0
        frac = np.full(self.n_bins, np.nan)
        frac[valid] = num[valid] / den[valid]

        n_eff = effective_n(den, den_w2)
        if self.error_method == "normal":
            # Plain weighted-binomial sigma. Kept for comparison; note it is
            # identically zero wherever the fraction saturates at 0 or 1.
            err = np.full(self.n_bins, np.nan)
            good = n_eff > 0
            err[good] = np.sqrt(np.clip(frac[good] * (1.0 - frac[good]), 0.0, None) / n_eff[good])
            err_lo, err_hi = err, err
        else:
            err_lo, err_hi = wilson_interval(frac, n_eff, z=self.z)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        ax.errorbar(
            centers[valid], frac[valid],
            yerr=np.vstack([err_lo[valid], err_hi[valid]]),
            fmt="o-", ms=4, lw=1.2, capsize=2, elinewidth=1.0, label="classifier",
        )
        ax.set_xlabel(r"Predicted score $\hat{s}$")
        ax.set_ylabel("Weighted target fraction")
        ax.set_title(
            f"Calibration curve ({'Wilson' if self.error_method != 'normal' else 'binomial'} "
            f"{self.z:g}$\\sigma$)"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path


@register_plot("reweighting")
class ReweightingPlot(Plot):
    """1D reweighting closure: histogram of a feature for the target sample,
    the reference sample, and the reference sample reweighted by the density
    ratio r = s/(1-s). If the ratio is exact, reference*r matches the target.
    """

    def __init__(self, feature: str | None = None, n_bins: int = 40, log_y: bool = False):
        self.feature = feature
        self.n_bins = n_bins
        self.log_y = log_y

    def plot(self, ctx: EvaluationContext) -> Path:
        if not ctx.feature_names:
            raise ValueError("ReweightingPlot needs features + feature_names in the context.")
        feature = self.feature or ctx.feature_names[0]
        f_idx = ctx.feature_names.index(feature)

        x = ctx.features_raw()[:, f_idx]
        y, w, r = ctx.labels, ctx.weights.astype(np.float64), ctx.ratio

        finite = np.isfinite(x) & np.isfinite(w) & np.isfinite(r)
        x, y, w, r = x[finite], y[finite], w[finite], r[finite]

        tgt, ref = y == 1.0, y == 0.0
        lo, hi = np.quantile(x, [0.001, 0.999])
        bins = np.linspace(lo, hi, self.n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        h_target, _ = np.histogram(x[tgt], bins=bins, weights=w[tgt])
        h_ref, _ = np.histogram(x[ref], bins=bins, weights=w[ref])
        h_reweighted, _ = np.histogram(x[ref], bins=bins, weights=w[ref] * r[ref])

        # Σw² per bin: the variance of a weighted histogram, and the only
        # honest input to the ratio uncertainty. Note the reweighted
        # histogram's variance uses (w·r)², not w² — a reference event pulled
        # up by a large ratio carries a correspondingly large uncertainty,
        # which is precisely where closure plots tend to mislead.
        h_target_w2, _ = np.histogram(x[tgt], bins=bins, weights=w[tgt] ** 2)
        h_reweighted_w2, _ = np.histogram(x[ref], bins=bins, weights=(w[ref] * r[ref]) ** 2)

        path = ctx.out_path(f"reweighting_{feature}")
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax.step(centers, h_ref, where="mid", label="reference")
        ax.step(centers, h_target, where="mid", label="target")
        ax.step(centers, h_reweighted, where="mid", linestyle="--", label=r"reference $\times\; r$")
        ax.set_ylabel("Weighted events")
        ax.set_title(f"Density-ratio reweighting closure: {feature}")
        if self.log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Ratio and its uncertainty. Numerator and denominator come from
        # DISJOINT event samples (reference vs target), so they are
        # independent and the relative errors add in quadrature:
        #     (dR/R)^2 = (dN/N)^2 + (dD/D)^2,   dX = sqrt(Sum w^2) of that histogram.
        with np.errstate(divide="ignore", invalid="ignore"):
            good = (h_target > 0) & (h_reweighted > 0)
            ratio = np.where(h_target > 0, h_reweighted / h_target, np.nan)
            rel_num = np.where(good, np.sqrt(h_reweighted_w2) / h_reweighted, np.nan)
            rel_den = np.where(good, np.sqrt(h_target_w2) / h_target, np.nan)
            ratio_err = np.where(good, np.abs(ratio) * np.sqrt(rel_num**2 + rel_den**2), np.nan)

        axr.axhline(1.0, linestyle="--", color="grey")
        axr.errorbar(centers, ratio, yerr=ratio_err, fmt="o", ms=3, capsize=2, elinewidth=1.0)
        axr.set_ylim(0.5, 1.5)
        axr.set_xlabel(feature)
        axr.set_ylabel("reweighted / target")
        axr.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
