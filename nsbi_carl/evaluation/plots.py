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


@register_plot("calibration_curve")
class CalibrationCurvePlot(Plot):
    """Weighted fraction of target events per score bin vs the score itself.
    A perfectly calibrated classifier lies on the diagonal."""

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins

    def plot(self, ctx: EvaluationContext) -> Path:
        path = ctx.out_path("calibration_curve")
        s = np.clip(ctx.scores, 0.0, 1.0)
        y, w = ctx.labels, ctx.weights.astype(np.float64)

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        idx = np.clip(np.digitize(s, bins) - 1, 0, self.n_bins - 1)

        num = np.zeros(self.n_bins)
        den = np.zeros(self.n_bins)
        np.add.at(num, idx[y == 1.0], w[y == 1.0])
        np.add.at(den, idx, w)
        valid = den > 0
        frac = np.full(self.n_bins, np.nan)
        frac[valid] = num[valid] / den[valid]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        ax.plot(centers[valid], frac[valid], "o-", ms=4, label="classifier")
        ax.set_xlabel(r"Predicted score $\hat{s}$")
        ax.set_ylabel("Weighted target fraction")
        ax.set_title("Calibration curve")
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

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_target > 0, h_reweighted / h_target, np.nan)
        axr.axhline(1.0, linestyle="--", color="grey")
        axr.plot(centers, ratio, "o", ms=3)
        axr.set_ylim(0.5, 1.5)
        axr.set_xlabel(feature)
        axr.set_ylabel("reweighted / target")
        axr.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
