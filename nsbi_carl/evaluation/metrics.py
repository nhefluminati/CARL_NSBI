"""Auxiliary evaluation metrics. Each metric is a small standalone class so it
can also be used directly on any dataset outside the pipeline::

    DensityRatioIntegral().evaluate(EvaluationContext(scores=s, labels=y, weights=w))
"""

from __future__ import annotations

import numpy as np

from ..registry import register_metric
from .base import EvaluationContext, Metric


@register_metric("density_ratio_integral")
class DensityRatioIntegral(Metric):
    r"""Checks that the implied p_target integrates to 1.

    From the classifier output s the density ratio is r = s / (1 - s)
    (~ p_target / p_ref). If the ratio is exact, then

        \int p_target dx = \int r(x) p_ref(x) dx
                         ≈ Σ_{i in reference} r_i * w_i / Σ_{i in reference} w_i
                         = 1

    The sum runs over REFERENCE events only, weighted by their normalized
    event weights. Returns the integral and its deviation from 1.
    """

    def __init__(self, ratio_clip: float = 1e13):
        self.ratio_clip = ratio_clip

    def evaluate(self, ctx: EvaluationContext) -> dict[str, float]:
        ref = ctx.labels == 0.0
        if not np.any(ref):
            return {"density_ratio_integral": float("nan"), "density_ratio_integral_deviation": float("nan")}

        r = np.clip(ctx.ratio[ref], 0.0, self.ratio_clip)
        w = ctx.weights[ref].astype(np.float64)
        w_sum = w.sum()
        if w_sum <= 0:
            return {"density_ratio_integral": float("nan"), "density_ratio_integral_deviation": float("nan")}

        integral = float(np.sum(r * (w / w_sum)))
        return {
            "density_ratio_integral": integral,
            "density_ratio_integral_deviation": integral - 1.0,
        }
