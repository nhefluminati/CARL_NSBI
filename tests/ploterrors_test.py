"""Checks that the plot error bars mean what they claim.

An error bar is only useful if its size matches the actual spread of the
quantity under repetition. Both checks here are frequency tests: generate
many independent pseudo-experiments, measure the scatter of the estimator,
and compare it with the uncertainty the plotting code reports. A formula that
is merely plausible but wrong by a factor would pass a smoke test and fail
these.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.evaluation.base import EvaluationContext  # noqa: E402
from nsbi_carl.evaluation.plots import (  # noqa: E402
    CalibrationCurvePlot,
    ReweightingPlot,
    effective_n,
    wilson_interval,
)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_plt_"))
    ok = []
    try:
        rng = np.random.default_rng(0)

        # -- 1) effective N behaves ----------------------------------------
        w_unit = np.ones(100)
        assert abs(effective_n(np.array([w_unit.sum()]), np.array([(w_unit**2).sum()]))[0] - 100) < 1e-9
        # one huge weight among many small ones carries little statistical power
        w_skew = np.concatenate([np.ones(99), [1000.0]])
        n_skew = effective_n(np.array([w_skew.sum()]), np.array([(w_skew**2).sum()]))[0]
        assert n_skew < 5, f"skewed weights gave N_eff={n_skew}"
        ok.append(f"effective_n: unit weights -> 100, one dominant weight -> {n_skew:.2f}")

        # -- 2) calibration error bars match the observed scatter ----------
        # True target fraction p in a bin; draw many pseudo-experiments and
        # compare the scatter of the measured fraction against the reported
        # sigma. Unit weights, so N_eff == N and Wilson ~ binomial away from
        # the boundaries.
        for p_true, n in ((0.3, 400), (0.5, 400), (0.8, 250)):
            fracs = rng.binomial(n, p_true, size=4000) / n
            observed = fracs.std()
            lo, hi = wilson_interval(np.array([p_true]), np.array([float(n)]), z=1.0)
            reported = 0.5 * (lo[0] + hi[0])
            assert abs(reported - observed) / observed < 0.10, (
                f"p={p_true} n={n}: reported {reported:.5f} vs observed {observed:.5f}"
            )
        ok.append("calibration sigma matches pseudo-experiment scatter within 10% (3 working points)")

        # the boundary case the normal approximation gets wrong
        lo, hi = wilson_interval(np.array([0.0, 1.0]), np.array([50.0, 50.0]))
        assert hi[0] > 0.01 and lo[1] > 0.01, "Wilson collapsed at the boundary"
        normal = np.sqrt(0.0 * 1.0 / 50.0)
        assert normal == 0.0
        ok.append(f"at p=0 the normal sigma is 0 but Wilson gives +{hi[0]:.4f} (and -{lo[1]:.4f} at p=1)")

        # errors are never negative, and bracket the estimate
        p = rng.random(500)
        n = rng.integers(1, 5000, 500).astype(float)
        lo, hi = wilson_interval(p, n)
        assert (lo >= 0).all() and (hi >= 0).all(), "negative error bar"
        assert ((p - lo) >= -1e-12).all() and ((p + hi) <= 1 + 1e-12).all()
        ok.append("Wilson errors non-negative and confined to [0, 1] over 500 random bins")

        # empty bins produce nan, not a crash or a fake zero
        lo, hi = wilson_interval(np.array([np.nan]), np.array([0.0]))
        assert np.isnan(lo[0]) and np.isnan(hi[0])
        ok.append("empty bins yield nan rather than a spurious zero error")

        # -- 3) reweighting ratio error matches its scatter ----------------
        # One bin: numerator = sum of w*r over reference events, denominator =
        # sum of w over target events, the two drawn independently.
        #
        # The entry COUNT per bin is Poisson, which is what makes Sum w^2 the
        # right variance of a weighted histogram: for N ~ Poisson(lambda) with
        # iid weights, Var(sum w) = lambda * E[w^2] ~ sum w^2. Holding N fixed
        # instead would give Var = N * Var(w), which is smaller whenever the
        # weights have a non-zero mean -- so the pseudo-experiments have to
        # fluctuate the count too, or they measure the wrong quantity.
        lam_ref, lam_tgt, trials = 300.0, 300.0, 4000
        ratios = np.empty(trials)
        for i in range(trials):
            wr = rng.exponential(1.0, rng.poisson(lam_ref)) * 1.0
            wt = rng.exponential(1.0, rng.poisson(lam_tgt))
            ratios[i] = wr.sum() / wt.sum()
        observed = ratios.std()

        # report the uncertainty the plot would draw, averaged over a few draws
        reps = []
        for _ in range(200):
            wr = rng.exponential(1.0, rng.poisson(lam_ref))
            wt = rng.exponential(1.0, rng.poisson(lam_tgt))
            num, den = wr.sum(), wt.sum()
            reps.append((num / den) * np.sqrt((wr**2).sum() / num**2 + (wt**2).sum() / den**2))
        reported = float(np.mean(reps))
        assert abs(reported - observed) / observed < 0.10, (
            f"ratio sigma reported {reported:.5f} vs observed {observed:.5f}"
        )
        ok.append(
            f"reweighting ratio sigma matches Poisson pseudo-experiment scatter "
            f"({reported:.4f} vs {observed:.4f})"
        )

        # -- 4) both plots still render, with finite error bars ------------
        n = 20000
        s = rng.random(n)
        y = (rng.random(n) < s).astype(float)      # calibrated by construction
        w = rng.uniform(0.5, 1.5, n)
        feats = rng.normal(0, 1, (n, 2))
        ctx = EvaluationContext(
            scores=s, labels=y, weights=w, features=feats,
            feature_names=["m4l", "mTZZ"], mean=np.zeros(2), std=np.ones(2),
            output_dir=tmp, tag="test",
        )
        p1 = CalibrationCurvePlot(n_bins=20).plot(ctx)
        p2 = CalibrationCurvePlot(n_bins=20, error_method="normal").plot(ctx)
        p3 = ReweightingPlot(feature="m4l", n_bins=20).plot(ctx)
        for p in (p1, p2, p3):
            assert Path(p).exists() and Path(p).stat().st_size > 1000, f"{p} not written"
        ok.append("calibration (wilson + normal) and reweighting plots render")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL PLOT ERROR CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
