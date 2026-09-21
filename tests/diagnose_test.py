"""The diagnostic must separate a good ratio from a normalised-but-wrong one.

The claim being tested: E_ref[r] = 1 can hold while the ratio is badly wrong
where the target lives, and E_target[1/r] catches exactly that case. If the
second identity were no more sensitive than the first, the diagnostic would
be decoration.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.diagnose import ratio_report, weight_report  # noqa: E402


class FakeScorer:
    """Scores from an explicit ratio function, bypassing any network."""

    def __init__(self, r_fn):
        self.r_fn = r_fn

    def score(self, x):
        r = np.clip(self.r_fn(np.asarray(x, dtype=np.float64)), 1e-12, 1e12)
        return r / (1.0 + r)


def main():
    ok = []
    rng = np.random.default_rng(0)
    n = 200_000

    # target ~ N(1, 1), reference ~ N(0, 1); the exact ratio is known
    x_t = rng.normal(1.0, 1.0, (n, 1))
    x_r = rng.normal(0.0, 1.0, (n, 1))
    w_t = np.full(n, 1.0 / n)
    w_r = np.full(n, 1.0 / n)

    def true_r(x):
        z = x[:, 0]
        return np.exp(-0.5 * (z - 1.0) ** 2) / np.exp(-0.5 * z**2)

    # -- 1) the exact ratio satisfies BOTH identities -------------------
    rep = ratio_report(FakeScorer(true_r), x_t, w_t, x_r, w_r)
    assert abs(rep["E_ref[r]"] - 1) < 0.02, rep["E_ref[r]"]
    assert abs(rep["E_target[1/r]"] - 1) < 0.02, rep["E_target[1/r]"]
    ok.append(f"exact ratio: E_ref[r]={rep['E_ref[r]']:.4f}, "
              f"E_target[1/r]={rep['E_target[1/r]']:.4f} — both 1")

    # -- 2) a ratio that is right on the reference, wrong on the target --
    # Distort r only in the upper tail, where the TARGET has weight but the
    # reference has little. Rescale so the reference-side identity is
    # restored -- exactly the situation that makes E_ref[r] look healthy.
    def broken_r(x):
        z = x[:, 0]
        r = true_r(x)
        return np.where(z > 1.5, r * 6.0, r)

    s_r = broken_r(x_r)
    norm = float(np.average(s_r, weights=w_r))          # force E_ref[r] = 1
    rep_b = ratio_report(FakeScorer(lambda x: broken_r(x) / norm), x_t, w_t, x_r, w_r)

    assert abs(rep_b["E_ref[r]"] - 1) < 0.02, (
        f"fixture failed: E_ref[r]={rep_b['E_ref[r]']} should look healthy")
    assert abs(rep_b["E_target[1/r]"] - 1) > 0.05, (
        f"E_target[1/r]={rep_b['E_target[1/r]']} did not flag the distortion")
    ok.append(f"distorted ratio: E_ref[r]={rep_b['E_ref[r]']:.4f} still looks fine, but "
              f"E_target[1/r]={rep_b['E_target[1/r]']:.4f} flags it "
              f"({100*(rep_b['E_target[1/r]']-1):+.1f}%)")

    # -- 3) negative target weights are reported -------------------------
    w_neg = w_t.copy()
    w_neg[rng.random(n) < 0.15] *= -1.0
    wr = weight_report("target", w_neg)
    assert wr["n_negative"] > 0 and wr["frac_weight_negative"] > 0.1
    assert weight_report("target", w_t)["n_negative"] == 0
    ok.append(f"negative target weights reported: {wr['frac_negative']:.1%} of events, "
              f"{wr['frac_weight_negative']:.1%} of |weight|")

    # -- 4) the ratio ceiling flags a target outside the reference -------
    # Target shifted far from the reference: r saturates and a real share of
    # the target piles up against the ceiling.
    x_far = rng.normal(4.0, 1.0, (n, 1))
    rep_f = ratio_report(FakeScorer(true_r), x_far, w_t, x_r, w_r)
    assert rep_f["frac_target_beyond_ref_range"] > 0.20, rep_f
    near = ratio_report(FakeScorer(true_r), x_t, w_t, x_r, w_r)
    assert near["frac_target_beyond_ref_range"] < 0.05, near
    ok.append(f"support check: {rep_f['frac_target_beyond_ref_range']:.1%} of a displaced "
              f"target lies beyond the reference's ratio range, vs "
              f"{near['frac_target_beyond_ref_range']:.2%} for an overlapping one")

    for line in ok:
        print(f"[ok] {line}")
    print("\nALL DIAGNOSTIC CHECKS PASSED")


if __name__ == "__main__":
    main()
