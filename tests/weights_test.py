"""Checks on the global weight normalization (``data.normalize_weights``).

What it must do:      mean weight over the combined train split becomes 1,
                      applied to train/val/test with one identical scalar.
What it must NOT do:  leak into the persisted reference cache, change the
                      target/reference balance, or change what the network
                      learns.

The last point is asserted rather than assumed: the weighted BCE used here is
scale invariant, so the normalization is a no-op for the optimisation. The
test pins that, both so the claim is checked and so that a future change to
the loss that breaks the invariance is caught here.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.data.loading import DatasetBuilder, save_reference_cache  # noqa: E402
from nsbi_carl.data.reweighting import ReweightStep  # noqa: E402
from nsbi_carl.data.splitting import SplitStep  # noqa: E402
from nsbi_carl.model import weighted_bce_with_logits  # noqa: E402


def make_h5(path, n, loc, seed, weight_scale=1.0):
    rng = np.random.default_rng(seed)
    with h5.File(path, "w") as f:
        f["m4l"] = rng.normal(loc, 1.0, n)
        f["mTZZ"] = rng.normal(loc * 0.5, 1.0, n)
        f["weight"] = rng.uniform(0.5, 1.5, n) * weight_scale


def build(tmp, **kw):
    return DatasetBuilder(
        target_paths=[str(tmp / "target.h5")],
        reference_paths=[str(tmp / "ref_a.h5"), str(tmp / "ref_b.h5")],
        features=["m4l", "mTZZ"],
        **kw,
    ).build()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_w_"))
    ok = []
    try:
        make_h5(tmp / "target.h5", 8000, 0.6, 1, weight_scale=1e3)
        make_h5(tmp / "ref_a.h5", 9000, 0.0, 2)
        make_h5(tmp / "ref_b.h5", 6000, 0.0, 3)
        split = SplitStep(0.8, 0.1, seed=52)

        # -- 1) mean weight over the combined train split is 1 -------------
        d = build(tmp)
        s = split.split(d)
        info = ReweightStep(normalize_weights=True).apply(d, s)
        w = d.w.numpy().reshape(-1)
        mean = w[s.train].mean()
        assert abs(mean - 1.0) < 1e-10, f"train mean weight is {mean}, not 1"
        assert abs(info["train_mean_weight"] - 1.0) < 1e-10
        ok.append(f"mean weight over the combined train split == 1 (got {mean:.12f})")

        # one scalar, so val/test are weighted on the same footing as train
        plain = build(tmp)
        s2 = split.split(plain)
        ReweightStep(normalize_weights=False).apply(plain, s2)
        wp = plain.w.numpy().reshape(-1)
        ratio = w / wp
        assert np.allclose(ratio, ratio[0], rtol=1e-12), "normalization was not a single scalar"
        assert abs(ratio[0] - info["global_normalization_scale"]) / ratio[0] < 1e-12
        ok.append("one identical scalar applied to every event of both classes")

        # -- 2) the target/reference balance is untouched -------------------
        y = d.y.numpy().reshape(-1)
        for tag, idx in (("train", s.train), ("val", s.val), ("test", s.test)):
            a = w[idx][y[idx] == 1.0].sum() / w[idx][y[idx] == 0.0].sum()
            b = wp[idx][y[idx] == 1.0].sum() / wp[idx][y[idx] == 0.0].sum()
            assert abs(a - b) / b < 1e-12, f"{tag}: target/reference balance changed"
        ok.append("target/reference balance unchanged in train, val and test")

        # -- 3) it does not reach the persisted reference cache -------------
        # The cache is written before reweighting, so a run with the flag on
        # must produce a byte-identical file to one with it off.
        c_on, c_off = tmp / "ref_on.h5", tmp / "ref_off.h5"
        a, b = build(tmp), build(tmp)
        sa, sb = split.split(a), split.split(b)
        save_reference_cache(c_on, a, absolute_weights=False)
        save_reference_cache(c_off, b, absolute_weights=False)
        ReweightStep(normalize_weights=True).apply(a, sa)   # after the save
        ReweightStep(normalize_weights=False).apply(b, sb)
        with h5.File(c_on) as f1, h5.File(c_off) as f2:
            assert np.array_equal(f1["weight"][:], f2["weight"][:]), "cache weights differ"
            assert np.array_equal(f1["split_label"][:], f2["split_label"][:])
        ok.append("persisted reference cache is identical with the flag on or off")

        # a cache written by a normalized run reloads unscaled
        reloaded = DatasetBuilder(
            target_paths=[str(tmp / "target.h5")], reference_paths=[],
            features=["m4l", "mTZZ"], load_reference=str(c_on),
        ).build()
        yr = reloaded.y.numpy().reshape(-1)
        with h5.File(c_on) as f1:
            assert np.array_equal(
                reloaded.w.numpy().reshape(-1)[yr == 0.0], f1["weight"][:]
            ), "reloaded reference weights were rescaled"
        ok.append("reloading that cache gives the raw, unnormalized reference weights")

        # -- 4) training is unchanged up to float32 rounding ----------------
        # The weighted BCE is (bce*w).sum()/w.sum(), so a common factor
        # cancels analytically. In float32 it cancels only to a few ULP.
        # Asserting the size of that residual is what makes the claim
        # "this cannot change the physics" checkable rather than asserted.
        eps = float(np.finfo(np.float32).eps)
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))
        x = torch.as_tensor(d.x.numpy()[s.train][:4096])
        yt = torch.as_tensor(y[s.train][:4096], dtype=torch.float32)
        losses, grads, sums = [], [], []
        for weights in (w, wp):
            net.zero_grad()
            wt = torch.as_tensor(weights[s.train][:4096], dtype=torch.float64).float()
            loss = weighted_bce_with_logits(net(x).flatten(), yt, wt)
            loss.backward()
            losses.append(loss.item())
            grads.append(net[0].weight.grad.clone())
            sums.append(wt.sum().item())

        d_loss = abs(losses[0] - losses[1]) / abs(losses[1])
        d_grad = ((grads[0] - grads[1]).abs() / grads[1].abs().clamp_min(1e-30)).max().item()
        assert d_loss < 10 * eps, f"loss moved by {d_loss:.2e}, more than float32 rounding"
        assert d_grad < 10 * eps, f"gradients moved by {d_grad:.2e}, more than float32 rounding"
        ok.append(
            f"loss and gradients agree to float32 rounding "
            f"(loss {d_loss:.1e}, grad {d_grad:.1e}, eps {eps:.1e}) — a no-op for the optimum"
        )

        # ... and the normalized run is the better conditioned one, which is
        # the actual reason to switch it on. The accumulator lands at the
        # batch size regardless of which side it started on: with a large
        # balance factor the raw weights are huge, without one they are tiny
        # (each reference sample sums to 1 over all its events), and either
        # extreme costs float32 precision.
        # Tolerance is loose on purpose: the mean weight is 1 over the whole
        # train split, and this is one 4096-event slice of it, so a sub-percent
        # fluctuation is expected. The claim being tested is that the
        # accumulator sits at the batch size rather than decades away from it.
        n_batch = 4096
        assert abs(sums[0] - n_batch) / n_batch < 0.05, (
            f"normalized accumulator {sums[0]:.3e} is not ~the batch size {n_batch}"
        )
        drift = abs(np.log10(sums[1] / n_batch))
        ok.append(
            f"float32 weight-sum accumulator {sums[1]:.3e} -> {sums[0]:.3e} "
            f"(~batch size; was {drift:.1f} decades off)"
        )

        # -- 5) the classes end up balanced, and r recovers p_t/p_r ---------
        # This is the invariant that matters: CARL's optimum is
        #   s = w_t p_t / (w_t p_t + w_r p_r)
        # so r = s/(1-s) is the density ratio ONLY when the two classes carry
        # equal total weight. A balance factor far from 1 also collapses the
        # loss towards 0, because the lighter class stops contributing.
        d3 = build(tmp)
        s3 = split.split(d3)
        ReweightStep().apply(d3, s3)
        w3 = d3.w.numpy().reshape(-1)
        y3 = d3.y.numpy().reshape(-1)
        T = w3[s3.train][y3[s3.train] == 1.0].sum()
        R = w3[s3.train][y3[s3.train] == 0.0].sum()
        assert abs(T / R - 1.0) < 1e-9, f"classes not balanced: target/reference = {T / R:.3e}"
        ok.append(f"default balance puts the classes on equal total weight (T/R = {T / R:.6f})")

        # Closed-form check on separable 1-D Gaussians: fit the analytic
        # optimum and confirm the recovered ratio is p_t/p_r, not a multiple.
        rng = np.random.default_rng(0)
        n = 200_000
        xt, xr = rng.normal(0.5, 1.0, n), rng.normal(0.0, 1.0, n)
        xs = np.concatenate([xt, xr])
        ys = np.concatenate([np.ones(n), np.zeros(n)])
        for factor, expect in ((1.0, 1.0), (50.0, 50.0)):
            wt = np.where(ys == 1.0, factor, 1.0)
            # optimal s at each x for these known densities
            pt = np.exp(-0.5 * (xs - 0.5) ** 2)
            pr = np.exp(-0.5 * xs**2)
            s_opt = (factor * pt) / (factor * pt + pr)
            r = s_opt / (1.0 - s_opt)
            got = np.median(r / (pt / pr))
            assert abs(got - expect) / expect < 1e-9, f"factor {factor}: r/(p_t/p_r) = {got}"
        ok.append("r = s/(1-s) equals p_t/p_r at factor 1, and factor*p_t/p_r otherwise")

        # an extreme factor must warn rather than fail silently
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            d4 = build(tmp)
            s4 = split.split(d4)
            ReweightStep(target_balance_factor=1e6).apply(d4, s4)
        assert any(issubclass(c.category, RuntimeWarning) for c in caught), "no warning raised"
        ok.append("an extreme target_balance_factor raises a RuntimeWarning")

        # -- 6) bootstrap_fraction must not unbalance the classes ------------
        # The bootstrap resamples the TARGET only, so at fraction f the target
        # carries ~f of its weight against a whole reference. Without the
        # per-member rebalance every member would learn f * p_t/p_r -- the
        # same defect as an unbalanced target_balance_factor, reintroduced
        # once per member.
        from nsbi_carl.training.ensemble import bootstrap_train_indices, rebalance_scale

        d5 = build(tmp)
        s5 = split.split(d5)
        ReweightStep().apply(d5, s5)
        w5 = d5.w.numpy().reshape(-1)
        y5 = d5.y.numpy().reshape(-1)
        for frac in (0.5, 0.8, 1.0):
            idx = bootstrap_train_indices(s5, y5, seed=100, fraction=frac)
            raw = w5[idx]
            lab = y5[idx]
            before = raw[lab == 1.0].sum() / raw[lab == 0.0].sum()
            scale = rebalance_scale(w5, y5, idx)
            after = (raw[lab == 1.0].sum() * scale) / raw[lab == 0.0].sum()
            assert abs(after - 1.0) < 1e-9, f"fraction {frac}: balance {after} after rebalance"
            if frac < 1.0:
                assert abs(before - frac) < 0.05, f"fraction {frac}: expected ~{frac}, got {before}"
        ok.append("target-only bootstrap rebalanced to 1:1 for every bootstrap_fraction")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL WEIGHT CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
