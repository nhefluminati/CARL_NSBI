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
        # the actual reason to switch it on.
        assert sums[0] < sums[1], "normalization did not reduce the float32 accumulator"
        ok.append(f"float32 weight-sum accumulator {sums[1]:.3e} -> {sums[0]:.3e}")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL WEIGHT CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
