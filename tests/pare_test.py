"""Checks on the PARE loss option (``model.loss: pare``).

Reference: Drnevich, Jiggins, Katzy, Cranmer, Mach. Learn.: Sci. Technol. 6
(2025) 045023 ("Neural quasiprobabilistic likelihood ratio estimation with
negatively weighted data").

What it must do:
  * decode the raw output with r = t0(1 - t0 s) / (t1(t1 s - 1)), the inverse
    of the paper's s* = (t0 + t1 r)/(t0^2 + t1^2 r), and hand downstream a
    score with s/(1-s) == r, negative r included;
  * learn a ratio that is NEGATIVE where the weighted target density is, from
    +-1 weighted target events, in both training paths (Lightning single
    network and vectorized ensemble);
  * store the loss settings in the checkpoint, so ensemble scoring and
    inference.EnsembleScorer decode exactly as training did.

What it must refuse: half precision, t0 == t1, stacking members trained with
different loss settings.

The toy is 1-D: reference N(0, 1); target the signed mixture
q = c N(0, 1) - (c - 1) N(0, sigma2) with c = 2, sigma2 = 0.4, sampled as in
the paper's algorithm 3 with weights +-1. Then r(x) = c - (c - 1) N(x; 0,
sigma2) / N(x; 0, 1), which is -0.5 at x = 0 and tends to 2 in the tails.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl import Pipeline  # noqa: E402
from nsbi_carl.model import (  # noqa: E402
    CARL,
    common_loss_hparams,
    pare_ratio,
    scores_from_output,
    validate_loss,
    weighted_pare,
)

C, SIGMA2 = 2.0, 0.4
T0, T1 = 25619.0, 58.0      # the paper's (and the default) t; pole at r ~ -1.95e5


def true_ratio(x):
    n = lambda x, s: np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
    return C - (C - 1.0) * n(x, SIGMA2) / n(x, 1.0)


def make_inputs(data, n_target=200_000, n_ref=150_000):
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    pos = rng.random(n_target) < C / (2 * C - 1)
    x = np.where(pos, rng.normal(0, 1.0, n_target), rng.normal(0, SIGMA2, n_target))
    with h5.File(data / "target.h5", "w") as f:
        f["m4l"] = x
        f["weight"] = np.where(pos, 1.0, -1.0)
    with h5.File(data / "ref.h5", "w") as f:
        f["m4l"] = np.random.default_rng(12).normal(0, 1.0, n_ref)
        f["weight"] = np.ones(n_ref)


def config(tmp, name, loss="pare", ensemble=None, precision="32-true", param="ratio"):
    cfg = {
        "run_name": name,
        "output_dir": str(tmp / f"out_{name}"),
        "seed": 52,
        "data": {
            "target_paths": [str(tmp / "data" / "target.h5")],
            "reference_paths": [str(tmp / "data" / "ref.h5")],
            "features": ["m4l"],
        },
        "split": {"train_fraction": 0.8, "val_fraction": 0.1},
        "model": {"n_layers": 3, "n_nodes": 32, "loss": loss,
                  "pare": {"t0": T0, "t1": T1, "param": param}},
        "training": {
            "batch_size": 4096, "val_batch_size": 65536, "learning_rate": 3e-3,
            "optimizer": "adam", "max_epochs": 60, "early_stopping_patience": 60, "gpus": [],
        },
        "performance": {"device_batches": True, "progress_bar": False, "precision": precision},
        "evaluation": {},
    }
    if ensemble:
        cfg["ensemble"] = {"enabled": True, "n_members": 2, "start_seed": 100,
                           "mode": ensemble, "bootstrap": False}
    return cfg


def ratio_on_grid(model: CARL, x: np.ndarray, mean, std) -> np.ndarray:
    xs = torch.as_tensor(((x - mean) / std)[:, None], dtype=torch.float32)
    with torch.no_grad():
        s = model.eval()(xs).double().flatten().numpy()
    return s / (1.0 - s)   # no clipping: negative ratios must survive


def check_ratio(tag, r_hat, x, ok):
    r = true_ratio(x)
    err = np.abs(r_hat - r)
    win = np.abs(x) < 0.25
    centre, true_centre = r_hat[win].mean(), r[win].mean()
    print(f"[{tag}] mean r_hat on |x|<0.25: {centre:+.3f} (true {true_centre:+.3f}); "
          f"max |r_hat - r| on |x|<2.5: {err.max():.3f}, mean {err.mean():.3f}")
    assert centre < -0.15 and abs(centre - true_centre) < 0.12, (
        f"[{tag}] the negative-ratio region was not learned (r ~ {centre} vs {true_centre})"
    )
    assert err.mean() < 0.12 and err.max() < 0.45, f"[{tag}] ratio off: mean {err.mean()}"
    ok.append(f"{tag}: learned the signed ratio (r ~ {centre:+.2f} vs {true_centre:+.2f} near 0, "
              f"mean err {err.mean():.3f})")


def main():
    ok = []

    # -- 1) analytic: optimum, inverse, score convention ------------------
    r = torch.tensor([-50.0, -3.0, -1.5, -0.5, 0.0, 0.3, 1.0, 7.0, 40.0], dtype=torch.float64)
    for t0, t1 in ((10.0, 1.0), (T0, T1), (2.0, 1.0)):
        s_star = (t0 + t1 * r) / (t0**2 + t1**2 * r)
        assert torch.allclose(pare_ratio(s_star, t0, t1), r, rtol=1e-9, atol=1e-9)
        sc = scores_from_output(s_star, "pare", t0, t1)
        assert torch.allclose(sc / (1 - sc), r, rtol=1e-9, atol=1e-9)
        # ratio parametrization: s = (1 + rho u)/t0 is the same estimator
        rho = t1 / t0
        u_star = (t0 * s_star - 1.0) / rho
        assert torch.allclose(u_star, r * (1 - rho) / (1 + rho**2 * r), rtol=1e-6, atol=1e-6)
        assert torch.allclose(pare_ratio(u_star, t0, t1, "ratio"), r, rtol=1e-6, atol=1e-6)
        sc = scores_from_output(u_star, "pare", t0, t1, "ratio")
        assert torch.allclose(sc / (1 - sc), r, rtol=1e-6, atol=1e-6)
    ok.append("decode inverts s* = (t0+t1 r)/(t0^2+t1^2 r) in both parametrizations, r<0 too")

    # the pointwise minimiser of the weighted PARE loss is s*, eq. (9)
    q0, q1 = 0.7, -0.35                       # r = -0.5
    t0, t1 = 10.0, 1.0
    s = torch.linspace(-0.5, 0.5, 200_001, dtype=torch.float64)
    loss = q0 * (1 - s * t0) ** 2 + q1 * (1 - s * t1) ** 2
    s_min = float(s[loss.argmin()])
    expected = (t0 * q0 + t1 * q1) / (t0**2 * q0 + t1**2 * q1)
    assert abs(s_min - expected) < 1e-5, (s_min, expected)
    # the ratio form, through the package's own loss: minimiser u* ~ r
    u = torch.linspace(-3, 3, 600_001, dtype=torch.float64)
    yy, ww = torch.tensor([0.0, 1.0], dtype=torch.float64), torch.tensor([q0, q1], dtype=torch.float64)
    lu = torch.stack([weighted_pare(torch.stack([v, v]), yy, ww, T0, T1, "ratio") for v in u[::100]])
    u_min = float(u[::100][lu.argmin()])
    rho = T1 / T0
    assert abs(u_min - (-0.5) * (1 - rho) / (1 + rho**2 * -0.5)) < 2e-3, u_min
    ok.append(f"loss minimum sits at s* (raw) and u* ~ r (ratio) with a negative target density")

    # -- 2) refusals ------------------------------------------------------
    for bad in (dict(name="pare", t0=3.0, t1=3.0), dict(name="pare", t0=0.0, t1=1.0),
                dict(name="mse")):
        try:
            validate_loss(bad.pop("name"), **bad)
        except ValueError:
            continue
        raise AssertionError(f"validate_loss accepted {bad}")
    a = CARL(n_features=1, n_layers=1, n_nodes=4)
    b = CARL(n_features=1, n_layers=1, n_nodes=4, loss="pare", pare_t0=T0, pare_t1=T1)
    try:
        common_loss_hparams([a, b])
        raise AssertionError("BCE and PARE members were accepted into one stack")
    except ValueError:
        pass
    ok.append("t0 == t1, t = 0, unknown losses and mixed-loss stacks are refused")

    tmp = Path(tempfile.mkdtemp(prefix="nsbi_pare_"))
    try:
        make_inputs(tmp / "data")
        try:
            Pipeline(config(tmp, "bf16", precision="bf16-mixed"))
            raise AssertionError("PARE with bf16 precision was accepted")
        except ValueError:
            pass
        ok.append("half precision is refused for PARE")

        x = np.linspace(-2.5, 2.5, 201)

        # -- 3) single network, Lightning path ----------------------------
        p = Pipeline(config(tmp, "single"))
        res = p.run()
        ckpt = res["summaries"][0]["checkpoint"]
        model = CARL.load_from_checkpoint(ckpt, map_location="cpu")
        assert model.loss_name == "pare" and model.pare_t0 == T0 and model.pare_t1 == T1
        mean, std = p.dataset.mean, p.dataset.std
        check_ratio("lightning", ratio_on_grid(model, x, mean, std), x, ok)
        assert res["summaries"][0]["val_loss"], "no val loss recorded"

        # -- 4) vectorized ensemble, checkpoints, stacked scoring ---------
        p = Pipeline(config(tmp, "vec", ensemble="vectorized"))
        res = p.run()
        ens = res["ensemble"]
        for m in ens.models:
            assert m.hparams["loss"] == "pare", "vectorized checkpoint lost the loss setting"
            check_ratio("vectorized member", ratio_on_grid(m, x, p.dataset.mean, p.dataset.std),
                        x, ok)
        xs = torch.as_tensor(((x - p.dataset.mean) / p.dataset.std)[:, None], dtype=torch.float32)
        stacked = ens.member_predictions(xs, device="cpu", stacked=True)
        looped = ens.member_predictions(xs, device="cpu", stacked=False)
        dev = float(np.abs(stacked - looped).max())
        # float32 noise in the raw output is amplified ~(t0/t1)^2/|1+r| by the decode
        assert dev < 5e-5, f"stacked vs looped PARE scoring differ by {dev}"
        direct = np.stack([m.eval()(xs).flatten().detach().numpy() for m in ens.models])
        assert np.abs(stacked - direct).max() < 5e-5, "ensemble scoring does not decode like CARL"
        ok.append(f"stacked ensemble scoring decodes PARE outputs like CARL.forward ({dev:.1e})")

        from nsbi_carl.inference import EnsembleScorer, read_record

        scorer = EnsembleScorer(read_record(tmp / "out_vec", "vec"))
        ms = scorer.member_scores(x[:, None])
        assert np.abs(ms - direct).max() < 5e-5, "EnsembleScorer does not decode PARE outputs"
        ok.append("inference.EnsembleScorer reads the loss from the checkpoints and decodes it")

        # -- 5) contrast: BCE cannot go negative ---------------------------
        cfg = config(tmp, "bce", loss="bce")
        cfg["training"]["max_epochs"] = 15
        p = Pipeline(cfg)
        res = p.run()
        m = CARL.load_from_checkpoint(res["summaries"][0]["checkpoint"], map_location="cpu")
        r_bce = ratio_on_grid(m, x, p.dataset.mean, p.dataset.std)
        assert r_bce.min() >= 0.0
        ok.append(f"(contrast) BCE on the same data: min r = {r_bce.min():.2e}, cannot go negative")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    for line in ok:
        print(f"[ok] {line}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
