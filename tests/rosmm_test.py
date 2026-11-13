"""Checks on the RoSMM mode (``rosmm.enabled``), Drnevich et al., MLST 6
(2025) 045023, Sec. 3.

Toy: reference N(0, 1); target the signed mixture
    q = C N(0, 1) - (C - 1) N(0, SIGMA2),   C = 2, SIGMA2 = 0.4,
sampled with weights +-1 (algorithm 3 of the paper). Then
    p+ = N(0, 1), p- = N(0, SIGMA2), c = C = 2,
    r(x) = C - (C - 1) N(x; 0, SIGMA2) / N(x; 0, 1)    (-0.5 at x = 0, -> 2 in the tails)
so r+ = 1 and r- is the Gaussian ratio: RoSMM must recover a NEGATIVE ratio
near x = 0 from two all-positive BCE trainings.

What must hold, in the fixed-split and the k-fold mode:
  * c from eq. 16 is ~2; every flavor (rosmm, rosmmc, rosmmr) learns r,
    including its negative region, and integrates to ~1 over the reference;
  * the sub-runs are ordinary runs: own records, same scaler and reference
    fingerprint as the parent;
  * inference.RoSMMScorer reproduces the pipeline's scores for every flavor,
    in-sample and out-of-fold, and load_templates returns it and validates;
  * the analytic pieces (pair combination, eq. 16, the PARE fit of c) are
    exact on known inputs;
  * misconfigurations are refused.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl import Pipeline  # noqa: E402
from nsbi_carl.combine import ratio_from_scores  # noqa: E402
from nsbi_carl.inference import (  # noqa: E402
    RoSMMScorer,
    load_templates,
    read_record,
    template_record,
    validate_templates,
)
from nsbi_carl.rosmm import (  # noqa: E402
    combine_pair_ratios,
    fit_coefficient,
    mixture_coefficient,
    validate_rosmm_config,
)

C, SIGMA2 = 2.0, 0.4
N_TARGET, N_REF = 120_000, 100_000


def npdf(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))


def true_ratio(x):
    return C - (C - 1.0) * npdf(x, SIGMA2) / npdf(x, 1.0)


def make_inputs(data: Path, negative_reference: bool = False, n_target: int = N_TARGET):
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    pos = rng.random(n_target) < C / (2 * C - 1)
    x = np.where(pos, rng.normal(0, 1.0, n_target), rng.normal(0, SIGMA2, n_target))
    with h5.File(data / "target.h5", "w") as f:
        f["m4l"] = x
        f["weight"] = np.where(pos, 1.0, -1.0)
    w_ref = np.ones(N_REF)
    if negative_reference:
        w_ref[::10] = -0.5
    with h5.File(data / "ref.h5", "w") as f:
        f["m4l"] = np.random.default_rng(2).normal(0, 1.0, N_REF)
        f["weight"] = w_ref


def config(tmp: Path, name: str, kfold: bool = False, tune: str = "full", **over) -> dict:
    cfg = {
        "run_name": name,
        "output_dir": str(tmp / f"out_{name}"),
        "seed": 5,
        "data": {
            "target_paths": [str(tmp / "data" / "target.h5")],
            "reference_paths": [str(tmp / "data" / "ref.h5")],
            "features": ["m4l"],
        },
        "split": {"train_fraction": 0.7, "val_fraction": 0.15},
        "model": {"n_layers": 3, "n_nodes": 32},
        "training": {
            "batch_size": 4096, "val_batch_size": 65536, "learning_rate": 3e-3,
            "optimizer": "adam", "max_epochs": 25, "early_stopping_patience": 25, "gpus": [],
        },
        "performance": {"device_batches": True, "progress_bar": False},
        "ensemble": {"enabled": True, "n_members": 2, "start_seed": 100,
                     "mode": "vectorized", "bootstrap": False, "combiner": "mean_ratio"},
        "rosmm": {"enabled": True, "tune": tune, "pare": {"t0": 10, "t1": 1},
                  "tuning": {"learning_rate": 1e-4, "max_epochs": 3, "patience": 2,
                             "log_every_n_epochs": 0}},
        "evaluation": {"after_training": {
            "metrics": [{"name": "density_ratio_integral"}],
            "plots": [{"name": "reweighting", "kwargs": {"feature": "m4l"}}],
        }},
    }
    if kfold:
        cfg["split"].update(k_folds=3, kfold_test_fraction=0.15)
    for k, v in over.items():
        cfg[k] = v
    return cfg


def check_flavors(tag, res, out_dir, name, ok, folds_mode):
    rec = yaml.safe_load(open(out_dir / f"run_config_{name}.yaml"))
    mean = np.array(rec["preprocessing"]["scaler_mean"])
    std = np.array(rec["preprocessing"]["scaler_std"])
    assert abs(res["c_hat"] - C) < 0.06, f"[{tag}] c from eq. 16 = {res['c_hat']}, expected ~{C}"
    assert set(res["models"]) == {"rosmm", "rosmmc", "rosmmr"}, res["models"].keys()

    g = np.linspace(-2.5, 2.5, 201)[:, None]
    win = np.abs(g[:, 0]) < 0.25
    for flavor, model in res["models"].items():
        r = ratio_from_scores(model.scores((g - mean) / std))
        err = np.abs(r - true_ratio(g[:, 0]))
        centre, truth = r[win].mean(), true_ratio(g[win, 0]).mean()
        print(f"[{tag}] {flavor}: c={np.round(model.c[:2], 4)}  r near 0: {centre:+.3f} "
              f"(true {truth:+.3f})  mean |err| {err.mean():.3f}  max {err.max():.3f}")
        assert centre < -0.2 and abs(centre - truth) < 0.1, (
            f"[{tag}] {flavor} missed the negative region ({centre} vs {truth})")
        assert err.mean() < 0.06, f"[{tag}] {flavor} ratio off (mean err {err.mean()})"

        m = res["metrics"][flavor]
        integrals = [m["test"], m["out_of_fold"]] if folds_mode else [m]
        for mm in integrals:
            assert abs(mm["density_ratio_integral_deviation"]) < 0.03, (tag, flavor, mm)
    ok.append(f"{tag}: c ~ {res['c_hat']:.3f}; rosmm/rosmmc/rosmmr learn the signed ratio "
              "(negative region included) and integrate to ~1")

    # sub-runs: ordinary records, same scaler + reference as the parent
    for part in ("pos", "neg"):
        sub = read_record(out_dir / part, f"{name}_{part}")
        assert np.allclose(sub.scaler_mean, mean) and np.allclose(sub.scaler_std, std)
        assert sub.reference_fingerprint == rec["reference_fingerprint"], part
        assert abs(sub.class_balance - 1.0) < 1e-6, (part, sub.class_balance)
    ok.append(f"{tag}: pos/neg sub-runs carry the parent's scaler and reference fingerprint")

    # inference reproduces the pipeline, per flavor
    x_raw = np.random.default_rng(3).normal(0, 1.3, (5000, 1))
    for flavor, model in res["models"].items():
        sc = RoSMMScorer(out_dir, name, flavor=flavor)
        s_inf = sc.score(x_raw)
        s_pipe = model.scores((x_raw - mean) / std)
        d = np.max(np.abs(ratio_from_scores(s_inf) - ratio_from_scores(s_pipe)))
        assert d < 1e-4, f"[{tag}] RoSMMScorer({flavor}) differs from the pipeline by {d}"
        if folds_mode:
            folds = np.random.default_rng(4).integers(-1, 3, len(x_raw))
            d = np.max(np.abs(sc.ratio_out_of_fold(x_raw, folds)
                              - ratio_from_scores(model.scores_out_of_fold(
                                  (x_raw - mean) / std, folds))))
            assert d < 1e-4, f"[{tag}] out-of-fold RoSMMScorer({flavor}) differs by {d}"
    ok.append(f"{tag}: RoSMMScorer reproduces every flavor"
              + (" (in-sample and out-of-fold)" if folds_mode else ""))

    loaded = load_templates({name: out_dir})
    assert isinstance(loaded[name], RoSMMScorer) and loaded[name].flavor == "rosmmr"
    assert validate_templates([template_record(out_dir, name)]) == []
    ok.append(f"{tag}: load_templates returns a RoSMMScorer and validates the template")


def main():
    ok = []

    # -- 1) analytic pieces -------------------------------------------------
    w = np.r_[np.ones(300), -np.ones(100)]
    assert abs(mixture_coefficient(w) - 1.5) < 1e-12           # 300 / 200
    rp, rn = np.array([[1.0, 2.0]]), np.array([[3.0, 0.5]])
    assert np.allclose(combine_pair_ratios(rp, rn, 2.0), [[-1.0, 3.5]])
    assert np.allclose(combine_pair_ratios(np.vstack([rp, rp]), np.vstack([rn, rn]),
                                           np.array([1.0, 2.0])), [[1.0, 2.0], [-1.0, 3.5]])
    # PARE fit of c on exact sub-ratios recovers c
    rng = np.random.default_rng(0)
    xr = rng.normal(0, 1, 200_000)
    pos = rng.random(200_000) < C / (2 * C - 1)
    xt = np.where(pos, rng.normal(0, 1, 200_000), rng.normal(0, SIGMA2, 200_000))
    x = np.r_[xt, xr]
    y = np.r_[np.ones(200_000), np.zeros(200_000)]
    wt = np.r_[np.where(pos, 1.0, -1.0), np.ones(200_000)]
    wt[y == 1] *= wt[y == 0].sum() / wt[y == 1].sum()
    r_neg = npdf(x, SIGMA2) / npdf(x, 1.0)
    fit = fit_coefficient(np.ones_like(x), r_neg, y, wt, c_hat=1.7, t0=10.0, t1=1.0)
    assert abs(fit["c_fit"] - C) < 0.05, fit["c_fit"]
    assert fit["loss_at_c_fit"] <= fit["loss_at_c_hat"]
    ok.append(f"eq. 16, pair combination exact; PARE fit recovers c = {fit['c_fit']:.3f} (true 2)")

    tmp = Path(tempfile.mkdtemp(prefix="nsbi_rosmm_"))
    try:
        make_inputs(tmp / "data")

        # -- 2) refusals --------------------------------------------------
        for bad, why in (
            ({"ensemble": {"enabled": False}}, "no ensemble"),
            ({"rosmm": {"enabled": True, "tune": "everything"}}, "unknown tune"),
            ({"rosmm": {"enabled": True, "pare": {"t0": 1, "t1": 10}}}, "|t0| < |t1|"),
            ({"ensemble": {"enabled": True, "combiner": "mean_logit"}}, "mean_logit"),
        ):
            try:
                Pipeline(config(tmp, "bad", **bad))
            except ValueError:
                continue
            raise AssertionError(f"accepted a rosmm config with {why}")
        assert validate_rosmm_config(config(tmp, "x"))["tune"] == "full"
        cfg = config(tmp, "few")
        cfg["rosmm"]["min_negative_events"] = 10**9
        try:
            Pipeline(cfg).run()
            raise AssertionError("ran with too few negative events")
        except ValueError:
            pass
        neg_ref = tmp / "negref"
        make_inputs(neg_ref / "data", negative_reference=True, n_target=2000)
        cfg = config(neg_ref, "negref")
        cfg["data"]["reference_unit_weights"] = False
        try:
            Pipeline(cfg).run()
            raise AssertionError("ran with a signed reference")
        except NotImplementedError:
            pass
        ok.append("refused: no ensemble, bad tune / t, mean_logit, too few negatives, "
                  "signed reference")

        # -- 3) fixed split ---------------------------------------------------
        res = Pipeline(config(tmp, "fixed")).run()
        check_flavors("fixed", res, tmp / "out_fixed", "fixed", ok, folds_mode=False)

        # -- 4) k-fold --------------------------------------------------------
        res = Pipeline(config(tmp, "kf", kfold=True)).run()
        check_flavors("kfold", res, tmp / "out_kf", "kf", ok, folds_mode=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    for line in ok:
        print(f"[ok] {line}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
