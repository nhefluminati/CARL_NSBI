"""Checks on the neural resampler (nsbi_carl/resample.py).

1. ``positive_weights`` with the TRUE moments: no negative weights, and per
   phase-space bin both sum(w) and sum(w^2) are preserved (resample mode) or
   sum(w) is preserved (mean mode).
2. End to end on a toy with 20% negative weights and a known <w|x>: the
   cross-fitted estimate tracks the truth, the branch is written next to the
   input branches, signed and positive histograms agree, and an existing
   branch is not overwritten without --overwrite.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.resample import ResamplerConfig, main, positive_weights  # noqa: E402


def toy(n, seed):
    """w = a(x) with prob 0.8, -1 with prob 0.2; a = 1.5 + 0.5 tanh(x0).
    => <w|x> = 1 + 0.4 tanh(x0),  <w^2|x> = 0.8 a^2 + 0.2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    a = 1.5 + 0.5 * np.tanh(x[:, 0])
    neg = rng.random(n) < 0.2
    w = np.where(neg, -1.0, a)
    m1 = 1.0 + 0.4 * np.tanh(x[:, 0])
    m2 = 0.8 * a ** 2 + 0.2
    return x, w, m1, m2


def binned(x0, w, edges):
    return np.histogram(x0, edges, weights=w)[0], np.histogram(x0, edges, weights=w ** 2)[0]


def check_true_moments(ok):
    x, w, m1, m2 = toy(400_000, 1)
    edges = np.linspace(-2.5, 2.5, 11)
    h, h2 = binned(x[:, 0], w, edges)
    for mode in ("resample", "mean"):
        cfg = ResamplerConfig(mode=mode, preserve_total=False)
        w_pos, d = positive_weights(w, m1, m2, cfg, np.random.default_rng(2))
        assert (w_pos >= 0).all(), mode
        g, g2 = binned(x[:, 0], w_pos, edges)
        pull = (g - h) / np.sqrt(h2 + g2)
        assert np.abs(pull).max() < 4, (mode, pull)
        if mode == "resample":
            # second moment preserved per bin (to ~statistical accuracy)
            assert np.allclose(g2, h2, rtol=0.05), (g2 / h2)
            assert 0 < d["n_dropped"] < len(w)
        else:
            assert d["n_dropped"] == 0
            assert (g2 < h2).all()  # mean mode underestimates the variance
    ok.append("true moments: sum(w) per bin preserved; sum(w^2) preserved in resample mode")


def check_end_to_end(ok, tmp):
    x, w, m1_true, _ = toy(60_000, 3)
    path = tmp / "toy_qq.h5"
    with h5.File(path, "w") as f:
        f["m4l"] = x[:, 0]
        f["mZ1"] = x[:, 1]
        f["weight"] = w
    cfg_path = tmp / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump({
            "run_name": "toy", "output_dir": str(tmp / "out"),
            "data": {"target_paths": [str(path)], "features": ["m4l", "mZ1"], "weight_key": "weight"},
            "resampler": {"k_folds": 3, "n_layers": 2, "n_nodes": 32, "batch_size": 4096,
                          "max_epochs": 200, "patience": 10, "device": "cpu", "plots": True},
        }, f)

    main(["--config", str(cfg_path)])
    with h5.File(path, "r") as f:
        assert "weight_positive" in f and "weight" in f
        w_pos = f["weight_positive"][:]
        assert len(w_pos) == len(w)
    assert (w_pos >= 0).all()
    assert np.isclose(w_pos.sum(), w.sum(), rtol=1e-9)  # preserve_total

    edges = np.linspace(-2.5, 2.5, 11)
    h, h2 = binned(x[:, 0], w, edges)
    g, g2 = binned(x[:, 0], w_pos, edges)
    pull = (g - h) / np.sqrt(h2 + g2)
    assert np.abs(pull).max() < 4, pull

    report = yaml.safe_load(open(tmp / "out" / "resampler" / "resampler_toy_qq.yaml"))
    assert report["n_negative_before"] > 0
    assert (tmp / "out" / "resampler" / "closure_toy_qq.png").exists()
    ok.append(f"end to end: branch written, no negatives, max |pull| {np.abs(pull).max():.2f}")

    try:
        main(["--config", str(cfg_path)])
        raise AssertionError("existing branch was overwritten without --overwrite")
    except FileExistsError:
        ok.append("existing branch protected without --overwrite")


def check_mean_estimate(ok):
    """The cross-fitted regression itself recovers <w|x>."""
    import torch
    from nsbi_carl.resample import cross_fit
    x, w, m1_true, _ = toy(60_000, 5)
    cfg = ResamplerConfig(k_folds=3, n_layers=2, n_nodes=32, batch_size=4096, max_epochs=200, patience=10)
    m1 = cross_fit(x, w, cfg, torch.device("cpu"), seed=0, tag="test")
    err = np.sqrt(np.mean((m1 - m1_true) ** 2))
    assert err < 0.05, err
    ok.append(f"cross-fitted <w|x> rms error vs truth {err:.3f}")


def check_knn(ok):
    """The tiled GPU/CPU k-NN matches brute force."""
    import torch
    from nsbi_carl.resample import knn
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3000, 5)).astype(np.float32)
    q = rng.choice(3000, 200, replace=False)
    d, i = knn(torch.as_tensor(X), q, 9, q_chunk=64, n_chunk=700)
    ref = np.sqrt(((X[q, None, :] - X[None, :, :]) ** 2).sum(-1))
    ref_i = np.argsort(ref, axis=1)[:, :9]
    assert np.allclose(d, np.take_along_axis(ref, ref_i, 1), atol=1e-4)
    assert (i[:, 0] == q).all()                       # the point itself comes first
    ok.append("k-NN matches brute force")


def check_cell(ok, tmp):
    """Cell resampling: no negatives left, total conserved to machine precision,
    closure within (paired) statistics, radius cap leaves seeds unresolved."""
    import torch
    from nsbi_carl.resample import cell_resample
    x, w, _, _ = toy(60_000, 11)
    xs = (x - x.mean(0)) / x.std(0)
    cfg = ResamplerConfig(method="cell", cell_k=8, cell_k_retry=64)
    w_pos, d = cell_resample(x, xs, w, ["m4l", "mZ1"], cfg, torch.device("cpu"), seed=0)
    d.pop("_cells")
    assert (w_pos >= 0).all() and d["n_unresolved"] == 0
    assert abs(w_pos.sum() - w.sum()) < 1e-9 * abs(w).sum()
    assert d["n_retry"] > 0                           # exercised the retry path
    edges = np.linspace(-2.5, 2.5, 21)
    h = np.histogram(x[:, 0], edges, weights=w)[0]
    g = np.histogram(x[:, 0], edges, weights=w_pos)[0]
    v = np.histogram(x[:, 0], edges, weights=(w_pos - w) ** 2)[0]
    chi2 = ((g - h) ** 2 / v).mean()
    assert chi2 < 2.0, chi2
    ok.append(f"cell: no negatives, total exact, paired chi2/ndf in m4l {chi2:.2f}, "
              f"median cell size {d['cell_size']['p50']:.0f}")

    w_cap, dc = cell_resample(x, xs, w, ["m4l", "mZ1"],
                              ResamplerConfig(method="cell", cell_max_radius=0.01), torch.device("cpu"), 0)
    assert dc["n_unresolved"] > 0 and (w_cap < 0).any()
    assert abs(w_cap.sum() - w.sum()) < 1e-9 * abs(w).sum()
    ok.append(f"cell radius cap: {dc['n_unresolved']:,} seeds left unresolved, total still exact")

    # end to end through the CLI, writing the branch
    path = tmp / "toy_cell.h5"
    with h5.File(path, "w") as f:
        f["m4l"] = x[:, 0]
        f["mZ1"] = x[:, 1]
        f["weight"] = w
    cfg_path = tmp / "cfg_cell.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump({"output_dir": str(tmp / "out_cell"),
                        "data": {"target_paths": [str(path)], "features": ["m4l", "mZ1"]},
                        "resampler": {"method": "cell", "device": "cpu",
                                      "cell_feature_scales": {"mZ1": 2.0}}}, f)
    main(["--config", str(cfg_path)])
    with h5.File(path, "r") as f:
        assert (f["weight_positive"][:] >= 0).all()
        assert "cell resampling" in f["weight_positive"].attrs["method"]
    assert (tmp / "out_cell" / "resampler" / "cells_toy_cell.png").exists()
    ok.append("cell end to end: branch written, cell diagnostics plotted")


def main_test():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_rs_"))
    ok = []
    try:
        check_true_moments(ok)
        check_mean_estimate(ok)
        check_end_to_end(ok, tmp)
        check_knn(ok)
        check_cell(ok, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for line in ok:
        print("OK:", line)


if __name__ == "__main__":
    main_test()
