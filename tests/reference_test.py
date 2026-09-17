"""Checks on reference-sample identity and handling.

These are the guarantees the analysis depends on: every template network
(S, B, SBI, qq, SBI_EW) and every ensemble member within them must train
against the *same* reference events, or the ratios the likelihood forms
inherit the mismatch.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.data.loading import DatasetBuilder, save_reference_cache  # noqa: E402
from nsbi_carl.data.splitting import SplitStep, reference_fingerprint  # noqa: E402
from nsbi_carl.training.ensemble import bootstrap_train_indices  # noqa: E402


def make_h5(path, n, loc, seed, negative_weights=False):
    rng = np.random.default_rng(seed)
    with h5.File(path, "w") as f:
        f["m4l"] = rng.normal(loc, 1.0, n)
        f["mTZZ"] = rng.normal(loc * 0.5, 1.0, n)
        w = rng.uniform(0.5, 1.5, n)
        if negative_weights:
            w[rng.random(n) < 0.3] *= -1.0
        f["weight"] = w


def builder(tmp, targets, refs, **kw):
    return DatasetBuilder(
        target_paths=[str(tmp / t) for t in targets],
        reference_paths=[str(tmp / r) for r in refs],
        features=["m4l", "mTZZ"],
        **kw,
    )


def ref_events(dataset, splits, which):
    idx = getattr(splits, which)
    y = dataset.y.numpy().reshape(-1)
    x = dataset.x.numpy()[idx[y[idx] == 0.0]]
    return x[np.lexsort(x.T[::-1])]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_ref_"))
    ok = []
    try:
        make_h5(tmp / "target_a.h5", 9000, 0.6, 1)
        make_h5(tmp / "target_b.h5", 4000, 1.2, 2)   # different size AND content
        make_h5(tmp / "ref_a.h5", 12000, 0.0, 3, negative_weights=True)
        make_h5(tmp / "ref_b.h5", 7000, 0.0, 4)
        refs = ["ref_a.h5", "ref_b.h5"]
        split = SplitStep(0.8, 0.1, seed=52)

        # -- 1) reference split is independent of the target --------------
        d1 = builder(tmp, ["target_a.h5"], refs).build()
        s1 = split.split(d1)
        d2 = builder(tmp, ["target_b.h5", "target_a.h5"], refs).build()
        s2 = split.split(d2)

        for which in ("train", "val", "test"):
            a, b = ref_events(d1, s1, which), ref_events(d2, s2, which)
            assert a.shape == b.shape and np.array_equal(a, b), f"reference {which} differs"
        assert reference_fingerprint(d1, s1) == reference_fingerprint(d2, s2)
        ok.append("reference split identical across different target samples (+ fingerprint)")

        # -- 2) bootstrap touches the target only -------------------------
        labels = d1.y.numpy().reshape(-1)
        n_ref = int((labels[s1.train] == 0.0).sum())
        boots = [bootstrap_train_indices(s1, labels, seed, 1.0) for seed in (100, 101, 102)]
        for b in boots:
            assert len(b) == len(s1.train), "bootstrap changed the train size"
            assert np.array_equal(b[-n_ref:], boots[0][-n_ref:]), "members disagree on reference"
            assert set(b[-n_ref:].tolist()) == set(s1.train[labels[s1.train] == 0.0].tolist())
        assert not np.array_equal(boots[0][:-n_ref], boots[1][:-n_ref]), "target was not resampled"
        ok.append("bootstrap resamples the target only; reference identical across members")

        # opt-in reference bootstrap still works
        rb = [bootstrap_train_indices(s1, labels, s, 1.0, True) for s in (100, 101)]
        assert not np.array_equal(rb[0][-n_ref:], rb[1][-n_ref:])
        ok.append("bootstrap_reference: true restores reference resampling")

        # -- 3) absolute weights, reference only ---------------------------
        signed = builder(tmp, ["target_a.h5"], refs).build()
        absed = builder(tmp, ["target_a.h5"], refs, absolute_weights=True).build()
        y = absed.y.numpy().reshape(-1)
        wa = absed.w.numpy().reshape(-1)
        ws = signed.w.numpy().reshape(-1)
        assert (ws[y == 0.0] < 0).any(), "test fixture has no negative reference weights"
        assert (wa[y == 0.0] >= 0).all(), "reference weights not made positive"
        assert np.array_equal(wa[y == 0.0], np.abs(ws[y == 0.0]))
        assert np.array_equal(wa[y == 1.0], ws[y == 1.0]), "target weights were modified"
        ok.append("absolute_weights applies to the reference only, target untouched")

        # -- 4) save / load round trip -------------------------------------
        cache = tmp / "reference_sample.h5"
        info = save_reference_cache(cache, d1, absolute_weights=False)
        assert info["n_events"] == 12000 + 7000

        loaded_b = builder(tmp, ["target_b.h5"], [], load_reference=str(cache))
        dl = loaded_b.build()
        # A deliberately different seed/fractions: the cached split must win.
        sl = SplitStep(0.5, 0.25, seed=999).split(dl)
        for which in ("train", "val", "test"):
            a, b = ref_events(d1, s1, which), ref_events(dl, sl, which)
            assert a.shape == b.shape and np.array_equal(a, b), f"cached reference {which} differs"
        assert reference_fingerprint(dl, sl) == reference_fingerprint(d1, s1)
        ok.append("load_reference reproduces the split exactly, ignoring seed/fraction changes")

        # feature-order mismatch must be caught, not silently permuted
        try:
            DatasetBuilder(
                target_paths=[str(tmp / "target_a.h5")], reference_paths=[],
                features=["mTZZ", "m4l"], load_reference=str(cache),
            ).build()
            raise AssertionError("permuted feature order was accepted")
        except ValueError as e:
            assert "ORDER" in str(e)
        ok.append("load_reference rejects a permuted feature list")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL REFERENCE CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
