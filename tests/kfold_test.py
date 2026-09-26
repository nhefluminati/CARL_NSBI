"""Checks on k-fold ensemble training (split.k_folds) and out-of-fold scoring.

What must hold for the NSBI fit to be clean:
  * every event is in exactly one fold, the assignment is reproducible from
    (fold_seed, file name, row) and balanced;
  * a member of fold f never trains or validates on fold f, draws WITHOUT
    replacement, and shares its reference rows with every other member of
    fold f -- and of every other template trained on that reference;
  * a test set (fold -1) is kept out of every fold's training, validation,
    reweighting and scaler fit, is the same events for every template, and
    is scored by the whole ensemble;
  * the bootstrap settings are ignored in k-fold mode;
  * the analysis side scores each event only with its own fold's members, and
    recovers the fold of a cached reference event exactly as training did.
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
from nsbi_carl.data.splitting import TEST_FOLD, KFoldStep, fold_assignment  # noqa: E402
from nsbi_carl.inference import (  # noqa: E402
    EnsembleScorer,
    load_reference_sample,
    read_record,
    validate_templates,
)

FEATURES = ["m4l", "mTZZ"]
K = 3
TEST_FRAC = 0.15


def make_h5(path, n, loc, seed):
    rng = np.random.default_rng(seed)
    with h5.File(path, "w") as f:
        f["m4l"] = rng.normal(loc, 1.0, n)
        f["mTZZ"] = rng.normal(loc * 0.5, 1.0, n)
        f["weight"] = rng.uniform(0.5, 1.5, n)


def config(tmp, name, target, mode="vectorized", seed=52, **data):
    return {
        "run_name": name,
        "output_dir": str(tmp / name),
        "seed": seed,                       # differs per template on purpose
        "data": {
            "target_paths": [str(tmp / "data" / target)],
            "features": FEATURES,
            "absolute_weights": True,
            "reference_unit_weights": False,
            **data,
        },
        "split": {"train_fraction": 0.7, "val_fraction": 0.15,
                  "k_folds": K, "fold_seed": 777, "kfold_val_fraction": 0.2,
                  "kfold_test_fraction": TEST_FRAC},
        "model": {"n_layers": 2, "n_nodes": 16},
        "training": {"batch_size": 2048, "val_batch_size": 8192, "learning_rate": 1e-2,
                     "max_epochs": 3, "early_stopping_patience": 50, "gpus": []},
        "performance": {"device_batches": True, "progress_bar": False},
        "ensemble": {"enabled": True, "n_members": 2, "start_seed": 100, "mode": mode,
                     # must be ignored under k-fold:
                     "bootstrap": True, "bootstrap_fraction": 0.5},
        "evaluation": {"after_training": {"metrics": [{"name": "density_ratio_integral"}]}},
    }


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_kfold_"))
    ok = []
    try:
        (tmp / "data").mkdir()
        make_h5(tmp / "data" / "S.h5", 4000, 0.7, 1)
        make_h5(tmp / "data" / "B.h5", 3000, 0.2, 2)
        make_h5(tmp / "data" / "ref_a.h5", 5000, 0.0, 3)
        make_h5(tmp / "data" / "ref_b.h5", 3000, 0.1, 4)
        refs = [str(tmp / "data" / r) for r in ("ref_a.h5", "ref_b.h5")]
        cache = str(tmp / "ref_cache.h5")

        # -- 1) fold assignment -------------------------------------------
        f1 = fold_assignment("x.h5", 1001, K, 5)
        assert np.array_equal(f1, fold_assignment("x.h5", 1001, K, 5)), "not reproducible"
        assert np.bincount(f1).max() - np.bincount(f1).min() <= 1, np.bincount(f1)
        assert not np.array_equal(f1, fold_assignment("y.h5", 1001, K, 5)), "name ignored"
        assert not np.array_equal(f1, fold_assignment("x.h5", 1001, K, 6)), "seed ignored"
        ok.append(f"folds reproducible and balanced ({np.bincount(f1).tolist()})")
        f2 = fold_assignment("x.h5", 1000, K, 5, test_fraction=TEST_FRAC)
        assert (f2 == TEST_FOLD).sum() == 150, (f2 == TEST_FOLD).sum()
        cvc = np.bincount(f2[f2 >= 0])
        assert cvc.max() - cvc.min() <= 1, cvc
        ok.append(f"test set carved out first: {int((f2 == TEST_FOLD).sum())} test, "
                  f"folds {cvc.tolist()}")

        # -- 2) train two templates on one reference, different run seeds --
        # S builds the reference cache; B loads it. S is also a reference
        # sample, as in the real analysis.
        pS = Pipeline(config(tmp, "S", "S.h5", seed=16, reference_paths=refs + [str(tmp / "data" / "S.h5")],
                             save_reference=cache))
        resS = pS.run()
        pB = Pipeline(config(tmp, "B", "B.h5", seed=15, load_reference=cache))
        resB = pB.run()
        assert len(resS["summaries"]) == K * 2, len(resS["summaries"])
        assert sorted({s["fold"] for s in resS["summaries"]}) == list(range(K))
        assert set(resS["metrics"]) == {"test", "out_of_fold"}, resS["metrics"]
        assert resS["metrics"]["test"] and resS["metrics"]["out_of_fold"]
        ok.append(f"pipeline trained {K} folds x 2 members; evaluation ran on the untouched "
                  f"test set and out-of-fold ({sorted(resS['metrics'])})")

        # -- 3) per-member index sets -------------------------------------
        ds, folds = pS.dataset, resS["folds"]
        step = KFoldStep(K, 777, 0.2, TEST_FRAC)
        assert abs((folds == TEST_FOLD).mean() - TEST_FRAC) < 0.01
        # scaler fitted on the cross-validated events only
        raw = ds.x.numpy().astype(np.float64) * ds.std + ds.mean
        assert np.allclose(ds.mean, raw[folds != TEST_FOLD].mean(axis=0), atol=1e-4), \
            "scaler saw the test set"
        y = ds.y.numpy().reshape(-1)
        for f in range(K):
            a = step.member_indices(ds, folds, f, 100)
            b = step.member_indices(ds, folds, f, 101)
            for sp in (a, b):
                assert np.all(folds[sp.train] != f) and np.all(folds[sp.val] != f), "holdout leaked"
                assert np.all(folds[sp.train] != TEST_FOLD), "test set used in training"
                assert np.all(folds[sp.val] != TEST_FOLD), "test set used in validation"
                assert np.array_equal(np.sort(sp.test), np.flatnonzero(folds == f))
                assert len(np.unique(sp.train)) == len(sp.train), "drawn with replacement"
                assert not np.intersect1d(sp.train, sp.val).size, "train/val overlap"
                pool = np.flatnonzero((folds != f) & (folds != TEST_FOLD))
                assert np.array_equal(np.sort(np.concatenate([sp.train, sp.val])), pool), \
                    "pool not fully used"
            ra, rb = a.train[y[a.train] == 0], b.train[y[b.train] == 0]
            assert np.array_equal(ra, rb), "members of a fold disagree on reference rows"
            assert np.array_equal(a.val[y[a.val] == 0], b.val[y[b.val] == 0])
            ta, tb = a.train[y[a.train] == 1], b.train[y[b.train] == 1]
            assert not np.array_equal(np.sort(ta), np.sort(tb)), "target split not redrawn"
            assert a.train.shape == b.train.shape and a.val.shape == b.val.shape
        ok.append("members: holdout and test set excluded, scaler fitted without the test "
                  "set, no replacement, whole pool used, reference rows shared, target split "
                  "redrawn per member")

        # reference events: identical fold in both templates (via the cache)
        def ref_folds(p, res):
            d = p.dataset
            yy = d.y.numpy().reshape(-1)
            x = d.x.numpy()[yy == 0] * d.std + d.mean
            order = np.lexsort(x.T[::-1])
            return x[order], res["folds"][yy == 0][order]
        xs_, fs_ = ref_folds(pS, resS)
        xb_, fb_ = ref_folds(pB, resB)
        assert np.allclose(xs_, xb_, atol=1e-4) and np.array_equal(fs_, fb_), \
            "a reference event sits in different folds for S and B"
        # S.h5 is both the target and a reference sample: one assignment
        s_ids = [i for i, n in enumerate(ds.sample_names) if n == "S.h5"]
        assert len(s_ids) == 2, ds.sample_names
        assert np.array_equal(folds[ds.sample_id == s_ids[0]], folds[ds.sample_id == s_ids[1]]), \
            "S has different folds as target vs reference"
        ok.append("reference folds identical across templates; a target that is also in the "
                  "reference keeps one fold assignment")

        # -- 4) analysis side ---------------------------------------------
        recS, recB = read_record(tmp / "S", "S"), read_record(tmp / "B", "B")
        assert recS.kfold and recS.k_folds == K and recS.fold_seed == 777
        assert recS.kfold_test_fraction == TEST_FRAC
        assert [len(m) for m in recS.fold_members] == [2] * K
        probs = validate_templates([recS, recB])
        assert not any("k-fold" in p for p in probs), probs
        ok.append("record carries the fold scheme; validate_templates accepts matching schemes")

        scorer = EnsembleScorer(recS, device="cpu")
        x, w, rf = load_reference_sample(cache, recS, split="all", return_folds=True)
        assert len(rf) == len(x) and abs(w.sum() - 1) < 1e-12
        with h5.File(tmp / "data" / "ref_a.h5") as fh:
            na = len(fh["weight"])
        assert np.array_equal(rf[:na], recS.folds_for("ref_a.h5", na)), \
            "cached reference folds != folds recomputed from the file"
        assert (rf == TEST_FOLD).any()
        oof = scorer.score_out_of_fold(x, rf)
        # manual: each fold's members only; the test set by ALL members
        xs = torch.as_tensor(scorer.scale(x), dtype=torch.float32)
        want = np.empty(len(x))
        for f, members in enumerate(recS.fold_members):
            sel = rf == f
            want[sel] = np.mean([scorer.models[i](xs[sel]).flatten().detach().numpy()
                                 for i in members], axis=0)
        sel = rf == TEST_FOLD
        want[sel] = np.mean([m(xs[sel]).flatten().detach().numpy() for m in scorer.models], axis=0)
        dev = float(np.abs(oof - want).max())
        assert dev < 1e-5, f"out-of-fold scoring deviates by {dev}"
        pooled = scorer.score(x)
        assert np.abs(pooled - oof).max() > 1e-4, "out-of-fold equals pooled -> folds unused"
        cvm = rf >= 0
        assert scorer.fold_member_scores(x[cvm], rf[cvm]).shape == (2, int(cvm.sum()))
        try:
            scorer.fold_member_scores(x, rf)
            raise AssertionError("fold_member_scores accepted test-set events")
        except ValueError:
            pass
        assert scorer.member_scores(x).shape == (2 * K, len(x))
        ok.append(f"score_out_of_fold uses only each event's own fold, and all members for "
                  f"the test set (max dev {dev:.1e}); pooled score differs")

        xt, wt, ft = load_reference_sample(cache, recS, split="test", return_folds=True)
        assert np.all(ft == TEST_FOLD) and len(xt) == int((rf == TEST_FOLD).sum())
        assert abs(wt.sum() - 1) < 1e-12
        xc, _, fc = load_reference_sample(cache, recS, split="cv", return_folds=True)
        assert np.all(fc >= 0) and len(xc) + len(xt) == len(x)
        try:
            load_reference_sample(cache, recS, split="train")
            raise AssertionError("split='train' accepted for a k-fold record")
        except ValueError:
            pass
        ok.append(f"load_reference_sample(split='test') returns the {len(xt)} untouched "
                  "reference events; 'cv' the rest")

        # -- 5) mismatched fold schemes are flagged -------------------------
        recB.fold_seed = 778
        probs = validate_templates([recS, recB])
        assert any("k-fold scheme" in p for p in probs), probs
        ok.append("a template with a different fold_seed is flagged")

        # -- 6) process mode trains too ------------------------------------
        cfg = config(tmp, "Sp", "S.h5", mode="process", load_reference=cache)
        cfg["ensemble"]["n_members"] = 1
        resP = Pipeline(cfg).run()
        assert sorted(s["fold"] for s in resP["summaries"]) == list(range(K))
        ok.append("process mode trains one member per fold")

        # -- 7) k-fold without an ensemble is rejected ----------------------
        cfg = config(tmp, "X", "S.h5", load_reference=cache)
        cfg["ensemble"]["enabled"] = False
        try:
            Pipeline(cfg)
            raise AssertionError("k-fold without ensemble accepted")
        except ValueError as e:
            assert "ensemble" in str(e)
        ok.append("split.k_folds without ensemble.enabled is rejected")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL K-FOLD CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
