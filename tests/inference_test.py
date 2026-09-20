"""Checks on nsbi_carl.inference, the analysis-facing loader.

Trains two small templates against a shared reference, then verifies that the
loader reproduces the training-time scoring, rebuilds the reference weights
from the record, and — the part that matters — detects the two silent failure
modes a multi-template fit cannot otherwise see: templates that trained
against different references, and templates trained with unbalanced classes.
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
from nsbi_carl.inference import (  # noqa: E402
    EnsembleScorer,
    load_templates,
    read_record,
    reference_weights,
    validate_templates,
)

FEATURES = ["m4l", "mTZZ"]


def make_h5(path, n, loc, seed, negative=False):
    rng = np.random.default_rng(seed)
    with h5.File(path, "w") as f:
        f["m4l"] = rng.normal(loc, 1.0, n)
        f["mTZZ"] = rng.normal(loc * 0.5, 1.0, n)
        w = rng.uniform(0.5, 1.5, n)
        if negative:
            w[rng.random(n) < 0.2] *= -1.0
        f["weight"] = w


def config(tmp, name, target, refs, **data):
    cfg = {
        "run_name": name,
        "output_dir": str(tmp / name),
        "seed": 52,
        "data": {
            "target_paths": [str(tmp / "data" / target)],
            "reference_paths": [str(tmp / "data" / r) for r in refs],
            "features": FEATURES,
            **data,
        },
        "split": {"train_fraction": 0.8, "val_fraction": 0.1},
        "model": {"n_layers": 2, "n_nodes": 16},
        "training": {"batch_size": 4096, "val_batch_size": 8192, "learning_rate": 1e-2,
                     "max_epochs": 3, "early_stopping_patience": 50, "gpus": []},
        "performance": {"device_batches": True, "progress_bar": False},
        "ensemble": {"enabled": True, "n_members": 3, "start_seed": 100},
    }
    return cfg


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_inf_"))
    ok = []
    try:
        (tmp / "data").mkdir()
        make_h5(tmp / "data" / "S.h5", 6000, 0.7, 1)
        make_h5(tmp / "data" / "B.h5", 6000, 0.2, 2)
        make_h5(tmp / "data" / "ref_a.h5", 7000, 0.0, 3, negative=True)
        make_h5(tmp / "data" / "ref_b.h5", 5000, 0.0, 4)
        refs = ["ref_a.h5", "ref_b.h5"]

        for name, tgt in (("S", "S.h5"), ("B", "B.h5")):
            Pipeline(config(tmp, name, tgt, refs, absolute_weights=True)).run()

        # -- 1) the record round-trips ------------------------------------
        rec = read_record(tmp / "S", "S")
        assert rec.features == FEATURES
        assert len(rec.checkpoints) == 3, rec.checkpoints
        assert rec.absolute_weights is True
        assert abs(rec.class_balance - 1.0) < 1e-9, rec.class_balance
        assert rec.reference_fingerprint.get("train_sha1")
        ok.append(f"record parsed: {len(rec.checkpoints)} members, balance "
                  f"{rec.class_balance:.6f}, fingerprint present")

        # -- 2) stacked scoring == the per-member mean --------------------
        scorer = EnsembleScorer(rec, device="cpu")
        x = np.random.default_rng(0).normal(0, 1, (500, 2))
        got = scorer.score(x)
        xs = torch.as_tensor(scorer.scale(x), dtype=torch.float32)
        want = np.mean([m(xs).flatten().detach().numpy() for m in scorer.models], axis=0)
        dev = float(np.abs(got - want).max())
        assert dev < 1e-5, f"stacked scoring deviates by {dev}"
        assert scorer.member_scores(x).shape == (3, 500)
        ok.append(f"stacked scoring matches the per-member mean (max dev {dev:.2e})")

        # scaler actually applied, and a wrong column count is caught
        assert not np.allclose(scorer.scale(x), x), "scaler was not applied"
        try:
            scorer.score(np.zeros((10, 5)))
            raise AssertionError("wrong feature count accepted")
        except ValueError as e:
            assert "features" in str(e)
        ok.append("scaler applied from the record; wrong feature count rejected")

        # -- 3) reference weights rebuilt from the record ------------------
        raw = [np.asarray(h5.File(tmp / "data" / r)["weight"][:]) for r in refs]
        w = reference_weights(rec, raw)
        assert abs(w.sum() - 1.0) < 1e-12
        assert (w >= 0).all(), "negative reference weights survived"
        # this record has reference_unit_weights=True -> every event equal
        # within a sample, each sample carrying half the total
        for i, r in enumerate(refs):
            share = w[sum(len(a) for a in raw[:i]): sum(len(a) for a in raw[:i + 1])].sum()
            assert abs(share - 0.5) < 1e-12, f"{r} carries {share}, not half"
        ok.append("reference weights rebuilt from the record (samples equalized, w>=0)")

        # -- 4) the two templates pass validation together ------------------
        recs = [read_record(tmp / n, n) for n in ("S", "B")]
        assert validate_templates(recs) == [], "clean templates reported problems"
        ok.append("two templates against one reference validate clean")

        # -- 5) a DIFFERENT reference is detected --------------------------
        Pipeline(config(tmp, "odd", "S.h5", ["ref_a.h5"], absolute_weights=True)).run()
        bad = validate_templates(recs + [read_record(tmp / "odd", "odd")])
        assert any("same reference" in p for p in bad), bad
        ok.append("a template trained against a different reference is caught")

        # -- 6) an unbalanced template is detected -------------------------
        cfg = config(tmp, "unbal", "S.h5", refs, absolute_weights=True,
                     target_balance_factor=1e6)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            Pipeline(cfg).run()
        bad = validate_templates([read_record(tmp / "unbal", "unbal")])
        assert any("not 1" in p for p in bad), bad
        ok.append("a template trained with unbalanced classes is caught")

        # -- 7) load_templates convenience --------------------------------
        loaded = load_templates({"S": tmp / "S", "B": tmp / "B"}, validate=True)
        assert set(loaded) == {"S", "B"} and loaded["S"].n_members == 3
        ok.append("load_templates returns ready scorers for the whole set")

        for line in ok:
            print(f"[ok] {line}")
        print("\nALL INFERENCE CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
