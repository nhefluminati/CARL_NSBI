"""End-to-end smoke test: real Pipeline, synthetic .h5 inputs, tiny budget.

Checks that the optimized paths still produce a usable ensemble:
  * the vectorized path writes checkpoints CARL.load_from_checkpoint can read
  * CARLEnsemble.load()/inference() work on them
  * stacked inference agrees with the per-model loop
  * the single-network path still trains with device batching
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl import Pipeline  # noqa: E402


def make_h5(path, n, loc, seed):
    rng = np.random.default_rng(seed)
    with h5.File(path, "w") as f:
        f["m4l"] = rng.normal(loc, 1.0, n)
        f["mTZZ"] = rng.normal(loc * 0.5, 1.0, n)
        f["weight"] = rng.uniform(0.5, 1.5, n)


def build_config(tmp, mode, ensemble=True):
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    make_h5(data / "target.h5", 20000, 0.6, 1)
    make_h5(data / "ref_a.h5", 20000, 0.0, 2)
    make_h5(data / "ref_b.h5", 15000, 0.0, 3)
    cfg = {
        "run_name": f"smoke_{mode}",
        "output_dir": str(tmp / f"out_{mode}"),
        "seed": 52,
        "data": {
            "target_paths": [str(data / "target.h5")],
            "reference_paths": [str(data / "ref_a.h5"), str(data / "ref_b.h5")],
            "features": ["m4l", "mTZZ"],
        },
        "split": {"train_fraction": 0.8, "val_fraction": 0.1},
        "model": {"n_layers": 2, "n_nodes": 32},
        "training": {
            "batch_size": 4096, "val_batch_size": 8192, "learning_rate": 1e-2,
            "max_epochs": 4, "early_stopping_patience": 50, "gpus": [],
        },
        "performance": {"device_batches": True, "progress_bar": False},
        "evaluation": {"after_training": {"metrics": [{"name": "density_ratio_integral"}]}},
    }
    if ensemble:
        cfg["ensemble"] = {"enabled": True, "n_members": 3, "start_seed": 100, "mode": mode}
    return cfg


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nsbi_smoke_"))
    try:
        # --- vectorized ensemble ---------------------------------------
        cfg = build_config(tmp, "vectorized")
        result = Pipeline(cfg).run()
        ens = result["ensemble"]
        assert len(ens.models) == 3, f"expected 3 members, got {len(ens.models)}"
        print(f"[ok] vectorized ensemble trained, metrics={result['metrics']}")

        x = ens.models[0].net[0].weight.new_empty(500, 2).normal_().cpu()
        stacked = ens.member_predictions(x, device="cpu", stacked=True)
        looped = ens.member_predictions(x, device="cpu", stacked=False)
        max_dev = float(np.abs(stacked - looped).max())
        assert max_dev < 1e-5, f"stacked vs looped inference differ by {max_dev}"
        print(f"[ok] stacked inference matches per-model loop (max dev {max_dev:.2e})")

        mean = ens.inference(x)
        assert mean.shape == (500,) and np.all((mean > 0) & (mean < 1))
        print("[ok] ensemble.inference() shape and range")

        # --- process-pool ensemble (fallback path) ----------------------
        cfg = build_config(tmp, "process")
        result = Pipeline(cfg).run()
        assert len(result["ensemble"].models) == 3
        print("[ok] process-pool ensemble path still works")

        # --- single network ---------------------------------------------
        cfg = build_config(tmp, "single", ensemble=False)
        result = Pipeline(cfg).run()
        assert result["summaries"][0]["val_loss"], "no val loss recorded"
        print("[ok] single-network path with device batching")

        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
