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

        # --- shared reference across two templates -----------------------
        import yaml as _yaml

        cache = tmp / "reference_sample.h5"
        cfg = build_config(tmp, "sbi", ensemble=False)
        cfg["data"]["save_reference"] = str(cache)
        Pipeline(cfg).run()
        assert cache.exists(), "reference cache not written"

        prints = []
        for tag, target in (("tmpl_a", "target.h5"), ("tmpl_b", "target2.h5")):
            make_h5(tmp / "data" / "target2.h5", 9000, 1.3, 77)
            cfg = build_config(tmp, tag, ensemble=False)
            cfg["data"]["target_paths"] = [str(tmp / "data" / target)]
            cfg["data"]["reference_paths"] = []
            cfg["data"]["load_reference"] = str(cache)
            cfg["seed"] = 7 if tag == "tmpl_b" else 52   # deliberately different
            Pipeline(cfg).run()
            record = _yaml.safe_load(open(tmp / f"out_{tag}" / f"run_config_smoke_{tag}.yaml"))
            prints.append(record["reference_fingerprint"])

        assert prints[0] == prints[1], f"templates disagree on the reference: {prints}"
        assert prints[0]["train_n"] > 0
        print(f"[ok] two templates with different targets AND seeds share one reference "
              f"(train sha1 {prints[0]['train_sha1'][:12]}..., {prints[0]['train_n']} events)")

        # --- global weight normalization through the real pipeline --------
        cfg = build_config(tmp, "norm", ensemble=False)
        cfg["data"]["normalize_weights"] = True
        Pipeline(cfg).run()
        record = _yaml.safe_load(open(tmp / "out_norm" / "run_config_smoke_norm.yaml"))
        rw = record["reweighting"]
        assert abs(rw["train_mean_weight"] - 1.0) < 1e-9, rw["train_mean_weight"]
        assert rw["normalize_weights"] is True
        print(f"[ok] normalize_weights through the pipeline "
              f"(scale {rw['global_normalization_scale']:.4g}, train mean weight "
              f"{rw['train_mean_weight']:.10f})")

        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
