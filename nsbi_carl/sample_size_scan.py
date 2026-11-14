"""Training-sample-size scan: how much does the network gain from more MC?

Trains the configured network (single model or ensemble, exactly as the
config says) several times, each time with only a fraction of the available
training events, and overlays the calibration curves of all runs::

    python -m nsbi_carl.sample_size_scan --config configs/base/S.yaml
    python -m nsbi_carl.sample_size_scan --config configs/base/S.yaml \\
        --fractions 0.1 0.2 0.5 1.0 --out outputs/S_size_scan
    # re-draw the comparison from saved scores, no training:
    python -m nsbi_carl.sample_size_scan --config configs/base/S.yaml --plot-only

What is varied, and what is held fixed
--------------------------------------
Only the amount of training data changes between the runs:

* The split (fixed train/val/test, or the k-fold assignment) is drawn on the
  FULL dataset first, exactly as a normal run would. Only then are events
  dropped, and only from the train/validation part (k-fold: from the
  cross-validated folds). The test set (k-fold: the untouched fold -1) is
  never thinned, so every run is evaluated on the identical test events; the
  script checks this before plotting.
* The thinning is per input sample and per split, so the sample composition
  of the reference and the train/val proportions are unchanged.
* The subsets are nested: a fixed permutation per (sample, split) is drawn
  from ``--subsample-seed`` and the first ``fraction`` of it is kept, so the
  10% events are contained in the 20% ones, and so on. Differences between
  the curves are then due to the added events, not to a different draw.
* Seeds (split, initialisation, bootstrap) are those of the config in every
  run.
* Reweighting and the scaler are refitted per run on that run's training
  events, just as they would be if the smaller sample were all you had.

``--apply-to target|reference`` thins only one class. ``--keep-full-val``
keeps the whole validation split (fixed-split mode only; in k-fold mode the
validation share is drawn per member out of the thinned folds).

Each run's own pipeline output (checkpoints, run record, per-run evaluation
plots) goes to ``<out>/frac_XXX/``. Its test-set scores are saved to
``<out>/frac_XXX/test_scores.npz`` and the comparison plots to ``<out>/``:

* ``calibration_comparison.pdf``: overlaid calibration curves (Wilson
  error bars, same binning and weights as the ``calibration_curve`` plot of
  the pipeline) with a residual panel, observed - predicted.
* ``score_distributions.pdf``: the test-set score histograms per class.
* ``sample_size_scan.json``: per-run event counts and the weighted
  expected calibration error (ECE) per fraction.

The run never writes a reference cache (``data.save_reference`` is ignored),
so a cache the templates share is never overwritten by a thinned reference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

DEFAULT_FRACTIONS = (0.1, 0.2, 0.5, 1.0)
_SUBSAMPLE_STREAM = 0x5A3  # keeps these draws independent of the split's own rng


# ---------------------------------------------------------------------------
# Row thinning
# ---------------------------------------------------------------------------
def thin_mask(
    sample_id: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_names: list[str],
    fraction: float,
    seed: int,
    thin_groups: tuple[int, ...],
    apply_to: str = "both",
) -> np.ndarray:
    """Boolean keep-mask over the rows.

    Rows whose ``groups`` value is in ``thin_groups`` are thinned to
    ``fraction`` per (sample, group); every other row is kept. Nested in
    ``fraction``: the permutation depends only on (seed, sample name, group)
    and the group's size, never on the fraction.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    keep = np.ones(len(sample_id), dtype=bool)
    if fraction == 1.0:
        return keep
    for sid, name in enumerate(sample_names):
        in_sample = sample_id == sid
        if not in_sample.any():
            continue
        is_ref = bool(labels[in_sample][0] == 0.0)
        if (apply_to == "target" and is_ref) or (apply_to == "reference" and not is_ref):
            continue
        for g in thin_groups:
            rows = np.flatnonzero(in_sample & (groups == g))
            if rows.size == 0:
                continue
            rng = np.random.default_rng(
                [int(seed), zlib.crc32(name.encode()), int(g) + 16, _SUBSAMPLE_STREAM]
            )
            perm = rows[rng.permutation(rows.size)]
            n_keep = max(1, int(round(fraction * rows.size)))
            keep[perm[n_keep:]] = False
    return keep


def _take_rows(dataset, keep: np.ndarray) -> np.ndarray:
    """Reduce ``dataset`` in place to the kept rows. Returns old->new index map
    (-1 for dropped rows)."""
    import torch

    kept = np.flatnonzero(keep)
    sel = torch.as_tensor(kept, dtype=torch.long)
    dataset.x = dataset.x.index_select(0, sel)
    dataset.y = dataset.y.index_select(0, sel)
    dataset.w = dataset.w.index_select(0, sel)
    dataset.sample_id = dataset.sample_id[kept]
    dataset.split_label = dataset.split_label[kept]
    remap = np.full(len(keep), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.size, dtype=np.int64)
    return remap


def _remap(idx: np.ndarray, remap: np.ndarray) -> np.ndarray:
    new = remap[np.asarray(idx, dtype=np.int64)]
    return new[new >= 0]  # keeps the (shuffled) order of the survivors


def _counts(dataset, rows: np.ndarray) -> dict:
    y = dataset.y.numpy().reshape(-1)[rows]
    return {"target": int((y == 1.0).sum()), "reference": int((y == 0.0).sum())}


def install_thinning(pipeline, fraction: float, seed: int, apply_to: str,
                     keep_full_val: bool) -> dict:
    """Hook the thinning into ``pipeline`` right after its split assignment.

    Returns a dict that is filled during ``prepare_data`` with the original
    row index of every kept event (``kept_rows``) and the event counts.
    """
    from .data.dataset import SplitIndices
    from .data.splitting import TEST_FOLD, TRAIN, VAL

    info: dict = {"fraction": fraction, "apply_to": apply_to, "subsample_seed": seed}

    if pipeline.kfold_step is None:
        orig_split = pipeline.split_step.split

        def split(dataset):
            splits = orig_split(dataset)
            n_before = {"train": _counts(dataset, splits.train),
                        "val": _counts(dataset, splits.val)}
            groups = (TRAIN,) if keep_full_val else (TRAIN, VAL)
            keep = thin_mask(
                dataset.sample_id, dataset.y.numpy().reshape(-1),
                dataset.split_label.astype(np.int64), dataset.sample_names,
                fraction, seed, groups, apply_to,
            )
            remap = _take_rows(dataset, keep)
            out = SplitIndices(
                train=_remap(splits.train, remap),
                val=_remap(splits.val, remap),
                test=_remap(splits.test, remap),
            )
            if len(out.test) != len(splits.test):
                raise RuntimeError("thinning touched the test split")
            info.update(
                kept_rows=np.flatnonzero(keep),
                n_before=n_before,
                n_after={"train": _counts(dataset, out.train),
                         "val": _counts(dataset, out.val),
                         "test": _counts(dataset, out.test)},
            )
            _report(pipeline.run_name, info)
            pipeline.record.update(sample_size_scan={k: v for k, v in info.items()
                                                     if k != "kept_rows"})
            return out

        pipeline.split_step.split = split
    else:
        if keep_full_val:
            print("[sample_size_scan] --keep-full-val has no effect with k-fold: the "
                  "validation share is drawn per member from the thinned folds.", flush=True)
        orig_assign = pipeline.kfold_step.assign

        def assign(dataset):
            folds = orig_assign(dataset).astype(np.int64)
            cv_before = np.flatnonzero(folds != TEST_FOLD)
            n_before = {"cv": _counts(dataset, cv_before)}
            # One group for all cross-validated events, one for the test fold.
            groups = np.where(folds == TEST_FOLD, -1, 0)
            keep = thin_mask(
                dataset.sample_id, dataset.y.numpy().reshape(-1), groups,
                dataset.sample_names, fraction, seed, (0,), apply_to,
            )
            if not keep[folds == TEST_FOLD].all():
                raise RuntimeError("thinning touched the k-fold test set")
            _take_rows(dataset, keep)
            new_folds = folds[keep].astype(np.int16)
            info.update(
                kept_rows=np.flatnonzero(keep),
                n_before=n_before,
                n_after={"cv": _counts(dataset, np.flatnonzero(new_folds != TEST_FOLD)),
                         "test": _counts(dataset, np.flatnonzero(new_folds == TEST_FOLD)),
                         "per_fold": np.bincount(new_folds[new_folds != TEST_FOLD],
                                                 minlength=pipeline.kfold_step.k).tolist()},
            )
            _report(pipeline.run_name, info)
            pipeline.record.update(sample_size_scan={k: v for k, v in info.items()
                                                     if k != "kept_rows"})
            return new_folds

        pipeline.kfold_step.assign = assign
    return info


def _report(tag: str, info: dict) -> None:
    print(f"[{tag}] sample-size scan: keeping {info['fraction']:.0%} of the "
          f"training events ({info['apply_to']})", flush=True)
    for split, after in info["n_after"].items():
        before = info["n_before"].get(split)
        if isinstance(after, dict) and before:
            print(f"[{tag}]   {split:5s} target {before['target']:>10,d} -> {after['target']:>10,d}"
                  f"   reference {before['reference']:>10,d} -> {after['reference']:>10,d}",
                  flush=True)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_fraction(base_cfg: dict, fraction: float, out_dir: Path, args) -> Path:
    from .data.splitting import TEST_FOLD
    from .pipeline import Pipeline

    cfg = copy.deepcopy(base_cfg)
    pct = int(round(fraction * 100))
    run_dir = out_dir / f"frac_{pct:03d}"
    cfg["run_name"] = f"{base_cfg.get('run_name', 'carl')}_frac{pct:03d}"
    cfg["output_dir"] = str(run_dir)
    cfg.setdefault("data", {})["save_reference"] = None   # never overwrite a shared cache
    if args.single:
        cfg.setdefault("ensemble", {})["enabled"] = False
        gpus = (cfg.get("training") or {}).get("gpus") or []
        # One device: a multi-GPU single model would run under DDP, which
        # re-launches this script and with it the whole scan.
        cfg.setdefault("training", {})["gpus"] = list(gpus[:1])
    if args.gpus is not None:
        cfg.setdefault("training", {})["gpus"] = list(args.gpus)

    pipeline = Pipeline(cfg)
    info = install_thinning(pipeline, fraction, args.subsample_seed, args.apply_to,
                            args.keep_full_val)
    result = pipeline.run()

    dataset = pipeline.dataset
    if pipeline.folds is not None:
        test_idx = np.flatnonzero(np.asarray(pipeline.folds) == TEST_FOLD)
    else:
        test_idx = np.asarray(pipeline.splits.test)
    if test_idx.size == 0:
        raise RuntimeError("empty test set: nothing to evaluate the calibration on")
    x_test = dataset.x.numpy()[test_idx]

    if result.get("ensemble") is not None:
        scores = result["ensemble"].inference(x_test)
        result["ensemble"].release_gpu()
    else:
        from .model import CARL

        ckpt = result["summaries"][0]["checkpoint"]
        model = CARL.load_from_checkpoint(ckpt, map_location="cpu")
        scores = pipeline.trainer.predict(model, dataset, test_idx)

    original_rows = info["kept_rows"][test_idx]
    path = run_dir / "test_scores.npz"
    np.savez(
        path,
        fraction=fraction,
        scores=np.asarray(scores, dtype=np.float64).reshape(-1),
        labels=dataset.y.numpy().reshape(-1)[test_idx],
        weights=dataset.w.numpy().reshape(-1)[test_idx].astype(np.float64),
        test_fingerprint=_fingerprint(original_rows),
        counts=json.dumps(info["n_after"]),
    )
    print(f"[{cfg['run_name']}] test scores -> {path}", flush=True)
    return path


def _fingerprint(rows: np.ndarray) -> str:
    return hashlib.sha1(np.sort(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def calibration_points(scores, labels, weights, n_bins: int, z: float = 1.0) -> dict:
    """Same binning, weights and Wilson errors as ``CalibrationCurvePlot``."""
    from .evaluation.plots import effective_n, wilson_interval

    s = np.clip(scores, 0.0, 1.0)
    y, w = labels, weights.astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(s, bins) - 1, 0, n_bins - 1)
    num, den, den_w2 = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    np.add.at(num, idx[y == 1.0], w[y == 1.0])
    np.add.at(den, idx, w)
    np.add.at(den_w2, idx, w**2)
    valid = den > 0
    frac = np.full(n_bins, np.nan)
    frac[valid] = num[valid] / den[valid]
    lo, hi = wilson_interval(frac, effective_n(den, den_w2), z=z)
    # Weighted expected calibration error: bin weight share x |observed - predicted|.
    share = den / den.sum() if den.sum() > 0 else den
    ece = float(np.nansum(share[valid] * np.abs(frac[valid] - centers[valid])))
    return {"centers": centers, "frac": frac, "lo": lo, "hi": hi, "valid": valid,
            "ece": ece, "bins": bins}


def plot_comparison(paths: list[Path], out_dir: Path, n_bins: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted((dict(np.load(p, allow_pickle=False)) for p in paths),
                  key=lambda r: float(r["fraction"]))
    prints = {str(r["test_fingerprint"]) for r in runs}
    if len(prints) != 1:
        raise RuntimeError(
            "the runs were not evaluated on the same test events; re-run them with "
            "the same config, seed and split settings"
        )

    # Ordered quantity -> one hue, light to dark; markers as a second cue.
    cmap = plt.get_cmap("Blues")
    colors = [cmap(v) for v in np.linspace(0.45, 1.0, len(runs))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    offsets = (np.arange(len(runs)) - (len(runs) - 1) / 2) * (0.25 / n_bins)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    axr.axhline(0.0, ls="--", color="grey", lw=1)
    summary = []
    for i, run in enumerate(runs):
        f = float(run["fraction"])
        c = calibration_points(run["scores"], run["labels"], run["weights"], n_bins)
        v = c["valid"]
        x = c["centers"][v] + offsets[i]   # small dodge so the error bars stay readable
        yerr = np.vstack([c["lo"][v], c["hi"][v]])
        label = f"{f:.0%} of train sample  (ECE {c['ece']:.4f})"
        kw = dict(color=colors[i], marker=markers[i % len(markers)], ms=4, lw=1.2,
                  capsize=2, elinewidth=1.0)
        ax.errorbar(x, c["frac"][v], yerr=yerr, label=label, **kw)
        axr.errorbar(x, c["frac"][v] - c["centers"][v], yerr=yerr, **kw)
        counts = json.loads(str(run["counts"]))
        summary.append({"fraction": f, "ece": c["ece"], "counts": counts})

    ax.set_ylabel("Weighted target fraction")
    ax.set_title(f"Calibration vs training-sample size: {title}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    axr.set_xlabel(r"Predicted score $\hat{s}$")
    axr.set_ylabel("observed - predicted")
    axr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_comparison.pdf")
    fig.savefig(out_dir / "calibration_comparison.png", dpi=150)
    plt.close(fig)

    # Score distributions: where the events sit tells which calibration bins matter.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    hbins = np.linspace(0, 1, 51)
    for i, run in enumerate(runs):
        s = np.clip(run["scores"], 0, 1)
        for ax_, cls, name in ((axes[0], 1.0, "target"), (axes[1], 0.0, "reference")):
            m = run["labels"] == cls
            ax_.hist(s[m], bins=hbins, weights=run["weights"][m], histtype="step",
                     density=True, lw=1.4, color=colors[i],
                     label=f"{float(run['fraction']):.0%}")
            ax_.set_title(f"{name} test events")
            ax_.set_xlabel(r"$\hat{s}$")
    axes[0].set_ylabel("normalised weighted events")
    axes[0].set_yscale("log")
    for ax_ in axes:
        ax_.grid(alpha=0.3)
    axes[1].legend(title="train fraction", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.pdf")
    plt.close(fig)

    with open(out_dir / "sample_size_scan.json", "w") as fh:
        json.dump({"test_fingerprint": prints.pop(), "runs": summary}, fh, indent=2)
    print(f"[sample_size_scan] plots -> {out_dir}/calibration_comparison.pdf, "
          "score_distributions.pdf", flush=True)
    for s_ in summary:
        print(f"  {s_['fraction']:>5.0%}  ECE = {s_['ece']:.5f}", flush=True)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from .config import load_config

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="pipeline YAML, e.g. configs/base/S.yaml")
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS),
                   help="fractions of the training events to keep (default: 0.1 0.2 0.5 1.0)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <output_dir>/sample_size_scan)")
    p.add_argument("--apply-to", choices=["both", "target", "reference"], default="both",
                   help="which class to thin (default: both)")
    p.add_argument("--keep-full-val", action="store_true",
                   help="thin only the train split, keep the whole validation split")
    p.add_argument("--subsample-seed", type=int, default=None,
                   help="seed of the nested subsets (default: the config's seed)")
    p.add_argument("--single", action="store_true",
                   help="train one network per fraction instead of the configured ensemble")
    p.add_argument("--gpus", type=int, nargs="*", default=None,
                   help="override training.gpus")
    p.add_argument("--n-bins", type=int, default=30, help="calibration bins (default 30)")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; re-plot from <out>/frac_*/test_scores.npz")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if (cfg.get("rosmm") or {}).get("enabled", False):
        raise SystemExit("rosmm runs are not supported by the sample-size scan")
    if args.single and int((cfg.get("split") or {}).get("k_folds", 0) or 0):
        raise SystemExit("--single cannot be combined with split.k_folds (k-fold needs an ensemble)")
    if args.subsample_seed is None:
        args.subsample_seed = int(cfg.get("seed", 52))
    out_dir = Path(args.out or Path(cfg.get("output_dir", "outputs/carl")) / "sample_size_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = sorted(set(args.fractions))
    if args.plot_only:
        paths = sorted(out_dir.glob("frac_*/test_scores.npz"))
        if not paths:
            raise SystemExit(f"no test_scores.npz under {out_dir}")
    else:
        paths = [run_fraction(cfg, f, out_dir, args) for f in fractions]
    plot_comparison(paths, out_dir, args.n_bins, cfg.get("run_name", Path(args.config).stem))


if __name__ == "__main__":
    main()
