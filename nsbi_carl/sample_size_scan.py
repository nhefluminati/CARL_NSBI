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

* ``calibration_frac_XXX.pdf``: one calibration curve per fraction (Wilson
  error bars, same binning and weights as the pipeline's
  ``calibration_curve`` plot) with a residual panel, observed - predicted.
  All files share one residual scale, so they can be compared side by side.
  The box quotes the ECE and the ``density_ratio_integral`` with its
  statistical error.
* ``metrics_vs_fraction.pdf``: the weighted expected calibration error (ECE)
  and the density-ratio integral ``sum_ref r w / sum_ref w`` (should be 1)
  against the fraction, both on the common test set.
* ``score_distributions.pdf``: the test-set score histograms per class.
* ``sample_size_scan.json``: per-run event counts, ECE and ratio integral.

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


def ratio_integral(scores, labels, weights) -> tuple[float, float]:
    """``density_ratio_integral`` on the test reference events, with its
    statistical uncertainty.

    I = sum_ref r_i w_i / sum_ref w_i should be 1. Its sampling error is
    sqrt(sum_ref (w_i/W)^2 (r_i - I)^2): with that, a deviation from 1 can be
    told apart from test-sample noise, which is the same for every fraction
    here because every run is scored on the same test events.
    """
    from .evaluation.base import EvaluationContext
    from .evaluation.metrics import DensityRatioIntegral

    ctx = EvaluationContext(scores=np.asarray(scores, dtype=np.float64),
                            labels=labels, weights=weights)
    integral = DensityRatioIntegral().evaluate(ctx)["density_ratio_integral"]
    ref = labels == 0.0
    r = np.clip(ctx.ratio[ref], -1e13, 1e13)
    p = weights[ref].astype(np.float64)
    p = p / p.sum()
    err = float(np.sqrt(np.sum(p**2 * (r - integral) ** 2)))
    return float(integral), err


def _train_counts_label(counts: dict) -> str:
    part = counts.get("train") or counts.get("cv") or {}
    if not part:
        return ""
    return (f"train events: target {part.get('target', 0):,d}, "
            f"reference {part.get('reference', 0):,d}")


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

    # Everything first, so all per-fraction files share one residual scale
    # and can be compared side by side.
    results = []
    for run in runs:
        c = calibration_points(run["scores"], run["labels"], run["weights"], n_bins)
        integral, integral_err = ratio_integral(run["scores"], run["labels"], run["weights"])
        results.append({"run": run, "cal": c, "integral": integral,
                        "integral_err": integral_err,
                        "counts": json.loads(str(run["counts"]))})
    resid_max = max(
        float(np.nanmax(np.abs(r["cal"]["frac"] - r["cal"]["centers"])[r["cal"]["valid"]]))
        for r in results
    )
    resid_lim = max(0.02, 1.3 * resid_max)

    # -- one calibration file per fraction --------------------------------
    for r in results:
        f = float(r["run"]["fraction"])
        c, v = r["cal"], r["cal"]["valid"]
        yerr = np.vstack([c["lo"][v], c["hi"][v]])
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(6.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
        )
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        kw = dict(color="#1f5fa8", marker="o", ms=4, lw=1.2, capsize=2, elinewidth=1.0)
        ax.errorbar(c["centers"][v], c["frac"][v], yerr=yerr, label="classifier", **kw)
        axr.axhline(0.0, ls="--", color="grey", lw=1)
        axr.errorbar(c["centers"][v], c["frac"][v] - c["centers"][v], yerr=yerr, **kw)

        text = (f"ECE = {c['ece']:.4f}\n"
                f"$\\Sigma_{{ref}}\\, r\\,w / \\Sigma_{{ref}}\\, w$ = "
                f"{r['integral']:.4f} $\\pm$ {r['integral_err']:.4f}\n"
                f"{_train_counts_label(r['counts'])}")
        ax.text(0.97, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
        ax.set_title(f"Calibration, {f:.0%} of the training sample: {title}")
        ax.set_ylabel("Weighted target fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        axr.set_ylim(-resid_lim, resid_lim)
        axr.set_xlabel(r"Predicted score $\hat{s}$")
        axr.set_ylabel("observed - predicted")
        axr.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"calibration_frac_{int(round(f * 100)):03d}.pdf")
        plt.close(fig)

    # -- metrics vs fraction ---------------------------------------------
    fr = np.array([float(r["run"]["fraction"]) for r in results])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
    a1.plot(fr, [r["cal"]["ece"] for r in results], "o-", color="#1f5fa8")
    a1.set_ylabel("ECE (test set)")
    a1.set_title(f"Calibration and reweighting vs training-sample size: {title}")
    a1.set_ylim(bottom=0)
    a1.grid(alpha=0.3)
    a2.axhline(1.0, ls="--", color="grey", lw=1)
    a2.errorbar(fr, [r["integral"] for r in results],
                yerr=[r["integral_err"] for r in results],
                fmt="o-", color="#1f5fa8", capsize=3)
    a2.set_ylabel(r"$\Sigma_{ref}\, r\,w / \Sigma_{ref}\, w$ (test set)")
    a2.set_xlabel("fraction of the training sample used")
    a2.set_xscale("log")
    a2.set_xticks(fr)
    a2.set_xticklabels([f"{x:.0%}" for x in fr])
    a2.minorticks_off()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_fraction.pdf")
    plt.close(fig)

    # -- score distributions (all fractions; histograms overlay cleanly) --
    cmap = plt.get_cmap("Blues")
    colors = [cmap(x) for x in np.linspace(0.45, 1.0, len(results))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    hbins = np.linspace(0, 1, 51)
    for i, r in enumerate(results):
        run = r["run"]
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

    summary = [{"fraction": float(r["run"]["fraction"]), "ece": r["cal"]["ece"],
                "density_ratio_integral": r["integral"],
                "density_ratio_integral_deviation": r["integral"] - 1.0,
                "density_ratio_integral_stat_error": r["integral_err"],
                "counts": r["counts"]} for r in results]
    with open(out_dir / "sample_size_scan.json", "w") as fh:
        json.dump({"test_fingerprint": prints.pop(), "runs": summary}, fh, indent=2)
    print(f"[sample_size_scan] plots -> {out_dir}/calibration_frac_*.pdf, "
          "metrics_vs_fraction.pdf, score_distributions.pdf", flush=True)
    print(f"  {'fraction':>8s}  {'ECE':>8s}  {'ratio integral':>22s}", flush=True)
    for s_ in summary:
        print(f"  {s_['fraction']:>8.0%}  {s_['ece']:>8.5f}  "
              f"{s_['density_ratio_integral']:>10.5f} +- "
              f"{s_['density_ratio_integral_stat_error']:.5f}", flush=True)

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
