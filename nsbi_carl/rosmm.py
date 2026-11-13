"""Ratio of signed mixtures model (RoSMM).

Drnevich, Jiggins, Katzy, Cranmer, Mach. Learn.: Sci. Technol. 6 (2025)
045023, Sec. 3 and Algorithms 1-2. Enabled with ``rosmm.enabled: true``.

Idea
----
A weighted target with negative weights has a signed density. Split its events
by the sign of the weight (eqs. 11-12):

    q(x|target) = c p+(x) + (1 - c) p-(x),      c = Σ_{w>=0} w / Σ w  >= 1   (eq. 16)

where p+ is the density of the positive-weight events and p- that of the
negative-weight events taken with |w|; both are ordinary densities. With an
all-positive reference p_r the ratio decomposes as (eq. 28)

    r(x) = q(x|target) / p_r(x) = c r+(x) - (c - 1) r-(x),
    r+ = p+ / p_r,   r- = p- / p_r.

r+ and r- are standard CARL problems with positive weights only, so they are
trained with the weighted BCE: no negative weight ever enters a training loss,
which removes both the unbounded loss and most of the gradient variance that
signed weights cause.

What this module does
---------------------
1. ``Pipeline.prepare_data`` builds, splits, reweights and scales the FULL
   signed dataset once, exactly as a normal run.
2. Two sub-datasets are cut from it -- (target w >= 0 vs reference) and
   (target |w| for w < 0 vs reference). Every event keeps its split label /
   k-fold fold and the features keep the one scaler, so both sub-ratios are
   evaluated on identical inputs.
3. Each sub-dataset is trained as an ordinary child run with BCE, in
   ``<output_dir>/pos`` and ``<output_dir>/neg`` (own run record, manifest,
   checkpoints and evaluation plots). Both use the parent's model/training/
   ensemble settings.
4. Members are paired (member i of ``pos`` with member i of ``neg``; in k-fold
   within each fold) and combined per pair, r_i = c_i r+_i - (c_i - 1) r-_i;
   the pairs are then combined with ``ensemble.combiner`` in signed ratio
   space. Flavors (``rosmm.tune``):

   * ``none``          RoSMM:  c from eq. 16.
   * ``coefficients``  RoSMMc: c refitted by minimising the PARE loss of the
                       combined model with the sub-networks frozen (eq. 18), on
                       events the networks did not train on.
   * ``full``          RoSMMr: every pair fine-tuned jointly (both networks
                       and its c) with the PARE loss on the full signed data,
                       early-stopped on validation (Algorithm 2, full tuning).

   Every flavor up to the requested one is evaluated with the configured
   after-training suite (tags ``rosmm_test``, ``rosmmc_test``, ``rosmmr_test``
   and their ``_out_of_fold`` versions in k-fold mode), so they can be
   compared directly.
5. The parent run record gets a ``rosmm`` section that
   ``inference.RoSMMScorer`` / ``inference.load_templates`` read.

Scope: the reference must have no negative weights (the four-sub-ratio case
of eq. 14 is not implemented). With the default ``reference_unit_weights``
this always holds.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .combine import combine_scores, ratio_from_scores
from .data.dataset import NSBIDataset, SplitIndices
from .data.splitting import TEST_FOLD
from .evaluation.base import EvaluationContext
from .model import (
    PARE_DEFAULT_T0,
    PARE_DEFAULT_T1,
    pare_pole,
    pare_u_from_ratio,
    validate_loss,
    weighted_pare_sums,
)

TUNE_LEVELS = ("none", "coefficients", "full")
FLAVOR_OF = {"none": "rosmm", "coefficients": "rosmmc", "full": "rosmmr"}
PARTS = ("pos", "neg")
_LOGIT_CLAMP = 40.0   # exp(40) ~ 2e17: far beyond any ratio a template needs


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def validate_rosmm_config(config: dict) -> dict:
    """Normalised ``rosmm`` block; raises on settings the mode cannot run."""
    cfg = dict(config.get("rosmm") or {})
    ens = config.get("ensemble") or {}
    if not ens.get("enabled", False):
        raise ValueError(
            "rosmm needs ensemble.enabled: true (use ensemble.n_members: 1 for one network "
            "per sub-ratio). The sub-ratios are trained and paired as ensemble members."
        )
    if ens.get("combiner") == "mean_logit":
        raise ValueError(
            "rosmm combines signed ratios, so ensemble.combiner: mean_logit is not defined. "
            "Use mean_ratio / mean_score (same thing here), trimmed_ratio or median_ratio."
        )
    tune = str(cfg.get("tune", "coefficients")).lower()
    if tune not in TUNE_LEVELS:
        raise ValueError(f"rosmm.tune must be one of {TUNE_LEVELS}, got {tune!r}")
    pare = cfg.get("pare") or {}
    t0 = float(pare.get("t0", PARE_DEFAULT_T0))
    t1 = float(pare.get("t1", PARE_DEFAULT_T1))
    validate_loss("pare", t0, t1)
    if abs(t0) <= abs(t1):
        raise ValueError(
            f"rosmm.pare needs |t0| > |t1| (got t0={t0:g}, t1={t1:g}): the pole sits at "
            "-(t0/t1)^2 and must lie below -1 for negative target weights."
        )
    tuning = dict(cfg.get("tuning") or {})
    out = {
        "tune": tune,
        "t0": t0,
        "t1": t1,
        "min_negative_events": int(cfg.get("min_negative_events", 100)),
        "coefficient_range": cfg.get("coefficient_range"),   # [lo, hi] or None
        "tuning": {
            "learning_rate": float(tuning.get("learning_rate", 1e-5)),
            "batch_size": tuning.get("batch_size"),            # None -> training.batch_size
            "max_epochs": int(tuning.get("max_epochs", 100)),
            "patience": int(tuning.get("patience", 10)),
            "weight_decay": float(tuning.get("weight_decay", 0.0)),
            "tune_coefficient": bool(tuning.get("tune_coefficient", True)),
            "device": tuning.get("device"),
            "log_every_n_epochs": int(tuning.get("log_every_n_epochs", 1)),
        },
    }
    if out["tuning"]["patience"] < 1 or out["tuning"]["max_epochs"] < 1:
        raise ValueError("rosmm.tuning.patience and max_epochs must be >= 1")
    return out


def child_config(config: dict, part: str) -> dict:
    """Config of the ``pos`` / ``neg`` sub-ratio run: the parent's settings,
    trained with BCE, in its own sub-directory."""
    cfg = copy.deepcopy(config)
    run_name = cfg.get("run_name", "carl")
    out_dir = Path(cfg.get("output_dir", f"outputs/{run_name}"))
    cfg["run_name"] = f"{run_name}_{part}"
    cfg["output_dir"] = str(out_dir / part)
    cfg.pop("rosmm", None)
    model = cfg.setdefault("model", {})
    model["loss"] = "bce"
    model.pop("pare", None)
    # Each child writes its own cache-free record; the parent already wrote
    # the reference cache if one was requested.
    data = cfg.setdefault("data", {})
    data.pop("save_reference", None)
    return cfg


# ---------------------------------------------------------------------------
# dataset surgery
# ---------------------------------------------------------------------------
def mixture_coefficient(w_target: np.ndarray) -> float:
    """``c = Σ_{w>=0} w / Σ w`` (eq. 16): 1 without negative weights, larger
    the more of the positive weight is cancelled."""
    w = np.asarray(w_target, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError(f"target weights sum to {total:.6g} <= 0; the signed density "
                         "cannot be normalised")
    return float(w[w >= 0].sum() / total)


def subset_dataset(dataset: NSBIDataset, idx: np.ndarray, weights: np.ndarray) -> NSBIDataset:
    """Rows ``idx`` of a prepared (scaled) dataset, with the given weights."""
    sub = NSBIDataset(
        x=dataset.x.numpy()[idx],
        y=dataset.y.numpy().reshape(-1)[idx],
        w=np.asarray(weights, dtype=np.float64),
        sample_id=dataset.sample_id[idx],
        sample_names=dataset.sample_names,
        feature_names=dataset.feature_names,
        split_label=dataset.split_label[idx],
    )
    sub.mean, sub.std = dataset.mean, dataset.std
    return sub


def remap_splits(splits: SplitIndices, idx: np.ndarray, n_parent: int) -> SplitIndices:
    """Parent split indices -> indices into the subset ``idx`` (order kept)."""
    pos = np.full(n_parent, -1, dtype=np.int64)
    pos[idx] = np.arange(len(idx))

    def m(a):
        p = pos[np.asarray(a, dtype=np.int64)]
        return p[p >= 0]

    return SplitIndices(train=m(splits.train), val=m(splits.val), test=m(splits.test))


# ---------------------------------------------------------------------------
# the combined model on top of two trained child ensembles
# ---------------------------------------------------------------------------
def logits_to_ratio(z: np.ndarray) -> np.ndarray:
    """BCE logit -> density ratio ``exp(z)``, in float64."""
    return np.exp(np.clip(np.asarray(z, dtype=np.float64), -_LOGIT_CLAMP, _LOGIT_CLAMP))


def combine_pair_ratios(r_pos: np.ndarray, r_neg: np.ndarray, c) -> np.ndarray:
    """``c r+ - (c - 1) r-``; ``c`` scalar or one value per row (pair)."""
    c = np.asarray(c, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    return c * r_pos - (c - 1.0) * r_neg


def ratios_to_scores(r: np.ndarray) -> np.ndarray:
    """Package score convention ``s = r / (1 + r)`` (signed ratios allowed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return r / (1.0 + r)


@dataclass
class RoSMMEnsemble:
    """Pairs member i of ``pos`` with member i of ``neg``; ``c`` per pair."""

    pos: object          # CARLEnsemble
    neg: object          # CARLEnsemble
    c: np.ndarray        # (n_pairs,) in manifest order
    combiner: str
    trim: float

    def __post_init__(self):
        tp = [m["member"] for m in self.pos.manifest.get("members", [])]
        tn = [m["member"] for m in self.neg.manifest.get("members", [])]
        if tp != tn:
            raise ValueError(
                f"pos/neg ensembles have different members ({tp[:6]}... vs {tn[:6]}...); "
                "they must be trained with identical ensemble settings to be paired"
            )
        self.c = np.broadcast_to(np.asarray(self.c, dtype=np.float64), (len(tp),)).copy()

    @property
    def n_pairs(self) -> int:
        return len(self.c)

    def with_c(self, c) -> "RoSMMEnsemble":
        return RoSMMEnsemble(self.pos, self.neg, c, self.combiner, self.trim)

    # -- member level --------------------------------------------------
    def member_ratios(self, x: np.ndarray, device=None) -> np.ndarray:
        """``(n_pairs, n)`` pair ratios, every pair scoring every event."""
        rp = logits_to_ratio(self.pos.member_predictions(x, device=device, raw=True))
        rn = logits_to_ratio(self.neg.member_predictions(x, device=device, raw=True))
        return combine_pair_ratios(rp, rn, self.c)

    def oof_member_ratios(self, x: np.ndarray, folds: np.ndarray, device=None) -> np.ndarray:
        """``(pairs_per_fold, n)``: column i from the pairs of event i's fold."""
        rp = logits_to_ratio(self.pos.out_of_fold_member_predictions(x, folds, device=device,
                                                                     raw=True))
        rn = logits_to_ratio(self.neg.out_of_fold_member_predictions(x, folds, device=device,
                                                                     raw=True))
        per_fold = self.pos.fold_members()
        c = np.empty_like(rp)
        for f, members in enumerate(per_fold):
            sel = folds == f
            if sel.any():
                c[:, sel] = self.c[np.asarray(members)][:, None]
        return c * rp - (c - 1.0) * rn

    # -- combined -------------------------------------------------------
    def combine(self, member_ratios: np.ndarray) -> np.ndarray:
        """Combined score of each event from its ``(n_pairs, n)`` ratios."""
        return combine_scores(ratios_to_scores(member_ratios), self.combiner, self.trim,
                              signed=True)

    def scores(self, x: np.ndarray, device=None, chunk: int = 1_000_000) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        for i in range(0, len(x), chunk):
            out[i : i + chunk] = self.combine(self.member_ratios(x[i : i + chunk], device))
        return out

    def scores_out_of_fold(self, x: np.ndarray, folds: np.ndarray, device=None,
                           chunk: int = 1_000_000) -> np.ndarray:
        """Each event scored only by pairs that never trained on it (its own
        fold's pairs; all pairs for the test set, fold -1)."""
        folds = np.asarray(folds)
        out = np.empty(len(x), dtype=np.float64)
        for i in range(0, len(x), chunk):
            xs, fs = x[i : i + chunk], folds[i : i + chunk]
            o = np.empty(len(xs), dtype=np.float64)
            test = fs == TEST_FOLD
            if test.any():
                o[test] = self.combine(self.member_ratios(xs[test], device))
            if (~test).any():
                o[~test] = self.combine(self.oof_member_ratios(xs[~test], fs[~test], device))
            out[i : i + chunk] = o
        return out

    def release_gpu(self) -> None:
        self.pos.release_gpu()
        self.neg.release_gpu()


# ---------------------------------------------------------------------------
# evaluation of one flavor with the configured after-training suite
# ---------------------------------------------------------------------------
def evaluate_flavor(parent, dataset: NSBIDataset, splits: SplitIndices, folds, model: RoSMMEnsemble,
                    flavor: str, histories: list[dict]) -> dict:
    suite = parent.trainer.final_evaluation
    x = dataset.x.numpy()
    y = dataset.y.numpy().reshape(-1)
    w = dataset.w.numpy().reshape(-1).astype(np.float64)

    def run(idx: np.ndarray, scores: np.ndarray, tag: str) -> dict:
        ctx = EvaluationContext(
            scores=scores, labels=y[idx], weights=w[idx], features=x[idx],
            feature_names=dataset.feature_names, mean=dataset.mean, std=dataset.std,
            member_scores=None, histories=histories,
            output_dir=parent.output_dir / "evaluation", tag=tag,
        )
        res = suite.run(ctx) if len(suite) else {}
        if res:
            out = parent.output_dir / "evaluation"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"metrics_{tag}.json", "w") as f:
                json.dump(res, f, indent=2)
        return res

    try:
        if folds is not None:
            results: dict = {}
            test_idx = np.flatnonzero(folds == TEST_FOLD)
            cv_idx = np.flatnonzero(folds != TEST_FOLD)
            if test_idx.size:
                results["test"] = run(test_idx, model.scores(x[test_idx]), f"{flavor}_test")
            if cv_idx.size:
                results["out_of_fold"] = run(
                    cv_idx, model.scores_out_of_fold(x[cv_idx], folds[cv_idx]),
                    f"{flavor}_test_out_of_fold",
                )
            return results
        idx = np.asarray(splits.test)
        return run(idx, model.scores(x[idx]), f"{flavor}_test")
    finally:
        model.release_gpu()


# ---------------------------------------------------------------------------
# RoSMMc: refit c with the PARE loss, sub-networks frozen (eq. 18)
# ---------------------------------------------------------------------------
def _pare_mean_loss(r: np.ndarray, y: np.ndarray, w: np.ndarray, t0: float, t1: float) -> float:
    rho = t1 / t0
    u = pare_u_from_ratio(r, t0, t1)
    per = np.where(y == 1.0, rho * rho * u * u - 2.0 * (1.0 - rho) * u, u * u)
    return float(np.sum(per * w) / np.sum(w))


def fit_coefficient(r_pos: np.ndarray, r_neg: np.ndarray, y: np.ndarray, w: np.ndarray,
                    c_hat: float, t0: float, t1: float, c_range=None) -> dict:
    """1-D minimisation of the PARE loss of ``c r+ - (c-1) r-`` over c.

    ``r_pos`` / ``r_neg`` are the (ensemble-combined) sub-ratios on held-out
    events, ``w`` their weights rebalanced to equal class totals. A coarse
    grid finds the basin (the loss need not be convex in c once some events
    approach the pole), golden-section search refines it.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if c_range is None:
        span = max(3.0 * (c_hat - 1.0), 0.5)
        lo, hi = 1.0, 1.0 + span
    else:
        lo, hi = float(c_range[0]), float(c_range[1])
    d = r_pos - r_neg

    def loss(c: float) -> float:
        return _pare_mean_loss(r_neg + c * d, y, w, t0, t1)

    grid = np.linspace(lo, hi, 61)
    vals = np.array([loss(c) for c in grid])
    k = int(np.nanargmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = loss(c1), loss(c2)
    for _ in range(60):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - g * (b - a)
            f1 = loss(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + g * (b - a)
            f2 = loss(c2)
    c_fit = float((a + b) / 2.0)
    at_edge = k in (0, len(grid) - 1)
    r_fit = r_neg + c_fit * d
    return {
        "c_fit": c_fit,
        "loss_at_c_hat": loss(c_hat),
        "loss_at_c_fit": loss(c_fit),
        "search_range": [lo, hi],
        "at_search_edge": bool(at_edge),
        "n_events": int(len(y)),
        "n_below_pole": int((r_fit <= pare_pole(t0, t1)).sum()),
        "grid": grid.tolist(),
        "grid_loss": vals.tolist(),
    }


def _plot_coefficient_scan(path: Path, fit: dict, c_hat: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fit["grid"], fit["grid_loss"], color="C0")
    ax.axvline(c_hat, color="C1", ls="--", label=f"eq. 16: c = {c_hat:.4f}")
    ax.axvline(fit["c_fit"], color="C2", ls=":", label=f"PARE fit: c = {fit['c_fit']:.4f}")
    ax.set_xlabel("mixture coefficient c")
    ax.set_ylabel("PARE loss (held-out events)")
    ax.set_title("RoSMMc coefficient scan")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _child_combined_ratio(ens, x, folds=None) -> np.ndarray:
    """Ensemble-combined sub-ratio of each event (out-of-fold if ``folds``)."""
    if folds is None:
        raw = ens.member_predictions(x, raw=True)
    else:
        raw = ens.out_of_fold_member_predictions(x, folds, raw=True)
    s = combine_scores(ratios_to_scores(logits_to_ratio(raw)), ens.config.combiner,
                       ens.config.combiner_trim)
    return ratio_from_scores(s)


def rebalanced(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Target weights scaled so both classes carry equal total weight."""
    w = np.asarray(w, dtype=np.float64).copy()
    t, r = w[y == 1.0].sum(), w[y == 0.0].sum()
    if t > 0 and r > 0:
        w[y == 1.0] *= r / t
    return w


def coefficient_fit_events(splits: SplitIndices, folds) -> tuple[np.ndarray, str]:
    """Events c is fitted on: ones the sub-networks did not train on."""
    if folds is not None:
        return np.flatnonzero(folds != TEST_FOLD), "out_of_fold"
    if len(splits.val):
        return np.asarray(splits.val), "validation"
    return np.asarray(splits.train), "train"


# ---------------------------------------------------------------------------
# RoSMMr: joint PARE fine-tuning of every pair (Algorithm 2, full tuning)
# ---------------------------------------------------------------------------
def _pair_train_val(parent, dataset, splits, folds, member: dict) -> tuple[np.ndarray, np.ndarray]:
    """The pair's own train/val events in the FULL dataset, drawn the way its
    sub-networks' members were (same fold, same member seed)."""
    y = dataset.y.numpy().reshape(-1)
    if folds is not None:
        s = parent.kfold_step.member_indices(dataset, folds, int(member["fold"]),
                                             int(member["seed"]))
        return s.train, s.val
    ec = parent.ensemble.config
    if ec.bootstrap:
        from .training.ensemble import bootstrap_train_indices

        train = bootstrap_train_indices(splits, y, int(member["seed"]), ec.bootstrap_fraction,
                                        ec.bootstrap_reference)
    else:
        train = np.asarray(splits.train)
    return train, np.asarray(splits.val)


def _tuning_device(parent, rcfg) -> torch.device:
    dev = rcfg["tuning"]["device"]
    if dev:
        return torch.device(dev)
    gpus = parent.trainer.config.gpus or []
    if gpus and torch.cuda.is_available():
        g = gpus[0]
        return torch.device(g if isinstance(g, str) else f"cuda:{int(g)}")
    return torch.device("cpu")


def finetune_pair(pos_model, neg_model, c_init: float, x, y, w, train, val, t0, t1, tcfg,
                  batch_size: int, device, tag: str) -> dict:
    """Fine-tune one (r+, r-, c) triple jointly with the PARE loss."""
    tw = rebalanced(w[train], y[train])
    vw = rebalanced(w[val], y[val]) if len(val) else tw
    if not len(val):
        val, vw = train, tw
    to = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    xt, yt, wt = to(x[train]), to(y[train]), to(tw)
    xv, yv, wv = to(x[val]), to(y[val]), to(vw, torch.float64)

    net_p = copy.deepcopy(pos_model.net).to(device).train()
    net_n = copy.deepcopy(neg_model.net).to(device).train()
    c = torch.nn.Parameter(torch.tensor(float(c_init), device=device))
    groups = [{"params": list(net_p.parameters()) + list(net_n.parameters()),
               "weight_decay": tcfg["weight_decay"]}]
    if tcfg["tune_coefficient"]:
        groups.append({"params": [c], "weight_decay": 0.0})
    else:
        c.requires_grad_(False)
    opt = torch.optim.Adam(groups, lr=tcfg["learning_rate"])

    def ratio(xb):
        rp = torch.exp(torch.clamp(net_p(xb).flatten(), -_LOGIT_CLAMP, _LOGIT_CLAMP))
        rn = torch.exp(torch.clamp(net_n(xb).flatten(), -_LOGIT_CLAMP, _LOGIT_CLAMP))
        return c * rp - (c - 1.0) * rn

    def val_loss() -> float:
        net_p.eval(), net_n.eval()
        num = torch.zeros((), dtype=torch.float64, device=device)
        den = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for i in range(0, len(xv), 4 * batch_size):
                u = pare_u_from_ratio(ratio(xv[i : i + 4 * batch_size]).double(), t0, t1)
                n, d = weighted_pare_sums(u, yv[i : i + 4 * batch_size],
                                          wv[i : i + 4 * batch_size], t0, t1,
                                          accumulate_dtype=None, param="ratio")
                num, den = num + n, den + d
        net_p.train(), net_n.train()
        return float(num / den)

    def snapshot():
        return ({k: v.detach().cpu().clone() for k, v in net_p.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in net_n.state_dict().items()},
                float(c.detach()))

    best_val = val_loss()
    best = snapshot()
    hist_train, hist_val, stale = [], [best_val], 0
    gen = torch.Generator(device="cpu").manual_seed(0)
    for epoch in range(1, tcfg["max_epochs"] + 1):
        perm = torch.randperm(len(xt), generator=gen).to(device)
        tot, cnt = 0.0, 0
        for i in range(0, len(xt), batch_size):
            b = perm[i : i + batch_size]
            u = pare_u_from_ratio(ratio(xt[b]), t0, t1)
            num, den = weighted_pare_sums(u, yt[b], wt[b], t0, t1, accumulate_dtype=None,
                                          param="ratio")
            loss = num / den
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            cnt += len(b)
        vl = val_loss()
        hist_train.append(tot / max(cnt, 1))
        hist_val.append(vl)
        if vl < best_val:
            best_val, best, stale = vl, snapshot(), 0
        else:
            stale += 1
        if tcfg["log_every_n_epochs"] and epoch % tcfg["log_every_n_epochs"] == 0:
            print(f"[{tag}] tune epoch {epoch:4d}  train {hist_train[-1]:.8f}  val {vl:.8f}  "
                  f"c {float(c.detach()):.5f}", flush=True)
        if not math.isfinite(vl):
            print(f"[{tag}] validation loss is not finite; keeping the best state", flush=True)
            break
        if stale >= tcfg["patience"]:
            break

    return {"pos_state": best[0], "neg_state": best[1], "c": best[2],
            "train_loss": hist_train, "val_loss": hist_val, "best_val_loss": best_val,
            "epochs": len(hist_train)}


def _save_tuned(src_ckpt: str, model, state: dict, path: Path) -> str:
    """A copy of ``src_ckpt`` whose weights are the tuned ``state``."""
    tuned = copy.deepcopy(model).cpu()
    tuned.net.load_state_dict(state)
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    ck["state_dict"] = tuned.state_dict()
    ck["rosmm_tuned"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return str(path)


def finetune_all(parent, dataset, splits, folds, base: RoSMMEnsemble, rcfg) -> tuple:
    """Fine-tune every pair; returns (tuned RoSMMEnsemble, record dict, histories)."""
    tcfg = rcfg["tuning"]
    device = _tuning_device(parent, rcfg)
    batch_size = int(tcfg["batch_size"] or parent.trainer.config.batch_size)
    x = dataset.x.numpy()
    y = dataset.y.numpy().reshape(-1)
    w = dataset.w.numpy().reshape(-1).astype(np.float64)
    t0, t1 = rcfg["t0"], rcfg["t1"]

    pos, neg = base.pos, base.neg
    if not pos.models:
        pos.load()
    if not neg.models:
        neg.load()
    members = pos.manifest["members"]
    out_root = parent.output_dir / "rosmmr"
    tuned_pos, tuned_neg, cs, histories, ckpts = [], [], [], [], {"pos": [], "neg": []}
    print(f"[{parent.run_name}] RoSMMr: fine-tuning {len(members)} pair(s) on {device} "
          f"(lr {tcfg['learning_rate']:g}, batch {batch_size}, patience {tcfg['patience']})",
          flush=True)
    for i, m in enumerate(members):
        tag = m["tag"]
        train, val = _pair_train_val(parent, dataset, splits, folds, m)
        res = finetune_pair(pos.models[i], neg.models[i], float(base.c[i]), x, y, w, train, val,
                            t0, t1, tcfg, batch_size, device, f"{parent.run_name} {tag}")
        for part, ens, state_key, out_list in (("pos", pos, "pos_state", tuned_pos),
                                               ("neg", neg, "neg_state", tuned_neg)):
            path = out_root / part / tag / "checkpoints" / f"best_{tag}.ckpt"
            src = ens.manifest["members"][i]["checkpoint"]
            ckpts[part].append(_save_tuned(src, ens.models[i], res[state_key], path))
            model = copy.deepcopy(ens.models[i]).cpu()
            model.net.load_state_dict(res[state_key])
            out_list.append(model.eval())
        cs.append(res["c"])
        histories.append({"member": m["member"], "train_loss": res["train_loss"],
                          "val_loss": res["val_loss"]})
        print(f"[{parent.run_name}] {tag}: c {base.c[i]:.5f} -> {res['c']:.5f}, "
              f"best val {res['best_val_loss']:.8f} after {res['epochs']} epochs", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def tuned_ens(ens, models):
        t = copy.copy(ens)
        t.models = models
        t._stacked, t._stacked_device, t._fold_stacks, t._stack_cache = None, None, {}, {}
        return t

    tuned = RoSMMEnsemble(tuned_ens(pos, tuned_pos), tuned_ens(neg, tuned_neg), np.array(cs),
                          base.combiner, base.trim)
    rel = {p: [str(Path(c).relative_to(parent.output_dir)) for c in ckpts[p]] for p in PARTS}
    record = {"c": [float(v) for v in cs], "checkpoints": rel,
              "best_val_loss": [h["val_loss"][int(np.argmin(h["val_loss"]))] for h in histories]}
    with open(out_root / "tuning_history.json", "w") as f:
        json.dump({"members": [m["tag"] for m in members], "c": record["c"],
                   "histories": histories}, f, indent=2)
    return tuned, record, histories


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _child_histories(ens, part: str) -> list[dict]:
    return [{"member": f"{part}{m['member']}", "train_loss": m.get("train_loss", []),
             "val_loss": m.get("val_loss", [])} for m in ens.manifest.get("members", [])]


def run_rosmm(parent) -> dict:
    """Train and evaluate a RoSMM template; see the module docstring."""
    from .pipeline import Pipeline

    rcfg = parent.rosmm_config
    dataset, splits = parent.prepare_data()
    folds = parent.folds
    y = dataset.y.numpy().reshape(-1)
    w = dataset.w.numpy().reshape(-1).astype(np.float64)
    n = len(y)

    if np.any(w[y == 0.0] < 0):
        raise NotImplementedError(
            "rosmm: the reference has negative weights. Only the target may be signed here "
            "(the four-sub-ratio case of the paper's eq. 14 is not implemented); use "
            "data.reference_unit_weights: true or data.absolute_weights: true."
        )
    tgt = np.flatnonzero(y == 1.0)
    neg_t = tgt[w[tgt] < 0]
    pos_t = tgt[w[tgt] >= 0]
    ref = np.flatnonzero(y == 0.0)
    if len(neg_t) < rcfg["min_negative_events"]:
        raise ValueError(
            f"rosmm: only {len(neg_t)} negative-weight target events (rosmm.min_negative_events "
            f"= {rcfg['min_negative_events']}); r- cannot be learned from so few. Train this "
            "template without rosmm."
        )
    train_t = np.asarray(splits.train)[y[splits.train] == 1.0]
    c_hat = mixture_coefficient(w[train_t])
    frac = float(len(neg_t) / len(tgt))
    print(
        f"[{parent.run_name}] RoSMM: {len(pos_t):,} positive / {len(neg_t):,} negative target "
        f"events ({100 * frac:.1f}% negative); c = Σw+/Σw = {c_hat:.6f} (train split). "
        f"r = c r+ - (c-1) r-",
        flush=True,
    )
    parent.record.update(rosmm={
        "c_hat": c_hat, "tune": rcfg["tune"], "t0": rcfg["t0"], "t1": rcfg["t1"],
        "n_positive_target": int(len(pos_t)), "n_negative_target": int(len(neg_t)),
        "children": {p: {"dir": p, "name": f"{parent.run_name}_{p}"} for p in PARTS},
        "status": "training sub-ratios",
    })

    children: dict = {}
    child_results: dict = {}
    for part, t_idx in (("pos", pos_t), ("neg", neg_t)):
        idx = np.sort(np.concatenate([t_idx, ref]))
        sub_w = w[idx].copy()
        if part == "neg":
            sub_w[y[idx] == 1.0] = np.abs(sub_w[y[idx] == 1.0])
        sub = subset_dataset(dataset, idx, sub_w)
        sub_splits = remap_splits(splits, idx, n)
        sub_folds = None if folds is None else np.asarray(folds)[idx]
        print(f"\n[{parent.run_name}] ==== sub-ratio r{'+' if part == 'pos' else '-'} "
              f"({part}): {int((sub.y.numpy() == 1).sum()):,} target vs "
              f"{len(ref):,} reference events ====", flush=True)
        child = Pipeline(child_config(parent.config, part))
        child_results[part] = child.run_prepared(
            sub, sub_splits, sub_folds,
            preprocessing=parent.record.data["preprocessing"],
            split_record=parent.record.data.get("split", {}),
            # The parent hashed the reference before scaling; reuse that hash so
            # the children match every other template trained against it.
            extra_record={"rosmm_part": part, "rosmm_parent": parent.run_name,
                          "reference_fingerprint": parent.record.data["reference_fingerprint"]},
        )
        children[part] = child
        del sub

    ec = parent.ensemble.config
    base = RoSMMEnsemble(children["pos"].ensemble, children["neg"].ensemble, c_hat,
                         ec.combiner, ec.combiner_trim)
    hist = _child_histories(base.pos, "pos") + _child_histories(base.neg, "neg")
    metrics: dict = {}
    print(f"\n[{parent.run_name}] ==== evaluating RoSMM (c = {c_hat:.6f}) ====", flush=True)
    metrics["rosmm"] = evaluate_flavor(parent, dataset, splits, folds, base, "rosmm", hist)
    rec = {"rosmm": {"c": [c_hat] * base.n_pairs}}
    models = {"rosmm": base}

    if rcfg["tune"] in ("coefficients", "full"):
        fit_idx, where = coefficient_fit_events(splits, folds)
        xf = dataset.x.numpy()[fit_idx]
        ff = None if folds is None else np.asarray(folds)[fit_idx]
        rp = _child_combined_ratio(base.pos, xf, ff)
        rn = _child_combined_ratio(base.neg, xf, ff)
        base.release_gpu()
        fit = fit_coefficient(rp, rn, y[fit_idx], rebalanced(w[fit_idx], y[fit_idx]), c_hat,
                              rcfg["t0"], rcfg["t1"], rcfg["coefficient_range"])
        fit["events"] = where
        _plot_coefficient_scan(parent.output_dir / "evaluation" / "rosmm_coefficient_scan.pdf",
                               fit, c_hat)
        print(f"[{parent.run_name}] RoSMMc: c {c_hat:.6f} -> {fit['c_fit']:.6f} "
              f"(PARE loss {fit['loss_at_c_hat']:.8f} -> {fit['loss_at_c_fit']:.8f}, "
              f"{fit['n_events']:,} {where} events)", flush=True)
        if fit["at_search_edge"]:
            print(f"[{parent.run_name}] WARNING: the PARE minimum lies at the edge of the "
                  f"search range {fit['search_range']}; widen rosmm.coefficient_range",
                  flush=True)
        model_c = base.with_c(fit["c_fit"])
        print(f"\n[{parent.run_name}] ==== evaluating RoSMMc ====", flush=True)
        metrics["rosmmc"] = evaluate_flavor(parent, dataset, splits, folds, model_c, "rosmmc",
                                            hist)
        rec["rosmmc"] = {"c": [fit["c_fit"]] * base.n_pairs,
                         "fit": {k: v for k, v in fit.items() if k not in ("grid", "grid_loss")}}
        models["rosmmc"] = model_c

        if rcfg["tune"] == "full":
            tuned, rec_r, hist_r = finetune_all(parent, dataset, splits, folds, model_c, rcfg)
            print(f"\n[{parent.run_name}] ==== evaluating RoSMMr ====", flush=True)
            metrics["rosmmr"] = evaluate_flavor(parent, dataset, splits, folds, tuned, "rosmmr",
                                                hist_r)
            rec["rosmmr"] = rec_r
            models["rosmmr"] = tuned

    rosmm_rec = dict(parent.record.data["rosmm"])
    rosmm_rec.update(status="done", flavors=rec, default_flavor=FLAVOR_OF[rcfg["tune"]],
                     combiner=ec.combiner, combiner_trim=ec.combiner_trim)
    parent.record.update(rosmm=rosmm_rec, final_metrics=metrics)
    return {"metrics": metrics, "models": models, "children": child_results,
            "c_hat": c_hat, "folds": folds}
