"""Neural resampler for negative MC weights (Nachman & Thaler, arXiv:2007.11586).

Why
---
CARL only ever sees the *signed density* of a sample in its input features x,

    p(x) = p_unweighted(x) * <w | x>,

so replacing every weight w_i by an estimate of ``<w | x_i>`` leaves the
population loss -- and therefore s/(1-s) = p_target/p_ref -- unchanged, while
removing the negative weights that make the weighted BCE unbounded below. This
is exact in the feature space the conditional mean is taken in, which is why
the resampler uses the CARL ``data.features`` by default: averaging over
anything the network does not see (generator-level variables, the NLO
subtraction structure) is free.

What is computed (per input file)
---------------------------------
Two regressions on the event features, both with k-fold *cross-fitting* (every
event is predicted by a network that never saw it, so the network cannot just
memorise individual signed weights):

    m1(x) ~ <w   | x>      the local mean weight
    m2(x) ~ <w^2 | x>      the local second moment

and then, following Nachman & Thaler,

    K(x) = m2(x) / m1(x)^2            (>= 1 by Jensen; clipped to [1, k_max])

``mode: resample`` (default, the paper's method)
    keep each event with probability 1/K(x_i); kept events get
    w_pos = m1(x_i) * K(x_i), dropped events get w_pos = 0. Per phase-space
    patch this preserves both sum(w) and sum(w^2), i.e. the yield AND the
    statistical uncertainty of the original sample.

``mode: mean``
    w_pos = m1(x_i) for every event. Same expectation, no events dropped and
    lower variance -- but sum(w_pos^2) underestimates the MC stat uncertainty
    of the original sample. Usually better for *training* only; do not use the
    result for MC stat uncertainties.

Dropped events are kept in the file with ``weight_positive = 0`` so that the
branch stays aligned with every other branch; a zero weight simply drops out
of the weighted CARL loss.

Two losses are available for the regressions:

``loss: mse`` (default)
    f(x) = softplus(net(x)), minimise mean (f - w)^2. The minimiser is
    <w|x>, the loss is bounded below even with signed w, and the softplus
    makes the estimate positive by construction. This is the stable choice.

``loss: bce`` (the paper's original formulation)
    a classifier between the sample weighted by w and the *same* events with
    unit weight; <w|x> = g/(1-g) = exp(logit). This is itself a weighted BCE
    with signed weights, i.e. it can show exactly the instability you are
    trying to get rid of. Provided for comparison / fidelity to the paper.

``method: cell`` (Andersen & Maier, arXiv:2109.07851)
    No networks. For every negative-weight seed (random order), add nearest
    neighbours in the scaled feature space until the cell's summed weight is
    >= 0, then set each member to |w_i| * sum(w) / sum(|w|) over the cell. The
    total weight of every cell is conserved exactly. Its bias is smearing on
    the scale of a cell -- effectively a convolution, which broadens the
    distributions (deficit in the core, excess in the tails). In many
    dimensions cells stay wide even with few members: look at
    ``cells_<file>.png``, which shows how far the cells reach in every feature.
    Options: ``cell_feature_scales`` (stretch a feature's axis to make cells
    narrower in it, e.g. {n_jets: 10} keeps cells at one jet multiplicity),
    ``cell_max_radius``, ``cell_k`` / ``cell_k_retry``.

Usage
-----
Point it at an ordinary CARL config; it reads ``data.target_paths``,
``data.features`` and ``data.weight_key`` and an optional ``resampler:``
section (all keys optional)::

    resampler:
      output_key: weight_positive
      method: neural          # neural | cell
      mode: resample          # resample | mean   (neural only)
      loss: mse               # mse | bce
      k_folds: 5              # cross-fitting folds
      val_fraction: 0.1       # of the training folds, for early stopping
      n_layers: 4
      n_nodes: 256
      batch_size: 16384
      learning_rate: 1.0e-3
      max_epochs: 400
      patience: 40             # the MSE val loss is noise dominated; stop late
      k_max: 100.0            # cap on K; capped events keep the mean, lose some variance
      preserve_total: true    # rescale so sum(w_pos) == sum(w) exactly
      n_members: 1            # networks averaged per fold (lowers the variance of <w|x>)
      sentinel_below: null    # e.g. -900: values below are "undefined" fill values (see below)
      seed: 1234
      device: null            # default: cuda:<training.gpus[0]> if available, else cpu
      plots: true             # closure histograms per feature

Sentinel values. Features such as ``dijet_invmass = -999`` for events without
two jets wreck the standardisation: the physical range is squashed into a
sliver next to the sentinel spike and the regression cannot resolve it. With
``sentinel_below`` set, sentinel entries of a feature are mapped to a fill
value just below that feature's physical range (min - 1 std) before scaling.
The map is one-to-one, so the regression conditions on exactly the same
information as CARL and <w|x> is still taken in the CARL feature space.

    python -m nsbi_carl.resample --config configs/qq_4l.yaml
    python -m nsbi_carl.resample --config configs/qq_4l.yaml --files other.h5 --overwrite

Then train with ``data.weight_key: weight_positive``.

Outputs: the ``weight_positive`` dataset in every input file (with the settings
as HDF5 attributes), ``<output_dir>/resampler/resampler_<file>.yaml`` with the
diagnostics, closure plots ``<output_dir>/resampler/closure_<file>.png``
(signed ``weight`` vs ``weight_positive`` in every feature), train/validation
loss curves ``loss_m1_<file>.png`` and ``loss_m2_<file>.png`` (one panel per
fold, one line pair per member) and the raw curves in
``loss_history_<file>.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .config import _to_builtin, load_config

METHODS = ("neural", "cell")
MODES = ("resample", "mean")
LOSSES = ("mse", "bce")


@dataclass
class ResamplerConfig:
    output_key: str = "weight_positive"
    method: str = "neural"
    mode: str = "resample"
    loss: str = "mse"
    k_folds: int = 5
    val_fraction: float = 0.1
    n_layers: int = 4
    n_nodes: int = 256
    batch_size: int = 16384
    learning_rate: float = 1e-3
    max_epochs: int = 400
    patience: int = 40
    k_max: float = 100.0
    m1_floor: float = 1e-6          # in units of the sample's mean weight
    n_members: int = 1
    sentinel_below: float | None = None
    # --- method: cell (Andersen & Maier, arXiv:2109.07851) ---
    cell_max_radius: float | None = None         # in scaled-feature units; None = no cap
    cell_feature_scales: dict | None = None      # {feature: factor} on the standardised coordinate
    cell_k: int = 32                             # neighbours precomputed per seed (first pass)
    cell_k_retry: int = 1024                     # neighbours for seeds the first pass could not close
    preserve_total: bool = True
    seed: int = 1234
    device: str | None = None
    plots: bool = True

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"resampler.method must be one of {METHODS}, got {self.method!r}")
        if self.cell_k < 1 or self.cell_k_retry < self.cell_k:
            raise ValueError("resampler.cell_k must be >= 1 and cell_k_retry >= cell_k.")
        if self.cell_max_radius is not None and self.cell_max_radius <= 0:
            raise ValueError("resampler.cell_max_radius must be positive (or null).")
        if self.mode not in MODES:
            raise ValueError(f"resampler.mode must be one of {MODES}, got {self.mode!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"resampler.loss must be one of {LOSSES}, got {self.loss!r}")
        if self.k_folds < 2:
            raise ValueError("resampler.k_folds must be >= 2 (cross-fitting is not optional).")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("resampler.val_fraction must be in (0, 0.5).")
        if self.k_max < 1.0:
            raise ValueError("resampler.k_max must be >= 1.")
        if self.n_members < 1:
            raise ValueError("resampler.n_members must be >= 1.")

    @classmethod
    def from_dict(cls, d: dict | None) -> "ResamplerConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise KeyError(f"Unknown resampler option(s) {unknown}. Known: {sorted(known)}")
        return cls(**d)


# ---------------------------------------------------------------------------
# regression of <t | x> with k-fold cross-fitting
# ---------------------------------------------------------------------------
def _mlp(n_in: int, n_layers: int, n_nodes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, n_nodes), nn.SiLU()]
        d = n_nodes
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _loss(z: torch.Tensor, t: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-event loss whose population minimiser encodes <t | x>.

    mse: softplus(z) -> <t|x>.
    bce: event once in class 1 with weight t, once in class 0 with weight 1;
         exp(z) -> <t|x> (t is normalised to mean 1, so the classes balance).
    """
    if kind == "mse":
        return (F.softplus(z) - t) ** 2
    return t * F.softplus(-z) + F.softplus(z)


def _predict(net: nn.Module, x: torch.Tensor, kind: str, batch: int) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z = net(x[i:i + batch]).squeeze(-1)
            out.append((F.softplus(z) if kind == "mse" else torch.exp(z.clamp(max=30.0))).double().cpu())
    return torch.cat(out).numpy()


def _fit(x_tr, t_tr, x_va, t_va, cfg: ResamplerConfig, seed: int, tag: str,
         history: list | None = None) -> nn.Module:
    torch.manual_seed(seed)
    net = _mlp(x_tr.shape[1], cfg.n_layers, cfg.n_nodes).to(x_tr.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(cfg.patience // 3, 1))
    gen = torch.Generator(device=x_tr.device).manual_seed(seed)

    best, best_state, bad, best_epoch = float("inf"), None, 0, 0
    train_curve, val_curve, lr_curve = [], [], []
    n = len(x_tr)
    for epoch in range(cfg.max_epochs):
        net.train()
        perm = torch.randperm(n, device=x_tr.device, generator=gen)
        running = torch.zeros((), device=x_tr.device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = _loss(net(x_tr[idx]).squeeze(-1), t_tr[idx], cfg.loss).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            running += loss.detach() * len(idx)   # stays on the device; one sync per epoch
        train_curve.append(running.item() / n)

        net.eval()
        with torch.no_grad():
            v = sum(
                _loss(net(x_va[i:i + 4 * cfg.batch_size]).squeeze(-1), t_va[i:i + 4 * cfg.batch_size], cfg.loss).sum().item()
                for i in range(0, len(x_va), 4 * cfg.batch_size)
            ) / len(x_va)
        if not np.isfinite(v):
            raise FloatingPointError(
                f"[{tag}] validation loss is {v} at epoch {epoch + 1}. With loss: bce and "
                "negative weights this is the instability the resampler exists to avoid; "
                "use loss: mse or a lower learning_rate."
            )
        val_curve.append(v)
        lr_curve.append(opt.param_groups[0]["lr"])
        sched.step(v)
        if best_state is None or v < best - 1e-7 * abs(best):
            best, bad, best_epoch = v, 0, epoch
            best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    print(f"    [{tag}] stopped after {epoch + 1} epochs, best val loss {best:.6g}", flush=True)
    if history is not None:
        history.append({"tag": tag, "train": train_curve, "val": val_curve,
                        "lr": lr_curve, "best_epoch": best_epoch})
    net.load_state_dict(best_state)
    return net


def cross_fit(x: np.ndarray, t: np.ndarray, cfg: ResamplerConfig, device: torch.device,
              seed: int, tag: str, history: list | None = None) -> np.ndarray:
    """Out-of-fold estimate of <t | x> for every event. If ``history`` is a
    list, one entry per trained network (fold, member, per-epoch losses) is
    appended to it."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(t)) % cfg.k_folds
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    tt = torch.as_tensor(t, dtype=torch.float32, device=device)
    out = np.empty(len(t), dtype=np.float64)
    for k in range(cfg.k_folds):
        hold = np.flatnonzero(fold == k)
        pool = rng.permutation(np.flatnonzero(fold != k))
        n_va = max(int(round(cfg.val_fraction * len(pool))), 1)
        va, tr = (torch.as_tensor(a, device=device) for a in (pool[:n_va], pool[n_va:]))
        xh = xt[torch.as_tensor(hold, device=device)]
        pred = np.zeros(len(hold))
        for m in range(cfg.n_members):
            mtag = f"{tag} fold {k + 1}/{cfg.k_folds}" + (f" member {m + 1}/{cfg.n_members}" if cfg.n_members > 1 else "")
            n_before = len(history) if history is not None else 0
            net = _fit(xt[tr], tt[tr], xt[va], tt[va], cfg, seed + 1000 * (k + 1) + 17 * m, mtag, history)
            if history is not None and len(history) > n_before:
                history[-1].update(fold=k, member=m)
            pred += _predict(net, xh, cfg.loss, 4 * cfg.batch_size)
        out[hold] = pred / cfg.n_members
    return out


# ---------------------------------------------------------------------------
# the resampling itself (pure numpy, testable without a network)
# ---------------------------------------------------------------------------
def positive_weights(w: np.ndarray, m1: np.ndarray, m2: np.ndarray, cfg: ResamplerConfig,
                     rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Turn the moment estimates into the new weights. All inputs in the
    original weight units; returns (w_pos, diagnostics)."""
    scale = float(np.mean(np.abs(w)))
    floor = cfg.m1_floor * scale
    n_floor = int((m1 < floor).sum())
    m1 = np.maximum(m1, floor)

    if cfg.mode == "mean":
        w_pos = m1.copy()
        K = np.ones_like(m1)
        n_capped = 0
    else:
        K_raw = m2 / m1 ** 2
        n_capped = int((K_raw > cfg.k_max).sum())
        K = np.clip(K_raw, 1.0, cfg.k_max)
        keep = rng.random(len(w)) < 1.0 / K
        w_pos = np.where(keep, m1 * K, 0.0)

    total_scale = 1.0
    if cfg.preserve_total:
        if w_pos.sum() <= 0:
            raise ValueError("Sum of the new weights is not positive; nothing to rescale.")
        total_scale = float(w.sum() / w_pos.sum())
        w_pos = w_pos * total_scale

    def n_eff(a):
        return float(a.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0

    diag = {
        "n_events": int(len(w)),
        "n_negative_before": int((w < 0).sum()),
        "negative_fraction_before": float((w < 0).mean()),
        "sum_w_before": float(w.sum()),
        "sum_w2_before": float((w ** 2).sum()),
        "sum_w_after": float(w_pos.sum()),
        "sum_w2_after": float((w_pos ** 2).sum()),
        "n_eff_before": n_eff(w),
        "n_eff_after": n_eff(w_pos),
        "n_kept": int((w_pos > 0).sum()),
        "n_dropped": int((w_pos == 0).sum()),
        "mean_K": float(K.mean()),
        "n_K_capped": n_capped,
        "n_m1_floored": n_floor,
        "total_rescale": total_scale,
    }
    return w_pos, diag


def map_sentinels(x: np.ndarray, below: float | None, names: list[str]) -> np.ndarray:
    """Move sentinel entries (< ``below``) of each feature to just below its
    physical range. One-to-one, so no information is lost."""
    x = x.copy()
    if below is None:
        return x
    for j in range(x.shape[1]):
        sent = x[:, j] < below
        if sent.any() and (~sent).any():
            real = x[~sent, j]
            fill = real.min() - max(real.std(), 1e-12)
            x[sent, j] = fill
            print(f"[resampler]   {names[j]}: {sent.mean():.1%} sentinel values -> {fill:.4g}", flush=True)
    return x


def plot_loss_curves(history: list, title: str, path: Path) -> None:
    """One panel per fold: train (solid) and validation (dashed) loss per
    epoch for every member, with the restored best epoch marked. Panels have
    their own y range -- with heavy-tailed weights the loss level depends on
    which outliers landed in a fold, so only the shape is comparable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    folds = sorted({h["fold"] for h in history})
    ncol = min(len(folds), 5)
    nrow = int(np.ceil(len(folds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, k in zip(axes.flat, folds):
        runs = [h for h in history if h["fold"] == k]
        for h in runs:
            c = colors[h["member"] % len(colors)]
            ep = np.arange(1, len(h["train"]) + 1)
            ax.plot(ep, h["train"], color=c, lw=1.0, label=f"train, member {h['member'] + 1}")
            ax.plot(ep, h["val"], color=c, lw=1.0, ls="--", label=f"val, member {h['member'] + 1}")
            ax.axvline(h["best_epoch"] + 1, color=c, lw=0.6, ls=":")
        vals = np.concatenate([np.r_[h["train"], h["val"]] for h in runs])
        if (vals > 0).all() and vals.max() / vals.min() > 20:
            ax.set_yscale("log")   # only when the range is wide; narrow log axes are unreadable
        ax.set_title(f"fold {k + 1}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(folds):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6, ncol=1 if len(history) <= len(folds) else 2)
    fig.suptitle(f"{title}   (solid: train, dashed: val, dotted: best epoch)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# cell resampling (Andersen & Maier, arXiv:2109.07851)
# ---------------------------------------------------------------------------
def knn(X: torch.Tensor, queries: np.ndarray, k: int, q_chunk: int = 1024,
        n_chunk: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours (Euclidean) of ``X[queries]`` among all of
    ``X``, the query point itself included, sorted by distance. Brute force on
    the device in (q_chunk x n_chunk) tiles with a running top-k, so memory
    stays bounded (~q_chunk * n_chunk * 4 bytes)."""
    N = X.shape[0]
    k = min(k, N)
    X2 = (X * X).sum(1)
    dist = np.empty((len(queries), k), dtype=np.float32)
    idx = np.empty((len(queries), k), dtype=np.int64)
    q_all = torch.as_tensor(queries, device=X.device)
    for a in range(0, len(queries), q_chunk):
        q = q_all[a:a + q_chunk]
        Q, Q2 = X[q], X2[q]
        best_d = torch.full((len(q), k), float("inf"), device=X.device)
        best_i = torch.full((len(q), k), -1, dtype=torch.long, device=X.device)
        for b in range(0, N, n_chunk):
            d = (Q2[:, None] + X2[None, b:b + n_chunk] - 2.0 * Q @ X[b:b + n_chunk].T).clamp_min_(0.0)
            kk = min(k, d.shape[1])
            dv, di = d.topk(kk, dim=1, largest=False)
            cat_d = torch.cat([best_d, dv], 1)
            cat_i = torch.cat([best_i, di + b], 1)
            best_d, order = cat_d.topk(k, dim=1, largest=False)
            best_i = cat_i.gather(1, order)
        # The expansion |a|^2 + |b|^2 - 2ab loses precision in float32 for close
        # pairs (self-distance ~1e-3 instead of 0). It is only used to select the
        # candidates; their distances are recomputed directly and re-sorted.
        exact = (X[best_i] - Q[:, None, :]).norm(dim=-1)
        exact, order = exact.sort(dim=1)
        dist[a:a + len(q)] = exact.cpu().numpy()
        idx[a:a + len(q)] = best_i.gather(1, order).cpu().numpy()
    return dist, idx


def _close_cells(w, x_phys, seeds, nb_idx, nb_dist, max_radius, cells, rows_of, group):
    """Sequentially build cells around ``seeds`` (in the given order) from
    their precomputed neighbour lists, updating ``w`` in place. Returns the
    seeds whose cell could not be closed within the list / radius."""
    unresolved = []
    for s in seeds:
        if w[s] >= 0.0:                 # already absorbed by an earlier cell
            continue
        r = rows_of[s]
        nb, nd = nb_idx[r], nb_dist[r]
        keep = nb != s
        if max_radius is not None:
            keep &= nd <= max_radius
        nb, nd = nb[keep], nd[keep]
        cs = w[s] + np.cumsum(w[nb])
        hit = np.flatnonzero(cs >= 0.0)
        if hit.size == 0:
            unresolved.append(s)
            continue
        j = hit[0]
        members = np.concatenate(([s], nb[:j + 1]))
        a = np.abs(w[members])
        total, abs_total = cs[j], a.sum()
        w[members] = a * (total / abs_total) if abs_total > 0 else 0.0
        fresh = members[group[members] < 0]          # first cell an event joins owns it
        group[fresh] = len(cells["size"])
        cells["size"].append(j + 2)
        cells["radius"].append(float(nd[j]))
        cells["spread"].append(np.abs(x_phys[members] - x_phys[s]).max(axis=0))
    return np.asarray(unresolved, dtype=np.int64)


def cell_resample(x_phys: np.ndarray, xs: np.ndarray, w: np.ndarray, names: list[str],
                  cfg: "ResamplerConfig", device: torch.device, seed: int) -> tuple[np.ndarray, dict]:
    """Andersen-Maier cell resampling in the (scaled) CARL feature space.

    For every negative-weight seed (random order): add nearest neighbours in
    increasing distance until the cell's summed weight is >= 0, then set each
    member to |w_i| * sum(w) / sum(|w|) over the cell. The cell total is
    conserved exactly, every member ends up >= 0. Positions never change, so
    the neighbour lists are computed once, up front, on ``device``."""
    w = w.astype(np.float64).copy()
    w_in = w.copy()
    scales = np.ones(xs.shape[1])
    for name, f in (cfg.cell_feature_scales or {}).items():
        if name not in names:
            raise KeyError(f"resampler.cell_feature_scales: unknown feature {name!r}")
        scales[names.index(name)] = float(f)
    X = torch.as_tensor(xs * scales, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    seeds = rng.permutation(np.flatnonzero(w < 0))
    cells = {"size": [], "radius": [], "spread": []}
    print(f"[resampler]   {len(seeds):,} negative seeds; neighbour search (k={cfg.cell_k}) on {device}",
          flush=True)

    rows_of = np.full(len(w), -1, dtype=np.int64)
    rows_of[seeds] = np.arange(len(seeds))
    group = np.full(len(w), -1, dtype=np.int64)
    d1, i1 = knn(X, seeds, cfg.cell_k + 1)
    left = _close_cells(w, x_phys, seeds, i1, d1, cfg.cell_max_radius, cells, rows_of, group)
    del d1, i1

    n_retry = len(left)
    if n_retry:
        print(f"[resampler]   {n_retry:,} cells need more than {cfg.cell_k} neighbours; "
              f"retrying with k={cfg.cell_k_retry}", flush=True)
        rows_of[left] = np.arange(len(left))
        d2, i2 = knn(X, left, cfg.cell_k_retry + 1, q_chunk=256)
        left = _close_cells(w, x_phys, left, i2, d2, cfg.cell_max_radius, cells, rows_of, group)
    left = left[w[left] < 0] if len(left) else left

    size = np.asarray(cells["size"])
    radius = np.asarray(cells["radius"])
    spread = np.asarray(cells["spread"]).reshape(-1, len(names))

    def q(a, ps=(50, 90, 99)):
        return {f"p{p}": float(np.percentile(a, p)) for p in ps} if len(a) else {}

    def n_eff(a):
        return float(a.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0

    diag = {
        "n_events": int(len(w)),
        "n_negative_before": int((w_in < 0).sum()),
        "negative_fraction_before": float((w_in < 0).mean()),
        "n_negative_after": int((w < 0).sum()),
        "negative_weight_left": float(w[w < 0].sum()),
        "n_cells": int(len(size)),
        "n_retry": int(n_retry),
        "n_unresolved": int(len(left)),
        "n_events_changed": int((w != w_in).sum()),
        "cell_size": q(size),
        "cell_radius_scaled": q(radius),
        "cell_spread_physical": {n: q(spread[:, j]) for j, n in enumerate(names)},
        "sum_w_before": float(w_in.sum()),
        "sum_w_after": float(w.sum()),
        "sum_w2_before": float((w_in ** 2).sum()),
        "sum_w2_after": float((w ** 2).sum()),
        "n_eff_before": n_eff(w_in),
        "n_eff_after": n_eff(w),
        "n_kept": int((w > 0).sum()),
        "n_dropped": int((w == 0).sum()),
    }
    diag["_cells"] = {"size": size, "radius": radius, "spread": spread, "group": group}
    return w, diag


def plot_cells(cells: dict, names: list[str], feature_std: np.ndarray, title: str, path: Path) -> None:
    """Cell size and radius distributions, and how far cells reach in every
    feature relative to that feature's spread (median and 90% quantile)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    size, radius, spread = cells["size"], cells["radius"], cells["spread"]
    if not len(size):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(size, bins=np.arange(1.5, size.max() + 1.5), log=True)
    axes[0].set_xlabel("events per cell")
    axes[0].set_ylabel("cells")
    axes[1].hist(radius, bins=60, log=True)
    axes[1].set_xlabel("cell radius (scaled-feature units)")
    rel = spread / np.where(feature_std > 0, feature_std, 1.0)
    pos = np.arange(len(names))
    axes[2].bar(pos - 0.2, np.median(rel, 0), 0.4, label="median")
    axes[2].bar(pos + 0.2, np.percentile(rel, 90, 0), 0.4, label="90% quantile")
    axes[2].set_xticks(pos, names, rotation=60, ha="right", fontsize=8)
    axes[2].set_ylabel("max |x_member - x_seed| / std(x)")
    axes[2].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# closure plots
# ---------------------------------------------------------------------------
def _paired_var(col, edges, d, group):
    """Per-bin variance of sum(d). Events in the same ``group`` (a cell) are
    correlated -- their changes sum to zero -- so their contributions inside
    a bin are summed before squaring. group < 0 (or None): independent."""
    if group is None:
        return np.histogram(col, edges, weights=d ** 2)[0]
    b = np.searchsorted(edges, col, side="right") - 1
    b[col == edges[-1]] = len(edges) - 2
    inside = (b >= 0) & (b < len(edges) - 1)
    b, d, g = b[inside], d[inside], group[inside]
    solo = g < 0
    var = np.bincount(b[solo], weights=d[solo] ** 2, minlength=len(edges) - 1)
    if (~solo).any():
        key = g[~solo] * (len(edges) - 1) + b[~solo]
        uk, inv = np.unique(key, return_inverse=True)
        sums = np.bincount(inv, weights=d[~solo])
        var += np.bincount(uk % (len(edges) - 1), weights=sums ** 2, minlength=len(edges) - 1)
    return var


def closure(x: np.ndarray, names: list[str], w: np.ndarray, w_pos: np.ndarray,
            path: Path | None, sentinel_below: float | None = None,
            group: np.ndarray | None = None) -> dict:
    """Per-feature chi2/ndf between the signed and the new weights.

    Both histograms are filled from the SAME events, so their difference is
    tested with the paired per-event difference d = w_pos - w: the
    uncertainty of a bin's difference is sqrt(sum d^2). Adding the two
    histograms' own errors in quadrature instead would ignore the strong
    positive correlation and make the test far too forgiving. For cell
    resampling the changes inside one cell are themselves correlated (they
    sum to zero), so with ``group`` (cell id per event) they are summed per
    cell and bin before squaring.

    Sentinel entries (< ``sentinel_below``) are left out of the binning so
    the physical range gets all the bins; their total is checked as a
    separate one-bin entry ``<name>[sentinel]``."""
    chi2 = {}
    panels = []
    d = w_pos - w
    for j, name in enumerate(names):
        col = x[:, j]
        if sentinel_below is not None:
            sent = col < sentinel_below
            if sent.any() and (~sent).any():
                hs, hp = w[sent].sum(), w_pos[sent].sum()
                vd = _paired_var(np.zeros(sent.sum()), np.array([-1.0, 1.0]), d[sent],
                                 None if group is None else group[sent])[0]
                chi2[f"{name}[sentinel]"] = float((hp - hs) ** 2 / vd) if vd > 0 else 0.0
                col = col[~sent]
                wj, wpj, dj = w[~sent], w_pos[~sent], d[~sent]
                gj = None if group is None else group[~sent]
            else:
                wj, wpj, dj, gj = w, w_pos, d, group
        else:
            wj, wpj, dj, gj = w, w_pos, d, group
        uniq = np.unique(col)
        if len(uniq) <= 25:
            edges = np.concatenate([uniq - 0.5, [uniq[-1] + 0.5]]) if np.allclose(uniq, np.round(uniq)) \
                else np.unique(np.concatenate([uniq, [uniq[-1] + 1e-9]]))
        else:
            lo, hi = np.percentile(col, [0.5, 99.5])
            edges = np.linspace(lo, hi, 41)
        h0, _ = np.histogram(col, edges, weights=wj)
        h1, _ = np.histogram(col, edges, weights=wpj)
        var = _paired_var(col, edges, dj, gj)
        ok = var > 0
        chi2[name] = float(((h0 - h1)[ok] ** 2 / var[ok]).sum() / max(ok.sum(), 1))
        panels.append((name, edges, h0, h1, var))

    if path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return chi2
        n = len(panels)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(4.4 * ncol, 4.8 * nrow))
        outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.3)
        for i, (name, edges, h0, h1, var) in enumerate(panels):
            r, c = divmod(i, ncol)
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(inner[0])
            rx = fig.add_subplot(inner[1], sharex=ax)
            ax.stairs(h0, edges, label="signed weight", color="black")
            ax.stairs(h1, edges, label="weight_positive", color="tab:blue")
            ax.set_title(f"{name}  (chi2/ndf {chi2[name]:.2f})", fontsize=9)
            ax.tick_params(labelbottom=False)
            if i == 0:
                ax.legend(fontsize=8)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = h1 / h0
                err = np.sqrt(var) / np.abs(h0)
            centers = 0.5 * (edges[1:] + edges[:-1])
            rx.errorbar(centers, ratio, yerr=err, fmt=".", ms=3, color="tab:blue")
            rx.axhline(1.0, color="black", lw=0.8)
            rx.set_ylim(0.85, 1.15)
            rx.set_xlabel(name, fontsize=8)
            rx.set_ylabel("pos / signed", fontsize=8)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return chi2


# ---------------------------------------------------------------------------
def _device(cfg: ResamplerConfig, full_config: dict) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        gpus = (full_config.get("training", {}) or {}).get("gpus") or [0]
        return torch.device(f"cuda:{int(gpus[0])}")
    return torch.device("cpu")


def resample_file(path: str | Path, features: list[str], weight_key: str, cfg: ResamplerConfig,
                  device: torch.device, out_dir: Path, seed: int, overwrite: bool = False) -> dict:
    path = Path(path)
    with h5.File(path, "r") as f:
        missing = [k for k in features + [weight_key] if k not in f]
        if missing:
            raise KeyError(f"{path} is missing keys {missing}")
        if cfg.output_key in f and not overwrite:
            raise FileExistsError(
                f"{path} already has '{cfg.output_key}'. Pass --overwrite to replace it."
            )
        x = np.stack([f[k][:] for k in features], axis=1).astype(np.float64)
        w = f[weight_key][:].astype(np.float64)
    if len(w) != len(x):
        raise ValueError(f"{path}: weight length != feature length")
    if w.sum() <= 0:
        raise ValueError(f"{path}: total weight {w.sum():.6g} is not positive; "
                         "a positive local mean weight cannot exist.")

    print(f"[resampler] {path.name}: {len(w):,} events, "
          f"{(w < 0).mean():.2%} negative weights, sum(w) = {w.sum():.6g}", flush=True)

    x_mapped = map_sentinels(x, cfg.sentinel_below, features)
    mean = x_mapped.mean(axis=0)
    std = x_mapped.std(axis=0)
    std[std == 0.0] = 1.0
    xs = (x_mapped - mean) / std

    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.method == "cell":
        # cell reach is measured on the sentinel-mapped features, so a cell that
        # pairs a -999 entry with a physical value does not report a reach of ~1000
        w_pos, diag = cell_resample(x_mapped, xs, w, features, cfg, device, seed)
        cells = diag.pop("_cells")
        if cfg.plots:
            plot_cells(cells, features, x_mapped.std(axis=0), f"{path.name}: cell resampling",
                       out_dir / f"cells_{path.stem}.png")
        return _finish(path, x, w, w_pos, diag, features, weight_key, cfg, out_dir, seed,
                       "cell resampling (Andersen & Maier, arXiv:2109.07851)", group=cells["group"])

    # Work in units of the mean weight so the regression targets are O(1)
    # and the bce classes are balanced.
    s1 = float(w.mean())
    u = w / s1
    histories: dict[str, list] = {"m1": [], "m2": []}
    m1 = cross_fit(xs, u, cfg, device, seed, "<w|x>", histories["m1"]) * s1

    if cfg.mode == "resample":
        s2 = float((w ** 2).mean())
        m2 = cross_fit(xs, w ** 2 / s2, cfg, device, seed + 7, "<w^2|x>", histories["m2"]) * s2
    else:
        m2 = m1 ** 2

    w_pos, diag = positive_weights(w, m1, m2, cfg, np.random.default_rng(seed + 13))

    with open(out_dir / f"loss_history_{path.stem}.json", "w") as f:
        json.dump(histories, f)
    if cfg.plots:
        plot_loss_curves(histories["m1"], f"{path.name}: <w|x> regression ({cfg.loss})",
                         out_dir / f"loss_m1_{path.stem}.png")
        if histories["m2"]:
            plot_loss_curves(histories["m2"], f"{path.name}: <w^2|x> regression ({cfg.loss})",
                             out_dir / f"loss_m2_{path.stem}.png")
    return _finish(path, x, w, w_pos, diag, features, weight_key, cfg, out_dir, seed,
                   "neural resampler (Nachman & Thaler, arXiv:2007.11586)")


def _finish(path, x, w, w_pos, diag, features, weight_key, cfg, out_dir, seed, method_label,
            group: np.ndarray | None = None) -> dict:
    """Closure test, h5 output, report and summary printout (both methods)."""
    chi2 = closure(x, features, w, w_pos,
                   out_dir / f"closure_{path.stem}.png" if cfg.plots else None,
                   sentinel_below=cfg.sentinel_below, group=group)
    diag["closure_chi2_ndf"] = chi2

    with h5.File(path, "a") as f:
        if cfg.output_key in f:
            del f[cfg.output_key]
        ds = f.create_dataset(cfg.output_key, data=w_pos.astype(np.float64))
        ds.attrs["method"] = method_label
        ds.attrs["source_weight"] = weight_key
        ds.attrs["features"] = json.dumps(list(features))
        ds.attrs["config"] = json.dumps(asdict(cfg))
        ds.attrs["seed"] = int(seed)

    report = {"file": str(path), "weight_key": weight_key, "features": list(features),
              "seed": int(seed), "config": asdict(cfg), **diag}
    with open(out_dir / f"resampler_{path.stem}.yaml", "w") as f:
        yaml.safe_dump(_to_builtin(report), f, sort_keys=False)

    print(f"[resampler] {path.name}: wrote '{cfg.output_key}'  "
          f"kept {diag['n_kept']:,}/{diag['n_events']:,}  "
          f"N_eff {diag['n_eff_before']:.4g} -> {diag['n_eff_after']:.4g}  "
          f"sum(w^2) {diag['sum_w2_before']:.4g} -> {diag['sum_w2_after']:.4g}", flush=True)
    print("[resampler]   closure chi2/ndf (paired errors): " + ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(chi2.items(), key=lambda kv: -kv[1])), flush=True)
    if cfg.method == "cell":
        cs = diag["cell_size"]
        print(f"[resampler]   {diag['n_cells']:,} cells, size median {cs.get('p50', 0):.0f} / "
              f"p90 {cs.get('p90', 0):.0f} / p99 {cs.get('p99', 0):.0f}; "
              f"{diag['n_events_changed']:,} events reweighted", flush=True)
        if diag["n_unresolved"]:
            print(f"[resampler]   WARNING: {diag['n_unresolved']:,} negative seeds could not be "
                  f"closed (cell_max_radius / cell_k_retry); {diag['n_negative_after']:,} events "
                  f"keep negative weights, summing to {diag['negative_weight_left']:.4g}.", flush=True)
    if diag.get("n_m1_floored"):
        print(f"[resampler]   NOTE: {diag['n_m1_floored']:,} events had <w|x> <= 0 (floored). "
              "The signed density is not positive there and no positive weighting can "
              "reproduce it; check the closure plots in that region.", flush=True)
    if diag.get("n_K_capped"):
        print(f"[resampler]   NOTE: {diag['n_K_capped']:,} events had K > k_max={cfg.k_max:g}; "
              "their mean is preserved, their variance slightly underestimated.", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Neural resampler: add a positive weight branch to .h5 samples.")
    p.add_argument("--config", required=True, help="CARL pipeline config (data.* and optional resampler:).")
    p.add_argument("--files", nargs="*", default=None, help="Override data.target_paths.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output branch.")
    args = p.parse_args(argv)

    config = load_config(args.config)
    data = config["data"]
    cfg = ResamplerConfig.from_dict(config.get("resampler"))
    files = args.files if args.files else data["target_paths"]
    if isinstance(files, str):
        files = [files]
    features = list(data["features"])
    weight_key = data.get("weight_key", "weight")
    if weight_key == cfg.output_key:
        raise ValueError(
            f"data.weight_key is already '{weight_key}'. The resampler reads the ORIGINAL signed "
            "weight; point weight_key back at it (or override resampler.output_key)."
        )
    out_dir = Path(config.get("output_dir", "outputs")) / "resampler"
    device = _device(cfg, config)
    print(f"[resampler] device {device}, mode {cfg.mode}, loss {cfg.loss}, "
          f"{cfg.k_folds}-fold cross-fitting, features {features}", flush=True)

    for i, path in enumerate(files):
        resample_file(path, features, weight_key, cfg, device, out_dir,
                      seed=cfg.seed + 101 * i, overwrite=args.overwrite)
    print(f"[resampler] done. Set data.weight_key: {cfg.output_key} to train on it.", flush=True)


if __name__ == "__main__":
    main()
