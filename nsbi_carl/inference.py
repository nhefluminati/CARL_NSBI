"""Loading trained templates for inference (fits, NLL scans, Neyman toys).

Downstream analysis code needs three things from a finished run, and all three
live in the run record rather than in the checkpoint: the feature list *and its
order*, the fitted scaler, and the reweighting that was in force during
training. Getting any of them from somewhere else — a hand-maintained list in
the analysis script, say — is how a script and its networks drift apart.

This module reads them back, and validates the assumptions an NSBI fit makes
but usually cannot check:

* **All templates share one reference.** The likelihood forms ratios of the
  templates' outputs, so a reference that differs between them biases those
  ratios. Every run writes a ``reference_fingerprint``;
  :func:`validate_templates` compares them.
* **The target and reference were balanced.** CARL's optimum is
  ``s = w_t p_t / (w_t p_t + w_r p_r)``, so ``r = s/(1-s)`` is the density
  ratio only when the two classes carried equal total weight. A template
  trained with an unbalanced factor yields ``factor * p_t/p_r``, silently
  rescaling every likelihood term.
* **The reference weights used downstream match the ones trained against.**
  :func:`reference_weights` rebuilds them from the record rather than from a
  flag the analyst has to remember to keep in sync.

Ensemble members are fused into a single stacked model, so scoring makes one
pass over the events instead of one per member — the difference between a
fast likelihood scan and a slow one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .model import CARL


@dataclass
class TemplateRecord:
    """Everything an analysis needs from ``run_config_<name>.yaml``."""

    name: str
    run_dir: Path
    features: list[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    member_tags: list[str]
    checkpoints: list[str]

    # -- reweighting as it was applied during training -------------------
    reference_unit_weights: bool = True
    absolute_weights: bool = False
    target_balance_scale: float = 1.0
    target_balance_factor: float = 1.0
    normalize_weights: bool = False
    global_normalization_scale: float = 1.0
    reference_sample_scales: dict = field(default_factory=dict)
    train_target_yield: float | None = None
    train_reference_yield: float | None = None

    # -- provenance ------------------------------------------------------
    reference_fingerprint: dict = field(default_factory=dict)
    target_paths: list[str] = field(default_factory=list)
    reference_paths: list[str] = field(default_factory=list)
    load_reference: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def class_balance(self) -> float | None:
        """Ratio of target to reference total weight on the train split."""
        if not self.train_reference_yield:
            return None
        return self.train_target_yield / self.train_reference_yield


def read_record(run_dir: str | Path, name: str) -> TemplateRecord:
    """Parse ``<run_dir>/run_config_<name>.yaml``.

    Tolerates records written by older versions of the pipeline: keys that did
    not exist then fall back to what the code did at the time, so an old run
    still loads (``validate_templates`` is what flags it as suspect).
    """
    run_dir = Path(run_dir)
    path = run_dir / f"run_config_{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"[{name}] no run record at {path}")
    rec = load_config(path) or {}

    prep = rec.get("preprocessing") or {}
    if "scaler_mean" not in prep:
        raise KeyError(f"[{name}] {path} has no preprocessing/scaler_mean; run incomplete?")

    data_cfg = ((rec.get("config") or {}).get("data")) or {}
    rw = rec.get("reweighting") or {}

    tags = list(rec.get("ensemble_members") or [])
    checkpoints = _resolve_checkpoints(run_dir, name, tags)

    return TemplateRecord(
        name=name,
        run_dir=run_dir,
        features=list(rec["features"]),
        scaler_mean=np.asarray(prep["scaler_mean"], dtype=np.float64),
        scaler_std=np.asarray(prep["scaler_std"], dtype=np.float64),
        member_tags=tags,
        checkpoints=checkpoints,
        # `reference_unit_weights` replaced an unconditional w=1; a record
        # without it came from that era, so the effective value was True.
        reference_unit_weights=bool(rw.get("reference_unit_weights", True)),
        absolute_weights=bool(data_cfg.get("absolute_weights", False)),
        target_balance_scale=float(rw.get("target_balance_scale", 1.0)),
        # Older records have no factor; infer it from the yields they do carry.
        target_balance_factor=float(
            rw.get("target_balance_factor", _infer_balance_factor(rw))
        ),
        normalize_weights=bool(rw.get("normalize_weights", False)),
        global_normalization_scale=float(rw.get("global_normalization_scale", 1.0)),
        reference_sample_scales=dict(rw.get("reference_sample_scales") or {}),
        train_target_yield=_maybe_float(rw.get("train_target_yield")),
        train_reference_yield=_maybe_float(rw.get("train_reference_yield")),
        reference_fingerprint=dict(rec.get("reference_fingerprint") or {}),
        target_paths=list(data_cfg.get("target_paths") or []),
        reference_paths=list(data_cfg.get("reference_paths") or []),
        load_reference=data_cfg.get("load_reference"),
        raw=rec,
    )


def _maybe_float(v):
    return None if v is None else float(v)


def _infer_balance_factor(rw: dict) -> float:
    """Recover the target/reference balance from the recorded yields."""
    t, r = rw.get("train_target_yield"), rw.get("train_reference_yield")
    if t is None or not r:
        return 1.0
    return float(t) / float(r)


def _resolve_checkpoints(run_dir: Path, name: str, tags: list[str]) -> list[str]:
    """Checkpoint paths for an ensemble run, or for a single-network run."""
    out: list[str] = []
    for tag in tags:
        ckpt_dir = run_dir / tag / "checkpoints"
        hits = sorted(ckpt_dir.glob("best_*.ckpt")) or sorted(ckpt_dir.glob("*.ckpt"))
        if not hits:
            raise FileNotFoundError(f"[{name}] no checkpoint for member {tag!r} under {ckpt_dir}")
        out.append(str(hits[0]))
    if out:
        return out

    for cand in (run_dir / name / "checkpoints", run_dir):
        hits = sorted(cand.glob("best_*.ckpt")) or sorted(cand.glob("**/checkpoints/*.ckpt"))
        if hits:
            return [str(hits[0])]
    raise FileNotFoundError(f"[{name}] no checkpoint found under {run_dir}")


# ---------------------------------------------------------------------------
class EnsembleScorer:
    """Scores raw (unscaled) events with a trained ensemble.

    Applies the run's own scaler, then evaluates every member in a single
    stacked pass. ``score`` returns the ensemble mean; ``member_scores``
    returns the ``(n_members, n_events)`` matrix, which is what an ensemble
    spread / systematic needs.
    """

    def __init__(self, record: TemplateRecord, device: str | torch.device = "cpu"):
        self.record = record
        self.device = torch.device(device)
        self.models = [
            CARL.load_from_checkpoint(p, map_location="cpu").eval() for p in record.checkpoints
        ]
        for m in self.models:
            for p in m.parameters():
                p.requires_grad_(False)
        self._stacked = self._build_stacked()

    @property
    def n_members(self) -> int:
        return len(self.models)

    def _build_stacked(self):
        from .training.vectorized import StackedMLP

        hp = self.models[0].hparams
        model = StackedMLP(
            n_members=len(self.models),
            n_features=hp["n_features"],
            n_layers=hp["n_layers"],
            n_nodes=hp["n_nodes"],
            dropout=hp.get("dropout", 0.0),
        )
        stride = 3 if hp.get("dropout", 0.0) > 0.0 else 2
        with torch.no_grad():
            for m, member in enumerate(self.models):
                sd = member.state_dict()
                for layer in range(len(model.weights)):
                    idx = layer * stride
                    model.weights[layer][m].copy_(sd[f"net.{idx}.weight"].t())
                    model.biases[layer][m, 0].copy_(sd[f"net.{idx}.bias"])
        return model.to(self.device).eval()

    def scale(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"[{self.record.name}] expected a 2D array, got shape {x.shape}")
        if x.shape[1] != len(self.record.features):
            raise ValueError(
                f"[{self.record.name}] expected {len(self.record.features)} features "
                f"({self.record.features}), got {x.shape[1]} columns"
            )
        return (x - self.record.scaler_mean) / self.record.scaler_std

    @torch.no_grad()
    def member_scores(self, x_raw: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        xs = torch.as_tensor(self.scale(x_raw), dtype=torch.float32)
        out = [
            torch.sigmoid(self._stacked(xs[i : i + batch_size].to(self.device))).cpu()
            for i in range(0, len(xs), batch_size)
        ]
        if not out:
            return np.zeros((self.n_members, 0))
        return torch.cat(out, dim=1).numpy()

    def score(self, x_raw: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return self.member_scores(x_raw, batch_size=batch_size).mean(axis=0)

    __call__ = score


# ---------------------------------------------------------------------------
def reference_weights(record: TemplateRecord, weight_arrays, equalize: bool = True):
    """Reference weights matching what the networks were trained against.

    ``weight_arrays`` are the physical per-sample weight arrays, in the same
    order as the reference samples. The record decides what happens to them:
    ``absolute_weights`` takes ``|w|``, ``reference_unit_weights`` replaces
    them with 1, and each sample is then equalized to unit total — exactly
    stage 1 of ``ReweightStep``.

    Deriving this from the record rather than from a flag in the analysis
    script is the point: the two cannot disagree.
    """
    parts = []
    for w in weight_arrays:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if record.absolute_weights:
            w = np.abs(w)
        if record.reference_unit_weights:
            w = np.ones_like(w)
        if equalize:
            total = w.sum()
            if total <= 0:
                raise ValueError("a reference sample has non-positive total weight")
            w = w / total
        parts.append(w)
    out = np.concatenate(parts)
    return out / out.sum()


# ---------------------------------------------------------------------------
def load_reference_sample(
    path: str | Path,
    record: TemplateRecord,
    split: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the reference events and weights from a saved reference cache.

    This is the analysis-side counterpart of ``data.save_reference``: the fit
    reads the *same file* the networks were trained against, instead of
    restacking the individual MC samples and hoping the list, the order and
    the weight handling still match. A feature list that differs from the
    cache's is rejected by the loader rather than silently permuted.

    ``split`` selects by the cached train/val/test assignment:
      * ``"all"``  — every cached event; best statistics for p_ref.
      * ``"test"`` — only events no network trained on, which is what you
        want if the closure needs to be free of any memorisation.

    Returns ``(features, weights)`` with the weights built exactly as
    :func:`reference_weights` builds them — per-sample equalisation included —
    and normalised to sum to 1.
    """
    from .data.loading import load_reference_cache
    from .data.splitting import TEST, TRAIN, VAL

    cache = load_reference_cache(path, record.features)
    x = np.asarray(cache["x"], dtype=np.float64)
    w = np.asarray(cache["w"], dtype=np.float64)
    sid = np.asarray(cache["sample_id"], dtype=np.int64)
    lab = np.asarray(cache["split_label"], dtype=np.int8)

    key = str(split).lower()
    if key != "all":
        want = {"train": TRAIN, "val": VAL, "test": TEST}.get(key)
        if want is None:
            raise ValueError(f"split must be 'all', 'train', 'val' or 'test', got {split!r}")
        keep = lab == want
        if not keep.any():
            raise ValueError(f"reference cache {path} has no events in the {key} split")
        x, w, sid = x[keep], w[keep], sid[keep]

    # Rebuild the weights per sample, in the cache's own sample order, so the
    # per-sample equalisation matches what training applied.
    order = np.argsort(sid, kind="stable")
    x, w, sid = x[order], w[order], sid[order]
    groups = [w[sid == s] for s in np.unique(sid)]
    weights = reference_weights(record, groups)
    return x, weights


def reference_sample_groups(path: str | Path, record: TemplateRecord):
    """Per-sample ``(name, features, raw weights)`` from a reference cache.

    For code that needs the reference split back into its constituent
    samples — calibration, per-sample diagnostics — rather than as one pool.
    """
    from .data.loading import load_reference_cache

    cache = load_reference_cache(path, record.features)
    x = np.asarray(cache["x"], dtype=np.float64)
    w = np.asarray(cache["w"], dtype=np.float64)
    sid = np.asarray(cache["sample_id"], dtype=np.int64)
    names = list(cache["sample_names"])
    out = []
    for s in np.unique(sid):
        m = sid == s
        name = names[int(s)] if int(s) < len(names) else f"sample_{int(s)}"
        out.append((name, x[m], w[m]))
    return out


def validate_templates(records, balance_tol: float = 0.01, raise_on_error: bool = False) -> list[str]:
    """Check the assumptions a multi-template NSBI fit silently relies on.

    Returns the list of problems found (empty when all is well), printing each.
    These are exactly the failures that do not announce themselves: the fit
    still runs and still produces a number.
    """
    problems: list[str] = []
    records = list(records)
    if not records:
        return problems

    # -- 1) one reference for all templates ------------------------------
    prints = {r.name: r.reference_fingerprint.get("train_sha1") for r in records}
    known = {n: p for n, p in prints.items() if p}
    if not known:
        problems.append(
            "no template records carry a reference_fingerprint, so it cannot be verified "
            "that they trained against the same reference (records predate the fingerprint)"
        )
    elif len(set(known.values())) > 1:
        groups: dict[str, list[str]] = {}
        for n, p in known.items():
            groups.setdefault(p, []).append(n)
        detail = "; ".join(f"{p[:12]}...: {sorted(v)}" for p, v in groups.items())
        problems.append(
            f"templates did NOT train against the same reference sample -> {detail}. "
            "Every ratio formed between these templates is biased."
        )
    if known and len(known) < len(prints):
        missing = sorted(n for n, p in prints.items() if not p)
        problems.append(f"no reference fingerprint for {missing}; cannot confirm they match")

    # -- 2) target/reference balance -------------------------------------
    for r in records:
        bal = r.class_balance
        if bal is None:
            continue
        if abs(bal - 1.0) > balance_tol:
            problems.append(
                f"[{r.name}] trained with target/reference weight ratio {bal:.4g}, not 1. "
                f"r = s/(1-s) then estimates {bal:.4g} * p_t/p_r, so every likelihood term "
                "carries that factor. Retrain with data.target_balance_factor: 1.0."
            )

    # -- 3) feature order must agree across templates ---------------------
    feats = {r.name: tuple(r.features) for r in records}
    if len(set(feats.values())) > 1:
        problems.append(f"templates disagree on the feature list/order: {feats}")

    # -- 4) reference construction must agree ------------------------------
    for key in ("reference_unit_weights", "absolute_weights"):
        vals = {r.name: getattr(r, key) for r in records}
        if len(set(vals.values())) > 1:
            problems.append(f"templates disagree on {key}: {vals}")

    for p in problems:
        print(f"  *** {p}")
    if problems and raise_on_error:
        raise ValueError(f"{len(problems)} template validation problem(s); see above.")
    if not problems:
        ref = next(iter(known.values()), "")
        n_ev = records[0].reference_fingerprint.get("train_n")
        print(
            f"  all {len(records)} templates share reference {ref[:12]}... "
            f"({n_ev} train events) and were trained with balanced classes"
        )
    return problems


def load_templates(spec, device: str | torch.device = "cpu", validate: bool = True):
    """Load several templates at once.

    ``spec`` maps template name -> run directory. Returns
    ``{name: EnsembleScorer}`` and, unless disabled, validates them together
    first — which is the only moment the whole set is in view.
    """
    records = [read_record(run_dir, name) for name, run_dir in spec.items()]
    if validate:
        print("\n== TEMPLATE VALIDATION ==")
        validate_templates(records)
    return {r.name: EnsembleScorer(r, device=device) for r in records}
