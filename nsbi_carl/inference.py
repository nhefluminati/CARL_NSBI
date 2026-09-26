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

from .combine import DEFAULT_COMBINER, combine_scores, validate_combiner
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

    # -- ensemble combination (ensemble.combiner in the training config) --
    combiner: str = DEFAULT_COMBINER
    combiner_trim: float = 0.1

    # -- k-fold cross-validation (split.k_folds in the training config) ---
    # k_folds == 0 means the run used the fixed split + bootstrap pooling.
    # fold_members[f] indexes into ``checkpoints``: the members of fold f,
    # i.e. the ONLY networks allowed to score fold f's events.
    k_folds: int = 0
    fold_seed: int | None = None
    kfold_test_fraction: float = 0.0     # share kept as the untouched test set (fold -1)
    fold_members: list = field(default_factory=list)

    @property
    def kfold(self) -> bool:
        return self.k_folds >= 2

    def folds_for(self, sample_name: str, n_events: int) -> np.ndarray:
        """Fold of every row of the sample file ``sample_name`` (its basename):
        -1 for the untouched test set, else 0..k-1.

        Rows must be in the file's own order: the assignment is a function of
        (fold_seed, file name, row index), identical to what training used.
        """
        if not self.kfold:
            raise ValueError(f"[{self.name}] this run was not trained with k-fold")
        from .data.splitting import fold_assignment

        return fold_assignment(
            Path(sample_name).name, int(n_events), self.k_folds, self.fold_seed,
            self.kfold_test_fraction,
        )

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
    ens_cfg = ((rec.get("config") or {}).get("ensemble")) or {}
    rw = rec.get("reweighting") or {}

    tags = list(rec.get("ensemble_members") or [])
    checkpoints = _resolve_checkpoints(run_dir, name, tags)

    kf = rec.get("kfold") or {}
    fold_members: list[list[int]] = []
    if kf:
        pos = {t: i for i, t in enumerate(tags)}
        missing = [t for fold in kf["members"] for t in fold if t not in pos]
        if missing:
            raise KeyError(f"[{name}] k-fold members {missing[:5]} not in ensemble_members")
        fold_members = [[pos[t] for t in fold] for fold in kf["members"]]

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
        # Records written before combiners existed averaged the scores.
        combiner=str(ens_cfg.get("combiner", DEFAULT_COMBINER)),
        combiner_trim=float(ens_cfg.get("combiner_trim", 0.1)),
        k_folds=int(kf.get("k_folds", 0)),
        fold_seed=kf.get("fold_seed"),
        # Runs written before the test set existed had none.
        kfold_test_fraction=float(kf.get("kfold_test_fraction", 0.0)),
        fold_members=fold_members,
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
    stacked pass. ``score`` returns the combined score (see
    :mod:`nsbi_carl.combine`); ``member_scores`` returns the
    ``(n_members, n_events)`` matrix, which is what an ensemble spread /
    systematic needs.

    The combiner defaults to the one in the run record (``ensemble.combiner``
    at training time). Pass ``combiner=`` / ``combiner_trim=`` to override it
    at analysis time without retraining; ``set_combiner`` changes it later.

    k-fold runs: ``score_out_of_fold(x, folds)`` scores each event only with
    networks that never saw it -- its own fold's members, or ALL members for
    the untouched test set (fold -1). Use it for every MC sample (the Asimov,
    the reference pool, closure tests), with the folds from
    ``record.folds_for(file_name, n_rows)`` or ``load_reference_sample(...,
    return_folds=True)``. ``score`` pools ALL members of all folds, which is
    right only for events no network has seen: real data, or the test set.
    """

    def __init__(
        self,
        record: TemplateRecord,
        device: str | torch.device = "cpu",
        combiner: str | None = None,
        combiner_trim: float | None = None,
    ):
        self.record = record
        self.device = torch.device(device)
        self.set_combiner(combiner, combiner_trim)
        self.models = [
            CARL.load_from_checkpoint(p, map_location="cpu").eval() for p in record.checkpoints
        ]
        for m in self.models:
            for p in m.parameters():
                p.requires_grad_(False)
        if record.kfold:
            if len(record.fold_members) != record.k_folds or not all(record.fold_members):
                raise ValueError(f"[{record.name}] run record lists incomplete k-fold members")
            self._stacks = [self._build_stacked(idx) for idx in record.fold_members]
        else:
            self._stacks = [self._build_stacked(list(range(len(self.models))))]
        # One scoring pass evaluates one of these stacks at a time; exposed as
        # `_stacked` for code that sizes batches from its shape.
        self._stacked = self._stacks[0]

    def set_combiner(self, combiner: str | None = None, trim: float | None = None) -> None:
        self.combiner_trim = float(self.record.combiner_trim if trim is None else trim)
        self.combiner = validate_combiner(
            self.record.combiner if combiner is None else combiner, self.combiner_trim
        )

    @property
    def n_members(self) -> int:
        return len(self.models)

    def _build_stacked(self, member_idx: list[int]):
        from .training.vectorized import StackedMLP

        models = [self.models[i] for i in member_idx]
        hp = models[0].hparams
        model = StackedMLP(
            n_members=len(models),
            n_features=hp["n_features"],
            n_layers=hp["n_layers"],
            n_nodes=hp["n_nodes"],
            dropout=hp.get("dropout", 0.0),
        )
        stride = 3 if hp.get("dropout", 0.0) > 0.0 else 2
        with torch.no_grad():
            for m, member in enumerate(models):
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
    def _batches(self, xs: torch.Tensor, stacks, batch_size: int):
        """Yield ``(start, (M, B) scores)`` with ``stacks`` concatenated."""
        for i in range(0, len(xs), batch_size):
            xb = xs[i : i + batch_size].to(self.device)
            yield i, torch.cat([torch.sigmoid(s(xb)) for s in stacks], dim=0).cpu().numpy()

    def member_scores(self, x_raw: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        """``(n_members, n_events)``; in a k-fold run, all folds' members."""
        xs = torch.as_tensor(self.scale(x_raw), dtype=torch.float32)
        out = [b for _, b in self._batches(xs, self._stacks, batch_size)]
        if not out:
            return np.zeros((self.n_members, 0))
        return np.concatenate(out, axis=1)

    def score(self, x_raw: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        """Combined score from ALL members (all folds, for a k-fold run).

        Combined batch by batch, so the full member matrix is never held.
        """
        xs = torch.as_tensor(self.scale(x_raw), dtype=torch.float32)
        out = np.empty(len(xs), dtype=np.float64)
        for i, b in self._batches(xs, self._stacks, batch_size):
            out[i : i + b.shape[1]] = combine_scores(b, self.combiner, self.combiner_trim)
        return out

    def fold_member_scores(
        self, x_raw: np.ndarray, folds: np.ndarray, batch_size: int = 200_000
    ) -> np.ndarray:
        """``(members_per_fold, n_events)``: column i from event i's own fold.

        Not defined for test-set events (fold -1): every member is
        out-of-sample for them; use :meth:`member_scores`.
        """
        folds = self._check_folds(x_raw, folds)
        if np.any(folds < 0):
            raise ValueError(
                f"[{self.record.name}] test-set events (fold -1) have no own fold; "
                "use member_scores() for them"
            )
        sizes = {len(m) for m in self.record.fold_members}
        if len(sizes) != 1:
            raise ValueError(f"[{self.record.name}] folds have unequal member counts")
        out = np.empty((sizes.pop(), len(folds)), dtype=np.float32)
        for f, stack in enumerate(self._stacks):
            sel = np.flatnonzero(folds == f)
            if sel.size:
                xs = torch.as_tensor(self.scale(np.asarray(x_raw)[sel]), dtype=torch.float32)
                for i, b in self._batches(xs, [stack], batch_size):
                    out[:, sel[i : i + b.shape[1]]] = b
        return out

    def score_out_of_fold(
        self, x_raw: np.ndarray, folds: np.ndarray, batch_size: int = 200_000
    ) -> np.ndarray:
        """Combined score of every event using only networks that never saw
        it: its own fold's members, or ALL members for the test set (fold -1).
        Use this for all MC."""
        folds = self._check_folds(x_raw, folds)
        out = np.empty(len(folds), dtype=np.float64)
        test = np.flatnonzero(folds < 0)
        if test.size:
            out[test] = self.score(np.asarray(x_raw)[test], batch_size=batch_size)
        for f, stack in enumerate(self._stacks):
            sel = np.flatnonzero(folds == f)
            if sel.size:
                xs = torch.as_tensor(self.scale(np.asarray(x_raw)[sel]), dtype=torch.float32)
                for i, b in self._batches(xs, [stack], batch_size):
                    out[sel[i : i + b.shape[1]]] = combine_scores(
                        b, self.combiner, self.combiner_trim
                    )
        return out

    def _check_folds(self, x_raw, folds) -> np.ndarray:
        if not self.record.kfold:
            raise ValueError(f"[{self.record.name}] out-of-fold scoring needs a k-fold run")
        folds = np.asarray(folds).reshape(-1)
        if len(folds) != len(x_raw):
            raise ValueError(f"[{self.record.name}] {len(folds)} folds for {len(x_raw)} events")
        if folds.size and (folds.min() < -1 or folds.max() >= self.record.k_folds):
            raise ValueError(
                f"[{self.record.name}] fold ids outside -1..{self.record.k_folds - 1}"
            )
        return folds

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
    return_folds: bool = False,
):
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

    For a k-fold ``record`` the selection follows the FOLDS instead of the
    cached labels: ``"test"`` is the untouched test set (fold -1), ``"cv"``
    the cross-validated events, ``"all"`` both. (``"train"``/``"val"`` have
    no meaning there.)

    Returns ``(features, weights)`` with the weights built exactly as
    :func:`reference_weights` builds them — per-sample equalisation included —
    and normalised to sum to 1. With ``return_folds=True`` (k-fold runs)
    returns ``(features, weights, folds)``, the folds aligned with the rows,
    for :meth:`EnsembleScorer.score_out_of_fold`. In a k-fold run use
    ``split="all"``: out-of-fold scoring is what keeps it clean.
    """
    from .data.loading import load_reference_cache
    from .data.splitting import TEST, TRAIN, VAL

    cache = load_reference_cache(path, record.features)
    x = np.asarray(cache["x"], dtype=np.float64)
    w = np.asarray(cache["w"], dtype=np.float64)
    sid = np.asarray(cache["sample_id"], dtype=np.int64)
    lab = np.asarray(cache["split_label"], dtype=np.int8)
    if record.kfold:
        folds = _cache_folds(cache, record)
    else:
        folds = np.zeros(len(sid), np.int16)

    key = str(split).lower()
    if record.kfold and key != "all":
        keep = {"test": folds < 0, "cv": folds >= 0}.get(key)
        if keep is None:
            raise ValueError(f"for a k-fold run split must be 'all', 'cv' or 'test', got {split!r}")
        if not keep.any():
            raise ValueError(f"no reference events in the k-fold {key} set "
                             f"(kfold_test_fraction={record.kfold_test_fraction})")
        x, w, sid, folds = x[keep], w[keep], sid[keep], folds[keep]
    elif key != "all":
        want = {"train": TRAIN, "val": VAL, "test": TEST}.get(key)
        if want is None:
            raise ValueError(f"split must be 'all', 'train', 'val' or 'test', got {split!r}")
        keep = lab == want
        if not keep.any():
            raise ValueError(f"reference cache {path} has no events in the {key} split")
        x, w, sid, folds = x[keep], w[keep], sid[keep], folds[keep]

    # Rebuild the weights per sample, in the cache's own sample order, so the
    # per-sample equalisation matches what training applied.
    order = np.argsort(sid, kind="stable")
    x, w, sid, folds = x[order], w[order], sid[order], folds[order]
    groups = [w[sid == s] for s in np.unique(sid)]
    weights = reference_weights(record, groups)
    if return_folds:
        return x, weights, folds
    return x, weights


def _cache_folds(cache: dict, record: TemplateRecord) -> np.ndarray:
    """Fold of every cached reference event.

    The cache stores each sample's events in the order of its source file,
    so an event's row within its sample is its row in that file -- the same
    index training used when it assigned folds.
    """
    sid = np.asarray(cache["sample_id"], dtype=np.int64)
    names = list(cache["sample_names"])
    folds = np.empty(len(sid), dtype=np.int16)
    for s in np.unique(sid):
        rows = np.flatnonzero(sid == s)
        folds[rows] = record.folds_for(names[int(s)], rows.size)
    return folds


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

    # -- 5) one fold scheme for all templates -----------------------------
    # A reference event is only scored cleanly if it sits in the same fold
    # for every template, which needs the same k and fold_seed everywhere.
    schemes = {
        r.name: (r.k_folds, r.fold_seed, r.kfold_test_fraction) if r.kfold else (0, None, None)
        for r in records
    }
    if len(set(schemes.values())) > 1:
        problems.append(
            f"templates disagree on the k-fold scheme (k_folds, fold_seed, "
            f"kfold_test_fraction): {schemes}. Out-of-fold scoring of the shared reference "
            "is then not clean, and the test sets are not the same events."
        )

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
