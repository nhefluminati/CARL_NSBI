"""Neural-network ensemble.

All members are pooled from ONE common dataset: each member trains on a
bootstrap resample of the shared train split (validation split is shared and
untouched). Member ``i`` uses seed ``start_seed + i``, so any member is
reproducible in isolation and runs can be extended later.

* Training reuses :class:`CARLTrainer` per member and adds parallelism: the
  members are distributed round-robin over the configured GPUs with a
  process pool.
* ``inference(x)`` / ``predict(dataset, idx)`` return the ensembled output,
  combined as ``ensemble.combiner`` says (default: mean of member scores).
* After training, the *ensemble-level* evaluation suite runs once on the
  test split; the :class:`EvaluationContext` then carries ``member_scores``
  so metrics/plots can use the full ensemble information.

k-fold mode (``split.k_folds: k``) replaces the bootstrap pooling: every event
is assigned to one of k folds, and fold f gets its own ``n_members`` members,
each trained on a fresh train/validation split of the other k-1 folds drawn
WITHOUT replacement. Fold f's members are then the only ones that ever score
fold f's events (``inference_out_of_fold``), so no event is scored by a
network that trained on it. The bootstrap settings are ignored in this mode.
A ``split.kfold_test_fraction`` of every sample is kept out of ALL folds as a
final test set; no member trains or validates on it, so the whole ensemble
can be evaluated on it.
"""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.dataset import NSBIDataset, SplitIndices
from ..data.splitting import DEFAULT_FOLD_SEED, TEST_FOLD, KFoldStep
from ..evaluation.base import EvaluationContext, EvaluationSuite
from ..combine import combine_scores
from ..model import CARL
from .trainer import CARLTrainer, ModelConfig, TrainerConfig


@dataclass
class EnsembleConfig:
    n_members: int = 8
    start_seed: int = 100
    bootstrap: bool = True
    bootstrap_fraction: float = 1.0
    bootstrap_reference: bool = False  # resample the reference too (off: every
                                       # member shares the identical reference)
    workers_per_gpu: int = 1
    start_member: int = 0  # resume an ensemble by training only members >= this

    # -- performance -----------------------------------------------------
    # "vectorized": all members assigned to a GPU train together as one
    #   batched matmul (fastest by a wide margin for small networks).
    # "process":    the original one-member-per-process pool.
    mode: str = "vectorized"
    members_per_group: int = 0  # 0 -> all members on a GPU in one group

    # -- how member scores become one score (see nsbi_carl/combine.py) ---
    combiner: str = "mean_score"   # mean_score | mean_ratio | mean_logit |
                                   # median_ratio | trimmed_ratio
    combiner_trim: float = 0.1     # trimmed_ratio: fraction cut at EACH end

    # -- k-fold cross-validation (set from the `split:` config block) -----
    # k_folds >= 2 switches the bootstrap off: n_members is then the number
    # of members PER FOLD, and each member trains on a without-replacement
    # train/val split of the other k-1 folds.
    k_folds: int = 0
    fold_seed: int = DEFAULT_FOLD_SEED
    kfold_val_fraction: float = 0.2
    kfold_test_fraction: float = 0.1   # untouched final test set (fold -1)

    @property
    def kfold(self) -> bool:
        return self.k_folds >= 2

    def kfold_step(self) -> KFoldStep:
        return KFoldStep(
            self.k_folds, self.fold_seed, self.kfold_val_fraction, self.kfold_test_fraction
        )


def bootstrap_train_indices(
    splits: SplitIndices,
    labels: np.ndarray,
    seed: int,
    fraction: float,
    bootstrap_reference: bool = False,
) -> np.ndarray:
    """Bootstrap-resample the train split — by default the TARGET only.

    Every CARL network in the analysis is trained against the same reference,
    so the ensemble spread is meant to capture the statistical uncertainty of
    the target sample, not of the reference. Resampling the reference as well
    would make each member see a different denominator, and the ratios of
    different templates' outputs (which is what the likelihood actually uses)
    would inherit that mismatch. So the reference train split is passed
    through whole and unchanged: every member of every process trains on the
    numerically identical reference events.

    ``bootstrap_reference=True`` restores the old behaviour, in which the
    reference is resampled too with its own name-seeded generator.

    The reference events always occupy the tail of the returned array in
    their original order, so two members built from the same splits share
    byte-identical reference rows.
    """
    train = splits.train
    tgt = train[labels[train] == 1.0]
    ref = train[labels[train] == 0.0]

    rng_t = np.random.default_rng([seed, zlib.crc32(b"target")])
    n_t = max(1, int(round(fraction * len(tgt))))
    tgt_out = rng_t.choice(tgt, n_t, replace=True)

    if bootstrap_reference:
        rng_r = np.random.default_rng([seed, zlib.crc32(b"reference")])
        n_r = max(1, int(round(fraction * len(ref))))
        ref_out = rng_r.choice(ref, n_r, replace=True)
    else:
        ref_out = ref  # identical for every member, in a fixed order

    return np.concatenate([tgt_out, ref_out])


def rebalance_scale(weights: np.ndarray, labels: np.ndarray, idx: np.ndarray) -> float:
    """Factor restoring equal total weight to the two classes on ``idx``.

    The bootstrap resamples the TARGET only, so with
    ``bootstrap_fraction < 1`` the target carries only that fraction of its
    weight while the reference is passed through whole: a member trained on
    ``bootstrap_fraction: 0.8`` would see a 0.8:1 class balance and therefore
    learn ``0.8 * p_t/p_r`` instead of the density ratio. Even at fraction 1.0
    drawing with replacement leaves a small statistical imbalance.

    Multiplying the member's target weights by this factor puts the classes
    back on equal footing, so every member is a valid CARL estimator on its
    own resample. It is a per-member weight scale, which cancels from nothing
    the ensemble mean cares about.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)[idx]
    y = np.asarray(labels).reshape(-1)[idx]
    t, r = w[y == 1.0].sum(), w[y == 0.0].sum()
    if t <= 0 or r <= 0:
        return 1.0
    return float(r / t)


# ---------------------------------------------------------------------------
# Worker executed in a spawned subprocess: rebuilds the dataset from an .npz
# snapshot, trains one member on one GPU.
# ---------------------------------------------------------------------------
def _train_member_worker(
    snapshot_path: str,
    member_id: int,
    seed: int,
    gpu_id: int | None,
    model_config: dict,
    trainer_config: dict,
    ensemble_config: dict,
    output_dir: str,
    run_name: str,
    fold: int | None = None,
) -> dict:
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    if trainer_config.get("disable_native_jit"):
        os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

    dataset, splits = _load_snapshot(snapshot_path)
    tconf = TrainerConfig(**trainer_config)
    tconf.gpus = [gpu_id] if gpu_id is not None else []
    tconf.resume_from = None  # member-level resume is handled via start_member
    trainer = CARLTrainer(
        model_config=ModelConfig(**model_config),
        trainer_config=tconf,
        output_dir=output_dir,
        run_name=run_name,
    )

    econf = EnsembleConfig(**ensemble_config)
    labels = dataset.y.numpy().reshape(-1)
    if fold is not None:
        # k-fold: this member's own without-replacement split of the other
        # folds; validation on its own val rows, test = the held-out fold.
        splits = econf.kfold_step().member_indices(dataset, _load_folds(snapshot_path), fold, seed)
        train_idx = splits.train
    elif econf.bootstrap:
        train_idx = bootstrap_train_indices(
            splits, labels, seed, econf.bootstrap_fraction, econf.bootstrap_reference
        )
    else:
        train_idx = splits.train
    summary = trainer.fit(dataset, splits, train_idx=train_idx, seed=seed, tag=f"member_{member_id:03d}")
    summary["member"] = member_id
    summary["gpu"] = gpu_id
    if fold is not None:
        summary["fold"] = int(fold)
    return summary


def _save_snapshot(
    path: Path, dataset: NSBIDataset, splits: SplitIndices, folds: np.ndarray | None = None
) -> None:
    """Write the prepared dataset as a directory of plain ``.npy`` files.

    ``.npz`` forces every worker to parse and fully materialise its own copy
    of the arrays: with 8 members that is 8x the dataset in RAM and 8x the
    load time. Plain ``.npy`` files can be memory-mapped instead, so the
    workers share one set of pages from the OS cache. The feature/label/weight
    arrays are stored in the dtypes ``NSBIDataset`` wants, which makes the
    subsequent ``torch.as_tensor`` a zero-copy view of the mapping.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": dataset.x.numpy().astype(np.float32, copy=False),
        "y": dataset.y.numpy().astype(np.float32, copy=False),
        "w": dataset.w.numpy().astype(np.float64, copy=False),
        "sample_id": dataset.sample_id,
        "mean": np.asarray(dataset.mean),
        "std": np.asarray(dataset.std),
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    if folds is not None:
        arrays["folds"] = np.asarray(folds, dtype=np.int16)
    for name, arr in arrays.items():
        np.save(path / f"{name}.npy", arr)
    with open(path / "names.json", "w") as f:
        json.dump(
            {"sample_names": list(dataset.sample_names), "feature_names": list(dataset.feature_names)}, f
        )


def _load_snapshot(path: str) -> tuple[NSBIDataset, SplitIndices]:
    p = Path(path)
    load = lambda n: np.load(p / f"{n}.npy", mmap_mode="r")  # noqa: E731
    with open(p / "names.json") as f:
        names = json.load(f)
    dataset = NSBIDataset(
        x=load("x"), y=load("y"), w=load("w"), sample_id=np.asarray(load("sample_id")),
        sample_names=names["sample_names"],
        feature_names=names["feature_names"],
    )
    dataset.mean, dataset.std = np.asarray(load("mean")), np.asarray(load("std"))
    return dataset, SplitIndices(
        train=np.asarray(load("train")), val=np.asarray(load("val")), test=np.asarray(load("test"))
    )


def _load_folds(path: str) -> np.ndarray:
    p = Path(path) / "folds.npy"
    if not p.exists():
        raise FileNotFoundError(f"k-fold training needs {p}; the snapshot was written without folds")
    return np.array(np.load(p))


# ---------------------------------------------------------------------------
def _train_group_worker(
    snapshot_path: str,
    member_ids: list[int],
    seeds: list[int],
    gpu_id: int | None,
    model_config: dict,
    trainer_config: dict,
    ensemble_config: dict,
    output_dir: str,
    run_name: str,
    fold: int | None = None,
) -> list[dict]:
    """Train a whole group of members together, vectorized, on one GPU.

    With ``fold`` set (k-fold mode) every member of the group belongs to that
    fold and gets its own without-replacement train/val rows.
    """
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    if trainer_config.get("disable_native_jit"):
        os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

    import torch as _torch

    from .vectorized import VectorizedConfig, VectorizedEnsembleTrainer

    dataset, splits = _load_snapshot(snapshot_path)
    tconf = TrainerConfig(**trainer_config)
    econf = EnsembleConfig(**ensemble_config)
    device = f"cuda:{gpu_id}" if (gpu_id is not None and _torch.cuda.is_available()) else "cpu"
    _torch.set_float32_matmul_precision(tconf.matmul_precision)

    vconf = VectorizedConfig(
        batch_size=tconf.batch_size,
        val_batch_size=tconf.val_batch_size,
        learning_rate=tconf.learning_rate,
        momentum=tconf.momentum,
        weight_decay=tconf.weight_decay,
        optimizer=tconf.optimizer,
        max_epochs=tconf.max_epochs,
        early_stopping_patience=tconf.early_stopping_patience,
        scheduler_t0=model_config.get("scheduler_t0", 25),
        scheduler_t_mult=model_config.get("scheduler_t_mult", 1),
        scheduler_eta_min=model_config.get("scheduler_eta_min", 1e-8),
        compile=tconf.compile,
        log_weights=tconf.log_weights,
        amp_dtype="bf16" if tconf.precision.startswith("bf16") else "none",
    )
    trainer = VectorizedEnsembleTrainer(
        config=vconf, output_dir=output_dir, run_name=run_name, device=device
    )
    train_matrix = val_matrix = None
    if fold is not None:
        folds = _load_folds(snapshot_path)
        step = econf.kfold_step()
        member_splits = [step.member_indices(dataset, folds, fold, s) for s in seeds]
        train_matrix = np.stack([m.train for m in member_splits])
        val_matrix = np.stack([m.val for m in member_splits])
        splits = member_splits[0]
    summaries = trainer.train(
        dataset,
        splits,
        member_ids=list(member_ids),
        seeds=list(seeds),
        bootstrap=econf.bootstrap,
        bootstrap_fraction=econf.bootstrap_fraction,
        bootstrap_reference=econf.bootstrap_reference,
        model_config=ModelConfig(**model_config),
        train_matrix=train_matrix,
        val_matrix=val_matrix,
    )
    if fold is not None:
        for s in summaries:
            s["fold"] = int(fold)
    return summaries


def _run_jobs_sequentially(jobs: list[dict], common: dict) -> list[dict]:
    """Run several group jobs one after another in one process (one GPU)."""
    out: list[dict] = []
    for job in jobs:
        out.extend(_train_group_worker(**job, **common))
    return out


# ---------------------------------------------------------------------------
class CARLEnsemble:
    def __init__(
        self,
        trainer: CARLTrainer,
        ensemble_config: EnsembleConfig,
        output_dir: str | Path,
        run_name: str = "carl",
    ):
        self.trainer = trainer
        self.config = ensemble_config
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.models: list[CARL] = []
        self.manifest: dict = {}
        self._stacked = None
        self._stacked_device: str | None = None
        self._fold_stacks: dict = {}

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"ensemble_manifest_{self.run_name}.json"

    # -- training -------------------------------------------------------
    def train(
        self, dataset: NSBIDataset, splits: SplitIndices, folds: np.ndarray | None = None
    ) -> list[dict]:
        cfg = self.config
        if cfg.kfold and folds is None:
            raise ValueError("k-fold ensemble training needs the per-event fold array")
        member_ids = list(range(cfg.start_member, cfg.n_members))
        if not member_ids:
            return self._merge_manifest([])

        gpus = self.trainer.config.gpus or [None]
        snapshot = self.output_dir / f"dataset_snapshot_{self.run_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _save_snapshot(snapshot, dataset, splits, folds if cfg.kfold else None)

        common = dict(
            snapshot_path=str(snapshot),
            model_config=asdict(self.trainer.model_config),
            trainer_config={**asdict(self.trainer.config), "gpus": []},
            ensemble_config=asdict(cfg),
            output_dir=str(self.output_dir),
            run_name=self.run_name,
        )

        if cfg.kfold and cfg.mode == "vectorized":
            summaries = self._train_kfold_vectorized(member_ids, gpus, common)
        elif cfg.kfold:
            summaries = self._train_kfold_process_pool(member_ids, gpus, common)
        elif cfg.mode == "vectorized":
            summaries = self._train_vectorized(member_ids, gpus, common)
        else:
            summaries = self._train_process_pool(member_ids, gpus, common)

        summaries = self._merge_manifest(summaries)
        self.load()  # load all trained members for inference
        return summaries

    # ------------------------------------------------------------------
    def _groups(self, member_ids: list[int], gpus: list) -> list[tuple[list[int], object]]:
        """Split the members into one group per GPU (or smaller groups).

        One group == one process == one stacked model. Members inside a group
        train simultaneously; groups run in parallel across GPUs.
        """
        cfg = self.config
        per_gpu: list[list[int]] = [[] for _ in gpus]
        for i, m in enumerate(member_ids):
            per_gpu[i % len(gpus)].append(m)

        groups: list[tuple[list[int], object]] = []
        for gpu, members in zip(gpus, per_gpu):
            if not members:
                continue
            size = cfg.members_per_group or len(members)
            for s in range(0, len(members), size):
                groups.append((members[s : s + size], gpu))
        return groups

    def _train_vectorized(self, member_ids: list[int], gpus: list, common: dict) -> list[dict]:
        cfg = self.config
        groups = self._groups(member_ids, gpus)
        jobs = [
            (members, [cfg.start_seed + m for m in members], gpu) for members, gpu in groups
        ]

        if len(jobs) <= 1:
            members, seeds, gpu = jobs[0]
            return _train_group_worker(member_ids=members, seeds=seeds, gpu_id=gpu, **common)

        ctx = torch.multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            futures = [
                pool.submit(_train_group_worker, member_ids=m, seeds=s, gpu_id=g, **common)
                for m, s, g in jobs
            ]
            out: list[dict] = []
            for f in futures:
                out.extend(f.result())
        return out

    def _train_process_pool(self, member_ids: list[int], gpus: list, common: dict) -> list[dict]:
        cfg = self.config
        gpu_slots = [g for g in gpus for _ in range(cfg.workers_per_gpu)]
        assignments = [
            (m, cfg.start_seed + m, gpu_slots[i % len(gpu_slots)]) for i, m in enumerate(member_ids)
        ]
        max_workers = min(len(assignments), len(gpu_slots))
        if max_workers <= 1:
            return [
                _train_member_worker(member_id=m, seed=s, gpu_id=g, **common)
                for m, s, g in assignments
            ]
        ctx = torch.multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_train_member_worker, member_id=m, seed=s, gpu_id=g, **common)
                for m, s, g in assignments
            ]
            return [f.result() for f in futures]

    # -- k-fold -----------------------------------------------------------
    def fold_member_id(self, fold: int, member: int) -> int:
        """Global member id (and seed offset) of ``member`` in ``fold``."""
        return fold * self.config.n_members + member

    def _train_kfold_vectorized(self, member_ids: list[int], gpus: list, common: dict) -> list[dict]:
        """One vectorized group per fold (split by ``members_per_group``).

        Folds are dealt round-robin to the GPUs, and each GPU works through
        its folds one after another, so at most one group per GPU is resident.
        """
        cfg = self.config
        size = cfg.members_per_group or len(member_ids)
        jobs = []
        for fold in range(cfg.k_folds):
            gids = [self.fold_member_id(fold, m) for m in member_ids]
            for s in range(0, len(gids), size):
                chunk = gids[s : s + size]
                jobs.append(dict(member_ids=chunk, seeds=[cfg.start_seed + g for g in chunk],
                                 fold=fold))
        per_gpu: dict[int, list[dict]] = {i: [] for i in range(len(gpus))}
        for i, job in enumerate(jobs):
            slot = i % len(gpus)
            per_gpu[slot].append({**job, "gpu_id": gpus[slot]})

        work = [js for js in per_gpu.values() if js]
        if len(work) <= 1:
            return _run_jobs_sequentially(work[0], common) if work else []
        ctx = torch.multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(work), mp_context=ctx) as pool:
            futures = [pool.submit(_run_jobs_sequentially, js, common) for js in work]
            out: list[dict] = []
            for fut in futures:
                out.extend(fut.result())
        return out

    def _train_kfold_process_pool(self, member_ids: list[int], gpus: list, common: dict) -> list[dict]:
        cfg = self.config
        gpu_slots = [g for g in gpus for _ in range(cfg.workers_per_gpu)]
        assignments = []
        for fold in range(cfg.k_folds):
            for m in member_ids:
                gid = self.fold_member_id(fold, m)
                assignments.append((gid, cfg.start_seed + gid, fold))
        max_workers = min(len(assignments), len(gpu_slots))
        if max_workers <= 1:
            return [
                _train_member_worker(member_id=g, seed=s, gpu_id=gpu_slots[0], fold=f, **common)
                for g, s, f in assignments
            ]
        ctx = torch.multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_train_member_worker, member_id=g, seed=s,
                            gpu_id=gpu_slots[i % len(gpu_slots)], fold=f, **common)
                for i, (g, s, f) in enumerate(assignments)
            ]
            return [fut.result() for fut in futures]

    def fold_members(self) -> list[list[int]]:
        """Indices into ``self.models`` of each fold's members, fold by fold."""
        members = self.manifest.get("members", [])
        out: list[list[int]] = [[] for _ in range(self.config.k_folds)]
        for i, m in enumerate(members):
            if "fold" not in m:
                raise ValueError(f"manifest member {m.get('member')} carries no fold")
            out[int(m["fold"])].append(i)
        return out

    def _merge_manifest(self, new_summaries: list[dict]) -> list[dict]:
        members: dict[int, dict] = {}
        if self.manifest_path.exists():  # keep previously trained members
            with open(self.manifest_path) as f:
                members = {m["member"]: m for m in json.load(f)["members"]}
        members.update({s["member"]: s for s in new_summaries})
        ordered = [members[k] for k in sorted(members)]
        self.manifest = {"run_name": self.run_name, "n_members": len(ordered), "members": ordered}
        if self.config.kfold:
            self.manifest.update(
                k_folds=self.config.k_folds,
                fold_seed=self.config.fold_seed,
                kfold_test_fraction=self.config.kfold_test_fraction,
                members_per_fold=self.config.n_members,
            )
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return ordered

    # -- inference ------------------------------------------------------
    def load(self, map_location: str = "cpu") -> "CARLEnsemble":
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        self.models = [
            CARL.load_from_checkpoint(m["checkpoint"], map_location=map_location)
            for m in self.manifest["members"]
        ]
        self._stacked, self._stacked_device = None, None
        self._fold_stacks = {}
        return self

    @torch.no_grad()
    def member_predictions(self, x: torch.Tensor | np.ndarray, device: str | None = None,
                           batch_size: int | None = None, stacked: bool = True,
                           mem_fraction: float = 0.5) -> np.ndarray:
        """(n_members, n_events) matrix of member scores for SCALED features.

        With ``stacked`` (the default) the members are fused into a single
        batched module and every member is evaluated in the same pass, which
        is what the likelihood scan wants: it turns M sequential sweeps over
        the events into one. The per-model loop is kept as a fallback.

        ``batch_size=None`` sizes the event chunk from the free GPU memory and
        the ensemble size: the stacked forward keeps ``(M, B, n_nodes)``
        activations alive, so a fixed chunk that fits a small ensemble runs
        out of memory for a large one.
        """
        if not self.models:
            self.load()
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, dtype=torch.float32)

        if stacked and len(self.models) > 1:
            model = self.stacked(device=device)
            if batch_size is None:
                batch_size = self._auto_batch_size(model, device, mem_fraction)
            out = [
                torch.sigmoid(model(x[i : i + batch_size].to(device))).cpu()
                for i in range(0, len(x), batch_size)
            ]
            return torch.cat(out, dim=1).numpy()

        batch_size = batch_size or 65536
        preds = []
        for model in self.models:
            model = model.to(device).eval()
            out = [model(x[i : i + batch_size].to(device)).flatten().cpu() for i in range(0, len(x), batch_size)]
            preds.append(torch.cat(out).numpy())
            model.to("cpu")
        return np.stack(preds)

    @staticmethod
    def _auto_batch_size(model, device: str | torch.device, mem_fraction: float = 0.5) -> int:
        """Largest event chunk whose ``(M, B, H)`` activations fit in free GPU memory."""
        dev = torch.device(device)
        if dev.type != "cuda":
            return 65536
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(dev)
        width = max(model.n_nodes, model.n_features)
        per_event = 3 * model.n_members * width * 4  # ~3 live fp32 (M, B, H) tensors per layer
        return int(max(1024, min(262144, mem_fraction * free // per_event)))

    def stacked(self, device: str | torch.device = "cpu"):
        """Fuse the loaded members into one :class:`StackedMLP` for inference."""
        if self._stacked is not None and self._stacked_device == str(device):
            return self._stacked
        if not self.models:
            self.load()
        model = stack_members(self.models, device)
        self._stacked, self._stacked_device = model, str(device)
        return model

    # -- k-fold inference -------------------------------------------------
    @torch.no_grad()
    def out_of_fold_member_predictions(
        self, x: torch.Tensor | np.ndarray, folds: np.ndarray, device: str | None = None,
        batch_size: int = 262144,
    ) -> np.ndarray:
        """``(members_per_fold, n_events)`` scores for SCALED features, where
        column i comes only from the members of event i's own fold -- the
        networks that never trained on it. Row m is member m of that fold."""
        if not self.config.kfold:
            raise ValueError("out-of-fold prediction needs a k-fold ensemble")
        if not self.models:
            self.load()
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        per_fold = self.fold_members()
        sizes = {len(p) for p in per_fold}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(f"folds have unequal/empty member counts {[len(p) for p in per_fold]}")
        x = torch.as_tensor(x, dtype=torch.float32)
        folds = np.asarray(folds).reshape(-1)
        if np.any(folds == TEST_FOLD):
            raise ValueError(
                "events of the untouched test set (fold -1) have no 'own fold'; every member "
                "is out-of-sample for them, so score them with member_predictions()/inference()"
            )
        out = np.empty((sizes.pop(), len(x)), dtype=np.float32)
        for f, members in enumerate(per_fold):
            sel = np.flatnonzero(folds == f)
            if sel.size == 0:
                continue
            key = (f, str(device))
            if key not in self._fold_stacks:
                self._fold_stacks[key] = stack_members([self.models[i] for i in members], device)
            model = self._fold_stacks[key]
            xs = x[torch.as_tensor(sel)]
            out[:, sel] = torch.cat(
                [torch.sigmoid(model(xs[i : i + batch_size].to(device))).cpu()
                 for i in range(0, len(xs), batch_size)], dim=1
            ).numpy()
        return out

    def inference_out_of_fold(self, x, folds, device: str | None = None) -> np.ndarray:
        """Combined score of every event using only networks that never saw it:
        its own fold's members, or ALL members for the test set (fold -1)."""
        folds = np.asarray(folds).reshape(-1)
        out = np.empty(len(folds), dtype=np.float64)
        test = folds == TEST_FOLD
        x = np.asarray(x)
        if test.any():
            out[test] = self.inference(x[test], device=device)
        if (~test).any():
            out[~test] = combine_scores(
                self.out_of_fold_member_predictions(x[~test], folds[~test], device=device),
                self.config.combiner, self.config.combiner_trim,
            )
        return out

    def inference(self, x: torch.Tensor | np.ndarray, device: str | None = None) -> np.ndarray:
        """Ensembled output, combined per ``EnsembleConfig.combiner``."""
        return combine_scores(self.member_predictions(x, device=device),
                              self.config.combiner, self.config.combiner_trim)

    __call__ = inference

    # -- ensemble-level evaluation ---------------------------------------
    def evaluate(self, dataset: NSBIDataset, idx: np.ndarray, tag: str = "ensemble_test",
                 device: str | None = None, folds: np.ndarray | None = None) -> dict:
        """Run the after-training suite on ``idx``.

        k-fold mode (pass ``folds``, and usually every event as ``idx``) runs
        it twice:

        * ``<tag>`` on the untouched test set (fold -1), scored by the WHOLE
          ensemble -- no member ever trained or validated on these events;
        * ``<tag>_out_of_fold`` on the cross-validated events, each scored by
          its own fold's members only.

        and returns ``{"test": ..., "out_of_fold": ...}``.
        """
        if self.config.kfold:
            if folds is None:
                raise ValueError("k-fold evaluation needs the per-event fold array")
            idx = np.asarray(idx)
            f = np.asarray(folds)[idx]
            results: dict = {}
            test_idx, cv_idx = idx[f == TEST_FOLD], idx[f != TEST_FOLD]
            if test_idx.size:
                results["test"] = self._run_suite(
                    dataset, test_idx,
                    self.member_predictions(dataset.x.numpy()[test_idx], device=device), tag,
                )
            if cv_idx.size:
                results["out_of_fold"] = self._run_suite(
                    dataset, cv_idx,
                    self.out_of_fold_member_predictions(
                        dataset.x.numpy()[cv_idx], np.asarray(folds)[cv_idx], device=device
                    ),
                    f"{tag}_out_of_fold",
                )
            return results

        member_scores = self.member_predictions(dataset.x.numpy()[idx], device=device)
        return self._run_suite(dataset, idx, member_scores, tag)

    def _run_suite(self, dataset: NSBIDataset, idx: np.ndarray, member_scores: np.ndarray,
                   tag: str) -> dict:
        histories = [
            {"member": m["member"], "train_loss": m["train_loss"], "val_loss": m["val_loss"]}
            for m in self.manifest.get("members", [])
        ]
        ctx = EvaluationContext(
            scores=combine_scores(member_scores, self.config.combiner,
                                  self.config.combiner_trim).astype(np.float64),
            labels=dataset.y.numpy().reshape(-1)[idx],
            weights=dataset.w.numpy().reshape(-1)[idx].astype(np.float64),
            features=dataset.x.numpy()[idx],
            feature_names=dataset.feature_names,
            mean=dataset.mean,
            std=dataset.std,
            member_scores=member_scores,
            histories=histories,
            output_dir=self.output_dir / "evaluation",
            tag=tag,
        )
        results = self.trainer.final_evaluation.run(ctx)
        if results:
            out = self.output_dir / "evaluation"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"metrics_{tag}.json", "w") as f:
                json.dump(results, f, indent=2)
        return results


def stack_members(models: list, device: str | torch.device = "cpu"):
    """Fuse CARL members into one :class:`StackedMLP` (one pass for all)."""
    from .vectorized import StackedMLP

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
    return model.to(device).eval()
