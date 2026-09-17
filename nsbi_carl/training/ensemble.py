"""Neural-network ensemble.

All members are pooled from ONE common dataset: each member trains on a
bootstrap resample of the shared train split (validation split is shared and
untouched). Member ``i`` uses seed ``start_seed + i``, so any member is
reproducible in isolation and runs can be extended later.

* Training reuses :class:`CARLTrainer` per member and adds parallelism: the
  members are distributed round-robin over the configured GPUs with a
  process pool.
* ``inference(x)`` / ``predict(dataset, idx)`` return the ensembled output:
  the average of the member predictions.
* After training, the *ensemble-level* evaluation suite runs once on the
  test split; the :class:`EvaluationContext` then carries ``member_scores``
  so metrics/plots can use the full ensemble information.
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
from ..evaluation.base import EvaluationContext, EvaluationSuite
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
    train_idx = (
        bootstrap_train_indices(
            splits, labels, seed, econf.bootstrap_fraction, econf.bootstrap_reference
        )
        if econf.bootstrap
        else splits.train
    )
    summary = trainer.fit(dataset, splits, train_idx=train_idx, seed=seed, tag=f"member_{member_id:03d}")
    summary["member"] = member_id
    summary["gpu"] = gpu_id
    return summary


def _save_snapshot(path: Path, dataset: NSBIDataset, splits: SplitIndices) -> None:
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
) -> list[dict]:
    """Train a whole group of members together, vectorized, on one GPU."""
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
        amp_dtype="bf16" if tconf.precision.startswith("bf16") else "none",
    )
    trainer = VectorizedEnsembleTrainer(
        config=vconf, output_dir=output_dir, run_name=run_name, device=device
    )
    return trainer.train(
        dataset,
        splits,
        member_ids=list(member_ids),
        seeds=list(seeds),
        bootstrap=econf.bootstrap,
        bootstrap_fraction=econf.bootstrap_fraction,
        bootstrap_reference=econf.bootstrap_reference,
        model_config=ModelConfig(**model_config),
    )


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

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"ensemble_manifest_{self.run_name}.json"

    # -- training -------------------------------------------------------
    def train(self, dataset: NSBIDataset, splits: SplitIndices) -> list[dict]:
        cfg = self.config
        member_ids = list(range(cfg.start_member, cfg.n_members))
        if not member_ids:
            return self._merge_manifest([])

        gpus = self.trainer.config.gpus or [None]
        snapshot = self.output_dir / f"dataset_snapshot_{self.run_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _save_snapshot(snapshot, dataset, splits)

        common = dict(
            snapshot_path=str(snapshot),
            model_config=asdict(self.trainer.model_config),
            trainer_config={**asdict(self.trainer.config), "gpus": []},
            ensemble_config=asdict(cfg),
            output_dir=str(self.output_dir),
            run_name=self.run_name,
        )

        if cfg.mode == "vectorized":
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

    def _merge_manifest(self, new_summaries: list[dict]) -> list[dict]:
        members: dict[int, dict] = {}
        if self.manifest_path.exists():  # keep previously trained members
            with open(self.manifest_path) as f:
                members = {m["member"]: m for m in json.load(f)["members"]}
        members.update({s["member"]: s for s in new_summaries})
        ordered = [members[k] for k in sorted(members)]
        self.manifest = {"run_name": self.run_name, "n_members": len(ordered), "members": ordered}
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
        return self

    @torch.no_grad()
    def member_predictions(self, x: torch.Tensor | np.ndarray, device: str | None = None,
                           batch_size: int = 262144, stacked: bool = True) -> np.ndarray:
        """(n_members, n_events) matrix of member scores for SCALED features.

        With ``stacked`` (the default) the members are fused into a single
        batched module and every member is evaluated in the same pass, which
        is what the likelihood scan wants: it turns M sequential sweeps over
        the events into one. The per-model loop is kept as a fallback.
        """
        if not self.models:
            self.load()
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, dtype=torch.float32)

        if stacked and len(self.models) > 1:
            model = self.stacked(device=device)
            out = [
                torch.sigmoid(model(x[i : i + batch_size].to(device))).cpu()
                for i in range(0, len(x), batch_size)
            ]
            return torch.cat(out, dim=1).numpy()

        preds = []
        for model in self.models:
            model = model.to(device).eval()
            out = [model(x[i : i + batch_size].to(device)).flatten().cpu() for i in range(0, len(x), batch_size)]
            preds.append(torch.cat(out).numpy())
            model.to("cpu")
        return np.stack(preds)

    def stacked(self, device: str | torch.device = "cpu"):
        """Fuse the loaded members into one :class:`StackedMLP` for inference."""
        from .vectorized import StackedMLP

        if self._stacked is not None and self._stacked_device == str(device):
            return self._stacked
        if not self.models:
            self.load()

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
        model = model.to(device).eval()
        self._stacked, self._stacked_device = model, str(device)
        return model

    def inference(self, x: torch.Tensor | np.ndarray, device: str | None = None) -> np.ndarray:
        """Ensembled output: average of all member predictions."""
        return self.member_predictions(x, device=device).mean(axis=0)

    __call__ = inference

    # -- ensemble-level evaluation ---------------------------------------
    def evaluate(self, dataset: NSBIDataset, idx: np.ndarray, tag: str = "ensemble_test",
                 device: str | None = None) -> dict[str, float]:
        member_scores = self.member_predictions(dataset.x.numpy()[idx], device=device)
        histories = [
            {"member": m["member"], "train_loss": m["train_loss"], "val_loss": m["val_loss"]}
            for m in self.manifest.get("members", [])
        ]
        ctx = EvaluationContext(
            scores=member_scores.mean(axis=0).astype(np.float64),
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
