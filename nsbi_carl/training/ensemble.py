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
    workers_per_gpu: int = 1
    start_member: int = 0  # resume an ensemble by training only members >= this


def bootstrap_train_indices(
    splits: SplitIndices, labels: np.ndarray, seed: int, fraction: float
) -> np.ndarray:
    """Bootstrap-resample the train split, target and reference separately.

    Separate, name-seeded generators keep the reference draws independent of
    the target sample — the same guarantee the splitter gives.
    """
    train = splits.train
    tgt = train[labels[train] == 1.0]
    ref = train[labels[train] == 0.0]
    rng_t = np.random.default_rng([seed, zlib.crc32(b"target")])
    rng_r = np.random.default_rng([seed, zlib.crc32(b"reference")])
    n_t = max(1, int(round(fraction * len(tgt))))
    n_r = max(1, int(round(fraction * len(ref))))
    out = np.concatenate([rng_t.choice(tgt, n_t, replace=True), rng_r.choice(ref, n_r, replace=True)])
    return np.random.default_rng(seed).permutation(out)


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
        bootstrap_train_indices(splits, labels, seed, econf.bootstrap_fraction)
        if econf.bootstrap
        else splits.train
    )
    summary = trainer.fit(dataset, splits, train_idx=train_idx, seed=seed, tag=f"member_{member_id:03d}")
    summary["member"] = member_id
    summary["gpu"] = gpu_id
    return summary


def _save_snapshot(path: Path, dataset: NSBIDataset, splits: SplitIndices) -> None:
    np.savez(
        path,
        x=dataset.x.numpy(),
        y=dataset.y.numpy(),
        w=dataset.w.numpy(),
        sample_id=dataset.sample_id,
        sample_names=np.array(dataset.sample_names),
        feature_names=np.array(dataset.feature_names),
        mean=dataset.mean,
        std=dataset.std,
        train=splits.train,
        val=splits.val,
        test=splits.test,
    )


def _load_snapshot(path: str) -> tuple[NSBIDataset, SplitIndices]:
    z = np.load(path, allow_pickle=False)
    dataset = NSBIDataset(
        x=z["x"], y=z["y"], w=z["w"], sample_id=z["sample_id"],
        sample_names=[str(s) for s in z["sample_names"]],
        feature_names=[str(s) for s in z["feature_names"]],
    )
    dataset.mean, dataset.std = z["mean"], z["std"]
    return dataset, SplitIndices(train=z["train"], val=z["val"], test=z["test"])


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

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"ensemble_manifest_{self.run_name}.json"

    # -- training -------------------------------------------------------
    def train(self, dataset: NSBIDataset, splits: SplitIndices) -> list[dict]:
        cfg = self.config
        member_ids = list(range(cfg.start_member, cfg.n_members))
        gpus = self.trainer.config.gpus or [None]
        gpu_slots = [g for g in gpus for _ in range(cfg.workers_per_gpu)]
        assignments = [
            (m, cfg.start_seed + m, gpu_slots[i % len(gpu_slots)]) for i, m in enumerate(member_ids)
        ]

        snapshot = self.output_dir / f"dataset_snapshot_{self.run_name}.npz"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _save_snapshot(snapshot, dataset, splits)

        worker_args = dict(
            snapshot_path=str(snapshot),
            model_config=asdict(self.trainer.model_config),
            trainer_config={**asdict(self.trainer.config), "gpus": []},
            ensemble_config=asdict(cfg),
            output_dir=str(self.output_dir),
            run_name=self.run_name,
        )

        max_workers = min(len(assignments), len(gpu_slots))
        if max_workers <= 1:
            summaries = [
                _train_member_worker(member_id=m, seed=s, gpu_id=g, **worker_args)
                for m, s, g in assignments
            ]
        else:
            ctx = torch.multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_train_member_worker, member_id=m, seed=s, gpu_id=g, **worker_args)
                    for m, s, g in assignments
                ]
                summaries = [f.result() for f in futures]

        summaries = self._merge_manifest(summaries)
        self.load()  # load all trained members for inference
        return summaries

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
        return self

    @torch.no_grad()
    def member_predictions(self, x: torch.Tensor | np.ndarray, device: str | None = None,
                           batch_size: int = 4096) -> np.ndarray:
        """(n_members, n_events) matrix of member scores for SCALED features."""
        if not self.models:
            self.load()
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, dtype=torch.float32)
        preds = []
        for model in self.models:
            model = model.to(device).eval()
            out = [model(x[i : i + batch_size].to(device)).flatten().cpu() for i in range(0, len(x), batch_size)]
            preds.append(torch.cat(out).numpy())
            model.to("cpu")
        return np.stack(preds)

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
