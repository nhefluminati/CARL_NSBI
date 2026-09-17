"""Single-network trainer.

Owns everything about *how* one CARL network is trained: model
hyperparameters, batch size, learning rate, epochs, early stopping, device
placement (multi-GPU via a plain list of GPU ids), checkpointing/resuming,
and the auxiliary evaluation suites (during and after training).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset

from ..data.dataset import NSBIDataset, SplitIndices
from ..data.fastloader import DeviceBatches, resolve_device
from ..evaluation.base import EvaluationContext, EvaluationSuite
from ..model import CARL


@dataclass
class TrainerConfig:
    batch_size: int = 512
    val_batch_size: int = 2048
    learning_rate: float = 1e-4
    momentum: float = 0.0
    max_epochs: int = 400
    early_stopping_patience: int = 25
    num_workers: int = 4
    gpus: list[int] = field(default_factory=list)  # empty -> CPU
    precision: str = "32-true"
    resume_from: str | None = None                  # checkpoint path to continue from

    # -- performance knobs (see configs/example.yaml) --------------------
    device_batches: bool = True       # keep the split on the GPU, skip DataLoader
    matmul_precision: str = "high"    # "highest" | "high" | "medium"  (TF32)
    compile: bool = False             # torch.compile the network
    optimizer: str = "sgd"            # "sgd" | "adam" | "adamw"
    weight_decay: float = 0.0
    save_last: bool = False           # writing last.ckpt every epoch is pure IO
    checkpoint_every_n_epochs: int = 1
    progress_bar: bool = True
    log_every_n_steps: int = 50


@dataclass
class ModelConfig:
    n_layers: int = 5
    n_nodes: int = 128
    dropout: float = 0.0
    scheduler_t0: int = 25
    scheduler_t_mult: int = 1
    scheduler_eta_min: float = 1e-8


class LossHistory(Callback):
    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_loss.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking and "val_loss" in trainer.callback_metrics:
            self.val_loss.append(trainer.callback_metrics["val_loss"].item())


class CARLTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        output_dir: str | Path,
        run_name: str = "carl",
        training_evaluation: EvaluationSuite | None = None,
        final_evaluation: EvaluationSuite | None = None,
    ):
        self.model_config = model_config
        self.config = trainer_config
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.training_evaluation = training_evaluation or EvaluationSuite()
        self.final_evaluation = final_evaluation or EvaluationSuite()

    # ------------------------------------------------------------------
    def build_model(self, n_features: int) -> CARL:
        mc = self.model_config
        return CARL(
            n_features=n_features,
            n_layers=mc.n_layers,
            n_nodes=mc.n_nodes,
            dropout=mc.dropout,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            scheduler_t0=mc.scheduler_t0,
            scheduler_t_mult=mc.scheduler_t_mult,
            scheduler_eta_min=mc.scheduler_eta_min,
            optimizer=self.config.optimizer,
            weight_decay=self.config.weight_decay,
        )

    def make_loaders(
        self, dataset: NSBIDataset, train_idx: np.ndarray, val_idx: np.ndarray, seed: int = 0
    ):
        """Batch iterators for train and validation.

        With ``device_batches`` (the default) the two splits are uploaded to
        the GPU once and batched by tensor slicing — for tabular data this is
        the single biggest win available, because the default DataLoader path
        makes one Python ``__getitem__`` call per *event* and then collates,
        which for these datasets costs far more than the network itself.
        """
        if self.config.device_batches:
            device = resolve_device(self.config.gpus)
            world_size = len(self.config.gpus) if len(self.config.gpus) > 1 else 1
            rank = 0
            if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                device = torch.device(f"cuda:{self.config.gpus[rank]}")
            common = dict(device=device, rank=rank, world_size=world_size, seed=seed)
            train_loader = DeviceBatches(
                dataset, train_idx, self.config.batch_size, shuffle=True, drop_last=True, **common
            )
            val_loader = DeviceBatches(
                dataset, val_idx, self.config.val_batch_size, shuffle=False, drop_last=False, **common
            )
            return train_loader, val_loader

        common = dict(
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )
        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()), batch_size=self.config.batch_size, shuffle=True, **common
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()), batch_size=self.config.val_batch_size, shuffle=False, **common
        )
        return train_loader, val_loader

    def _lightning_trainer(self, checkpoint_dir: Path, tag: str) -> tuple[L.Trainer, LossHistory]:
        history = LossHistory()
        callbacks: list[Callback] = [
            history,
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=self.config.save_last,
                every_n_epochs=self.config.checkpoint_every_n_epochs,
                dirpath=checkpoint_dir,
                filename=f"best_{tag}",
            ),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.config.early_stopping_patience,
                min_delta=1e-6,
                check_finite=True,
            ),
        ]
        gpus = self.config.gpus
        use_gpu = torch.cuda.is_available() and len(gpus) > 0

        strategy = "auto"
        if use_gpu and len(gpus) > 1:
            # For a network this small the gradient all-reduce is a real cost,
            # so: no unused-parameter scan, a static graph (lets DDP fuse and
            # reuse its buckets), and gradients viewed directly in the bucket
            # instead of copied into it.
            strategy = DDPStrategy(
                find_unused_parameters=False,
                static_graph=True,
                gradient_as_bucket_view=True,
            )

        trainer = L.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=gpus if use_gpu else 1,
            strategy=strategy,
            precision=self.config.precision,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=self.config.progress_bar,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=self.config.log_every_n_steps,
        )
        return trainer, history

    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: NSBIDataset,
        splits: SplitIndices,
        train_idx: np.ndarray | None = None,
        seed: int = 52,
        tag: str | None = None,
        gpus_override: list[int] | None = None,
    ) -> dict:
        """Train one network. Returns a summary dict (checkpoint, history).

        ``train_idx`` allows the ensemble to pass bootstrapped indices while
        validation always uses ``splits.val``.
        """
        tag = tag or self.run_name
        if gpus_override is not None:
            self.config.gpus = gpus_override

        L.seed_everything(seed, workers=True)
        # TF32 on the tensor cores: for fp32 MLPs this is close to free speed.
        torch.set_float32_matmul_precision(self.config.matmul_precision)
        torch.backends.cudnn.benchmark = True

        model = self.build_model(dataset.n_features)
        model.evaluation = self.training_evaluation
        model.aux_dir = self.output_dir / tag / "aux"
        model.feature_names = dataset.feature_names
        model.scaler_mean = dataset.mean
        model.scaler_std = dataset.std
        if self.config.compile:
            model.net = torch.compile(model.net, dynamic=False)

        train_loader, val_loader = self.make_loaders(
            dataset, splits.train if train_idx is None else train_idx, splits.val, seed=seed
        )
        checkpoint_dir = self.output_dir / tag / "checkpoints"
        trainer, history = self._lightning_trainer(checkpoint_dir, tag)
        trainer.fit(model, train_loader, val_loader, ckpt_path=self.config.resume_from)

        checkpoint_cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
        summary = {
            "tag": tag,
            "seed": seed,
            "checkpoint": checkpoint_cb.best_model_path or checkpoint_cb.last_model_path,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
        }
        with open(self.output_dir / tag / "history.json", "w") as f:
            json.dump(summary, f, indent=2)

        if len(self.final_evaluation) > 0 and trainer.is_global_zero:
            self.evaluate(model, dataset, splits.test, histories=[summary], tag=f"{tag}_test")
        return summary

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, model: CARL, dataset: NSBIDataset, idx: np.ndarray, device: str | None = None) -> np.ndarray:
        """Score events by slicing device tensors — no DataLoader, no collate."""
        device = device or ("cuda:0" if torch.cuda.is_available() and self.config.gpus else "cpu")
        model = model.to(device).eval()
        sel = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        x = dataset.x.index_select(0, sel)
        bs = max(1, self.config.val_batch_size)
        out = [
            model(x[i : i + bs].to(device, non_blocking=True)).flatten().cpu()
            for i in range(0, x.shape[0], bs)
        ]
        return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)

    def evaluate(
        self,
        model: CARL,
        dataset: NSBIDataset,
        idx: np.ndarray,
        histories: list[dict] | None = None,
        tag: str = "final",
        suite: EvaluationSuite | None = None,
    ) -> dict[str, float]:
        """Run the after-training evaluation suite on the given split."""
        scores = self.predict(model, dataset, idx)
        ctx = EvaluationContext(
            scores=scores.astype(np.float64),
            labels=dataset.y.numpy().reshape(-1)[idx],
            weights=dataset.w.numpy().reshape(-1)[idx].astype(np.float64),
            features=dataset.x.numpy()[idx],
            feature_names=dataset.feature_names,
            mean=dataset.mean,
            std=dataset.std,
            histories=histories or [],
            output_dir=self.output_dir / "evaluation",
            tag=tag,
        )
        results = (suite or self.final_evaluation).run(ctx)
        if results:
            eval_dir = self.output_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_dir / f"metrics_{tag}.json", "w") as f:
                json.dump(results, f, indent=2)
        return results
