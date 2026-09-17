"""The CARL classifier as a LightningModule.

Architecture: configurable feed-forward network, swish (SiLU) activations in
the hidden layers, single sigmoid output. Trained with a weighted binary
cross entropy and a CosineAnnealingWarmRestarts schedule.

Auxiliary diagnostics: an :class:`~nsbi_carl.evaluation.base.EvaluationSuite`
can be attached (``model.evaluation = suite``). At the end of every
validation epoch the suite runs whatever is due at that epoch (each metric /
plot carries its own ``every_n_epochs``); metric values are logged, plots are
written to ``aux_dir`` tagged with the epoch.

Performance notes
-----------------
* ``val_loss`` is accumulated on the device as two running scalars
  (Σ w·bce and Σ w) and reduced across ranks with a single two-element
  ``all_reduce``. The previous implementation moved *every* validation
  feature, label, weight and logit to the CPU each epoch and, under DDP,
  ``all_gather``-ed the full feature matrix — on a 1M-event validation split
  that is hundreds of MB of traffic per epoch to compute one number.
* The full arrays are only materialised on epochs where the evaluation suite
  actually has something due (``evaluation_due``), so a diagnostic scheduled
  ``every_n_epochs: 25`` costs nothing on the other 24 epochs.
* The loss is computed in float32 throughout; the float64 casts only happen
  once, on the numpy arrays handed to the diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .evaluation.base import EvaluationContext, EvaluationSuite


def weighted_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """BCE where every event contributes proportionally to its weight."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * weights).sum() / weights.sum()


def weighted_bce_sums(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(Σ w·bce, Σ w)`` — the two accumulators a weighted mean needs.

    Keeping numerator and denominator separate is what lets the validation
    loss be accumulated batch by batch and reduced across ranks exactly,
    rather than averaging per-batch means (which is wrong for unequal
    batches) or gathering everything.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * weights).sum(), weights.sum()


def build_mlp(n_features: int, n_layers: int, n_nodes: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = n_features
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, n_nodes), nn.SiLU()]  # SiLU == swish
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = n_nodes
    layers.append(nn.Linear(in_dim, 1))  # logit head; sigmoid in forward()
    return nn.Sequential(*layers)


class CARL(L.LightningModule):
    def __init__(
        self,
        n_features: int,
        n_layers: int = 5,
        n_nodes: int = 128,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        momentum: float = 0.0,
        scheduler_t0: int = 25,
        scheduler_t_mult: int = 1,
        scheduler_eta_min: float = 1e-8,
        optimizer: str = "sgd",
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = build_mlp(n_features, n_layers, n_nodes, dropout)

        # Attached by the trainer; not a hyperparameter.
        self.evaluation: EvaluationSuite = EvaluationSuite()
        self.aux_dir: Path = Path(".")
        self.feature_names: list[str] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

        # Running validation accumulators (device tensors, no host sync).
        self._val_num: torch.Tensor | None = None
        self._val_den: torch.Tensor | None = None
        # Only populated on epochs where a diagnostic is due.
        self._val_buffer: list[tuple[torch.Tensor, ...]] = []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        loss = weighted_bce_with_logits(self.net(x).flatten(), y.flatten(), w.flatten())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    @property
    def evaluation_due(self) -> bool:
        """True when the attached suite has anything scheduled this epoch."""
        if len(self.evaluation) == 0 or self.trainer.sanity_checking:
            return False
        return self.evaluation.due_at(self.current_epoch)

    def on_validation_epoch_start(self):
        self._val_num = None
        self._val_den = None
        self._val_buffer.clear()

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y, w = y.flatten(), w.flatten()
        logits = self.net(x).flatten()

        num, den = weighted_bce_sums(logits, y, w)
        if self._val_num is None:
            self._val_num, self._val_den = num, den
        else:
            self._val_num = self._val_num + num
            self._val_den = self._val_den + den

        # Full arrays are expensive; only keep them when something needs them.
        if self.evaluation_due:
            self._val_buffer.append((x.detach(), y.detach(), w.detach(), logits.detach()))
        return None

    def on_validation_epoch_end(self):
        if self._val_num is None:
            return

        # One tiny collective instead of gathering the whole validation set.
        stats = torch.stack([self._val_num, self._val_den])
        if self.trainer.world_size > 1:
            stats = self.all_gather(stats).sum(dim=0)
        val_loss = stats[0] / stats[1]
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=False)

        if not self._val_buffer:
            return

        x = torch.cat([b[0] for b in self._val_buffer])
        y = torch.cat([b[1] for b in self._val_buffer])
        w = torch.cat([b[2] for b in self._val_buffer])
        logits = torch.cat([b[3] for b in self._val_buffer])
        self._val_buffer.clear()

        if self.trainer.world_size > 1:  # multi-GPU: gather full validation set
            x = self.all_gather(x).reshape(-1, x.shape[-1])
            y = self.all_gather(y).reshape(-1)
            w = self.all_gather(w).reshape(-1)
            logits = self.all_gather(logits).reshape(-1)

        if self.trainer.world_size > 1 and self.trainer.global_rank != 0:
            return

        # Single host transfer, at the very end, only on rank 0.
        ctx = EvaluationContext(
            scores=torch.sigmoid(logits).double().cpu().numpy(),
            labels=y.cpu().numpy(),
            weights=w.double().cpu().numpy(),
            features=x.float().cpu().numpy(),
            feature_names=self.feature_names,
            mean=self.scaler_mean,
            std=self.scaler_std,
            output_dir=self.aux_dir,
            tag=f"epoch_{self.current_epoch:04d}",
        )
        for name, value in self.evaluation.run(ctx, epoch=self.current_epoch).items():
            self.log(f"aux/{name}", value, on_epoch=True, prog_bar=False, sync_dist=False)

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return torch.sigmoid(self.net(x)).flatten()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        name = str(getattr(self.hparams, "optimizer", "sgd")).lower()
        wd = float(getattr(self.hparams, "weight_decay", 0.0))
        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=wd,
                foreach=True,
            )
        elif name in ("adam", "adamw"):
            cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
            # `fused` keeps the whole step on the GPU; it is a large win for
            # small models where the optimizer step rivals the forward pass.
            kwargs = dict(lr=self.hparams.learning_rate, weight_decay=wd)
            try:
                optimizer = cls(self.parameters(), fused=True, **kwargs)
            except (RuntimeError, TypeError):
                optimizer = cls(self.parameters(), foreach=True, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer {name!r}; use 'sgd', 'adam' or 'adamw'.")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.scheduler_t0,
            T_mult=self.hparams.scheduler_t_mult,
            eta_min=self.hparams.scheduler_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
