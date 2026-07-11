"""The CARL classifier as a LightningModule.

Architecture: configurable feed-forward network, swish (SiLU) activations in
the hidden layers, single sigmoid output. Trained with a weighted binary
cross entropy, plain SGD and a CosineAnnealingWarmRestarts schedule.

Auxiliary diagnostics: an :class:`~nsbi_carl.evaluation.base.EvaluationSuite`
can be attached (``model.evaluation = suite``). At the end of every
validation epoch the suite runs whatever is due at that epoch (each metric /
plot carries its own ``every_n_epochs``); metric values are logged, plots are
written to ``aux_dir`` tagged with the epoch.
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
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, n_nodes), nn.SiLU()]  # SiLU == swish
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = n_nodes
        layers.append(nn.Linear(in_dim, 1))  # logit head; sigmoid in forward()
        self.net = nn.Sequential(*layers)

        # Attached by the trainer; not a hyperparameter.
        self.evaluation: EvaluationSuite = EvaluationSuite()
        self.aux_dir: Path = Path(".")
        self.feature_names: list[str] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

        self._val_buffer: list[tuple[torch.Tensor, ...]] = []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        loss = weighted_bce_with_logits(self.net(x).flatten(), y.flatten(), w.flatten())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self.net(x).flatten()
        loss = weighted_bce_with_logits(logits, y.flatten(), w.flatten())
        self._val_buffer.append(
            (x.detach().cpu(), y.flatten().detach().cpu(), w.flatten().detach().cpu(), logits.detach().cpu())
        )
        return loss

    def on_validation_epoch_end(self):
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

        val_loss = weighted_bce_with_logits(logits, y, w)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=False)

        # ---- auxiliary metrics / plots (rank 0 only) ----------------------
        if len(self.evaluation) == 0 or self.trainer.sanity_checking:
            return
        if self.trainer.world_size > 1 and self.trainer.global_rank != 0:
            return

        ctx = EvaluationContext(
            scores=torch.sigmoid(logits).numpy().astype(np.float64),
            labels=y.numpy(),
            weights=w.numpy().astype(np.float64),
            features=x.numpy(),
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
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum
        )
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
