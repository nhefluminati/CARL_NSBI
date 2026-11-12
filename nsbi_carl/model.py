"""The CARL classifier as a LightningModule.

Architecture: selected by ``architecture``:

* ``"dnn"`` (default): configurable feed-forward network, swish (SiLU)
  activations in the hidden layers.
* ``"transformer"``: a small feature-tokenising transformer
  (:class:`FeatureTransformer`): every input feature becomes one token, a
  learned [CLS] token attends over them, and the [CLS] state is read out.

Both end in a single output unit and train with a CosineAnnealingWarmRestarts
schedule. The loss is selected by ``loss``:

* ``"bce"`` (default): the output is a logit, trained with a weighted binary
  cross entropy; ``forward`` applies the sigmoid.
* ``"pare"``: the output is the raw PARE estimator, trained with the weighted
  pole-adjustable ratio estimation loss (see ``LOSSES`` below), which keeps
  negatively weighted events and negative ratios inside a well-posed problem.
  ``forward`` decodes it to the same score convention as BCE.

Auxiliary diagnostics: an :class:`~nsbi_carl.evaluation.base.EvaluationSuite`
can be attached (``model.evaluation = suite``). At the end of every
validation epoch the suite runs whatever is due at that epoch (each metric /
plot carries its own ``every_n_epochs``); metric values are logged, plots are
written to ``aux_dir`` tagged with the epoch.

Performance notes
-----------------
* ``val_loss`` is accumulated on the device as two running scalars
  (Σ w·loss and Σ w; float64 for PARE) and reduced across ranks with a single two-element
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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
# "bce":  weighted binary cross entropy on a logit head; s = sigmoid(z) and the
#         ratio is r = s / (1 - s). Cannot represent r < 0.
# "pare": pole-adjustable ratio estimation (Drnevich, Jiggins, Katzy, Cranmer,
#         Mach. Learn.: Sci. Technol. 6 (2025) 045023, eq. 8-10):
#
#             L(s, y; t) = (1 - s * t_y)^2,   t = (t0, t1)
#
#         The raw network output s is unbounded (no sigmoid). Its optimum is
#
#             s* = (t0 + t1 r) / (t0^2 + t1^2 r)
#
#         with r = q(x|target) / q(x|reference), invertible to
#
#             r  = t0 (1 - t0 s) / (t1 (t1 s - 1)).
#
#         The transform's only pole sits at r_pole = -(t0/t1)^2 instead of at
#         r = -1 (MSE) or r = 0 (BCE), so negative ratios and negative weights
#         stay inside a smooth, bounded-below problem as long as the true ratio
#         stays above r_pole. Per eq. (32)-(34) of the paper, pick t0/t1 so that
#         |t0/t1| comfortably exceeds the largest |r| expected; the
#         defaults (25619, 58) are the paper's, |t0/t1| ~ 442.
#
#         Output parametrization (``pare_param``):
#
#         "raw"   the network output IS s, exactly as in the paper. With a
#                 large |t0/t1| the whole useful range r in [0, 100] maps into
#                 an output window ~1e-5 wide around 1/t0, next to the decode
#                 pole at s = 1/t1. Optimiser steps on a wide network are far
#                 coarser than that window, so from-scratch training jitters
#                 and can cross the pole (ratios of 1e16). Kept for checkpoints
#                 trained that way.
#         "ratio" (default) the network outputs u with s = (1 + rho u) / t0,
#                 rho = t1 / t0, and the loss is PARE / rho^2 minus its
#                 constant:
#
#                     l(u, y=0) = u^2,   l(u, y=1) = rho^2 u^2 - 2 (1 - rho) u
#
#                 An affine map of the output and a positive rescaling of the
#                 loss, so the minimiser and the pole are the paper's:
#                 u* = r (1 - rho) / (1 + rho^2 r) ~ r, and the decode is
#                 r = u / (1 - rho - rho^2 u). The output is O(r), the loss
#                 and gradients are O(1), and nothing cancels numerically. As
#                 rho -> 0 this is least-squares density-ratio fitting (uLSIF).
LOSSES = ("bce", "pare")
PARE_PARAMS = ("raw", "ratio")
PARE_DEFAULT_T0 = 25619.0
PARE_DEFAULT_T1 = 58.0


def validate_pare_param(name: str) -> str:
    key = str(name).lower()
    if key not in PARE_PARAMS:
        raise ValueError(f"Unknown PARE parametrization {name!r}; use one of {PARE_PARAMS}.")
    return key


def validate_loss(name: str, t0: float = PARE_DEFAULT_T0, t1: float = PARE_DEFAULT_T1) -> str:
    key = str(name).lower()
    if key not in LOSSES:
        raise ValueError(f"Unknown loss {name!r}; use one of {LOSSES}.")
    if key == "pare":
        t0, t1 = float(t0), float(t1)
        if t0 == 0.0 or t1 == 0.0:
            raise ValueError(f"PARE needs non-zero t0 and t1, got t0={t0}, t1={t1}.")
        if t0 == t1:
            # s* = 1/t for every r: the optimum carries no information.
            raise ValueError(f"PARE needs t0 != t1 (got {t0}); the optimum is then constant.")
    return key


def pare_pole(t0: float, t1: float) -> float:
    """``r_pole = -(t0/t1)^2``: the one ratio the PARE optimum cannot express."""
    return -((float(t0) / float(t1)) ** 2)


def weighted_pare(s: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                  t0: float, t1: float, param: str = "raw") -> torch.Tensor:
    """Weighted PARE loss on the network output (see ``weighted_pare_sums``)."""
    num, den = weighted_pare_sums(s, targets, weights, t0, t1, param=param,
                                  accumulate_dtype=None)
    return num / den


def weighted_pare_sums(
    s: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, t0: float, t1: float,
    accumulate_dtype: torch.dtype | None = torch.float64, param: str = "raw",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(Σ w l(out, y), Σ w)``, optionally accumulated in float64.

    ``param="raw"``:   l = (1 - s t_y)^2 on the raw output s (the paper).
    ``param="ratio"``: l = u^2 (y=0), rho^2 u^2 - 2 (1 - rho) u (y=1) on the
                       reparametrized output u; same minimiser (module notes).

    float64 matters for monitoring the raw form: its loss sits near a
    constant and improves by O((t1/t0)^2). Training uses the dtype of ``s``.
    """
    if accumulate_dtype is not None:
        s, weights = s.to(accumulate_dtype), weights.to(accumulate_dtype)
    if param == "ratio":
        rho = float(t1) / float(t0)
        per_event = torch.where(targets > 0.5, rho * rho * s * s - 2.0 * (1.0 - rho) * s, s * s)
        return (per_event * weights).sum(dim=-1), weights.sum(dim=-1)
    t = torch.where(targets > 0.5, torch.as_tensor(float(t1), dtype=s.dtype, device=s.device),
                    torch.as_tensor(float(t0), dtype=s.dtype, device=s.device))
    per_event = (1.0 - s * t) ** 2
    return (per_event * weights).sum(dim=-1), weights.sum(dim=-1)


def pare_ratio(s: torch.Tensor, t0: float, t1: float, param: str = "raw") -> torch.Tensor:
    """Density ratio from a PARE output (inverse of eq. 9), in float64."""
    s = s.double()
    if param == "ratio":
        rho = float(t1) / float(t0)
        return s / (1.0 - rho - rho * rho * s)
    return t0 * (1.0 - t0 * s) / (t1 * (t1 * s - 1.0))


def scores_from_output(out: torch.Tensor, loss: str = "bce",
                       t0: float = PARE_DEFAULT_T0, t1: float = PARE_DEFAULT_T1,
                       param: str = "raw") -> torch.Tensor:
    """Map the raw network output to the package-wide *score* convention.

    Everything downstream (combiners, evaluation, inference, diagnostics)
    works with a score ``s`` and the ratio ``r = s / (1 - s)``. For BCE that
    score is the sigmoid. For PARE the ratio is decoded first and then
    re-encoded as ``s = r / (1 + r)``, so ``s / (1 - s)`` returns exactly the
    PARE ratio and no consumer needs to know which loss trained the network.

    For ``r >= 0`` this score lies in [0, 1) like a sigmoid. A negative PARE
    ratio gives a score outside [0, 1] (``s < 0`` for ``-1 < r < 0``, ``s > 1``
    for ``r < -1``); it still round-trips through ``s/(1-s)``, but the
    existing consumers clip scores to (0, 1) and therefore treat it as r ~ 0.
    Returned in the dtype of ``out`` (float64 internally for PARE).
    """
    if loss == "pare":
        r = pare_ratio(out, t0, t1, param)
        return (r / (1.0 + r)).to(out.dtype)
    return torch.sigmoid(out)


def loss_hparams(hp) -> dict:
    """``loss``, ``pare_t0``, ``pare_t1``, ``pare_param`` of a CARL ``hparams``.

    Checkpoints written before PARE existed carry none of these; they are BCE.
    PARE checkpoints without ``pare_param`` predate the reparametrization and
    are ``raw``.
    """
    return dict(
        loss=validate_loss(hp.get("loss", "bce"), hp.get("pare_t0", PARE_DEFAULT_T0),
                           hp.get("pare_t1", PARE_DEFAULT_T1)),
        pare_t0=float(hp.get("pare_t0", PARE_DEFAULT_T0)),
        pare_t1=float(hp.get("pare_t1", PARE_DEFAULT_T1)),
        pare_param=validate_pare_param(hp.get("pare_param", "raw")),
    )


def common_loss_hparams(models: list) -> dict:
    """The shared loss settings of a list of CARL members.

    Members are fused into one stacked module and decoded by one rule, so a
    mix (BCE members next to PARE members, or two different t) is refused.
    """
    settings = [loss_hparams(m.hparams) for m in models]
    for other in settings[1:]:
        if other != settings[0]:
            raise ValueError(
                f"Ensemble members disagree on the loss / PARE parameters ({settings[0]} vs "
                f"{other}); their outputs cannot be decoded by one rule."
            )
    return settings[0]


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


# ---------------------------------------------------------------------------
# Transformer variant
# ---------------------------------------------------------------------------
ARCHITECTURES = ("dnn", "transformer")


def validate_architecture(name: str) -> str:
    key = str(name).lower()
    if key in ("mlp", "ffnn", "fcnn"):
        key = "dnn"
    if key not in ARCHITECTURES:
        raise ValueError(f"Unknown model architecture {name!r}; use one of {ARCHITECTURES}.")
    return key


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: self-attention + SiLU feed-forward.

    Attention is written out explicitly (no ``nn.MultiheadAttention``) so the
    block has no fused "fast path" and runs unchanged under ``torch.func.vmap``,
    which is how the vectorized ensemble trainer batches the members.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)

    def _drop(self, t: torch.Tensor) -> torch.Tensor:
        return F.dropout(t, p=self.dropout, training=self.training) if self.dropout > 0.0 else t

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        a = (self._drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        h = h + self._drop(self.proj(a))
        h = h + self._drop(self.ff2(self._drop(F.silu(self.ff1(self.norm2(h))))))
        return h


class FeatureTransformer(nn.Module):
    """Simple transformer for tabular inputs (FT-Transformer style).

    ``x`` ``(B, F)`` -> tokens ``(B, F, d)`` via a per-feature affine embedding
    ``x_f * W_f + b_f``, a learned [CLS] token is prepended, ``n_layers``
    encoder blocks run over the ``F + 1`` tokens and the final [CLS] state is
    mapped to one logit. Returns ``(B, 1)``, the same shape as ``build_mlp``.

    The logit head is a broadcast multiply + sum rather than ``nn.Linear(d, 1)``
    for the same reason as ``StackedMLP``'s head: a batched matmul with inner
    dimension 1 in the backward is routed to a runtime-compiled Triton kernel.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        bound = d_model ** -0.5
        self.tok_weight = nn.Parameter(torch.empty(n_features, d_model).uniform_(-bound, bound))
        self.tok_bias = nn.Parameter(torch.empty(n_features, d_model).uniform_(-bound, bound))
        self.cls = nn.Parameter(torch.empty(d_model).uniform_(-bound, bound))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_weight = nn.Parameter(torch.empty(d_model).uniform_(-bound, bound))
        self.head_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.tok_weight + self.tok_bias           # (B, F, d)
        cls = self.cls.expand(x.shape[0], 1, self.d_model)                    # (B, 1, d)
        h = torch.cat([cls, tokens], dim=1)                                   # (B, F+1, d)
        for block in self.blocks:
            h = block(h)
        z = self.norm(h[:, 0])                                                # (B, d)
        return (z * self.head_weight).sum(-1, keepdim=True) + self.head_bias  # (B, 1)


def build_network(
    n_features: int,
    architecture: str = "dnn",
    n_layers: int = 5,
    n_nodes: int = 128,
    dropout: float = 0.0,
    tf_d_model: int = 32,
    tf_n_heads: int = 4,
    tf_n_layers: int = 3,
    tf_ff_dim: int = 64,
) -> nn.Module:
    """The network behind ``CARL.net`` for either architecture."""
    if validate_architecture(architecture) == "transformer":
        return FeatureTransformer(
            n_features, d_model=tf_d_model, n_heads=tf_n_heads,
            n_layers=tf_n_layers, ff_dim=tf_ff_dim, dropout=dropout,
        )
    return build_mlp(n_features, n_layers, n_nodes, dropout)


def architecture_hparams(hp) -> dict:
    """The network-defining hyperparameters of a CARL ``hparams`` mapping.

    Checkpoints written before the transformer option existed carry no
    ``architecture`` key; they are DNNs.
    """
    return dict(
        architecture=validate_architecture(hp.get("architecture", "dnn")),
        n_layers=hp.get("n_layers", 5),
        n_nodes=hp.get("n_nodes", 128),
        dropout=hp.get("dropout", 0.0),
        tf_d_model=hp.get("tf_d_model", 32),
        tf_n_heads=hp.get("tf_n_heads", 4),
        tf_n_layers=hp.get("tf_n_layers", 3),
        tf_ff_dim=hp.get("tf_ff_dim", 64),
    )


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
        architecture: str = "dnn",
        tf_d_model: int = 32,
        tf_n_heads: int = 4,
        tf_n_layers: int = 3,
        tf_ff_dim: int = 64,
        loss: str = "bce",
        pare_t0: float = PARE_DEFAULT_T0,
        pare_t1: float = PARE_DEFAULT_T1,
        pare_param: str = "raw",   # "raw" only so old PARE checkpoints load; see ModelConfig
    ):
        super().__init__()
        validate_loss(loss, pare_t0, pare_t1)
        validate_pare_param(pare_param)
        self.save_hyperparameters()
        lh = loss_hparams(self.hparams)
        self.loss_name: str = lh["loss"]
        self.pare_t0: float = lh["pare_t0"]
        self.pare_t1: float = lh["pare_t1"]
        self.pare_param: str = lh["pare_param"]

        self.net = build_network(
            n_features,
            architecture=architecture,
            n_layers=n_layers,
            n_nodes=n_nodes,
            dropout=dropout,
            tf_d_model=tf_d_model,
            tf_n_heads=tf_n_heads,
            tf_n_layers=tf_n_layers,
            tf_ff_dim=tf_ff_dim,
        )

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
        return self.scores(self.net(x))

    def scores(self, out: torch.Tensor) -> torch.Tensor:
        """Raw network output -> score with ``r = s/(1-s)`` (see ``scores_from_output``)."""
        return scores_from_output(out, self.loss_name, self.pare_t0, self.pare_t1,
                                  self.pare_param)

    def _loss(self, out: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "pare":
            return weighted_pare(out, y, w, self.pare_t0, self.pare_t1, self.pare_param)
        return weighted_bce_with_logits(out, y, w)

    def _loss_sums(self, out: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        if self.loss_name == "pare":
            return weighted_pare_sums(out, y, w, self.pare_t0, self.pare_t1,
                                      param=self.pare_param)
        return weighted_bce_sums(out, y, w)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        loss = self._loss(self.net(x).flatten(), y.flatten(), w.flatten())
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

        num, den = self._loss_sums(logits, y, w)
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
            scores=self.scores(logits.double()).cpu().numpy(),
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
        return self.scores(self.net(x)).flatten()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.parameters(),
            name=str(getattr(self.hparams, "optimizer", "sgd")),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=float(getattr(self.hparams, "weight_decay", 0.0)),
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


OPTIMIZERS = ("sgd", "adam", "adamw", "nadam")


def validate_optimizer(name: str) -> str:
    key = str(name).lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; use one of {OPTIMIZERS}.")
    return key


def build_optimizer(params, name: str, lr: float, momentum: float = 0.0,
                    weight_decay: float = 0.0):
    """One optimizer factory for the Lightning and the vectorized trainers,
    so a name can never mean different things in the two paths."""
    name = validate_optimizer(name)
    params = list(params)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, foreach=True)
    cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
           "nadam": torch.optim.NAdam}[name]   # NAdam: as in the ATLAS NSBI analysis
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    # `fused` keeps the whole step on the GPU (a large win for small models);
    # fall back to `foreach` where the optimizer or device has no fused kernel.
    try:
        return cls(params, fused=True, **kwargs)
    except (RuntimeError, TypeError, ValueError):
        return cls(params, foreach=True, **kwargs)
