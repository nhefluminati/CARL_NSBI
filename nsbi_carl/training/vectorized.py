"""Vectorized ensemble training: every member in one batched matmul.

Why
---
A CARL network here is tiny — a handful of features into 5x256 — while the
ensemble is large (8+ members per template, 5 templates). Training one such
network on one A100 leaves the GPU almost entirely idle: the kernels are too
small to fill it, and the wall time is dominated by launch overhead, not
arithmetic. Running one member per GPU (the process-pool path) therefore
wastes most of every GPU, and DDP across GPUs for a *single* member is worse
still: the all-reduce costs more than the gradient it synchronises.

The fix is to stop treating the members as separate jobs. All members share
an architecture and differ only in their initial weights and their bootstrap
resample, so they can be stacked into one module whose weights carry a
leading member axis:

    x  (M, B, F)  @  W  (M, F, H)  ->  (M, B, H)      [torch.bmm]

One kernel launch trains all M members. The arithmetic per launch grows by a
factor M while the launch count stays fixed, which is exactly the direction
a launch-bound workload needs. Members stay mathematically independent: their
parameters are disjoint slices of the stacked tensors, and summing the
per-member losses gives each member exactly the gradient it would have had on
its own.

This keeps every NSBI guarantee the rest of the package makes: member ``i``
still initialises from ``start_seed + i`` and still trains on its own
bootstrap resample drawn by the same seeded generators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import lightning as L

from ..data.dataset import NSBIDataset, SplitIndices
from ..model import build_mlp


# ---------------------------------------------------------------------------
class StackedMLP(nn.Module):
    """``n_members`` independent MLPs evaluated as one batched matmul.

    Parameters are stored with a leading member axis. ``weights[i]`` holds
    layer ``i`` as ``(M, in, out)`` — transposed relative to ``nn.Linear``
    (which stores ``(out, in)``) because ``bmm`` wants it that way.
    """

    def __init__(self, n_members: int, n_features: int, n_layers: int, n_nodes: int, dropout: float = 0.0):
        super().__init__()
        self.n_members = int(n_members)
        self.n_features = int(n_features)
        self.n_layers = int(n_layers)
        self.n_nodes = int(n_nodes)
        self.dropout = float(dropout)

        dims = [n_features] + [n_nodes] * n_layers + [1]
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(self.n_members, dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(self.n_members, 1, dims[i + 1])) for i in range(len(dims) - 1)]
        )

    # -- initialisation -----------------------------------------------------
    def init_from_seeds(self, seeds: list[int]) -> None:
        """Initialise member ``m`` exactly as a standalone ``CARL`` would.

        Each member's weights are drawn by building a throwaway reference MLP
        under ``torch.manual_seed(seed)``, so a member trained here and the
        same member trained through ``CARLTrainer`` start from identical
        parameters.
        """
        if len(seeds) != self.n_members:
            raise ValueError(f"expected {self.n_members} seeds, got {len(seeds)}")
        with torch.no_grad():
            for m, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                ref = build_mlp(self.n_features, self.n_layers, self.n_nodes, self.dropout)
                linears = [mod for mod in ref if isinstance(mod, nn.Linear)]
                for layer, lin in enumerate(linears):
                    self.weights[layer][m].copy_(lin.weight.t())
                    self.biases[layer][m, 0].copy_(lin.bias)

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(M, B, F)`` (per-member batches) or ``(B, F)`` (shared).

        Returns logits of shape ``(M, B)``. A shared ``(B, F)`` input is
        broadcast against the member axis without being copied M times.
        """
        h = x
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.matmul(h, w) + b  # broadcasts (B,F)x(M,F,H) -> (M,B,H)
            if i != last:
                h = F.silu(h)
                if self.dropout > 0.0 and self.training:
                    h = F.dropout(h, p=self.dropout, training=True)
        return h.squeeze(-1)

    # -- export -------------------------------------------------------------
    def member_state_dict(self, m: int) -> dict[str, torch.Tensor]:
        """One member's parameters in ``CARL.net`` (``nn.Sequential``) naming."""
        state: dict[str, torch.Tensor] = {}
        stride = 3 if self.dropout > 0.0 else 2
        for layer in range(len(self.weights)):
            idx = layer * stride
            state[f"net.{idx}.weight"] = self.weights[layer][m].t().detach().cpu().clone()
            state[f"net.{idx}.bias"] = self.biases[layer][m, 0].detach().cpu().clone()
        return state


# ---------------------------------------------------------------------------
@dataclass
class VectorizedConfig:
    batch_size: int = 32768
    val_batch_size: int = 262144
    learning_rate: float = 1e-4
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    max_epochs: int = 400
    early_stopping_patience: int = 25
    min_delta: float = 1e-6
    scheduler_t0: int = 25
    scheduler_t_mult: int = 1
    scheduler_eta_min: float = 1e-8
    compile: bool = False
    amp_dtype: str = "none"  # "none" | "bf16" | "fp16"
    log_every_n_epochs: int = 10


def _member_bootstrap_matrix(
    splits: SplitIndices, labels: np.ndarray, seeds: list[int], fraction: float, bootstrap: bool
) -> np.ndarray:
    """``(M, n_train)`` index matrix, one bootstrap resample per member.

    Reuses the exact draw logic of ``ensemble.bootstrap_train_indices`` so the
    vectorized path and the process-pool path see identical data.
    """
    from .ensemble import bootstrap_train_indices

    if not bootstrap:
        return np.tile(splits.train[None, :], (len(seeds), 1))
    rows = [bootstrap_train_indices(splits, labels, s, fraction) for s in seeds]
    width = min(len(r) for r in rows)
    return np.stack([r[:width] for r in rows])


class VectorizedEnsembleTrainer:
    """Trains ``M`` CARL members simultaneously on one device."""

    def __init__(
        self,
        config: VectorizedConfig,
        output_dir: str | Path,
        run_name: str = "carl",
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def _autocast(self):
        dt = self.config.amp_dtype.lower()
        if dt == "none" or self.device.type != "cuda":
            return torch.autocast("cuda", enabled=False)
        dtype = torch.bfloat16 if dt == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def _build_optimizer(self, model: nn.Module):
        cfg = self.config
        name = cfg.optimizer.lower()
        if name == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay, foreach=True,
            )
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        try:
            return cls(model.parameters(), fused=True, **kwargs)
        except (RuntimeError, TypeError):
            return cls(model.parameters(), foreach=True, **kwargs)

    # ------------------------------------------------------------------
    def train(
        self,
        dataset: NSBIDataset,
        splits: SplitIndices,
        member_ids: list[int],
        seeds: list[int],
        bootstrap: bool = True,
        bootstrap_fraction: float = 1.0,
        model_config=None,
    ) -> list[dict]:
        cfg = self.config
        dev = self.device
        M = len(member_ids)

        n_layers = getattr(model_config, "n_layers", 5)
        n_nodes = getattr(model_config, "n_nodes", 128)
        dropout = getattr(model_config, "dropout", 0.0)

        labels = dataset.y.numpy().reshape(-1)
        boot = _member_bootstrap_matrix(splits, labels, seeds, bootstrap_fraction, bootstrap)
        boot_t = torch.as_tensor(boot, dtype=torch.long, device=dev)
        n_train = boot_t.shape[1]

        # Whole dataset resident on the device: no host traffic in the loop.
        x_all = dataset.x.to(dev)
        y_all = dataset.y.reshape(-1).to(dev)
        w_all = dataset.w.reshape(-1).float().to(dev)

        val_idx = torch.as_tensor(splits.val, dtype=torch.long, device=dev)
        x_val, y_val, w_val = x_all[val_idx], y_all[val_idx], w_all[val_idx]

        model = StackedMLP(M, dataset.n_features, n_layers, n_nodes, dropout).to(dev)
        model.init_from_seeds(seeds)
        forward = model
        if cfg.compile:
            forward = torch.compile(model, dynamic=False)

        optimizer = self._build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.scheduler_t0, T_mult=cfg.scheduler_t_mult, eta_min=cfg.scheduler_eta_min
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp_dtype.lower() == "fp16" and dev.type == "cuda"))

        best_val = torch.full((M,), float("inf"), device=dev)
        stale = torch.zeros(M, dtype=torch.long, device=dev)
        best_state: dict[int, dict[str, torch.Tensor]] = {}
        train_hist: list[list[float]] = []
        val_hist: list[list[float]] = []

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seeds[0]))

        n_batches = max(1, n_train // cfg.batch_size)

        for epoch in range(cfg.max_epochs):
            # ---- train ------------------------------------------------
            model.train()
            perm = torch.argsort(torch.rand(M, n_train, device=dev, generator=gen), dim=1)
            epoch_num = torch.zeros(M, device=dev)
            epoch_den = torch.zeros(M, device=dev)

            for b in range(n_batches):
                sel = perm[:, b * cfg.batch_size : (b + 1) * cfg.batch_size]
                idx = torch.gather(boot_t, 1, sel)              # (M, B) event ids
                xb = x_all[idx]                                  # (M, B, F)
                yb = y_all[idx]
                wb = w_all[idx]

                with self._autocast():
                    logits = forward(xb)                         # (M, B)
                    bce = F.binary_cross_entropy_with_logits(logits.float(), yb, reduction="none")
                    num = (bce * wb).sum(dim=1)
                    den = wb.sum(dim=1)
                    per_member = num / den
                    # Disjoint parameters => summing is per-member independent.
                    loss = per_member.sum()

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_num += num.detach()
                epoch_den += den.detach()

            scheduler.step()
            train_loss = epoch_num / epoch_den

            # ---- validate --------------------------------------------
            model.eval()
            vnum = torch.zeros(M, device=dev)
            vden = torch.zeros(M, device=dev)
            with torch.no_grad():
                for s in range(0, x_val.shape[0], cfg.val_batch_size):
                    xv = x_val[s : s + cfg.val_batch_size]
                    yv = y_val[s : s + cfg.val_batch_size]
                    wv = w_val[s : s + cfg.val_batch_size]
                    with self._autocast():
                        logits = forward(xv)                     # (M, B) broadcast, no copy
                    bce = F.binary_cross_entropy_with_logits(logits.float(), yv.expand_as(logits), reduction="none")
                    vnum += (bce * wv).sum(dim=1)
                    vden += wv.sum()
            val_loss = vnum / vden

            train_hist.append(train_loss.tolist())
            val_hist.append(val_loss.tolist())

            # ---- per-member checkpointing / early stopping -------------
            improved = val_loss < (best_val - cfg.min_delta)
            if bool(improved.any()):
                best_val = torch.where(improved, val_loss, best_val)
                for m in improved.nonzero(as_tuple=True)[0].tolist():
                    best_state[m] = model.member_state_dict(m)
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)

            if cfg.log_every_n_epochs and epoch % cfg.log_every_n_epochs == 0:
                print(
                    f"[{self.run_name}] epoch {epoch:4d}  "
                    f"train {train_loss.mean().item():.6f}  val {val_loss.mean().item():.6f}  "
                    f"converged {int((stale >= cfg.early_stopping_patience).sum())}/{M}",
                    flush=True,
                )

            if bool((stale >= cfg.early_stopping_patience).all()):
                print(f"[{self.run_name}] all members early-stopped at epoch {epoch}", flush=True)
                break

        # ---- write one Lightning-compatible checkpoint per member ------
        summaries = []
        for i, member_id in enumerate(member_ids):
            tag = f"member_{member_id:03d}"
            ckpt_dir = self.output_dir / tag / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"best_{tag}.ckpt"

            state = best_state.get(i) or model.member_state_dict(i)
            torch.save(
                {
                    "state_dict": state,
                    "hyper_parameters": {
                        "n_features": dataset.n_features,
                        "n_layers": n_layers,
                        "n_nodes": n_nodes,
                        "dropout": dropout,
                        "learning_rate": cfg.learning_rate,
                        "momentum": cfg.momentum,
                        "scheduler_t0": cfg.scheduler_t0,
                        "scheduler_t_mult": cfg.scheduler_t_mult,
                        "scheduler_eta_min": cfg.scheduler_eta_min,
                        "optimizer": cfg.optimizer,
                        "weight_decay": cfg.weight_decay,
                    },
                    "pytorch-lightning_version": L.__version__,
                    "epoch": len(val_hist),
                    "global_step": len(val_hist) * n_batches,
                },
                ckpt_path,
            )

            summary = {
                "tag": tag,
                "member": member_id,
                "seed": seeds[i],
                "checkpoint": str(ckpt_path),
                "train_loss": [e[i] for e in train_hist],
                "val_loss": [e[i] for e in val_hist],
                "best_val_loss": float(best_val[i].item()),
                "device": str(self.device),
                "vectorized": True,
            }
            with open(self.output_dir / tag / "history.json", "w") as f:
                json.dump(summary, f, indent=2)
            summaries.append(summary)

        return summaries
