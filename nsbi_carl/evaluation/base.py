"""Common interface for auxiliary evaluation metrics and plots.

Metrics implement  ``evaluate(ctx) -> dict[str, float]``
Plots   implement  ``plot(ctx)     -> Path``

Both receive an :class:`EvaluationContext`, which is intentionally the ONLY
interface between the training/ensembling machinery and the diagnostics. The
same context works for

* a single network during training (scores from the current epoch),
* a single finished network,
* a finished ensemble (``scores`` is the ensemble mean and ``member_scores``
  additionally carries every member's prediction).

This is what makes every metric/plot automatically ensemble-compatible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EvaluationContext:
    scores: np.ndarray                       # NN output s in (0, 1); ensemble mean if applicable
    labels: np.ndarray                       # 1 = target, 0 = reference
    weights: np.ndarray                      # event weights (post-reweighting)
    features: np.ndarray | None = None       # scaled features, shape (n, n_features)
    feature_names: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None            # scaler mean (to unscale features for plotting)
    std: np.ndarray | None = None
    member_scores: np.ndarray | None = None   # (n_members, n_events) for ensembles
    histories: list[dict] = field(default_factory=list)  # per-model {train_loss: [...], val_loss: [...]}
    output_dir: Path = Path(".")
    tag: str = ""                              # e.g. "epoch_0050", "ensemble_test"

    @property
    def ratio(self) -> np.ndarray:
        """Density ratio r = p_target / p_ref = s / (1 - s)."""
        s = np.clip(self.scores, 1e-12, 1.0 - 1e-12)
        return s / (1.0 - s)

    def features_raw(self) -> np.ndarray:
        """Features in physical units (inverse of the standard scaler)."""
        if self.features is None:
            raise ValueError("Context carries no features.")
        if self.mean is None or self.std is None:
            return self.features
        return self.features * self.std + self.mean

    def out_path(self, stem: str, suffix: str = ".pdf") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_{self.tag}" if self.tag else ""
        return self.output_dir / f"{stem}{tag}{suffix}"


class Metric(ABC):
    """Auxiliary scalar diagnostic. Register with @register_metric("name")."""

    @abstractmethod
    def evaluate(self, ctx: EvaluationContext) -> dict[str, float]:
        ...


class Plot(ABC):
    """Auxiliary figure. Register with @register_plot("name")."""

    @abstractmethod
    def plot(self, ctx: EvaluationContext) -> Path:
        ...


@dataclass
class _Scheduled:
    obj: Metric | Plot
    every_n_epochs: int = 1


def _due(item: _Scheduled, epoch: int | None) -> bool:
    return epoch is None or (epoch + 1) % max(1, item.every_n_epochs) == 0


class EvaluationSuite:
    """An ordered, schedulable collection of metrics and plots.

    Built from config by :func:`build_suite`; used both inside the training
    loop (with an epoch argument controlling the schedule) and once at the
    end of a training/ensemble run (epoch=None runs everything).
    """

    def __init__(self):
        self._metrics: list[_Scheduled] = []
        self._plots: list[_Scheduled] = []

    def add_metric(self, metric: Metric, every_n_epochs: int = 1) -> "EvaluationSuite":
        self._metrics.append(_Scheduled(metric, every_n_epochs))
        return self

    def add_plot(self, plot: Plot, every_n_epochs: int = 1) -> "EvaluationSuite":
        self._plots.append(_Scheduled(plot, every_n_epochs))
        return self

    def __len__(self) -> int:
        return len(self._metrics) + len(self._plots)

    def due_at(self, epoch: int | None) -> bool:
        """Is anything scheduled to run at this epoch?

        Lets the training loop skip the expensive bookkeeping (buffering and
        transferring the whole validation set) on epochs where no diagnostic
        would consume it.
        """
        return any(_due(item, epoch) for item in self._metrics + self._plots)

    def run(self, ctx: EvaluationContext, epoch: int | None = None) -> dict[str, float]:
        """Run everything due. Returns the merged metric results."""

        def due(item: _Scheduled) -> bool:
            return _due(item, epoch)

        results: dict[str, float] = {}
        for item in self._metrics:
            if due(item):
                results.update(item.obj.evaluate(ctx))
        for item in self._plots:
            if due(item):
                item.obj.plot(ctx)
        return results


def build_suite(config: dict | None) -> EvaluationSuite:
    """Build a suite from a config block like::

        metrics:
          - name: density_ratio_integral
            every_n_epochs: 5
        plots:
          - name: calibration_curve
            every_n_epochs: 25
            kwargs: {n_bins: 40}
    """
    from ..registry import METRICS, PLOTS, build

    suite = EvaluationSuite()
    if not config:
        return suite
    for spec in config.get("metrics", []) or []:
        suite.add_metric(build(METRICS, spec), spec.get("every_n_epochs", 1))
    for spec in config.get("plots", []) or []:
        suite.add_plot(build(PLOTS, spec), spec.get("every_n_epochs", 1))
    return suite
