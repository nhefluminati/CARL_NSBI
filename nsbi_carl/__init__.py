"""nsbi_carl: composable training workspace for CARL/NSBI classifier ensembles."""

from .pipeline import Pipeline
from .model import CARL
from .data.dataset import NSBIDataset, SplitIndices
from .data.loading import DatasetBuilder
from .data.splitting import SplitStep
from .data.reweighting import ReweightStep
from .data.scaling import StandardScalerStep
from .training.trainer import CARLTrainer, ModelConfig, TrainerConfig
from .training.ensemble import CARLEnsemble, EnsembleConfig
from .evaluation.base import EvaluationContext, EvaluationSuite, Metric, Plot, build_suite
from .registry import register_metric, register_plot

__all__ = [
    "Pipeline", "CARL", "NSBIDataset", "SplitIndices", "DatasetBuilder",
    "SplitStep", "ReweightStep", "StandardScalerStep", "CARLTrainer",
    "ModelConfig", "TrainerConfig", "CARLEnsemble", "EnsembleConfig",
    "EvaluationContext", "EvaluationSuite", "Metric", "Plot", "build_suite",
    "register_metric", "register_plot",
]
