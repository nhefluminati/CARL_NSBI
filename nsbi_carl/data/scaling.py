"""Pipeline step: standardize features to mean 0, variance 1.

The scaler is fitted on the TRAIN split only and applied to everything, so
validation/test remain honest holdouts. The fitted mean/std (in the exact
feature order used for training) are returned for the run record YAML.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import NSBIDataset, SplitIndices


class StandardScalerStep:
    def apply(self, dataset: NSBIDataset, splits: SplitIndices) -> dict:
        x = dataset.x.numpy()

        x_train = x[splits.train]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0.0] = 1.0

        dataset.x = torch.as_tensor((x - mean) / std, dtype=torch.float32)
        dataset.mean = mean.astype(np.float64)
        dataset.std = std.astype(np.float64)

        return {
            "features": dataset.feature_names,
            "scaler_mean": mean,
            "scaler_std": std,
        }

    @staticmethod
    def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply a previously fitted scaler (e.g. loaded from a run record)."""
        return (np.asarray(x) - np.asarray(mean)) / np.asarray(std)

    @staticmethod
    def inverse_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.asarray(x) * np.asarray(std) + np.asarray(mean)
