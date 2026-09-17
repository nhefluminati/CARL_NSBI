"""In-memory dataset container used by all pipeline steps.

The container is deliberately dumb: it holds tensors plus bookkeeping arrays
(labels, per-event sample ids) and exposes them to torch. All logic that
*changes* the data (reweighting, scaling, splitting) lives in the dedicated
step classes in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class NSBIDataset(Dataset):
    """Events from target (label 1) and reference (label 0) samples.

    ``sample_id`` maps every event to the file it came from (index into
    ``sample_names``) so per-sample reweighting stays possible after
    concatenation.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        sample_id: np.ndarray,
        sample_names: list[str],
        feature_names: list[str],
        split_label: np.ndarray | None = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.w = torch.as_tensor(w, dtype=torch.float64).reshape(-1, 1)
        self.sample_id = np.asarray(sample_id, dtype=np.int64)
        self.sample_names = list(sample_names)
        self.feature_names = list(feature_names)

        # Per-event split assignment: 0 = train, 1 = val, 2 = test, -1 = not
        # yet assigned (the usual case; SplitStep then draws it). Events that
        # arrive with a label keep it — this is how a reference sample loaded
        # from a cache file carries its original split across runs.
        if split_label is None:
            self.split_label = np.full(len(self.sample_id), -1, dtype=np.int8)
        else:
            self.split_label = np.asarray(split_label, dtype=np.int8)

        # Filled by the preprocessing step:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    # -- convenience masks -------------------------------------------------
    @property
    def target_mask(self) -> np.ndarray:
        return self.y.numpy().reshape(-1) == 1.0

    @property
    def reference_mask(self) -> np.ndarray:
        return self.y.numpy().reshape(-1) == 0.0

    @property
    def n_target(self) -> int:
        return int(self.target_mask.sum())

    @property
    def n_reference(self) -> int:
        return int(self.reference_mask.sum())

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    # -- torch Dataset API --------------------------------------------------
    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx].float()
