"""Pipeline step: build an NSBIDataset from target/reference .h5 files."""

from __future__ import annotations

from pathlib import Path

import h5py as h5
import numpy as np

from .dataset import NSBIDataset


class DatasetBuilder:
    """Reads the requested features and weights from .h5 samples.

    * Events from ``target_paths``   -> label 1
    * Events from ``reference_paths`` -> label 0
    * ``features`` fixes both the selection AND the column order of the
      network inputs; it is written to the run record by the pipeline.
    * ``weight_key`` is the name of the weight dataset inside each file.
    """

    def __init__(
        self,
        target_paths: list[str],
        reference_paths: list[str],
        features: list[str],
        weight_key: str = "weight",
    ):
        if isinstance(target_paths, str):
            target_paths = [target_paths]
        if isinstance(reference_paths, str):
            reference_paths = [reference_paths]
        self.target_paths = [str(p) for p in target_paths]
        self.reference_paths = [str(p) for p in reference_paths]
        self.features = list(features)
        self.weight_key = weight_key

    # ------------------------------------------------------------------
    def _read_sample(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        with h5.File(path, "r") as f:
            missing = [k for k in self.features + [self.weight_key] if k not in f]
            if missing:
                raise KeyError(f"{path} is missing keys {missing}")
            x = np.stack([f[k][:] for k in self.features], axis=1).astype(np.float64)
            w = f[self.weight_key][:].astype(np.float64)
        if len(w) != len(x):
            raise ValueError(f"{path}: weight length != feature length")
        return x, w

    def build(self) -> NSBIDataset:
        xs, ws, ys, sids, names = [], [], [], [], []
        sid = 0
        for path in self.target_paths + self.reference_paths:
            is_target = path in self.target_paths
            x, w = self._read_sample(path)
            xs.append(x)
            ws.append(w)
            ys.append(np.full(len(w), 1.0 if is_target else 0.0))
            sids.append(np.full(len(w), sid, dtype=np.int64))
            names.append(Path(path).name)
            sid += 1

        return NSBIDataset(
            x=np.concatenate(xs, axis=0),
            y=np.concatenate(ys),
            w=np.concatenate(ws),
            sample_id=np.concatenate(sids),
            sample_names=names,
            feature_names=self.features,
        )
